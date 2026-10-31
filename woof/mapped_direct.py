"""Mapped-source initialization and atomic stock-WRF hierarchy export."""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import threading
import time
from types import SimpleNamespace
from typing import Mapping, Sequence
import uuid

import numpy as np
from woof.ingest.preparation_evidence import scientific_preprocessing

from woof.boundary_fields import mapping_boundary_species
from woof.config import soil_layer_count
from woof.core.grid import make_vertical_coord
from woof.explain import warn
from woof.vertical_adaptation import (
    adapt_experiment_for_statics,
    vertical_coordinate_receipt as _vertical_coordinate_receipt,
)
from woof.experiment import load_experiment, validate_boundary_timing
from woof.fortran_namelist import parse_namelist
from woof.ingest.horiz import (
    declared_grid_pairing, interpolate_era5_to_lambert, with_inland_air_temperature)
from woof.ingest.soil_downscale import (
    declared_soil_texture_downscale, soil_mesh_plan_from_case)
from woof.ingest.lateral_bc import (
    StateBoundaryFrames,
    attach_lateral_boundaries,
    start_last_forcing_order,
)
from woof.ingest.cg_topo import RootTerrainBlend
from woof.ingest.boundary_stream import (
    HIERARCHY_HEAD_DIRNAME,
    PreparedTreeWriter,
    TreeStartStates,
    chained_enabled,
    domain_tree_head_fields,
    prepared_head_urban_columns,
    producer_device_bytes,
    remove_unfinished_tree,
    input_plan_sha256,
)
from woof.ingest.prepared_cache import (
    prepared_cache_identity,
)
from woof.ingest.memory_refusal import InitializationMemoryRefused
from woof.ingest.preparation_price import (
    price_forcing_preparation, price_preparation_floor)
from woof.ingest.preprocess_backend import (
    preprocess_math_call,
    admit_preparation,
    move_start_result_to_host,
    start_time_backend,
    preprocess_identity,
    release_backend_memory,
    resolve_preprocess_backend,
)
from woof.ingest.real import aerosol_donor_snapshot, initialize_real
from woof.ingest.soil import (
    door_reconciled_soil_category, soil_temperature_repair_proof)
from woof.ingest.source_coverage import (
    ForcingSeriesRefusal,
    PreparationRefusal,
    RunInputRefusal,
    VerticalLadderRefusal,
    existing_output_root_refusal,
    owns_source_coverage_refusal,
)
from woof.ingest.land_state import land_state_context
from woof.ingest.ruc_soil import preprocess_land_surface_soil
from woof.ingest.water_temperature import WaterTemperatureStatics
from woof.mapped_composition import (
    MappedSourceBundle,
    _decoder_inventory,
    _path_inventory,
    composition_receipt_identity_sha256,
    decode_composed_source,
    decoded_vertical_ladder,
    mapped_composition_receipt,
)
from woof.mapped_engine_bridge import (
    ENGINE_ENV as _ENGINE_ENV,
    ENGINE_RUST as _ENGINE_RUST,
    ENGINES as _ENGINES,
    MAPPED_ROUTE_SUBCOMMAND as _ROUTE_SUBCOMMAND,
)
from woof.mapped_source import (
    _load_json_bytes,
    _load_json_document,
    _mapped_engine_choice,
    _require_authority_snapshot,
    _sha256,
    _snapshot_authority,
    load_mapping,
    read_input_list,
    warn_regular_join_drops,
)
from woof.moisture_floor_receipt import moisture_floor_proof_entry
from woof.native_wrf_contract import (
    canonical_noah_surface,
    load_native_static_cache,
    native_static_export_fields,
    validate_native_lambert_contract,
    validate_native_lambert_contracts,
    verify_native_static_receipt,
    write_native_geometry_receipt,
    write_native_static_cache,
)
from woof.source_authorities import (
    BOUNDARY_MULTIPLES_KEY, boundary_interval_refusal)
from woof.source_hierarchy import (
    initialize_and_export_regular_source_hierarchy,
    prepare_regular_source_hierarchy_head,
    seal_regular_source_hierarchy,
)
from woof.static.build import GeogSelection, build_static
from woof.vertical_contract import validate_explicit_eta_grid
from woof.wrf_direct import (
    StockWrfExportUnsupported, export_prepared_wrf,
    stock_wrf_export_not_requested, stock_wrf_export_refused,
    validate_stock_wrf_export_config, validate_stock_wrf_export_hierarchy,
)
from woof.native_domain_artifacts import (
    published_path_refusal,
    root_domain_artifact_binding,
    write_child_domain_artifacts,
    write_domain_static_files,
)
from woof.native_hierarchy import (
    STOCK_WRF_EXPORT_MODES,
    hierarchy_moisture_floor_receipts,
)
from woof.progress import prep_stage


PROOF_SCHEMA = "gpuwm-mapped-direct-wrf-proof-v1"
HIERARCHY_PROOF_SCHEMA = "gpuwm-mapped-native-hierarchy-proof-v1"


def _canonical(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    )


#: The mapped/rw-wps route's name in every water-temperature refusal
#: and receipt.  One name for every composition that reaches this
#: adapter, 20CRv3 included.
_WATER_ROUTE = "the mapped rw-wps route"


def _forcing_series(bundle):
    """The bundle's snapshots in valid-time order.

    A composed bundle answers with a sequence that packs ONE valid time
    when that time is asked for and drops it when the next is, which is
    what keeps a seven-time preparation from holding seven valid times
    (the measured 15.9 GiB per forcing time on a 3 km CONUS source).  A
    caller that hands over a plain tuple -- an adapter fixture, a route
    that already has its snapshots -- is sorted the way it always was.
    """

    snapshots = bundle.regular_snapshots()
    ordered = getattr(snapshots, "sorted_by_valid_time", None)
    if ordered is not None:
        return ordered()
    return tuple(sorted(
        snapshots, key=lambda snapshot: snapshot.valid_time))


def _forcing_valid_times(snapshots) -> tuple:
    """Every forcing valid time, without packing a snapshot to read one."""

    declared = getattr(snapshots, "valid_times", None)
    if declared is not None:
        return tuple(declared)
    return tuple(snapshot.valid_time for snapshot in snapshots)


def _posted_window_leads(posted_source, times) -> tuple:
    """The posted leads of the run's own forcing window, in its order.

    The fetch can schedule more leads than the run reads: ``[fetch] hours``
    is forcing coverage, and the packaged HRRR recipe fetches 3 h for a 2 h
    run.  ``_within_run_window`` keeps only the window's times, so the
    boundary schedule has one interval per pair of those times.  Naming
    every scheduled lead in the as-posted head refused the run at "an
    as-posted head names 4 forcing leads for 2 boundary intervals" (full
    CONUS HRRR from HRRR f00 + RAP f03-f06, 2026-10-21).  The
    head binds exactly the window's leads; a later scheduled lead is not
    forcing for this run.
    """

    valid = tuple(posted_source.valid_times)
    try:
        positions = [valid.index(value) for value in times]
    except ValueError:
        raise ValueError(
            f"the run's forcing times {[str(value) for value in times]} are "
            f"not all scheduled leads of the as-posted fetch "
            f"({[str(value) for value in valid]})") from None
    if positions != list(range(positions[0], positions[0] + len(positions))):
        raise ValueError(
            "the run's forcing times are not consecutive scheduled leads of "
            "the as-posted fetch")
    return tuple(posted_source.leads[index] for index in positions)


def _within_run_window(snapshots, start, run_seconds):
    """Keep the run's forcing window through its final bracketing time.

    One source file can hold a whole forecast: Earth2Studio writes every
    lead of a run into one NetCDF (lead 0 to 96 h, or four months of a
    seasonal run), and a domain starts at any lead of it.  The leads
    before the start and beyond its final boundary are not forcing for this run;
    leaving them in refused the file with "mapped forcing must begin at
    START, got [lead 0]".  The window's own times still have to be there,
    uniformly spaced, which the checks after this one keep refusing.
    """

    from datetime import timedelta

    end = start + timedelta(seconds=float(run_seconds))
    times = _forcing_valid_times(snapshots)
    keep = [index for index, time in enumerate(times) if start <= time <= end]
    # A run shorter than the source's step (1 h from a 6-hourly Earth-2 run) still needs the first
    # forcing time at or after its end: without it only the start survived and prep refused
    # "mapped forecast preparation requires at least two forcing times" (2.9.3 Earth-2 door leg).
    if keep and times[keep[-1]] < end:
        later = [index for index, time in enumerate(times) if time > end]
        if later:
            keep.append(min(later, key=lambda index: times[index]))
    keep = tuple(keep)
    if len(keep) == len(times) or not keep:
        return snapshots
    select = getattr(snapshots, "select", None)
    if select is not None:
        return select(keep)
    return tuple(snapshots[index] for index in keep)


def _source_top_pressure_pa(snapshots, count: int | None = None, *,
                            mapping_contract=None) -> float:
    """The highest source level the whole forcing series reaches, in Pa.

    Read from the pressure field alone where the series can do that:
    packing a whole valid time to answer one number would read every
    array a frame carries, for every forcing time, before the first one
    is interpolated.  ``count`` limits it to the first times (an
    as-posted head reads the leads decoded so far).
    """

    count = len(snapshots) if count is None else int(count)
    levels = getattr(snapshots, "source_pressure_hpa", None)
    if levels is None:
        mass_top = max(
            float(np.min(snapshots[index].levels_hpa) * 100.0)
            for index in range(count))
    else:
        mass_top = max(
            float(np.min(levels(index)) * 100.0)
            for index in range(count))
    # Native files publish mass levels, while p_top is the interface
    # bounding the top layer.  Comparing those unlike coordinates refuses
    # a run on the source's own ladder.  The mapped table supplies the
    # interface; the interpolation operator separately enforces every
    # target mass level's support in the decoded source column.
    vertical = (mapping_contract or {}).get("coordinates", {}).get("vertical", {})
    model_top = vertical.get("model_top_pressure_pa")
    if model_top is None:
        return mass_top
    model_top = float(model_top)
    if not np.isfinite(model_top) or model_top <= 0.0 or model_top > mass_top:
        raise ValueError("native model-top interface must be positive and above its decoded mass levels")
    return model_top


def _soil_floor_receipts(soil) -> dict[str, object]:
    """The root soil's conditional receipts, for the proof AND the cache.

    ``soil_moisture_floor`` (real.exe's RUC layer-source
    ``SMOIS = MAX(SMOIS, 0.005)``, :class:`woof.ingest.ruc_soil.RucSoilState`)
    is present only when the floor raised a value.  The front door binds
    every :data:`woof.ingest.prepared_cache.CONDITIONAL_PREPARATION_RECEIPTS`
    key from the proof and compares the cache's user metadata EXACTLY, so
    the one dict this returns goes into both: a receipt in the proof and
    not in the header made every RUC run whose floor fired refuse its own
    preparation (2.8.8 acceptance D-02).
    """

    floor = getattr(soil, "soil_moisture_floor", None)
    return {"soil_moisture_floor": dict(floor)} if floor else {}


def _file_receipt(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _copy_bound_authority(
    source: Path,
    destination: Path,
    expected_sha256: str,
) -> None:
    """Copy one small evidence authority and prove the copied bytes."""

    shutil.copy2(source, destination)
    if _sha256(destination) != expected_sha256:
        raise ValueError(
            f"mapped evidence changed before publication: {source}"
        )


def _bound_provenance_authorities(
    bundle: MappedSourceBundle,
    composition: Mapping[str, object],
) -> dict[str, tuple[Path, str]]:
    """Resolve declared roles to the identities actually used by the decoder.

    ``composition`` is the already hash-verified publication copy.  A donor
    record names a binding, not its provenance role; only the bound declaration
    can supply that relationship.  Terrain may itself come from a donor.
    """

    terrain_identity = (
        bundle.terrain_provenance_path.resolve(),
        bundle.terrain_provenance_sha256,
    )
    authorities: dict[str, tuple[Path, str]] = {}

    def add(role, identity):
        previous = authorities.get(role)
        if previous is not None and previous != identity:
            raise ValueError(f"decoded provenance identities conflict for role {role!r}")
        authorities[role] = identity

    terrain = composition["supplements"].get("terrain_height")
    if terrain is not None:
        add(terrain["provenance_role"], terrain_identity)
    bindings = composition.get("field_sources", {})
    seen = set()
    for record in bundle.contributing_sources:
        name = record["binding"]
        if name not in bindings or name in seen:
            raise ValueError("decoded contributing source inventory differs from composition")
        seen.add(name)
        declaration = bindings[name]
        provenance = record["provenance"]
        identity = (Path(provenance["path"]).resolve(), provenance["sha256"])
        if "terrain_height" in declaration["fields"] and identity != terrain_identity:
            raise ValueError("decoded terrain provenance differs from its contributing source")
        add(declaration["provenance_role"], identity)
    if seen != set(bindings):
        raise ValueError("decoded contributing source inventory differs from composition")
    return authorities


def _provenance_evidence_name(role: str, suffix: str) -> str:
    """Return a short deterministic filename for one provenance role.

    Composition role names remain recorded in the bound composition receipt;
    repeating an arbitrary role verbatim in the filesystem only burns scarce
    Windows path budget.  Sixteen SHA-256 hex digits keep names stable and
    collision-resistant, while the create-only destination check below still
    fails closed if two roles ever collide.
    """

    role_digest = hashlib.sha256(role.encode("utf-8")).hexdigest()[:16]
    return f"provenance-{role_digest}{suffix}"


def _bound_decoder_receipts(
    paths: Mapping[str, Path],
    expected_sha256: Mapping[str, str],
) -> dict[str, dict[str, object]]:
    """Recheck small decoder executables before emitting proof evidence."""

    if set(paths) != set(expected_sha256):
        raise ValueError("decoded decoder inventory is internally inconsistent")
    receipts = {}
    for role, path in paths.items():
        receipt = _file_receipt(path)
        if receipt["sha256"] != expected_sha256[role]:
            raise ValueError(f"mapped decoder changed after decode: {role}")
        receipts[role] = receipt
    return receipts


def _plan_review_spacing(
    target: Mapping[str, object], wps_namelist: Path,
) -> tuple[int | None, str | None]:
    """The boundary spacing plan review holds the experiment to, and where
    it was read, before any source is decoded.

    A target that takes only its declared spacing: that spacing, the one a
    series it decodes can carry.  A target that takes whole multiples of it
    (A173): the namelist's ``&share/interval_seconds`` where the target
    takes it, because that is the spacing this run is prepared at -- the
    GDT-101 normalization holds it equal to the gap between the fetched
    leads, and ``woof domain`` and the catalog write the cadence they
    fetch.  Holding such a run to the target's base spacing instead
    refused a root step that divides the run's spacing and not the base:
    icon-global at its default 3 h with dt = 54 s was refused naming the
    3600 s the publisher posts and the run never uses.  Without a usable
    namelist interval the spacing is ``None``: it is not known until the
    decode, and the decoded series' own spacing is checked whole there.
    """

    declared = target.get("boundary_interval_seconds")
    if target.get(BOUNDARY_MULTIPLES_KEY) is not True:
        return declared, None
    values = parse_namelist(wps_namelist).get("share", {}).get(
        "interval_seconds") or ()
    interval = values[0] if values else None
    if (isinstance(interval, int) and not isinstance(interval, bool)
            and interval > 0
            and boundary_interval_refusal(target, interval) is None):
        return interval, f"&share/interval_seconds of {wps_namelist}"
    return None, None


def _validate_target_contract(
    mapping: Mapping[str, object],
    exp,
    boundary_interval_seconds: int | None,
    *,
    hierarchy: bool,
    experiment_config: Path | None = None,
    spacing_origin: str | None = None,
    before_decode: bool = False,
) -> dict[str, object]:
    """Bind mapped target limits to one resolved experiment and cadence.

    ``spacing_origin`` names where a spacing read before the decode came
    from, so a refusal says which number to change.  ``before_decode``
    with no spacing is plan review of a series whose spacing only the
    decode knows (:func:`_plan_review_spacing`): every other limit is
    checked, and the cadence and the timing law wait for the series.

    The vertical dimension is the EXPERIMENT'S, not the mapping's: the
    whole route interpolates onto the experiment's explicit eta ladder
    (``make_vertical_coord``/``validate_explicit_eta_grid`` read only
    ``exp``), so a config that carries a valid ladder is adopted at its
    own level count and the mapping's ``target_vertical_levels`` is
    recorded as the reference it is.  Refusing a valid ladder over the
    count named no breakage, and paired with the later shape-(0,) error
    it formed the circle of UX finding N6: 44 levels refused for 49,
    then 49 refused for an explicit ladder nothing the user ran had
    ever written.  What IS refused, in one message with both counts and
    both reconciling doors, is a config with no ladder at all -- a
    count alone does not define the interpolation target and WRF's
    automatic level generator is not implemented.
    """

    target = mapping["target"]
    if not isinstance(target, dict):
        raise TypeError("mapping target must be an object")
    domain_count = len(exp.domains)
    max_dom = target["max_dom"]
    if domain_count > max_dom:
        raise ValueError(
            f"mapped target allows max_dom={max_dom}, experiment requests "
            f"{domain_count} domains"
        )
    target_vertical_levels = target["target_vertical_levels"]
    nz = int(exp.root.run.nz)
    eta_levels = tuple(getattr(exp.vertical, "eta_levels", ()) or ())
    config_name = (
        "the experiment config" if experiment_config is None
        else str(experiment_config))
    if not eta_levels:
        counts = (
            f"declares nz={nz} mass levels (WRF e_vert={nz + 1}) and no "
            "explicit eta_levels ladder")
        against = (
            f"matching this mapping's reference count"
            if nz == target_vertical_levels else
            f"and this mapping's reference target is "
            f"{target_vertical_levels} levels "
            f"(WRF e_vert={target_vertical_levels + 1})")
        raise VerticalLadderRefusal(
            f"mapped vertical ladder is missing: {config_name} {counts}, "
            f"{against}.  The mapped route interpolates every forcing "
            "time onto an explicit full-level eta ladder; a level count "
            "alone does not define one for this entry point",
            remedy=(
                f"remedy: two doors reconcile this.  Keep your {nz} "
                f"levels: add an explicit eta_levels ladder of {nz + 1} "
                "interfaces -- `eta_levels = [1.0, ..., 0.0]`, strictly "
                f"decreasing -- to the [shared] block of {config_name}; "
                "prep adopts your ladder at your level count.  Or use "
                "the packaged reference ladder: `woof domain` authors "
                "a config whose [shared] block carries the certified "
                f"{target_vertical_levels}-level ladder; copy its "
                "nz/p_top/eta_levels lines into your imported config."))
    try:
        validate_explicit_eta_grid(
            eta_levels, nz=nz, p_top=exp.vertical.p_top,
            context="mapped experiment vertical ladder",
        )
    except ValueError as error:
        raise VerticalLadderRefusal(
            str(error),
            remedy=(
                f"remedy: fix the [shared] eta_levels ladder in "
                f"{config_name}: nz + 1 entries (WRF e_vert), running "
                "1.0 (surface) to 0.0 (top), strictly decreasing, with "
                "p_top in pascals inside the source atmosphere."),
        ) from error
    # A mapping may decode an isolated analysis as well as a forcing series.
    # Its decode-time minimum does not prohibit forecasting from a complete
    # series. This operation checks the actual times, coverage and cadence
    # before static preparation and always records its boundary requirement.
    if boundary_interval_seconds is not None or not before_decode:
        if (isinstance(boundary_interval_seconds, bool)
                or not isinstance(boundary_interval_seconds, int)
                or boundary_interval_seconds <= 0):
            raise ValueError(
                "mapped boundary interval must be a positive integer")
        refusal = boundary_interval_refusal(
            target, boundary_interval_seconds,
            subject="mapped boundary interval")
        if refusal is not None:
            raise ValueError(refusal)
        label = "mapped hierarchy target" if hierarchy else "mapped target"
        validate_boundary_timing(
            exp, boundary_interval_seconds,
            source=(label if spacing_origin is None
                    else f"{label} at {spacing_origin}"))
    # Plan review is where a declared field the target join has no consumer
    # for is named: the mapping document and the experiment are both in hand
    # here, ahead of every decode and fetch.  It is a notice, not a refusal.
    # Carrying more than the target consumes is a drop, and the same
    # function answers the question at the frame join.
    dropped = warn_regular_join_drops(
        mapping.get("fields") or (), subject="this mapping")
    return {
        "status": "PASS",
        "regular_join_dropped_fields": list(dropped),
        "domain_count": domain_count,
        "domain_ids": [domain.grid_id for domain in exp.domains],
        "domain_start_times": {
            f"d{domain.grid_id:02d}": (
                exp.domain_start_time(domain.grid_id).isoformat()
                if hasattr(exp, "domain_start_time")
                else (
                    exp.start_time
                    if getattr(domain, "start_time", None) is None
                    else domain.start_time
                ).isoformat())
            for domain in exp.domains
        },
        "mapping_max_dom": max_dom,
        # The ADOPTED count -- the experiment's, which is what every
        # array downstream is allocated at; the mapping's declared
        # count is recorded beside it as the reference it is.
        "target_vertical_levels": nz,
        "mapping_reference_vertical_levels": target_vertical_levels,
        "vertical_levels_adopted_from": (
            "mapping-reference-count" if nz == target_vertical_levels
            else "experiment-eta-ladder"),
        "require_lateral_boundaries": True,
        "boundary_interval_seconds": boundary_interval_seconds,
        "hierarchy": hierarchy,
    }


#: Which subprocess tools each mapped format's Python engine launches.
#: The same table ``mapped_composition._decoder_inventory`` enforces;
#: named here only to say what a MISSING role means to a reader.
_FORMAT_TOOL_ROLES = {
    "grib1": ("grib1_bridge",),
    "grib2": ("grib2_inventory", "grib2_dump"),
    "netcdf": (),
}


#: The tool-role flag each role is spelled with on the command line.
_TOOL_ROLE_FLAGS = {
    "grib1_bridge": "--grib1-bridge",
    "grib2_inventory": "--grib2-inventory",
    "grib2_dump": "--grib2-dump",
}


def _decoder_inventory_refusal(message, formats, *, missing=(), extra=()):
    """The decoder-contract fault, as a refusal this door can deliver.

    Named breakage: a bare ``woof prep`` of any composed GRIB2 source
    reached ``ValueError: grib2 decoder inventory differs from the
    contract; missing=['grib2_dump', 'grib2_inventory']`` as an
    unhandled traceback, so a reader met this package's line numbers
    before meeting a remedy.  The contract is unchanged and still
    refuses; this gives the same sentence a delivery.

    The remedy is composed for THIS install, and the two directions get
    opposite ones because they are opposite faults with the same
    message shape: a MISSING tool gets the estate answer ``woof
    doctor`` would print (staged bundle, or a clone and a build), and a
    tool supplied to a route that decodes in process gets told to drop
    the pin -- there is nothing here that would ever launch it.
    """

    from woof import bridges
    from woof.ingest.source_coverage import DecoderInventoryRefusal
    from woof.mapped_engine_bridge import ENGINE_ENV, ENGINE_PYTHON

    text = str(message)
    missing = tuple(sorted(missing))
    extra = tuple(sorted(extra))
    label = "+".join(sorted({str(name) for name in formats}))
    if missing:
        remedy = (
            f"remedy: this route composes {label} on the Python engine, "
            f"which reads it through {', '.join(missing)}.\n"
            + bridges.bridge_remedy(missing[0])
            + "\n  # `woof prep` resolves these through the same ladder "
            "with no flags once they are staged; "
            + " / ".join(_TOOL_ROLE_FLAGS[role] for role in missing)
            + " override it.")
    elif extra:
        remedy = (
            "remedy: drop "
            + " / ".join(_TOOL_ROLE_FLAGS[role] for role in extra)
            + f" -- this route reads {label} with no such tool and has "
            "nothing to launch it for.  To decode with those exact "
            "executables instead, ask for the engine that runs them: "
            f"{ENGINE_ENV}={ENGINE_PYTHON} (equivalently --mapped-engine "
            f"{ENGINE_PYTHON}).")
    else:
        remedy = DecoderInventoryRefusal.remedy
    return DecoderInventoryRefusal(text, remedy=remedy)


def _announce_vertical_ladder(entry: Mapping[str, object]) -> None:
    """Say, once, that the source files carry fewer levels than declared."""

    units = str(entry["units"])

    def label(value: object) -> str:
        if units == "Pa":
            return f"{float(value) / 100.0:g} hPa"
        return f"{float(value):g} {units}"

    carried = len(entry["decoded_levels"])
    declared = len(entry["declared_levels"])
    absent = ", ".join(label(value) for value in entry["absent_levels"])
    warn(f"the source files carry {carried} of the {declared} vertical "
         f"levels this source declares (absent: {absent}); the column is "
         f"built from the {carried} they carry",
         "The mapping lists this ladder in vertical.era_ladders: another "
         "publication of the same product carries it.  Every field is read "
         "on the ladder all of them share, the receipt records it under "
         "source_vertical_ladder, and a model top above the highest carried "
         "level is still refused.")


def _announce_derived_terrain(entry: Mapping[str, object]) -> None:
    """Say, once, that the source's terrain was derived, not read."""

    cells = int(entry["cells"])
    below = int(entry["cells_below_lowest_level"])
    warn(f"the source files carry no surface geopotential, so terrain is "
         f"derived at all {cells} source cells as each column's height at "
         f"its surface pressure ({below} of them below the lowest pressure "
         f"level, from the 2 m temperature and dewpoint); it spans "
         f"{float(entry['minimum_m']):.0f} to {float(entry['maximum_m']):.0f} m",
         "The mapping declares this under fields.terrain_height.when_absent. "
         "The receipt records it under source_composition.alignment.derived.")


def _announce_adaptation(sentence: str) -> None:
    """Say, once, that the run is not on the configured vertical coordinate."""

    warn(sentence,
         "WRF v4.6.1 dyn_em/nest_init_utils.F:1158-1182 calls a column "
         "the coordinate cannot order fatal and names reducing etac as "
         "the remedy, and a column it only just orders keeps one layer "
         "too thin to integrate, which a lower etac thickens; the etac is "
         "derived here from the terrain this run can actually touch and "
         "applied, so the prepared inputs, their receipt and the forecast "
         "all carry the same coordinate.  p_top is untouched.")


def _survey_static_catalog(exp, wps_namelist, geog_root, static_highres=None):
    """The WPS_GEOG catalog the terrain survey needs, or None."""

    if geog_root is None or len(exp.domains) < 2:
        return None
    from woof.hrrr_native_static import verified_static_catalog
    from woof.static.terrain_smoothing import selection_carrier_kwargs

    catalog, _ = verified_static_catalog(
        Path(wps_namelist), Path(geog_root),
        [domain.grid_id for domain in exp.domains],
        **selection_carrier_kwargs(static_highres))
    return catalog


#: The proof keys that read every lead of a composed window, so an
#: as-posted head leaves them out and its seal writes them: the whole
#: window's composition receipt (its manifest, alignment and every frame)
#: and the vertical ladder its frames carry.
_AS_POSTED_SEAL_KEYS = ("source_composition", "source_vertical_ladder")
#: Where an as-posted mapped proof names the input manifest it sealed.
_AS_POSTED_PROOF_MANIFEST_KEY = "source_composition.input_manifest.sha256"
#: The identity keys that carry the input manifest's own digest.
_AS_POSTED_MANIFEST_BOUND = ("bridge_manifest_sha256", "source_manifest_sha256",
                             "input_manifest_sha256")
#: The as-posted fetch's published manifest, whose ``prep`` block names
#: every file the route plans for the window (the lead objects).
_FETCH_MANIFEST_NAME = "fetch-manifest.json"


def _naive_time(value):
    from datetime import datetime, timezone

    moment = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if moment.tzinfo is not None:
        moment = moment.astimezone(timezone.utc).replace(tzinfo=None)
    return moment


#: The most leads one as-posted batch decodes when its caller names no bound
#: (:meth:`_PostedMappedSource.through`): six hourly leads is one decode
#: child per six forcing hours, decoded ahead of the builds that read them.
POSTED_BATCH_LEADS = 6


class _PostedMappedSource:
    """A composed source decoded lead batch by lead batch as its leads post.

    DESIGN A136 2.4 item 3 for the mapped engine.  Indexing a forcing time
    waits for its lead's marker (``PostedLeads``), then decodes every lead
    whose marker is there and is not decoded yet in one
    ``decode_composed_source`` run, on all cores, under a sub-manifest
    authored for that batch.  The batches run one at a time in the
    preparation's thread, so the compose never holds more than one batch,
    and a batch is never more than the whole window.

    What a batch reads: the leads' own objects (each held to the digest
    its marker names), every object the route fetched with the first lead
    group but that belongs to no lead (a cycle-invariant field, step-0
    statics), and every input the route does not plan (a donor analysis
    fetched before the first lead).  Where the route concatenated parts
    into the window's first lead's file alone (the cycle's step-0 statics),
    a batch without that lead decodes it again and keeps only its own
    times (:meth:`_needs_first_lead`).  A supplement role drawn from the
    primary inventory (a source whose own files carry its surface) follows
    the batch; any other supplement is read whole by every batch and must
    be there when the first batch is decoded.

    At the seal :meth:`finish` authors the window's input manifest through
    the one-shot author (``author_input_manifest``) and returns the bundle
    one decode of the window would have: every batch's frames as one
    frameset, the batches' alignment receipts merged
    (:func:`woof.mapped_composition.posted_composition_bundle`).
    """

    def __init__(self, *, posting, input_manifest, composition, mapping,
                 primary, supplements, provenance, contributing, decoders,
                 grids, workers, output_root, source_format,
                 physical_members=False):
        from woof.ingest.boundary_stream import (
            POSTING_SCHEDULE_NAME, PostedLeads)
        from woof.mapped_composition import PostedFrames

        self.posting = Path(posting).resolve()
        self.fetch_root = self.posting.parent
        self.input_manifest = Path(input_manifest).resolve()
        if self.input_manifest.parent != self.fetch_root:
            raise ValueError(
                f"an as-posted mapped preparation seals its input manifest "
                f"beside the fetched files ({self.fetch_root}), not at "
                f"{self.input_manifest}: the manifest names each file by its "
                "path from its own folder, and the lead markers name them "
                "from the fetch folder, so the seal holds a row to its marker "
                "only when the two folders are one")
        schedule_path = self.posting / POSTING_SCHEDULE_NAME
        try:
            schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ValueError(
                f"an as-posted preparation reads the fetch's posting "
                f"schedule, and {schedule_path} is not readable: "
                f"{error}") from None
        rows = list(schedule.get("leads") or ())
        if not rows:
            raise ValueError(f"{schedule_path} schedules no lead")
        self.leads = tuple(int(row["lead"]) for row in rows)
        self.valid_times = tuple(_naive_time(row["valid_time"]) for row in rows)
        self.schedule = schedule
        self.posted = PostedLeads(self.posting, source=str(schedule["source"]),
                                  cycle=str(schedule["cycle"]))
        self.composition = Path(composition)
        self.mapping = Path(mapping)
        self.primary = tuple(Path(path) for path in primary)
        self.supplements = {role: tuple(Path(path) for path in paths)
                            for role, paths in supplements.items()}
        self.provenance = dict(provenance)
        self.contributing = dict(contributing)
        self.decoders = dict(decoders)
        self.grids = grids
        self.workers = workers
        self.output_root = Path(output_root)
        self.source_format = str(source_format)
        self.physical_members = physical_members
        #: Every marker read, by lead (what segments bind).
        self.markers: dict[int, dict] = {}
        #: A posted object's lead, or ``None`` for one every batch reads.
        self.lead_of: dict[Path, int | None] = {}
        #: A posted object's ``(bytes, sha256)`` as its marker names it.
        self.expected: dict[Path, tuple] = {}
        #: A posted object's route role, as its marker names it.
        self.role_of: dict[Path, object] = {}
        #: The parts the route concatenated into each lead's file(s).
        self.parts_of: dict[int, tuple[Path, ...]] = {}
        #: Whether a later batch decoded the first lead's file again
        #: (:meth:`_needs_first_lead`), so more than one batch read it.
        self.carried_any = False
        self.planned: set[Path] = set()
        self.batches: list = []
        self.batch_leads: list[tuple[int, ...]] = []
        self.decode_seconds = 0.0
        self.decode_memory_budget_bytes: int | None = None
        self._where: dict[int, tuple] = {}
        self._physical_evidence: dict[int, dict] = {}
        import threading
        self._decode_lock = threading.RLock()
        self._next = 0
        self._scratch = self.output_root.parent / (
            f".posted-{uuid.uuid4().hex[:8]}")
        self.frames = PostedFrames(self.valid_times, self._locate,
                                   prefetch=self.prefetch_through,
                                   decoded=lambda: self._next,
                                   posted=self.posted_count)

    # -- reading ---------------------------------------------------------

    def _read_marker(self, lead: int, marker: Mapping[str, object]) -> None:
        from woof.fetch_as_posted import marker_files

        self.markers[int(lead)] = dict(marker)
        for item in marker_files(marker):
            path = (self.fetch_root / str(item["name"])).resolve()
            own = item.get("lead", lead)
            self.lead_of[path] = int(lead) if own == lead else None
            self.expected[path] = (item.get("bytes"), item.get("sha256"))
            if "role" in item:
                self.role_of[path] = item.get("role")
        self.parts_of[int(lead)] = tuple(
            (self.fetch_root / str(part)).resolve()
            for item in (marker.get("composed") or ())
            if isinstance(item, Mapping)
            for part in (item.get("parts") or ()))

    def _read_plan(self) -> None:
        path = self.fetch_root / _FETCH_MANIFEST_NAME
        try:
            fetched = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ValueError(
                f"an as-posted mapped preparation reads which files the "
                f"route plans from {path}, which the fetch publishes with "
                f"its first lead, and it is not readable: {error}") from None
        prep = fetched.get("prep") or {}
        self.planned = {
            (self.fetch_root / str(name)).resolve()
            for name in (*(prep.get("primary_files") or ()),
                         *(prep.get("supplement_files") or ()))}
        if not self.planned:
            raise ValueError(
                f"{path} names no planned file (prep.primary_files); an "
                "as-posted mapped preparation needs the route's plan to tell "
                "a lead's objects from the inputs its head reads whole")

    @property
    def fixed(self) -> tuple[Path, ...]:
        """The data inputs the route does not plan: read whole at the head."""

        data = (*self.primary,
                *(path for paths in self.supplements.values() for path in paths))
        return tuple(dict.fromkeys(
            path for path in data if path.resolve() not in self.planned))

    def _available(self, path: Path) -> bool:
        path = path.resolve()
        return path not in self.planned or path in self.lead_of

    def _in_batch(self, path: Path, leads) -> bool:
        path = path.resolve()
        if path not in self.planned:
            return True
        if path not in self.lead_of:
            return False
        owner = self.lead_of[path]
        return owner is None or owner in leads

    # -- decoding --------------------------------------------------------

    def start(self):
        """Wait for the first lead, read the route's plan, decode the first batch."""

        self._read_marker(self.leads[0], self.posted.wait(self.leads[0]))
        self._read_plan()
        missing = [str(path) for path in self.primary
                   if path.resolve() not in self.planned
                   and not path.is_file()]
        if missing:
            raise ValueError(
                f"the as-posted preparation's inputs {missing} are neither "
                "planned by the route nor present")
        # The first batch is what the first step needs: the start time and
        # the forcing time after it.  Every later lead is decoded when it is
        # asked for (prefetch_through), or beside the forecast by the
        # remaining-forcing stage (prepare_independent_times).
        self.through(0, until=min(1, len(self.leads) - 1))
        return self.batches[0]

    def prefetch_through(self, position: int) -> None:
        """Decode every lead through ``leads[position]`` in one batch, and no further.

        For a reader about to ask for several times at once (the inland
        water's daily-mean 2 m air temperature reads the first 24 h): one
        batch through the last of them, where reads arriving side by side
        would each extend the batch to every posted lead.
        """

        self.through(int(position), until=int(position))

    def through(self, position: int, *, until: int | None = None) -> None:
        """Decode every lead through ``leads[position]``, waiting as they post.

        A batch also takes later leads already posted: up to
        ``leads[until]`` when ``until`` is given, otherwise at most
        :data:`POSTED_BATCH_LEADS` leads in all, so a window that posted
        long ago is still decoded a few leads at a time, ahead of the
        forcing times that read them.

        Named breakages the bound prevents (V59/V60, 2026-10-21): with every
        posted lead in one batch, the start state's inland-water mean asked
        for f013 and the root stage composed all 36 boundary frames of a
        48 h RAP window before the first step, which needs two; and once
        those leads moved after the head, the remaining-forcing stage
        composed all of them in one budgeted child before its first
        boundary time, so the forecast waited ~24 min with every card idle.
        """

        with self._decode_lock:
            while self._next <= position:
                lead = self.leads[self._next]
                if lead not in self.markers:
                    self._read_marker(lead, self.posted.wait(lead))
                end = self._next
                # A NetCDF decode holds its whole batch at once (it is priced
                # per batch, not per time), so its later batches are one lead:
                # each then fits the cap the first decode's price set.
                batch = 1 if self.source_format == "netcdf" else POSTED_BATCH_LEADS
                last = (max(int(until), position) if until is not None
                        else self._next + batch - 1)
                last = min(last, len(self.leads) - 1)
                while end + 1 <= last:
                    later = self.leads[end + 1]
                    record = self.posted.marker(later)
                    if record is None:
                        break
                    self._read_marker(later, record)
                    end += 1
                self._decode_batch(self._next, end)
                self._next = end + 1

    def posted_count(self, now=None) -> int:
        """How many of the window's leads, from the first, the source has posted now.

        What a start-time reader may take without holding the head for a
        lead the source has not published yet (the inland water's daily
        mean, woof.ingest.horiz).  Read from the posting schedule the fetch
        published before its first lead: a lead counts once its
        ``expected_at`` (the source's publication rule) has passed, whether
        or not this fetch has moved it, and the reader then waits for its
        marker (:meth:`through`), so a lead that never comes ends the wait
        loudly (``failed.json``) instead of leaving the mean short.  A row
        with no readable ``expected_at`` counts as posted, so it is waited
        for.  ``now`` is naive UTC (default: the clock).

        Breakage it prevents: counted from this fetch's own markers, an
        archived cycle's daily mean averaged the leads the download had
        finished when the start state was built -- 1 of 3 from a cold cache,
        all 3 from a warm one (HRRR native 2025-03-14 12Z, 2026-10-30) -- so
        one case prepared lake temperatures up to 2.17 K apart on its 14
        inland-water cells and forecast differently by cache state.
        """

        if now is None:
            from datetime import datetime, timezone

            now = datetime.now(timezone.utc).replace(tzinfo=None)
        count = 0
        for row in self.schedule.get("leads") or ():
            due = row.get("expected_at")
            try:
                due = None if due is None else _naive_time(due)
            except ValueError:
                due = None
            if due is not None and due > now:
                break
            count += 1
        return count

    def prepare_independent_times(self) -> bool:
        """Whether an already posted window may build its times in parallel.

        A live window stays ordered while any lead still needs a wait.
        Otherwise every lead's marker is read here, on the caller's thread,
        before workers are priced, so the segment markers are fixed; the
        leads themselves are decoded by the builds that read them, a
        bounded batch at a time under the decode lock (:meth:`through`),
        so the first boundary time is ready after one batch rather than
        after the whole window.  Snapshot packing still uses the
        preparation's snapshot_lock.
        """

        records = {lead: self.posted.marker(lead) for lead in self.leads[self._next:]}
        if any(record is None for record in records.values()):
            return False
        with self._decode_lock:
            for lead, record in records.items():
                if lead not in self.markers:
                    self._read_marker(lead, record)
        return True

    def _needs_first_lead(self, leads) -> bool:
        """Whether a batch without the window's first lead decodes it again.

        A route that concatenates the cycle's step-0 objects into the
        window's first lead's file only (roles composed into no other
        lead's file) carries cycle-invariant fields there, and the decode
        binds such a field from the earliest time that carries it to every
        other time, naming that record as each frame's source.  A batch
        without that lead decoded its times without them: a 3-lead GDPS
        window as posted was refused at f003 with "mapped frame at
        2026-10-01 03:00:00 lacks required fields ['land_fraction']".
        Reading only those parts filled the arrays but named another file
        as their source, so the frame headers and the composition receipt
        differed from one decode of the window.  Decoding the first lead's
        file again binds the same records the whole window binds; the
        batch keeps only its own times (:meth:`_decode_batch`).
        """

        first = self.leads[0]
        if first in leads:
            return False
        roles = {self.role_of.get(path)
                 for lead in leads for path in self.parts_of.get(lead, ())}
        return any(self.role_of.get(path) not in roles
                   for path in self.parts_of.get(first, ()))

    def _batch_inventory(self, leads):
        primary = tuple(path for path in self.primary
                        if self._in_batch(path, leads))
        chosen = {path.resolve() for path in primary}
        whole_primary = {path.resolve() for path in self.primary}
        supplements = {}
        for role, paths in self.supplements.items():
            if {path.resolve() for path in paths} <= whole_primary:
                # Drawn from the primary inventory (the source's own files
                # carry its surface: all of them, or one file per lead such
                # as RRFS's 2dfld): it follows the batch.  Read whole, a
                # live RRFS window was refused at its first batch, because
                # the later leads' 2dfld files had not posted.
                supplements[role] = tuple(
                    path for path in paths if path.resolve() in chosen)
                continue
            late = [str(path) for path in paths if not self._available(path)]
            if late:
                raise ValueError(
                    f"supplement role {role!r} is read whole by every lead "
                    f"batch, and {late} belong to leads not posted yet; the "
                    "route fetches such a supplement with its first lead "
                    "group, so this window cannot be decoded as it posts")
            supplements[role] = paths
        return primary, supplements

    def _decode_batch(self, start: int, end: int) -> None:
        from woof.mapped_authoring import author_input_manifest

        leads = self.leads[start:end + 1]
        # The first lead decoded again, ahead of the batch's own times, where
        # its file alone carries the cycle's step-0 statics.
        again = 1 if self._needs_first_lead(set(leads)) else 0
        primary, supplements = self._batch_inventory(
            {*leads, *self.leads[:again]})
        member_verification = None
        if getattr(self, "physical_members", False):
            from woof.prep_handoff import verify_posted_member_batch
            member_verification = verify_posted_member_batch(
                self.fetch_root, leads=(*self.leads[:again], *leads),
                primary_files=primary, trajectory=self.physical_trajectory())
        started = time.perf_counter()
        self._scratch.mkdir(parents=True, exist_ok=True)
        authored = author_input_manifest(
            self._scratch / f"batch-{len(self.batches):03d}.json",
            mapping_path=self.mapping, composition_path=self.composition,
            primary_files=primary, supplement_files=supplements,
            provenance_files=self.provenance,
            grib1_bridge=self.decoders.get("grib1_bridge"),
            grib2_inventory=self.decoders.get("grib2_inventory"),
            grib2_dump=self.decoders.get("grib2_dump"),
            contributing_mappings=self.contributing,
            expected_format=self.source_format)
        manifest_path = Path(authored["manifest"]["path"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        from woof.ingest.boundary_stream import composition_data_rows
        for row in composition_data_rows(manifest):
            path = (manifest_path.parent / str(row["path"])).resolve()
            if path in self.expected and (
                    row.get("bytes"), row.get("sha256")) != self.expected[path]:
                raise ValueError(
                    f"{path} is not the object its lead's posted marker named "
                    f"({self.expected[path]}); it read {row.get('bytes')} "
                    f"bytes, {row.get('sha256')}")
        bundle = decode_composed_source(
            self.composition, self.mapping, primary, supplements,
            self.provenance, input_manifest=manifest_path,
            input_manifest_sha256=str(authored["manifest"]["sha256"]),
            contributing_mappings=self.contributing,
            grib1_bridge=self.decoders.get("grib1_bridge"),
            grib2_inventory=self.decoders.get("grib2_inventory"),
            grib2_dump=self.decoders.get("grib2_dump"),
            scratch_destination=self.output_root,
            atmospheric_grids=self.grids,
            workers=self.workers,
            # A batch may be one lead; the window's series rules are held
            # over the planned times (prepare_mapped_wrf) and each batch's
            # times are held to them below.
            lead_batch=True,
            **({} if self.decode_memory_budget_bytes is None else {
                "memory_budget_bytes": self.decode_memory_budget_bytes}),
        )
        frames = bundle.frames
        decoded = tuple(getattr(frames, "valid_times", None)
                        or (frame.valid_time for frame in frames))
        if decoded != (self.valid_times[:again]
                       + self.valid_times[start:end + 1]):
            bundle.close()
            raise ValueError(
                f"the lead batch {', '.join(f'f{lead:03d}' for lead in leads)} "
                f"decoded valid times {[str(value) for value in decoded]}, not "
                "the ones the posting schedule names for those leads")
        for local, index in enumerate(range(start, end + 1)):
            self._where[index] = (frames, local + again)
        if again:
            from woof.mapped_composition import without_earlier_batch_times

            # The first lead's time is the first batch's: one decode of the
            # window lists it once in every alignment time list.
            self.carried_any = True
            bundle = without_earlier_batch_times(bundle, self.batches[0])
        self.frames.add_part(frames)
        self.batches.append(bundle)
        self.batch_leads.append(tuple(leads))
        self.decode_seconds += time.perf_counter() - started
        # Bind each physical knot to the actual ordinary batch decode. The
        # batch receipt is complete even while later raw leads are absent.
        from woof.ingest.boundary_stream import posted_lead_marker_sha256
        evidence = {"input_manifest_sha256": str(authored["manifest"]["sha256"]),
                    "bundle": bundle,
                    "member_verification": member_verification,
                    "primary_rows": tuple(((manifest_path.parent / row["path"]).resolve(),
                                           row["sha256"]) for row in manifest["primary_files"]),
                    "posted_leads": {str(lead): posted_lead_marker_sha256(self.markers[lead])
                                     for lead in (*self.leads[:again], *leads)}}
        if getattr(self, "physical_members", False):
            evidence["verified_trajectory"] = self.physical_trajectory().identity
        for index in range(start, end + 1):
            self.__dict__.setdefault("_physical_evidence", {})[index] = evidence

    def _locate(self, index: int):
        self.through(int(index))
        return self._where[int(index)]

    @property
    def decoded_count(self) -> int:
        return self._next

    def future_decode_host_bytes(self) -> int | None:
        """One native batch's priced peak while a forecast holds the head.

        A later batch may use the entire input grid even when the target
        is small. Its native inventory price, not target array geometry,
        reserves the child. The same bytes become its whole-process cap.
        """
        if self._next >= len(self.leads):
            remaining = [getattr(batch, "future_decode_host_bytes", lambda: 0)()
                         for batch in self.batches]
            return None if None in remaining else max(remaining, default=0)
        prices = [getattr(batch, "native_decode_memory", None) for batch in self.batches]
        if not prices or any(not isinstance(price, Mapping)
                             or not price.get("budget_supported")
                             or not isinstance(price.get("one_time_peak_bytes"), int)
                             or price["one_time_peak_bytes"] <= 0 for price in prices):
            return None
        return max(int(price["one_time_peak_bytes"]) for price in prices)

    def limit_future_decode(self, budget_bytes: int) -> None:
        self.decode_memory_budget_bytes = budget_bytes
        for batch in self.batches:
            limit = getattr(batch, "limit_future_decode", None)
            if limit is not None:
                limit(budget_bytes)

    def regular_snapshots(self):
        from woof.mapped_composition import _RegularSnapshots

        return _RegularSnapshots(SimpleNamespace(
            frames=self.frames,
            soil_layer_contract=self.batches[0].soil_layer_contract))

    def physical_trajectory(self):
        """The byte-verified route's canonical source, cycle and member."""
        from datetime import datetime, timedelta, timezone
        from woof.ensemble.recipes import SourceTrajectory
        cycle = datetime.fromisoformat(str(self.schedule["cycle"]).replace("Z", "+00:00"))
        if cycle.tzinfo is None:
            cycle = cycle.replace(tzinfo=timezone.utc)
        trajectory = SourceTrajectory(str(self.schedule["source"]), cycle,
                                      self.schedule.get("member"))
        for lead, marker in self.markers.items():
            recorded = SourceTrajectory(str(marker["source"]),
                _naive_time(marker["cycle"]).replace(tzinfo=timezone.utc),
                marker.get("member"))
            position = self.leads.index(lead)
            if (recorded != trajectory
                    or _naive_time(marker["valid_time"]) != self.valid_times[position]
                    or self.valid_times[position] != _naive_time(cycle) + timedelta(hours=lead)):
                raise ValueError("mapped physical lead differs from its posted source trajectory")
        return trajectory

    def physical_evidence(self, index):
        self.through(index)
        trajectory = self.physical_trajectory()
        from woof.ingest.boundary_stream import posted_lead_marker_sha256
        evidence = self._physical_evidence[index]
        raw = evidence["posted_leads"]
        if any(posted_lead_marker_sha256(self.markers[int(lead)]) != digest
               for lead, digest in raw.items()):
            raise ValueError("mapped physical raw marker changed after its batch was decoded")
        if _naive_time(self.frames.header(index).source_cycle) != _naive_time(trajectory.cycle):
            raise ValueError("mapped physical decoded source cycle differs from the posted trajectory")
        if evidence.get("verified_trajectory") != trajectory.identity:
            from woof.forcing_member import member_contract
            from woof.member_prep import verify_member_file
            contract = member_contract(trajectory.source, trajectory.member)
            for path, digest in evidence["primary_rows"]:
                if _sha256(path) != digest:
                    raise ValueError("mapped physical primary bytes changed after their batch decode")
                if contract is not None:
                    _adapter, grammar, member = contract
                    verify_member_file(grammar, member, path)
                    if _sha256(path) != digest:
                        raise ValueError("mapped physical member bytes changed during native verification")
            evidence["verified_trajectory"] = trajectory.identity
        decoded_digest = hashlib.sha256(_canonical({
            "input_manifest_sha256": evidence["input_manifest_sha256"],
            "composition_receipt_sha256": composition_receipt_identity_sha256(
                mapped_composition_receipt(evidence["bundle"])),
            "member_verification": evidence.get("member_verification"),
        }).encode()).hexdigest()
        return {"posted_leads": raw, "decoded_leads": {str(self.leads[index]): decoded_digest}}

    # -- the seal --------------------------------------------------------

    def finish(self):
        """Author the window's manifest and return ``(path, digest, bundle)``."""

        from woof.mapped_authoring import author_input_manifest
        from woof.mapped_composition import posted_composition_bundle

        self.through(len(self.leads) - 1)
        authored = author_input_manifest(
            self.input_manifest, mapping_path=self.mapping,
            composition_path=self.composition, primary_files=self.primary,
            supplement_files=self.supplements,
            provenance_files=self.provenance,
            grib1_bridge=self.decoders.get("grib1_bridge"),
            grib2_inventory=self.decoders.get("grib2_inventory"),
            grib2_dump=self.decoders.get("grib2_dump"),
            # Every batch and the one-shot author name the contributing
            # mappings, which add a donor's format to the decoder rows; the
            # window's manifest must seal the same rows.
            contributing_mappings=self.contributing,
            expected_format=self.source_format)
        path = Path(authored["manifest"]["path"])
        digest = str(authored["manifest"]["sha256"])
        if path != self.input_manifest:
            raise ValueError(
                f"the as-posted seal's input manifest landed at {path}, not "
                f"at {self.input_manifest}, because a manifest sealed with "
                "other mapping, composition or decoder bytes is there")
        shared = (self.carried_any
                  or any(owner is None for owner in self.lead_of.values()))
        bundle = posted_composition_bundle(
            self.batches, self.frames, input_manifest_path=path,
            input_manifest_sha256=digest,
            supplement_files=self.supplements, shared_primary=shared)
        return path, digest, bundle

    def close(self) -> None:
        try:
            self.frames.close()
        finally:
            shutil.rmtree(self._scratch, ignore_errors=True)


def _posted_input_plan(posted_source, *, mapping, composition, primary,
                       supplements, provenance, decoders) -> dict:
    """The input plan an as-posted mapped head binds (DESIGN A136 2.4 item 5).

    The manifest the seal will write with the one-shot author, every
    planned lead object's size and digest left out
    (:func:`woof.mapped_authoring.planned_input_manifest`); the rows of
    the inputs the route does not plan (a donor analysis) are read whole
    now and bound as they are (``fixed_rows``).  Also binds the route
    table the fetch planned its leads under.
    """

    from woof.ingest.boundary_stream import (
        as_posted_placeholder, input_plan, input_plan_sha256)
    from woof.mapped_authoring import (
        manifest_row_path, planned_input_manifest)

    data = (*primary, *(path for paths in supplements.values()
                        for path in paths))
    manifest = planned_input_manifest(
        posted_source.input_manifest, mapping_path=mapping,
        composition_path=composition, primary_files=primary,
        supplement_files=supplements, provenance_files=provenance,
        planned=[path for path in data
                 if path.resolve() in posted_source.planned],
        grib1_bridge=decoders.get("grib1_bridge"),
        grib2_inventory=decoders.get("grib2_inventory"),
        grib2_dump=decoders.get("grib2_dump"),
        contributing_mappings=posted_source.contributing)
    fixed_rows = sorted({manifest_row_path(path, posted_source.input_manifest)
                         for path in posted_source.fixed})
    route_table_sha256 = posted_source.posted.route_table_sha256()
    plan = input_plan(manifest, lead_role_prefix="",
                      route_table_sha256=route_table_sha256,
                      fixed_rows=fixed_rows)
    return {"plan": plan, "fixed_rows": fixed_rows,
            "route_table_sha256": route_table_sha256,
            "placeholder": as_posted_placeholder(input_plan_sha256(plan))}


def _require_source_top(exp, cfg, source_top_pressure_pa: float,
                        experiment_config) -> None:
    try:
        validate_explicit_eta_grid(
            exp.vertical.eta_levels,
            nz=cfg.nz,
            p_top=exp.vertical.p_top,
            source_top_pressure_pa=source_top_pressure_pa,
            context="mapped direct adapter",
        )
    except ValueError as error:
        # The ladder's shape and values were validated at the door;
        # what this re-check adds is the SOURCE TOP, so the only
        # user-reachable raise left here is a model top the staged
        # atmosphere does not reach.
        raise VerticalLadderRefusal(
            str(error),
            remedy=(
                "remedy: lower the model top -- raise p_top in the "
                f"experiment config {experiment_config} to a "
                "pressure the staged source atmosphere reaches -- "
                "or prepare from a source whose levels reach this "
                "top (`woof prep --show-source NAME` prints each "
                "source's vertical coverage)."),
        ) from error


def _seal_posted_mapped(posted_source, *, writer, plan, mapping_contract,
                        snapshots, exp, cfg, source_identity,
                        static_cache_sha256, namelist_sha256,
                        forcing_identity, experiment_config=None,
                        physical_provider_seal=None,
                        identity_builder=None) -> dict:
    """What an as-posted mapped seal writes (DESIGN A136 2.4 item 6).

    The window's input manifest through the one-shot author, refused
    unless it is the plan the head bound and every lead row is the object
    its lead's marker named; the composition receipt and vertical ladder
    of every lead batch read as one decode; the whole window's source top
    checked as the head checked its first leads; the one-shot identity,
    which the writer holds to the head's (only the manifest and receipt
    digests move); and the record of the leads consumed.

    A hierarchy supplies its ordinary root artifact identity builder. The
    same source checks then precede sealing the root's streamed cache.
    """

    from woof.ingest.boundary_stream import (
        hold_composition_rows, input_plan)

    path, digest, whole = posted_source.finish()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    # Re-read at the seal: leads fetched under another route table than
    # the one the head planned with are another plan.
    route_table_sha256 = posted_source.posted.route_table_sha256()
    if input_plan(manifest, lead_role_prefix="",
                  route_table_sha256=route_table_sha256,
                  fixed_rows=plan["fixed_rows"]) != plan["plan"]:
        raise ValueError(
            f"the input manifest the seal wrote at {path} is not the input "
            "plan the head bound: a file, a non-lead input or the fetch's "
            "route table differs")
    hold_composition_rows(
        manifest, {str(lead): marker
                   for lead, marker in posted_source.markers.items()},
        fixed_rows=plan["fixed_rows"])
    receipt = mapped_composition_receipt(whole)
    ladder = decoded_vertical_ladder(whole, mapping_contract)
    if ladder is not None:
        _announce_vertical_ladder(ladder)
    _require_source_top(exp, cfg, _source_top_pressure_pa(
        snapshots, mapping_contract=mapping_contract),
                        experiment_config)
    sealed_source_identity = {
        **source_identity,
        "input_manifest_sha256": digest,
        "composition_receipt_sha256": composition_receipt_identity_sha256(
            receipt),
    }
    if identity_builder is None:
        identity = prepared_cache_identity(
            bridge_manifest_sha256=digest,
            source_manifest_sha256=digest,
            static_cache_sha256=static_cache_sha256,
            namelist_sha256=namelist_sha256,
            domain_config=exp.root,
            **forcing_identity,
            source_identity=sealed_source_identity,
        )
    else:
        identity = identity_builder(sealed_source_identity, digest)
    _copy_bound_authority(
        path, writer.root / "source-evidence" / "input-manifest.json", digest)
    writer.write_posted_leads(posted_source.markers,
                              route_table_sha256=route_table_sha256)
    cache_receipt = dict(writer.seal_cache(identity=identity,
        manifest_sha256=digest,
        **({"physical_provider_seal": physical_provider_seal}
           if physical_provider_seal is not None else {})))
    return {
        "cache_receipt": cache_receipt,
        "identity": identity,
        "bundle": whole,
        "source_identity": sealed_source_identity,
        "manifest_sha256": digest,
        "proof": {
            "source_composition": receipt,
            **({"source_vertical_ladder": ladder}
               if ladder is not None else {}),
            "posting": {
                "as_posted": True,
                "waits": list(posted_source.posted.waits),
                "leads_late": [],
            },
        },
    }


@preprocess_math_call
class _EarlyInitialDecode:
    """The separate initial analysis, decoded beside the boundary source.

    ``decode_initial_analysis`` reads only the declared analysis files, the
    packaged authorities, the grids and the start time, none of which the
    boundary source's decode or its start time's build touches.  On the
    literal HRRR start it took 45.6 s of the 234 s root stage, one after the
    other with the 137 s boundary decode while the card sat idle.  It now
    starts as soon as the grids are known, in its own engine process, and the
    root stage enters the same context it would have entered, so every array
    and receipt it hands on is the one the serial decode produced.
    """

    def __init__(self, request, **arguments):
        from contextlib import ExitStack
        self._stack = ExitStack()
        self._analysis = None
        self._error = None
        self._thread = threading.Thread(
            target=self._run, args=(request, arguments),
            name="initial-analysis-decode", daemon=True)
        self._thread.start()

    def _run(self, request, arguments):
        from woof.initial_source import decode_initial_analysis
        try:
            self._analysis = self._stack.enter_context(
                decode_initial_analysis(request, **arguments))
        except BaseException as error:  # re-raised on the root stage's thread
            self._error = error

    def close(self):
        self._thread.join()
        self._stack.close()

    @contextlib.contextmanager
    def analysis(self):
        self._thread.join()
        try:
            if self._error is not None:
                raise self._error
            yield self._analysis
        finally:
            self._stack.close()


def prepare_mapped_wrf(
    *,
    composition: str | Path,
    mapping: str | Path,
    primary_files: Sequence[str | Path],
    supplement_files: Mapping[
        str, str | Path | Sequence[str | Path]
    ],
    provenance_files: Mapping[str, str | Path],
    input_manifest: str | Path,
    input_manifest_sha256: str,
    contributing_mappings: Mapping[str, str | Path] | None = None,
    grib1_bridge: str | Path | None = None,
    grib2_inventory: str | Path | None = None,
    grib2_dump: str | Path | None = None,
    wps_namelist: str | Path,
    geog_root: str | Path,
    static_input: str | Path | None = None,
    static_receipt: str | Path | None = None,
    experiment_config: str | Path,
    output_root: str | Path,
    preprocess_backend: str = "auto",
    preprocess_workers: int | None = None,
    cpu_preprocess_bridge: str | Path | None = None,
    hierarchy_workers: int | None = None,
    stock_wrf_export: str = "optional",
    statics_corridor=None,
    as_posted: str | Path | None = None,
    initial_inputs: str | Path | None = None,
    _source_manifest: str | Path | None = None,
    _source_manifest_sha256: str | None = None,
    _source_adapter: str = "rw-wps-mapped-composition-v2",
    physical_input_store: str | Path | None = None,
    physical_output_store: str | Path | None = None,
    physical_input_provider=None,
    physical_member_index: int | None = None,
    inland_air_mean_hours: float | None = None,
) -> dict[str, object]:
    """Build native WRF inputs from one complete mapped-source composition.

    All mixed-product semantics are resolved before target interpolation.
    The root owns the complete external-LBC sequence; mapped child inputs are
    initialized independently and finalized parent-before-child.  Products
    are published atomically only after source and run-control authorities are
    reverified.

    ``stock_wrf_export`` defaults to an optional companion WRF file set.
    ``required`` refuses unsupported configurations before ingest; ``off``
    prepares only the native forecast artifacts. Only the named unsupported
    export condition is recoverable; I/O and corrupt-array failures still fail.

    ``static_input``/``static_receipt`` supply a previously published
    native-static NPZ instead of rebuilding the root domain's geography from
    WPS_GEOG, exactly as the ERA5/GFS/HRRR adapters already accept.  The
    receipt binds the target geometry contract and the NPZ SHA-256, and the
    loader re-derives MAPFAC/F/E/SINALPHA/COSALPHA from the grid, so a cache
    that does not belong to this domain is refused rather than trusted.
    ``geog_root`` remains required either way: the proof's geog_datasets
    binding and any child-domain statics are still resolved from the tree.

    ``statics_corridor`` opts this preparation into emitting the sealed
    child-resolution statics corridors
    (:func:`woof.static.corridor.emit_statics_corridor_set`), exactly as
    the GFS and HRRR hierarchy stages already do.  It is what a RELOCATING
    nest needs and cannot rebuild for itself: a prepared bundle runs with
    no GEOG source, so the ground a nest has not visited yet has to be
    sealed at preparation time or the move has nothing to stand on.
    ``woof.prepared_domain_tree_forecast`` refuses a configured
    ``[relocation]`` against a bundle that carries no corridor for the
    mover, by name -- so a mapped tree that means to move MUST pass this.
    ``None`` selects nothing, ``"all"`` every child, and a sequence
    exactly those child grid ids; the resolution itself belongs to
    :func:`woof.static.corridor.validated_corridor_selection`, which the
    hierarchy call applies, so this route cannot resolve a bare flag to a
    different set than `run-plan --estimate` priced.

    ``_source_manifest``/``_source_manifest_sha256`` are the seam for a
    NAMED-SOURCE route whose user-facing sealed authority is its own
    manifest schema (the 20CRv3 member manifest, with its filename-bound
    member identity): the route verifies its own document, authors the
    generic composition-inputs manifest FROM it, and hands both here.
    The composition decodes against the bridged manifest -- which the
    composition receipt seals -- while the prepared tree's evidence copy,
    identity chain and proof stay bound to the route's own document, the
    one its forecast leg re-verifies with the route's own reader.  Absent,
    the two manifests are the same document and nothing changes.

    ``as_posted`` is an as-posted fetch's ``posting/`` folder (DESIGN A136
    2.4): the preparation starts on the window's first lead and decodes the
    later leads batch by batch as their markers appear
    (:class:`_PostedMappedSource`), and ``input_manifest`` is where its
    seal writes the window's manifest (beside the fetched files) with the
    one-shot author, so ``input_manifest_sha256`` is ``None``.  The head
    binds the input plan and the first leads' markers; the seal writes the
    composition receipt, the vertical ladder and the one-shot identity, and
    the prepared arrays are the one-shot preparation's byte for byte.
    """

    if stock_wrf_export not in STOCK_WRF_EXPORT_MODES:
        raise ValueError(f"stock_wrf_export must be one of {STOCK_WRF_EXPORT_MODES}")
    started = time.perf_counter()
    try:
        from woof.boundary_prestage import current_boundary_prestage
    except ImportError:
        # The standalone preparation package carries no standby door, so no
        # prestage can be current there (tools/build_rw_wps_release.py).
        boundary_prestage = None
    else:
        boundary_prestage = current_boundary_prestage()
    composition = Path(composition).resolve()
    mapping = Path(mapping).resolve()
    primary = tuple(Path(path).resolve() for path in primary_files)
    supplements = {
        str(role): _path_inventory(path, f"supplement {role!r}")
        for role, path in supplement_files.items()
    }
    provenance = {
        str(role): Path(path).resolve() for role, path in provenance_files.items()
    }
    contributing = {
        str(role): Path(path).resolve()
        for role, path in (contributing_mappings or {}).items()
    }
    input_manifest = Path(input_manifest).resolve()
    if as_posted is not None:
        if input_manifest_sha256 is not None:
            raise ValueError(
                "an as-posted preparation writes its input manifest at the "
                "seal, so it takes no manifest digest")
        if _source_manifest is not None:
            raise ValueError(
                "an as-posted preparation seals the composition inputs "
                "manifest itself; a named-source route's own manifest is "
                "verified whole and cannot be written lead by lead")
    if (_source_manifest is None) != (_source_manifest_sha256 is None):
        raise ValueError(
            "source manifest and source manifest SHA are an atomic pair"
        )
    if _source_manifest is None:
        source_manifest_path: Path | None = None
        source_manifest_sha256: str | None = None
    else:
        source_manifest_path = Path(_source_manifest).resolve()
        source_manifest_sha256 = str(_source_manifest_sha256).lower()
        if not source_manifest_path.is_file():
            raise RunInputRefusal(
                f"mapped run input is missing: --source-manifest names "
                f"{source_manifest_path}, which does not exist")
        if _sha256(source_manifest_path) != source_manifest_sha256:
            raise ValueError(
                "source manifest bytes differ from the declared SHA-256, so "
                "the identity chain would be sealed to a document nobody "
                "verified"
            )
    mapping_snapshot = _snapshot_authority(mapping, retain_bytes=True)
    mapping_contract = load_mapping(
        mapping,
        _raw=_load_json_bytes(mapping_snapshot.data, "mapping", mapping),
    )
    physical_provider = None
    if physical_input_provider is not None:
        from woof.ensemble.posted_physical import PostedPhysicalProvider
        if as_posted is None:
            raise ValueError("posted physical providers require an as-posted preparation")
        if physical_input_store is not None or physical_output_store is not None:
            raise ValueError("a posted physical consumer cannot also capture or read a complete physical store")
        if type(physical_member_index) is not int or physical_member_index < 0:
            raise ValueError("a posted physical provider requires the original member index")
        physical_provider = (physical_input_provider if isinstance(physical_input_provider, PostedPhysicalProvider)
                             else PostedPhysicalProvider.open(physical_input_provider,
                                  cpu_bridge=cpu_preprocess_bridge, workers=preprocess_workers or 1))
    elif physical_member_index is not None:
        raise ValueError("physical_member_index requires a posted physical provider")
    # The decoder role set is the union across the primary's format and
    # every contributing source's format.  Only the format key is probed
    # here; ``decode_composed_source`` fully validates each contributing
    # mapping against the composition's pinned hash before decoding.
    decode_formats = {str(mapping_contract["format"])}
    for role, contributing_path in contributing.items():
        raw = _load_json_document(
            contributing_path, f"contributing mapping {role!r}",
        )
        observed_format = raw.get("format") if isinstance(raw, dict) else None
        if observed_format in ("grib1", "grib2", "netcdf"):
            decode_formats.add(str(observed_format))
    engine_binary = None
    # This route ALWAYS composes, and every contributing format has to be
    # one the engine can read: a union with one unported format in it is
    # a Python-engine job whole, because half a composition decoded by
    # each engine would be one frameset with two provenances.
    if physical_provider is None and all(
        _mapped_engine_choice(
            grib1_bridge=grib1_bridge,
            grib2_inventory=grib2_inventory,
            grib2_dump=grib2_dump,
            subcommand=_ROUTE_SUBCOMMAND,
            source_format=source_format,
        ) == _ENGINE_RUST
        for source_format in sorted(decode_formats)
    ):
        # Resolved HERE, before any output directory exists, so an
        # unstaged engine costs a refusal at the door instead of a
        # half-built preparation: this is the same pre-flight the
        # subprocess decoders get below.
        from woof.mapped_engine_bridge import require_engine

        engine_binary = require_engine()
    try:
        decoders = {} if physical_provider is not None else _decoder_inventory(
            sorted(decode_formats),
            grib1_bridge=grib1_bridge,
            grib2_inventory=grib2_inventory,
            grib2_dump=grib2_dump,
            engine=engine_binary,
        )
    except ValueError as error:
        # Delivered, not demoted: the contract still refuses and still
        # names the same breakage.  What changes is that a user reading
        # it meets a remedy instead of nineteen frames of this package's
        # internals, which is what they met on a bare default prep of
        # every composed source until 2.5.0.
        present = {
            role for role, path in (
                ("grib1_bridge", grib1_bridge),
                ("grib2_inventory", grib2_inventory),
                ("grib2_dump", grib2_dump),
            ) if path is not None
        }
        required = set() if engine_binary is not None else {
            role for name in decode_formats
            for role in _FORMAT_TOOL_ROLES[str(name)]
        }
        raise _decoder_inventory_refusal(
            str(error), decode_formats,
            missing=required - present, extra=present - required,
        ) from error
    wps_namelist = Path(wps_namelist).resolve()
    geog_root = Path(geog_root).resolve()
    experiment_config = Path(experiment_config).resolve()
    output_root = Path(output_root).resolve()
    cpu_bridge = (
        None if cpu_preprocess_bridge is None
        else Path(cpu_preprocess_bridge).resolve()
    )
    if (static_input is None) != (static_receipt is None):
        raise ValueError(
            "mapped static-input and static-receipt must be supplied together"
        )
    prebuilt_static = (
        None if static_input is None else Path(static_input).resolve())
    prebuilt_receipt = (
        None if static_receipt is None else Path(static_receipt).resolve())
    # Every file the caller named, checked under the FLAG that named
    # it and reported together: the old anonymous loop relayed the
    # first miss as a bare ``FileNotFoundError`` holding only a path,
    # which is how a pasted prep command with one wrong working
    # directory answered a user with a traceback (UX finding N6).
    # As posted, the manifest is the seal's to write and the later leads'
    # files arrive as they post; each lead batch checks its own inputs.
    data_labels = () if as_posted is not None else (
        ("--input-manifest", input_manifest),
        *(("--input/--input-list", path) for path in primary),
        *((f"--supplement {role}", path)
          for role, paths in supplements.items() for path in paths),
    )
    labeled = [
        ("--composition", composition),
        ("--mapping", mapping),
        ("--wps-namelist", wps_namelist),
        ("--experiment-config", experiment_config),
        *((f"decoder tool {role}", path) for role, path in decoders.items()),
        *data_labels,
        *((f"--provenance {role}", path)
          for role, path in provenance.items()),
        *((f"--contributing-mapping {role}", path)
          for role, path in contributing.items()),
        *(() if prebuilt_static is None
          else (("--static-input", prebuilt_static),
                ("--static-receipt", prebuilt_receipt))),
        *(() if cpu_bridge is None
          else (("--cpu-preprocess-bridge", cpu_bridge),)),
    ]
    missing = [
        (flag, path) for flag, path in labeled if not path.is_file()
    ]
    if missing:
        listing = "; ".join(
            f"{flag} names {path}, which does not exist"
            for flag, path in missing)
        remedy = RunInputRefusal.remedy
        if any(flag == "--experiment-config" for flag, _path in missing):
            remedy += (
                "  For --experiment-config: `woof import-namelist WPS "
                "INPUT --output FILE.toml` translates an existing WRF "
                "namelist pair into one, and `woof domain` authors one "
                "from scratch.")
        raise RunInputRefusal(
            f"mapped run inputs are missing: {listing}", remedy=remedy)
    if not geog_root.is_dir():
        raise RunInputRefusal(
            f"--geog-root names {geog_root}, which is not a directory",
            remedy=(
                "remedy: point --geog-root at a staged WPS_GEOG tree.  "
                "`woof fetch-geog` stages one (~16 GB) and prints the "
                "root to pass."))
    # A head published early whose producer failed, was stopped or went
    # silent is this tool's own unfinished product, not a finished run.
    remove_unfinished_tree(output_root)
    refusal = existing_output_root_refusal(output_root)
    if refusal is not None:
        raise refusal
    run_control_before = {
        "wps_namelist": _file_receipt(wps_namelist),
        "experiment_config": _file_receipt(experiment_config),
    }
    if cpu_bridge is not None:
        run_control_before["cpu_preprocess_bridge"] = _file_receipt(cpu_bridge)

    from woof.ingest.preparation_workers import configure_preparation_workers
    worker_policy = configure_preparation_workers(preprocess_workers)
    exp = load_experiment(experiment_config)
    # THE FLOOR, BEFORE THE DECODE (A98).  The backend is resolved here,
    # weighed against the lower bound the domains alone set: an explicit
    # cuda they cannot fit is refused in seconds instead of after the
    # decode and the statics, and auto moves to the CPU here.  Resolving
    # allocates nothing on the card; the decoded price below stays the
    # binding check.
    preprocess = resolve_preprocess_backend(
        preprocess_backend, workers=preprocess_workers,
        cpu_bridge=cpu_bridge,
        price=lambda: price_preparation_floor("mapped", exp),
    )
    from woof.static.highres_production import (
        load_static_highres, apply_prepared_highres, static_highres_identity)
    static_highres = load_static_highres(experiment_config)
    from woof.case_data import optional_case_data_from_config, preparation_case_policy
    case_data = optional_case_data_from_config(experiment_config)
    case_policy = preparation_case_policy(case_data)
    from woof.ingest.water_overlay import (
        load_bound_water_overlay, overlay_snapshot_sequence, verify_overlay_sequence)
    water_overlay, water_overlay_binding = load_bound_water_overlay(
        None if case_data is None else case_data.water_temperature_overlay)
    from woof.experiment import (
        deferred_initial_perturbation, dormant_domain_ids,
    )
    # A dormant (spawn-declared) nest is prepared at its declared placeholder
    # like any t = 0 child -- that prices it and gives the prepared tree its
    # bundle -- and the corridor reaches the whole parent frame for it
    # (woof.static.corridor.corridor_reach), which is the ground the
    # prepared tree route crops a newborn from.  A single-domain bundle has
    # no tree route to spawn on, so a dormant slot there still refuses.
    if dormant_domain_ids(exp) and len(exp.domains) < 2:
        from woof.experiment import refuse_unrouted_spawn
        refuse_unrouted_spawn(exp, "mapped-adapter single-domain prepared-cache")
    initial_perturbation = deferred_initial_perturbation(
        exp, "mapped-adapter prepared-cache")
    hierarchy = len(exp.domains) > 1
    initial_request = None
    if initial_inputs is not None or boundary_prestage is not None:
        # The independent start must be applied to every child before a tree
        # can use it. Refuse a tree here instead of silently starting children
        # from the boundary product while their parent uses the analysis.
        if hierarchy:
            raise ValueError("separate initial inputs currently require a complete "
                             "single-domain forcing window; a tree needs the "
                             "analysis propagated into each child's initial state")
        if as_posted is not None:
            # Named breakage (2.9.4 rolling, HRRR recipe with --initial-inputs on
            # the rap-native route): this refusal used to name a tree that was
            # not there.  The as-posted head binds forcing leads as they post,
            # so the window the separate start is checked against is not whole.
            raise ValueError("separate initial inputs need the complete forcing window "
                             "before the start, and this run fetches its forcing as it "
                             "posts; run with --whole-cycle to fetch the whole window "
                             "first")
        if physical_input_store is not None or physical_input_provider is not None:
            # A physical member store replaces every knot's interpolated
            # state, the start included, so the separate analysis would be
            # read and then silently discarded.
            raise ValueError("separate initial inputs cannot be combined with a "
                             "physical member input; the member store already "
                             "carries the initial state")
        from woof.initial_source import read_initial_inputs
        initial_request = (boundary_prestage if boundary_prestage is not None
                           else read_initial_inputs(initial_inputs))
        if boundary_prestage is not None and physical_output_store is not None:
            raise ValueError("resident boundary prestage cannot capture a physical "
                             "member store while suspending its initial analysis")
    # A domain tree is published through staging deeper than its
    # output root; refused here, the first point this door knows it
    # prepares a tree, before any source is decoded.  A single-domain
    # bundle publishes no tree and is not measured against it.
    if hierarchy:
        refusal = published_path_refusal(
            output_root, wrf_export=stock_wrf_export != "off")
        if refusal is not None:
            raise ValueError(refusal)
    # Refused HERE rather than left to the hierarchy call, because the
    # hierarchy call is the thing that does not happen on a single-domain
    # preparation: without this the flag would be accepted, silently emit
    # nothing, and the missing corridor would surface hours later as the
    # forecast runner's relocation refusal.  Same guard, same reason, as
    # `woof.gfs_direct` (statics_corridor / len(exp.domains) < 2).
    if statics_corridor is not None and not hierarchy:
        raise ValueError(
            "--statics-corridor prepares child-resolution statics over the "
            "ground a child domain can reach, and this experiment has no "
            "child domain; "
            "remove the flag or prepare a domain tree")
    # Plan review: the run's own spacing where it is known before the
    # decode, never a finer one the run does not use (A173).
    review_spacing, review_origin = _plan_review_spacing(
        mapping_contract["target"], wps_namelist)
    target_contract = _validate_target_contract(
        mapping_contract,
        exp,
        review_spacing,
        hierarchy=hierarchy,
        experiment_config=experiment_config,
        spacing_origin=review_origin,
        before_decode=True,
    )
    cfg = exp.root.run

    # A physical source replacement enters before native real initialization,
    # at both the initial state and every future boundary knot.
    physical_input = physical_output = None
    if physical_input_store is not None or physical_input_provider is not None:
        from woof.ensemble.posted_preparation import require_exclusive_native_consumer
        require_exclusive_native_consumer(physical_input_store=physical_input_store,
                                         physical_input_provider=physical_input_provider)
    if physical_provider is not None:
        if hierarchy:
            raise ValueError("posted physical providers require a single-domain as-posted preparation")
    if physical_input_store is not None or physical_output_store is not None:
        from woof.ensemble.physical_store import NativePhysicalStore
        if hierarchy or physical_input_store is not None and as_posted is not None:
            raise ValueError(
                "sealed physical stores currently describe a complete single-domain "
                "forcing window; a hierarchy or an unsealed as-posted head needs "
                "domain-local stores and its own final source identity")
        if physical_input_store is not None:
            physical_input = NativePhysicalStore(physical_input_store)
    physics_selection = None
    if not hierarchy:
        from woof.physics_compat import (
            acknowledgement_delivery, single_domain_physics_selection,
        )

        acknowledgements, ack_provenance = acknowledgement_delivery(
            toml=getattr(exp, "acknowledgements", ()))
        physics_selection = single_domain_physics_selection(
            cfg, expert_acknowledgements=acknowledgements,
            acknowledgement_provenance=ack_provenance)
    if stock_wrf_export == "required":
        with prep_stage("export_config", label="Validate requested WRF export"):
            # A requested WRF product must be representable before expensive
            # decode/initialization. The exporter reuses this same authority.
            if hierarchy:
                validate_stock_wrf_export_hierarchy(exp)
            for domain in exp.domains:
                validate_stock_wrf_export_config(
                    domain.run, root=domain.grid_id == 1,
                    configured_suite=not hierarchy,
                    label=("direct-export" if not hierarchy
                           else f"d{domain.grid_id:02d} direct-export"))
    if hierarchy:
        grids = validate_native_lambert_contracts(
            exp, wps_namelist, source_name="mapped source",
        )
        grid = grids[0]
    else:
        grid = validate_native_lambert_contract(
            exp, wps_namelist, source_name="mapped source",
        )
        grids = (grid,)
    selection = GeogSelection.from_case_data(
        SimpleNamespace(wps_namelist=wps_namelist, geog_root=geog_root,
                        static_highres=static_highres), 1,
    )

    posted_source = None
    early_initial = None
    if initial_request is not None and boundary_prestage is None:
        early_initial = _EarlyInitialDecode(
            initial_request, output_parent=output_root.parent, grids=grids,
            valid_time=exp.start_time, workers=preprocess_workers,
            decoders=decoders)
        import atexit
        atexit.register(early_initial.close)
    with prep_stage("root_initialize" if physical_provider is not None else "source_decode",
                    label=("Initialize member from shared source" if physical_provider is not None
                           else "Decode and compose source"),
                    backend=(str(preprocess.receipt()["backend"]) if physical_provider is not None
                             else "rust" if engine_binary is not None else "python")):
        decode_started = time.perf_counter()
        if as_posted is not None:
            posted_source = _PostedMappedSource(
                posting=as_posted, input_manifest=input_manifest,
                composition=composition, mapping=mapping, primary=primary,
                supplements=supplements, provenance=provenance,
                contributing=contributing, decoders=decoders, grids=grids,
                workers=preprocess_workers, output_root=output_root,
                source_format=str(mapping_contract["format"]),
                **({"physical_members": True} if physical_output_store is not None
                   or physical_provider is not None else {}))
            if physical_provider is not None:
                # The pinned ordinary source has already decoded, mapped and
                # initialized its base. Reconstruct its acquisition plan here
                # without reading future raw files or repeating that work.
                posted_source._read_plan()
                from woof.ensemble.mapped_posted_reuse import (
                    prepare_posted_mapped_member, shared_input_plan)
                shared_plan = shared_input_plan(physical_provider.source_context(physical_member_index), posted_source,
                    mapping=mapping, composition=composition, primary=primary,
                    supplements=supplements, provenance=provenance, contributing=contributing)
                return prepare_posted_mapped_member(physical_provider, physical_member_index,
                    posted_plan=shared_plan, mapping_contract=mapping_contract,
                    mapping=mapping, composition=composition,
                    experiment_config=experiment_config, wps_namelist=wps_namelist,
                    output_root=output_root, preprocess=preprocess,
                    preprocess_workers=preprocess_workers, run_control_before=run_control_before,
                    stock_wrf_export=stock_wrf_export, physics_selection=physics_selection,
                    source_adapter=_source_adapter, case_policy=case_policy,
                    water_overlay_binding=water_overlay_binding,
                    static_input=prebuilt_static, static_receipt=prebuilt_receipt)
            try:
                bundle = posted_source.start()
            except BaseException:
                posted_source.close()
                raise
        else:
            bundle = decode_composed_source(
                composition, mapping, primary, supplements, provenance,
                input_manifest=input_manifest,
                input_manifest_sha256=input_manifest_sha256,
                contributing_mappings=contributing,
                grib1_bridge=decoders.get("grib1_bridge"),
                grib2_inventory=decoders.get("grib2_inventory"),
                grib2_dump=decoders.get("grib2_dump"),
                # The engine's compose scratch holds the whole composed frame
                # stream; naming this run's destination keeps that stream on the
                # output's disk-backed filesystem instead of a tmpfs system temp
                # with a quota smaller than the biggest registered sources.
                scratch_destination=output_root,
                atmospheric_grids=grids,
                # An explicit --preprocess-workers is the count every host
                # step takes, the source decode included; without one the
                # engine uses every core it may run on.  Either way the
                # engine narrows to what free memory holds.
                workers=preprocess_workers,
            )
    decode_seconds = time.perf_counter() - decode_started
    if bundle.mapping_sha256 != mapping_snapshot.sha256:
        raise ValueError(
            "mapped contract changed between target validation and decode"
        )
    _require_authority_snapshot(mapping_snapshot)
    if dict(bundle.decoder_paths) != decoders:
        raise ValueError("decoded decoder paths differ from requested decoders")
    if posted_source is None:
        composition_receipt = mapped_composition_receipt(bundle)
        receipt_identity_sha256 = composition_receipt_identity_sha256(
            composition_receipt)
        vertical_ladder = decoded_vertical_ladder(bundle, mapping_contract)
        if vertical_ladder is not None:
            _announce_vertical_ladder(vertical_ladder)
        derived_terrain = dict(
            composition_receipt.get("alignment") or {}).get("derived")
        if derived_terrain is not None:
            _announce_derived_terrain(derived_terrain)
        snapshots = _forcing_series(bundle)
    else:
        # The receipt, its alignment and the vertical ladder read every
        # lead: the seal writes them from every batch (_AS_POSTED_SEAL_KEYS).
        composition_receipt = receipt_identity_sha256 = vertical_ladder = None
        if dict(bundle.alignment_receipt).get("derived") is not None:
            posted_source.close()
            raise ValueError(
                "this source's terrain is derived from the window's first "
                "valid time (the mapping's when_absent), so a later lead "
                "batch would derive its own; it is prepared whole, not as "
                "posted")
        snapshots = posted_source.regular_snapshots().sorted_by_valid_time()
    for_grids = getattr(snapshots, "for_grids", None)
    if for_grids is not None:
        snapshots = for_grids(grids)
    snapshots = _within_run_window(snapshots, exp.start_time, exp.run_seconds)
    # The backend resolves later; an explicit --preprocess-workers is the
    # count every host step takes, and without one the automatic count.
    snapshots = overlay_snapshot_sequence(snapshots, water_overlay,
                                          binding=water_overlay_binding,
                                          workers=preprocess_workers)
    times = _forcing_valid_times(snapshots)
    if len(times) < 2:
        raise ForcingSeriesRefusal(
            "mapped forecast preparation requires at least two forcing times")
    if not times or times[0] != exp.start_time:
        raise ValueError(
            f"mapped forcing must begin at {exp.start_time}, got {times[:1]}"
        )
    forcing_offsets = tuple((value - times[0]).total_seconds() for value in times)
    if any(not value.is_integer() for value in forcing_offsets):
        raise ValueError("mapped forcing times must use whole-second offsets")
    forcing_seconds = tuple(int(value) for value in forcing_offsets)
    if any(value < 0 for value in forcing_seconds) \
            or forcing_seconds[-1] < exp.run_seconds:
        raise ValueError("mapped forcing does not cover the configured run")
    deltas = {
        later - earlier
        for earlier, later in zip(forcing_seconds, forcing_seconds[1:])
    }
    if len(deltas) != 1 or next(iter(deltas), 0) <= 0:
        raise ValueError("mapped forcing cadence must be positive and uniform")
    boundary_interval_seconds = deltas.pop()
    target_contract = _validate_target_contract(
        mapping_contract,
        exp,
        boundary_interval_seconds,
        hierarchy=hierarchy,
        experiment_config=experiment_config,
    )
    # As posted, the head reads the leads decoded so far and the seal
    # checks the whole window again (_check_posted_top).
    source_top_pressure_pa = _source_top_pressure_pa(
        snapshots, count=None if posted_source is None
        else posted_source.decoded_count, mapping_contract=mapping_contract)
    if hierarchy:
        grids = validate_native_lambert_contracts(
            exp,
            wps_namelist,
            source_name="mapped source",
            source_top_pressure_pa=source_top_pressure_pa,
        )
        grid = grids[0]
    else:
        _require_source_top(exp, cfg, source_top_pressure_pa,
                            experiment_config)

    with prep_stage("root_static", label="Prepare root static fields"):
        static_started = time.perf_counter()
        if prebuilt_static is None:
            # WRF's topo_wind / gwd_opt read sub-grid orographic statistics
            # geogrid writes; the build adds them only when either is on.
            from woof.static.orographic import with_terrain_drag_statics
            static = build_static(
                grid, geog_root,
                selection=with_terrain_drag_statics(selection, cfg))
            root_static_provider = "native-wps-geog"
            # The terrain-smoothing root seam reads this attestation.
            from woof.static.terrain_smoothing import smoothing_receipt
            smoothing = smoothing_receipt(static_highres)
            # The static-source root seam reads this attestation too.
            from woof.static.external_source import static_source_receipt
            static_source = static_source_receipt(static_highres)
            root_static_receipt = ({
                **({"terrain_smoothing": smoothing} if smoothing else {}),
                **({"static_source": static_source} if static_source else {}),
            } or None)
        else:
            # The receipt binds native_geometry_contract(grid, cfg) AND the NPZ
            # SHA-256; the loader then re-derives the geometry fields from the
            # grid and refuses a stored copy that disagrees.  A cache built for
            # another domain cannot survive either check.
            root_static_receipt = verify_native_static_receipt(
                prebuilt_receipt, prebuilt_static, grid, cfg)
            static = load_native_static_cache(
                prebuilt_static, grid, cfg.ny, cfg.nx)
            root_static_provider = "prebuilt-hash-bound-cache"
        static, root_static_receipt = apply_prepared_highres(
            static, grid, config=static_highres, domain_id=1,
            case_date=exp.start_time.date(),
            landuse_attrs=(selection.landuse_global_attrs()
                           if static_highres is not None and static_highres.enabled else None),
            baseline_receipt=root_static_receipt, run=cfg)
    static_seconds = time.perf_counter() - static_started

    # THE COORDINATE, BEFORE ANYTHING IS BUILT ON IT.  The same call the
    # other source doors make, in the same place: root terrain in hand,
    # nothing yet built on a vertical coordinate.
    exp, vertical_adaptation = adapt_experiment_for_statics(
        exp, grids, root_terrain=static["HGT_M"],
        static_catalog=_survey_static_catalog(
            exp, wps_namelist, geog_root, static_highres),
        static_highres=static_highres, announce=_announce_adaptation,
        statics_corridor=statics_corridor)
    cfg = exp.root.run

    # Priced from the decoded composition, before the first device
    # allocation: a large CUDA build stages bounded batches while completed
    # arrays stay on the host. A card that cannot hold one batch retains the
    # named CPU fallback or refusal. A
    # backend already on the CPU is not priced.  The card is read again
    # here: the decode took minutes and the card may be shared.
    preprocess = admit_preparation(
        preprocess,
        lambda: price_forcing_preparation(
            "mapped", exp, snapshots,
            boundary_species=mapping_boundary_species(mapping_contract)),
        workers=preprocess_workers)
    # The start time is built before the head exists, so no forecast sits
    # beside it: it takes the whole card when the build fits there, and
    # its result moves to the host before the head (start_time_backend).
    start_preprocess = start_time_backend(preprocess)
    preprocess_receipt = preprocess.receipt()
    posted_plan = (None if posted_source is None else _posted_input_plan(
        posted_source, mapping=mapping, composition=composition,
        primary=primary, supplements=supplements,
        provenance=provenance, decoders=decoders))
    # CHAINED TREES.  A tree's children need only the start time and the
    # root's initial state, so a tree is chained exactly like a single
    # domain, on either backend: the start time first, the children into
    # the head, one root interval per segment, the one-shot tree at the
    # seal.  No start state stays resident between head and seal (the seal
    # re-reads each from the head, boundary_stream.TreeStartStates), so a
    # later forcing time is built on the card with nothing under it, as in
    # the start-last order the unchained tree keeps.
    chain_tree = hierarchy and (posted_source is not None or chained_enabled())
    if getattr(exp, 'nudging', None) is not None:
        if posted_source is not None:
            raise ValueError('NUDGING_UNSEALED_ANALYSIS_IDENTITY: an as-posted input plan changes its source digest at seal; prepare the complete source window before nudging')
        chain_tree = False  # selected nests need analyses from the complete hierarchy join
    # The same declared policy reaches the root and every child catalog.
    water_statics = WaterTemperatureStatics.for_route(
        route=_WATER_ROUTE, policy=case_policy["water_temperature_policy"],
        landmask=static["LANDMASK"], lu_index=static["LU_INDEX"],
        landuse_attrs=selection.landuse_global_attrs())
    # Inland water the source does not resolve takes the forcing's
    # daily-mean 2 m air temperature (WRF use_tavg_for_tsk).
    water_statics = with_inland_air_temperature(
        water_statics, snapshots, hours=inland_air_mean_hours)
    with prep_stage("root_initialize", label="Initialize root forcing states",
                    backend=str(preprocess_receipt["backend"]),
                    count=(len(snapshots) if hierarchy and not chain_tree
                           else 1)):
        initialize_started = time.perf_counter()
        # ONE forcing time is ever resident.  A single domain builds the
        # start time FIRST, writes it into the published head and releases
        # it, then builds each later time and writes each interval as soon
        # as its two times exist (woof.ingest.boundary_stream).  A chained
        # hierarchy does the same (chain_tree, above) and its seal re-reads
        # the start states from the head.  An unchained hierarchy builds
        # the start time LAST (start_last_forcing_order) and
        # keeps only that met/state; every other time contributes its
        # perimeter frames against its own position and is released before
        # the next one is interpolated.  Walking the times in order instead
        # meant holding the start time -- which nothing reads until the
        # boundaries are complete -- while each later time was built
        # underneath it.  At 800x800x49
        # with mp=10 and three GFS times that second resident time is 14.67
        # GiB of device residency against 7.66, a priced peak envelope of
        # 23.92 GiB against 15.86: the difference between preparing the
        # domain on a 16 GiB card and OOMing after the whole forcing chain
        # had already been fetched.
        initial_result = None
        initial_met = None
        initial_source_receipt = None
        initial_source_evidence = None
        initial_soil_contract = bundle.soil_layer_contract
        initial_soil_mesh = None
        forcing = StateBoundaryFrames(
            spec_bdy_width=cfg.spec_bdy_width,
            spec_zone=cfg.spec_zone, relax_zone=cfg.relax_zone)
        coord = make_vertical_coord(
            cfg.nz, hybrid_opt=cfg.hybrid_opt, etac=cfg.etac,
            eta_levels=exp.vertical.eta_levels,
        )
        mapfac_m, mapfac_u, mapfac_v = (
            grid.mapfac_m(), grid.mapfac_u(), grid.mapfac_v(),
        )
        coriolis_f, coriolis_e = grid.coriolis_m()
        rotation_sin, rotation_cos = grid.rotation_m()
        # The hydrometeor masses this mapping declares are decoded from
        # every frame, so the root's specified boundary carries them
        # (woof.boundary_fields); a mapping that declares none keeps
        # water vapour only.
        boundary_species = mapping_boundary_species(mapping_contract)
        # WRF's smooth_cg_topo (woof.ingest.cg_topo): the root terrain is
        # blended toward the source's once, before the first
        # initialization reads it.  Off, this does nothing.
        terrain_blend = RootTerrainBlend(exp, static, route="mapped")
        if physical_output_store is not None or physical_input is not None:
            from woof.native_wrf_contract import native_geometry_contract
            from woof.ensemble.mapped_physical_contract import (
                mapped_physical_field_contract, require_mapped_physical_field_contract)
            physical_source_identity = {
                    "adapter": _source_adapter,
                    "mapping_sha256": bundle.mapping_sha256,
                    "composition_sha256": bundle.composition_sha256,
                    "input_manifest_sha256": (posted_plan["placeholder"] if posted_plan is not None
                                              else source_manifest_sha256 or bundle.input_manifest_sha256),
                    "composition_receipt_sha256": (posted_plan["placeholder"] if posted_plan is not None
                                                   else receipt_identity_sha256),
                    "preparation_case_policy": case_policy,
                    "water_temperature_overlay": water_overlay_binding,
                    **({"static_highres": static_highres_identity(static_highres)}
                       if static_highres is not None else {})}
            physical_geometry = native_geometry_contract(grid, cfg)
            expected_physical_fields = mapped_physical_field_contract(
                physical_geometry, mapping_path=mapping, composition_path=composition,
                source_identity=physical_source_identity,
                extra_evidence={("raw_input_plan" if posted_plan is not None else "raw_source_manifest"):
                                (input_plan_sha256(posted_plan["plan"]) if posted_plan is not None
                                 else physical_source_identity["input_manifest_sha256"]),
                                **{"native_decoder_"+key: value for key, value in bundle.decoder_sha256.items()}})
            if physical_output_store is not None and posted_source is None:
                physical_output = NativePhysicalStore(
                    physical_output_store, grid_identity=physical_geometry,
                    source_identity=physical_source_identity, field_contract=expected_physical_fields)
        if physical_input is not None:
            from woof.native_wrf_contract import native_geometry_contract
            if physical_input.document["grid"] != native_geometry_contract(grid, cfg):
                raise ValueError("physical input geometry differs from native preparation geometry")
            if tuple(physical_input.times) != tuple(times):
                raise ValueError("physical input valid times differ from the full initial/boundary window")
            require_mapped_physical_field_contract(physical_input, expected_physical_fields)
        forcing_timings = []
        # Forcing times may be built on several threads (stream_forcing_times
        # workers); the lazy snapshot sequence keeps a one-deep cache and its
        # frames read engine scratch, so a time's source is packed one at a
        # time and only the interpolation and initialization run side by side.
        snapshot_lock = threading.Lock()
        nudging_writer = None
        if getattr(exp, "nudging", None) is not None:
            from woof.spectral_nudging.prepared import analysis_writer
            nudging_writer = analysis_writer(output_root, exp, times, source_manifest_sha256)

        def build_forcing_time(index, *, source_override=None, aerosol_snapshot=None,
                               backend=None):
            nonlocal physical_output
            backend = preprocess if backend is None else backend
            # One forcing time's build, unchanged: interpolate, initialize,
            # attach the map factors.  The single-domain route calls it start
            # first (woof.ingest.boundary_stream), as does a chained
            # hierarchy; an unchained hierarchy keeps the start time last.
            build_started = time.perf_counter()
            if source_override is None:
                from woof.ingest.boundary_stream import concurrent_forcing_build
                pack = getattr(snapshots, "pack_uncached", None)
                if pack is not None and concurrent_forcing_build():
                    # Beside other builds each packs its own copy, the same
                    # snapshot the cached read returns. Named breakage: 17
                    # forcing-time workers ran at ~3 cores, every one waiting
                    # here for the next time's pack (Oklahoma City, 2026-10-23).
                    source = pack(index)
                else:
                    with snapshot_lock:
                        source = snapshots[index]
            else:
                source = source_override
            timing = {"forcing_index": index, "valid_time": str(source.valid_time)}
            horizontal_started = time.perf_counter()
            # Metgrid classifies masked-field TARGET cells by the model
            # (geogrid) landmask; the mapped lane declares it like the ERA5
            # lanes so soil, skin, snow, and physics share one surface.
            if physical_input is None:
                met = interpolate_era5_to_lambert(
                    source, grid, backend=backend,
                    target_landmask=np.asarray(static["LANDMASK"]) >= 0.5,
                    water_temperature_statics=water_statics)
            else:
                if source_override is not None:
                    # The member's captured fields would silently replace
                    # the separately decoded initial analysis.
                    raise ValueError("member physical input cannot replace a separate "
                                     "initial analysis source")
                met = physical_input.read(index)
            timing["horizontal_seconds"] = time.perf_counter() - horizontal_started
            column_timing = {}
            terrain_blend.before_initialize(
                met.fields.get("SOURCE_OROGRAPHY"))
            if physical_output_store is not None:
                from woof.ensemble.physical_store import physical_static_identity
                from woof.native_wrf_contract import native_landuse_identity
                actual_static = physical_static_identity(
                    native_static_export_fields(static, grid),
                    native_landuse_identity(static))
            if physical_output_store is not None and posted_source is not None:
                from datetime import timezone
                from woof.ensemble.posted_physical import PostedPhysicalStream, posted_source_identity
                if physical_output is None:
                    physical_output = PostedPhysicalStream.create(
                        physical_output_store, trajectory=posted_source.physical_trajectory(),
                        valid_times=tuple(value.replace(tzinfo=timezone.utc) for value in times),
                        grid_identity=physical_geometry,
                        source_identity=posted_source_identity(
                            {**physical_source_identity, "static_identity": actual_static},
                            input_plan=posted_plan["plan"]),
                        field_contract=expected_physical_fields,
                        input_plan_sha256=input_plan_sha256(posted_plan["plan"]))
                physical_output.publish(met, **posted_source.physical_evidence(index))
            elif physical_output is not None:
                if index == 0:
                    from woof.ensemble.physical_store import physical_static_identity
                    from woof.native_wrf_contract import native_landuse_identity
                    physical_output.document["source"]["static_identity"] = physical_static_identity(
                        native_static_export_fields(static, grid),
                    native_landuse_identity(static))
                physical_output.write(met)
            from woof.ensemble.posted_preparation import (
                current_posted_preparation, replace_current_native_snapshot)
            if current_posted_preparation() is not None:
                member_plan = (None if posted_source is None else _posted_input_plan(
                    posted_source, mapping=mapping, composition=composition,
                    primary=primary, supplements=supplements, provenance=provenance,
                    decoders=decoders))
                met = replace_current_native_snapshot(met, grid=grid, cfg=cfg,
                    static_fields=static, landuse_attrs=selection.landuse_global_attrs(),
                    domain_id=exp.root.grid_id, metadata={
                        "bundle": bundle, "mapping_path": mapping, "composition_path": composition,
                        "source_manifest_sha256": source_manifest_sha256,
                        "input_manifest": input_manifest, "posted_plan": member_plan,
                        "source_snapshot": source, "source_adapter": _source_adapter,
                        "preprocessing": preprocess_identity(preprocess_receipt),
                        "preparation_case_policy": case_policy,
                        "water_temperature_overlay": water_overlay_binding})
            initialized = initialize_real(
                met, cfg, coord, static["HGT_M"], grid=grid,
                landmask=static["LANDMASK"],
                p_top=exp.vertical.p_top, sfcp_to_sfcp=case_policy["sfcp_to_sfcp"],
                preprocess_backend=backend,
                state_backend="preprocess",
                # Only the start time's result is kept below; every later
                # time contributes its state to the boundaries.
                boundary_only=(index != 0 or (
                    initial_request is not None and source_override is None)),
                boundary_species=boundary_species,
                timing_report=column_timing,
                **({"aerosol_snapshot": aerosol_snapshot}
                   if aerosol_snapshot is not None else {}),
                # THE FRONT DOOR for the mp=28 aerosol default is the "grid="
                # above (lane/static-dataset-door), and it is the SAME door the
                # other ten real routes now use.
                #
                # lane/wif-default wired this one route by EVALUATING both
                # runtime inputs here -- wif_grid_latlon=grid.latlon_mass() and
                # wif_valid_date=source.valid_time.isoformat() -- and doing it
                # unconditionally, on the argument that both were already to
                # hand.  Two things were wrong with that, and the merge removes
                # it rather than carrying both mechanisms:
                #
                #  * It is EAGER.  grid.latlon_mass() was called on every mapped
                #    preparation regardless of mp_physics, so any grid object
                #    without that method -- the mapped hierarchy tests' own
                #    _Grid among them -- died with an AttributeError on a line
                #    no configuration in that test had selected.  Nine tests in
                #    tests/test_mapped_direct.py were red on it.
                #  * It is PER-ROUTE.  Repeating it at eleven call sites is
                #    eleven chances to spell the derivation differently.
                #
                # initialize_real now derives both itself, lazily, inside the
                # mp=28 climatology branch and only there: the valid date from
                # snapshot.valid_time (the same value source.valid_time
                # produced) and the lat/lon from the "grid=" carrier.  Nothing
                # is lost -- this route still reaches the climatology by
                # default -- and the explicit keywords remain available as
                # overrides for a caller with a reason to disagree.
            )
            initialized.state.set_map_coriolis(
                mapfac_m, mapfac_u, mapfac_v, coriolis_f, coriolis_e,
                sina=rotation_sin, cosa=rotation_cos,
            )
            timing["column_stages_seconds"] = column_timing
            timing["total_seconds"] = time.perf_counter() - build_started
            forcing_timings.append(timing)
            if nudging_writer is not None:
                nudging_writer.write(index, initialized)
            return met, initialized

        if hierarchy and not chain_tree:
            for index in start_last_forcing_order(len(snapshots)):
                met, initialized = build_forcing_time(index)
                forcing.add_state(initialized.state, index=index)
                if index == 0:
                    initial_met = met
                    initial_result = initialized
                else:
                    del met, initialized
                    release_backend_memory(preprocess)
            boundaries = forcing.build(times)
            attach_lateral_boundaries(initial_result.state, boundaries)
        else:
            # START FIRST.  The start time makes the head (state, surface,
            # static fields, receipts); the later times are built after it is
            # published, one resident at a time, and each interval is written
            # as soon as its two times exist.
            if initial_request is None and start_preprocess is not preprocess:
                initial_met, initial_result = build_forcing_time(
                    0, backend=start_preprocess)
                moved = move_start_result_to_host(initial_met, initial_result)
                release_backend_memory(start_preprocess)
                preprocess.selection["chunking"]["start_time"]["moved_to_host_bytes"] = moved
            else:
                initial_met, initial_result = build_forcing_time(0)
            start_preprocess = None
            forcing.add_state(initial_result.state, index=0)
            if initial_request is not None:
                from woof.initial_source import decode_initial_analysis
                boundary_head_backend = preprocess
                boundary_head_preprocessing = copy.deepcopy(preprocess.receipt())
                # Only the donor's aerosol and its coordinate stay held for
                # the initial analysis' build; the rest of the boundary
                # analysis is released here.
                aerosol_met = (aerosol_donor_snapshot(initial_met)
                               if cfg.use_rap_aero_icbc else None)
                initial_met = None
                del initial_result
                release_backend_memory(preprocess)
                if boundary_prestage is not None:
                    from woof.ingest.preparation_workers import forcing_time_workers
                    prestage_workers = forcing_time_workers(
                        len(snapshots) - 1,
                        cpu_workers=worker_policy.get("effective_workers"),
                        backend=str(preprocess_receipt["backend"]),
                        selection=preprocess_receipt.get("selection"))
                    worker_policy["forcing_time_workers"] = prestage_workers
                    boundary_prestage.receipt["stages"].extend([
                        {"name": "decode_and_compose", "wall_seconds": decode_seconds,
                         "parallelism": "Rust worker pool"},
                        {"name": "static", "wall_seconds": static_seconds,
                         "parallelism": str(preprocess_receipt["backend"])},
                    ])
                    initial_request = boundary_prestage.prepare_independent(
                        forcing=forcing, times=times, build=build_forcing_time,
                        release=lambda: release_backend_memory(preprocess),
                        workers=prestage_workers["workers"], output_root=output_root)
                try:
                    analysis_started = time.perf_counter()
                    with (early_initial.analysis() if early_initial is not None
                          else decode_initial_analysis(
                            initial_request, output_parent=output_root.parent,
                            grids=grids, valid_time=exp.start_time,
                            workers=preprocess_workers, decoders=decoders)) as analysis:
                        if boundary_prestage is not None:
                            boundary_prestage.receipt["stages"].append({
                                "name": "initial_analysis_decode",
                                "wall_seconds": time.perf_counter() - analysis_started,
                                "parallelism": "Rust worker pool"})
                        _require_source_top(
                            exp, cfg, _source_top_pressure_pa(
                                (analysis.snapshot,), mapping_contract=analysis.mapping),
                            experiment_config)
                        # Re-read free memory while the aerosol donor is still
                        # held, and price the second source before mapping it.
                        preprocess = admit_preparation(
                            preprocess,
                            lambda: price_forcing_preparation(
                                "mapped", exp, (analysis.snapshot,),
                                boundary_species=boundary_species),
                            workers=preprocess_workers)
                        preprocess_receipt = preprocess.receipt()
                        initial_met, initial_result = build_forcing_time(
                            0, source_override=analysis.snapshot,
                            aerosol_snapshot=aerosol_met)
                        # The first boundary record is the start state's
                        # own coupling, its tendency running to the next
                        # boundary-source record: WRFDA da_update_bc's rule,
                        # which operational HRRR runs after GSI.  It was the
                        # boundary source's atmosphere at the start time.
                        forcing.replace_start_state(
                            initial_result.state, index=0)
                        if boundary_prestage is not None:
                            boundary_prestage.complete(forcing, times)
                            boundary_prestage.receipt["forcing_stage_timings"] = forcing_timings
                        initial_soil_contract = analysis.soil_layer_contract
                        initial_soil_mesh = soil_mesh_plan_from_case(
                            analysis.snapshot, grid, experiment_config)
                        initial_source_receipt = {
                            **analysis.receipt,
                            "boundary_head_preprocessing": boundary_head_preprocessing,
                            "aerosol_source": ("boundary-analysis" if aerosol_met is not None
                                               else "initial-analysis"),
                        }
                        initial_source_evidence = analysis.evidence
                finally:
                    aerosol_met = None
                    release_backend_memory(boundary_head_backend)
            boundaries = None
        # The chem processes' preparation-time arrays (dust statics, sulfur
        # lat/lon), before the head is published; nothing on a chem-off
        # state (woof/core/chem_statics.py).
        from woof.core.chem_statics import attach_chem_statics
        attach_chem_statics(initial_result.state, grid, geog_root, selection)
        # No lake skin override: the masked=both SKINTEMP chain with
        # static-landmask targets already yields water-source skin at lakes,
        # matching real.exe's no-TAVGSFC behavior, except inland water with no
        # source water within INLAND_WATER_SOURCE_REACH_M, which takes the
        # daily-mean 2 m air temperature (use_tavg_for_tsk).  The static landmask drives
        # WRF's process_soil_real land/water branches, and terrain plus the
        # composition's canonical source orography enable adjust_soil_temp_new's
        # elevation lapse on skin and soil temperature inputs.  The router
        # forwards this exact argument list to preprocess_noah_soil for
        # Noah-geometry schemes.
        # The reconciled category, on the one rulebook every door follows
        # (woof/ingest/soil.py: door_reconciled_soil_category); the raw
        # SCT_DOM let a land column carry the water soil category into RUC
        # (ENG-009).
        from woof.core.landuse import (
            ruc_fractional_seaice as _ruc_fractional_seaice,
            seaice_threshold_of as _seaice_threshold_of)
        soil = preprocess_land_surface_soil(
            initial_met.fields,
            # real.exe's adjust_for_seaice_pre/post keep the fraction under
            # fractional_seaice = 1 (threshold 0.02) and snap to 0/1 at 0.5
            # otherwise (module_soil_pre.F:216-219, :337-343, :392-393 of the HRRR
            # v4.1.21 fork).
            fractional_seaice=_ruc_fractional_seaice(cfg),
            seaice_threshold=_seaice_threshold_of(cfg),
            sf_surface_physics=int(cfg.sf_surface_physics),
            # Resolved, not defaulted: see woof/ingest/hrrr_physics.py.
            num_soil_layers=soil_layer_count(cfg),
            soil_type=door_reconciled_soil_category(
                static, initial_met.fields, selection.landuse_global_attrs(),
                route="mapped source"),
            deep_soil_temperature=static["TMN"],
            soil_layer_contract=initial_soil_contract,
            landmask=static["LANDMASK"],
            # Land the source holds no land for takes the column the
            # router builds (woof/ingest/soil.py: island_soil_columns).
            soil_no_source_land=getattr(
                initial_met, "soil_no_source_land", None),
            terrain=static["HGT_M"],
            source_orography=initial_met.fields["SOURCE_OROGRAPHY"],
            water_temperature=getattr(initial_met, "water_temperature", None),
            water_temperature_policy=water_statics.policy,
            # Same seam as the elevation lapse above, for moisture and for the
            # deep temperature: a mapped source's mesh can be coarser than this
            # grid, and where it is, the soil state gets the target grid's own
            # soil texture instead of the source cell's average.
            soil_mesh=(initial_soil_mesh if initial_request is not None
                       else soil_mesh_plan_from_case(
                           snapshots[0], grid, experiment_config)),
            # Where and when, for a same-model soil donor (woof/ingest/
            # land_state.py: HRRR's RUC soil for a RUC run over CONUS).
            land_state=land_state_context(grid, times[0]),
            route=_WATER_ROUTE,
        )
        soil_temperature_repair = soil_temperature_repair_proof(soil, grid)
    initialize_seconds = time.perf_counter() - initialize_started

    # Keep atomic siblings short.  Repeating a user-supplied output name at
    # every nested transaction exceeded legacy Windows MAX_PATH before the
    # first hierarchy artifact could be written.
    staging = output_root.with_name(f".tmp-{uuid.uuid4().hex[:8]}")
    if staging.exists():
        raise FileExistsError(f"mapped staging output already exists: {staging}")
    staging.mkdir(parents=True)
    writer = None
    try:
        static_path = staging / "native-static.npz"
        geometry_path = staging / "geometry-receipt.json"
        prepared_path = staging / "prepared-cache"
        wrf_path = staging / "wrf-native-input"
        evidence_path = staging / "source-evidence"
        evidence_path.mkdir()
        if initial_source_evidence is not None:
            from woof.initial_source import write_initial_evidence
            write_initial_evidence(evidence_path, initial_source_evidence)
        # The USER-FACING manifest: the route's own sealed document when
        # the route bridged one in, otherwise the composition manifest
        # itself.  The evidence copy, the identity chain and the proof
        # bind this one -- it is what the forecast leg re-verifies with
        # the route's own reader -- while the bridged twin stays sealed
        # inside the composition receipt.
        published_manifest = (
            input_manifest if source_manifest_path is None
            else source_manifest_path)
        published_manifest_sha256 = (
            bundle.input_manifest_sha256 if source_manifest_sha256 is None
            else source_manifest_sha256)
        if posted_source is not None:
            # The manifest the seal will write, with every planned lead
            # object's size and digest not known yet; the head binds its
            # plan, and each digest the manifest carries is this plan's
            # placeholder until the seal (boundary_stream.input_plan).
            published_manifest_sha256 = receipt_identity_sha256 = (
                posted_plan["placeholder"])
        for source, name, digest in (
            (mapping, "mapping.json", bundle.mapping_sha256),
            (
                composition,
                "composition.json",
                bundle.composition_sha256,
            ),
            # As posted, the seal writes the manifest and copies it here.
            *(() if posted_source is not None else ((
                published_manifest,
                "input-manifest.json",
                published_manifest_sha256,
            ),)),
            # The bridged composition-inputs manifest travels TOO on a
            # named-source route: the composition receipt seals its
            # digest, and the forecast leg re-hashes this copy against
            # that seal -- without the bytes, the receipt's manifest
            # record would be a digest nothing can re-check once the raw
            # inputs move.
            *(
                ()
                if source_manifest_path is None
                else ((
                    input_manifest,
                    "composition-inputs.json",
                    bundle.input_manifest_sha256,
                ),)
            ),
        ):
            _copy_bound_authority(source, evidence_path / name, digest)
        provenance_authorities = _bound_provenance_authorities(
            bundle,
            _load_json_document(
                evidence_path / "composition.json", "published mapped composition"),
        )
        if set(provenance) != set(provenance_authorities):
            raise ValueError("decoded provenance role inventory differs from requested roles")
        for role, source in sorted(provenance.items()):
            expected_path, expected_digest = provenance_authorities[role]
            destination = evidence_path / _provenance_evidence_name(
                role, source.suffix,
            )
            if destination.exists():
                raise ValueError(
                    f"provenance roles collide after filename encoding: {role!r}"
                )
            if source != expected_path:
                raise ValueError(
                    f"decoded provenance path differs for role {role!r}"
                )
            _copy_bound_authority(
                source,
                destination,
                expected_digest,
            )
        static_output_receipt = write_native_static_cache(
            static_path, native_static_export_fields(static, grid),
        )
        geometry_receipt = write_native_geometry_receipt(
            geometry_path, grid, cfg, static_path,
            **({"terrain_autosmooth": root_static_receipt["terrain_autosmooth"]}
               if root_static_receipt and "terrain_autosmooth" in root_static_receipt
               else {}),
        )
        source_identity = {
            "adapter": _source_adapter,
            **({"static_highres": static_highres_identity(static_highres)}
               if static_highres is not None else {}),
            "mapping_sha256": bundle.mapping_sha256,
            "composition_sha256": bundle.composition_sha256,
            "input_manifest_sha256": published_manifest_sha256,
            # The receipt less where this run's files sat (A151), as the
            # preprocessing receipt is bound less what was measured (A138):
            # the proof keeps the whole receipt.
            "composition_receipt_sha256": receipt_identity_sha256,
            # What ran, without what was measured (A138): the proof keeps
            # the whole receipt.
            "preprocessing": preprocess_identity(preprocess_receipt),
            "preparation_case_policy": case_policy,
            "water_temperature_overlay": water_overlay_binding,
            **({"initial_source": initial_source_receipt}
               if initial_source_receipt is not None else {}),
        }
        if physical_input is not None:
            from woof.ensemble.physical_store import physical_input_binding, physical_static_identity
            from woof.native_wrf_contract import native_landuse_identity
            source_identity["ensemble_physical_input"] = physical_input_binding(
                physical_input, grid, cfg, source_identity,
                input_manifest_sha256=published_manifest_sha256,
                static_identity=physical_static_identity(
                    native_static_export_fields(static, grid),
                    native_landuse_identity(static)))
        from woof.ensemble.posted_preparation import bind_current_source_identity
        source_identity = bind_current_source_identity(source_identity,
            domain_id=exp.root.grid_id)
        forcing_identity = (
            {"forcing_hours": tuple(
                value // 3600 for value in forcing_seconds)}
            if all(value % 3600 == 0 for value in forcing_seconds)
            else {"forcing_offsets_seconds": forcing_seconds})
        forcing_key, forcing_axis = next(iter(forcing_identity.items()))
        if hierarchy and chain_tree:
            chained_tree = SimpleNamespace(
                exp=exp, grids=grids, cfg=cfg, snapshots=snapshots,
                times=times, forcing_seconds=forcing_seconds,
                forcing_identity=forcing_identity, forcing_key=forcing_key,
                forcing_axis=forcing_axis,
                boundary_interval_seconds=boundary_interval_seconds,
                target_contract=target_contract,
                vertical_adaptation=vertical_adaptation, soil=soil,
                soil_temperature_repair=soil_temperature_repair,
                initial_result=initial_result, initial_met=initial_met,
                forcing=forcing, build_forcing_time=build_forcing_time,
                preprocess=preprocess, preprocess_receipt=preprocess_receipt,
                static=static, static_output_receipt=static_output_receipt,
                geometry_receipt=geometry_receipt,
                root_static_provider=root_static_provider,
                root_static_receipt=root_static_receipt,
                bundle=bundle, composition_receipt=composition_receipt,
                vertical_ladder=vertical_ladder,
                source_identity=source_identity,
                published_manifest_sha256=published_manifest_sha256,
                staging=staging, output_root=output_root,
                experiment_config=experiment_config,
                wps_namelist=wps_namelist, geog_root=geog_root,
                cpu_bridge=cpu_bridge, hierarchy_workers=hierarchy_workers,
                statics_corridor=statics_corridor,
                static_highres=static_highres, case_policy=case_policy,
                stock_wrf_export=stock_wrf_export,
                run_control_before=run_control_before,
                static_seconds=static_seconds, decode_seconds=decode_seconds,
                initialize_seconds=initialize_seconds, started=started,
                verify_overlay_sequence=verify_overlay_sequence,
                initial_perturbation=initial_perturbation,
                mapping_contract=mapping_contract,
                posted_source=posted_source, posted_plan=posted_plan,
                worker_policy=worker_policy,
                # Physical stores need the later times in order, as on the
                # single-domain route; anything else builds them side by side.
                forcing_times_independent=(physical_input is None
                                           and physical_output is None
                                           and physical_output_store is None),
            )
            # The chained tree owns the start state from here and releases
            # it at its head; these names would keep it resident.
            del initial_result, initial_met
            return _prepare_chained_mapped_tree(chained_tree)
        if hierarchy:
            selected_workers = hierarchy_workers
            if selected_workers is None:
                selected_workers = (
                    8 if preprocess_receipt["backend"] == "cpu" else 1
                )
            hierarchy_started = time.perf_counter()
            hierarchy_result = initialize_and_export_regular_source_hierarchy(
                nudging_context=((output_root, build_forcing_time)
                    if getattr(exp, 'nudging', None) is not None else None),
                exp=exp,
                grids=grids,
                snapshots=snapshots,
                # A child sees the catalog, not the config.
                soil_texture_downscale=declared_soil_texture_downscale(
                    experiment_config),
                **forcing_identity,
                wps_namelist=wps_namelist,
                geog_root=geog_root,
                source_name="RW-WPS-MAPPED",
                artifact_output=staging / "hierarchy-artifacts",
                wrf_output=wrf_path,
                root_initial_result=initial_result,
                root_met=initial_met,
                root_soil=soil,
                root_static_fields=static,
                root_boundaries=boundaries,
                bridge_manifest_sha256=published_manifest_sha256,
                source_manifest_sha256=published_manifest_sha256,
                namelist_sha256=_sha256(experiment_config),
                source_identity={
                    **source_identity,
                    "target_contract": target_contract,
                },
                source_inventory=tuple(snapshots[0].fields),
                workers=selected_workers,
                preprocess_backend=preprocess_receipt["backend"],
                cpu_bridge=cpu_bridge,
                soil_layer_contract=bundle.soil_layer_contract,
                root_metadata={
                    "composition_receipt_sha256": receipt_identity_sha256,
                    "mapped_target_contract": target_contract,
                    **_soil_floor_receipts(soil),
                },
                input_provenance={
                    "mapping_sha256": bundle.mapping_sha256,
                    "composition_sha256": bundle.composition_sha256,
                    "input_manifest_sha256": published_manifest_sha256,
                    "decoder_sha256": dict(bundle.decoder_sha256),
                    "preprocessing": scientific_preprocessing(preprocess_receipt),
                    "mapped_target_contract": target_contract,
                },
                artifact_manifest_reference=(
                    "../hierarchy-artifacts/domain-artifacts.json"
                ),
                statics_corridor=statics_corridor,
                static_highres=static_highres,
                sfcp_to_sfcp=case_policy["sfcp_to_sfcp"],
                water_temperature_policy=case_policy["water_temperature_policy"],
                stock_wrf_export=stock_wrf_export,
            )
            hierarchy_seconds = time.perf_counter() - hierarchy_started
            verify_overlay_sequence(snapshots)
            run_control_after = {
                "wps_namelist": _file_receipt(wps_namelist),
                "experiment_config": _file_receipt(experiment_config),
            }
            if cpu_bridge is not None:
                run_control_after["cpu_preprocess_bridge"] = _file_receipt(
                    cpu_bridge
                )
            if run_control_after != run_control_before:
                raise ValueError(
                    "mapped run-control bytes changed during preparation"
                )
            proof = {
                "schema": HIERARCHY_PROOF_SCHEMA,
                "status": "READY_NOT_YET_STOCK_WRF_GATED",
                "stock_wrf_export": stock_wrf_export,
                "domain_count": len(exp.domains),
                "vertical_coordinate": _vertical_coordinate_receipt(
                    exp, vertical_adaptation),
                "forcing_times": [value.isoformat() for value in times],
                # The soil-state SOURCE resolution and whether the
                # sub-source-cell reconstitution ran on it.
                "soil_texture_downscale": dict(
                    getattr(soil, "soil_texture_downscale", {}) or {}),
                # Present only when a soil temperature rebuild (real.exe's
                # band, or the snow-covered rule beside it) touched a land
                # column of the root.
                **_soil_floor_receipts(soil),
                **({"soil_temperature_repair": soil_temperature_repair}
                   if soil_temperature_repair is not None else {}),
                forcing_key: list(forcing_axis),
                "boundary_interval_seconds": boundary_interval_seconds,
                "target_contract": target_contract,
                "execution_inputs": {
                    "decoders": _bound_decoder_receipts(
                        bundle.decoder_paths,
                        bundle.decoder_sha256,
                    ),
                    **run_control_before,
                    "geog_root": str(geog_root),
                    "geog_source_binding": (
                        "per_domain_resolved_dataset_paths_plus_native_"
                        "static_output_sha256"
                    ),
                    "root_static_provider": root_static_provider,
                    "root_static_receipt": root_static_receipt,
                },
                "source_composition": composition_receipt,
                # Present only when the source files carried fewer vertical
                # levels than the mapping declares (an era ladder), as in
                # the single-domain proof.
                **({"source_vertical_ladder": vertical_ladder}
                   if vertical_ladder is not None else {}),
                "preprocessing": scientific_preprocessing(preprocess_receipt),
                "hierarchy_workers": selected_workers,
                "root_static": static_output_receipt,
                "root_geometry": geometry_receipt,
                "static_catalog": dict(
                    hierarchy_result.static_catalog_receipt
                ),
                "source_coverage": dict(
                    hierarchy_result.source_coverage_receipt
                ),
                "artifact_receipt": dict(
                    hierarchy_result.hierarchy.artifacts.receipt
                ),
                # WHETHER EACH DOMAIN'S INITIALIZATION MODIFIED VAPOUR ON
                # THE WAY IN, root and children alike.  Unconditional, and
                # stated even when no floor fired: an absent key would read
                # as "prepared before the receipt existed", a different
                # claim and one no reader of the bundle could check.  Old
                # bundles that predate it are still accepted -- the reader
                # discards the requirement when the key is absent, the way
                # it already does for `stock_wrf_export`.
                **dict(hierarchy_result.hierarchy.moisture_floor_receipts),
                "wrf_manifest": dict(
                    hierarchy_result.hierarchy.wrf_manifest
                ),
                # Present only when the experiment carries a
                # [perturbation] block: the bubbles this tree's forecast
                # applies at start, recorded as deferred because the
                # prepared arrays stay unperturbed.  Absent, the proof is
                # byte-for-byte what it always was.
                **({"initial_perturbation": initial_perturbation}
                   if initial_perturbation is not None else {}),
                # Present only when the preparation opted in: the sealed
                # statics-corridor set, digest-bound here the way every
                # other sealed artifact is.  Absent, the proof is
                # byte-for-byte what it always was -- which is what keeps
                # every retained mapped hash valid across this change.
                **(
                    {"statics_corridor": dict(
                        hierarchy_result.statics_corridor_receipt)}
                    if hierarchy_result.statics_corridor_receipt is not None
                    else {}),
                "timing_seconds": {
                    "static_root": static_seconds,
                    "decode_and_compose": decode_seconds,
                    "initialize_all_root_times": initialize_seconds,
                    **dict(hierarchy_result.hierarchy.timings_seconds),
                    "hierarchy_call_wall": hierarchy_seconds,
                    "total": time.perf_counter() - started,
                },
            }
            from woof.ingest.preparation_evidence import write_runtime_preparation
            write_runtime_preparation(staging, preprocessing=preprocess_receipt,
                parallelism={"hierarchy_workers": selected_workers})
            proof["proof_content_sha256"] = hashlib.sha256(
                _canonical(proof).encode("utf-8")
            ).hexdigest()
            (staging / "proof.json").write_text(
                json.dumps(
                    proof, indent=2, sort_keys=True, allow_nan=False
                ) + "\n",
                encoding="utf-8",
            )
            os.replace(staging, output_root)
            # The composed frame stream is spent: every valid time has
            # been interpolated and the tree is published, so the engine
            # scratch it streamed from goes now rather than at collection.
            bundle.close()
            return proof
        identity = prepared_cache_identity(
            bridge_manifest_sha256=published_manifest_sha256,
            source_manifest_sha256=published_manifest_sha256,
            static_cache_sha256=static_output_receipt["sha256"],
            namelist_sha256=_sha256(experiment_config),
            domain_config=exp.root,
            **forcing_identity,
            source_identity=source_identity,
        )
        writer = PreparedTreeWriter(
            staging=staging, output_root=output_root, identity=identity)
        # Everything the proof says that the start time already knows.  The
        # seal adds only the cache receipt, the export and the wall times
        # (boundary_stream.SEAL_ONLY_PROOF_KEYS), and refuses a proof that
        # differs from this head anywhere else.
        proof_head = {
            "schema": PROOF_SCHEMA,
            **({"initial_source": initial_source_receipt}
               if initial_source_receipt is not None else {}),
            "status": "READY_NOT_YET_STOCK_WRF_GATED",
            "stock_wrf_export": stock_wrf_export,
            "vertical_coordinate": _vertical_coordinate_receipt(
                exp, vertical_adaptation),
            "forcing_times": [value.isoformat() for value in times],
            # The soil-state SOURCE resolution and whether the
            # sub-source-cell reconstitution ran on it.
            "soil_texture_downscale": dict(
                getattr(soil, "soil_texture_downscale", {}) or {}),
            # Present only when a soil temperature rebuild (real.exe's
            # band, or the snow-covered rule beside it) touched a land
            # column, so a healthy proof is unchanged.
            **_soil_floor_receipts(soil),
            **({"soil_temperature_repair": soil_temperature_repair}
               if soil_temperature_repair is not None else {}),
            forcing_key: list(forcing_axis),
            "boundary_interval_seconds": boundary_interval_seconds,
            "execution_inputs": {
                "decoders": _bound_decoder_receipts(
                    bundle.decoder_paths,
                    bundle.decoder_sha256,
                ),
                **run_control_before,
                "geog_root": str(geog_root),
                "geog_source_binding": (
                    "resolved_dataset_paths_plus_native_static_output_sha256"
                ),
                "root_static_provider": root_static_provider,
                "root_static_receipt": root_static_receipt,
                "geog_resolution_tokens": list(selection.resolution_tokens),
                "geog_datasets": {
                    field: str(selection.path(field))
                    for field in (
                        "terrain", "landuse", "soil_top", "soil_bottom",
                        "greenfrac", "lai", "albedo", "snow_albedo",
                        "soil_temperature",
                    )
                },
            },
            "source_composition": composition_receipt,
            # Present only when the source files carried fewer vertical
            # levels than the mapping declares (an era ladder), so a
            # full-ladder preparation's receipt is unchanged.
            **({"source_vertical_ladder": vertical_ladder}
               if vertical_ladder is not None else {}),
            "preprocessing": scientific_preprocessing(preprocess.receipt()),
            # WHETHER THIS INITIALIZATION MODIFIED VAPOUR ON THE WAY IN.
            # Unconditional, and stated even when no floor fired.
            **moisture_floor_proof_entry(
                initial_result,
                when_unrecorded=(
                    "this preparation's initialization result carries "
                    "no moisture-floor field, so it came from an ingest "
                    "predating the receipt; re-prepare the case to "
                    "record whether its vapour was floored on the way "
                    "in")),
            "static": static_output_receipt,
            "geometry": geometry_receipt,
            # Present only when the root coordinates pair with their own
            # source cells (horiz.declared_grid_pairing), so every
            # interpolated preparation's proof is unchanged.
            **({"source_pairing": "identity"}
               if declared_grid_pairing(mapping_contract.get("grid"), grid)
               == "identity" else {}),
        }
        if posted_source is not None:
            # They read every lead: the seal writes them from every lead
            # batch (_AS_POSTED_SEAL_KEYS), and the head leaves them out.
            for key in _AS_POSTED_SEAL_KEYS:
                proof_head.pop(key, None)
        # One machine, one card: the forecast may start beside this
        # producer only when both fit (boundary_stream.chained_admission).
        writer.admit(
            experiment=exp, backend=str(preprocess_receipt["backend"]),
            device_bytes=producer_device_bytes(str(preprocess_receipt["backend"]),
                                               selection=preprocess_receipt.get("selection")),
            preprocess_selection=preprocess_receipt.get("selection"),
            future_decode_host_bytes=(getattr(bundle, "future_decode_host_bytes", lambda: 0)()
                                      if posted_source is None else
                                      posted_source.future_decode_host_bytes()),
            source=mapping_contract,
            urban_columns=prepared_head_urban_columns(exp, static))
        as_posted_head = None
        if posted_source is not None:
            as_posted_head = {
                "input_plan": posted_plan["plan"],
                # The markers of every lead the head's own decode read.
                "start_markers": dict(posted_source.markers),
                "forcing_leads": _posted_window_leads(posted_source, times),
                "seal_authored_proof_keys": _AS_POSTED_SEAL_KEYS,
                "manifest_path": "source-evidence/input-manifest.json",
                "lead_role_prefix": "",
                "manifest_bound_identity_keys": _AS_POSTED_MANIFEST_BOUND,
                "fixed_rows": posted_plan["fixed_rows"],
                "proof_manifest_key": _AS_POSTED_PROOF_MANIFEST_KEY,
                "posted_user_metadata": ("composition_receipt_sha256",),
            }
        with prep_stage("prepared_head", label="Publish prepared head"):
            head_started = time.perf_counter()
            writer.write_head(
                initial_result=initial_result, met=initial_met,
                surface=canonical_noah_surface(soil),
                metadata={
                    "source_adapter": "mapped",
                    "initial_valid_time": times[0].isoformat(),
                    "last_valid_time": times[-1].isoformat(),
                    forcing_key: list(forcing_axis),
                    "boundary_interval_seconds": boundary_interval_seconds,
                    "composition_receipt_sha256": receipt_identity_sha256,
                    **_soil_floor_receipts(soil),
                },
                lbc={
                    "spec_bdy_width": cfg.spec_bdy_width,
                    "spec_zone": cfg.spec_zone,
                    "relax_zone": cfg.relax_zone,
                    "schedule": [
                        [float(earlier), float(later)] for earlier, later
                        in zip(forcing_seconds, forcing_seconds[1:])],
                    "fields": forcing.inventory,
                },
                proof_head=proof_head,
                input_manifest_sha256=(None if posted_source is not None
                                       else published_manifest_sha256),
                forcing=forcing,
                as_posted=as_posted_head,
            )
            if posted_source is not None:
                # A missing lead keeps its build waiting_for_source; each
                # segment binds the markers of the leads it spans.
                posted_source.posted.writer = writer
                writer.bind_posted_leads(posted_source.markers)
            if writer.chained:
                future = bundle if posted_source is None else posted_source
                future_bytes = getattr(future, "future_decode_host_bytes", lambda: 0)()
                if future_bytes:
                    future.limit_future_decode(future_bytes)
        head_seconds = time.perf_counter() - head_started
        # The start state has done its work: it is in the head.
        del initial_result, initial_met
        release_backend_memory(preprocess)
        with prep_stage("root_boundaries",
                        label="Initialize remaining forcing states",
                        backend=str(preprocess_receipt["backend"]),
                        count=len(snapshots) - 1):
            boundaries_started = time.perf_counter()
            from woof.ensemble.posted_preparation import current_posted_preparation
            from woof.ingest.preparation_workers import forcing_time_workers
            # Physical stores and member replacement still need time order.
            # A posted window with no missing lead is independent too: finish
            # its decode before workers start, so no build advances batches or
            # changes the markers the ordered segment writer binds.
            independent_times = (physical_input is None and physical_output is None
                                 and current_posted_preparation() is None)
            if independent_times and posted_source is not None:
                independent_times = posted_source.prepare_independent_times()
            forcing_workers = forcing_time_workers(
                len(snapshots) - 1,
                cpu_workers=worker_policy.get("effective_workers"),
                backend=str(preprocess_receipt["backend"]),
                selection=preprocess_receipt.get("selection"),
            ) if independent_times else {"workers": 1, "basis": "ordered physical source or pending posted lead"}
            worker_policy["forcing_time_workers"] = forcing_workers
            if boundary_prestage is not None:
                writer.write_intervals(boundary_prestage.boundaries.intervals)
                forcing.release(0)
                forcing.release(1)
            else:
                writer.stream_forcing_times(
                    count=len(snapshots), build_forcing_time=build_forcing_time,
                    forcing=forcing, times=times,
                    release=lambda: release_backend_memory(preprocess),
                    workers=forcing_workers["workers"])
        initialize_seconds += time.perf_counter() - boundaries_started
        forcing_timings.sort(key=lambda entry: entry["forcing_index"])
        posted_proof = {}
        # Every deferred full re-read of the inputs has matched before the
        # cache seals (woof.authority_digests): a changed input never seals.
        from woof import authority_digests
        authority_digests.join()
        with prep_stage("prepared_cache", label="Seal prepared cache"):
            cache_started = time.perf_counter()
            if posted_source is None:
                cache_receipt = dict(writer.seal_cache())
            else:
                sealed = _seal_posted_mapped(
                    posted_source, writer=writer, plan=posted_plan,
                    mapping_contract=mapping_contract, snapshots=snapshots,
                    exp=exp, cfg=cfg, source_identity=source_identity,
                    static_cache_sha256=static_output_receipt["sha256"],
                    namelist_sha256=_sha256(experiment_config),
                    forcing_identity=forcing_identity,
                    experiment_config=experiment_config)
                cache_receipt = sealed["cache_receipt"]
                posted_proof = sealed["proof"]
                bundle = sealed["bundle"]
                decode_seconds = posted_source.decode_seconds
            cache_receipt["path"] = "prepared-cache"
        cache_seconds = head_seconds + time.perf_counter() - cache_started
        prepared_path = writer.cache_path
        static_path = writer.root / "native-static.npz"
        geometry_path = writer.root / "geometry-receipt.json"
        wrf_path = writer.root / "wrf-native-input"
        with prep_stage("wrf_export", label="Companion WRF files") as export_stage:
            export_started = time.perf_counter()
            export_schema = "gpuwm-native-direct-wrf-export-v3"
            export_stage["outcome"] = "produced"
            if stock_wrf_export == "off":
                export_receipt = stock_wrf_export_not_requested(schema=export_schema)
                export_stage["outcome"] = "not_requested"
            else:
                try:
                    export_receipt = export_prepared_wrf(
                        prepared_path, static_path, geometry_path, wrf_path,
                        valid_time=times[0],
                        boundary_interval_seconds=boundary_interval_seconds,
                        experiment_config_suite=True,
                        expert_acknowledgements=tuple(physics_selection["acknowledgements"]),
                        acknowledgement_provenance=physics_selection[
                            "acknowledgement_provenance"],
                    )
                except StockWrfExportUnsupported as error:
                    if stock_wrf_export == "required":
                        raise
                    export_receipt = stock_wrf_export_refused(error, schema=export_schema)
                    export_stage.update(outcome="refused", reason=str(error))
                else:
                    if (export_receipt.get("schema") != export_schema
                            or export_receipt.get("physics") != physics_selection):
                        raise RuntimeError(
                            "mapped export physics differs from the experiment config")
        export_seconds = time.perf_counter() - export_started
        verify_overlay_sequence(snapshots)
        run_control_after = {
            "wps_namelist": _file_receipt(wps_namelist),
            "experiment_config": _file_receipt(experiment_config),
        }
        if cpu_bridge is not None:
            run_control_after["cpu_preprocess_bridge"] = _file_receipt(cpu_bridge)
        if run_control_after != run_control_before:
            raise ValueError("mapped run-control bytes changed during preparation")
        proof = {
            **proof_head,
            # As posted, the leads the preparation waited for (DESIGN A136
            # 2.4 item 6); a preparation of a complete window has none.
            **({"posting": posted_proof.get("posting")}
               if posted_source is not None else {}),
            "prepared_cache": cache_receipt,
            "export": export_receipt,
            "boundary_stream": writer.boundary_stream_proof(),
            "timing_seconds": {
                "static": static_seconds,
                "decode_and_compose": decode_seconds,
                "initialize_all_times": initialize_seconds,
                "write_prepared_cache": cache_seconds,
                "direct_wrf_export": export_seconds,
                "total": time.perf_counter() - started,
            },
        }
        # Runtime headroom and inherited thread defaults vary between identical
        # preparations. Keep the existing scientific preprocessing identity in
        # proof/head and retain all measurements in the adjacent runtime receipt.
        from woof.ingest.preparation_evidence import write_runtime_preparation
        write_runtime_preparation(writer.root, preprocessing=preprocess.receipt(),
                                  parallelism=worker_policy, forcing_stage_timings=forcing_timings)
        # The seal-authored receipt keys, from every lead batch as one
        # decode (the head left them out).
        proof.update({key: posted_proof[key] for key in _AS_POSTED_SEAL_KEYS
                      if key in posted_proof})
        proof["proof_content_sha256"] = hashlib.sha256(
            _canonical(proof).encode("utf-8")
        ).hexdigest()
        writer.publish(proof)
        if boundary_prestage is not None:
            boundary_prestage.cache_content_sha256 = cache_receipt["content_sha256"]
        if physical_output is not None:
            physical_output.seal()
        # The composed frame stream is spent: every valid time has been
        # interpolated and the tree is published, so the engine scratch
        # it streamed from goes now rather than at collection.
        bundle.close()
        if posted_source is not None:
            posted_source.close()
        return proof
    except BaseException as error:
        try:
            if writer is not None:
                # A head already published stays, marked failed, so a
                # waiting forecast ends with this reason and the next
                # preparation of this output root rebuilds it.  Recorded
                # first, so a close that raises cannot leave it unwritten.
                writer.fail(error)
        finally:
            try:
                bundle.close()
                if posted_source is not None:
                    posted_source.close()
            finally:
                shutil.rmtree(staging, ignore_errors=True)
        raise


def _prepare_chained_mapped_tree(c) -> dict[str, object]:
    """A mapped domain tree, chained: head, one root interval per segment, seal.

    The head holds the root's static files and start state (streamed cache
    under ``hierarchy-head/domains/d01/prepared-cache``, its header written
    at the seal) and every child's complete artifact set under
    ``hierarchy-head/domains/dNN`` (a child needs only the start time),
    and a moving nest's statics corridor under
    ``hierarchy-head/statics-corridor``, bound by the head's proof.
    Segment k is the root's boundary interval k.  No start state stays
    resident between head and seal, on either backend: the seal re-reads
    the root (with its whole boundary set) and every child from the head
    (:class:`woof.ingest.boundary_stream.TreeStartStates`) and writes the
    one-shot ``hierarchy-artifacts/`` tree from them, through the same
    writer and the same arguments as the unchained tree, then the
    companion WRF files and the corridor (copied from the head), then
    ``proof.json``.
    """

    exp = c.exp
    cfg = c.cfg
    bundle = c.bundle
    staging = c.staging
    backend = str(c.preprocess_receipt["backend"])
    selected_workers = c.hierarchy_workers
    if selected_workers is None:
        selected_workers = 8 if backend == "cpu" else 1
    namelist_sha256 = _sha256(c.experiment_config)
    manifest_sha256 = c.published_manifest_sha256
    posted = c.posted_source
    receipt_sha256 = (c.source_identity["composition_receipt_sha256"]
        if posted is not None else
        composition_receipt_identity_sha256(c.composition_receipt))
    initialize_seconds = c.initialize_seconds
    # The forcing axis under its own name, as the unchained proof spells
    # it (the proof inventory gate reads the key by that name).
    forcing_key, forcing_axis = c.forcing_key, c.forcing_axis
    # Exactly the three the unchained tree hands its hierarchy call.
    tree_identity = {**c.source_identity, "target_contract": c.target_contract}
    root_metadata = {"composition_receipt_sha256": receipt_sha256,
                     "mapped_target_contract": c.target_contract,
                     **_soil_floor_receipts(c.soil)}
    input_provenance = {
        "mapping_sha256": bundle.mapping_sha256,
        "composition_sha256": bundle.composition_sha256,
        "input_manifest_sha256": manifest_sha256,
        "decoder_sha256": dict(bundle.decoder_sha256),
        "preprocessing": scientific_preprocessing(c.preprocess_receipt),
        "mapped_target_contract": c.target_contract,
    }
    writer = None
    try:
        hierarchy_started = time.perf_counter()
        tree_head = prepare_regular_source_hierarchy_head(
            exp=exp,
            grids=c.grids,
            snapshots=c.snapshots,
            # A child sees the catalog, not the config.
            soil_texture_downscale=declared_soil_texture_downscale(
                c.experiment_config),
            **c.forcing_identity,
            wps_namelist=c.wps_namelist,
            geog_root=c.geog_root,
            source_name="RW-WPS-MAPPED",
            root_initial_result=c.initial_result,
            source_inventory=tuple(c.snapshots[0].fields),
            workers=selected_workers,
            preprocess_backend=backend,
            cpu_bridge=c.cpu_bridge,
            soil_layer_contract=bundle.soil_layer_contract,
            preprocess_selection=c.preprocess_receipt.get("selection"),
            source_manifest_sha256=manifest_sha256,
            statics_corridor=c.statics_corridor,
            static_highres=c.static_highres,
            sfcp_to_sfcp=c.case_policy["sfcp_to_sfcp"],
            water_temperature_policy=c.case_policy[
                "water_temperature_policy"],
            # The statics corridor goes into the head, so a nest that
            # moves from the first step has its ground there.
            head_artifacts=staging / HIERARCHY_HEAD_DIRNAME,
        )
        bound_identity = tree_head.bound_source_identity(tree_identity)
        head_domains = staging / HIERARCHY_HEAD_DIRNAME / "domains"
        root_directory = head_domains / "d01"
        root_directory.mkdir(parents=True)
        root_static_receipt, _root_geometry = write_domain_static_files(
            root_directory, domain=exp.domains[0], grid=c.grids[0],
            static_fields=c.static)
        child_builds = write_child_domain_artifacts(
            head_domains, exp=exp, child_results=tree_head.child_results,
            bridge_manifest_sha256=manifest_sha256,
            source_manifest_sha256=manifest_sha256,
            namelist_sha256=namelist_sha256, **tree_head.forcing_identity,
            source_identity=bound_identity, valid_time=exp.start_time)
        binding = root_domain_artifact_binding(
            exp=exp, static_cache_sha256=root_static_receipt["sha256"],
            bridge_manifest_sha256=manifest_sha256,
            source_manifest_sha256=manifest_sha256,
            namelist_sha256=namelist_sha256, **tree_head.forcing_identity,
            source_identity=bound_identity, valid_time=exp.start_time,
            root_metadata=root_metadata)
        head_hierarchy_seconds = time.perf_counter() - hierarchy_started
        cache_name = f"{HIERARCHY_HEAD_DIRNAME}/domains/d01/prepared-cache"
        writer = PreparedTreeWriter(
            staging=staging, output_root=c.output_root,
            identity=binding.identity, cache_name=cache_name)
        head_moisture_floor_receipts = hierarchy_moisture_floor_receipts(
            exp, c.initial_result, tree_head.child_results)
        # Everything the tree's proof says that the start time already
        # knows, the statics corridor included (built into the head, so a
        # moving nest's forecast starts there); the seal adds the artifact
        # records, the export, the stream and the wall times
        # (boundary_stream.SEAL_ONLY_PROOF_KEYS).
        proof_head = {
            "schema": HIERARCHY_PROOF_SCHEMA,
            "status": "READY_NOT_YET_STOCK_WRF_GATED",
            "stock_wrf_export": c.stock_wrf_export,
            "domain_count": len(exp.domains),
            "vertical_coordinate": _vertical_coordinate_receipt(
                exp, c.vertical_adaptation),
            "forcing_times": [value.isoformat() for value in c.times],
            # The soil-state SOURCE resolution and whether the
            # sub-source-cell reconstitution ran on it.
            "soil_texture_downscale": dict(
                getattr(c.soil, "soil_texture_downscale", {}) or {}),
            # Present only when a soil temperature rebuild (real.exe's
            # band, or the snow-covered rule beside it) touched a land
            # column of the root.
            **_soil_floor_receipts(c.soil),
            **({"soil_temperature_repair": c.soil_temperature_repair}
               if c.soil_temperature_repair is not None else {}),
            forcing_key: list(forcing_axis),
            "boundary_interval_seconds": c.boundary_interval_seconds,
            "target_contract": c.target_contract,
            "execution_inputs": {
                "decoders": _bound_decoder_receipts(
                    bundle.decoder_paths,
                    bundle.decoder_sha256,
                ),
                **c.run_control_before,
                "geog_root": str(c.geog_root),
                "geog_source_binding": (
                    "per_domain_resolved_dataset_paths_plus_native_"
                    "static_output_sha256"
                ),
                "root_static_provider": c.root_static_provider,
                "root_static_receipt": c.root_static_receipt,
            },
            "source_composition": c.composition_receipt,
            # Present only when the source files carried fewer vertical
            # levels than the mapping declares (an era ladder), as in the
            # unchained tree.
            **({"source_vertical_ladder": c.vertical_ladder}
               if c.vertical_ladder is not None else {}),
            "preprocessing": scientific_preprocessing(c.preprocess_receipt),
            "hierarchy_workers": selected_workers,
            "root_static": c.static_output_receipt,
            "root_geometry": c.geometry_receipt,
            "static_catalog": dict(tree_head.static_receipt),
            "source_coverage": dict(tree_head.source_coverage_receipt),
            **head_moisture_floor_receipts,
            # Present only when the experiment carries a [perturbation]
            # block: the bubbles this tree's forecast applies at start,
            # recorded as deferred as in the unchained tree.
            **({"initial_perturbation": c.initial_perturbation}
               if c.initial_perturbation is not None else {}),
            # Present only when the preparation opted in, as in the
            # unchained tree.
            **({"statics_corridor": dict(tree_head.statics_corridor_receipt)}
               if tree_head.statics_corridor_receipt is not None else {}),
        }
        as_posted_head = None
        if posted is not None:
            for key in _AS_POSTED_SEAL_KEYS:
                proof_head.pop(key, None)
            as_posted_head = {
                "input_plan": c.posted_plan["plan"],
                "start_markers": dict(posted.markers),
                "forcing_leads": _posted_window_leads(posted, c.times),
                "seal_authored_proof_keys": _AS_POSTED_SEAL_KEYS,
                "manifest_path": "source-evidence/input-manifest.json",
                "lead_role_prefix": "",
                "manifest_bound_identity_keys": _AS_POSTED_MANIFEST_BOUND,
                "fixed_rows": c.posted_plan["fixed_rows"],
                "proof_manifest_key": _AS_POSTED_PROOF_MANIFEST_KEY,
                "posted_user_metadata": ("composition_receipt_sha256",),
            }
        # One machine, one card: a forecast started on this head beside
        # its producer is admitted only when both fit, priced with the
        # hydrometeor boundary tables this mapping publishes, as the
        # single domain is.
        writer.admit(
            experiment=exp, backend=backend,
            device_bytes=producer_device_bytes(backend, selection=c.preprocess_receipt.get("selection")),
            preprocess_selection=c.preprocess_receipt.get("selection"),
            future_decode_host_bytes=getattr(c.bundle, "future_decode_host_bytes", lambda: 0)(),
            source=c.mapping_contract,
            urban_columns=prepared_head_urban_columns(
                exp, c.static, child_results=tree_head.child_results))
        with prep_stage("prepared_head", label="Publish prepared head"):
            head_started = time.perf_counter()
            writer.write_head(
                initial_result=c.initial_result, met=c.initial_met,
                surface=canonical_noah_surface(c.soil),
                metadata=dict(binding.metadata),
                lbc={
                    "spec_bdy_width": cfg.spec_bdy_width,
                    "spec_zone": cfg.spec_zone,
                    "relax_zone": cfg.relax_zone,
                    "schedule": [
                        [float(earlier), float(later)] for earlier, later
                        in zip(c.forcing_seconds, c.forcing_seconds[1:])],
                    "fields": c.forcing.inventory,
                },
                proof_head=proof_head,
                input_manifest_sha256=(None if posted is not None
                                       else manifest_sha256),
                forcing=c.forcing,
                as_posted=as_posted_head,
                tree=domain_tree_head_fields(
                    [f"d{int(domain.grid_id):02d}" for domain in exp.domains],
                    root_cache=cache_name,
                    children_receipts={
                        f"d{int(build.receipt['grid_id']):02d}": _sha256(
                            head_domains
                            / f"d{int(build.receipt['grid_id']):02d}"
                            / "receipt.json")
                        for build in child_builds}),
                extra_head_payload_bytes=sum(
                    int(build.receipt["artifacts"]["prepared_cache"][
                        "payload_bytes"]) for build in child_builds),
            )
        head_seconds = time.perf_counter() - head_started
        if posted is not None:
            posted.posted.writer = writer
            writer.bind_posted_leads(posted.markers)
        future_bytes = getattr(c.bundle, "future_decode_host_bytes", lambda: 0)()
        if writer.chained and future_bytes:
            c.bundle.limit_future_decode(future_bytes)
        # Every start state is in the head now: the root's in its streamed
        # cache, each child's in hierarchy-head/domains/dNN.  None stays
        # resident while the later times are built (on the card that is the
        # residency start_last_forcing_order avoids); the seal re-reads them.
        start_states = TreeStartStates.release(
            root_result=c.initial_result, root_met=c.initial_met,
            child_results=tree_head.child_results,
            child_content_sha256={
                f"d{int(build.receipt['grid_id']):02d}": build.receipt[
                    "artifacts"]["prepared_cache"]["content_sha256"]
                for build in child_builds})
        c.initial_result = c.initial_met = None
        tree_head.child_results = ()
        release_backend_memory(c.preprocess)
        with prep_stage("root_boundaries",
                        label="Initialize remaining forcing states",
                        backend=backend, count=len(c.snapshots) - 1):
            boundaries_started = time.perf_counter()
            # The later root times side by side, as the single-domain route
            # builds them (forcing_time_workers; written in index order, so
            # the bytes are the serial loop's). Named breakage: a domain tree
            # built them one at a time -- the WOOF Oklahoma City nest pair
            # spent 82-112 s here at about 3 of 32 cores (2026-10-23).
            from woof.ensemble.posted_preparation import current_posted_preparation
            from woof.ingest.preparation_workers import forcing_time_workers
            independent_times = (c.forcing_times_independent
                                 and current_posted_preparation() is None)
            if independent_times and posted is not None:
                independent_times = posted.prepare_independent_times()
            forcing_workers = forcing_time_workers(
                len(c.snapshots) - 1,
                cpu_workers=c.worker_policy.get("effective_workers"),
                backend=backend,
                selection=c.preprocess_receipt.get("selection"),
            ) if independent_times else {
                "workers": 1, "basis": "ordered physical source or pending posted lead"}
            c.worker_policy["forcing_time_workers"] = forcing_workers
            print(f"preparation workers: forcing times {forcing_workers['workers']} "
                  f"({forcing_workers.get('caps', forcing_workers['basis'])})", file=sys.stderr, flush=True)
            writer.stream_forcing_times(
                count=len(c.snapshots),
                build_forcing_time=c.build_forcing_time,
                forcing=c.forcing, times=c.times,
                release=lambda: release_backend_memory(c.preprocess),
                workers=forcing_workers["workers"])
        initialize_seconds += time.perf_counter() - boundaries_started
        root_identity = binding.identity
        posted_proof = {}
        decode_seconds = c.decode_seconds
        # Every deferred full re-read of the inputs has matched before the
        # cache seals (woof.authority_digests): a changed input never seals.
        from woof import authority_digests
        authority_digests.join()
        with prep_stage("prepared_cache", label="Seal prepared cache"):
            if posted is None:
                root_content_sha256 = str(writer.seal_cache()["content_sha256"])
            else:
                def identity_builder(source, digest):
                    return root_domain_artifact_binding(
                        exp=exp, static_cache_sha256=root_static_receipt["sha256"],
                        bridge_manifest_sha256=digest,
                        source_manifest_sha256=digest,
                        namelist_sha256=namelist_sha256,
                        **tree_head.forcing_identity,
                        source_identity=tree_head.bound_source_identity(source),
                        valid_time=exp.start_time,
                        root_metadata={"composition_receipt_sha256":
                            source["composition_receipt_sha256"],
                            "mapped_target_contract": c.target_contract,
                            **_soil_floor_receipts(c.soil)}).identity
                sealed = _seal_posted_mapped(posted, writer=writer,
                    plan=c.posted_plan, mapping_contract=c.mapping_contract,
                    snapshots=c.snapshots, exp=exp, cfg=cfg,
                    source_identity=tree_identity,
                    static_cache_sha256=root_static_receipt["sha256"],
                    namelist_sha256=namelist_sha256,
                    forcing_identity=tree_head.forcing_identity,
                    experiment_config=c.experiment_config,
                    identity_builder=identity_builder)
                manifest_sha256 = sealed["manifest_sha256"]
                tree_identity = sealed["source_identity"]
                root_metadata = {"composition_receipt_sha256":
                    tree_identity["composition_receipt_sha256"],
                    "mapped_target_contract": c.target_contract,
                    **_soil_floor_receipts(c.soil)}
                input_provenance = {**input_provenance,
                    "input_manifest_sha256": manifest_sha256}
                root_identity = sealed["identity"]
                root_content_sha256 = str(sealed["cache_receipt"]["content_sha256"])
                posted_proof = sealed["proof"]
                decode_seconds = posted.decode_seconds
                initialize_seconds -= decode_seconds - c.decode_seconds
        hierarchy_seal_started = time.perf_counter()
        with prep_stage("tree_start_states",
                        label="Re-read start states from the head"):
            (root_result, root_met, boundaries,
             tree_head.child_results) = start_states.reread(
                writer.root, exp=exp, grids=c.grids,
                root_identity=root_identity,
                root_content_sha256=root_content_sha256)
        hierarchy_result = seal_regular_source_hierarchy(
            tree_head,
            artifact_output=writer.root / "hierarchy-artifacts",
            wrf_output=writer.root / "wrf-native-input",
            root_initial_result=root_result,
            root_met=root_met,
            root_soil=c.soil,
            root_static_fields=c.static,
            root_boundaries=boundaries,
            bridge_manifest_sha256=manifest_sha256,
            source_manifest_sha256=manifest_sha256,
            namelist_sha256=namelist_sha256,
            source_identity=tree_identity,
            root_metadata=root_metadata,
            input_provenance=input_provenance,
            artifact_manifest_reference=(
                "../hierarchy-artifacts/domain-artifacts.json"
            ),
            stock_wrf_export=c.stock_wrf_export,
            head_artifacts=writer.root / HIERARCHY_HEAD_DIRNAME,
            snapshots=c.snapshots,
        )
        hierarchy_seconds = (head_hierarchy_seconds
                             + time.perf_counter() - hierarchy_seal_started)
        if dict(hierarchy_result.hierarchy.moisture_floor_receipts) \
                != head_moisture_floor_receipts:
            raise RuntimeError(
                "the sealed tree's moisture-floor receipts differ from the "
                "head's, which were computed from the same initial states")
        start_states.require_sealed_is_head(
            hierarchy_result.hierarchy.artifacts.receipt,
            root_content_sha256=root_content_sha256,
            **({} if posted is None else {"as_posted": {
                "root": writer.root, "head": writer.head,
                "manifest_sha256": manifest_sha256}}))
        c.verify_overlay_sequence(c.snapshots)
        run_control_after = {
            "wps_namelist": _file_receipt(c.wps_namelist),
            "experiment_config": _file_receipt(c.experiment_config),
        }
        if c.cpu_bridge is not None:
            run_control_after["cpu_preprocess_bridge"] = _file_receipt(
                c.cpu_bridge)
        if run_control_after != c.run_control_before:
            raise ValueError(
                "mapped run-control bytes changed during preparation"
            )
        proof = {
            **proof_head,
            **posted_proof,
            "artifact_receipt": dict(
                hierarchy_result.hierarchy.artifacts.receipt
            ),
            "wrf_manifest": dict(
                hierarchy_result.hierarchy.wrf_manifest
            ),
            # Present only when the preparation opted in, as in the
            # unchained tree.
            **(
                {"statics_corridor": dict(
                    hierarchy_result.statics_corridor_receipt)}
                if hierarchy_result.statics_corridor_receipt is not None
                else {}),
            "boundary_stream": writer.boundary_stream_proof(),
            "timing_seconds": {
                "static_root": c.static_seconds,
                "decode_and_compose": decode_seconds,
                "initialize_all_root_times": initialize_seconds,
                **dict(hierarchy_result.hierarchy.timings_seconds),
                "hierarchy_call_wall": hierarchy_seconds,
                "prepared_head": head_seconds,
                "head_published_after": writer.head_seconds,
                "total": time.perf_counter() - c.started,
            },
        }
        proof["proof_content_sha256"] = hashlib.sha256(
            _canonical(proof).encode("utf-8")
        ).hexdigest()
        from woof.ingest.preparation_evidence import write_runtime_preparation
        write_runtime_preparation(writer.root, preprocessing=c.preprocess_receipt,
            parallelism={"hierarchy_workers": selected_workers})
        writer.publish(proof)
    except BaseException as error:
        if writer is not None:
            # A head already published stays, marked failed, so a waiting
            # forecast ends with this reason and the next preparation of
            # this output root rebuilds it.  The caller closes the frame
            # stream and removes a staging tree that was never published.
            writer.fail(error)
        raise
    # The composed frame stream is spent: every valid time has been
    # interpolated and the tree is sealed, so the engine scratch it
    # streamed from goes now rather than at collection.
    bundle.close()
    if posted is not None:
        posted.close()
    return proof


_ROLE_PATTERN = re.compile(r"[A-Za-z0-9_.-]+")


def _role_bindings(
    values: Sequence[str], *, multiple: bool,
) -> dict[str, Path | tuple[Path, ...]]:
    """Parse repeatable ROLE=PATH bindings without silent replacement."""

    grouped: dict[str, list[Path]] = {}
    for value in values:
        role, separator, raw_path = value.partition("=")
        if not separator or not _ROLE_PATTERN.fullmatch(role) or not raw_path:
            raise ValueError(
                "role bindings must use non-empty ROLE=PATH with a portable "
                "role name"
            )
        if not multiple and role in grouped:
            raise ValueError(f"duplicate binding for role {role!r}")
        grouped.setdefault(role, []).append(Path(raw_path))
    return {
        role: tuple(paths) if multiple else paths[0]
        for role, paths in grouped.items()
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rw-wps mapped",
        description=(
            "Prepare atomic stock-WRF initial and boundary files from a "
            "strict mapped GRIB/NetCDF composition without WPS or real.exe."
        ),
    )
    parser.add_argument(
        "--stock-wrf-export", choices=STOCK_WRF_EXPORT_MODES, default="optional",
        help="unchanged-WRF file output: optional (native forecast remains usable "
             "if its configuration is unrepresentable), required, or off")
    parser.add_argument(
        "--source-format", choices=("grib1", "grib2", "netcdf"),
        required=True,
    )
    parser.add_argument("--composition", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    # One set of primary files, two transports.  The repeated flag IS
    # the grammar; the list file carries the same ordered paths for the
    # command lines Windows cannot launch (CreateProcess caps the whole
    # line at 32 KB, and a field-per-file source needs hundreds of
    # inputs per state).
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument(
        "--input", dest="primary_files", action="append", type=Path,
    )
    inputs.add_argument(
        "--input-list", dest="input_list", type=Path,
        help="file naming the --input files, one path per line, in the "
             "same deterministic time/file order",
    )
    parser.add_argument(
        "--supplement", action="append", default=[], metavar="ROLE=PATH",
        help="repeat a role for a deterministic multi-file supplement",
    )
    parser.add_argument(
        "--provenance", action="append", default=[], metavar="ROLE=PATH",
    )
    parser.add_argument(
        "--contributing-mapping", action="append", default=[],
        metavar="ROLE=PATH",
        help=(
            "a contributing source's own mapping document for a "
            "cross-source composition, under the mapping_role its "
            "field_sources binding declares; the bytes must hash to the "
            "SHA-256 the composition pins"
        ),
    )
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--initial-inputs", type=Path)
    parser.add_argument("--input-manifest-sha256")
    parser.add_argument(
        "--inland-air-mean-hours", type=float, default=None, metavar="HOURS",
        help="hours of source 2 m air temperature that inland water beyond "
             "the source reach averages (the source row's "
             "inland_air_mean_hours; 0 = the start time's own; default 24)")
    parser.add_argument(
        "--as-posted", type=Path, default=None, metavar="POSTING_DIR",
        help=("an as-posted fetch's posting/ folder: prepare as the leads "
              "post, decoding each lead batch as its marker appears, and "
              "write the window's input manifest at --input-manifest (beside "
              "the fetched files) at the seal, in place of "
              "--input-manifest-sha256"))
    parser.add_argument("--grib1-bridge", type=Path)
    parser.add_argument("--grib2-inventory", type=Path)
    parser.add_argument("--grib2-dump", type=Path)
    parser.add_argument(
        "--mapped-engine", choices=_ENGINES, default=None,
        help=(
            "which engine decodes the source bytes; omitted, the default "
            "engine runs. `python` is a WORKAROUND, not a mode: it runs "
            "the slower Python decode path and is here so a decode the "
            "Rust engine gets wrong has a way around it while the defect "
            "is fixed"),
    )
    parser.add_argument("--wps-namelist", type=Path, required=True)
    # --geog-root stays required even with a prebuilt static cache: the proof
    # binds the resolved geog dataset paths, and child domains in a hierarchy
    # still build their own statics from the tree.
    parser.add_argument("--geog-root", type=Path, required=True)
    parser.add_argument(
        "--static-input", type=Path,
        help=(
            "previously published native-static.npz to load instead of "
            "rebuilding root geography; requires --static-receipt"),
    )
    parser.add_argument(
        "--static-receipt", type=Path,
        help=(
            "geometry-receipt.json binding --static-input by geometry "
            "contract and SHA-256"),
    )
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--physical-input-store", type=Path,
                        help="sealed native physical fields to initialize, with base and donor identity")
    parser.add_argument("--physical-output-store", type=Path,
                        help="capture native mapped physical fields before real initialization")
    parser.add_argument("--physical-input-provider", type=Path,
                        help="posted native physical provider with a frozen source and member plan")
    parser.add_argument("--physical-member-index", type=int,
                        help="original recipe member index resolved by the posted physical provider")
    parser.add_argument(
        "--preprocess-backend", choices=("cuda", "cpu", "auto"),
        default="auto",
        help="where the source-grid/WRF-real setup transforms run "
             "(default auto: CUDA when the certified runtime is usable, "
             "otherwise the deterministic parallel CPU backend, "
             "announced in one line)",
    )
    parser.add_argument("--preprocess-workers", type=int)
    parser.add_argument("--cpu-preprocess-bridge", type=Path)
    parser.add_argument("--hierarchy-workers", type=int)
    parser.add_argument(
        "--statics-corridor", nargs="?", const="all", default=None,
        metavar="GRID_IDS",
        help="seal child-resolution statics over the ground each child "
             "can reach, for every child (bare flag) or for a "
             "comma-separated list of child grid ids.  REQUIRED when "
             "the experiment configures "
             "[relocation]: a prepared bundle runs without WPS_GEOG, so "
             "a nest that moves onto ground it has not visited has no "
             "other source of terrain, landuse and soil category.  Costs "
             "host RAM and disk only, never VRAM",
    )
    parser.add_argument(
        "--prepared-forecast-source", default=None, metavar="ID",
        help="the source id the finished bundle is run under, so this "
             "door can print the complete hash-bound forecast command "
             "when it is done; `woof prep`/`rw-wps` forward the "
             "--source you gave them, and a hand-written call that "
             "omits it gets prose instead of a command",
    )
    return parser


def _next_command_lines(args, proof: Mapping[str, object]) -> list[str]:
    """What this door says after the proof document, on stderr.

    The GFS, ERA5 and 20CRv3 front doors all finish by printing the
    prepared-forecast command with its three digests filled in, and the
    mapped route -- the one every packaged source runs on -- printed
    nothing at all.  A user who had just prepared a cycle was left to
    hand-extract three SHA-256 values from a 42 KB JSON document, and
    the first value the document offers under a hash-shaped name,
    ``proof_content_sha256``, is the one that can never be right.

    Where a complete command cannot be printed, this prints prose saying
    so.  It never prints a partial one: the whole value of the line is
    that it runs when pasted, and the two ways it could not are a
    preparation made from a user's own mapping (which the prepared
    runners do not accept by name) and a hand-written call to this
    module that never said which source id it is preparing.
    """

    # `packaged_profile_sources()`, NOT the forecast runner's own
    # SUPPORTED_SOURCES, and the reason is a packaging boundary rather
    # than a preference: the RW-WPS standalone distribution stages this
    # preprocessing module and deliberately EXCLUDES the forecast
    # executor, so importing it here breaks that wheel
    # (tests/test_native_wrf_distribution.py catches it).  The two sets
    # cannot drift: the runner builds `_MAPPED_PACKAGED_PROFILE` from
    # this same table, and every id in it is one it accepts -- held by
    # test_every_packaged_source_is_a_source_the_runner_accepts.
    from woof.gfs_direct import prepared_forecast_next_command
    from woof.source_adapters import packaged_profile_sources

    source = args.prepared_forecast_source
    if source is None:
        return [
            "",
            "rw-wps: preparation complete.  No forecast command is "
            "printed because this call named no prepared-forecast source "
            "id; pass --prepared-forecast-source ID (the same id `woof "
            "prep --source` takes) to get the complete hash-bound "
            "command here.",
        ]
    if source not in packaged_profile_sources():
        return [
            "",
            f"rw-wps: preparation complete, and there is no forecast "
            f"command to print for --source {source}: the "
            f"prepared-forecast runners bind a bundle to a PACKAGED "
            f"source certificate, and this preparation was made from a "
            f"mapping of your own.  The bundle under "
            f"{args.output_root} is complete; a run of it goes through "
            f"the same runner with a packaged source id.",
        ]
    return prepared_forecast_next_command(
        proof, output_root=args.output_root,
        experiment_config=args.experiment_config,
        wps_namelist=args.wps_namelist, source=source)


@owns_source_coverage_refusal
def main(argv: Sequence[str] | None = None) -> int:
    """The mapped preparation, inside its input-digest scope.

    This process decodes the inputs the fetch recorded, so its snapshots
    take those digests and the full re-reads run beside the decode
    (:mod:`woof.authority_digests`); the cache seals only after they match.
    """
    from woof import authority_digests

    with authority_digests.deferred_checks():
        return _main(argv)


def _main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.mapped_engine is not None:
        # Published into the environment rather than threaded through
        # `prepare_mapped_wrf`: the design freezes that signature, and
        # the environment variable is the SAME switch a library caller
        # uses, so the flag and the documented workaround cannot drift
        # into meaning different things.
        os.environ[_ENGINE_ENV] = args.mapped_engine
    if (args.as_posted is None) == (args.input_manifest_sha256 is None):
        parser.error(
            "give --input-manifest-sha256 for a complete window, or "
            "--as-posted POSTING_DIR to prepare as the leads post and write "
            "the manifest at the seal, and not both")
    if args.input_list is not None:
        # Resolved before any other input is opened, so a bad list file
        # is refused at the door rather than after the mapping loads.
        try:
            args.primary_files = read_input_list(args.input_list)
        except ValueError as error:
            parser.error(str(error))
    # Resolved beside the input list and for the same reason: `all` or an
    # explicit child list is an ARGV question, so a typo belongs in the
    # usage complaint rather than eleven frames into the hierarchy behind
    # a mapping that had to load first.  Spelled exactly as `rw-wps
    # --source gfs` spells it, so both front doors take the same string.
    statics_corridor = args.statics_corridor
    if statics_corridor is not None and statics_corridor != "all":
        try:
            statics_corridor = tuple(
                int(part) for part in statics_corridor.split(",") if part)
        except ValueError:
            parser.error(
                "--statics-corridor accepts 'all' or a comma-separated "
                f"list of child grid ids, not {args.statics_corridor!r}")
        if not statics_corridor:
            parser.error(
                "--statics-corridor was given an empty grid-id list; "
                "pass the bare flag for every child, or name ids")
    mapping = load_mapping(args.mapping)
    if mapping["format"] != args.source_format:
        parser.error(
            f"--source-format {args.source_format} differs from mapping "
            f"format {mapping['format']}"
        )
    decoder_values = {
        "grib1_bridge": args.grib1_bridge,
        "grib2_inventory": args.grib2_inventory,
        "grib2_dump": args.grib2_dump,
    }
    present_decoders = {
        name for name, value in decoder_values.items() if value is not None
    }
    # On the Rust engine no subprocess decoder runs, so demanding paths
    # to one is the staged-tool papercut again: a refusal asking for
    # executables nothing is going to launch.  The flags stay ACCEPTED
    # (supplying one is how a caller pins a tool, which routes the call
    # to the Python engine) but they stop being required.
    # ``compose`` unconditionally, because `prepare_mapped_wrf` composes
    # unconditionally -- contributing mappings only widen the format
    # union, they are not what makes this route a composition.  Asking
    # ``decode`` for the single-source spelling is what let this
    # validator wave a bare GRIB2 call through on the Rust engine that
    # the route then handed to the Python engine with no tools, so the
    # refusal arrived nineteen frames deep in the decoder contract
    # instead of here.
    engine_decodes_in_process = (
        not present_decoders
        and _mapped_engine_choice(
            grib1_bridge=args.grib1_bridge,
            grib2_inventory=args.grib2_inventory,
            grib2_dump=args.grib2_dump,
            subcommand=_ROUTE_SUBCOMMAND,
            source_format=args.source_format,
        ) == _ENGINE_RUST
    )
    required_decoders = set() if engine_decodes_in_process else {
        "grib1": {"grib1_bridge"},
        "grib2": {"grib2_inventory", "grib2_dump"},
        "netcdf": set(),
    }[args.source_format]
    # Stated as the decoder-contract refusal rather than as an argparse
    # usage error, and in the same sentence the route's own contract
    # uses.  A bare `usage: ... error: grib2 requires decoder flags` is
    # the staged-tool papercut in its original form: it demands a flag
    # pointing at a file the reader may not have, and says nothing about
    # getting one.  `woof prep` resolves these through the shared
    # ladder and forwards them, so this door is reached bare only by a
    # hand-written call -- which deserves the estate answer, not a
    # vocabulary complaint.
    if not args.contributing_mapping:
        unsatisfied = present_decoders != required_decoders
    else:
        # A contributing source may decode a different format, so extra
        # decoder flags are arbitrated by the exact format union inside
        # prepare_mapped_wrf; the primary's own decoders stay mandatory.
        unsatisfied = not required_decoders <= present_decoders
    if unsatisfied:
        raise _decoder_inventory_refusal(
            f"{args.source_format} decoder inventory differs from the "
            f"contract; missing={sorted(required_decoders - present_decoders)}"
            f", extra={sorted(present_decoders - required_decoders)}",
            (args.source_format,),
            missing=required_decoders - present_decoders,
            extra=present_decoders - required_decoders,
        )
    if args.preprocess_workers is not None and args.preprocess_workers <= 0:
        parser.error("--preprocess-workers must be positive")
    if args.hierarchy_workers is not None \
            and args.hierarchy_workers not in range(1, 33):
        parser.error("--hierarchy-workers must be between 1 and 32")
    if args.preprocess_backend != "cpu" \
            and args.cpu_preprocess_bridge is not None:
        parser.error(
            "--cpu-preprocess-bridge requires --preprocess-backend cpu"
        )
    if (args.static_input is None) != (args.static_receipt is None):
        parser.error("--static-input and --static-receipt must be given together")
    try:
        supplements = _role_bindings(args.supplement, multiple=True)
        provenance = _role_bindings(args.provenance, multiple=False)
        contributing = _role_bindings(
            args.contributing_mapping, multiple=False,
        )
    except ValueError as error:
        parser.error(str(error))
    try:
        proof = prepare_mapped_wrf(
            composition=args.composition,
            mapping=args.mapping,
            primary_files=args.primary_files,
            supplement_files=supplements,
            provenance_files=provenance,
            contributing_mappings=contributing,
            input_manifest=args.input_manifest,
            input_manifest_sha256=args.input_manifest_sha256,
            **({"initial_inputs": args.initial_inputs}
               if args.initial_inputs is not None else {}),
            # Named only as posted, so every other call is what it was.
            **({"as_posted": args.as_posted}
               if args.as_posted is not None else {}),
            grib1_bridge=args.grib1_bridge,
            grib2_inventory=args.grib2_inventory,
            grib2_dump=args.grib2_dump,
            wps_namelist=args.wps_namelist,
            geog_root=args.geog_root,
            static_input=args.static_input,
            static_receipt=args.static_receipt,
            experiment_config=args.experiment_config,
            output_root=args.output_root,
            **({"physical_input_store": args.physical_input_store}
               if args.physical_input_store is not None else {}),
            **({"physical_output_store": args.physical_output_store}
               if args.physical_output_store is not None else {}),
            **({"physical_input_provider": args.physical_input_provider}
               if args.physical_input_provider is not None else {}),
            **({"physical_member_index": args.physical_member_index}
               if args.physical_member_index is not None else {}),
            preprocess_backend=args.preprocess_backend,
            preprocess_workers=args.preprocess_workers,
            cpu_preprocess_bridge=args.cpu_preprocess_bridge,
            hierarchy_workers=args.hierarchy_workers,
            stock_wrf_export=args.stock_wrf_export,
            statics_corridor=statics_corridor,
            **({"inland_air_mean_hours": args.inland_air_mean_hours}
               if args.inland_air_mean_hours is not None else {}),
        )
    except InitializationMemoryRefused as error:
        # A preparation the card cannot hold under an explicit
        # --preprocess-backend cuda (PreparationDeviceRefused), or any
        # other measured initialization budget that refuses the case, is
        # a refusal with its own remedy line: one message at exit 2, as
        # `woof` itself answers it, not a traceback that buries the
        # remedy at the bottom of a stack.
        print(f"rw-wps: {error}", file=sys.stderr)
        return 2
    print(json.dumps(proof, indent=2, sort_keys=True, allow_nan=False))
    export = proof.get("export", proof.get("wrf_manifest", {}))
    if export.get("status") == "REFUSED":
        print("rw-wps: native preparation is complete; the optional WRF "
              f"export was refused: {export.get('reason')}", file=sys.stderr)
    elif export.get("status") == "NOT_REQUESTED":
        print("rw-wps: native preparation is complete; WRF export is off.",
              file=sys.stderr)
    for line in _next_command_lines(args, proof):
        print(line, file=sys.stderr)
    return 0


__all__ = [
    "HIERARCHY_PROOF_SCHEMA",
    "PROOF_SCHEMA",
    "main",
    "prepare_mapped_wrf",
]


if __name__ == "__main__":
    raise SystemExit(main())
