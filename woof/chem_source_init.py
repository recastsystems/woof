"""Fill a forcing frame's chem fields from the enabled boundary sources.

Every forcing frame of a real-data run -- the initial state and each
boundary time -- is initialized through :func:`woof.ingest.real.initialize_real`,
which ends with that frame's total hydrostatic pressure.  When the run carries chem
(``state.chem`` is set, i.e. ``cfg.chem_sets`` is non-empty), this module
gives every active row whose FIRST enabled boundary source is a data-store
composition source its field for that frame, through
:func:`woof.chem_boundary_remap.boundary_fields_for_grid`.  The initial
frame's fields are the initial condition; the later frames' fields are what
the lateral-boundary tables are cut from, exactly as the met fields of the
same frames are.  Frames of a 3-hourly source are blended to each frame's
valid time before the remap (WRF-Chem's boundary preprocessors do the same).

THE BYTES ARE FOUND, NEVER FETCHED.  Initialization reads the content-
addressed data-store cache (:mod:`woof.data_store_fetch`): every published
file sits beside a receipt that records the exact request it answers.  This
module picks the receipts whose request covers the frame (same dataset, the
variables the rows need, a cycle and lead list that bracket the valid time),
so the initialization of every frame finds the same files the fetch stage
wrote without either stage passing a file list to the other.  A frame no
receipt covers is a refusal naming the fetch command; downloading from
inside initialization would make a model run depend on the network and on a
credential halfway through.

Decoded stacks are cached next to the files, keyed by raw content, complete
selectors and actual selected Rust decoder bytes. A forcing frame reuses only
the same decode authority, including when an input is replaced at its path.

NOTHING HERE NAMES A SOURCE OR A SPECIES: which sources, which rows and
which variables are the chem table's rows and the config's
``chem_sources``.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

from woof.chem_boundary_remap import (
    BoundaryRemapError, SourceFrames, TargetGrid, boundary_fields_for_grid,
    boundary_rows, decode_source_files,
)

__all__ = ["ChemSourceUnavailable", "chem_boundary_fields",
           "data_store_cache_root", "data_store_sources",
           "fill_boundary_sources", "covering_files", "files_cycle"]

#: Environment variable naming the data-store cache (one authority:
#: :data:`woof.data_store_fetch.CACHE_ENV`).
CACHE_ENV = "WOOF_DATA_STORE_CACHE"


class ChemSourceUnavailable(BoundaryRemapError):
    """No fetched file covers a frame an enabled chem source must fill."""


def data_store_cache_root() -> Path:
    """The root the fetch stage publishes into (the same function)."""
    from woof.data_store_fetch import default_cache_root
    return default_cache_root()


def _enabled(cfg) -> tuple[str, ...]:
    """``cfg.chem_sources`` as names (a comma-separated string on RunConfig)."""
    from woof.chem_table import chem_names
    return chem_names(getattr(cfg, "chem_sources", None), "chem_sources")


def data_store_sources(table, cfg) -> tuple:
    """Enabled sources of the table that a data store serves, in cfg order."""
    enabled = _enabled(cfg)
    out = []
    for name in enabled:
        row = table.sources.get(name)
        if row is not None and row.acquisition is not None:
            out.append(row)
    return tuple(out)


def chem_boundary_fields(table, cfg) -> tuple[str, ...]:
    """State names of the transported rows this run's forcing frames fill.

    WRF-Chem's ``have_bcs_chem`` rows: every transported row whose first
    enabled boundary source is a data-store source.  Each forcing frame's
    initialization writes their field (:func:`fill_boundary_sources`), so
    the root's sealed lateral forcing carries them and their inflow is the
    source's (:func:`woof.core.chem_transport.apply_chem_flow_boundaries`).
    In the table's transported order.
    """
    enabled = _enabled(cfg)
    filled = set()
    for source_row in data_store_sources(table, cfg):
        filled.update(row.name for row in boundary_rows(
            table, source_row.name, available=enabled))
    return tuple(row.state_attr for row in table.transported
                 if row.name in filled)


def _utc(moment) -> datetime:
    if isinstance(moment, str):
        moment = datetime.fromisoformat(moment.replace("Z", "+00:00"))
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def _request_names(source_row, variables: Iterable[str]) -> dict[str, set]:
    """``{group: {request name}}`` the variables need."""
    groups: dict[str, set] = {}
    for key in variables:
        spec = source_row.variables[key]
        groups.setdefault(spec["group"], set()).add(spec["request"])
    return groups


def _needed_times(source_row, valid_time: datetime) -> tuple[datetime, ...]:
    cadence = float(source_row.time["cadence_s"])
    interp = source_row.time.get("interpolation", "linear")
    if interp == "daily_mean":
        day = valid_time.replace(hour=0, minute=0, second=0, microsecond=0)
        count = int(round(86400.0 / cadence))
        return tuple(day + timedelta(seconds=cadence * i) for i in range(count))
    epoch = valid_time.replace(hour=0, minute=0, second=0, microsecond=0)
    offset = (valid_time - epoch).total_seconds()
    lower = epoch + timedelta(seconds=cadence * (offset // cadence))
    if lower == valid_time:
        return (lower,)
    return (lower, lower + timedelta(seconds=cadence))


def _receipt_times(request: dict, keys: dict) -> tuple[datetime | None, tuple]:
    date = str(request.get(keys["date"], "")).split("/")[0]
    cycle = request.get(keys["cycle"]) or []
    if isinstance(cycle, str):
        cycle = [cycle]
    try:
        base = datetime.strptime(f"{date} {cycle[0]}", "%Y-%m-%d %H:%M").replace(
            tzinfo=timezone.utc)
    except (ValueError, IndexError):
        return None, ()
    leads = request.get(keys["lead"]) or []
    return base, tuple(base + timedelta(hours=float(h)) for h in leads)


def _area_contains(request: dict, keys: dict, bounds) -> bool:
    """Whether the request's area holds the target box (south, west, north, east)."""
    if bounds is None:
        return True
    area = request.get(keys["area"])
    order = keys.get("area_order", ["north", "west", "south", "east"])
    try:
        box = {name: float(value) for name, value in zip(order, area)}
    except (TypeError, ValueError):
        return False
    south, west, north, east = bounds
    if not (box["south"] <= south and north <= box["north"]):
        return False
    if box["west"] > box["east"]:          # the request crosses 180E
        return True
    # Longitudes on either convention (-180..180 or 0..360): move the target
    # onto the request's branch before comparing.
    shift = 360.0 * round(((box["west"] + box["east"]) - (west + east)) / 720.0)
    return box["west"] <= west + shift and east + shift <= box["east"]


def _levelled_request_keys(acquisition) -> set[str]:
    """Every request key that marks a levelled fetch of this source."""
    keys = {"model_level", "pressure_level"}
    for spec in (acquisition.get("requests") or {}).values():
        if spec.get("levels_key"):
            keys.add(spec["levels_key"])
    return keys


def covering_files(source_row, variables: Iterable[str], valid_time, *,
                   cache_root: Path | None = None, bounds=None) -> dict[str, Path]:
    """The cached files that together cover ``variables`` at ``valid_time``.

    One file per request group, from the NEWEST cycle whose receipts of
    every group carry every needed variable and every needed time over an
    area that holds ``bounds`` (the target's south, west, north, east).
    Raises :class:`ChemSourceUnavailable` naming what is missing.
    """
    root = Path(cache_root) if cache_root is not None else data_store_cache_root()
    if source_row.format == "netcdf":
        from woof.composition_source import covering_files as covering_composition
        return covering_composition(source_row, variables, valid_time, root)
    acq = source_row.acquisition
    keys = acq["keys"]
    need = _request_names(source_row, variables)
    times = _needed_times(source_row, _utc(valid_time))
    candidates: dict[datetime, dict[str, Path]] = {}
    coverage_sizes = {}
    for receipt in sorted(root.glob("*.json")) if root.is_dir() else ():
        try:
            record = json.loads(receipt.read_text(encoding="utf-8"))
            request = record["request"]
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if record.get("dataset") != acq["dataset"]:
            continue
        required = acq.get("research_only_licence")
        if required and record.get("accepted_licence") != required:
            continue
        cycle, lead_times = _receipt_times(request, keys)
        if cycle is None or not set(times) <= set(lead_times):
            continue
        if not _area_contains(request, keys, bounds):
            continue
        names = set(request.get(keys["variable"]) or ())
        for group, wanted in need.items():
            spec = acq["requests"][group]
            levels_key = spec.get("levels_key")
            if levels_key and levels_key not in request:
                continue
            # A single-level group never takes a file fetched for a levelled
            # group: the levelled file names the same variable (HRRR wrfnat's
            # PRES on hybrid levels beside PRES at the surface) but holds
            # none of its single-level records.  Breakage this prevents: the
            # surface pressure was stacked from the hybrid-level file
            # whenever its content-addressed name sorted first, "key PSFC:
            # selector matches zero records" (a rented box, 2026-10-21 06Z and
            # 2024-04-27 18Z).
            if not levels_key and any(k in request for k in
                                      _levelled_request_keys(acq)):
                continue
            if wanted <= names:
                path = receipt.with_suffix(".grib2")
                if path.is_file():
                    key = (cycle, group)
                    # At the same source cycle, avoid decoding an unrelated
                    # full-day stack when a shorter receipt covers this frame.
                    size = (len(lead_times), len(names), path.stat().st_size)
                    if key not in coverage_sizes or size < coverage_sizes[key]:
                        candidates.setdefault(cycle, {})[group] = path
                        coverage_sizes[key] = size
    for cycle in sorted(candidates, reverse=True):
        files = candidates[cycle]
        if set(files) == set(need):
            return {group: files[group] for group in sorted(files)}
    raise ChemSourceUnavailable(
        f"chem source {source_row.name!r}: no fetched file in {root} covers "
        f"{sorted(k for g in need.values() for k in g)} at "
        f"{', '.join(t.isoformat() for t in times)}. Fetch them first: "
        f"python -m woof.data_store_fetch fetch --source {source_row.name} "
        "--area S,W,N,E --start START --end END (the domain's box and the "
        "run's window); initialization never downloads.")


def files_cycle(source_row, files: dict[str, Path]) -> datetime | None:
    """The one cycle (reference time) the fetch receipts of ``files`` record.

    Each published file sits beside its receipt (same name, ``.json``),
    whose request names the date and cycle it answers.  Returns None when no
    file has a readable receipt (a composition NetCDF source, or files placed
    by hand); a source variable whose units depend on the cycle then refuses
    the read by name (:func:`woof.chem_conversions.units_epoch_scale`).
    Files of two different cycles are refused: their units could differ.
    """
    acq = source_row.acquisition or {}
    keys = acq.get("keys")
    if not keys or source_row.format == "netcdf":
        return None
    cycles = set()
    for path in files.values():
        try:
            record = json.loads(Path(path).with_suffix(".json").read_text(encoding="utf-8"))
            cycle, _ = _receipt_times(record["request"], keys)
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if cycle is not None:
            cycles.add(cycle)
    if len(cycles) > 1:
        raise ChemSourceUnavailable(
            f"chem source {source_row.name!r}: the covering files come from "
            f"cycles {sorted(c.isoformat() for c in cycles)}; one frame must be "
            "read from one cycle, whose units the source row names")
    return next(iter(cycles), None)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


_DECODED: dict[tuple, SourceFrames] = {}


def _frames(source_row, files: dict[str, Path], variables: tuple[str, ...],
            cache_root: Path) -> SourceFrames:
    from woof.chem_boundary_remap import grib2_selection, _stack_once
    from woof.grib2_stack import stack_cache_digest, stack_cache_identity
    from woof.ingest.cpu_backend import (
        cpu_bridge_scope, resolve_cpu_bridge, _selected_cpu_worker_cap)
    if source_row.format == "netcdf":
        from woof.composition_source import registered_frames
        return registered_frames(source_row, cache_root)
    if source_row.format != "grib2":
        raise BoundaryRemapError(
            f"source {source_row.name!r} is {source_row.format}; the boundary "
            "remap decodes GRIB2 sources")
    variables = tuple(variables)
    files = dict(files)
    cache_root = Path(cache_root).resolve()

    def current_authority():
        # Raw bytes precede memory admission. Fetch names identify a request,
        # and an import can replace its file at the same path.
        inputs = []
        for group, path in files.items():
            wanted = tuple(name for name in variables
                           if source_row.variables[name]["group"] == group)
            inputs.append({"group": group,
                           "decode": stack_cache_identity(path, grib2_selection(source_row, wanted))})
        acquisition = source_row.acquisition or {}
        value = {
            "schema": "gpuwm-chem-source-cache-v2",
            "source": {"name": source_row.name, "kind": source_row.kind,
                       "format": source_row.format,
                       "provider": {name: acquisition.get(name)
                                    for name in ("provider", "service", "dataset")}},
            "variables": list(variables), "inputs": inputs,
        }
        return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))

    # Every native decode/axis helper uses the resolver selection whose bytes
    # were hashed. Preserve the preparation's existing worker cap too.
    with cpu_bridge_scope(resolve_cpu_bridge(), worker_cap=_selected_cpu_worker_cap.get()):
        authority = current_authority()
        digest = stack_cache_digest(authority)
        # Lazy field paths belong to this output namespace. Input paths also
        # preserve current SourceFrames.files provenance in memory.
        key = (digest, str(cache_root),
               tuple((group, str(Path(path).resolve())) for group, path in files.items()))
        cached = _DECODED.get(key)
        expected = {(str(Path(path).resolve()), stack_cache_digest(item["decode"]["specification"])): item["decode"]
                    for path, item in zip(files.values(), authority["inputs"])}

        def admitted_stack(path, select, out):
            specification = {"schema": "gpuwm-grib2-stack-v1", "select": [dict(row) for row in select]}
            expected_key = (str(path.resolve()), stack_cache_digest(specification))
            if expected_key not in expected:
                raise BoundaryRemapError("GRIB2 selectors changed before decode")
            return _stack_once(path, select, out, expected_authority=expected[expected_key])

        work = cache_root / "decoded" / digest[:16]
        # The same atomic cache reader validates sidecars and lazy field
        # sizes/hashes on memory hits too. It performs no native re-decode
        # when the authority and payload remain unchanged.
        frames = decode_source_files(source_row, files, variables, work, stacker=admitted_stack)
        if current_authority() != authority:
            raise BoundaryRemapError("GRIB2 source authority changed during decode")
        if cached is not None:
            return cached
        _DECODED[key] = frames
        return frames


def fill_boundary_sources(state, cfg, *, valid_time, latlon, pressure,
                          cache_root: Path | None = None, backend=None,
                          frame=None) -> list[dict]:
    """Write every data-store source's rows into ``state`` for this frame.

    Returns one receipt per source (empty when the run has no chem or no such
    source).  ``latlon`` is ``(lat2d, lon2d)`` of the mass points and
    ``pressure`` the frame's ``(nz, ny, nx)`` total hydrostatic half-level
    pressure (WRF's ``p + pb``).
    """
    chem = getattr(state, "chem", None)
    if chem is None:
        return []
    sources = data_store_sources(chem.table, cfg)
    if not sources:
        return []
    if latlon is None or valid_time is None:
        raise ChemSourceUnavailable(
            "chem boundary sources need the frame's mass-point latitude/"
            "longitude and valid time, and this initialization carries "
            "neither; the chem fields cannot be placed on the grid")
    root = Path(cache_root) if cache_root is not None else data_store_cache_root()
    lat, lon = (np.asarray(a.get() if hasattr(a, "get") else a, dtype=np.float64)
                for a in latlon)
    target = TargetGrid(lat, lon, np.asarray(
        pressure.get() if hasattr(pressure, "get") else pressure,
        dtype=np.float32))
    enabled = _enabled(cfg)
    receipts = []
    for source_row in sources:
        rows = boundary_rows(chem.table, source_row.name, available=enabled,
                             frame=frame)
        if not rows:
            continue
        from woof.chem_boundary_remap import _needed_variables
        variables = _needed_variables(source_row, rows)
        bounds = (float(lat.min()), float(lon.min()),
                  float(lat.max()), float(lon.max()))
        files = covering_files(source_row, variables, valid_time,
                               cache_root=root, bounds=bounds)
        frames = _frames(source_row, files, variables, root)
        cycle = files_cycle(source_row, files)
        fields, receipt = boundary_fields_for_grid(
            chem.table, source_row, frames, target, _utc(valid_time),
            backend=backend, available=enabled, frame=frame, cycle=cycle)
        for name, values in fields.items():
            destination = getattr(state, chem.table.row(name).state_attr)
            # NumPy 2 arrays carry .device too, so ask the type's module.
            xp = (__import__("cupy")
                  if type(destination).__module__.split(".")[0] == "cupy"
                  else np)
            destination[...] = xp.asarray(values, dtype=np.float32)
        receipt["files"] = {group: str(p) for group, p in files.items()}
        receipt["files_cycle"] = None if cycle is None else cycle.isoformat()
        receipts.append(receipt)
    from woof.core.chem_mp_coupling import initialize_from_sources
    number_receipt = initialize_from_sources(state, cfg, receipts)
    if number_receipt is not None:
        for receipt in receipts:
            receipt["thompson_initialization"] = number_receipt
    # The frame's sealed lateral forcing carries these rows beside water
    # vapour (woof.ingest.lateral_bc._coupled_device_fields, uncoupled as
    # WRF-Chem's chem_b/chem_bt are), so their inflow is the source's.
    carried = chem_boundary_fields(chem.table, cfg)
    if carried:
        present = tuple(
            name for name in getattr(state, "_external_scalar_boundary_fields", ())
            if name not in carried)
        state._external_scalar_boundary_fields = (*present, *carried)
    return receipts
