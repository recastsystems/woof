"""The lateral boundary after an analysis replaces the start state (WRFDA's da_update_bc).

real.exe's first ``wrfbdy_d01`` record is always the coupling of its own wrfinput
(main/real_em.F:866-899), and the WRF-input door refuses a pair whose first record is another
atmosphere (:func:`woof.ingest.wrfinput._check_initial_boundary_pair`: a boundary from another
real.exe run silently forcing this initial state).  An analysis is exactly such a start: the
background's boundary, forcing the analysed state.  Left as it is, the specified zone snaps the
rim to the background at the first step and the relaxation zone pulls the analysed columns back
toward it through the whole first interval.

WRF's rule after an analysis replaces the start is WRFDA's ``da_update_bc``, which operational
HRRR runs after GSI: the first record becomes the analysis's own coupling and its tendency is
re-derived so the END of the first interval stays the boundary's own,
``BT' = (B + BT*dt - B') / dt``.  :func:`rebase_first_record` writes that file: the first
record's values from :func:`woof.ingest.wrfinput.wrf_initial_boundary_head` (the one
definition the door checks a pair against, and the stock-WRF exporter writes from) applied to
the start file, the tendencies re-derived in float64, every later record and every other byte
the original's.  The same rule the stock exporter applies to a replaced start
(:func:`woof.wrf_direct._wrfbdy_fields`).

A cold start inside the first boundary interval (an analysis valid between boundary records,
WRF analysis door gap G8b) is rebased the same way, onto the instant the start is valid: the
first record is restamped to the start, its values are the start's own coupling, and its
tendency is re-derived so the end of the interval is still the boundary's own,
``BT' = (B + BT*T - B') / (T - (t - t0))``.  The record then covers ``[t, t0 + T)``, a lead
interval shorter than the cadence, which the WRF-input door reads as such
(:func:`woof.ingest.wrfinput.wrfbdy_coverage`); every later record and every other byte is the
original's.  WRF's own evaluation of the record at ``t`` (``B + dtbc*BT`` from a record left at
``t0``) gives the same coupled value, but WOOF-ArWen's default boundary path re-normalises each
interval's END POINTS by the column mass at that instant and interpolates between them, so a
record left at ``t0`` with back-extrapolated values would put the rim at ``t`` somewhere other
than the start's own coupling; the restamped record has both ends at real states.  A start in a
later interval is cropped to the record that holds it first (records before it never force the
run).  The cycling-restart route reads the boundary at its own instant and the door does not
rewrite it there.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

#: wrfbdy side suffix -> woof side name.
SIDES = {"XS": "west", "XE": "east", "YS": "south", "YE": "north"}
#: The dycore core's wrfbdy base names -> the logical key the boundary head uses.
CORE_KEYS = {"U": "u", "V": "v", "T": "theta", "PH": "phi", "MU": "mu", "QVAPOR": "qv"}
_NEXT_TIME = "md___nextbdytimee_x_t_d_o_m_a_i_n_m_e_t_a_data_"
_TIME_FORMAT = "%Y-%m-%d_%H:%M:%S"


class BoundaryUpdateRefused(ValueError):
    """The boundary cannot be rebased onto this start; the message names why."""


def _first(variable) -> np.ndarray:
    variable.set_auto_mask(False)
    values = np.asarray(variable[...])
    if variable.dimensions and variable.dimensions[0] == "Time":
        values = values[0]
    return values


def _times(dataset) -> list[datetime]:
    variable = dataset.variables["Times"]
    variable.set_auto_mask(False)
    rows = np.asarray(variable[...])
    if rows.ndim == 1:
        rows = rows[None]
    out = []
    for row in rows:
        text = b"".join(bytes(c) for c in row).decode("ascii", "replace").strip("\x00 ")
        out.append(datetime.strptime(text, _TIME_FORMAT))
    return out


def boundary_keys(dataset, start=None) -> tuple[dict[str, str], dict[str, str]]:
    """``(rebased, kept)``: the tables rebuilt from the start, wrfbdy base name -> logical key,
    and the tables left as the boundary's own, base name -> why.

    Rebuilt: the dycore core and vapour (what the WRF-input door's boundary always reads), and
    every moisture species and number the start file carries with a mass-coupled boundary rule
    (the scalars the door reads when the run declares them, :func:`woof.boundary_fields.
    external_scalar_fields`).  Kept: tables with no such rule here (real.exe also writes
    ``W``, ``HT_SHAD``, ``PC`` and the MYNN ``qke_adv``; the WRF-input door reads none of them
    but ``qke_adv`` under ``bl_mynn_tkeadvect``, and an analysis moves none of them).  A kept
    table the run does read is held to the start by the door's own pair check at restore
    (named gap G8c when that check refuses one).
    """

    from woof.ingest.lateral_bc import COUPLED_SCALAR_STATE_FIELDS
    from woof.ingest.wrfinput import MOISTURE_MAP, NSSL_MOISTURE_MAP

    bases = sorted({name[:-4] for name in dataset.variables if name.endswith("_BXS")})
    names = set() if start is None else set(start.variables)
    keys, kept = {}, {}
    for base in bases:
        if base in CORE_KEYS:
            keys[base] = CORE_KEYS[base]
            continue
        state = MOISTURE_MAP.get(base) or NSSL_MOISTURE_MAP.get(base)
        if state is None or state not in COUPLED_SCALAR_STATE_FIELDS or base.upper().startswith("QKE"):
            kept[base] = ("no mass-coupled boundary rule for it here; the WRF-input door does not read it, "
                          "or (qke_adv under bl_mynn_tkeadvect) holds it to the start by its own pair check")
        elif start is not None and base not in names:
            kept[base] = "the start file does not carry it, so there is no analysed value to couple"
        else:
            keys[base] = state
    missing = sorted(set(CORE_KEYS) - set(keys))
    if missing:
        raise BoundaryUpdateRefused(f"the boundary lacks {missing}; it is not a real.exe wrfbdy")
    return keys, kept


def first_interval_seconds(dataset) -> float:
    times = _times(dataset)
    if len(times) > 1:
        return float((times[1] - times[0]).total_seconds())
    if _NEXT_TIME not in dataset.variables:
        raise BoundaryUpdateRefused("a single-record wrfbdy needs its nextbdytime to say how long the "
                                    "first interval is")
    variable = dataset.variables[_NEXT_TIME]
    variable.set_auto_mask(False)
    rows = np.asarray(variable[...])
    if rows.ndim == 1:
        rows = rows[None]
    text = b"".join(bytes(c) for c in rows[0]).decode("ascii", "replace").strip("\x00 ")
    return float((datetime.strptime(text, _TIME_FORMAT) - times[0]).total_seconds())


def rebase_first_record(wrfbdy: str | Path, start: str | Path, out: str | Path) -> dict[str, Any]:
    """``wrfbdy`` with its first record rebased onto ``start`` (a wrfinput-form file at the
    boundary's first time), written to ``out``.  Returns the receipt."""

    from woof import netcdf_bridge
    from woof.io.analysis_bundle import _patched_copy, require_classic
    from woof.ingest.wrfinput import _boundary_strip, wrf_initial_boundary_head
    from woof.wrf_direct import _side_to_wrf

    wrfbdy, start, out = Path(wrfbdy), Path(start), Path(out)
    require_classic(wrfbdy, "the boundary")
    bdy = netcdf_bridge.open_dataset(wrfbdy)
    first = netcdf_bridge.open_dataset(start)
    bdy_times = _times(bdy)
    start_times = _times(first)
    offset = float((start_times[0] - bdy_times[0]).total_seconds())
    if offset:
        # A start inside the first interval (G8b): the record is restamped to the start, see the
        # module docstring.  Any other start time is not this boundary's to rebase.
        span = first_interval_seconds(bdy)
        if not 0.0 < offset < span:
            raise BoundaryUpdateRefused(
                f"the start is valid {start_times[0]:%Y-%m-%d_%H:%M:%S} and the boundary's first record "
                f"{bdy_times[0]:%Y-%m-%d_%H:%M:%S} covers {span:g} s from there; a cold start from an "
                "analysis needs the boundary cropped to the record whose interval holds the analysis "
                "time (a start before the first record, or after its interval, has no record to rebase)")
    for name in ("USE_THETA_M", "GRID_ID", "DX", "DY"):
        if float(first.getncattr(name)) != float(bdy.getncattr(name)):
            raise BoundaryUpdateRefused(f"the start says {name} {first.getncattr(name)} and the boundary "
                                        f"{bdy.getncattr(name)}; they are not one run's files")
    keys, kept = boundary_keys(bdy, first)
    width = len(bdy.dimensions["bdy_width"])
    duration = first_interval_seconds(bdy)
    if not duration > 0:
        raise BoundaryUpdateRefused(f"the first boundary interval is {duration} s long")
    lead = duration - offset
    needed = set(keys) | {"MU", "MUB", "C1H", "C2H", "C1F", "C2F", "MAPFAC_M", "MAPFAC_U", "MAPFAC_V",
                          "QVAPOR"}
    missing = sorted(name for name in needed if name not in first.variables)
    if missing:
        raise BoundaryUpdateRefused(f"{start.name} lacks {missing}, which the boundary coupling reads")
    # unmasked BEFORE the batch: _first switches each variable to set_auto_mask(False), which drops a cache
    # decoded masked, so a masked prefetch was thrown away and every variable was decoded again on its own
    for name in sorted(needed):
        first.variables[name].set_auto_mask(False)
    first.prefetch(sorted(needed))
    raw = {name: np.asarray(_first(first.variables[name]), dtype=np.float32) for name in needed}
    layouts = {key: (base,) for base, key in keys.items()}
    moist_theta = int(first.getncattr("USE_THETA_M")) == 1
    head = wrf_initial_boundary_head(raw, layouts, moist_theta=moist_theta)
    tables = [f"{base}_B{suffix}" for base in keys for suffix in SIDES]
    tables += [f"{base}_BT{suffix}" for base in keys for suffix in SIDES]
    # unmasked BEFORE the batch: _first switches each variable to set_auto_mask(False), which drops a cache
    # decoded masked, so a masked prefetch was thrown away and every variable was decoded again on its own
    for name in tables:
        bdy.variables[name].set_auto_mask(False)
    bdy.prefetch(tables)
    updates: dict[str, np.ndarray] = {}
    departure: dict[str, float] = {}
    for base, key in keys.items():
        largest = 0.0
        for suffix, side in SIDES.items():
            value = np.asarray(_first(bdy.variables[f"{base}_B{suffix}"]), dtype=np.float64)
            tendency = np.asarray(_first(bdy.variables[f"{base}_BT{suffix}"]), dtype=np.float64)
            strip = _side_to_wrf(_boundary_strip(head[key], side, width), key, side)
            strip = np.asarray(strip, dtype=np.float32)
            if strip.shape != value.shape:
                raise BoundaryUpdateRefused(
                    f"{base} {side}: the start's strip is {strip.shape}, the boundary table "
                    f"{value.shape}; the start is not on the boundary's grid")
            end = value + duration * tendency
            updates[f"{base}_B{suffix}"] = strip
            updates[f"{base}_BT{suffix}"] = ((end - strip.astype(np.float64)) / lead).astype(np.float32)
            at_start = value + offset * tendency if offset else value
            largest = max(largest, float(np.max(np.abs(strip.astype(np.float64) - at_start))))
        departure[base] = largest
    receipt = {"schema": "gpuwm.wrfbdy-rebase/1", "boundary": str(out), "start": str(start),
               "rule": "first record = coupling of the start (wrf_initial_boundary_head); BT' = (B + BT*dt - B')/dt, "
                       "so the end of the first interval is the boundary's own (WRFDA da_update_bc)",
               "first_interval_seconds": duration, "fields": sorted(keys), "kept": kept,
               "max_abs_first_record_change": departure, "moist_theta": moist_theta}
    if offset:
        stamp = np.frombuffer(f"{start_times[0]:%Y-%m-%d_%H:%M:%S}".encode("ascii"), dtype="S1")
        for name in ("Times", "md___thisbdytimee_x_t_d_o_m_a_i_n_m_e_t_a_data_"):
            if name in bdy.variables:
                updates[name] = stamp
        receipt.update(
            rule=("a start inside the first interval: the first record restamped to the start, its values the "
                  "coupling of the start (wrf_initial_boundary_head), BT' = (B + BT*T - B')/(T - (t - t0)), so the "
                  "end of the interval is the boundary's own; every later record the original's"),
            start_inside_interval={"record_time": f"{bdy_times[0]:%Y-%m-%d_%H:%M:%S}",
                                   "start_time": f"{start_times[0]:%Y-%m-%d_%H:%M:%S}",
                                   "offset_seconds": offset, "lead_interval_seconds": lead},
            max_abs_first_record_change_note=("against the original record carried to the start by its own "
                                              "tendency (B + BT*(t - t0))"))
    _patched_copy(wrfbdy, out, updates)
    return receipt


def same_start(run_wrfinput: str | Path, start: str | Path) -> bool:
    """Whether ``start`` is byte for byte the wrfinput the boundary was made with."""

    from woof.io.analysis_bundle import sha256_file

    run_wrfinput, start = Path(run_wrfinput), Path(start)
    if run_wrfinput.stat().st_size != start.stat().st_size:
        return False
    return sha256_file(run_wrfinput) == sha256_file(start)


__all__ = ["BoundaryUpdateRefused", "boundary_keys", "first_interval_seconds", "rebase_first_record",
           "same_start"]
