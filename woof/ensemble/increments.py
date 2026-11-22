"""The DA interface: applying analysis increments (EXPERIMENTAL).

The assimilation step -- whoever implements it -- hands the engine a
plain ``dict[field_name -> ndarray]`` whose arrays match the state's
shapes exactly.  This module is the only thing that touches state on the
DA side, and it is deliberately dumb: add, check, receipt.  It does no
balancing, no localisation, no covariance work of any kind.

Increments are checked before anything is written: an unknown field, a
shape mismatch, a non-finite value, or a dtype that is not a float kind
is a refusal, not a warning.  Half-applied increments are the one
outcome that would silently corrupt a cycling ensemble, so the checks
run over the whole dict first and only then does the apply loop start.

**Finiteness is checked where the value lands, not only where it came
from.**  A ``float64`` increment of ``1e300`` is finite; cast to the
``float32`` a state field actually stores it is ``inf``.  So is a
``float32``-representable increment added to a large enough background.
Validating only the source dtype let both through and wrote ``inf`` into
the analysis, which is exactly the outcome the "a non-finite value is a
refusal" contract exists to prevent.  Every write path therefore checks
the cast addend AND the sum, in the target's own dtype, before anything
is stored -- and computes the sum into a temporary first, so a refusal
leaves the target byte-identical rather than half-poisoned.

**Moment consistency is checked here for the same reason finiteness is.**
This module is the only thing that writes a DA analysis, so it is the
only place a guard cannot be skipped by using a different driver.  A
multi-moment state's mass and number are a pair; an update that moves one
and not the other produces cells holding ``q > 0`` with ``N = 0``, which
the scheme's slope closure evaluates to NaN and the reflectivity operator
correctly refuses to smooth over.  Every write therefore validates the
increment's field set against the pairs the background actually carries
and, after the sum, repairs or refuses the pairs the analysis broke --
see :mod:`woof.da.moments` for the authority each scheme's repair comes
from and for the real-radar cycle that made this necessary.

**Vapour is capped at saturation where the increment moved it.**  An
ensemble increment of theta and vapour knows nothing of the saturation
curve, and nothing downstream of this writer took the excess back until
the scheme's first step condensed it.  A storm-scale child's first radar
analysis left 2.8 to 6.9 Mt of vapour above liquid saturation in a
3,449 km2 box whose backgrounds held under 0.01 Mt; condensed at the first
step it warmed 1 to 3 km by 1.4 to 2.1 K over the whole box, the likely
source of the updrafts that followed.  So
every write here caps the resulting vapour at saturation over liquid water
at the resulting temperature (:func:`_saturation_cap`), at the cells whose
theta or vapour the increment moved, and never below the background's own
supersaturation ratio there (the scheme's state between two of its calls
is not the analysis's to rewrite).  The excess is removed, not moved into
cloud: condensing it into cloud water conserves water only by leaving its
latent heat out, and putting that heat back is the very warming the cap
exists to prevent.  The receipt records the cells and the vapour removed,
and a caller can record it as its own stage (``saturation_observer``).
The ensemble analysis keeps its mean under the same limit first
(:func:`mean_preserving_saturation_bound`), so on an analysed ensemble this
cap is the last guard, not a one-signed sink.
"""

from __future__ import annotations

import hashlib
import os
import uuid
from pathlib import Path
from typing import Mapping

import numpy as np
from woof.ensemble.state_sha import _host as _host

#: Versioned contract label recorded in every increment receipt.
INCREMENT_CONTRACT = "gpuwm-da-increments.v1"

#: Receipt schema of the saturation cap (:func:`_saturation_cap`).
SATURATION_CAP_SCHEMA = "gpuwm-da.saturation-cap.v1"

#: What the cap reads off a state or checkpoint: full pressure, the
#: inverse dry density and vapour give the background's temperature
#: through the equation of state, so no setup array (``thb``) is needed.
SATURATION_CAP_READS = ("p", "alt", "qv")

#: Cells per block of the cap's arithmetic (:func:`_saturation_cap`), so
#: its float64 scratch is a few arrays of the moved cells of one block on
#: the state's device, whatever the grid.
SATURATION_CAP_BLOCK_CELLS = 1 << 18


class SaturationCap:
    """What the cap writes: the flat cell ``index`` (on the state's array
    module) and the capped vapour ``limit`` there (float64)."""

    __slots__ = ("index", "limit")

    def __init__(self, index, limit):
        self.index = index
        self.limit = limit

    def write(self, vapour):
        """Write the capped vapour into ``vapour`` (a fresh, C-contiguous
        array the writer owns, in the state's dtype) in place; return the
        change it made at ``index`` as host float64, measured on the
        stored values (never positive)."""
        if not vapour.flags.c_contiguous:
            raise ValueError("the saturation cap writes in place and needs "
                             "a C-contiguous vapour array the writer owns")
        flat = vapour.reshape(-1)
        before = flat[self.index]
        flat[self.index] = self.limit.astype(vapour.dtype)
        return _host(flat[self.index]).astype(np.float64) \
            - _host(before).astype(np.float64)

    def host_index(self):
        return _host(self.index)


def _flat(values):
    """A flat view of an array (a copy only if it is not contiguous)."""
    return values.reshape(-1)


def _to_module(xp, values):
    """``values`` in the array module ``xp``."""
    if xp is np and hasattr(values, "__cuda_array_interface__"):
        return values.get()
    return xp.asarray(values)


def _saturation_cap(read, staged, checked):
    """``(SaturationCap or None, receipt)`` for one write.

    ``read(name)`` returns the BACKGROUND's array for a state field (or
    ``None``); ``staged`` holds the resulting arrays of the fields the
    increment names; ``checked`` is the increment.  The background's
    temperature is the equation of state's, ``T = p * alt / (Rd * (1 +
    (Rv/Rd) qv))`` (the form :func:`woof.core.diagnostics.update_diagnostics`
    evaluates, inverted; it agrees with ``(thb + thp) * Pi`` to 8e-5 K on
    storm-scale child members), so a checkpoint without the setup
    ``thb`` is capped exactly as a live state is.  Its theta is ``T / Pi``
    with ``Pi = (p/P0)**(Rd/cp)``, and the resulting temperature is
    ``(theta + dthp) * Pi`` at the background's pressure (the write
    precedes the refresh that moves ``p``).  Saturation is over liquid
    water (:func:`liquid_saturation_mixing_ratio`, Bolton), the
    bound a scheme's own condensation acts on; supersaturation over ice is
    left to the scheme.  The limit at a cell is that saturation times the
    background's own supersaturation ratio where it held one, and the cap
    acts only at cells where the increment moved theta or vapour.

    The arithmetic runs on the moved cells only, block by block
    (:data:`SATURATION_CAP_BLOCK_CELLS`), and hands back the capped cells
    and their values rather than a whole field: the first version formed
    about ten whole-grid float64 temporaries on the state's device inside
    every write (about 480 MB each on a 1000x1000x60 child, 5 GB beside a
    member pool of three on one card).
    """
    receipt = {"schema": SATURATION_CAP_SCHEMA,
               "rule": ("resulting vapour at most saturation over liquid "
                        "water at the resulting temperature, or the "
                        "background's own supersaturation ratio times it "
                        "where the background held one, at the cells whose "
                        "thp or qv the increment moved; the excess is "
                        "removed"),
               "reference": ("liquid water, Bolton "
                             "(woof.ensemble.increments.liquid_saturation_mixing_ratio)"),
               "temperature": ("equation of state of the background, "
                               "T = p*alt/(Rd*(1+(Rv/Rd)*qv)), plus the "
                               "increment's thp times the Exner function "
                               "at the background's pressure")}
    if "qv" not in checked and "thp" not in checked:
        return None, {**receipt, "evaluated": False,
                      "reason": "the increment moves neither thp nor qv"}
    arrays = {name: read(name) for name in SATURATION_CAP_READS}
    missing = sorted(name for name, value in arrays.items() if value is None)
    if missing:
        return None, {**receipt, "evaluated": False,
                      "reason": (f"the state carries no {missing}; the "
                                 "background temperature cannot be formed")}
    xp = _array_module(arrays["qv"])
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return _saturation_cap_blocks(xp, arrays, staged, checked, receipt)


def liquid_saturation_mixing_ratio(temperature, pressure):
    """Saturation mixing ratio over liquid water (kg/kg), Bolton/Tetens.

    ``es = 1000*SVP1*exp(SVP2*(T-SVPT0)/(T-SVP3))`` and ``qs = EP2*es/(p - es)``,
    with the denominator held at 1 Pa or more so a cell whose saturation
    vapour pressure exceeds its pressure never gives a negative ratio.  The
    same arithmetic as the liquid phase of the reflectivity nudging module
    that 3.0.3 removed with the local data assimilation.
    """
    from woof.core import constants as c

    xp = _array_module(temperature)
    es = (1000.0 * c.SVP1
          * xp.exp(c.SVP2 * (temperature - c.SVPT0) / (temperature - c.SVP3)))
    return c.EP2 * es / xp.maximum(pressure - es, 1.0)


def saturation_limit(xp, pressure, inverse_density, vapour, theta_increment,
                     *, with_ratio=False):
    """The most vapour a cell may hold after an increment, float64 in ``xp``.

    The background's temperature is the equation of state's,
    ``T = p * alt / (Rd * (1 + (Rv/Rd) qv))``; the resulting temperature is
    that plus the theta increment times the Exner function at the
    background's pressure (``theta_increment`` ``None``: unchanged).  The
    limit is liquid saturation at the resulting temperature times the
    background's own supersaturation ratio where it held one.  ONE owner of
    the formula: the applier's cap (:func:`_saturation_cap`) and the
    analysis's ensemble bound (:func:`mean_preserving_saturation_bound`)
    both read it here, so a member the bound left at its limit is a member
    the cap finds at its limit.
    """
    from woof.core import constants as c

    p = pressure.astype(xp.float64)
    q0 = vapour.astype(xp.float64)
    temperature_before = p * inverse_density.astype(xp.float64) / (
        c.RD * (1.0 + c.RVOVRD * q0))
    temperature_after = temperature_before
    if theta_increment is not None:
        temperature_after = (temperature_before
                             + theta_increment.astype(xp.float64)
                             * (p / c.P0) ** c.RCP)
    ratio = xp.maximum(
        q0 / liquid_saturation_mixing_ratio(temperature_before, p), 1.0)
    limit = liquid_saturation_mixing_ratio(temperature_after, p) * ratio
    return (limit, ratio) if with_ratio else limit


#: Receipt schema of the analysis's ensemble saturation bound
#: (:func:`mean_preserving_saturation_bound`).
SATURATION_BOUND_SCHEMA = "gpuwm-da.saturation-bound.v1"


def mean_preserving_saturation_bound(prior_vapour, vapour_increment,
                                     pressure, inverse_density,
                                     theta_increment=None):
    """``(vapour increment, receipt)``: the ensemble's analysed vapour held
    at or below each member's saturation limit while the filter's own
    ensemble-mean analysis is kept.

    Why.  The applier's cap
    (:func:`_saturation_cap`) bounds ONE member at a time: at the cells a
    member's analysed vapour exceeds its limit it removes the excess, and
    at the cells a member is below its limit it adds nothing.  An analysis
    spreads its members about its mean, so wherever the mean sits near
    saturation (the top of an afternoon mixed layer, every cloud) the cap
    cuts the upper tail and keeps the lower one: a one-signed sink of water
    at every analysis, in every member and never in the unanalysed control.
    On a four-minute storm-scale cycle's 1 km child it removed 30 to 44
    g m-2 per member per analysis (1.1 to 1.5 kg m-2 over thirty analyses),
    most of it at 1 to 2.5 km, and the members' 0-2 km air ended 0.6 to 1.2
    g/kg drier than the unanalysed control's (2 m dewpoint 2.6 K below 24
    warm-sector surface stations).  Only about a quarter of it was vapour
    the ensemble mean itself held above saturation.

    The bound is a mean-preserving rule applied to the headroom
    ``s = limit - analysed vapour``, which must be non-negative: wherever
    any member's headroom is negative, every member's non-negative headroom
    is scaled by one factor so the ensemble keeps the filter's mean
    headroom -- and, the limits being fixed, the filter's mean vapour --
    and a member over its limit ends exactly at it.  Where the filter's
    mean is itself over the mean limit every member ends at its limit,
    which is what the cap does for the supersaturation it was installed for
    (2.8 to 6.9 Mt above saturation from a radial-velocity batch's
    vapour).  The limit is
    :func:`saturation_limit`, the cap's own.

    ``prior_vapour``, ``vapour_increment``, ``pressure`` and
    ``inverse_density`` are ``(R, nz, ny, nx)`` host arrays with ``R`` of
    two or more; ``theta_increment`` likewise or ``None``.  The returned
    increment is a new array in ``vapour_increment``'s dtype; cells no
    member exceeds come back bit for bit.
    """
    increment = np.asarray(vapour_increment)
    if increment.ndim != 4 or increment.shape[0] < 2:
        raise ValueError(
            f"the ensemble saturation bound keeps the mean over a leading "
            f"member axis and this increment is {increment.shape}; bound one "
            "member with the applier's cap instead")
    members = int(increment.shape[0])
    receipt = {"schema": SATURATION_BOUND_SCHEMA,
               "rule": ("a mean-preserving rule on the "
                        "headroom (saturation limit minus analysed vapour): "
                        "members over their limit end at it and the vapour "
                        "goes to the members with headroom at that cell, so "
                        "the filter's ensemble-mean vapour is kept unless "
                        "that mean is itself over the mean limit. A cell no "
                        "member's vapour or theta increment moved is never "
                        "written or counted: member_cells_over_limit and "
                        "cells_touched count moved cells only"),
               "limit": "woof.ensemble.increments.saturation_limit",
               "members": members}
    # Block by block, on the cells some member's increment moved (a cell no
    # increment moved holds its background vapour, which its own limit
    # admits): a block is rows of one level, or whole levels on a small
    # grid, holding about SATURATION_CAP_BLOCK_CELLS member values, so the
    # float64 scratch is a few (members, moved cells) arrays of one block
    # whatever the grid.  The first version formed whole-grid float64
    # temporaries per member, twice (about 760 MB each per member on a
    # 1799x1059x50 grid).
    prior_vapour = np.asarray(prior_vapour)
    pressure = np.asarray(pressure)
    inverse_density = np.asarray(inverse_density)
    theta = None if theta_increment is None else np.asarray(theta_increment)
    nz, ny, nx = (int(n) for n in increment.shape[1:])
    rows = max(1, SATURATION_CAP_BLOCK_CELLS // max(1, nx * members))
    if rows >= ny:
        levels = max(1, rows // ny)
        spans = [(k0, min(nz, k0 + levels), 0, ny)
                 for k0 in range(0, nz, levels)]
    else:
        spans = [(k0, k0 + 1, j0, min(ny, j0 + rows))
                 for k0 in range(nz) for j0 in range(0, ny, rows)]
    bounded = None
    over_cells = touched_cells = mean_over = 0
    per_member_removal = removed = 0.0
    for k0, k1, j0, j1 in spans:
        moved = np.any(increment[:, k0:k1, j0:j1] != 0, axis=0)
        if theta is not None:
            moved |= np.any(theta[:, k0:k1, j0:j1] != 0, axis=0)
        k, j, i = np.nonzero(moved)
        if not k.size:
            continue
        k, j = k + k0, j + j0
        q0 = prior_vapour[:, k, j, i].astype(np.float64)
        dq = increment[:, k, j, i].astype(np.float64)
        limit = saturation_limit(
            np, pressure[:, k, j, i], inverse_density[:, k, j, i], q0,
            None if theta is None else theta[:, k, j, i])
        room = limit - (q0 + dq)
        over = room < 0.0
        count = int(np.count_nonzero(over))
        if not count:
            continue
        over_cells += count
        per_member_removal += float(-room[over].sum())
        hit = over.any(axis=0)
        touched_cells += int(np.count_nonzero(hit))
        base, limits, room = q0[:, hit], limit[:, hit], room[:, hit]
        mean = room.mean(axis=0)
        held = np.maximum(room, 0.0)
        held_mean = held.mean(axis=0)
        scale = np.where(mean > 0.0,
                         mean / np.where(held_mean > 0.0, held_mean, 1.0), 0.0)
        np.minimum(scale, 1.0, out=scale)
        if bounded is None:
            bounded = np.array(increment, copy=True)
        bounded[:, k[hit], j[hit], i[hit]] = (
            limits - scale[None, :] * held - base).astype(increment.dtype)
        removed += float(members * np.maximum(-mean, 0.0).sum())
        mean_over += int(np.count_nonzero(mean <= 0.0))
    receipt.update(member_cells_over_limit=over_cells,
                   cells_touched=touched_cells,
                   per_member_cap_would_remove_kg_kg_sum=per_member_removal)
    if bounded is None:
        receipt.update(vapour_removed_kg_kg_sum=0.0,
                       vapour_returned_kg_kg_sum=0.0,
                       cells_mean_over_limit=0)
        return increment, receipt
    receipt.update(vapour_removed_kg_kg_sum=removed,
                   vapour_returned_kg_kg_sum=per_member_removal - removed,
                   cells_mean_over_limit=mean_over)
    return bounded, receipt


def _saturation_cap_blocks(xp, arrays, staged, checked, receipt):
    """The cap's arithmetic (:func:`_saturation_cap`), in ``xp``, on the
    moved cells of each block of the flat grid."""
    pressure = _flat(arrays["p"])
    inverse_density = _flat(arrays["alt"])
    vapour_before = _flat(arrays["qv"])
    moves = [_flat(checked[name]) for name in ("thp", "qv") if name in checked]
    theta_increment = _flat(checked["thp"]) if "thp" in checked else None
    vapour_after = _flat(staged["qv"]) if "qv" in checked else None
    size = int(vapour_before.size)
    moved_total = without_state = kept_super = 0
    removed_sum = removed_max = 0.0
    indices, limits = [], []
    for start in range(0, size, SATURATION_CAP_BLOCK_CELLS):
        stop = min(start + SATURATION_CAP_BLOCK_CELLS, size)
        moved = None
        for values in moves:
            here = _to_module(xp, values[start:stop]) != 0
            moved = here if moved is None else moved | here
        cells = xp.flatnonzero(moved)
        if not int(cells.size):
            continue
        p = pressure[start:stop][cells].astype(xp.float64)
        alt = inverse_density[start:stop][cells].astype(xp.float64)
        # A cell with no positive pressure or volume has no temperature to
        # saturate at (a synthetic state of zeros); it is counted, not capped.
        thermodynamic = (p > 0.0) & (alt > 0.0)
        without_state += int(xp.count_nonzero(~thermodynamic))
        if not bool(thermodynamic.all()):
            cells, p, alt = (cells[thermodynamic], p[thermodynamic],
                             alt[thermodynamic])
        moved_total += int(cells.size)
        if not int(cells.size):
            continue
        q0 = vapour_before[start:stop][cells].astype(xp.float64)
        dthp = None
        if theta_increment is not None:
            dthp = _to_module(xp, theta_increment[start:stop])[cells]
        q1 = (vapour_after[start:stop][cells].astype(xp.float64)
              if vapour_after is not None else q0)
        limit, ratio = saturation_limit(xp, p, alt, q0, dthp, with_ratio=True)
        capped = q1 > limit
        count = int(xp.count_nonzero(capped))
        if not count:
            continue
        removed = (q1 - limit)[capped]
        removed_sum += float(removed.sum())
        removed_max = max(removed_max, float(removed.max()))
        kept_super += int(xp.count_nonzero(ratio[capped] > 1.0))
        indices.append(cells[capped].astype(xp.int64) + start)
        limits.append(limit[capped])
    capped_total = sum(int(part.size) for part in indices)
    receipt["cells_without_thermodynamic_state"] = without_state
    receipt.update({"evaluated": True, "cells_moved": moved_total,
                    "cells_capped": capped_total})
    if not capped_total:
        receipt.update({"vapour_removed_kg_kg_sum": 0.0,
                        "vapour_removed_kg_kg_max": 0.0})
        return None, receipt
    receipt.update({
        "vapour_removed_kg_kg_sum": removed_sum,
        "vapour_removed_kg_kg_max": removed_max,
        "background_supersaturated_cells_kept": kept_super,
    })
    return SaturationCap(xp.concatenate(indices), xp.concatenate(limits)), \
        receipt


def _moment_guard(resulting: Mapping[str, object],
                  updated_fields, available_fields, *,
                  moment_policy: str, moment_repair: bool,
                  mp_physics: int | None, morr_rimed_ice: int,
                  where: str) -> tuple[dict, dict]:
    """``(number fields to overwrite, receipt)`` for one analysis.

    Three checks, in the order the failure happens.  First the field set:
    an update that moves a mass field and leaves a paired moment the
    background carries is refused under ``full-moment`` before anything
    is written.  Then the RESULT, in two parts.

    A moment that is not FINITE where the scheme reads it is refused
    whatever the policy and whatever ``moment_repair`` says, because no
    limiter repairs a NaN and IEEE comparison hides one from every
    ordering test the depleted-number check makes: ``qr = 1e-3`` beside
    ``nr = NaN`` is not above-threshold-with-a-non-positive-number, and
    was attested consistent.  ``_require_finite_result`` does not cover
    it either -- it checks the fields the INCREMENT names, and the
    pair's other half is exactly the field the increment did not name.

    Then the depleted pairs: a cell holding mass above the scheme's
    activity threshold with a non-positive number is repaired through the
    scheme's own limiter, or -- with the repair switched off, or for a
    scheme whose limiter is not ported -- is a refusal naming the counts.
    """
    from woof.da.moments import (moment_consistency_report,
                                  nonfinite_moment_refusal, pairs_present,
                                  repair_moments, validate_analysis_fields)

    policy_receipt = validate_analysis_fields(
        tuple(updated_fields), available=tuple(available_fields),
        mp_physics=mp_physics, policy=moment_policy)
    pairs = pairs_present(tuple(available_fields), mp_physics=mp_physics)
    if not pairs:
        return {}, {**policy_receipt, "pairs_checked": [],
                    "offending_cells_total": 0, "nonfinite_cells_total": 0,
                    "consistent": True, "repaired": False}
    finite_report = moment_consistency_report(
        resulting, mp_physics=mp_physics, pairs=pairs)
    if finite_report["nonfinite_cells_total"]:
        raise ValueError(nonfinite_moment_refusal(finite_report, where=where))
    if not moment_repair:
        report = finite_report
        if not report["consistent"]:
            raise ValueError(
                f"the analysis for {where} leaves "
                f"{report['offending_cells_total']} cell(s) holding mass "
                f"above {report['q_threshold_kg_kg']:g} kg/kg with a number "
                "moment at or below zero ("
                + ", ".join(
                    f"{entry['mass_field']}: {entry['offending_cells']}"
                    for entry in report["species"]
                    if entry["offending_cells"])
                + "), and moment_repair is off. That state is one the "
                "scheme's own slope closure evaluates to NaN, so the "
                "reflectivity operator will refuse it rather than invent a "
                "clear-air floor. Enable the repair or analyse the number "
                "moments with the mass.")
        return {}, {**policy_receipt, **report, "repaired": False}
    repaired, report = repair_moments(resulting, mp_physics=mp_physics,
                                      pairs=pairs,
                                      morr_rimed_ice=morr_rimed_ice)
    return repaired, {**policy_receipt, **report}


def _array_sha256(value) -> str:
    host = _host(value)
    digest = hashlib.sha256()
    digest.update(str(host.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(repr(host.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(host.tobytes())
    return digest.hexdigest()


def _validate(name: str, target, increment, *, where: str) -> np.ndarray:
    if target is None:
        raise ValueError(
            f"increment names field {name!r}, which {where} does not "
            "carry; the DA interface is keyed by prognostic field name "
            "and every key must resolve")
    checked = np.asarray(increment)
    target_shape = tuple(np.shape(target))
    if checked.shape != target_shape:
        raise ValueError(
            f"increment for {name!r} has shape {checked.shape}, but "
            f"{where} field has shape {target_shape}; increments must "
            "match the state's shapes exactly")
    if checked.dtype.kind != "f":
        raise ValueError(
            f"increment for {name!r} has dtype {checked.dtype}; "
            "increments must be a float kind")
    if not np.all(np.isfinite(checked)):
        raise ValueError(
            f"increment for {name!r} contains non-finite values; a NaN "
            "or Inf increment would poison the whole member")
    return checked


def _nonfinite_count(value) -> int:
    """How many entries of ``value`` are not finite.  Backend-agnostic."""
    module = _array_module(value)
    return int(module.count_nonzero(~module.isfinite(value)))


def _array_module(value):
    if hasattr(value, "__cuda_array_interface__"):
        import cupy as cp

        return cp
    return np


def _require_finite_result(name: str, target, addend, updated, *,
                           where: str) -> None:
    """Refuse an addend or a sum that is not finite in the target's dtype.

    ``_validate`` proved the increment finite in the dtype the caller
    handed over.  This proves it is still finite after the cast the write
    performs, and that the sum it produces is finite too -- an increment
    that overflows on cast, or that overflows against the background it
    lands on, is a poisoned analysis and not a rounding detail.
    """
    dtype = getattr(target, "dtype", None)
    cast_bad = _nonfinite_count(addend)
    if cast_bad:
        raise ValueError(
            f"increment for {name!r} is finite as given but has "
            f"{cast_bad} non-finite value(s) once cast to the {dtype} of "
            f"{where} field; a cast that overflows is a refusal, not a "
            "silent inf")
    sum_bad = _nonfinite_count(updated)
    if sum_bad:
        raise ValueError(
            f"increment for {name!r} is finite and casts finitely, but "
            f"adding it to {where} field produces {sum_bad} non-finite "
            "value(s); the analysis would carry inf/nan into the next "
            "leg, so this is a refusal")


def _sorted_items(increments: Mapping[str, object]):
    if not isinstance(increments, Mapping):
        raise TypeError(
            "increments must be a mapping of field name to ndarray, got "
            f"{type(increments).__name__}")
    if not increments:
        raise ValueError(
            "increments mapping is empty; an assimilation step that "
            "changes nothing must be recorded as such by its caller, not "
            "passed here as an empty dict")
    # Sorted so the receipt is order-independent: two callers handing the
    # same increments in different dict orders get the same receipt.
    return sorted(increments.items())


def apply_increments(state, increments: Mapping[str, object], *,
                     moment_policy: str = "full-moment",
                     moment_repair: bool = True,
                     mp_physics: int | None = None,
                     morr_rimed_ice: int = 1,
                     saturation_observer=None) -> dict:
    """Add ``increments`` to the matching attributes of a live ``state``.

    Returns a receipt: per-field increment sha256 and the resulting
    field sha256, plus the pre/post whole-state hashes, the
    moment-consistency block and the saturation cap's block
    (``saturation``).  ``saturation_observer``, when given, is called with
    ``{"qv": change}`` (host float64, the change the cap made on top of the
    increment, never positive) whenever the cap wrote a cell, so a caller
    can record it as its own stage.
    """
    from woof.ensemble.state_sha import (live_state_sha256,
                                          serialized_state_attrs)

    items = _sorted_items(increments)
    checked = {}
    for name, increment in items:
        target = getattr(state, name, None)
        checked[name] = _validate(name, target, increment, where="the state")

    # Two passes: prove every field's cast addend and resulting sum are
    # finite before ANY field is written.  A refusal on the last field
    # must not leave the first three applied.
    staged = {}
    # The overflow is the thing being detected, so numpy's warning about
    # it is noise: the refusal below is the report.
    with np.errstate(over="ignore", invalid="ignore"):
        for name, increment in checked.items():
            target = getattr(state, name)
            addend = _as_target_array(target, increment)
            updated = target + addend
            _require_finite_result(name, target, addend, updated,
                                   where="the state")
            staged[name] = updated

    # Vapour at saturation, on the result, before the moment guard reads
    # it and before anything is written.
    cap, saturation = _saturation_cap(
        lambda name: getattr(state, name, None), staged, checked)
    saturation_write = None
    if cap is not None:
        target = getattr(state, "qv")
        # The resulting vapour the writer owns: the staged sum, or a copy
        # of the state's when a theta increment alone made the cap write.
        vapour = staged["qv"] if "qv" in staged else target.copy()
        change = cap.write(vapour)
        if saturation_observer is not None:
            observed = np.zeros(int(np.prod(np.shape(target))),
                                dtype=np.float64)
            observed[cap.host_index()] = change
            saturation_observer({"qv": observed.reshape(np.shape(target))})
        if "qv" not in staged:
            saturation_write = vapour

    # The moment guard sees the RESULT, not the increment: a pair breaks
    # where prior + increment lands, and the whole point is the cell the
    # background left clear.
    available = tuple(name for name in serialized_state_attrs()
                      if getattr(state, name, None) is not None)
    resulting = {name: staged.get(name, getattr(state, name))
                 for name in available}
    repaired, moments = _moment_guard(
        resulting, tuple(checked), available,
        moment_policy=moment_policy, moment_repair=moment_repair,
        mp_physics=mp_physics, morr_rimed_ice=morr_rimed_ice,
        where="this state")

    before = live_state_sha256(state)
    fields = []
    for name, increment in checked.items():
        target = getattr(state, name)
        target[...] = staged[name]
        fields.append({
            "field": name,
            "shape": list(np.shape(target)),
            "increment_sha256": _array_sha256(increment),
            "field_sha256": _array_sha256(target),
        })
    if saturation_write is not None:
        # A theta increment alone cooled cells past saturation: the cap
        # writes vapour the increment did not name.
        target = getattr(state, "qv")
        target[...] = saturation_write
        saturation["qv_written_beyond_the_increment"] = True
        saturation["qv_field_sha256"] = _array_sha256(target)
    for name, values in repaired.items():
        target = getattr(state, name)
        target[...] = _as_target_array(target, values)
    return {
        "contract": INCREMENT_CONTRACT,
        "stability": "experimental",
        "field_count": len(fields),
        "fields": fields,
        "moments": moments,
        "saturation": saturation,
        "state_sha256_before": before,
        "state_sha256_after": live_state_sha256(state),
    }


def _as_target_array(target, increment: np.ndarray):
    """``increment`` cast to the target's dtype and memory space."""
    dtype = getattr(target, "dtype", None)
    if hasattr(target, "__cuda_array_interface__"):
        import cupy as cp

        return cp.asarray(increment, dtype=dtype)
    return np.asarray(increment, dtype=dtype)


#: Suffix of the file an unpublished analysis is staged at.  Named, not
#: incidental: :func:`publish_staged_analysis` is the only thing that
#: turns one into an ``analysis.npz``, and a leftover ``.staged`` file is
#: the visible trace of a crash between staging and publication.
STAGED_SUFFIX = ".staged"


def publish_staged_analysis(staged: str | Path,
                            destination: str | Path) -> Path:
    """Rename a staged analysis into place.  The commit half of the seam."""
    src = Path(staged)
    dst = Path(destination)
    if not src.is_file():
        raise ValueError(
            f"no staged analysis at {src}; nothing to publish")
    src.replace(dst)
    return dst


def apply_increments_to_checkpoint(
        source: str | Path, increments: Mapping[str, object],
        destination: str | Path, *, publish: bool = True,
        moment_policy: str = "full-moment",
        moment_repair: bool = True,
        mp_physics: int | None = None,
        morr_rimed_ice: int = 1) -> dict:
    """Write a copy of a checkpoint with ``increments`` added.

    The source checkpoint is never modified: the analysis is a new file,
    so a failed assimilation always leaves the background recoverable.
    Every key of the source is carried through unchanged except the
    ``state/<field>`` arrays that the increments name.

    ``publish=False`` writes the analysis to ``destination`` +
    :data:`STAGED_SUFFIX` and stops there, so a caller with more than one
    member to analyse can prove every member's write succeeded before any
    member's ``analysis.npz`` becomes visible.  The receipt then carries
    ``published: false`` and ``staged``; :func:`publish_staged_analysis`
    is the commit.  A one-member caller has no use for it and the default
    is the whole operation.
    """
    from woof.ensemble.state_sha import (checkpoint_state_sha256,
                                          serialized_state_attrs)

    src = Path(source)
    dst = Path(destination)
    if not src.is_file():
        raise ValueError(f"no checkpoint at {src}")
    if dst.resolve() == src.resolve():
        raise ValueError(
            f"refusing to write the analysis over its own background "
            f"{src}; pass a distinct destination")
    items = _sorted_items(increments)

    with np.load(src, allow_pickle=False) as data:
        payload = {key: data[key] for key in data.files}

    contract = set(serialized_state_attrs())
    checked = {}
    for name, increment in items:
        if name not in contract:
            raise ValueError(
                f"increment names field {name!r}, which is not in the "
                "restart prognostic contract; the DA interface may only "
                "rewrite serialised prognostic state")
        key = f"state/{name}"
        if key not in payload:
            raise ValueError(
                f"increment names field {name!r}, which checkpoint {src} "
                "does not carry")
        checked[name] = _validate(name, payload[key], increment,
                                  where=f"checkpoint {src}")

    before = checkpoint_state_sha256(src)
    staged = {}
    with np.errstate(over="ignore", invalid="ignore"):
        for name, increment in checked.items():
            key = f"state/{name}"
            original = payload[key]
            addend = increment.astype(original.dtype, copy=False)
            updated = np.asarray(original + addend, dtype=original.dtype)
            _require_finite_result(name, original, addend, updated,
                                   where=f"checkpoint {src}")
            staged[name] = updated

    # Vapour at saturation, the same cap a live state takes; the
    # background temperature comes from the file's own p, alt and qv.
    cap, saturation = _saturation_cap(
        lambda name: payload.get(f"state/{name}"), staged, checked)
    if cap is not None:
        if "qv" not in staged:
            staged["qv"] = np.array(payload["state/qv"], copy=True)
        cap.write(staged["qv"])
        if "qv" not in checked:
            saturation["qv_written_beyond_the_increment"] = True

    # The moment guard reads the RESULT the checkpoint is about to carry,
    # over every state array the file holds -- not only the ones the
    # increment names, because the pair's other half is exactly the one
    # the increment did not name.
    available = tuple(key[len("state/"):] for key in payload
                      if key.startswith("state/"))
    resulting = {name: staged.get(name, payload[f"state/{name}"])
                 for name in available}
    repaired, moments = _moment_guard(
        resulting, tuple(checked), available,
        moment_policy=moment_policy, moment_repair=moment_repair,
        mp_physics=mp_physics, morr_rimed_ice=morr_rimed_ice,
        where=f"checkpoint {src.name}")

    fields = []
    for name, increment in checked.items():
        key = f"state/{name}"
        updated = staged[name]
        payload[key] = updated
        fields.append({
            "field": name,
            "shape": list(updated.shape),
            "increment_sha256": _array_sha256(increment),
            "field_sha256": _array_sha256(updated),
        })
    if cap is not None and "qv" not in checked:
        payload["state/qv"] = staged["qv"]
    for name, values in repaired.items():
        key = f"state/{name}"
        original = payload[key]
        payload[key] = np.asarray(values, dtype=original.dtype)

    dst.parent.mkdir(parents=True, exist_ok=True)
    landed = dst if publish else dst.with_name(dst.name + STAGED_SUFFIX)
    # Unique tmp name: a fixed one is a collision between two writers
    # aimed at the same destination, and the loser silently wins.
    tmp = landed.with_name(f"{landed.name}.{os.getpid()}."
                           f"{uuid.uuid4().hex}.tmp")
    try:
        with open(tmp, "wb") as stream:
            np.savez(stream, **payload)
            stream.flush()
            os.fsync(stream.fileno())
        tmp.replace(landed)
    finally:
        if tmp.exists():
            tmp.unlink()
    receipt = {
        "contract": INCREMENT_CONTRACT,
        "stability": "experimental",
        "source": str(src),
        "destination": str(dst),
        "published": bool(publish),
        "field_count": len(fields),
        "fields": fields,
        "moments": moments,
        "saturation": saturation,
        "state_sha256_before": before,
        "state_sha256_after": checkpoint_state_sha256(landed),
    }
    if not publish:
        receipt["staged"] = str(landed)
    return receipt
