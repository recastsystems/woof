"""Putting an observation and a forecast on one grid, reproducibly.

Two remap operators, both registered, because the battery scores under one
and qualifies the choice against the other:

``nearest``
    every destination cell takes the value of the source cell whose centre is
    closest to it on the sphere.  Cheap, monotone, invents nothing -- and it
    is the registered operator for reflectivity, where the quantity is a
    maximum rather than an average and smoothing it would be a physical
    statement nobody made.

``cell_average``
    every destination cell takes the mean of the source cells that fall in
    it, where "fall in it" is spelled as "whose nearest destination centre is
    this one".  For a locally uniform destination grid that partition is the
    destination cells' footprints, so the operator conserves the field's mean
    over any union of destination cells whose sources are all valid -- the
    budget-style alternative the qualification receipt measures the registered
    choice against.

Three properties this module exists to keep:

* **The plan is data, computed once.**  A remap between two fixed grids is a
  fixed integer mapping.  Building it once per case and reusing it across
  arms is not only faster, it is what makes the arms comparable: every arm is
  remapped by the identical integer array, so no score can differ because a
  neighbour search broke a tie differently.
* **Distance is bounded and the bound is enforced.**  A destination cell with
  no source centre within ``max_distance_m`` is marked *invalid*, not filled
  from far away.  Without that, a domain corner outside the observation's
  coverage silently borrows the nearest edge observation and gets scored.
* **Validity is remapped with the values.**  A destination cell built only
  from invalid sources is invalid.  Missing does not become zero at a grid
  change.

Geometry is done on the unit sphere in Cartesian coordinates, so longitude
wrapping and the poles are not special cases; distances are converted between
great-circle and chord form with :data:`EARTH_RADIUS_M`.

**Where the arithmetic runs.**  WOOF-ArWen's Python boundary names
"regrid/transform" as data-path processing, so both halves of this module
run on the Rust crate ``tools/rustwx/crates/obs-regrid`` by default,
reached through :mod:`woof.obs_regrid_bridge`.  The bodies below
stay as the parity reference the crate's goldens are extracted from, and
as an explicit fallback (``WOOF_OBSREGRID_PYTHON=1``) that announces
itself as a workaround.

The two engines agree bit for bit on real observation and model grids --
the crate's parity suite compares IEEE bit patterns, not tolerances, on
goldens this module produced -- with exactly one documented divergence.
When two source cells are exactly equidistant from one destination cell,
``cKDTree`` returns whichever the traversal reached first (measured: 232
of 400 trials returned the lowest index, 168 returned another), while
the Rust engine defines the answer as the lowest flat source index.
That is deliberate rather than a parity failure: this module's own
premise, three paragraphs up, is that every arm is remapped by the
identical integer array so no score can differ because a neighbour
search broke a tie differently, and a tie resolved by traversal order is
that defect waiting for a library upgrade.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

#: Mean Earth radius (IUGG), used only to turn a metre bound into a chord
#: bound on the unit sphere.  Nothing here is sensitive to the fourth digit.
EARTH_RADIUS_M = 6371008.8

NEAREST = "nearest"
CELL_AVERAGE = "cell_average"
METHODS = (NEAREST, CELL_AVERAGE)


def unit_vectors(latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
    """``(n, 3)`` unit-sphere positions for a lat/lon grid, row-major."""
    lat = np.deg2rad(np.asarray(latitude, dtype=np.float64)).ravel()
    lon = np.deg2rad(np.asarray(longitude, dtype=np.float64)).ravel()
    if lat.shape != lon.shape or lat.size == 0:
        raise ValueError("latitude and longitude must be one non-empty shape")
    cos_lat = np.cos(lat)
    return np.stack(
        (cos_lat * np.cos(lon), cos_lat * np.sin(lon), np.sin(lat)), axis=1)


def chord_from_arc(distance_m: float) -> float:
    """Straight-line unit-sphere distance for a great-circle metre distance."""
    if not math.isfinite(distance_m) or distance_m <= 0.0:
        raise ValueError("a remap distance bound must be positive and finite")
    arc = float(distance_m) / EARTH_RADIUS_M
    if arc >= math.pi:
        return 2.0
    return 2.0 * math.sin(arc / 2.0)


def arc_from_chord(chord: float) -> float:
    """Great-circle metres for a unit-sphere chord length."""
    value = min(max(float(chord) / 2.0, -1.0), 1.0)
    return 2.0 * math.asin(value) * EARTH_RADIUS_M


@dataclass(frozen=True, eq=False)
class RegridPlan:
    """A fixed integer mapping from one grid to another.

    ``source_index`` is per destination cell for :data:`NEAREST` and per
    *source* cell for :data:`CELL_AVERAGE`; ``reachable`` is the destination
    mask the distance bound leaves usable in either case.  The two are kept in
    one type because callers only ever want the pair, and separating them
    invites remapping values with one and validity with the other.
    """

    method: Literal["nearest", "cell_average"]
    source_index: np.ndarray
    reachable: np.ndarray
    destination_shape: tuple[int, int]
    source_shape: tuple[int, int]
    max_distance_m: float
    max_used_distance_m: float

    def record(self) -> dict[str, object]:
        return {
            "method": self.method,
            "destination_shape": list(self.destination_shape),
            "source_shape": list(self.source_shape),
            "max_distance_m": float(self.max_distance_m),
            "max_used_distance_m": float(self.max_used_distance_m),
            "unreachable_destination_cells": int(
                self.reachable.size - int(np.count_nonzero(self.reachable))),
        }


def _tree(points: np.ndarray):
    try:
        from scipy.spatial import cKDTree
    except ImportError as error:
        # Not "install the [obs] extra" any more: scipy is a runtime
        # dependency, so [obs] is an empty compatibility alias and running
        # it would install nothing.  Reaching here means the environment
        # lost a declared dependency.
        raise ImportError(
            "building a remap plan needs scipy (the cKDTree neighbour "
            "search). scipy is a required dependency of woof, so this "
            "environment is incomplete: repair it with `pip install "
            "--upgrade --force-reinstall woof`, or install the one "
            "package with `pip install 'scipy>=1.11'`") from error

    return cKDTree(points)


def build_plan(*, source_latitude: np.ndarray, source_longitude: np.ndarray,
               destination_latitude: np.ndarray,
               destination_longitude: np.ndarray, method: str,
               max_distance_m: float) -> RegridPlan:
    """Compute the remap once, for reuse across every arm of a case."""
    if method not in METHODS:
        raise ValueError(f"unknown remap method {method!r}; expected {METHODS}")
    source_shape = np.asarray(source_latitude).shape
    destination_shape = np.asarray(destination_latitude).shape
    if len(source_shape) != 2 or len(destination_shape) != 2:
        raise ValueError("both grids must be 2-D")

    from woof import obs_regrid_bridge as regrid_bridge

    engine = regrid_bridge.route("remap plan build")
    if engine is not None:
        index, reachable, used = engine.build_plan(
            method=method, source_latitude=source_latitude,
            source_longitude=source_longitude,
            destination_latitude=destination_latitude,
            destination_longitude=destination_longitude,
            max_distance_m=max_distance_m)
        return RegridPlan(
            method=method, source_index=index, reachable=reachable,
            destination_shape=(int(destination_shape[0]),
                               int(destination_shape[1])),
            source_shape=(int(source_shape[0]), int(source_shape[1])),
            max_distance_m=float(max_distance_m),
            max_used_distance_m=used)

    return _build_plan_python(
        source_latitude=source_latitude, source_longitude=source_longitude,
        destination_latitude=destination_latitude,
        destination_longitude=destination_longitude, method=method,
        max_distance_m=max_distance_m)


def _build_plan_python(*, source_latitude: np.ndarray,
                       source_longitude: np.ndarray,
                       destination_latitude: np.ndarray,
                       destination_longitude: np.ndarray, method: str,
                       max_distance_m: float) -> RegridPlan:
    """The scipy body: the parity reference and the announced fallback.

    Kept verbatim rather than deleted because the crate's goldens are
    extracted from THIS code on real staged bytes, and a reference that
    has been rewritten is no longer the thing the port was proven
    against.
    """
    source_shape = np.asarray(source_latitude).shape
    destination_shape = np.asarray(destination_latitude).shape
    source_points = unit_vectors(source_latitude, source_longitude)
    destination_points = unit_vectors(destination_latitude,
                                      destination_longitude)
    bound = chord_from_arc(max_distance_m)

    if method == NEAREST:
        distance, index = _tree(source_points).query(
            destination_points, k=1, distance_upper_bound=bound)
        reachable = np.isfinite(distance) & (index < source_points.shape[0])
        index = np.where(reachable, index, 0).astype(np.int64)
        used = float(arc_from_chord(distance[reachable].max())
                     ) if reachable.any() else 0.0
        return RegridPlan(
            method=NEAREST, source_index=index.reshape(destination_shape),
            reachable=reachable.reshape(destination_shape),
            destination_shape=(int(destination_shape[0]),
                               int(destination_shape[1])),
            source_shape=(int(source_shape[0]), int(source_shape[1])),
            max_distance_m=float(max_distance_m), max_used_distance_m=used)

    distance, index = _tree(destination_points).query(
        source_points, k=1, distance_upper_bound=bound)
    assigned = np.isfinite(distance) & (index < destination_points.shape[0])
    index = np.where(assigned, index, -1).astype(np.int64)
    reachable = np.zeros(destination_points.shape[0], dtype=bool)
    reachable[index[assigned]] = True
    used = float(arc_from_chord(distance[assigned].max())
                 ) if assigned.any() else 0.0
    return RegridPlan(
        method=CELL_AVERAGE, source_index=index.reshape(source_shape),
        reachable=reachable.reshape(destination_shape),
        destination_shape=(int(destination_shape[0]),
                           int(destination_shape[1])),
        source_shape=(int(source_shape[0]), int(source_shape[1])),
        max_distance_m=float(max_distance_m), max_used_distance_m=used)


def apply_plan(plan: RegridPlan, values: np.ndarray, valid: np.ndarray
               ) -> tuple[np.ndarray, np.ndarray]:
    """Remap a field and its validity together.

    Returns ``(values, valid)`` on the destination grid.  Values under a
    False destination mask are zero and must not be read; the mask is the
    answer to "is there an observation here", and the scorer asks it.
    """
    values = np.asarray(values, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if values.shape != plan.source_shape or valid.shape != plan.source_shape:
        raise ValueError(
            f"field shape {values.shape} does not match the plan's source "
            f"grid {plan.source_shape}")

    from woof import obs_regrid_bridge as regrid_bridge

    engine = regrid_bridge.route("remap apply")
    if engine is not None:
        return engine.apply_plan(
            method=plan.method, source_index=plan.source_index,
            reachable=plan.reachable, source_shape=plan.source_shape,
            destination_shape=plan.destination_shape, values=values,
            valid=valid)

    return _apply_plan_python(plan, values, valid)


def _apply_plan_python(plan: RegridPlan, values: np.ndarray,
                       valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The numpy body: the parity reference and the announced fallback."""
    if plan.method == NEAREST:
        flat_values = values.ravel()
        flat_valid = valid.ravel()
        index = plan.source_index
        out_values = flat_values[index]
        out_valid = flat_valid[index] & plan.reachable
        return np.where(out_valid, out_values, 0.0), out_valid

    destination_cells = int(np.prod(plan.destination_shape))
    index = plan.source_index.ravel()
    contributing = (index >= 0) & valid.ravel()
    totals = np.zeros(destination_cells, dtype=np.float64)
    counts = np.zeros(destination_cells, dtype=np.int64)
    np.add.at(totals, index[contributing], values.ravel()[contributing])
    np.add.at(counts, index[contributing], 1)
    out_valid = (counts > 0).reshape(plan.destination_shape)
    out_values = np.zeros(destination_cells, dtype=np.float64)
    np.divide(totals, counts, out=out_values, where=counts > 0)
    return out_values.reshape(plan.destination_shape), out_valid


def regrid_field(*, source_latitude: np.ndarray, source_longitude: np.ndarray,
                 destination_latitude: np.ndarray,
                 destination_longitude: np.ndarray, values: np.ndarray,
                 valid: np.ndarray, method: str, max_distance_m: float
                 ) -> tuple[np.ndarray, np.ndarray, RegridPlan]:
    """Build a plan and apply it, for callers with a single field to move."""
    plan = build_plan(
        source_latitude=source_latitude, source_longitude=source_longitude,
        destination_latitude=destination_latitude,
        destination_longitude=destination_longitude, method=method,
        max_distance_m=max_distance_m)
    remapped, remapped_valid = apply_plan(plan, values, valid)
    return remapped, remapped_valid, plan


__all__ = [
    "CELL_AVERAGE", "EARTH_RADIUS_M", "METHODS", "NEAREST", "RegridPlan",
    "apply_plan", "arc_from_chord", "build_plan", "chord_from_arc",
    "regrid_field", "unit_vectors",
]
