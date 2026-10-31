"""Shared assembly and evidence plumbing for the idealized N2 rung cases."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from types import SimpleNamespace

import numpy as np

from woof.core.clock import build_schedule, resolve_clock
from woof.core.model import (DomainNode, ExperimentState, ModelRuntimeStatus,
                              execute_experiment)
from woof.core.nest import NestCoupler as ConcreteNestCoupler
from woof.core.preflight import nest_field_kinds
from woof.ingest.lateral_bc import (attach_nest_boundaries,
                                     couple_nest_field)
from woof.ingest.nest_init import parent_only_init
from woof.static.lambert import grids_from_projection_config
from woof.verify.nest_gates import gate
from woof.verify.npref import np_specified_relaxation, np_w_relaxation_current
from woof.verify.state_equiv import fp32_state_equiv_report


# WRF Registry scalar names versus woof's compact state attributes.
FIELD_ALIASES = MappingProxyType({
    "qnr": "nr",
    "qni": "ni",
    "qns": "ns",
    "qng": "ng",
})

# SINT is exact at ratio one, but its production registration still reads a
# +/-2 donor stencil.  A full-parent identity child therefore needs this
# setup-only source halo even though every noncentral coefficient is zero.
IDENTITY_SINT_HALO = 2

# F19's numeric identity-oracle contract.  These values deliberately live in
# the verification case rather than a production surface: nest_gates owns the
# registered prose and this module is its executable comparator.
IDENTITY_COMPONENTS = (
    "t0_full_state_bit_identity",
    "force_table_oracle",
    "rk_davies_tendency_oracle",
    "nested_w_substep_bit_identity",
    "raw_specified_row_guardrail",
)
DAVIES_RTOL = 5.0e-7
DAVIES_ATOL = MappingProxyType({
    "u": 8.0e-6,
    "v": 8.0e-6,
    "w": 6.0e-6,
    "theta": 4.0e-6,
    "phi": 2.0e-3,
    "mu": 1.0e-7,
    "qv": 5.0e-10,
})
RAW_SPECIFIED_CAPS = MappingProxyType({
    "u": 8.0e-6,
    "v": 1.0e-6,
    "w": 4.0e-9,
    "thp": 6.5e-5,
    "php": 6.5e-5,
    "mup": 2.0e-6,
    "qv": 2.5e-10,
})
INACTIVE_MOIST_FIELDS = (
    "qc", "qr", "qi", "qs", "qg", "qnr", "qni", "qns", "qng",
)

_APPLICATION_NAME = {"t": "theta", "ph": "phi"}
_RAW_NAME = {"t": "thp", "ph": "php", "mu": "mup"}
_SCALAR_APPLICATION_NAMES = {
    "qv", "qc", "qr", "qi", "qs", "qg", "nr", "ni", "ns", "ng",
}


def _host_fp32(value) -> np.ndarray:
    """Return a C-order host view of the stored FP32 bits."""
    if hasattr(value, "get"):
        value = value.get()
    return np.ascontiguousarray(value, dtype=np.float32)


def _bit_mismatch_count(actual, expected) -> int:
    actual = _host_fp32(actual)
    expected = _host_fp32(expected)
    if actual.shape != expected.shape:
        return max(int(actual.size), int(expected.size), 1)
    return int(np.count_nonzero(
        actual.view(np.uint32) != expected.view(np.uint32)))


def _finite_pair(actual, expected) -> bool:
    return bool(np.isfinite(actual).all() and np.isfinite(expected).all())


def _distance_map(ny: int, nx: int) -> np.ndarray:
    j = np.arange(ny)[:, None]
    i = np.arange(nx)[None, :]
    return np.minimum(np.minimum(j, ny - 1 - j),
                      np.minimum(i, nx - 1 - i))


def _side_view(array, side: str, width: int) -> np.ndarray:
    value = _host_fp32(array)
    if side == "west":
        return np.ascontiguousarray(value[..., :width])
    if side == "east":
        return np.ascontiguousarray(value[..., -width:][..., ::-1])
    if side == "south":
        return np.ascontiguousarray(value[..., :width, :])
    if side == "north":
        return np.ascontiguousarray(value[..., -width:, :][..., ::-1, :])
    raise KeyError(side)


def _unreversed_side_view(array, side: str, width: int) -> np.ndarray:
    value = _host_fp32(array)
    if side == "east":
        return np.ascontiguousarray(value[..., -width:])
    if side == "north":
        return np.ascontiguousarray(value[..., -width:, :])
    return _side_view(value, side, width)


def _host_boundary(boundary):
    """Copy a resident boundary object into the independent NumPy mirror."""
    return SimpleNamespace(**{
        side: SimpleNamespace(
            value=_host_fp32(getattr(boundary, side).value),
            tendency=_host_fp32(getattr(boundary, side).tendency))
        for side in ("west", "east", "south", "north")
    })


def bit_identity_report(candidate_state, reference_state,
                        field_names) -> dict[str, object]:
    """Exact finite FP32 identity, including signed-zero bit patterns."""
    candidate = state_field_view(candidate_state, field_names)
    reference = state_field_view(reference_state, field_names)
    fields = {}
    for name in field_names:
        try:
            actual = _host_fp32(candidate[name])
            expected = _host_fp32(reference[name])
        except (AttributeError, KeyError) as exc:
            fields[name] = {
                "count": 0, "bit_mismatches": None, "finite": False,
                "pass": False, "reason": f"missing field: {exc}",
            }
            continue
        same_shape = actual.shape == expected.shape
        finite = same_shape and _finite_pair(actual, expected)
        mismatches = (_bit_mismatch_count(actual, expected)
                      if same_shape else None)
        fields[name] = {
            "count": int(actual.size),
            "bit_mismatches": mismatches,
            "finite": bool(finite),
            "pass": bool(finite and mismatches == 0 and actual.size > 0),
            "reason": (None if same_shape else
                       f"shape mismatch: {actual.shape} != {expected.shape}"),
        }
    return {
        "fields": fields,
        "pass": bool(fields and all(item["pass"] for item in fields.values())),
    }


def force_table_oracle(parent_fields, child_fields, tables, *,
                       parent_dt_fp32) -> dict[str, object]:
    """Score one complete FORCE transaction under F19 component (2)."""
    dt = np.float32(parent_dt_fp32)
    valid_dt = bool(np.isfinite(dt) and dt > 0.0)
    fields: dict[str, object] = {}
    for name, table_sides in tables.items():
        parent = _host_fp32(parent_fields[name])
        child = _host_fp32(child_fields[name])
        side_reports = {}
        for side, pair in table_sides.items():
            value, tendency = map(_host_fp32, pair)
            width = int(value.shape[-1] if side in ("west", "east")
                        else value.shape[-2])
            expected_child = _side_view(child, side, width)
            expected_parent = _side_view(parent, side, width)
            difference = np.subtract(
                expected_parent, expected_child, dtype=np.float32)
            expected_tendency = np.asarray(
                difference.astype(np.float64) / np.float64(dt),
                dtype=np.float32) if valid_dt else np.full_like(
                    difference, np.nan)
            successor = np.add(
                value, np.multiply(dt, tendency, dtype=np.float32),
                dtype=np.float32)
            error = np.abs(successor.astype(np.float64)
                           - expected_parent.astype(np.float64))
            # F19a: the registered basis is the parent FIELD max (the
            # attribution run's measured basis), not the per-side max --
            # a near-zero side makes the per-side bound denormal-tight.
            scale = np.float32(np.max(np.abs(parent.astype(np.float64))))
            tolerance = (0.0 if scale == 0.0 else
                         2.0 * float(np.spacing(scale)))
            finite = (valid_dt and _finite_pair(value, expected_child)
                      and _finite_pair(tendency, expected_tendency)
                      and _finite_pair(successor, expected_parent))
            value_mismatches = _bit_mismatch_count(value, expected_child)
            tendency_mismatches = _bit_mismatch_count(
                tendency, expected_tendency)
            successor_pass = bool(
                finite and error.size > 0
                and float(np.max(error)) <= tolerance)
            side_reports[side] = {
                "value_bit_mismatches": value_mismatches,
                "tendency_bit_mismatches": tendency_mismatches,
                "successor_max_abs_error": (
                    float(np.max(error)) if error.size else None),
                "parent_field_scale": float(scale),
                "successor_abs_tolerance": tolerance,
                "successor_pass": successor_pass,
                "finite": bool(finite),
                "pass": bool(finite and value_mismatches == 0
                             and tendency_mismatches == 0
                             and successor_pass),
            }
            if side in ("east", "north"):
                unreversed = _unreversed_side_view(parent, side, width)
                side_reports[side]["distance_reversal_pass"] = bool(
                    value_mismatches == 0 and tendency_mismatches == 0)
                side_reports[side]["successor_vs_unreversed_max_abs"] = float(
                    np.max(np.abs(successor.astype(np.float64)
                                  - unreversed.astype(np.float64))))

        # The CUDA consumer gives south/north ownership of all four corners.
        # Pin the selected table entries explicitly rather than relying on the
        # whole-side equality above to imply the ownership convention.
        corners = {
            "south_west": ("south", (..., 0, 0), (..., 0, 0)),
            "south_east": ("south", (..., 0, -1), (..., 0, -1)),
            "north_west": ("north", (..., 0, 0), (..., -1, 0)),
            "north_east": ("north", (..., 0, -1), (..., -1, -1)),
        }
        corner_reports = {}
        for corner, (side, table_index, child_index) in corners.items():
            value, tendency = map(_host_fp32, table_sides[side])
            pcorner = parent[child_index]
            ccorner = child[child_index]
            difference = np.subtract(pcorner, ccorner, dtype=np.float32)
            expected_tendency = np.asarray(
                difference.astype(np.float64) / np.float64(dt),
                dtype=np.float32) if valid_dt else np.full_like(
                    difference, np.nan)
            corner_reports[corner] = {
                "owner": side,
                "value_bit_mismatches": _bit_mismatch_count(
                    value[table_index], ccorner),
                "tendency_bit_mismatches": _bit_mismatch_count(
                    tendency[table_index], expected_tendency),
            }
            corner_reports[corner]["pass"] = bool(
                corner_reports[corner]["value_bit_mismatches"] == 0
                and corner_reports[corner]["tendency_bit_mismatches"] == 0)
        fields[name] = {
            "sides": side_reports,
            "y_corner_ownership": corner_reports,
            "east_distance_reversal_pass": bool(
                side_reports["east"].get("distance_reversal_pass")),
            "north_distance_reversal_pass": bool(
                side_reports["north"].get("distance_reversal_pass")),
            "y_corner_ownership_pass": all(
                item["pass"] for item in corner_reports.values()),
        }
        fields[name]["pass"] = bool(
            all(item["pass"] for item in side_reports.values())
            and fields[name]["east_distance_reversal_pass"]
            and fields[name]["north_distance_reversal_pass"]
            and fields[name]["y_corner_ownership_pass"])
    return {
        "parent_dt_fp32": float(dt),
        "fields": fields,
        "pass": bool(valid_dt and fields
                     and all(item["pass"] for item in fields.values())),
    }


def _applied_boundary_component(before, after, spec_zone: int) -> np.ndarray:
    """Recover the boundary-owned increment from replacement+add semantics."""
    before = _host_fp32(before)
    after = _host_fp32(after)
    applied = np.subtract(after, before, dtype=np.float32)
    ny, nx = after.shape[-2:]
    for d in range(spec_zone):
        js, jn = d, ny - 1 - d
        applied[..., js, d:nx - d] = after[..., js, d:nx - d]
        applied[..., jn, d:nx - d] = after[..., jn, d:nx - d]
        iw, ie = d, nx - 1 - d
        applied[..., d + 1:ny - d - 1, iw] = after[
            ..., d + 1:ny - d - 1, iw]
        applied[..., d + 1:ny - d - 1, ie] = after[
            ..., d + 1:ny - d - 1, ie]
    return applied


def davies_tendency_oracle(*, field_name: str, current, applied, boundary,
                           dtbc, dt, spec_zone: int, relax_zone: int,
                           apply_relax: bool, inactive: bool = False,
                           divide_msf=None) -> dict[str, object]:
    """Compare an applied component with the independent five-point oracle."""
    current = _host_fp32(current)
    applied = _host_fp32(applied)
    host_boundary = _host_boundary(boundary)
    expected = np_specified_relaxation(
        current, np.zeros_like(current, dtype=np.float64), host_boundary,
        dtbc=float(np.float32(dtbc)), dt=float(np.float32(dt)),
        spec_zone=spec_zone, relax_zone=relax_zone, spec_exp=0.0,
        apply_relax=apply_relax)
    if divide_msf is not None:
        expected = expected / _host_fp32(divide_msf)[None]
    ny, nx = applied.shape[-2:]
    distance = _distance_map(ny, nx)
    active = distance < relax_zone
    zero_ramp = distance == relax_zone
    outside = distance >= relax_zone
    finite = bool(_finite_pair(applied, expected))
    residual = np.abs(applied.astype(np.float64) - expected)
    atol = DAVIES_ATOL.get(field_name)
    if inactive:
        exact_zero = np.zeros_like(applied, dtype=np.float32)
        inactive_pass = bool(
            finite and _bit_mismatch_count(applied, exact_zero) == 0)
        formula_pass = inactive_pass
    elif atol is None:
        inactive_pass = True
        formula_pass = False
    else:
        inactive_pass = True
        # F19a: rtol binds against the TABLE-MAX held tendency (the
        # attribution run's residual basis) -- the FP32-vs-FP64 formula
        # residual scales with the table max, so a per-element rtol
        # misreads growing tendencies whose elements pass through zero.
        table_scale = (float(np.max(np.abs(expected[..., active])))
                       if np.any(active) else 0.0)
        tolerance = atol + DAVIES_RTOL * table_scale
        formula_pass = bool(
            finite and (not np.any(active) or float(np.max(
                residual[..., active])) <= tolerance))
    zeros = np.zeros_like(applied, dtype=np.float32)
    zero_ramp_mismatches = _bit_mismatch_count(
        applied[..., zero_ramp], zeros[..., zero_ramp])
    support_mismatches = _bit_mismatch_count(
        applied[..., outside], zeros[..., outside])
    report = {
        "field": field_name,
        "apply_relax": bool(apply_relax),
        "inactive": bool(inactive),
        "rtol": DAVIES_RTOL,
        "atol": (0.0 if inactive else atol),
        "max_abs_residual": (
            float(np.max(residual)) if residual.size else None),
        "zero_ramp_row": int(relax_zone),
        "zero_ramp_bit_mismatches": zero_ramp_mismatches,
        "row_support_bit_mismatches": support_mismatches,
        "formula_pass": bool(formula_pass),
        "zero_ramp_pass": zero_ramp_mismatches == 0,
        "row_support_pass": support_mismatches == 0,
        "inactive_pass": bool(inactive_pass),
        "finite": finite,
    }
    report["pass"] = bool(
        finite and report["formula_pass"] and report["zero_ramp_pass"]
        and report["row_support_pass"] and report["inactive_pass"])
    return report


def nested_w_substep_oracle(before, after, rw_t, *, dtau,
                            spec_zone: int) -> dict[str, object]:
    """F19 bit oracle for every nested specified-row acoustic update."""
    before = _host_fp32(before)
    after = _host_fp32(after)
    rw_t = _host_fp32(rw_t)
    expected = np.add(
        before, np.multiply(np.float32(dtau), rw_t, dtype=np.float32),
        dtype=np.float32)
    distance = _distance_map(after.shape[-2], after.shape[-1])
    specified = distance < spec_zone
    actual_row = after[..., specified]
    expected_row = expected[..., specified]
    finite = _finite_pair(actual_row, expected_row)
    mismatches = _bit_mismatch_count(actual_row, expected_row)
    return {
        "dtau": float(np.float32(dtau)),
        "count": int(actual_row.size),
        "bit_mismatches": mismatches,
        "finite": bool(finite),
        "pass": bool(finite and actual_row.size > 0 and mismatches == 0),
    }


def raw_specified_row_guardrail(candidate_state, reference_state, *,
                                active_caps, inactive_fields,
                                spec_zone: int) -> dict[str, object]:
    """F19 final-uncoupling raw caps and inactive-species identity."""
    active = {}
    for name, cap in active_caps.items():
        actual = _host_fp32(getattr(candidate_state, FIELD_ALIASES.get(name, name)))
        expected = _host_fp32(getattr(reference_state, FIELD_ALIASES.get(name, name)))
        distance = _distance_map(actual.shape[-2], actual.shape[-1])
        mask = distance < spec_zone
        av, ev = actual[..., mask], expected[..., mask]
        finite = _finite_pair(av, ev)
        maximum = (float(np.max(np.abs(
            av.astype(np.float64) - ev.astype(np.float64))))
                   if av.size else None)
        active[name] = {
            "count": int(av.size), "cap": float(cap), "max_abs": maximum,
            "finite": bool(finite),
            "pass": bool(finite and av.size > 0 and maximum <= cap),
        }
    inactive = {}
    for name in inactive_fields:
        actual = _host_fp32(getattr(candidate_state, FIELD_ALIASES.get(name, name)))
        expected = _host_fp32(getattr(reference_state, FIELD_ALIASES.get(name, name)))
        distance = _distance_map(actual.shape[-2], actual.shape[-1])
        mask = distance < spec_zone
        av, ev = actual[..., mask], expected[..., mask]
        finite = _finite_pair(av, ev)
        mismatches = _bit_mismatch_count(av, ev)
        inactive[name] = {
            "count": int(av.size), "bit_mismatches": mismatches,
            "finite": bool(finite),
            "pass": bool(finite and av.size > 0 and mismatches == 0),
        }
    return {
        "active_caps": active,
        "inactive_bit_identity": inactive,
        "h_diabatic_excluded": True,
        "pass": bool(active and all(item["pass"] for item in active.values())
                     and all(item["pass"] for item in inactive.values())),
    }


def identity_oracle_verdict(components) -> bool:
    """The gate passes iff all five registered blocking components pass."""
    return bool(set(components) == set(IDENTITY_COMPONENTS)
                and all(bool(components[name]["pass"])
                        for name in IDENTITY_COMPONENTS))


class RatioOneIdentityCoupler:
    """Coincident-grid specialization of the production force transaction.

    ``ConcreteNestCoupler`` requires WRF SINT's +/-2 parent donor halo.
    The deliberately non-production N2a/N2b child fills its same-size parent,
    so that stencil cannot exist (and the concrete constructor correctly
    rejects it).  At ratio one SINT is the identity by definition.  This
    specialization keeps the remaining production path intact: coupled
    field units, child-current boundary values, parent-step tendencies,
    rolling nested boundary attachment, dtbc reset, and Davies application.
    """

    _APPLICATION_NAME = {"t": "theta", "ph": "phi"}

    def __init__(self, child_node):
        child = child_node.cfg
        if child.parent_grid_ratio != 1 or child.parent_time_step_ratio != 1:
            raise ValueError("RatioOneIdentityCoupler requires ratios 1/1")
        if child_node.parent is None:
            raise ValueError("identity coupler requires a parent")
        parent = child_node.parent.cfg.run
        run = child.run
        if (run.nx, run.ny, run.nz) != (parent.nx, parent.ny, parent.nz):
            raise ValueError("ratio-1 identity grids must have equal shapes")
        self.child_node = child_node
        self._valid = False
        self._last_tables = None

    @property
    def valid(self) -> bool:
        return self._valid

    def invalidate(self) -> None:
        self._valid = False
        resident = getattr(
            self.child_node.state, "_lateral_boundary_device", None)
        if resident is not None:
            resident.valid = False

    @staticmethod
    def _coupled(state, kind):
        import cupy as cp

        if kind == "mu":
            shape = (1, *state.mup.shape)
        else:
            name = {"t": "thp", "ph": "php"}.get(kind, kind)
            shape = getattr(state, name).shape
        out = cp.empty(shape, dtype=cp.float32)
        couple_nest_field(state, kind, out=out)
        return out

    @staticmethod
    def _sides(value, tendency, width):
        def packed(array):
            # The Davies raw kernels consume these tables as linear C-order
            # buffers, exactly like bdy_interp1's manifest slots.  Materialize
            # every side at each FORCE so reversed/strided views cannot leak
            # through attach_nest_boundaries and the values remain current.
            return array.copy(order="C")

        return {
            "west": (packed(value[..., :width]),
                     packed(tendency[..., :width])),
            "east": (packed(value[..., -width:][..., ::-1]),
                     packed(tendency[..., -width:][..., ::-1])),
            "south": (packed(value[..., :width, :]),
                      packed(tendency[..., :width, :])),
            "north": (packed(value[..., -width:, :][..., ::-1, :]),
                      packed(tendency[..., -width:, :][..., ::-1, :])),
        }

    def force(self, node) -> None:
        import cupy as cp

        if node is not self.child_node:
            raise ValueError("identity force called with a different node")
        parent = node.parent
        lead = int(parent.clock.ticks) - int(node.clock.ticks)
        interval_ticks = int(parent.clock.spec.step_ticks)
        if lead != interval_ticks:
            raise RuntimeError(
                "parent must lead child by one interval before identity FORCE")
        run = node.cfg.run
        width = min(max(run.spec_zone, run.relax_zone + 1),
                    run.spec_bdy_width)
        rdt = np.float64(1.0) / np.float64(parent.clock.spec.dt_fp32)
        fields = {}
        for kind in nest_field_kinds(run):
            parent_field = self._coupled(parent.state, kind)
            child_field = self._coupled(node.state, kind)
            # interp_fcn.F:2583: form the difference in REAL, widen it for
            # the REAL*8 reciprocal multiply, then round the stored table.
            difference = cp.asarray(parent_field - child_field,
                                    dtype=cp.float32)
            tendency = cp.asarray(
                difference.astype(cp.float64) * rdt, dtype=cp.float32)
            fields[self._APPLICATION_NAME.get(kind, kind)] = self._sides(
                child_field, tendency, width)
        attach_nest_boundaries(
            node.state, fields, clock=node.clock,
            spec_bdy_width=run.spec_bdy_width,
            spec_zone=run.spec_zone, relax_zone=run.relax_zone)
        node.clock.mark_force()
        self._last_tables = MappingProxyType(fields)
        self._valid = True

    def feedback_prepare(self, node, out) -> None:
        if node is not self.child_node:
            raise ValueError("feedback node differs from identity child")
        out.payload = None

    def feedback_commit(self, node, *, start: bool = False) -> None:
        if node is not self.child_node:
            raise ValueError("feedback node differs from identity child")

    def feedback_finalize(self, node, *, start: bool = False) -> None:
        if node is not self.child_node:
            raise ValueError("feedback node differs from identity child")


class IdentityOracleCoupler(RatioOneIdentityCoupler):
    """Ratio-one coupler that scores every FORCE without altering it."""

    def __init__(self, child_node):
        super().__init__(child_node)
        self.oracle_forces: list[dict[str, object]] = []

    def force(self, node) -> None:
        super().force(node)
        parent_fields = {}
        child_fields = {}
        for application_name in self._last_tables:
            kind = {"theta": "t", "phi": "ph"}.get(
                application_name, application_name)
            parent_fields[application_name] = _host_fp32(
                self._coupled(node.parent.state, kind))
            child_fields[application_name] = _host_fp32(
                self._coupled(node.state, kind))
        result = force_table_oracle(
            parent_fields, child_fields, self._last_tables,
            parent_dt_fp32=node.parent.clock.spec.dt_fp32)
        result.update({
            "ordinal": len(self.oracle_forces) + 1,
            "target_ticks": int(node.parent.clock.ticks),
            "target_seconds": float(node.parent.clock.elapsed_seconds),
        })
        self.oracle_forces.append(result)


def _relaxation_current(state, name: str, source) -> np.ndarray:
    """Independent FP32 transcription of the coupled current-side value."""
    if name == "w":
        return np_w_relaxation_current(
            state, source_field=source, dtype=np.float32)
    mup = _host_fp32(state.mup)
    mub = _host_fp32(state.mub2d)
    mass = np.add(mub, mup, dtype=np.float32)
    if name == "mu":
        return mup[None]
    c1h = _host_fp32(state.c1h)[:, None, None]
    c2h = _host_fp32(state.c2h)[:, None, None]
    c1f = _host_fp32(state.c1f)[:, None, None]
    c2f = _host_fp32(state.c2f)[:, None, None]
    if name == "u":
        face = np.empty((mass.shape[0], mass.shape[1] + 1), dtype=np.float32)
        face[:, 0], face[:, -1] = mass[:, 0], mass[:, -1]
        face[:, 1:-1] = np.multiply(
            np.float32(0.5),
            np.add(mass[:, 1:], mass[:, :-1], dtype=np.float32),
            dtype=np.float32)
        weight = np.add(
            np.multiply(c1h, face[None], dtype=np.float32), c2h,
            dtype=np.float32)
        result = np.multiply(weight, _host_fp32(source), dtype=np.float32)
        if state.has_msf:
            result = np.divide(
                result, _host_fp32(state.msfu)[None], dtype=np.float32)
        return result
    if name == "v":
        face = np.empty((mass.shape[0] + 1, mass.shape[1]), dtype=np.float32)
        face[0], face[-1] = mass[0], mass[-1]
        face[1:-1] = np.multiply(
            np.float32(0.5),
            np.add(mass[1:], mass[:-1], dtype=np.float32),
            dtype=np.float32)
        weight = np.add(
            np.multiply(c1h, face[None], dtype=np.float32), c2h,
            dtype=np.float32)
        result = np.multiply(weight, _host_fp32(source), dtype=np.float32)
        if state.has_msf:
            result = np.divide(
                result, _host_fp32(state.msfv)[None], dtype=np.float32)
        return result
    if name == "phi":
        weight = np.add(
            np.multiply(c1f, mass[None], dtype=np.float32), c2f,
            dtype=np.float32)
        return np.multiply(weight, _host_fp32(source), dtype=np.float32)
    weight = np.add(
        np.multiply(c1h, mass[None], dtype=np.float32), c2h,
        dtype=np.float32)
    if name == "theta":
        thb = _host_fp32(state.thb)
        if thb.ndim == 1:
            thb = thb[:, None, None]
        value = np.subtract(
            np.add(thb, _host_fp32(source), dtype=np.float32),
            np.float32(300.0), dtype=np.float32)
    else:
        value = _host_fp32(source)
    return np.multiply(weight, value, dtype=np.float32)


class IdentityOracleTrace:
    """Case-local observers for F19 stage, scalar, and acoustic components."""

    _DRY_TENDENCIES = {
        "u": "ru_t", "v": "rv_t", "theta": "rth_t",
        "phi": "rph_t", "w": "rw_t", "mu": "rmu_t",
    }
    _SOURCES = {
        "u": "u0", "v": "v0", "theta": "thp0", "phi": "php0",
        "w": "w0", "mu": "mup",
    }

    def __init__(self, child_state, child_cfg, *, inactive_fields):
        self.child_state = child_state
        self.child_cfg = child_cfg
        self.inactive_fields = frozenset(
            FIELD_ALIASES.get(name, name) for name in inactive_fields)
        self.davies_checks: list[dict[str, object]] = []
        self.w_substeps: list[dict[str, object]] = []
        self.current_stage: int | None = None
        self._originals = {}

    def _active(self):
        from woof.ingest import lateral_bc as lbc
        return lbc._active_device_interval(self.child_state, self.child_cfg)

    def _record_davies(self, *, name, before, after, current, boundary,
                        dtbc, dt, stage, apply_relax, divide_msf=None):
        observed = _applied_boundary_component(
            before, after, self.child_cfg.spec_zone)
        check = davies_tendency_oracle(
            field_name=name, current=current, applied=observed,
            boundary=boundary, dtbc=dtbc, dt=dt,
            spec_zone=self.child_cfg.spec_zone,
            relax_zone=self.child_cfg.relax_zone,
            apply_relax=apply_relax,
            inactive=name in self.inactive_fields,
            divide_msf=divide_msf)
        check.update({
            "ordinal": len(self.davies_checks) + 1,
            "rk_stage": int(stage) + 1,
        })
        self.davies_checks.append(check)

    def install(self) -> None:
        from woof.core import dycore
        from woof.ingest import lateral_bc as lbc

        self._originals = {
            "dry": dycore.apply_state_lateral_boundaries,
            "scalar": lbc.apply_state_scalar_lateral_boundary,
            "acoustic": dycore.acoustic_substep,
        }

        def dry_wrapper(state, cfg, *, rk_stage):
            if state is not self.child_state:
                return self._originals["dry"](
                    state, cfg, rk_stage=rk_stage)
            self.current_stage = int(rk_stage)
            interval, dtbc, dt, _spec_exp = self._active()
            before = {
                name: _host_fp32(
                    getattr(state, tendency)[None]
                    if name == "mu" else getattr(state, tendency))
                for name, tendency in self._DRY_TENDENCIES.items()
            }
            current = {
                name: _relaxation_current(
                    state, name, getattr(state, self._SOURCES[name]))
                for name in self._DRY_TENDENCIES
            }
            result = self._originals["dry"](
                state, cfg, rk_stage=rk_stage)
            for name, tendency in self._DRY_TENDENCIES.items():
                after = _host_fp32(
                    getattr(state, tendency)[None]
                    if name == "mu" else getattr(state, tendency))
                self._record_davies(
                    name=name, before=before[name], after=after,
                    current=current[name], boundary=interval.fields[name],
                    dtbc=dtbc, dt=dt, stage=rk_stage,
                    apply_relax=(name != "mu" or rk_stage == 0),
                    divide_msf=(state.msft if state.has_msf
                                and name in ("theta", "phi") else None))
            return result

        def scalar_wrapper(state, cfg, field_name, tendency, *,
                           apply_relax=True, source_field=None):
            if state is not self.child_state:
                return self._originals["scalar"](
                    state, cfg, field_name, tendency,
                    apply_relax=apply_relax, source_field=source_field)
            interval, dtbc, dt, _spec_exp = self._active()
            before = _host_fp32(tendency)
            source = (getattr(state, field_name)
                      if source_field is None else source_field)
            current = _relaxation_current(state, field_name, source)
            result = self._originals["scalar"](
                state, cfg, field_name, tendency,
                apply_relax=apply_relax, source_field=source_field)
            self._record_davies(
                name=field_name, before=before, after=_host_fp32(tendency),
                current=current, boundary=interval.fields[field_name],
                dtbc=dtbc, dt=dt,
                stage=(0 if self.current_stage is None
                       else self.current_stage),
                apply_relax=bool(apply_relax))
            return result

        def acoustic_wrapper(state, cfg, dtau, first, coefficients=None):
            if state is not self.child_state:
                return self._originals["acoustic"](
                    state, cfg, dtau, first, coefficients=coefficients)
            before = _host_fp32(state.w_pp)
            rw_t = _host_fp32(state.rw_t)
            result = self._originals["acoustic"](
                state, cfg, dtau, first, coefficients=coefficients)
            check = nested_w_substep_oracle(
                before, _host_fp32(state.w_pp), rw_t, dtau=dtau,
                spec_zone=cfg.spec_zone)
            check.update({
                "ordinal": len(self.w_substeps) + 1,
                "rk_stage": (None if self.current_stage is None
                             else self.current_stage + 1),
            })
            self.w_substeps.append(check)
            return result

        dycore.apply_state_lateral_boundaries = dry_wrapper
        lbc.apply_state_scalar_lateral_boundary = scalar_wrapper
        dycore.acoustic_substep = acoustic_wrapper

    def restore(self) -> None:
        if not self._originals:
            return
        from woof.core import dycore
        from woof.ingest import lateral_bc as lbc
        dycore.apply_state_lateral_boundaries = self._originals["dry"]
        lbc.apply_state_scalar_lateral_boundary = self._originals["scalar"]
        dycore.acoustic_substep = self._originals["acoustic"]
        self._originals = {}

    def davies_component(self, *, steps: int) -> dict[str, object]:
        kinds = tuple(_APPLICATION_NAME.get(kind, kind)
                      for kind in nest_field_kinds(self.child_cfg))
        expected_checks = int(steps) * 3 * len(kinds)
        inventory = {
            name: {
                str(stage): sum(
                    check["field"] == name and check["rk_stage"] == stage
                    for check in self.davies_checks)
                for stage in (1, 2, 3)
            }
            for name in kinds
        }
        inventory_pass = all(
            count == steps for stages in inventory.values()
            for count in stages.values())
        return {
            "rtol": DAVIES_RTOL,
            "atol_floors": dict(DAVIES_ATOL),
            "expected_checks": expected_checks,
            "checks": self.davies_checks,
            "checks_by_field_and_stage": inventory,
            "stage_policy": {
                "recomputed_each_stage": [
                    "u", "v", "theta", "phi", "w", "qv"],
                "mu_relaxation": "stage-1-only",
                "inactive_species": "exact-zero",
            },
            "pass": bool(
                len(self.davies_checks) == expected_checks
                and inventory_pass
                and all(check["pass"] for check in self.davies_checks)),
        }

    def w_component(self, *, steps: int) -> dict[str, object]:
        sound_steps = int(self.child_cfg.time_step_sound)
        expected = int(steps) * (
            1 + max(sound_steps // 2, 1) + sound_steps)
        return {
            "expected_substeps": expected,
            "observed_substeps": len(self.w_substeps),
            "checks": self.w_substeps,
            "pass": bool(
                len(self.w_substeps) == expected
                and all(check["pass"] for check in self.w_substeps)),
        }


def gate_record(metric: str) -> dict[str, object]:
    """Return one N2 ledger record through the binding ``gate()`` lookup."""
    return dataclasses.asdict(gate("N2", metric))


def state_field_view(state, field_names) -> dict[str, object]:
    """Expose report names, including WRF qn* aliases, without copying."""
    return {
        name: getattr(state, FIELD_ALIASES.get(name, name))
        for name in field_names
    }


def _preserve_parent_map_policy(parent_state, child_state) -> None:
    """Keep the frozen Cartesian WK82 map policy after parent-only init.

    ``parent_only_init`` normally installs the child's Lambert map fields,
    which is correct for a real-data parent.  The frozen WK82 builder is an
    f=0, unit-map idealized case; when its parent carries that identity policy,
    reproduce it on the child so the hodograph motion used for N2c placement
    remains the same case.  A projected parent is left untouched because the
    initializer has already installed its child's own projected fields.
    """
    if not hasattr(child_state, "set_map_coriolis"):
        return
    if bool(getattr(parent_state, "has_msf", False)
            or getattr(parent_state, "rotational", False)):
        return
    child_state.set_map_coriolis(
        np.ones(child_state.msft.shape),
        np.ones(child_state.msfu.shape),
        np.ones(child_state.msfv.shape),
        np.zeros(child_state.f.shape), np.zeros(child_state.e.shape),
        sina=np.zeros(child_state.sina.shape),
        cosa=np.ones(child_state.cosa.shape))


def prepare_idealized_domain(state, dc, grid, start_time):
    """Apply the production post-initialization seams to one ideal domain.

    Production root and child preparation re-derive the EOS diagnostics
    before validation and attach a distinct per-domain physics driver when
    any configured scheme needs persistent state.  Idealized cases have no
    real-data surface products, so ``initialize_physics`` receives its
    documented neutral surface defaults; radiation, when configured, still
    receives the domain's own mass-grid coordinates and common start time.
    """
    from woof.core.diagnostics import update_diagnostics
    from woof.core.physics import (initialize_physics,
                                    physics_driver_required)

    cfg = dc.run
    update_diagnostics(state, cfg.hypsometric_opt)
    if not physics_driver_required(cfg):
        return None
    lat, lon = grid.latlon_mass()
    return initialize_physics(
        state, cfg, radiation_start_time=start_time,
        radiation_latitude=lat, radiation_longitude=lon)


#: ``mp_physics`` values whose microphysics call stages a scheme-native
#: REFL_10CM field.  Identical to ``woof.runtime.REFL_10CM_MICROPHYSICS``
#: (held equal by tests/test_mp28_runtime_reachability.py) and named here
#: for the same reason: this handoff is what makes the ideal-nest cases
#: exercise production's consume-once contract, and a scheme that stages a
#: field this function does not consume raises at the NEXT history frame,
#: far from the omission.  28 belongs here because WRF's calc_refl10cm has
#: no aerosol-aware arm (module_mp_thompson.F:5710-5711, reached from the
#: single site :1458 gated on diagflag alone) and woof's mp=28 adapter
#: stages the field through the same seam as mp=8.  16 belongs here for the
#: identical structural reason: ``wdm6`` reaches ``refl10cm_wdm6`` from the
#: single site module_mp_wdm6.F:291, gated on
#: ``diagflag .and. do_radar_ref == 1`` (:279-280), and woof's mp=16
#: adapter stages the field through the same seam.  Omitting it here while
#: woof/runtime.py admitted it is exactly the failure this comment warns
#: about -- the stage happens, the consume does not, and the raise lands at
#: the next history frame.
from woof.core.physics_inventory import REFL_10CM_MICROPHYSICS


def consume_history_reflectivity(node, ticks: int) -> None:
    """Complete production's one-frame REFL handoff without writing a file.

    The due tick is THE DOMAIN'S, read off its own clock, exactly as
    ``woof.runtime._submit_tree_history_frame`` reads it: a nest that
    activates later has its analysis frame due at the activation epoch,
    not at the experiment's 0, and consuming there takes a stash no step
    of that domain has produced (#205).  A node with no clock -- every
    hand-built CPU fixture -- starts with its experiment, which is the
    0 this defaults to.
    """
    from woof.core.refl import domain_start_ticks_of, refl_10cm_stash_is_due

    if (not refl_10cm_stash_is_due(
                ticks, domain_start_ticks=domain_start_ticks_of(node))
            or node.cfg.run.mp_physics not in REFL_10CM_MICROPHYSICS):
        return
    if getattr(node.state, "physics", None) is None:
        raise RuntimeError(
            f"d{node.cfg.grid_id:02d} history has microphysics but no driver")
    from woof.core.refl import consume_refl_10cm
    consume_refl_10cm(node.state)


def assemble_idealized_tree(exp, root_state, *, grids=None,
                            child_initializer=parent_only_init,
                            coupler_factory=ConcreteNestCoupler,
                            domain_preparer=prepare_idealized_domain,
                            ) -> ExperimentState:
    """Hand-assemble the public Task-14 tree around real or mock states.

    Injectable initializer/coupler hooks keep the topology path CPU-testable;
    controller execution uses the defaults, hence real ``parent_only_init``
    and the production one-way coupler.
    """
    tick_clock = resolve_clock(exp)
    schedule = build_schedule(exp, tick_clock)
    clocks = tick_clock.clocks()
    grids = (grids_from_projection_config(exp) if grids is None else grids)
    if len(grids) != len(exp.domains):
        raise ValueError("grid inventory differs from experiment domains")

    root_dc = exp.root
    domain_preparer(root_state, root_dc, grids[0], exp.start_time)
    root = DomainNode(
        root_dc, grids[0], root_state, clocks[root_dc.grid_id],
        None, [], None)
    nodes = {root_dc.grid_id: root}
    for dc, grid in zip(exp.domains[1:], grids[1:]):
        parent = nodes[dc.parent_id]
        initialized = child_initializer(dc, parent)
        _preserve_parent_map_policy(parent.state, initialized.state)
        initialized_grid = getattr(initialized, "grid", grid)
        domain_preparer(
            initialized.state, dc, initialized_grid, exp.start_time)
        node = DomainNode(
            dc, initialized_grid, initialized.state, clocks[dc.grid_id],
            parent, [], None)
        node.coupler = coupler_factory(node)
        parent.children.append(node)
        node.state._nest_restart_classification = "REBUILT"
        nodes[dc.grid_id] = node

    model = ExperimentState(
        root, MappingProxyType(nodes), schedule, None,
        f"idealized:{exp.name}")
    model._runtime_status = ModelRuntimeStatus()
    model._resumed = False
    model._resume_committed_history_grid_ids = frozenset()
    model._scratch_arena = None
    # States are supplied to this hand-assembly path preconstructed, so it
    # cannot retroactively bind B-1 views without leaving their old blocks.
    model._dycore_state_workspace = None
    model._io_manager = None
    model._last_checkpoint = None
    return model


def identity_halo_run(exp):
    """Return the setup-only root RunConfig with SINT's donor halo."""
    run = exp.root.run
    width = 2 * IDENTITY_SINT_HALO
    return replace(run, nx=run.nx + width, ny=run.ny + width)


def ratio_one_parent_initializer(exp, halo_state):
    """Use ``parent_only_init`` with a disposable ratio-1 SINT halo.

    The returned child is on the actual coincident scaffold grid.  The
    padded parent exists only while ``parent_only_init`` evaluates SINT; its
    central 48-by-48 construction is the same idealized state as the real
    parent and is checked by the ordinary t=0 state-equivalence sample.
    """
    root_dc, child_dc = exp.domains
    if (child_dc.parent_grid_ratio, child_dc.parent_time_step_ratio) != (1, 1):
        raise ValueError("identity halo initializer requires ratios 1/1")
    halo_run = identity_halo_run(exp)
    halo_dc = replace(root_dc, run=halo_run)
    halo_exp = replace(
        exp, name=f"{exp.name}_sint_halo", domains=(halo_dc,))
    halo_grid = grids_from_projection_config(halo_exp)[0]
    actual_child_grid = grids_from_projection_config(exp)[1]
    donor_child = replace(
        child_dc,
        i_parent_start=IDENTITY_SINT_HALO + 1,
        j_parent_start=IDENTITY_SINT_HALO + 1)
    source = [halo_state]

    def initialize(dc, parent):
        if dc is not child_dc or parent.cfg is not root_dc:
            raise ValueError("identity initializer called for another tree")
        if source[0] is None:
            raise RuntimeError("identity halo initializer is single-use")
        halo_parent = SimpleNamespace(
            cfg=halo_dc, grid=halo_grid, state=source[0])
        result = parent_only_init(donor_child, halo_parent)
        source[0] = None
        return replace(result, grid=actual_child_grid)

    return initialize


def synchronized_identity_config(exp):
    """Make both identity-domain history alarms ring every parent step."""
    cadence = float(exp.root.run.dt)
    domains = tuple(replace(
        dc, history_interval_s=cadence,
        run=replace(dc.run, output_interval_s=cadence))
        for dc in exp.domains)
    return replace(exp, domains=domains)


def run_identity_case(*, exp, root_state, halo_state, metric: str, field_names,
                      inactive_fields=(), outdir: str | Path
                      ) -> dict[str, object]:
    """Execute and score all five blocking F19 identity-oracle components."""
    record = gate("N2", metric)
    if record.kind != "identity_oracle":
        raise RuntimeError(
            f"N2 metric {metric!r} changed kind to {record.kind!r}")
    grids = grids_from_projection_config(exp)
    model = assemble_idealized_tree(
        exp, root_state, grids=grids,
        child_initializer=ratio_one_parent_initializer(exp, halo_state),
        coupler_factory=IdentityOracleCoupler)
    root = model.root
    child = model.node(exp.domains[1].grid_id)
    diagnostics: list[dict[str, object]] = []
    raw_checks: list[dict[str, object]] = []
    t0_anchor = None
    active_caps = {
        name: cap for name, cap in RAW_SPECIFIED_CAPS.items()
        if name in field_names
    }
    trace = IdentityOracleTrace(
        child.state, child.cfg.run, inactive_fields=inactive_fields)

    def snapshot(_model, node, ticks) -> None:
        nonlocal t0_anchor
        consume_history_reflectivity(node, ticks)
        if node is not child:
            return
        if ticks != child.clock.ticks:
            raise RuntimeError("history tick differs from child clock")
        if child.clock.ticks != root.clock.ticks:
            raise RuntimeError(
                "identity comparison attempted while parent leads child")
        equiv = fp32_state_equiv_report(
            state_field_view(child.state, field_names),
            state_field_view(root.state, field_names), field_names)
        diagnostics.append({
            "ticks": int(ticks),
            "elapsed_seconds": float(child.clock.elapsed_seconds),
            **equiv,
        })
        if int(ticks) == 0:
            if t0_anchor is not None:
                raise RuntimeError("identity t=0 anchor sampled more than once")
            t0_anchor = bit_identity_report(
                child.state, root.state, field_names)
            return
        raw = raw_specified_row_guardrail(
            child.state, root.state, active_caps=active_caps,
            inactive_fields=inactive_fields,
            spec_zone=child.cfg.run.spec_zone)
        raw.update({
            "ticks": int(ticks),
            "elapsed_seconds": float(child.clock.elapsed_seconds),
        })
        raw_checks.append(raw)

    trace.install()
    try:
        execution = execute_experiment(model, history_handler=snapshot)
    finally:
        trace.restore()
    expected_samples = int(round(exp.run_seconds / exp.root.run.dt)) + 1
    expected_forces = int(execution.forces)
    components = {
        "t0_full_state_bit_identity": {
            "registered_fields": list(field_names),
            "sample": t0_anchor,
            "pass": bool(t0_anchor is not None and t0_anchor["pass"]),
        },
        "force_table_oracle": {
            "expected_forces": expected_forces,
            "observed_forces": len(child.coupler.oracle_forces),
            "forces": child.coupler.oracle_forces,
            "pass": bool(
                len(child.coupler.oracle_forces) == expected_forces
                and all(item["pass"]
                        for item in child.coupler.oracle_forces)),
        },
        "rk_davies_tendency_oracle": trace.davies_component(
            steps=expected_forces),
        "nested_w_substep_bit_identity": trace.w_component(
            steps=expected_forces),
        "raw_specified_row_guardrail": {
            "active_caps": active_caps,
            "inactive_fields": list(inactive_fields),
            "h_diabatic_excluded": True,
            "expected_post_step_samples": expected_samples - 1,
            "samples": raw_checks,
            "pass": bool(
                len(raw_checks) == expected_samples - 1
                and all(item["pass"] for item in raw_checks)),
        },
    }
    passed = identity_oracle_verdict(components)
    report = {
        "case": exp.name,
        "gate": dataclasses.asdict(record),
        "compared_fields": list(field_names),
        "expected_synchronized_instants": expected_samples,
        "components": components,
        "diagnostics": {
            "pre_f19_full_state_equivalence": {
                "blocking": False,
                "description": (
                    "Periodic-parent versus Davies-child full-state ULP "
                    "comparison; retained for attribution only."),
                "synchronized_instants": diagnostics,
            },
        },
        "executor": {
            "steps": int(execution.steps),
            "forces": int(execution.forces),
            "feedback_calls": int(execution.feedback_calls),
        },
        "initializer": "parent_only_init",
        "identity_sint_source_halo_cells": IDENTITY_SINT_HALO,
        "coupler": "IdentityOracleCoupler",
        "verdict_rule": "pass iff every registered component passes",
        "pass": bool(passed),
    }
    write_json(Path(outdir) / "report.json", report)
    return report


def write_json(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    return path


__all__ = [
    "DAVIES_ATOL", "DAVIES_RTOL", "FIELD_ALIASES", "IDENTITY_COMPONENTS",
    "IDENTITY_SINT_HALO", "INACTIVE_MOIST_FIELDS", "IdentityOracleCoupler",
    "IdentityOracleTrace", "RAW_SPECIFIED_CAPS", "RatioOneIdentityCoupler",
    "assemble_idealized_tree", "bit_identity_report",
    "consume_history_reflectivity", "davies_tendency_oracle",
    "force_table_oracle", "gate_record", "identity_halo_run",
    "identity_oracle_verdict", "nested_w_substep_oracle",
    "prepare_idealized_domain", "raw_specified_row_guardrail",
    "ratio_one_parent_initializer", "run_identity_case",
    "state_field_view", "synchronized_identity_config", "write_json",
]
