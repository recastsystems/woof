#!/usr/bin/env python3
# ruff: noqa: E402
"""Run the synthetic v8.4.1 CPU-to-CUDA numerical and mutation ruler.

This is deliberately an implementation ruler, not native MPAS authority.  It
uses one prepared float32 x1.2562 state for both implementations, captures the
outputs of the production operators reached by one complete step, and then
requires named CUDA-source mutations to exceed a CPU-authority-derived budget.
The CUDA step receipt must retain its exact four-byte safety D2H claim; ruler
downloads happen only after ``step_device`` returns.
"""

from __future__ import annotations
import os as _hex_os
_hex_os.environ["WOOF_HEX_NUMERICS"] = "strict"  # a bitwise proof against native MPAS runs strict numerics (fast is the default, strict the opt-in)

import argparse
import contextlib
import hashlib
import json
import os
import re
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DEFAULT_GRID = ROOT / "data" / "meshes" / "x1.2562" / "x1.2562.grid.nc"
DEFAULT_STATIC = ROOT / "data" / "meshes" / "x1.2562" / "x1.2562.static.nc"
COMPILED_REPORT = (
    ROOT
    / "receipts"
    / "mpas-v841-compiled-endpoint"
    / "v841-compiled-endpoint-report.json"
)
NUMERIC_TEST = ROOT / "tests" / "test_cuda_v841_numeric_ruler.py"

STATE_FIELDS = ("rho", "rho_theta", "rho_u", "rho_w", "scalars")
SAVED_FIELDS = (
    "theta_m",
    "exner",
    "density_perturbation",
    "rho_theta_perturbation",
    "pressure_perturbation",
    "normal_velocity",
    "vertical_velocity",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def numeric_authority_snapshot() -> dict[str, Any]:
    paths = sorted((ROOT / "src" / "hexcore").rglob("*.py"))
    paths.extend((Path(__file__).resolve(), NUMERIC_TEST, COMPILED_REPORT))
    files = {
        str(path.relative_to(ROOT)).replace("\\", "/"): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(set(paths))
    }
    return {
        "file_count": len(files),
        "files_sha256": canonical_json_sha256(files),
        "files": files,
    }


def input_file_snapshot(grid: Path, static: Path) -> dict[str, Any]:
    return {
        "grid": {
            "path": str(grid),
            "bytes": grid.stat().st_size,
            "sha256": sha256_file(grid),
        },
        "static": {
            "path": str(static),
            "bytes": static.stat().st_size,
            "sha256": sha256_file(static),
        },
    }


def prepared_case_fingerprint(case: SimpleNamespace) -> dict[str, Any]:
    arrays: dict[str, str] = {}
    for name in STATE_FIELDS:
        arrays[f"state.{name}"] = array_sha256(getattr(case.state, name))
    for name in SAVED_FIELDS:
        arrays[f"saved.{name}"] = array_sha256(getattr(case.saved, name))
    for name in case.vertical.__slots__:
        arrays[f"vertical.{name}"] = array_sha256(getattr(case.vertical, name))
    for name in case.reference.__slots__:
        arrays[f"reference.{name}"] = array_sha256(getattr(case.reference, name))
    for name in ("zb_cell", "zb3_cell"):
        arrays[f"terrain.{name}"] = array_sha256(getattr(case.terrain, name))
    for name in ("u_init", "v_init"):
        arrays[f"profiles.{name}"] = array_sha256(getattr(case.profiles, name))
    for name in (
        "adv_coefs",
        "adv_coefs_3rd",
        "n_adv_cells_for_edge",
        "adv_cells_for_edge",
    ):
        arrays[f"advection.{name}"] = array_sha256(getattr(case.coefficients, name))
    payload = {
        "configuration": asdict(case.config),
        "time_seconds": float(case.state.time_seconds),
        "arrays": arrays,
    }
    return {**payload, "sha256": canonical_json_sha256(payload)}


@dataclass(frozen=True, slots=True)
class ErrorPolicy:
    policy_id: str
    operation_rounding_sites: int
    exact_bits: bool
    derivation: str
    local_scale_rule: str = (
        "per-element abs(authority); one binary32 spacing at that exact local "
        "scale; zero uses the smallest binary32 subnormal; no global-scale or "
        "measured-gap floor"
    )


# Counts are declared once from maximum sequential source-order operations in
# this fixed 3-level, x1.2562, split3/substep2 deck.  They are conservative
# operation counts, never fitted CPU/CUDA gaps.
_DIRECT_SITES = {
    "acoustic": 192,
    "theta_tendency": 144,
    "w_tendency": 144,
    "vector_momentum": 160,
    "scalar_transport": 192,
}
CUDA_STANDARD_TRIG_MAX_ULP = 2
DIRECT_VECTOR_CANCELLATION_INDEX = (2, 7262)
DIRECT_VECTOR_CANCELLATION_CELLS = (273, 1936)
DIRECT_VECTOR_MESH_ARRAY_SHA256 = {
    "cellsOnEdge": "ea5acb5876e434f78eba71a9b6852f8ae79f833ae38a591f901f804c00b23e6d",
    "edgesOnEdge": "a6f7de34d4146d9b5c86c72adb4f8c8871e43fa6914b0d4e34eb9064d046e15a",
    "nEdgesOnEdge": "587337b2ec0fe25f6daa48db3d5c88b530855776a3e71136a02d98af8da9c13a",
    "angleEdge": "c0d975aa98e9a67a992b963ff4509f28bd24242b8f60555652538205781bec32",
    "weightsOnEdge": "bc277ae2f8a396d2e5d2612b3d3d61fa15d29b2e58067f9564b6d918a0331aea",
    "fEdge": "fbe44921cfff37d2c213243e0c15e87ffa876728b16df6d9410fae14d495a94f",
    "dcEdge": "ea4169d0a13102b8bce85676f2990ed859b363be746aac36cc841ca2ead6a6cc",
}
DIRECT_VECTOR_MESH_SHA256 = (
    "c2eecf9f349ca9e319bbf328f51e10982840c86ffc0ef5492751797d9e243640"
)
CUDA_STANDARD_TRIG_ACCURACY_SOURCE = (
    "https://docs.nvidia.com/cuda/cuda-programming-guide/"
    "05-appendices/mathematical-functions.html#trigonometric-functions"
)
_DIRECT_TRIG_PROBE_MODULE = "hexcore.cuda_v841_numeric_trig_probe"
_DIRECT_TRIG_PROBE_SOURCE = r"""
extern "C" __global__ void reached_angle_sincos_f32(
    const int count, const float *angle, float *sine, float *cosine)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= count) return;
    sine[index] = sinf(angle[index]);
    cosine[index] = cosf(angle[index]);
}
"""
_WHOLE_STEP_SITES = (
    3
    * 3
    * (
        2 * _DIRECT_SITES["acoustic"]
        + _DIRECT_SITES["theta_tendency"]
        + _DIRECT_SITES["w_tendency"]
        + _DIRECT_SITES["vector_momentum"]
    )
    + 3 * _DIRECT_SITES["scalar_transport"]
    + 6
)

ERROR_POLICIES: dict[str, ErrorPolicy] = {
    "one_step_state": ErrorPolicy(
        "one_step_state",
        _WHOLE_STEP_SITES,
        False,
        "conservative max-per-stage envelope: 3 splits * 3 RK stages * "
        "(2 acoustic substeps * 192 + theta 144 + w 144 + vector 160) + "
        "3 scalar RK stages * 192 + 6 split reductions; the exact admitted "
        "stage schedule is [1,1,2] (4, not 6, acoustic calls per split)",
    ),
    "one_step_saved": ErrorPolicy(
        "one_step_saved",
        _WHOLE_STEP_SITES,
        False,
        "same carried production graph as one_step_state; saved diagnostics are "
        "published from the final dynamics stage",
    ),
    "direct_acoustic": ErrorPolicy(
        "direct_acoustic",
        _DIRECT_SITES["acoustic"],
        False,
        "fixed x1.2562 max 6 cell-edge reduction plus 3-level "
        "RHS/Thomas/recovery chain; 192 is the audited sequential-site ceiling",
    ),
    "direct_theta_tendency": ErrorPolicy(
        "direct_theta_tendency",
        _DIRECT_SITES["theta_tendency"],
        False,
        "max 10 advection cells per edge plus max 6 edges per cell and vertical flux",
    ),
    "direct_w_tendency": ErrorPolicy(
        "direct_w_tendency",
        _DIRECT_SITES["w_tendency"],
        False,
        "max 10 advection cells per edge plus max 6 edges per cell and vertical flux",
    ),
    "direct_vector_momentum": ErrorPolicy(
        "direct_vector_momentum",
        _DIRECT_SITES["vector_momentum"],
        False,
        "20 neighboring-edge/reference terms with at most 8 rounded ops each",
    ),
    "direct_scalar_transport": ErrorPolicy(
        "direct_scalar_transport",
        _DIRECT_SITES["scalar_transport"],
        False,
        "max 10 advection cells per edge, max 6 edges per cell, vertical order-3 "
        "flux, RK numerator, and density divide",
    ),
    "direct_split_flux": ErrorPolicy(
        "direct_split_flux",
        2,
        False,
        "one float32 accumulator add and one reciprocal multiply per captured reduction",
    ),
    "trajectory_dynamics_checkpoint": ErrorPolicy(
        "trajectory_dynamics_checkpoint",
        9,
        False,
        "one companion-parent-scale binary32 allowance for each of the nine "
        "executed dynamics RK stages",
    ),
    "trajectory_acoustic_checkpoint": ErrorPolicy(
        "trajectory_acoustic_checkpoint",
        12,
        False,
        "one companion-parent-scale binary32 allowance for each of the twelve "
        "executed acoustic calls in the three split dynamics subcycles",
    ),
    "trajectory_split_flux_checkpoint": ErrorPolicy(
        "trajectory_split_flux_checkpoint",
        12,
        False,
        "nine dynamics RK stages, two ordered split-flux additions, and one "
        "typed reciprocal finalization at a companion momentum scale",
    ),
    "exact_rw_endpoints": ErrorPolicy(
        "exact_rw_endpoints",
        0,
        True,
        "both implementations assign exact +0.0f endpoint bits",
    ),
    "exact_uploaded_profile": ErrorPolicy(
        "exact_uploaded_profile",
        0,
        True,
        "copy-only host-to-device float32 profile anchor",
    ),
}

POLICY_SOURCE_AUDIT = {
    "fixed_deck": {
        "n_vert_levels": 3,
        "max_n_edges_on_cell": 6,
        "max_n_edges_on_edge": 10,
        "max_advection_cells_per_edge": 10,
        "dynamics_split_steps": 3,
        "stage_acoustic_steps": [1, 1, 2],
    },
    "policy_operation_sites": {
        name: policy.operation_rounding_sites for name, policy in ERROR_POLICIES.items()
    },
    "whole_step_max_per_stage_note": (
        "whole-step policy budgets two acoustic calls at each of three RK stages; "
        "the actual [1,1,2] schedule performs four calls per split, so the six-call "
        "term is a declared conservative ceiling, not a measured fit"
    ),
}


@dataclass(frozen=True, slots=True)
class AuthorityErrorBudget:
    """A fixed-operation, element-local budget derived only from CPU authority."""

    policy_id: str
    dtype: str
    authority_scale: float
    operation_rounding_sites: int
    exact_bits: bool
    local_scale_rule: str
    derivation: str
    minimum_local_ulp: float
    maximum_local_ulp: float
    minimum_local_absolute_budget: float
    max_absolute_error: float
    smallest_defect_that_must_fail: float


@dataclass(frozen=True, slots=True)
class ArrayComparison:
    name: str
    authority_sha256: str
    candidate_sha256: str
    shape: tuple[int, ...]
    dtype: str
    max_absolute_error: float
    different_elements: int
    failing_elements: int
    passed: bool
    budget: AuthorityErrorBudget


def _local_ulp_and_budget(
    authority: np.ndarray, policy: ErrorPolicy
) -> tuple[np.ndarray, np.ndarray]:
    magnitude = np.abs(authority)
    local_ulp = (
        np.nextafter(magnitude, np.float32(np.inf), dtype=np.float32) - magnitude
    ).astype(np.float64)
    if not np.all(np.isfinite(local_ulp)):
        raise ValueError("authority magnitude is too large for a finite local ULP")
    return local_ulp, local_ulp * policy.operation_rounding_sites


def authority_error_budget(
    authority: Any,
    *,
    policy_id: str,
) -> AuthorityErrorBudget:
    """Declare a ruler without consulting a CUDA candidate or measured gap."""

    array = np.asarray(authority)
    if array.dtype != np.dtype(np.float32):
        raise TypeError("the v8.4.1 ruler accepts exact float32 authority arrays")
    if not np.all(np.isfinite(array)):
        raise ValueError("authority array must be finite")
    try:
        policy = ERROR_POLICIES[policy_id]
    except KeyError:
        raise ValueError(f"unknown authority error policy {policy_id!r}") from None
    local_ulp, local_budget = _local_ulp_and_budget(array, policy)
    minimum_budget = float(np.min(local_budget, initial=np.inf))
    maximum_budget = float(np.max(local_budget, initial=0.0))
    return AuthorityErrorBudget(
        policy_id=policy.policy_id,
        dtype="float32",
        authority_scale=float(np.max(np.abs(array), initial=np.float32(0.0))),
        operation_rounding_sites=policy.operation_rounding_sites,
        exact_bits=policy.exact_bits,
        local_scale_rule=policy.local_scale_rule,
        derivation=policy.derivation,
        minimum_local_ulp=float(np.min(local_ulp, initial=np.inf)),
        maximum_local_ulp=float(np.max(local_ulp, initial=0.0)),
        minimum_local_absolute_budget=minimum_budget,
        max_absolute_error=maximum_budget,
        smallest_defect_that_must_fail=float(np.nextafter(maximum_budget, np.inf)),
    )


def compare_array(
    name: str,
    authority: Any,
    candidate: Any,
    *,
    policy_id: str,
) -> ArrayComparison:
    expected = np.asarray(authority)
    actual = np.asarray(candidate)
    if expected.shape != actual.shape:
        raise ValueError(f"{name}: shape {actual.shape} != authority {expected.shape}")
    if expected.dtype != np.dtype(np.float32) or actual.dtype != expected.dtype:
        raise TypeError(
            f"{name}: authority/candidate must share exact float32, got "
            f"{expected.dtype}/{actual.dtype}"
        )
    if not np.all(np.isfinite(actual)):
        raise FloatingPointError(f"{name}: candidate contains non-finite values")
    budget = authority_error_budget(expected, policy_id=policy_id)
    policy = ERROR_POLICIES[policy_id]
    _, local_budget = _local_ulp_and_budget(expected, policy)
    absolute = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    different_bits = actual.view(np.uint32) != expected.view(np.uint32)
    failures = different_bits if policy.exact_bits else absolute > local_budget
    return ArrayComparison(
        name=name,
        authority_sha256=array_sha256(expected),
        candidate_sha256=array_sha256(actual),
        shape=tuple(expected.shape),
        dtype=str(expected.dtype),
        max_absolute_error=float(np.max(absolute, initial=0.0)),
        different_elements=int(np.count_nonzero(different_bits)),
        failing_elements=int(np.count_nonzero(failures)),
        passed=not bool(np.any(failures)),
        budget=budget,
    )


def upward_float32_activity(activity: Any) -> np.ndarray:
    """Round a nonnegative float64 authority envelope toward ``+inf`` in f32."""

    source = np.asarray(activity)
    if source.dtype != np.dtype(np.float64):
        raise TypeError("source activity must be accumulated in float64")
    if not np.all(np.isfinite(source)) or np.any(source < 0.0):
        raise ValueError("source activity must be finite and nonnegative")
    nearest = source.astype(np.float32)
    if not np.all(np.isfinite(nearest)):
        raise ValueError("source activity exceeds finite float32")
    rounded_up = np.nextafter(
        nearest,
        np.float32(np.inf),
        dtype=np.float32,
    )
    carrier = np.where(nearest.astype(np.float64) < source, rounded_up, nearest)
    if np.any(carrier.astype(np.float64) < source):
        raise RuntimeError("float32 activity carrier rounded below its authority")
    return np.ascontiguousarray(carrier, dtype=np.float32)


def _mesh_authority_array(mesh: Any, name: str) -> np.ndarray:
    try:
        return np.asarray(getattr(mesh, name))
    except AttributeError:
        arrays = getattr(mesh, "arrays", None)
        if arrays is None or name not in arrays:
            raise AttributeError(f"mesh has no authority field {name!r}") from None
        return np.asarray(arrays[name])


def vector_momentum_source_activity_v841(
    mesh: Any,
    *,
    normal_velocity: Any,
    rho_edge: Any,
    pv_edge: Any,
    kinetic_energy: Any,
    horizontal_divergence: Any,
    reference_wind: Any,
) -> np.ndarray:
    """Return the per-output CPU source-activity envelope for v8.4.1 momentum.

    Every source operand is first admitted as exact float32.  The absolute-term
    envelope is then accumulated in float64, so cancellation in ``q``, the
    kinetic-energy gradient, reference Coriolis, or divergence term cannot
    shrink the scale used by the direct CPU-to-CUDA ruler.
    """

    def f32(name: str, value: Any, shape: tuple[int, ...]) -> np.ndarray:
        array = np.asarray(value)
        if array.dtype != np.dtype(np.float32) or array.shape != shape:
            raise TypeError(
                f"{name} must be exact float32 with shape {shape}, got "
                f"{array.dtype}/{array.shape}"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be finite")
        return array

    u = np.asarray(normal_velocity)
    if u.dtype != np.dtype(np.float32) or u.ndim != 2:
        raise TypeError("normal_velocity must be an exact float32 (level,edge) array")
    nlev, nedges = map(int, u.shape)
    cells = _mesh_authority_array(mesh, "cellsOnEdge")
    edges = _mesh_authority_array(mesh, "edgesOnEdge")
    counts = _mesh_authority_array(mesh, "nEdgesOnEdge")
    if cells.shape != (nedges, 2) or counts.shape != (nedges,):
        raise ValueError("vector-momentum mesh edge topology shape changed")
    if edges.ndim != 2 or edges.shape[0] != nedges:
        raise ValueError("edgesOnEdge shape changed")
    ncells = int(np.max(cells, initial=-1)) + 1
    if ncells < 1:
        raise ValueError("cellsOnEdge has no owned cells")
    rho = f32("rho_edge", rho_edge, (nlev, nedges))
    pv = f32("pv_edge", pv_edge, (nlev, nedges))
    kinetic = f32("kinetic_energy", kinetic_energy, (nlev, ncells))
    divergence = f32(
        "horizontal_divergence", horizontal_divergence, (nlev, ncells)
    )
    weights = f32(
        "weightsOnEdge",
        _mesh_authority_array(mesh, "weightsOnEdge").astype(np.float32, copy=False),
        edges.shape,
    )
    angle = f32(
        "angleEdge",
        _mesh_authority_array(mesh, "angleEdge").astype(np.float32, copy=False),
        (nedges,),
    )
    coriolis = f32(
        "fEdge",
        _mesh_authority_array(mesh, "fEdge").astype(np.float32, copy=False),
        (nedges,),
    )
    from woof.hex.dynamics_v841 import precomputed_mesh_inverse_v841

    inv_dc = precomputed_mesh_inverse_v841(
        mesh, "dcEdge", np.dtype(np.float32)
    )
    inv_dc = f32("invDcEdge", inv_dc, (nedges,))
    reference_wind.validate(nlev, np.dtype(np.float32))
    u_init = f32("u_init", reference_wind.u_init, (nlev,))
    v_init = f32("v_init", reference_wind.v_init, (nlev,))

    # These are the CPU authority's float32 transcendental results.  Converting
    # them to float64 below decodes their exact bits; it does not recompute trig.
    cosine = np.cos(angle, dtype=np.float32)
    sine = np.sin(angle, dtype=np.float32)
    cosine_magnitude = np.abs(cosine)
    sine_magnitude = np.abs(sine)
    cosine_ulp = (
        np.nextafter(cosine_magnitude, np.float32(np.inf), dtype=np.float32)
        - cosine_magnitude
    ).astype(np.float64)
    sine_ulp = (
        np.nextafter(sine_magnitude, np.float32(np.inf), dtype=np.float32)
        - sine_magnitude
    ).astype(np.float64)
    cosine_bound = cosine_magnitude.astype(np.float64) + (
        CUDA_STANDARD_TRIG_MAX_ULP * cosine_ulp
    )
    sine_bound = sine_magnitude.astype(np.float64) + (
        CUDA_STANDARD_TRIG_MAX_ULP * sine_ulp
    )
    activity = np.empty((nlev, nedges), dtype=np.float64)
    for edge, (cell0_raw, cell1_raw) in enumerate(cells):
        cell0 = int(cell0_raw)
        cell1 = int(cell1_raw)
        count = int(counts[edge])
        if count < 0 or count > edges.shape[1]:
            raise ValueError(f"nEdgesOnEdge[{edge}]={count} is invalid")
        neighbors = np.asarray(edges[edge, :count], dtype=np.int64)
        if np.any(neighbors < 0) or np.any(neighbors >= nedges):
            raise ValueError(f"edgesOnEdge[{edge}] has an invalid neighbor")
        weight = weights[edge, :count].astype(np.float64)
        f_value = float(np.float64(coriolis[edge]))
        inv_dc_value = abs(float(np.float64(inv_dc[edge])))
        for level in range(nlev):
            pv_activity = np.sum(
                np.abs(weight * u[level, neighbors].astype(np.float64))
                * np.float64(0.5)
                * (
                    abs(float(np.float64(pv[level, edge])))
                    + np.abs(pv[level, neighbors].astype(np.float64))
                ),
                dtype=np.float64,
            )
            reference_activity = np.sum(
                np.abs(weight * f_value)
                * (
                    abs(float(np.float64(u_init[level])))
                    * cosine_bound[neighbors]
                    + abs(float(np.float64(v_init[level])))
                    * sine_bound[neighbors]
                ),
                dtype=np.float64,
            )
            gradient_activity = (
                abs(float(np.float64(kinetic[level, cell1])))
                + abs(float(np.float64(kinetic[level, cell0])))
            ) * inv_dc_value
            divergence_activity = (
                abs(float(np.float64(u[level, edge])))
                * np.float64(0.5)
                * (
                    abs(float(np.float64(divergence[level, cell0])))
                    + abs(float(np.float64(divergence[level, cell1])))
                )
            )
            activity[level, edge] = (
                abs(float(np.float64(rho[level, edge])))
                * (pv_activity + reference_activity + gradient_activity)
                + divergence_activity
            )
    if not np.all(np.isfinite(activity)) or np.any(activity < 0.0):
        raise RuntimeError("vector-momentum source activity is invalid")
    return np.ascontiguousarray(activity)


def compare_array_at_activity_scale(
    name: str,
    authority: Any,
    candidate: Any,
    *,
    policy_id: str,
    activity: Any,
    activity_name: str,
) -> ArrayComparison:
    """Compare a shared-input operator at a source-linked activity scale."""

    expected = np.asarray(authority)
    actual = np.asarray(candidate)
    if expected.shape != actual.shape:
        raise ValueError(f"{name}: shape {actual.shape} != authority {expected.shape}")
    if expected.dtype != np.dtype(np.float32) or actual.dtype != expected.dtype:
        raise TypeError(
            f"{name}: authority/candidate must share exact float32, got "
            f"{expected.dtype}/{actual.dtype}"
        )
    if not np.all(np.isfinite(expected)) or not np.all(np.isfinite(actual)):
        raise FloatingPointError(f"{name}: authority or candidate is non-finite")
    carrier = upward_float32_activity(activity)
    if carrier.shape != expected.shape:
        raise ValueError(
            f"{name}: activity {activity_name} shape {carrier.shape} != {expected.shape}"
        )
    try:
        policy = ERROR_POLICIES[policy_id]
    except KeyError:
        raise ValueError(f"unknown authority error policy {policy_id!r}") from None
    if policy.exact_bits:
        raise ValueError("activity-scale comparison cannot use an exact-bit policy")
    local_ulp = (
        np.nextafter(carrier, np.float32(np.inf), dtype=np.float32) - carrier
    ).astype(np.float64)
    if not np.all(np.isfinite(local_ulp)) or np.any(local_ulp <= 0.0):
        raise ValueError(f"{name}: activity {activity_name} has invalid spacing")
    local_budget = local_ulp * policy.operation_rounding_sites
    maximum_budget = float(np.max(local_budget, initial=0.0))
    budget = AuthorityErrorBudget(
        policy_id=policy.policy_id,
        dtype="float32",
        authority_scale=float(np.max(carrier, initial=np.float32(0.0))),
        operation_rounding_sites=policy.operation_rounding_sites,
        exact_bits=False,
        local_scale_rule=(
            f"per-element upward-float32 CPU authority source activity "
            f"{activity_name}; candidate values and measured gaps are excluded"
        ),
        derivation=policy.derivation,
        minimum_local_ulp=float(np.min(local_ulp, initial=np.inf)),
        maximum_local_ulp=float(np.max(local_ulp, initial=0.0)),
        minimum_local_absolute_budget=float(
            np.min(local_budget, initial=np.inf)
        ),
        max_absolute_error=maximum_budget,
        smallest_defect_that_must_fail=float(
            np.nextafter(maximum_budget, np.inf)
        ),
    )
    absolute = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    different_bits = actual.view(np.uint32) != expected.view(np.uint32)
    failures = absolute > local_budget
    return ArrayComparison(
        name=name,
        authority_sha256=array_sha256(expected),
        candidate_sha256=array_sha256(actual),
        shape=tuple(expected.shape),
        dtype=str(expected.dtype),
        max_absolute_error=float(np.max(absolute, initial=0.0)),
        different_elements=int(np.count_nonzero(different_bits)),
        failing_elements=int(np.count_nonzero(failures)),
        passed=not bool(np.any(failures)),
        budget=budget,
    )


def compare_array_at_parent_scale(
    name: str,
    authority: Any,
    candidate: Any,
    *,
    policy_id: str,
    parent: Any,
    parent_name: str,
) -> ArrayComparison:
    """Compare a cancellation residual at a declared authority-parent scale."""

    expected = np.asarray(authority)
    actual = np.asarray(candidate)
    basis = np.asarray(parent)
    if expected.shape != actual.shape:
        raise ValueError(f"{name}: shape {actual.shape} != authority {expected.shape}")
    if expected.dtype != np.dtype(np.float32) or actual.dtype != expected.dtype:
        raise TypeError(
            f"{name}: authority/candidate must share exact float32, got "
            f"{expected.dtype}/{actual.dtype}"
        )
    if basis.shape != expected.shape:
        raise ValueError(
            f"{name}: parent basis {parent_name} shape {basis.shape} != {expected.shape}"
        )
    if basis.dtype != np.dtype(np.float32):
        raise TypeError(f"{name}: parent basis {parent_name} must be exact float32")
    if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(basis)):
        raise FloatingPointError(f"{name}: candidate or parent basis is non-finite")
    try:
        policy = ERROR_POLICIES[policy_id]
    except KeyError:
        raise ValueError(f"unknown authority error policy {policy_id!r}") from None
    if policy.exact_bits:
        raise ValueError("parent-scale comparison cannot use an exact-bit policy")
    condition_scale = np.maximum(np.abs(expected), np.abs(basis))
    local_ulp = (
        np.nextafter(
            condition_scale,
            np.float32(np.inf),
            dtype=np.float32,
        )
        - condition_scale
    ).astype(np.float64)
    if not np.all(np.isfinite(local_ulp)) or np.any(local_ulp <= 0.0):
        raise ValueError(f"{name}: parent basis {parent_name} has invalid spacing")
    local_budget = local_ulp * policy.operation_rounding_sites
    minimum_ulp = float(np.min(local_ulp, initial=np.inf))
    maximum_ulp = float(np.max(local_ulp, initial=0.0))
    minimum_budget = float(np.min(local_budget, initial=np.inf))
    maximum_budget = float(np.max(local_budget, initial=0.0))
    budget = AuthorityErrorBudget(
        policy_id=policy.policy_id,
        dtype="float32",
        authority_scale=float(
            np.max(condition_scale, initial=np.float32(0.0))
        ),
        operation_rounding_sites=policy.operation_rounding_sites,
        exact_bits=False,
        local_scale_rule=(
            f"per-element max(abs(authority output), abs({parent_name})); used "
            "only for a declared cancellation/propagated-trajectory field"
        ),
        derivation=policy.derivation,
        minimum_local_ulp=minimum_ulp,
        maximum_local_ulp=maximum_ulp,
        minimum_local_absolute_budget=minimum_budget,
        max_absolute_error=maximum_budget,
        smallest_defect_that_must_fail=float(np.nextafter(maximum_budget, np.inf)),
    )
    absolute = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    different_bits = actual.view(np.uint32) != expected.view(np.uint32)
    failures = absolute > local_budget
    return ArrayComparison(
        name=name,
        authority_sha256=array_sha256(expected),
        candidate_sha256=array_sha256(actual),
        shape=tuple(expected.shape),
        dtype=str(expected.dtype),
        max_absolute_error=float(np.max(absolute, initial=0.0)),
        different_elements=int(np.count_nonzero(different_bits)),
        failing_elements=int(np.count_nonzero(failures)),
        passed=not bool(np.any(failures)),
        budget=budget,
    )


@dataclass(frozen=True, slots=True)
class SourceMutation:
    name: str
    family: str
    module_key: str
    before: str
    after: str
    required_capture_prefix: str


SOURCE_MUTATIONS: tuple[SourceMutation, ...] = (
    SourceMutation(
        "acoustic-rhs-sign",
        "acoustic_rhs",
        "hexcore.cuda_acoustic_v841",
        "r = mpas_sub(r, flux);",
        "r = mpas_add(r, flux);",
        "acoustic.",
    ),
    SourceMutation(
        "acoustic-forward-solve-sign",
        "acoustic_solve",
        "hexcore.cuda_acoustic_v841",
        "rw_p[i] = mpas_mul(mpas_sub(rw_p[i], mpas_mul(mpas_mul(",
        "rw_p[i] = mpas_mul(mpas_add(rw_p[i], mpas_mul(mpas_mul(",
        "acoustic.",
    ),
    SourceMutation(
        "theta-edge-flux-sign",
        "theta_transport",
        "hexcore.cuda_driver",
        "float value = mpas_mul(ru[index], edge_theta);",
        "float value = mpas_sub(0.0f, mpas_mul(ru[index], edge_theta));",
        "theta_tendency.",
    ),
    SourceMutation(
        "w-edge-flux-sign",
        "w_transport",
        "hexcore.cuda_driver",
        "value = mpas_mul(ru_interface, edge_w);",
        "value = mpas_sub(0.0f, mpas_mul(ru_interface, edge_w));",
        "w_tendency.",
    ),
    SourceMutation(
        "scalar-rk-numerator-sign",
        "scalar_transport",
        "hexcore.cuda_transport_v841",
        "const float numerator = mpas_add(\n                mpas_mul(old[index], rho_old[C2(k, cell, ncells)]),\n                mpas_mul(dt, mpas_sub(tendency, vertical)));",
        "const float numerator = mpas_sub(\n                mpas_mul(old[index], rho_old[C2(k, cell, ncells)]),\n                mpas_mul(dt, mpas_sub(tendency, vertical)));",
        "scalar_transport.",
    ),
    SourceMutation(
        "split-average-times-two",
        "split_reduction",
        "hexcore.cuda_dynamics_v841",
        "average[index] = mpas_mul(accumulator[index], reciprocal);",
        "average[index] = mpas_mul(accumulator[index], mpas_mul(2.0f, reciprocal));",
        "split_flux.",
    ),
    SourceMutation(
        "rw-bottom-endpoint-one",
        "rw_endpoints",
        "hexcore.cuda_dynamics_v841",
        "rw[C2(0, cell, ncells)] = 0.0f;",
        "rw[C2(0, cell, ncells)] = 1.0f;",
        "rw_endpoints.",
    ),
    SourceMutation(
        "reference-coriolis-sign",
        "reference_wind",
        "hexcore.cuda_dynamics_v841",
        "q = mpas_sub(q, mpas_mul(mpas_mul(\n                weights_on_edge[offset], reference_u), f_edge[edge]));",
        "q = mpas_add(q, mpas_mul(mpas_mul(\n                weights_on_edge[offset], reference_u), f_edge[edge]));",
        "vector_momentum.",
    ),
)


DIRECT_VECTOR_MUTATIONS: tuple[SourceMutation, ...] = (
    SourceMutation(
        "reference-cosine-to-sine",
        "reference_wind_trigonometry",
        "hexcore.cuda_dynamics_v841",
        "mpas_mul(u_init[k], cosf(angle_edge[neighbor]))",
        "mpas_mul(u_init[k], sinf(angle_edge[neighbor]))",
        "direct_vector_momentum.",
    ),
    SourceMutation(
        "reference-coriolis-double",
        "reference_wind_source",
        "hexcore.cuda_dynamics_v841",
        "weights_on_edge[offset], reference_u), f_edge[edge]));",
        (
            "weights_on_edge[offset], reference_u), "
            "mpas_mul(2.0f, f_edge[edge])));"
        ),
        "direct_vector_momentum.",
    ),
)


class MutatingKernelCache:
    """Exact-one-token source mutation proxy around a fresh KernelCache."""

    def __init__(self, base: Any, mutation: SourceMutation) -> None:
        self.base = base
        self.mutation = mutation
        self.applied_calls = 0
        self.original_source_sha256: str | None = None
        self.mutated_source_sha256: str | None = None
        self.mutated_source_text: str | None = None

    def raw_kernel(
        self,
        name: str,
        source: str,
        *,
        module_key: str,
        options: Sequence[str] = (),
    ) -> Any:
        selected = source
        if module_key == self.mutation.module_key:
            count = source.count(self.mutation.before)
            if count != 1:
                raise RuntimeError(
                    f"{self.mutation.name}: expected one source token in "
                    f"{module_key}, found {count}"
                )
            selected = source.replace(self.mutation.before, self.mutation.after, 1)
            self.applied_calls += 1
            self.original_source_sha256 = hashlib.sha256(source.encode()).hexdigest()
            self.mutated_source_sha256 = hashlib.sha256(selected.encode()).hexdigest()
            self.mutated_source_text = selected
        return self.base.raw_kernel(
            name, selected, module_key=module_key, options=tuple(options)
        )

    def raw_kernels(self, *args: Any, **kwargs: Any) -> Any:
        return self.base.raw_kernels(*args, **kwargs)

    def compile_manifest(self) -> Any:
        return self.base.compile_manifest()

    def evidence(self) -> dict[str, Any]:
        if self.applied_calls < 1:
            raise RuntimeError(f"mutation {self.mutation.name} was never reached")
        return {
            "name": self.mutation.name,
            "family": self.mutation.family,
            "module_key": self.mutation.module_key,
            "applied_kernel_requests": self.applied_calls,
            "original_source_sha256": self.original_source_sha256,
            "mutated_source_sha256": self.mutated_source_sha256,
        }


def _validate_direct_vector_mutation_provenance(
    production_source: str,
    mutation: SourceMutation,
    proxy: MutatingKernelCache,
) -> dict[str, Any]:
    """Prove an exact one-token direct-vector delta from live production source."""

    if mutation.module_key != "hexcore.cuda_dynamics_v841":
        raise RuntimeError("direct vector mutation targets the wrong production module")
    token_count = production_source.count(mutation.before)
    if token_count != 1:
        raise RuntimeError(
            f"direct vector mutation {mutation.name} source token count is {token_count}"
        )
    expected_mutated = production_source.replace(
        mutation.before, mutation.after, 1
    )
    original_sha = hashlib.sha256(production_source.encode("utf-8")).hexdigest()
    mutated_sha = hashlib.sha256(expected_mutated.encode("utf-8")).hexdigest()
    if (
        proxy.mutation != mutation
        or proxy.applied_calls != 1
        or proxy.original_source_sha256 != original_sha
        or proxy.mutated_source_sha256 != mutated_sha
        or proxy.mutated_source_text != expected_mutated
    ):
        raise RuntimeError(
            f"direct vector mutation {mutation.name} is not the exact live-source delta"
        )
    return {
        **proxy.evidence(),
        "production_before_token_count": token_count,
        "exact_single_replacement": True,
        "production_source_sha256": original_sha,
        "expected_mutated_source_sha256": mutated_sha,
    }


def ruler_config(**updates: Any) -> "V841DryDycoreConfig":  # noqa: F821
    from woof.hex.config_v841 import V841DryDycoreConfig

    values: dict[str, Any] = {
        "config_dt": 1.0,
        "config_number_of_sub_steps": 2,
        "config_dynamics_split_steps": 3,
        "config_split_dynamics_transport": True,
        "config_scalar_advection": True,
        "config_monotonic": False,
        "config_positive_definite": False,
        "config_apvm_upwinding": 0.5,
        "config_horiz_mixing": "off",
        "config_xnutr": 0.25,
        "config_zd": 8_000.0,
        "config_epssm_minimum": 0.1,
        "config_epssm_maximum": 0.6,
        "config_epssm_transition_bottom_z": 3_000.0,
        "config_epssm_transition_top_z": 20_000.0,
    }
    values.update(updates)
    config = V841DryDycoreConfig(**values)
    config.validate()
    return config


def _f32_record(record: Any, names: Sequence[str]) -> Any:
    values: dict[str, Any] = {}
    for name in names:
        value = np.asarray(getattr(record, name))
        if value.ndim == 0:
            if value.dtype.kind in "iu":
                values[name] = int(value)
            elif value.dtype.kind == "b":
                values[name] = bool(value)
            else:
                values[name] = np.float32(value.item())
        else:
            values[name] = np.ascontiguousarray(value, dtype=np.float32)
    return replace(
        record,
        **values,
    )


def prepare_synthetic_case(
    grid: Path = DEFAULT_GRID,
    static: Path = DEFAULT_STATIC,
    *,
    config: "V841DryDycoreConfig | None" = None,  # noqa: F821
    require_nonconstant_eps: bool = True,
    require_active_dss: bool = True,
) -> SimpleNamespace:
    """Build the non-vacuous implementation deck without touching CUDA."""

    from woof.hex.driver import (
        DryDycoreDriver,
        TerrainMetrics,
        make_synthetic_x1_case,
    )
    from woof.hex.dynamics_v841 import V841ReferenceWindProfiles
    from woof.hex.integration import RKSchedule
    from woof.hex.mesh import Mesh
    from woof.hex.transport import build_advection_coefficients

    selected = ruler_config() if config is None else config
    mesh = Mesh.from_netcdf(grid, static_path=static)
    case = make_synthetic_x1_case(
        mesh,
        n_vert_levels=3,
        perturbation_amplitude=0.02,
        wind_speed=9.0,
        n_scalars=1,
    )
    state = _f32_record(case.state, STATE_FIELDS)
    ncells = mesh.dimensions["nCells"]
    tracer = np.empty_like(state.scalars)
    tracer[0] = (
        np.float32(0.001)
        + np.linspace(np.float32(0.0), np.float32(0.0003), ncells, dtype=np.float32)[
            None, :
        ]
        + np.arange(3, dtype=np.float32)[:, None] * np.float32(0.00005)
    )
    rho_w = np.array(state.rho_w, copy=True, order="C")
    cell_phase = np.linspace(
        np.float32(-1.0), np.float32(1.0), ncells, dtype=np.float32
    )
    rho_w[1] = np.float32(0.012) * state.rho[0] * cell_phase
    rho_w[2] = np.float32(-0.008) * state.rho[1] * cell_phase[::-1]
    rho_w[0] = np.float32(0.0)
    rho_w[-1] = np.float32(0.0)
    state = replace(
        state,
        rho_w=np.ascontiguousarray(rho_w),
        scalars=np.ascontiguousarray(tracer),
    )

    vertical_names = tuple(case.vertical_grid.__slots__)
    vertical = _f32_record(case.vertical_grid, vertical_names)
    vertical_values = {
        name: np.array(getattr(vertical, name), copy=True, order="C")
        for name in ("dzu", "rdzu", "rdzwp", "rdzwm", "fzp", "fzm")
    }
    for name in ("dzu", "rdzu", "rdzwp", "rdzwm", "fzp", "fzm"):
        vertical_values[name][0] = np.float32(0.0)
    vertical = replace(vertical, **vertical_values)
    reference = _f32_record(case.reference, tuple(case.reference.__slots__))
    terrain = TerrainMetrics(
        zb_cell=np.zeros((4, ncells, mesh.dimensions["maxEdges"]), dtype=np.float32),
        zb3_cell=np.zeros((4, ncells, mesh.dimensions["maxEdges"]), dtype=np.float32),
    )
    profiles = V841ReferenceWindProfiles(
        u_init=np.ascontiguousarray(np.array([7.0, -3.0, 11.0], np.float32)),
        v_init=np.ascontiguousarray(np.array([-2.0, 5.0, -4.0], np.float32)),
    )
    coefficients = build_advection_coefficients(
        mesh,
        config_scalar_adv_order=3,
        n_vert_levels=3,
        source_order_v841=True,
    )
    cpu = DryDycoreDriver(
        mesh,
        vertical,
        reference,
        selected,
        terrain_metrics=terrain,
        advection_coefficients=coefficients,
        reference_wind_profiles=profiles,
    )
    saved = cpu._rebuild_saved_diagnostics(state)
    schedule = RKSchedule.from_mpas(
        selected.config_dt,
        order=selected.config_time_integration_order,
        acoustic_substeps=selected.config_number_of_sub_steps,
        dynamics_splits=selected.config_dynamics_split_steps,
    )
    deck_audit = {
        "n_vert_levels": int(state.rho.shape[0]),
        "max_n_edges_on_cell": int(np.max(np.asarray(mesh.arrays["nEdgesOnCell"]))),
        "max_n_edges_on_edge": int(np.max(np.asarray(mesh.arrays["nEdgesOnEdge"]))),
        "max_advection_cells_per_edge": int(
            np.max(np.asarray(coefficients.n_adv_cells_for_edge))
        ),
        "dynamics_split_steps": selected.config_dynamics_split_steps,
        "configured_acoustic_substeps": selected.config_number_of_sub_steps,
        "stage_acoustic_steps": [stage.acoustic_steps for stage in schedule.stages],
        "actual_acoustic_calls_per_split": sum(
            stage.acoustic_steps for stage in schedule.stages
        ),
        "whole_step_policy_acoustic_calls_per_split": 3 * 2,
        "whole_step_policy_is_conservative_max_per_stage": True,
    }
    if deck_audit != {
        "n_vert_levels": 3,
        "max_n_edges_on_cell": 6,
        "max_n_edges_on_edge": 10,
        "max_advection_cells_per_edge": 10,
        "dynamics_split_steps": 3,
        "configured_acoustic_substeps": 2,
        "stage_acoustic_steps": [1, 1, 2],
        "actual_acoustic_calls_per_split": 4,
        "whole_step_policy_acoustic_calls_per_split": 6,
        "whole_step_policy_is_conservative_max_per_stage": True,
    }:
        raise RuntimeError(
            f"numeric ruler deck topology/schedule drifted: {deck_audit}"
        )
    nonvacuity = {
        "tracer_nonzero_count": int(np.count_nonzero(state.scalars)),
        "tracer_min": float(np.min(state.scalars)),
        "tracer_max": float(np.max(state.scalars)),
        "rho_w_interior_nonzero_count": int(np.count_nonzero(state.rho_w[1:-1])),
        "u_init": profiles.u_init.tolist(),
        "v_init": profiles.v_init.tolist(),
        "dss_nonzero_count": int(np.count_nonzero(cpu.damping_coefficients)),
        "dss_maximum": float(np.max(cpu.damping_coefficients)),
        "etp": np.asarray(cpu.acoustic_offcentering.etp).tolist(),
        "ewp": np.asarray(cpu.acoustic_offcentering.ewp).tolist(),
    }
    if (
        nonvacuity["tracer_nonzero_count"] == 0
        or nonvacuity["rho_w_interior_nonzero_count"] == 0
        or (require_active_dss and nonvacuity["dss_nonzero_count"] == 0)
        or (require_nonconstant_eps and np.ptp(cpu.acoustic_offcentering.etp) == 0.0)
        or np.count_nonzero(profiles.u_init) != 3
        or np.count_nonzero(profiles.v_init) != 3
    ):
        raise RuntimeError("synthetic v8.4.1 ruler deck became vacuous")
    return SimpleNamespace(
        mesh=mesh,
        state=state,
        vertical=vertical,
        reference=reference,
        terrain=terrain,
        profiles=profiles,
        coefficients=coefficients,
        config=selected,
        cpu=cpu,
        saved=saved,
        nonvacuity=nonvacuity,
        deck_audit=deck_audit,
    )


def prepare_shared_input_vector_deck(case: SimpleNamespace) -> SimpleNamespace:
    """Build one CPU-only nonzero-reference deck with a declared cancellation lane."""

    from woof.hex.dynamics_v841 import vector_invariant_momentum_tendency_v841

    mesh = case.mesh
    nlev = int(case.profiles.u_init.size)
    nedges = int(mesh.dimensions["nEdges"])
    ncells = int(mesh.dimensions["nCells"])
    mesh_array_sha256 = {
        name: array_sha256(_mesh_authority_array(mesh, name))
        for name in DIRECT_VECTOR_MESH_ARRAY_SHA256
    }
    mesh_sha256 = canonical_json_sha256(mesh_array_sha256)
    if (
        mesh_array_sha256 != DIRECT_VECTOR_MESH_ARRAY_SHA256
        or mesh_sha256 != DIRECT_VECTOR_MESH_SHA256
    ):
        raise RuntimeError(
            "fixed direct-vector cancellation lane requires the pinned x1.2562 mesh"
        )
    edge_index = np.arange(nlev * nedges, dtype=np.float32).reshape(nlev, nedges)
    cell_index = np.arange(nlev * ncells, dtype=np.float32).reshape(nlev, ncells)
    normal_velocity = np.ascontiguousarray(
        np.sin(edge_index * np.float32(0.001), dtype=np.float32)
        * np.float32(0.2)
    )
    rho_edge = np.ascontiguousarray(
        np.float32(0.9)
        + np.cos(edge_index * np.float32(0.0007), dtype=np.float32)
        * np.float32(0.1)
    )
    pv_edge = np.ascontiguousarray(
        np.float32(1.0e-4)
        + np.sin(edge_index * np.float32(0.0003), dtype=np.float32)
        * np.float32(2.0e-5)
    )
    kinetic_energy = np.ascontiguousarray(
        np.float32(0.01)
        + np.sin(cell_index * np.float32(0.002), dtype=np.float32)
        * np.float32(0.005)
    )
    horizontal_divergence = np.zeros((nlev, ncells), dtype=np.float32)
    common = {
        "normal_velocity": normal_velocity,
        "rho_edge": rho_edge,
        "pv_edge": pv_edge,
        "kinetic_energy": kinetic_energy,
        "reference_wind": case.profiles,
    }
    uncancelled = vector_invariant_momentum_tendency_v841(
        mesh,
        horizontal_divergence=horizontal_divergence,
        **common,
    )
    level, edge = DIRECT_VECTOR_CANCELLATION_INDEX
    if (
        level >= nlev
        or edge >= nedges
        or np.abs(normal_velocity[level, edge]) < np.float32(0.05)
    ):
        raise RuntimeError("fixed x1.2562 vector cancellation lane is unavailable")
    cells = _mesh_authority_array(mesh, "cellsOnEdge")
    cell0, cell1 = (int(value) for value in cells[edge])
    if (cell0, cell1) != DIRECT_VECTOR_CANCELLATION_CELLS:
        raise RuntimeError(
            "fixed x1.2562 vector cancellation lane topology changed"
        )
    cancellation_value = np.float32(
        uncancelled[level, edge] / normal_velocity[level, edge]
    )
    horizontal_divergence[level, cell0] = cancellation_value
    horizontal_divergence[level, cell1] = cancellation_value
    horizontal_divergence = np.ascontiguousarray(horizontal_divergence)
    authority = vector_invariant_momentum_tendency_v841(
        mesh,
        horizontal_divergence=horizontal_divergence,
        **common,
    )
    activity = vector_momentum_source_activity_v841(
        mesh,
        horizontal_divergence=horizontal_divergence,
        **common,
    )
    carrier = upward_float32_activity(activity)
    if np.any(carrier.astype(np.float64) < np.abs(authority).astype(np.float64)):
        raise RuntimeError("vector source activity is smaller than CPU authority output")
    carrier_spacing = (
        np.nextafter(carrier, np.float32(np.inf), dtype=np.float32) - carrier
    )
    authority_magnitude = np.abs(authority[level, edge])
    authority_spacing = (
        np.nextafter(
            authority_magnitude,
            np.float32(np.inf),
            dtype=np.float32,
        )
        - authority_magnitude
    )
    if authority_magnitude != np.float32(0.0):
        raise RuntimeError("declared vector cancellation lane is not exact CPU zero")
    if carrier_spacing[level, edge] <= np.float32(128.0) * authority_spacing:
        raise RuntimeError("declared vector cancellation lane is not ill-conditioned")
    inputs = {
        **common,
        "horizontal_divergence": horizontal_divergence,
    }
    return SimpleNamespace(
        **inputs,
        authority=np.ascontiguousarray(authority),
        activity=activity,
        carrier=carrier,
        cancellation_index=(int(level), int(edge)),
        cancellation_selection=(
            "fixed predeclared x1.2562 level/edge lane; the reached-angle probe "
            "must independently prove standard-sinf/cosf bit drift on its neighbors"
        ),
        cancellation_cells=(cell0, cell1),
        cancellation_value=cancellation_value,
        mesh_array_sha256=mesh_array_sha256,
        mesh_sha256=mesh_sha256,
        input_sha256={
            name: array_sha256(value)
            for name, value in inputs.items()
            if isinstance(value, np.ndarray)
        }
        | {
            "u_init": array_sha256(case.profiles.u_init),
            "v_init": array_sha256(case.profiles.v_init),
        },
    )


def execute_shared_input_vector_cuda(
    deck: SimpleNamespace,
    driver: Any,
    kernel_cache: Any,
    cp: Any,
) -> np.ndarray:
    """Execute the production CUDA vector kernel on the deck's exact host inputs."""

    from woof.hex.cuda_dynamics_v841 import vector_momentum_tendency_cuda_v841

    context = driver.v841_context
    if context is None:
        raise RuntimeError("v8.4.1 CUDA context is absent")

    def resident(value: np.ndarray) -> Any:
        return cp.ascontiguousarray(cp.asarray(value))

    result = vector_momentum_tendency_cuda_v841(
        driver.atmosphere.mesh,
        context,
        normal_velocity=resident(deck.normal_velocity),
        rho_edge=resident(deck.rho_edge),
        pv_edge=resident(deck.pv_edge),
        kinetic_energy=resident(deck.kinetic_energy),
        horizontal_divergence=resident(deck.horizontal_divergence),
        kernel_cache=kernel_cache,
    )
    return np.ascontiguousarray(cp.asnumpy(result))


def _single_tu_compile_evidence(
    manifest: Mapping[str, Any],
    *,
    module_key: str,
    source: str,
    resolved_kernels: Sequence[str],
    require_only_module: bool,
    expected_platform_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate exact NVRTC source/image evidence for one reached translation unit."""

    from woof.hex.cuda_backend.compile_contract import (
        validate_compile_platform_fingerprint,
    )

    if manifest.get("schema") != "mpas-port.cuda-compile-manifest/v1":
        raise RuntimeError("direct compile manifest schema changed")
    platform = manifest.get("compile_platform")
    if not isinstance(platform, Mapping) or set(platform) != {"fingerprint", "sha256"}:
        raise RuntimeError("direct compile manifest lacks exact platform binding")
    fingerprint = platform.get("fingerprint")
    if not isinstance(fingerprint, Mapping):
        raise RuntimeError("direct compile platform fingerprint is absent")
    validated = validate_compile_platform_fingerprint(fingerprint)
    if dict(fingerprint) != validated:
        raise RuntimeError("direct compile platform fingerprint is not canonical")
    fingerprint_sha = canonical_json_sha256(fingerprint)
    if (
        platform.get("sha256") != fingerprint_sha
        or fingerprint.get("device_compute_capability") != "120"
    ):
        raise RuntimeError("direct compile platform digest or SM is false")
    if expected_platform_sha256 is not None and fingerprint_sha != expected_platform_sha256:
        raise RuntimeError("direct compile platform differs from baseline production")
    modules = manifest.get("modules")
    if not isinstance(modules, Mapping) or module_key not in modules:
        raise RuntimeError("direct compile manifest omitted its reached translation unit")
    if require_only_module and set(modules) != {module_key}:
        raise RuntimeError("direct compile must contain exactly one translation unit")
    module = modules[module_key]
    if not isinstance(module, Mapping):
        raise RuntimeError("direct compile module declaration is invalid")
    source_sha = hashlib.sha256(source.encode("utf-8")).hexdigest()
    expected_cache = hashlib.sha256()
    expected_cache.update(b"sm_120")
    expected_cache.update(b"\0compile-platform\0")
    expected_cache.update(fingerprint_sha.encode("ascii"))
    expected_cache.update(source.encode("utf-8"))
    for option in ("--std=c++17", "--fmad=false"):
        expected_cache.update(b"\0")
        expected_cache.update(option.encode("utf-8"))
    if (
        module.get("source_sha256") != source_sha
        or module.get("module_cache_key") != expected_cache.hexdigest()
        or module.get("compile_platform_fingerprint_sha256") != fingerprint_sha
        or module.get("requested_options") != ["--std=c++17", "--fmad=false"]
        or module.get("resolved_kernels") != list(resolved_kernels)
    ):
        raise RuntimeError("direct compile source/options/cache/kernel relation is false")
    effective = module.get("effective_compile")
    method = (
        "wrapped cupy.cuda.compiler._compile_using_nvrtc_no_warning "
        "at the NVRTC entry point"
    )
    observations = effective.get("observations") if isinstance(effective, Mapping) else None
    if (
        not isinstance(effective, Mapping)
        or set(effective) != {"status", "method", "observations"}
        or effective.get("status") != "resolved"
        or effective.get("method") != method
        or not isinstance(observations, list)
        or len(observations) != 1
    ):
        raise RuntimeError("direct compile lacks one exact NVRTC-entry observation")
    observation = observations[0]
    expected_observation_keys = {
        "source_sha256",
        "effective_flags",
        "include_path_count",
        "include_paths_omitted",
        "compiled_image",
    }
    omitted = (
        "-I entries describe this machine's toolkit layout and are counted but not "
        "copied into the arithmetic contract"
    )
    if (
        not isinstance(observation, Mapping)
        or set(observation) != expected_observation_keys
        or observation.get("source_sha256") != source_sha
        or observation.get("effective_flags")
        != ["--std=c++17", "--fmad=false", "-ftz=true"]
        or not isinstance(observation.get("include_path_count"), int)
        or observation.get("include_path_count", -1) < 0
        or observation.get("include_paths_omitted") != omitted
    ):
        raise RuntimeError("direct compile has false NVRTC-entry evidence")
    image = observation.get("compiled_image")
    if (
        not isinstance(image, Mapping)
        or set(image) != {"status", "kind", "sha256"}
        or image.get("status") != "resolved"
        or image.get("kind") != "cubin"
        or re.fullmatch(r"[0-9a-f]{64}", str(image.get("sha256", ""))) is None
    ):
        raise RuntimeError("direct compile has no exact resolved final CUBIN image")
    return {
        "module_key": module_key,
        "source_sha256": source_sha,
        "module_cache_key": expected_cache.hexdigest(),
        "requested_options": ["--std=c++17", "--fmad=false"],
        "effective_flags": ["--std=c++17", "--fmad=false", "-ftz=true"],
        "resolved_kernels": list(resolved_kernels),
        "compiled_image": dict(image),
        "compile_platform": {
            "fingerprint": dict(fingerprint),
            "sha256": fingerprint_sha,
        },
        "compile_manifest_sha256": canonical_json_sha256(manifest),
    }


def _reached_angle_trig_component(
    name: str,
    authority: Any,
    candidate: Any,
) -> tuple[dict[str, Any], np.ndarray]:
    """Validate one exact float32 reached-angle trigonometric observation."""

    expected = np.asarray(authority)
    actual = np.asarray(candidate)
    if (
        expected.dtype != np.dtype(np.float32)
        or actual.dtype != np.dtype(np.float32)
        or expected.shape != actual.shape
    ):
        raise TypeError(
            f"reached-angle {name} authority/candidate must be same-shape float32"
        )
    if not np.all(np.isfinite(expected)) or not np.all(np.isfinite(actual)):
        raise FloatingPointError(
            f"reached-angle standard {name} contains non-finite values"
        )
    magnitude = np.abs(expected)
    spacing = (
        np.nextafter(magnitude, np.float32(np.inf), dtype=np.float32) - magnitude
    ).astype(np.float64)
    absolute = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    normalized = np.divide(
        absolute,
        spacing,
        out=np.full_like(absolute, np.inf),
        where=spacing > 0.0,
    )
    failures = normalized > CUDA_STANDARD_TRIG_MAX_ULP
    if np.any(failures):
        raise RuntimeError(
            f"reached-angle standard {name} exceeded "
            f"{CUDA_STANDARD_TRIG_MAX_ULP} ULP"
        )
    different = expected.view(np.uint32) != actual.view(np.uint32)
    return (
        {
            "authority_sha256": array_sha256(expected),
            "candidate_sha256": array_sha256(actual),
            "different_elements": int(np.count_nonzero(different)),
            "different_index_sha256": array_sha256(
                np.ascontiguousarray(np.flatnonzero(different), dtype=np.int64)
            ),
            "max_absolute_error": float(np.max(absolute, initial=0.0)),
            "max_error_in_authority_ulps": float(
                np.max(normalized, initial=0.0)
            ),
            "failing_elements": 0,
        },
        different,
    )


def verify_reached_angle_standard_trig(
    case: SimpleNamespace,
    *,
    capability: Any,
    cache_root: Path,
    cp: Any,
    expected_platform_sha256: str,
    required_angle_indices: Sequence[int],
) -> dict[str, Any]:
    """Measure the predeclared two-ULP sinf/cosf contract on reached angles."""

    from woof.hex.cuda_backend import KernelCache

    edges = _mesh_authority_array(case.mesh, "edgesOnEdge")
    counts = _mesh_authority_array(case.mesh, "nEdgesOnEdge")
    angle = _mesh_authority_array(case.mesh, "angleEdge").astype(
        np.float32, copy=False
    )
    reached = np.zeros(angle.shape, dtype=np.bool_)
    for edge, count_raw in enumerate(counts):
        count = int(count_raw)
        neighbors = np.asarray(edges[edge, :count], dtype=np.int64)
        if np.any(neighbors < 0) or np.any(neighbors >= angle.size):
            raise ValueError("reached-angle probe found invalid edgesOnEdge")
        reached[neighbors] = True
    reached_indices = np.ascontiguousarray(np.flatnonzero(reached), dtype=np.int64)
    reached_angle = np.ascontiguousarray(angle[reached_indices], dtype=np.float32)
    if reached_angle.size == 0:
        raise RuntimeError("vector momentum reaches no reference angles")
    required = np.ascontiguousarray(
        np.asarray(tuple(required_angle_indices), dtype=np.int64)
    )
    if required.ndim != 1 or required.size == 0:
        raise ValueError("trig probe requires fixed cancellation-lane angle indices")
    if (
        np.any(required < 0)
        or np.any(required >= angle.size)
        or not np.all(reached[required])
    ):
        raise ValueError("required cancellation-lane angle is not reached")
    compact_by_edge = np.full(angle.shape, -1, dtype=np.int64)
    compact_by_edge[reached_indices] = np.arange(reached_indices.size, dtype=np.int64)
    required_compact = compact_by_edge[required]
    sine_authority = np.sin(reached_angle, dtype=np.float32)
    cosine_authority = np.cos(reached_angle, dtype=np.float32)

    cache = KernelCache(
        capability=capability,
        cache_dir=cache_root / "direct-vector-trig-probe",
    )
    kernel = cache.raw_kernel(
        "reached_angle_sincos_f32",
        _DIRECT_TRIG_PROBE_SOURCE,
        module_key=_DIRECT_TRIG_PROBE_MODULE,
    )
    angle_device = cp.ascontiguousarray(cp.asarray(reached_angle))
    sine_device = cp.empty_like(angle_device)
    cosine_device = cp.empty_like(angle_device)
    threads = 256
    kernel(
        ((int(reached_angle.size) + threads - 1) // threads,),
        (threads,),
        (
            np.int32(reached_angle.size),
            angle_device,
            sine_device,
            cosine_device,
        ),
    )
    cp.cuda.runtime.deviceSynchronize()
    sine_candidate = np.ascontiguousarray(cp.asnumpy(sine_device))
    cosine_candidate = np.ascontiguousarray(cp.asnumpy(cosine_device))

    sine, sine_different = _reached_angle_trig_component(
        "sinf", sine_authority, sine_candidate
    )
    cosine, cosine_different = _reached_angle_trig_component(
        "cosf", cosine_authority, cosine_candidate
    )
    if sine["different_elements"] + cosine["different_elements"] == 0:
        raise RuntimeError("reached-angle probe did not exercise host/device trig")
    required_different = np.logical_or(
        sine_different[required_compact], cosine_different[required_compact]
    )
    if not np.any(required_different):
        raise RuntimeError(
            "fixed cancellation lane did not reach a host/device trig bit drift"
        )
    manifest = cache.compile_manifest()
    compile_evidence = _single_tu_compile_evidence(
        manifest,
        module_key=_DIRECT_TRIG_PROBE_MODULE,
        source=_DIRECT_TRIG_PROBE_SOURCE,
        resolved_kernels=("reached_angle_sincos_f32",),
        require_only_module=True,
        expected_platform_sha256=expected_platform_sha256,
    )
    return {
        "contract": "case-specific reached-angle standard sinf/cosf verification",
        "predeclared_max_ulp": CUDA_STANDARD_TRIG_MAX_ULP,
        "accuracy_source": CUDA_STANDARD_TRIG_ACCURACY_SOURCE,
        "accuracy_source_limitation": (
            "NVIDIA states that the two-ULP figure is testing-derived rather than "
            "a universal guarantee; this receipt therefore claims only the exact "
            "angles, image, flags, and platform measured here"
        ),
        "authority_nonclaim": (
            "NumPy float32 sin/cos are the selected CPU implementation authority; "
            "this probe does not claim a universal correctly-rounded host libm"
        ),
        "reached_angle_count": int(reached_angle.size),
        "reached_angle_sha256": array_sha256(reached_angle),
        "reached_angle_index_sha256": array_sha256(reached_indices),
        "required_angle_indices": required.tolist(),
        "required_angle_sha256": array_sha256(
            np.ascontiguousarray(angle[required], dtype=np.float32)
        ),
        "required_trig_different_elements": int(
            np.count_nonzero(required_different)
        ),
        "sinf": sine,
        "cosf": cosine,
        "diagnostic_d2h_bytes": int(
            sine_candidate.nbytes + cosine_candidate.nbytes
        ),
        "compile_evidence": compile_evidence,
        "compile_manifest": manifest,
    }


def certify_shared_input_vector_momentum(
    case: SimpleNamespace,
    *,
    capability: Any,
    baseline_cache: Any,
    baseline_compile_manifest: Mapping[str, Any],
    cache_root: Path,
    cp: Any,
) -> dict[str, Any]:
    """Certify only the common-input v8.4.1 vector-momentum implementation."""

    from woof.hex.cuda_backend import KernelCache
    from woof.hex.cuda_ftz import v841_reached_translation_units

    deck = prepare_shared_input_vector_deck(case)
    production_source, production_kernels = v841_reached_translation_units()[
        "hexcore.cuda_dynamics_v841"
    ]
    baseline_compile_evidence = _single_tu_compile_evidence(
        baseline_compile_manifest,
        module_key="hexcore.cuda_dynamics_v841",
        source=production_source,
        resolved_kernels=production_kernels,
        require_only_module=False,
    )
    platform_sha = baseline_compile_evidence["compile_platform"]["sha256"]
    level, edge = deck.cancellation_index
    edges = _mesh_authority_array(case.mesh, "edgesOnEdge")
    counts = _mesh_authority_array(case.mesh, "nEdgesOnEdge")
    lane_neighbors = np.ascontiguousarray(
        edges[edge, : int(counts[edge])], dtype=np.int64
    )
    trig_contract = verify_reached_angle_standard_trig(
        case,
        capability=capability,
        cache_root=cache_root,
        cp=cp,
        expected_platform_sha256=platform_sha,
        required_angle_indices=lane_neighbors,
    )
    driver = _new_cuda_driver(case, baseline_cache)
    baseline = execute_shared_input_vector_cuda(deck, driver, baseline_cache, cp)
    comparison = compare_array_at_activity_scale(
        "direct_vector_momentum.shared_input",
        deck.authority,
        baseline,
        policy_id="direct_vector_momentum",
        activity=deck.activity,
        activity_name="vector_momentum_source_activity_v841",
    )
    if not comparison.passed:
        raise RuntimeError("shared-input vector momentum exceeded its source activity")
    if comparison.different_elements == 0:
        raise RuntimeError("shared-input vector deck did not exercise CUDA trigonometry")
    if np.any(deck.carrier.astype(np.float64) < np.abs(deck.authority).astype(np.float64)):
        raise RuntimeError("vector source activity is smaller than CPU authority output")
    output_local = compare_array(
        "direct_vector_momentum.output_local_control",
        deck.authority,
        baseline,
        policy_id="direct_vector_momentum",
    )
    if output_local.passed:
        raise RuntimeError("shared-input vector deck lost its cancellation adversary")
    cancellation_output_local = compare_array(
        "direct_vector_momentum.output_local_cancellation_lane",
        deck.authority[level : level + 1, edge : edge + 1],
        baseline[level : level + 1, edge : edge + 1],
        policy_id="direct_vector_momentum",
    )
    if cancellation_output_local.passed:
        raise RuntimeError("fixed vector cancellation lane did not fail output-local ruler")
    carrier_spacing = (
        np.nextafter(deck.carrier, np.float32(np.inf), dtype=np.float32)
        - deck.carrier
    ).astype(np.float64)
    baseline_absolute = np.abs(
        baseline.astype(np.float64) - deck.authority.astype(np.float64)
    )
    measured_activity_ulps = np.divide(
        baseline_absolute,
        carrier_spacing,
        out=np.full_like(baseline_absolute, np.inf),
        where=carrier_spacing > 0.0,
    )
    maximum_measured_activity_ulps = float(
        np.max(measured_activity_ulps, initial=0.0)
    )
    if maximum_measured_activity_ulps > _DIRECT_SITES["vector_momentum"]:
        raise RuntimeError("measured vector activity ratio exceeds its fixed ceiling")

    mutants = []
    for index, mutation in enumerate(DIRECT_VECTOR_MUTATIONS):
        base = KernelCache(
            capability=capability,
            cache_dir=cache_root / f"direct-vector-mutant-{index:02d}-{mutation.name}",
        )
        proxy = MutatingKernelCache(base, mutation)
        candidate = execute_shared_input_vector_cuda(deck, driver, proxy, cp)
        mutant_comparison = compare_array_at_activity_scale(
            f"direct_vector_momentum.{mutation.name}",
            deck.authority,
            candidate,
            policy_id="direct_vector_momentum",
            activity=deck.activity,
            activity_name="vector_momentum_source_activity_v841",
        )
        if mutant_comparison.passed:
            raise RuntimeError(
                f"direct vector mutation {mutation.name} did not exceed its activity"
            )
        changed_comparison = compare_array_at_activity_scale(
            f"direct_vector_momentum.{mutation.name}.changed_from_baseline",
            baseline,
            candidate,
            policy_id="direct_vector_momentum",
            activity=deck.activity,
            activity_name="vector_momentum_source_activity_v841",
        )
        if changed_comparison.passed:
            raise RuntimeError(
                f"direct vector mutation {mutation.name} did not go red against "
                "the production baseline under the frozen carrier"
            )
        manifest = base.compile_manifest()
        mutated_source = proxy.mutated_source_text
        if mutated_source is None:
            raise RuntimeError(f"direct vector mutation {mutation.name} has no source")
        mutation_evidence = _validate_direct_vector_mutation_provenance(
            production_source, mutation, proxy
        )
        mutants.append(
            {
                "mutation": mutation_evidence,
                "comparison": _as_receipts([mutant_comparison])[0],
                "changed_from_baseline": _as_receipts([changed_comparison])[0],
                "candidate_sha256": array_sha256(candidate),
                "diagnostic_d2h_bytes": int(candidate.nbytes),
                "compile_evidence": _single_tu_compile_evidence(
                    manifest,
                    module_key="hexcore.cuda_dynamics_v841",
                    source=mutated_source,
                    resolved_kernels=("vector_momentum_v841_f32",),
                    require_only_module=True,
                    expected_platform_sha256=platform_sha,
                ),
                "compile_manifest": manifest,
            }
        )

    vector_d2h_bytes = int(baseline.nbytes)
    mutant_d2h_bytes = int(sum(row["diagnostic_d2h_bytes"] for row in mutants))
    trig_d2h_bytes = int(trig_contract["diagnostic_d2h_bytes"])
    return {
        "claim": (
            "certification-grade shared-input float32 vector-momentum comparison; "
            "no independently evolved trajectory claim"
        ),
        "policy_id": "direct_vector_momentum",
        "operation_rounding_sites": _DIRECT_SITES["vector_momentum"],
        "measured_max_error_in_activity_ulps": maximum_measured_activity_ulps,
        "source_activity_formula": (
            "abs(rho_edge)*(sum(abs(w*u))*0.5*(abs(pv_e)+abs(pv_n)) + "
            "sum(abs(w*f))*(abs(u_init)*(abs(cos32)+2*ulp(cos32)) + "
            "abs(v_init)*(abs(sin32)+2*ulp(sin32))) + "
            "(abs(ke_c1)+abs(ke_c0))*abs(invDc32)) + "
            "abs(u)*0.5*(abs(div_c0)+abs(div_c1))"
        ),
        "activity_accumulation_dtype": "float64",
        "activity_carrier_rounding": "per-element ceiling to float32",
        "activity_dominates_cpu_authority_per_element": True,
        "reached_angle_trig_contract": trig_contract,
        "diagnostic_d2h": {
            "production_vector_bytes": vector_d2h_bytes,
            "mutant_vector_bytes": mutant_d2h_bytes,
            "trig_probe_bytes": trig_d2h_bytes,
            "total_bytes": vector_d2h_bytes + mutant_d2h_bytes + trig_d2h_bytes,
            "production_step_safety_receipt_bytes": 4,
            "scope": (
                "certification-only vector and trig downloads occur after the "
                "production step and are outside its four-byte safety D2H receipt"
            ),
        },
        "mesh_array_sha256": deck.mesh_array_sha256,
        "mesh_sha256": deck.mesh_sha256,
        "inputs_sha256": deck.input_sha256,
        "authority_sha256": array_sha256(deck.authority),
        "activity_float64_sha256": array_sha256(deck.activity),
        "activity_upward_float32_sha256": array_sha256(deck.carrier),
        "activity_minimum": float(np.min(deck.activity)),
        "activity_maximum": float(np.max(deck.activity)),
        "cancellation_lane": {
            "index": [level, edge],
            "cells": list(deck.cancellation_cells),
            "authority": float(deck.authority[level, edge]),
            "candidate": float(baseline[level, edge]),
            "activity": float(deck.activity[level, edge]),
            "output_local_policy_failed": True,
            "output_local_failing_elements": output_local.failing_elements,
            "fixed_lane_output_local": _as_receipts(
                [cancellation_output_local]
            )[0],
        },
        "baseline": _as_receipts([comparison])[0],
        "baseline_compile_evidence": baseline_compile_evidence,
        "mutants": mutants,
    }


def compiled_fixture_nonclaims(path: Path = COMPILED_REPORT) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    claims = payload["vacuities_and_nonclaims"]
    required_true = (
        "qv_identically_zero",
        "u_init_identically_zero",
        "v_init_identically_zero",
        "dss_identically_zero",
    )
    required_false = (
        "nonzero_tracer_compiled_certified",
        "nonzero_reference_coriolis_compiled_certified",
        "nonzero_dss_compiled_certified",
    )
    if any(claims.get(name) is not True for name in required_true):
        raise ValueError("compiled fixture vacuity declaration changed")
    if any(claims.get(name) is not False for name in required_false):
        raise ValueError("compiled fixture nonclaim declaration changed")
    return {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "sha256": sha256_file(path),
        "vacuities_and_nonclaims": claims,
    }


def _copy_host_result(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True, order="C")
    if hasattr(value, "__slots__"):
        return {
            name: _copy_host_result(getattr(value, name))
            for name in value.__slots__
            if isinstance(getattr(value, name), np.ndarray)
        }
    return value


def _copy_device_result(value: Any, cp: Any) -> Any:
    if isinstance(value, cp.ndarray):
        return cp.array(value, copy=True)
    if hasattr(value, "__slots__"):
        return {
            name: _copy_device_result(getattr(value, name), cp)
            for name in value.__slots__
            if isinstance(getattr(value, name), cp.ndarray)
        }
    return value


@contextlib.contextmanager
def capture_cpu_operators() -> Iterator[dict[str, list[Any]]]:
    import woof.hex.driver as module

    captures: dict[str, list[Any]] = {
        name: []
        for name in (
            "theta_tendency",
            "w_tendency",
            "vector_momentum",
            "acoustic",
            "scalar_transport",
            "split_flux",
            "rw_endpoints",
            "eps_profile",
            "reference_profile",
            "dss_profile",
        )
    }
    originals = {
        "_rho_theta_tendency": module._rho_theta_tendency,
        "_vertical_momentum_transport": module._vertical_momentum_transport,
        "vector_invariant_momentum_tendency_v841": module.vector_invariant_momentum_tendency_v841,
        "advance_acoustic_step_v841": module.advance_acoustic_step_v841,
        "advance_scalar_transport": module.advance_scalar_transport,
        "finish_split_flux": module.finish_split_flux,
        "enforce_recovered_rw_endpoints_v841": module.enforce_recovered_rw_endpoints_v841,
    }

    def theta(*args: Any, **kwargs: Any) -> Any:
        result = originals["_rho_theta_tendency"](*args, **kwargs)
        captures["theta_tendency"].append(_copy_host_result(result))
        return result

    def w(*args: Any, **kwargs: Any) -> Any:
        result = originals["_vertical_momentum_transport"](*args, **kwargs)
        captures["w_tendency"].append(_copy_host_result(result))
        return result

    def acoustic(*args: Any, **kwargs: Any) -> Any:
        result = originals["advance_acoustic_step_v841"](*args, **kwargs)
        captures["acoustic"].append(_copy_host_result(result))
        return result

    def vector(*args: Any, **kwargs: Any) -> Any:
        result = originals["vector_invariant_momentum_tendency_v841"](*args, **kwargs)
        captures["vector_momentum"].append(_copy_host_result(result))
        return result

    def scalar(*args: Any, **kwargs: Any) -> Any:
        result = originals["advance_scalar_transport"](*args, **kwargs)
        captures["scalar_transport"].append(_copy_host_result(result))
        return result

    def split(*args: Any, **kwargs: Any) -> Any:
        result = originals["finish_split_flux"](*args, **kwargs)
        captures["split_flux"].append(_copy_host_result(result))
        return result

    def endpoints(*args: Any, **kwargs: Any) -> Any:
        result = originals["enforce_recovered_rw_endpoints_v841"](*args, **kwargs)
        captures["rw_endpoints"].append(
            np.ascontiguousarray(np.asarray(args[0])[[0, -1]])
        )
        return result

    module._rho_theta_tendency = theta
    module._vertical_momentum_transport = w
    module.vector_invariant_momentum_tendency_v841 = vector
    module.advance_acoustic_step_v841 = acoustic
    module.advance_scalar_transport = scalar
    module.finish_split_flux = split
    module.enforce_recovered_rw_endpoints_v841 = endpoints
    try:
        yield captures
    finally:
        for name, original in originals.items():
            setattr(module, name, original)


@contextlib.contextmanager
def capture_cuda_operators(cp: Any) -> Iterator[dict[str, list[Any]]]:
    import woof.hex.cuda_driver as module

    captures: dict[str, list[Any]] = {
        name: []
        for name in (
            "theta_tendency",
            "w_tendency",
            "vector_momentum",
            "acoustic",
            "scalar_transport",
            "split_flux",
            "rw_endpoints",
            "eps_profile",
            "reference_profile",
            "dss_profile",
        )
    }
    cls = module.CudaDryDycoreDriver
    originals = {
        "_theta_tendency": cls._theta_tendency,
        "_w_tendency": cls._w_tendency,
        "vector_momentum_tendency_cuda_v841": module.vector_momentum_tendency_cuda_v841,
        "advance_acoustic_step_cuda_v841": module.advance_acoustic_step_cuda_v841,
        "advance_scalar_transport_cuda_v841": module.advance_scalar_transport_cuda_v841,
        "finish_split_flux_cuda_v841": module.finish_split_flux_cuda_v841,
        "enforce_rw_endpoints_cuda_v841": module.enforce_rw_endpoints_cuda_v841,
    }

    def theta(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = originals["_theta_tendency"](self, *args, **kwargs)
        captures["theta_tendency"].append(_copy_device_result(result, cp))
        return result

    def w(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = originals["_w_tendency"](self, *args, **kwargs)
        captures["w_tendency"].append(_copy_device_result(result, cp))
        return result

    def acoustic(*args: Any, **kwargs: Any) -> Any:
        result = originals["advance_acoustic_step_cuda_v841"](*args, **kwargs)
        captures["acoustic"].append(_copy_device_result(result, cp))
        return result

    def vector(*args: Any, **kwargs: Any) -> Any:
        result = originals["vector_momentum_tendency_cuda_v841"](*args, **kwargs)
        captures["vector_momentum"].append(_copy_device_result(result, cp))
        return result

    def scalar(*args: Any, **kwargs: Any) -> Any:
        result = originals["advance_scalar_transport_cuda_v841"](*args, **kwargs)
        captures["scalar_transport"].append(_copy_device_result(result, cp))
        return result

    def split(*args: Any, **kwargs: Any) -> Any:
        result = originals["finish_split_flux_cuda_v841"](*args, **kwargs)
        captures["split_flux"].append(_copy_device_result(result, cp))
        return result

    def endpoints(*args: Any, **kwargs: Any) -> Any:
        result = originals["enforce_rw_endpoints_cuda_v841"](*args, **kwargs)
        captures["rw_endpoints"].append(
            cp.stack(
                (
                    cp.array(args[0][0], copy=True),
                    cp.array(args[0][-1], copy=True),
                )
            )
        )
        return result

    cls._theta_tendency = theta
    cls._w_tendency = w
    module.vector_momentum_tendency_cuda_v841 = vector
    module.advance_acoustic_step_cuda_v841 = acoustic
    module.advance_scalar_transport_cuda_v841 = scalar
    module.finish_split_flux_cuda_v841 = split
    module.enforce_rw_endpoints_cuda_v841 = endpoints
    try:
        yield captures
    finally:
        cls._theta_tendency = originals["_theta_tendency"]
        cls._w_tendency = originals["_w_tendency"]
        for name in (
            "advance_acoustic_step_cuda_v841",
            "vector_momentum_tendency_cuda_v841",
            "advance_scalar_transport_cuda_v841",
            "finish_split_flux_cuda_v841",
            "enforce_rw_endpoints_cuda_v841",
        ):
            setattr(module, name, originals[name])


def download_cuda_captures(
    captures: Mapping[str, list[Any]], cp: Any
) -> dict[str, list[Any]]:
    def download(value: Any) -> Any:
        if isinstance(value, cp.ndarray):
            return np.ascontiguousarray(cp.asnumpy(value))
        if isinstance(value, dict):
            return {name: download(item) for name, item in value.items()}
        return value

    return {
        name: [download(item) for item in values] for name, values in captures.items()
    }


def flatten_captures(captures: Mapping[str, list[Any]]) -> dict[str, np.ndarray]:
    flattened: dict[str, np.ndarray] = {}
    for family, calls in captures.items():
        for index, value in enumerate(calls):
            prefix = f"{family}.{index}"
            if isinstance(value, dict):
                for name, item in value.items():
                    flattened[f"{prefix}.{name}"] = np.asarray(item)
            else:
                flattened[prefix] = np.asarray(value)
    return flattened


def _host_state(
    atmosphere: Any, cp: Any
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    state = {
        name: np.ascontiguousarray(cp.asnumpy(getattr(atmosphere.state, name)))
        for name in STATE_FIELDS
    }
    saved = {
        name: np.ascontiguousarray(cp.asnumpy(getattr(atmosphere.saved, name)))
        for name in SAVED_FIELDS
    }
    return state, saved


def _compare_mapping(
    authority: Mapping[str, np.ndarray],
    candidate: Mapping[str, np.ndarray],
    *,
    policy_id: str,
) -> list[ArrayComparison]:
    if set(authority) != set(candidate):
        raise ValueError(
            f"{policy_id}: candidate fields differ from authority: "
            f"{sorted(set(authority) ^ set(candidate))}"
        )
    return [
        compare_array(
            name,
            authority[name],
            candidate[name],
            policy_id=policy_id,
        )
        for name in sorted(authority)
    ]


def _edge_levels_to_cell_interfaces(
    mesh: Any,
    edge_values: Any,
    *,
    parent_name: str,
) -> np.ndarray:
    """Build a same-lane cell/interface carrier from incident edge magnitudes."""

    values = np.asarray(edge_values)
    edges_on_cell = np.asarray(mesh.edgesOnCell)
    counts = np.asarray(mesh.nEdgesOnCell)
    if values.dtype != np.dtype(np.float32) or values.ndim != 2:
        raise TypeError(f"{parent_name} must be a two-dimensional float32 field")
    if edges_on_cell.ndim != 2 or counts.shape != (edges_on_cell.shape[0],):
        raise ValueError("mesh cell-edge carrier topology is malformed")
    nlev, nedges = values.shape
    if np.any(edges_on_cell[:, : int(np.max(counts))] >= nedges):
        raise ValueError(f"{parent_name} carrier topology exceeds its edge extent")
    cell_levels = np.zeros((nlev, edges_on_cell.shape[0]), dtype=np.float32)
    magnitude = np.abs(values)
    for slot in range(edges_on_cell.shape[1]):
        valid = counts > slot
        if not np.any(valid):
            continue
        selected = magnitude[:, edges_on_cell[valid, slot]]
        cell_levels[:, valid] = np.maximum(cell_levels[:, valid], selected)
    interfaces = np.empty((nlev + 1, edges_on_cell.shape[0]), dtype=np.float32)
    interfaces[0] = cell_levels[0]
    interfaces[-1] = cell_levels[-1]
    if nlev > 1:
        interfaces[1:-1] = np.maximum(cell_levels[:-1], cell_levels[1:])
    return np.ascontiguousarray(interfaces)


def _momentum_interface_parent(case: SimpleNamespace) -> np.ndarray:
    return _edge_levels_to_cell_interfaces(
        case.mesh,
        _edge_momentum_parent(case),
        parent_name="CPU initial/final state.rho_u carrier",
    )


def _edge_momentum_parent(case: SimpleNamespace) -> np.ndarray:
    initial = np.asarray(case.state.rho_u)
    authority_state = getattr(case, "authority_state", None)
    if authority_state is None:
        return np.ascontiguousarray(np.abs(initial))
    final = np.asarray(authority_state.rho_u)
    if final.shape != initial.shape or final.dtype != initial.dtype:
        raise ValueError("CPU authority rho_u carrier changed shape or RKIND")
    return np.ascontiguousarray(np.maximum(np.abs(initial), np.abs(final)))


def _velocity_interface_parent(case: SimpleNamespace) -> np.ndarray:
    authority_saved = getattr(case, "authority_saved", None)
    edge_velocity = (
        case.saved.normal_velocity
        if authority_saved is None
        else authority_saved.normal_velocity
    )
    return _edge_levels_to_cell_interfaces(
        case.mesh,
        edge_velocity,
        parent_name="CPU saved.normal_velocity",
    )


def _compare_whole_state(
    authority: Mapping[str, np.ndarray],
    candidate: Mapping[str, np.ndarray],
    case: SimpleNamespace,
) -> list[ArrayComparison]:
    if set(authority) != set(candidate):
        raise ValueError("one-step state inventory differs between CPU and CUDA")
    comparisons: list[ArrayComparison] = []
    for name in sorted(authority):
        if name == "rho_w":
            comparisons.append(
                compare_array_at_parent_scale(
                    name,
                    authority[name],
                    candidate[name],
                    policy_id="trajectory_dynamics_checkpoint",
                    parent=_momentum_interface_parent(case),
                    parent_name=(
                        "incident-edge initial state.rho_u mapped to cell interfaces"
                    ),
                )
            )
        else:
            comparisons.append(
                compare_array(
                    name,
                    authority[name],
                    candidate[name],
                    policy_id="one_step_state",
                )
            )
    return comparisons


def _compare_whole_saved(
    authority: Mapping[str, np.ndarray],
    candidate: Mapping[str, np.ndarray],
    case: SimpleNamespace,
) -> list[ArrayComparison]:
    if set(authority) != set(candidate):
        raise ValueError("one-step saved inventory differs between CPU and CUDA")
    parents: dict[str, tuple[Any, str]] = {
        "density_perturbation": (
            case.state.rho,
            "initial state.rho (density perturbation parent)",
        ),
        "rho_theta_perturbation": (
            case.state.rho_theta,
            "initial state.rho_theta (rtheta perturbation parent)",
        ),
        "pressure_perturbation": (
            case.reference.pressure_base,
            "reference.pressure_base (pressure perturbation parent)",
        ),
        "vertical_velocity": (
            _velocity_interface_parent(case),
            "incident-edge saved.normal_velocity mapped to cell interfaces",
        ),
    }
    comparisons: list[ArrayComparison] = []
    for name in sorted(authority):
        if name in parents:
            parent, parent_name = parents[name]
            comparisons.append(
                compare_array_at_parent_scale(
                    name,
                    authority[name],
                    candidate[name],
                    policy_id="trajectory_dynamics_checkpoint",
                    parent=parent,
                    parent_name=parent_name,
                )
            )
        else:
            comparisons.append(
                compare_array(
                    name,
                    authority[name],
                    candidate[name],
                    policy_id="one_step_saved",
                )
            )
    return comparisons


def _operator_parent_policy(
    name: str, case: SimpleNamespace
) -> tuple[str, Any, str] | None:
    family = name.split(".", 1)[0]
    if family == "acoustic":
        leaf = name.rsplit(".", 1)[-1]
        if leaf == "rho_pp":
            return (
                "trajectory_acoustic_checkpoint",
                case.state.rho,
                "initial state.rho (acoustic density parent)",
            )
        if leaf in ("rtheta_pp", "rtheta_pp_old"):
            return (
                "trajectory_acoustic_checkpoint",
                case.state.rho_theta,
                "initial state.rho_theta (acoustic thermodynamic parent)",
            )
        if leaf in ("ru_p", "ru_avg"):
            return (
                "trajectory_acoustic_checkpoint",
                _edge_momentum_parent(case),
                "per-edge CPU initial/final state.rho_u carrier",
            )
        if leaf in ("rw_p", "ww_avg"):
            return (
                "trajectory_acoustic_checkpoint",
                _momentum_interface_parent(case),
                "incident-edge initial state.rho_u mapped to cell interfaces",
            )
    if family == "theta_tendency":
        return (
            "trajectory_dynamics_checkpoint",
            case.state.rho_theta,
            "initial state.rho_theta (theta tendency parent)",
        )
    if family == "vector_momentum":
        return (
            "trajectory_dynamics_checkpoint",
            _edge_momentum_parent(case),
            "per-edge CPU initial/final state.rho_u carrier",
        )
    if family == "w_tendency":
        return (
            "trajectory_dynamics_checkpoint",
            _momentum_interface_parent(case),
            "incident-edge initial state.rho_u mapped to cell interfaces",
        )
    if family == "split_flux":
        parent = (
            _edge_momentum_parent(case)
            if name == "split_flux.0"
            else _momentum_interface_parent(case)
        )
        parent_name = (
            "per-edge CPU initial/final state.rho_u carrier"
            if name == "split_flux.0"
            else "incident-edge initial state.rho_u mapped to cell interfaces"
        )
        return (
            "trajectory_split_flux_checkpoint",
            parent,
            parent_name,
        )
    return None


def _compare_operator_capture(
    name: str,
    authority: np.ndarray,
    candidate: np.ndarray,
    case: SimpleNamespace,
) -> ArrayComparison:
    parent_policy = _operator_parent_policy(name, case)
    if parent_policy is not None:
        policy_id, parent, parent_name = parent_policy
        return compare_array_at_parent_scale(
            name,
            authority,
            candidate,
            policy_id=policy_id,
            parent=parent,
            parent_name=parent_name,
        )
    family = name.split(".", 1)[0]
    if family in ("eps_profile", "reference_profile", "dss_profile"):
        policy_id = "exact_uploaded_profile"
    elif family == "rw_endpoints":
        policy_id = "exact_rw_endpoints"
    elif family == "scalar_transport":
        policy_id = "direct_scalar_transport"
    else:
        raise ValueError(f"captured operator family has no policy: {family}")
    return compare_array(name, authority, candidate, policy_id=policy_id)


def _compare_operator_captures(
    authority: Mapping[str, np.ndarray],
    candidate: Mapping[str, np.ndarray],
    case: SimpleNamespace,
) -> list[ArrayComparison]:
    """Compare live trajectory checkpoints and exact copy-only anchors."""

    if set(authority) != set(candidate):
        raise ValueError(
            "operator captures: candidate fields differ from authority: "
            f"{sorted(set(authority) ^ set(candidate))}"
        )

    return [
        _compare_operator_capture(
            name,
            authority[name],
            candidate[name],
            case,
        )
        for name in sorted(authority)
    ]


def _new_cuda_driver(case: SimpleNamespace, cache: Any) -> "CudaDryDycoreDriver":  # noqa: F821
    from woof.hex.cuda_driver import CudaDryDycoreDriver

    return CudaDryDycoreDriver.from_host(
        case.mesh,
        case.state.copy(),
        case.vertical,
        case.reference,
        case.config,
        saved_diagnostics=case.saved,
        terrain_metrics=case.terrain,
        advection_coefficients=case.coefficients,
        kernel_cache=cache,
        reference_wind_profiles=case.profiles,
    )


def _validate_mutant_compile_manifest(
    manifest: Mapping[str, Any],
    mutation_cache: MutatingKernelCache,
    baseline_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one controlled source delta over the exact live eight-TU graph."""

    from woof.hex.cuda_ftz import (
        V841_REACHED_TRANSLATION_UNITS,
        canonical_sha256,
        validate_v841_compile_manifest_relation,
        v841_compiled_translation_units,
        v841_reached_translation_units,
    )

    mutation = mutation_cache.mutation
    baseline_relation = validate_v841_compile_manifest_relation(baseline_manifest)
    baseline_modules = baseline_manifest.get("modules")
    if (
        not isinstance(baseline_modules, Mapping)
        or baseline_relation["reached_kernel_count"] != 46
        or baseline_relation["compiled_kernel_count"] != 95
    ):
        raise RuntimeError("mutant has no exact validated baseline compile relation")
    mutated_source = mutation_cache.mutated_source_text
    if mutated_source is None:
        raise RuntimeError(f"mutation {mutation.name} has no compiled source")
    platform = manifest.get("compile_platform")
    modules = manifest.get("modules")
    if (
        manifest.get("schema") != "mpas-port.cuda-compile-manifest/v1"
        or not isinstance(platform, Mapping)
        or not isinstance(modules, Mapping)
        or tuple(sorted(modules)) != V841_REACHED_TRANSLATION_UNITS
    ):
        raise RuntimeError("mutant did not compile the exact eight reached TUs")
    fingerprint = platform.get("fingerprint")
    fingerprint_sha = platform.get("sha256")
    if (
        not isinstance(fingerprint, Mapping)
        or fingerprint.get("device_compute_capability") != "120"
        or fingerprint_sha != canonical_json_sha256(fingerprint)
        or baseline_manifest.get("compile_platform") != platform
    ):
        raise RuntimeError("mutant compile platform is not canonical sm_120")
    reached = v841_reached_translation_units()
    compiled = v841_compiled_translation_units()
    live_target_source = reached[mutation.module_key][0]
    exact_mutated_source = live_target_source.replace(
        mutation.before, mutation.after, 1
    )
    if (
        live_target_source.count(mutation.before) != 1
        or mutated_source != exact_mutated_source
        or mutation_cache.applied_calls < 1
        or mutation_cache.original_source_sha256
        != hashlib.sha256(live_target_source.encode()).hexdigest()
        or mutation_cache.mutated_source_sha256
        != hashlib.sha256(mutated_source.encode()).hexdigest()
    ):
        raise RuntimeError("mutant cache is not the declared exact single-token delta")
    relation: dict[str, Any] = {}
    declaration_fields = {
        "source_sha256",
        "module_cache_key",
        "compile_platform_fingerprint_sha256",
        "requested_options",
        "resolved_kernels",
        "effective_compile",
    }
    observation_fields = {
        "source_sha256",
        "effective_flags",
        "include_path_count",
        "include_paths_omitted",
        "compiled_image",
    }
    for module_key in V841_REACHED_TRANSLATION_UNITS:
        declaration = modules[module_key]
        if (
            not isinstance(declaration, Mapping)
            or set(declaration) != declaration_fields
        ):
            raise RuntimeError(f"mutant declaration fields changed for {module_key}")
        source = (
            mutated_source
            if module_key == mutation.module_key
            else reached[module_key][0]
        )
        source_sha = hashlib.sha256(source.encode("utf-8")).hexdigest()
        expected_cache = hashlib.sha256()
        expected_cache.update(b"sm_120")
        expected_cache.update(b"\0compile-platform\0")
        expected_cache.update(str(fingerprint_sha).encode("ascii"))
        expected_cache.update(source.encode("utf-8"))
        for option in ("--std=c++17", "--fmad=false"):
            expected_cache.update(b"\0")
            expected_cache.update(option.encode("utf-8"))
        if (
            declaration["source_sha256"] != source_sha
            or declaration["module_cache_key"] != expected_cache.hexdigest()
            or declaration["compile_platform_fingerprint_sha256"] != fingerprint_sha
            or declaration["requested_options"] != ["--std=c++17", "--fmad=false"]
            or declaration["resolved_kernels"] != list(reached[module_key][1])
        ):
            raise RuntimeError(f"mutant compile relation is false for {module_key}")
        effective = declaration["effective_compile"]
        if module_key == mutation.module_key:
            if (
                not isinstance(effective, Mapping)
                or set(effective) != {"status", "method", "observations"}
                or effective.get("status") != "resolved"
                or effective.get("method")
                != "wrapped cupy.cuda.compiler._compile_using_nvrtc_no_warning at the NVRTC entry point"
                or not isinstance(effective.get("observations"), list)
                or len(effective["observations"]) != 1
            ):
                raise RuntimeError(
                    f"mutant NVRTC evidence is invalid for {module_key}"
                )
            observation = effective["observations"][0]
            evidence_source = "mutant NVRTC compile"
        else:
            if (
                isinstance(effective, Mapping)
                and effective.get("status") == "resolved"
                and isinstance(effective.get("observations"), list)
                and len(effective["observations"]) == 1
            ):
                observation = effective["observations"][0]
                evidence_source = "fresh unchanged-TU NVRTC compile"
            elif (
                isinstance(effective, Mapping)
                and set(effective) == {"status", "reason"}
                and effective.get("status") == "unavailable"
                and effective.get("reason")
                == "NVRTC did not fire while the RawModule function was resolved; "
                "the compiled image may have been a CuPy disk-cache hit; effective "
                "terminal -ftz=true is not claimed without a real NVRTC-entry capture"
            ):
                baseline_effective = baseline_modules[module_key]["effective_compile"]
                if (
                    not isinstance(baseline_effective, Mapping)
                    or baseline_effective.get("status") != "resolved"
                    or not isinstance(baseline_effective.get("observations"), list)
                    or len(baseline_effective["observations"]) != 1
                ):
                    raise RuntimeError(
                        f"baseline NVRTC evidence is unavailable for {module_key}"
                    )
                observation = baseline_effective["observations"][0]
                evidence_source = "validated same-process baseline compile"
            else:
                raise RuntimeError(
                    f"unchanged mutant TU evidence is neither compiled nor an exact "
                    f"cache reuse: {module_key}"
                )
        image = (
            observation.get("compiled_image")
            if isinstance(observation, Mapping)
            else None
        )
        if (
            not isinstance(observation, Mapping)
            or set(observation) != observation_fields
            or observation.get("source_sha256") != source_sha
            or observation.get("effective_flags")
            != ["--std=c++17", "--fmad=false", "-ftz=true"]
            or not isinstance(observation.get("include_path_count"), int)
            or observation.get("include_path_count", -1) < 0
            or observation.get("include_paths_omitted")
            != "-I entries describe this machine's toolkit layout and are counted but not copied into the arithmetic contract"
            or not isinstance(image, Mapping)
            or set(image) != {"status", "kind", "sha256"}
            or image.get("status") != "resolved"
            or image.get("kind") not in {"cubin", "ptx"}
            or not isinstance(image.get("sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", image["sha256"]) is None
        ):
            raise RuntimeError(f"mutant compile image is invalid for {module_key}")
        relation[module_key] = {
            "source_sha256": source_sha,
            "resolved_kernels": list(reached[module_key][1]),
            "compiled_kernel_surface": list(compiled[module_key][1]),
            "compiled_image": dict(image),
            "evidence_source": evidence_source,
        }
    if sum(len(row["resolved_kernels"]) for row in relation.values()) != 46:
        raise RuntimeError("mutant compile relation did not resolve exact 46 kernels")
    if sum(len(row["compiled_kernel_surface"]) for row in relation.values()) != 95:
        raise RuntimeError("mutant compile relation lost the 95-kernel surface")
    return {
        "source_release": "v8.4.1-mutant",
        "compile_manifest_sha256": canonical_sha256(manifest),
        "translation_units": relation,
        "reached_kernel_count": 46,
        "compiled_kernel_count": 95,
        "mutation": mutation.name,
        "mutated_module_key": mutation.module_key,
        "authority_claim": False,
    }


def _validate_reused_compile_manifest(
    manifest: Mapping[str, Any],
    baseline_compile_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind same-process cache hits to an already validated baseline image."""
    from woof.hex.cuda_ftz import (
        canonical_sha256,
        validate_v841_compile_manifest_relation,
    )

    baseline_relation = validate_v841_compile_manifest_relation(
        baseline_compile_manifest
    )
    if (
        manifest.get("schema") != baseline_compile_manifest.get("schema")
        or manifest.get("compile_platform")
        != baseline_compile_manifest.get("compile_platform")
    ):
        raise RuntimeError("reused compile manifest changed its schema or platform")
    modules = manifest.get("modules")
    baseline_modules = baseline_compile_manifest.get("modules")
    if (
        not isinstance(modules, Mapping)
        or not isinstance(baseline_modules, Mapping)
        or set(modules) != set(baseline_modules)
    ):
        raise RuntimeError("reused compile manifest changed its module inventory")

    unavailable = {
        "status": "unavailable",
        "reason": (
            "NVRTC did not fire while the RawModule function was resolved; "
            "the compiled image may have been a CuPy disk-cache hit; effective "
            "terminal -ftz=true is not claimed without a real NVRTC-entry capture"
        ),
    }
    translation_units: dict[str, Any] = {}
    for module_key in sorted(modules):
        declaration = modules[module_key]
        baseline_declaration = baseline_modules[module_key]
        if not isinstance(declaration, Mapping) or not isinstance(
            baseline_declaration, Mapping
        ):
            raise RuntimeError(f"invalid reused compile declaration for {module_key}")
        if {
            key: value
            for key, value in declaration.items()
            if key != "effective_compile"
        } != {
            key: value
            for key, value in baseline_declaration.items()
            if key != "effective_compile"
        }:
            raise RuntimeError(f"reused compile declaration changed for {module_key}")
        effective = declaration.get("effective_compile")
        baseline_effective = baseline_declaration.get("effective_compile")
        if effective == baseline_effective:
            evidence_source = "fresh identical NVRTC compile"
        elif effective == unavailable:
            evidence_source = "validated same-process baseline compile"
        else:
            raise RuntimeError(
                f"unchanged TU evidence is neither identical nor an exact cache reuse: "
                f"{module_key}"
            )
        row = dict(baseline_relation["translation_units"][module_key])
        row["evidence_source"] = evidence_source
        translation_units[module_key] = row

    return {
        **baseline_relation,
        "compile_manifest_sha256": canonical_sha256(manifest),
        "translation_units": translation_units,
        "evidence_reused_from_validated_baseline": True,
    }


def _require_receipt(
    receipt: Any,
    case: SimpleNamespace,
    cache: Any,
    *,
    mutation_cache: MutatingKernelCache | None = None,
    baseline_compile_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    from woof.hex.cuda_ftz import (
        canonical_sha256,
        validate_v841_compile_manifest_relation,
    )

    if receipt.source_release != "v8.4.1":
        raise RuntimeError("CUDA ruler did not execute the v8.4.1 source lane")
    if receipt.d2h.bytes != 4:
        raise RuntimeError(
            "production CUDA step performed more than the 4-byte safety D2H"
        )
    config = receipt.configuration
    required = {
        "config_scalar_advection": True,
        "config_monotonic": False,
        "config_positive_definite": False,
        "config_dynamics_split_steps": 3,
        "config_split_dynamics_transport": True,
    }
    if any(config.get(name) != value for name, value in required.items()):
        raise RuntimeError("CUDA receipt lost a required non-vacuous configuration")
    if float(config.get("config_apvm_upwinding", 0.0)) <= 0.0:
        raise RuntimeError("CUDA receipt lost positive APVM upwinding")
    if case.state.scalars.shape[0] < 1:
        raise RuntimeError("CUDA ruler requires at least one tracer")
    manifest = cache.compile_manifest()
    if receipt.compile_manifest != manifest:
        raise RuntimeError("CUDA receipt compile manifest differs from executed cache")
    manifest_sha = canonical_sha256(manifest)
    if receipt.compile_manifest_sha256 != manifest_sha:
        raise RuntimeError("CUDA receipt compile manifest SHA is false")
    if mutation_cache is None and baseline_compile_manifest is None:
        relation = validate_v841_compile_manifest_relation(manifest)
    elif mutation_cache is None:
        relation = _validate_reused_compile_manifest(
            manifest, baseline_compile_manifest
        )
    else:
        if baseline_compile_manifest is None:
            raise RuntimeError("mutant receipt lacks its validated baseline manifest")
        relation = _validate_mutant_compile_manifest(
            manifest,
            mutation_cache,
            baseline_compile_manifest,
        )
    if (
        relation["compile_manifest_sha256"] != manifest_sha
        or relation["reached_kernel_count"] != 46
        or relation["compiled_kernel_count"] != 95
        or len(relation["translation_units"]) != 8
    ):
        raise RuntimeError("CUDA actual-step compile relation is incomplete")
    return {
        "source_release": receipt.source_release,
        "d2h_bytes": receipt.d2h.bytes,
        "configuration_sha256": receipt.configuration_sha256,
        "compile_manifest_sha256": receipt.compile_manifest_sha256,
        "compile_relation": relation,
        "stage_acoustic_steps": list(receipt.stage_acoustic_steps),
        "dynamics_split_steps": receipt.dynamics_split_steps,
    }


def _run_cpu(case: SimpleNamespace) -> tuple[Any, dict[str, np.ndarray]]:
    with capture_cpu_operators() as captured:
        profile = case.cpu.acoustic_offcentering
        for name in ("etp", "etm", "ewp", "ewm"):
            captured["eps_profile"].append(
                np.array(getattr(profile, name), copy=True, order="C")
            )
        for name in ("u_init", "v_init"):
            captured["reference_profile"].append(
                np.array(getattr(case.profiles, name), copy=True, order="C")
            )
        captured["dss_profile"].append(
            np.array(case.cpu.damping_coefficients, copy=True, order="C")
        )
        result = case.cpu.step(case.state.copy(), saved_diagnostics=case.saved)
    return result, flatten_captures(captured)


def _run_cuda(
    case: SimpleNamespace,
    cache: Any,
    cp: Any,
    *,
    mutation_cache: MutatingKernelCache | None = None,
    baseline_compile_manifest: Mapping[str, Any] | None = None,
) -> tuple[Any, dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    driver = _new_cuda_driver(case, cache)
    with capture_cuda_operators(cp) as captured:
        context = driver.v841_context
        if context is None:
            raise RuntimeError("v8.4.1 CUDA context is absent")
        for name in ("etp", "etm", "ewp", "ewm"):
            captured["eps_profile"].append(cp.array(getattr(context, name), copy=True))
        for name in ("u_init", "v_init"):
            captured["reference_profile"].append(
                cp.array(getattr(context, name), copy=True)
            )
        captured["dss_profile"].append(
            cp.array(driver.atmosphere.vertical.dss, copy=True)
        )
        result = driver.step_device()
    receipt = _require_receipt(
        result.receipt,
        case,
        cache,
        mutation_cache=mutation_cache,
        baseline_compile_manifest=baseline_compile_manifest,
    )
    captures = flatten_captures(download_cuda_captures(captured, cp))
    state, saved = _host_state(result.atmosphere, cp)
    return receipt, state, saved, captures


def _as_receipts(values: Sequence[ArrayComparison]) -> list[dict[str, Any]]:
    return [{**asdict(item), "budget": asdict(item.budget)} for item in values]


def _mutant_red(
    mutation: SourceMutation,
    authority: Mapping[str, np.ndarray],
    candidate: Mapping[str, np.ndarray],
    case: SimpleNamespace,
) -> tuple[bool, list[ArrayComparison]]:
    selected_authority = {
        name: value
        for name, value in authority.items()
        if name.startswith(mutation.required_capture_prefix)
    }
    selected_candidate = {
        name: candidate[name] for name in selected_authority if name in candidate
    }
    if not selected_authority or set(selected_candidate) != set(selected_authority):
        raise RuntimeError(
            f"{mutation.name}: required direct capture family was not reached"
        )
    comparisons = [
        _compare_operator_capture(
            name,
            selected_authority[name],
            selected_candidate[name],
            case,
        )
        for name in sorted(selected_authority)
    ]
    return any(not item.passed for item in comparisons), comparisons


def run_numeric_ruler(
    *,
    grid: Path,
    static: Path,
    cache_root: Path,
    authority_before: Mapping[str, Any] | None = None,
    inputs_before: Mapping[str, Any] | None = None,
    engineering: bool = False,
) -> dict[str, Any]:
    from woof.hex.cuda_backend import KernelCache, require_cuda

    measured_authority = numeric_authority_snapshot()
    if authority_before is None:
        authority_before = measured_authority
    elif measured_authority != authority_before:
        raise RuntimeError("numeric authority changed before production imports")
    measured_inputs = input_file_snapshot(grid, static)
    if inputs_before is None:
        inputs_before = measured_inputs
    elif measured_inputs != inputs_before:
        raise RuntimeError("numeric grid/static bytes changed before case load")
    capability = require_cuda(
        min_compute=(12, 0),
        cache_dir=cache_root / "baseline",
    )
    import cupy as cp

    case = prepare_synthetic_case(grid, static)
    cpu_result, cpu_captures = _run_cpu(case)
    case.authority_state = cpu_result.state
    case.authority_saved = cpu_result.saved_diagnostics
    baseline_cache = KernelCache(
        capability=capability, cache_dir=cache_root / "baseline"
    )
    baseline_receipt, gpu_state, gpu_saved, gpu_captures = _run_cuda(
        case, baseline_cache, cp
    )
    baseline_compile_manifest = baseline_cache.compile_manifest()
    direct_vector_certification = certify_shared_input_vector_momentum(
        case,
        capability=capability,
        baseline_cache=baseline_cache,
        baseline_compile_manifest=baseline_compile_manifest,
        cache_root=cache_root,
        cp=cp,
    )
    cpu_state = {
        name: np.asarray(getattr(cpu_result.state, name)) for name in STATE_FIELDS
    }
    cpu_saved = {
        name: np.asarray(getattr(cpu_result.saved_diagnostics, name))
        for name in SAVED_FIELDS
    }
    state_comparisons = _compare_whole_state(cpu_state, gpu_state, case)
    saved_comparisons = _compare_whole_saved(cpu_saved, gpu_saved, case)
    operator_comparisons = _compare_operator_captures(
        cpu_captures,
        gpu_captures,
        case,
    )
    baseline_comparisons = (
        *state_comparisons,
        *saved_comparisons,
        *operator_comparisons,
    )
    failed_baseline = [item for item in baseline_comparisons if not item.passed]
    if failed_baseline and not engineering:
        details = "; ".join(
            f"{item.name}: max_abs={item.max_absolute_error:.9g}, "
            f"budget={item.budget.max_absolute_error:.9g}, "
            f"failing={item.failing_elements}/{np.prod(item.shape, dtype=np.int64)}"
            for item in failed_baseline
        )
        raise RuntimeError(
            "baseline CPU-v8.4.1 to CUDA numerical ruler failed: " + details
        )
    if float(cpu_result.state.time_seconds) != float(case.config.config_dt):
        raise RuntimeError("CPU authority ended at the wrong model time")

    mutants: list[dict[str, Any]] = []
    for index, mutation in enumerate(SOURCE_MUTATIONS):
        mutation_dir = cache_root / f"mutant-{index:02d}-{mutation.name}"
        base = KernelCache(capability=capability, cache_dir=mutation_dir)
        proxy = MutatingKernelCache(base, mutation)
        receipt, _state, _saved, captures = _run_cuda(
            case,
            proxy,
            cp,
            mutation_cache=proxy,
            baseline_compile_manifest=baseline_compile_manifest,
        )
        candidate = captures
        authority = cpu_captures
        red, comparisons = _mutant_red(mutation, authority, candidate, case)
        baseline_family = {
            name: gpu_captures[name]
            for name in authority
            if name.startswith(mutation.required_capture_prefix)
        }
        changed_gpu_baseline = any(
            array_sha256(candidate[name]) != array_sha256(baseline_family[name])
            for name in baseline_family
        )
        if not changed_gpu_baseline:
            raise RuntimeError(f"mutation {mutation.name} did not alter its GPU family")
        if not red and not engineering:
            raise RuntimeError(f"mutation {mutation.name} did not go red")
        baseline_family_comparisons = [
            _compare_operator_capture(
                name,
                authority[name],
                baseline_family[name],
                case,
            )
            for name in sorted(baseline_family)
        ]
        baseline_family_passed = all(
            item.passed for item in baseline_family_comparisons
        )
        mutants.append(
            {
                "mutation": proxy.evidence(),
                "receipt": receipt,
                "compile_manifest": base.compile_manifest(),
                "required_capture_prefix": mutation.required_capture_prefix,
                "changed_gpu_baseline": True,
                "baseline_family_strict_passed": baseline_family_passed,
                "went_red_under_strict_ruler": red if baseline_family_passed else None,
                "baseline_family_comparisons": _as_receipts(
                    baseline_family_comparisons
                ),
                "comparisons": _as_receipts(comparisons),
            }
        )

    # Two authority-side input adversaries close the nonconstant EPS profile
    # and active DSS inputs.  Each modified input must still mirror CPU on GPU,
    # while producing a different direct acoustic answer than the baseline.
    input_mutants: list[dict[str, Any]] = []
    for name, config in (
        (
            "constant-eps-profile",
            ruler_config(
                config_epssm_minimum=0.3,
                config_epssm_maximum=0.3,
                config_epssm_transition_bottom_z=3_000.0,
                config_epssm_transition_top_z=20_000.0,
            ),
        ),
        ("zero-dss", ruler_config(config_xnutr=0.0)),
    ):
        variant = prepare_synthetic_case(
            grid,
            static,
            config=config,
            require_nonconstant_eps=name != "constant-eps-profile",
            require_active_dss=name != "zero-dss",
        )
        variant_cpu, variant_cpu_captures = _run_cpu(variant)
        variant.authority_state = variant_cpu.state
        variant.authority_saved = variant_cpu.saved_diagnostics
        variant_cache = KernelCache(
            capability=capability, cache_dir=cache_root / f"input-{name}"
        )
        receipt, variant_state, variant_saved, variant_gpu_captures = _run_cuda(
            variant,
            variant_cache,
            cp,
            baseline_compile_manifest=baseline_compile_manifest,
        )
        comparisons = (
            _compare_whole_state(
                {
                    field: np.asarray(getattr(variant_cpu.state, field))
                    for field in STATE_FIELDS
                },
                variant_state,
                variant,
            )
            + _compare_whole_saved(
                {
                    field: np.asarray(getattr(variant_cpu.saved_diagnostics, field))
                    for field in SAVED_FIELDS
                },
                variant_saved,
                variant,
            )
            + _compare_operator_captures(
                variant_cpu_captures,
                variant_gpu_captures,
                variant,
            )
        )
        mirror_passed = all(item.passed for item in comparisons)
        if not mirror_passed and not engineering:
            raise RuntimeError(f"input mutant {name} failed its CPU-CUDA mirror")
        baseline_acoustic = {
            key: value
            for key, value in cpu_captures.items()
            if key.startswith("acoustic.")
        }
        variant_acoustic = {
            key: value
            for key, value in variant_cpu_captures.items()
            if key.startswith("acoustic.")
        }
        difference = _compare_mapping(
            baseline_acoustic,
            variant_acoustic,
            policy_id="direct_acoustic",
        )
        if not any(not item.passed for item in difference):
            raise RuntimeError(
                f"input adversary {name} did not change acoustic answers"
            )
        input_mutants.append(
            {
                "name": name,
                "cpu_cuda_strict_passed": mirror_passed,
                "changed_baseline_acoustic_answer": True,
                "receipt": receipt,
                "compile_manifest": variant_cache.compile_manifest(),
                "comparisons": _as_receipts(comparisons),
                "baseline_difference": _as_receipts(difference),
                "nonvacuity": variant.nonvacuity,
                "deck_audit": variant.deck_audit,
                "prepared_case": prepared_case_fingerprint(variant),
            }
        )

    authority_after = numeric_authority_snapshot()
    if authority_after != authority_before:
        raise RuntimeError(
            "numeric authority source/test bytes changed during execution"
        )
    if input_file_snapshot(grid, static) != inputs_before:
        raise RuntimeError("numeric grid/static bytes changed during execution")
    return {
        "schema": "mpas-port.cuda-v841-numeric-ruler.v2",
        "claim": (
            "work-only engineering synthetic float32 x1.2562 CPU-v8.4.1 to CUDA "
            "one-step mirror; live trajectory carriers are provisional and are not "
            "a certification-grade source-term forward-error envelope; adversarial "
            "mutation coverage is "
            "limited to the eight explicitly inventoried acoustic-RHS, acoustic-solve, "
            "theta-transport, w-transport, scalar-transport, split-reduction, "
            "rw-endpoint, and reference-wind families plus EPS/DSS input adversaries; "
            "this is not native or compiled MPAS authority and makes no claim for "
            "unmutated recovery, horizontal, pressure/APVM, mass-divergence, or "
            "inverse-metric kernels; the separately identified shared-input direct "
            "vector-momentum result is certification-grade only for that operator"
        ),
        "certified_native_authority": False,
        "promotion_status": (
            "not eligible: source-linked conditioning envelopes for the remaining "
            "operators and an isolated pre-import trust capsule are still required"
        ),
        "engineering_mode": engineering,
        "conditioning_envelope_status": (
            "certification-grade only for the separately reported shared-input direct "
            "vector-momentum operator; live trajectory carriers remain provisional "
            "engineering localization and are not numerical certification"
        ),
        "baseline_strict_passed": not failed_baseline,
        "baseline_strict_failures": _as_receipts(failed_baseline),
        "engineering_run_completed": True,
        "error_policies": {
            name: asdict(policy) for name, policy in sorted(ERROR_POLICIES.items())
        },
        "policy_source_audit": {
            "deck": case.deck_audit,
            "direct_operation_sites": dict(_DIRECT_SITES),
            "whole_step_operation_sites": _WHOLE_STEP_SITES,
            "whole_step_formula": ERROR_POLICIES["one_step_state"].derivation,
            "static_audit": POLICY_SOURCE_AUDIT,
        },
        "mutation_coverage": {
            "covered_source_families": [
                mutation.family for mutation in SOURCE_MUTATIONS
            ],
            "covered_input_families": ["eps_profile", "dss"],
            "certified_shared_input_direct_mutations": [
                mutation.family for mutation in DIRECT_VECTOR_MUTATIONS
            ],
            "explicit_nonclaims": [
                "recovery kernels",
                "horizontal diagnostic kernels",
                "pressure-gradient and APVM kernels",
                "mass-divergence kernels",
                "inverse-metric kernels",
            ],
        },
        "platform": {
            name: value
            for name, value in capability.as_dict().items()
            if name != "cache_directory"
        },
        "inputs": {
            "grid": dict(inputs_before["grid"]),
            "static": dict(inputs_before["static"]),
            "tool": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "compiled_fixture": compiled_fixture_nonclaims(),
            "numeric_authority": authority_before,
            "prepared_case": prepared_case_fingerprint(case),
        },
        "configuration": asdict(case.config),
        "nonvacuity": case.nonvacuity,
        "deck_audit": case.deck_audit,
        "baseline_receipt": baseline_receipt,
        "baseline_compile_manifest": baseline_compile_manifest,
        "comparisons": {
            "state": _as_receipts(state_comparisons),
            "saved_diagnostics": _as_receipts(saved_comparisons),
            "live_trajectory_checkpoints": _as_receipts(operator_comparisons),
            "certified_shared_input_direct": {
                "vector_momentum": direct_vector_certification
            },
        },
        "source_mutants": mutants,
        "input_mutants": input_mutants,
    }


def _require_absent_directory(path: Path, name: str) -> None:
    if path.exists():
        raise FileExistsError(f"{name} must be absent: {path}")
    if not path.parent.is_dir():
        raise FileNotFoundError(f"{name} parent does not exist: {path.parent}")


def _reject_symlink_ancestry(path: Path, name: str) -> None:
    selected = path.expanduser().absolute()
    for parent in (selected, *selected.parents):
        is_junction = getattr(parent, "is_junction", lambda: False)
        if parent.exists() and (parent.is_symlink() or is_junction()):
            raise ValueError(f"{name} ancestry contains a symlink: {parent}")


def validate_destination_paths(
    cache_root: Path,
    output: Path,
    *,
    protected_inputs: Sequence[Path] = (),
) -> tuple[Path, Path]:
    """Close fresh destination scope before imports or CUDA probing."""

    _reject_symlink_ancestry(cache_root, "cache root")
    _reject_symlink_ancestry(output, "output directory")
    cache = cache_root.expanduser().resolve()
    receipt = output.expanduser().resolve()
    _require_absent_directory(cache, "cache root")
    _require_absent_directory(receipt, "output directory")
    if cache == receipt or cache in receipt.parents or receipt in cache.parents:
        raise ValueError("cache root and output directory must not overlap")
    protected = (
        ROOT / "src",
        ROOT / "tests",
        ROOT / "tools",
        COMPILED_REPORT,
        *protected_inputs,
    )
    for destination_name, destination in (
        ("cache root", cache),
        ("output directory", receipt),
    ):
        for raw in protected:
            authority = raw.resolve()
            if destination == authority or authority in destination.parents:
                raise ValueError(
                    f"{destination_name} overlaps protected authority path {authority}"
                )
            if destination in authority.parents:
                raise ValueError(
                    f"{destination_name} contains protected authority path {authority}"
                )
    return cache, receipt


def _write_exclusive_json(path: Path, payload: Any) -> None:
    encoded = (
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, default=DEFAULT_GRID)
    parser.add_argument("--static", type=Path, default=DEFAULT_STATIC)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--engineering",
        action="store_true",
        help=(
            "complete the work-only localization deck while recording, but not "
            "waiving, strict conditioning-envelope failures"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    grid = args.grid.resolve(strict=True)
    static = args.static.resolve(strict=True)
    cache_root, output = validate_destination_paths(
        args.cache_root,
        args.output,
        protected_inputs=(grid, static),
    )
    authority_before = numeric_authority_snapshot()
    inputs_before = input_file_snapshot(grid, static)
    cache_root.mkdir()
    output.mkdir()
    payload = run_numeric_ruler(
        grid=grid,
        static=static,
        cache_root=cache_root,
        authority_before=authority_before,
        inputs_before=inputs_before,
        engineering=args.engineering,
    )
    if numeric_authority_snapshot() != payload["inputs"]["numeric_authority"]:
        raise RuntimeError("numeric authority changed before receipt publication")
    if input_file_snapshot(grid, static) != inputs_before:
        raise RuntimeError("numeric grid/static changed before receipt publication")
    payload["receipt_sha256"] = canonical_json_sha256(payload)
    receipt = output / "cuda-v841-numeric-ruler.json"
    _write_exclusive_json(receipt, payload)
    if numeric_authority_snapshot() != payload["inputs"]["numeric_authority"]:
        raise RuntimeError("numeric authority changed during receipt publication")
    if input_file_snapshot(grid, static) != inputs_before:
        raise RuntimeError("numeric grid/static changed during receipt publication")
    print(
        json.dumps(
            {
                "receipt": str(receipt),
                "receipt_file_sha256": sha256_file(receipt),
                "receipt_payload_sha256": payload["receipt_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
