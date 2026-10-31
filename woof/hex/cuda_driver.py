"""Device-resident dry MPAS-A RK3 orchestration.

This module composes the CUDA backend containers, horizontal C-grid kernels,
split-explicit acoustic kernels, and scalar FCT kernels in the frozen order at
``mpas_atm_time_integration.F:638-1224``.  The admitted paths are the
closed/global no-mix ruler and the frozen original-JW horizontal
Smagorinsky/divergence/upper-damping branch.

Host fields cross PCIe only in :meth:`CudaDryDycoreDriver.from_host`; a full
step remains resident until :meth:`CudaDryDycoreDriver.step` materializes the
result.  Kernels use float32 logical ``(level, entity)`` storage and compile
without fast math or fused multiply-add contraction.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, fields, replace
import hashlib
import os
import re
import time
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

from . import host_parallel
from .host_parallel import all_finite, active_indices_in_range
from .cuda_fp32 import CUDA_FTZ_HELPERS
from .native_powf import CUDA_NATIVE_POWF

from .cuda_acoustic import (
    CudaAcousticForcing,
    CudaAcousticState,
    advance_acoustic_step_cuda,
    compute_vertical_implicit_coefficients_cuda,
    convert_w_tendency_to_omega_cuda,
)
from . import cuda_acoustic_v841 as _acoustic_v841_module
from .cuda_acoustic_v841 import (
    advance_acoustic_step_cuda_v841,
    compute_vertical_implicit_coefficients_cuda_v841,
    short_lifetimes_enabled,
)
from .cuda_backend import (
    canonical_sha256,
    DeviceAtmosphere,
    DevicePrognosticState,
    DeviceSavedDiagnostics,
    KernelCache,
    TransferStats,
    recover_state,
    require_cuda,
)
from .cuda_horizontal import CudaHorizontal
from .cuda_horizontal_v841 import CudaHorizontalV841
from .cuda_dynamics_v841 import (
    accumulate_finite_array_cuda_v841,
    accumulate_split_flux_cuda_v841,
    enforce_rw_endpoints_cuda_v841,
    finish_split_flux_cuda_v841,
    theta_finish_cuda_v841,
    validate_recovered_state_cuda_v841,
    vector_momentum_tendency_cuda_v841,
    w_finish_cuda_v841,
)
from .cuda_transport import (
    CudaAdvectionCoefficients,
    advance_scalar_transport_cuda,
)
from .cuda_transport_v841 import advance_scalar_transport_cuda_v841
from .driver import (
    DryDycoreConfig,
    DryDycoreDriver,
    DrySavedDiagnostics,
    SPLIT_FLUX_REDUCTION,
    _frozen_vertical_damping,
    _mixing_config,
)
from .damping_v841 import build_v841_vertical_velocity_damping
from .arwen_les import model_row as arwen_model_row
from .species_row import (
    species_row_for_names,
    species_row_for_scheme as species_row_for_selector,
)
from .dynamics_v841 import V841ReferenceWindProfiles
from .errors import ConfigurationRefusal
from .integration import RKSchedule
from .offcentering_v841 import build_v841_acoustic_offcentering
from .prep_cache import cached
from .state import PrognosticState
from .transport import build_advection_coefficients
from .cuda_v841 import CudaV841Context

if TYPE_CHECKING:
    from .cuda_physics_v841 import CudaPhaseOneExecutionProvenanceV841


CUDA_WHOLE_STEP_EVIDENCE = "frozen-jw-nomix-cuda-step-linked"
CUDA_ORIGINAL_JW_BRANCH_EVIDENCE = "frozen-jw-original-mixed-cuda-step-linked"
CUDA_IMPLEMENTED_UNLINKED_EVIDENCE = "implemented-cuda-dry-rk3-unlinked"
CUDA_V841_IMPLEMENTED_UNLINKED_EVIDENCE = (
    "implemented-v841-cuda-closed-dry-split3-unlinked-no-authority-claim"
)
CUDA_V841_AUTHORITY_NONCLAIMS = (
    "native nonzero tracer",
    "native nonzero u_init/v_init",
    "native nonzero dss",
    "mixing",
    "physics",
)
CUDA_V841_PHYSICS_IMPLEMENTED_EVIDENCE = (
    "implemented-v841-cuda-held-column-physics-uncommitted-wsm6-"
    "no-authority-claim"
)
CUDA_V841_PHYSICS_COMMITTED_EVIDENCE = (
    "implemented-v841-cuda-held-column-physics-post-wsm6-committed-"
    "no-authority-claim"
)
CUDA_V841_PHYSICS_GWDO_IMPLEMENTED_EVIDENCE = (
    "implemented-v841-cuda-external-ysu-gwdo-held-column-physics-"
    "uncommitted-wsm6-no-authority-claim"
)
CUDA_V841_PHYSICS_GWDO_COMMITTED_EVIDENCE = (
    "implemented-v841-cuda-external-ysu-gwdo-held-column-physics-"
    "post-wsm6-committed-no-authority-claim"
)
CUDA_V841_PHYSICS_AUTHORITY_NONCLAIMS = (
    "native full-physics CUDA authority",
    "native moist-dynamics CUDA authority",
    "gravity-wave drag",
    "final forecast authority",
)
CUDA_V841_PHYSICS_COMPONENTS = (
    "legacy_rrtmg_lw",
    "legacy_rrtmg_sw",
    "cld_fraction",
    "sf_monin_obukhov_rev",
    "sf_noahmp",
    "bl_ysu",
    "cu_grell_freitas",
    "mp_wsm6_post_rk",
)
CUDA_V841_PHYSICS_GWDO_AUTHORITY_NONCLAIMS = (
    "native full-physics CUDA authority",
    "native moist-dynamics CUDA authority",
    "final forecast authority",
    "WOOF GWDO execution",
)
CUDA_V841_PHYSICS_GWDO_COMPONENTS = (
    "legacy_rrtmg_lw",
    "legacy_rrtmg_sw",
    "cld_fraction",
    "sf_monin_obukhov_rev",
    "sf_noahmp",
    "bl_ysu",
    "cu_grell_freitas",
    "bl_ysu_gwdo_external_backend",
    "mp_wsm6_post_rk",
)
V841_WSM6_DYNAMICS_SCALAR_NAMES = ("qv", "qc", "qr", "qi", "qs", "qg")
V841_MOIST_DYNAMICS_NEGATIVE_QV_POLICY = (
    "raw-start-state-qv-included-before-post-rk-wsm6-clamp"
)
CUDA_AUTHORITY_RULER_SCHEMA = "mpas-port.cuda-authority-ruler/v2"
CUDA_AUTHORITY_INITIAL_SCHEMA = "mpas-port.cuda-authority-initial/v2"
FROZEN_CUDA_SOURCE = "MPAS-A v8.2.3 mpas_atm_time_integration.F:638-1224"
V841_CUDA_SOURCE = (
    "MPAS-A v8.4.2 tag-object=2eee75b147d83d67835cecfb30c6fb3793e3042b; "
    "commit=f34984b2c89353cf31c235b08d3b4acd96b37bb3; CUDA implementation-only"
)
V841_MOIST_DYNAMICS_SOURCE = (
    "MPAS-A v8.4.2 mpas_atm_time_integration.F:3188-3283,"
    "6458-6495,6778-6787"
)
V841_MOIST_DYNAMICS_SOURCE_SHA256 = (
    "579acb2819b565760822194401d5af9e33c129a4114ac03e6882d7abdbabc757"
)
CUDA_T0_EXACT_SIDECAR = "uploaded-exact-sidecar"
CUDA_T0_REBUILT_DIAGNOSTICS = "device-rebuilt-public-t0"
CUDA_LAYOUT_CONTRACT = {
    "logical_order": "[level,entity]",
    "storage": "C-contiguous; entity fastest",
    "launch": "one thread per horizontal owner; ascending level loop",
}

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_AUTHORITY_STATE_FIELDS = ("rho", "rho_theta", "rho_u", "rho_w", "scalars")
_AUTHORITY_SIDECAR_FIELDS = (
    "theta_m",
    "exner",
    "density_perturbation",
    "rho_theta_perturbation",
    "pressure_perturbation",
    "normal_velocity",
    "vertical_velocity",
)
_AUTHORITY_MESH_INDEX_FIELDS = (
    "cellsOnEdge",
    "edgesOnCell",
    "nEdgesOnCell",
    "cellsOnCell",
    "verticesOnEdge",
    "edgesOnEdge",
    "nEdgesOnEdge",
    "verticesOnCell",
    "edgesOnVertex",
    "cellsOnVertex",
)
_AUTHORITY_MESH_FLOAT_FIELDS = (
    "weightsOnEdge",
    "dcEdge",
    "dvEdge",
    "areaCell",
    "areaTriangle",
    "kiteAreasOnVertex",
    "latCell",
    "lonCell",
    "latEdge",
    "lonEdge",
    "angleEdge",
    "meshDensity",
    "defc_a",
    "defc_b",
    "fVertex",
    "fEdge",
    "spec_zone_mask_edge",
)
_AUTHORITY_VERTICAL_FIELDS = (
    "zw",
    "dzw",
    "rdzw",
    "zu",
    "dzu",
    "rdzu",
    "rdzwp",
    "rdzwm",
    "fzp",
    "fzm",
    "ah",
    "hx",
    "zgrid",
    "zz",
    "zxu",
    "dss",
)
_AUTHORITY_REFERENCE_FIELDS = (
    "rho_base",
    "rho_theta_base",
    "pressure_base",
    "exner_base",
)
_AUTHORITY_TERRAIN_FIELDS = ("zb_cell", "zb3_cell")
_AUTHORITY_ADVECTION_FLOAT_FIELDS = ("adv_coefs", "adv_coefs_3rd")
_AUTHORITY_ADVECTION_INDEX_FIELDS = (
    "n_adv_cells_for_edge",
    "adv_cells_for_edge",
)


def cuda_configuration_payload(config: DryDycoreConfig) -> dict[str, Any]:
    """Materialize every registered config field in declaration order."""

    return {field.name: getattr(config, field.name) for field in fields(config)}

def _v841_physics_receipt_lane(
    config: DryDycoreConfig,
    execution_provenance: "CudaPhaseOneExecutionProvenanceV841",
) -> dict[str, Any]:
    """Derive exact off/on receipt evidence from validated execution provenance."""

    from .config_v841 import (
        V841MpasColumnPhysicsGwdoConfig,
    )
    from .cuda_physics_v841 import CudaPhaseOneExecutionProvenanceV841

    from .arwen_les import validate
    validate(config)
    if not isinstance(
        execution_provenance, CudaPhaseOneExecutionProvenanceV841
    ):
        raise TypeError(
            "full physics requires CudaPhaseOneExecutionProvenanceV841"
        )
    execution_provenance.validate()
    aggregate_config = {
        "phase1_arwen_commit": execution_provenance.arwen_commit,
        "phase1_aggregate_factory": execution_provenance.aggregate_factory,
        "phase1_orchestration": execution_provenance.phase1_orchestration,
        "phase1_source_manifest": execution_provenance.source_manifest,
        "phase1_h_diabatic_applied": execution_provenance.h_diabatic_applied,
    }
    def components(names):
        if getattr(config, "config_les_model", "none") == "none":
            return names
        return tuple(name for name in names if name != "bl_ysu" and not (
            name == "cu_grell_freitas" and config.config_convection_scheme == "off"
        )) + ("cuda_les_v842",)
    for name, executed_value in aggregate_config.items():
        if getattr(config, name, None) != executed_value:
            raise ValueError(
                f"phase-one execution provenance does not match config field {name}"
            )

    if not isinstance(config, V841MpasColumnPhysicsGwdoConfig):
        if execution_provenance.gwd_selector != "off":
            raise ValueError(
                "GWD-off full-physics config requires exact GWD-off execution provenance"
            )
        return {
            "candidate_evidence": CUDA_V841_PHYSICS_IMPLEMENTED_EVIDENCE,
            "committed_evidence": CUDA_V841_PHYSICS_COMMITTED_EVIDENCE,
            "authority_nonclaims": CUDA_V841_PHYSICS_AUTHORITY_NONCLAIMS,
            "physics_components": components(CUDA_V841_PHYSICS_COMPONENTS),
            "gwd_scheme": "off",
            "gwd_evidence": None,
        }

    if execution_provenance.gwd_selector != config.config_gwdo_scheme:
        raise ValueError(
            "GWD-on full-physics config requires exact composed GWDO execution provenance"
        )
    validation = execution_provenance.gwdo_validation_d2h
    if validation is None or validation.bytes != 4:
        raise ValueError("GWD-on provenance must carry its exact four-byte validation")
    return {
        "candidate_evidence": CUDA_V841_PHYSICS_GWDO_IMPLEMENTED_EVIDENCE,
        "committed_evidence": CUDA_V841_PHYSICS_GWDO_COMMITTED_EVIDENCE,
        "authority_nonclaims": CUDA_V841_PHYSICS_GWDO_AUTHORITY_NONCLAIMS,
        "physics_components": components(CUDA_V841_PHYSICS_GWDO_COMPONENTS),
        "gwd_scheme": execution_provenance.gwd_selector,
        "gwd_evidence": {
            "composition_owner": execution_provenance.gwdo_composer,
            "composition_phase": execution_provenance.gwdo_composition_phase,
            "executed": execution_provenance.gwdo_executed,
            "composed_once": execution_provenance.gwdo_composed_once,
            "result_module": execution_provenance.gwdo_result_module,
            "result_class": execution_provenance.gwdo_result_class,
            "contract_sha256": execution_provenance.gwdo_contract_sha256,
            "kernel_sha256": execution_provenance.gwdo_kernel_sha256,
            "validation_d2h": validation.as_dict(),
            "input_du_is_arwen_output": (
                execution_provenance.gwdo_input_du_is_arwen_output
            ),
            "input_dv_is_arwen_output": (
                execution_provenance.gwdo_input_dv_is_arwen_output
            ),
            "raw_du_is_gwdo_output": execution_provenance.raw_du_is_gwdo_output,
            "raw_dv_is_gwdo_output": execution_provenance.raw_dv_is_gwdo_output,
            "arwen_execution_claim": False,
        },
    }


def _v841_physics_cadences(config: DryDycoreConfig) -> dict[str, float | None]:
    """The cadence each physics component is called on, in seconds.

    ``convection`` is ``None`` when no cumulus scheme is selected, and that
    is the truthful value rather than a zero: zero would read as "called
    every instant" in a table whose other entries are call intervals.  It
    was ruled on 2026-08-26 that convection is switched off below 3 km, so
    this is a configuration the lane runs, not an absence to paper over.

    THE BREAKAGE THIS PREVENTS, MEASURED (2026-08-26, RTX 5070 Ti):
    this function called ``float(config.config_cudt_seconds)``
    unconditionally, so the first convection-off arm bound clean, admitted
    its timestep, built its sealed constructor and died inside composite
    step 0 with ``TypeError: float() argument must be a string or a real
    number, not 'NoneType'`` -- surfacing as ``composite step at 0.0 s was
    aborted without publication``, which names neither the field nor the
    reason.
    """

    cudt = config.config_cudt_seconds
    return {
        "radiation_lw": float(config.config_radt_seconds),
        "radiation_sw": float(config.config_radt_seconds),
        "surface_pbl": float(config.config_bldt_seconds),
        "convection": None if cudt is None else float(cudt),
        "microphysics": float(config.config_dt),
    }


def _forecast_checks_active() -> bool:
    """Whether an MPAS-physics forecast loop collects device refusals off the step path (deferred_checks_v842)."""
    from .mpasphys.deferred_checks_v842 import active
    return active() is not None


def _sum_transfer_stats(*values: TransferStats | None) -> TransferStats:
    selected = tuple(value for value in values if value is not None)
    return TransferStats(
        sum(value.bytes for value in selected),
        sum(value.seconds for value in selected),
    )


def _authority_array_fingerprint(
    value: Any,
    *,
    dtype: Any | None = None,
) -> dict[str, Any]:
    array = np.asarray(value, dtype=dtype)
    if array.dtype.hasobject:
        raise TypeError("authority arrays cannot use object dtype")
    contiguous = np.ascontiguousarray(array)
    # A byte view keeps the exact C-order representation without a whole-array
    # bytes allocation. Flatten before viewing to handle empty/structured and
    # datetime arrays as well as the numeric forecast fields.
    raw = memoryview(contiguous.reshape(-1).view(np.uint8))
    return {
        "dtype": contiguous.dtype.str,
        "shape": list(contiguous.shape),
        "c_bytes": len(raw),
        "c_bytes_sha256": hashlib.sha256(raw).hexdigest(),
    }


def _authority_array_fingerprints(
    arrays: Mapping[str, tuple[Any, Any]],
) -> dict[str, Any]:
    """Hash independent, stable host inputs with bounded CPU parallelism.

    The 30 km preflight profile spent 5.1 s in array fingerprinting after
    advection preparation was threaded. SHA-256 releases the GIL; each
    worker preserves one array's entire digest and the caller's NumPy error
    policy. Ordered map retains record order and first-refusal order. Small
    groups stay serial, and no pool survives the fingerprint call.
    """
    workers = min(len(arrays), max(1, int(os.environ.get(
        "WOOF_HEX_HOSTPREP_THREADS", "4"))))
    large = sum(value.nbytes for value, _ in arrays.values()
                if isinstance(value, np.ndarray)) >= 16 * 1024 * 1024
    error_policy = np.geterr()

    def record(item):
        name, (value, dtype) = item
        with np.errstate(**error_policy):
            return name, _authority_array_fingerprint(value, dtype=dtype)

    if workers <= 1 or not large:
        return dict(map(record, arrays.items()))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return dict(pool.map(record, arrays.items()))


def _authority_member(value: Any, name: str, default: Any = None) -> Any:
    if hasattr(value, name):
        return getattr(value, name)
    arrays = getattr(value, "arrays", None)
    if isinstance(arrays, Mapping) and name in arrays:
        return arrays[name]
    attrs = getattr(value, "attrs", None)
    if isinstance(attrs, Mapping) and name in attrs:
        return attrs[name]
    return default


def _authority_group_fingerprint(
    arrays: Mapping[str, tuple[Any, Any]],
    scalars: Mapping[str, Any],
) -> dict[str, Any]:
    normalized_scalars: dict[str, int | float] = {}
    for name, value in scalars.items():
        selected = value.item() if isinstance(value, np.generic) else value
        if not isinstance(selected, (int, float)) or not np.isfinite(float(selected)):
            raise ValueError(f"authority scalar {name} must be finite numeric")
        normalized_scalars[name] = selected
    return {
        "arrays": _authority_array_fingerprints(arrays),
        "scalars": normalized_scalars,
    }


def cuda_authority_initial_fingerprint(
    state: PrognosticState,
    saved_diagnostics: DrySavedDiagnostics,
    *,
    mesh: Any | None = None,
    vertical: Any | None = None,
    reference: Any | None = None,
    terrain_metrics: Any | None = None,
    config: DryDycoreConfig | None = None,
    advection_coefficients: Any | None = None,
) -> dict[str, Any]:
    """Fingerprint exact host t0 and, when supplied, every execution input.

    The two-argument form remains useful for low-level binder validation.  A
    linked :meth:`from_host` execution always measures the complete form, so a
    state-only binder cannot attach authority evidence to a real CUDA step.
    """

    start_time = float(state.time_seconds)
    if not np.isfinite(start_time):
        raise ValueError("authority initial model time must be finite")
    result = {
        "schema": CUDA_AUTHORITY_INITIAL_SCHEMA,
        "start_time_seconds": start_time,
        "state": _authority_array_fingerprints({
            name: (getattr(state, name), None) for name in _AUTHORITY_STATE_FIELDS
        }),
        "saved_diagnostics": _authority_array_fingerprints({
            name: (getattr(saved_diagnostics, name), None)
            for name in _AUTHORITY_SIDECAR_FIELDS
        }),
    }
    execution_values = (
        mesh,
        vertical,
        reference,
        terrain_metrics,
        config,
    )
    if any(value is not None for value in execution_values):
        if any(value is None for value in execution_values):
            raise ValueError(
                "complete authority execution fingerprint requires mesh, vertical, "
                "reference, terrain_metrics, and config"
            )
        assert mesh is not None
        assert vertical is not None
        assert reference is not None
        assert terrain_metrics is not None
        assert config is not None
        coefficients = advection_coefficients
        if coefficients is None:
            coefficients = build_advection_coefficients(
                mesh,
                config_scalar_adv_order=config.config_scalar_adv_order,
                n_vert_levels=state.rho.shape[0],
            )
        selected_vertical = replace(
            vertical,
            dss=_frozen_vertical_damping(mesh, vertical, config),
        )
        dimensions = getattr(mesh, "dimensions", {})
        n_cells = int(
            dimensions.get(
                "nCells", np.asarray(_authority_member(mesh, "latCell")).size
            )
        )
        n_edges = int(
            dimensions.get("nEdges", np.asarray(_authority_member(mesh, "dcEdge")).size)
        )
        n_vertices = int(
            dimensions.get(
                "nVertices",
                np.asarray(_authority_member(mesh, "areaTriangle")).size,
            )
        )
        max_edges = int(
            dimensions.get(
                "maxEdges",
                np.asarray(_authority_member(mesh, "edgesOnCell")).shape[1],
            )
        )
        max_edges2 = int(
            dimensions.get(
                "maxEdges2",
                np.asarray(_authority_member(mesh, "edgesOnEdge")).shape[1],
            )
        )
        vertex_degree = int(
            dimensions.get(
                "vertexDegree",
                np.asarray(_authority_member(mesh, "cellsOnVertex")).shape[1],
            )
        )
        nominal_min_dc = _authority_member(mesh, "nominalMinDc")
        if nominal_min_dc is None:
            nominal_min_dc = _authority_member(mesh, "nominal_min_dc")
        if nominal_min_dc is None:
            raise ValueError("authority mesh nominalMinDc is missing")
        spec_zone = _authority_member(mesh, "spec_zone_mask_edge")
        if spec_zone is None:
            spec_zone = np.zeros(n_edges, dtype=np.float32)
        mesh_arrays = {
            name: (_authority_member(mesh, name), np.int32)
            for name in _AUTHORITY_MESH_INDEX_FIELDS
        }
        mesh_arrays.update(
            {
                name: (
                    spec_zone
                    if name == "spec_zone_mask_edge"
                    else _authority_member(mesh, name),
                    np.float32,
                )
                for name in _AUTHORITY_MESH_FLOAT_FIELDS
            }
        )
        result["execution_inputs"] = {
            "mesh": _authority_group_fingerprint(
                mesh_arrays,
                {
                    "n_cells": n_cells,
                    "n_edges": n_edges,
                    "n_vertices": n_vertices,
                    "max_edges": max_edges,
                    "max_edges2": max_edges2,
                    "vertex_degree": vertex_degree,
                    "nominal_min_dc": float(np.asarray(nominal_min_dc)),
                },
            ),
            "vertical": _authority_group_fingerprint(
                {
                    name: (getattr(selected_vertical, name), np.float32)
                    for name in _AUTHORITY_VERTICAL_FIELDS
                },
                {
                    "cf1": float(selected_vertical.cf1),
                    "cf2": float(selected_vertical.cf2),
                    "cf3": float(selected_vertical.cf3),
                    "first_height_level": int(selected_vertical.first_height_level),
                    "n_vert_levels": int(np.asarray(selected_vertical.zz).shape[0]),
                },
            ),
            "reference": _authority_group_fingerprint(
                {
                    name: (getattr(reference, name), np.float32)
                    for name in _AUTHORITY_REFERENCE_FIELDS
                },
                {},
            ),
            "terrain": _authority_group_fingerprint(
                {
                    name: (getattr(terrain_metrics, name), np.float32)
                    for name in _AUTHORITY_TERRAIN_FIELDS
                },
                {},
            ),
            "advection": _authority_group_fingerprint(
                {
                    **{
                        name: (getattr(coefficients, name), np.float32)
                        for name in _AUTHORITY_ADVECTION_FLOAT_FIELDS
                    },
                    **{
                        name: (getattr(coefficients, name), np.int32)
                        for name in _AUTHORITY_ADVECTION_INDEX_FIELDS
                    },
                },
                {"horizontal_order": int(coefficients.horizontal_order)},
            ),
        }
    return result


def _validate_authority_array_record(record: Any, name: str) -> None:
    if not isinstance(record, Mapping) or set(record) != {
        "dtype",
        "shape",
        "c_bytes",
        "c_bytes_sha256",
    }:
        raise ValueError(f"authority initial fingerprint {name} is incomplete")
    if not isinstance(record["dtype"], str) or not record["dtype"]:
        raise ValueError(f"authority fingerprint dtype is invalid for {name}")
    shape = record["shape"]
    if not isinstance(shape, list) or any(
        not isinstance(extent, int) or extent < 0 for extent in shape
    ):
        raise ValueError(f"authority fingerprint shape is invalid for {name}")
    if not isinstance(record["c_bytes"], int) or record["c_bytes"] < 0:
        raise ValueError(f"authority fingerprint byte count is invalid for {name}")
    digest = record["c_bytes_sha256"]
    if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
        raise ValueError(f"authority fingerprint digest is invalid for {name}")


def _validate_initial_fingerprint(value: Mapping[str, Any]) -> None:
    if value.get("schema") != CUDA_AUTHORITY_INITIAL_SCHEMA:
        raise ValueError("authority initial fingerprint schema changed")
    start_time = value.get("start_time_seconds")
    if not isinstance(start_time, (int, float)) or not np.isfinite(float(start_time)):
        raise ValueError("authority initial fingerprint start time is invalid")
    for group_name, expected_fields in (
        ("state", _AUTHORITY_STATE_FIELDS),
        ("saved_diagnostics", _AUTHORITY_SIDECAR_FIELDS),
    ):
        group = value.get(group_name)
        if not isinstance(group, Mapping) or set(group) != set(expected_fields):
            raise ValueError(
                f"authority initial fingerprint {group_name} is incomplete"
            )
        for name in expected_fields:
            record = group[name]
            _validate_authority_array_record(record, f"{group_name}.{name}")
    execution = value.get("execution_inputs")
    if execution is not None:
        expected_arrays = {
            "mesh": set(_AUTHORITY_MESH_INDEX_FIELDS)
            | set(_AUTHORITY_MESH_FLOAT_FIELDS),
            "vertical": set(_AUTHORITY_VERTICAL_FIELDS),
            "reference": set(_AUTHORITY_REFERENCE_FIELDS),
            "terrain": set(_AUTHORITY_TERRAIN_FIELDS),
            "advection": set(_AUTHORITY_ADVECTION_FLOAT_FIELDS)
            | set(_AUTHORITY_ADVECTION_INDEX_FIELDS),
        }
        expected_scalars = {
            "mesh": {
                "n_cells",
                "n_edges",
                "n_vertices",
                "max_edges",
                "max_edges2",
                "vertex_degree",
                "nominal_min_dc",
            },
            "vertical": {
                "cf1",
                "cf2",
                "cf3",
                "first_height_level",
                "n_vert_levels",
            },
            "reference": set(),
            "terrain": set(),
            "advection": {"horizontal_order"},
        }
        if not isinstance(execution, Mapping) or set(execution) != set(expected_arrays):
            raise ValueError("authority execution-input fingerprint is incomplete")
        for group_name in expected_arrays:
            group = execution[group_name]
            if not isinstance(group, Mapping) or set(group) != {"arrays", "scalars"}:
                raise ValueError(
                    f"authority execution fingerprint {group_name} is incomplete"
                )
            arrays = group["arrays"]
            scalars = group["scalars"]
            if (
                not isinstance(arrays, Mapping)
                or set(arrays) != expected_arrays[group_name]
            ):
                raise ValueError(
                    f"authority execution arrays {group_name} are incomplete"
                )
            if (
                not isinstance(scalars, Mapping)
                or set(scalars) != expected_scalars[group_name]
            ):
                raise ValueError(
                    f"authority execution scalars {group_name} are incomplete"
                )
            for name, record in arrays.items():
                _validate_authority_array_record(
                    record,
                    f"execution_inputs.{group_name}.{name}",
                )
            if any(
                not isinstance(item, (int, float)) or not np.isfinite(float(item))
                for item in scalars.values()
            ):
                raise ValueError(
                    f"authority execution scalars {group_name} are invalid"
                )


@dataclass(frozen=True, slots=True)
class CudaAuthorityRulerBinder:
    """Explicit, configuration-pinned authority admitted by a CUDA receipt.

    Merely selecting JW-like numbers never creates this object.  A caller must
    name a checked ruler, pin its fixture manifest, and state the exact full
    configuration digest that the ruler admits.  The driver revalidates all of
    those claims at the execution boundary before it emits linked evidence.
    """

    evidence: str
    payload: dict[str, Any]
    payload_sha256: str

    @classmethod
    def validated(
        cls,
        *,
        evidence: str,
        identity: str,
        fixture_manifest_sha256: str,
        admitted_configuration_sha256: str,
        admitted_initial_fingerprint: Mapping[str, Any],
        ruler: Mapping[str, Any] | None = None,
    ) -> "CudaAuthorityRulerBinder":
        if evidence not in (
            CUDA_WHOLE_STEP_EVIDENCE,
            CUDA_ORIGINAL_JW_BRANCH_EVIDENCE,
        ):
            raise ValueError(
                "authority ruler requires a registered linked evidence label"
            )
        if not isinstance(identity, str) or not identity.strip():
            raise ValueError("authority ruler identity must be a nonempty string")
        for name, value in (
            ("fixture_manifest_sha256", fixture_manifest_sha256),
            ("admitted_configuration_sha256", admitted_configuration_sha256),
        ):
            if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        extra = {} if ruler is None else dict(ruler)
        _validate_initial_fingerprint(admitted_initial_fingerprint)
        initial_fingerprint = deepcopy(dict(admitted_initial_fingerprint))
        initial_sha256 = canonical_sha256(initial_fingerprint)
        reserved = {
            "schema",
            "identity",
            "fixture_manifest_sha256",
            "admitted_configuration_sha256",
            "admitted_initial_fingerprint",
            "admitted_initial_fingerprint_sha256",
            "admitted_start_time_seconds",
        }
        overlap = reserved.intersection(extra)
        if overlap:
            raise ValueError(
                "authority ruler extras cannot replace reserved fields: "
                + ", ".join(sorted(overlap))
            )
        payload = {
            "schema": CUDA_AUTHORITY_RULER_SCHEMA,
            "identity": identity.strip(),
            "fixture_manifest_sha256": fixture_manifest_sha256,
            "admitted_configuration_sha256": admitted_configuration_sha256,
            "admitted_initial_fingerprint": initial_fingerprint,
            "admitted_initial_fingerprint_sha256": initial_sha256,
            "admitted_start_time_seconds": initial_fingerprint["start_time_seconds"],
            **extra,
        }
        return cls(
            evidence=evidence,
            payload=payload,
            payload_sha256=canonical_sha256(payload),
        )

    def validate(
        self,
        configuration_sha256: str,
        initial_fingerprint: Mapping[str, Any],
    ) -> None:
        if self.evidence not in (
            CUDA_WHOLE_STEP_EVIDENCE,
            CUDA_ORIGINAL_JW_BRANCH_EVIDENCE,
        ):
            raise ValueError("authority ruler carries an unregistered evidence label")
        if self.payload.get("schema") != CUDA_AUTHORITY_RULER_SCHEMA:
            raise ValueError("authority ruler schema changed")
        identity = self.payload.get("identity")
        if not isinstance(identity, str) or not identity.strip():
            raise ValueError("authority ruler identity is missing")
        fixture_sha256 = self.payload.get("fixture_manifest_sha256")
        if (
            not isinstance(fixture_sha256, str)
            or _SHA256_RE.fullmatch(fixture_sha256) is None
        ):
            raise ValueError("authority ruler fixture manifest SHA-256 is invalid")
        admitted = self.payload.get("admitted_configuration_sha256")
        if admitted != configuration_sha256:
            raise ValueError(
                "authority ruler configuration mismatch: "
                f"{configuration_sha256} != {admitted}"
            )
        _validate_initial_fingerprint(initial_fingerprint)
        measured_initial = deepcopy(dict(initial_fingerprint))
        admitted_initial = self.payload.get("admitted_initial_fingerprint")
        if not isinstance(admitted_initial, Mapping):
            raise ValueError("authority ruler has no admitted initial fingerprint")
        _validate_initial_fingerprint(admitted_initial)
        admitted_initial_sha256 = canonical_sha256(admitted_initial)
        if self.payload.get("admitted_initial_fingerprint_sha256") != (
            admitted_initial_sha256
        ):
            raise ValueError("authority ruler admitted-initial digest is false")
        measured_initial_sha256 = canonical_sha256(measured_initial)
        if measured_initial_sha256 != admitted_initial_sha256:
            raise ValueError(
                "authority ruler initial execution input mismatch: "
                f"{measured_initial_sha256} != {admitted_initial_sha256}"
            )
        admitted_start = self.payload.get("admitted_start_time_seconds")
        measured_start = measured_initial["start_time_seconds"]
        if admitted_start != measured_start:
            raise ValueError(
                "authority ruler start time mismatch: "
                f"{measured_start} != {admitted_start}"
            )
        measured = canonical_sha256(self.payload)
        if self.payload_sha256 != measured:
            raise ValueError(
                "authority ruler payload SHA-256 mismatch: "
                f"{self.payload_sha256} != {measured}"
            )


def _resolve_cuda_receipt_provenance(
    config: DryDycoreConfig,
    authority_ruler: CudaAuthorityRulerBinder | None,
    initial_fingerprint: Mapping[str, Any] | None,
) -> tuple[
    dict[str, Any],
    str,
    str,
    dict[str, Any] | None,
    str | None,
]:
    configuration = cuda_configuration_payload(config)
    configuration_sha256 = canonical_sha256(configuration)
    if authority_ruler is None:
        return (
            configuration,
            configuration_sha256,
            CUDA_IMPLEMENTED_UNLINKED_EVIDENCE,
            None,
            None,
        )
    if not isinstance(authority_ruler, CudaAuthorityRulerBinder):
        raise TypeError("authority_ruler must be a validated CudaAuthorityRulerBinder")
    if initial_fingerprint is None:
        raise ValueError(
            "linked authority requires from_host t0 fingerprint validation"
        )
    authority_ruler.validate(configuration_sha256, initial_fingerprint)
    return (
        configuration,
        configuration_sha256,
        authority_ruler.evidence,
        deepcopy(authority_ruler.payload),
        authority_ruler.payload_sha256,
    )


_CUDA_SOURCE = (
    CUDA_FTZ_HELPERS + CUDA_NATIVE_POWF
    + r"""
#define C2(k,c,nc) ((k)*(nc) + (c))
#define E2(k,e,ne) ((k)*(ne) + (e))
#define CES(c,s,me) ((c)*(me) + (s))
/* Level-independent kernels walk a flat (level, owner) element index, owner
   fastest, in a grid-stride loop; each element runs the former per-owner
   loop body for its level unchanged, so the launch geometry is not in the
   arithmetic.  The forecast launches one thread per element. */
#define DRIVER_ELEMENT_LOOP(total) \
    for (int element = blockDim.x * blockIdx.x + threadIdx.x; \
         element < (total); element += gridDim.x * blockDim.x)
#define ADV(e,s,w) ((e)*(w) + (s))

// The flux3/flux4 statement-function denominator of
// mpas_atm_time_integration.F:4625-4638, held as a translation-unit constant
// rather than written as a source literal.  MEASURED: NVRTC rewrites
// ``x / <float literal>`` as ``x * (1/<literal>)`` for every target at or
// above compute_100, which is one ulp off the quotient the CPU authority
// computes; a symbol the host can write cannot legally be folded, so the
// division survives.  See tools/audit_nvrtc_reciprocal_rewrite.py.
__constant__ float mpas_third_order_denominator = 12.0f;

extern "C" __global__ void euler_w_f32(
    const int nlev, const int ncells, const float *pressure_p,
    const float *dpdz, const float *rdzu, const float *fzm,
    const float *fzp, float *result)
{
    DRIVER_ELEMENT_LOOP((nlev + 1) * ncells) {
        const int cell = element % ncells;
        const int k = element / ncells;
        const int index = C2(k, cell, ncells);
        if (k == 0 || k == nlev) {
            result[index] = 0.0f;
        } else {
            result[index] = mpas_sub(0.0f, mpas_sub(mpas_mul(rdzu[k], mpas_sub(
                pressure_p[C2(k, cell, ncells)],
                pressure_p[C2(k - 1, cell, ncells)])), mpas_add(
                    mpas_mul(fzm[k], dpdz[C2(k, cell, ncells)]),
                    mpas_mul(fzp[k],
                        dpdz[C2(k - 1, cell, ncells)]))));
        }
    }
}

extern "C" __global__ void vertical_u_flux_f32(
    const int nlev, const int ncells, const int nedges,
    const float *u, const float *rw, const int *cells_on_edge,
    const float *fzm, const float *fzp, float *flux)
{
    DRIVER_ELEMENT_LOOP((nlev + 1) * nedges) {
        const int edge = element % nedges;
        const int k = element / nedges;
        const int c0 = cells_on_edge[2 * edge];
        const int c1 = cells_on_edge[2 * edge + 1];
        const int index = E2(k, edge, nedges);
        float value = 0.0f;
        if (k > 0 && k < nlev) {
            const float velocity = mpas_mul(0.5f, mpas_add(
                rw[C2(k, c0, ncells)], rw[C2(k, c1, ncells)]));
            if (k == 1 || k == nlev - 1) {
                value = mpas_mul(velocity, mpas_add(
                    mpas_mul(fzm[k], u[E2(k, edge, nedges)]),
                    mpas_mul(fzp[k], u[E2(k - 1, edge, nedges)])));
            } else {
                const float qm2 = u[E2(k - 2, edge, nedges)];
                const float qm1 = u[E2(k - 1, edge, nedges)];
                const float qi = u[E2(k, edge, nedges)];
                const float qp1 = u[E2(k + 1, edge, nedges)];
                value = mpas_add(mpas_div(mpas_mul(velocity, mpas_sub(
                    mpas_mul(7.0f, mpas_add(qi, qm1)),
                    mpas_add(qp1, qm2))),
                    mpas_third_order_denominator), mpas_div(mpas_mul(
                        mpas_abs(velocity), mpas_sub(mpas_sub(qp1, qm2),
                            mpas_mul(3.0f, mpas_sub(qi, qm1)))),
                        mpas_third_order_denominator));
            }
        }
        flux[index] = value;
    }
}

extern "C" __global__ void vertical_u_finish_f32(
    const int nlev, const int nedges, const float *rdzw,
    const float *flux, float *result)
{
    DRIVER_ELEMENT_LOOP(nlev * nedges) {
        const int edge = element % nedges;
        const int k = element / nedges;
        const int index = E2(k, edge, nedges);
        result[index] = mpas_mul(mpas_sub(0.0f, rdzw[k]), mpas_sub(
            flux[E2(k + 1, edge, nedges)], flux[index]));
    }
}

extern "C" __global__ void vector_momentum_f32(
    const int nlev, const int ncells, const int nedges, const int max_edges2,
    const float *u, const float *rho_edge, const float *pv_edge,
    const float *kinetic, const float *mass_divergence,
    const int *cells_on_edge, const int *edges_on_edge,
    const int *n_edges_on_edge, const float *weights_on_edge,
    const float *dc_edge, float *result)
{
    const int edge = blockDim.x * blockIdx.x + threadIdx.x;
    if (edge >= nedges) return;
    const int c0 = cells_on_edge[2 * edge];
    const int c1 = cells_on_edge[2 * edge + 1];
    for (int k = 0; k < nlev; ++k) {
        const int index = E2(k, edge, nedges);
        float q = 0.0f;
        for (int slot = 0; slot < n_edges_on_edge[edge]; ++slot) {
            const int neighbor = edges_on_edge[edge * max_edges2 + slot];
            const float work_pv = mpas_mul(0.5f, mpas_add(
                pv_edge[index], pv_edge[E2(k, neighbor, nedges)]));
            q = mpas_add(q, mpas_mul(mpas_mul(
                weights_on_edge[edge * max_edges2 + slot],
                u[E2(k, neighbor, nedges)]), work_pv));
        }
        result[index] = mpas_sub(mpas_mul(rho_edge[index], mpas_sub(q,
            mpas_div(mpas_sub(kinetic[C2(k, c1, ncells)],
                kinetic[C2(k, c0, ncells)]), dc_edge[edge]))),
            mpas_mul(mpas_mul(u[index], 0.5f), mpas_add(
                mass_divergence[C2(k, c0, ncells)],
                mass_divergence[C2(k, c1, ncells)])));
    }
}

extern "C" __global__ void theta_edge_flux_f32(
    const int nlev, const int ncells, const int nedges, const int width,
    const int rk_step, const float coefficient,
    const float *theta, const float *theta_saved,
    const float *ru, const float *ru_saved, const float *dv_edge,
    const int *cells_on_edge, const float *adv, const float *adv3,
    const int *n_adv, const int *adv_cells, float *flux)
{
    DRIVER_ELEMENT_LOOP(nlev * nedges) {
        const int edge = element % nedges;
        const int k = element / nedges;
        const int c0 = cells_on_edge[2 * edge];
        const int c1 = cells_on_edge[2 * edge + 1];
        const int index = E2(k, edge, nedges);
        float edge_theta = 0.0f;
        const float sign = mpas_copysign(1.0f, ru[index]);
        for (int slot = 0; slot < n_adv[edge]; ++slot) {
            const int cell = adv_cells[ADV(edge, slot, width)];
            const float weight = mpas_add(adv[ADV(edge, slot, width)],
                mpas_mul(mpas_mul(coefficient, sign),
                    adv3[ADV(edge, slot, width)]));
            edge_theta = mpas_add(edge_theta, mpas_mul(
                weight, theta[C2(k, cell, ncells)]));
        }
        float value = mpas_mul(ru[index], edge_theta);
        if (rk_step > 1) {
            value = mpas_add(value, mpas_mul(mpas_mul(mpas_mul(
                dv_edge[edge], mpas_sub(ru_saved[index], ru[index])), 0.5f),
                mpas_add(theta_saved[C2(k, c0, ncells)],
                    theta_saved[C2(k, c1, ncells)])));
        }
        flux[index] = value;
    }
}

extern "C" __global__ void theta_vertical_flux_f32(
    const int nlev, const int ncells, const float coefficient,
    const float *theta, const float *theta_saved,
    const float *rw, const float *rw_saved,
    const float *fzm, const float *fzp, float *flux)
{
    DRIVER_ELEMENT_LOOP((nlev + 1) * ncells) {
        const int cell = element % ncells;
        const int k = element / ncells;
        const int index = C2(k, cell, ncells);
        float value = 0.0f;
        if (k > 0 && k < nlev) {
            if (k == 1) {
                value = mpas_mul(rw[index], mpas_add(
                    mpas_mul(fzm[k], theta[C2(k, cell, ncells)]),
                    mpas_mul(fzp[k],
                        theta[C2(k - 1, cell, ncells)])));
            } else if (k == nlev - 1) {
                value = mpas_mul(rw_saved[index], mpas_add(
                    mpas_mul(fzm[k], theta[C2(k, cell, ncells)]),
                    mpas_mul(fzp[k],
                        theta[C2(k - 1, cell, ncells)])));
            } else {
                const float qm2 = theta[C2(k - 2, cell, ncells)];
                const float qm1 = theta[C2(k - 1, cell, ncells)];
                const float qi = theta[C2(k, cell, ncells)];
                const float qp1 = theta[C2(k + 1, cell, ncells)];
                const float velocity = rw[index];
                value = mpas_add(mpas_div(mpas_mul(velocity, mpas_sub(
                    mpas_mul(7.0f, mpas_add(qi, qm1)),
                    mpas_add(qp1, qm2))),
                    mpas_third_order_denominator), mpas_div(mpas_mul(
                        mpas_mul(coefficient, mpas_abs(velocity)),
                        mpas_sub(mpas_sub(qp1, qm2),
                            mpas_mul(3.0f, mpas_sub(qi, qm1)))),
                        mpas_third_order_denominator));
            }
            if (k < nlev - 1) {
                value = mpas_add(value, mpas_mul(
                    mpas_sub(rw_saved[index], rw[index]), mpas_add(
                        mpas_mul(fzm[k],
                            theta_saved[C2(k, cell, ncells)]),
                        mpas_mul(fzp[k],
                            theta_saved[C2(k - 1, cell, ncells)]))));
            }
        }
        flux[index] = value;
    }
}

extern "C" __global__ void theta_finish_f32(
    const int nlev, const int ncells, const int nedges, const int max_edges,
    const int *n_edges_on_cell, const int *edges_on_cell,
    const float *acoustic_sign, const float *area_cell, const float *rdzw,
    const float *edge_flux, const float *vertical_flux, float *result)
{
    const int cell = blockDim.x * blockIdx.x + threadIdx.x;
    if (cell >= ncells) return;
    for (int k = 0; k < nlev; ++k) {
        const int index = C2(k, cell, ncells);
        float value = 0.0f;
        for (int slot = 0; slot < n_edges_on_cell[cell]; ++slot) {
            const int edge = edges_on_cell[CES(cell, slot, max_edges)];
            value = mpas_sub(value, mpas_mul(
                acoustic_sign[CES(cell, slot, max_edges)],
                edge_flux[E2(k, edge, nedges)]));
        }
        value = mpas_div(value, area_cell[cell]);
        value = mpas_sub(value, mpas_mul(rdzw[k], mpas_sub(
            vertical_flux[C2(k + 1, cell, ncells)], vertical_flux[index])));
        result[index] = value;
    }
}

extern "C" __global__ void w_edge_flux_f32(
    const int nlev, const int ncells, const int nedges, const int width,
    const float coefficient, const float *w, const float *ru,
    const float *fzm, const float *fzp, const float *adv, const float *adv3,
    const int *n_adv, const int *adv_cells, float *flux)
{
    DRIVER_ELEMENT_LOOP((nlev + 1) * nedges) {
        const int edge = element % nedges;
        const int k = element / nedges;
        const int index = E2(k, edge, nedges);
        float value = 0.0f;
        if (k > 0 && k < nlev) {
            const float ru_interface = mpas_add(
                mpas_mul(fzm[k], ru[E2(k, edge, nedges)]),
                mpas_mul(fzp[k], ru[E2(k - 1, edge, nedges)]));
            const float sign = mpas_copysign(1.0f, ru_interface);
            float edge_w = 0.0f;
            for (int slot = 0; slot < n_adv[edge]; ++slot) {
                const int cell = adv_cells[ADV(edge, slot, width)];
                edge_w = mpas_add(edge_w, mpas_mul(mpas_add(
                    adv[ADV(edge, slot, width)], mpas_mul(
                        mpas_mul(coefficient, sign),
                        adv3[ADV(edge, slot, width)])),
                    w[C2(k, cell, ncells)]));
            }
            value = mpas_mul(ru_interface, edge_w);
        }
        flux[index] = value;
    }
}

#ifndef HEX_HOIST_ADV
#define HEX_HOIST_ADV 10
#endif
/* Thread-per-edge forms of theta_edge_flux_f32 and w_edge_flux_f32.  The
   edge's advection stencil (cells, both coefficient rows) is read once into
   registers and the thread walks the levels; each level runs the element
   body unchanged, so each flux carries the same bits. */
extern "C" __global__ void theta_edge_flux_hoist_f32(
    const int nlev, const int ncells, const int nedges, const int width,
    const int rk_step, const float coefficient,
    const float *theta, const float *theta_saved,
    const float *ru, const float *ru_saved, const float *dv_edge,
    const int *cells_on_edge, const float *adv, const float *adv3,
    const int *n_adv, const int *adv_cells, float *flux, const int lpt)
{
    const int hex_rows = nlev;
    const int hex_chunks = (hex_rows + lpt - 1) / lpt;
    const int hex_element = blockDim.x * blockIdx.x + threadIdx.x;
    if (hex_element >= nedges * hex_chunks) return;
    const int edge = hex_element % nedges;
    const int hex_k0 = (hex_element / nedges) * lpt;
    const int hex_k1 = min(hex_rows, hex_k0 + lpt);
    const int c0 = cells_on_edge[2 * edge];
    const int c1 = cells_on_edge[2 * edge + 1];
    const int count = n_adv[edge];
    if (count > HEX_HOIST_ADV) {
        for (int k = hex_k0; k < hex_k1; ++k) {
            const int index = E2(k, edge, nedges);
            float edge_theta = 0.0f;
            const float sign = mpas_copysign(1.0f, ru[index]);
            for (int slot = 0; slot < count; ++slot) {
                const int cell = adv_cells[ADV(edge, slot, width)];
                const float weight = mpas_add(adv[ADV(edge, slot, width)],
                    mpas_mul(mpas_mul(coefficient, sign),
                        adv3[ADV(edge, slot, width)]));
                edge_theta = mpas_add(edge_theta, mpas_mul(
                    weight, theta[C2(k, cell, ncells)]));
            }
            float value = mpas_mul(ru[index], edge_theta);
            if (rk_step > 1) {
                value = mpas_add(value, mpas_mul(mpas_mul(mpas_mul(
                    dv_edge[edge], mpas_sub(ru_saved[index], ru[index])), 0.5f),
                    mpas_add(theta_saved[C2(k, c0, ncells)],
                        theta_saved[C2(k, c1, ncells)])));
            }
            flux[index] = value;
        }
        return;
    }
    int cl[HEX_HOIST_ADV];
    float a1[HEX_HOIST_ADV];
    float a3[HEX_HOIST_ADV];
#pragma unroll
    for (int slot = 0; slot < HEX_HOIST_ADV; ++slot) {
        cl[slot] = slot < count ? adv_cells[ADV(edge, slot, width)] : c0;
        a1[slot] = slot < count ? adv[ADV(edge, slot, width)] : 0.0f;
        a3[slot] = slot < count ? adv3[ADV(edge, slot, width)] : 0.0f;
    }
    const float dv = dv_edge[edge];
    for (int k = hex_k0; k < hex_k1; ++k) {
        const int index = E2(k, edge, nedges);
        float edge_theta = 0.0f;
        const float ru_k = ru[index];
        const float sign = mpas_copysign(1.0f, ru_k);
#pragma unroll
        for (int slot = 0; slot < HEX_HOIST_ADV; ++slot) {
            if (slot < count) {
                const float weight = mpas_add(a1[slot],
                    mpas_mul(mpas_mul(coefficient, sign), a3[slot]));
                edge_theta = mpas_add(edge_theta, mpas_mul(
                    weight, theta[C2(k, cl[slot], ncells)]));
            }
        }
        float value = mpas_mul(ru_k, edge_theta);
        if (rk_step > 1) {
            value = mpas_add(value, mpas_mul(mpas_mul(mpas_mul(
                dv, mpas_sub(ru_saved[index], ru_k)), 0.5f),
                mpas_add(theta_saved[C2(k, c0, ncells)],
                    theta_saved[C2(k, c1, ncells)])));
        }
        flux[index] = value;
    }
}

extern "C" __global__ void w_edge_flux_hoist_f32(
    const int nlev, const int ncells, const int nedges, const int width,
    const float coefficient, const float *w, const float *ru,
    const float *fzm, const float *fzp, const float *adv, const float *adv3,
    const int *n_adv, const int *adv_cells, float *flux, const int lpt)
{
    /* rows 0..nlev of one edge; rows 0 and nlev are the element kernel's
       zero rows, every other row runs its body unchanged. */
    const int hex_rows = nlev + 1;
    const int hex_chunks = (hex_rows + lpt - 1) / lpt;
    const int hex_element = blockDim.x * blockIdx.x + threadIdx.x;
    if (hex_element >= nedges * hex_chunks) return;
    const int edge = hex_element % nedges;
    const int hex_k0 = (hex_element / nedges) * lpt;
    const int hex_k1 = min(hex_rows, hex_k0 + lpt);
    const int count = n_adv[edge];
    const bool hoisted = count <= HEX_HOIST_ADV;
    int cl[HEX_HOIST_ADV];
    float a1[HEX_HOIST_ADV];
    float a3[HEX_HOIST_ADV];
#pragma unroll
    for (int slot = 0; slot < HEX_HOIST_ADV; ++slot) {
        const bool live = hoisted && slot < count;
        cl[slot] = live ? adv_cells[ADV(edge, slot, width)] : 0;
        a1[slot] = live ? adv[ADV(edge, slot, width)] : 0.0f;
        a3[slot] = live ? adv3[ADV(edge, slot, width)] : 0.0f;
    }
    for (int k = hex_k0; k < hex_k1; ++k) {
        const int index = E2(k, edge, nedges);
        float value = 0.0f;
        if (k > 0 && k < nlev) {
            const float ru_interface = mpas_add(
                mpas_mul(fzm[k], ru[E2(k, edge, nedges)]),
                mpas_mul(fzp[k], ru[E2(k - 1, edge, nedges)]));
            const float sign = mpas_copysign(1.0f, ru_interface);
            float edge_w = 0.0f;
            if (hoisted) {
#pragma unroll
                for (int slot = 0; slot < HEX_HOIST_ADV; ++slot) {
                    if (slot < count) {
                        edge_w = mpas_add(edge_w, mpas_mul(mpas_add(
                            a1[slot], mpas_mul(mpas_mul(coefficient, sign), a3[slot])),
                            w[C2(k, cl[slot], ncells)]));
                    }
                }
            } else {
                for (int slot = 0; slot < count; ++slot) {
                    const int cell = adv_cells[ADV(edge, slot, width)];
                    edge_w = mpas_add(edge_w, mpas_mul(mpas_add(
                        adv[ADV(edge, slot, width)], mpas_mul(
                            mpas_mul(coefficient, sign),
                            adv3[ADV(edge, slot, width)])),
                        w[C2(k, cell, ncells)]));
                }
            }
            value = mpas_mul(ru_interface, edge_w);
        }
        flux[index] = value;
    }
}

extern "C" __global__ void w_vertical_flux_f32(
    const int nlev, const int ncells, const float *w,
    const float *rw, float *flux)
{
    DRIVER_ELEMENT_LOOP((nlev + 1) * ncells) {
        const int cell = element % ncells;
        const int k = element / ncells;
        const int index = C2(k, cell, ncells);
        float value = 0.0f;
        if (k > 0 && k < nlev) {
            const float velocity = mpas_mul(0.5f, mpas_add(
                rw[index], rw[C2(k - 1, cell, ncells)]));
            if (k == 1 || k == nlev - 1) {
                value = mpas_mul(mpas_mul(0.5f, velocity), mpas_add(
                    w[index], w[C2(k - 1, cell, ncells)]));
            } else {
                const float qm2 = w[C2(k - 2, cell, ncells)];
                const float qm1 = w[C2(k - 1, cell, ncells)];
                const float qi = w[index];
                const float qp1 = w[C2(k + 1, cell, ncells)];
                value = mpas_add(mpas_div(mpas_mul(velocity, mpas_sub(
                    mpas_mul(7.0f, mpas_add(qi, qm1)),
                    mpas_add(qp1, qm2))),
                    mpas_third_order_denominator), mpas_div(mpas_mul(
                        mpas_abs(velocity), mpas_sub(mpas_sub(qp1, qm2),
                            mpas_mul(3.0f, mpas_sub(qi, qm1)))),
                        mpas_third_order_denominator));
            }
        }
        flux[index] = value;
    }
}

extern "C" __global__ void w_finish_f32(
    const int nlev, const int ncells, const int nedges, const int max_edges,
    const int *n_edges_on_cell, const int *edges_on_cell,
    const float *acoustic_sign, const float *area_cell, const float *rdzu,
    const float *edge_flux, const float *vertical_flux, float *result)
{
    const int cell = blockDim.x * blockIdx.x + threadIdx.x;
    if (cell >= ncells) return;
    for (int k = 0; k <= nlev; ++k) {
        const int index = C2(k, cell, ncells);
        if (k == 0 || k == nlev) {
            result[index] = 0.0f;
        } else {
            float value = 0.0f;
            for (int slot = 0; slot < n_edges_on_cell[cell]; ++slot) {
                const int edge = edges_on_cell[CES(cell, slot, max_edges)];
                value = mpas_sub(value, mpas_mul(
                    acoustic_sign[CES(cell, slot, max_edges)],
                    edge_flux[E2(k, edge, nedges)]));
            }
            value = mpas_div(value, area_cell[cell]);
            value = mpas_sub(value, mpas_mul(rdzu[k], mpas_sub(
                vertical_flux[C2(k + 1, cell, ncells)],
                vertical_flux[index])));
            result[index] = value;
        }
    }
}

extern "C" __global__ void add_inplace_f32(
    const int nlevels, const int nowners,
    const float *increment, float *target)
{
    DRIVER_ELEMENT_LOOP(nlevels * nowners) {
        target[element] = mpas_add(target[element], increment[element]);
    }
}

extern "C" __global__ void scale_f32(
    const int nlevels, const int nowners, const float scale,
    const float *source, float *target)
{
    DRIVER_ELEMENT_LOOP(nlevels * nowners) {
        target[element] = mpas_mul(scale, source[element]);
    }
}

extern "C" __global__ void recover_cells_f32(
    const int nlev, const int ncells, const int stage, const int native_gnu_powf,
    const float rgas, const float cp, const float p0,
    const float *rho_p_saved, const float *rtheta_p_saved,
    const float *rho_pp, const float *rtheta_pp,
    const int has_diabatic, const float dt, const float *diabatic_theta,
    const float *rho_base, const float *rtheta_base,
    const float *exner_base, const float *zz,
    const float *exner_saved, const float *pressure_p_saved,
    float *rho, float *rtheta, float *rho_p, float *rtheta_p,
    float *theta, float *exner, float *pressure_p)
{
    DRIVER_ELEMENT_LOOP(nlev * ncells) {
        const int index = element;
        const float density_p = mpas_add(
            rho_p_saved[index], rho_pp[index]);
        float theta_p = mpas_add(
            rtheta_p_saved[index], rtheta_pp[index]);
        const float full_rho = mpas_add(density_p, rho_base[index]);
        // mpas_atm_time_integration.F:4436-4437: remove lagged microphysics
        // heating only at the final RK recovery, after the density update.
        if (stage == 3 && has_diabatic)
            theta_p = mpas_sub(theta_p,
                mpas_mul(mpas_mul(dt, full_rho), diabatic_theta[index]));
        const float full_rtheta = mpas_add(theta_p, rtheta_base[index]);
        rho_p[index] = density_p;
        rtheta_p[index] = theta_p;
        rho[index] = full_rho;
        rtheta[index] = full_rtheta;
        theta[index] = mpas_div(full_rtheta, full_rho);
        if (stage == 3) {
            const float argument = mpas_mul(mpas_mul(
                zz[index], mpas_div(rgas, p0)), full_rtheta);
            // v8.4.1 selects the GNU native powf polynomial; legacy retains double pow.
            const float value = native_gnu_powf ? mpas_native_powf_positive(
                argument, mpas_div(rgas, mpas_sub(cp, rgas))) : __double2float_rn(pow(
                mpas_f32_to_f64_ieee(argument),
                mpas_f32_to_f64_ieee(mpas_div(rgas, mpas_sub(cp, rgas)))));
            exner[index] = value;
            pressure_p[index] = mpas_mul(mpas_mul(zz[index], rgas),
                mpas_add(mpas_mul(value, theta_p),
                    mpas_mul(rtheta_base[index],
                        mpas_sub(value, exner_base[index]))));
        } else {
            exner[index] = exner_saved[index];
            pressure_p[index] = pressure_p_saved[index];
        }
    }
}

extern "C" __global__ void recover_edges_f32(
    const int nlev, const int nedges, const int acoustic_steps,
    const float *ru_saved, const float *ru_p, const float *ru_avg_p,
    float *ru, float *flux_u)
{
    DRIVER_ELEMENT_LOOP(nlev * nedges) {
        const int index = element;
        ru[index] = mpas_add(ru_saved[index], ru_p[index]);
        flux_u[index] = mpas_add(ru_saved[index],
            mpas_mul(ru_avg_p[index], mpas_div(1.0f, (float)acoustic_steps)));
    }
}

extern "C" __global__ void recover_interfaces_f32(
    const int nlev, const int ncells, const int acoustic_steps,
    const float *rw_saved, const float *rw_p, const float *ww_avg_p,
    float *rw, float *flux_w)
{
    DRIVER_ELEMENT_LOOP((nlev + 1) * ncells) {
        const int k = element / ncells;
        const int index = element;
        if (k == 0 || k == nlev) {
            rw[index] = 0.0f;
        } else {
            rw[index] = mpas_add(rw_saved[index], rw_p[index]);
        }
        flux_w[index] = mpas_add(rw_saved[index],
            mpas_mul(ww_avg_p[index], mpas_div(1.0f, (float)acoustic_steps)));
    }
}
"""
)

CUDA_V841_PHYSICS_DRIVER_SOURCE = CUDA_FTZ_HELPERS + r"""
#define C2(k,c,nc) ((k)*(nc) + (c))
#define E2(k,e,ne) ((k)*(ne) + (e))
#define S3(s,k,c,nl,nc) ((((s)*(nl) + (k))*(nc)) + (c))

extern "C" __global__ void moist_cell_coefficients_v841_f32(
    const int nlev, const int ncells, const int nmass, const float *scalars,
    float *qtot, float *cqw, int *invalid)
{
    const int cell = blockDim.x * blockIdx.x + threadIdx.x;
    if (cell >= ncells) return;
    for (int k = 0; k < nlev; ++k) {
        const int i = C2(k, cell, ncells);
        float total = 0.0f;
        // nmass, not nScalars: only the mass-LOADING species enter qtot.
        // The species row orders them as a contiguous leading run, so one
        // count is the whole rule (hexcore.species_row).  A number
        // concentration folded in here would collapse 1/(1+qtot); a rime
        // MASS that is a component of another species would double-count it.
        for (int species = 0; species < nmass; ++species) {
            const float value = scalars[S3(species, k, cell, nlev, ncells)];
            if (!isfinite(value)) atomicExch(invalid, 1);
            total = mpas_add(total, value);
        }
        if (!isfinite(total)) atomicExch(invalid, 1);
        qtot[i] = total;
    }
    // Native Fortran assigns cqw only for k=2..nVertLevels.  The unused
    // k=1/zero-based boundary is intentionally left untouched here.
    for (int k = 1; k < nlev; ++k) {
        const int i = C2(k, cell, ncells);
        const float total = mpas_mul(0.5f, mpas_add(
            qtot[i], qtot[C2(k - 1, cell, ncells)]));
        const float value = mpas_div(1.0f, mpas_add(1.0f, total));
        if (!isfinite(total) || !isfinite(value)) atomicExch(invalid, 1);
        cqw[i] = value;
    }
}

extern "C" __global__ void moist_edge_coefficients_v841_f32(
    const int nlev, const int ncells, const int nedges, const int nmass,
    const int *cells_on_edge, const float *scalars,
    float *cqu, int *invalid)
{
    const int edge = blockDim.x * blockIdx.x + threadIdx.x;
    if (edge >= nedges) return;
    const int cell1 = cells_on_edge[2 * edge];
    const int cell2 = cells_on_edge[2 * edge + 1];
    for (int k = 0; k < nlev; ++k) {
        float total = 0.0f;
        // See moist_cell_coefficients_v841_f32: nmass is the loading run.
        // The per-species average is summed in species order, NOT summed
        // then averaged -- that ordering is FP32-observable and is what the
        // frozen WSM6 arm's bytes were measured under.
        for (int species = 0; species < nmass; ++species) {
            const float q1 = scalars[S3(species, k, cell1, nlev, ncells)];
            const float q2 = scalars[S3(species, k, cell2, nlev, ncells)];
            if (!isfinite(q1) || !isfinite(q2)) atomicExch(invalid, 1);
            total = mpas_add(total, mpas_mul(0.5f, mpas_add(q1, q2)));
        }
        const float value = mpas_div(1.0f, mpas_add(1.0f, total));
        if (!isfinite(total) || !isfinite(value)) atomicExch(invalid, 1);
        cqu[E2(k, edge, nedges)] = value;
    }
}

extern "C" __global__ void moist_dpdz_v841_f32(
    const int nlev, const int ncells, const float gravity,
    const float *rho_base, const float *density_perturbation,
    const float *qtot, float *dpdz, int *invalid)
{
    const int cell = blockDim.x * blockIdx.x + threadIdx.x;
    if (cell >= ncells) return;
    for (int k = 0; k < nlev; ++k) {
        const int i = C2(k, cell, ncells);
        const float moist_base = mpas_mul(rho_base[i], qtot[i]);
        const float moist_perturbation = mpas_mul(density_perturbation[i],
            mpas_add(1.0f, qtot[i]));
        const float value = mpas_mul(mpas_sub(0.0f, gravity),
            mpas_add(moist_base, moist_perturbation));
        if (!isfinite(rho_base[i]) || !isfinite(density_perturbation[i])
                || !isfinite(qtot[i]) || !isfinite(value))
            atomicExch(invalid, 1);
        dpdz[i] = value;
    }
}

extern "C" __global__ void euler_w_moist_v841_f32(
    const int nlev, const int ncells, const float *pressure_p,
    const float *dpdz, const float *cqw, const float *rdzu,
    const float *fzm, const float *fzp, const int has_mixing,
    const float *mixing, float *result, int *invalid)
{
    const int cell = blockDim.x * blockIdx.x + threadIdx.x;
    if (cell >= ncells) return;
    for (int k = 0; k <= nlev; ++k) {
        const int index = C2(k, cell, ncells);
        if (k == 0 || k == nlev) {
            result[index] = 0.0f;
        } else {
            const float pressure_term = mpas_mul(rdzu[k], mpas_sub(
                pressure_p[C2(k, cell, ncells)],
                pressure_p[C2(k - 1, cell, ncells)]));
            const float buoyancy = mpas_add(
                mpas_mul(fzm[k], dpdz[C2(k, cell, ncells)]),
                mpas_mul(fzp[k], dpdz[C2(k - 1, cell, ncells)]));
            const float value = mpas_sub(has_mixing ? mixing[index] : 0.0f, mpas_mul(cqw[index],
                mpas_sub(pressure_term, buoyancy)));
            if (!isfinite(cqw[index]) || !isfinite(value))
                atomicExch(invalid, 1);
            result[index] = value;
        }
    }
}

// mpas_atm_time_integration.F:6931-6937: resolved tendency diagnostic
// precedes the carried microphysics heating, Euler mixing and held physics.
extern "C" __global__ void add_lagged_diabatic_theta_v842_f32(
    const int nlev, const int ncells, const float *rho,
    const float *diabatic_theta, float *tend_rt)
{
    const int cell = blockDim.x * blockIdx.x + threadIdx.x;
    if (cell >= ncells) return;
    for (int k = 0; k < nlev; ++k) {
        const int i = C2(k, cell, ncells);
        tend_rt[i] = mpas_add(tend_rt[i], mpas_mul(rho[i], diabatic_theta[i]));
    }
}

// mpas_atm_time_integration.F:7949 (atm_bdy_adjust_dynamics_speczone_tend):
// a limited-area run zeroes rt_diabatic_tend in the specified zone at every
// RK stage, in place, so the final recovery (:4436-4437) removes no carry
// there.  Launched only when a carry exists.
extern "C" __global__ void zero_speczone_lagged_diabatic_theta_v842_f32(
    const int nlev, const int ncells, const int nspec,
    const int *spec_cells, float *diabatic_theta)
{
    const int element = blockDim.x * blockIdx.x + threadIdx.x;
    if (element >= nlev * nspec) return;
    const int slot = element % nspec;
    const int k = element / nspec;
    diabatic_theta[C2(k, spec_cells[slot], ncells)] = 0.0f;
}

extern "C" __global__ void modified_theta_dynamics_rate_v841_f32(
    const int nlev, const int ncells, const float *tend_rtheta,
    const float *tend_rho, const float *rho, const float *theta_m,
    float *result, int *invalid)
{
    const int cell = blockDim.x * blockIdx.x + threadIdx.x;
    if (cell >= ncells) return;
    for (int k = 0; k < nlev; ++k) {
        const int i = C2(k, cell, ncells);
        const float value = mpas_div(mpas_sub(tend_rtheta[i],
            mpas_mul(tend_rho[i], theta_m[i])), rho[i]);
        if (!isfinite(rho[i]) || rho[i] <= 0.0f || !isfinite(value))
            atomicExch(invalid, 1);
        result[i] = value;
    }
}

extern "C" __global__ void gf_dry_theta_dynamics_rate_v841_f32(
    const int nlev, const int ncells, const float rvord,
    const float *modified_rate, const float *theta_m,
    const float *qv, float *rthdynten, float *rqvdynten, int *invalid)
{
    const int cell = blockDim.x * blockIdx.x + threadIdx.x;
    if (cell >= ncells) return;
    for (int k = 0; k < nlev; ++k) {
        const int i = C2(k, cell, ncells);
        const float rqv = 0.0f;
        const float fac_m = mpas_div(1.0f,
            mpas_add(1.0f, mpas_mul(rvord, qv[i])));
        const float theta_local = mpas_mul(theta_m[i], fac_m);
        const float value = mpas_mul(fac_m, mpas_sub(modified_rate[i],
            mpas_mul(mpas_mul(theta_local, rvord), rqv)));
        if (!isfinite(qv[i]) || !isfinite(fac_m) || fac_m <= 0.0f
                || !isfinite(value)) atomicExch(invalid, 1);
        rqvdynten[i] = rqv;
        rthdynten[i] = value;
    }
}
"""


@dataclass(frozen=True, slots=True)
class CudaStepReceipt:
    evidence: str
    configuration: dict[str, Any]
    configuration_sha256: str
    authority_ruler: dict[str, Any] | None
    authority_ruler_sha256: str | None
    frozen_source: str
    t0_diagnostics_source: str
    stage_acoustic_steps: tuple[int, int, int]
    start_time_seconds: float
    end_time_seconds: float
    h2d: TransferStats
    d2h: TransferStats
    compile_manifest: dict[str, Any]
    compile_manifest_sha256: str
    layout_contract: dict[str, str]


@dataclass(frozen=True, slots=True)
class CudaV841StepReceipt(CudaStepReceipt):
    """Additive receipt schema for the implementation-only v8.4.1 lane."""

    source_release: str
    dynamics_split_steps: int
    dynamics_timestep_seconds: float
    dynamics_stage_timesteps: tuple[float, float, float]
    scalar_transport_stage_timesteps: tuple[float, float, float] | None
    split_flux_reduction: str
    authority_nonclaims: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CudaV841PhysicsStepReceipt(CudaV841StepReceipt):
    """Truthful two-phase receipt; candidate and committed states are distinct."""

    bulk_physics_contract_sha256: str
    bulk_physics_tendencies_applied: bool
    physics_components: tuple[str, ...]
    physics_cadences_seconds: dict[str, float | None]
    gwd_scheme: str
    gwd_evidence: dict[str, Any] | None
    phase_one_execution_provenance: "CudaPhaseOneExecutionProvenanceV841"
    gwdo_validation_d2h: TransferStats | None
    scalar_order: tuple[str, ...]
    held_tendency_time_seconds: float
    held_tendency_validation_d2h: TransferStats
    dycore_validation_d2h: TransferStats
    phase2_validation_d2h: TransferStats | None
    final_commit_validation_d2h: TransferStats | None
    moist_dynamics_coefficients_applied: bool
    moist_dynamics_scalar_order: tuple[str, ...]
    moist_dynamics_negative_qv_policy: str
    moist_dynamics_source: str
    moist_dynamics_source_sha256: str
    post_wsm6_h_diabatic_reapplied: bool
    composite_committed: bool
    final_authority_claim: bool


def _validate_v841_physics_candidate_receipt(
    receipt: CudaV841PhysicsStepReceipt,
    *,
    config: DryDycoreConfig,
    configuration: dict[str, Any],
    configuration_sha256: str,
) -> dict[str, Any]:
    """Re-derive and verify every mutable full-physics evidence carrier."""

    from .cuda_physics_v841 import (
        CUDA_PHYSICS_V841_CONTRACT_SHA256,
        CudaPhaseOneExecutionProvenanceV841,
    )

    # The RUN's scalar order, not the frozen lane's.  A receipt that names
    # WSM6's six on a P3 step is a receipt that lies about what integrated,
    # which is the stale-relay class this program has been bitten by.
    receipt_scalar_order = arwen_model_row(config).names()

    if not isinstance(receipt, CudaV841PhysicsStepReceipt):
        raise TypeError("pending full-physics candidate receipt type changed")
    provenance = receipt.phase_one_execution_provenance
    if not isinstance(provenance, CudaPhaseOneExecutionProvenanceV841):
        raise TypeError("candidate receipt lost typed phase-one provenance")
    provenance.validate()
    lane = _v841_physics_receipt_lane(config, provenance)
    expected_cadences = _v841_physics_cadences(config)
    expected_gwd_evidence = lane["gwd_evidence"]

    if receipt.configuration != configuration:
        raise ValueError("candidate configuration payload changed")
    if receipt.configuration_sha256 != configuration_sha256:
        raise ValueError("candidate configuration digest changed")
    if canonical_sha256(receipt.configuration) != configuration_sha256:
        raise ValueError("candidate mutable configuration payload failed its digest")
    if canonical_sha256(receipt.compile_manifest) != receipt.compile_manifest_sha256:
        raise ValueError("candidate mutable compile manifest failed its digest")
    if receipt.layout_contract != dict(CUDA_LAYOUT_CONTRACT):
        raise ValueError("candidate mutable layout contract changed")
    if receipt.physics_cadences_seconds != expected_cadences:
        raise ValueError("candidate mutable physics cadence receipt changed")
    if receipt.gwd_evidence != expected_gwd_evidence:
        raise ValueError("candidate mutable GWD execution evidence changed")

    exact = {
        "evidence": lane["candidate_evidence"],
        "authority_ruler": None,
        "authority_ruler_sha256": None,
        "frozen_source": V841_CUDA_SOURCE,
        "source_release": "v8.4.1",
        "authority_nonclaims": lane["authority_nonclaims"],
        "bulk_physics_contract_sha256": CUDA_PHYSICS_V841_CONTRACT_SHA256,
        "bulk_physics_tendencies_applied": True,
        "physics_components": lane["physics_components"],
        "gwd_scheme": lane["gwd_scheme"],
        "gwdo_validation_d2h": provenance.gwdo_validation_d2h,
        "scalar_order": receipt_scalar_order,
        "held_tendency_time_seconds": receipt.start_time_seconds,
        "phase2_validation_d2h": None,
        "final_commit_validation_d2h": None,
        "moist_dynamics_coefficients_applied": True,
        "moist_dynamics_scalar_order": receipt_scalar_order,
        "moist_dynamics_negative_qv_policy": (
            V841_MOIST_DYNAMICS_NEGATIVE_QV_POLICY
        ),
        "moist_dynamics_source": V841_MOIST_DYNAMICS_SOURCE,
        "moist_dynamics_source_sha256": V841_MOIST_DYNAMICS_SOURCE_SHA256,
        "post_wsm6_h_diabatic_reapplied": False,
        "composite_committed": False,
        "final_authority_claim": False,
    }
    for name, required in exact.items():
        if getattr(receipt, name) != required:
            raise ValueError(f"candidate full-physics receipt field {name} changed")
    if receipt.held_tendency_validation_d2h.bytes != 4:
        raise ValueError("candidate held-tendency validation must remain four bytes")
    if receipt.dycore_validation_d2h.bytes != 4:
        raise ValueError("candidate dycore validation must remain four bytes")
    if float(receipt.end_time_seconds) - float(receipt.start_time_seconds) != float(
        config.config_dt
    ):
        raise ValueError("candidate receipt endpoint no longer matches config_dt")
    expected_d2h = _sum_transfer_stats(
        receipt.held_tendency_validation_d2h,
        receipt.dycore_validation_d2h,
        provenance.gwdo_validation_d2h,
    )
    if receipt.d2h != expected_d2h:
        raise ValueError("candidate validation transfer receipt changed")
    return lane


@dataclass(frozen=True, slots=True)
class _CudaV841MoistDynamicsCoefficients:
    """Resident native cqu/cqw/qtot held for one full dynamics step."""

    qtot: Any
    cqw: Any
    cqu: Any

    def validate(
        self, *, cp: Any, n_vert_levels: int, n_cells: int, n_edges: int
    ) -> None:
        expected = {
            "qtot": (n_vert_levels, n_cells),
            "cqw": (n_vert_levels, n_cells),
            "cqu": (n_vert_levels, n_edges),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if not isinstance(value, cp.ndarray):
                raise TypeError(f"{name} must remain a resident cupy.ndarray")
            if value.dtype != cp.dtype(cp.float32) or tuple(value.shape) != shape:
                raise TypeError(f"{name} must be FP32 with shape {shape}")
            if not value.flags.c_contiguous:
                raise ValueError(f"{name} must be C-contiguous")


@dataclass(frozen=True, slots=True)
class CudaV841GfDynamicsTendencies:
    """Resident endpoint dry-theta/qv dynamics rates consumed by next-step GF."""

    rthdynten: Any
    rqvdynten: Any
    time_seconds: float

    def validate(self, *, cp: Any, n_vert_levels: int, n_cells: int) -> None:
        if not np.isfinite(self.time_seconds) or self.time_seconds < 0.0:
            raise ValueError("GF dynamics tendency time must be finite and non-negative")
        expected = (n_vert_levels, n_cells)
        for name in ("rthdynten", "rqvdynten"):
            value = getattr(self, name)
            if not isinstance(value, cp.ndarray):
                raise TypeError(f"{name} must remain a resident cupy.ndarray")
            if value.dtype != cp.dtype(cp.float32) or tuple(value.shape) != expected:
                raise TypeError(f"{name} must be FP32 [level,cell]")
            if not value.flags.c_contiguous:
                raise ValueError(f"{name} must be C-contiguous")


@dataclass(frozen=True, slots=True)
class CudaV841PhysicsStepCandidate:
    """Validated dycore endpoint intentionally not committed before WSM6."""

    atmosphere: DeviceAtmosphere
    dynamics_tendencies: CudaV841GfDynamicsTendencies
    receipt: CudaV841PhysicsStepReceipt


@dataclass(frozen=True, slots=True)
class CudaV841CommittedPhysicsStepResult:
    atmosphere: DeviceAtmosphere
    dynamics_tendencies: CudaV841GfDynamicsTendencies
    surface_updates: Mapping[str, Any]
    effective_radii: Mapping[str, Any]
    receipt: CudaV841PhysicsStepReceipt


@dataclass(frozen=True, slots=True)
class CudaDeviceStepResult:
    atmosphere: DeviceAtmosphere
    receipt: CudaStepReceipt


@dataclass(frozen=True, slots=True)
class CudaHostStepResult:
    state: PrognosticState
    saved_diagnostics: DrySavedDiagnostics
    receipt: CudaStepReceipt


@dataclass(frozen=True, slots=True)
class _CudaDynamicsSubcycleResult:
    state: DevicePrognosticState
    saved: DeviceSavedDiagnostics
    diagnostics: Any
    mass_flux_u: Any
    mass_flux_w: Any
    modified_theta_dynamics_rate: Any | None = None
    stage_physics_tendencies: Any | None = None


def diagnostics_lifetime_lean() -> bool:
    """Whether a v8.4.1 step holds no solve-diagnostics set between substeps.

    Default yes (memory lane, 2026-10-15).  The old flow formed the step-start
    solve in ``_step_device_v841`` and carried each substep's stage-3 solve
    into the next substep, so the caller held one full set (about 4.7 KB per
    column at 55 levels) while every substep ran, and the last substep's set
    -- read by nothing -- stayed live through the scalar transport and the
    physics callbacks, which is where the moist step peaks.  Lean, each
    substep forms its start solve from its own inputs and skips the unread
    stage-3 solve; the arrays and kernels that form every set a stage reads
    are unchanged, so the step's words are unchanged.  On the CUDA-graph path
    the set is no longer a staged graph input either.
    ``WOOF_HEX_DIAG_LIFETIME=0`` restores the carried flow for A/B runs.
    """

    return os.environ.get("WOOF_HEX_DIAG_LIFETIME", "1") != "0"


def _zero_transfer() -> TransferStats:
    return TransferStats(0, 0.0)


def _level_owner_shape(value: Any, name: str) -> tuple[int, int]:
    """Return ``(vertical extent, horizontal owners)`` for a resident field."""

    if getattr(value, "ndim", None) != 2:
        raise ValueError(f"{name} must have shape (levels,owners)")
    nlevels, nowners = map(int, value.shape)
    if nlevels < 1 or nowners < 1:
        raise ValueError(f"{name} must have non-empty levels and owners")
    return nlevels, nowners


def _copy_device_dataclass(value: Any, cp: Any) -> Any:
    """Clone every resident array in one device dataclass without D2H."""

    replacements = {
        field.name: cp.array(item, copy=True)
        for field in fields(value)
        if isinstance((item := getattr(value, field.name)), cp.ndarray)
    }
    return replace(value, **replacements)


def state_levels_enabled() -> bool:
    """Bind read-only time levels by reference instead of private copies.

    Default on.  ``WOOF_HEX_MEM_STATE_LEVELS=0`` restores the private
    images of the step-start and subcycle-start levels, and keeps the
    dynamics-only arrays alive through transport, for A/B measurement.
    """

    return os.environ.get("WOOF_HEX_MEM_STATE_LEVELS", "1") != "0"


def regional_state_levels_enabled() -> bool:
    """The limited-area lane binds its start levels as well.

    Default on.  ``WOOF_HEX_MEM_STATE_LEVELS_REGIONAL=0`` restores the
    regional lane's private images alone (the global lanes keep binding), for
    A/B word and memory measurement on limited-area meshes.
    """

    return os.environ.get("WOOF_HEX_MEM_STATE_LEVELS_REGIONAL", "1") != "0"


class _StateLevelsWitness:
    """Opt-in proof that the bound time levels are never written in place.

    ``WOOF_HEX_MEM_STATE_WITNESS=1`` keeps a private image of every array a
    binding shares and compares the live words against it, as uint32 so a
    NaN payload or a signed zero counts, at the binding's last reader.  A
    changed word raises: binding by reference would then have changed the
    forecast.  Off by default; it costs the very copies the binding removes.
    """

    checks = 0
    arrays = 0
    _registered = False

    def __init__(self, cp: Any, label: str, **arrays: Any) -> None:
        if not type(self)._registered:
            import atexit
            import sys

            type(self)._registered = True
            atexit.register(lambda: print(type(self).report(), file=sys.stderr, flush=True))
        self.cp, self.label = cp, label
        self.images = {
            name: (value, cp.array(value, copy=True))
            for name, value in arrays.items()
            if isinstance(value, cp.ndarray)
        }

    @staticmethod
    def enabled() -> bool:
        return os.environ.get("WOOF_HEX_MEM_STATE_WITNESS", "0") == "1"

    def check(self, where: str) -> None:
        cp = self.cp
        changed = [
            name for name, (live, image) in self.images.items()
            if not bool(cp.array_equal(live.view(cp.uint32), image.view(cp.uint32)))
        ]
        type(self).checks += 1
        type(self).arrays += len(self.images)
        if changed:
            raise RuntimeError(
                f"state-levels witness: {self.label} changed in place before {where}: "
                + ", ".join(changed)
            )

    @classmethod
    def report(cls) -> str:
        return (f"state-levels witness: {cls.checks} checks, "
                f"{cls.arrays} array comparisons, 0 changed")


def regional_bdy_mask_digest(mesh: Any) -> str | None:
    """SHA-256 of the three 7-ring masks, or ``None`` on a closed mesh.

    The same digest the mesh registry stores as ``bdy_mask_sha256``
    (``tools/mpas_mesh_binding.MeshBinding``), so a regional anchor and a
    registered mesh row are comparable without a second convention.

    That sentence was FALSE until 2026-08-26 and this function is where it
    was false.  There were two conventions: ``mesh.regional_boundary_mask_
    digest`` frames each mask name with a NUL before its little-endian int32
    payload and refuses a partial triple, and this function hashed the three
    payloads bare and silently skipped any that were missing.  Measured on
    ``r4.75.11020``, same three arrays, same file: ``2baf091d...`` against
    ``0c2d9feb...``.

    THE BREAKAGE THAT COST: ``require_regional_anchor`` cross-checks a named
    row's stored digest against the digest of the bytes in hand, and the two
    sides drew from different conventions -- the registry rows carried the
    ``mesh.py`` one, the anchor rows carried this one.  The cross-check could
    therefore never be satisfied by a registry-derived digest, and every
    regional run resolved its anchor by digest alone.  Nothing had failed yet
    only because nothing wrote ``registry_row`` onto a mesh, so the name
    lookup was always skipped.  Two classifiers that disagree about which
    mesh they are looking at is the shape of a defect this project has
    already lost a day to.

    There is now one definition, and it lives with the mesh contract.  The
    partial-triple refusal comes with it: a mesh carrying two of the three
    masks cannot identify its rings, and returning a digest for it would
    admit an anchor nobody checked.
    """

    from .mesh import REGIONAL_BOUNDARY_MASK_NAMES, regional_boundary_mask_digest

    masks = {}
    for name in REGIONAL_BOUNDARY_MASK_NAMES:
        value = _authority_member(mesh, name)
        if value is not None:
            masks[name] = value
    if not masks:
        return None
    return regional_boundary_mask_digest(masks)


def _refuse_regional_execution(mesh: Any, n_cells: int) -> None:
    """Refuse a limited-area cull handed to the GLOBAL device path, by name.

    RETIRED 2026-10-10 as an anchor gate: this used to pass a regional mesh
    through when a registered regional anchor existed and refuse it when
    none did.  Regional runs are built by the regional assembler
    (woof.hex.cuda_regional_forecast_v841), which pads the ring-7 sentinels
    to an explicit garbage cell before this validator sees the mesh, so a
    sentinel HERE means a cull reached the global path.  THE BREAKAGE: the
    global kernels index cellsOnEdge directly and would read the sentinel
    as a cell, and nothing would drive the boundary zone.  That is refused
    whether or not any anchor exists.
    """

    raise ConfigurationRefusal(
        "config_apply_lbcs",
        True,
        f"this {int(n_cells)}-cell mesh is a limited-area cull (stored-0 "
        "cellsOnEdge sentinels or a nonzero specified zone) and reached the "
        "global CUDA path, whose kernels would index the sentinels as cells "
        "and leave the boundary zone undriven",
        "a regional run through the forecast door with --lbc-dir (the "
        "regional assembler pads the rings), or a global mesh",
    )


def _parallel_row_result(values, active, operation, dtype):
    """Independent row reductions, with at most four private scratch buffers."""

    if values.size < host_parallel.MIN_PARALLEL_ELEMENTS or host_parallel.threads() == 1:
        return operation(values, active)
    result = np.empty(values.shape[0], dtype=dtype)

    def reduce(rows):
        result[rows] = operation(values[rows], active[rows])

    host_parallel.run_blocks(reduce, len(values), elements=values.size, worker_limit=4)
    return result


def _rows_outside(values: np.ndarray, active: np.ndarray, extent: int) -> np.ndarray:
    """Rows with an active entry outside ``[0, extent)``."""

    return _parallel_row_result(
        values, active,
        lambda block, used: np.any(used & ((block < 0) | (block >= extent)), axis=1),
        bool,
    )


_ROW_SENTINEL = int(np.iinfo(np.int64).min)


def _rows_not_unique(values: np.ndarray, active: np.ndarray) -> np.ndarray:
    """Rows whose active entries repeat (``np.unique(row[:count]).size != count``)."""

    if values.shape[1] < 2:
        return np.zeros(values.shape[0], dtype=bool)
    sentinel = _ROW_SENTINEL + np.arange(values.shape[1], dtype=np.int64)
    def repeated(block, used):
        ordered = np.sort(np.where(used, block, sentinel[None, :]), axis=1)
        return np.any(ordered[:, 1:] == ordered[:, :-1], axis=1)

    return _parallel_row_result(values, active, repeated, bool)


def _rows_distinct_block(values: np.ndarray, active: np.ndarray) -> np.ndarray:
    width = values.shape[1]
    sentinel = _ROW_SENTINEL + np.arange(width, dtype=np.int64)
    ordered = np.sort(np.where(active, values, sentinel[None, :]), axis=1)
    real = ordered >= _ROW_SENTINEL + width
    new = np.ones(ordered.shape, dtype=bool)
    new[:, 1:] = ordered[:, 1:] != ordered[:, :-1]
    return np.sum(new & real, axis=1)


def _rows_distinct(values: np.ndarray, active: np.ndarray) -> np.ndarray:
    """The number of distinct active entries per row (its set size)."""

    return _parallel_row_result(values, active, _rows_distinct_block, np.int64)


def _raise_first_row_failure(checks, rows_checked: np.ndarray | None) -> None:
    """Raise what the scalar row loop raised: the first failing check of the
    first failing checked row, rows in index order, checks in loop order."""

    failing = None
    for mask, _message in checks:
        failing = np.array(mask, dtype=bool) if failing is None else failing | mask
    if rows_checked is not None:
        failing &= rows_checked
    if failing is None or not failing.any():
        return
    row = int(np.argmax(failing))
    for mask, message in checks:
        if mask[row]:
            raise ValueError(message)


@cached("v841-host-mesh-validation")
def _validate_v841_host_mesh(mesh: Any) -> tuple[int, int, int, int]:
    """Validate every v8.4.1 active connectivity row before CUDA probing."""

    area_cell = np.asarray(_authority_member(mesh, "areaCell"))
    dc_edge = np.asarray(_authority_member(mesh, "dcEdge"))
    area_triangle = np.asarray(_authority_member(mesh, "areaTriangle"))
    n_cells = int(area_cell.size)
    n_edges = int(dc_edge.size)
    n_vertices = int(area_triangle.size)
    if min(n_cells, n_edges, n_vertices) < 1:
        raise ValueError("the v8.4.1 CUDA mesh dimensions must be non-empty")
    declared = getattr(mesh, "dimensions", {})
    for name, actual in (
        ("nCells", n_cells),
        ("nEdges", n_edges),
        ("nVertices", n_vertices),
    ):
        if name in declared and int(declared[name]) != actual:
            raise ValueError(f"mesh declared {name} differs from array extent")

    def index(name: str, shape: tuple[int, ...]) -> np.ndarray:
        value = np.asarray(_authority_member(mesh, name))
        if value.dtype.kind not in "iu":
            raise TypeError(f"{name} must be integer before CUDA upload")
        if value.shape != shape:
            raise ValueError(f"{name} shape {value.shape} != {shape}")
        return value.astype(np.int64, copy=False)

    # Partition-local meshes (build_local_mesh) carry their layout: rows the
    # brief-1 masks mark INCOMPLETE hold deterministic clamped references
    # (absent neighbour -> local index 0) whose outputs are never consumed as
    # owned truth -- the halo exchange overwrites them.  Topology laws below
    # are therefore asserted on COMPLETE rows only; on a whole mesh (layout
    # None) every row is checked exactly as before.
    _layout = getattr(mesh, "partition_layout", None)
    if _layout is None:
        _cell_ok = _edge_ok = _vertex_ok = None
    else:
        _cell_ok = np.asarray(_layout.cell_row_complete, dtype=bool)
        _edge_ok = np.asarray(_layout.edge_row_complete, dtype=bool)
        _vertex_ok = np.asarray(_layout.vertex_row_complete, dtype=bool)
        for mask, extent, what in (
            (_cell_ok, n_cells, "cell"),
            (_edge_ok, n_edges, "edge"),
            (_vertex_ok, n_vertices, "vertex"),
        ):
            if mask.shape != (extent,):
                raise ValueError(
                    f"partition {what} completeness mask does not match the mesh"
                )

    cells_on_edge = index("cellsOnEdge", (n_edges, 2))
    if np.any(cells_on_edge >= n_cells):
        raise ConfigurationRefusal(
            "cellsOnEdge",
            "out of range",
            "v8.4.1 CUDA requires every cellsOnEdge entry inside [0,nCells)",
            "a mesh whose connectivity indexes its own cells",
        )
    if np.any(cells_on_edge < 0):
        # A negative entry here is the regional cull's stored-0 sentinel on a
        # ring-7 row: a cull on the global path (see
        # _refuse_regional_execution for why that cannot run).
        _refuse_regional_execution(mesh, n_cells)
    _same_endpoints = cells_on_edge[:, 0] == cells_on_edge[:, 1]
    if _edge_ok is not None:
        _same_endpoints = _same_endpoints & _edge_ok
    if np.any(_same_endpoints):
        raise ValueError("cellsOnEdge endpoints must be distinct")
    raw_edges_on_cell = np.asarray(_authority_member(mesh, "edgesOnCell"))
    if raw_edges_on_cell.ndim != 2:
        raise ValueError("edgesOnCell must have shape (nCells,maxEdges)")
    max_edges = int(raw_edges_on_cell.shape[1])
    if "maxEdges" in declared and int(declared["maxEdges"]) != max_edges:
        raise ValueError("mesh declared maxEdges differs from edgesOnCell")
    edges_on_cell = index("edgesOnCell", (n_cells, max_edges))
    counts_cell = index("nEdgesOnCell", (n_cells,))
    if np.any(counts_cell < 0) or np.any(counts_cell > max_edges):
        raise ValueError("nEdgesOnCell is outside [0,maxEdges]")
    cells_on_cell = index("cellsOnCell", (n_cells, max_edges))
    vertices_on_cell = index("verticesOnCell", (n_cells, max_edges))
    # The row laws below are evaluated over whole tables at once.  They are
    # the scalar row loops they replace, check for check and in the same
    # order: the first failing COMPLETE row (in row order) raises the message
    # of its first failing check, exactly as the loops did.  THE COST THIS
    # REMOVES (measured 2026-10-13, RTX PRO 6000 box, derecho 3 km patch,
    # 167k cells): every ensemble member's rate classes are member-dependent
    # (fullstep_lts_admission reads the initial state), so each member's
    # class-local meshes missed the preparation cache and these loops cost
    # 30 s + 5 s per member on the host.  tests/test_mesh_validation_rows.py
    # holds the scalar loops and proves the same verdict and message on the
    # same meshes, including mutated ones.
    cell_index = np.arange(n_cells, dtype=np.int64)[:, None]
    active_cell = np.arange(max_edges, dtype=np.int64)[None, :] < counts_cell[:, None]
    clipped_edges = np.clip(edges_on_cell, 0, n_edges - 1)
    endpoint0 = cells_on_edge[clipped_edges, 0]
    endpoint1 = cells_on_edge[clipped_edges, 1]
    incident = (endpoint0 == cell_index) | (endpoint1 == cell_index)
    opposite = np.where(endpoint0 == cell_index, endpoint1, endpoint0)
    incidence_fail = active_cell & ~incident
    opposite_fail = active_cell & incident & (cells_on_cell != opposite)
    slot_fail = incidence_fail | opposite_fail
    first_slot = np.argmax(slot_fail, axis=1)
    slot_row = slot_fail.any(axis=1)
    incidence_first = slot_row & incidence_fail[np.arange(n_cells), first_slot]
    _raise_first_row_failure((
        (_rows_outside(edges_on_cell, active_cell, n_edges),
         "active edgesOnCell entry is outside [0,nEdges)"),
        (_rows_outside(cells_on_cell, active_cell, n_cells),
         "active cellsOnCell entry is outside [0,nCells)"),
        (_rows_outside(vertices_on_cell, active_cell, n_vertices),
         "active verticesOnCell entry is outside [0,nVertices)"),
        (_rows_not_unique(edges_on_cell, active_cell),
         "active edgesOnCell entries must be unique per cell"),
        (_rows_not_unique(cells_on_cell, active_cell),
         "active cellsOnCell entries must be unique per cell"),
        (_rows_not_unique(vertices_on_cell, active_cell),
         "active verticesOnCell entries must be unique per cell"),
        (incidence_first, "active edgesOnCell entry is not incident to its cell"),
        (slot_row & ~incidence_first, "cellsOnCell is not the opposite cell on edgesOnCell"),
    ), _cell_ok)

    raw_edges_on_edge = np.asarray(_authority_member(mesh, "edgesOnEdge"))
    if raw_edges_on_edge.ndim != 2:
        raise ValueError("edgesOnEdge must have shape (nEdges,maxEdges2)")
    max_edges2 = int(raw_edges_on_edge.shape[1])
    if "maxEdges2" in declared and int(declared["maxEdges2"]) != max_edges2:
        raise ValueError("mesh declared maxEdges2 differs from edgesOnEdge")
    edges_on_edge = index("edgesOnEdge", (n_edges, max_edges2))
    counts_edge = index("nEdgesOnEdge", (n_edges,))
    if np.any(counts_edge < 0) or np.any(counts_edge > max_edges2):
        raise ValueError("nEdgesOnEdge is outside [0,maxEdges2]")
    edge_index = np.arange(n_edges, dtype=np.int64)[:, None]
    active_edge = np.arange(max_edges2, dtype=np.int64)[None, :] < counts_edge[:, None]
    share_failure = np.empty(n_edges, dtype=bool)
    union_failure = np.empty(n_edges, dtype=bool)
    def check_edge_topology(block):
        # Gather endpoint and neighbor tables inside bounded worker blocks.
        # Only final row verdicts survive the block, not whole-mesh integer
        # gather tables. Failure selection remains global and ordered below.
        for start in range(block.start, block.stop, 65536):
            rows = slice(start, min(start + 65536, block.stop))
            neighbors = edges_on_edge[rows]
            endpoints = cells_on_edge[rows]
            neighbor = np.clip(neighbors, 0, n_edges - 1)
            neighbor_cell0 = cells_on_edge[neighbor, 0]
            neighbor_cell1 = cells_on_edge[neighbor, 1]
            share = ((neighbor_cell0 == endpoints[:, 0, None])
                     | (neighbor_cell0 == endpoints[:, 1, None])
                     | (neighbor_cell1 == endpoints[:, 0, None])
                     | (neighbor_cell1 == endpoints[:, 1, None]))
            used = active_edge[rows]
            share_failure[rows] = np.any(used & ~share, axis=1)
            # Set semantics: all active edges of either endpoint, minus
            # this edge itself. Inactive slots keep distinct sentinels.
            union = np.concatenate((edges_on_cell[endpoints[:, 0]],
                                    edges_on_cell[endpoints[:, 1]]), axis=1)
            union_active = np.concatenate((active_cell[endpoints[:, 0]],
                                           active_cell[endpoints[:, 1]]), axis=1)
            union_size = _rows_distinct_block(union, union_active) - np.any(
                union_active & (union == edge_index[rows]), axis=1)
            in_union = np.any((neighbors[:, :, None] == union[:, None, :])
                              & union_active[:, None, :], axis=2)
            union_failure[rows] = (np.any(used & ~in_union, axis=1)
                                   | (counts_edge[rows] != union_size))

    host_parallel.run_blocks(check_edge_topology, n_edges, elements=edges_on_edge.size, worker_limit=4)
    _raise_first_row_failure((
        (_rows_outside(edges_on_edge, active_edge, n_edges),
         "active edgesOnEdge entry is outside [0,nEdges)"),
        (np.any(active_edge & (edges_on_edge == edge_index), axis=1)
         | _rows_not_unique(edges_on_edge, active_edge),
         "active edgesOnEdge neighbors must be unique and non-self"),
        (share_failure, "edgesOnEdge neighbors must share a cell endpoint"),
        (union_failure,
         "edgesOnEdge must contain the complete endpoint-cell edge union"),
    ), _edge_ok)

    vertices_on_edge = index("verticesOnEdge", (n_edges, 2))
    if np.any(vertices_on_edge < 0) or np.any(vertices_on_edge >= n_vertices):
        raise ValueError("verticesOnEdge is outside [0,nVertices)")
    _same_vertices = vertices_on_edge[:, 0] == vertices_on_edge[:, 1]
    if _edge_ok is not None:
        _same_vertices = _same_vertices & _edge_ok
    if np.any(_same_vertices):
        raise ValueError("verticesOnEdge endpoints must be distinct")
    edges_on_vertex = index("edgesOnVertex", (n_vertices, 3))
    cells_on_vertex = index("cellsOnVertex", (n_vertices, 3))
    if "vertexDegree" in declared and int(declared["vertexDegree"]) != 3:
        raise ValueError("v8.4.1 CUDA requires mesh vertexDegree=3")
    if np.any(edges_on_vertex < 0) or np.any(edges_on_vertex >= n_edges):
        raise ValueError("v8.4.1 CUDA requires exactly three valid edges per vertex")
    if np.any(cells_on_vertex < 0) or np.any(cells_on_vertex >= n_cells):
        raise ValueError("v8.4.1 CUDA requires exactly three valid cells per vertex")
    vertex_index = np.arange(n_vertices, dtype=np.int64)[:, None]
    every_slot = np.ones((n_vertices, 3), dtype=bool)
    through = ((vertices_on_edge[edges_on_vertex, 0] == vertex_index)
               | (vertices_on_edge[edges_on_vertex, 1] == vertex_index))
    _raise_first_row_failure((
        (_rows_not_unique(edges_on_vertex, every_slot), "edgesOnVertex entries must be distinct"),
        (_rows_not_unique(cells_on_vertex, every_slot), "cellsOnVertex entries must be distinct"),
        (np.any(~through, axis=1), "edgesOnVertex is not incident through verticesOnEdge"),
    ), _vertex_ok)

    edge_column = np.arange(n_edges, dtype=np.int64)
    unreciprocated = np.zeros(n_edges, dtype=bool)
    for side in (0, 1):
        vertex = vertices_on_edge[:, side]
        listed = np.any(edges_on_vertex[vertex] == edge_column[:, None], axis=1)
        checked = np.ones(n_edges, dtype=bool) if _vertex_ok is None else _vertex_ok[vertex]
        unreciprocated |= checked & ~listed
    _raise_first_row_failure((
        (unreciprocated, "verticesOnEdge is not reciprocal with edgesOnVertex"),
    ), _edge_ok)

    kite_vertex = np.clip(vertices_on_cell, 0, n_vertices - 1)
    kite = np.any(cells_on_vertex[kite_vertex] == cell_index[:, :, None], axis=2)
    _raise_first_row_failure((
        (np.any(active_cell & ~kite, axis=1),
         "every active cell-vertex pair requires a kite-area association"),
    ), _cell_ok)

    float_shapes = {
        "weightsOnEdge": (n_edges, max_edges2),
        "dcEdge": (n_edges,),
        "dvEdge": (n_edges,),
        "areaCell": (n_cells,),
        "areaTriangle": (n_vertices,),
        "kiteAreasOnVertex": (n_vertices, 3),
        "angleEdge": (n_edges,),
        "fVertex": (n_vertices,),
        "fEdge": (n_edges,),
        "latCell": (n_cells,),
        "lonCell": (n_cells,),
        "latEdge": (n_edges,),
        "lonEdge": (n_edges,),
        "meshDensity": (n_cells,),
        "defc_a": (n_cells, max_edges),
        "defc_b": (n_cells, max_edges),
    }
    for name, shape in float_shapes.items():
        value = np.asarray(_authority_member(mesh, name))
        if value.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise TypeError(f"{name} must use float32 or float64 source storage")
        if value.shape != shape or not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must be finite with shape {shape}")
        if not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous before CUDA upload")
        if name in ("dcEdge", "dvEdge", "areaCell", "areaTriangle"):
            rounded = value.astype(np.float32, copy=False)
            if np.any(~np.isfinite(rounded)) or np.any(rounded <= 0.0):
                raise ValueError(
                    f"{name} must remain finite and positive after float32 rounding"
                )
    specified = _authority_member(mesh, "spec_zone_mask_edge")
    if specified is not None:
        value = np.asarray(specified)
        if value.shape != (n_edges,) or not np.all(np.isfinite(value)):
            raise ValueError(
                "spec_zone_mask_edge must be finite with shape (nEdges,)"
            )
        if value.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise TypeError("spec_zone_mask_edge must use float32 or float64")
        if not value.flags.c_contiguous:
            raise ValueError("spec_zone_mask_edge must be C-contiguous")
        if np.any(value != 0.0):
            _refuse_regional_execution(mesh, n_cells)
    nominal = _authority_member(mesh, "nominalMinDc")
    if nominal is None:
        nominal = _authority_member(mesh, "nominal_min_dc")
    if nominal is None:
        raise ValueError("mesh.nominalMinDc must be present")
    nominal_array = np.asarray(nominal)
    if nominal_array.shape != ():
        raise ValueError("mesh.nominalMinDc must be scalar")
    if nominal_array.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError("mesh.nominalMinDc must use float32 or float64")
    nominal_rkind = np.float32(nominal_array)
    if not np.isfinite(nominal_rkind) or nominal_rkind <= np.float32(0.0):
        raise ValueError(
            "mesh.nominalMinDc must remain finite and positive in float32"
        )
    return n_cells, n_edges, n_vertices, max_edges


def _validate_v841_host_execution(
    mesh: Any,
    state: PrognosticState,
    vertical: Any,
    reference: Any,
    saved_diagnostics: DrySavedDiagnostics | None,
    terrain_metrics: Any,
) -> tuple[int, int, int]:
    """Close host shapes/dtypes before the first CUDA runtime interaction."""

    n_cells, n_edges, n_vertices, max_edges = _validate_v841_host_mesh(mesh)
    rho = np.asarray(state.rho)
    if rho.ndim != 2:
        raise ValueError("state.rho must have shape (nVertLevels,nCells)")
    nlev = int(rho.shape[0])
    if nlev < 3:
        raise ValueError(
            "v8.4.1 CUDA terrain recovery requires nVertLevels>=3"
        )
    state.validate(n_cells=n_cells, n_edges=n_edges, n_vert_levels=nlev)
    if not np.isfinite(float(state.time_seconds)):
        raise ValueError("state.time_seconds must be finite")
    if np.any(np.asarray(state.rho_theta) <= 0.0):
        raise ValueError("state.rho_theta must be strictly positive")
    reference.validate((nlev, n_cells))

    def f32(name: str, value: Any, shape: tuple[int, ...]) -> np.ndarray:
        array = np.asarray(value)
        if array.dtype != np.dtype(np.float32):
            raise TypeError(f"{name} must use exact float32 RKIND storage")
        if array.shape != shape or not all_finite(array):
            raise ValueError(f"{name} must be finite with shape {shape}")
        if not array.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous before CUDA upload")
        return array

    for name, shape in {
        "rho": (nlev, n_cells),
        "rho_theta": (nlev, n_cells),
        "rho_u": (nlev, n_edges),
        "rho_w": (nlev + 1, n_cells),
        "scalars": (int(np.asarray(state.scalars).shape[0]), nlev, n_cells),
    }.items():
        f32(f"state.{name}", getattr(state, name), shape)
    if int(np.asarray(state.scalars).shape[0]) < 1:
        raise ConfigurationRefusal(
            "state.scalars",
            0,
            "the exact v8.4.1 CUDA execution inventory includes scalar transport",
            "at least one transported scalar with config_scalar_advection=True",
        )
    for name in ("rho_base", "rho_theta_base", "pressure_base", "exner_base"):
        f32(f"reference.{name}", getattr(reference, name), (nlev, n_cells))
    vertical_shapes = {
        "zw": (nlev + 1,),
        "dzw": (nlev,),
        "rdzw": (nlev,),
        "zu": (nlev,),
        "dzu": (nlev,),
        "rdzu": (nlev,),
        "rdzwp": (nlev,),
        "rdzwm": (nlev,),
        "fzp": (nlev,),
        "fzm": (nlev,),
        "ah": (nlev + 1,),
        "hx": (nlev + 1, n_cells),
        "zgrid": (nlev + 1, n_cells),
        "zz": (nlev, n_cells),
        "zxu": (nlev, n_edges),
    }
    for name, shape in vertical_shapes.items():
        f32(f"vertical.{name}", getattr(vertical, name), shape)
    f32("vertical.dss", getattr(vertical, "dss"), (nlev, n_cells))
    for name in ("cf1", "cf2", "cf3"):
        value = np.asarray(getattr(vertical, name))
        if value.shape != () or value.dtype not in (
            np.dtype(np.float32),
            np.dtype(np.float64),
        ):
            raise TypeError(f"vertical.{name} must be a floating RKIND scalar")
        rounded = np.float32(value)
        if not np.isfinite(rounded):
            raise ValueError(f"vertical.{name} must remain finite in float32")
    first_height_level = getattr(vertical, "first_height_level")
    if (
        isinstance(first_height_level, (bool, np.bool_))
        or not isinstance(first_height_level, (int, np.integer))
        or not 1 <= int(first_height_level) <= nlev + 1
    ):
        raise ValueError(
            "vertical.first_height_level must be an integer in [1,nVertLevels+1]"
        )
    if saved_diagnostics is not None:
        saved_diagnostics.validate((nlev, n_cells), rho.dtype, n_edges)
        for name in saved_diagnostics.__slots__:
            shape = (
                (nlev, n_edges)
                if name == "normal_velocity"
                else (nlev + 1, n_cells)
                if name == "vertical_velocity"
                else (nlev, n_cells)
            )
            f32(f"saved.{name}", getattr(saved_diagnostics, name), shape)
    if terrain_metrics is None:
        raise ConfigurationRefusal(
            "config_terrain_following",
            False,
            "the CUDA whole step requires explicit terrain metrics",
            "terrain_metrics=<native TerrainMetrics>",
        )
    terrain_metrics.validate(
        nlev=nlev, ncells=n_cells, max_edges=max_edges
    )
    f32(
        "terrain.zb_cell",
        terrain_metrics.zb_cell,
        (nlev + 1, n_cells, max_edges),
    )
    f32(
        "terrain.zb3_cell",
        terrain_metrics.zb3_cell,
        (nlev + 1, n_cells, max_edges),
    )
    return nlev, n_cells, n_edges


@cached("v841-host-coefficient-validation")
def _validate_v841_host_coefficients(
    coefficients: Any,
    *,
    n_cells: int,
    n_edges: int,
) -> None:
    """Refuse malformed v8.4.1 advection stencils before CUDA upload."""

    adv = np.asarray(coefficients.adv_coefs)
    adv3 = np.asarray(coefficients.adv_coefs_3rd)
    counts = np.asarray(coefficients.n_adv_cells_for_edge)
    cells = np.asarray(coefficients.adv_cells_for_edge)
    if adv.ndim != 2 or adv.shape[0] != n_edges:
        raise ValueError("adv_coefs must have shape (nEdges,stencilWidth)")
    if adv.dtype != np.dtype(np.float32) or adv3.dtype != np.dtype(np.float32):
        raise TypeError("v8.4.1 CUDA advection coefficients must use float32 RKIND")
    if adv3.shape != adv.shape or cells.shape != adv.shape:
        raise ValueError("v8.4.1 CUDA advection coefficient shapes disagree")
    if counts.shape != (n_edges,) or counts.dtype.kind not in "iu":
        raise TypeError("n_adv_cells_for_edge must be one integer per edge")
    if cells.dtype.kind not in "iu":
        raise TypeError("adv_cells_for_edge must be integer")
    if not all_finite(adv) or not all_finite(adv3):
        raise ValueError("v8.4.1 CUDA advection coefficients must be finite")
    for name, value in (
        ("adv_coefs", adv),
        ("adv_coefs_3rd", adv3),
        ("n_adv_cells_for_edge", counts),
        ("adv_cells_for_edge", cells),
    ):
        if not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous before CUDA upload")
    width = int(adv.shape[1])
    if np.any(counts < 0) or np.any(counts > width):
        raise ValueError("n_adv_cells_for_edge is outside coefficient width")
    if not active_indices_in_range(cells, counts, n_cells):
        raise ValueError("active adv_cells_for_edge entry is out of range")


class CudaDryDycoreDriver:
    """One resident CUDA atmosphere and its frozen dry RK3 step."""

    def __init__(
        self,
        atmosphere: DeviceAtmosphere,
        config: DryDycoreConfig,
        advection_coefficients: CudaAdvectionCoefficients,
        *,
        kernel_cache: KernelCache | None = None,
        t0_diagnostics_source: str = CUDA_T0_EXACT_SIDECAR,
        authority_ruler: CudaAuthorityRulerBinder | None = None,
        v841_context: CudaV841Context | None = None,
        les_dissipation: Any | None = None,
        _authority_initial_fingerprint: Mapping[str, Any] | None = None,
    ) -> None:
        config.validate(**({"les_dissipation": les_dissipation} if les_dissipation is not None else {}))
        self.les_dissipation = les_dissipation
        self._validate_config(config)
        source_release = getattr(config, "source_release", "v8.2.3")
        if source_release == "v8.4.1" and v841_context is None:
            raise ConfigurationRefusal(
                "u_init",
                None,
                "the v8.4.1 CUDA mirror requires its complete resident sidecar",
                "reference_wind_profiles with u_init/v_init and a CudaV841Context",
            )
        if source_release == "v8.2.3" and v841_context is not None:
            raise ConfigurationRefusal(
                "source_release",
                source_release,
                "v8.2.3 must not consume v8.4.1 numerical sidecars",
                "omit v841_context",
            )
        if source_release == "v8.4.1" and authority_ruler is not None:
            raise ConfigurationRefusal(
                "authority_ruler",
                "provided",
                "the first v8.4.1 CUDA mirror is implementation-only pending a native nonzero-tracer ruler",
                "authority_ruler=None",
            )
        require_cuda(min_compute=(12, 0))
        import cupy as cp

        if np.dtype(atmosphere.state.dtype) != np.dtype(np.float32):
            raise TypeError("CUDA whole-step authority requires float32 state")
        if atmosphere.terrain is None:
            raise ConfigurationRefusal(
                "config_terrain_following",
                False,
                "the portable CUDA JW gate requires uploaded terrain metrics",
                "config_terrain_following=True",
            )
        self.cp = cp
        self.atmosphere = atmosphere
        self.atmosphere.validate()
        if t0_diagnostics_source not in (
            CUDA_T0_EXACT_SIDECAR,
            CUDA_T0_REBUILT_DIAGNOSTICS,
        ):
            raise ValueError(
                "t0_diagnostics_source must identify the exact or rebuilt path"
            )
        self.t0_diagnostics_source = t0_diagnostics_source
        self.config = config
        self.source_release = source_release
        self.v841_context = v841_context
        (
            self.configuration,
            self.configuration_sha256,
            self.evidence,
            self.authority_ruler,
            self.authority_ruler_sha256,
        ) = _resolve_cuda_receipt_provenance(
            config,
            authority_ruler,
            _authority_initial_fingerprint,
        )
        # ``from_host`` arms a complete private device snapshot only after a
        # possibly absent sidecar has been rebuilt.  Until then no linked
        # execution is current.
        self._consume_authority_snapshot = lambda: None
        self._authority_snapshot_pending = False
        self.mixing_config = (
            _mixing_config(config)
            if (
                config.config_horiz_mixing == "2d_smagorinsky"
                and source_release != "v8.4.1"
            )
            else None
        )
        self.mixing_config_v841 = None
        if (
            source_release == "v8.4.1"
            and config.config_horiz_mixing == "2d_smagorinsky"
            and les_dissipation is None
        ):
            from .mixing_v841 import V841MixingConfig

            self.mixing_config_v841 = V841MixingConfig(
                config_horiz_mixing=config.config_horiz_mixing,
                config_len_disp=config.config_len_disp,
                config_visc4_2dsmag=config.config_visc4_2dsmag,
                config_smagorinsky_coef=config.config_smagorinsky_coef,
                config_del4u_div_factor=config.config_del4u_div_factor,
                config_h_ScaleWithMesh=config.config_h_ScaleWithMesh,
                config_mpas_cam_coef=config.config_mpas_cam_coef,
            )
            self.mixing_config_v841.validate()
        # Treatment proof (A/B rule): count of RK1 mixing computations, one
        # per dynamics subcycle, i.e. 3 per whole step at split-three.
        self.mixing_calls_v841 = 0
        # Two-rank halo executor hook (partition_executor_v841.HaloExchanger).
        # None on every single-device run: each hook site below is a guarded
        # no-op, so the whole-mesh step path is bitwise untouched.  When set,
        # the driver runs on a partition-local mesh and the exchanger delivers
        # owner-truth halo values at the design's B/C/D/E round sites (the A
        # round is the runner's, at the committed step boundary).
        self.halo_exchanger_v841: Any | None = None
        # Regional (limited-area) hook, mirroring the halo-exchanger pattern
        # directly above: None on every global run, so each guarded site
        # below is a no-op and the whole-mesh step path is bitwise untouched.
        # When set (by woof.hex.cuda_regional_forecast_v841, behind the
        # earned-anchor gate) the driver runs on native's padded memory model
        # and the runtime supplies the lateral-boundary stages at the sites
        # the v8.4.1 CPU authority puts them.
        self.regional_v841: Any | None = None
        self.deformation_weights_receipt_v841: dict[str, Any] | None = None
        self.coefficients = advection_coefficients
        self.coefficients.validate()
        self.h2d = TransferStats(
            atmosphere.h2d.bytes
            + advection_coefficients.h2d.bytes
            + (0 if v841_context is None else v841_context.h2d.bytes),
            atmosphere.h2d.seconds
            + advection_coefficients.h2d.seconds
            + (0.0 if v841_context is None else v841_context.h2d.seconds),
        )
        self.cache = KernelCache() if kernel_cache is None else kernel_cache
        if v841_context is None:
            self.horizontal = CudaHorizontal(
                atmosphere.mesh,
                atmosphere.vertical.n_vert_levels,
                kernel_cache=self.cache,
            )
        else:
            self.horizontal = CudaHorizontalV841(
                atmosphere.mesh,
                atmosphere.vertical.n_vert_levels,
                v841_context,
                kernel_cache=self.cache,
            )
        self._kernels: dict[str, Any] = {}
        self._physics_driver_kernels: dict[str, Any] = {}
        self._pending_v841_physics_candidate: CudaV841PhysicsStepCandidate | None = None
        self.nlev = int(atmosphere.vertical.n_vert_levels)
        self.ncells = int(atmosphere.mesh.n_cells)
        self.nedges = int(atmosphere.mesh.n_edges)
        if v841_context is not None:
            v841_context.validate(
                n_vert_levels=self.nlev,
                n_cells=self.ncells,
                n_edges=self.nedges,
                n_vertices=int(atmosphere.mesh.n_vertices),
            )

    @classmethod
    def from_host(
        cls,
        mesh: Any,
        state: PrognosticState,
        vertical: Any,
        reference: Any,
        config: DryDycoreConfig,
        *,
        saved_diagnostics: DrySavedDiagnostics | None = None,
        terrain_metrics: Any,
        advection_coefficients: Any | None = None,
        kernel_cache: KernelCache | None = None,
        authority_ruler: CudaAuthorityRulerBinder | None = None,
        reference_wind_profiles: V841ReferenceWindProfiles | None = None,
        les_dissipation: Any | None = None,
        upload_cache: Any | None = None,
    ) -> "CudaDryDycoreDriver":
        """Upload every state/mesh/metric field once at the API boundary."""

        config.validate(**({"les_dissipation": les_dissipation} if les_dissipation is not None else {}))
        cls._validate_config(config)
        source_release = getattr(config, "source_release", "v8.2.3")
        is_v841 = source_release == "v8.4.1"
        if is_v841:
            from .config_v841 import V841DryDycoreConfig

            if not isinstance(config, V841DryDycoreConfig):
                raise ConfigurationRefusal(
                    "source_release",
                    source_release,
                    "receipt relabeling cannot select v8.4.1 CUDA numerics",
                    "V841DryDycoreConfig(source_release='v8.4.1')",
                )
            if authority_ruler is not None:
                raise ConfigurationRefusal(
                    "authority_ruler",
                    "provided",
                    "the v8.4.1 CUDA path has no released native nonzero-tracer ruler",
                    "authority_ruler=None",
                )
            if reference_wind_profiles is None:
                raise ConfigurationRefusal(
                    "u_init",
                    None,
                    "v8.4.1 unconditionally subtracts initialized reference Coriolis flow",
                    "reference_wind_profiles=V841ReferenceWindProfiles(u_init=..., v_init=...)",
                )
            nlev_host, ncells_host, nedges_host = (
                _validate_v841_host_execution(
                    mesh,
                    state,
                    vertical,
                    reference,
                    saved_diagnostics,
                    terrain_metrics,
                )
            )
            if np.dtype(state.rho.dtype) != np.dtype(np.float32):
                raise TypeError("the v8.4.1 CUDA mirror requires float32 state")
            reference_wind_profiles.validate(nlev_host, np.dtype(np.float32))
            offcentering = build_v841_acoustic_offcentering(
                np.asarray(vertical.rdzw),
                minimum=config.config_epssm_minimum,
                maximum=config.config_epssm_maximum,
                transition_bottom_z=config.config_epssm_transition_bottom_z,
                transition_top_z=config.config_epssm_transition_top_z,
            )
        else:
            if reference_wind_profiles is not None:
                raise ConfigurationRefusal(
                    "u_init",
                    "provided",
                    "v8.2.3 CUDA does not consume v8.4.1 reference profiles",
                    "omit reference_wind_profiles",
                )
            offcentering = None
        if authority_ruler is not None and not isinstance(
            authority_ruler, CudaAuthorityRulerBinder
        ):
            raise TypeError(
                "authority_ruler must be a validated CudaAuthorityRulerBinder"
            )
        coefficients = advection_coefficients
        if coefficients is None:
            coefficients = build_advection_coefficients(
                mesh,
                config_scalar_adv_order=config.config_scalar_adv_order,
                n_vert_levels=state.rho.shape[0],
                source_order_v841=is_v841,
            )
        if is_v841:
            _validate_v841_host_coefficients(
                coefficients,
                n_cells=ncells_host,
                n_edges=nedges_host,
            )
        # The initialized file's dss is only a placeholder.  Core always
        # rebuilds it from config_xnutr/config_zd; doing this even for xnutr=0
        # prevents an input file from silently activating upper damping.
        selected_dss = (
            build_v841_vertical_velocity_damping(
                np.asarray(vertical.zgrid),
                xnutr=config.config_xnutr,
                damping_start_z=config.config_zd,
            )
            if is_v841
            else _frozen_vertical_damping(mesh, vertical, config)
        )
        selected_vertical = replace(
            vertical,
            dss=selected_dss,
            **(
                {
                    name: float(np.float32(getattr(vertical, name)))
                    for name in ("cf1", "cf2", "cf3")
                }
                if is_v841
                else {}
            ),
        )
        authority_initial_fingerprint = None
        if authority_ruler is not None:
            authority_saved = saved_diagnostics
            if authority_saved is None:
                authority_saved = DryDycoreDriver(
                    mesh,
                    selected_vertical,
                    reference,
                    config,
                    advection_coefficients=coefficients,
                    terrain_metrics=terrain_metrics,
                    reference_wind_profiles=reference_wind_profiles,
                )._rebuild_saved_diagnostics(state)
            authority_initial_fingerprint = cuda_authority_initial_fingerprint(
                state,
                authority_saved,
                mesh=mesh,
                vertical=vertical,
                reference=reference,
                terrain_metrics=terrain_metrics,
                config=config,
                advection_coefficients=coefficients,
            )
            authority_ruler.validate(
                canonical_sha256(cuda_configuration_payload(config)),
                authority_initial_fingerprint,
            )
        v841_context = None
        if is_v841:
            assert offcentering is not None
            assert reference_wind_profiles is not None
            v841_context = CudaV841Context.from_host(
                mesh,
                offcentering,
                reference_wind_profiles,
                n_vert_levels=state.rho.shape[0],
                dtype=np.float32,
            )
        atmosphere = DeviceAtmosphere.from_host(
            mesh,
            state,
            selected_vertical,
            reference,
            saved_diagnostics,
            terrain_metrics,
            dtype=np.float32,
            index_dtype=np.int32,
            **({'upload_cache': upload_cache} if upload_cache is not None else {}),
        )
        device_coefficients = CudaAdvectionCoefficients.from_host(coefficients)
        diagnostics_source = (
            CUDA_T0_REBUILT_DIAGNOSTICS
            if saved_diagnostics is None
            else CUDA_T0_EXACT_SIDECAR
        )
        driver = cls(
            atmosphere,
            config,
            device_coefficients,
            kernel_cache=kernel_cache,
            t0_diagnostics_source=diagnostics_source,
            authority_ruler=authority_ruler,
            v841_context=v841_context,
            les_dissipation=les_dissipation,
            _authority_initial_fingerprint=authority_initial_fingerprint,
        )
        if driver.mixing_config_v841 is not None:
            from .mixing_v841 import initialize_deformation_weights_v841

            weights = initialize_deformation_weights_v841(
                mesh, dtype=np.float32
            )
            driver.deformation_weights_receipt_v841 = (
                driver.horizontal.attach_deformation_weights_v841(weights)
            )
        if saved_diagnostics is None:
            driver.atmosphere.saved = driver._rebuild_saved_diagnostics()
        if authority_ruler is not None:
            driver._arm_authority_snapshot()
        return driver

    def _arm_authority_snapshot(self) -> None:
        """Isolate every linked execution input in a one-use D2D snapshot."""

        if self.authority_ruler is None:
            raise ValueError("cannot arm an authority snapshot without a ruler")
        if self._authority_snapshot_pending:
            raise ValueError("authority snapshot is already armed")
        cp = self.cp
        atmosphere = self.atmosphere
        private_atmosphere = DeviceAtmosphere(
            mesh=_copy_device_dataclass(atmosphere.mesh, cp),
            state=_copy_device_dataclass(atmosphere.state, cp),
            vertical=_copy_device_dataclass(atmosphere.vertical, cp),
            reference=_copy_device_dataclass(atmosphere.reference, cp),
            saved=_copy_device_dataclass(atmosphere.saved, cp),
            terrain=(
                None
                if atmosphere.terrain is None
                else _copy_device_dataclass(atmosphere.terrain, cp)
            ),
            h2d=atmosphere.h2d,
        )
        private_coefficients = _copy_device_dataclass(self.coefficients, cp)
        private_atmosphere.validate()
        private_coefficients.validate()
        pending_snapshot = [(private_atmosphere, private_coefficients)]

        def consume_authority_snapshot() -> (
            tuple[DeviceAtmosphere, CudaAdvectionCoefficients] | None
        ):
            return pending_snapshot.pop() if pending_snapshot else None

        self._consume_authority_snapshot = consume_authority_snapshot
        self._authority_snapshot_pending = True

    def _rebuild_saved_diagnostics(
        self, atmosphere: DeviceAtmosphere | None = None
    ) -> DeviceSavedDiagnostics:
        """Recover a resident sidecar without publishing the candidate state."""

        selected = self.atmosphere if atmosphere is None else atmosphere
        # Untimed: one launch per kernel, no event and no synchronize.  The
        # timed path launched every recovery kernel twice with two event
        # syncs, on the commit's rebuild as on every RK stage.
        recovered = recover_state(
            selected,
            cache=self.cache,
            warmup=0,
            timing_repeats=0,
        )
        result = DeviceSavedDiagnostics(
            theta_m=recovered.theta_m,
            exner=recovered.exner,
            density_perturbation=recovered.density_perturbation,
            rho_theta_perturbation=recovered.rho_theta_perturbation,
            pressure_perturbation=recovered.pressure_perturbation,
            normal_velocity=recovered.normal_velocity,
            vertical_velocity=recovered.vertical_velocity,
            dtype=np.dtype(np.float32),
            h2d=_zero_transfer(),
        )
        result.validate(
            n_vert_levels=self.nlev,
            n_cells=self.ncells,
            n_edges=self.nedges,
        )
        return result

    @staticmethod
    def _validate_config(config: DryDycoreConfig) -> None:
        source_release = getattr(config, "source_release", "v8.2.3")
        if source_release not in ("v8.2.3", "v8.4.1"):
            raise ConfigurationRefusal(
                "source_release",
                source_release,
                "the CUDA kernels are source-pinned only for v8.2.3 and v8.4.1",
                "source_release='v8.2.3' or V841DryDycoreConfig",
            )
        if source_release == "v8.4.1":
            from .config_v841 import V841DryDycoreConfig

            if not isinstance(config, V841DryDycoreConfig):
                raise ConfigurationRefusal(
                    "source_release",
                    source_release,
                    "receipt relabeling cannot select v8.4.1 CUDA numerics",
                    "V841DryDycoreConfig(source_release='v8.4.1')",
                )
            if config.config_dynamics_split_steps != 3:
                # A missing code path, not missing evidence: the v8.4.1 CUDA
                # dynamics subcycle is written for three subcycles per scalar
                # transport step and stops at its first call with "v8.4.1
                # CUDA subcycle requires the outer split-three schedule".
                raise ConfigurationRefusal(
                    "config_dynamics_split_steps",
                    config.config_dynamics_split_steps,
                    "the v8.4.1 CUDA dynamics subcycle is implemented for three "
                    "subcycles per transport step only (MPAS-A's default, "
                    "Registry.xml:109) and would stop at its first call",
                    "config_dynamics_split_steps=3",
                )
            if not config.config_split_dynamics_transport:
                raise ConfigurationRefusal(
                    "config_split_dynamics_transport",
                    config.config_split_dynamics_transport,
                    "native split-three performs one scalar RK after flux averaging",
                    "config_split_dynamics_transport=True",
                )
            # RETIRED 2026-10-10 (owner ruling): three refusals that named an evidence inventory, not a
            # breakage.  config_scalar_advection=False skips scalar transport
            # (advance_scalar_transport_cuda_v841 returns the stage
            # unchanged, as native does); config_positive_definite=True runs
            # the monotonic FCT branch, which is exactly what native v8.4.1
            # does with it (mpas_atm_dynamics_checks warns that the option
            # "will enable monotonic advection"); and config_apvm_upwinding is
            # a coefficient the APVM kernels read.  What stays refused is a
            # value outside the Registry's own domain.
            if (
                not np.isfinite(config.config_apvm_upwinding)
                or config.config_apvm_upwinding < 0.0
            ):
                raise ConfigurationRefusal(
                    "config_apvm_upwinding",
                    config.config_apvm_upwinding,
                    "the APVM upwinding weight is a finite non-negative "
                    "fraction (MPAS-A Registry.xml:269, default 0.5)",
                    "finite config_apvm_upwinding>=0",
                )
            # The v8.4.1 deformation-coefficient 2-D Smagorinsky branch IS
            # ported: kdiff mirrors smagorinsky_2d
            # (mpas_atm_dissipation_models.F:119-204) with the deformation
            # weights of atm_initialize_deformation_weights
            # (mpas_atm_core.F:1620-1850), and the RK1 saved-Euler mixing of
            # u/w/theta runs once per dynamics subcycle exactly as native's
            # rk_step==1 branch (mpas_atm_time_integration.F:6329-6410,
            # 6585-6598, 6725-6738, 6885-6901).
            # The v8.4.1 CUDA divergence-damping branch IS ported: it runs in
            # _step_device_v841's acoustic sub-step loop at native's position
            # (mpas_atm_time_integration.F:2401), through the same
            # divergence_damping_f32 kernel the v8.2.3 lane uses.  Refusing it
            # here silently ran the lane at coefficient zero while native runs
            # the term unconditionally at config_smdiv=0.1.
        elif config.config_dynamics_split_steps != 1:
            raise ConfigurationRefusal(
                "config_dynamics_split_steps",
                config.config_dynamics_split_steps,
                "the frozen v8.2.3 CUDA lane retains its single dynamics step",
                "config_dynamics_split_steps=1",
            )
        if config.config_horiz_mixing not in (
            "off",
            "2d_fixed",
            "2d_smagorinsky",
        ):
            raise ConfigurationRefusal(
                "config_horiz_mixing",
                config.config_horiz_mixing,
                "the CUDA step admits the frozen dry horizontal branches",
                "config_horiz_mixing='2d_smagorinsky' or a supported no-mixing value",
            )
        for knob in (
            "config_h_theta_eddy_visc2",
            "config_v_theta_eddy_visc2",
            "config_h_mom_eddy_visc2",
            "config_v_mom_eddy_visc2",
            "config_h_theta_eddy_visc4",
            "config_h_mom_eddy_visc4",
        ):
            if float(getattr(config, knob)) != 0.0:
                raise ConfigurationRefusal(
                    knob,
                    getattr(config, knob),
                    "the CUDA whole step has no saved vertical/fixed Euler branch",
                    f"{knob}=0.0",
                )
        if config.config_scalar_adv_order != 3:
            raise ConfigurationRefusal(
                "config_scalar_adv_order",
                config.config_scalar_adv_order,
                "CUDA transport currently admits order three",
                "config_scalar_adv_order=3",
            )
        if config.config_scalar_vadv_order != 3:
            raise ConfigurationRefusal(
                "config_scalar_vadv_order",
                config.config_scalar_vadv_order,
                "CUDA transport currently admits order three",
                "config_scalar_vadv_order=3",
            )

    def _kernel(self, name: str) -> Any:
        result = self._kernels.get(name)
        if result is None:
            result = self.cache.raw_kernel(
                name,
                _CUDA_SOURCE,
                module_key="hexcore.cuda_driver",
            )
            self._kernels[name] = result
        return result

    def _launch(self, name: str, count: int, args: tuple[Any, ...]) -> None:
        threads = 128
        self._kernel(name)(((count + threads - 1) // threads,), (threads,), args)

    def _physics_driver_kernel(self, name: str) -> Any:
        if name not in (
            "moist_cell_coefficients_v841_f32",
            "moist_edge_coefficients_v841_f32",
            "moist_dpdz_v841_f32",
            "euler_w_moist_v841_f32",
            "modified_theta_dynamics_rate_v841_f32",
            "add_lagged_diabatic_theta_v842_f32",
            "zero_speczone_lagged_diabatic_theta_v842_f32",
            "gf_dry_theta_dynamics_rate_v841_f32",
        ):
            raise KeyError(name)
        result = self._physics_driver_kernels.get(name)
        if result is None:
            result = self.cache.raw_kernel(
                name,
                CUDA_V841_PHYSICS_DRIVER_SOURCE,
                module_key="hexcore.cuda_driver.physics_v841",
            )
            self._physics_driver_kernels[name] = result
        return result

    def _launch_physics_driver(
        self, name: str, count: int, args: tuple[Any, ...]
    ) -> None:
        threads = 128
        self._physics_driver_kernel(name)(
            ((count + threads - 1) // threads,), (threads,), args
        )

    def _zero_speczone_lagged_heating(self, diabatic_theta: Any) -> None:
        """SG2: ``rt_diabatic_tend(k,iCell) = 0`` in the specified zone.

        Native ``atm_bdy_adjust_dynamics_speczone_tend``
        (mpas_atm_time_integration.F:7938-7949) writes the zero into the
        shared tend pool, so the carry stays zero there for the rest of the
        step and the final recovery removes nothing in those cells.  This
        writes the same zero into the carry plane the dycore was handed (the
        backend's committed plane, which the next microphysics call replaces
        whole, as native's microphysics_to_MPAS does).  THE BREAKAGE: without
        it the stage-3 recovery subtracted dt*rho*carry in specified-zone
        cells whose tendency had been replaced by the driving tendency, and
        exner/pressure_p formed from that reduced rtheta reached WSM6 in the
        two outermost rows (verifier, correct/storm-growth, 2026-10-13).
        """

        regional = self.regional_v841
        if regional is None:
            return
        expected = (self.nlev, self.ncells)
        if tuple(diabatic_theta.shape) != expected or diabatic_theta.dtype != self.cp.float32:
            raise ValueError(
                f"lagged heating carry must be float32 {expected}; got "
                f"{diabatic_theta.dtype} {tuple(diabatic_theta.shape)}")
        masks = regional.masks
        nspec = int(masks.n_spec_cells)
        if nspec == 0:
            return
        self._launch_physics_driver(
            "zero_speczone_lagged_diabatic_theta_v842_f32",
            self.nlev * nspec,
            (np.int32(self.nlev), np.int32(self.ncells), np.int32(nspec),
             masks.spec_cells, diabatic_theta),
        )

    def _compute_moist_dynamics_coefficients_v841(
        self,
        scalars: Any,
        *,
        scalar_names: tuple[str, ...] | None,
        validation_flag: Any,
        n_mass_loading: int | None = None,
    ) -> _CudaV841MoistDynamicsCoefficients:
        """Reproduce atm_compute_moist_coefficients from raw start-state q.

        ``scalar_names`` names a microphysics block; the species row says how
        many leading species load the air.  The physics-off path carries no
        scheme row and passes ``n_mass_loading`` instead (every species it can
        carry is in native's moist group; see _step_device_v841).
        """

        cp = self.cp
        # The species row replaces the WSM6 order assertion that stood here.
        # It answers the question the assertion was standing in for -- how
        # many LEADING species load the air -- for any declared scheme, and
        # it refuses a block that matches no row at all (woof.hex.species_row).
        if (scalar_names is None) == (n_mass_loading is None):
            raise ValueError("pass exactly one of scalar_names or n_mass_loading")
        if scalar_names is not None:
            n_species = len(tuple(scalar_names))
            n_mass_loading = species_row_for_names(scalar_names).n_mass_loading
        else:
            n_species = int(scalars.shape[0])
            if not 1 <= int(n_mass_loading) <= n_species:
                raise ValueError("n_mass_loading must name a leading run of the scalar block")
        expected = (n_species, self.nlev, self.ncells)
        if not isinstance(scalars, cp.ndarray):
            raise TypeError("moist dynamics scalars must remain a resident cupy.ndarray")
        if scalars.dtype != cp.dtype(cp.float32) or tuple(scalars.shape) != expected:
            raise TypeError(f"moist dynamics scalars must be FP32 with shape {expected}")
        if not scalars.flags.c_contiguous:
            raise ValueError("moist dynamics scalars must be C-contiguous")
        if (
            not isinstance(validation_flag, cp.ndarray)
            or validation_flag.dtype != cp.dtype(cp.int32)
            or tuple(validation_flag.shape) != (1,)
            or not validation_flag.flags.c_contiguous
        ):
            raise TypeError("moist dynamics validation flag must be resident int32[1]")

        qtot = cp.empty((self.nlev, self.ncells), dtype=cp.float32)
        # The native routine never assigns the lowest-interface cqw row.  Keep
        # this allocation uninitialized and never inspect/read row zero.
        cqw = cp.empty_like(qtot)
        cqu = cp.empty((self.nlev, self.nedges), dtype=cp.float32)
        self._launch_physics_driver(
            "moist_cell_coefficients_v841_f32",
            self.ncells,
            (
                np.int32(self.nlev),
                np.int32(self.ncells),
                np.int32(n_mass_loading),
                scalars,
                qtot,
                cqw,
                validation_flag,
            ),
        )
        self._launch_physics_driver(
            "moist_edge_coefficients_v841_f32",
            self.nedges,
            (
                np.int32(self.nlev),
                np.int32(self.ncells),
                np.int32(self.nedges),
                np.int32(n_mass_loading),
                self.atmosphere.mesh.cells_on_edge,
                scalars,
                cqu,
                validation_flag,
            ),
        )
        result = _CudaV841MoistDynamicsCoefficients(qtot=qtot, cqw=cqw, cqu=cqu)
        result.validate(
            cp=cp,
            n_vert_levels=self.nlev,
            n_cells=self.ncells,
            n_edges=self.nedges,
        )
        return result

    def _moist_dpdz_v841(
        self, density_perturbation: Any, qtot: Any, validation_flag: Any
    ) -> Any:
        result = self.cp.empty_like(density_perturbation)
        self._launch_physics_driver(
            "moist_dpdz_v841_f32",
            self.ncells,
            (
                np.int32(self.nlev),
                np.int32(self.ncells),
                np.float32(9.80616),
                self.atmosphere.reference.rho_base,
                density_perturbation,
                qtot,
                result,
                validation_flag,
            ),
        )
        return result

    def _moist_euler_w_v841(
        self,
        pressure_perturbation: Any,
        dpdz: Any,
        cqw: Any,
        validation_flag: Any,
        mixing_seed: Any | None = None,
    ) -> Any:
        result = self.cp.empty((self.nlev + 1, self.ncells), dtype=self.cp.float32)
        self._launch_physics_driver(
            "euler_w_moist_v841_f32",
            self.ncells,
            (
                np.int32(self.nlev),
                np.int32(self.ncells),
                pressure_perturbation,
                dpdz,
                cqw,
                self.atmosphere.vertical.rdzu,
                self.atmosphere.vertical.fzm,
                self.atmosphere.vertical.fzp,
                np.int32(mixing_seed is not None),
                dpdz if mixing_seed is None else mixing_seed,
                result,
                validation_flag,
            ),
        )
        return result

    def _add(self, target: Any, increment: Any) -> None:
        nlevels, nowners = _level_owner_shape(target, "target")
        if tuple(increment.shape) != tuple(target.shape):
            raise ValueError("increment shape differs from target")
        self._launch(
            "add_inplace_f32",
            nlevels * nowners,
            (np.int32(nlevels), np.int32(nowners), increment, target),
        )

    def _scale(self, source: Any, scale: float) -> Any:
        nlevels, nowners = _level_owner_shape(source, "source")
        result = self.cp.empty_like(source)
        self._launch(
            "scale_f32",
            nlevels * nowners,
            (np.int32(nlevels), np.int32(nowners), np.float32(scale), source, result),
        )
        return result

    def _vertical_coefficients(
        self,
        state: DevicePrognosticState,
        dts: float,
        theta_m: Any,
        rtheta_p: Any,
        *,
        exner: Any | None = None,
        validation_flag: Any | None = None,
        cqw: Any | None = None,
        qtot: Any | None = None,
    ) -> Any:
        cp = self.cp
        if (cqw is None) != (qtot is None):
            raise ValueError("cqw and qtot must select dry or moist coefficients together")
        selected_exner = self.atmosphere.saved.exner if exner is None else exner
        if self.v841_context is not None:
            return compute_vertical_implicit_coefficients_cuda_v841(
                dts=dts,
                context=self.v841_context,
                zz=self.atmosphere.vertical.zz,
                cqw=cp.ones_like(state.rho) if cqw is None else cqw,
                exner=selected_exner,
                theta=theta_m,
                rho_base=self.atmosphere.reference.rho_base,
                rho_theta_base=self.atmosphere.reference.rho_theta_base,
                exner_base=self.atmosphere.reference.exner_base,
                rho_theta_perturbation=rtheta_p,
                qtot=cp.zeros_like(state.rho) if qtot is None else qtot,
                rdzw=self.atmosphere.vertical.rdzw,
                fzm=self.atmosphere.vertical.fzm,
                fzp=self.atmosphere.vertical.fzp,
                rdzu=self.atmosphere.vertical.rdzu,
                validation_flag=validation_flag,
                kernel_cache=self.cache,
            )
        return compute_vertical_implicit_coefficients_cuda(
            dts=dts,
            epssm=self.config.config_epssm,
            zz=self.atmosphere.vertical.zz,
            cqw=cp.ones_like(state.rho) if cqw is None else cqw,
            pressure=selected_exner,
            theta=theta_m,
            rho_base=self.atmosphere.reference.rho_base,
            rho_theta_base=self.atmosphere.reference.rho_theta_base,
            pressure_base=self.atmosphere.reference.exner_base,
            rho_theta_perturbation=rtheta_p,
            qtot=cp.zeros_like(state.rho) if qtot is None else qtot,
            rdzw=self.atmosphere.vertical.rdzw,
            fzm=self.atmosphere.vertical.fzm,
            fzp=self.atmosphere.vertical.fzp,
            rdzu=self.atmosphere.vertical.rdzu,
            kernel_cache=self.cache,
        )

    def _vertical_u(self, u: Any, rw: Any) -> Any:
        cp = self.cp
        flux = cp.empty((self.nlev + 1, self.nedges), dtype=cp.float32)
        result = cp.empty((self.nlev, self.nedges), dtype=cp.float32)
        self._launch(
            "vertical_u_flux_f32",
            self.nedges * (self.nlev + 1),
            (
                np.int32(self.nlev),
                np.int32(self.ncells),
                np.int32(self.nedges),
                u,
                rw,
                self.atmosphere.mesh.cells_on_edge,
                self.atmosphere.vertical.fzm,
                self.atmosphere.vertical.fzp,
                flux,
            ),
        )
        self._launch(
            "vertical_u_finish_f32",
            self.nedges * self.nlev,
            (
                np.int32(self.nlev),
                np.int32(self.nedges),
                self.atmosphere.vertical.rdzw,
                flux,
                result,
            ),
        )
        return result

    def _vector_momentum(
        self,
        u: Any,
        rho_edge: Any,
        pv_edge: Any,
        kinetic: Any,
        mass_divergence: Any,
        initial_tendency: Any | None = None,
    ) -> Any:
        if self.v841_context is not None:
            return vector_momentum_tendency_cuda_v841(
                self.atmosphere.mesh,
                self.v841_context,
                normal_velocity=u,
                rho_edge=rho_edge,
                pv_edge=pv_edge,
                kinetic_energy=kinetic,
                horizontal_divergence=mass_divergence,
                initial_tendency=initial_tendency,
                kernel_cache=self.cache,
            )
        result = self.cp.empty_like(u)
        mesh = self.atmosphere.mesh
        self._launch(
            "vector_momentum_f32",
            self.nedges,
            (
                np.int32(self.nlev),
                np.int32(self.ncells),
                np.int32(self.nedges),
                np.int32(mesh.max_edges2),
                u,
                rho_edge,
                pv_edge,
                kinetic,
                mass_divergence,
                mesh.cells_on_edge,
                mesh.edges_on_edge,
                mesh.n_edges_on_edge,
                mesh.weights_on_edge,
                mesh.dc_edge,
                result,
            ),
        )
        if initial_tendency is not None:
            self._add(result, initial_tendency)
        return result

    def _theta_tendency(
        self,
        state: DevicePrognosticState,
        theta: Any,
        saved_state: DevicePrognosticState,
        theta_saved: Any,
        rk_step: int,
    ) -> Any:
        cp = self.cp
        coefficient = self.coefficients
        width = int(coefficient.adv_coefs.shape[1])
        edge_flux = cp.empty_like(state.rho_u)
        vertical_flux = cp.empty_like(state.rho_w)
        result = cp.empty_like(state.rho)
        # v8.4.1 accumulates the RK correction in a separate cell loop
        # (native F:6814-6874; dycore-match 8322a3f).  A full-step LTS class
        # driver keeps it inside the edge flux instead: its interface capture
        # and the matched heat flux read the third-stage edge flux as the
        # whole heat moved through an edge (rk_step 3 in argument 4), and with
        # the correction moved to the cell loop every LTS step stopped with
        # "the third-stage theta edge flux was not observed" (golden 0.4.0
        # integration, G3, 2026-10-12).  LTS is a different time scheme by
        # design and is not part of the byte-identity claim.
        separate_correction = self.v841_context is not None and not getattr(
            self, "lts_theta_correction_in_edge_flux", False)
        theta_arguments = (
            np.int32(self.nlev),
            np.int32(self.ncells),
            np.int32(self.nedges),
            np.int32(width),
            np.int32(1 if separate_correction else rk_step),
            np.float32(self.config.config_coef_3rd_order),
            theta,
            theta_saved,
            state.rho_u,
            saved_state.rho_u,
            self.atmosphere.mesh.dv_edge,
            self.atmosphere.mesh.cells_on_edge,
            coefficient.adv_coefs,
            coefficient.adv_coefs_3rd,
            coefficient.n_adv_cells_for_edge,
            coefficient.adv_cells_for_edge,
        )
        from . import cuda_edge_hoist
        if cuda_edge_hoist.enabled(self.nedges):
            lpt = cuda_edge_hoist.levels_per_thread(
                "theta_edge_flux_hoist_f32", self.nedges, self.nlev, lambda value: self._launch(
                    "theta_edge_flux_hoist_f32", self.nedges * cuda_edge_hoist.chunks(self.nlev, value),
                    (*theta_arguments, edge_flux, np.int32(value))), cp)
            self._launch("theta_edge_flux_hoist_f32", self.nedges * cuda_edge_hoist.chunks(self.nlev, lpt),
                         (*theta_arguments, edge_flux, np.int32(lpt)))
            if cuda_edge_hoist.check_wanted("theta_edge_flux_hoist_f32", cp):
                element = cp.empty_like(edge_flux)
                self._launch("theta_edge_flux_f32", self.nedges * self.nlev, (*theta_arguments, element))
                cuda_edge_hoist.compare("theta_edge_flux_hoist_f32", edge_flux, element, cp)
        else:
            self._launch("theta_edge_flux_f32", self.nedges * self.nlev, (*theta_arguments, edge_flux))
        self._launch(
            "theta_vertical_flux_f32",
            self.ncells * (self.nlev + 1),
            (
                np.int32(self.nlev),
                np.int32(self.ncells),
                np.float32(self.config.config_coef_3rd_order),
                theta,
                theta_saved,
                state.rho_w,
                saved_state.rho_w,
                self.atmosphere.vertical.fzm,
                self.atmosphere.vertical.fzp,
                vertical_flux,
            ),
        )
        if self.v841_context is not None:
            return theta_finish_cuda_v841(
                self.atmosphere.mesh,
                self.v841_context,
                edge_flux=edge_flux,
                vertical_flux=vertical_flux,
                rdzw=self.atmosphere.vertical.rdzw,
                n_vert_levels=self.nlev,
                correction=(state.rho_u, saved_state.rho_u, theta_saved)
                if rk_step > 1 and separate_correction else None,
                kernel_cache=self.cache,
            )
        self._launch(
            "theta_finish_f32",
            self.ncells,
            (
                np.int32(self.nlev),
                np.int32(self.ncells),
                np.int32(self.nedges),
                np.int32(self.atmosphere.mesh.max_edges),
                self.atmosphere.mesh.n_edges_on_cell,
                self.atmosphere.mesh.edges_on_cell,
                self.atmosphere.mesh.edge_sign_on_cell,
                self.atmosphere.mesh.area_cell,
                self.atmosphere.vertical.rdzw,
                edge_flux,
                vertical_flux,
                result,
            ),
        )
        return result

    def _w_tendency(self, state: DevicePrognosticState, w: Any) -> Any:
        cp = self.cp
        coefficient = self.coefficients
        width = int(coefficient.adv_coefs.shape[1])
        edge_flux = cp.empty((self.nlev + 1, self.nedges), dtype=cp.float32)
        vertical_flux = cp.empty_like(state.rho_w)
        result = cp.empty_like(state.rho_w)
        w_arguments = (
            np.int32(self.nlev),
            np.int32(self.ncells),
            np.int32(self.nedges),
            np.int32(width),
            np.float32(self.config.config_coef_3rd_order),
            w,
            state.rho_u,
            self.atmosphere.vertical.fzm,
            self.atmosphere.vertical.fzp,
            coefficient.adv_coefs,
            coefficient.adv_coefs_3rd,
            coefficient.n_adv_cells_for_edge,
            coefficient.adv_cells_for_edge,
        )
        from . import cuda_edge_hoist
        if cuda_edge_hoist.enabled(self.nedges):
            lpt = cuda_edge_hoist.levels_per_thread(
                "w_edge_flux_hoist_f32", self.nedges, self.nlev + 1, lambda value: self._launch(
                    "w_edge_flux_hoist_f32", self.nedges * cuda_edge_hoist.chunks(self.nlev + 1, value),
                    (*w_arguments, edge_flux, np.int32(value))), cp)
            self._launch("w_edge_flux_hoist_f32", self.nedges * cuda_edge_hoist.chunks(self.nlev + 1, lpt),
                         (*w_arguments, edge_flux, np.int32(lpt)))
            if cuda_edge_hoist.check_wanted("w_edge_flux_hoist_f32", cp):
                element = cp.empty_like(edge_flux)
                self._launch("w_edge_flux_f32", self.nedges * (self.nlev + 1), (*w_arguments, element))
                cuda_edge_hoist.compare("w_edge_flux_hoist_f32", edge_flux, element, cp)
        else:
            self._launch("w_edge_flux_f32", self.nedges * (self.nlev + 1), (*w_arguments, edge_flux))
        self._launch(
            "w_vertical_flux_f32",
            self.ncells * (self.nlev + 1),
            (np.int32(self.nlev), np.int32(self.ncells), w, state.rho_w, vertical_flux),
        )
        if self.v841_context is not None:
            return w_finish_cuda_v841(
                self.atmosphere.mesh,
                self.v841_context,
                edge_flux=edge_flux,
                vertical_flux=vertical_flux,
                rdzu=self.atmosphere.vertical.rdzu,
                n_vert_levels=self.nlev,
                kernel_cache=self.cache,
            )
        self._launch(
            "w_finish_f32",
            self.ncells,
            (
                np.int32(self.nlev),
                np.int32(self.ncells),
                np.int32(self.nedges),
                np.int32(self.atmosphere.mesh.max_edges),
                self.atmosphere.mesh.n_edges_on_cell,
                self.atmosphere.mesh.edges_on_cell,
                self.atmosphere.mesh.edge_sign_on_cell,
                self.atmosphere.mesh.area_cell,
                self.atmosphere.vertical.rdzu,
                edge_flux,
                vertical_flux,
                result,
            ),
        )
        return result

    def _device_state(
        self, rho: Any, rtheta: Any, ru: Any, rw: Any, scalars: Any
    ) -> DevicePrognosticState:
        return DevicePrognosticState(
            rho,
            rtheta,
            ru,
            rw,
            scalars,
            self.atmosphere.state.time_seconds,
            np.dtype(np.float32),
            _zero_transfer(),
        )

    def _device_saved(
        self,
        theta: Any,
        exner: Any,
        rho_p: Any,
        rtheta_p: Any,
        pressure_p: Any,
        u: Any,
        w: Any,
    ) -> DeviceSavedDiagnostics:
        return DeviceSavedDiagnostics(
            theta,
            exner,
            rho_p,
            rtheta_p,
            pressure_p,
            u,
            w,
            np.dtype(np.float32),
            _zero_transfer(),
        )

    def _recover_candidate(
        self,
        saved_state: DevicePrognosticState,
        saved_diag: DeviceSavedDiagnostics,
        acoustic: CudaAcousticState,
        stage: int,
        acoustic_steps: int,
        scalars: Any,
        *, diabatic_theta: Any | None = None, large_timestep: float = 0.0,
    ) -> tuple[DevicePrognosticState, DeviceSavedDiagnostics, Any, Any]:
        cp = self.cp
        rho = cp.empty_like(saved_state.rho)
        rtheta = cp.empty_like(saved_state.rho_theta)
        rho_p = cp.empty_like(saved_state.rho)
        rtheta_p = cp.empty_like(saved_state.rho_theta)
        theta = cp.empty_like(saved_state.rho)
        exner = cp.empty_like(saved_diag.exner)
        pressure_p = cp.empty_like(saved_diag.pressure_perturbation)
        self._launch(
            "recover_cells_f32",
            self.ncells * self.nlev,
            (
                np.int32(self.nlev),
                np.int32(self.ncells),
                np.int32(stage),
                np.int32(self.source_release == "v8.4.1"),
                np.float32(287.0),
                np.float32(1004.5),
                np.float32(100_000.0),
                saved_diag.density_perturbation,
                saved_diag.rho_theta_perturbation,
                acoustic.rho_pp,
                acoustic.rtheta_pp,
                np.int32(diabatic_theta is not None), np.float32(large_timestep),
                saved_diag.density_perturbation if diabatic_theta is None else diabatic_theta,
                self.atmosphere.reference.rho_base,
                self.atmosphere.reference.rho_theta_base,
                self.atmosphere.reference.exner_base,
                self.atmosphere.vertical.zz,
                saved_diag.exner,
                saved_diag.pressure_perturbation,
                rho,
                rtheta,
                rho_p,
                rtheta_p,
                theta,
                exner,
                pressure_p,
            ),
        )
        ru = cp.empty_like(saved_state.rho_u)
        flux_u = cp.empty_like(saved_state.rho_u)
        self._launch(
            "recover_edges_f32",
            self.nedges * self.nlev,
            (
                np.int32(self.nlev),
                np.int32(self.nedges),
                np.int32(acoustic_steps),
                saved_state.rho_u,
                acoustic.ru_p,
                acoustic.ru_avg,
                ru,
                flux_u,
            ),
        )
        rw = cp.empty_like(saved_state.rho_w)
        flux_w = cp.empty_like(saved_state.rho_w)
        self._launch(
            "recover_interfaces_f32",
            self.ncells * (self.nlev + 1),
            (
                np.int32(self.nlev),
                np.int32(self.ncells),
                np.int32(acoustic_steps),
                saved_state.rho_w,
                acoustic.rw_p,
                acoustic.ww_avg,
                rw,
                flux_w,
            ),
        )
        if self.v841_context is not None:
            enforce_rw_endpoints_cuda_v841(rw, kernel_cache=self.cache)
        regional_u_overwrite = None
        if self.regional_v841 is not None:
            # driver.py:2228-2265 and 2390-2394: every regional recover runs
            # in native's padded memory model with rho_zz's garbage cell at
            # 1.0 (F:4385-4392), so the shared recover_edges kernel divides a
            # one-cell ring-7 edge by rho(present)+1.0 exactly as
            # regional_normal_velocity does on the host.
            self.regional_v841.bind_state_rho(rho)
            # driver.py:2935-2937 -- the ru half of atm_srk3:2442-2485,
            # before the candidate is validated and before u is recovered.
            regional_u_overwrite = self.regional_v841.overwrite_speczone_rho_u(
                ru, stage
            )
        # The candidate carries the caller's scalar block by reference.  Nothing
        # between here and the transport stage writes through it: the acoustic
        # solver, the tendency helpers and the recovery kernels all read the
        # scalars and allocate their own outputs, and both callers REBIND
        # ``.scalars`` to a transport result rather than filling it in place.
        # Copying it here allocated a whole six-species block (nScalars x nlev x
        # nCells) per RK stage, three times per dynamics subcycle, and every one
        # of those copies was dropped unmodified.
        state = self._device_state(rho, rtheta, ru, rw, scalars)
        temporary = DeviceAtmosphere(
            self.atmosphere.mesh,
            state,
            self.atmosphere.vertical,
            self.atmosphere.reference,
            saved_diag,
            self.atmosphere.terrain,
            self.atmosphere.h2d,
        )
        # Only normal_velocity and vertical_velocity are read below; theta,
        # exner, pressure, rho_p, rtheta_p and pressure_p come from
        # recover_cells_f32 above, so asking recover_state for its own copies
        # allocated and filled six cell fields per stage for the bin.
        recovered = recover_state(
            temporary,
            cache=self.cache,
            warmup=0,
            timing_repeats=0,
            include_pressure=False,
        )
        if self.regional_v841 is not None:
            # driver.py:3005-3009 -- the u half of the same native overwrite,
            # with the SAME interpolated driving values the ru half used --
            # and driver.py:2405, atm_zero_gradient_w_bdy (F:7868-7902).
            self.regional_v841.overwrite_speczone_u(
                recovered.normal_velocity, regional_u_overwrite
            )
            # Recovery skips specified-zone terrain flux. Whole-column zeroing
            # matches native at the step boundary; nearest is applied after solve.
            self.regional_v841.zero_speczone_w(recovered.vertical_velocity)
        diagnostics = self._device_saved(
            theta,
            exner,
            rho_p,
            rtheta_p,
            pressure_p,
            recovered.normal_velocity,
            recovered.vertical_velocity,
        )
        return state, diagnostics, flux_u, flux_w

    def _copy_state(
        self, source: DevicePrognosticState, *, share_scalars: bool = False
    ) -> DevicePrognosticState:
        """Private image of a prognostic state.

        ``share_scalars=True`` binds the caller's scalar block by reference
        instead of copying it.  The dynamics only ever READ the scalars -- the
        transport stage rebinds ``.scalars`` to a fresh transport output -- so
        inside one dynamics subcycle a copy is an nScalars x nlev x nCells
        allocation that is written by nobody.
        """

        cp = self.cp
        return self._device_state(
            cp.array(source.rho, copy=True),
            cp.array(source.rho_theta, copy=True),
            cp.array(source.rho_u, copy=True),
            cp.array(source.rho_w, copy=True),
            source.scalars if share_scalars else cp.array(source.scalars, copy=True),
        )

    def _copy_saved(
        self, source: DeviceSavedDiagnostics
    ) -> DeviceSavedDiagnostics:
        cp = self.cp
        return self._device_saved(
            cp.array(source.theta_m, copy=True),
            cp.array(source.exner, copy=True),
            cp.array(source.density_perturbation, copy=True),
            cp.array(source.rho_theta_perturbation, copy=True),
            cp.array(source.pressure_perturbation, copy=True),
            cp.array(source.normal_velocity, copy=True),
            cp.array(source.vertical_velocity, copy=True),
        )

    def _shares_start_levels(self) -> bool:
        """Whether this driver binds its start levels instead of copying them.

        The regional lane binds too (``regional_state_levels_enabled``).  Its
        garbage-cell discipline classifies device blocks by ADDRESS
        (bind_state_rho, _unit_pool), and a bound level shares its source's
        address: the resident ``rho_zz`` is registered at ``begin_step`` and
        every later subcycle input is a recovered candidate registered by
        ``bind_state_rho``, so the 1.0 pool cell is the same either way, and a
        scrub of a bound array writes only the garbage column, which no reader
        of the resident level takes (output reads the solved cells, the physics
        seam lends and restores its own garbage column).  Private images were
        ~10 KB per column on the limited-area route (step-start state with its
        scalar block, time level 1, and the subcycle-start images), ~6 GiB per
        card on NOAA conus3km at 4 ranks.  A multi-card rank binds only when its halo exchanger declares
        ``start_levels_read_only``: its in-step rounds (B stage entry, C
        acoustic, D transport, E limiter scales) write owner truth into the
        recovered candidate, the acoustic state, the transport stage block and
        the scale factors, never into the step-start level, and its step
        boundary round A writes the published level before the next step
        binds it.  The same-host exchanger declares it; the LTS block
        exchangers do not.  A driver with local time stepping attached
        (``local_timestep``, cuda_driver_lts) keeps the images as well: its
        rebound recovery reads the start level only, but no LTS deck has been
        run against the binding, so the binding stays off that path until one
        is.
        """

        exchanger = self.halo_exchanger_v841
        return (
            state_levels_enabled()
            and (self.regional_v841 is None or regional_state_levels_enabled())
            and getattr(self, "local_timestep", None) is None
            and (exchanger is None
                 or bool(getattr(exchanger, "start_levels_read_only", False)))
        )

    def _bind_state(
        self, source: DevicePrognosticState, *, copy_scalars: bool = False,
        copy_rho: bool = False,
    ) -> DevicePrognosticState:
        """A fresh container over ``source``'s arrays (no device copy).

        ``copy_rho`` / ``copy_scalars`` keep a private image of just that
        field, for the callers that read it after something may overwrite
        the source.
        """

        return self._device_state(
            self.cp.array(source.rho, copy=True) if copy_rho else source.rho,
            source.rho_theta,
            source.rho_u,
            source.rho_w,
            self.cp.array(source.scalars, copy=True) if copy_scalars else source.scalars,
        )

    def _bind_saved(
        self, source: DeviceSavedDiagnostics
    ) -> DeviceSavedDiagnostics:
        """A fresh container over ``source``'s arrays (no device copy)."""

        return self._device_saved(
            source.theta_m,
            source.exner,
            source.density_perturbation,
            source.rho_theta_perturbation,
            source.pressure_perturbation,
            source.normal_velocity,
            source.vertical_velocity,
        )

    def _advance_dynamics_subcycle_v841(
        self,
        state: DevicePrognosticState,
        *,
        time_level_one: DeviceSavedDiagnostics,
        initial_diagnostics: Any = None,
        schedule: RKSchedule,
        outer_dt: float,
        validation_flag: Any,
        physics_tendencies: Any | None = None,
        moist_coefficients: _CudaV841MoistDynamicsCoefficients | None = None,
        physics_stage_callback: Any | None = None,
        acoustic_output_callback: Any | None = None,
        dynamics_substep: int = 1,
        final_diagnostics: bool = True,
    ) -> _CudaDynamicsSubcycleResult:
        """Advance one native chained v8.4.1 dynamics subcycle on device.

        ``initial_diagnostics=None`` forms the substep-start solve here, from
        this substep's own inputs (``state`` and ``time_level_one``), instead
        of receiving a set the caller holds for the whole substep.
        ``final_diagnostics=False`` skips the stage-3 solve whose only reader
        was the next substep's start; the result then carries
        ``diagnostics=None``.  The caller that passes both
        (``_step_device_v841`` under ``diagnostics_lifetime_lean()``) forms
        each set from the same arrays and kernels as before, so every value
        a stage reads is the same word -- only when the set exists changes.
        """

        if self.v841_context is None:
            raise RuntimeError("v8.4.1 device context is missing")
        if (physics_tendencies is not None or physics_stage_callback is not None) and moist_coefficients is None:
            # Full physics without the moist coefficients would run the dry
            # pressure-gradient/buoyancy forms under a moist state.  The other
            # half of the old pairing law (coefficients only WITH physics) is
            # retired: native atm_srk3 (mpas_atm_time_integration.F:2125-2139)
            # forms them on every step whatever the physics suite, so the
            # physics-off global path and the regional lane both carry them.
            raise ValueError(
                "full-physics tendencies require native moist coefficients"
            )
        if moist_coefficients is not None:
            moist_coefficients.validate(
                cp=self.cp,
                n_vert_levels=self.nlev,
                n_cells=self.ncells,
                n_edges=self.nedges,
            )
        if schedule.full_timestep != outer_dt or schedule.dynamics_splits != 3:
            raise ValueError("v8.4.1 CUDA subcycle requires the outer split-three schedule")
        # Memory lane (WOOF_HEX_MEM_ACOUSTIC, default on): every working array
        # below is released at its last reader instead of at the end of the
        # stage or the subcycle.  No arithmetic launch moves or changes; the
        # moment a block returns to the pool does, and one copy launch per
        # acoustic sub-step goes (damping_reference_in_state).
        short_lifetimes = short_lifetimes_enabled()
        # The damping reference can be the acoustic state's own rtheta_pp_old
        # only where acoustic_prepare_v841 is the last writer of it before the
        # damping: the closed global sub-step on one card.  The regional,
        # fused, LTS-rebound and multi-card sub-steps keep the captured copy.
        damping_reference_in_state = (
            short_lifetimes
            and self.regional_v841 is None
            and self.halo_exchanger_v841 is None
            and advance_acoustic_step_cuda_v841
            is _acoustic_v841_module.advance_acoustic_step_cuda_v841
        )
        witness = None
        if self._shares_start_levels():
            # mem/state-levels: the subcycle-start level is bound, not copied.
            # Inside this method every read goes through saved_*/current_*
            # (``state`` and ``time_level_one`` are not touched again), and
            # no operation below writes through them -- each helper allocates
            # its own output, the acoustic solver owns CudaAcousticState, and
            # the recovery kernels read saved_* and write fresh candidates.
            # The caller reads only rho and scalars of a subcycle input after
            # the call (transport's old density), and nothing here writes
            # those.  The private images were five prognostic plus seven
            # diagnostic arrays per subcycle (3.3 KB per column; retained for
            # the graph's life when subcycle graphs are on).
            saved_state = self._bind_state(state)
            saved_diag = self._bind_saved(time_level_one)
            if (_StateLevelsWitness.enabled()
                    and getattr(self, "subcycle_graphs", None) is None):
                witness = _StateLevelsWitness(
                    self.cp, "subcycle start level",
                    rho=state.rho, rho_theta=state.rho_theta,
                    rho_u=state.rho_u, rho_w=state.rho_w, scalars=state.scalars,
                    **{f"saved.{f.name}": getattr(time_level_one, f.name)
                       for f in fields(time_level_one)},
                )
        else:
            saved_state = self._copy_state(state, share_scalars=True)
            saved_diag = self._copy_saved(time_level_one)
        # RK stage 1 begins at the substep-start state, so current_* and saved_*
        # are the same field values.  Stage 1 READS both handles and writes
        # through neither -- every tendency helper below allocates its own
        # output, the acoustic solver works on CudaAcousticState, and the loop
        # body REBINDS current_state/current_diag to the recovered candidate
        # before stage 2 reads them.  A private image here was a second copy of
        # a read-only field set: five prognostic arrays plus seven diagnostic
        # arrays, allocated once per dynamics subcycle and never written.
        current_state = saved_state
        current_diag = saved_diag
        if initial_diagnostics is None:
            # The substep-start solve, with the arguments the caller's
            # step-start solve used (outer state, time-level-one velocity,
            # rk_step=3) on substep 1, and on later substeps the same arrays
            # the previous substep's stage-3 solve read: its recovered state
            # and diagnostics ARE this substep's state and time_level_one.
            initial_diagnostics = self.horizontal.solve_diagnostics(
                state.rho,
                state.rho_u,
                dt=outer_dt,
                apvm_upwinding=self.config.config_apvm_upwinding,
                normal_velocity=time_level_one.normal_velocity,
                rk_step=3,
            )
        diagnostics = initial_diagnostics
        initial_diagnostics = None
        cached_v = diagnostics.tangential_velocity
        cp = self.cp
        final_modified_theta_dynamics_rate = None

        coefficients = self._vertical_coefficients(
            saved_state,
            schedule.stages[0].acoustic_timestep,
            saved_diag.theta_m,
            saved_diag.rho_theta_perturbation,
            exner=saved_diag.exner,
            validation_flag=validation_flag,
            cqw=None if moist_coefficients is None else moist_coefficients.cqw,
            qtot=None if moist_coefficients is None else moist_coefficients.qtot,
        )
        if moist_coefficients is None:
            cqu = cp.ones_like(saved_state.rho_u)
            dpdz = self._scale(saved_diag.density_perturbation, -9.80616)
        else:
            cqu = moist_coefficients.cqu
            dpdz = self._moist_dpdz_v841(
                saved_diag.density_perturbation,
                moist_coefficients.qtot,
                validation_flag,
            )
        if self.les_dissipation is not None:
            if physics_stage_callback is not None:
                physics_tendencies = physics_stage_callback(
                    rk_step=1, dynamics_substep=dynamics_substep,
                    state=current_state, saved=current_diag, diagnostics=diagnostics)
            if physics_tendencies is None or moist_coefficients is None:
                raise RuntimeError("LES requires scalar tendency and mass-loading carriers")
            self.les_dissipation.begin(self, saved_state, saved_diag, diagnostics,
                moist_coefficients, physics_tendencies, dynamics_substep,
                self.les_dissipation.winds)
        saved_euler_ru = self.horizontal.pressure_gradient_euler_tendency(
            saved_diag.pressure_perturbation,
            dpdz,
            cqu,
            self.atmosphere.vertical.zz,
            self.atmosphere.vertical.zxu,
        )
        tend_rho_dynamics = self.horizontal.density_tendency(
            saved_state.rho_u,
            saved_state.rho_w,
            self.atmosphere.vertical.rdzw,
        )
        tend_rho_saved = (
            tend_rho_dynamics
            if physics_tendencies is None
            else self.horizontal.density_tendency(
                saved_state.rho_u,
                saved_state.rho_w,
                self.atmosphere.vertical.rdzw,
                physics_tendency=physics_tendencies.rho,
            )
        )
        if self.regional_v841 is not None:
            # atm_compute_dyn_tend_work F:6455-6466 writes tend_rho only at
            # rk_step==1, so the pool carries stage 1's values AND every
            # regional adjustment applied to them into stages 2 and 3, where
            # the regional stages adjust the already-adjusted array again.
            # The CUDA subcycle already forms tend_rho once per subcycle, so
            # the native quirk is inherited; the private image only keeps the
            # in-place regional mutation out of tend_rho_dynamics.
            tend_rho_saved = cp.array(tend_rho_saved, copy=True)
        if moist_coefficients is None:
            saved_euler_rw = cp.empty_like(saved_state.rho_w)
            self._launch(
                "euler_w_f32",
                self.ncells * (self.nlev + 1),
                (
                    np.int32(self.nlev),
                    np.int32(self.ncells),
                    saved_diag.pressure_perturbation,
                    dpdz,
                    self.atmosphere.vertical.rdzu,
                    self.atmosphere.vertical.fzm,
                    self.atmosphere.vertical.fzp,
                    saved_euler_rw,
                ),
            )
        else:
            saved_euler_rw = self._moist_euler_w_v841(
                saved_diag.pressure_perturbation,
                dpdz,
                moist_coefficients.cqw,
                validation_flag,
            )
        saved_euler_theta = None
        if self.les_dissipation is not None:
            saved_euler_ru, mixing_w, saved_euler_theta = self.les_dissipation.tendencies(
                saved_euler_ru, physics_tendencies)
            # Native F:6784 subtracts the pressure/buoyancy term directly
            # from the LES pool, retaining its rounding and signed zero.
            saved_euler_rw = self._moist_euler_w_v841(
                saved_diag.pressure_perturbation, dpdz, moist_coefficients.cqw,
                validation_flag, mixing_seed=mixing_w)
        if short_lifetimes:
            # LES rebuilds the w tendency with its mixing seed, so dpdz must
            # survive that final pressure-term read as well.
            dpdz = None
        if self.mixing_config_v841 is not None:
            # Native computes the horizontal-mixing Euler tendencies once per
            # dynamics substep at rk_step==1 from the substep-start state
            # (mpas_atm_time_integration.F:6329,6585,6725,6885) and reuses
            # them for RK2/RK3.  ``diagnostics`` here is the substep-start
            # solve (tangential velocity, divergence, vorticity, rho_edge).
            saved_mixing = self.horizontal.compute_dry_mixing_tendencies_v841(
                saved_diag.normal_velocity,
                diagnostics.tangential_velocity,
                saved_diag.vertical_velocity,
                saved_diag.theta_m,
                diagnostics.h_edge,
                diagnostics.divergence,
                diagnostics.vorticity,
                dt=outer_dt,
                config=self.mixing_config_v841,
                cam_damping_levels=getattr(self.config,"config_number_cam_damping_levels",4),
                n_cells_solve=None if self.regional_v841 is None else self.regional_v841.n_cells_solve,
                initial_u_tendency=saved_euler_ru,
            )
            # The u filter accumulates into the pressure-gradient seed in
            # native order: (pressure + del2) - del4.  Its return is the
            # complete saved pool, so do not add the seed a second time.
            saved_euler_ru = saved_mixing.tend_u_euler
            self._add(saved_euler_rw, saved_mixing.tend_w_euler)
            saved_euler_theta = cp.array(
                saved_mixing.tend_theta_euler, copy=True
            )
            if getattr(self.config, 'config_scalar_eddy_mix', False):
                # Reuse the final subcycle's RK1 viscosity for the NOAA
                # scalar hook (44c0849 still calls this absent pool kdiff).
                self._scalar_eddy_kdiff = saved_mixing.kdiff
            del saved_mixing
            self.mixing_calls_v841 += 1
        final_flux_u = None
        final_flux_w = None
        final_stage = schedule.stages[-1]

        for stage in schedule.stages:
            # mpas_atm_time_integration.F:2140-2176: assemble sources at the
            # first callback and hold them through every RK/dynamics substep.
            # The callback retains outer time-level-one theta/scalars and
            # native halo/projection results. Density physics is zero.
            if physics_stage_callback is not None:
                physics_tendencies = physics_stage_callback(
                    rk_step=stage.stage, dynamics_substep=dynamics_substep,
                    state=current_state, saved=current_diag,
                    diagnostics=diagnostics,
                )
                if physics_tendencies is None:
                    raise RuntimeError("active MPAS RK callback returned no sources")
            if stage.stage == 2 and self.config.config_time_integration_order == 3:
                coefficients = self._vertical_coefficients(
                    current_state,
                    stage.acoustic_timestep,
                    current_diag.theta_m,
                    current_diag.rho_theta_perturbation,
                    exner=saved_diag.exner,
                    validation_flag=validation_flag,
                    cqw=None if moist_coefficients is None else moist_coefficients.cqw,
                    qtot=None if moist_coefficients is None else moist_coefficients.qtot,
                )
            mass_divergence = self.horizontal.mass_flux_divergence(
                current_state.rho_u
            )
            tend_ru = self._vertical_u(
                current_diag.normal_velocity, current_state.rho_w
            )
            tend_ru = self._vector_momentum(
                current_diag.normal_velocity,
                diagnostics.h_edge,
                diagnostics.pv_edge,
                diagnostics.kinetic_energy,
                mass_divergence,
                initial_tendency=tend_ru,
            )
            del mass_divergence
            if self.config.config_rayleigh_damp_u:
                from .cuda_upper_damping_v841 import apply_rayleigh_u_cuda
                apply_rayleigh_u_cuda(tend_ru,diagnostics.h_edge,current_diag.normal_velocity,
                    levels=getattr(self.config,"config_number_rayleigh_damp_u_levels",6),
                    days=getattr(self.config,"config_rayleigh_damp_u_timescale_days",5.0),
                    solved=None if self.regional_v841 is None else self.regional_v841.n_edges_solve,
                    cache=self.cache)
            if short_lifetimes:
                # The stage's solve diagnostics are last read by the momentum
                # tendency and the Rayleigh damping just above (h_edge,
                # pv_edge, kinetic_energy); only the tangential velocity
                # survives, as cached_v.  In RK stages 2 and 3 this frame owns
                # the set (formed at the end of the previous stage), so
                # releasing it here rather than after the acoustic state is
                # allocated takes one whole diagnostic set off the stage's
                # acoustic high-water.  (Stage 1's set is the caller's.)
                diagnostics = None
            self._add(tend_ru, saved_euler_ru)
            if physics_tendencies is not None:
                self._add(tend_ru, physics_tendencies.rho_u)
            tend_rt = self._theta_tendency(
                current_state,
                current_diag.theta_m,
                saved_state,
                saved_diag.theta_m,
                stage.stage,
            )
            if physics_tendencies is not None and stage.stage == 3:
                final_modified_theta_dynamics_rate = cp.empty_like(tend_rt)
                self._launch_physics_driver(
                    "modified_theta_dynamics_rate_v841_f32",
                    self.ncells,
                    (
                        np.int32(self.nlev),
                        np.int32(self.ncells),
                        tend_rt,
                        # Both MPAS-A v8.4.2 and NOAA 44c0849 form regional
                        # heating from the boundary-adjusted density tendency.
                        tend_rho_saved if self.regional_v841 is not None else tend_rho_dynamics,
                        current_state.rho,
                        current_diag.theta_m,
                        final_modified_theta_dynamics_rate,
                        validation_flag,
                    ),
                )
            diabatic_theta = getattr(physics_tendencies, "diabatic_theta", None)
            if diabatic_theta is not None:
                self._launch_physics_driver(
                    "add_lagged_diabatic_theta_v842_f32", self.ncells,
                    (np.int32(self.nlev), np.int32(self.ncells),
                     current_state.rho, diabatic_theta, tend_rt),
                )
            if saved_euler_theta is not None:
                # native mpas_atm_time_integration.F:6948: tend_theta =
                # tend_theta + tend_theta_euler + tend_rtheta_physics -- the
                # saved mixing increment lands before the physics increment,
                # and after rthdynten is formed (line 6941).
                self._add(tend_rt, saved_euler_theta)
            if physics_tendencies is not None:
                self._add(tend_rt, physics_tendencies.rho_theta)
            tend_rw = self._w_tendency(
                current_state, current_diag.vertical_velocity
            )
            self._add(tend_rw, saved_euler_rw)
            tend_omega = convert_w_tendency_to_omega_cuda(
                self.atmosphere.mesh,
                tend_rw,
                tend_ru,
                fzm=self.atmosphere.vertical.fzm,
                fzp=self.atmosphere.vertical.fzp,
                zz=self.atmosphere.vertical.zz,
                zb_cell=self.atmosphere.terrain.zb_cell,
                zb3_cell=self.atmosphere.terrain.zb3_cell,
                kernel_cache=self.cache,
            )
            del tend_rw
            if self.regional_v841 is not None:
                # atm_srk3:2300-2351 -- the specified-zone tendency
                # assignment and the relaxation-zone Rayleigh/Laplacian
                # stages, against the SHARED tend_rho pool and the omega
                # tendency (not tend_rw), exactly as driver.py:2734-2782.
                self.regional_v841.adjust_dynamics_tendencies(
                    tend_ru=tend_ru,
                    tend_rho=tend_rho_saved,
                    tend_rt=tend_rt,
                    tend_omega=tend_omega,
                    rho_u=current_state.rho_u,
                    theta_m=current_diag.theta_m,
                    rho_zz=current_state.rho,
                    rk_step=stage.stage,
                )
                if diabatic_theta is not None:
                    # SG2 on a limited-area run: native zeroes the carry in
                    # the specified zone here (F:7949), so the stage-3
                    # recovery below subtracts nothing in those cells.
                    self._zero_speczone_lagged_heating(diabatic_theta)
            acoustic = CudaAcousticState.zeros(self.nlev, self.ncells, self.nedges)
            forcing = CudaAcousticForcing(
                rho_zz=current_state.rho,
                theta_m=saved_diag.theta_m,
                zz=self.atmosphere.vertical.zz,
                exner=saved_diag.exner,
                cqu=cqu,
                zxu=self.atmosphere.vertical.zxu,
                dss=self.atmosphere.vertical.dss,
                tend_ru=tend_ru,
                tend_rho=tend_rho_saved,
                tend_rt=tend_rt,
                tend_rw=tend_omega,
                w=current_diag.vertical_velocity,
                rw=current_state.rho_w,
                rw_save=saved_state.rho_w,
            )
            # Only cached_v survives this stage's diagnostic solve inputs.
            diagnostics = None
            for small_step in range(1, stage.acoustic_steps + 1):
                # native mpas_atm_time_integration.F:3959 / :3968 -- the damping
                # reference is zeroed on the first sub-step and otherwise holds
                # rtheta_pp as it stood BEFORE this sub-step's advance.
                if damping_reference_in_state:
                    # acoustic_prepare_v841, inside the sub-step below, writes
                    # acoustic.rtheta_pp_old = 0 on sub-step 1 and otherwise
                    # = rtheta_pp as it stood before this sub-step's advance
                    # (acoustic_ru_v841 before it only reads rtheta_pp), and
                    # no later kernel of the sub-step writes it: the same
                    # words capture_rtheta_pp_old copied into a fresh array
                    # every sub-step.  The damping reads that field instead.
                    rtheta_pp_old = None
                else:
                    rtheta_pp_old = self.horizontal.capture_rtheta_pp_old(
                        acoustic.rtheta_pp,
                        small_step=small_step,
                    )
                if self.regional_v841 is None:
                    acoustic = advance_acoustic_step_cuda_v841(
                        self.atmosphere.mesh,
                        acoustic,
                        forcing,
                        coefficients,
                        context=self.v841_context,
                        dts=stage.acoustic_timestep,
                        small_step=small_step,
                        fzm=self.atmosphere.vertical.fzm,
                        fzp=self.atmosphere.vertical.fzp,
                        rdzw=self.atmosphere.vertical.rdzw,
                        in_place=True,
                        kernel_cache=self.cache,
                    )
                else:
                    # The same substep with L5's three specified-zone-aware
                    # entrypoints substituted: the pgrad masking (F:3909),
                    # the rs/ts skip and the implicit-solve skip
                    # (F:4093-4103).  The shared kernels it still launches
                    # are byte-unchanged.
                    from .cuda_regional_forecast_v841 import (
                        advance_acoustic_step_regional_cuda_v841,
                    )

                    acoustic = advance_acoustic_step_regional_cuda_v841(
                        self.regional_v841,
                        self.atmosphere.mesh,
                        acoustic,
                        forcing,
                        coefficients,
                        context=self.v841_context,
                        dts=stage.acoustic_timestep,
                        small_step=small_step,
                        fzm=self.atmosphere.vertical.fzm,
                        fzp=self.atmosphere.vertical.fzp,
                        rdzw=self.atmosphere.vertical.rdzw,
                    )
                if damping_reference_in_state:
                    rtheta_pp_old = acoustic.rtheta_pp_old
                interior_mask = getattr(self.halo_exchanger_v841, "interior_edge_mask", None)
                if self.halo_exchanger_v841 is not None:
                    # Round C: between the advance and the damping, because the
                    # damping at an owned cut edge reads THIS sub-step's ring-1
                    # rtheta_pp, which only its owner computes exactly (the
                    # ring-1 column's divergence reads ring-2 edges outside the
                    # K=2 cone).  ru_p moves in the same frame so the peer
                    # damps owner-truth momentum.
                    interior_work = None
                    if interior_mask is not None:
                        def interior_work():
                            self.horizontal.divergence_damping_3d(
                                acoustic.ru_p, saved_diag.theta_m,
                                acoustic.rtheta_pp, rtheta_pp_old,
                                native_stored_inverse=True,
                                dts=stage.acoustic_timestep,
                                config_smdiv=self.config.config_smdiv,
                                config_len_disp=self.config.config_len_disp,
                                in_place=True, _partition_edges=(interior_mask, 1),
                                **({} if self.regional_v841 is None else
                                   {"n_cells_solve": self.regional_v841.n_cells_solve}),
                            )
                    if interior_work is None:
                        self.halo_exchanger_v841.round_acoustic(acoustic)
                    else:
                        self.halo_exchanger_v841.round_acoustic(acoustic, interior_work=interior_work)
                # native :2395-2406 -- "complete update of horizontal momentum by
                # including 3d divergence damping at the end of the acoustic
                # step".  The authority's call carries no branch and no config
                # flag, so this one carries none either: config_smdiv alone sets
                # the strength (Registry.xml:264 default 0.1) and a zero
                # coefficient is the only way to switch the term off, which is
                # exactly how native behaves.
                acoustic.ru_p = self.horizontal.divergence_damping_3d(
                    acoustic.ru_p,
                    saved_diag.theta_m,
                    acoustic.rtheta_pp,
                    rtheta_pp_old,
                    native_stored_inverse=True,
                    dts=stage.acoustic_timestep,
                    config_smdiv=self.config.config_smdiv,
                    config_len_disp=self.config.config_len_disp,
                    in_place=True,
                    **({} if interior_mask is None else {"_partition_edges": (interior_mask, 0)}),
                    # atm_divergence_damping_3d F:4183 scales the update by
                    # (1 - specZoneMaskEdge), which the shared kernel already
                    # carries; the regional lane only has to name the SOLVE
                    # cell count, because the kernel's two-cell early-out
                    # compares against it and the padded garbage cell sits at
                    # exactly that index (driver.py:2875-2898).
                    **(
                        {}
                        if self.regional_v841 is None
                        else {
                            "n_cells_solve": self.regional_v841.n_cells_solve
                        }
                    ),
                )
            # All acoustic tendency reads are queued on this rank stream.
            # Retire their owners before allocating the recovered endpoint.
            forcing = None
            interior_work = None
            del tend_ru, tend_rt, tend_omega, rtheta_pp_old
            if short_lifetimes and stage is final_stage:
                # After the final stage's acoustic sub-steps nothing reads the
                # implicit coefficients, the dry cqu, dpdz or the two density
                # tendencies again.  The saved Euler tendencies stay: they are
                # live to the return (tests/mpasdiag_forecast_v842.py observes
                # them in the returning frame for the PV budget).
                coefficients = None
                cqu = None
                dpdz = None
                tend_rho_saved = None
                tend_rho_dynamics = None
            stage_scalars = current_state.scalars
            if short_lifetimes:
                # The stage's input candidate was last read by the forcing of
                # the sub-steps above (rho_zz, w, rw); the recovery reads the
                # subcycle-start images, the acoustic state and the scalar
                # block, which stage_scalars keeps.  In stages 2 and 3 this
                # releases the previous stage's recovered candidate before the
                # new one is allocated; in stage 1 it aliases saved_* and
                # frees nothing.
                current_state = current_diag = None
            current_state, current_diag, flux_u, flux_w = self._recover_candidate(
                saved_state,
                saved_diag,
                acoustic,
                stage.stage,
                stage.acoustic_steps,
                stage_scalars,
                diabatic_theta=diabatic_theta, large_timestep=stage.large_timestep,
            )
            stage_scalars = None
            if stage.stage == 3 and acoustic_output_callback is not None:
                # Registry.xml:1911-1921 exposes the final acoustic words in
                # restart output. Retain references before their owner dies;
                # recovering them by subtraction loses native rounding.
                acoustic_output_callback(ru_p=acoustic.ru_p, rw_p=acoustic.rw_p)
            acoustic = None
            if short_lifetimes and stage is final_stage:
                # The final recovery was the last reader of the subcycle-start
                # images; drop them before the final solve diagnostics allocate.
                saved_state = None
                saved_diag = None
            if self.regional_v841 is None:
                validate_recovered_state_cuda_v841(
                    current_state,
                    current_diag,
                    invalid_flag=validation_flag,
                    kernel_cache=self.cache,
                )
            else:
                # The same test, term for term, over the elements native
                # solves: the shared kernel is launched over the padded
                # extent and would refuse the garbage element for holding
                # the native allocation zero where it demands a positive
                # density.  See CudaRegionalRuntimeV841.validate_recovered.
                self.regional_v841.validate_recovered(
                    current_state, current_diag, validation_flag
                )
            if self.halo_exchanger_v841 is not None:
                # Round B (design 44-round schedule): the recovered state and
                # saved diagnostics become owner truth on the halo BEFORE
                # solve_diagnostics reads ring-1 rho at owned cut edges.
                self.halo_exchanger_v841.round_stage_entry(
                    current_state, current_diag
                )
            if stage.stage == 3:
                final_flux_u = flux_u
                final_flux_w = flux_w
            elif short_lifetimes:
                # A non-final stage's recovered mass fluxes have no reader;
                # release them before this stage's solve diagnostics allocate
                # rather than after.
                flux_u = flux_w = None
            if stage.stage == 3 and not final_diagnostics:
                # Nothing in this substep reads the stage-3 solve; the next
                # substep forms its own start solve from the same arrays.
                self._apply_configured_w_boundary(current_diag.vertical_velocity)
                cached_v = None
                continue
            diagnostics = self.horizontal.solve_diagnostics(
                current_state.rho,
                current_state.rho_u,
                dt=outer_dt,
                apvm_upwinding=self.config.config_apvm_upwinding,
                normal_velocity=current_diag.normal_velocity,
                cached_tangential_velocity=cached_v,
                rk_step=stage.stage,
            )
            self._apply_configured_w_boundary(current_diag.vertical_velocity)
            cached_v = diagnostics.tangential_velocity
            if stage.stage != 3:
                del flux_u, flux_w

        if final_flux_u is None or final_flux_w is None:
            raise RuntimeError("v8.4.1 dynamics subcycle did not reach its third RK stage")
        if witness is not None:
            witness.check("the subcycle returned")
        return _CudaDynamicsSubcycleResult(
            state=current_state,
            saved=current_diag,
            diagnostics=diagnostics,
            mass_flux_u=final_flux_u,
            mass_flux_w=final_flux_w,
            modified_theta_dynamics_rate=final_modified_theta_dynamics_rate,
            stage_physics_tendencies=physics_tendencies,
        )

    def flush_deferred_validation_v841(self) -> int:
        """Read every deferred step-validation flag in one transfer; refuse if any step set one.

        Only the one-card full-step LTS executor defers (``deferred_validation_v841`` is a list there and absent
        elsewhere).  Returns how many flags were read.  The refusal is the per-step one's, naming the first
        refused step's clock and the state's current ranges.
        """

        pending = getattr(self, "deferred_validation_v841", None)
        if not pending:
            return 0
        cp = self.cp
        flags = [flag for flag, _ in pending]
        clocks = [clock for _, clock in pending]
        pending.clear()
        values = cp.asnumpy(flags[0] if len(flags) == 1 else cp.concatenate(flags)).tolist()
        refused = [index for index, value in enumerate(values) if int(value) != 0]
        if refused:
            current = self.atmosphere.state
            invalid_ranges = {}
            for name in ("rho", "rho_theta", "rho_u", "scalars"):
                array = getattr(current, name)
                invalid_ranges[name] = {
                    "min": float(cp.nanmin(array).item()),
                    "max": float(cp.nanmax(array).item()),
                    "nonfinite": int(cp.count_nonzero(~cp.isfinite(array)).item()),
                }
            first = refused[0]
            raise FloatingPointError(
                "v8.4.1 CUDA validation flag refused the outer step before publish: "
                f"flag={int(values[first])}, time_seconds={clocks[first]}, "
                f"candidate_ranges={invalid_ranges} (read at the LTS macro boundary; "
                f"{len(refused)} of {len(flags)} class steps refused)"
            )
        return len(flags)

    def _apply_configured_w_boundary(self, vertical_velocity):
        """NOAA F:8030-8105: configured interior w, after solve diagnostics."""
        regional = self.regional_v841
        if regional is None or getattr(self.config, 'config_lbc_w', 'zero') != 'nearest':
            return
        if not hasattr(self, '_boundary_nearest_relaxation'):
            nearest = regional.masks_host.nearest_relaxation_cell
            self._boundary_nearest_relaxation = self.cp.full(self.ncells, self.ncells - 1, dtype=self.cp.int32)
            self._boundary_nearest_relaxation[:len(nearest)] = self.cp.asarray(nearest, dtype=self.cp.int32)
        from .cuda_regional_rrfs import apply_w_boundary_cuda_rrfs
        apply_w_boundary_cuda_rrfs(vertical_velocity, regional.bdy_mask_cell,
            self._boundary_nearest_relaxation, config_lbc_w=self.config.config_lbc_w,
            cell_solve_end=regional.n_cells_solve, kernel_cache=self.cache)

    def _scalar_eddy_source(self, scalar_old, source):
        """NOAA F:4849-4948: update the held source before transport RK1.

        The existing fork mixing kernels implement this unchanged delta;
        advection itself continues through the shared v8.4.x kernels.
        """
        if not getattr(self.config, 'config_scalar_eddy_mix', False):
            return source
        from .cuda_transport_rrfs import apply_gsl_scalar_mixing_cuda_rrfs
        cp = self.cp
        source = cp.zeros_like(scalar_old) if source is None else cp.array(source, copy=True)
        # 44c0849 F:4671 retains the removed kdiff pool name. Use the
        # upstream eddy_visc_horz carrier; an enabled native replay remains
        # unprovable until that dangling native pool lookup is resolved.
        kdiff = (cp.ascontiguousarray(self.les_dissipation.frame['eddy_visc_horz'][:, :self.ncells])
                 if self.les_dissipation is not None else getattr(self, '_scalar_eddy_kdiff', None))
        if kdiff is None:
            kdiff = cp.full_like(self.atmosphere.state.rho, np.float32(self.config.config_h_theta_eddy_visc2))
        if not hasattr(self, '_scalar_eddy_scaling'):
            del2 = cp.empty(self.nedges, dtype=cp.float32)
            del4 = cp.empty_like(del2)
            self.horizontal._launch('mixing_scaling_f32', self.nedges,
                (self.atmosphere.mesh.cells_on_edge, self.atmosphere.mesh.mesh_density,
                 np.int32(self.nedges), np.int32(self.config.config_h_ScaleWithMesh), del2, del4))
            self._scalar_eddy_scaling = del2, del4
        del2, del4 = self._scalar_eddy_scaling
        length = np.float32(self.config.config_len_disp or self.atmosphere.mesh.nominal_min_dc)
        h4 = np.float32(np.float32(self.config.config_h_scalar_filter4) * ((length * length) * length))
        regional = self.regional_v841
        apply_gsl_scalar_mixing_cuda_rrfs(self.atmosphere.mesh, self.v841_context, scalar_old, source,
            kdiff=kdiff, rho_edge=self._scalar_eddy_rho_edge,
            mesh_scaling_del2=del2, mesh_scaling_del4=del4, h_scalar_filter4=h4,
            cell_count=None if regional is None else int(regional.padded.garbage_cell),
            cell_solve_count=None if regional is None else regional.n_cells_solve, kernel_cache=self.cache)
        return source

    def _step_device_v841(
        self, physics_tendencies: Any | None = None, *,
        physics_stage_callback: Any | None = None,
        physics_post_transport_callback: Any | None = None,
        physics_mass_loading_count: int | None = None,
        acoustic_output_callback: Any | None = None,
    ) -> CudaDeviceStepResult | CudaV841PhysicsStepCandidate:
        """Run resident v8.4.1 fields with one final four-byte safety check."""

        if self.v841_context is None:
            raise RuntimeError("v8.4.1 device context is missing")
        from .config_v841 import V841MpasColumnPhysicsConfig

        full_physics = isinstance(self.config, V841MpasColumnPhysicsConfig)
        physics_lane = None
        phase_one_provenance = None
        if full_physics and physics_tendencies is None:
            raise ConfigurationRefusal(
                "held_physics_tendencies",
                None,
                "the full-physics lane must compute and validate phase one before RK",
                "step_device_with_physics(CudaHeldMpasPhysicsTendenciesV841)",
            )
        if not full_physics and physics_tendencies is not None:
            raise ConfigurationRefusal(
                "held_physics_tendencies",
                "provided",
                "a dry v8.4.1 config cannot be relabelled by supplying physics arrays",
                "V841MpasColumnPhysicsConfig()",
            )
        cp = self.cp
        validation_flag = cp.zeros((1,), dtype=cp.int32)
        if self.regional_v841 is not None:
            # driver.py:3085-3089 -- mpas_atm_core.F:735-781 reads lbc_in and
            # re-forms the boundary tendencies at the start of any step whose
            # clock reached an interval end, before any dynamics.
            self.regional_v841.begin_step(self.atmosphere.state)
        step_witness = None
        shares_start_levels = self._shares_start_levels()
        if shares_start_levels:
            # mem/state-levels: native MPAS keeps two time levels; the
            # private image of time level 1 here was a third.  Nothing in the
            # step writes the resident level in place: the subcycles read it
            # (see _advance_dynamics_subcycle_v841), transport reads the old
            # scalars and density and writes fresh stage blocks, the
            # post-transport callbacks write the CANDIDATE (current/carried,
            # recovered fresh) and read outer_state, and the step publishes a
            # new atmosphere rather than filling the resident one.  The one
            # in-place writer of a start-level array is the scalar clamp (and
            # the post-transport scalar fill) when transport does NOT rebind
            # .scalars, so the scalar block keeps its private image exactly
            # then.
            #
            # With subcycle graphs attached the resident level can be the
            # graph's own result buffers, which the first subcycle's replay
            # overwrites.  Every read of the start level up to that replay
            # happens first (the step-start solve, the moist coefficients,
            # and the graph's staging copy of the subcycle inputs, all on
            # this stream), and after it the step reads only outer_saved.rho
            # (transport's old density) and the scalar block.  So with graphs
            # the density alone keeps its private image; rho_theta, rho_u,
            # rho_w and the seven time-level-1 diagnostics are bound.
            graphs = getattr(self, "subcycle_graphs", None)
            transport_rebinds_scalars = (
                bool(self.config.config_scalar_advection)
                and int(self.atmosphere.state.scalars.shape[0]) > 0
            )
            outer_saved = self._bind_state(
                self.atmosphere.state,
                copy_scalars=not transport_rebinds_scalars,
                copy_rho=graphs is not None,
            )
            time_level_one = self._bind_saved(self.atmosphere.saved)
            if graphs is None and _StateLevelsWitness.enabled():
                resident_state = self.atmosphere.state
                resident_saved = self.atmosphere.saved
                step_witness = _StateLevelsWitness(
                    cp, "resident time level 1",
                    rho=resident_state.rho, rho_theta=resident_state.rho_theta,
                    rho_u=resident_state.rho_u, rho_w=resident_state.rho_w,
                    scalars=resident_state.scalars,
                    **{f"saved.{f.name}": getattr(resident_saved, f.name)
                       for f in fields(resident_saved)},
                )
        else:
            outer_saved = self._copy_state(self.atmosphere.state)
            time_level_one = self._copy_saved(self.atmosphere.saved)
        if self.regional_v841 is not None:
            self.regional_v841.bind_state_rho(outer_saved.rho)
        physics_contract_sha256 = None
        scalar_names: tuple[str, ...] = ()
        moist_coefficients = None
        if full_physics:
            from .cuda_physics_v841 import (
                CUDA_PHYSICS_V841_CONTRACT_SHA256,
                CudaHeldMpasPhysicsTendenciesV841,
                CudaPhaseOneExecutionProvenanceV841,
            )

            if not isinstance(
                physics_tendencies, CudaHeldMpasPhysicsTendenciesV841
            ):
                raise TypeError(
                    "full physics requires CudaHeldMpasPhysicsTendenciesV841"
                )
            if physics_tendencies.contract_sha256 != (
                CUDA_PHYSICS_V841_CONTRACT_SHA256
            ):
                raise ValueError("held physics contract digest changed")
            # The row the RUN selected, not the frozen lane's: the scalar
            # block's own width decides which declared row this is, and the
            # dynamics, the preparation and the seam all read the same one.
            run_row = arwen_model_row(self.config)
            if int(outer_saved.scalars.shape[0]) != run_row.n_species():
                raise ValueError(
                    f"{run_row.scheme_name!r} declares "
                    f"{run_row.n_species()} scalars and the state carries "
                    f"{int(outer_saved.scalars.shape[0])}"
                )
            physics_tendencies.validate(
                n_vert_levels=self.nlev,
                n_cells=self.ncells,
                n_edges=self.nedges,
                n_scalars=run_row.n_species(),
            )
            if float(physics_tendencies.time_seconds) != float(
                outer_saved.time_seconds
            ):
                raise ValueError(
                    "held physics time must equal the candidate start exactly: "
                    f"{physics_tendencies.time_seconds} != {outer_saved.time_seconds}"
                )
            phase_one_provenance = physics_tendencies.execution_provenance
            if not isinstance(
                phase_one_provenance, CudaPhaseOneExecutionProvenanceV841
            ):
                raise TypeError("held physics lost typed phase-one provenance")
            physics_lane = _v841_physics_receipt_lane(
                self.config, phase_one_provenance
            )
            physics_contract_sha256 = CUDA_PHYSICS_V841_CONTRACT_SHA256
            scalar_names = run_row.names()
            moist_coefficients = self._compute_moist_dynamics_coefficients_v841(
                outer_saved.scalars,
                scalar_names=scalar_names,
                validation_flag=validation_flag,
            )
        outer_dt = float(self.config.config_dt)
        dynamics_splits = int(self.config.config_dynamics_split_steps)
        dynamics_schedule = RKSchedule.from_mpas(
            outer_dt,
            order=self.config.config_time_integration_order,
            acoustic_substeps=self.config.config_number_of_sub_steps,
            dynamics_splits=dynamics_splits,
        )
        scalar_schedule = RKSchedule.from_mpas(
            outer_dt,
            order=self.config.config_time_integration_order,
            acoustic_substeps=self.config.config_number_of_sub_steps,
            dynamics_splits=1,
        )
        lean_diagnostics = diagnostics_lifetime_lean()
        diagnostics = None
        if not lean_diagnostics:
            diagnostics = self.horizontal.solve_diagnostics(
                outer_saved.rho,
                outer_saved.rho_u,
                dt=outer_dt,
                apvm_upwinding=self.config.config_apvm_upwinding,
                normal_velocity=time_level_one.normal_velocity,
                rk_step=3,
            )
        # A limited-area MPAS suite run (the post-transport callback with its
        # Registry mass-loading count) loads the air with every moist species
        # exactly as the global suite route does (atm_compute_moist_coefficients,
        # mpas_atm_time_integration.F:2125-2139, has no limited-area branch).
        # The regional qv-only path below stays for the dycore's own regional
        # runs, which carry one species.  THE BREAKAGE THIS PREVENTS: with
        # Thompson's condensate the regional path would weigh qv alone.
        regional_suite_loading = (
            self.regional_v841 is not None
            and physics_mass_loading_count is not None
        )
        if (
            self.regional_v841 is not None
            and not regional_suite_loading
            and moist_coefficients is None
            and bool(getattr(self.config, "config_moist_physics", False))
        ):
            # driver.py:3090-3105 -- formed once per outer step from the
            # step-start scalars and reused by every dynamics subcycle.
            moist_coefficients = self.regional_v841.moist_coefficients(
                outer_saved.scalars
            )
        if (self.regional_v841 is None or regional_suite_loading) and moist_coefficients is None:
            # Native atm_srk3 calls atm_compute_moist_coefficients on EVERY
            # step (mpas_atm_time_integration.F:2125-2139), physics suite
            # 'none' included: qtot/cqw/cqu enter dpdz, the Euler w and u
            # pressure terms, the implicit coefficients and the acoustic
            # pressure gradient.  Without them a physics-off run carrying
            # water vapour integrated the dry equations against native's
            # moist ones.  With physics off native's Registry activates only
            # moist-group species (qc..qg need a scheme package; number
            # concentrations need Thompson, which is the physics config), so
            # every carried species loads the air.  On a dry state qtot=0,
            # cqu=cqw=1 and every product/divide is exact: dry bits are kept.
            moist_coefficients = self._compute_moist_dynamics_coefficients_v841(
                outer_saved.scalars,
                scalar_names=None,
                # Registry moist group ends before Thompson number carriers;
                # mpas_atm_time_integration.F:2125-2139, Registry.xml:1705-1747.
                n_mass_loading=(int(outer_saved.scalars.shape[0])
                    if physics_mass_loading_count is None
                    else int(physics_mass_loading_count)),
                validation_flag=validation_flag,
            )
        current = outer_saved
        carried = time_level_one
        split_flux_u_sum = None
        split_flux_w_sum = None
        final_modified_theta_dynamics_rate = None
        for _subcycle in range(dynamics_splits):
            if self.regional_v841 is not None:
                # The in-step offset of every driving interpolation is
                # dt_dynamics*(substep-1) + rk_timestep (F:491-551 through
                # regional_v841.dynamics_time_offset); the substep index is
                # one-based, as in driver.py's loop.
                self.regional_v841.begin_dynamics_substep(_subcycle + 1)
            final_modified_theta_dynamics_rate = None
            result = self._advance_dynamics_subcycle_v841(
                current,
                time_level_one=carried,
                initial_diagnostics=diagnostics,
                schedule=dynamics_schedule,
                outer_dt=outer_dt,
                validation_flag=validation_flag,
                physics_tendencies=physics_tendencies,
                moist_coefficients=moist_coefficients,
                physics_stage_callback=physics_stage_callback,
                dynamics_substep=_subcycle + 1,
                **({"acoustic_output_callback": acoustic_output_callback}
                   if acoustic_output_callback is not None and
                   _subcycle == dynamics_splits - 1 else {}),
                # Lean: each substep forms its start solve and skips its
                # unread stage-3 solve, so no diagnostics set is held here
                # across a substep, the scalar transport or the physics.
                **({"final_diagnostics": False} if lean_diagnostics and not (
                    getattr(self.config, "config_scalar_eddy_mix", False) and
                    _subcycle == dynamics_splits - 1) else {}),
            )
            if physics_stage_callback is not None:
                physics_tendencies = result.stage_physics_tendencies
            current = result.state
            carried = result.saved
            diagnostics = result.diagnostics
            final_modified_theta_dynamics_rate = (
                result.modified_theta_dynamics_rate
            )
            split_flux_u_sum = accumulate_split_flux_cuda_v841(
                result.mass_flux_u,
                split_flux_u_sum,
                kernel_cache=self.cache,
            )
            split_flux_w_sum = accumulate_split_flux_cuda_v841(
                result.mass_flux_w,
                split_flux_w_sum,
                kernel_cache=self.cache,
            )
            del result
        if split_flux_u_sum is None or split_flux_w_sum is None:
            raise RuntimeError("v8.4.1 dynamics subcycle loop did not execute")
        if shares_start_levels:
            # mem/state-levels: the dynamics-only arrays die with the last
            # subcycle instead of riding through transport, the ordinary
            # step's memory peak.  Nothing below reads them: transport takes
            # outer_saved, current, the split fluxes and the physics scalar
            # source; the post-transport callback takes current, carried and
            # outer_saved; the receipts take times and shapes.  Dropping a
            # reference changes no arithmetic.  The solve diagnostics are
            # 5.5 KB per column, the moist coefficients 1.1, the time-level-1
            # diagnostics 2.0 when they were a private image.  Same scope as
            # the bindings: the regional lane classifies device blocks by
            # address, so freeing earlier stays out of it until a regional
            # deck proves it.
            diagnostics = moist_coefficients = time_level_one = None
            if physics_stage_callback is not None:
                release = getattr(getattr(physics_stage_callback, "__self__", None),
                                  "release_dynamics_sources", None)
                if release is not None:
                    physics_tendencies = release()
        if step_witness is not None:
            step_witness.check("transport")
        split_flux_u = finish_split_flux_cuda_v841(
            split_flux_u_sum, dynamics_splits, kernel_cache=self.cache
        )
        split_flux_w = finish_split_flux_cuda_v841(
            split_flux_w_sum, dynamics_splits, kernel_cache=self.cache
        )

        del split_flux_u_sum, split_flux_w_sum
        scalar_stage_timesteps = None
        if (
            self.config.config_scalar_advection
            and current.scalars.shape[0] > 0
        ):
            if getattr(self.config, "config_scalar_eddy_mix", False):
                self._scalar_eddy_rho_edge = diagnostics.h_edge
            scalar_source = self._scalar_eddy_source(outer_saved.scalars,
                None if physics_tendencies is None else physics_tendencies.scalars)
            scalar_stage = cp.array(outer_saved.scalars, copy=True)
            for stage in scalar_schedule.stages:
                if self.halo_exchanger_v841 is not None:
                    # Round D: the six-species stage block becomes owner truth
                    # before the stage's edge fluxes read it at ring<=2.
                    self.halo_exchanger_v841.round_transport(scalar_stage)
                if self.regional_v841 is not None:
                    # driver.py:3186-3257: the regional split transport, with
                    # the mask-4/5 first-order edge downgrade and the
                    # specified-zone cell skip of F:4764-4861, then the
                    # per-stage boundary adjust of atm_srk3:2688-2717.
                    from .cuda_regional_forecast_v841 import (
                        advance_scalars_regional_cuda_v841,
                    )

                    # Native takes the limited routine on the last stage when
                    # EITHER switch is on (mpas_atm_time_integration.F:2953,
                    # "rk_step < 3 .or. (.not. config_monotonic .and. .not.
                    # config_positive_definite)" selects the unlimited one),
                    # as the global path does (cuda_transport_v841).  Testing
                    # config_monotonic alone left a regional run with
                    # positive_definite=True and monotonic off unlimited
                    # while its receipt recorded the switch as on.
                    if (
                        self.config.config_monotonic
                        or self.config.config_positive_definite
                    ) and stage.stage == 3:
                        transported = advance_scalar_transport_cuda_v841(
                            self.atmosphere.mesh, self.v841_context,
                            outer_saved.scalars, scalar_stage, outer_saved.rho,
                            current.rho, split_flux_u, split_flux_w, stage.large_timestep,
                            rk_step=3, config_scalar_advection=True,
                            config_monotonic=bool(self.config.config_monotonic),
                            config_positive_definite=bool(self.config.config_positive_definite),
                            config_split_dynamics_transport=True,
                            config_time_integration_order=self.config.config_time_integration_order,
                            config_scalar_adv_order=3, config_scalar_vadv_order=3,
                            config_coef_3rd_order=self.config.config_coef_3rd_order,
                            coefficients=self.coefficients, fzm=self.atmosphere.vertical.fzm,
                            fzp=self.atmosphere.vertical.fzp, rdzw=self.atmosphere.vertical.rdzw,
                            validation_flag=validation_flag, indices_prevalidated=True,
                            rho_old_prevalidated=True, kernel_cache=self.cache,
                            config_apply_lbcs=True,
                            bdy_mask_cell=self.regional_v841.bdy_mask_cell,
                            bdy_mask_edge=self.regional_v841.bdy_mask_edge,
                            garbage_cell=int(self.regional_v841.padded.garbage_cell),
                            bind_density_pool=self.regional_v841.bind_state_rho,
                            # Round E on N ranks, as on the global path: the
                            # limiter reads both endpoint cells' scale factors
                            # at owned seam edges.  One card has no exchanger.
                            halo_exchange=self.halo_exchanger_v841,
                            **({} if scalar_source is None else {"scalar_tendency": scalar_source}),
                        )
                        scalar_stage = transported.scalars
                        self.regional_v841.bdy_adjust_scalars(scalar_stage, stage.stage)
                        accumulate_finite_array_cuda_v841(
                            scalar_stage, invalid_flag=validation_flag, kernel_cache=self.cache,
                        )
                        continue
                    scalar_stage, _target = advance_scalars_regional_cuda_v841(
                        self.regional_v841,
                        self.atmosphere.mesh,
                        self.v841_context,
                        outer_saved.scalars,
                        scalar_stage,
                        outer_saved.rho,
                        current.rho,
                        split_flux_u,
                        split_flux_w,
                        stage.large_timestep,
                        coefficients=self.coefficients,
                        fzm=self.atmosphere.vertical.fzm,
                        fzp=self.atmosphere.vertical.fzp,
                        rdzw=self.atmosphere.vertical.rdzw,
                        rk_step=stage.stage,
                        config_time_integration_order=self.config.config_time_integration_order,
                        config_coef_3rd_order=(
                            self.config.config_coef_3rd_order
                        ),
                        validation_flag=validation_flag,
                        scalar_tendency=scalar_source,
                    )
                    self.regional_v841.bdy_adjust_scalars(
                        scalar_stage, stage.stage
                    )
                    accumulate_finite_array_cuda_v841(
                        scalar_stage,
                        invalid_flag=validation_flag,
                        kernel_cache=self.cache,
                    )
                    continue
                transported = advance_scalar_transport_cuda_v841(
                    self.atmosphere.mesh,
                    self.v841_context,
                    outer_saved.scalars,
                    scalar_stage,
                    outer_saved.rho,
                    current.rho,
                    split_flux_u,
                    split_flux_w,
                    stage.large_timestep,
                    rk_step=stage.stage,
                    config_scalar_advection=True,
                    config_monotonic=self.config.config_monotonic,
                    config_positive_definite=self.config.config_positive_definite,
                    config_split_dynamics_transport=True,
                    config_time_integration_order=self.config.config_time_integration_order,
                    coefficients=self.coefficients,
                    fzm=self.atmosphere.vertical.fzm,
                    fzp=self.atmosphere.vertical.fzp,
                    rdzw=self.atmosphere.vertical.rdzw,
                    config_scalar_adv_order=3,
                    config_scalar_vadv_order=3,
                    config_coef_3rd_order=self.config.config_coef_3rd_order,
                    validation_flag=validation_flag,
                    indices_prevalidated=True,
                    rho_old_prevalidated=True,
                    kernel_cache=self.cache,
                    halo_exchange=self.halo_exchanger_v841,
                    **(
                        {}
                        if scalar_source is None
                        else {"scalar_tendency": scalar_source}
                    ),
                )
                scalar_stage = transported.scalars
                accumulate_finite_array_cuda_v841(
                    scalar_stage,
                    invalid_flag=validation_flag,
                    kernel_cache=self.cache,
                )
            if self.halo_exchanger_v841 is not None:
                # Final D round: the last stage's transported scalars become
                # owner truth on the halo BEFORE phase-2 (WSM6) consumes the
                # halo columns.  Without it, phase-2 writes backend-held
                # per-column products (h_diabatic, effc/effi) from garbage
                # halo scalars; the next radiation call freezes that garbage
                # into held tendencies whose ring-1 values owned cut edges
                # consume -- the measured step-12 divergence of the first
                # cross-node run.
                self.halo_exchanger_v841.round_transport(scalar_stage)
            current.scalars = scalar_stage
            scalar_stage_timesteps = tuple(
                stage.large_timestep for stage in scalar_schedule.stages
            )
        if physics_post_transport_callback is not None:
            # Native unclamped transport handoff, dry-theta recovery, clamp,
            # microphysics and returned diagnostics; time_integration.F:2770-2816.
            physics_post_transport_callback(
                state=current, saved=carried, outer_state=outer_saved,
                modified_theta_dynamics_rate=final_modified_theta_dynamics_rate,
            )
        if (self.regional_v841 is None and not full_physics
                and physics_post_transport_callback is None):
            # atm_srk3 F:2799-2800 (DO_PHYSICS builds, every suite): negative
            # mixing ratios of every species are set to zero after transport.
            # A where-assignment, not max(): -0.0 and NaN keep native's bits.
            cp.putmask(current.scalars, current.scalars < 0, 0)
        if self.regional_v841 is not None:
            # driver.py:3259-3301, in this order: the unconditional clamp
            # (atm_srk3:2798-2800), then the specified-zone resets and the
            # F:8238 perturbation write and bdy_set_scalars, all reading the
            # PRE-advance clock, then the clock advance last.
            if self.config.config_moist_physics:
                self.regional_v841.clamp_negative_scalars(current.scalars)
            self.regional_v841.reset_speczone_values(
                theta_m=carried.theta_m,
                rho_theta=current.rho_theta,
                rho_theta_perturbation=carried.rho_theta_perturbation,
                scalars=current.scalars,
            )
            self.regional_v841.advance_clock(outer_dt)

        gf_rate_arrays = None
        if full_physics:
            if final_modified_theta_dynamics_rate is None:
                raise RuntimeError(
                    "full-physics split loop did not capture final RK dynamics heating"
                )
            rthdynten = cp.empty_like(current.rho)
            rqvdynten = cp.empty_like(current.rho)
            self._launch_physics_driver(
                "gf_dry_theta_dynamics_rate_v841_f32",
                self.ncells,
                (
                    np.int32(self.nlev),
                    np.int32(self.ncells),
                    np.float32(np.float32(461.6) / np.float32(287.0)),
                    final_modified_theta_dynamics_rate,
                    carried.theta_m,
                    current.scalars[0],
                    rthdynten,
                    rqvdynten,
                    validation_flag,
                ),
            )
            gf_rate_arrays = (rthdynten, rqvdynten)

        if step_witness is not None:
            step_witness.check("publish")
            step_witness = None
        deferred = getattr(self, "deferred_validation_v841", None)
        if deferred is not None:
            # A one-card full-step LTS class step (cuda_fullstep_lts_active.FullStepLtsActive): the flag is
            # read with every other class step's at the macro boundary (flush_deferred_validation_v841), so the
            # host is not held here until the card drains the step.  Breakage the deferral prevents
            # (HEX-LTS-REGRESSION-2026-10-27): this four-byte read was a device synchronize in every class step,
            # so the card idled through each step's host tail and the next class's first launches (85.5 %
            # busy on the 96 -> 3 km zoom globe against 96.5 % for the fixed step).  Nothing between the step
            # and the macro boundary is written out; a set flag refuses the macro, before any frame.
            deferred.append((validation_flag, float(outer_saved.time_seconds)))
            validation_value = 0
            validation_seconds = 0.0
        elif _forecast_checks_active():
            # Inside an MPAS-physics forecast loop (runtime_v842), the word goes through deferred_checks_v842,
            # as cuda_driver_rrfs has done since cb2ae586: reduced on the device now, copied to pinned memory
            # at the step's flush, inspected when it lands.  The forecast still stops with this
            # FloatingPointError at most one step later and before any output (before_output/prefer_earlier).
            # THE COST THIS ENDS (rrfs review 2026-10-29, conus3km 8 cards): the shared driver serves the
            # 44c0849 RRFS row and MPAS-A v8.4.2, and it read this word with a blocking copy every step, so each
            # rank's host waited for its card to drain the whole step (dycore + TEMPO tail) before it could queue
            # the next step's outer physics and post its halo rounds; HB15 ran 863.8 ms/step at 77.6 % busy
            # against 688.4 ms for the old pin, whose RRFS driver defers the same word.  Outside a forecast
            # (replays, referee helpers, the ArWen route) the word is read at once, with the ranges, as before.
            from .mpasphys.deferred_checks_v842 import defer
            refused_clock = float(outer_saved.time_seconds)
            defer(validation_flag,
                  lambda value: ("v8.4.1 CUDA validation flag refused the outer step before publish: "
                                 f"flag={value}, time_seconds={refused_clock}"),
                  error=FloatingPointError)
            validation_value = 0
            validation_seconds = 0.0
        else:
            validation_started = time.perf_counter()
            validation_value = int(cp.asnumpy(validation_flag)[0])
            validation_seconds = time.perf_counter() - validation_started
        if validation_value != 0:
            invalid_ranges = {}
            for name in ("rho", "rho_theta", "rho_u", "scalars"):
                values = getattr(current, name)
                invalid_ranges[name] = {
                    "min": float(cp.nanmin(values).item()),
                    "max": float(cp.nanmax(values).item()),
                    "nonfinite": int(cp.count_nonzero(~cp.isfinite(values)).item()),
                }
            raise FloatingPointError(
                "v8.4.1 CUDA validation flag refused the outer step before publish: "
                f"flag={validation_value}, time_seconds={outer_saved.time_seconds}, "
                f"candidate_ranges={invalid_ranges}"
            )
        validation_transfer = TransferStats(
            int(validation_flag.nbytes), validation_seconds
        )
        current.time_seconds = outer_saved.time_seconds + outer_dt
        dynamics_tendencies = None
        if full_physics:
            assert gf_rate_arrays is not None
            dynamics_tendencies = CudaV841GfDynamicsTendencies(
                rthdynten=gf_rate_arrays[0],
                rqvdynten=gf_rate_arrays[1],
                time_seconds=current.time_seconds,
            )
            dynamics_tendencies.validate(
                cp=cp, n_vert_levels=self.nlev, n_cells=self.ncells
            )
        result_atmosphere = DeviceAtmosphere(
            self.atmosphere.mesh,
            current,
            self.atmosphere.vertical,
            self.atmosphere.reference,
            carried,
            self.atmosphere.terrain,
            self.atmosphere.h2d,
        )
        compile_manifest = self.cache.compile_manifest()
        dynamics_stage_timesteps = tuple(
            stage.large_timestep for stage in dynamics_schedule.stages
        )
        if full_physics:
            assert physics_tendencies is not None
            assert physics_contract_sha256 is not None
            assert dynamics_tendencies is not None
            held_validation = physics_tendencies.validation_d2h
            assert phase_one_provenance is not None
            gwdo_validation = phase_one_provenance.gwdo_validation_d2h
            candidate_d2h = _sum_transfer_stats(
                held_validation,
                validation_transfer,
                gwdo_validation,
            )
            assert physics_lane is not None
            receipt = CudaV841PhysicsStepReceipt(
                evidence=physics_lane["candidate_evidence"],
                configuration=dict(self.configuration),
                configuration_sha256=self.configuration_sha256,
                authority_ruler=None,
                authority_ruler_sha256=None,
                frozen_source=V841_CUDA_SOURCE,
                t0_diagnostics_source=self.t0_diagnostics_source,
                stage_acoustic_steps=tuple(
                    stage.acoustic_steps for stage in dynamics_schedule.stages
                ),
                start_time_seconds=outer_saved.time_seconds,
                end_time_seconds=current.time_seconds,
                h2d=self.h2d,
                d2h=candidate_d2h,
                compile_manifest=compile_manifest,
                compile_manifest_sha256=canonical_sha256(compile_manifest),
                layout_contract=dict(CUDA_LAYOUT_CONTRACT),
                source_release="v8.4.1",
                dynamics_split_steps=dynamics_splits,
                dynamics_timestep_seconds=outer_dt / dynamics_splits,
                dynamics_stage_timesteps=dynamics_stage_timesteps,
                scalar_transport_stage_timesteps=scalar_stage_timesteps,
                split_flux_reduction=SPLIT_FLUX_REDUCTION,
                authority_nonclaims=physics_lane["authority_nonclaims"],
                bulk_physics_contract_sha256=physics_contract_sha256,
                bulk_physics_tendencies_applied=True,
                physics_components=physics_lane["physics_components"],
                physics_cadences_seconds=_v841_physics_cadences(self.config),
                gwd_scheme=physics_lane["gwd_scheme"],
                gwd_evidence=(
                    None
                    if physics_lane["gwd_evidence"] is None
                    else dict(physics_lane["gwd_evidence"])
                ),
                phase_one_execution_provenance=phase_one_provenance,
                gwdo_validation_d2h=gwdo_validation,
                scalar_order=scalar_names,
                held_tendency_time_seconds=float(
                    physics_tendencies.time_seconds
                ),
                held_tendency_validation_d2h=held_validation,
                dycore_validation_d2h=validation_transfer,
                phase2_validation_d2h=None,
                final_commit_validation_d2h=None,
                moist_dynamics_coefficients_applied=True,
                moist_dynamics_scalar_order=arwen_model_row(self.config).names(),
                moist_dynamics_negative_qv_policy=(
                    V841_MOIST_DYNAMICS_NEGATIVE_QV_POLICY
                ),
                moist_dynamics_source=V841_MOIST_DYNAMICS_SOURCE,
                moist_dynamics_source_sha256=(
                    V841_MOIST_DYNAMICS_SOURCE_SHA256
                ),
                post_wsm6_h_diabatic_reapplied=False,
                composite_committed=False,
                final_authority_claim=False,
            )
            candidate = CudaV841PhysicsStepCandidate(
                atmosphere=result_atmosphere,
                dynamics_tendencies=dynamics_tendencies,
                receipt=receipt,
            )
            self._pending_v841_physics_candidate = candidate
            return candidate

        receipt = CudaV841StepReceipt(
            evidence=CUDA_V841_IMPLEMENTED_UNLINKED_EVIDENCE,
            configuration=dict(self.configuration),
            configuration_sha256=self.configuration_sha256,
            authority_ruler=None,
            authority_ruler_sha256=None,
            frozen_source=V841_CUDA_SOURCE,
            t0_diagnostics_source=self.t0_diagnostics_source,
            stage_acoustic_steps=tuple(
                stage.acoustic_steps for stage in dynamics_schedule.stages
            ),
            start_time_seconds=outer_saved.time_seconds,
            end_time_seconds=current.time_seconds,
            h2d=self.h2d,
            d2h=validation_transfer,
            compile_manifest=compile_manifest,
            compile_manifest_sha256=canonical_sha256(compile_manifest),
            layout_contract=dict(CUDA_LAYOUT_CONTRACT),
            source_release="v8.4.1",
            dynamics_split_steps=dynamics_splits,
            dynamics_timestep_seconds=outer_dt / dynamics_splits,
            dynamics_stage_timesteps=dynamics_stage_timesteps,
            scalar_transport_stage_timesteps=scalar_stage_timesteps,
            split_flux_reduction=SPLIT_FLUX_REDUCTION,
            authority_nonclaims=CUDA_V841_AUTHORITY_NONCLAIMS,
        )
        return CudaDeviceStepResult(result_atmosphere, receipt)

    def step_device_with_physics(
        self, physics_tendencies: Any
    ) -> CudaV841PhysicsStepCandidate:
        """Advance to an uncommitted endpoint using one held phase-one source."""

        from .config_v841 import V841MpasColumnPhysicsConfig

        if not isinstance(self.config, V841MpasColumnPhysicsConfig):
            raise ConfigurationRefusal(
                "config_physics_suite",
                self.config.config_physics_suite,
                "held column tendencies require the distinct full-physics config",
                "V841MpasColumnPhysicsConfig()",
            )
        if self._pending_v841_physics_candidate is not None:
            raise RuntimeError(
                "a prior full-physics candidate still awaits post-WSM6 commit"
            )
        current_configuration = cuda_configuration_payload(self.config)
        if (
            current_configuration != self.configuration
            or canonical_sha256(current_configuration) != self.configuration_sha256
        ):
            raise ValueError("CUDA configuration changed after admission")
        result = self._step_device_v841(physics_tendencies)
        if not isinstance(result, CudaV841PhysicsStepCandidate):
            raise RuntimeError("full-physics step did not return a transactional candidate")
        return result

    def commit_post_wsm6_candidate(
        self,
        candidate: CudaV841PhysicsStepCandidate,
        recovery: Any,
    ) -> CudaV841CommittedPhysicsStepResult:
        """Publish only the externally validated phase-two resident endpoint."""

        from .config_v841 import V841MpasColumnPhysicsConfig
        from .cuda_physics_v841 import (
            CUDA_PHYSICS_V841_CONTRACT_SHA256,
            CudaPostRkWsm6RecoveryV841,
        )

        if not isinstance(self.config, V841MpasColumnPhysicsConfig):
            raise ConfigurationRefusal(
                "config_physics_suite",
                self.config.config_physics_suite,
                "post-WSM6 commit exists only for the full-physics lane",
                "V841MpasColumnPhysicsConfig()",
            )
        if candidate is not self._pending_v841_physics_candidate:
            raise ValueError("candidate is not the driver's pending transaction")
        self.config.validate(**({"les_dissipation": self.les_dissipation}
                                if self.les_dissipation is not None else {}))
        current_configuration = cuda_configuration_payload(self.config)
        current_configuration_sha256 = canonical_sha256(current_configuration)
        if (
            current_configuration != self.configuration
            or current_configuration_sha256 != self.configuration_sha256
            or canonical_sha256(self.configuration) != self.configuration_sha256
        ):
            raise ValueError("CUDA configuration changed after candidate admission")
        physics_lane = _validate_v841_physics_candidate_receipt(
            candidate.receipt,
            config=self.config,
            configuration=current_configuration,
            configuration_sha256=current_configuration_sha256,
        )
        execution_provenance = candidate.receipt.phase_one_execution_provenance
        if not isinstance(recovery, CudaPostRkWsm6RecoveryV841):
            raise TypeError("commit requires CudaPostRkWsm6RecoveryV841")
        if recovery.contract_sha256 != CUDA_PHYSICS_V841_CONTRACT_SHA256:
            raise ValueError("post-WSM6 recovery contract digest changed")
        endpoint = float(candidate.atmosphere.state.time_seconds)
        if endpoint != float(candidate.receipt.end_time_seconds):
            raise ValueError("candidate state and receipt endpoint times differ")
        if endpoint != float(recovery.time_seconds):
            raise ValueError(
                "post-WSM6 recovery time must equal the candidate endpoint exactly"
            )
        if recovery.state is not candidate.atmosphere.state:
            raise ValueError(
                "post-WSM6 recovery must validate the pending candidate in place"
            )
        if float(self.atmosphere.state.time_seconds) != float(
            candidate.receipt.start_time_seconds
        ):
            raise ValueError("resident driver start state changed during phase two")
        if recovery.validation_d2h.bytes != 4:
            raise ValueError("phase-two recovery must carry its exact four-byte flag")

        staged = DeviceAtmosphere(
            candidate.atmosphere.mesh,
            recovery.state,
            candidate.atmosphere.vertical,
            candidate.atmosphere.reference,
            candidate.atmosphere.saved,
            candidate.atmosphere.terrain,
            candidate.atmosphere.h2d,
        )
        staged.validate()
        rebuilt_saved = self._rebuild_saved_diagnostics(staged)
        committed = DeviceAtmosphere(
            staged.mesh,
            staged.state,
            staged.vertical,
            staged.reference,
            rebuilt_saved,
            staged.terrain,
            staged.h2d,
        )
        committed.validate()

        # This is deliberately a fresh flag, separate from the held, dycore,
        # and WSM6 recovery gates.  Validate every committed prognostic and
        # rebuilt saved field while all arrays are still resident.
        final_validation_flag = self.cp.zeros((1,), dtype=self.cp.int32)
        if self.regional_v841 is None:
            validate_recovered_state_cuda_v841(
                committed.state,
                committed.saved,
                invalid_flag=final_validation_flag,
                kernel_cache=self.cache,
            )
        else:
            # The same test, term for term, over the elements native solves.
            # The shared kernel is launched over the PADDED extent and would
            # refuse the garbage element for holding the native allocation
            # zero where it demands a positive density -- the identical
            # reason the dycore's own recovered-state gate is routed through
            # this method at the end of every RK stage.  This site is the
            # full-physics commit, and before 2026-08-26 it was the only
            # recovered-state gate the regional lane did NOT route, because
            # no limited-area run had ever reached it.
            self.regional_v841.validate_recovered(
                committed.state, committed.saved, final_validation_flag
            )
        accumulate_finite_array_cuda_v841(
            committed.state.scalars,
            invalid_flag=final_validation_flag,
            kernel_cache=self.cache,
        )
        final_validation_started = time.perf_counter()
        # Forecast can read this word with its endpoint health envelope.
        # The hook must finish before the existing numeric refusal or publish.
        reader = getattr(self, "final_validation_reader", None)
        final_validation_value = (
            int(self.cp.asnumpy(final_validation_flag)[0]) if reader is None
            else reader(committed, final_validation_flag)
        )
        final_validation_seconds = time.perf_counter() - final_validation_started
        final_validation_transfer = TransferStats(
            int(final_validation_flag.nbytes), final_validation_seconds
        )
        if final_validation_transfer.bytes != 4:
            raise ValueError("final committed-state validation must transfer four bytes")
        if final_validation_value != 0:
            raise FloatingPointError(
                "final rebuilt v8.4.1 full-physics state failed numeric validation"
            )

        compile_manifest = self.cache.compile_manifest()
        total_d2h = _sum_transfer_stats(
            candidate.receipt.d2h,
            recovery.validation_d2h,
            final_validation_transfer,
        )
        receipt = replace(
            candidate.receipt,
            evidence=physics_lane["committed_evidence"],
            configuration=dict(current_configuration),
            configuration_sha256=current_configuration_sha256,
            authority_ruler=None,
            authority_ruler_sha256=None,
            frozen_source=V841_CUDA_SOURCE,
            d2h=total_d2h,
            compile_manifest=compile_manifest,
            compile_manifest_sha256=canonical_sha256(compile_manifest),
            layout_contract=dict(CUDA_LAYOUT_CONTRACT),
            source_release="v8.4.1",
            authority_nonclaims=physics_lane["authority_nonclaims"],
            bulk_physics_contract_sha256=CUDA_PHYSICS_V841_CONTRACT_SHA256,
            bulk_physics_tendencies_applied=True,
            physics_components=physics_lane["physics_components"],
            physics_cadences_seconds=_v841_physics_cadences(self.config),
            gwd_scheme=physics_lane["gwd_scheme"],
            gwd_evidence=(
                None
                if physics_lane["gwd_evidence"] is None
                else dict(physics_lane["gwd_evidence"])
            ),
            phase_one_execution_provenance=execution_provenance,
            gwdo_validation_d2h=execution_provenance.gwdo_validation_d2h,
            scalar_order=arwen_model_row(self.config).names(),
            held_tendency_time_seconds=candidate.receipt.start_time_seconds,
            held_tendency_validation_d2h=(
                candidate.receipt.held_tendency_validation_d2h
            ),
            dycore_validation_d2h=candidate.receipt.dycore_validation_d2h,
            phase2_validation_d2h=recovery.validation_d2h,
            final_commit_validation_d2h=final_validation_transfer,
            moist_dynamics_coefficients_applied=True,
            moist_dynamics_scalar_order=arwen_model_row(self.config).names(),
            moist_dynamics_negative_qv_policy=(
                V841_MOIST_DYNAMICS_NEGATIVE_QV_POLICY
            ),
            moist_dynamics_source=V841_MOIST_DYNAMICS_SOURCE,
            moist_dynamics_source_sha256=V841_MOIST_DYNAMICS_SOURCE_SHA256,
            post_wsm6_h_diabatic_reapplied=False,
            composite_committed=True,
            final_authority_claim=False,
        )
        result = CudaV841CommittedPhysicsStepResult(
            atmosphere=committed,
            dynamics_tendencies=candidate.dynamics_tendencies,
            surface_updates=recovery.surface_updates,
            effective_radii=recovery.effective_radii,
            receipt=receipt,
        )
        self.atmosphere = committed
        self._pending_v841_physics_candidate = None
        return result

    def abort_post_wsm6_candidate(
        self, candidate: CudaV841PhysicsStepCandidate
    ) -> None:
        """Discard exactly the pending candidate without publishing any state."""

        pending = self._pending_v841_physics_candidate
        if pending is None or candidate is not pending:
            raise ValueError("candidate is not the driver's pending transaction")
        self._pending_v841_physics_candidate = None

    def step_device(self) -> CudaDeviceStepResult:
        """Execute one admitted dry RK3 branch without a host field copy."""

        cp = self.cp
        current_configuration = cuda_configuration_payload(self.config)
        current_configuration_sha256 = canonical_sha256(current_configuration)
        if (
            current_configuration != self.configuration
            or current_configuration_sha256 != self.configuration_sha256
        ):
            raise ValueError("CUDA configuration changed after admission")
        if self.source_release == "v8.4.1":
            from .config_v841 import V841MpasColumnPhysicsConfig

            if isinstance(self.config, V841MpasColumnPhysicsConfig):
                raise ConfigurationRefusal(
                    "held_physics_tendencies",
                    None,
                    "plain step_device cannot omit phase-one column physics",
                    "step_device_with_physics(validated_held_tendencies)",
                )
            return self._step_device_v841()
        self.mixing_config = (
            _mixing_config(self.config)
            if self.config.config_horiz_mixing == "2d_smagorinsky"
            else None
        )
        authority_payload = (
            self._consume_authority_snapshot()
            if self.authority_ruler is not None and self._authority_snapshot_pending
            else None
        )
        ruler_is_current = authority_payload is not None
        if ruler_is_current:
            self.atmosphere, self.coefficients = authority_payload
            self.horizontal = CudaHorizontal(
                self.atmosphere.mesh,
                self.atmosphere.vertical.n_vert_levels,
                kernel_cache=self.cache,
            )
            self._authority_snapshot_pending = False
        saved_state = self.atmosphere.state
        saved_diag = self.atmosphere.saved
        schedule = RKSchedule.from_mpas(
            self.config.config_dt,
            order=self.config.config_time_integration_order,
            acoustic_substeps=self.config.config_number_of_sub_steps,
            dynamics_splits=1,
        )
        coefficients = self._vertical_coefficients(
            saved_state,
            schedule.stages[0].acoustic_timestep,
            saved_diag.theta_m,
            saved_diag.rho_theta_perturbation,
        )
        current_state = self._device_state(
            cp.array(saved_state.rho, copy=True),
            cp.array(saved_state.rho_theta, copy=True),
            cp.array(saved_state.rho_u, copy=True),
            cp.array(saved_state.rho_w, copy=True),
            cp.array(saved_state.scalars, copy=True),
        )
        current_diag = self._device_saved(
            cp.array(saved_diag.theta_m, copy=True),
            cp.array(saved_diag.exner, copy=True),
            cp.array(saved_diag.density_perturbation, copy=True),
            cp.array(saved_diag.rho_theta_perturbation, copy=True),
            cp.array(saved_diag.pressure_perturbation, copy=True),
            cp.array(saved_diag.normal_velocity, copy=True),
            cp.array(saved_diag.vertical_velocity, copy=True),
        )
        diagnostics = self.horizontal.solve_diagnostics(
            current_state.rho,
            current_state.rho_u,
            dt=self.config.config_dt,
            apvm_upwinding=self.config.config_apvm_upwinding,
            normal_velocity=current_diag.normal_velocity,
            rk_step=3,
        )
        cached_v = diagnostics.tangential_velocity
        cqu = cp.ones_like(saved_state.rho_u)
        dpdz = self._scale(saved_diag.density_perturbation, -9.80616)
        euler_ru = self.horizontal.pressure_gradient_euler_tendency(
            saved_diag.pressure_perturbation,
            dpdz,
            cqu,
            self.atmosphere.vertical.zz,
            self.atmosphere.vertical.zxu,
        )
        tend_rho_saved = self.horizontal.density_tendency(
            saved_state.rho_u,
            saved_state.rho_w,
            self.atmosphere.vertical.rdzw,
        )
        euler_rw = cp.empty_like(saved_state.rho_w)
        self._launch(
            "euler_w_f32",
            self.ncells * (self.nlev + 1),
            (
                np.int32(self.nlev),
                np.int32(self.ncells),
                saved_diag.pressure_perturbation,
                dpdz,
                self.atmosphere.vertical.rdzu,
                self.atmosphere.vertical.fzm,
                self.atmosphere.vertical.fzp,
                euler_rw,
            ),
        )
        saved_euler_ru = euler_ru
        saved_euler_rw = euler_rw
        saved_euler_theta = None
        if self.mixing_config is not None:
            saved_mixing = self.horizontal.compute_dry_mixing_tendencies(
                current_diag.normal_velocity,
                diagnostics.tangential_velocity,
                current_diag.vertical_velocity,
                saved_diag.theta_m,
                diagnostics.h_edge,
                diagnostics.divergence,
                diagnostics.vorticity,
                dt=self.config.config_dt,
                config=self.mixing_config,
            )
            # Frozen RK1 builds complete saved Euler pools once, then reuses
            # those exact arrays during RK2/RK3.  Form the pool in the same
            # binary32 addition order as CPU apply_saved_euler_mixing().
            saved_euler_ru = cp.array(euler_ru, copy=True)
            self._add(saved_euler_ru, saved_mixing.tend_u_euler)
            saved_euler_rw = cp.array(euler_rw, copy=True)
            self._add(saved_euler_rw, saved_mixing.tend_w_euler)
            saved_euler_theta = cp.array(
                saved_mixing.tend_theta_euler,
                copy=True,
            )
        dss = self.atmosphere.vertical.dss
        split_flux_u = cp.array(saved_state.rho_u, copy=True)
        split_flux_w = cp.array(saved_state.rho_w, copy=True)

        for stage in schedule.stages:
            if stage.stage == 2 and self.config.config_time_integration_order == 3:
                coefficients = self._vertical_coefficients(
                    current_state,
                    stage.acoustic_timestep,
                    current_diag.theta_m,
                    current_diag.rho_theta_perturbation,
                )
            mass_divergence = self.horizontal.mass_flux_divergence(current_state.rho_u)
            tend_ru = self._vertical_u(
                current_diag.normal_velocity, current_state.rho_w
            )
            tend_ru = self._vector_momentum(
                current_diag.normal_velocity,
                diagnostics.h_edge,
                diagnostics.pv_edge,
                diagnostics.kinetic_energy,
                mass_divergence,
                initial_tendency=tend_ru,
            )
            self._add(tend_ru, saved_euler_ru)
            tend_rt = self._theta_tendency(
                current_state,
                current_diag.theta_m,
                saved_state,
                saved_diag.theta_m,
                stage.stage,
            )
            if saved_euler_theta is not None:
                self._add(tend_rt, saved_euler_theta)
            tend_rw = self._w_tendency(current_state, current_diag.vertical_velocity)
            self._add(tend_rw, saved_euler_rw)
            tend_omega = convert_w_tendency_to_omega_cuda(
                self.atmosphere.mesh,
                tend_rw,
                tend_ru,
                fzm=self.atmosphere.vertical.fzm,
                fzp=self.atmosphere.vertical.fzp,
                zz=self.atmosphere.vertical.zz,
                zb_cell=self.atmosphere.terrain.zb_cell,
                zb3_cell=self.atmosphere.terrain.zb3_cell,
                kernel_cache=self.cache,
            )
            acoustic = CudaAcousticState.zeros(self.nlev, self.ncells, self.nedges)
            forcing = CudaAcousticForcing(
                rho_zz=current_state.rho,
                theta_m=saved_diag.theta_m,
                zz=self.atmosphere.vertical.zz,
                exner=saved_diag.exner,
                cqu=cqu,
                zxu=self.atmosphere.vertical.zxu,
                dss=dss,
                tend_ru=tend_ru,
                tend_rho=tend_rho_saved,
                tend_rt=tend_rt,
                tend_rw=tend_omega,
                w=current_diag.vertical_velocity,
                rw=current_state.rho_w,
                rw_save=saved_state.rho_w,
            )
            for small_step in range(1, stage.acoustic_steps + 1):
                rtheta_pp_old = None
                if self.config.config_divergence_damping:
                    rtheta_pp_old = self.horizontal.capture_rtheta_pp_old(
                        acoustic.rtheta_pp,
                        small_step=small_step,
                    )
                acoustic = advance_acoustic_step_cuda(
                    self.atmosphere.mesh,
                    acoustic,
                    forcing,
                    coefficients,
                    dts=stage.acoustic_timestep,
                    small_step=small_step,
                    epssm=self.config.config_epssm,
                    fzm=self.atmosphere.vertical.fzm,
                    fzp=self.atmosphere.vertical.fzp,
                    rdzw=self.atmosphere.vertical.rdzw,
                    in_place=True,
                    kernel_cache=self.cache,
                )
                if self.config.config_divergence_damping:
                    assert rtheta_pp_old is not None
                    acoustic.ru_p = self.horizontal.divergence_damping_3d(
                        acoustic.ru_p,
                        saved_diag.theta_m,
                        acoustic.rtheta_pp,
                        rtheta_pp_old,
                        dts=stage.acoustic_timestep,
                        config_smdiv=self.config.config_smdiv,
                        config_len_disp=self.config.config_len_disp,
                        in_place=True,
                    )
            current_state, current_diag, flux_u, flux_w = self._recover_candidate(
                saved_state,
                saved_diag,
                acoustic,
                stage.stage,
                stage.acoustic_steps,
                current_state.scalars,
            )
            if stage.stage == 3:
                split_flux_u = flux_u
                split_flux_w = flux_w
            if (
                self.config.config_scalar_advection
                and not self.config.config_split_dynamics_transport
                and current_state.scalars.shape[0] > 0
            ):
                transported = advance_scalar_transport_cuda(
                    self.atmosphere.mesh,
                    saved_state.scalars,
                    current_state.scalars,
                    saved_state.rho,
                    current_state.rho,
                    flux_u,
                    flux_w,
                    stage.large_timestep,
                    rk_step=stage.stage,
                    config_scalar_advection=True,
                    config_monotonic=self.config.config_monotonic,
                    config_positive_definite=self.config.config_positive_definite,
                    config_split_dynamics_transport=False,
                    config_time_integration_order=self.config.config_time_integration_order,
                    coefficients=self.coefficients,
                    fzm=self.atmosphere.vertical.fzm,
                    fzp=self.atmosphere.vertical.fzp,
                    rdzw=self.atmosphere.vertical.rdzw,
                    config_scalar_adv_order=3,
                    config_scalar_vadv_order=3,
                    config_coef_3rd_order=self.config.config_coef_3rd_order,
                    kernel_cache=self.cache,
                )
                current_state.scalars = transported.scalars
            diagnostics = self.horizontal.solve_diagnostics(
                current_state.rho,
                current_state.rho_u,
                dt=self.config.config_dt,
                apvm_upwinding=self.config.config_apvm_upwinding,
                normal_velocity=current_diag.normal_velocity,
                cached_tangential_velocity=cached_v,
                rk_step=stage.stage,
            )
            cached_v = diagnostics.tangential_velocity

        if (
            self.config.config_scalar_advection
            and self.config.config_split_dynamics_transport
            and current_state.scalars.shape[0] > 0
        ):
            scalar_stage = cp.array(saved_state.scalars, copy=True)
            for stage in schedule.stages:
                transported = advance_scalar_transport_cuda(
                    self.atmosphere.mesh,
                    saved_state.scalars,
                    scalar_stage,
                    saved_state.rho,
                    current_state.rho,
                    split_flux_u,
                    split_flux_w,
                    stage.large_timestep,
                    rk_step=stage.stage,
                    config_scalar_advection=True,
                    config_monotonic=self.config.config_monotonic,
                    config_positive_definite=self.config.config_positive_definite,
                    config_split_dynamics_transport=True,
                    config_time_integration_order=self.config.config_time_integration_order,
                    coefficients=self.coefficients,
                    fzm=self.atmosphere.vertical.fzm,
                    fzp=self.atmosphere.vertical.fzp,
                    rdzw=self.atmosphere.vertical.rdzw,
                    config_scalar_adv_order=3,
                    config_scalar_vadv_order=3,
                    config_coef_3rd_order=self.config.config_coef_3rd_order,
                    kernel_cache=self.cache,
                )
                scalar_stage = transported.scalars
            current_state.scalars = scalar_stage

        current_state.time_seconds = saved_state.time_seconds + self.config.config_dt
        result_atmosphere = DeviceAtmosphere(
            self.atmosphere.mesh,
            current_state,
            self.atmosphere.vertical,
            self.atmosphere.reference,
            current_diag,
            self.atmosphere.terrain,
            self.atmosphere.h2d,
        )
        compile_manifest = self.cache.compile_manifest()
        ruler_is_current = bool(
            ruler_is_current
            and saved_state.time_seconds
            == self.authority_ruler["admitted_start_time_seconds"]
        )
        receipt = CudaStepReceipt(
            evidence=(
                self.evidence
                if ruler_is_current
                else CUDA_IMPLEMENTED_UNLINKED_EVIDENCE
            ),
            configuration=dict(self.configuration),
            configuration_sha256=self.configuration_sha256,
            authority_ruler=(dict(self.authority_ruler) if ruler_is_current else None),
            authority_ruler_sha256=(
                self.authority_ruler_sha256 if ruler_is_current else None
            ),
            frozen_source=FROZEN_CUDA_SOURCE,
            t0_diagnostics_source=self.t0_diagnostics_source,
            stage_acoustic_steps=tuple(
                stage.acoustic_steps for stage in schedule.stages
            ),
            start_time_seconds=saved_state.time_seconds,
            end_time_seconds=current_state.time_seconds,
            h2d=self.h2d,
            d2h=_zero_transfer(),
            compile_manifest=compile_manifest,
            compile_manifest_sha256=canonical_sha256(compile_manifest),
            layout_contract=dict(CUDA_LAYOUT_CONTRACT),
        )
        return CudaDeviceStepResult(result_atmosphere, receipt)

    def step(self) -> CudaHostStepResult:
        """Execute one resident step, then download only the final gate state."""

        device = self.step_device()
        state, state_transfer = device.atmosphere.state.to_host_timed()
        started = time.perf_counter()
        saved = device.atmosphere.saved
        host_saved = DrySavedDiagnostics(
            theta_m=self.cp.asnumpy(saved.theta_m),
            exner=self.cp.asnumpy(saved.exner),
            density_perturbation=self.cp.asnumpy(saved.density_perturbation),
            rho_theta_perturbation=self.cp.asnumpy(saved.rho_theta_perturbation),
            pressure_perturbation=self.cp.asnumpy(saved.pressure_perturbation),
            normal_velocity=self.cp.asnumpy(saved.normal_velocity),
            vertical_velocity=self.cp.asnumpy(saved.vertical_velocity),
        )
        self.cp.cuda.get_current_stream().synchronize()
        elapsed = time.perf_counter() - started
        diagnostic_bytes = sum(
            int(getattr(saved, name).nbytes)
            for name in (
                "theta_m",
                "exner",
                "density_perturbation",
                "rho_theta_perturbation",
                "pressure_perturbation",
                "normal_velocity",
                "vertical_velocity",
            )
        )
        receipt = replace(
            device.receipt,
            d2h=TransferStats(
                device.receipt.d2h.bytes + state_transfer.bytes + diagnostic_bytes,
                device.receipt.d2h.seconds + state_transfer.seconds + elapsed,
            ),
        )
        return CudaHostStepResult(state, host_saved, receipt)


__all__ = [
    "CUDA_AUTHORITY_INITIAL_SCHEMA",
    "CUDA_AUTHORITY_RULER_SCHEMA",
    "CUDA_IMPLEMENTED_UNLINKED_EVIDENCE",
    "CUDA_ORIGINAL_JW_BRANCH_EVIDENCE",
    "CUDA_T0_EXACT_SIDECAR",
    "CUDA_T0_REBUILT_DIAGNOSTICS",
    "CUDA_WHOLE_STEP_EVIDENCE",
    "CUDA_LAYOUT_CONTRACT",
    "CudaAuthorityRulerBinder",
    "CudaDeviceStepResult",
    "CudaDryDycoreDriver",
    "CudaHostStepResult",
    "CudaStepReceipt",
    "CudaV841CommittedPhysicsStepResult",
    "CudaV841GfDynamicsTendencies",
    "CudaV841PhysicsStepCandidate",
    "CudaV841PhysicsStepReceipt",
    "CudaV841StepReceipt",
    "cuda_authority_initial_fingerprint",
    "cuda_configuration_payload",
    "FROZEN_CUDA_SOURCE",
]
