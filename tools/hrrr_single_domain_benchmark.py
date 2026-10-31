#!/usr/bin/env python3
"""Run native HRRR preprocessing/forecasting for a validated target domain."""

from __future__ import annotations

from woof.physics_registry import canonical_template_id

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from dataclasses import asdict, replace
from datetime import datetime, timedelta
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from collections.abc import Mapping
from types import MappingProxyType, SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from woof import runtime_manifest  # noqa: E402
from woof.ingest.hrrr import _read_gate, aerosol_fill_receipt  # noqa: E402
from woof.ingest.preprocess_backend import preprocess_math_call
from woof.aerosol_source_receipt import (  # noqa: E402
    AEROSOL_SOURCE_KEY,
    aerosol_source_report_entry,
)
from tools.hrrr_build_native_static import (  # noqa: E402
    array_sha256,
    benchmark_grid,
    sha256_file,
    validate_static,
)
from woof.hrrr_route_inputs import (  # noqa: E402
    ROUTE_DEFAULT_PHYSICS_PROFILE,
)
from woof.ingest.hrrr_target import (  # noqa: E402
    HrrrTargetDomain,
    load_hrrr_target_domain,
    required_hrrr_source_window,
)
from woof.physics_compat import (  # noqa: E402
    ASYMMETRIC_RADIATION_NOCTURNAL_ACK,
    CONSTANT_DOWNWARD_LONGWAVE_ACK,
    EXPERIMENTAL_THOMPSON_ENV,
    KESSLER_PROFILE_ID,
    route_physics_profiles,
    MORRISON_PROFILE_ID,
    MYNN_NOAHMP_PROFILE_ID,
    MYNN_NOAHMP_RTE_RRTMGP_PROFILE_ID,
    MYNN_PROFILE_ID,
    MYNN_RTE_RRTMGP_PROFILE_ID,
    MYNN_RUC_PROFILE_ID,
    MYNN_RUC_RTE_RRTMGP_PROFILE_ID,
    NOAHMP_PROFILE_ID,
    NSSL2_LEGACY_RRTMG_PROFILE_ID,
    NSSL2_PROFILE_ID,
    P3_LEGACY_RRTMG_PROFILE_ID,
    RUC_PROFILE_ID,
    THOMPSON_LEGACY_RRTMG_PROFILE_ID,
    THOMPSON_MYNN_RUC_DUDHIA_PROFILE_ID,
    THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID,
    THOMPSON_PROFILE_ID,
    THOMPSON_RTE_RRTMGP_PROFILE_ID,
    THOMPSON_SHINHONG_LEGACY_RRTMG_PROFILE_ID,
    THOMPSON_TABLE_ROOT_ENV,
    WRF_RRTMG_LEGACY,
    WRF_RRTMG_TO_RTE_RRTMGP,
    WSM6_PROFILE_ID,
    downward_longwave_disposition,
    first_local_night_time,
    single_domain_runtime_switches,
    thompson_guard_exports,
    thompson_runtime_requirements,
    validate_single_domain_physics_profile,
)
from woof.core.thompson_contract import (  # noqa: E402
    MP_PHYSICS as THOMPSON_MP_PHYSICS,
)
from woof.core.nssl2_contract import (  # noqa: E402
    CONTRACT_ID as NSSL2_CONTRACT_ID,
    DEFAULT_MODE as NSSL2_DEFAULT_MODE,
    MP_PHYSICS as NSSL2_MP_PHYSICS,
    WRF_NAMELIST_DEFAULTS as NSSL2_WRF_NAMELIST_DEFAULTS,
    nssl2_contract_receipt,
    resolve_nssl2_mode,
)
from woof import explain  # noqa: E402
from woof.progress import prep_progress, prep_stage  # noqa: E402
from woof.hrrr_forecast import (  # noqa: E402
    hrrr_source_window, resolve_cycle_flags)
from woof.namelist_seal import namelist_extension_invariant  # noqa: E402
from woof.vertical_contract import (  # noqa: E402
    explicit_vertical_from_wrf_namelist,
)
MAX_HISTORY_CFL = 10.0
MAX_HISTORY_W_MS = 150.0
REPORT_SCHEMA = "gpuwm-native-hrrr-benchmark-v2"
PREPARATION_REPORT_SCHEMA = "gpuwm-native-hrrr-preparation-v2"
RUNNER_CAPABILITIES_SCHEMA = "gpuwm-runner-capabilities-v1"
_FORECAST_EXECUTOR_MODULES = (
    "woof.core.clock",
    "woof.core.dycore",
    "woof.core.health",
    "woof.core.model",
    "woof.core.refl",
    "woof.io.wrfout",
    "woof.state_digest",
)
_NATIVE_REFLECTIVITY_MP_PHYSICS = frozenset({1, 6, 8, 10, 18})


def _missing_forecast_executor_modules() -> list[str]:
    missing = []
    for module in _FORECAST_EXECUTOR_MODULES:
        try:
            available = importlib.util.find_spec(module) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            available = False
        if not available:
            missing.append(module)
    return missing


def _history_uses_native_reflectivity(
        *, ticks: int, has_moisture: bool, mp_physics: int) -> bool:
    """Return whether this history frame owns a native REFL_10CM stash."""

    return (
        ticks != 0
        and has_moisture
        and mp_physics in _NATIVE_REFLECTIVITY_MP_PHYSICS
    )


def runner_capabilities() -> dict[str, object]:
    """Return the import-only public contract advertised to launchers."""

    missing_executor_modules = _missing_forecast_executor_modules()
    forecast_available = not missing_executor_modules
    thompson = thompson_runtime_requirements()
    # The guarded evidence runtime's two-variable launch contract is
    # ``thompson`` above, and it describes ONE profile.  The mp8 suites
    # this route stages itself resolve their tables through the project's
    # packaged ladder instead (_microphysics_table_authority), so their
    # capability rows say that rather than advertising an environment
    # gate they do not have.  Both rows quote the same pinned table
    # authority, because it is the same bytes either way.
    staged_thompson = {
        "readiness": "WRF_MATCHED_RUN_CANDIDATE",
        "table_staging": "route-staged-at-profile-binding",
        "table_root_resolution": (
            f"{THOMPSON_TABLE_ROOT_ENV} override, then user staging, "
            "then packaged package-data root"),
        "table_authority": thompson["table_authority"],
        "runtime_guards": [
            "exact-size-and-sha256 before GPU setup",
        ],
        "external_table_assets": [
            asset["filename"]
            for asset in thompson["table_authority"]["assets"]
        ],
    }
    return {
        "schema": RUNNER_CAPABILITIES_SCHEMA,
        "runner": ROUTE_ID,
        "supported_sources": ["hrrr"],
        "physics_profile_ids": list(NATIVE_BENCHMARK_PHYSICS_PROFILES),
        "report_schema": REPORT_SCHEMA,
        "preparation_report_schema": PREPARATION_REPORT_SCHEMA,
        "readiness": (
            "FORECAST_IMPLEMENTATION_PRESENT_RUNTIME_PREFLIGHT_REQUIRED"
            if forecast_available
            else "PREPARATION_ONLY_FORECAST_EXECUTOR_OMITTED"
        ),
        "modes": {
            "prepare-only": {
                "available": True,
                "availability_scope": "implementation-present",
                "launch_ready": None,
                "included_in_standalone_rw_wps_wheel": True,
                "can_run_without_cupy": True,
                "cupy_requirement": "cuda-preprocess-backend-only",
                "preprocess_backends": {
                    "cpu": "available-with-native-cpu-backend",
                    "cuda": "requires-cupy-and-compatible-cuda-runtime",
                    "auto": "resolves-runtime-backend",
                },
                "requires_prepared_cache_output": True,
            },
            "forecast": {
                "available": forecast_available,
                "availability_scope": "executor-module-presence-only",
                "launch_ready": None,
                "launch_readiness_check": (
                    "validate CuPy, CUDA, GPU allocation, inputs, and profile "
                    "guards before launch"
                ),
                "requires_cupy": True,
                "requires_compatible_cuda_gpu": True,
                "missing_executor_modules": missing_executor_modules,
                "included_in_standalone_rw_wps_wheel": False,
                "unavailable_reason": (
                    None
                    if forecast_available
                    else "standalone RW-WPS omits the GPUWM forecast executor"
                ),
            },
        },
        "standalone_rw_wps_wheel": {
            "runner_included": True,
            "prepare_only_available": True,
            "forecast_executor_included": False,
        },
        "physics_profiles": {
            WSM6_PROFILE_ID: {
                "selector": 6,
                "readiness": "SUPPORTED_RUNNER_PROFILE",
                "explicit_expert_consent_required": False,
                "runtime_guards": [],
            },
            KESSLER_PROFILE_ID: {
                "selector": 1,
                "readiness": "IMPLEMENTED_UNVERIFIED",
                "explicit_expert_consent_required": False,
                "runtime_guards": [],
                "source_scope": ["hrrr"],
                "frozen_species_policy": (
                    "retain QC/QR; discard source QI/QS/QG with a receipt"),
            },
            THOMPSON_PROFILE_ID: {
                "selector": 8,
                **thompson,
                # Warnings are informational for executable profiles.  The
                # exact env/table guards still fail closed at launch.
                "explicit_expert_consent_required": False,
            },
            THOMPSON_LEGACY_RRTMG_PROFILE_ID: {
                "selector": 8,
                **staged_thompson,
                "explicit_expert_consent_required": False,
                "radiation_solver": "legacy RRTMG",
            },
            # The route default since the owner ruling of 2026-09-19.  Same
            # composition as the row above and the same staged tables; the
            # readiness is its own because the registry ranks it at its
            # composition ceiling, no receipt covering the composed suite
            # on this engine existing yet.  A default whose receipt carries
            # no readiness key is a default nobody can audit, which is why
            # the row is written rather than inherited.
            THOMPSON_RTE_RRTMGP_PROFILE_ID: {
                "selector": 8,
                **staged_thompson,
                "readiness": "IMPLEMENTED_UNVERIFIED",
                "explicit_expert_consent_required": False,
                "radiation_solver": "RTE+RRTMGP",
                "route_default": True,
            },
            THOMPSON_SHINHONG_LEGACY_RRTMG_PROFILE_ID: {
                "selector": 8,
                **staged_thompson,
                "explicit_expert_consent_required": False,
                "radiation_solver": "legacy RRTMG",
                # The one component that differs from the row above.  The
                # microphysics requirements are identical because the
                # microphysics is identical.
                "pbl_solver": "Shin-Hong 2015 scale-aware",
            },
            MORRISON_PROFILE_ID: {
                "selector": 10,
                "readiness": "WRF_MATCHED_RUN_RUNTIME_PROFILE",
                "explicit_expert_consent_required": False,
                "runtime_guards": [],
            },
            NSSL2_PROFILE_ID: {
                "selector": NSSL2_MP_PHYSICS,
                "readiness": "WRF_MATCHED_RUN_CANDIDATE",
                "explicit_expert_consent_required": False,
                "runtime_guards": [],
                "external_table_assets": [],
                "contract_id": NSSL2_CONTRACT_ID,
                "resolved_fixed_preset": True,
            },
            NSSL2_LEGACY_RRTMG_PROFILE_ID: {
                "selector": NSSL2_MP_PHYSICS,
                "readiness": "WRF_MATCHED_RUN_CANDIDATE",
                "explicit_expert_consent_required": False,
                "runtime_guards": [],
                "external_table_assets": [],
                "contract_id": NSSL2_CONTRACT_ID,
                "radiation_solver": "legacy RRTMG",
                "resolved_fixed_preset": True,
            },
            P3_LEGACY_RRTMG_PROFILE_ID: {
                "selector": 50,
                "readiness": "IMPLEMENTED_UNVERIFIED",
                "explicit_expert_consent_required": False,
                # The lookup table ships INSIDE the woof wheel
                # (woof/data/p3/tables) and is byte-validated at profile
                # binding, so nothing is external -- the guard names the
                # check, not an environment variable.
                "runtime_guards": [
                    "exact-size-and-sha256 p3_lookupTable_1 before GPU "
                    "setup",
                ],
                "external_table_assets": [],
                "radiation_solver": "legacy RRTMG",
                "source_scope": ["hrrr"],
                "frozen_species_policy": (
                    "retain QC/QR/QI; discard source QS/QG with a "
                    "receipt (P3 carries one ice category)"),
            },
            MYNN_PROFILE_ID: {
                "selector": 6,
                "readiness": "IMPLEMENTED_UNVERIFIED",
                "explicit_expert_consent_required": False,
                "runtime_guards": [],
            },
            MYNN_RTE_RRTMGP_PROFILE_ID: {
                "selector": 6,
                "readiness": "IMPLEMENTED_UNVERIFIED",
                "explicit_expert_consent_required": False,
                "runtime_guards": [],
                "radiation_solver": "RTE+RRTMGP",
            },
            MYNN_RUC_PROFILE_ID: {
                "selector": 6,
                "readiness": "IMPLEMENTED_UNVERIFIED",
                "explicit_expert_consent_required": False,
                "runtime_guards": [],
            },
            MYNN_RUC_RTE_RRTMGP_PROFILE_ID: {
                "selector": 6,
                "readiness": "IMPLEMENTED_UNVERIFIED",
                "explicit_expert_consent_required": False,
                "runtime_guards": [],
                "radiation_solver": "RTE+RRTMGP",
            },
            # The Thompson members of the pair above: the staged Thompson
            # table block, because the microphysics is Thompson, and the
            # readiness of the composition, which the registry ranks at
            # the implemented-unverified ceiling MYNN and RUC set.
            THOMPSON_MYNN_RUC_DUDHIA_PROFILE_ID: {
                "selector": 8,
                **staged_thompson,
                "readiness": "IMPLEMENTED_UNVERIFIED",
                "explicit_expert_consent_required": False,
            },
            THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID: {
                "selector": 8,
                **staged_thompson,
                "readiness": "IMPLEMENTED_UNVERIFIED",
                "explicit_expert_consent_required": False,
                "radiation_solver": "RTE+RRTMGP",
            },
            THOMPSON_MYNN_RUC_MONTHLY_LEGACY_RRTMG_PROFILE_ID: {
                "selector": 8,
                **staged_thompson,
                "readiness": "IMPLEMENTED_UNVERIFIED",
                "explicit_expert_consent_required": False,
                "radiation_solver": "legacy RRTMG",
            },
            THOMPSON_MYNN_RUC_MONTHLY_SOLAR_LEGACY_RRTMG_PROFILE_ID: {
                "selector": 8,
                **staged_thompson,
                "readiness": "IMPLEMENTED_UNVERIFIED",
                "explicit_expert_consent_required": False,
                "radiation_solver": "legacy RRTMG",
                "surface_albedo": "solar-geometry-selected",
            },
            THOMPSON_MYNN_GSD41_PROFILE_ID: {
                "selector": 28,
                "readiness": "IMPLEMENTED_UNVERIFIED",
                "explicit_expert_consent_required": False,
                "radiation_solver": "legacy RRTMG",
                "table_staging": "route-staged-at-profile-binding",
                "runtime_guards": [
                    "classic and aerosol tables size- and SHA256-checked before GPU setup",
                    "staged WIF climatology for the declared aerosol source",
                ],
            },
            RUC_PROFILE_ID: {
                "selector": 6,
                "readiness": "IMPLEMENTED_UNVERIFIED",
                "explicit_expert_consent_required": False,
                "runtime_guards": [],
            },
            NOAHMP_PROFILE_ID: {
                "selector": 6,
                "readiness": "IMPLEMENTED_UNVERIFIED",
                "explicit_expert_consent_required": True,
                "expert_acknowledgement_id":
                    "noahmp-host-column-throughput-v1",
                "runtime_guards": [
                    "measured column ceiling or explicit accepted budget",
                    "glacier columns refused",
                ],
            },
            MYNN_NOAHMP_PROFILE_ID: {
                "selector": 6,
                "readiness": "IMPLEMENTED_UNVERIFIED",
                "explicit_expert_consent_required": True,
                "expert_acknowledgement_id":
                    "noahmp-host-column-throughput-v1",
                "runtime_guards": [
                    "measured column ceiling or explicit accepted budget",
                    "glacier columns refused",
                ],
            },
            MYNN_NOAHMP_RTE_RRTMGP_PROFILE_ID: {
                "selector": 6,
                "readiness": "IMPLEMENTED_UNVERIFIED",
                "explicit_expert_consent_required": True,
                "expert_acknowledgement_id":
                    "noahmp-host-column-throughput-v1",
                "runtime_guards": [
                    "measured column ceiling or explicit accepted budget",
                    "glacier columns refused",
                ],
                "radiation_solver": "RTE+RRTMGP",
            },
        },
        "window": {
            "kind": "absolute-cycle-relative-contiguous-hourly-source-leads",
            "minimum_frame_count": 2,
            "maximum_source_forecast_hour": 48,
            "maximum_run_seconds": 172_800,
            "cycle_horizon_hours": {
                "00Z": 48, "06Z": 48, "12Z": 48, "18Z": 48,
                "other": 18,
            },
            "model_forcing_rebased_to_zero": True,
            "source_forcing_cadence_seconds": 3600,
            "run_seconds": {
                "finite_positive_required": True,
                "whole_hour_required": False,
                "source_end_policy": "start-plus-ceiling-run-seconds-over-3600",
                "explicit_end_must_equal_derived_end": True,
            },
        },
        "output": {
            "io_modes": ["none", "history"],
            "history_interval_seconds": {
                "required_for_mode": "history",
                "accepted_for_prepare_only_cache_identity": True,
                "finite_positive_required": True,
                "must_be_whole_model_steps": True,
                "must_evenly_divide_run": False,
                "schedule_policy": (
                    "initial-and-floor-multiples-at-or-before-run-end"),
                "initial_frame_required": True,
                "frame_at_run_end_required": False,
                "last_scheduled_frame_may_precede_run_end": True,
            },
            "configurable_cadence": True,
            "restart_output": False,
        },
        "capability_query": {
            "flag": "--show-capabilities",
            "side_effect_free": True,
            "requires_cupy": False,
            "validates_gpu_or_runtime_assets": False,
        },
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_json(value):
    if isinstance(value, dict):
        return {str(key): _strict_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if np.isfinite(value) else None
    return value


def _history_period_count(run_seconds: float, cadence_seconds: float) -> int:
    """Return complete output periods due at or before model stop."""

    run_seconds = float(run_seconds)
    cadence_seconds = float(cadence_seconds)
    if not math.isfinite(run_seconds) or run_seconds <= 0.0:
        raise ValueError("run-seconds must be finite and positive")
    if not math.isfinite(cadence_seconds) or cadence_seconds <= 0.0:
        raise ValueError(
            "history-interval-seconds must be finite and positive")
    return Fraction(run_seconds) // Fraction(cadence_seconds)


def _history_output_schedule(
        *, start_time: datetime, run_seconds: float, cadence_seconds: float,
        domain_id: int = 1,
) -> tuple[tuple[float, datetime, str], ...]:
    """Resolve model-relative output seconds, valid times, and WRF names."""

    periods = _history_period_count(run_seconds, cadence_seconds)
    cadence = Fraction(float(cadence_seconds))
    records = []
    for index in range(periods + 1):
        offset_seconds = float(index * cadence)
        valid_time = start_time + timedelta(seconds=offset_seconds)
        if valid_time.microsecond != 0:
            raise ValueError(
                "history cadence produces sub-second valid times that cannot "
                "be represented by second-complete WRF history filenames")
        records.append((
            offset_seconds,
            valid_time,
            valid_time.strftime(
                f"wrfout_d{int(domain_id):02d}_%Y-%m-%d_%H_%M_%S"),
        ))
    names = [record[2] for record in records]
    if len(names) != len(set(names)):
        raise ValueError("history cadence produces duplicate WRF filenames")
    return tuple(records)


def _validate_history_output_cadence(
        exp, history_interval_seconds: float,
) -> dict[str, object]:
    """Validate the explicit history cadence against the generated model."""

    requested = float(history_interval_seconds)
    domain = exp.root
    configured = float(domain.history_interval_s)
    derived = float(domain.run.output_interval_s)
    if (not math.isfinite(requested) or requested <= 0.0
            or requested != configured or requested != derived):
        raise ValueError(
            "history-interval-seconds must be finite, positive, and exactly "
            "match the generated experiment history cadence")
    exact_steps = Fraction(configured) / exp.dt_exact(domain.grid_id)
    if exact_steps.denominator != 1 or exact_steps < 1:
        raise ValueError(
            "history cadence is not a positive whole number of exact model "
            "time steps")
    schedule = _history_output_schedule(
        start_time=exp.start_time, run_seconds=exp.run_seconds,
        cadence_seconds=configured)
    periods = len(schedule) - 1
    first = schedule[0][1]
    last_offset, last_scheduled, _name = schedule[-1]
    run_end = first + timedelta(seconds=float(exp.run_seconds))
    last_equals_run_end = (
        Fraction(periods) * Fraction(configured)
        == Fraction(float(exp.run_seconds)))
    return {
        "schema": "gpuwm-native-hrrr-history-cadence-v1",
        "requested_seconds": requested,
        "configured_seconds": configured,
        "exact_model_steps_per_interval": int(exact_steps),
        "complete_intervals": periods,
        "expected_frame_count": periods + 1,
        "initial_valid_time": first.isoformat(),
        "last_scheduled_offset_seconds": last_offset,
        "last_scheduled_valid_time": last_scheduled.isoformat(),
        "run_end_offset_seconds": float(exp.run_seconds),
        "run_end_valid_time": run_end.isoformat(),
        "last_scheduled_equals_run_end": last_equals_run_end,
        "initial_frame_required": True,
        "run_end_frame_scheduled": last_equals_run_end,
    }


def _partition_preprocess_worker_budget(
        total_workers: int, job_slots: int) -> tuple[int, ...]:
    """Partition one native-thread budget across fixed concurrent slots."""

    for value, label in (
            (total_workers, "total workers"), (job_slots, "job slots")):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"preprocess {label} must be a positive integer")
    slots = min(total_workers, job_slots)
    quotient, remainder = divmod(total_workers, slots)
    return tuple(
        quotient + (1 if slot < remainder else 0)
        for slot in range(slots))


class _PreprocessWorkerBudget:
    """Deterministic accounting for concurrent native CPU transform jobs.

    Under the CUDA backend the budget has no slots to partition, but it is
    not empty: the masked surface fields (soil, snow, skin temperature and
    sea ice) and the native route's soil stencil run in the Rust library
    on the host, on ``host_workers`` threads (``--preprocess-workers``, or
    every CPU the process may use).  They run in the controller's mapping
    of one forcing time at a time, so that count is also the peak.
    """

    schema = "gpuwm-preprocess-worker-budget-v1"
    _phase_order = {
        "full_domain_initialization": 0,
        "boundary_mapping": 1,
        "boundary_initialization": 2,
    }

    def __init__(self, *, backend: str, requested_total,
                 effective_total: int | None, requested_job_slots: int,
                 future_job_count: int, clock_origin: float,
                 host_workers: int | None = None):
        self.backend = backend
        self.requested_total = requested_total
        self.effective_total = effective_total
        self.requested_job_slots = requested_job_slots
        self.future_job_count = future_job_count
        self.clock_origin = float(clock_origin)
        self.jobs: list[dict[str, object]] = []
        if backend == "cpu":
            if effective_total is None:
                raise ValueError("CPU preprocessing requires a worker budget")
            wanted_slots = min(requested_job_slots, future_job_count)
            self.slot_workers = (
                _partition_preprocess_worker_budget(
                    effective_total, wanted_slots)
                if wanted_slots else ())
        else:
            if backend != "cuda":
                raise ValueError(f"unsupported preprocessing backend {backend!r}")
            if effective_total is not None:
                raise ValueError("CUDA preprocessing has no native CPU budget")
            if (isinstance(host_workers, bool)
                    or not isinstance(host_workers, int) or host_workers < 1):
                # A receipt without it would say the host steps took no
                # threads, which is what the CUDA budget said before they
                # moved to the Rust library.
                raise ValueError(
                    "CUDA preprocessing needs the worker count of its host "
                    "steps (the masked surface fields)")
            self.slot_workers = ()
        self.host_workers = host_workers if backend == "cuda" else None

    @property
    def concurrent_job_slots(self) -> int:
        return len(self.slot_workers)

    def allocation_for_hour(self, hour: int) -> tuple[int | None, int]:
        if self.backend != "cpu" or self.effective_total is None:
            raise RuntimeError("native worker allocations apply only to CPU")
        if hour == 0:
            return None, self.effective_total
        if not self.slot_workers:
            raise RuntimeError("no future preprocessing job slots were allocated")
        slot = (int(hour) - 1) % len(self.slot_workers)
        return slot, self.slot_workers[slot]

    def record(self, *, forecast_hour: int, phase: str,
               slot: int | None, native_workers: int,
               started: float, finished: float) -> None:
        if self.backend != "cpu" or self.effective_total is None:
            raise RuntimeError("cannot record native workers for CUDA")
        if phase not in self._phase_order:
            raise ValueError(f"unknown preprocessing job phase {phase!r}")
        if not (math.isfinite(started) and math.isfinite(finished)
                and finished > started):
            raise ValueError("preprocessing job interval must be finite and positive")
        expected_slot, expected_workers = self.allocation_for_hour(
            int(forecast_hour))
        if phase == "full_domain_initialization":
            if int(forecast_hour) != 0:
                raise ValueError("full-domain preprocessing must be f00")
        elif int(forecast_hour) < 1:
            raise ValueError("boundary preprocessing requires f01 or later")
        if slot != expected_slot or native_workers != expected_workers:
            raise RuntimeError(
                "preprocessing job used a worker allocation outside its slot")
        identity = (int(forecast_hour), phase)
        if any((job["forecast_hour"], job["phase"]) == identity
               for job in self.jobs):
            raise RuntimeError(f"duplicate preprocessing job receipt {identity}")
        self.jobs.append({
            "forecast_hour": int(forecast_hour),
            "phase": phase,
            "slot": slot,
            "effective_native_workers": int(native_workers),
            "started_seconds_from_preparation_start": (
                float(started) - self.clock_origin),
            "finished_seconds_from_preparation_start": (
                float(finished) - self.clock_origin),
            "wall_seconds": float(finished) - float(started),
            "_started": float(started),
            "_finished": float(finished),
        })

    def receipt(self) -> dict[str, object]:
        ordered = sorted(
            self.jobs,
            key=lambda job: (
                int(job["forecast_hour"]),
                self._phase_order[str(job["phase"])]))
        if self.backend == "cpu":
            expected = set()
            if self.future_job_count:
                expected.add((0, "full_domain_initialization"))
                for hour in range(1, self.future_job_count + 1):
                    expected.add((hour, "boundary_mapping"))
                    expected.add((hour, "boundary_initialization"))
            observed = {
                (int(job["forecast_hour"]), str(job["phase"]))
                for job in ordered
            }
            if observed != expected:
                raise RuntimeError(
                    "native preprocessing worker receipt has incomplete jobs: "
                    f"expected {sorted(expected)!r}, observed {sorted(observed)!r}")
        peak_workers = 0
        peak_jobs = 0
        active_workers = 0
        active_jobs = 0
        if self.backend == "cpu":
            by_slot: dict[int | None, list[dict[str, object]]] = {}
            events = []
            for job in ordered:
                by_slot.setdefault(job["slot"], []).append(job)
                workers = int(job["effective_native_workers"])
                events.append((float(job["_started"]), 1, workers))
                events.append((float(job["_finished"]), 0, workers))
            for slot, jobs in by_slot.items():
                previous_finished = None
                for job in sorted(jobs, key=lambda item: float(item["_started"])):
                    started = float(job["_started"])
                    if previous_finished is not None and started < previous_finished:
                        raise RuntimeError(
                            f"preprocessing worker slot {slot} overlapped itself")
                    previous_finished = float(job["_finished"])
            # End events sort before start events at a shared boundary.
            for _, kind, workers in sorted(events):
                if kind == 0:
                    active_workers -= workers
                    active_jobs -= 1
                else:
                    active_workers += workers
                    active_jobs += 1
                    peak_workers = max(peak_workers, active_workers)
                    peak_jobs = max(peak_jobs, active_jobs)
                if active_workers < 0 or active_jobs < 0:
                    raise RuntimeError("invalid preprocessing allocation timeline")
            if active_workers or active_jobs:
                raise RuntimeError("unclosed preprocessing allocation timeline")
            if peak_workers > int(self.effective_total):
                raise RuntimeError(
                    "native preprocessing exceeded its total worker budget")
        if self.backend == "cuda":
            peak_workers = int(self.host_workers)
        public_jobs = []
        for job in ordered:
            public_jobs.append({
                key: value for key, value in job.items()
                if not key.startswith("_")
            })
        host_steps = {}
        if self.backend == "cuda":
            host_steps = {
                "host_step_native_workers": int(self.host_workers),
                "host_step_scope": (
                    "masked surface fields in the Rust preprocessing "
                    "library, one forcing time at a time in the "
                    "controller"),
            }
        return {
            "schema": self.schema,
            "backend": self.backend,
            "applicable": self.backend == "cpu",
            "scope": "native CPU horizontal and WRF-real transforms",
            **host_steps,
            "requested_total_native_workers": self.requested_total,
            "effective_total_native_workers": self.effective_total,
            "requested_prepare_job_slots": self.requested_job_slots,
            "effective_concurrent_job_slots": len(self.slot_workers),
            "slot_native_workers": [
                {"slot": slot, "native_workers": workers}
                for slot, workers in enumerate(self.slot_workers)],
            "effective_allocation_per_job": public_jobs,
            "peak_active_native_workers": peak_workers,
            "peak_active_preprocessing_jobs": peak_jobs,
            "peak_policy": (
                "sum of reserved native-worker allocations across "
                "overlapping controller job intervals"
                if self.backend == "cpu" else
                "the host steps' worker count: they run one forcing time "
                "at a time"),
            "pipeline_decoder_workers_included": False,
        }


def _proc_io() -> dict[str, int]:
    values = {}
    try:
        for line in Path("/proc/self/io").read_text().splitlines():
            key, value = line.split(":", 1)
            values[key] = int(value.strip())
    except OSError:
        pass
    return values


def _device_name(cp) -> str:
    """The attached device's name as text, not as a bytes repr.

    ``getDeviceProperties(0)["name"]`` is ``bytes``, and ``str()`` of
    bytes is its repr: a 1.5.0 field report carries the device as
    ``b'NVIDIA GeForce RTX 5070 Ti'`` -- quotes, ``b`` prefix and all --
    in a receipt whose whole job is to say which card produced the
    numbers.  Decoded the way the identity blocks in
    :mod:`woof.gpu_stack_identity` and :mod:`woof.report_bundle` decode
    it, ``errors="replace"`` included, so a driver that returns
    something undecodable degrades to a readable string instead of
    raising inside a receipt.
    """

    name = cp.cuda.runtime.getDeviceProperties(0).get("name", b"")
    if isinstance(name, (bytes, bytearray)):
        name = bytes(name).decode("utf-8", errors="replace")
    return str(name)



def _configured_soil_mesh(grid, experiment_tables):
    """Declare the decoder's actual geometry to the shared soil operator."""
    from woof.ingest.hrrr import hrrr_source_grid
    from woof.ingest.soil_downscale import soil_mesh_plan_from_case
    return soil_mesh_plan_from_case(
        None, grid, experiment_tables, source_grid=hrrr_source_grid())


def _fetch_receipt_path(args):
    """The fetch receipt beside the source files, where ``woof fetch``
    records each runtime surface row (appended, or its fallback)."""
    from woof.fetch import FETCH_MANIFEST_NAME
    root = getattr(args, "source_root", None)
    if root is not None:
        return Path(root) / FETCH_MANIFEST_NAME
    return getattr(args, "source_manifest", None)


def _source_identity() -> dict[str, object]:
    paths = (
        REPO / "woof/hrrr_forecast.py",
        REPO / "woof/ingest/hrrr.py",
        REPO / "woof/ingest/hrrr_physics.py",
        REPO / "woof/ingest/hrrr_surface.py",
        REPO / "woof/ingest/real.py",
        REPO / "woof/ingest/soil.py",
        REPO / "woof/ingest/soil_downscale.py",
        REPO / "woof/static/highres_production.py",
        REPO / "woof/static/highres.py",
        REPO / "woof/static/highres_fetch.py",
        REPO / "woof/ingest/ruc_soil.py",
        REPO / "woof/ingest/lateral_bc.py",
        REPO / "woof/ingest/prepared_cache.py",
        REPO / "woof/state_serialization_contract.py",
        REPO / "tools/hrrr_single_domain_benchmark.py",
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"HRRR installed source helpers are missing: {missing}")
    # ``.as_posix()``, not ``str()``: this dict is written into the
    # prepared cache's ``source_identity`` and into the proof beside it,
    # and the forecast reader looks those keys up by the forward-slash
    # names in ``_HRRR_DECODE_SOURCES``.  ``str()`` of a relative Path
    # emits backslashes on Windows, which sealed caches keyed
    # ``woof\hrrr_forecast.py`` and made the reader refuse all ten.
    source_sha256 = {
        path.relative_to(REPO).as_posix(): _sha256(path) for path in paths
    }
    # Three real installs, three identities -- resolved in one shared
    # place (woof.runtime_manifest.provenance).  This used to be a
    # sealed-manifest branch and an unguarded `git rev-parse` with the
    # working directory set to REPO, which for every pip install is
    # site-packages: `CalledProcessError: returned non-zero exit status
    # 128`, before a single byte of the user's data was read.  A wheel
    # has an identity -- the distribution version and the digests pip
    # wrote into RECORD -- and that is what it binds now.
    identity = runtime_manifest.provenance(REPO)
    manifest_sha256 = identity.pop("distribution_manifest_sha256", None)
    if manifest_sha256 is not None:
        source_sha256["distribution/manifest.json"] = manifest_sha256
    for key in ("installed_wheel", "installed_editable"):
        bound = identity.pop(key, None)
        if bound is not None:
            identity[key] = bound
    return {**identity, "source_sha256": source_sha256}


def _physics_receipt(driver, cp) -> dict[str, object]:
    return {
        "resolved_lw_sw": [
            int(driver.ra_lw_physics), int(driver.ra_sw_physics)],
        "radiation_update_count": int(
            driver.radiation_callable.update_count),
        "microphysics_update_count": int(driver.microphysics_updates),
        "swdown_min_wm2": float(cp.min(driver.fields["swdown"]).get()),
        "swdown_max_wm2": float(cp.max(driver.fields["swdown"]).get()),
        "rainnc_max_mm": float(cp.max(driver.microphysics.rainnc).get()),
    }


def _atomic_json(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_strict_json(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _peak_rss_bytes() -> int:
    """Return process peak RSS on POSIX and Windows without a hard dependency."""

    try:
        import resource
    except ModuleNotFoundError:
        if os.name != "nt":
            return 0
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = (
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
        )
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        process = kernel32.GetCurrentProcess()
        if not psapi.GetProcessMemoryInfo(
                process, ctypes.byref(counters), counters.cb):
            raise OSError("GetProcessMemoryInfo failed")
        return int(counters.PeakWorkingSetSize)
    scale = 1 if sys.platform == "darwin" else 1024
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale)


# These are complete fixed runner products, not bare microphysics selectors.
# Validate the selected (last, or only) WRF domain before constructing the
# corresponding woof profile so a changed surrounding suite can never be
# accepted and silently ignored.  The historical WSM6/Thompson contracts are
# preserved byte-semantically; Morrison and NSSL use the accepted real74
# YSU/classic-MM5/Noah/KF/RTE+RRTMGP surrounding suite.
#: This runner's own registry route id.
ROUTE_ID = "tools.hrrr_single_domain_benchmark"

#: The templates THIS route declares, in its own declared order -- not the
#: shared single-domain menu.
#:
#: Every per-profile table below is keyed by profile id, and each one is
#: something only THIS route needs: a native WRF namelist contract to gate
#: the operator's namelist against, the initialization contract for the
#: species the analysis does not supply, and the switch-home map that
#: forwards a profile into an experiment payload.  Keying them off the
#: shared menu is what let another route's declaration grow the menu past
#: these tables: six composition suites reached ``--physics-profile`` as
#: offers and every one of them refused here with ``unsupported native
#: HRRR physics profile``.  The route declares what it can replay; this
#: reads that declaration.
NATIVE_BENCHMARK_PHYSICS_PROFILES = route_physics_profiles(ROUTE_ID)
THOMPSON_MYNN_RUC_MONTHLY_LEGACY_RRTMG_PROFILE_ID = (
    "thompson-mp8-mynn-mynn-ruc-monthly-rrtmg-legacy-v1"
)
THOMPSON_MYNN_RUC_MONTHLY_SOLAR_LEGACY_RRTMG_PROFILE_ID = (
    "thompson-mp8-mynn-mynn-ruc-monthly-solar-rrtmg-legacy-v1"
)
THOMPSON_MYNN_GSD41_PROFILE_ID = (
    "thompson-mp28-mynn-gsd41-mynn-ruc-rrtmg-legacy-v1"
)

_NATIVE_HRRR_NAMELIST_CONTRACTS = MappingProxyType({
    WSM6_PROFILE_ID: MappingProxyType({
        "physics": MappingProxyType({
            "mp_physics": 6.0,
            "ra_lw_physics": 0.0,
            "ra_sw_physics": 1.0,
            "radt": 1.0,
            "sf_sfclay_physics": 91.0,
            "sf_surface_physics": 2.0,
            "bl_pbl_physics": 1.0,
            "cu_physics": 0.0,
        }),
        "dynamics": MappingProxyType({
            "km_opt": 4.0,
            "diff_6th_opt": 2.0,
            "diff_6th_factor": 0.08,
            "diff_6th_slopeopt": 1.0,
        }),
    }),
    KESSLER_PROFILE_ID: MappingProxyType({
        "physics": MappingProxyType({
            "mp_physics": 1.0,
            "ra_lw_physics": 0.0,
            "ra_sw_physics": 1.0,
            "radt": 1.0,
            "sf_sfclay_physics": 91.0,
            "sf_surface_physics": 2.0,
            "bl_pbl_physics": 1.0,
            "cu_physics": 0.0,
        }),
        "dynamics": MappingProxyType({
            "km_opt": 4.0,
            "diff_6th_opt": 2.0,
            "diff_6th_factor": 0.08,
            "diff_6th_slopeopt": 1.0,
        }),
    }),
    MYNN_PROFILE_ID: MappingProxyType({
        "physics": MappingProxyType({
            "mp_physics": 6.0,
            "ra_lw_physics": 0.0,
            "ra_sw_physics": 1.0,
            "radt": 1.0,
            "sf_sfclay_physics": 5.0,
            "sf_surface_physics": 2.0,
            "bl_pbl_physics": 5.0,
            "cu_physics": 0.0,
            "num_soil_layers": 4.0,
        }),
        "dynamics": MappingProxyType({
            "km_opt": 4.0,
            "diff_6th_opt": 2.0,
            "diff_6th_factor": 0.08,
            "diff_6th_slopeopt": 1.0,
        }),
    }),
    # Each radiation-bearing MYNN twin carries its OWN row for the same
    # reason the legacy-RRTMG Thompson twin does (see the note further
    # down): this table is a hard per-field equality gate on the supplied
    # namelist, and the twins genuinely move the fields it pins --
    # ra_lw_physics 0 -> 4, ra_sw_physics 1 -> 4, radt 1.0 -> 12.0.  A
    # twin "simplified" to an alias of its sibling would pin the
    # sibling's radiation against the profile's own and refuse the
    # namelist one gate later.
    MYNN_RTE_RRTMGP_PROFILE_ID: MappingProxyType({
        "physics": MappingProxyType({
            "mp_physics": 6.0,
            "ra_lw_physics": 4.0,
            "ra_sw_physics": 4.0,
            "radt": 12.0,
            "sf_sfclay_physics": 5.0,
            "sf_surface_physics": 2.0,
            "bl_pbl_physics": 5.0,
            "cu_physics": 0.0,
            "num_soil_layers": 4.0,
        }),
        "dynamics": MappingProxyType({
            "km_opt": 4.0,
            "diff_6th_opt": 2.0,
            "diff_6th_factor": 0.08,
            "diff_6th_slopeopt": 1.0,
        }),
    }),
    MYNN_RUC_PROFILE_ID: MappingProxyType({
        "physics": MappingProxyType({
            "mp_physics": 6.0,
            "ra_lw_physics": 0.0,
            "ra_sw_physics": 1.0,
            "radt": 1.0,
            "sf_sfclay_physics": 5.0,
            "sf_surface_physics": 3.0,
            "bl_pbl_physics": 5.0,
            "cu_physics": 0.0,
            "num_soil_layers": 9.0,
        }),
        "dynamics": MappingProxyType({
            "km_opt": 4.0,
            "diff_6th_opt": 2.0,
            "diff_6th_factor": 0.08,
            "diff_6th_slopeopt": 1.0,
        }),
    }),
    MYNN_RUC_RTE_RRTMGP_PROFILE_ID: MappingProxyType({
        "physics": MappingProxyType({
            "mp_physics": 6.0,
            "ra_lw_physics": 4.0,
            "ra_sw_physics": 4.0,
            "radt": 12.0,
            "sf_sfclay_physics": 5.0,
            "sf_surface_physics": 3.0,
            "bl_pbl_physics": 5.0,
            "cu_physics": 0.0,
            "num_soil_layers": 9.0,
        }),
        "dynamics": MappingProxyType({
            "km_opt": 4.0,
            "diff_6th_opt": 2.0,
            "diff_6th_factor": 0.08,
            "diff_6th_slopeopt": 1.0,
        }),
    }),
    # The Thompson members of the two rows above, TRANSCRIBED rather than
    # aliased for the reason the Shin-Hong row gives: this table is a hard
    # per-field equality gate, and each moves one field it pins
    # (mp_physics 6.0 -> 8.0).  They differ from their WSM6 rows in that
    # field and no other.
    THOMPSON_MYNN_RUC_DUDHIA_PROFILE_ID: MappingProxyType({
        "physics": MappingProxyType({
            "mp_physics": 8.0,
            "ra_lw_physics": 0.0,
            "ra_sw_physics": 1.0,
            "radt": 1.0,
            "sf_sfclay_physics": 5.0,
            "sf_surface_physics": 3.0,
            "bl_pbl_physics": 5.0,
            "cu_physics": 0.0,
            "num_soil_layers": 9.0,
        }),
        "dynamics": MappingProxyType({
            "km_opt": 4.0,
            "diff_6th_opt": 2.0,
            "diff_6th_factor": 0.08,
            "diff_6th_slopeopt": 1.0,
        }),
    }),
    THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID: MappingProxyType({
        "physics": MappingProxyType({
            "mp_physics": 8.0,
            "ra_lw_physics": 4.0,
            "ra_sw_physics": 4.0,
            "radt": 12.0,
            "sf_sfclay_physics": 5.0,
            "sf_surface_physics": 3.0,
            "bl_pbl_physics": 5.0,
            "cu_physics": 0.0,
            "num_soil_layers": 9.0,
        }),
        "dynamics": MappingProxyType({
            "km_opt": 4.0,
            "diff_6th_opt": 2.0,
            "diff_6th_factor": 0.08,
            "diff_6th_slopeopt": 1.0,
        }),
    }),
    # Monthly surface fields change WRF namelist values, so this profile
    # has its own contract rather than borrowing the unselected surface
    # switches of the radiation sibling. The row binds the supplied
    # configuration; it does not certify a native forecast comparison.
    THOMPSON_MYNN_RUC_MONTHLY_LEGACY_RRTMG_PROFILE_ID: MappingProxyType({
        "physics": MappingProxyType({
            "mp_physics": 8.0,
            "ra_lw_physics": 4.0,
            "ra_sw_physics": 4.0,
            "radt": 12.0,
            "sf_sfclay_physics": 5.0,
            "sf_surface_physics": 3.0,
            "bl_pbl_physics": 5.0,
            "cu_physics": 0.0,
            "num_soil_layers": 9.0,
            "usemonalb": True,
            "rdlai2d": True,
            "fractional_seaice": 1.0,
        }),
        "dynamics": MappingProxyType({
            "km_opt": 4.0,
            "diff_6th_opt": 2.0,
            "diff_6th_factor": 0.08,
            "diff_6th_slopeopt": 1.0,
        }),
    }),
    # Solar albedo is a separate requested native fork switch. This row
    # binds the input composition and makes no native trajectory claim.
    THOMPSON_MYNN_RUC_MONTHLY_SOLAR_LEGACY_RRTMG_PROFILE_ID: MappingProxyType({
        "physics": MappingProxyType({
            "mp_physics": 8.0,
            "ra_lw_physics": 4.0,
            "ra_sw_physics": 4.0,
            "radt": 12.0,
            "sf_sfclay_physics": 5.0,
            "sf_surface_physics": 3.0,
            "bl_pbl_physics": 5.0,
            "cu_physics": 0.0,
            "num_soil_layers": 9.0,
            "usemonalb": True,
            "rdlai2d": True,
            "fractional_seaice": 1.0,
            "alb_sol": 1.0,
        }),
        "dynamics": MappingProxyType({
            "km_opt": 4.0,
            "diff_6th_opt": 2.0,
            "diff_6th_factor": 0.08,
            "diff_6th_slopeopt": 1.0,
        }),
    }),
    # This pins the declared source-version composition. It does not
    # certify an operational native trajectory or the remaining fork rows.
    THOMPSON_MYNN_GSD41_PROFILE_ID: MappingProxyType({
        "physics": MappingProxyType({
            "mp_physics": 28.0,
            "ra_lw_physics": 4.0,
            "ra_sw_physics": 4.0,
            "radt": 15.0,
            "sf_sfclay_physics": 5.0,
            "sf_surface_physics": 3.0,
            "bl_pbl_physics": 5.0,
            "bl_mynn_mixlength": 2.0,
            "scalar_pblmix": 1.0,
            "cu_physics": 0.0,
            "num_soil_layers": 9.0,
        }),
        "dynamics": MappingProxyType({
            "km_opt": 4.0,
            "diff_6th_opt": 2.0,
            "diff_6th_factor": 0.08,
            "diff_6th_slopeopt": 1.0,
        }),
    }),
    RUC_PROFILE_ID: MappingProxyType({
        "physics": MappingProxyType({
            "mp_physics": 6.0,
            "ra_lw_physics": 0.0,
            "ra_sw_physics": 1.0,
            "radt": 1.0,
            "sf_sfclay_physics": 91.0,
            "sf_surface_physics": 3.0,
            "bl_pbl_physics": 1.0,
            "cu_physics": 0.0,
            "num_soil_layers": 9.0,
        }),
        "dynamics": MappingProxyType({
            "km_opt": 4.0,
            "diff_6th_opt": 2.0,
            "diff_6th_factor": 0.08,
            "diff_6th_slopeopt": 1.0,
        }),
    }),
    NOAHMP_PROFILE_ID: MappingProxyType({
        "physics": MappingProxyType({
            "mp_physics": 6.0,
            "ra_lw_physics": 0.0,
            "ra_sw_physics": 1.0,
            "radt": 1.0,
            "sf_sfclay_physics": 91.0,
            "sf_surface_physics": 4.0,
            "bl_pbl_physics": 1.0,
            "cu_physics": 0.0,
            "num_soil_layers": 4.0,
        }),
        "dynamics": MappingProxyType({
            "km_opt": 4.0,
            "diff_6th_opt": 2.0,
            "diff_6th_factor": 0.08,
            "diff_6th_slopeopt": 1.0,
        }),
    }),
    MYNN_NOAHMP_PROFILE_ID: MappingProxyType({
        "physics": MappingProxyType({
            "mp_physics": 6.0,
            "ra_lw_physics": 0.0,
            "ra_sw_physics": 1.0,
            "radt": 1.0,
            "sf_sfclay_physics": 5.0,
            "sf_surface_physics": 4.0,
            "bl_pbl_physics": 5.0,
            "cu_physics": 0.0,
            "num_soil_layers": 4.0,
        }),
        "dynamics": MappingProxyType({
            "km_opt": 4.0,
            "diff_6th_opt": 2.0,
            "diff_6th_factor": 0.08,
            "diff_6th_slopeopt": 1.0,
        }),
    }),
    MYNN_NOAHMP_RTE_RRTMGP_PROFILE_ID: MappingProxyType({
        "physics": MappingProxyType({
            "mp_physics": 6.0,
            "ra_lw_physics": 4.0,
            "ra_sw_physics": 4.0,
            "radt": 12.0,
            "sf_sfclay_physics": 5.0,
            "sf_surface_physics": 4.0,
            "bl_pbl_physics": 5.0,
            "cu_physics": 0.0,
            "num_soil_layers": 4.0,
        }),
        "dynamics": MappingProxyType({
            "km_opt": 4.0,
            "diff_6th_opt": 2.0,
            "diff_6th_factor": 0.08,
            "diff_6th_slopeopt": 1.0,
        }),
    }),
    THOMPSON_PROFILE_ID: MappingProxyType({
        "physics": MappingProxyType({
            "mp_physics": 8.0,
            "ra_lw_physics": 0.0,
            "ra_sw_physics": 1.0,
            "radt": 1.0,
            "sf_sfclay_physics": 91.0,
            "sf_surface_physics": 2.0,
            "bl_pbl_physics": 1.0,
            "cu_physics": 0.0,
        }),
        "dynamics": MappingProxyType({
            "km_opt": 4.0,
            "diff_6th_opt": 2.0,
            "diff_6th_factor": 0.08,
            "diff_6th_slopeopt": 1.0,
        }),
    }),
    # The legacy-RRTMG Thompson twin carries its OWN namelist contract, not
    # an alias: this table is a hard per-field equality gate on the supplied
    # namelist, and the twin's registered composition genuinely differs from
    # the validation row here (ra 0/1 -> 4/4, radt 1.0 -> 12.0,
    # diff_6th_factor 0.08 -> 0.12; the values are the profile row's, which
    # is the battery config's as registered).  The NSSL-2 twin CAN alias
    # because its base row already pins the identical 4/4 + radt 12 values.
    # The species/cold-start tables below are the radiation-independent
    # part, and there the twin does alias -- see
    # _initialization_contract_profile.
    THOMPSON_LEGACY_RRTMG_PROFILE_ID: MappingProxyType({
        "physics": MappingProxyType({
            "mp_physics": 8.0,
            "ra_lw_physics": 4.0,
            "ra_sw_physics": 4.0,
            "radt": 12.0,
            "sf_sfclay_physics": 91.0,
            "sf_surface_physics": 2.0,
            "bl_pbl_physics": 1.0,
            "cu_physics": 0.0,
        }),
        "dynamics": MappingProxyType({
            "km_opt": 4.0,
            "diff_6th_opt": 2.0,
            "diff_6th_factor": 0.12,
            "diff_6th_slopeopt": 1.0,
        }),
    }),
    # The gray-zone sibling of the row above, TRANSCRIBED for the same
    # reason that row is not an alias: this table is a hard per-field
    # equality gate on the supplied namelist, and the two rows genuinely
    # differ (bl_pbl_physics 1 -> 11).  An alias would pin YSU against a
    # Shin-Hong namelist and refuse it one gate later on value drift.
    # They differ in that field and no other, which is the property the
    # paired comparison rests on.
    THOMPSON_SHINHONG_LEGACY_RRTMG_PROFILE_ID: MappingProxyType({
        "physics": MappingProxyType({
            "mp_physics": 8.0,
            "ra_lw_physics": 4.0,
            "ra_sw_physics": 4.0,
            "radt": 12.0,
            "sf_sfclay_physics": 91.0,
            "sf_surface_physics": 2.0,
            "bl_pbl_physics": 11.0,
            "cu_physics": 0.0,
        }),
        "dynamics": MappingProxyType({
            "km_opt": 4.0,
            "diff_6th_opt": 2.0,
            "diff_6th_factor": 0.12,
            "diff_6th_slopeopt": 1.0,
        }),
    }),
    MORRISON_PROFILE_ID: MappingProxyType({
        "physics": MappingProxyType({
            "mp_physics": 10.0,
            "morr_rimed_ice": 1.0,
            "ra_lw_physics": 4.0,
            "ra_sw_physics": 4.0,
            "radt": 12.0,
            "sf_sfclay_physics": 91.0,
            "sf_surface_physics": 2.0,
            "bl_pbl_physics": 1.0,
            "cu_physics": 1.0,
            "cudt": 5.0,
            "num_soil_layers": 4.0,
        }),
        "dynamics": MappingProxyType({
            "epssm": 0.5,
            "top_lid": False,
            "km_opt": 4.0,
            "diff_6th_opt": 2.0,
            "diff_6th_factor": 0.12,
            "diff_6th_slopeopt": 1.0,
        }),
    }),
    NSSL2_PROFILE_ID: MappingProxyType({
        "physics": MappingProxyType({
            "mp_physics": 18.0,
            "ra_lw_physics": 4.0,
            "ra_sw_physics": 4.0,
            "radt": 12.0,
            "sf_sfclay_physics": 91.0,
            "sf_surface_physics": 2.0,
            "bl_pbl_physics": 1.0,
            "cu_physics": 1.0,
            "cudt": 5.0,
            "num_soil_layers": 4.0,
        }),
        "dynamics": MappingProxyType({
            "epssm": 0.5,
            "top_lid": False,
            "km_opt": 4.0,
            "diff_6th_opt": 2.0,
            "diff_6th_factor": 0.12,
            "diff_6th_slopeopt": 1.0,
        }),
    }),
    # The Thompson legacy-RRTMG contract row with ONE field moved
    # (mp_physics 8.0 -> 50.0), TRANSCRIBED for the reason the Shin-Hong
    # row above states: this table is a hard per-field equality gate on
    # the supplied namelist, and an alias would pin mp=8 against a P3
    # namelist and refuse it one gate later on value drift.
    P3_LEGACY_RRTMG_PROFILE_ID: MappingProxyType({
        "physics": MappingProxyType({
            "mp_physics": 50.0,
            "ra_lw_physics": 4.0,
            "ra_sw_physics": 4.0,
            "radt": 12.0,
            "sf_sfclay_physics": 91.0,
            "sf_surface_physics": 2.0,
            "bl_pbl_physics": 1.0,
            "cu_physics": 0.0,
        }),
        "dynamics": MappingProxyType({
            "km_opt": 4.0,
            "diff_6th_opt": 2.0,
            "diff_6th_factor": 0.12,
            "diff_6th_slopeopt": 1.0,
        }),
    }),
})

_NATIVE_HRRR_RUNTIME_SWITCHES = MappingProxyType({
    profile: MappingProxyType(single_domain_runtime_switches(profile))
    for profile in NATIVE_BENCHMARK_PHYSICS_PROFILES
})

from woof.ingest.microphysics_cold_start import (
    cold_start_seeded_numbers, source_absent_microphysics,
)

# Compatibility views for old receipt callers. The species/default authority is
# shared and selector-based; production passes the actual RunConfig below.
_HRRR_SOURCE_ABSENT_STATE_DEFAULTS = MappingProxyType({
    profile: MappingProxyType(source_absent_microphysics(
        SimpleNamespace(**single_domain_runtime_switches(profile)))[1])
    for profile in NATIVE_BENCHMARK_PHYSICS_PROFILES
})
_HRRR_SOURCE_ABSENT_WRF_FIELDS = MappingProxyType({
    profile: source_absent_microphysics(
        SimpleNamespace(**single_domain_runtime_switches(profile)))[0]
    for profile in NATIVE_BENCHMARK_PHYSICS_PROFILES
})


#: Which profile's SPECIES/COLD-START tables serve each legacy-RRTMG twin.
#: The HRRR initialization contract -- which species the source supplies,
#: which absent WRF fields cold-start, and their exact FP32 defaults -- is
#: a microphysics property and is radiation-variant-independent: a twin
#: selects a different 4/4 radiation IMPLEMENTATION and changes nothing
#: about the analyzed or absent species (the same reasoning as
#: tools/prepare_hrrr_wrf.py's verbatim _HRRR_COLD_START_CONTRACT reuse;
#: the NSSL-2 twin was the in-tree precedent, aliased inline at three
#: sites until the Thompson twin missed all three).  The NAMELIST contract
#: is deliberately NOT served by this map for the Thompson twin: that
#: table pins radiation/radt/diffusion values the twins genuinely change,
#: so each twin either aliases there because its values are identical
#: (NSSL-2) or carries its own row (Thompson).
_INITIALIZATION_CONTRACT_ALIASES = MappingProxyType({
    NSSL2_LEGACY_RRTMG_PROFILE_ID: NSSL2_PROFILE_ID,
    THOMPSON_LEGACY_RRTMG_PROFILE_ID: THOMPSON_PROFILE_ID,
    # The engine twin of the row above: which species HRRR supplies and
    # what an absent one cold-starts to is a microphysics property, and
    # the twin moves the radiation engine only.  Its NAMELIST contract
    # aliases its own base rather than this one, because that table pins
    # radiation values the validation row does not carry.
    THOMPSON_RTE_RRTMGP_PROFILE_ID: THOMPSON_PROFILE_ID,
    # Same reasoning one component further: which species HRRR supplies
    # and what the absent ones cold-start to is a microphysics property,
    # and the gray-zone sibling changes the PBL closure, not the
    # microphysics.  Its NAMELIST contract is its own row above, because
    # that table does pin the switch it moves.
    THOMPSON_SHINHONG_LEGACY_RRTMG_PROFILE_ID: THOMPSON_PROFILE_ID,
    # The radiation-bearing MYNN twins, on exactly the same reasoning:
    # each selects a different radiation composition and changes nothing
    # about which species HRRR supplies or what an absent one cold-starts
    # to.  All three are WSM6, whose entry is the empty pair.
    MYNN_RTE_RRTMGP_PROFILE_ID: MYNN_PROFILE_ID,
    MYNN_RUC_RTE_RRTMGP_PROFILE_ID: MYNN_RUC_PROFILE_ID,
    MYNN_NOAHMP_RTE_RRTMGP_PROFILE_ID: MYNN_NOAHMP_PROFILE_ID,
    # The Thompson members of the MYNN + RUC pair: the species HRRR
    # supplies and the cold start of the absent ones are Thompson's, so
    # both read the Thompson validation row's tables.
    THOMPSON_MYNN_RUC_DUDHIA_PROFILE_ID: THOMPSON_PROFILE_ID,
    THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID: THOMPSON_PROFILE_ID,
    THOMPSON_MYNN_RUC_MONTHLY_LEGACY_RRTMG_PROFILE_ID: THOMPSON_PROFILE_ID,
    THOMPSON_MYNN_RUC_MONTHLY_SOLAR_LEGACY_RRTMG_PROFILE_ID: THOMPSON_PROFILE_ID,
})


def _initialization_contract_profile(profile: str) -> str:
    """The profile whose species/cold-start tables serve ``profile``."""
    profile = canonical_template_id(profile)
    return _INITIALIZATION_CONTRACT_ALIASES.get(profile, profile)


def _unsupported_profile(profile: str) -> ValueError:
    """The door refusal, naming the breakage AND the way out.

    This runner replays a native WRF run of the named composition and
    gates the operator's namelist field for field against a transcribed
    contract, so a suite with no native run behind it has nothing to be
    replayed against -- which is a real limit of THIS route and not of the
    engine.  The bare ``unsupported native HRRR physics profile 'x'`` said
    none of that: an operator who picked a registered, launchable suite
    met one line that named neither what was missing nor anywhere else to
    run it, having already paid for preparation.  The registry states the
    same refusal at plan review (``refused_template_ids``); this is the
    last line of the same one, and it carries the offered list because the
    offered list is this module's own and nothing upstream can print it.
    """

    return ValueError(
        f"unsupported native HRRR physics profile {profile!r}: this route "
        "replays a native WRF run of one immutable composition and gates "
        "it field for field against a transcribed namelist contract, so it "
        "offers only the suites that have such a run behind them "
        f"({', '.join(NATIVE_BENCHMARK_PHYSICS_PROFILES)}). Run this suite "
        "on tools.prepared_single_domain_forecast, which resolves every "
        "switch of it from the registry, or state it per domain on "
        "tools.prepared_domain_tree_forecast")


#: Which profile's NATIVE NAMELIST contract serves another.  A row
#: belongs here only when the two suites pin the SAME fields at the SAME
#: values in the namelist, because this table is a hard per-field equality
#: gate: an alias whose values differ would pin one suite's radiation
#: against the other's namelist and refuse it one gate later.  The two
#: rows below qualify for the same reason -- the namelist has no field for
#: the 4/4 radiation IMPLEMENTATION (``ra_rrtmg_variant`` is a woof
#: setting, not a WRF one), so an engine twin's namelist is its base's
#: namelist, field for field.  The Thompson legacy twin does NOT alias its
#: validation base, and the gray-zone sibling does not alias either: both
#: move fields this table pins.
#:
#: It was an inline conditional naming one pair, which is what a second
#: pair had to become a second conditional to join.
_NAMELIST_CONTRACT_ALIASES = MappingProxyType({
    NSSL2_LEGACY_RRTMG_PROFILE_ID: NSSL2_PROFILE_ID,
    THOMPSON_RTE_RRTMGP_PROFILE_ID: THOMPSON_LEGACY_RRTMG_PROFILE_ID,
})


def _native_hrrr_profile_contract(profile: str) -> dict[str, object]:
    profile = canonical_template_id(profile)
    contract_profile = _NAMELIST_CONTRACT_ALIASES.get(profile, profile)
    if contract_profile not in _NATIVE_HRRR_NAMELIST_CONTRACTS:
        raise _unsupported_profile(profile)
    return {
        section: dict(fields)
        for section, fields in _NATIVE_HRRR_NAMELIST_CONTRACTS[
            contract_profile].items()
    }


def _native_hrrr_runtime_switches(profile: str) -> dict[str, object]:
    profile = canonical_template_id(profile)
    try:
        return dict(_NATIVE_HRRR_RUNTIME_SWITCHES[profile])
    except KeyError:
        raise _unsupported_profile(profile) from None


def _guarded_launch_remedy(missing: tuple[str, ...]) -> str:
    """Name BOTH variables, with values, whichever one is missing.

    The guard is deliberate and unchanged: this runner's launch contract
    is that the operator states the experimental selection and the table
    root explicitly.  What was not deliberate is meeting it as two
    consecutive one-line refusals that named one variable each and
    carried no value -- a field run of the shipped 1.5.0 wheel set the
    first, re-ran, was refused for the second, and then had to work out
    the table root by hand having already downloaded it.

    So a single message names the pair and prints both export lines for
    the reader's platform.  Nothing is relaxed: an absent variable is
    still a refusal, and the root that gets set is still byte-validated
    against the pinned asset set before any GPU setup.
    """

    exports = "\n".join(f"    {line}"
                        for line in thompson_guard_exports())
    return (
        f"{THOMPSON_PROFILE_ID} is gated on {EXPERIMENTAL_THOMPSON_ENV}=1 "
        f"and {THOMPSON_TABLE_ROOT_ENV}; this process has neither"
        if len(missing) == 2 else
        f"{THOMPSON_PROFILE_ID} is gated on {EXPERIMENTAL_THOMPSON_ENV}=1 "
        f"and {THOMPSON_TABLE_ROOT_ENV}; this process is missing "
        f"{missing[0]}") + (
        ".\n  Export BOTH in the shell that starts the chain -- "
        "preparation launches this\n"
        "  runner as a subprocess and it inherits the environment:\n"
        f"{exports}\n"
        "  # the root above is this install's own resolution; every "
        "asset in it is\n"
        "  # size- and SHA-256-checked before GPU setup, so a wrong root "
        "still fails closed.")


def _thompson_runtime_authority() -> dict[str, object]:
    """Validate the guarded canonical Thompson table bytes before setup."""

    missing = tuple(
        name for name, satisfied in (
            (EXPERIMENTAL_THOMPSON_ENV,
             os.environ.get(EXPERIMENTAL_THOMPSON_ENV) == "1"),
            (THOMPSON_TABLE_ROOT_ENV,
             bool(os.environ.get(THOMPSON_TABLE_ROOT_ENV))),
        ) if not satisfied)
    if missing:
        raise RuntimeError(_guarded_launch_remedy(missing))
    raw_root = os.environ.get(THOMPSON_TABLE_ROOT_ENV)
    from woof.core.thompson_contract import (
        TABLE_SET_ID,
        WRF_REFERENCE_COMMIT,
        WRF_REFERENCE_VERSION,
        validate_table_assets,
    )

    root = Path(raw_root).resolve()
    assets = validate_table_assets(root)
    return {
        "experimental_guard": {
            "environment": EXPERIMENTAL_THOMPSON_ENV,
            "required_value": "1",
            "observed_value": "1",
        },
        "table_root": str(root),
        "table_set": TABLE_SET_ID,
        "wrf_reference_version": WRF_REFERENCE_VERSION,
        "wrf_reference_commit": WRF_REFERENCE_COMMIT,
        "assets": [
            {"filename": item.filename, "bytes": item.bytes,
             "sha256": item.sha256}
            for item in assets
        ],
    }


def _with_thompson_process_generation_authority(authority: dict, switches: dict) -> dict:
    if int(switches["mp_physics"]) != 28:
        return authority
    version = switches.get("thompson_version", "wrf_461")
    if version == "wrf_461":
        return authority
    if version != "wrf_39_noaa":
        raise ValueError(f"MP28 has no process table authority for thompson_version={version!r}")
    from woof.core.thompson_contract import (
        FORK_REFERENCE_SOURCE, TABLE_SETS_BY_VERSION, validate_table_assets,
    )
    from woof.physics_compat import thompson_fork_table_root

    _, expected_assets, table_set = TABLE_SETS_BY_VERSION[version]
    # Configuration preview declares the pinned process set without downloading
    # or compiling it. Actual fork execution acquires and byte-validates through
    # woof.thompson_fork_assets.ensure_thompson_fork_tables before first use.
    root = Path(thompson_fork_table_root()).resolve()
    assets = expected_assets
    return {
        "schema": authority["schema"], "mp_physics": 28,
        "thompson_version": version,
        "table_root": str(root), "table_set": table_set,
        "asset_validation": "deferred_to_runtime_before_first_use",
        "wrf_reference_version": "3.9-noaa-fork", "wrf_reference_source": FORK_REFERENCE_SOURCE,
        "assets": [{"filename": item.filename, "bytes": item.bytes, "sha256": item.sha256}
                   for item in assets],
        "classic_aerosol_authority": {
            **authority,
            "runtime_role": "The fork reads CCN activation from this validated classic root and process tables from the fork root.",
        },
    }


def _microphysics_table_authority(profile: str) -> dict[str, object] | None:
    """Stage and byte-validate the lookup tables THIS profile's mp reads.

    THE FIX FOR THE CONSTRAINT THAT KEPT THIS ROUTE'S DEFAULT ASYMMETRIC.
    Until 1.8 the only microphysics tables this route ever resolved were
    the ones :func:`_thompson_runtime_authority` resolves, and that
    function fires for exactly one profile id --
    ``thompson-mp8-ysu-mm5-noah-dudhia-daytime-v1``, the guarded evidence
    runtime -- behind two environment variables.  Every other mp8 suite,
    including BOTH legacy-RRTMG twins (the only full-radiation
    compositions this route's physics gate admits), reached GPU setup
    with no table resolution and no byte check on this route at all.  The
    wsm6 and Kessler families need no tables, so "the default is wsm6
    because it needs no staged tables" was true and self-perpetuating:
    the route could not default to full radiation because full radiation
    here means mp8, and mp8 was not staged.

    Staging is what removes it, not a relaxed guard.  The tables ship as
    package data (``woof_data/data/thompson/tables`` in the ``recast-woof-data``
    companion distribution, SHA-256 pinned) and the
    ladder that finds them -- env override, user staging, packaged root
    -- is the project's one resolver
    (:func:`woof.physics_compat.thompson_table_root`).  This calls it
    for ANY profile whose resolved ``mp_physics`` reads those tables, at
    profile-binding time, which on this route is before the fetch and
    before preprocessing; an install that never staged them is refused in
    one sentence naming ``woof fetch-tables`` rather than by a
    ``FileNotFoundError`` at the top of a paid GPU run.  The returned
    block goes into the physics receipt, so a reader can see WHICH bytes
    a prepared tree was built against.

    ``None`` for a profile whose microphysics reads no tables; the caller
    omits the key rather than writing an empty one.
    """

    switches = (_native_hrrr_runtime_switches(profile)
                if isinstance(profile, str) else asdict(profile))
    mp_physics = int(switches["mp_physics"])
    if mp_physics in (THOMPSON_MP_PHYSICS, 28):
        from woof.core.thompson_contract import (
            CLASSIC_TABLE_ASSETS,
            TABLE_SET_ID,
            WRF_REFERENCE_COMMIT,
            WRF_REFERENCE_VERSION,
            validate_table_assets,
        )
        from woof.table_assets import require_thompson_tables

        # Presence first, in a sentence naming ``woof fetch-tables``; then
        # exact bytes.  ``validate_table_assets`` defaults to the pinned
        # classic set and fails closed on an absent, resized or substituted
        # asset, so it IS the contract -- re-comparing its return value to
        # the same constant would only ever catch a test double.
        required_assets = CLASSIC_TABLE_ASSETS
        table_set = TABLE_SET_ID
        if mp_physics == 28:
            from woof.core.thompson_aerosol_contract import (
                AEROSOL_TABLE_ASSETS, AEROSOL_TABLE_SET_ID,
            )
            required_assets = (*CLASSIC_TABLE_ASSETS, *AEROSOL_TABLE_ASSETS)
            table_set = AEROSOL_TABLE_SET_ID
        root = Path(require_thompson_tables(assets=required_assets)).resolve()
        assets = (validate_table_assets(root, required_assets)
                  if mp_physics == 28 else validate_table_assets(root))
        authority = {
            "schema": "gpuwm-prepared-microphysics-table-authority-v1",
            "mp_physics": mp_physics,
            "table_root": str(root),
            "table_set": table_set,
            "wrf_reference_version": WRF_REFERENCE_VERSION,
            "wrf_reference_commit": WRF_REFERENCE_COMMIT,
            "assets": [
                {"filename": item.filename, "bytes": item.bytes,
                 "sha256": item.sha256}
                for item in assets
            ],
        }
        return _with_thompson_process_generation_authority(authority, switches)
    if mp_physics == 50:
        # P3's single lookup table ships INSIDE the woof wheel and its
        # loader (woof.core.p3_tables) is the byte authority; p3_init
        # READS the file and computes nothing, so an absent or modified
        # copy would otherwise surface as a hard stop at the top of a
        # paid GPU run.  The loader's own validator runs here, at
        # profile binding, the same stage the Thompson arm above chose
        # for the same reason.
        from woof.core.p3_tables import (
            TABLE_1_2MOM_ASSET,
            _validate_asset_bytes,
            p3_table_root,
        )

        root = Path(p3_table_root()).resolve()
        _validate_asset_bytes(root / TABLE_1_2MOM_ASSET.filename,
                              TABLE_1_2MOM_ASSET)
        return {
            "schema": "gpuwm-prepared-microphysics-table-authority-v1",
            "mp_physics": 50,
            "table_root": str(root),
            "table_set": "wrf-v4.6.1-p3-lookuptable1-2momi-v1",
            "wrf_reference_version": "v4.6.1",
            "wrf_reference_commit":
                "d66e442fccc04111067e29274c9f9eaccc3cef28",
            "assets": [
                {"filename": TABLE_1_2MOM_ASSET.filename,
                 "bytes": TABLE_1_2MOM_ASSET.size,
                 "sha256": TABLE_1_2MOM_ASSET.sha256},
            ],
        }
    return None


def _configured_physics_receipt(cfg, profile=None, *, acknowledgements=()):
    from woof.hrrr_configuration import resolved_run_settings
    from woof.physics_compat import single_domain_physics_selection
    fields, defaults = source_absent_microphysics(cfg)
    resolved = resolved_run_settings(cfg)
    selection = single_domain_physics_selection(cfg, profile=profile,
        expert_acknowledgements=tuple(acknowledgements))
    receipt = {
        "schema": "gpuwm-prepared-physics-profile-v1", "profile": profile,
        "front_door_selection": selection, "selection": "configured d01",
        # The registry's 4/4 radiation option names the spectral pair
        # ('rte-rrtmgp'); the engine that computes it is ra_rrtmg_variant.
        "radiation_engine": getattr(cfg, "ra_rrtmg_variant", None),
        "resolved": resolved,
        "hrrr_initialization_contract": {
            "analyzed_mass_fields": ["QC", "QR", "QI", "QS", "QG"],
            "source_absent_fields": list(fields),
            "source_absent_number_fields": [name for name in fields if name.startswith("QN")],
            "source_absent_state_defaults_fp32": defaults,
            "number_moment_policy": "exact active-scheme FP32 cold-start values",
        },
    }
    tables = _microphysics_table_authority(cfg)
    if tables is not None:
        receipt["microphysics_table_authority"] = tables
    return receipt


def _validate_native_hrrr_physics_profile(
        path: Path, profile: str = ROUTE_DEFAULT_PHYSICS_PROFILE, *,
        expert_acknowledgements: tuple[str, ...] = (),
        ) -> dict[str, object]:
    """Bind the namelist to one explicit HRRR runner profile.

    This preparer prepares d01.  It therefore reads d01's column entry --
    the FIRST value of every WRF per-domain array -- for every key it
    validates, and for a max_dom = 1 namelist that is simply the only
    value.

    It used to read the LAST value, on the reasoning that the historical
    two-domain input's inner domain was what this standalone controller
    represented.  On the public nested route that is the wrong domain and
    it fails in the most confusing way available: given a two-domain
    namelist whose d01 states the profile's ``diff_6th_factor = 0.08`` and
    whose d02 states a nest's 0.10, it read 0.10 and refused with a
    sentence about the value, never mentioning that it had been looking at
    d02.  A refusal on a column that is not uniform now says so and names
    the fix.
    """

    profile = canonical_template_id(profile)
    from woof.namelist_import import (
        MULTI_DOMAIN_ROOT_VIEW_HINT,
        parse_namelist,
        root_domain_namelist_view,
    )

    front_door_selection = validate_single_domain_physics_profile(
        profile, expert_acknowledgements=expert_acknowledgements,
        acknowledgement_provenance={
            value: ["--ack"] for value in expert_acknowledgements
        })
    expected_profile = _native_hrrr_profile_contract(profile)
    sections = parse_namelist(path)
    root_view, nonuniform_columns = root_domain_namelist_view(
        sections, source=str(path))
    selected: dict[str, dict[str, float | bool]] = {}
    for section_name, expected_fields in expected_profile.items():
        section = root_view.get(section_name)
        if section is None:
            raise ValueError(
                f"native HRRR profile {profile!r} requires "
                f"&{section_name}")
        selected[section_name] = {}
        for key, expected in expected_fields.items():
            if key not in section:
                raise ValueError(
                    f"native HRRR profile {profile!r} requires "
                    f"&{section_name}/{key}")
            raw = section[key]
            if isinstance(expected, bool):
                if not isinstance(raw, (bool, np.bool_)):
                    raise ValueError(
                        f"native HRRR &{section_name}/{key} must be logical")
                actual: float | bool = bool(raw)
                matches = actual is expected
                expected_text = str(expected)
            else:
                if isinstance(raw, bool) or not isinstance(
                        raw, (int, float, np.integer, np.floating)):
                    raise ValueError(
                        f"native HRRR &{section_name}/{key} must be numeric")
                actual = float(raw)
                matches = math.isfinite(actual) and actual == expected
                expected_text = f"{expected:g}"
            if not matches:
                column = f"&{section_name}/{key}"
                message = (
                    f"native HRRR profile {profile!r} drift at "
                    f"{column}: d01's value must be {expected_text}, got "
                    f"{raw!r}")
                if column in nonuniform_columns:
                    message += (
                        f".  {MULTI_DOMAIN_ROOT_VIEW_HINT}.  Columns that "
                        "differ across domains here: "
                        + ", ".join(nonuniform_columns))
                raise ValueError(message)
            selected[section_name][key] = actual
    if profile == THOMPSON_MYNN_GSD41_PROFILE_ID:
        # The source spelling and aerosol source are part of this version,
        # not optional namelist fields to silently replace with its preset.
        for key, expected in {"bl_mynn_tkebudget": 0,
                              "use_aero_icbc": True}.items():
            raw = root_view["physics"].get(key)
            if type(raw) is not type(expected) or raw != expected:
                raise ValueError(
                    f"native HRRR profile {profile!r} requires "
                    f"&physics/{key}={expected!r}, got {raw!r}")
            selected["physics"][key] = raw
        wif_input = root_view.get("domains", {}).get("wif_input_opt")
        if type(wif_input) is not int or wif_input != 1:
            raise ValueError(
                f"native HRRR profile {profile!r} requires "
                f"&domains/wif_input_opt=1, got {wif_input!r}")
        selected["domains"] = {"wif_input_opt": wif_input}
    if profile in (NSSL2_PROFILE_ID, NSSL2_LEGACY_RRTMG_PROFILE_ID):
        physics_section = root_view["physics"]
        nssl_values: dict[str, int | float] = {}
        selector_names = {
            "nssl_2moment_on", "nssl_hail_on", "nssl_ccn_on",
            "nssl_density_on", "nssl_3moment",
        }
        for key, default in NSSL2_WRF_NAMELIST_DEFAULTS.items():
            # d01's column entry, for the same reason as the loop above.
            raw = physics_section.get(key, default)
            if isinstance(default, int):
                if isinstance(raw, bool) or not isinstance(
                        raw, (int, np.integer)):
                    raise ValueError(
                        f"native HRRR NSSL-2 &physics/{key} must be an "
                        "integer selector")
                actual = int(raw)
            else:
                if isinstance(raw, bool) or not isinstance(
                        raw, (int, float, np.integer, np.floating)):
                    raise ValueError(
                        f"native HRRR NSSL-2 &physics/{key} must be numeric")
                actual = float(raw)
                if not math.isfinite(actual):
                    raise ValueError(
                        f"native HRRR NSSL-2 &physics/{key} must be finite")
            if key not in selector_names and actual != default:
                raise ValueError(
                    f"native HRRR NSSL-2 fixed preset requires "
                    f"&physics/{key}={default!r}, got {raw!r}")
            nssl_values[key] = actual
        mode = resolve_nssl2_mode(**{
            key: int(nssl_values[key]) for key in selector_names})
        if mode != NSSL2_DEFAULT_MODE:
            raise ValueError(
                "native HRRR NSSL-2 fixed preset does not implement the "
                f"requested optional selectors: resolved={asdict(mode)}, "
                f"required={asdict(NSSL2_DEFAULT_MODE)}")
        selected["physics"].update(nssl_values)
    resolved = _native_hrrr_runtime_switches(profile)
    resolved["radiation_scheme_ids"] = [
        resolved["ra_lw_physics"], resolved["ra_sw_physics"]]
    contract_profile = _initialization_contract_profile(profile)
    source_absent_defaults = _HRRR_SOURCE_ABSENT_STATE_DEFAULTS[
        contract_profile]
    receipt = {
        "schema": "gpuwm-prepared-physics-profile-v1",
        "profile": profile,
        "front_door_selection": front_door_selection,
        "selection": "d01 (first-or-only) WRF domain value",
        "resolved": resolved,
        "validated_namelist": selected,
        "hrrr_initialization_contract": {
            "analyzed_mass_fields": ["QC", "QR", "QI", "QS", "QG"],
            "mass_policy": (
                "WRF-Q linear pressure interpolation with zero surface "
                "pseudo-level; retain analyzed nonnegative FP32 masses"),
            "source_absent_fields": list(
                _HRRR_SOURCE_ABSENT_WRF_FIELDS[contract_profile]),
            # Backward-compatible narrow view retained for Thompson/older
            # receipt consumers; source_absent_fields is the full inventory.
            "source_absent_number_fields": [
                name for name in _HRRR_SOURCE_ABSENT_WRF_FIELDS[
                    contract_profile]
                if name.startswith("QN")
            ],
            "source_absent_state_defaults_fp32": dict(
                source_absent_defaults),
            "number_moment_policy": (
                "exact profile-defined FP32 cold-start values when absent "
                "from HRRR"
                if source_absent_defaults else "not applicable"),
        },
    }
    # Every profile that READS microphysics tables stages and validates
    # them here, whatever its radiation pairing -- see
    # _microphysics_table_authority.  The guarded evidence runtime below
    # keeps its own two-variable launch contract on top of this; it is a
    # launch contract for that one profile, not the route's table
    # resolution, and it no longer stands between this route and a
    # full-radiation default.
    table_authority = _microphysics_table_authority(profile)
    if table_authority is not None:
        receipt["microphysics_table_authority"] = table_authority
    if profile == THOMPSON_PROFILE_ID:
        receipt["readiness"] = "WRF_MATCHED_RUN_EXPERIMENTAL"
        receipt["readiness_scope"] = (
            "Legacy readiness identifier. The July matched run is historical "
            "and does not cover this current daytime-only suite.")
        receipt["thompson_contract"] = _thompson_runtime_authority()
    elif profile == MORRISON_PROFILE_ID:
        receipt["readiness"] = "WRF_MATCHED_RUN_RUNTIME_PROFILE"
        receipt["readiness_scope"] = (
            "Legacy readiness identifier for a composition exemption; no "
            "current matched-run manifest or decay tables cover this exact suite.")
        receipt["morrison_contract"] = {
            "selector": 10,
            "morr_rimed_ice": 1,
            "rimed_ice_category": "hail",
        }
    elif profile in (THOMPSON_LEGACY_RRTMG_PROFILE_ID,
                     THOMPSON_SHINHONG_LEGACY_RRTMG_PROFILE_ID):
        # Registry maturity ``wrf-matched-run-candidate``, and no env
        # launch contract: these resolve their tables through the staged
        # authority above, which is what lets one of them be this
        # route's default.  Stating the readiness is part of that -- a
        # default whose receipt carries no readiness key is a default
        # nobody can audit.
        receipt["readiness"] = "WRF_MATCHED_RUN_CANDIDATE"
    elif profile in (NSSL2_PROFILE_ID, NSSL2_LEGACY_RRTMG_PROFILE_ID):
        receipt["readiness"] = "WRF_MATCHED_RUN_CANDIDATE"
        # This preset pins the default lane -- the selector check above
        # refuses anything else -- but the receipt is built by the one
        # shared emitter so every NSSL receipt in the tree reads the same
        # way and none of them can drift into describing a mode that did
        # not run.
        receipt["nssl2_contract"] = nssl2_contract_receipt(
            NSSL2_DEFAULT_MODE)
    elif profile in (
            KESSLER_PROFILE_ID, MYNN_PROFILE_ID, MYNN_RUC_PROFILE_ID,
            RUC_PROFILE_ID,
            NOAHMP_PROFILE_ID, MYNN_NOAHMP_PROFILE_ID,
            # P3: the port is oracle-measured but no receipt covers the
            # composed suite; the registry template says the same.
            P3_LEGACY_RRTMG_PROFILE_ID,
            # The route default's engine twin, at its composition ceiling
            # for the same reason: every component is measured and no
            # receipt covers the composed suite on this engine.
            THOMPSON_RTE_RRTMGP_PROFILE_ID,
            # The Thompson members of the MYNN + RUC pair, at the ceiling
            # MYNN and RUC set.
            THOMPSON_MYNN_RUC_DUDHIA_PROFILE_ID,
            THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID,
            THOMPSON_MYNN_RUC_MONTHLY_LEGACY_RRTMG_PROFILE_ID,
            THOMPSON_MYNN_RUC_MONTHLY_SOLAR_LEGACY_RRTMG_PROFILE_ID,
            THOMPSON_MYNN_GSD41_PROFILE_ID):
        receipt["readiness"] = "IMPLEMENTED_UNVERIFIED"
    if profile in (MORRISON_PROFILE_ID, NSSL2_PROFILE_ID,
                   THOMPSON_RTE_RRTMGP_PROFILE_ID):
        receipt["radiation_substitution"] = {
            "contract": WRF_RRTMG_TO_RTE_RRTMGP,
            "requested_wrf_scheme_ids": [4, 4],
            "resolved_gpuwm_scheme_ids": [4, 4],
            "resolved_gpuwm_solver": "RTE+RRTMGP",
        }
    elif profile in (NSSL2_LEGACY_RRTMG_PROFILE_ID,
                     THOMPSON_MYNN_GSD41_PROFILE_ID):
        receipt["radiation_identity"] = {
            "contract": WRF_RRTMG_LEGACY,
            "requested_wrf_scheme_ids": [4, 4],
            "resolved_gpuwm_scheme_ids": [4, 4],
            "resolved_gpuwm_solver": "legacy RRTMG",
        }
    return receipt


#: Where every switch a shipped physics profile declares has to land in
#: the experiment payload :func:`_experiment` builds: ``"shared"`` for the
#: hierarchy-wide ``[shared]`` table, ``"domain"`` for the root
#: ``[[domain]]`` table.
#:
#: This map exists because the literal payload below is not a contract.
#: ``ra_rrtmg_variant`` was added to the RRTMG-paired profile rows and
#: nothing here forwarded it, so a ``RunConfig`` built on this route kept
#: the ``"rte-rrtmgp"`` default; the paired-switch validator in
#: ``woof/config.py`` then refused the ratified legacy-RRTMG profile in
#: 0.7 s while its RTE sibling built fine purely because that sibling's
#: declared value happened to EQUAL the default.  A profile that grows a
#: switch this map does not name is now a named refusal here rather than
#: a silent default three layers down.
_PROFILE_SWITCH_HOMES = MappingProxyType({
    'oml_gamma': "shared",
    'oml_hml0': "shared",
    'oml_relaxation_time': "shared",
    'sf_ocean_physics': "shared",
    "aer_init_opt": "shared",
    "alb_sol": "shared",
    "bl_mynn_gsd41_unsquared_qtke": "shared",
    "bl_mynn_mixlength": "shared",
    # Admitted with the MYNN TKE-advection port (QKE_ADV scalar transport);
    # the HRRR profiles declare it false, as the operational namelists do.
    "bl_mynn_tkeadvect": "shared",
    "bl_mynn_version": "shared",
    "bl_pbl_physics": "shared",
    "cu_physics": "shared",
    "cudt_minutes": "shared",
    "diff_6th_factor": "domain",
    "diff_6th_opt": "shared",
    "diff_6th_slopeopt": "shared",
    "epssm": "shared",
    "fractional_seaice": "shared",
    "km_opt": "shared",
    "mosaic_lu": "shared",
    "mosaic_soil": "shared",
    "moist": "shared",
    "moist_cq": "shared",
    "morr_rimed_ice": "shared",
    "mp_physics": "shared",
    "mynn_sfclay_variant": "shared",
    "num_soil_layers": "shared",
    "ra_lw_physics": "shared",
    "ra_physics": "shared",
    "ra_rrtmg_variant": "shared",
    "ra_sw_physics": "shared",
    "radt": "domain",
    "rdlai2d": "shared",
    "sf_sfclay_physics": "shared",
    "sf_surface_physics": "shared",
    # Every template declares the urban component (none = 0) since the
    # urban canopy models joined the registry.
    "sf_urban_physics": "shared",
    "scalar_pblmix": "shared",
    "terrain_opt": "shared",
    "thompson_fork_snow_fall": "shared",
    "thompson_version": "shared",
    "top_lid": "shared",
    "usemonalb": "shared",
    "wif_input_opt": "shared",
    "wrf_rrtmg_compatibility": "shared",
    "wsm6_hail_opt": "shared",
})


def _forward_profile_switches(
        raw: dict[str, object], switches: Mapping[str, object]) -> None:
    """Carry EVERY switch the selected profile declares into ``raw``.

    The literal payload in :func:`_experiment` documents the common ones;
    this makes the forward complete.  Profiles do not all declare the same
    switch set (only the RRTMG-paired rows carry ``ra_rrtmg_variant``), so
    the forward is driven by the selected profile's own inventory rather
    than by a hand-maintained list that can fall behind it.
    """

    shared = raw["shared"]
    domain = raw["domain"][0]
    unmapped = sorted(set(switches) - set(_PROFILE_SWITCH_HOMES))
    if unmapped:
        raise ValueError(
            "the native HRRR experiment builder has no declared home for "
            f"physics-profile switch(es) {unmapped}; add them to "
            "_PROFILE_SWITCH_HOMES so they reach the RunConfig instead of "
            "silently taking its default")
    for name, value in switches.items():
        target = shared if _PROFILE_SWITCH_HOMES[name] == "shared" else domain
        target[name] = value


def _experiment_tables(
        vertical, *, run_seconds: float,
        start_time: datetime = datetime(2026, 7, 18),
        target: HrrrTargetDomain | None = None,
        physics_profile: str = ROUTE_DEFAULT_PHYSICS_PROFILE,
        history_interval_seconds: float = 300.0,
        usemonalb: bool | None = None, rdlai2d: bool | None = None):
    """The raw tables this route hands ``build_experiment``, plus the target.

    Split out of :func:`_experiment` so the preparation can PUBLISH the
    document it built rather than describing it.  A stage after this one
    -- the cycling DA driver -- has to supply an experiment config whose
    ``prepared_domain_config_identity`` equals the one baked into the
    prepared cache, and the only reliable way to supply it is to render
    these exact tables (:mod:`woof.experiment_document`).
    """

    target = target or HrrrTargetDomain.legacy_500x500()

    switches = _native_hrrr_runtime_switches(physics_profile)
    raw = {
        "experiment": {
            "name": target.name,
            "start_time": start_time,
            "run_seconds": float(run_seconds),
            "feedback": 0,
            "smooth_option": 0,
            "blend_width": 5,
            "spec_bdy_width": target.spec_bdy_width,
            "restart_interval_s": 0.0,
        },
        "projection": {
            "map_proj": "lambert",
            "ref_lat": target.ref_lat,
            "ref_lon": target.ref_lon,
            "truelat1": target.truelat1,
            "truelat2": target.truelat2,
            "stand_lon": target.stand_lon,
        },
        "shared": {
            "nz": target.nz, "ztop": 20000.0,
            "p_top": vertical.p_top,
            "eta_levels": list(map(float, vertical.eta_levels)),
            "hybrid_opt": vertical.hybrid_opt,
            "etac": vertical.etac, "base_temp": 290.0,
            "time_step_sound": 4, "epssm": switches["epssm"],
            "emdiv": 0.01,
            "hypsometric_opt": 2, "h_sca_adv_order": 5, "smdiv": 0.1,
            "top_lid": switches["top_lid"],
            "moist": switches["moist"],
            "moist_cq": switches["moist_cq"],
            "mp_physics": switches["mp_physics"],
            "morr_rimed_ice": switches["morr_rimed_ice"],
            "wsm6_hail_opt": switches["wsm6_hail_opt"],
            "moist_adv_opt": 1, "km_opt": switches["km_opt"],
            "c_s": 0.25,
            "diff_6th_opt": switches["diff_6th_opt"],
            "diff_6th_slopeopt": switches["diff_6th_slopeopt"],
            "diff_6th_thresh": 0.10,
            "w_damping": 1, "damp_opt": 3, "zdamp": 5000.0,
            "dampcoef": 0.2, "khdif": 0.0, "kvdif": 0.0,
            "spec_zone": target.spec_zone,
            "relax_zone": target.relax_zone,
            "terrain_opt": switches["terrain_opt"],
            "map_proj": 1,
            "sf_sfclay_physics": switches["sf_sfclay_physics"],
            "sf_surface_physics": switches["sf_surface_physics"],
            "bl_pbl_physics": switches["bl_pbl_physics"], "bldt": 0.0,
            "num_soil_layers": switches["num_soil_layers"],
            "ra_physics": switches["ra_physics"],
            "ra_lw_physics": switches["ra_lw_physics"],
            "ra_sw_physics": switches["ra_sw_physics"],
            "wrf_rrtmg_compatibility": switches[
                "wrf_rrtmg_compatibility"],
            "cu_physics": switches["cu_physics"],
            "cudt_minutes": switches["cudt_minutes"],
        },
        "domain": [{
            "grid_id": 1, "parent_id": 0,
            "i_parent_start": 1, "j_parent_start": 1,
            "parent_grid_ratio": 1, "parent_time_step_ratio": 1,
            "nx": target.nx, "ny": target.ny,
            # The three rational-clock keys, exactly as the experiment
            # TOML spells them; a 7.5 s root clock is 7 + 1/2, never a
            # float in a "time_step" the loader types as integer.
            "time_step": target.time_step_seconds,
            "time_step_fract_num": target.time_step_fract_num,
            "time_step_fract_den": target.time_step_fract_den,
            "dx": target.dx_m, "dy": target.dy_m,
            "history_interval_s": float(history_interval_seconds),
            "specified": True, "nested": False,
            "radt": switches["radt"],
            "radt_minutes": switches["radt"],
            "diff_6th_factor": switches["diff_6th_factor"],
            "spec_exp": 0.0,
        }],
    }
    _forward_profile_switches(raw, switches)
    # A frozen replay can state its historical surface controls explicitly.
    # None retains the selected source/profile's prescribed defaults.
    for name, value in (("usemonalb", usemonalb), ("rdlai2d", rdlai2d)):
        if value is not None:
            if type(value) is not bool:
                raise ValueError(f"{name} override must be boolean")
            raw["shared"][name] = value
    from woof.physics_source_defaults import (
        MP_SCOPED_RECIPE_SETTINGS, recipe_physics_defaults,
        recipe_root_defaults, with_recipe_root_defaults)
    defaults = recipe_physics_defaults("hrrr")
    for name, value in defaults.items():
        if name in ("usemonalb", "rdlai2d"):
            # 3e5245839, lane/ruc-evap-gap: prescribed monthly fields have
            # consumers in Noah (2) and RUC (3). Preserve each profile's
            # declared controls, including explicit false.
            if int(switches["sf_surface_physics"]) in (2, 3):
                raw["shared"].setdefault(name, value)
        elif name in MP_SCOPED_RECIPE_SETTINGS:
            # The fork Thompson generation exists for mp_physics = 28 only.
            if int(switches["mp_physics"]) in MP_SCOPED_RECIPE_SETTINGS[name]:
                raw["shared"][name] = value
        else:
            raw["shared"][name] = value
    with_recipe_root_defaults(
        raw["shared"], raw["domain"], recipe_root_defaults("hrrr"))
    if (target.dx_m == 3000.0 and target.dy_m == 3000.0
            and target.time_step_exact == Fraction(20)):
        # A 20 s native-grid recipe retains the full-source operational
        # acoustic bound: HRRR's 20/20/20 clock derives six substeps over
        # its entire 3 km source grid. This is a retained source bound,
        # not a fresh derivation from the smaller crop's map factors.
        raw["shared"]["time_step_sound"] = 6
        raw["shared"]["use_adaptive_time_step"] = False
    _declare_asymmetric_radiation(
        raw, switches, target=target, start_time=start_time,
        run_seconds=run_seconds)
    return raw, target


def _declare_asymmetric_radiation(
        raw: dict[str, object], switches: Mapping[str, object], *,
        target: HrrrTargetDomain, start_time: datetime,
        run_seconds: float) -> None:
    """Declare this route's asymmetric radiation, per claim, per guard.

    Eight of the thirteen profiles this route stages run
    ``ra_sw_physics 1`` (Dudhia) with ``ra_lw_physics 0`` -- the whole
    wsm6 no-radiation family, ``kessler-mp1-ysu-mm5-noah-dudhia-v1``
    and ``thompson-mp8-ysu-mm5-noah-dudhia-daytime-v1``.  1.7.1's
    nocturnal-radiation guard refuses that pairing at config load for
    any window that includes local night
    (:func:`woof.physics_compat.nocturnal_radiation_refusal`), which is
    correct and is why the declaration is written HERE, where this
    route's physics is chosen, rather than left to a caller who has no
    config file to write it in: this preparation builds its experiment
    in code, so the code is the only place the config-side declaration
    can be spelled.  It is the same declaration the shipped configs
    carry (``configs/gfs_wrf_hierarchy_proof.toml`` and the LES cases,
    1.7.1) and it reaches the published authority in ink -- the
    experiment document this route publishes carries the line, so a
    downstream reader sees the declaration rather than inheriting it
    silently.

    WHAT CHANGED IN 1.8: this used to fire for the route's own DEFAULT.
    The default was :data:`WSM6_PROFILE_ID` -- asymmetric -- so a
    preparation that named no profile got the declaration written on the
    operator's behalf, and a declaration written by silence declares
    nothing.  The default is now
    :data:`woof.hrrr_route_inputs.ROUTE_DEFAULT_PHYSICS_PROFILE`, a 4/4
    suite, so this function returns without writing anything unless an
    operator EXPLICITLY selected an asymmetric profile.  Auto-declaration
    by silence died with the default that needed it; the eight asymmetric
    profiles stay fully selectable and still get the declaration, which
    is now what it was always supposed to mean -- a deliberate choice,
    recorded.

    TWO CLAIMS, TWO CONDITIONS.  The pairing raises two separate
    questions and each token answers exactly one of them, so each is
    written under its own condition rather than under a shared one:

    * :data:`ASYMMETRIC_RADIATION_NOCTURNAL_ACK` -- "this window has
      night in it".  Written only when the resolved pairing is
      asymmetric AND :func:`first_local_night_time` finds night inside
      the window, matching the wizard's emission rule exactly
      (:func:`woof.domain_wizard.render_config`).  An all-daylight
      window carries nothing, so this token in a published document
      always means a real nocturnal asymmetric run.
    * :data:`CONSTANT_DOWNWARD_LONGWAVE_ACK` -- "this flux is
      fabricated".  Written whenever
      :func:`woof.physics_compat.downward_longwave_disposition`
      classifies the resolved selectors as ``consumed`` or
      ``published``, which is EXACTLY the set
      :func:`woof.physics_compat.constant_longwave_refusal` refuses.
      The sun has nothing to do with it: a fixed 300 W m-2 under a land
      surface is fabricated at noon just as it is at midnight.

    THE BUG THIS FIXES (1.8.8 round 2).  Both tokens used to be written
    below the night check, so an ALL-DAYLIGHT window under any of the
    eight asymmetric profiles reached the constant-GLW guard with no
    declaration and refused at config build -- with no escape hatch,
    because this route synthesizes its config in code and the refusal
    correctly says the acknowledgement is config-side (not ``--ack``).
    Thirteen nodes of ``tests/test_hrrr_single_domain_benchmark.py``
    were red on it.  Reading the disposition classifier rather than
    re-deriving the rule here is the same discipline the guard, the
    initializer and the receipt already share: one decider, so this
    route cannot declare a different set from the one the door refuses.
    """

    # Same one-decider discipline, one level down: the split/combined
    # resolution is the engine's rule and this had a third hand-copy of
    # it, blind to the aggregate option's explicit -1 in both split keys.
    # No template this route declares sits on that option today, so this
    # is a swept sibling rather than a live defect; it is swept because
    # the copy is what made the live one on the prepared route possible.
    from woof.config import radiation_scheme_ids_from_settings

    lw, sw = radiation_scheme_ids_from_settings(switches)
    surface = int(switches.get("sf_surface_physics", 0))

    asymmetric = sw > 0 and lw == 0
    nocturnal = asymmetric and first_local_night_time(
        start_time, float(run_seconds),
        ref_lat=target.ref_lat, ref_lon=target.ref_lon) is not None
    kind, _consumer = downward_longwave_disposition(
        ra_lw_physics=lw, ra_sw_physics=sw, sf_surface_physics=surface)
    fabricated = kind in ("consumed", "published")
    if not (nocturnal or fabricated):
        return

    experiment = raw["experiment"]
    assert isinstance(experiment, dict)          # built above, in this file
    declared = list(experiment.get("acknowledgements", ()))
    required = []
    if nocturnal:
        required.append(ASYMMETRIC_RADIATION_NOCTURNAL_ACK)
    if fabricated:
        required.append(CONSTANT_DOWNWARD_LONGWAVE_ACK)
    for acknowledgement in required:
        if acknowledgement not in declared:
            declared.append(acknowledgement)
    experiment["acknowledgements"] = declared


def _experiment(
        vertical, *, run_seconds: float,
        start_time: datetime = datetime(2026, 7, 18),
        target: HrrrTargetDomain | None = None,
        physics_profile: str = ROUTE_DEFAULT_PHYSICS_PROFILE,
        history_interval_seconds: float = 300.0,
        usemonalb: bool | None = None, rdlai2d: bool | None = None):
    from woof.experiment import build_experiment

    raw, resolved = _experiment_tables(
        vertical, run_seconds=run_seconds, start_time=start_time,
        target=target, physics_profile=physics_profile,
        history_interval_seconds=history_interval_seconds,
        usemonalb=usemonalb, rdlai2d=rdlai2d)
    return build_experiment(
        raw, f"programmatic:native-HRRR:{resolved.identity_sha256()}")


def _validate_resolved_hrrr_profile(exp, receipt: dict[str, object]) -> None:
    """Prove the generated GPUWM config is the profile that was validated."""

    from woof.config import radiation_scheme_ids

    cfg = exp.root.run
    resolved = receipt["resolved"]
    names = (
        "moist", "moist_cq", "mp_physics", "top_lid", "epssm", "radt",
        "morr_rimed_ice", "wsm6_hail_opt", "ra_physics",
        "wrf_rrtmg_compatibility", "cudt_minutes", "num_soil_layers",
        "terrain_opt",
        "sf_sfclay_physics", "sf_surface_physics", "bl_pbl_physics",
        "cu_physics", "km_opt", "diff_6th_opt", "diff_6th_factor",
        "diff_6th_slopeopt",
    )
    observed = {name: getattr(cfg, name) for name in names}
    expected = {name: resolved[name] for name in names}
    observed["ra_lw_physics"], observed["ra_sw_physics"] = (
        radiation_scheme_ids(cfg))
    expected["ra_lw_physics"] = resolved["ra_lw_physics"]
    expected["ra_sw_physics"] = resolved["ra_sw_physics"]
    if observed != expected:
        raise RuntimeError(
            "generated native HRRR physics differs from the selected "
            f"profile: observed={observed}, expected={expected}")


def _initial_hrrr_microphysics_receipt(
        state, profile: str,
        initialization: object = None) -> dict[str, object]:
    """Gate source-to-state mass identity plus exact scheme cold start.

    The emitted receipt also states its own evidentiary strength.  The
    Current producers bind every positive decoded source sample to either a
    target-influencing WRF stencil or an explicit WRF column/support
    exclusion.  The legacy v1 correspondence retains its original broad
    source-nonzero/state-zero refusal so old in-process callers neither gain
    nor lose admission through a schema reinterpretation.
    """
    profile = canonical_template_id(profile)
    from woof.ingest.real import (
        HRRR_HYDROMETEOR_CORRESPONDENCE_SCHEMA_V1,
        HRRR_HYDROMETEOR_CORRESPONDENCE_SCHEMA_V2,
        array_correspondence_fingerprint,
        validate_hrrr_hydrometeor_vertical_disposition,
    )

    def scalar(value):
        if hasattr(value, "get"):
            value = value.get()
        return np.asarray(value).item()

    if not isinstance(initialization, Mapping):
        raise ValueError(
            "native HRRR analyzed hydrometeor receipt lacks decoded-source "
            "to initialized-state correspondence evidence")
    correspondence_schema = initialization.get("schema")
    if correspondence_schema not in {
            HRRR_HYDROMETEOR_CORRESPONDENCE_SCHEMA_V1,
            HRRR_HYDROMETEOR_CORRESPONDENCE_SCHEMA_V2}:
        raise ValueError(
            "native HRRR analyzed hydrometeor correspondence schema is "
            "missing or unsupported")
    source_species = initialization.get("decoded_source_species")
    correspondence = initialization.get("retained_correspondence")
    initialized_species = initialization.get("initialized_state_species")
    discarded = initialization.get("discarded_source_species")
    if (
        not isinstance(source_species, Mapping)
        or not isinstance(correspondence, Mapping)
        or not isinstance(initialized_species, Mapping)
        or not isinstance(discarded, Mapping)
    ):
        raise ValueError(
            "native HRRR hydrometeor correspondence evidence is incomplete")
    expected_source = {"QC", "QR", "QI", "QS", "QG"}
    if set(source_species) != expected_source:
        raise ValueError(
            "native HRRR decoded hydrometeor receipt must bind exactly "
            f"{sorted(expected_source)}, got {sorted(source_species)}")
    if (
        set(correspondence) | set(discarded) != expected_source
        or set(correspondence) & set(discarded)
    ):
        raise ValueError(
            "native HRRR hydrometeor policy does not account for every "
            "decoded source species")

    masses = {}
    retained_source_fingerprints = {}
    live_by_source = {}
    for source_name, name in sorted(correspondence.items()):
        if not isinstance(name, str):
            raise ValueError(
                f"native HRRR correspondence for {source_name} is invalid")
        value = getattr(state, name, None)
        if value is None:
            raise ValueError(
                f"native HRRR profile {profile!r} omitted state.{name}")
        finite = bool(scalar((value == value).all()))
        minimum = float(scalar(value.min()))
        maximum = float(scalar(value.max()))
        if not finite or minimum < 0.0 or not math.isfinite(maximum):
            raise ValueError(
                f"native HRRR analyzed hydrometeor {name} is invalid")
        live = array_correspondence_fingerprint(value)
        expected_live = initialized_species.get(name)
        if live != expected_live:
            raise ValueError(
                "native HRRR analyzed hydrometeor source-to-state "
                f"correspondence failed for {source_name}->{name}: "
                "initialized state checksum, nonzero mask, or extrema "
                "changed")
        source = source_species.get(source_name)
        if not isinstance(source, Mapping):
            raise ValueError(
                f"native HRRR decoded-source fingerprint {source_name} "
                "is missing")
        for key in (
                "sha256", "nonzero_mask_sha256", "nonzero_count",
                "minimum", "maximum"):
            if key not in source:
                raise ValueError(
                    f"native HRRR decoded-source fingerprint {source_name} "
                    f"omits {key}")
        if (
            not math.isfinite(float(source["minimum"]))
            or not math.isfinite(float(source["maximum"]))
            or float(source["minimum"]) < 0.0
        ):
            raise ValueError(
                f"native HRRR decoded-source {source_name} extrema are "
                "invalid")
        if (
            correspondence_schema
            == HRRR_HYDROMETEOR_CORRESPONDENCE_SCHEMA_V1
            and
            int(source["nonzero_count"]) > 0
            and int(live["nonzero_count"]) == 0
        ):
            raise ValueError(
                "native HRRR analyzed hydrometeor source-to-state "
                f"correspondence lost all nonzero mass for "
                f"{source_name}->{name}")
        if maximum > float(source["maximum"]) * (1.0 + 4.0e-7):
            raise ValueError(
                f"native HRRR initialized {name} maximum exceeds decoded "
                f"{source_name} maximum")
        masses[name] = {
            **live,
            "decoded_source_field": source_name,
            "decoded_source": source,
        }
        retained_source_fingerprints[source_name] = source
        live_by_source[source_name] = live

    if (correspondence_schema
            == HRRR_HYDROMETEOR_CORRESPONDENCE_SCHEMA_V2):
        disposition = initialization.get("vertical_disposition")
        disposition_validation = \
            validate_hrrr_hydrometeor_vertical_disposition(
                retained_source_fingerprints, live_by_source, disposition)
    else:
        disposition = None
        disposition_validation = None

    retention_evidence = {}
    for source_name, name in sorted(correspondence.items()):
        source = retained_source_fingerprints[source_name]
        live = live_by_source[source_name]
        source_nonzero = int(source["nonzero_count"])
        state_nonzero = int(live["nonzero_count"])
        if disposition_validation is None:
            strength = "PROVEN" if source_nonzero > 0 else "VACUOUS"
            influencing_count = source_nonzero
            excluded_count = 0
            labels_sha256 = None
        else:
            disposition_species = disposition_validation["species"][source_name]
            strength = disposition_species["strength"]
            influencing_count = disposition_species[
                "target_influencing_source_count"]
            excluded_count = disposition_species["wrf_excluded_source_count"]
            labels_sha256 = disposition_species["labels_sha256"]
        if strength == "VACUOUS":
            reason = (
                "decoded source carried no nonzero mass: the retention "
                "gate had nothing to check and this species is not proven")
        elif strength == "WRF_EXCLUDED":
            reason = (
                "every positive decoded source sample was exhaustively "
                "partitioned into an explicit WRF column/support exclusion; "
                "no sample influenced a target state cell")
        elif strength == "PARTIALLY_WRF_EXCLUDED":
            reason = (
                "target-influencing decoded source mass was retained while "
                "the remaining positive samples were exhaustively assigned "
                "explicit WRF column/support exclusions")
        else:
            reason = (
                "every positive decoded source sample influenced a WRF "
                "target stencil and initialized state retained nonzero mass")
        retention_evidence[source_name] = {
            "state_field": name,
            "strength": strength,
            "reason": reason,
            "source_nonzero_count": source_nonzero,
            "source_maximum": float(source["maximum"]),
            "target_influencing_source_count": influencing_count,
            "wrf_excluded_source_count": excluded_count,
            "state_nonzero_count": state_nonzero,
            "state_maximum": float(live["maximum"]),
            "vertical_disposition_schema": (
                None if disposition_validation is None
                else disposition_validation["schema"]),
            "vertical_disposition_labels_sha256": labels_sha256,
        }
    for source_name, policy in sorted(discarded.items()):
        if (
            not isinstance(policy, Mapping)
            or policy.get("source") != source_species[source_name]
            or policy.get("policy") != (
                "discard-source-species-absent-from-active-moist-package")
            or not isinstance(policy.get("wrf_commit"), str)
            or not isinstance(policy.get("registry_citation"), str)
            or not isinstance(policy.get("real_citation"), str)
        ):
            raise ValueError(
                f"native HRRR discarded source species {source_name} lacks "
                "a source-bound WRF-real policy receipt")
    cfg = (SimpleNamespace(**_native_hrrr_runtime_switches(profile))
           if isinstance(profile, str) else profile)
    absent_wrf_fields, defaults = source_absent_microphysics(cfg)
    selection_label = profile if isinstance(profile, str) else f"mp_physics={cfg.mp_physics}"
    exact_fields = {}
    numbers = {}
    number_names = frozenset({
        "nc", "nr", "ni", "ns", "ng", "qndrop", "qnr", "qni",
        "qns", "qng", "qnh", "qnn", "nn", "nh",
    })
    # The Thompson numbers (nr and ni, and mp=28's nc) are not held to
    # "exact zero everywhere": that premise, real.exe leaving a
    # source-absent number at zero, was retired by A99's port of real.exe's
    # make_DropletNumber, make_RainNumber and make_IceNumber
    # (module_initialize_real.F:4829-4852), after which the old check
    # refused every native Thompson preparation whose analysis carried
    # rain or ice (A110).  They are held to that rule instead: the exact
    # allocation zero where the paired mass is zero, finite and above zero
    # where it is above zero, and exactly the seeded cell count the
    # cold-start closure's seed receipt records.
    seeded_numbers = cold_start_seeded_numbers(cfg)
    closure = (initialization.get("cold_start_moment_closure")
               if seeded_numbers else None)
    from woof.ingest.handover_numbers import closure_active as _hn1_active
    morrison_hn1 = int(cfg.mp_physics) == 10 and _hn1_active(cfg)
    for name, raw_expected in defaults.items():
        value = getattr(state, name, None)
        expected = np.float32(raw_expected)
        dtype = None if value is None else np.dtype(value.dtype)
        if morrison_hn1 and name == "nc":
            # Correct-mode HN1 (woof.ingest.handover_numbers): Morrison's
            # droplet number starts at the scheme's own NDCNST diagnostic
            # everywhere (morrison.cu nc3d = NDCNST*1e6/rho), not zero.
            if (value is None or dtype != np.dtype(np.float32)
                    or not bool(scalar(((value > 0) & (value < np.inf)).all()))):
                raise ValueError(
                    f"native HRRR selection {selection_label!r} Morrison nc must be "
                    "the finite, positive NDCNST droplet number under correct-mode "
                    "hydrometeor_handover_numbers")
            exact_fields[name] = {"ndcnst_droplet_number": True,
                                  "all_finite_above_zero": True}
            numbers[name] = {"ndcnst_droplet_number": True}
            continue
        if name in seeded_numbers:
            mass_name, seed_key = seeded_numbers[name]
            label = (f"native HRRR selection {selection_label!r} Thompson "
                     f"number moment {name}")
            mass = getattr(state, mass_name, None)
            if (value is None or dtype != np.dtype(np.float32)
                    or mass is None
                    or tuple(mass.shape) != tuple(value.shape)):
                raise ValueError(
                    f"{label} is missing, is not FP32, or is not shaped "
                    f"like its mass {mass_name}")
            has_mass = mass > 0
            if not bool(scalar((has_mass | (value == expected)).all())):
                raise ValueError(
                    f"{label} must be exact FP32 {float(expected)!r} where "
                    f"{mass_name} is zero")
            if not bool(scalar(
                    (~has_mass | ((value > 0) & (value < np.inf))).all())):
                raise ValueError(
                    f"{label} must be finite and above zero where "
                    f"{mass_name} is above zero, as real.exe seeds it")
            seeded_cells = int(scalar(has_mass.sum()))
            seed = (closure.get(seed_key)
                    if isinstance(closure, Mapping) else None)
            recorded = (seed.get("seeded_cells")
                        if isinstance(seed, Mapping) else None)
            if (not isinstance(closure, Mapping)
                    or closure.get("mp_physics") != int(cfg.mp_physics)
                    or isinstance(recorded, bool)
                    or recorded != seeded_cells):
                raise ValueError(
                    f"{label} holds {seeded_cells} seeded cell(s) where "
                    f"{mass_name} is above zero, but the cold-start "
                    f"closure's {seed_key} records {recorded!r}")
            exact_fields[name] = {
                "expected_float32": float(expected),
                "expected_uint32_bits": int(expected.view(np.uint32)),
                "all_exact_expected_where_mass_is_zero": True,
                "paired_mass_field": mass_name,
                "seed_receipt": seed_key,
                "seeded_cells": seeded_cells,
                "seeded_all_finite_above_zero": True,
            }
            numbers[name] = {
                "all_exact_zero_where_mass_is_zero": True,
                "paired_mass_field": mass_name,
                "seed_receipt": seed_key,
                "seeded_cells": seeded_cells,
            }
            continue
        matches = (
            value is not None
            and dtype == np.dtype(np.float32)
            and bool(scalar((value == expected).all()))
        )
        if not matches:
            raise ValueError(
                f"native HRRR selection {selection_label!r} source-absent state "
                f"{name} must initialize to exact FP32 "
                f"{float(expected)!r}")
        field_receipt = {
            "expected_float32": float(expected),
            "expected_uint32_bits": int(expected.view(np.uint32)),
            "all_exact_expected": True,
        }
        exact_fields[name] = field_receipt
        if name in number_names:
            numbers[name] = (
                {"all_exact_zero": True}
                if expected == np.float32(0.0)
                else {
                    "expected_float32": float(expected),
                    "all_exact_expected": True,
                }
            )
    if morrison_hn1:
        number_policy = (
            "correct-mode HN1 Morrison: nc at the NDCNST droplet number; "
            "QNRAIN/QNICE/QNSNOW/QNGRAUPEL exact zero where the paired mass "
            "is zero and closed from it where it is above zero")
    elif seeded_numbers:
        makers = {"nc": "make_DropletNumber", "nr": "make_RainNumber",
                  "ni": "make_IceNumber"}
        number_policy = (
            "real.exe Thompson cold start: "
            + "/".join(sorted(seeded_numbers))
            + " exact zero where the paired mass is zero and seeded by "
            + "/".join(makers[name] for name in sorted(seeded_numbers))
            + " where it is above zero")
    elif not isinstance(profile, str):
        number_policy = ("exact active-scheme source-absent FP32 initial values"
                         if defaults else "not applicable")
    elif profile == MORRISON_PROFILE_ID:
        number_policy = (
            "exact-zero Morrison QNRAIN/QNICE/QNSNOW/QNGRAUPEL; "
            "runtime-diagnosed nc begins at exact zero")
    elif profile in (NSSL2_PROFILE_ID, NSSL2_LEGACY_RRTMG_PROFILE_ID):
        number_policy = (
            "exact-zero NSSL number moments except WRF default predicted "
            "CCN qnn=408163264.0 # kg-1")
    else:
        number_policy = "not applicable"
    vacuous = sorted(
        name for name, evidence in retention_evidence.items()
        if evidence["strength"] == "VACUOUS")
    proven = sorted(
        name for name, evidence in retention_evidence.items()
        if evidence["strength"] == "PROVEN")
    partially_excluded = sorted(
        name for name, evidence in retention_evidence.items()
        if evidence["strength"] == "PARTIALLY_WRF_EXCLUDED")
    excluded = sorted(
        name for name, evidence in retention_evidence.items()
        if evidence["strength"] == "WRF_EXCLUDED")
    if partially_excluded or (excluded and (proven or partially_excluded)):
        overall = "PARTIALLY_WRF_EXCLUDED"
    elif excluded:
        overall = "WRF_EXCLUDED"
    elif not proven:
        overall = "VACUOUS"
    elif vacuous:
        overall = "PARTIALLY_VACUOUS"
    else:
        overall = "PROVEN"
    return {
        "schema": "gpuwm-hrrr-microphysics-initialization-v4",
        "source_mass_fields": ["QC", "QR", "QI", "QS", "QG"],
        "state_mass_fields": masses,
        "source_to_state_correspondence": dict(sorted(correspondence.items())),
        "retention_evidence": dict(sorted(retention_evidence.items())),
        "retention_evidence_summary": {
            "strength": overall,
            "proven_species": proven,
            "partially_excluded_species": partially_excluded,
            "excluded_species": excluded,
            "vacuous_species": vacuous,
            "statement": (
                "every retained species carried nonzero decoded source mass"
                if overall == "PROVEN" else
                "all positive decoded source samples were exhaustively "
                "accounted for by WRF column/support exclusions; no target "
                "retention claim was applicable"
                if overall == "WRF_EXCLUDED" else
                "retention is proven for target-influencing source mass; "
                "other positive samples were exhaustively accounted for by "
                "WRF column/support exclusions"
                if overall == "PARTIALLY_WRF_EXCLUDED" else
                "no retained species carried nonzero decoded source mass: "
                "this receipt proves nothing about analyzed-hydrometeor "
                "retention on this domain"
                if overall == "VACUOUS" else
                "retention is proven only for "
                f"{', '.join(proven)}; {', '.join(vacuous)} had no nonzero "
                "decoded source mass and are not proven"),
        },
        "vertical_disposition": disposition,
        "discarded_source_species": dict(discarded),
        "source_absent_wrf_fields": list(absent_wrf_fields),
        "source_absent_state_policy": (
            "active-scheme exact FP32 WRF-real-style cold start"),
        "state_source_absent_fields": exact_fields,
        "source_absent_number_policy": number_policy,
        "state_number_fields": numbers,
    }


def _load_static(
        cache: Path, receipt_path: Path,
        target: HrrrTargetDomain | None = None):
    from woof.hrrr_native_static import (require_static_coverage,
                                          verify_geog_source_evidence)

    target = target or HrrrTargetDomain.legacy_500x500()
    started = time.perf_counter()
    receipt = json.loads(receipt_path.read_text())
    accepted_schemas = {"gpuwm-native-hrrr-static-v2"}
    if target == HrrrTargetDomain.legacy_500x500():
        accepted_schemas.add("gpuwm-native-hrrr-static-500x500-v1")
    if (receipt.get("status") != "PASS"
            or receipt.get("schema") not in accepted_schemas):
        raise ValueError("native static receipt is not a supported PASS receipt")
    verify_geog_source_evidence(receipt)
    if receipt.get("schema") == "gpuwm-native-hrrr-static-v2":
        if receipt.get("target_domain_sha256") != target.identity_sha256():
            raise ValueError("native static target-domain identity mismatch")
        # A pinned static source (the HRRR route's default since
        # 6a69b356f) records its own grid window, not the HRRR crop; the
        # shared check reads which one this receipt must carry.
        require_static_coverage(receipt, target,
                                prefix="native static HRRR receipt")
    expected = receipt.get("cache", {}).get("sha256")
    actual = sha256_file(cache)
    if actual != expected:
        raise ValueError(
            f"native static cache SHA mismatch: expected {expected}, got {actual}")
    with np.load(cache, allow_pickle=False) as stored:
        fields = {name: np.asarray(stored[name], dtype=np.float64)
                  for name in stored.files}
    validation = validate_static(fields, target)
    expected_arrays = receipt.get("array_sha256", {})
    observed_arrays = {
        name: array_sha256(value) for name, value in sorted(fields.items())}
    if observed_arrays != expected_arrays:
        raise ValueError("native static array inventory differs from receipt")
    attrs = {
        "MMINLU": "MODIFIED_IGBP_MODIS_NOAH",
        "ISWATER": 17, "ISLAKE": 21, "ISICE": 15,
        "CEN_LAT": float(receipt["geometry"]["center_lat"]),
    }
    return fields, attrs, {
        "wall_seconds": time.perf_counter() - started,
        "cache_sha256": actual,
        "validation": validation,
    }


def _map_snapshot(
        snapshot, grid, static, mapping_report, *,
        surface_fallback_radius: int = 8, preprocess_backend="cuda"):
    from woof.ingest.hrrr import interpolate_hrrr_to_lambert

    started = time.perf_counter()
    met = interpolate_hrrr_to_lambert(
        snapshot, grid, target_landmask=static["LANDMASK"],
        soil_mapping_report=mapping_report,
        surface_fallback_radius=surface_fallback_radius,
        backend=preprocess_backend, target_name="domain 1")
    return met, time.perf_counter() - started


def _crop_horizontal_snapshot(snapshot, *, y0, y1, x0, x1,
                              full_shape, detach=False):
    """Crop one mapped C-grid rectangle without changing array values."""
    ny, nx = map(int, full_shape)
    y0, y1, x0, x1 = map(int, (y0, y1, x0, x1))
    if not (0 <= y0 < y1 <= ny and 0 <= x0 < x1 <= nx):
        raise ValueError("boundary rectangle lies outside the mapped domain")
    cropped = {}
    for name, value in snapshot.fields.items():
        shape = tuple(value.shape)
        if len(shape) < 2:
            raise ValueError(
                f"mapped HRRR field {name!r} has no horizontal dimensions")
        if shape[-2:] == (ny, nx):
            selected = value[..., y0:y1, x0:x1]
        elif shape[-2:] == (ny, nx + 1):
            selected = value[..., y0:y1, x0:x1 + 1]
        elif shape[-2:] == (ny + 1, nx):
            selected = value[..., y0:y1 + 1, x0:x1]
        else:
            raise ValueError(
                f"mapped HRRR field {name!r} has unsupported C-grid shape "
                f"{shape[-2:]}")
        if detach:
            # Spawned ProcessPool jobs must not inherit CuPy objects or the
            # MappingProxyType installed by HorizontalSnapshot.__post_init__.
            # A byte-exact host copy is both pickle-safe and releases the
            # full mapped hour's device allocation before worker launch.
            if hasattr(selected, "get"):
                selected = selected.get()
            selected = np.array(selected, copy=True, order="C", subok=False)
        cropped[name] = selected
    return SimpleNamespace(
        valid_time=snapshot.valid_time,
        levels_hpa=np.array(snapshot.levels_hpa, copy=True),
        fields=cropped,
        withheld_fields=dict(getattr(snapshot, "withheld_fields", None) or {}))


def _crop_boundary_static(static, *, y0, y1, x0, x1, full_shape):
    ny, nx = map(int, full_shape)
    result = {}
    for name in ("HGT_M", "LANDMASK", "MAPFAC_M", "MAPFAC_U", "MAPFAC_V",
                 "F", "E", "SINALPHA", "COSALPHA"):
        value = static[name]
        if value.shape == (ny, nx):
            result[name] = value[y0:y1, x0:x1]
        elif value.shape == (ny, nx + 1):
            result[name] = value[y0:y1, x0:x1 + 1]
        elif value.shape == (ny + 1, nx):
            result[name] = value[y0:y1 + 1, x0:x1]
        else:
            raise ValueError(
                f"static field {name!r} has unsupported C-grid shape "
                f"{value.shape}")
    return result


def _compact_boundary_inputs(met, dc, *, width):
    """Detach four compact slabs so the full mapped hour can be released."""
    ny, nx = dc.run.ny, dc.run.nx
    rectangles = {
        "west": (0, ny, 0, width),
        "east": (0, ny, nx - width, nx),
        "south": (0, width, 0, nx),
        "north": (ny - width, ny, 0, nx),
    }
    # WRF makes FLAG_SH's surface fallback decision once at the first valid
    # point, then applies it to the complete domain.  Preserve that global
    # decision when the four compact side rectangles are initialized in
    # isolation; otherwise each side could silently make a different choice.
    flag_sh_surface_fallback = bool(float(met.fields["Q2"][0, 0]) < 1.0e-6)
    # Keep this as a normal dict: the compact payload is intentionally
    # serializable so independent forcing hours can be initialized in spawned
    # worker processes.  MappingProxyType cannot cross that process boundary.
    result = {}
    for side, (y0, y1, x0, x1) in rectangles.items():
        strip = _crop_horizontal_snapshot(
            met, y0=y0, y1=y1, x0=x0, x1=x1,
            full_shape=(ny, nx), detach=True)
        strip.flag_sh_surface_fallback = flag_sh_surface_fallback
        result[side] = strip
    return result


def _boundary_mapping_targets(grid, static, run_cfg, *, width):
    """Build exact Lambert subgrids for the four specified-boundary strips."""
    from woof.static.lambert import LambertGrid

    ny, nx = int(run_cfg.ny), int(run_cfg.nx)
    rectangles = {
        "west": (0, ny, 0, width),
        "east": (0, ny, nx - width, nx),
        "south": (0, width, 0, nx),
        "north": (ny - width, ny, 0, nx),
    }
    landmask = np.asarray(static["LANDMASK"])
    if landmask.shape != (ny, nx):
        raise ValueError("target LANDMASK shape differs from the run domain")
    result = {}
    for side, (y0, y1, x0, x1) in rectangles.items():
        subgrid = LambertGrid(
            grid.ref_lat, grid.ref_lon, grid.truelat1, grid.truelat2,
            grid.stand_lon, grid.dx, grid.dy,
            (x1 - x0) + 1, (y1 - y0) + 1,
            known_x=grid.known_x - x0, known_y=grid.known_y - y0,
            moad_cen_lat=grid.moad_cen_lat,
            moad_cen_lon=grid.moad_cen_lon)
        result[side] = (
            subgrid, np.ascontiguousarray(landmask[y0:y1, x0:x1]))
    return result


def _detach_mapped_snapshot(met):
    """Copy one already-compact mapped snapshot to pickle-safe host arrays."""
    fields = {}
    for name, value in met.fields.items():
        if hasattr(value, "get"):
            value = value.get()
        fields[name] = np.array(value, copy=True, order="C", subok=False)
    return SimpleNamespace(
        valid_time=met.valid_time,
        levels_hpa=np.array(met.levels_hpa, copy=True),
        fields=fields,
        # A boundary strip that asks for a withheld field is refused with
        # the decoder's reason, as the start state is.
        withheld_fields=dict(getattr(met, "withheld_fields", None) or {}))


def _map_boundary_snapshot(
        snapshot, targets, mapping_report, *,
        surface_fallback_radius: int = 8, preprocess_backend="cuda",
        domain_grid=None):
    """Map only four boundary strips, avoiding a disposable full-domain hour.

    ``domain_grid`` is the domain the strips were cut from.  Each strip is
    mapped with it as its lattice parent, so the strips of a domain that
    IS the native HRRR grid take the identity route its start state took;
    without it they were interpolated and their edge points refused
    (acceptance D-05, 2.8.8: the full 1800 x 1060 grid died in its first
    boundary hour, "HRRR bridge window lacks the bilinear donor cell").
    """
    from woof.ingest.hrrr import interpolate_hrrr_to_lambert

    started = time.perf_counter()
    compact = {}
    side_reports = {}
    for side, (subgrid, target_landmask) in targets.items():
        side_report = {}
        met = interpolate_hrrr_to_lambert(
            snapshot, subgrid, target_landmask=target_landmask,
            soil_mapping_report=side_report,
            surface_fallback_radius=surface_fallback_radius,
            backend=preprocess_backend,
            # Four strips are mapped independently here, so a soil
            # refusal that named only "soil mapping" left a coastal
            # domain's owner with four identical-looking suspects.
            target_name=f"the {side} boundary strip of domain 1",
            lattice_parent=domain_grid)
        compact[side] = _detach_mapped_snapshot(met)
        side_reports[side] = side_report
        del met

    # WRF makes FLAG_SH's surface fallback decision once at the first global
    # target point.  West contains that point; apply its decision to all four
    # independently mapped strips just as _compact_boundary_inputs does.
    flag_sh_surface_fallback = bool(
        float(compact["west"].fields["Q2"][0, 0]) < 1.0e-6)
    for strip in compact.values():
        strip.flag_sh_surface_fallback = flag_sh_surface_fallback
    if len(mapping_report):
        raise ValueError("boundary mapping report must be empty")
    mapping_report.update({
        "policy": "direct exact Lambert boundary-strip interpolation",
        "sides": side_reports,
    })
    return compact, time.perf_counter() - started


def _compact_boundary_static(static, run_cfg, *, width, latlon_mass=None):
    """Copy the four static side rectangles once for worker initialization.

    ``latlon_mass`` is the domain's mass-point ``(lat2d, lon2d)``; each
    side then carries its own rectangle of it as ``XLAT_M``/``XLONG_M``,
    the geodesy :func:`woof.ingest.real.initialize_real` reads (through
    ``grid=``) to build an mp=28 strip's aerosol from the monthly WIF
    climatology.  Without it an mp=28 tree on this route asking for the
    climatology refused in every boundary worker ("could not derive the
    model mass-point latitudes/longitudes"), although the start state
    beside it had them.
    """
    ny, nx = run_cfg.ny, run_cfg.nx
    rectangles = {
        "west": (0, ny, 0, width),
        "east": (0, ny, nx - width, nx),
        "south": (0, width, 0, nx),
        "north": (ny - width, ny, 0, nx),
    }
    sides = {
        side: {
            name: np.ascontiguousarray(value)
            for name, value in _crop_boundary_static(
                static, y0=y0, y1=y1, x0=x0, x1=x1,
                full_shape=(ny, nx)).items()
        }
        for side, (y0, y1, x0, x1) in rectangles.items()
    }
    if latlon_mass is not None:
        latitude, longitude = (np.asarray(value, dtype=np.float64)
                               for value in latlon_mass)
        if latitude.shape != (ny, nx) or longitude.shape != (ny, nx):
            raise ValueError(
                "boundary strip geodesy must be the mass grid "
                f"({ny} x {nx}); got {latitude.shape} and {longitude.shape}")
        for side, (y0, y1, x0, x1) in rectangles.items():
            sides[side]["XLAT_M"] = np.ascontiguousarray(latitude[y0:y1, x0:x1])
            sides[side]["XLONG_M"] = np.ascontiguousarray(longitude[y0:y1, x0:x1])
    return sides


def _boundary_static_sides(static, grid, run_cfg, *, width):
    """The controller's boundary strip statics, geodesy included.

    The coordinates come from ``grid``, never from ``static``: a root whose
    statics come from a static source (``[static] source =
    "hrrr-conus-v4"``) carries no ``XLAT_M``/``XLONG_M`` in its field set,
    so a strip that looked there found nothing.  With no coordinates every
    mp=28 boundary worker refused the WIF climatology at forecast start
    ("could not derive the model mass-point latitudes/longitudes"),
    blocking every plain HRRR regional run on 2.8.6.  Only an mp=28 domain
    reads strip geodesy (its aerosol climatology); other schemes keep the
    small payload.
    """
    return _compact_boundary_static(
        static, run_cfg, width=width,
        latlon_mass=(grid.latlon_mass() if int(run_cfg.mp_physics) == 28
                     else None))


@preprocess_math_call
def _initialize_boundary_sides(
        compact_mets, run_cfg, static_sides, eta, *, p_top, width,
        preprocess_backend="cuda", preprocess_workers=None,
        cpu_preprocess_bridge=None, sfcp_to_sfcp=True,
        preprocess_selection=None):
    """Initialize one hour's four side rectangles on one explicit backend.

    ``preprocess_selection`` is the controller's record of how its
    backend was chosen.  A spawned worker resolves the backend again by
    name, which is the controller's choice and not a second one, so its
    receipt carries the controller's selection rather than "named by the
    caller" about a backend auto chose.
    """
    from woof.ingest.preprocess_backend import resolve_preprocess_backend

    from woof.core.grid import make_vertical_coord
    from woof.ingest.lateral_bc import (
        _coupled_device_fields, extract_lateral_side)
    from woof.ingest.real import initialize_real

    preprocess = resolve_preprocess_backend(
        preprocess_backend, workers=preprocess_workers,
        cpu_bridge=cpu_preprocess_bridge)
    if preprocess_selection is not None:
        preprocess.selection = dict(preprocess_selection)
    xp = preprocess.array_module
    is_cuda = getattr(preprocess, "name", None) == "cuda"

    ny, nx = run_cfg.ny, run_cfg.nx
    rectangles = {
        "west": (0, ny, 0, width),
        "east": (0, ny, nx - width, nx),
        "south": (0, width, 0, nx),
        "north": (ny - width, ny, 0, nx),
    }
    sides = {}
    timings = {}
    gpu_peak_used = 0
    pool_peak_total = 0

    def record_memory():
        nonlocal gpu_peak_used, pool_peak_total
        if is_cuda:
            free, total = xp.cuda.runtime.memGetInfo()
            gpu_peak_used = max(gpu_peak_used, int(total - free))
            pool_peak_total = max(
                pool_peak_total,
                int(xp.get_default_memory_pool().total_bytes()))

    stream = xp.cuda.Stream(non_blocking=True) if is_cuda else None
    with stream if stream is not None else nullcontext():
        for side, (y0, y1, x0, x1) in rectangles.items():
            started = time.perf_counter()
            strip_met = compact_mets[side]
            strip_static = static_sides[side]
            strip_cfg = replace(run_cfg, nx=x1 - x0, ny=y1 - y0)
            coord = make_vertical_coord(
                strip_cfg.nz, hybrid_opt=strip_cfg.hybrid_opt,
                etac=strip_cfg.etac, eta_levels=eta)
            # The strip's own mass-point geodesy, when the controller
            # shipped it (``_compact_boundary_static(latlon_mass=...)``):
            # an mp=28 domain's boundary rows carry nwfa/nifa
            # (woof.boundary_fields.AEROSOL_BOUNDARY_FIELDS), and under the
            # monthly WIF climatology initialize_real builds them from this
            # lat/lon, lazily and only on its mp=28 climatology branch, the
            # same door the start state takes with ``grid=grid``.  Every
            # other scheme reads none of it.
            strip_geodesy = (
                {"XLAT_M": strip_static["XLAT_M"],
                 "XLONG_M": strip_static["XLONG_M"]}
                if "XLAT_M" in strip_static else None)
            result = initialize_real(
                strip_met, strip_cfg, coord, strip_static["HGT_M"],
                landmask=strip_static["LANDMASK"],
                **({} if strip_geodesy is None else {"grid": strip_geodesy}),
                p_top=p_top, sfcp_to_sfcp=sfcp_to_sfcp,
                preprocess_backend=preprocess,
                state_backend="preprocess",
                boundary_only=True,
                flag_sh_surface_fallback=(
                    strip_met.flag_sh_surface_fallback),
                boundary_species=_native_boundary_species())
            result.state.set_map_coriolis(
                strip_static["MAPFAC_M"], strip_static["MAPFAC_U"],
                strip_static["MAPFAC_V"], strip_static["F"], strip_static["E"],
                sina=strip_static["SINALPHA"], cosa=strip_static["COSALPHA"])
            coupled_device = _coupled_device_fields(result.state)
            # No host extraction may race unfinished work on this stream.
            if stream is not None:
                stream.synchronize()
            coupled = {
                name: np.asarray(
                    value.get() if hasattr(value, "get") else value
                ).astype(np.float64, copy=False)
                for name, value in coupled_device.items()}
            sides[side] = dict(extract_lateral_side(coupled, side, width))
            timings[side] = time.perf_counter() - started
            record_memory()
            del coupled, coupled_device, result, strip_static
    if stream is not None:
        stream.synchronize()
    record_memory()
    memory = {
        # memGetInfo is device-global, so concurrent worker observations also
        # capture the aggregate VRAM high-water at their sampling points.
        "gpu_used_bytes_observed": gpu_peak_used,
        "cupy_pool_total_bytes": pool_peak_total,
        "worker_peak_rss_bytes": _peak_rss_bytes(),
    }
    return sides, timings, memory, preprocess.receipt()


_PREPARE_WORKER_CONTEXT = None


def _exit_with_parent():
    """End this spawned worker the moment its parent process dies.

    Breakage it prevents (Updraft factory, Vast V8X, 2026-10-27): when the
    preparation process was killed (cgroup OOM killer, a pool child barrier,
    a restart), its boundary workers blocked forever on the executor's call
    queue, because each worker holds a copy of that queue's write end, so no
    EOF ever arrives.  Two such workers kept 866 MiB CUDA contexts each on a
    card for 8.5 h, invisible to the card pool.  The parent sentinel is the
    pipe multiprocessing gives every spawned child for exactly this; it reads
    EOF only when the parent is gone.  No-op outside a spawned child.
    """
    import multiprocessing.connection
    import threading

    parent = multiprocessing.parent_process()
    if parent is None or parent.sentinel is None:
        return

    def watch(sentinel=parent.sentinel):
        multiprocessing.connection.wait([sentinel])
        os._exit(70)

    threading.Thread(target=watch, name="exit-with-parent", daemon=True).start()


def _prepare_worker_init(
        run_cfg, static_sides, eta, p_top, width,
        preprocess_backend, cpu_preprocess_bridge, sfcp_to_sfcp=True,
        host_workers=None, preprocess_selection=None):
    """Install immutable per-domain inputs in one spawned worker process.

    ``host_workers`` is ``--preprocess-workers`` under the CUDA backend:
    the thread count of the host steps that backend runs in the Rust
    library, which every worker's backend must carry so its receipt is
    the controller's.  Under the CPU backend each job's count comes
    with the job, from its slot of the budget.  ``preprocess_selection``
    is the controller backend's ``selection``, which every worker's
    backend carries for the same reason.
    """
    global _PREPARE_WORKER_CONTEXT
    _exit_with_parent()
    _PREPARE_WORKER_CONTEXT = (
        run_cfg, static_sides, tuple(float(value) for value in eta),
        float(p_top), int(width), preprocess_backend,
        cpu_preprocess_bridge, sfcp_to_sfcp, host_workers,
        None if preprocess_selection is None else dict(preprocess_selection))


def _prepare_boundary_hour(
        hour, compact_mets, preprocess_workers, worker_slot):
    """Initialize one forcing hour's four side slabs in an isolated process."""
    if _PREPARE_WORKER_CONTEXT is None:
        raise RuntimeError("boundary preparation worker was not initialized")
    (run_cfg, static_sides, eta, p_top, width, preprocess_backend,
     cpu_preprocess_bridge, sfcp_to_sfcp, host_workers,
     preprocess_selection) = _PREPARE_WORKER_CONTEXT
    started = time.perf_counter()
    sides, side_timings, memory, preprocess_receipt = _initialize_boundary_sides(
        compact_mets, run_cfg, static_sides, eta,
        p_top=p_top, width=width,
        preprocess_backend=preprocess_backend,
        preprocess_workers=(
            preprocess_workers if preprocess_workers is not None
            else host_workers),
        cpu_preprocess_bridge=cpu_preprocess_bridge, sfcp_to_sfcp=sfcp_to_sfcp,
        preprocess_selection=preprocess_selection)
    finished = time.perf_counter()
    return {
        "forecast_hour": int(hour),
        "sides": sides,
        "side_state_seconds": side_timings,
        "worker_wall_seconds": finished - started,
        "worker_started_monotonic": started,
        "worker_finished_monotonic": finished,
        "worker_pid": os.getpid(),
        "preprocess_worker_slot": worker_slot,
        "effective_native_workers": preprocess_workers,
        "memory": memory,
        "preprocess_backend": preprocess_receipt,
    }


def _prep_step(args, stage: str, *, label: str):
    """A step record for a preparation, and nothing for a forecast run.

    A ``--prepare-only`` run is a preparer program: its parent
    (``tools/prepare_hrrr_wrf.py`` under ``woof go``) reads each step's
    ``GPUWM_PREP_EVENT`` line off its stderr and puts it on the run's
    stream.  A forecast run's stderr is a person's terminal.
    """

    return (prep_stage(stage, label=label) if args.prepare_only
            else nullcontext())


def _native_boundary_species():
    """The hydrometeor masses the native HRRR decoder publishes per frame.

    Read off the source row (``boundary_species`` on ``hrrr`` in
    :mod:`woof.source_adapters`), so the start state and every boundary
    strip carry the same inventory and the root's specified boundary
    holds the analysed cloud, rain, ice, snow and graupel.
    """
    from woof.boundary_fields import source_boundary_species

    return source_boundary_species("hrrr")


@preprocess_math_call
def _initialize_state(
        snapshot, dc, grid, static, eta, mapping_report, *,
        p_top, column_workers=1, surface_fallback_radius: int = 8,
        preprocess_backend="cuda", state_backend="cuda",
        sfcp_to_sfcp=True, water_temperature_statics=None,
        physical_input=None, physical_output=None):
    """Full-domain f00/reference initialization with split timing."""
    from woof.core.grid import make_vertical_coord
    from woof.ingest.real import initialize_real

    if physical_input is None:
        met, horizontal_seconds = _map_snapshot(
            snapshot, grid, static, mapping_report,
            surface_fallback_radius=surface_fallback_radius,
            preprocess_backend=preprocess_backend)
    else:
        started = time.perf_counter()
        met = physical_input.read(0)
        if met.valid_time != snapshot.valid_time:
            raise ValueError("physical start snapshot valid time differs from the native source")
        from woof.ensemble.hrrr_physical_contract import validate_hrrr_physical_snapshot
        validate_hrrr_physical_snapshot(met)
        horizontal_seconds = time.perf_counter() - started
        mapping_report.update(_physical_mapping_receipt(physical_input, preprocess_backend))
    if water_temperature_statics is not None and physical_input is None:
        from woof.ingest.cpu_backend import host_step_workers
        from woof.ingest.water_temperature import assemble_horizontal_water_temperature
        met = assemble_horizontal_water_temperature(
            met, water_temperature_statics,
            workers=host_step_workers(preprocess_backend))
    if physical_output is not None:
        physical_output.write(met)
    started = time.perf_counter()
    coord = make_vertical_coord(
        dc.run.nz, hybrid_opt=dc.run.hybrid_opt, etac=dc.run.etac,
        eta_levels=eta)
    state_timing = {}
    result = initialize_real(
        met, dc.run, coord, static["HGT_M"], grid=grid, p_top=p_top,
        landmask=static["LANDMASK"],
        sfcp_to_sfcp=sfcp_to_sfcp, column_workers=column_workers,
        preprocess_backend=preprocess_backend,
        state_backend=state_backend,
        timing_report=state_timing,
        boundary_species=_native_boundary_species())
    result.state.set_map_coriolis(
        static["MAPFAC_M"], static["MAPFAC_U"], static["MAPFAC_V"],
        static["F"], static["E"], sina=static["SINALPHA"],
        cosa=static["COSALPHA"])
    return (result, met, horizontal_seconds, time.perf_counter() - started,
            state_timing)


def _physical_mapping_receipt(store, preprocess):
    """The mapping authority belongs to a verified native physical store."""
    from woof.ensemble.physical_store import digest_file
    return {"policy": "sealed native physical snapshot input",
            "physical_store_sha256": digest_file(store.manifest_path),
            "preprocess_backend": preprocess.receipt()}


def _overlay_acquirer(acquire, hours, overlay, binding, *, workers,
                      physical_input=None):
    """Apply source-grid overlays only before the first native mapping.

    A physical input has already passed physical_input_binding, including
    equality of the requested overlay bytes and source policy. Its mapped
    fields therefore carry that application and must not be overlaid again.
    """
    if overlay is None or physical_input is not None:
        return acquire, None
    from woof.ingest.water_overlay import overlay_snapshot_sequence

    class ForcingSequence:
        def __len__(self):
            return len(hours)

        def __getitem__(self, index):
            return acquire(hours[index])

    sequence = overlay_snapshot_sequence(
        ForcingSequence(), overlay, binding=binding, workers=workers)
    return lambda hour: sequence[hours.index(hour)], sequence


def _verify_preparation_overlay(sequence, *, physical_input, binding):
    from woof.ingest.water_overlay import (
        WaterOverlayError, overlay_file_identity, verify_overlay_sequence)
    result = verify_overlay_sequence(sequence)
    if physical_input is not None and binding is not None:
        if overlay_file_identity(binding["path"]) != binding:
            raise WaterOverlayError("water-temperature overlay bytes changed during physical replay")
    return result


class _LbcPayloadDigest:
    """The LBC payload digest, fed one interval at a time in order.

    A chained preparation writes each interval as a segment and lets it go,
    so it cannot hash the whole set at the end; fed the same intervals in
    the same order this is :func:`_lbc_payload_sha256` of that set.
    """

    def __init__(self):
        self._digest = hashlib.sha256()
        self.count = 0

    def add(self, index, interval) -> None:
        if int(index) != self.count:
            raise ValueError(
                f"LBC interval {index} hashed out of order (next is "
                f"{self.count})")
        digest = self._digest
        digest.update(f"{index}:{interval.start_seconds}:{interval.end_seconds}\n".encode())
        for name in sorted(interval.fields):
            field = interval.fields[name]
            for side_name in ("west", "east", "south", "north"):
                side = getattr(field, side_name)
                for role, value in (("value", side.value),
                                    ("tendency", side.tendency)):
                    array = np.ascontiguousarray(value)
                    digest.update(
                        f"{name}/{side_name}/{role};{array.dtype.str};"
                        f"{list(array.shape)};".encode())
                    digest.update(array.tobytes(order="C"))
        self.count += 1

    def hexdigest(self) -> str:
        return self._digest.hexdigest()


def _lbc_payload_sha256(boundaries):
    """Hash the exact ordered LBC values/tendencies independent of cache path."""
    digest = _LbcPayloadDigest()
    for index, interval in enumerate(boundaries.intervals):
        digest.add(index, interval)
    return digest.hexdigest()


#: The document ``tools/prepare_hrrr_wrf.py`` hands this preparation when
#: it chains (``--chained-bundle``): the bundle root and the authorities
#: the portable proof binds, so this process can publish the head.
CHAINED_BUNDLE_SCHEMA = "gpuwm-hrrr-chained-bundle-v1"
#: Where the native bundle keeps its prepared cache (the forecast runner's
#: ``HRRR_BUNDLE_PATHS["prepared_cache"]``).
CHAINED_CACHE_NAME = "native/prepared-cache"
#: The user metadata a native head holds for its start lead only; the
#: seal completes it with every boundary lead's report.
CHAINED_SEAL_COMPLETES = ("mapping_reports",)


def _chained_bundle(args):
    """The bundle a chained native preparation publishes into, or ``None``.

    ``None`` unless ``--chained-bundle`` was given.  A chained head is
    published only by a new prepare-only pipeline preparation; the prefix
    sealed cache the stream controller extends one hour at a time is
    published at its seal.
    """

    path = getattr(args, "chained_bundle", None)
    if path is None:
        return None
    if (not args.prepare_only or args.pipeline_series is None
            or args.sealed_prepared_cache):
        raise ValueError(
            "--chained-bundle publishes a head for a new prepare-only "
            "pipeline preparation, and this is not one")
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if (not isinstance(document, dict)
            or document.get("schema") != CHAINED_BUNDLE_SCHEMA):
        raise ValueError(f"{path} is not a {CHAINED_BUNDLE_SCHEMA} document")
    return document


def _interval_host_pricing(sides_by_side):
    """What one written interval of these side snapshots holds in host RAM.

    Its value and its tendency, each a float64 copy of the four sides
    (the same count :class:`woof.ingest.lateral_bc.StateBoundaryFrames`
    gives a route that builds from whole states).
    """

    return SimpleNamespace(interval_host_bytes=2 * sum(
        int(np.asarray(array).size) * np.dtype(np.float64).itemsize
        for side in sides_by_side.values() for array in side.values()))


def _bundle_root_in_place(staging, output_root) -> None:
    """The native bundle root already exists: a head is published in place.

    ``tools/prepare_hrrr_wrf.py`` creates the bundle root (and refuses an
    existing one) before it builds the static fields this preparation
    reads, so there is no staging tree to rename; the head is published by
    ``boundary-stream/head.json`` being written last.
    """

    if Path(staging).resolve() != Path(output_root).resolve():
        raise RuntimeError(
            f"a native bundle is published in place, not from {staging} "
            f"to {output_root}")


class _SealedBridgeLeads:
    """Leads of a sealed native bridge, each mapped only when it is taken.

    :meth:`open` verifies the bridge's ``SHA256SUMS`` once, and
    :meth:`take` then maps one lead, which the caller lets go once that
    lead is mapped onto the domain.  A mapped lead holds one open file
    descriptor per field (at least 24), so mapping every later lead at
    once held over 1,100 descriptors on a 48 h window and failed with
    ``[Errno 24] Too many open files`` under the ordinary 1024 soft
    ``RLIMIT_NOFILE``.  One lead at a time is what the pipeline route
    holds.
    """

    def __init__(self):
        self._load = None
        self._pending = {}

    def __bool__(self):
        return self._load is not None

    def open(self, bridge, source_hours, *, expected_manifest_sha256):
        """Verify ``bridge`` once; ``source_hours`` maps forcing hour to lead."""

        from woof.ingest.hrrr import verified_hrrr_native_bridge

        self._load = verified_hrrr_native_bridge(
            bridge, expected_manifest_sha256=expected_manifest_sha256)
        self._pending = {int(hour): int(source)
                         for hour, source in source_hours.items()}

    def take(self, hour):
        """Map forcing hour ``hour``'s lead; each hour is taken once."""

        return self._load(self._pending.pop(int(hour)))


def _write_chained_head(
        args, *, chain, exp, dc, grid, static, soil_mesh, pipeline_producer,
        source_hash_receipt, source_window, timing, make_identity,
        sealed_leads, requested_cycle, source_forecast_hours,
        model_forcing_hours, requested_hours, initial_snapshot, root_result,
        root_met, mapping_reports, boundary_sides, preprocess_receipt,
        source_identity):
    """Publish a chained native preparation's head once its start state exists.

    The bridge is sealed first: the prepared cache identity, the portable
    source manifest and the proof all bind its ``SHA256SUMS``, and the
    decoder only seals it once every lead is decoded.  Then the start
    state's surface, the portable authorities
    (:func:`woof.hrrr_prepared_bundle.publish_hrrr_bundle_head`) and the
    head.  Returns ``(writer, lbc_digest, identity, root_surface,
    soil_temperature_repair, pipeline_report)``; ``writer`` is ``None``
    when the portable authorities cannot be published, and the
    preparation then finishes one-shot and says why, as before.
    """

    from woof.hrrr_prepared_bundle import (
        HrrrBundleError, publish_hrrr_bundle_head)
    from woof.ingest.boundary_stream import PreparedTreeWriter
    from woof.ingest.hrrr_physics import resolve_prepared_noah_surface
    from woof.ingest.preprocess_backend import preprocess_reports_identity
    from woof.ingest.soil import soil_temperature_repair_proof

    started = time.perf_counter()
    with _prep_step(args, "root_bridge_seal",
                    label="Decode and seal every source hour"):
        pipeline_report = pipeline_producer.finish()
        seal_process, seal_receipt, seal_started = _start_seal(
            args, pipeline_report, source_hash_receipt, source_window)
        stdout, stderr = seal_process.communicate()
        timing["pipeline_bridge_seal_wall"] = (
            time.perf_counter() - seal_started)
        if seal_process.returncode != 0:
            raise RuntimeError("pipeline bridge seal failed: " + stderr[-4000:])
        if seal_receipt is None or not seal_receipt.is_file():
            raise RuntimeError("pipeline bridge seal omitted receipt")
        seal = json.loads(seal_receipt.read_text())
    args.manifest_sha256 = seal["manifest_sha256"]
    pipeline_report["seal"] = seal
    pipeline_report["seal_stdout"] = stdout.strip()
    identity = make_identity(args.manifest_sha256)
    # Verified here, once; each later lead is mapped when its hour is.
    sealed_leads.open(
        args.bridge, {hour: source_forecast_hours[hour]
                      for hour in requested_hours[1:]},
        expected_manifest_sha256=args.manifest_sha256)
    timing["chained_head_bridge_seconds"] = time.perf_counter() - started

    # The same surface call the one-shot writer makes, on the same start
    # state; it reads no boundary hour.
    root_surface = resolve_prepared_noah_surface(
        root_met, dc.run, static, soil_mesh=soil_mesh)
    soil_temperature_repair = soil_temperature_repair_proof(root_surface, grid)
    start_key = f"f{source_forecast_hours[0]:02d}"
    metadata = {
        "initial_valid_time": initial_snapshot.valid_time.isoformat(),
        "last_valid_time": (initial_snapshot.valid_time + timedelta(
            hours=int(requested_hours[-1]))).isoformat(),
        "source_cycle": requested_cycle.isoformat(),
        "source_forecast_hours": list(source_forecast_hours),
        "model_forcing_hours": list(model_forcing_hours),
        "forcing_hours": list(requested_hours),
        # The start lead's report; the seal adds every boundary lead's.
        "mapping_reports": _strict_json(preprocess_reports_identity(
            {start_key: mapping_reports[start_key]})),
        "soil_texture_downscale": _strict_json(
            root_surface.soil_texture_downscale),
        **({"soil_temperature_repair": _strict_json(
            soil_temperature_repair)}
           if soil_temperature_repair is not None else {}),
    }
    root = Path(chain["output_root"]).resolve()

    def optional_path(key):
        value = chain.get(key)
        return None if value is None else Path(value)

    try:
        head_bundle = publish_hrrr_bundle_head(
            output_root=root, prepared_cache=root / CHAINED_CACHE_NAME,
            static_cache=Path(chain["static_cache"]),
            static_receipt=Path(chain["static_receipt"]),
            geometry_receipt=Path(chain["geometry_receipt"]),
            bridge_manifest=args.bridge / "SHA256SUMS",
            namelist_input=args.namelist_input,
            wps_namelist=optional_path("wps_namelist"),
            source_manifest=Path(chain["source_manifest"]),
            experiment_config=Path(chain["experiment_config"]),
            source_cycle=requested_cycle,
            source_forecast_hours=source_forecast_hours,
            model_forcing_hours=model_forcing_hours,
            preprocessing=_strict_json(preprocess_receipt),
            source_identity=identity["source_identity"],
            physics_profile=chain.get("physics_profile"),
            cache_user_metadata=metadata,
            expert_acknowledgements=tuple(chain.get("acknowledgements") or ()),
            domain_spec=optional_path("domain_spec"))
    except HrrrBundleError as error:
        # The one-shot route publishes no portable bundle in this case and
        # says so (tools/prepare_hrrr_wrf.py); nothing can bind a head.
        print("prepare: chained preparation not used: the portable bundle's "
              f"head could not be published ({error}); the forecast starts "
              "after preparation", file=sys.stderr, flush=True)
        return (None, None, identity, root_surface, soil_temperature_repair,
                pipeline_report)

    writer = PreparedTreeWriter(
        staging=root, output_root=root, identity=identity,
        cache_name=CHAINED_CACHE_NAME, publish=_bundle_root_in_place)
    backend = str(preprocess_receipt["backend"])
    # A card producer is priced at the preparation's whole device price
    # (the start-state build and every spawned boundary worker), which
    # covers what it still holds while the forecast runs beside it.
    device_bytes = (int(native_preparation_price(
        dc.run, forcing_times=len(requested_hours),
        prepare_workers=args.prepare_workers).need_bytes)
        if backend == "cuda" else None)
    writer.admit(experiment=exp, backend=backend, device_bytes=device_bytes,
                 source="hrrr")
    writer.write_head(
        initial_result=root_result, met=root_met,
        surface=root_surface.fields, metadata=metadata,
        # The configuration's zones, which the forecast door compares
        # against this head (prepared_single_domain_forecast: "prepared
        # cache LBC width differs from experiment"); a literal 1/4 refused
        # every HRRR-route tree on other zones, HRRR v4's own 1/9 among them.
        lbc={"spec_bdy_width": int(dc.run.spec_bdy_width),
             "spec_zone": int(dc.run.spec_zone),
             "relax_zone": int(dc.run.relax_zone),
             "schedule": [[float(k * 3600), float((k + 1) * 3600)]
                          for k in range(len(requested_hours) - 1)],
             "fields": sorted(boundary_sides["west"])},
        proof_head=head_bundle["proof_head"],
        input_manifest_sha256=head_bundle["handoff"]["source_manifest_sha256"],
        forcing=_interval_host_pricing(boundary_sides),
        seal_completes=CHAINED_SEAL_COMPLETES)
    writer.native_bundle_head = head_bundle
    return (writer, _LbcPayloadDigest(), identity, root_surface,
            soil_temperature_repair, pipeline_report)


#: An as-posted native preparation (A136 L7c): the portable manifest has
#: no lead role (each lead's two objects are rows of the source manifest,
#: a document of their own), so its plan blanks only the two documents the
#: seal writes from the leads: the decoded bridge's SHA256SUMS and the
#: source manifest.  The prefix names no role of that manifest.
POSTED_NATIVE_LEAD_ROLE_PREFIX = "hrrr-f"
POSTED_NATIVE_DERIVED_ROLES = ("bridge", "source_manifest")
#: The two cache identity keys that carry those documents' digests, and the
#: manifest role each is bound to (boundary_stream.DOCUMENT_ROW_RULES).
POSTED_NATIVE_DOCUMENT_KEYS = {"bridge_manifest_sha256": "bridge",
                               "source_manifest_sha256": "source_manifest"}
#: The bridge rows that are no lead's decoded files: written from the
#: reference lead and the series (the sealer holds the gate to them).
POSTED_NATIVE_BRIDGE_FIXED_ROWS = ("./gate.txt", "./inventory.tsv")
#: Where an as-posted bundle keeps the source manifest its seal authors
#: from the leads' markers (``tools/prepare_hrrr_wrf.py`` passes this path
#: as ``--source-manifest``).
POSTED_SOURCE_MANIFEST = "native/posted-source/SHA256SUMS"


def _posted_native_leads(posting, *, cycle, source_forecast_hours):
    """The lead wait of an as-posted native preparation, held to its window.

    ``posting`` is the as-posted fetch's ``posting/`` folder; its schedule
    must be this cycle's native HRRR window and schedule every lead this
    preparation reads, so no lead is waited for that the fetch will never
    fetch.
    """

    from woof.ingest.boundary_stream import (
        POSTING_SCHEDULE_NAME, PostedLeads, read_replaced_json)

    path = Path(posting) / POSTING_SCHEDULE_NAME
    try:
        schedule = read_replaced_json(path)
    except (OSError, ValueError) as error:
        raise ValueError(
            f"an as-posted native preparation reads the fetch's posting "
            f"schedule, and {path} is not readable: {error}") from None
    label = cycle.strftime("%Y-%m-%dT%H")
    if (not isinstance(schedule, dict) or schedule.get("source") != "hrrr"
            or str(schedule.get("cycle", ""))[:13] != label):
        raise ValueError(
            f"{path} schedules {schedule.get('source')} "
            f"{schedule.get('cycle')}, not the hrrr {label} cycle this "
            "preparation reads")
    scheduled = {int(row["lead"]) for row in schedule.get("leads") or ()}
    unscheduled = sorted(set(source_forecast_hours) - scheduled)
    if unscheduled:
        raise ValueError(
            f"{path} does not schedule lead(s) "
            f"{', '.join(f'f{lead:03d}' for lead in unscheduled)} of this "
            "preparation's window, so nothing would ever post them here")
    return PostedLeads(Path(posting), source="hrrr",
                       cycle=str(schedule["cycle"]))


def _decoded_lead_record(root, hour, *, workers=8):
    """``{row: sha256}`` of one decoded lead's files, as the bridge seal names them.

    Read when the lead is taken from the decoder's staging, so each segment
    binds the decoded bytes its interval was built from, and the sealed
    bridge's ``SHA256SUMS`` rows are held to them (``decoded_leads``).
    """

    from concurrent.futures import ThreadPoolExecutor

    files = []
    for role in ("atmosphere", "soil"):
        directory = Path(root) / f"{role}-f{int(hour):02d}"
        files += [(f"./{role}-f{int(hour):02d}/{path.name}", path)
                  for path in sorted(directory.iterdir()) if path.is_file()]
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        digests = list(pool.map(lambda item: _sha256(item[1]), files))
    return {name: digest for (name, _), digest in zip(files, digests)}


def _await_admitted(admitter, producer, lead, *, poll_seconds=1.0,
                    between=None):
    """Lead ``lead``'s posted marker once it is admitted to the decoder.

    A lead may take its whole lateness budget to post, and the decoder can
    fail first (it exits at once on a series it cannot read).  The decoder
    is looked at between waits, so its failure is raised now, not once the
    lead posts: the breakage this prevents is a preparation that sat out a
    lead's whole budget beside a decoder that had already died (test host D,
    HRRR 2026-10-01T06, 30 minutes).  ``between`` is called between waits
    too (the build loop writes the intervals already built).
    """

    while True:
        marker = admitter.wait(lead, timeout=poll_seconds)
        if marker is not None:
            return marker
        producer.check()
        if between is not None:
            between()


def _seal_posted_bridge(args, *, admitter, pipeline_producer, source_window,
                        timing):
    """Finish an as-posted decode; author the source manifest; seal the bridge.

    The source manifest is written from the leads' posted markers (each
    object's name and digest, sorted by name), byte for byte the
    ``SHA256SUMS`` a one-shot fetch of the same window writes; its digest is
    what the seal binds where a one-shot preparation binds
    ``--source-manifest-sha256``.  Returns ``(pipeline_report,
    source_hash_receipt)``.
    """

    pipeline_report = pipeline_producer.finish()
    started = time.perf_counter()
    manifest = Path(args.source_manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(admitter.source_manifest_text())
    args.source_manifest_sha256 = _sha256(manifest)
    source_hash_receipt = admitter.receipt(manifest)
    seal_process, seal_receipt, seal_started = _start_seal(
        args, pipeline_report, source_hash_receipt, source_window)
    stdout, stderr = seal_process.communicate()
    timing["pipeline_bridge_seal_wall"] = time.perf_counter() - seal_started
    if seal_process.returncode != 0:
        raise RuntimeError("pipeline bridge seal failed: " + stderr[-4000:])
    if seal_receipt is None or not seal_receipt.is_file():
        raise RuntimeError("pipeline bridge seal omitted receipt")
    seal = json.loads(seal_receipt.read_text())
    args.manifest_sha256 = seal["manifest_sha256"]
    pipeline_report["seal"] = seal
    pipeline_report["seal_stdout"] = stdout.strip()
    timing["posted_source_manifest_and_bridge_seal"] = (
        time.perf_counter() - started)
    return pipeline_report, source_hash_receipt


def _posted_bundle_plan(args, *, chain, requested_cycle, source_forecast_hours,
                        model_forcing_hours, preprocess_receipt, source_identity,
                        posted_leads, metadata):
    """Publish the ordinary fixed authorities once, before a physical capture."""
    from woof.hrrr_prepared_bundle import publish_hrrr_bundle_head
    from woof.ingest.boundary_stream import input_plan
    root = Path(chain["output_root"]).resolve()
    def optional_path(key):
        value = chain.get(key)
        return None if value is None else Path(value)
    bundle = publish_hrrr_bundle_head(
        output_root=root, prepared_cache=root / CHAINED_CACHE_NAME,
        static_cache=Path(chain["static_cache"]),
        static_receipt=Path(chain["static_receipt"]),
        geometry_receipt=Path(chain["geometry_receipt"]),
        bridge_manifest=args.bridge / "SHA256SUMS",
        namelist_input=args.namelist_input,
        wps_namelist=optional_path("wps_namelist"),
        source_manifest=Path(chain["source_manifest"]),
        experiment_config=Path(chain["experiment_config"]),
        source_cycle=requested_cycle,
        source_forecast_hours=source_forecast_hours,
        model_forcing_hours=model_forcing_hours,
        preprocessing=_strict_json(preprocess_receipt),
        source_identity=source_identity,
        physics_profile=chain.get("physics_profile"),
        cache_user_metadata=metadata,
        expert_acknowledgements=tuple(chain.get("acknowledgements") or ()),
        domain_spec=optional_path("domain_spec"), as_posted=True)
    plan = input_plan(
        bundle["manifest"], lead_role_prefix=POSTED_NATIVE_LEAD_ROLE_PREFIX,
        route_table_sha256=posted_leads.route_table_sha256(),
        derived_roles=POSTED_NATIVE_DERIVED_ROLES)
    return bundle, plan


class _PostedHrrrCapture:
    """Capture a native knot only after its ordinary lead evidence is ready."""
    def __init__(self, stream, *, cycle, source_forecast_hours, admitter, decoded_records):
        self.stream = stream
        self.cycle = cycle
        self.source_forecast_hours = tuple(source_forecast_hours)
        self.admitter = admitter
        self.decoded_records = decoded_records

    def write(self, snapshot):
        from woof.ingest.boundary_stream import (
            posted_lead_marker_sha256, decoded_lead_record_sha256)
        seconds = (snapshot.valid_time - self.cycle).total_seconds()
        lead = int(seconds // 3600)
        if seconds != lead * 3600 or lead not in self.source_forecast_hours:
            raise ValueError("captured physical time is outside the native posted source plan")
        if lead not in self.admitter.markers or lead not in self.decoded_records:
            raise ValueError("physical capture requires actual posted and decoded native lead evidence")
        return self.stream.publish(snapshot,
            posted_leads={str(lead): posted_lead_marker_sha256(self.admitter.markers[lead])},
            decoded_leads={str(lead): decoded_lead_record_sha256(self.decoded_records[lead])})

    def seal(self):
        return self.stream.seal()


def _complete_posted_head_receipts(bundle, *, metadata, preprocess_receipt):
    """Fill observed preparation receipts before the ordinary head is hashed.

    Physical capture needs the fixed input plan before real initialization.
    The vertical route and soil outcomes become known during that call. They
    belong in the subsequently published proof, not in the fixed input plan.
    """
    from woof.hrrr_prepared_bundle import _canonical
    from woof.ingest.prepared_cache import SOIL_PREPARATION_RECEIPTS
    proof = bundle["proof_head"]
    observed = _strict_json(preprocess_receipt)
    proof["preprocessing"] = observed
    proof["preprocessing_receipt_sha256"] = hashlib.sha256(
        _canonical(observed).encode("utf-8")).hexdigest()
    for key in SOIL_PREPARATION_RECEIPTS:
        if key in metadata:
            proof[key] = metadata[key]


def _write_posted_head(
        args, *, chain, exp, dc, grid, static, soil_mesh, admitter,
        producer, posted_leads, decoded_records, timing, make_identity,
        requested_cycle, source_forecast_hours, model_forcing_hours,
        requested_hours, initial_snapshot, root_result, root_met,
        mapping_reports, boundary_sides, preprocess_receipt,
        source_identity, posted_bundle=None, ensemble_physical=None,
        physical_receipts=None, shared_source=None, reused_surface=None):
    """Publish an as-posted native preparation's head (A136 L7c (b)).

    Written once the start state exists and the first boundary lead is
    posted (the start needs, f(S) and f(S+1)), without waiting for any later
    lead: the head binds the input plan (the portable manifest with the
    bridge and source-manifest digests not yet known) and the two start
    leads' markers, and its cache identity carries the plan's placeholder
    in ``bridge_manifest_sha256`` and ``source_manifest_sha256``, which the
    seal writes as the digests of the documents it writes then.  Returns
    ``(writer, lbc_digest, identity, root_surface,
    soil_temperature_repair)``; ``writer`` is ``None`` when the portable
    authorities cannot be published, as the chained head says.
    """

    from woof.hrrr_prepared_bundle import (
        AS_POSTED_SEAL_KEYS, HrrrBundleError)
    from woof.ingest.boundary_stream import (
        PreparedTreeWriter, as_posted_placeholder,
        input_plan_sha256)
    from woof.ingest.hrrr_physics import resolve_prepared_noah_surface
    from woof.ingest.preprocess_backend import preprocess_reports_identity
    from woof.ingest.soil import soil_temperature_repair_proof

    started = time.perf_counter()
    start_leads = tuple(source_forecast_hours[:2])
    start_records = ({"start_markers": {
        lead: _await_admitted(admitter, producer, lead) for lead in start_leads}}
        if shared_source is None else {"start_marker_sha256":
            dict(shared_source.posted["start_marker_sha256"])})
    timing["posted_head_start_leads_wait"] = time.perf_counter() - started

    root_surface = resolve_prepared_noah_surface(
        root_met, dc.run, static, soil_mesh=soil_mesh, surface=reused_surface)
    soil_temperature_repair = soil_temperature_repair_proof(root_surface, grid)
    start_key = f"f{source_forecast_hours[0]:02d}"
    metadata = {
        "initial_valid_time": initial_snapshot.valid_time.isoformat(),
        "last_valid_time": (initial_snapshot.valid_time + timedelta(
            hours=int(requested_hours[-1]))).isoformat(),
        "source_cycle": requested_cycle.isoformat(),
        "source_forecast_hours": list(source_forecast_hours),
        "model_forcing_hours": list(model_forcing_hours),
        "forcing_hours": list(requested_hours),
        "mapping_reports": _strict_json(preprocess_reports_identity(
            {start_key: mapping_reports[start_key]})),
        "soil_texture_downscale": _strict_json(
            root_surface.soil_texture_downscale),
        **({"soil_temperature_repair": _strict_json(
            soil_temperature_repair)}
           if soil_temperature_repair is not None else {}),
    }
    root = Path(chain["output_root"]).resolve()

    try:
        if posted_bundle is None:
            head_bundle, plan = _posted_bundle_plan(
                args, chain=chain, requested_cycle=requested_cycle,
                source_forecast_hours=source_forecast_hours,
                model_forcing_hours=model_forcing_hours,
                preprocess_receipt=preprocess_receipt, source_identity=source_identity,
                posted_leads=posted_leads, metadata=metadata)
        else:
            head_bundle, plan = posted_bundle
            _complete_posted_head_receipts(head_bundle, metadata=metadata,
                                          preprocess_receipt=preprocess_receipt)
    except HrrrBundleError as error:
        if posted_bundle is not None or ensemble_physical is not None:
            raise
        print("prepare: chained preparation not used: the portable bundle's "
              f"head could not be published ({error}); the forecast starts "
              "after preparation", file=sys.stderr, flush=True)
        return None, None, None, root_surface, soil_temperature_repair

    placeholder = as_posted_placeholder(input_plan_sha256(plan))
    identity = make_identity(placeholder, placeholder)
    writer = PreparedTreeWriter(
        staging=root, output_root=root, identity=identity,
        cache_name=CHAINED_CACHE_NAME, publish=_bundle_root_in_place)
    backend = str(preprocess_receipt["backend"])
    device_bytes = (int(native_preparation_price(
        dc.run, forcing_times=len(requested_hours),
        prepare_workers=args.prepare_workers).need_bytes)
        if backend == "cuda" else None)
    writer.admit(experiment=exp, backend=backend, device_bytes=device_bytes,
                 source="hrrr")
    if shared_source is None:
        writer.bind_posted_leads(admitter.markers)
        writer.bind_decoded_leads(decoded_records)
    if physical_receipts is not None:
        writer.bind_physical_receipts(physical_receipts)
    bridge_relative = (Path(args.bridge).resolve() / "SHA256SUMS").relative_to(
        root).as_posix()
    source_relative = Path(chain["source_manifest"]).resolve().relative_to(
        root).as_posix()
    writer.write_head(
        initial_result=root_result, met=root_met,
        surface=root_surface.fields, metadata=metadata,
        # The configuration's zones, which the forecast door compares
        # against this head (prepared_single_domain_forecast: "prepared
        # cache LBC width differs from experiment"); a literal 1/4 refused
        # every HRRR-route tree on other zones, HRRR v4's own 1/9 among them.
        lbc={"spec_bdy_width": int(dc.run.spec_bdy_width),
             "spec_zone": int(dc.run.spec_zone),
             "relax_zone": int(dc.run.relax_zone),
             "schedule": [[float(k * 3600), float((k + 1) * 3600)]
                          for k in range(len(requested_hours) - 1)],
             "fields": sorted(boundary_sides["west"])},
        proof_head=head_bundle["proof_head"], input_manifest_sha256=None,
        forcing=_interval_host_pricing(boundary_sides),
        seal_completes=CHAINED_SEAL_COMPLETES,
        **({"ensemble_physical": ensemble_physical}
           if ensemble_physical is not None else {}),
        as_posted={
            "input_plan": plan,
            **start_records,
            "forcing_leads": list(source_forecast_hours),
            "seal_authored_proof_keys": AS_POSTED_SEAL_KEYS,
            "manifest_path": "source-input-manifest.json",
            "lead_role_prefix": POSTED_NATIVE_LEAD_ROLE_PREFIX,
            "derived_roles": POSTED_NATIVE_DERIVED_ROLES,
            "document_bound_identity_keys": POSTED_NATIVE_DOCUMENT_KEYS,
            "documents": {
                "bridge": {"path": bridge_relative,
                           "lead_rows": "decoded_leads",
                           "fixed_rows": POSTED_NATIVE_BRIDGE_FIXED_ROWS},
                "source_manifest": {"path": source_relative,
                                    "lead_rows": "posted_objects"},
            },
        })
    writer.native_bundle_head = head_bundle
    # From here a wait on a later lead says so on the producer heartbeat
    # (waiting_for_source), and a forecast that stopped ends it.
    posted_leads.writer = writer
    return (writer, _LbcPayloadDigest(), identity, root_surface,
            soil_temperature_repair)


def _seal_chained_cache(writer, *, timing, mapping_reports, last_valid_time,
                        **posted):
    """Seal the streamed cache: every lead's mapping report, then the header.

    ``posted`` (an as-posted head) is ``identity``, ``manifest_sha256`` and
    ``document_sha256``, the one-shot identity its seal writes
    (:meth:`woof.ingest.boundary_stream.PreparedTreeWriter.seal_cache`).
    """

    from woof.ingest.preprocess_backend import preprocess_reports_identity

    promised = writer.head["basis"]["cache"]["metadata"]["user"][
        "last_valid_time"]
    if promised != last_valid_time.isoformat():
        # The head named the last forcing time before the lead was decoded;
        # a source whose last lead is valid at another time is not the
        # cache that head described.
        raise RuntimeError(
            f"the last forcing time is {last_valid_time.isoformat()}, not "
            f"the {promised} the head was published with")
    started = time.perf_counter()
    receipt = writer.seal_cache(completed_metadata={
        "mapping_reports": _strict_json(
            preprocess_reports_identity(mapping_reports))}, **posted)
    timing["seal_streamed_prepared_cache"] = time.perf_counter() - started
    return receipt


def _publish_chained_proof(writer, args, *, chain, report, configured_run,
                           posted_proof=None):
    """Check the preparation's receipts, then write ``proof.json`` last.

    The receipts are the ones ``tools/prepare_hrrr_wrf.py`` checks before
    it publishes a one-shot bundle, checked here because a chained
    bundle's proof is written by this process; the wrapper checks them
    again after.  Returns the handoff the wrapper relays.
    """

    from woof.hrrr_prepared_bundle import (
        PROOF_NAME, seal_hrrr_bundle_proof, sealed_handoff)
    from tools.prepare_hrrr_wrf import (
        _validated_physics_receipt, _validated_worker_receipts)

    requested = chain["requested"]
    preprocessing, _, _ = _validated_worker_receipts(
        report, selected_backend=requested["preprocess_backend"],
        requested_preprocess_workers=requested["preprocess_workers"],
        requested_pipeline_workers=requested["pipeline_workers"],
        final_hour=int(report["model_forcing_hours"][-1]),
        reused_posted_source=getattr(args, "physical_input_provider", None) is not None)
    _validated_physics_receipt(
        report, requested_profile=chain.get("physics_profile"),
        expected_selection=configured_run)
    head = writer.native_bundle_head
    if (json.dumps(_strict_json(preprocessing), sort_keys=True)
            != json.dumps(head["proof_head"]["preprocessing"],
                          sort_keys=True)):
        # A boundary hour added a vertical route the start lead did not
        # meet, so the head's proof names a receipt the one-shot proof of
        # the same preparation would not.
        raise RuntimeError(
            "the preprocessing receipt changed after the head was published, "
            "so the sealed proof would differ from the head's")
    root = Path(chain["output_root"]).resolve()
    proof = seal_hrrr_bundle_proof(
        head, output_root=root, prepared_cache=root / CHAINED_CACHE_NAME,
        static_cache=Path(chain["static_cache"]),
        geometry_receipt=Path(chain["geometry_receipt"]))
    # An as-posted head's seal-authored keys (the portable manifest and the
    # bridge it names, written at the seal) and its posting record.
    proof.update(posted_proof or {})
    proof["boundary_stream"] = writer.boundary_stream_proof()
    writer.publish(proof)
    return sealed_handoff(head, proof_path=root / PROOF_NAME,
                          content_sha256=report["prepared_cache"][
                              "content_sha256"])


def _start_seal(args, producer_report, source_hash_receipt, source_window):
    from woof.ingest.native_supplements import verify_supplement_receipt
    verify_supplement_receipt(source_hash_receipt)
    from tools.hrrr_pipeline import write_pipeline_receipt

    controller = args.pipeline_signals / "controller.json"
    write_pipeline_receipt(controller, {
        "source_hash_preflight": source_hash_receipt,
        "producer": producer_report,
    })
    receipt = args.pipeline_signals / "seal.json"
    process = subprocess.Popen([
        sys.executable, str(REPO / "tools" / "seal_hrrr_native_bridge.py"),
        "--root", str(args.bridge), "--receipt", str(receipt),
        "--source-manifest-sha256", args.source_manifest_sha256,
        "--decoder", str(args.pipeline_decoder),
        "--series", str(args.pipeline_series),
        "--time-evidence", str(controller),
        "--expected-window-shape", f"{source_window.ny}x{source_window.nx}",
    ], cwd=REPO, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return process, receipt, time.perf_counter()


def _validated_namelist_extension_identity(args, *, cycle: datetime):
    """Derive the seal from actual bytes and the already validated window."""

    if args.sealed_prepared_cache:
        if args.forecast_start_hour != 0:
            raise ValueError(
                "sealed prepared-cache namelist must start at source hour 0")
        horizon_seconds = float(args.run_seconds)
    elif args.namelist_extension_suffix:
        if args.forecast_start_hour <= 0:
            raise ValueError(
                "namelist extension suffix must start after source hour 0")
        horizon_seconds = (
            args.forecast_start_hour * 3600.0 + float(args.run_seconds))
    else:
        return None
    if not horizon_seconds.is_integer():
        raise ValueError(
            "sealed namelist horizon must contain a whole number of seconds")
    return namelist_extension_invariant(
        args.namelist_input, cycle=cycle,
        run_seconds=int(horizon_seconds))


def _carry_vertical_routes(expected, actual, context):
    """Check a job's vertical routes against the controller's by depth.

    Pops ``vertical_interpolation`` from both receipt copies.  A job's
    route for a source level count the controller has recorded must be
    the controller's route for that count (the same engine, tier and
    reason); a route for a count the controller has not met is appended
    to the controller's record, which is the backend's own shared list.
    """

    controller_routes = expected.pop("vertical_interpolation", None)
    job_routes = actual.pop("vertical_interpolation", None)
    if controller_routes is None and job_routes is None:
        return
    if not isinstance(controller_routes, list) or not isinstance(
            job_routes, list):
        raise RuntimeError(
            f"{context} preprocessing backend receipt differs from the "
            "resolved public selector in vertical_interpolation")
    by_depth = {}
    for route in controller_routes:
        if isinstance(route, dict):
            by_depth.setdefault(route.get("source_levels"), []).append(route)
    for route in job_routes:
        if not isinstance(route, dict) or "source_levels" not in route:
            raise RuntimeError(
                f"{context} preprocessing backend receipt carries a "
                f"malformed vertical route {route!r}")
        known = by_depth.get(route["source_levels"])
        if known is None:
            controller_routes.append(dict(route))
            by_depth[route["source_levels"]] = [route]
        elif route not in known:
            raise RuntimeError(
                f"{context} preprocessing backend receipt differs from the "
                "resolved public selector in vertical_interpolation: it "
                f"routed {route['source_levels']} source levels as "
                f"{route!r}, the controller as {known[0]!r}")


def _within_worker_allocation(count, allocation):
    """Whether a job's native worker count is a positive count within its
    allocation (see :func:`_require_preprocess_receipt`)."""

    def whole(value):
        return (isinstance(value, (int, np.integer))
                and not isinstance(value, (bool, np.bool_)))

    return (whole(count) and whole(allocation)
            and 1 <= int(count) <= int(allocation))


def _require_preprocess_receipt(
        controller_receipt, observed, context, *,
        expected_native_workers=None):
    """Refuse a job whose backend receipt is not the controller's.

    Under the CPU backend every job runs on its slot's share of the
    worker budget, so the counts are the one place a job's receipt may
    differ from the controller's, and they are checked against the
    job's allocation instead: the top-level ``workers`` and the masked
    surface chain's ``workers``.  The chain's count used to be compared
    as part of the receipt, where the controller's names the whole
    budget and a slot's its share, so a CPU preparation with two or more
    boundary hours was refused at its first boundary hour.

    A count is held to AT MOST its allocation, not to equality with it.
    The bound is the breakage this check prevents (d31071e093, "bound
    preprocess workers"): a job on more than its share, such as a slot
    on the controller's whole budget, oversubscribes the host while the
    other slots run.  A job on fewer is the backend's own answer to the
    host: :func:`woof.ingest.preparation_workers.effective_workers`
    caps every count by the CPU and memory headroom it measures when the
    job's backend is built, and work splits only across independent
    columns, so the count moves no byte.  Equality turned load into a
    failure: on a busy host the CPU preparation was refused with
    "f00 mapping used 20 native workers; expected 24": the host narrowed
    the running scope's CPUs from 0-23 to 4-23 after
    :func:`woof.ingest.cpu_backend.automatic_workers` had taken (and
    cached) its 24-CPU budget and before the backend counted its own.

    The vertical routes are the other place: a spawned boundary worker
    keeps its own route record, one entry per source level count it
    prepared, where the controller's names the counts the controller and
    its in-process slots met.  A boundary hour whose source level count
    differs from f00's was refused on that difference although both
    backends route that depth the same way.  So a route is checked
    against the controller's route for the same source level count, and
    a depth the controller has not met is carried into the controller's
    record (the list its receipt shares), so the written receipt names
    every route that ran.
    """

    if not isinstance(observed, dict):
        raise RuntimeError(
            f"{context} omitted its preprocessing backend receipt")
    expected = dict(controller_receipt)
    actual = dict(observed)
    _carry_vertical_routes(expected, actual, context)
    if expected["backend"] == "cpu":
        expected.pop("workers")
        # Worker-share and host-headroom telemetry, measured per job.
        expected.pop("parallelism", None)
        actual.pop("parallelism", None)
        observed_workers = actual.pop("workers", None)
        if not _within_worker_allocation(
                observed_workers, expected_native_workers):
            raise RuntimeError(
                f"{context} used {observed_workers!r} native workers; "
                f"its allocation is at most {expected_native_workers!r}")
        expected_chain = expected.get("masked_surface_chain")
        if isinstance(expected_chain, dict):
            expected["masked_surface_chain"] = {
                key: value for key, value in expected_chain.items()
                if key != "workers"}
        observed_chain = actual.get("masked_surface_chain")
        if isinstance(observed_chain, dict):
            observed_chain = dict(observed_chain)
            chain_workers = observed_chain.pop("workers", None)
            if not _within_worker_allocation(
                    chain_workers, expected_native_workers):
                raise RuntimeError(
                    f"{context} ran the masked surface fields on "
                    f"{chain_workers!r} native workers; its allocation is "
                    f"at most {expected_native_workers!r}")
            actual["masked_surface_chain"] = observed_chain
    elif expected_native_workers is not None:
        raise RuntimeError(
            f"{context} assigned native workers to CUDA preprocessing")
    if actual != expected:
        differing = sorted(
            key for key in set(expected) | set(actual)
            if expected.get(key) != actual.get(key))
        raise RuntimeError(
            f"{context} preprocessing backend receipt differs from the "
            f"resolved public selector in {', '.join(differing)}")


def _budgeted_preprocess_backend(args, price=None):
    """The resolved backend and, for the CPU, its explicit worker total.

    ``price`` is the preparation's device price
    (:func:`native_preparation_price`): a CUDA answer that does not fit
    the card's free memory becomes the CPU under auto and is refused by
    name when cuda was named, before anything is allocated.

    Implicit CPU auto-selection is resolved to one explicit total budget,
    so every later concurrent job receives a deterministic partition of
    this one value rather than independently expanding "auto".  Pinning
    the budget re-resolves the CPU backend, which is not a second choice
    of backend: the receipt keeps the first resolution's selector and
    reason.  It used to say "named by the caller" about a CPU nobody
    named, once auto had fallen to it because no card was usable.
    """

    from woof.ingest.preprocess_backend import resolve_preprocess_backend

    preprocess = resolve_preprocess_backend(
        args.preprocess_backend, workers=args.preprocess_workers,
        cpu_bridge=args.cpu_preprocess_bridge, price=price)
    if getattr(preprocess, "name", None) == "cuda":
        return preprocess, None
    # The count the backend itself will run, through the same budget its
    # constructor applies (affinity, cgroup CPU quota, memory headroom,
    # WOOF_PREPROCESS_THREADS).  os.cpu_count() ignores the first three,
    # so on a host whose cgroup grants fewer CPUs than it has cores every
    # CPU preparation was refused at f00 ("used 245 native workers;
    # expected 256") by its own receipt check.
    from woof.ingest.cpu_backend import automatic_workers
    from woof.ingest.preparation_workers import effective_workers
    effective_preprocess_workers = (
        effective_workers(int(args.preprocess_workers))
        if args.preprocess_workers is not None
        else automatic_workers())
    if getattr(preprocess, "workers", None) != effective_preprocess_workers:
        chosen = getattr(preprocess, "selection", None)
        preprocess = resolve_preprocess_backend(
            "cpu", workers=effective_preprocess_workers,
            cpu_bridge=(
                args.cpu_preprocess_bridge
                if args.preprocess_backend == "cpu" else None))
        if chosen is not None:
            preprocess.selection = dict(chosen)
    return preprocess, effective_preprocess_workers


def native_preparation_price(run_cfg, *, forcing_times, prepare_workers):
    """The device price of this preparation: the f00 build, then the kept
    f00 state beside ``min(prepare_workers, forcing_times - 1)`` spawned
    boundary workers, each with its own CUDA context and one side strip.
    """
    from woof.ingest.preparation_price import (
        NOMINAL_SOURCE_INVENTORIES, price_preparation)

    workers = max(0, min(int(prepare_workers), int(forcing_times) - 1))
    return price_preparation(
        "hrrr-native", [run_cfg], NOMINAL_SOURCE_INVENTORIES["hrrr-native"],
        boundary_workers=workers)


@preprocess_math_call(options_parameter="args")
def run(args):
    from woof.ingest.cpu_backend import host_step_workers

    requested_preprocess_workers = (
        "auto" if args.preprocess_workers is None
        else int(args.preprocess_workers))

    from woof.ingest.hrrr import load_hrrr_pipeline_ready_window
    from woof.ingest.lateral_bc import (
        LateralBoundaries, attach_lateral_boundaries,
        build_lateral_interval_from_sides, domain_boundary_snapshot,
        extract_lateral_side)

    target = load_hrrr_target_domain(args.domain_spec)
    source_window = required_hrrr_source_window(target)
    requested_cycle, _legacy_cycle_flag = resolve_cycle_flags(
        args.cycle, args.valid_time,
        tool="hrrr_single_domain_benchmark",
        legacy_means="the HRRR cycle", warn=explain.warn)
    source_forecast_hours = hrrr_source_window(
        cycle=requested_cycle, start_hour=args.forecast_start_hour,
        run_seconds=args.run_seconds, end_hour=args.forecast_end_hour)
    namelist_invariant = _validated_namelist_extension_identity(
        args, cycle=requested_cycle)
    # The refusal lives in _check_outdir, at parse time; this is only the
    # creation.  exist_ok follows the explicit opt-in so a direct run()
    # caller that never went through the parser keeps the old strictness.
    args.outdir.mkdir(parents=True,
                      exist_ok=bool(getattr(args, "allow_existing", False)))
    progress_path = args.outdir / "progress.json"
    model_forcing_hours = tuple(range(len(source_forecast_hours)))
    model_start_time = requested_cycle + timedelta(
        hours=source_forecast_hours[0])
    vertical_grid = explicit_vertical_from_wrf_namelist(
        args.namelist_input,
        expected_nz=target.nz,
        context="native HRRR initializer",
    )
    from woof.hrrr_configuration import resolve_root_experiment
    exp, experiment_tables = resolve_root_experiment(
        target=target, vertical=vertical_grid, namelist_input=args.namelist_input,
        start_time=model_start_time, run_seconds=args.run_seconds,
        experiment_config=getattr(args, "experiment_config", None),
        wps_namelist=getattr(args, "wps_namelist", None),
        physics_profile=args.physics_profile, acknowledgements=tuple(args.ack),
        history_interval_seconds=args.history_interval_seconds)
    # [perturbation]: a tree's bubbles are applied by the domain-tree
    # forecast runner at start, so a preparation records them as deferred
    # in the root's identity, where the hierarchy stage reads them.  This
    # script's own forecast integrates d01 alone and applies no bubble, so
    # it refuses the block rather than run the unperturbed state under
    # the bubbles' name.
    if not args.prepare_only:
        from woof.experiment import refuse_unrouted_perturbation
        refuse_unrouted_perturbation(
            exp, "native HRRR single-domain benchmark forecast")
    from woof.hrrr_configuration import root_perturbation_deferral
    initial_perturbation = root_perturbation_deferral(
        getattr(args, "experiment_config", None))
    from woof.case_data import optional_case_data_from_tables
    companion_source = getattr(args, "experiment_config", None) or args.namelist_input
    declared_case = optional_case_data_from_tables(
        experiment_tables, source=str(companion_source),
        base_dir=Path(companion_source).parent)
    from woof.ingest.native_supplements import native_pressure_policy, require_native_pressure_field
    case_policy = native_pressure_policy(args.namelist_input, declared_case)
    if args.prepared_cache is None or not args.prepared_cache.exists():
        if args.pipeline_series is not None:
            from tools.hrrr_pipeline import _parse_series
            for row in _parse_series(args.pipeline_series):
                require_native_pressure_field(case_policy, bindings=row[3:])
        else:
            require_native_pressure_field(case_policy, bridge_root=args.bridge)
    trace_gas_overrides = ({"co2": declared_case.co2_vmr}
        if declared_case is not None and declared_case.co2_vmr is not None else None)
    from woof.ingest.water_overlay import load_bound_water_overlay
    water_overlay, water_overlay_binding = load_bound_water_overlay(
        None if declared_case is None else declared_case.water_temperature_overlay)
    from woof.static.highres_production import (resolve_static_highres, static_highres_identity)
    # The root's run picks the urban land-cover legend, as it does for the
    # static builder (load_static_highres); without it an urban run's
    # receipt and this carrier named different legends and the static was
    # refused, and the seal recorded no legend for a restore to check.
    static_highres = resolve_static_highres(experiment_tables,
        source=str(companion_source), base_dir=Path(companion_source).parent,
        run_config=exp.root.run)
    physics_profile = _configured_physics_receipt(
        exp.root.run, args.physics_profile, acknowledgements=tuple(args.ack))
    eta = np.asarray(vertical_grid.eta_levels, dtype=np.float64)
    p_top = vertical_grid.p_top
    history_interval_seconds = float(exp.root.history_interval_s)
    if args.publish_experiment_config is not None:
        from woof.experiment_document import publish_experiment_document
        published_config = publish_experiment_document(
            args.publish_experiment_config, experiment_tables, exp)
        print(f"published experiment authority: {published_config}", flush=True)
    output_cadence_receipt = (
        _validate_history_output_cadence(exp, history_interval_seconds)
        if args.io_mode == "history" or args.prepare_only else None)
    _validate_resolved_hrrr_profile(exp, physics_profile)
    dc = exp.domains[0]
    grid = benchmark_grid(target)
    soil_mesh = _configured_soil_mesh(grid, experiment_tables)
    if grid.latlon_mass()[0].shape != (target.ny, target.nx):
        raise ValueError("target Lambert geometry shape drift")

    # THE BACKEND, PRICED, BEFORE THE FIRST DEVICE ALLOCATION.  Everything
    # above is host work.  A restored prepared cache builds nothing on the
    # card, so it is not priced (A65).
    preparing = not (args.prepared_cache is not None
                     and args.prepared_cache.exists())
    preprocess, effective_preprocess_workers = _budgeted_preprocess_backend(
        args, price=(native_preparation_price(
            dc.run, forcing_times=len(model_forcing_hours),
            prepare_workers=args.prepare_workers) if preparing else None))
    preprocess_is_cuda = getattr(preprocess, "name", None) == "cuda"
    preprocess_receipt = preprocess.receipt()
    # CPU-only preparation/export must not import CuPy.  A benchmark that
    # continues into the GPU forecast still needs CuPy even when its mapping
    # and vertical transforms use the native CPU backend.
    cp = preprocess.array_module if preprocess_is_cuda else None
    if not args.prepare_only and cp is None:
        import cupy as cp

    static, attrs, static_load = _load_static(
        args.static_cache, args.static_receipt, target)
    from woof.static.highres_production import require_prepared_highres
    require_prepared_highres(
        json.loads(Path(args.static_receipt).read_text(encoding="utf-8")), grid,
        config=static_highres, domain_id=1, case_date=exp.start_time.date())
    timing = {"load_and_verify_cached_native_static": static_load["wall_seconds"]}
    io_before = _proc_io()
    total_started = time.perf_counter()
    pipeline_producer = None
    pipeline_report = None
    source_hash_receipt = None
    seal_process = None
    seal_receipt = None
    seal_started = None

    requested_hours = model_forcing_hours
    # The decoder reads HRRR's analyzed aerosol pair only for a
    # configuration that uses it (use_rap_aero_icbc, or
    # mp28_aerosol_source = "analysis"), the rule the fetch's record
    # subset already follows.  Read unasked, a pair NCEP masks with a
    # GRIB2 bitmap at a later lead stopped an as-posted default-profile
    # preparation that never used it.
    from woof.preparation_assets import analyzed_aerosol_domains
    decoder_reads_analyzed_aerosol = bool(analyzed_aerosol_domains(exp))
    source_identity = {
        **_source_identity(),
        "source_cycle": requested_cycle.isoformat(),
        "model_start_time": model_start_time.isoformat(),
        "source_forecast_hours": list(source_forecast_hours),
        "model_forcing_hours": list(model_forcing_hours),
        "ingest": {"soil_texture_downscale": soil_mesh.enabled},
        "trace_gas_overrides": trace_gas_overrides,
        "preparation_case_policy": case_policy,
        "water_temperature_overlay": water_overlay_binding,
        # Present only for a tree configuration carrying [perturbation];
        # absent, the identity is byte-for-byte what it always was.
        **({"initial_perturbation": initial_perturbation}
           if initial_perturbation is not None else {}),
    }

    if static_highres is not None:
        source_identity["static_highres"] = static_highres_identity(static_highres)
    physical_input = None
    physical_output = None
    physical_provider = None
    shared_posted = None
    physical_receipts = {}
    posted_bundle = None
    physical_output_receipt = None
    physical_input_path = getattr(args, "physical_input_store", None)
    physical_output_path = getattr(args, "physical_output_store", None)
    physical_provider_path = getattr(args, "physical_input_provider", None)
    physical_member_index = getattr(args, "physical_member_index", None)
    physical_requested = any(path is not None for path in (
        physical_input_path, physical_output_path, physical_provider_path))
    ordinary_physical_source_identity = dict(source_identity) if physical_requested else None
    if (physical_provider_path is None) != (physical_member_index is None):
        raise ValueError("physical-input-provider and physical-member-index must be supplied together")
    if physical_provider_path is not None:
        if args.as_posted is None or physical_input_path is not None or physical_output_path is not None:
            raise ValueError("a posted provider requires --as-posted and cannot also capture or replay a sealed physical store")
        from woof.ensemble.posted_physical import PostedPhysicalProvider
        physical_provider = PostedPhysicalProvider.open(
            physical_provider_path, cpu_bridge=args.cpu_preprocess_bridge,
            workers=host_step_workers(preprocess))
        from tools.hrrr_posted_reuse import SharedPostedHrrr
        shared_posted = SharedPostedHrrr(
            physical_provider.source_context(physical_member_index),
            cycle=requested_cycle, source_forecast_hours=source_forecast_hours)
        from woof.ensemble.posted_native import checked_source_inputs
        shared_posted.checked = checked_source_inputs(
            shared_posted.context, source="hrrr",
            experiment_config=shared_posted.context.prepared_root / "experiment.toml",
            wps_namelist=shared_posted.context.prepared_root / "namelist.wps",
            physics_profile=args.physics_profile, expert_acknowledgements=tuple(args.ack),
            history_interval_seconds=history_interval_seconds)
    if physical_input_path is not None and args.as_posted is not None:
        raise ValueError("sealed physical input cannot replace an as-posted provider")
    if physical_requested:
        if (not args.prepare_only
                or (args.prepared_cache is not None and args.prepared_cache.exists())):
            raise ValueError("physical input/output requires a fresh native preparation")
        from woof.ensemble.physical_store import (
            NativePhysicalStore, physical_input_binding, physical_static_identity)
        physical_static = physical_static_identity(static, attrs=attrs)
        if physical_input_path is not None:
            physical_input = NativePhysicalStore(physical_input_path)
            from woof.ensemble.hrrr_physical_contract import validate_hrrr_physical_field_contract
            validate_hrrr_physical_field_contract(physical_input.require_field_contract(), physical_input.document["grid"])
            expected_times = tuple(model_start_time + timedelta(hours=hour)
                                   for hour in model_forcing_hours)
            if physical_input.times != expected_times:
                raise ValueError("physical input must provide every native forcing valid time exactly")
            source_identity["ensemble_physical_input"] = physical_input_binding(
                physical_input, grid, dc.run, source_identity,
                input_manifest_sha256=args.source_manifest_sha256,
                static_identity=physical_static)
        if physical_output_path is not None and args.as_posted is None:
            from woof.native_wrf_contract import native_geometry_contract
            from woof.ensemble.hrrr_physical_contract import hrrr_physical_field_contract
            from woof.ingest import water_temperature
            geometry = native_geometry_contract(grid, dc.run)
            evidence = {
                "native_mapper": source_identity["source_sha256"]["woof/ingest/hrrr.py"],
                "raw_source_manifest": args.source_manifest_sha256,
                "water_temperature_assembly": _sha256(Path(water_temperature.__file__)),
                "target_static_cache": _sha256(args.static_cache),
            }
            if args.pipeline_decoder is not None:
                evidence["native_decoder_executable"] = _sha256(args.pipeline_decoder)
            else:
                evidence["sealed_native_bridge_manifest"] = _sha256(args.bridge / "SHA256SUMS")
            physical_output = NativePhysicalStore(
                physical_output_path, grid_identity=geometry,
                source_identity={**source_identity,
                                 "input_manifest_sha256": args.source_manifest_sha256,
                                 "static_identity": physical_static},
                field_contract=hrrr_physical_field_contract(geometry, evidence=evidence))
    namelist_sha256 = _sha256(args.namelist_input)
    prepared_cache_receipt = None
    prepared_cache_identity = None
    restore_cached = args.prepared_cache is not None \
        and args.prepared_cache.exists()
    chain = None if restore_cached else _chained_bundle(args)
    nudging_writer = None
    if getattr(exp, "nudging", None) is not None and not restore_cached:
        from woof.spectral_nudging.prepared import analysis_writer_from_request
        nudging_writer = analysis_writer_from_request(args, exp, chain,
            tuple(model_start_time + timedelta(hours=hour) for hour in requested_hours))
    sealed_leads = _SealedBridgeLeads()
    #: As posted: the lead admission, the fetch's lead wait, and each lead's
    #: decoded-file record by lead (_decoded_lead_record).
    admitter = None
    posted_leads = None
    decoded_records = {}
    writer = None
    lbc_digest = None
    if restore_cached and args.pipeline_series is not None:
        raise ValueError(
            "pipeline mode requires a new prepared-cache output path")
    preprocess_worker_budget = _PreprocessWorkerBudget(
        backend=str(preprocess_receipt["backend"]),
        requested_total=(
            requested_preprocess_workers
            if not preprocess_is_cuda else args.preprocess_workers),
        effective_total=effective_preprocess_workers,
        requested_job_slots=int(args.prepare_workers),
        future_job_count=(0 if restore_cached else len(requested_hours) - 1),
        clock_origin=total_started,
        host_workers=(preprocess.effective_host_workers
                      if preprocess_is_cuda else None),
    )
    if args.prepared_cache is not None:
        from woof.ingest.prepared_cache import prepared_cache_identity as identity

        def make_prepared_cache_identity(bridge_manifest_sha256,
                                         source_manifest_sha256=None):
            # An as-posted head names the plan's placeholder for both
            # documents; every other caller the source manifest it bound.
            return identity(
                bridge_manifest_sha256=bridge_manifest_sha256,
                source_manifest_sha256=(
                    args.source_manifest_sha256
                    if source_manifest_sha256 is None
                    else source_manifest_sha256),
                static_cache_sha256=static_load["cache_sha256"],
                namelist_sha256=namelist_sha256,
                domain_config=dc, forcing_hours=requested_hours,
                source_identity=source_identity,
                namelist_extension_invariant=namelist_invariant)

        if args.manifest_sha256 is not None:
            prepared_cache_identity = make_prepared_cache_identity(
                args.manifest_sha256)

    setup_records = []
    mapping_reports = {}
    setup_gpu_peak_used = 0
    setup_pool_peak_total = 0
    setup_worker_peak_rss_by_pid = {}
    preprocess_backends_by_workers = (
        {int(effective_preprocess_workers): preprocess}
        if effective_preprocess_workers is not None else {})

    def preprocess_backend_for_workers(workers):
        if preprocess_is_cuda:
            if workers is not None:
                raise RuntimeError("CUDA preprocessing received CPU workers")
            return preprocess
        workers = int(workers)
        if workers not in preprocess_backends_by_workers:
            # The controller's backend at the slot's share, not a second
            # resolution: its receipt must be the controller's.
            preprocess_backends_by_workers[workers] = preprocess.at_workers(
                workers)
        return preprocess_backends_by_workers[workers]

    def record_setup_memory():
        nonlocal setup_gpu_peak_used, setup_pool_peak_total
        if cp is not None:
            free, total = cp.cuda.runtime.memGetInfo()
            setup_gpu_peak_used = max(setup_gpu_peak_used, int(total - free))
            setup_pool_peak_total = max(
                setup_pool_peak_total,
                int(cp.get_default_memory_pool().total_bytes()))

    def synchronize_setup():
        if cp is not None:
            cp.cuda.Stream.null.synchronize()

    def free_setup_pool():
        if cp is not None:
            cp.get_default_memory_pool().free_all_blocks()

    def require_preprocess_receipt(
            observed, context, *, expected_native_workers=None):
        _require_preprocess_receipt(
            preprocess_receipt, observed, context,
            expected_native_workers=expected_native_workers)

    root_result = root_met = initial_snapshot = None
    # WHICH AEROSOL SOURCE THE f00 STATE CAME FROM, held across both
    # roads so the report can name it once.  The FRESH road gets it from
    # the ingest that made the decision; the RESTORE road gets it from
    # the prepared cache the deciding process wrote it into.  Never
    # re-resolved here: a second resolution over the same config field is
    # how a run reads one dataset and reports another.  ``{}`` for every
    # scheme with no aerosol number fields, which is what keeps the
    # report of such a run byte-identical.
    root_aerosol_initialization: dict = {}
    # The SOLVED Noah surface, when the state came from a prepared cache.
    # A cache stores it and therefore drops the native SOILT/SOILW pair
    # from the met contract beside it, so a restore that let physics setup
    # re-derive soil died on `missing soil input field(s): ['ST000007',
    # ...]` while the answer sat in surface/.  None on the fresh road,
    # which holds the native pair and no surface yet.
    root_surface = None
    # Recorded runtime surface fallbacks (fetch receipt rows); empty on a
    # restored cache, whose own prepare receipt already carries them.
    runtime_surface_fallbacks: dict = {}
    last_valid_time = None
    boundaries = None
    if restore_cached:
        from woof.ingest.prepared_cache import restore_prepared_cache

        started = time.perf_counter()
        actual_manifest = sha256_file(args.bridge / "SHA256SUMS")
        if actual_manifest != args.manifest_sha256:
            raise ValueError(
                "sealed bridge manifest differs from --manifest-sha256 "
                "before prepared-cache restore")
        timing["verify_bridge_manifest_for_prepared_cache"] = (
            time.perf_counter() - started)
        started = time.perf_counter()
        restored = restore_prepared_cache(
            args.prepared_cache, expected_identity=prepared_cache_identity,
            cfg=dc.run, static=static)
        synchronize_setup()
        timing["restore_prepared_state_and_all_lbc"] = (
            time.perf_counter() - started)
        timing["all_root_lbc_bound_seconds_from_startup"] = (
            time.perf_counter() - total_started)
        prepared_cache_receipt = dict(restored.receipt)
        root_result = restored.initial_result
        root_aerosol_initialization = dict(
            restored.metadata.get(AEROSOL_SOURCE_KEY, {}))
        root_met = restored.met
        root_surface = restored.surface
        boundaries = restored.boundaries
        cache_metadata = dict(restored.metadata)
        initial_valid_time = datetime.fromisoformat(
            cache_metadata["initial_valid_time"])
        last_valid_time = datetime.fromisoformat(
            cache_metadata["last_valid_time"])
        initial_snapshot = SimpleNamespace(valid_time=initial_valid_time)
        mapping_reports = dict(cache_metadata.get("mapping_reports", {}))
        available_hours = tuple(cache_metadata["forcing_hours"])
        if available_hours != requested_hours:
            raise ValueError(
                "prepared cache forcing-hour inventory differs from request")
        if tuple(cache_metadata.get("source_forecast_hours", ())) \
                != source_forecast_hours:
            raise ValueError(
                "prepared cache source-lead inventory differs from request")
        setup_records.append({
            "source": "prepared_cache",
            "restore_seconds": timing["restore_prepared_state_and_all_lbc"],
            "completed_seconds_from_startup": time.perf_counter() - total_started,
        })
        record_setup_memory()
        _atomic_json(progress_path, {
            "status": "PREPARED_CACHE_RESTORED",
            "elapsed_setup_wall_seconds": time.perf_counter() - total_started,
            "requested_run_seconds": args.run_seconds,
            "prepared_cache_content_sha256": restored.receipt[
                "content_sha256"],
        })
    elif shared_posted is not None:
        posted_leads = shared_posted
        available_hours = requested_hours
        acquire_snapshot = shared_posted.acquire
    elif args.pipeline_series is not None and args.as_posted is not None:
        # AS POSTED (A136 L7c (b)): no lead is hashed up front.  Each is
        # admitted to the decoder once its posted marker is there and its
        # two files are the marker's (PostedLeadAdmitter), and the decoder
        # inventories the first lead up front and each later one when it is
        # admitted (--series-workers-posted).
        from tools.hrrr_pipeline import (
            HrrrPipelineProducer, PostedLeadAdmitter)
        posted_leads = _posted_native_leads(
            args.as_posted, cycle=requested_cycle,
            source_forecast_hours=source_forecast_hours)
        admitter = PostedLeadAdmitter(
            posted=posted_leads, series=args.pipeline_series,
            admissions=args.pipeline_signals.with_name(
                args.pipeline_signals.name + "-admitted"))
        if admitter.hours != tuple(source_forecast_hours):
            raise ValueError(
                "pipeline source leads differ from the requested window: "
                f"{list(admitter.hours)} != {list(source_forecast_hours)}")
        pipeline_producer = HrrrPipelineProducer(
            decoder=args.pipeline_decoder, series=args.pipeline_series,
            output=args.bridge, signals=args.pipeline_signals,
            cycle=requested_cycle.strftime("%Y-%m-%d %H:%M:%S"),
            window=source_window.bridge_tuple(),
            workers=args.pipeline_workers,
            log=args.pipeline_signals.with_suffix(".decoder.log"),
            admissions=admitter.admissions,
            analyzed_aerosol=decoder_reads_analyzed_aerosol)
        started = time.perf_counter()
        pipeline_producer.start()
        admitter.start()

        def admitted(lead):
            return _await_admitted(admitter, pipeline_producer, lead)

        try:
            admitted(source_forecast_hours[0])
            timing["posted_start_lead_wait"] = time.perf_counter() - started
            pipeline_producer.wait_preflight()
        except BaseException:
            admitter.stop()
            pipeline_producer.cancel()
            raise
        timing["decoder_inventory_preflight_wait"] = time.perf_counter() - started
        available_hours = requested_hours

        def acquire_snapshot(hour):
            # Waits for the lead's admission (posted, held to its marker),
            # then for its decode; the decoded files are recorded as they
            # are taken, so each segment binds the bytes it was built from.
            source_hour = source_forecast_hours[hour]
            admitted(source_hour)
            root = pipeline_producer.wait_hour(source_hour)
            decoded_records[source_hour] = _decoded_lead_record(
                root, source_hour, workers=host_step_workers(preprocess))
            return load_hrrr_pipeline_ready_window(root, source_hour)
    elif args.pipeline_series is not None:
        from tools.hrrr_pipeline import (
            HrrrPipelineProducer, verify_source_tree)
        source_hash_receipt = verify_source_tree(
            source_root=args.source_root, manifest=args.source_manifest,
            expected_manifest_sha256=args.source_manifest_sha256,
            series=args.pipeline_series, workers=13)
        if tuple(source_hash_receipt["forecast_hours"]) \
                != source_forecast_hours:
            raise ValueError(
                "pipeline source leads differ from the requested window: "
                f"{source_hash_receipt['forecast_hours']} != "
                f"{source_forecast_hours}")
        timing["source_hash_preflight"] = source_hash_receipt["wall_seconds"]
        pipeline_producer = HrrrPipelineProducer(
            decoder=args.pipeline_decoder, series=args.pipeline_series,
            output=args.bridge, signals=args.pipeline_signals,
            cycle=requested_cycle.strftime("%Y-%m-%d %H:%M:%S"),
            window=source_window.bridge_tuple(),
            workers=args.pipeline_workers,
            log=args.pipeline_signals.with_suffix(".decoder.log"),
            analyzed_aerosol=decoder_reads_analyzed_aerosol)
        started = time.perf_counter()
        pipeline_producer.start()
        pipeline_producer.wait_preflight()
        timing["decoder_inventory_preflight_wait"] = time.perf_counter() - started
        available_hours = requested_hours

        def acquire_snapshot(hour):
            # A chained preparation seals the bridge before its head, and
            # the producer's staging goes with it: every later hour is read
            # from the sealed bridge, the same decoded bytes.
            if sealed_leads:
                return sealed_leads.take(hour)
            source_hour = source_forecast_hours[hour]
            root = pipeline_producer.wait_hour(source_hour)
            return load_hrrr_pipeline_ready_window(root, source_hour)
    else:
        started = time.perf_counter()
        sealed_leads.open(
            args.bridge, dict(zip(requested_hours, source_forecast_hours)),
            expected_manifest_sha256=args.manifest_sha256)
        timing["verify_native_bridge"] = time.perf_counter() - started
        available_hours = requested_hours

        def acquire_snapshot(hour):
            # Every hour is consumed exactly once, and mapped only then: a
            # long window's leads mapped at once exceed the ordinary open
            # file limit (_SealedBridgeLeads).
            if physical_input is not None:
                return SimpleNamespace(valid_time=physical_input.times[hour])
            return sealed_leads.take(hour)
        if physical_input is not None or physical_output is not None:
            pipeline_report = {
                "operation": "reused_sealed_native_bridge",
                "bridge_manifest_sha256": args.manifest_sha256,
                "workers": {"requested": str(args.pipeline_workers).strip().lower(),
                            "selected": 0, "operation": "reused_sealed_native_bridge"}}

    if args.as_posted is not None and physical_requested:
        if chain is None or posted_leads is None:
            raise ValueError("posted physical preparation requires the ordinary portable posted head")
        if water_overlay is not None:
            raise ValueError("a declared whole-window water overlay cannot publish a posted native head")
        posted_bundle = _posted_bundle_plan(
            args, chain=chain, requested_cycle=requested_cycle,
            source_forecast_hours=source_forecast_hours,
            model_forcing_hours=model_forcing_hours,
            preprocess_receipt=preprocess_receipt, source_identity=source_identity,
            posted_leads=posted_leads, metadata={})
        from woof.ingest.boundary_stream import input_plan_sha256, as_posted_placeholder
        from woof.ensemble.posted_physical import PostedPhysicalStream, posted_source_identity
        posted_plan = posted_bundle[1]
        posted_digest = input_plan_sha256(posted_plan)
        if physical_output_path is not None:
            from datetime import timezone
            from woof.ensemble.recipes import SourceTrajectory
            from woof.native_wrf_contract import native_geometry_contract
            from woof.ensemble.hrrr_physical_contract import hrrr_physical_field_contract
            from woof.ingest import water_temperature
            geometry = native_geometry_contract(grid, dc.run)
            captured_identity = posted_source_identity(
                {**source_identity, "input_manifest_sha256": as_posted_placeholder(posted_digest),
                 "static_identity": physical_static}, input_plan=posted_plan)
            contract = hrrr_physical_field_contract(geometry, evidence={
                "native_mapper": source_identity["source_sha256"]["woof/ingest/hrrr.py"],
                "ordinary_source_input_plan": posted_digest,
                "native_decoder_executable": _sha256(args.pipeline_decoder),
                "water_temperature_assembly": _sha256(Path(water_temperature.__file__)),
                "target_static_cache": _sha256(args.static_cache)})
            stream = PostedPhysicalStream.create(
                physical_output_path,
                trajectory=SourceTrajectory("hrrr", requested_cycle.replace(tzinfo=timezone.utc), None),
                valid_times=tuple((model_start_time + timedelta(hours=hour)).replace(tzinfo=timezone.utc)
                                  for hour in requested_hours),
                grid_identity=geometry, source_identity=captured_identity,
                field_contract=contract, input_plan_sha256=posted_digest)
            physical_output = _PostedHrrrCapture(
                stream, cycle=requested_cycle, source_forecast_hours=source_forecast_hours,
                admitter=admitter, decoded_records=decoded_records)

    def resolve_physical_hour(hour, valid_time):
        if physical_provider is None:
            return physical_input
        from woof.ensemble.posted_physical import bind_posted_physical_input
        from woof.ensemble.hrrr_physical_contract import validate_hrrr_physical_field_contract
        store, receipt = physical_provider.resolve(physical_member_index, valid_time)
        validate_hrrr_physical_field_contract(store.require_field_contract(), store.document["grid"])
        binding = bind_posted_physical_input(
            store, receipt, provider_plan=physical_provider.plan,
            member_index=physical_member_index, valid_time=valid_time, grid=grid, cfg=dc.run,
            source_identity={**ordinary_physical_source_identity,
                             "input_manifest_sha256": as_posted_placeholder(posted_digest)},
            input_plan=posted_plan, static_identity=physical_static)
        physical_receipts[hour] = binding
        if hour == 0:
            source_identity["ensemble_posted_physical_input"] = binding
        return store

    overlay_series = None
    if not restore_cached and water_overlay is not None:
        acquire_snapshot, overlay_series = _overlay_acquirer(
            acquire_snapshot, requested_hours, water_overlay, water_overlay_binding,
            workers=host_step_workers(preprocess), physical_input=physical_input)

    if not restore_cached:
        from woof.ingest.boundary_stream import say_prepared_sealed
        from woof.ingest.water_temperature import WaterTemperatureStatics
        if chain is not None and water_overlay is not None:
            # The overlay's receipt is verified over every forcing time
            # (verify_overlay_sequence) before the cache it binds may be
            # published, which a head cannot wait for.
            say_prepared_sealed("water_overlay")
            chain = None
        water_statics = WaterTemperatureStatics.for_route(
            route="native preparation", policy=case_policy["water_temperature_policy"],
            landmask=static["LANDMASK"], lu_index=static["LU_INDEX"], landuse_attrs=attrs)
        intervals = []
        boundary_sides_by_hour = {}
        valid_time_by_hour = {}
        setup_record_by_hour = {}
        width = int(dc.run.spec_bdy_width)
        static_sides = _boundary_static_sides(
            static, grid, dc.run, width=width)
        boundary_targets = _boundary_mapping_targets(
            grid, static, dc.run, width=width)
        try:
            # f00 is the sole full-domain initialization.  It is retained as
            # the integration state and supplies the first boundary snapshot.
            ready_started = time.perf_counter()
            snapshot = acquire_snapshot(0)
            ready_wait = time.perf_counter() - ready_started
            mapping = {}
            if preprocess_is_cuda:
                f00_worker_slot, f00_native_workers = None, None
            else:
                f00_worker_slot, f00_native_workers = (
                    preprocess_worker_budget.allocation_for_hour(0))
            f00_preprocess = preprocess_backend_for_workers(
                f00_native_workers)
            f00_job_started = time.perf_counter()
            initial_physical = resolve_physical_hour(0, snapshot.valid_time)
            if shared_posted is not None:
                from woof.ensemble.posted_native import shared_surface
                base_store, _ = shared_posted.context.physical_stream.require(
                    shared_posted.context.physical_stream.times[0])
                root_surface = shared_surface(shared_posted.checked,
                    base_met=base_store.read(0), member_met=initial_physical.read(0))
            # THE START STATE TAKES THE HOST'S COLUMN THREADS, NOT THE JOB
            # SLOTS.  --prepare-workers counts the forcing times prepared at
            # once (default 2); f00 runs alone before any of them, so handing
            # it that count left a 1262 x 1224 1 km start state at 1.1-1.6
            # busy cores for 108 s while the card waited. The column helpers are slab-exact
            # (tests/test_real_init.py), so the count moves no byte; the
            # backend's own host-step budget bounds the threads and their RAM.
            from woof.ingest.real import (
                _column_worker_count, _default_column_workers)
            f00_column_workers = _column_worker_count(
                _default_column_workers(f00_preprocess, f00_native_workers))
            with _prep_step(args, "root_initialize",
                            label="Initialize the start state"):
                result, met, horizontal, vertical, state_timing = _initialize_state(
                    snapshot, dc, grid, static, eta, mapping,
                    p_top=p_top, sfcp_to_sfcp=case_policy["sfcp_to_sfcp"],
                    water_temperature_statics=water_statics,
                    column_workers=f00_column_workers,
                    surface_fallback_radius=(
                        target.surface_fallback_radius_cells),
                    preprocess_backend=f00_preprocess,
                    state_backend=(
                        "preprocess" if args.prepare_only else "cuda"),
                    physical_input=initial_physical, physical_output=physical_output)
            f00_job_finished = time.perf_counter()
            if not preprocess_is_cuda:
                preprocess_worker_budget.record(
                    forecast_hour=0, phase="full_domain_initialization",
                    slot=f00_worker_slot,
                    native_workers=int(f00_native_workers),
                    started=f00_job_started, finished=f00_job_finished)
            record_setup_memory()
            require_preprocess_receipt(
                mapping.get("preprocess_backend"), "f00 mapping",
                expected_native_workers=f00_native_workers)
            mapping_reports[f"f{source_forecast_hours[0]:02d}"] = mapping
            from woof.ingest.prepared_cache import select_prepared_met_fields

            root_result = result
            if nudging_writer is not None:
                nudging_writer.write(0, root_result)
            root_aerosol_initialization = dict(
                getattr(result, "aerosol_initialization", {}) or {})
            # Physics setup/cache writing need only this detached host subset.
            # Keeping the complete mapped f00 snapshot here otherwise pins
            # multiple full-domain 3-D device arrays through all later hours.
            root_met = select_prepared_met_fields(met)
            initial_snapshot = SimpleNamespace(valid_time=snapshot.valid_time)
            valid_time_by_hour[0] = snapshot.valid_time
            coupled = domain_boundary_snapshot(root_result.state)
            boundary_sides_by_hour[0] = {
                side: dict(extract_lateral_side(coupled, side, width))
                for side in ("west", "east", "south", "north")
            }
            del coupled, met, snapshot
            free_setup_pool()
            setup_record_by_hour[0] = {
                "forecast_hour": 0,
                "model_forcing_hour": 0,
                "source_forecast_hour": source_forecast_hours[0],
                "preparation_kind": "full_initial_state",
                "ready_wait_seconds": ready_wait,
                "horizontal_seconds": horizontal,
                "state_seconds": vertical,
                "state_timing_seconds": state_timing,
                "column_workers": int(f00_column_workers),
                "preprocess_worker_slot": f00_worker_slot,
                "effective_native_workers": f00_native_workers,
                "completed_seconds_from_startup": (
                    time.perf_counter() - total_started),
            }
            _atomic_json(progress_path, {
                "status": "PREPARING",
                "completed_source_forecast_hours": [
                    source_forecast_hours[0]],
                "completed_model_forcing_hours": [0],
                "prepare_workers": args.prepare_workers,
                "elapsed_setup_wall_seconds": time.perf_counter() - total_started,
                "requested_run_seconds": args.run_seconds,
            })

            completed_hours = {0}
            if chain is not None and (admitter is not None or shared_posted is not None):
                (writer, lbc_digest, posted_identity, root_surface,
                 soil_temperature_repair) = _write_posted_head(
                    args, chain=chain, exp=exp, dc=dc, grid=grid,
                    static=static, soil_mesh=soil_mesh, admitter=admitter,
                    producer=pipeline_producer, posted_leads=posted_leads,
                    decoded_records=decoded_records, timing=timing,
                    make_identity=make_prepared_cache_identity,
                    requested_cycle=requested_cycle,
                    source_forecast_hours=source_forecast_hours,
                    model_forcing_hours=model_forcing_hours,
                    requested_hours=requested_hours,
                    initial_snapshot=initial_snapshot,
                    root_result=root_result, root_met=root_met,
                    mapping_reports=mapping_reports,
                    boundary_sides=boundary_sides_by_hour[0],
                    preprocess_receipt=preprocess_receipt,
                    source_identity=source_identity, posted_bundle=posted_bundle,
                    ensemble_physical=(None if physical_provider is None else {
                        "provider_plan": physical_provider.plan,
                        "member_index": physical_member_index,
                        "initial_receipt": physical_receipts[0]}),
                    physical_receipts=(physical_receipts if physical_provider is not None else None),
                    shared_source=shared_posted, reused_surface=root_surface)
                if shared_posted is not None:
                    from woof.ensemble.posted_native import writer_source_wait
                    physical_provider.set_wait_observer(writer_source_wait(writer))
            elif chain is not None:
                (writer, lbc_digest, prepared_cache_identity, root_surface,
                 soil_temperature_repair, pipeline_report) = _write_chained_head(
                    args, chain=chain, exp=exp, dc=dc, grid=grid,
                    static=static, soil_mesh=soil_mesh,
                    pipeline_producer=pipeline_producer,
                    source_hash_receipt=source_hash_receipt,
                    source_window=source_window, timing=timing,
                    make_identity=make_prepared_cache_identity,
                    sealed_leads=sealed_leads,
                    requested_cycle=requested_cycle,
                    source_forecast_hours=source_forecast_hours,
                    model_forcing_hours=model_forcing_hours,
                    requested_hours=requested_hours,
                    initial_snapshot=initial_snapshot,
                    root_result=root_result, root_met=root_met,
                    mapping_reports=mapping_reports,
                    boundary_sides=boundary_sides_by_hour[0],
                    preprocess_receipt=preprocess_receipt,
                    source_identity=source_identity)
                pipeline_producer = None
            next_segment = [0]

            def flush_segments():
                # Interval k closes once hours k and k+1 both have their
                # sides, in time order whatever order the workers finish
                # in; hour k's sides are let go once interval k is written.
                while (next_segment[0] + 1) in boundary_sides_by_hour:
                    k = next_segment[0]
                    interval = build_lateral_interval_from_sides(
                        boundary_sides_by_hour[k],
                        boundary_sides_by_hour[k + 1],
                        start_seconds=float(k * 3600),
                        end_seconds=float((k + 1) * 3600))
                    lbc_digest.add(k, interval)
                    if shared_posted is None:
                        writer.write_segment(k, interval)
                    else:
                        from woof.ensemble.posted_native import relay_source_segment
                        relay_source_segment(shared_posted.context, k, writer, interval)
                    del interval, boundary_sides_by_hour[k]
                    next_segment[0] = k + 1

            def map_boundary_hour(
                    hour, worker_slot, native_workers, hour_preprocess):
                source_hour = source_forecast_hours[hour]
                ready_started = time.perf_counter()
                hour_snapshot = acquire_snapshot(hour)
                ready_wait = time.perf_counter() - ready_started
                mapping = {}
                mapping_started = time.perf_counter()
                hour_physical = resolve_physical_hour(hour, hour_snapshot.valid_time)
                if nudging_writer is not None:
                    # The ordinary boundary path keeps its exact strip arrays.
                    # A selected analysis additionally needs the full domain;
                    # it uses the same native mapper and column initializer.
                    analysis, analysis_met, _, _, _ = _initialize_state(
                        hour_snapshot, dc, grid, static, eta, {}, p_top=p_top,
                        sfcp_to_sfcp=case_policy['sfcp_to_sfcp'],
                        water_temperature_statics=water_statics,
                        column_workers=f00_column_workers,
                        surface_fallback_radius=target.surface_fallback_radius_cells,
                        preprocess_backend=hour_preprocess, state_backend='preprocess')
                    nudging_writer.write(requested_hours.index(hour), analysis)
                    del analysis, analysis_met
                if hour_physical is not None:
                    met = hour_physical.read(0 if physical_provider is not None else hour)
                    if met.valid_time != hour_snapshot.valid_time:
                        raise ValueError("physical boundary valid time differs from the native source")
                    from woof.ensemble.hrrr_physical_contract import validate_hrrr_physical_snapshot
                    validate_hrrr_physical_snapshot(met)
                    compact = _compact_boundary_inputs(met, dc, width=width)
                    mapping.update(_physical_mapping_receipt(hour_physical, hour_preprocess))
                    if physical_output is not None:
                        physical_output.write(met)
                    del met
                    horizontal = time.perf_counter() - mapping_started
                else:
                    compact, horizontal = _map_boundary_snapshot(
                        hour_snapshot, boundary_targets, mapping,
                        surface_fallback_radius=(
                            target.surface_fallback_radius_cells),
                        preprocess_backend=hour_preprocess,
                        domain_grid=grid)
                    if physical_output is not None:
                        # Capture is observational: the ordinary boundary strips
                        # above remain the exact arrays the initializer consumes.
                        # The full map is made only when explicitly requested.
                        capture_mapping = {}
                        met, capture_seconds = _map_snapshot(
                            hour_snapshot, grid, static, capture_mapping,
                            surface_fallback_radius=target.surface_fallback_radius_cells,
                            preprocess_backend=hour_preprocess)
                        from woof.ingest.water_temperature import assemble_horizontal_water_temperature
                        met = assemble_horizontal_water_temperature(
                            met, water_statics, workers=host_step_workers(hour_preprocess))
                        physical_output.write(met)
                        mapping["physical_capture"] = capture_mapping
                        mapping["physical_capture_seconds"] = capture_seconds
                        del met
                mapping_finished = time.perf_counter()
                if not preprocess_is_cuda:
                    preprocess_worker_budget.record(
                        forecast_hour=hour, phase="boundary_mapping",
                        slot=worker_slot,
                        native_workers=int(native_workers),
                        started=mapping_started, finished=mapping_finished)
                record_setup_memory()
                for side, side_report in mapping.get("sides", {}).items():
                    require_preprocess_receipt(
                        side_report.get("preprocess_backend"),
                        f"f{source_hour:02d} {side} mapping",
                        expected_native_workers=native_workers)
                mapping_reports[f"f{source_hour:02d}"] = mapping
                valid_time_by_hour[hour] = hour_snapshot.valid_time
                setup_record_by_hour[hour] = {
                    "forecast_hour": int(hour),
                    "model_forcing_hour": int(hour),
                    "source_forecast_hour": int(source_hour),
                    "preparation_kind": "boundary_slabs",
                    "ready_wait_seconds": ready_wait,
                    "horizontal_seconds": horizontal,
                    "preprocess_worker_slot": worker_slot,
                    "effective_native_workers": native_workers,
                    "mapping_completed_seconds_from_startup": (
                        time.perf_counter() - total_started),
                }
                del hour_snapshot
                # Compact slabs are detached host arrays.  Return the now-idle
                # parent mapping allocation to CUDA before a worker initializes
                # the same hour, allowing mapping and slab setup to overlap on
                # a bounded VRAM footprint.
                free_setup_pool()
                return compact

            def record_boundary_worker(hour, worker):
                nonlocal setup_gpu_peak_used, setup_pool_peak_total
                if int(worker["forecast_hour"]) != int(hour):
                    raise RuntimeError("boundary worker returned the wrong hour")
                expected_slot = setup_record_by_hour[hour][
                    "preprocess_worker_slot"]
                expected_workers = setup_record_by_hour[hour][
                    "effective_native_workers"]
                if (worker["preprocess_worker_slot"] != expected_slot
                        or worker["effective_native_workers"]
                        != expected_workers):
                    raise RuntimeError(
                        "boundary worker used the wrong preprocessing slot")
                require_preprocess_receipt(
                    worker.pop("preprocess_backend"),
                    (f"f{source_forecast_hours[hour]:02d} boundary "
                     "initialization"),
                    expected_native_workers=expected_workers)
                if not preprocess_is_cuda:
                    preprocess_worker_budget.record(
                        forecast_hour=hour,
                        phase="boundary_initialization",
                        slot=int(expected_slot),
                        native_workers=int(expected_workers),
                        started=worker["worker_started_monotonic"],
                        finished=worker["worker_finished_monotonic"])
                boundary_sides_by_hour[hour] = worker.pop("sides")
                memory = worker["memory"]
                pid = int(worker["worker_pid"])
                setup_gpu_peak_used = max(
                    setup_gpu_peak_used,
                    int(memory["gpu_used_bytes_observed"]))
                setup_pool_peak_total = max(
                    setup_pool_peak_total,
                    int(memory["cupy_pool_total_bytes"]))
                setup_worker_peak_rss_by_pid[pid] = max(
                    setup_worker_peak_rss_by_pid.get(pid, 0),
                    int(memory["worker_peak_rss_bytes"]))
                setup_record_by_hour[hour].update({
                    "side_state_seconds": worker["side_state_seconds"],
                    "worker_wall_seconds": worker["worker_wall_seconds"],
                    "worker_started_seconds_from_startup": (
                        worker["worker_started_monotonic"] - total_started),
                    "worker_finished_seconds_from_startup": (
                        worker["worker_finished_monotonic"] - total_started),
                    "worker_pid": pid,
                    "worker_memory": memory,
                    "completed_seconds_from_startup": (
                        time.perf_counter() - total_started),
                })
                completed_hours.add(hour)
                if writer is not None:
                    writer.note_build_seconds(worker["worker_wall_seconds"])
                    flush_segments()
                if args.prepare_only:
                    # "Boundary times k of m" on a run page while the
                    # preparation builds them (the start state is f00).
                    prep_progress("root_boundaries", label="Boundary times",
                                  done=len(completed_hours) - 1,
                                  count=len(requested_hours) - 1)
                _atomic_json(progress_path, {
                    "status": "PREPARING",
                    "completed_model_forcing_hours": sorted(completed_hours),
                    "completed_source_forecast_hours": [
                        source_forecast_hours[value]
                        for value in sorted(completed_hours)],
                    "prepare_workers": args.prepare_workers,
                    "elapsed_setup_wall_seconds": (
                        time.perf_counter() - total_started),
                    "requested_run_seconds": args.run_seconds,
                })

            future_hours = requested_hours[1:]
            schedule_slots = (
                preprocess_worker_budget.concurrent_job_slots
                if not preprocess_is_cuda else
                min(int(args.prepare_workers), len(future_hours)))
            if schedule_slots < 1:
                raise RuntimeError("future HRRR hours have no preparation slots")
            if schedule_slots == 1:
                for index, hour in enumerate(future_hours):
                    worker_slot = index % schedule_slots
                    if preprocess_is_cuda:
                        native_workers = None
                    else:
                        allocated_slot, native_workers = (
                            preprocess_worker_budget.allocation_for_hour(hour))
                        if allocated_slot != worker_slot:
                            raise AssertionError(
                                "preprocess worker slot assignment drift")
                    hour_preprocess = preprocess_backend_for_workers(
                        native_workers)
                    compact = map_boundary_hour(
                        hour, worker_slot, native_workers, hour_preprocess)
                    started = time.perf_counter()
                    (sides, side_timings, memory,
                     worker_preprocess_receipt) = _initialize_boundary_sides(
                        compact, dc.run, static_sides, eta,
                        p_top=p_top, width=width,
                        sfcp_to_sfcp=case_policy["sfcp_to_sfcp"],
                        preprocess_backend=hour_preprocess)
                    finished = time.perf_counter()
                    del compact
                    record_boundary_worker(hour, {
                        "forecast_hour": hour,
                        "sides": sides,
                        "side_state_seconds": side_timings,
                        "worker_wall_seconds": finished - started,
                        "worker_started_monotonic": started,
                        "worker_finished_monotonic": finished,
                        "worker_pid": os.getpid(),
                        "preprocess_worker_slot": worker_slot,
                        "effective_native_workers": native_workers,
                        "memory": memory,
                        "preprocess_backend": worker_preprocess_receipt,
                    })
            else:
                context = multiprocessing.get_context("spawn")
                executor = ProcessPoolExecutor(
                    max_workers=schedule_slots, mp_context=context,
                    initializer=_prepare_worker_init,
                    initargs=(
                        dc.run, static_sides, tuple(eta), p_top, width,
                        preprocess_receipt["backend"],
                        (str(args.cpu_preprocess_bridge)
                         if args.cpu_preprocess_bridge is not None else None),
                        case_policy["sfcp_to_sfcp"],
                        (args.preprocess_workers if preprocess_is_cuda
                         else None),
                        getattr(preprocess, "selection", None)))
                futures = {}
                slot_futures = [None] * schedule_slots

                def collect(future):
                    hour, worker_slot = futures.pop(future)
                    if slot_futures[worker_slot] is future:
                        slot_futures[worker_slot] = None
                    record_boundary_worker(hour, future.result())

                def collect_finished():
                    for future in [future for future in slot_futures
                                   if future is not None and future.done()]:
                        collect(future)

                try:
                    for index, hour in enumerate(future_hours):
                        worker_slot = index % schedule_slots
                        if admitter is not None:
                            # As posted, lead ``hour`` may not be here yet.
                            # A finished hour is collected while it is
                            # awaited, so each interval is written once its
                            # two hours are built: collected only when its
                            # slot came round again, interval k waited for
                            # lead k + slots, not k + 1 (test host D, HRRR
                            # 2026-10-01T08, two slots: interval 3 waited for
                            # f005 although f004 was in at 09:02Z).  Collection
                            # order moves no byte (segments are written in
                            # time order, receipts sorted).
                            _await_admitted(
                                admitter, pipeline_producer,
                                source_forecast_hours[hour],
                                between=collect_finished)
                            collect_finished()
                        previous = slot_futures[worker_slot]
                        if previous is not None:
                            # A deterministic hour-to-slot binding makes the
                            # static thread partition independent of process
                            # completion order.  Mapping borrows the slot only
                            # after its preceding initializer has released it.
                            collect(previous)
                        if preprocess_is_cuda:
                            native_workers = None
                        else:
                            allocated_slot, native_workers = (
                                preprocess_worker_budget.allocation_for_hour(
                                    hour))
                            if allocated_slot != worker_slot:
                                raise AssertionError(
                                    "preprocess worker slot assignment drift")
                        hour_preprocess = preprocess_backend_for_workers(
                            native_workers)
                        compact = map_boundary_hour(
                            hour, worker_slot, native_workers,
                            hour_preprocess)
                        future = executor.submit(
                            _prepare_boundary_hour, hour, compact,
                            native_workers, worker_slot)
                        del compact
                        futures[future] = (hour, worker_slot)
                        slot_futures[worker_slot] = future
                        setup_record_by_hour[hour][
                            "submitted_seconds_from_startup"] = (
                                time.perf_counter() - total_started)
                    for future in tuple(slot_futures):
                        if future is not None:
                            collect(future)
                except BaseException:
                    for future in futures:
                        future.cancel()
                    executor.shutdown(wait=True, cancel_futures=True)
                    raise
                else:
                    executor.shutdown(wait=True)

            setup_records.extend(
                setup_record_by_hour[hour] for hour in requested_hours)
            if writer is not None:
                if next_segment[0] != len(requested_hours) - 1:
                    raise RuntimeError(
                        f"the chained preparation wrote {next_segment[0]} "
                        f"of {len(requested_hours) - 1} boundary intervals")
            else:
                for hour in requested_hours[1:]:
                    intervals.append(build_lateral_interval_from_sides(
                        boundary_sides_by_hour[hour - 1],
                        boundary_sides_by_hour[hour],
                        start_seconds=float((hour - 1) * 3600),
                        end_seconds=float(hour * 3600)))
            last_valid_time = valid_time_by_hour[requested_hours[-1]]
            if shared_posted is not None:
                pipeline_report, source_hash_receipt = shared_posted.seal(args)
                prepared_cache_identity = make_prepared_cache_identity(args.manifest_sha256)
            elif admitter is not None:
                # Every lead is decoded: the source manifest is authored
                # from the leads' markers and the bridge sealed, and the
                # one-shot identity of those bytes is what the seal writes.
                pipeline_report, source_hash_receipt = _seal_posted_bridge(
                    args, admitter=admitter,
                    pipeline_producer=pipeline_producer,
                    source_window=source_window, timing=timing)
                pipeline_producer = None
                prepared_cache_identity = make_prepared_cache_identity(
                    args.manifest_sha256)
        except BaseException as error:
            if admitter is not None:
                admitter.stop()
            if pipeline_producer is not None:
                pipeline_producer.cancel()
            if writer is not None:
                writer.fail(error)
            raise

        if root_result is None or root_met is None or initial_snapshot is None:
            raise AssertionError("f00 benchmark state was not retained")
        if (last_valid_time - initial_snapshot.valid_time).total_seconds() \
                < args.run_seconds:
            raise ValueError("native bridge does not cover requested forecast")
        if writer is None:
            boundaries = LateralBoundaries(tuple(intervals), width, 1, 4)
            started = time.perf_counter()
            attach_lateral_boundaries(root_result.state, boundaries)
            timing["attach_all_root_lbc"] = time.perf_counter() - started
        timing["all_root_lbc_bound_seconds_from_startup"] = (
            time.perf_counter() - total_started)

        _verify_preparation_overlay(overlay_series, physical_input=physical_input,
                                    binding=water_overlay_binding)
        # Every preparation road (posted bundle, sealed bridge, prepared
        # cache) checks the source-declared runtime surface fields here and
        # carries any fallback its fetch receipt recorded into the report.
        from woof.static.external_source import static_source_for
        from woof.runtime_surface_fetch import require_runtime_surface_fields
        from woof.source_adapters import get_source_adapter
        runtime_surface_fallbacks = require_runtime_surface_fields(
            root_met, get_source_adapter("hrrr"),
            source_manifest=_fetch_receipt_path(args),
            refuse_dropped=static_source_for(static_highres) is not None)
        posted_proof = None
        if writer is not None and (admitter is not None or shared_posted is not None):
            try:
                from woof.hrrr_prepared_bundle import seal_hrrr_posted_inputs

                posted_inputs = seal_hrrr_posted_inputs(
                    writer.native_bundle_head,
                    output_root=Path(chain["output_root"]),
                    bridge_manifest=args.bridge / "SHA256SUMS",
                    source_manifest=Path(args.source_manifest))
                window = set(source_forecast_hours)
                source_markers = (admitter.markers if shared_posted is None else shared_posted.markers)
                source_decoded = (decoded_records if shared_posted is None else shared_posted.decoded)
                writer.write_posted_leads(
                    {lead: marker for lead, marker in source_markers.items()
                     if lead in window},
                    route_table_sha256=posted_leads.route_table_sha256(),
                    decoded=source_decoded)
                prepared_cache_receipt = _seal_chained_cache(
                    writer, timing=timing, mapping_reports=mapping_reports,
                    last_valid_time=last_valid_time,
                    identity=prepared_cache_identity,
                    manifest_sha256=posted_inputs["input_manifest_sha256"],
                    document_sha256={
                        "bridge_manifest_sha256": args.manifest_sha256,
                        "source_manifest_sha256": args.source_manifest_sha256},
                    **({"physical_provider_seal": physical_provider.seal()}
                       if physical_provider is not None else {}))
                posted_proof = {
                    **posted_inputs,
                    "posting": {"as_posted": True,
                                "waits": list(posted_leads.waits),
                                "leads_late": []}}
            except BaseException as error:
                writer.fail(error)
                raise
        elif writer is not None:
            try:
                prepared_cache_receipt = _seal_chained_cache(
                    writer, timing=timing, mapping_reports=mapping_reports,
                    last_valid_time=last_valid_time)
            except BaseException as error:
                writer.fail(error)
                raise
        elif args.prepared_cache is not None:
            # In pipeline mode the canonical bridge manifest does not exist
            # until all f00..f12 payloads have been atomically published and
            # sealed.  Finish that producer here so the launch-ready cache is
            # cryptographically bound to the downloaded GRIB inputs and the
            # final bridge, rather than to an unknown/provisional identity.
            if prepared_cache_identity is None:
                if pipeline_producer is None:
                    raise AssertionError(
                        "prepared-cache identity was not constructed")
                pipeline_report = pipeline_producer.finish()
                seal_process, seal_receipt, seal_started = _start_seal(
                    args, pipeline_report, source_hash_receipt, source_window)
                stdout, stderr = seal_process.communicate()
                timing["pipeline_bridge_seal_wall"] = (
                    time.perf_counter() - seal_started)
                if seal_process.returncode != 0:
                    raise RuntimeError(
                        "pipeline bridge seal failed: " + stderr[-4000:])
                if seal_receipt is None or not seal_receipt.is_file():
                    raise RuntimeError(
                        "pipeline bridge seal omitted receipt")
                seal = json.loads(seal_receipt.read_text())
                args.manifest_sha256 = seal["manifest_sha256"]
                pipeline_report["seal"] = seal
                pipeline_report["seal_stdout"] = stdout.strip()
                prepared_cache_identity = make_prepared_cache_identity(
                    args.manifest_sha256)
                pipeline_producer = None
                seal_process = None

            from woof.ingest.prepared_cache import write_prepared_cache
            from woof.ingest.hrrr_physics import resolve_prepared_noah_surface
            from woof.ingest.preprocess_backend import (
                preprocess_reports_identity)

            # The cache and fresh forecast consume the very same solved
            # surface, including the declared sub-source-cell soil treatment.
            root_surface = resolve_prepared_noah_surface(
                root_met, dc.run, static, soil_mesh=soil_mesh)
            canonical_surface = root_surface.fields
            # real.exe's TSLB reasonableness rebuild and the snow-covered
            # rebuild beside it, recorded only when one touched a land column, so a healthy cycle's cache and proof
            # are byte for byte what they were; the bundle relays it into
            # the proof (SOIL_PREPARATION_RECEIPTS).
            from woof.ingest.soil import soil_temperature_repair_proof
            soil_temperature_repair = soil_temperature_repair_proof(
                root_surface, grid)

            started = time.perf_counter()
            prepared_cache_receipt = write_prepared_cache(
                args.prepared_cache, identity=prepared_cache_identity,
                initial_result=root_result, met=root_met,
                boundaries=boundaries, surface=canonical_surface,
                metadata={
                    "initial_valid_time": initial_snapshot.valid_time.isoformat(),
                    "last_valid_time": last_valid_time.isoformat(),
                    "source_cycle": requested_cycle.isoformat(),
                    "source_forecast_hours": list(source_forecast_hours),
                    "model_forcing_hours": list(model_forcing_hours),
                    "forcing_hours": list(requested_hours),
                    # Each report's backend receipt without what the
                    # preparation measured (A138); the preparation report
                    # keeps the whole receipt.
                    "mapping_reports": _strict_json(
                        preprocess_reports_identity(mapping_reports)),
                    "soil_texture_downscale": _strict_json(
                        root_surface.soil_texture_downscale),
                    **({"soil_temperature_repair": _strict_json(
                        soil_temperature_repair)}
                       if soil_temperature_repair is not None else {}),
                }, sealed_forcing_extension=args.sealed_prepared_cache)
            timing["write_prepared_state_and_all_lbc_cache"] = (
                time.perf_counter() - started)

    if physical_output is not None:
        physical_output_receipt = physical_output.seal()
    if root_result is None or root_met is None or initial_snapshot is None:
        raise AssertionError("benchmark preparation produced no initial state")
    if (last_valid_time - initial_snapshot.valid_time).total_seconds() \
            < args.run_seconds:
        raise ValueError("prepared forcing does not cover requested forecast")
    lbc_payload_sha256 = (lbc_digest.hexdigest() if writer is not None
                          else _lbc_payload_sha256(boundaries))
    preprocess_worker_budget_receipt = preprocess_worker_budget.receipt()
    physics_profile["hrrr_initialization"] = (
        _initial_hrrr_microphysics_receipt(
            root_result.state, exp.root.run,
            getattr(root_result, "hydrometeor_initialization", None)))

    _verify_preparation_overlay(overlay_series, physical_input=physical_input,
                                binding=water_overlay_binding)
    if args.prepare_only:
        if prepared_cache_receipt is None:
            raise RuntimeError("prepare-only completed without a cache receipt")
        record_setup_memory()
        timing["preparation_total"] = time.perf_counter() - total_started
        io_after = _proc_io()
        _atomic_json(progress_path, {
            "status": "PASS", "phase": "PREPARE_ONLY",
            "elapsed_setup_wall_seconds": timing["preparation_total"],
            "prepared_cache_content_sha256": prepared_cache_receipt[
                "content_sha256"],
        })
        report = {
            "schema": PREPARATION_REPORT_SCHEMA,
            "status": "PASS",
            "scope": (
                f"single {target.nx}x{target.ny}x{target.nz} HRRR "
                "prepared state and all LBCs"),
            "run_seconds": float(args.run_seconds),
            # Preparation produces no history files, but this exact cadence
            # is part of the cache's domain identity so a later forecast can
            # restore it without changing output_interval_s underneath the
            # prepared state.
            "history_interval_seconds": history_interval_seconds,
            "history_cadence": output_cadence_receipt,
            "source_cycle": requested_cycle.isoformat(),
            "model_start_time": model_start_time.isoformat(),
            "source_forecast_hours": list(source_forecast_hours),
            "model_forcing_hours": list(model_forcing_hours),
            "io_mode": "prepare-only",
            "device": (
                _device_name(cp)
                if preprocess_is_cuda else "CPU native preprocessing"),
            "geometry": target.to_payload(),
            "target_domain_sha256": target.identity_sha256(),
            "hrrr_source_coverage": source_window.to_dict(),
            "timing_seconds": timing,
            "setup_records": setup_records,
            "preparation": {
                "prepare_workers": int(args.prepare_workers),
                "preprocess_backend": preprocess_receipt,
                "preprocess_worker_budget": (
                    preprocess_worker_budget_receipt),
                "full_domain_initialized_hours": ([] if restore_cached else [0]),
                "boundary_slab_initialized_hours": (
                    [] if restore_cached else list(requested_hours[1:])),
                "source_full_domain_initialized_hours": (
                    [] if restore_cached else [source_forecast_hours[0]]),
                "source_boundary_slab_initialized_hours": (
                    [] if restore_cached else list(source_forecast_hours[1:])),
                "lbc_payload_sha256": lbc_payload_sha256,
            },
            "prepared_cache": prepared_cache_receipt,
            **({"physical_output_store": physical_output_receipt}
               if physical_output_receipt is not None else {}),
            "physics": physics_profile,
            "memory": {
                "gpu_peak_used_bytes_observed": setup_gpu_peak_used,
                "cupy_pool_peak_total_bytes_observed": setup_pool_peak_total,
                "cpu_peak_rss_bytes": _peak_rss_bytes(),
                "worker_peak_rss_bytes_by_pid": {
                    str(pid): value for pid, value in sorted(
                        setup_worker_peak_rss_by_pid.items())},
                "cpu_peak_rss_sum_upper_bound_bytes": (
                    _peak_rss_bytes()
                    + sum(value for pid, value in
                          setup_worker_peak_rss_by_pid.items()
                          if pid != os.getpid())),
            },
            "input": {
                "bridge": str(args.bridge.resolve()),
                "aerosol_missing": aerosol_fill_receipt(_read_gate(args.bridge)),
                "bridge_manifest_sha256": args.manifest_sha256,
                "source_manifest_sha256": args.source_manifest_sha256,
                # Present only when a runtime surface field started from its
                # fallback because the cycle publishes no record for it; the
                # rows are the fetch receipt's own (id, reason, cycle).
                **({"runtime_surface_fallback": runtime_surface_fallbacks}
                   if runtime_surface_fallbacks else {}),
                "source_cycle": requested_cycle.isoformat(),
                "model_start_time": model_start_time.isoformat(),
                "source_forecast_hours": list(source_forecast_hours),
                "model_forcing_hours": list(model_forcing_hours),
                "forcing_hours": list(requested_hours),
                "native_static_cache": str(args.static_cache.resolve()),
                "native_static_cache_sha256": static_load["cache_sha256"],
                "namelist_input_sha256": namelist_sha256,
                "native_physics_profile": physics_profile,
            },
            "mapping_reports": mapping_reports,
            "source_identity": source_identity,
            "pipeline": pipeline_report,
            "source_hash_preflight": source_hash_receipt,
            "process_io_delta": {
                key: int(io_after.get(key, 0) - io_before.get(key, 0))
                for key in sorted(set(io_before) | set(io_after))},
        }
        if writer is not None:
            try:
                report["portable_bundle"] = _publish_chained_proof(
                    writer, args, chain=chain, report=report,
                    configured_run=exp.root.run, posted_proof=posted_proof)
            except BaseException as error:
                writer.fail(error)
                raise
        return report

    # Everything below is forecast-only.  Keep these imports after the
    # prepare-only return so an installed CPU fallback has no CuPy dependency.
    import woof.core.dycore as dycore_module
    from woof.core.clock import build_schedule, resolve_clock
    from woof.core.dycore import stability_gate_failed, stability_report
    from woof.core.health import StateHealthValidator
    from woof.core.model import (
        DomainNode, ExperimentState, ModelRuntimeStatus, execute_experiment)
    from woof.state_digest import canonical_state_digest
    from woof.runtime import declared_constant_glw
    from woof.ingest.hrrr_physics import initialize_hrrr_physics

    if pipeline_producer is not None:
        pipeline_report = pipeline_producer.finish()
        seal_process, seal_receipt, seal_started = _start_seal(
            args, pipeline_report, source_hash_receipt, source_window)

    started = time.perf_counter()
    from woof.static.external_source import static_source_for
    from woof.runtime_surface_fetch import require_runtime_surface_fields
    from woof.source_adapters import get_source_adapter
    runtime_surface_fallbacks = require_runtime_surface_fields(
        root_met, get_source_adapter("hrrr"),
        source_manifest=_fetch_receipt_path(args),
        refuse_dropped=static_source_for(static_highres) is not None)
    driver = initialize_hrrr_physics(
        root_result, dc.run, root_met, static, attrs, grid,
        initial_snapshot.valid_time,
        constant_glw_wm2=declared_constant_glw(exp),
        surface=root_surface, soil_mesh=soil_mesh, p_top=exp.vertical.p_top,
        column_chunk=exp.column_chunk, trace_gas_overrides=trace_gas_overrides)
    timing["initialize_physics"] = time.perf_counter() - started

    clock = resolve_clock(exp, lbc_interval_s=3600.0)
    schedule = build_schedule(exp, clock)
    node = DomainNode(dc, grid, root_result.state, clock.clocks()[1],
                      None, [], None)
    model = ExperimentState(
        node, MappingProxyType({1: node}), schedule, None,
        f"native-hrrr-{target.name}")
    model._runtime_status = ModelRuntimeStatus()
    model._resumed = False
    model._resume_committed_history_grid_ids = frozenset()
    model._scratch_arena = None
    model._dycore_state_workspace = None
    model._io_manager = None
    model._last_checkpoint = None
    model._prepared_by_grid_id = MappingProxyType({
        1: SimpleNamespace(
            static_fields=static, geog_selection=None,
            initial_result=root_result)})

    initial_health = _strict_json(vars(
        StateHealthValidator(node.state).validate(phase="initialized.d01")))
    if not initial_health["ok"]:
        raise FloatingPointError(f"initial health failed: {initial_health}")

    original_step = dycore_module.step
    step_samples = []
    first_gpu_step_seconds = None

    def timed_step(state, cfg, *positional, **keywords):
        nonlocal first_gpu_step_seconds
        if len(step_samples) >= 3:
            return original_step(state, cfg, *positional, **keywords)
        cp.cuda.Stream.null.synchronize()
        before = time.perf_counter()
        value = original_step(state, cfg, *positional, **keywords)
        cp.cuda.Stream.null.synchronize()
        step_samples.append(time.perf_counter() - before)
        if first_gpu_step_seconds is None:
            first_gpu_step_seconds = time.perf_counter() - total_started
        return value

    dycore_module.step = timed_step
    history = []
    gpu_peak_used = setup_gpu_peak_used
    pool_peak_total = setup_pool_peak_total
    writers = None
    wrfout_paths = ()

    def record_memory():
        nonlocal gpu_peak_used, pool_peak_total
        free, total = cp.cuda.runtime.memGetInfo()
        gpu_peak_used = max(gpu_peak_used, int(total - free))
        pool_peak_total = max(
            pool_peak_total, int(cp.get_default_memory_pool().total_bytes()))

    def history_handler(_model, current, ticks):
        from woof.core.refl import consume_refl_10cm
        report = stability_report(
            current.state, current.cfg.run,
            boundary_width=current.cfg.run.spec_bdy_width)
        sample = {
            "ticks": int(ticks),
            "elapsed_seconds": float(current.clock.elapsed_seconds),
            **report,
        }
        history.append(sample)
        if stability_gate_failed(
                report, max_cfl=MAX_HISTORY_CFL,
                max_w_ms=MAX_HISTORY_W_MS):
            raise FloatingPointError(
                f"benchmark stability threshold failed: {sample}")
        refl = None
        if _history_uses_native_reflectivity(
                ticks=ticks,
                has_moisture=current.state.qv is not None,
                mp_physics=current.state.physics.mp_physics):
            refl = consume_refl_10cm(current.state)
        if writers is not None:
            writers.submit(current, ticks, refl_field=refl)
        record_memory()

    forecast_started = None

    def progress_callback(**event):
        if int(event["outer_step"]) == 1 or int(event["outer_step"]) % 60 == 0:
            record_memory()
            _atomic_json(progress_path, {
                "status": "RUNNING",
                "model_elapsed_seconds": event["model_elapsed_seconds"],
                "outer_step": event["outer_step"],
                "requested_run_seconds": args.run_seconds,
                "forecast_wall_seconds": time.perf_counter() - forecast_started,
                "gpu_peak_used_bytes_observed": gpu_peak_used,
            })

    try:
        if args.io_mode == "history":
            from woof.io.wrfout import PerDomainWrfoutWriters
            writers = PerDomainWrfoutWriters(
                model, args.outdir / "wrfout",
                start_time=exp.start_time,
                title=(
                    f"woof native HRRR {target.nx}x{target.ny} "
                    "easy-physics benchmark"))
            model._io_manager = writers
        forecast_started = time.perf_counter()
        if writers is None:
            execution = execute_experiment(
                model, history_handler=history_handler,
                progress_callback=progress_callback, validate_state=True,
                skip_feedback_path=True)
            cp.cuda.Stream.null.synchronize()
            timing["forecast_execution"] = time.perf_counter() - forecast_started
        else:
            with writers:
                execution = execute_experiment(
                    model, history_handler=history_handler,
                    progress_callback=progress_callback, validate_state=True,
                    skip_feedback_path=True)
                cp.cuda.Stream.null.synchronize()
                timing["forecast_execution_with_async_io"] = (
                    time.perf_counter() - forecast_started)
                drain_started = time.perf_counter()
                writers.drain()
                timing["final_writer_drain"] = time.perf_counter() - drain_started
                wrfout_paths = writers.paths
            timing["forecast_and_io_inclusive"] = (
                time.perf_counter() - forecast_started)
    finally:
        dycore_module.step = original_step
    record_memory()

    output_schedule = ()
    if writers is not None:
        output_schedule = _history_output_schedule(
            start_time=exp.start_time, run_seconds=exp.run_seconds,
            cadence_seconds=history_interval_seconds)
        if len(wrfout_paths) != len(output_schedule):
            raise RuntimeError(
                f"history writer published {len(wrfout_paths)} frames, "
                f"expected {len(output_schedule)}")
        expected_names = tuple(record[2] for record in output_schedule)
        if tuple(path.name for path in wrfout_paths) != expected_names:
            raise RuntimeError(
                "WRF history output filenames/cadence differ from the "
                "explicit request")

    final_health = _strict_json(vars(
        StateHealthValidator(node.state).validate(phase="final.d01")))
    final_stability = stability_report(
        node.state, node.cfg.run, boundary_width=node.cfg.run.spec_bdy_width)
    if not final_health["ok"]:
        raise FloatingPointError(f"final health failed: {final_health}")

    started = time.perf_counter()
    final_digest = canonical_state_digest(
        node.state, node.clock, scope="trajectory")
    timing["canonical_final_state_digest"] = time.perf_counter() - started

    if seal_process is not None:
        stdout, stderr = seal_process.communicate()
        timing["pipeline_bridge_seal_overlapped_wall"] = (
            time.perf_counter() - seal_started)
        if seal_process.returncode != 0:
            raise RuntimeError("pipeline bridge seal failed: " + stderr[-4000:])
        if seal_receipt is None or not seal_receipt.is_file():
            raise RuntimeError("pipeline bridge seal omitted receipt")
        seal = json.loads(seal_receipt.read_text())
        args.manifest_sha256 = seal["manifest_sha256"]
        pipeline_report["seal"] = seal
        pipeline_report["seal_stdout"] = stdout.strip()

    output_inventory = []
    started = time.perf_counter()
    for index, path in enumerate(wrfout_paths):
        row = {
            "path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "atomic_writer_readback_verified": True,
        }
        if output_schedule:
            offset_seconds, valid_time, _name = output_schedule[index]
            row.update({
                "model_elapsed_seconds": offset_seconds,
                "valid_time": valid_time.isoformat(),
            })
        output_inventory.append(row)
    timing["hash_gridded_output"] = time.perf_counter() - started

    io_after = _proc_io()
    integration_key = (
        "forecast_execution" if args.io_mode == "none"
        else "forecast_execution_with_async_io")
    warmed = _warmed_integration_rates(
        integration_seconds=timing[integration_key],
        run_seconds=float(args.run_seconds),
        steps=int(execution.steps),
        step_samples=step_samples)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "PASS",
        "scope": (
            f"single {target.nx}x{target.ny}x{target.nz} HRRR "
            f"{args.physics_profile}"),
        "run_seconds": float(args.run_seconds),
        "source_cycle": requested_cycle.isoformat(),
        "model_start_time": model_start_time.isoformat(),
        "source_forecast_hours": list(source_forecast_hours),
        "model_forcing_hours": list(model_forcing_hours),
        "io_mode": args.io_mode,
        "history_interval_seconds": (
            None if args.io_mode == "none" else history_interval_seconds),
        "device": _device_name(cp),
        "geometry": target.to_payload(),
        "target_domain_sha256": target.identity_sha256(),
        "hrrr_source_coverage": source_window.to_dict(),
        "timing_seconds": timing,
        "setup_records": setup_records,
        "preparation": {
            "prepare_workers": int(args.prepare_workers),
            "preprocess_backend": preprocess_receipt,
            "preprocess_worker_budget": preprocess_worker_budget_receipt,
            "full_domain_initialized_hours": ([] if restore_cached else [0]),
            "boundary_slab_initialized_hours": (
                [] if restore_cached else list(requested_hours[1:])),
            "source_full_domain_initialized_hours": (
                [] if restore_cached else [source_forecast_hours[0]]),
            "source_boundary_slab_initialized_hours": (
                [] if restore_cached else list(source_forecast_hours[1:])),
            "lbc_payload_sha256": lbc_payload_sha256,
        },
        "downloaded_hrrr_to_first_gpu_step_seconds": first_gpu_step_seconds,
        "integration_simulated_seconds_per_wall_second": (
            float(args.run_seconds) / timing[integration_key]),
        "integration_wall_seconds_per_simulated_hour": (
            timing[integration_key] / (float(args.run_seconds) / 3600.0)),
        "measured_initial_step_seconds": step_samples,
        # ADDED beside the two integration rates above, which keep their
        # exact meaning: whole-run averages including compilation.  These
        # separate the two populations a short run mixes together.
        **warmed,
        "executor": {
            "pool_trim": getattr(model, "_pool_trim_policy", None),
            "steps": int(execution.steps), "forces": int(execution.forces),
            "feedback_calls": int(execution.feedback_calls),
        },
        "health": {
            "initial": initial_health, "final": final_health,
            "final_stability": final_stability, "history": history,
            "sampling_interval_seconds": history_interval_seconds,
            "limits": {"max_cfl": MAX_HISTORY_CFL,
                       "max_w_ms": MAX_HISTORY_W_MS},
        },
        "final_state_digest": final_digest,
        "prepared_cache": prepared_cache_receipt,
        "physics": {
            **physics_profile,
            **_physics_receipt(driver, cp),
        },
        "memory": {
            "gpu_peak_used_bytes_observed": gpu_peak_used,
            "cupy_pool_peak_total_bytes_observed": pool_peak_total,
            "cpu_peak_rss_bytes": _peak_rss_bytes(),
            "worker_peak_rss_bytes_by_pid": {
                str(pid): value for pid, value in sorted(
                    setup_worker_peak_rss_by_pid.items())},
            "cpu_peak_rss_sum_upper_bound_bytes": (
                _peak_rss_bytes()
                + sum(value for pid, value in
                      setup_worker_peak_rss_by_pid.items()
                      if pid != os.getpid())),
            "decoder_max_rss_bytes_observed": (
                None if pipeline_report is None else pipeline_report[
                    "decoder_max_rss_bytes_observed"]),
        },
        "gridded_output": {
            "cadence_seconds": (
                None if args.io_mode == "none" else history_interval_seconds),
            "cadence_receipt": output_cadence_receipt,
            "expected_frame_count": (
                0 if args.io_mode == "none" else len(output_schedule)),
            "exact_frame_count_verified": True,
            "initial_frame_verified": args.io_mode == "history",
            "last_scheduled_frame_verified": args.io_mode == "history",
            "last_scheduled_offset_seconds": (
                None if output_cadence_receipt is None else
                output_cadence_receipt["last_scheduled_offset_seconds"]),
            "last_scheduled_valid_time": (
                None if output_cadence_receipt is None else
                output_cadence_receipt["last_scheduled_valid_time"]),
            "last_scheduled_equals_run_end": (
                None if output_cadence_receipt is None else
                output_cadence_receipt["last_scheduled_equals_run_end"]),
            "initial_and_final_frames_verified": (
                False if output_cadence_receipt is None else
                output_cadence_receipt["last_scheduled_equals_run_end"]),
            "all_frames_readback_verified": (
                None if args.io_mode == "none" else all(
                    item["atomic_writer_readback_verified"]
                    for item in output_inventory)),
            "completion_attribute": {
                "name": "GPUWM_WRITE_COMPLETE", "value": 1,
            },
            "frame_count": len(output_inventory),
            "total_bytes": sum(item["bytes"] for item in output_inventory),
            "files": output_inventory,
        },
        "input": {
            "bridge": str(args.bridge.resolve()),
                "aerosol_missing": aerosol_fill_receipt(_read_gate(args.bridge)),
            "bridge_manifest_sha256": args.manifest_sha256,
            "source_manifest_sha256": args.source_manifest_sha256,
            # Present only when a runtime surface field started from its
            # fallback because the cycle publishes no record for it; the
            # rows are the fetch receipt's own (id, reason, cycle).
            **({"runtime_surface_fallback": runtime_surface_fallbacks}
               if runtime_surface_fallbacks else {}),
            "source_cycle": requested_cycle.isoformat(),
            "model_start_time": model_start_time.isoformat(),
            "source_forecast_hours": list(source_forecast_hours),
            "model_forcing_hours": list(model_forcing_hours),
            "forcing_hours": list(requested_hours),
            "native_static_cache": str(args.static_cache.resolve()),
            "native_static_cache_sha256": static_load["cache_sha256"],
            "native_static_receipt": str(args.static_receipt.resolve()),
            "namelist_input_sha256": namelist_sha256,
            "native_physics_profile": physics_profile,
        },
        "pipeline": pipeline_report,
        "source_hash_preflight": source_hash_receipt,
        "mapping_reports": mapping_reports,
        "source_identity": source_identity,
        "process_io_delta": {
            key: int(io_after.get(key, 0) - io_before.get(key, 0))
            for key in sorted(set(io_before) | set(io_after))},
    }
    # WHICH AEROSOL INITIAL CONDITION THIS RUN STARTED FROM.  This report
    # is also the PREPARATION report the HRRR chain reads back
    # (tools/prepare_hrrr_wrf.py --prepare-only writes it to
    # native/preparation-report/report.json), so it is the record for the
    # whole downstream tree, not only for this process.  Merged rather
    # than assigned: nothing is added for a scheme with no aerosol number
    # fields, and every profile in this runner's registry except an
    # aerosol-aware one writes the report it wrote before.
    report.update(aerosol_source_report_entry(
        root_aerosol_initialization,
        mp_physics=dc.run.mp_physics,
        when_unrecorded=(
            "this run restored a prepared cache that carries no "
            "aerosol-initialization receipt, so the cache was written by a "
            "preparation predating the receipt being stored; re-prepare to "
            "record which source filled nwfa/nifa")))
    _atomic_json(progress_path, {
        "status": "PASS", "model_elapsed_seconds": float(args.run_seconds),
        "report": str((args.outdir / "report.json").resolve()),
    })
    return report


def _warmed_integration_rates(
        *, integration_seconds: float, run_seconds: float, steps: int,
        step_samples: list[float]) -> dict[str, object]:
    """The cold first step, and the rate the remaining steps ran at.

    A short run reports one number for two populations.  On the field
    report this runner produced -- 900 simulated seconds in 60 steps --
    the first step took 22.556 s and the next two took 0.283 s each,
    because step one pays for every NVRTC compilation the whole forecast
    needs.  The whole-run average was 2.7498 wall s per simulated
    minute; the same run excluding step one was 1.2672, and it is the
    second number that predicts what an hour of forecast costs.
    Reporting only the first invites a reader to extrapolate a
    fifteen-minute measurement into a figure that is more than twice the
    truth.

    Both existing rate fields keep their names and their meaning: they
    are the accurate cost of THIS run, compilation included, which is what
    a fifteen-minute run actually took.  These are additional.

    ``None`` rather than a fabricated number whenever the arithmetic has
    no meaning -- a single-step run has no warmed population, and a
    cold step that somehow exceeded the whole integration would produce
    a negative denominator.
    """

    simulated_per_step = (run_seconds / steps) if steps > 0 else 0.0
    cold = float(step_samples[0]) if step_samples else None
    result: dict[str, object] = {
        "cold_first_step_seconds": cold,
        "warmed_integration_excluded_steps": 1 if cold is not None else 0,
    }
    warmed_wall = (
        None if cold is None else integration_seconds - cold)
    warmed_simulated = run_seconds - simulated_per_step
    if (warmed_wall is None or warmed_wall <= 0.0
            or warmed_simulated <= 0.0 or steps <= 1):
        result["warmed_integration_seconds"] = None
        result["warmed_integration_simulated_seconds_per_wall_second"] = None
        result["warmed_integration_wall_seconds_per_simulated_hour"] = None
        return result
    result["warmed_integration_seconds"] = warmed_wall
    result["warmed_integration_simulated_seconds_per_wall_second"] = (
        warmed_simulated / warmed_wall)
    result["warmed_integration_wall_seconds_per_simulated_hour"] = (
        warmed_wall / (warmed_simulated / 3600.0))
    return result


#: Fixed-name documents this runner publishes into ``--outdir``.
_PUBLISHED_OUTPUT_NAMES = ("report.json", "progress.json")

#: WRF history frames, whose exact names come from the cadence and the
#: model start; matched by pattern here because ``--outdir`` is checked
#: before the namelist and the vertical grid have been read.
_PUBLISHED_OUTPUT_GLOB = "wrfout_d*"


def _existing_published_output(outdir: Path) -> list[str]:
    """Names in ``outdir`` this run would publish over."""

    names = {name for name in _PUBLISHED_OUTPUT_NAMES
             if (outdir / name).exists()}
    names.update(path.name for path in outdir.glob(_PUBLISHED_OUTPUT_GLOB))
    return sorted(names)


def _check_outdir(parser, args) -> None:
    """Refuse an occupied ``--outdir`` in a sentence, before anything runs.

    ``mkdir(exist_ok=False)`` used to fire in the middle of setup, as a
    bare ``FileExistsError`` traceback, after the target domain and the
    source window had already been resolved -- a field run of the shipped
    1.5.0 wheel met exactly that.  The same file also created its failure
    report with ``exist_ok=True``, so the two halves disagreed about
    whether an existing directory was allowed.

    Checked here, at parse time, where every other argument contract in
    this runner is checked.  ``--allow-existing`` is the explicit way to
    reuse a directory you made yourself or that holds unrelated files;
    it is NOT permission to clobber, because silently replacing a
    benchmark result is worse than either refusing or crashing.
    """

    outdir = args.outdir
    if not outdir.exists():
        return
    if not outdir.is_dir():
        parser.error(
            f"--outdir {outdir} exists and is not a directory; name a "
            "directory, or a path that does not exist yet")
    if not args.allow_existing:
        parser.error(
            f"--outdir {outdir} already exists; name a directory that does "
            "not exist yet, or pass --allow-existing to reuse this one "
            "(output files already in it are still never overwritten)")
    occupied = _existing_published_output(outdir)
    if occupied:
        parser.error(
            f"--outdir {outdir} already holds output this run publishes "
            f"({', '.join(occupied)}); benchmark results are not "
            "overwritten -- move them aside, or name a fresh --outdir")


def _positive_finite_seconds(value: str) -> float:
    try:
        seconds = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "must be a number of seconds") from error
    if not math.isfinite(seconds) or seconds <= 0.0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return seconds


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument(
        "--physics-profile", type=canonical_template_id,
        default=None,
        help="optional equality assertion against a named physics template; "
             "this route offers "
             + ", ".join(NATIVE_BENCHMARK_PHYSICS_PROFILES)
             + ".  Any other registered suite runs on "
             "tools.prepared_single_domain_forecast or per domain on "
             "tools.prepared_domain_tree_forecast",
    )
    parser.add_argument(
        "--ack", action="append", default=[],
        help="registry-owned expert physics acknowledgement id; repeatable")
    parser.add_argument(
        "--cycle",
        help="requested exact hourly HRRR cycle (YYYY-MM-DD_HH:00:00).  "
             "Model time zero is cycle + --forecast-start-hour")
    parser.add_argument(
        "--valid-time",
        help="deprecated spelling of --cycle (it always meant the cycle "
             "on this command); accepted unchanged for v1.4.0 scripts")
    parser.add_argument(
        "--forecast-start-hour", type=int, default=0,
        help="absolute cycle-relative HRRR lead used for model time zero")
    parser.add_argument(
        "--forecast-end-hour", type=int,
        help="inclusive absolute source lead; must cover --run-seconds exactly")
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--pipeline-series", type=Path)
    parser.add_argument("--pipeline-decoder", type=Path)
    parser.add_argument("--pipeline-signals", type=Path)
    parser.add_argument("--pipeline-workers", default="8")
    parser.add_argument(
        "--preprocess-backend", choices=("cuda", "cpu", "auto"),
        default="auto")
    parser.add_argument("--preprocess-workers", type=int)
    parser.add_argument("--cpu-preprocess-bridge", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--source-manifest-sha256")
    parser.add_argument("--static-cache", type=Path, required=True)
    parser.add_argument("--static-receipt", type=Path, required=True)
    parser.add_argument("--physical-input-provider", type=Path,
                        help="posted native physical provider with a frozen member plan")
    parser.add_argument("--physical-member-index", type=int,
                        help="original recipe member index in the posted provider")
    parser.add_argument("--physical-input-store", type=Path,
                        help="sealed native physical snapshots for this complete forcing window")
    parser.add_argument("--physical-output-store", type=Path,
                        help="capture mapped native physical snapshots before real initialization")
    parser.add_argument(
        "--domain-spec", type=Path,
        help=("strict gpuwm-hrrr-target-domain-v1 JSON; omission retains "
              "the sealed 500x500 benchmark target"),
    )
    parser.add_argument("--experiment-config", type=Path)
    parser.add_argument("--wps-namelist", type=Path)
    parser.add_argument("--namelist-input", type=Path, required=True)
    parser.add_argument(
        "--publish-experiment-config", type=Path,
        help=("render the experiment tables this route built into a TOML "
              "authority at PATH, verified to reload to the same prepared "
              "domain identity and vertical grid.  This is the only "
              "process that holds those tables, and a config-driven "
              "downstream stage (the cycling DA driver) cannot bind the "
              "prepared cache without one.  Create-only"))
    parser.add_argument(
        "--prepared-cache", type=Path,
        help="build once when absent, otherwise hash-validate and restore")
    parser.add_argument(
        "--prepare-only", action="store_true",
        help="prepare/restore --prepared-cache without integrating")
    parser.add_argument(
        "--chained-bundle", type=Path,
        help=("a gpuwm-hrrr-chained-bundle-v1 document (written by "
              "tools/prepare_hrrr_wrf.py): publish the portable bundle's "
              "head once the start state exists, one boundary interval per "
              "hour after it, and proof.json at the seal, so a forecast can "
              "start on the head"))
    parser.add_argument(
        "--as-posted", type=Path, dest="as_posted",
        help=("an as-posted fetch's posting/ folder (A136): each lead is "
              "decoded once its posted marker is there and its files are the "
              "marker's, the head is published on the first two leads, and "
              "the seal writes the source manifest at --source-manifest from "
              "the leads' markers (no --source-manifest-sha256: that "
              "manifest does not exist before the last lead)"))
    parser.add_argument(
        "--sealed-prepared-cache", action="store_true",
        help=("opt in to a prefix-sealed prepared cache that an operational "
              "controller may extend by one cryptographically joined hour"))
    parser.add_argument(
        "--namelist-extension-suffix", action="store_true",
        help=("recompute the immutable namelist identity for a nonzero-hour "
              "prepared-cache suffix"))
    parser.add_argument(
        "--prepare-workers", type=int, default=2,
        help=("spawn this many independent f01+ boundary-slab initializers; "
              "f00 uses the same count for exact CPU column parallelism "
              "(default: 2; use 1 as the conservative fallback)"))
    parser.add_argument("--run-seconds", type=float, required=True)
    parser.add_argument("--io-mode", choices=("none", "history"), default="none")
    parser.add_argument(
        "--history-interval-seconds", type=_positive_finite_seconds,
        help=("explicit WRF history cadence required with --io-mode history; "
              "with --prepare-only it binds the future forecast cadence into "
              "the prepared-cache identity without writing history output"))
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument(
        "--allow-existing", action="store_true",
        help=("reuse an existing --outdir instead of refusing it; output "
              "files already in it are still never overwritten"))
    args = parser.parse_args(argv)
    if args.as_posted is not None:
        # Each names the breakage it prevents.
        if args.pipeline_series is None or not args.prepare_only:
            parser.error(
                "--as-posted decodes each lead as it posts, which only a "
                "prepare-only pipeline preparation does (--pipeline-series, "
                "--prepare-only)")
        if args.sealed_prepared_cache:
            parser.error(
                "--as-posted and --sealed-prepared-cache are mutually "
                "exclusive: a prefix-sealed cache binds a source manifest "
                "before it decodes, which an as-posted window does not have")
        if args.source_manifest_sha256 is not None:
            parser.error(
                "--source-manifest-sha256 names a source manifest, and an "
                "as-posted preparation writes it at its seal from the leads' "
                "markers (at --source-manifest); omit the flag")
        if args.source_manifest is None or args.source_manifest.exists():
            parser.error(
                "--as-posted needs --source-manifest naming where its seal "
                "writes the source manifest, a path that does not exist yet")
    if args.pipeline_series is not None:
        required = {
            "pipeline_decoder": args.pipeline_decoder,
            "pipeline_signals": args.pipeline_signals,
            "source_root": args.source_root,
            "source_manifest": args.source_manifest,
            "source_manifest_sha256": args.source_manifest_sha256,
        }
        if args.as_posted is not None:
            required.pop("source_manifest_sha256")
        if args.physical_input_provider is not None:
            required.pop("pipeline_decoder")
            required.pop("pipeline_signals")
        missing = [key for key, value in required.items() if value is None]
        if missing:
            parser.error(f"pipeline mode is missing: {missing}")
        if args.manifest_sha256 is not None:
            parser.error("pipeline mode produces --manifest-sha256")
    elif args.manifest_sha256 is None:
        parser.error("--manifest-sha256 is required without pipeline mode")
    if (args.prepared_cache is not None and args.source_manifest_sha256 is None
            and args.as_posted is None):
        parser.error(
            "--source-manifest-sha256 is required with --prepared-cache")
    if args.prepare_only and args.prepared_cache is None:
        parser.error("--prepare-only requires --prepared-cache")
    if args.sealed_prepared_cache and not args.prepare_only:
        parser.error("--sealed-prepared-cache requires --prepare-only")
    if args.namelist_extension_suffix and not args.prepare_only:
        parser.error("--namelist-extension-suffix requires --prepare-only")
    if args.namelist_extension_suffix and args.sealed_prepared_cache:
        parser.error(
            "--namelist-extension-suffix and --sealed-prepared-cache are "
            "mutually exclusive")
    if args.io_mode == "history" and args.history_interval_seconds is None:
        parser.error(
            "--io-mode history requires --history-interval-seconds")
    if (args.io_mode == "none" and args.history_interval_seconds is not None
            and not args.prepare_only):
        parser.error(
            "--history-interval-seconds requires --io-mode history or "
            "--prepare-only")
    if args.prepare_only and args.io_mode != "none":
        parser.error("--prepare-only requires --io-mode none")
    if not 1 <= args.prepare_workers <= 32:
        parser.error("--prepare-workers must be between 1 and 32")
    if args.preprocess_workers is not None and args.preprocess_workers < 1:
        parser.error("--preprocess-workers must be positive")
    if (args.preprocess_backend != "cpu"
            and args.cpu_preprocess_bridge is not None):
        parser.error(
            "--cpu-preprocess-bridge requires --preprocess-backend cpu")
    _check_outdir(parser, args)
    return args


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv == ["--show-capabilities"]:
        print(json.dumps(runner_capabilities(), sort_keys=True))
        return 0
    args = _parse_args(argv)
    try:
        report = run(args)
    except BaseException as error:
        args.outdir.mkdir(parents=True, exist_ok=True)
        _atomic_json(args.outdir / "report.json", {
            "schema": REPORT_SCHEMA,
            "status": "FAIL", "error_type": type(error).__name__,
            "error": str(error), "traceback": traceback.format_exc(),
        })
        raise
    _atomic_json(args.outdir / "report.json", report)
    summary = {
        "status": report["status"], "run_seconds": report["run_seconds"],
        "io_mode": report["io_mode"],
        "history_interval_seconds": report.get("history_interval_seconds"),
        "gpu_peak_used_bytes": report["memory"]["gpu_peak_used_bytes_observed"],
    }
    if "integration_simulated_seconds_per_wall_second" in report:
        summary["simulated_seconds_per_wall_second"] = report[
            "integration_simulated_seconds_per_wall_second"]
        # Printed BESIDE the whole-run rate, never instead of it.  The
        # stdout summary is what a short run gets extrapolated from, and
        # on a fifteen-minute run the cold first step is most of it.
        summary["cold_first_step_seconds"] = report[
            "cold_first_step_seconds"]
        summary["warmed_simulated_seconds_per_wall_second"] = report[
            "warmed_integration_simulated_seconds_per_wall_second"]
    if report.get("prepared_cache") is not None:
        summary["prepared_cache_content_sha256"] = report[
            "prepared_cache"]["content_sha256"]
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
