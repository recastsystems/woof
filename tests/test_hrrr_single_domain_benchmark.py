"""Native HRRR single-domain benchmark controller regressions."""

import copy
import dataclasses
from datetime import datetime
import json
from types import SimpleNamespace

import numpy as np
import pytest

import tools.hrrr_single_domain_benchmark as hrrr_runner
from woof.static.lambert import LambertGrid
from woof.experiment import VerticalConfig
from woof.hrrr_route_inputs import ROUTE_DEFAULT_PHYSICS_PROFILE
from woof.ingest.hrrr_target import HrrrTargetDomain
from woof.ingest.prepared_cache import prepared_cache_identity
from woof.physics_compat import (
    KESSLER_PROFILE_ID,
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
    route_physics_profiles,
    THOMPSON_LEGACY_RRTMG_PROFILE_ID,
    THOMPSON_MYNN_RUC_DUDHIA_PROFILE_ID,
    THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID,
    THOMPSON_PROFILE_ID,
    THOMPSON_RTE_RRTMGP_PROFILE_ID,
    THOMPSON_SHINHONG_LEGACY_RRTMG_PROFILE_ID,
    WRF_RRTMG_LEGACY,
    WRF_RRTMG_TO_RTE_RRTMGP,
    WSM6_PROFILE_ID,
    downward_longwave_disposition,
    single_domain_runtime_switches,
)
from tools.hrrr_single_domain_benchmark import (
    _PreprocessWorkerBudget, _boundary_mapping_targets, _experiment,
    _initial_hrrr_microphysics_receipt,
    main as hrrr_runner_main,
    _map_boundary_snapshot, _map_snapshot, _parse_args,
    _partition_preprocess_worker_budget,
    _validated_namelist_extension_identity,
    _validate_native_hrrr_physics_profile,
    runner_capabilities,
)
from tools.prepare_hrrr_wrf import _validated_worker_receipts
from woof.stream import _materialize_input_namelist


def _required_args():
    return [
        "--bridge", "bridge",
        "--valid-time", "2026-07-20_06:00:00",
        "--manifest-sha256", "manifest-sha256",
        "--static-cache", "static.npz",
        "--static-receipt", "static.json",
        "--namelist-input", "namelist.input",
        "--run-seconds", "3600",
        "--outdir", "output",
    ]


def test_hrrr_runner_capability_query_is_side_effect_free_without_run_args(
        tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert hrrr_runner_main(["--show-capabilities"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == runner_capabilities()
    assert payload["schema"] == "gpuwm-runner-capabilities-v1"
    assert payload["supported_sources"] == ["hrrr"]
    assert payload["physics_profile_ids"] == [
        WSM6_PROFILE_ID, KESSLER_PROFILE_ID,
        THOMPSON_PROFILE_ID, THOMPSON_LEGACY_RRTMG_PROFILE_ID,
        THOMPSON_RTE_RRTMGP_PROFILE_ID,
        THOMPSON_SHINHONG_LEGACY_RRTMG_PROFILE_ID,
        MORRISON_PROFILE_ID,
        NSSL2_PROFILE_ID, NSSL2_LEGACY_RRTMG_PROFILE_ID,
        P3_LEGACY_RRTMG_PROFILE_ID,
        MYNN_PROFILE_ID, MYNN_RTE_RRTMGP_PROFILE_ID,
        RUC_PROFILE_ID,
        MYNN_RUC_PROFILE_ID, THOMPSON_MYNN_RUC_DUDHIA_PROFILE_ID,
        MYNN_RUC_RTE_RRTMGP_PROFILE_ID,
        THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID,
        hrrr_runner.THOMPSON_MYNN_RUC_MONTHLY_LEGACY_RRTMG_PROFILE_ID,
        hrrr_runner.THOMPSON_MYNN_RUC_MONTHLY_SOLAR_LEGACY_RRTMG_PROFILE_ID,
        hrrr_runner.THOMPSON_MYNN_GSD41_PROFILE_ID,
        NOAHMP_PROFILE_ID, MYNN_NOAHMP_PROFILE_ID,
        MYNN_NOAHMP_RTE_RRTMGP_PROFILE_ID]
    assert payload["report_schema"] == "gpuwm-native-hrrr-benchmark-v2"
    assert payload["window"]["maximum_source_forecast_hour"] == 48
    assert payload["window"]["maximum_run_seconds"] == 172_800
    assert payload["window"]["run_seconds"]["whole_hour_required"] is False
    assert payload["output"]["io_modes"] == ["none", "history"]
    assert payload["output"]["configurable_cadence"] is True
    cadence = payload["output"]["history_interval_seconds"]
    assert cadence["required_for_mode"] == "history"
    assert cadence["accepted_for_prepare_only_cache_identity"] is True
    assert cadence["must_be_whole_model_steps"] is True
    assert cadence["must_evenly_divide_run"] is False
    assert cadence["last_scheduled_frame_may_precede_run_end"] is True
    assert payload["readiness"] \
        == "FORECAST_IMPLEMENTATION_PRESENT_RUNTIME_PREFLIGHT_REQUIRED"
    assert payload["modes"]["prepare-only"]["available"] is True
    assert payload["modes"]["forecast"]["available"] is True
    assert payload["standalone_rw_wps_wheel"] == {
        "runner_included": True,
        "prepare_only_available": True,
        "forecast_executor_included": False,
    }
    thompson = payload["physics_profiles"][THOMPSON_PROFILE_ID]
    assert thompson["readiness"] == "WRF_MATCHED_RUN_EXPERIMENTAL_RUNTIME"
    assert thompson["runtime_guard"]["environment"] \
        == "WOOF_EXPERIMENTAL_THOMPSON_MP8"
    assert thompson["explicit_expert_consent_required"] is False
    assert thompson["table_root"]["environment"] \
        == "WOOF_THOMPSON_TABLE_ROOT"
    assert len(thompson["table_authority"]["assets"]) == 4
    assert payload["physics_profiles"][MORRISON_PROFILE_ID] == {
        "selector": 10,
        "readiness": "WRF_MATCHED_RUN_RUNTIME_PROFILE",
        "explicit_expert_consent_required": False,
        "runtime_guards": [],
    }
    assert payload["physics_profiles"][KESSLER_PROFILE_ID] == {
        "selector": 1,
        "readiness": "IMPLEMENTED_UNVERIFIED",
        "explicit_expert_consent_required": False,
        "runtime_guards": [],
        "source_scope": ["hrrr"],
        "frozen_species_policy": (
            "retain QC/QR; discard source QI/QS/QG with a receipt"),
    }
    nssl2 = payload["physics_profiles"][NSSL2_PROFILE_ID]
    assert nssl2["selector"] == 18
    assert nssl2["readiness"] == "WRF_MATCHED_RUN_CANDIDATE"
    assert nssl2["explicit_expert_consent_required"] is False
    legacy = payload["physics_profiles"][NSSL2_LEGACY_RRTMG_PROFILE_ID]
    assert legacy["readiness"] == "WRF_MATCHED_RUN_CANDIDATE"
    assert legacy["radiation_solver"] == "legacy RRTMG"
    assert nssl2["resolved_fixed_preset"] is True
    assert payload["physics_profiles"][MYNN_PROFILE_ID]["readiness"] \
        == "IMPLEMENTED_UNVERIFIED"
    assert payload["physics_profiles"][RUC_PROFILE_ID]["readiness"] \
        == "IMPLEMENTED_UNVERIFIED"
    assert payload["physics_profiles"][MYNN_RUC_PROFILE_ID]["readiness"] \
        == "IMPLEMENTED_UNVERIFIED"
    noahmp = payload["physics_profiles"][NOAHMP_PROFILE_ID]
    assert noahmp["explicit_expert_consent_required"] is True
    assert noahmp["expert_acknowledgement_id"] \
        == "noahmp-host-column-throughput-v1"
    assert payload["capability_query"]["requires_cupy"] is False
    assert list(tmp_path.iterdir()) == []


def test_hrrr_history_consumes_native_reflectivity_for_every_stashed_scheme():
    for mp_physics in (1, 6, 8, 10, 18):
        assert hrrr_runner._history_uses_native_reflectivity(
            ticks=1, has_moisture=True, mp_physics=mp_physics)
    assert not hrrr_runner._history_uses_native_reflectivity(
        ticks=0, has_moisture=True, mp_physics=18)
    assert not hrrr_runner._history_uses_native_reflectivity(
        ticks=1, has_moisture=False, mp_physics=18)


def test_hrrr_capabilities_fail_closed_when_forecast_executor_is_absent(
        monkeypatch):
    monkeypatch.setattr(
        hrrr_runner,
        "_missing_forecast_executor_modules",
        lambda: ["woof.core.model", "woof.io.wrfout"],
    )

    payload = hrrr_runner.runner_capabilities()

    assert payload["readiness"] == "PREPARATION_ONLY_FORECAST_EXECUTOR_OMITTED"
    assert payload["modes"]["prepare-only"]["available"] is True
    assert payload["modes"]["forecast"] == {
        "available": False,
        "availability_scope": "executor-module-presence-only",
        "launch_ready": None,
        "launch_readiness_check": (
            "validate CuPy, CUDA, GPU allocation, inputs, and profile guards "
            "before launch"),
        "requires_cupy": True,
        "requires_compatible_cuda_gpu": True,
        "missing_executor_modules": [
            "woof.core.model", "woof.io.wrfout"],
        "included_in_standalone_rw_wps_wheel": False,
        "unavailable_reason": (
            "standalone RW-WPS omits the GPUWM forecast executor"),
    }


def test_native_prepare_workers_defaults_to_validated_two_worker_path():
    assert _parse_args(_required_args()).prepare_workers == 2


def test_native_source_window_and_physics_profile_are_retained():
    args = _parse_args(_required_args() + [
        "--forecast-start-hour", "12",
        "--forecast-end-hour", "13",
        "--physics-profile", THOMPSON_PROFILE_ID,
    ])
    assert args.forecast_start_hour == 12
    assert args.forecast_end_hour == 13
    assert args.physics_profile == THOMPSON_PROFILE_ID


@pytest.mark.parametrize("workers", (1, 2, 4, 8))
def test_native_prepare_worker_override_is_retained(workers):
    args = _required_args() + ["--prepare-workers", str(workers)]
    assert _parse_args(args).prepare_workers == workers


def test_native_hrrr_cpu_preprocess_selector_is_retained():
    args = _parse_args(_required_args() + [
        "--preprocess-backend", "cpu",
        "--preprocess-workers", "8",
        "--cpu-preprocess-bridge", "libgpuwm_preprocess_cpu.so",
    ])
    assert args.preprocess_backend == "cpu"
    assert args.preprocess_workers == 8
    assert args.cpu_preprocess_bridge.name == "libgpuwm_preprocess_cpu.so"


def test_native_hrrr_accepts_cuda_host_workers():
    """--preprocess-workers under CUDA sets the host steps' threads.

    The CUDA backend runs the masked surface fields in the Rust library
    on the host, so the count has a meaning there; it used to be refused
    late, after the front door had accepted and forwarded it.
    """
    args = _parse_args(_required_args() + [
        "--preprocess-backend", "cuda",
        "--preprocess-workers", "8",
    ])
    assert args.preprocess_backend == "cuda"
    assert args.preprocess_workers == 8


def test_cuda_preprocess_budget_records_its_host_step_workers():
    budget = _PreprocessWorkerBudget(
        backend="cuda", requested_total=6, effective_total=None,
        requested_job_slots=3, future_job_count=3, clock_origin=0.0,
        host_workers=6)
    receipt = budget.receipt()
    assert receipt["applicable"] is False
    assert receipt["requested_total_native_workers"] == 6
    assert receipt["host_step_native_workers"] == 6
    assert receipt["peak_active_native_workers"] == 6
    assert receipt["effective_allocation_per_job"] == []
    # Without the count the receipt would say the host steps took none.
    for missing in (None, 0, True):
        with pytest.raises(ValueError, match="host steps"):
            _PreprocessWorkerBudget(
                backend="cuda", requested_total=None, effective_total=None,
                requested_job_slots=1, future_job_count=1,
                clock_origin=0.0, host_workers=missing)
    # The CPU receipt is unchanged: its slots already cover the host steps.
    cpu = _PreprocessWorkerBudget(
        backend="cpu", requested_total=2, effective_total=2,
        requested_job_slots=1, future_job_count=0, clock_origin=0.0)
    assert "host_step_native_workers" not in cpu.receipt()


@pytest.mark.parametrize(("backend", "slot_workers", "host_workers", "seen"), (
    ("cuda", None, 5, 5),
    ("cuda", None, None, None),
    ("cpu", 3, None, 3),
))
def test_boundary_pool_worker_takes_the_cuda_host_workers(
        monkeypatch, backend, slot_workers, host_workers, seen):
    """A spawned boundary worker resolves the controller's host workers.

    Its backend's receipt must equal the controller's, whose masked
    chain names the host-step count under CUDA.
    """
    import tools.hrrr_single_domain_benchmark as runner

    observed = []

    def initialize(*args, **kwargs):
        observed.append(kwargs["preprocess_workers"])
        return {}, {}, {}, {}

    monkeypatch.setattr(runner, "_initialize_boundary_sides", initialize)
    monkeypatch.setattr(runner, "_PREPARE_WORKER_CONTEXT", None)
    runner._prepare_worker_init(
        object(), {}, [1.0, 0.0], 5000.0, 5, backend, None, True,
        host_workers)
    result = runner._prepare_boundary_hour(1, {}, slot_workers, 0)
    assert observed == [seen]
    assert result["effective_native_workers"] == slot_workers


def test_a_boundary_pool_worker_carries_the_controllers_selection(
        monkeypatch):
    """A spawned boundary worker's backend records the controller's choice.

    It resolves its backend again by name, which recorded a backend auto
    chose as one "named by the caller", and its receipt was then refused
    against the controller's.
    """
    import tools.hrrr_single_domain_benchmark as runner

    chosen = {"requested": "auto", "backend": "cpu",
              "reason": "no CUDA device is visible here"}
    observed = []

    def initialize(*args, **kwargs):
        observed.append(kwargs["preprocess_selection"])
        return {}, {}, {}, {}

    monkeypatch.setattr(runner, "_initialize_boundary_sides", initialize)
    monkeypatch.setattr(runner, "_PREPARE_WORKER_CONTEXT", None)
    runner._prepare_worker_init(
        object(), {}, [1.0, 0.0], 5000.0, 5, "cpu", None, True, None, chosen)
    runner._prepare_boundary_hour(1, {}, 2, 0)
    assert observed == [chosen]
    assert observed[0] is not chosen


def _vertical_columns(source_levels=6):
    levels = np.linspace(95000.0, 20000.0, source_levels, dtype=np.float32)
    source = np.broadcast_to(
        levels[:, None, None], (source_levels, 2, 3)).copy()
    target = np.broadcast_to(
        np.linspace(94000.0, 25000.0, 4, dtype=np.float32)[:, None, None],
        (4, 2, 3)).copy()
    return source, np.full((2, 3), 100000.0, np.float32), target


@pytest.mark.parametrize("requested", ("auto", "cpu"))
def test_a_slot_receipt_is_the_controllers_but_for_its_worker_counts(
        requested):
    """Every slot of a CPU budget passes the check on its own share.

    The controller's backend ran f00 on the whole budget, so its route
    record holds a vertical route, and a slot runs on its share.  A slot
    resolved again by name was refused at the first boundary hour of
    every CPU preparation with two or more boundary hours: its masked
    surface chain named the share, its route record was empty, and a
    CPU auto chose came back as one "named by the caller".
    """
    from woof.ingest import preprocess_backend as backends

    reason = ("no CUDA device is visible here" if requested == "auto"
              else backends.NAMED_BY_CALLER)
    controller = backends._selection(
        requested, backends.ParallelCpuPreprocessBackend(workers=4), reason)
    controller.prepare_wrf_vertical(*_vertical_columns())
    expected = controller.receipt()
    assert expected["masked_surface_chain"]["workers"] == 4
    assert expected["vertical_interpolation"]

    slot = controller.at_workers(2)
    assert slot.workers == 2
    receipt = slot.receipt()
    assert receipt["masked_surface_chain"]["workers"] == 2
    assert receipt["selection"] == expected["selection"]
    hrrr_runner._require_preprocess_receipt(
        expected, receipt, "f01 west mapping", expected_native_workers=2)
    # A route the slot prepares is on the controller's record too.
    slot.prepare_wrf_vertical(*_vertical_columns(source_levels=7))
    assert [route["source_levels"] for route in
            controller.receipt()["vertical_interpolation"]] == [6, 7]

    resolved = backends.resolve_preprocess_backend("cpu", workers=2)
    if requested == "auto":
        # auto resolved again by name records a CPU nobody named.
        with pytest.raises(RuntimeError, match="differs from the resolved"):
            hrrr_runner._require_preprocess_receipt(
                expected, resolved.receipt(), "f01 west mapping",
                expected_native_workers=2)
    else:
        # A named cpu resolved again carries the controller's selection:
        # that is a spawned boundary worker, whose own route record starts
        # empty and is not a difference (A75).  A route at a depth the
        # controller met that differs from the controller's still is.
        hrrr_runner._require_preprocess_receipt(
            expected, resolved.receipt(), "f01 west mapping",
            expected_native_workers=2)
        stray = copy.deepcopy(resolved.receipt())
        stray["vertical_interpolation"] = [
            dict(expected["vertical_interpolation"][0], backend="cuda")]
        with pytest.raises(RuntimeError, match="differs from the resolved"):
            hrrr_runner._require_preprocess_receipt(
                expected, stray, "f01 west mapping",
                expected_native_workers=2)


def test_a_spawned_worker_at_another_source_depth_passes_and_is_recorded():
    """A spawned boundary worker keeps its own vertical route record.

    The controller's record names the depth f00 had; a worker process
    that prepared a boundary hour with another source level count was
    refused on that difference in ``vertical_interpolation``, although
    it routed that depth exactly as the controller would.  The worker's
    new route passes and lands on the controller's record; a route that
    differs from the controller's at the same depth is still refused.
    """
    from woof.ingest import preprocess_backend as backends

    reason = "no CUDA device is visible here"
    controller = backends._selection(
        "auto", backends.ParallelCpuPreprocessBackend(workers=4), reason)
    controller.prepare_wrf_vertical(*_vertical_columns(source_levels=6))
    expected = controller.receipt()

    # A spawned worker builds its own backend with the controller's
    # selection, so its route record starts empty (not at_workers).
    worker = backends._selection(
        "auto", backends.ParallelCpuPreprocessBackend(workers=2), reason)
    worker.prepare_wrf_vertical(*_vertical_columns(source_levels=6))
    worker.prepare_wrf_vertical(*_vertical_columns(source_levels=7))
    hrrr_runner._require_preprocess_receipt(
        expected, copy.deepcopy(worker.receipt()),
        "f01 boundary initialization", expected_native_workers=2)
    assert [route["source_levels"] for route in
            controller.receipt()["vertical_interpolation"]] == [6, 7]

    stray = copy.deepcopy(worker.receipt())
    stray["vertical_interpolation"][0]["backend"] = "cuda"
    with pytest.raises(RuntimeError, match="in vertical_interpolation"):
        hrrr_runner._require_preprocess_receipt(
            expected, stray, "f02 boundary initialization",
            expected_native_workers=2)


def test_the_receipt_check_holds_each_worker_count_to_the_allocation():
    """The counts left out of the comparison are checked, not dropped.

    Each is held to at most its allocation.  Fewer is the backend's own
    answer to a busy host (its memory headroom fell between the budget
    and the job): "f00 mapping used 20 native workers; expected 24"
    refused a CPU preparation on a loaded test host although the count moves
    no byte.  More than the allocation is the oversubscription the check
    exists for.
    """
    from woof.ingest import preprocess_backend as backends

    controller = backends._selection(
        "cpu", backends.ParallelCpuPreprocessBackend(workers=4),
        backends.NAMED_BY_CALLER)
    expected = controller.receipt()
    slot = controller.at_workers(2).receipt()
    # Fewer than the allocation: the job ran, inside its share.
    hrrr_runner._require_preprocess_receipt(
        expected, slot, "f00 mapping", expected_native_workers=3)
    # More than the allocation: the controller's whole budget on a slot.
    with pytest.raises(
            RuntimeError,
            match="used 4 native workers; its allocation is at most 2"):
        hrrr_runner._require_preprocess_receipt(
            expected, controller.receipt(), "f01 west mapping",
            expected_native_workers=2)
    for count in (0, None, True, "auto"):
        bad = dict(copy.deepcopy(slot), workers=count)
        with pytest.raises(RuntimeError, match="its allocation is at most 2"):
            hrrr_runner._require_preprocess_receipt(
                expected, bad, "f01 west mapping", expected_native_workers=2)
    stray = copy.deepcopy(slot)
    stray["masked_surface_chain"]["workers"] = 3
    with pytest.raises(RuntimeError,
                       match="masked surface fields on 3 native workers"):
        hrrr_runner._require_preprocess_receipt(
            expected, stray, "f01 west mapping", expected_native_workers=2)
    other = copy.deepcopy(slot)
    other["masked_surface_chain"]["entry"] = "another_entry"
    with pytest.raises(RuntimeError, match="in masked_surface_chain"):
        hrrr_runner._require_preprocess_receipt(
            expected, other, "f01 west mapping", expected_native_workers=2)
    # The controller's own receipt is left as it was.
    assert expected["masked_surface_chain"]["workers"] == 4
    assert expected["workers"] == 4


def test_native_preprocess_budget_is_partitioned_not_multiplied_per_job():
    budget = _PreprocessWorkerBudget(
        backend="cpu", requested_total=8, effective_total=8,
        requested_job_slots=3, future_job_count=3, clock_origin=0.0)
    assert _partition_preprocess_worker_budget(8, 3) == (3, 3, 2)
    assert budget.slot_workers == (3, 3, 2)
    assert [budget.allocation_for_hour(hour) for hour in (1, 2, 3)] == [
        (0, 3), (1, 3), (2, 2)]

    budget.record(
        forecast_hour=0, phase="full_domain_initialization", slot=None,
        native_workers=8, started=0.1, finished=1.0)
    budget.record(
        forecast_hour=1, phase="boundary_mapping", slot=0,
        native_workers=3, started=1.1, finished=1.2)
    budget.record(
        forecast_hour=1, phase="boundary_initialization", slot=0,
        native_workers=3, started=1.3, finished=5.0)
    budget.record(
        forecast_hour=2, phase="boundary_mapping", slot=1,
        native_workers=3, started=1.4, finished=2.0)
    budget.record(
        forecast_hour=2, phase="boundary_initialization", slot=1,
        native_workers=3, started=2.1, finished=5.0)
    budget.record(
        forecast_hour=3, phase="boundary_mapping", slot=2,
        native_workers=2, started=2.2, finished=2.4)
    budget.record(
        forecast_hour=3, phase="boundary_initialization", slot=2,
        native_workers=2, started=2.5, finished=5.0)

    receipt = budget.receipt()
    assert receipt["requested_total_native_workers"] == 8
    assert receipt["effective_total_native_workers"] == 8
    assert receipt["peak_active_native_workers"] == 8
    assert receipt["peak_active_preprocessing_jobs"] == 3
    assert len(receipt["effective_allocation_per_job"]) == 7
    assert receipt["pipeline_decoder_workers_included"] is False


def test_native_preprocess_budget_caps_concurrent_jobs_to_total_threads():
    budget = _PreprocessWorkerBudget(
        backend="cpu", requested_total=2, effective_total=2,
        requested_job_slots=8, future_job_count=4, clock_origin=0.0)
    assert budget.slot_workers == (1, 1)
    assert [budget.allocation_for_hour(hour) for hour in (1, 2, 3, 4)] == [
        (0, 1), (1, 1), (0, 1), (1, 1)]


def test_public_hrrr_receipt_keeps_decoder_and_native_budgets_separate():
    report = {
        "status": "PASS",
        "preparation": {
            "preprocess_backend": {"backend": "cpu", "workers": 8},
            "preprocess_worker_budget": {
                "schema": "gpuwm-preprocess-worker-budget-v1",
                "backend": "cpu",
                "applicable": True,
                "requested_total_native_workers": 8,
                "effective_total_native_workers": 8,
                "effective_allocation_per_job": [
                    {"effective_native_workers": 8},
                    {"effective_native_workers": 4},
                    {"effective_native_workers": 4},
                ],
                "peak_active_native_workers": 8,
                "pipeline_decoder_workers_included": False,
            },
        },
        "pipeline": {
            "workers": {"requested": "4", "selected": 4},
        },
    }

    _, native_budget, decoder_budget = _validated_worker_receipts(
        report, selected_backend="cpu", requested_preprocess_workers=8,
        requested_pipeline_workers="4", final_hour=1)

    assert native_budget["peak_active_native_workers"] == 8
    assert decoder_budget["selected"] == 4
    report["preparation"]["preprocess_worker_budget"][
        "peak_active_native_workers"] = 9
    with pytest.raises(RuntimeError, match="budget was not honored"):
        _validated_worker_receipts(
            report, selected_backend="cpu", requested_preprocess_workers=8,
            requested_pipeline_workers="4", final_hour=1)


def test_public_hrrr_receipt_takes_a_backend_below_its_budget():
    """The controller's backend may run on fewer threads than the budget.

    A busy host narrowed the CPUs between the budget (24) and the
    backend's own count (20): "f00 mapping used 20 native workers;
    expected 24", then "worker budget was not honored", although the
    count moves no byte.  More than the budget or the request is still
    refused.
    """
    def report(backend_workers):
        return {
            "status": "PASS",
            "preparation": {
                "preprocess_backend": {
                    "backend": "cpu", "workers": backend_workers},
                "preprocess_worker_budget": {
                    "schema": "gpuwm-preprocess-worker-budget-v1",
                    "backend": "cpu",
                    "applicable": True,
                    "requested_total_native_workers": 24,
                    "effective_total_native_workers": 24,
                    "effective_allocation_per_job": [
                        {"effective_native_workers": 24},
                        {"effective_native_workers": 12},
                        {"effective_native_workers": 12},
                    ],
                    "peak_active_native_workers": 24,
                    "pipeline_decoder_workers_included": False,
                },
            },
            "pipeline": {"workers": {"requested": "4", "selected": 4}},
        }

    options = dict(selected_backend="cpu", requested_pipeline_workers="4",
                   final_hour=1)
    for requested in (None, 24):
        _validated_worker_receipts(
            report(20), requested_preprocess_workers=requested, **options)
    with pytest.raises(RuntimeError, match="budget was not honored"):
        _validated_worker_receipts(
            report(25), requested_preprocess_workers=None, **options)
    with pytest.raises(RuntimeError, match="differs from the request"):
        _validated_worker_receipts(
            report(25), requested_preprocess_workers=24, **options)
    for bad in (0, None, True, "auto"):
        with pytest.raises(RuntimeError, match="budget was not honored"):
            _validated_worker_receipts(
                report(bad), requested_preprocess_workers=None, **options)


def test_pipeline_mode_accepts_launch_ready_prepared_cache():
    common = _required_args()
    manifest_index = common.index("--manifest-sha256")
    del common[manifest_index:manifest_index + 2]
    args = _parse_args(common + [
        "--pipeline-series", "series.tsv",
        "--pipeline-decoder", "decoder",
        "--pipeline-signals", "signals",
        "--source-root", "source",
        "--source-manifest", "SHA256SUMS",
        "--source-manifest-sha256", "source-manifest-sha256",
        "--prepared-cache", "prepared-cache",
        "--prepare-only",
        "--history-interval-seconds", "3600",
    ])
    assert args.pipeline_series.name == "series.tsv"
    assert args.prepare_only
    assert args.prepared_cache.name == "prepared-cache"
    assert args.io_mode == "none"
    assert args.history_interval_seconds == 3600.0


def test_prepare_only_cadence_is_bound_into_cache_identity_without_output():
    args = _parse_args(_required_args() + [
        "--source-manifest-sha256", "a" * 64,
        "--prepared-cache", "prepared-cache",
        "--prepare-only",
        "--history-interval-seconds", "3600",
    ])
    vertical = VerticalConfig(
        eta_levels=(1.0, 0.75, 0.5, 0.25, 0.0),
        p_top=5000.0, hybrid_opt=2, etac=0.2)
    target = dataclasses.replace(
        HrrrTargetDomain.legacy_500x500(), nz=4)
    exp = _experiment(
        vertical, run_seconds=3600.0, target=target,
        physics_profile=THOMPSON_RTE_RRTMGP_PROFILE_ID,
        history_interval_seconds=args.history_interval_seconds)

    def identity(domain):
        return prepared_cache_identity(
            bridge_manifest_sha256="b" * 64,
            source_manifest_sha256="a" * 64,
            static_cache_sha256="c" * 64,
            namelist_sha256="d" * 64,
            domain_config=domain,
            forcing_hours=(0, 1),
            source_identity={"source": "fixture"},
        )

    prepared_identity = identity(exp.root)
    forecast_exp = _experiment(
        vertical, run_seconds=3600.0, target=target,
        physics_profile=THOMPSON_RTE_RRTMGP_PROFILE_ID,
        history_interval_seconds=3600.0)
    legacy_default_exp = _experiment(
        vertical, run_seconds=3600.0, target=target,
        physics_profile=THOMPSON_RTE_RRTMGP_PROFILE_ID,
        history_interval_seconds=300.0)

    assert args.prepare_only is True
    assert args.io_mode == "none"
    assert identity(forecast_exp.root) == prepared_identity
    assert identity(legacy_default_exp.root) != prepared_identity
    assert prepared_identity["domain_config"]["history_interval_s"] == 3600.0
    assert prepared_identity["domain_config"]["run"][
        "output_interval_s"] == 3600.0


def _extension_namelist(tmp_path, *, lead: int):
    cycle = datetime(2026, 7, 20, 6)
    template = tmp_path / "namelist.template"
    template.write_text(
        "&time_control\n/\n&domains\n max_dom = 1,\n/\n",
        encoding="utf-8")
    output = tmp_path / f"namelist-f{lead:03d}.input"
    _materialize_input_namelist(
        template, output, cycle=cycle, lead=lead,
        domain_starts=[cycle])
    return cycle, output


def test_sealed_prepare_recomputes_namelist_invariant_from_bytes(tmp_path):
    cycle, namelist = _extension_namelist(tmp_path, lead=1)
    common = _required_args() + [
        "--source-manifest-sha256", "a" * 64,
        "--prepared-cache", "prepared-cache",
        "--namelist-input", str(namelist),
        "--prepare-only",
        "--history-interval-seconds", "3600",
        "--sealed-prepared-cache",
    ]
    args = _parse_args(common)
    invariant = _validated_namelist_extension_identity(args, cycle=cycle)
    assert invariant["schema"] == "gpuwm-namelist-extension-invariant-v1"
    assert len(invariant["sha256"]) == 64

    payload = namelist.read_text(encoding="utf-8")
    assert " end_hour = 7," in payload
    namelist.write_text(
        payload.replace(" end_hour = 7,", " end_hour = 8,"),
        encoding="utf-8")
    with pytest.raises(ValueError, match="end time differs"):
        _validated_namelist_extension_identity(args, cycle=cycle)

    with pytest.raises(SystemExit):
        _parse_args(common + [
            "--namelist-extension-invariant-sha256", "b" * 64,
        ])


def test_suffix_prepare_recomputes_full_horizon_namelist_invariant(tmp_path):
    cycle, namelist = _extension_namelist(tmp_path, lead=2)
    args = _parse_args(_required_args() + [
        "--source-manifest-sha256", "a" * 64,
        "--prepared-cache", "prepared-cache",
        "--namelist-input", str(namelist),
        "--forecast-start-hour", "1",
        "--forecast-end-hour", "2",
        "--prepare-only",
        "--history-interval-seconds", "3600",
        "--namelist-extension-suffix",
    ])
    assert args.sealed_prepared_cache is False
    assert _validated_namelist_extension_identity(
        args, cycle=cycle)["schema"] \
        == "gpuwm-namelist-extension-invariant-v1"


def test_prepare_only_main_summary_does_not_require_forecast_cadence(
        tmp_path, monkeypatch, capsys):
    report = {
        "status": "PASS",
        "run_seconds": 43_200.0,
        "io_mode": "prepare-only",
        "memory": {"gpu_peak_used_bytes_observed": 0},
        "prepared_cache": {"content_sha256": "a" * 64},
    }
    monkeypatch.setattr(
        "tools.hrrr_single_domain_benchmark._parse_args",
        lambda _argv: type("Args", (), {"outdir": tmp_path})())
    monkeypatch.setattr(
        "tools.hrrr_single_domain_benchmark.run", lambda _args: report)

    assert hrrr_runner_main([]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary == {
        "gpu_peak_used_bytes": 0,
        "history_interval_seconds": None,
        "io_mode": "prepare-only",
        "prepared_cache_content_sha256": "a" * 64,
        "run_seconds": 43_200.0,
        "status": "PASS",
    }


def test_boundary_mapping_targets_are_bitwise_full_grid_slices():
    ny, nx, width = 13, 17, 3
    grid = LambertGrid(
        35.5, -98.0, 30.0, 60.0, -97.0,
        1000.0, 1000.0, nx + 1, ny + 1)
    landmask = (np.arange(ny * nx).reshape(ny, nx) % 2).astype(np.float64)
    targets = _boundary_mapping_targets(
        grid, {"LANDMASK": landmask}, SimpleNamespace(nx=nx, ny=ny),
        width=width)
    rectangles = {
        "west": (0, ny, 0, width),
        "east": (0, ny, nx - width, nx),
        "south": (0, width, 0, nx),
        "north": (ny - width, ny, 0, nx),
    }
    full_mass = grid.latlon_mass()
    full_u = grid.latlon_u()
    full_v = grid.latlon_v()
    for side, (y0, y1, x0, x1) in rectangles.items():
        subgrid, target_landmask = targets[side]
        for actual, expected in zip(
                subgrid.latlon_mass(),
                (value[y0:y1, x0:x1] for value in full_mass)):
            np.testing.assert_array_equal(actual, expected)
        for actual, expected in zip(
                subgrid.latlon_u(),
                (value[y0:y1, x0:x1 + 1] for value in full_u)):
            np.testing.assert_array_equal(actual, expected)
        for actual, expected in zip(
                subgrid.latlon_v(),
                (value[y0:y1 + 1, x0:x1] for value in full_v)):
            np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(
            target_landmask, landmask[y0:y1, x0:x1])
        assert target_landmask.flags.c_contiguous


@pytest.mark.requires_capability("masked_stencil_bridge")
def test_f00_and_boundary_mapping_forward_explicit_target_radius(monkeypatch):
    observed = []
    parents = []

    def fake_interpolate(
            snapshot, grid, *, target_landmask, soil_mapping_report,
            surface_fallback_radius, backend, target_name,
            lattice_parent=None):
        observed.append((surface_fallback_radius, backend, target_name))
        parents.append(lattice_parent)
        soil_mapping_report.update({
            "land_stencil": {
                "fallback_radius_cells": surface_fallback_radius,
                "unresolved_target_count": 0,
                "cross_surface_donor_count": 0,
            },
        })
        return SimpleNamespace(
            valid_time="2026-07-20T06:00:00",
            levels_hpa=np.array([1.0]),
            fields={"Q2": np.ones(np.asarray(target_landmask).shape)},
        )

    monkeypatch.setattr(
        "woof.ingest.hrrr.interpolate_hrrr_to_lambert",
        fake_interpolate)
    grid = object()
    static = {"LANDMASK": np.ones((2, 2))}
    f00_report = {}
    _map_snapshot(
        object(), grid, static, f00_report, surface_fallback_radius=10,
        preprocess_backend="cpu-backend")
    targets = {
        side: (grid, np.ones((2, 2)))
        for side in ("west", "east", "south", "north")
    }
    boundary_report = {}
    _map_boundary_snapshot(
        object(), targets, boundary_report, surface_fallback_radius=10,
        preprocess_backend="cpu-backend", domain_grid=grid)

    assert [entry[:2] for entry in observed] == [(10, "cpu-backend")] * 5
    # Each of the five mappings names ITSELF.  A soil refusal from one of
    # four independently mapped boundary strips used to name neither the
    # strip nor the domain, which on a coastal domain left four
    # identical-looking suspects.
    assert [entry[2] for entry in observed] == [
        "domain 1",
        "the west boundary strip of domain 1",
        "the east boundary strip of domain 1",
        "the south boundary strip of domain 1",
        "the north boundary strip of domain 1",
    ]
    assert f00_report["land_stencil"]["fallback_radius_cells"] == 10
    assert all(
        report["land_stencil"]["fallback_radius_cells"] == 10
        for report in boundary_report["sides"].values())
    # Each strip names the domain it was cut from, so the strips of the
    # native grid take its identity route (acceptance D-05, 2.8.8).
    assert parents == [None] + [grid] * 4


@pytest.mark.parametrize("nz", [4, 17, 49, 80])
def test_native_hrrr_experiment_threads_admitted_explicit_vertical_grid(nz):
    # Every depth here runs the explicit YSU modern-radiation suite, so
    # the four-level geometry control remains independent of the MYNN
    # route default's five-level minimum. At p_top = 12_345 Pa each 4/4 arm builds
    # 31 cap layers above the model top, so the deepest case is
    # 80 + 31 = 111 radiation layers against a bound of 128 on either
    # engine (MAX_LEGACY_LONGWAVE_LAYERS and MAX_RRTMGP_LAYERS,
    # woof/physics_vertical_contract.py).  The deep case used to swap
    # to a shallower-bounded suite to dodge a 64-layer shortwave cap
    # that no engine holds: 64 is the layer count a 51-level ladder
    # reaches at p_top = 5000 Pa, not a cap.  The bound that does exist
    # is asserted in both directions, on both arms, by the test below.
    profile = THOMPSON_RTE_RRTMGP_PROFILE_ID
    target = dataclasses.replace(HrrrTargetDomain.legacy_500x500(), nz=nz)
    vertical = VerticalConfig(
        eta_levels=tuple(float(value)
                         for value in np.linspace(1.0, 0.0, nz + 1)),
        p_top=12_345.0,
        hybrid_opt=2,
        etac=0.37,
    )
    exp = _experiment(vertical, run_seconds=300.0, target=target,
                      physics_profile=profile)
    assert exp.root.run.nz == nz
    assert exp.vertical == vertical


def _deep_column_experiment(nz, profile):
    """This route's experiment at an explicit depth, on a named suite."""

    return _experiment(
        VerticalConfig(
            eta_levels=tuple(
                float(value)
                for value in np.linspace(1.0, 0.0, nz + 1)),
            p_top=12_345.0, hybrid_opt=2, etac=0.37),
        run_seconds=300.0,
        target=dataclasses.replace(
            HrrrTargetDomain.legacy_500x500(), nz=nz),
        physics_profile=profile)


@pytest.mark.parametrize(
    ("profile", "engine"),
    ((THOMPSON_LEGACY_RRTMG_PROFILE_ID, "legacy RRTMG"),
     (THOMPSON_RTE_RRTMGP_PROFILE_ID, "RTE+RRTMGP")))
def test_either_radiation_arm_bounds_a_longwave_column_at_the_same_number(
        profile, engine):
    """The one real limit these two suites carry, both directions.

    They differ in the radiation engine and nothing else, and the engine
    is NOT what the limit depends on: each 4/4 longwave engine bounds a
    column at 128 radiation layers -- the legacy one at WRF's
    RLW_MAXLAY, the modern one at the Planck-source kernel's fixed
    pfrac[128] -- and the resolved vertical preflight refuses either by
    number, before a preparation has been paid for.  A column is deeper
    than the ladder because a cap is built above the model top, 31
    layers at this p_top on both arms, so the depth that refuses is the
    same depth on both.  Binding this to the route default would make
    the test follow whichever arm the route currently names and stop
    covering the other; both are named instead.
    """
    from woof.physics_compat import PhysicsVerticalPreflightError

    # HRRR's own native depth builds, and so does a hand-authored deeper
    # one: the shortwave workspaces are sized to the run on both arms.
    assert _deep_column_experiment(49, profile).root.run.nz == 49
    assert _deep_column_experiment(80, profile).root.run.nz == 80
    # 97 + 31 = 128 reaches the bound exactly and still builds.
    assert _deep_column_experiment(97, profile).root.run.nz == 97
    # 98 + 31 = 129 is over it, and the refusal says so in numbers.
    with pytest.raises(PhysicsVerticalPreflightError) as caught:
        _deep_column_experiment(98, profile)
    assert (f"{engine} longwave requires model plus cap layers <= 128, "
            "got 98+31=129") in str(caught.value)


def test_the_route_default_activates_the_explicit_fork_composition():
    assert ROUTE_DEFAULT_PHYSICS_PROFILE == hrrr_runner.THOMPSON_MYNN_GSD41_PROFILE_ID
    run = _deep_column_experiment(49, ROUTE_DEFAULT_PHYSICS_PROFILE).root.run
    assert run.sf_sfclay_physics == run.bl_pbl_physics == 5
    assert run.mynn_sfclay_variant == "gsl_wrf39"
    assert run.sf_surface_physics == 3
    assert run.num_soil_layers == 9
    assert (run.ra_lw_physics, run.ra_sw_physics) == (4, 4)
    assert run.ra_rrtmg_variant == "rrtmg_legacy"
    assert run.bl_mynn_version == "gsd_41"
    assert run.bl_mynn_mixlength == 2
    assert run.mp_physics == 28
    assert (run.aer_init_opt, run.wif_input_opt) == (1, 1)


def test_source_version_namelist_contract_round_trips_and_refuses_drift(tmp_path, monkeypatch):
    from woof.hrrr_route_inputs import render_namelist_input
    from woof import thompson_fork_assets
    fork_root = tmp_path / "unacquired-fork-cache"
    monkeypatch.setenv("WOOF_THOMPSON_FORK_TABLE_ROOT", str(fork_root))
    monkeypatch.setattr(thompson_fork_assets, "ensure_thompson_fork_tables",
                        lambda *a, **k: pytest.fail("configuration preview acquired tables"))
    exp = _deep_column_experiment(49, hrrr_runner.THOMPSON_MYNN_GSD41_PROFILE_ID)
    text = render_namelist_input(exp)
    path = tmp_path / "namelist.input"
    path.write_text(text)
    receipt = _validate_native_hrrr_physics_profile(
        path, hrrr_runner.THOMPSON_MYNN_GSD41_PROFILE_ID)
    assert receipt["readiness"] == "IMPLEMENTED_UNVERIFIED"
    assert receipt["microphysics_table_authority"]["mp_physics"] == 28
    assert receipt["microphysics_table_authority"]["asset_validation"] == "deferred_to_runtime_before_first_use"
    assert not fork_root.exists()
    assert receipt["validated_namelist"]["physics"]["bl_mynn_tkebudget"] == 0
    import re
    changed = re.sub(r"(bl_mynn_tkebudget\s*=\s*)0", r"\g<1>1", text)
    assert changed != text
    path.write_text(changed)
    with pytest.raises(ValueError, match="bl_mynn_tkebudget"):
          _validate_native_hrrr_physics_profile(
              path, hrrr_runner.THOMPSON_MYNN_GSD41_PROFILE_ID)


@pytest.mark.parametrize("profile", [
    hrrr_runner.MYNN_NOAHMP_PROFILE_ID,
    hrrr_runner.MYNN_NOAHMP_RTE_RRTMGP_PROFILE_ID,
    hrrr_runner.MYNN_RUC_PROFILE_ID,
    hrrr_runner.MYNN_PROFILE_ID,
])
def test_ruc_prescribed_monthly_defaults_follow_the_selected_lsm(profile):
    exp = _deep_column_experiment(49, profile)
    expected = exp.root.run.sf_surface_physics in (2, 3)
    assert exp.root.run.usemonalb is expected
    assert exp.root.run.rdlai2d is expected


@pytest.mark.parametrize("profile", [
    hrrr_runner.MYNN_RUC_PROFILE_ID, hrrr_runner.MYNN_PROFILE_ID,
])
def test_ruc_source_monthly_defaults_preserve_explicit_false(monkeypatch, profile):
    original = hrrr_runner._native_hrrr_runtime_switches
    def switches(profile):
        result = dict(original(profile))
        result.update(usemonalb=False, rdlai2d=False)
        return result
    monkeypatch.setattr(hrrr_runner, "_native_hrrr_runtime_switches", switches)
    exp = _deep_column_experiment(49, profile)
    assert exp.root.run.usemonalb is False
    assert exp.root.run.rdlai2d is False


def test_benchmark_can_state_historical_monthly_false_controls():
    vertical = VerticalConfig(
        eta_levels=tuple(float(value) for value in np.linspace(1.0, 0.0, 50)),
        p_top=5000.0, hybrid_opt=2, etac=0.2)
    exp = _experiment(vertical, run_seconds=300.0,
                      physics_profile=THOMPSON_RTE_RRTMGP_PROFILE_ID,
                      usemonalb=False, rdlai2d=False)
    assert exp.root.run.usemonalb is False
    assert exp.root.run.rdlai2d is False


@pytest.mark.parametrize(
    ("run_seconds", "cadence_seconds", "expected_offsets", "final_suffix",
     "last_equals_run_end"),
    (
        (3600.0, 900.0, [0.0, 900.0, 1800.0, 2700.0, 3600.0],
         "01_00_00", True),
        (14400.0, 7200.0, [0.0, 7200.0, 14400.0], "04_00_00", True),
        (10800.0, 7200.0, [0.0, 7200.0], "02_00_00", False),
    ),
)
def test_native_hrrr_history_uses_floor_schedule_for_any_due_output(
        run_seconds, cadence_seconds, expected_offsets, final_suffix,
        last_equals_run_end):
    vertical = VerticalConfig(
        eta_levels=(1.0, 0.75, 0.5, 0.25, 0.0),
        p_top=5000.0, hybrid_opt=2, etac=0.2)
    target = dataclasses.replace(
        HrrrTargetDomain.legacy_500x500(), nz=4)
    exp = _experiment(
        vertical, run_seconds=run_seconds, target=target,
        physics_profile=THOMPSON_RTE_RRTMGP_PROFILE_ID,
        history_interval_seconds=cadence_seconds)

    receipt = hrrr_runner._validate_history_output_cadence(
        exp, cadence_seconds)
    schedule = hrrr_runner._history_output_schedule(
        start_time=exp.start_time, run_seconds=run_seconds,
        cadence_seconds=cadence_seconds)

    assert [record[0] for record in schedule] == expected_offsets
    assert schedule[-1][2].endswith(final_suffix)
    assert receipt["expected_frame_count"] == len(schedule)
    assert receipt["last_scheduled_offset_seconds"] == expected_offsets[-1]
    assert receipt["last_scheduled_valid_time"] == schedule[-1][1].isoformat()
    assert receipt["run_end_offset_seconds"] == run_seconds
    assert receipt["last_scheduled_equals_run_end"] is last_equals_run_end
    assert receipt["run_end_frame_scheduled"] is last_equals_run_end
    assert exp.root.run.output_interval_s == cadence_seconds


def test_native_hrrr_history_rejects_mismatch_and_nonstep_cadence():
    vertical = VerticalConfig(
        eta_levels=(1.0, 0.75, 0.5, 0.25, 0.0),
        p_top=5000.0, hybrid_opt=2, etac=0.2)
    target = dataclasses.replace(
        HrrrTargetDomain.legacy_500x500(), nz=4)
    exp = _experiment(
        vertical, run_seconds=10800.0, target=target,
        physics_profile=THOMPSON_RTE_RRTMGP_PROFILE_ID,
        history_interval_seconds=7200.0)

    with pytest.raises(ValueError, match="exactly match the generated"):
        hrrr_runner._validate_history_output_cadence(exp, 3600.0)
    domain = dataclasses.replace(
        exp.root, history_interval_s=71.0,
        run=dataclasses.replace(exp.root.run, output_interval_s=71.0))
    nonstep = dataclasses.replace(exp, domains=(domain,))
    with pytest.raises(ValueError, match="whole number of exact model"):
        hrrr_runner._validate_history_output_cadence(nonstep, 71.0)


def test_native_hrrr_cli_requires_explicit_history_cadence():
    args = _parse_args(_required_args() + [
        "--io-mode", "history", "--history-interval-seconds", "900"])
    assert args.io_mode == "history"
    assert args.history_interval_seconds == 900.0

    with pytest.raises(SystemExit):
        _parse_args(_required_args() + ["--io-mode", "history"])
    with pytest.raises(SystemExit):
        _parse_args(_required_args() + [
            "--io-mode", "history", "--history-interval-seconds", "nan"])
    with pytest.raises(SystemExit):
        _parse_args(_required_args() + [
            "--io-mode", "none", "--history-interval-seconds", "900"])


def _write_native_physics_namelist(
        path, *, mp_physics=6, include_diff_factor=True,
        full_real74_suite=False, morr_rimed_ice=1, cu_physics=1,
        nssl_controls="", sf_sfclay_physics=91,
        sf_surface_physics=2, bl_pbl_physics=1,
        num_soil_layers=None):
    # d01 comes FIRST in every per-domain column here, because this
    # preparer prepares the root: the profile-conforming value is d01's and
    # the second entry is a nest value that must NOT be selected.  The
    # columns stay deliberately non-uniform, which is what makes reading
    # the wrong end of a WRF array a visible failure rather than a
    # coincidence.
    diff_factor = (" diff_6th_factor = 0.08, 0.10,\n"
                   if include_diff_factor else "")
    if full_real74_suite:
        diff_factor = (" diff_6th_factor = 0.12, 0.10,\n"
                       if include_diff_factor else "")
        radiation = """ ra_lw_physics = 4, 4,
 ra_sw_physics = 4, 4,
 radt = 12, 12,
 sf_sfclay_physics = 91, 91,
 sf_surface_physics = 2, 2,
 bl_pbl_physics = 1, 1,
 cu_physics = {cu_physics}, 1,
 cudt = 5, 5,
 num_soil_layers = 4, 4,
""".format(cu_physics=cu_physics)
        dynamics = """ epssm = 0.5, 0.1,
 top_lid = .false., .true.,
"""
        morrison = (
            f" morr_rimed_ice = {morr_rimed_ice}, 1,\n"
            if mp_physics == 10 else "")
    else:
        soil = (
            f" num_soil_layers = {num_soil_layers}, 4,\n"
            if num_soil_layers is not None else "")
        radiation = f""" ra_lw_physics = 0, 0,
 ra_sw_physics = 1, 1,
 radt = 1, 3,
 sf_sfclay_physics = {sf_sfclay_physics}, 91,
 sf_surface_physics = {sf_surface_physics}, 2,
 bl_pbl_physics = {bl_pbl_physics}, 1,
 cu_physics = 0, 0,
{soil}"""
        dynamics = ""
        morrison = ""
    path.write_text(f"""&physics
 mp_physics = {mp_physics}, 6,
 {morrison}{nssl_controls}{radiation}\
/
&dynamics
{dynamics}\
 km_opt = 4, 4,
 diff_6th_opt = 2, 2,
{diff_factor} diff_6th_slopeopt = 1, 1,
/
""", encoding="ascii")


def test_native_hrrr_fixed_physics_profile_validates_the_root_domain(
        tmp_path):
    path = tmp_path / "namelist.input"
    _write_native_physics_namelist(path)
    # Named explicitly since 1.8: the route default is the full-radiation
    # suite, and this namelist describes WSM6.  The wsm6 family stays
    # fully selectable -- as a stated choice, which is the point.
    receipt = _validate_native_hrrr_physics_profile(path, WSM6_PROFILE_ID)
    assert receipt["profile"] == WSM6_PROFILE_ID
    assert receipt["selection"] == "d01 (first-or-only) WRF domain value"
    assert receipt["validated_namelist"]["physics"]["radt"] == 1.0
    assert receipt["validated_namelist"]["dynamics"][
        "diff_6th_factor"] == 0.08


def test_native_hrrr_root_preparer_reads_d01_and_names_the_nest_it_ignored(
        tmp_path):
    """The exact v1.3.1 field report, both halves.

    A user handed this single-domain root preparer their two-domain
    namelist.  d01 stated the WSM6 profile's diff_6th_factor = 0.08; d02
    stated a nest's 0.10.  The preparer read the LAST column, refused with
    "selected last-domain value must be 0.08, got 0.1", and the user
    changed d02 -- which is not what they meant and not what the root
    runs.  d01 is now what binds, and a refusal that happens to land on a
    per-domain column says which domain it read and how to fix it.
    """

    good = tmp_path / "d01-conforming.input"
    _write_native_physics_namelist(good)
    assert " diff_6th_factor = 0.08, 0.10,\n" in good.read_text()
    receipt = _validate_native_hrrr_physics_profile(good, WSM6_PROFILE_ID)
    assert receipt["validated_namelist"]["dynamics"][
        "diff_6th_factor"] == 0.08

    drifted = tmp_path / "d01-drifted.input"
    drifted.write_text(
        good.read_text().replace(
            " diff_6th_factor = 0.08, 0.10,",
            " diff_6th_factor = 0.10, 0.08,"),
        encoding="ascii")
    with pytest.raises(ValueError) as refusal:
        _validate_native_hrrr_physics_profile(drifted, WSM6_PROFILE_ID)
    message = str(refusal.value)
    assert "d01's value must be 0.08" in message
    assert ("multi-domain namelist passed to a single-domain root preparer"
            in message)
    assert "&dynamics/diff_6th_factor" in message


def test_native_hrrr_fixed_physics_profile_rejects_scheme_drift(tmp_path):
    path = tmp_path / "namelist.input"
    _write_native_physics_namelist(path, mp_physics=8)
    with pytest.raises(ValueError, match="mp_physics.*must be 6"):
        _validate_native_hrrr_physics_profile(path, WSM6_PROFILE_ID)


def test_native_hrrr_fixed_physics_profile_rejects_missing_control(tmp_path):
    path = tmp_path / "namelist.input"
    _write_native_physics_namelist(path, include_diff_factor=False)
    with pytest.raises(ValueError, match="requires.*diff_6th_factor"):
        _validate_native_hrrr_physics_profile(path, WSM6_PROFILE_ID)


@pytest.mark.parametrize(
    ("profile", "sfclay", "surface", "pbl", "soil_layers"),
    (
        (MYNN_PROFILE_ID, 5, 2, 5, 4),
        (RUC_PROFILE_ID, 91, 3, 1, 9),
        (MYNN_RUC_PROFILE_ID, 5, 3, 5, 9),
        (NOAHMP_PROFILE_ID, 91, 4, 1, 4),
        (MYNN_NOAHMP_PROFILE_ID, 5, 4, 5, 4),
    ),
)
def test_native_hrrr_new_front_door_families_prepare_exact_profile(
        tmp_path, profile, sfclay, surface, pbl, soil_layers):
    path = tmp_path / "namelist.input"
    _write_native_physics_namelist(
        path, sf_sfclay_physics=sfclay,
        sf_surface_physics=surface, bl_pbl_physics=pbl,
        num_soil_layers=soil_layers)
    acknowledgements = (
        ("noahmp-host-column-throughput-v1",)
        if profile in (NOAHMP_PROFILE_ID, MYNN_NOAHMP_PROFILE_ID) else ()
    )

    receipt = _validate_native_hrrr_physics_profile(
        path, profile, expert_acknowledgements=acknowledgements)

    assert receipt["profile"] == profile
    assert receipt["readiness"] == "IMPLEMENTED_UNVERIFIED"
    assert receipt["resolved"]["sf_sfclay_physics"] == sfclay
    assert receipt["resolved"]["sf_surface_physics"] == surface
    assert receipt["resolved"]["bl_pbl_physics"] == pbl
    assert receipt["resolved"]["num_soil_layers"] == soil_layers
    assert receipt["front_door_selection"]["profile"] == profile


@pytest.mark.parametrize("solar", [False, True], ids=["monthly", "monthly-solar"])
def test_native_hrrr_monthly_ruc_contract_reads_logicals_and_rejects_drift(
        tmp_path, pinned_thompson_tables, solar):
    profile = (hrrr_runner.THOMPSON_MYNN_RUC_MONTHLY_SOLAR_LEGACY_RRTMG_PROFILE_ID
               if solar else hrrr_runner.THOMPSON_MYNN_RUC_MONTHLY_LEGACY_RRTMG_PROFILE_ID)
    path = tmp_path / "monthly.input"
    _write_native_physics_namelist(
        path, mp_physics=8, sf_sfclay_physics=5,
        sf_surface_physics=3, bl_pbl_physics=5, num_soil_layers=9)
    text = path.read_text().replace(
        " ra_lw_physics = 0, 0,", " ra_lw_physics = 4, 4,").replace(
        " ra_sw_physics = 1, 1,", " ra_sw_physics = 4, 4,").replace(
        " radt = 1, 3,", " radt = 12, 12,").replace(
        "&physics\n", "&physics\n usemonalb = .true., .false.,\n"
        " rdlai2d = .true., .false.,\n fractional_seaice = 1, 0,\n")
    if solar:
        text = text.replace("&physics\n", "&physics\n alb_sol = 1, 0,\n")
    path.write_text(text, encoding="ascii")
    receipt = _validate_native_hrrr_physics_profile(path, profile)
    assert receipt["readiness"] == "IMPLEMENTED_UNVERIFIED"
    physics = receipt["validated_namelist"]["physics"]
    assert physics["usemonalb"] is physics["rdlai2d"] is True
    assert physics["fractional_seaice"] == 1
    assert receipt["resolved"]["ra_rrtmg_variant"] == "rrtmg_legacy"
    if solar:
        assert physics["alb_sol"] == receipt["resolved"]["alb_sol"] == 1
        path.write_text(text.replace("alb_sol = 1, 0,", "alb_sol = 0, 1,"), encoding="ascii")
        with pytest.raises(ValueError, match="alb_sol.*must be 1"):
            _validate_native_hrrr_physics_profile(path, profile)
    path.write_text(text.replace(
        "usemonalb = .true., .false.,", "usemonalb = .false., .true.,"),
        encoding="ascii")
    with pytest.raises(ValueError, match="usemonalb.*must be True"):
        _validate_native_hrrr_physics_profile(path, profile)


def test_native_hrrr_noahmp_warns_without_registry_acknowledgement(tmp_path, capsys):
    path = tmp_path / "namelist.input"
    _write_native_physics_namelist(
        path, sf_surface_physics=4, num_soil_layers=4)

    receipt = _validate_native_hrrr_physics_profile(path, NOAHMP_PROFILE_ID)
    assert receipt["profile"] == NOAHMP_PROFILE_ID
    assert "noahmp-host-column-throughput-v1" in capsys.readouterr().err


@pytest.fixture
def isolated_thompson_launch_tables(tmp_path, monkeypatch):
    """A complete table authority independent of installed or staged assets.

    These tests exercise the benchmark's additional launch contract after
    route-level staging. Tiny fixture assets still pass the real size and
    SHA-256 validator before the environment guard is reached.
    """
    import hashlib

    from woof import physics_compat
    from woof.core import thompson_contract

    root = tmp_path / "staged-tables"
    root.mkdir()
    data = b"launch guard fixture\n"
    assets = tuple(dataclasses.replace(
        asset, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        for asset in thompson_contract.CLASSIC_TABLE_ASSETS)
    for asset in assets:
        (root / asset.filename).write_bytes(data)
    validate_table_assets = thompson_contract.validate_table_assets
    monkeypatch.setattr(thompson_contract, "CLASSIC_TABLE_ASSETS", assets)
    monkeypatch.setattr(thompson_contract, "validate_table_assets",
                        lambda path: validate_table_assets(path, assets))
    monkeypatch.setattr(physics_compat, "packaged_thompson_table_root",
                        lambda: root)
    monkeypatch.setattr(physics_compat, "user_thompson_table_root",
                        lambda: root)
    monkeypatch.delenv(physics_compat.EXPERIMENTAL_THOMPSON_ENV, raising=False)
    monkeypatch.delenv(physics_compat.THOMPSON_TABLE_ROOT_ENV, raising=False)
    return root


def test_native_hrrr_thompson_profile_is_guarded_and_table_bound(
        tmp_path, monkeypatch, isolated_thompson_launch_tables):
    path = tmp_path / "namelist.input"
    _write_native_physics_namelist(path, mp_physics=8)
    with pytest.raises(RuntimeError, match="WOOF_EXPERIMENTAL_THOMPSON_MP8=1"):
        _validate_native_hrrr_physics_profile(path, THOMPSON_PROFILE_ID)

    monkeypatch.setenv("WOOF_EXPERIMENTAL_THOMPSON_MP8", "1")
    with pytest.raises(RuntimeError, match="WOOF_THOMPSON_TABLE_ROOT"):
        _validate_native_hrrr_physics_profile(path, THOMPSON_PROFILE_ID)

    table_root = tmp_path / "tables"
    monkeypatch.setenv("WOOF_THOMPSON_TABLE_ROOT", str(table_root))
    # 1.8: the route stages microphysics tables for EVERY mp8 profile,
    # before this profile's own environment launch contract, and staging
    # answers presence first (cheap) and bytes second.  A stand-in root
    # therefore has to exist on disk; its BYTES stay monkeypatched below,
    # which is what this test is actually about.
    table_root.mkdir()
    from woof.core.thompson_contract import CLASSIC_TABLE_ASSETS
    for pinned in CLASSIC_TABLE_ASSETS:
        (table_root / pinned.filename).write_bytes(b"")
    asset = SimpleNamespace(
        filename="freezeH2O.dat", bytes=123, sha256="a" * 64)
    monkeypatch.setattr(
        "woof.core.thompson_contract.validate_table_assets",
        lambda root: (asset,) if root == table_root.resolve() else ())
    receipt = _validate_native_hrrr_physics_profile(
        path, THOMPSON_PROFILE_ID)
    assert receipt["profile"] == THOMPSON_PROFILE_ID
    assert receipt["readiness"] == "WRF_MATCHED_RUN_EXPERIMENTAL"
    assert receipt["resolved"]["mp_physics"] == 8
    assert receipt["resolved"]["moist"] is True
    assert receipt["resolved"]["moist_cq"] is True
    assert receipt["resolved"]["top_lid"] is False
    assert receipt["thompson_contract"]["assets"] == [{
        "filename": "freezeH2O.dat", "bytes": 123, "sha256": "a" * 64}]


def test_the_guarded_mp8_refusal_names_both_variables_with_values(
        tmp_path, monkeypatch, isolated_thompson_launch_tables):
    """One refusal carries the whole launch contract, not half of it.

    A field run of the shipped 1.5.0 wheel set the first variable, was
    refused for the second, and then had to work out the table root by
    hand -- having already downloaded it.  The guard itself is unchanged:
    both variables are still required and the root is still byte-checked.
    """
    from woof.physics_compat import thompson_guard_exports

    path = tmp_path / "namelist.input"
    _write_native_physics_namelist(path, mp_physics=8)
    monkeypatch.delenv("WOOF_EXPERIMENTAL_THOMPSON_MP8", raising=False)
    monkeypatch.delenv("WOOF_THOMPSON_TABLE_ROOT", raising=False)

    for preset in ({}, {"WOOF_EXPERIMENTAL_THOMPSON_MP8": "1"}):
        for name, value in preset.items():
            monkeypatch.setenv(name, value)
        with pytest.raises(RuntimeError) as raised:
            _validate_native_hrrr_physics_profile(path, THOMPSON_PROFILE_ID)
        message = str(raised.value)
        assert "WOOF_EXPERIMENTAL_THOMPSON_MP8" in message
        assert "WOOF_THOMPSON_TABLE_ROOT" in message
        # Pasteable, and platform-correct: an `export` line in PowerShell
        # is a syntax error and the reader then debugs the instructions.
        for line in thompson_guard_exports():
            assert line in message


@pytest.mark.parametrize(
    ("profile", "selector"),
    ((MORRISON_PROFILE_ID, 10), (NSSL2_PROFILE_ID, 18)),
)
def test_native_hrrr_real74_suite_profiles_bind_exact_namelist_and_runtime(
        tmp_path, profile, selector):
    path = tmp_path / "namelist.input"
    _write_native_physics_namelist(
        path, mp_physics=selector, full_real74_suite=True)

    receipt = _validate_native_hrrr_physics_profile(path, profile)

    assert receipt["profile"] == profile
    assert receipt["resolved"] == {
        "moist": True,
        "moist_cq": True,
        "mp_physics": selector,
        "top_lid": False,
        "epssm": 0.5,
        "morr_rimed_ice": 1,
        "wsm6_hail_opt": 0,
        "ra_physics": 0,
        "ra_lw_physics": 4,
        "ra_sw_physics": 4,
            "radt": 12.0,
            "wrf_rrtmg_compatibility": WRF_RRTMG_TO_RTE_RRTMGP,
            "ra_rrtmg_variant": "rte-rrtmgp",
        "sf_sfclay_physics": 91,
        "sf_surface_physics": 2,
        "bl_pbl_physics": 1,
        "cu_physics": 1,
        "cudt_minutes": 5.0,
        "num_soil_layers": 4,
        "terrain_opt": 1,
        "km_opt": 4,
        "diff_6th_opt": 2,
        "diff_6th_factor": 0.12,
        "diff_6th_slopeopt": 1,
        # Every template states the urban component since the urban canopy
        # models joined the registry; the native run is urban-off.
        "sf_urban_physics": 0,
        "radiation_scheme_ids": [4, 4],
    }
    assert receipt["radiation_substitution"]["contract"] \
        == WRF_RRTMG_TO_RTE_RRTMGP
    if profile == MORRISON_PROFILE_ID:
        assert receipt["readiness"] == "WRF_MATCHED_RUN_RUNTIME_PROFILE"
        assert receipt["morrison_contract"]["morr_rimed_ice"] == 1
    else:
        assert receipt["readiness"] == "WRF_MATCHED_RUN_CANDIDATE"
        assert receipt["nssl2_contract"]["selector"] == 18
        assert receipt["nssl2_contract"]["is_default_lane"] is True
        assert receipt["nssl2_contract"]["resolved_mode"] == {
            "two_moment": True,
            "hail": True,
            "predicted_ccn": True,
            "density_moments": 2,
            "sixth_moments": 0,
        }


def test_native_hrrr_morrison_profile_rejects_rimed_ice_drift(tmp_path):
    path = tmp_path / "namelist.input"
    _write_native_physics_namelist(
        path, mp_physics=10, full_real74_suite=True, morr_rimed_ice=0)
    with pytest.raises(ValueError, match="morr_rimed_ice.*must be 1"):
        _validate_native_hrrr_physics_profile(path, MORRISON_PROFILE_ID)


def test_native_hrrr_nssl2_legacy_profile_binds_exact_solver_identity(
        tmp_path):
    path = tmp_path / "namelist.input"
    _write_native_physics_namelist(
        path, mp_physics=18, full_real74_suite=True)

    receipt = _validate_native_hrrr_physics_profile(
        path, NSSL2_LEGACY_RRTMG_PROFILE_ID)

    assert receipt["readiness"] == "WRF_MATCHED_RUN_CANDIDATE"
    assert receipt["resolved"]["wrf_rrtmg_compatibility"] == WRF_RRTMG_LEGACY
    assert receipt["resolved"]["ra_rrtmg_variant"] == "rrtmg_legacy"
    assert receipt["radiation_identity"] == {
        "contract": WRF_RRTMG_LEGACY,
        "requested_wrf_scheme_ids": [4, 4],
        "resolved_gpuwm_scheme_ids": [4, 4],
        "resolved_gpuwm_solver": "legacy RRTMG",
    }
    assert "radiation_substitution" not in receipt


def _profile_experiment(profile: str, **kwargs):
    """One HRRR experiment per profile, on a grid every profile accepts.

    nz=12 clears the Kain-Fritsch (>=8) and MYNN (>=5) vertical floors, so
    a refusal below is about the switch forward and nothing else.

    ``kwargs`` reaches :func:`_experiment` untouched, which is how the
    window tests below choose a start time: the default one is
    all-daylight over this target and the declaration rule differs
    between the two windows.
    """

    nz = 12
    vertical = VerticalConfig(
        eta_levels=tuple(float(value)
                         for value in np.linspace(1.0, 0.0, nz + 1)),
        p_top=5000.0, hybrid_opt=2, etac=0.2)
    target = dataclasses.replace(HrrrTargetDomain.legacy_500x500(), nz=nz)
    return _experiment(
        vertical, run_seconds=3600.0, target=target,
        physics_profile=profile, **kwargs)


#: The profiles THIS runner offers, which is its own route's declared
#: template list.  It used to be the menu shared with the other
#: fixed-template route; when that route declared six composition suites
#: this one cannot replay, every walk in this file demanded a native
#: namelist contract, a switch home and an initialization contract for
#: suites this runner never offers.  The walks below are about THIS
#: runner's per-profile tables, so they enumerate what THIS route
#: declares -- and the two facts are held equal by
#: ``test_the_route_declaration_is_the_namelist_contract_table``.
_ROUTE_PHYSICS_PROFILES = route_physics_profiles(
    "tools.hrrr_single_domain_benchmark")


#: Every shipped profile whose downward longwave is fabricated: no
#: longwave scheme, and a land-surface scheme that reads GLW every step.
#: Derived from the product's own classifier rather than listed by hand,
#: so a ninth profile joining the class arrives as coverage rather than
#: as a field report.
_FABRICATED_GLW_PROFILES = sorted(
    profile for profile in _ROUTE_PHYSICS_PROFILES
    if downward_longwave_disposition(
        ra_lw_physics=int(single_domain_runtime_switches(profile).get(
            "ra_lw_physics", 0)),
        ra_sw_physics=int(single_domain_runtime_switches(profile).get(
            "ra_sw_physics", 0)),
        sf_surface_physics=int(single_domain_runtime_switches(profile).get(
            "sf_surface_physics", 0)))[0] in ("consumed", "published"))

#: 18Z over the route's reference point (-98 E) is early afternoon: an
#: hour from here reaches no local night at all.  03Z is the small hours.
_ALL_DAYLIGHT_START = datetime(2026, 7, 18, 18)
_NOCTURNAL_START = datetime(2026, 7, 18, 3)


@pytest.mark.parametrize("profile", _FABRICATED_GLW_PROFILES)
def test_an_all_daylight_window_still_declares_the_fabricated_longwave(
        profile):
    """Daylight does not make a frozen 300 W m-2 into a computed flux.

    The 1.8.8 assembly wrote BOTH radiation tokens below the night
    check, so every one of these eight profiles refused at config BUILD
    on an all-daylight window -- and this route synthesizes its config in
    code, so there was no file for an operator to write the declaration
    into.  Thirteen nodes of this file were red on it.

    Two claims, two conditions: the constant-longwave token rides the
    GLW disposition (which the sun has nothing to do with), the
    nocturnal token rides the window.  Asserting the nocturnal token is
    ABSENT here is the half that keeps the fix from degenerating into
    "declare everything always", which would make a published
    declaration mean nothing.
    """

    from woof.physics_compat import (ASYMMETRIC_RADIATION_NOCTURNAL_ACK,
                                      CONSTANT_DOWNWARD_LONGWAVE_ACK)

    experiment = _profile_experiment(profile,
                                     start_time=_ALL_DAYLIGHT_START)

    declared = tuple(experiment.acknowledgements or ())
    assert CONSTANT_DOWNWARD_LONGWAVE_ACK in declared, (
        f"{profile} fabricates its downward longwave on any window; the "
        f"declaration cannot be conditional on nightfall")
    assert ASYMMETRIC_RADIATION_NOCTURNAL_ACK not in declared, (
        f"{profile} on an all-daylight window has no nocturnal claim to "
        f"make, and a token written by reflex declares nothing")


@pytest.mark.parametrize("profile", _FABRICATED_GLW_PROFILES)
def test_a_nocturnal_window_declares_both_radiation_claims(profile):
    """The night window carries both tokens, in the order it always did."""

    from woof.physics_compat import (ASYMMETRIC_RADIATION_NOCTURNAL_ACK,
                                      CONSTANT_DOWNWARD_LONGWAVE_ACK)

    experiment = _profile_experiment(profile, start_time=_NOCTURNAL_START)

    assert tuple(experiment.acknowledgements or ()) == (
        ASYMMETRIC_RADIATION_NOCTURNAL_ACK, CONSTANT_DOWNWARD_LONGWAVE_ACK)


def test_a_symmetric_profile_declares_nothing_on_either_window():
    """The control: the route's own default carries no token at all.

    Without this the two tests above pass just as well on a route that
    declares both tokens unconditionally, which is the failure mode the
    1.8 default change was made to end.
    """

    for start_time in (_ALL_DAYLIGHT_START, _NOCTURNAL_START):
        experiment = _profile_experiment(ROUTE_DEFAULT_PHYSICS_PROFILE,
                                         start_time=start_time)
        assert tuple(experiment.acknowledgements or ()) == ()


@pytest.mark.parametrize("profile", sorted(_ROUTE_PHYSICS_PROFILES))
def test_the_hrrr_experiment_forwards_every_switch_its_profile_declares(
        profile):
    """The forwarded set covers the profile's whole declared switch set.

    F1: ``_experiment`` copied ``wrf_rrtmg_compatibility`` and dropped
    ``ra_rrtmg_variant``, so the ratified
    ``nssl2-...-rrtmg-legacy-...-v1`` profile -- declared for the HRRR
    route and offered in this runner's ``--physics-profile`` choices --
    could not be built at all: the RunConfig kept the ``"rte-rrtmgp"``
    default and the paired-switch validator refused it in 0.7 s.  Its RTE
    sibling built only because its declared value equalled that default,
    which is exactly why a single-profile test would not have caught this.
    Parametrizing over every shipped profile kills the class.
    """

    switches = single_domain_runtime_switches(profile)
    run = _profile_experiment(profile).root.run

    missing = sorted(set(switches) - set(hrrr_runner._PROFILE_SWITCH_HOMES))
    assert missing == []
    observed = {name: getattr(run, name, "<absent from RunConfig>")
                for name in switches}
    assert observed == switches



@pytest.mark.parametrize("profile", sorted(_ROUTE_PHYSICS_PROFILES))
def test_monthly_surface_rows_follow_every_route_templates_land_scheme(profile):
    from woof.hrrr_route_inputs import render_namelist_input
    from woof.namelist_import import parse_namelist_text

    switches = single_domain_runtime_switches(profile)
    supported = switches["sf_surface_physics"] in (2, 3)
    exp = _profile_experiment(profile)
    target = dataclasses.replace(HrrrTargetDomain.legacy_500x500(),
                                 nz=exp.root.run.nz)
    raw, _ = hrrr_runner._experiment_tables(
        exp.vertical, run_seconds=3600.0, target=target,
        physics_profile=profile)
    physics = parse_namelist_text(render_namelist_input(exp))["physics"]
    for key in ("rdlai2d", "usemonalb"):
        assert (key in raw["shared"]) is supported, (profile, key)
        assert getattr(exp.root.run, key) is supported, (profile, key)
        assert (key in physics) is supported, (profile, key)
        if supported:
            assert raw["shared"][key] is True
            assert physics[key] == [True]
    if profile == ROUTE_DEFAULT_PHYSICS_PROFILE:
        domains = parse_namelist_text(render_namelist_input(exp))["domains"]
        assert (exp.root.run.aer_init_opt, exp.root.run.wif_input_opt) == (1, 1)
        assert physics["use_aero_icbc"] == [True]
        assert domains["wif_input_opt"] == [1]


def test_the_hrrr_switch_forward_refuses_a_profile_switch_it_has_no_home_for(
):
    """A profile that grows a switch fails loudly, not silently."""

    raw = {"shared": {}, "domain": [{}]}
    with pytest.raises(ValueError, match="no declared home"):
        hrrr_runner._forward_profile_switches(
            raw, {"ra_rrtmg_variant": "rrtmg_legacy",
                  "some_future_switch": 1})


def test_the_hrrr_route_builds_the_ratified_legacy_rrtmg_profile():
    """The exact instance F1 reported, and its RTE sibling beside it."""

    legacy = _profile_experiment(NSSL2_LEGACY_RRTMG_PROFILE_ID).root.run
    rte = _profile_experiment(NSSL2_PROFILE_ID).root.run

    assert legacy.wrf_rrtmg_compatibility == WRF_RRTMG_LEGACY
    assert legacy.ra_rrtmg_variant == "rrtmg_legacy"
    assert rte.wrf_rrtmg_compatibility == WRF_RRTMG_TO_RTE_RRTMGP
    assert rte.ra_rrtmg_variant == "rte-rrtmgp"


def test_native_hrrr_nssl2_profile_rejects_surrounding_suite_drift(tmp_path):
    path = tmp_path / "namelist.input"
    _write_native_physics_namelist(
        path, mp_physics=18, full_real74_suite=True, cu_physics=0)
    with pytest.raises(ValueError, match="cu_physics.*must be 1"):
        _validate_native_hrrr_physics_profile(path, NSSL2_PROFILE_ID)


def test_native_hrrr_nssl2_accepts_equivalent_explicit_optional_selectors(
        tmp_path):
    path = tmp_path / "namelist.input"
    controls = """ nssl_2moment_on = 1, -1,
 nssl_hail_on = 1, -1,
 nssl_ccn_on = 1, -1,
 nssl_density_on = 2, -1,
 nssl_3moment = 0, 0,
"""
    _write_native_physics_namelist(
        path, mp_physics=18, full_real74_suite=True,
        nssl_controls=controls)

    receipt = _validate_native_hrrr_physics_profile(
        path, NSSL2_PROFILE_ID)

    assert receipt["validated_namelist"]["physics"]["nssl_hail_on"] == 1
    assert receipt["validated_namelist"]["physics"][
        "nssl_density_on"] == 2


@pytest.mark.parametrize("controls,match", (
    (" nssl_hail_on = 0, -1,\n", "fixed preset does not implement"),
    (" nssl_rho_qhl = 800, 900,\n", "nssl_rho_qhl=900.0"),
))
def test_native_hrrr_nssl2_rejects_silently_substituted_optional_controls(
        tmp_path, controls, match):
    path = tmp_path / "namelist.input"
    _write_native_physics_namelist(
        path, mp_physics=18, full_real74_suite=True,
        nssl_controls=controls)

    with pytest.raises(ValueError, match=match):
        _validate_native_hrrr_physics_profile(path, NSSL2_PROFILE_ID)


@pytest.mark.parametrize(
    ("profile", "selector"),
    ((MORRISON_PROFILE_ID, 10), (NSSL2_PROFILE_ID, 18)),
)
def test_native_hrrr_real74_suite_experiment_switches_are_exact(
        profile, selector):
    vertical = VerticalConfig(
        eta_levels=tuple(float(value) for value in np.linspace(1.0, 0.0, 50)),
        p_top=5000.0,
        hybrid_opt=2,
        etac=0.2,
    )
    cfg = _experiment(
        vertical, run_seconds=3600.0, physics_profile=profile).root.run

    assert cfg.mp_physics == selector
    assert cfg.moist is True
    assert cfg.moist_cq is True
    assert cfg.top_lid is False
    assert cfg.epssm == 0.5
    assert cfg.morr_rimed_ice == 1
    assert cfg.sf_sfclay_physics == 91
    assert cfg.sf_surface_physics == 2
    assert cfg.bl_pbl_physics == 1
    assert cfg.cu_physics == 1
    assert cfg.cudt_minutes == 5.0
    assert cfg.ra_physics == 0
    assert (cfg.ra_lw_physics, cfg.ra_sw_physics) == (4, 4)
    assert cfg.wrf_rrtmg_compatibility == WRF_RRTMG_TO_RTE_RRTMGP
    assert cfg.radt == 12.0
    assert cfg.diff_6th_factor == 0.12


@pytest.mark.parametrize("profile", (
    WSM6_PROFILE_ID,
    THOMPSON_PROFILE_ID,
    MORRISON_PROFILE_ID,
    NSSL2_PROFILE_ID,
))
@pytest.mark.parametrize("hybrid_opt", (1, 2))
def test_native_hrrr_profile_preserves_explicit_vertical_hybrid_option(
        profile, hybrid_opt, monkeypatch):
    monkeypatch.setenv("WOOF_EXPERIMENTAL_THOMPSON_MP8", "1")
    monkeypatch.setenv("WOOF_THOMPSON_TABLE_ROOT", "config-only-fixture")
    vertical = VerticalConfig(
        eta_levels=tuple(float(value) for value in np.linspace(1.0, 0.0, 50)),
        p_top=5000.0,
        hybrid_opt=hybrid_opt,
        etac=0.2,
    )
    exp = _experiment(
        vertical, run_seconds=3600.0, physics_profile=profile)
    receipt = {
        "resolved": hrrr_runner._native_hrrr_runtime_switches(profile),
    }

    hrrr_runner._validate_resolved_hrrr_profile(exp, receipt)

    assert exp.root.run.hybrid_opt == hybrid_opt


class _ReferenceVerticalPlan:
    def __init__(self, source, surface, target):
        self.source = np.asarray(source, dtype=np.float32)
        self.surface = np.asarray(surface, dtype=np.float32)
        self.target = np.asarray(target, dtype=np.float32)

    def apply(self, field, surface_value, **options):
        from woof.verify.npref import np_wrf_real_vert_interp

        options.pop("values_are_finite", None)
        return np.asarray(np_wrf_real_vert_interp(
            field, surface_value, self.source, self.surface, self.target,
            **options), dtype=np.float32)


class _ReferencePreprocessBackend:
    """Independent CPU implementation of the production preprocessing ABI."""

    name = "cpu-reference-test"
    array_module = np

    @staticmethod
    def float32(value):
        return np.asarray(value, dtype=np.float32)

    @staticmethod
    def regular_plan(*args, **kwargs):
        raise AssertionError("regular horizontal preprocessing is unused here")

    masked_nearest = regular_plan
    rotate_earth_to_grid = regular_plan
    era5_rh_to_water = regular_plan

    @staticmethod
    def prepare_wrf_vertical(source, surface, target):
        return _ReferenceVerticalPlan(source, surface, target)

    @staticmethod
    def receipt():
        return {"backend": "cpu-reference-test"}


def _decoded_native_hrrr_initialization(
        mp_physics, *, analyzed_species=None,
        analyzed_source_levels=None, landmask=None,
        mp28_aerosol_source=None):
    """Drive decoded native fields through real initialization, never a fake state.

    ``analyzed_species`` selects which of QC/QR/QI/QS/QG carry mass in the
    decoded source.  It defaults to all five; passing a subset (or none)
    reproduces the meteorologically empty analysis the reopening battery hit,
    where four of five species had nothing to retain and the retention gate
    could not fire.
    """

    from woof.config import RunConfig
    from woof.core.grid import make_vertical_coord
    from woof.ingest.horiz import HorizontalSnapshot
    from woof.ingest.real import initialize_real

    ny, nx, nz = 2, 3, 8
    levels = np.array(
        [100.0, 300.0, 500.0, 700.0, 850.0, 1000.0],
        dtype=np.float64)
    pressure = np.broadcast_to(
        levels[:, None, None] * 100.0, (levels.size, ny, nx)).copy()
    temperature = np.broadcast_to(
        215.0 + 75.0 * (pressure / 100000.0) ** 0.22,
        pressure.shape).copy()
    height = np.broadcast_to(
        -7900.0 * np.log(pressure / 100000.0), pressure.shape).copy()
    level_index = np.arange(levels.size, dtype=np.float32)[:, None, None]
    row_index = np.arange(ny, dtype=np.float32)[None, :, None]
    column_index = np.arange(nx, dtype=np.float32)[None, None, :]
    carried = (("QC", "QR", "QI", "QS", "QG") if analyzed_species is None
               else tuple(analyzed_species))
    if analyzed_source_levels is not None:
        carried = tuple(analyzed_source_levels)
    analyzed = {}
    for species_index, name in enumerate(("QC", "QR", "QI", "QS", "QG"), 1):
        if name not in carried:
            analyzed[name] = np.zeros(pressure.shape, dtype=np.float32)
            continue
        if analyzed_source_levels is not None:
            value = np.zeros(pressure.shape, dtype=np.float32)
            value[int(analyzed_source_levels[name])] = np.float32(
                species_index * 1.0e-6)
            analyzed[name] = value
            continue
        value = np.asarray(
            species_index * 1.0e-6
            * (1.0 + level_index + 0.25 * row_index
               + 0.125 * column_index),
            dtype=np.float32)
        value[0, 0, 0] = np.float32(0.0)
        analyzed[name] = value
    fields = {
        "PRES": pressure.astype(np.float32),
        "SPFH": np.full(pressure.shape, 0.004, dtype=np.float32),
        "TT": temperature.astype(np.float32),
        "GHT": height.astype(np.float32),
        "UU": np.full((levels.size, ny, nx + 1), 8.0, dtype=np.float32),
        "VV": np.full((levels.size, ny + 1, nx), -2.0, dtype=np.float32),
        "PSFC": np.full((ny, nx), 100000.0, dtype=np.float32),
        "T2": np.full((ny, nx), 289.0, dtype=np.float32),
        "Q2": np.full((ny, nx), 0.004, dtype=np.float32),
        "U10": np.full((ny, nx + 1), 7.0, dtype=np.float32),
        "V10": np.full((ny + 1, nx), -1.0, dtype=np.float32),
        **analyzed,
    }
    snapshot = HorizontalSnapshot(
        valid_time=datetime(2026, 7, 20, 6),
        levels_hpa=levels,
        fields=fields,
    )
    extra = {} if mp28_aerosol_source is None else {
        "mp28_aerosol_source": mp28_aerosol_source}
    cfg = RunConfig(
        nx=nx, ny=ny, nz=nz, dx=12000.0, dy=12000.0,
        ztop=18000.0, dt=30.0, run_seconds=60.0,
        hybrid_opt=2, etac=0.2, moist=True, terrain_opt=1,
        mp_physics=mp_physics,
        **extra,
    )
    eta = np.linspace(1.0, 0.0, nz + 1)
    terrain = np.zeros((ny, nx), dtype=np.float64)
    return initialize_real(
        snapshot, cfg,
        make_vertical_coord(
            nz, hybrid_opt=2, etac=0.2, eta_levels=eta),
        terrain,
        source_orography=terrain,
        landmask=landmask,
        p_top=10000.0,
        use_sh_qv=True,
        preprocess_backend=_ReferencePreprocessBackend(),
        state_backend="cpu",
    )


@pytest.mark.parametrize(
    ("mp_physics", "retained", "discarded"),
    (
        (6, {"QC", "QR", "QI", "QS", "QG"}, set()),
        (18, {"QC", "QR", "QI", "QS", "QG"}, set()),
        (1, {"QC", "QR"}, {"QI", "QS", "QG"}),
        # P3 one-category (Registry.EM_COMMON:3038, moist:qv,qc,qr,qi) is the
        # second narrow package on this route and the only one narrow at the
        # frozen end: it keeps analyzed ice and cannot hold snow or graupel,
        # so it is the case that proves the partition is read from the
        # scheme's package and not from "Kessler or everything".
        (50, {"QC", "QR", "QI"}, {"QS", "QG"}),
        # Milbrandt-Yau (Registry.EM_COMMON:3025, moist:qv,qc,qr,qi,qs,qg,qh)
        # is the first package WIDER than the decoded inventory: everything
        # decoded is retained, nothing is discarded, and the seventh member
        # (hail) has no decoded source at all -- it stays at the allocator's
        # zero, which is what real.exe's flag_qh-gated arm
        # (module_initialize_real.F:1979-1997) leaves for a QH-less source.
        (9, {"QC", "QR", "QI", "QS", "QG"}, set()),
    ),
)
def test_native_hrrr_hydrometeor_admission_is_scheme_aware(
        mp_physics, retained, discarded):
    result = _decoded_native_hrrr_initialization(mp_physics)
    evidence = result.hydrometeor_initialization
    assert set(evidence["retained_correspondence"]) == retained
    assert set(evidence["discarded_source_species"]) == discarded
    assert set(evidence["decoded_source_species"]) == {
        "QC", "QR", "QI", "QS", "QG"}
    assert all(
        evidence["initialized_state_species"][state_name][
            "nonzero_count"] > 0
        for state_name in evidence["retained_correspondence"].values()
    )
    if mp_physics == 1:
        assert getattr(result.state, "qi", None) is None
    if mp_physics == 50:
        assert getattr(result.state, "qs", None) is None
        assert getattr(result.state, "qg", None) is None
    for policy in evidence["discarded_source_species"].values():
        assert policy["policy"] == (
            "discard-source-species-absent-from-active-moist-package")
        assert policy["wrf_commit"] == (
            "d66e442fccc04111067e29274c9f9eaccc3cef28")


def test_the_retention_receipt_states_when_it_proved_nothing():
    """The battery's vacuous-receipt lesson, bound to the producer.

    The first NSSL/RTE probe passed this receipt on a domain where four of
    five analyzed species were identically zero: the gate is ``source
    nonzero > 0 and live nonzero == 0 -> raise``, so it could not fire, and
    the receipt did not say so.  A green receipt on that domain was not
    evidence that B-H1 was fixed -- it would have passed with the fix
    reverted for QR/QI/QS/QG.  Now the receipt grades its own evidence.
    """

    # A cloud-free analysis: nothing to retain for any species.
    empty = _decoded_native_hrrr_initialization(18, analyzed_species=())
    receipt = _initial_hrrr_microphysics_receipt(
        empty.state, NSSL2_PROFILE_ID, empty.hydrometeor_initialization)
    assert receipt["retention_evidence_summary"]["strength"] == "VACUOUS"
    assert receipt["retention_evidence_summary"]["vacuous_species"] == [
        "QC", "QG", "QI", "QR", "QS"]
    assert receipt["retention_evidence_summary"]["proven_species"] == []
    assert "proves nothing" in \
        receipt["retention_evidence_summary"]["statement"]
    for species in ("QC", "QG", "QI", "QR", "QS"):
        evidence = receipt["retention_evidence"][species]
        assert evidence["strength"] == "VACUOUS"
        assert evidence["source_nonzero_count"] == 0
        assert "not proven" in evidence["reason"]

    # The battery's actual Ohio domain: QC alone carried mass.
    partial = _decoded_native_hrrr_initialization(
        18, analyzed_species=("QC",))
    receipt = _initial_hrrr_microphysics_receipt(
        partial.state, NSSL2_PROFILE_ID, partial.hydrometeor_initialization)
    summary = receipt["retention_evidence_summary"]
    assert summary["strength"] == "PARTIALLY_WRF_EXCLUDED"
    assert summary["proven_species"] == []
    assert summary["partially_excluded_species"] == ["QC"]
    assert summary["vacuous_species"] == ["QG", "QI", "QR", "QS"]
    assert receipt["retention_evidence"]["QC"]["strength"] == (
        "PARTIALLY_WRF_EXCLUDED")

    # The control: all five species carry target-influencing mass, while the
    # surface-close source row is explicitly excluded by WRF for each.
    full = _decoded_native_hrrr_initialization(18)
    receipt = _initial_hrrr_microphysics_receipt(
        full.state, NSSL2_PROFILE_ID, full.hydrometeor_initialization)
    assert receipt["retention_evidence_summary"]["strength"] == (
        "PARTIALLY_WRF_EXCLUDED")
    assert receipt["retention_evidence_summary"]["vacuous_species"] == []
    assert all(
        evidence["strength"] == "PARTIALLY_WRF_EXCLUDED"
        and evidence["source_nonzero_count"] > 0
        and evidence["state_nonzero_count"] > 0
        for evidence in receipt["retention_evidence"].values())


def test_sparse_wrf_exclusion_is_accepted_only_with_complete_v2_evidence():
    result = _decoded_native_hrrr_initialization(
        18, analyzed_source_levels={"QC": 5})

    receipt = _initial_hrrr_microphysics_receipt(
        result.state, NSSL2_PROFILE_ID, result.hydrometeor_initialization)

    qc = receipt["retention_evidence"]["QC"]
    assert qc["strength"] == "WRF_EXCLUDED"
    assert qc["source_nonzero_count"] == 6
    assert qc["target_influencing_source_count"] == 0
    assert qc["wrf_excluded_source_count"] == 6
    assert qc["state_nonzero_count"] == 0
    assert receipt["retention_evidence_summary"]["excluded_species"] == [
        "QC"]

    missing = copy.deepcopy(result.hydrometeor_initialization)
    missing.pop("vertical_disposition")
    with pytest.raises(ValueError, match="disposition evidence is missing"):
        _initial_hrrr_microphysics_receipt(
            result.state, NSSL2_PROFILE_ID, missing)

    # Schema v1 retains the historical broad predicate by design.  The same
    # sparse source/state pair is refused when the exhaustive v2 partition is
    # absent; old receipts are not silently reinterpreted under new rules.
    legacy = copy.deepcopy(result.hydrometeor_initialization)
    legacy["schema"] = "gpuwm-real-hydrometeor-correspondence-v1"
    legacy.pop("vertical_disposition")
    with pytest.raises(ValueError, match="lost all nonzero mass"):
        _initial_hrrr_microphysics_receipt(
            result.state, NSSL2_PROFILE_ID, legacy)


def test_native_hrrr_mp_off_refuses_unfaithful_analyzed_cloud_state():
    with pytest.raises(
            ValueError,
            match="cannot faithfully retain analyzed QC/QR/QI/QS/QG"):
        _decoded_native_hrrr_initialization(0)


def test_hrrr_thompson_initialization_holds_numbers_to_the_real_exe_seed():
    """nr and ni follow real.exe's cold start, not "exact zero everywhere".

    A99 ported make_RainNumber and make_IceNumber
    (module_initialize_real.F:4840-4852), so an analysed rain or ice cell
    starts with a number; the old exact-zero check then refused every
    native Thompson preparation with rain or ice in it (A110).  The rule
    the receipt now holds: exact FP32 zero where the paired mass is zero,
    finite and above zero where it is above zero, and exactly the seeded
    cell count the cold-start closure's seed receipt records.
    """
    result = _decoded_native_hrrr_initialization(8)
    state = result.state
    initialization = result.hydrometeor_initialization
    closure = initialization["cold_start_moment_closure"]
    receipt = _initial_hrrr_microphysics_receipt(
        state, THOMPSON_PROFILE_ID, initialization)
    for number, mass, seed_key in (("nr", "qr", "rain_number_seed"),
                                   ("ni", "qi", "ice_number_seed")):
        mass_value = getattr(state, mass)
        number_value = getattr(state, number)
        seeded = int(np.count_nonzero(mass_value > 0))
        # The fixture carries analysed rain and ice: the seeded half of
        # the rule is exercised (the massless half below).
        assert seeded > 0
        assert closure[seed_key]["seeded_cells"] == seeded
        assert np.all(number_value[mass_value == 0] == 0)
        assert np.all(number_value[mass_value > 0] > 0)
        assert receipt["state_number_fields"][number] == {
            "all_exact_zero_where_mass_is_zero": True,
            "paired_mass_field": mass,
            "seed_receipt": seed_key,
            "seeded_cells": seeded,
        }
        assert receipt["state_source_absent_fields"][number] == {
            "expected_float32": 0.0,
            "expected_uint32_bits": 0,
            "all_exact_expected_where_mass_is_zero": True,
            "paired_mass_field": mass,
            "seed_receipt": seed_key,
            "seeded_cells": seeded,
            "seeded_all_finite_above_zero": True,
        }
    assert set(receipt["state_number_fields"]) == {"nr", "ni"}
    assert "make_RainNumber" in receipt["source_absent_number_policy"]

    def refused(match, *, field=None, index=None, value=None,
                mutate_receipt=None, run=result):
        broken = copy.deepcopy(run.hydrometeor_initialization)
        if mutate_receipt is not None:
            mutate_receipt(broken)
        target = None if field is None else getattr(run.state, field)
        kept = None if target is None else target[index].copy()
        if target is not None:
            target[index] = np.float32(value)
        try:
            with pytest.raises(ValueError, match=match):
                _initial_hrrr_microphysics_receipt(
                    run.state, THOMPSON_PROFILE_ID, broken)
        finally:
            if target is not None:
                target[index] = kept

    def first(mask):
        return tuple(int(i) for i in np.argwhere(mask)[0])

    rainy = first(state.qr > 0)
    icy = first(state.qi > 0)

    # Mass with no number: exactly the cell real.exe fills.
    refused("number moment ni must be finite and above zero where qi",
            field="ni", index=icy, value=0.0)
    for bad in (np.nan, np.inf, -1.0):
        refused("number moment nr must be finite and above zero where qr",
                field="nr", index=rainy, value=bad)

    # The state and the closure's seed receipt must agree cell for cell.
    def miscounted(i):
        i["cold_start_moment_closure"]["rain_number_seed"][
            "seeded_cells"] += 1
    refused("closure's rain_number_seed records", mutate_receipt=miscounted)

    def no_closure(i):
        del i["cold_start_moment_closure"]
    refused("closure's (rain|ice)_number_seed records None",
            mutate_receipt=no_closure)

    def other_scheme(i):
        i["cold_start_moment_closure"]["mp_physics"] = 28
    refused("cold-start closure", mutate_receipt=other_scheme)

    # The untouched state still passes after every restore.
    assert _initial_hrrr_microphysics_receipt(
        state, THOMPSON_PROFILE_ID, initialization) == receipt

    # An analysis with rain and no ice: ni is the massless half of the
    # rule, exact zero everywhere, and a number there is refused.
    rain_only = _decoded_native_hrrr_initialization(
        8, analyzed_species=("QC", "QR"))
    assert not np.any(rain_only.state.qi > 0)
    dry = _initial_hrrr_microphysics_receipt(
        rain_only.state, THOMPSON_PROFILE_ID,
        rain_only.hydrometeor_initialization)
    assert dry["state_number_fields"]["ni"]["seeded_cells"] == 0
    assert np.all(rain_only.state.ni == 0)
    refused("number moment ni must be exact FP32 0.0 where qi is zero",
            field="ni", index=(0, 0, 0), value=1.0, run=rain_only)


def test_hrrr_morrison_initialization_receipts_every_source_absent_moment():
    result = _decoded_native_hrrr_initialization(10)
    state = result.state

    receipt = _initial_hrrr_microphysics_receipt(
        state, MORRISON_PROFILE_ID, result.hydrometeor_initialization)

    assert receipt["source_absent_wrf_fields"] == [
        "QNRAIN", "QNICE", "QNSNOW", "QNGRAUPEL"]
    from woof.ingest.handover_numbers import closure_active
    if not closure_active(None):
        # Exact mode (and a strict process): real.exe's zeros.
        assert receipt["state_number_fields"] == {
            name: {"all_exact_zero": True}
            for name in ("nc", "nr", "ni", "ns", "ng")
        }
        assert all(
            item["expected_uint32_bits"] == 0
            and item["all_exact_expected"] is True
            for item in receipt["state_source_absent_fields"].values())
        state.ng.flat[0] = np.float32(1.0)
        with pytest.raises(ValueError, match="source-absent state ng"):
            _initial_hrrr_microphysics_receipt(
                state, MORRISON_PROFILE_ID,
                result.hydrometeor_initialization)
        return
    # Correct-mode HN1 (2.9.7): nc starts at Morrison's NDCNST droplet number,
    # and each other number is exact zero where its mass is zero and closed
    # from the mass where it is above zero, as the closure receipt records.
    numbers = receipt["state_number_fields"]
    assert numbers["nc"] == {"ndcnst_droplet_number": True}
    for name, mass in (("nr", "qr"), ("ni", "qi"), ("ns", "qs"), ("ng", "qg")):
        assert numbers[name]["all_exact_zero_where_mass_is_zero"] is True
        assert numbers[name]["paired_mass_field"] == mass
    state.ng.flat[0] = np.float32(-1.0)
    with pytest.raises(ValueError, match="number moment ng must be"):
        _initial_hrrr_microphysics_receipt(
            state, MORRISON_PROFILE_ID,
            result.hydrometeor_initialization)


def test_hrrr_nssl2_initialization_receipts_zero_state_and_background_ccn():
    result = _decoded_native_hrrr_initialization(18)
    state = result.state
    zero_fields = (
        "qh", "qndrop", "qnr", "qni", "qns", "qng", "qnh",
        "qvolg", "qvolh",
    )

    receipt = _initial_hrrr_microphysics_receipt(
        state, NSSL2_PROFILE_ID, result.hydrometeor_initialization)

    exact = receipt["state_source_absent_fields"]
    assert set(exact) == {*zero_fields, "qnn"}
    assert exact["qnn"] == {
        "expected_float32": 408163264.0,
        "expected_uint32_bits": int(
            np.float32(408163264.0).view(np.uint32)),
        "all_exact_expected": True,
    }
    assert receipt["state_number_fields"]["qnn"] == {
        "expected_float32": 408163264.0,
        "all_exact_expected": True,
    }
    assert exact["qh"]["expected_uint32_bits"] == 0

    state.qnn.flat[0] = np.nextafter(
        np.float32(408163264.0), np.float32(np.inf), dtype=np.float32)
    with pytest.raises(ValueError, match="source-absent state qnn"):
        _initial_hrrr_microphysics_receipt(
            state, NSSL2_PROFILE_ID,
            result.hydrometeor_initialization)


@pytest.mark.parametrize(
    ("profile", "expected_fields"),
    (
        (MORRISON_PROFILE_ID, {"nc", "nr", "ni", "ns", "ng"}),
        (NSSL2_PROFILE_ID, {
            "qh", "qndrop", "qnr", "qni", "qns", "qng", "qnh",
            "qnn", "qvolg", "qvolh",
        }),
    ),
)
def test_real_domain_state_cold_start_satisfies_profile_receipt(
        profile, expected_fields):
    mp_physics = 10 if profile == MORRISON_PROFILE_ID else 18
    result = _decoded_native_hrrr_initialization(mp_physics)

    receipt = _initial_hrrr_microphysics_receipt(
        result.state, profile, result.hydrometeor_initialization)

    assert set(receipt["state_source_absent_fields"]) == expected_fields
    assert receipt["state_source_absent_fields"].get(
        "qnn", {"expected_float32": 0.0})["expected_float32"] in (
            0.0, 408163264.0)


def test_hrrr_receipt_rejects_zeroed_state_and_pre_fix_mp18_behavior():
    result = _decoded_native_hrrr_initialization(18)
    state = result.state
    _initial_hrrr_microphysics_receipt(
        state, NSSL2_PROFILE_ID, result.hydrometeor_initialization)

    for name in ("qc", "qr", "qi", "qs", "qg"):
        getattr(state, name)[...] = np.float32(0.0)
    with pytest.raises(ValueError, match="source-to-state correspondence"):
        _initial_hrrr_microphysics_receipt(
            state, NSSL2_PROFILE_ID, result.hydrometeor_initialization)

    # Regression anchor for the old mp_physics not-in-(6,8,10) path: it
    # produced the same all-zero MP18 state and no source correspondence.
    with pytest.raises(ValueError, match="lacks decoded-source"):
        _initial_hrrr_microphysics_receipt(
            state, NSSL2_PROFILE_ID, None)


def _outdir_args(outdir, *extra):
    args = [token for token in _required_args()]
    args[args.index("--outdir") + 1] = str(outdir)
    return args + list(extra)


def test_an_existing_outdir_is_refused_in_a_sentence_not_a_traceback(
        tmp_path, capsys):
    """`mkdir(exist_ok=False)` fired mid-setup as a bare FileExistsError,
    after the target domain and the source window had been resolved -- a
    field run of the shipped 1.5.0 wheel met exactly that.  The same file
    created its FAILURE report with exist_ok=True, so the two halves
    disagreed about whether an existing directory was allowed."""

    outdir = tmp_path / "already-here"
    outdir.mkdir()
    with pytest.raises(SystemExit) as raised:
        _parse_args(_outdir_args(outdir))
    assert raised.value.code == 2
    message = capsys.readouterr().err
    assert str(outdir) in message
    assert "already exists" in message
    assert "--allow-existing" in message
    assert "Traceback" not in message


def test_a_fresh_outdir_is_accepted_unchanged(tmp_path):
    args = _parse_args(_outdir_args(tmp_path / "fresh"))
    assert args.outdir == tmp_path / "fresh"
    assert args.allow_existing is False
    assert not (tmp_path / "fresh").exists()  # parsing creates nothing


def test_allow_existing_reuses_a_directory_that_holds_unrelated_files(
        tmp_path):
    outdir = tmp_path / "mine"
    outdir.mkdir()
    (outdir / "notes.txt").write_text("kept", encoding="utf-8")
    args = _parse_args(_outdir_args(outdir, "--allow-existing"))
    assert args.allow_existing is True
    assert (outdir / "notes.txt").read_text(encoding="utf-8") == "kept"


@pytest.mark.parametrize(
    "name", ("report.json", "progress.json", "wrfout_d01_2026-08-03_12_00_00"))
def test_allow_existing_still_refuses_to_overwrite_a_previous_result(
        tmp_path, capsys, name):
    """"Warn, don't block" is house style; silently replacing a benchmark
    result is not warning, it is losing the measurement."""

    outdir = tmp_path / "reused"
    outdir.mkdir()
    (outdir / name).write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit) as raised:
        _parse_args(_outdir_args(outdir, "--allow-existing"))
    assert raised.value.code == 2
    message = capsys.readouterr().err
    assert str(outdir) in message and name in message
    assert "not" in message and "overwritten" in message
    # The file is still there: a refusal that deleted the thing it was
    # protecting would be worse than the clobber.
    assert (outdir / name).exists()


def test_an_outdir_that_is_a_file_is_named_as_such(tmp_path, capsys):
    occupied = tmp_path / "not-a-directory"
    occupied.write_text("", encoding="utf-8")
    with pytest.raises(SystemExit):
        _parse_args(_outdir_args(occupied, "--allow-existing"))
    assert "is not a directory" in capsys.readouterr().err


def test_the_device_name_is_decoded_text_not_a_bytes_repr():
    """A 1.5.0 field report carries `b'NVIDIA GeForce RTX 5070 Ti'` --
    quotes, b prefix and all -- in the field whose whole job is to say
    which card produced the numbers."""
    from tools.hrrr_single_domain_benchmark import _device_name

    class _Runtime:
        @staticmethod
        def getDeviceProperties(index):
            return {"name": b"NVIDIA GeForce RTX 5070 Ti"}

    cp = SimpleNamespace(cuda=SimpleNamespace(runtime=_Runtime))
    assert _device_name(cp) == "NVIDIA GeForce RTX 5070 Ti"
    assert not _device_name(cp).startswith("b'")


def test_an_undecodable_device_name_degrades_instead_of_raising():
    from tools.hrrr_single_domain_benchmark import _device_name

    class _Runtime:
        @staticmethod
        def getDeviceProperties(index):
            return {"name": b"RTX \xff\xfe"}

    cp = SimpleNamespace(cuda=SimpleNamespace(runtime=_Runtime))
    assert _device_name(cp).startswith("RTX ")


def test_an_already_decoded_device_name_passes_through():
    from tools.hrrr_single_domain_benchmark import _device_name

    class _Runtime:
        @staticmethod
        def getDeviceProperties(index):
            return {"name": "NVIDIA GeForce RTX 5090"}

    cp = SimpleNamespace(cuda=SimpleNamespace(runtime=_Runtime))
    assert _device_name(cp) == "NVIDIA GeForce RTX 5090"


def test_the_cold_first_step_is_reported_apart_from_the_warmed_rate():
    """Reproduces the shipped-wheel field run: 900 simulated seconds in
    60 steps, 41.2476 s of integration, a 22.5562 s first step because
    step one pays for every NVRTC compilation the forecast needs.  The
    whole-run average is 2.7498 wall s per simulated minute; the same run
    excluding step one is 1.2672, and it is the second number that
    predicts what an hour of forecast costs."""
    from tools.hrrr_single_domain_benchmark import _warmed_integration_rates

    rates = _warmed_integration_rates(
        integration_seconds=41.247564, run_seconds=900.0, steps=60,
        step_samples=[22.556234, 0.283061, 0.283068])

    assert rates["cold_first_step_seconds"] == pytest.approx(22.556234)
    assert rates["warmed_integration_excluded_steps"] == 1
    assert rates["warmed_integration_seconds"] == pytest.approx(18.69133)
    # 1.2672 wall s per simulated MINUTE, as the field report computed it.
    per_minute = rates[
        "warmed_integration_wall_seconds_per_simulated_hour"] / 60.0
    assert per_minute == pytest.approx(1.2672, abs=5e-5)
    # And the whole-run figure this sits beside is the larger one.
    assert (900.0 / 41.247564) / 60.0 == pytest.approx(0.3637, abs=5e-5)
    assert rates["warmed_integration_simulated_seconds_per_wall_second"] \
        == pytest.approx(885.0 / 18.69133)


@pytest.mark.parametrize(
    ("steps", "samples", "integration"),
    ((1, [3.0], 3.0),          # no warmed population at all
     (60, [], 41.0),           # no per-step samples collected
     (60, [99.0], 41.0)),      # cold step exceeds the whole integration
)
def test_a_warmed_rate_with_no_meaning_is_none_not_a_fabricated_number(
        steps, samples, integration):
    from tools.hrrr_single_domain_benchmark import _warmed_integration_rates

    rates = _warmed_integration_rates(
        integration_seconds=integration, run_seconds=900.0, steps=steps,
        step_samples=samples)
    assert rates["warmed_integration_simulated_seconds_per_wall_second"] \
        is None
    assert rates["warmed_integration_wall_seconds_per_simulated_hour"] is None


def test_the_printed_summary_carries_both_rates(tmp_path, monkeypatch, capsys):
    """stdout is what a short run gets extrapolated from, so the split
    has to reach it -- and the whole-run field keeps its name and its
    value beside the new ones."""
    outdir = tmp_path / "run"

    def fake_run(args):
        args.outdir.mkdir(parents=True, exist_ok=True)
        return {
            "schema": "gpuwm-native-hrrr-benchmark-v2", "status": "PASS",
            "run_seconds": 900.0, "io_mode": "history",
            "history_interval_seconds": 900.0,
            "memory": {"gpu_peak_used_bytes_observed": 9_304_276_992},
            "integration_simulated_seconds_per_wall_second": 900.0 / 41.247564,
            "cold_first_step_seconds": 22.556234,
            "warmed_integration_simulated_seconds_per_wall_second":
                885.0 / 18.69133,
            "prepared_cache": None,
        }

    monkeypatch.setattr(hrrr_runner, "run", fake_run)
    assert hrrr_runner_main(_outdir_args(
        outdir, "--io-mode", "history",
        "--history-interval-seconds", "900")) == 0
    summary = json.loads(capsys.readouterr().out)

    assert summary["simulated_seconds_per_wall_second"] == pytest.approx(
        900.0 / 41.247564)
    assert summary["cold_first_step_seconds"] == pytest.approx(22.556234)
    assert summary["warmed_simulated_seconds_per_wall_second"] \
        == pytest.approx(885.0 / 18.69133)
    # The warmed rate is the larger one; a reader extrapolating from the
    # whole-run figure alone under-counts this machine by a factor of 2.
    assert (summary["warmed_simulated_seconds_per_wall_second"]
            > summary["simulated_seconds_per_wall_second"])


def test_the_route_declaration_is_the_namelist_contract_table():
    """The reverse leg: this route offers exactly what it can replay.

    The walk below proves every DECLARED profile resolves this runner's
    per-profile tables.  It cannot see the other direction, and the other
    direction is what went wrong: six composition suites were declared on
    this route by a registry pass, the runner refused all six with
    ``unsupported native HRRR physics profile``, and plan review had said
    launchable.  A contract row for a profile the route no longer
    declares is the mirror defect -- dead replay state nobody retires.

    So the two sets are held EQUAL here.  Growing the route is now a
    two-part edit by construction: declare the template and transcribe
    the native namelist it is replayed against, or do neither.
    """

    served = {
        profile: hrrr_runner._initialization_contract_profile(profile)
        for profile in _ROUTE_PHYSICS_PROFILES
    }
    missing = sorted(
        profile for profile, contract in served.items()
        if contract not in hrrr_runner._NATIVE_HRRR_NAMELIST_CONTRACTS
        and profile not in hrrr_runner._NATIVE_HRRR_NAMELIST_CONTRACTS)
    assert missing == [], (
        "declared on this route with no native namelist contract to "
        f"replay against: {missing}")

    reachable = set(_ROUTE_PHYSICS_PROFILES) | {
        hrrr_runner._initialization_contract_profile(profile)
        for profile in _ROUTE_PHYSICS_PROFILES
    }
    stale = sorted(set(hrrr_runner._NATIVE_HRRR_NAMELIST_CONTRACTS)
                   - reachable)
    assert stale == [], (
        "namelist contract rows for profiles this route no longer "
        f"declares: {stale}")

    unmapped = sorted({
        name
        for profile in _ROUTE_PHYSICS_PROFILES
        for name in hrrr_runner._native_hrrr_runtime_switches(profile)
    } - set(hrrr_runner._PROFILE_SWITCH_HOMES))
    assert unmapped == [], (
        "declared on this route with no home in _PROFILE_SWITCH_HOMES, "
        f"so the switch would silently take the RunConfig default: "
        f"{unmapped}")


@pytest.mark.parametrize("profile", sorted(_ROUTE_PHYSICS_PROFILES))
def test_every_shipped_profile_resolves_every_per_profile_table(profile):
    """B-04's refusal class, closed for the 13th profile too.

    The battery's first case run passed every static gate and every prior
    stage, then refused at root preparation: the offered menu
    had grown to 12 entries while _NATIVE_HRRR_NAMELIST_CONTRACTS held 10,
    and only the NSSL-2 legacy twin had an alias -- at three separate
    inline sites the Thompson twin missed.  This walk resolves EVERY
    shipped profile through every per-profile table this runner (and the
    root-preparation wrapper) keys by profile id, so the next profile
    cannot repeat the class.

    The value leg guards the opposite regression: the namelist contract is
    a hard per-field equality gate on the supplied namelist, so its values
    must equal the profile's own runtime switches -- a Thompson twin
    "simplified" to a bare alias of its base row would pin radt 1.0
    against the profile's 12.0 and refuse the battery's namelist one gate
    later.
    """

    from tools import prepare_hrrr_wrf

    contract = hrrr_runner._native_hrrr_profile_contract(profile)
    assert set(contract) == {"physics", "dynamics"}
    switches = hrrr_runner._native_hrrr_runtime_switches(profile)

    init_profile = hrrr_runner._initialization_contract_profile(profile)
    defaults = hrrr_runner._HRRR_SOURCE_ABSENT_STATE_DEFAULTS[init_profile]
    fields = hrrr_runner._HRRR_SOURCE_ABSENT_WRF_FIELDS[init_profile]
    assert isinstance(list(fields), list)
    assert isinstance(dict(defaults), dict)
    assert profile in prepare_hrrr_wrf._HRRR_COLD_START_CONTRACT
    assert profile in hrrr_runner.runner_capabilities()["physics_profiles"]

    for section, entries in contract.items():
        for key, expected in entries.items():
            name = {"cudt": "cudt_minutes"}.get(key, key)
            assert name in switches, (profile, section, key)
            observed = switches[name]
            if isinstance(expected, bool):
                assert observed is expected, (profile, section, key)
            else:
                assert float(observed) == float(expected), (
                    profile, section, key, observed, expected)


def test_battery_composition_namelist_passes_the_profile_contract(
        tmp_path, pinned_thompson_tables):
    """The exact gate the 2026-08-04 case run refused at, CPU-side.

    The battery composition's rendered native namelist, validated through
    _validate_native_hrrr_physics_profile under the legacy-RRTMG Thompson
    profile -- the call tools/prepare_hrrr_wrf.py makes at root
    preparation.  Before the twin's own contract row this raised
    'unsupported native HRRR physics profile'; a bare alias to the base
    Thompson row would instead refuse on radt/diff_6th_factor drift.
    """

    from pathlib import Path

    from tools import battery_wrf_node_plan as node_plan

    repo = Path(__file__).resolve().parents[1]
    config = (repo / "configs" / "battery"
              / "shape_3km_thompson_rrtmg_legacy.toml")
    outdir = tmp_path / "node"
    node_plan.build(config, outdir, ranks=24, repository_root=repo)

    receipt = hrrr_runner._validate_native_hrrr_physics_profile(
        outdir / "namelist.native.input",
        THOMPSON_LEGACY_RRTMG_PROFILE_ID)
    assert receipt["profile"] == THOMPSON_LEGACY_RRTMG_PROFILE_ID
    assert receipt["resolved"]["ra_rrtmg_variant"] == "rrtmg_legacy"
    assert receipt["resolved"]["radiation_scheme_ids"] == [4, 4]
    assert receipt["validated_namelist"]["physics"]["radt"] == 12.0
    assert receipt["validated_namelist"]["dynamics"][
        "diff_6th_factor"] == 0.12
    # The initialization contract is the Thompson one, verbatim -- a
    # microphysics property, radiation-variant-independent.
    contract = receipt["hrrr_initialization_contract"]
    assert contract["source_absent_fields"] == ["QNICE", "QNRAIN"]
    assert contract["source_absent_state_defaults_fp32"] == {
        "ni": 0.0, "nr": 0.0}


def test_a_prepare_only_run_says_its_steps_and_a_forecast_run_does_not(capsys):
    """A --prepare-only run is a preparer program whose parent reads its step lines.

    The breakage: the single-domain HRRR preparation wrote no step
    record, so the run page had nothing to show while it built the
    start state and the boundary times.  A forecast run's stderr is a
    person's terminal, so it writes none.
    """

    from woof.prep_progress import step_record

    with hrrr_runner._prep_step(SimpleNamespace(prepare_only=True), "root_initialize",
                                label="Initialize the start state"):
        pass
    said = [step_record(line) for line in capsys.readouterr().err.splitlines()]
    assert [(record["stage"], record["label"], record["event"]) for record in said] == [
        ("root_initialize", "Initialize the start state", "started"),
        ("root_initialize", "Initialize the start state", "finished")]

    with hrrr_runner._prep_step(SimpleNamespace(prepare_only=False), "root_initialize",
                                label="Initialize the start state"):
        pass
    assert "GPUWM_PREP_EVENT" not in capsys.readouterr().err


def _boundary_static(ny, nx):
    mass = np.zeros((ny, nx))
    return {"HGT_M": mass, "LANDMASK": mass, "MAPFAC_M": mass + 1.0,
            "MAPFAC_U": np.ones((ny, nx + 1)), "MAPFAC_V": np.ones((ny + 1, nx)),
            "F": mass, "E": mass, "SINALPHA": mass, "COSALPHA": mass + 1.0}


def test_an_mp28_boundary_strip_carries_its_own_geodesy():
    """Each side ships its rectangle of the mass-point lat/lon, which is
    what the WIF climatology needs (it refused in every boundary worker
    without it); without lat/lon the payload is the eight fields it was."""

    ny, nx, width = 12, 15, 3
    lat = np.arange(ny * nx, dtype=np.float64).reshape(ny, nx)
    lon = -lat
    sides = hrrr_runner._compact_boundary_static(
        _boundary_static(ny, nx), SimpleNamespace(ny=ny, nx=nx),
        width=width, latlon_mass=(lat, lon))
    rectangles = {"west": (0, ny, 0, width), "east": (0, ny, nx - width, nx),
                  "south": (0, width, 0, nx), "north": (ny - width, ny, 0, nx)}
    for side, (y0, y1, x0, x1) in rectangles.items():
        np.testing.assert_array_equal(sides[side]["XLAT_M"], lat[y0:y1, x0:x1])
        np.testing.assert_array_equal(sides[side]["XLONG_M"], lon[y0:y1, x0:x1])
    bare = hrrr_runner._compact_boundary_static(
        _boundary_static(ny, nx), SimpleNamespace(ny=ny, nx=nx), width=width)
    assert all("XLAT_M" not in fields for fields in bare.values())


def test_a_boundary_strip_hands_its_geodesy_to_initialize_real(monkeypatch):
    import woof.ingest.preprocess_backend as backends
    import woof.ingest.real as real
    from woof.config import RunConfig

    class Stop(Exception):
        pass

    seen = {}

    def initialize_real(*args, **kwargs):
        seen.update(kwargs)
        raise Stop

    monkeypatch.setattr(real, "initialize_real", initialize_real)
    monkeypatch.setattr(backends, "resolve_preprocess_backend",
                        lambda *a, **k: SimpleNamespace(
                            array_module=np, name="cpu", receipt=lambda: {}))
    ny, nx, width = 12, 15, 3
    lat = np.full((ny, nx), 40.0)
    lon = np.full((ny, nx), -100.0)
    sides = hrrr_runner._compact_boundary_static(
        _boundary_static(ny, nx), SimpleNamespace(ny=ny, nx=nx),
        width=width, latlon_mass=(lat, lon))
    cfg = RunConfig(nx=nx, ny=ny, nz=4, dx=3000.0, dy=3000.0, ztop=20000.0,
                    dt=10.0, run_seconds=60.0, mp_physics=28)
    mets = {side: SimpleNamespace(flag_sh_surface_fallback=False)
            for side in sides}
    with pytest.raises(Stop):
        hrrr_runner._initialize_boundary_sides(
            mets, cfg, sides, [1.0, 0.7, 0.4, 0.1, 0.0], p_top=5000.0,
            width=width, preprocess_backend="cpu")
    # A later forcing slab must use boundary-source roles, not the source
    # selected only to initialize the start state.
    assert seen["boundary_only"] is True
    grid = seen["grid"]
    np.testing.assert_array_equal(grid["XLAT_M"], lat[:, :width])
    np.testing.assert_array_equal(grid["XLONG_M"], lon[:, :width])


def test_production_initial_and_lateral_times_read_their_declared_smoke_source(monkeypatch, tmp_path):
    """The production preparer's slabs must not reread initial-only smoke.

    Only file IO and numerical remapping are stubbed. The source selection is
    the real chem table and fill_boundary_sources path used by initialize_real.
    """
    from datetime import timedelta, timezone
    from pathlib import Path
    import woof.chem_source_init as source_init
    import woof.ingest.lateral_bc as lateral
    import woof.ingest.preprocess_backend as backends
    import woof.ingest.real as real
    from woof.chem_table import load
    from woof.config import RunConfig

    ny, nx, width, nz = 12, 15, 3, 4
    cfg = RunConfig(nx=nx, ny=ny, nz=nz, dx=3000., dy=3000., ztop=20000.,
                    dt=12., run_seconds=10800., mp_physics=28,
                    chem_sets="smoke", chem_sources="rap-native-smoke,hrrr-native-smoke",
                    aerosol_mp_coupling="none")
    table = load(cfg)
    start = datetime(2023, 6, 7, tzinfo=timezone.utc)
    reads = []

    def covering(row, variables, valid_time, **kwargs):
        reads.append((valid_time, row.name))
        return {"hybrid_level": Path(row.name)}

    def remap(table, row, frames, target, valid_time, **kwargs):
        value = 11. if row.name == "hrrr-native-smoke" else 37.
        return {"smoke": np.full(target.pressure.shape, value, np.float32)}, {"source": row.name}

    monkeypatch.setattr(source_init, "covering_files", covering)
    monkeypatch.setattr(source_init, "_frames", lambda *a: object())
    monkeypatch.setattr(source_init, "files_cycle", lambda *a: start)
    monkeypatch.setattr(source_init, "boundary_fields_for_grid", remap)

    def initialize(snapshot, run, coord, terrain, *, grid, boundary_only=False, **kwargs):
        state = SimpleNamespace(chem=SimpleNamespace(table=table),
            chem_smoke=np.zeros((run.nz, run.ny, run.nx), np.float32),
            set_map_coriolis=lambda *a, **k: None)
        source_init.fill_boundary_sources(state, run, valid_time=snapshot.valid_time,
            latlon=(grid["XLAT_M"], grid["XLONG_M"]),
            pressure=np.ones_like(state.chem_smoke) * 80000., cache_root=tmp_path,
            frame="boundary" if boundary_only else "initial")
        return SimpleNamespace(state=state)

    monkeypatch.setattr(real, "initialize_real", initialize)
    monkeypatch.setattr(hrrr_runner, "_map_snapshot", lambda snapshot, *a, **k: (snapshot, 0.))
    monkeypatch.setattr(backends, "resolve_preprocess_backend", lambda *a, **k:
        SimpleNamespace(array_module=np, name="cpu", receipt=lambda: {}))
    monkeypatch.setattr(lateral, "_coupled_device_fields", lambda state: {"smoke": state.chem_smoke})
    monkeypatch.setattr(lateral, "extract_lateral_side", lambda fields, *a: fields)
    lat, lon = np.full((ny, nx), 41.), np.full((ny, nx), -75.)
    grid = {"XLAT_M": lat, "XLONG_M": lon}
    static = _boundary_static(ny, nx)
    initial = SimpleNamespace(valid_time=start, flag_sh_surface_fallback=False)
    result, *_ = hrrr_runner._initialize_state(initial, SimpleNamespace(run=cfg), grid,
        static, [1., .7, .4, .1, 0.], {}, p_top=5000., preprocess_backend="cpu",
        state_backend="preprocess")
    assert reads == [(start, "hrrr-native-smoke")]
    np.testing.assert_array_equal(result.state.chem_smoke, np.float32(11.))

    # Before the fix, omitting the role at a later time read the initial source.
    later = SimpleNamespace(valid_time=start + timedelta(hours=1), flag_sh_surface_fallback=False)
    initialize(later, cfg, None, static["HGT_M"], grid=grid)
    assert reads[-1] == (later.valid_time, "hrrr-native-smoke")

    sides = hrrr_runner._compact_boundary_static(static, cfg, width=width, latlon_mass=(lat, lon))
    for hour in (1, 2, 3):
        valid = start + timedelta(hours=hour)
        mets = {side: SimpleNamespace(valid_time=valid, flag_sh_surface_fallback=False)
                for side in sides}
        reads.clear()
        fields, *_ = hrrr_runner._initialize_boundary_sides(mets, cfg, sides,
            [1., .7, .4, .1, 0.], p_top=5000., width=width, preprocess_backend="cpu")
        assert reads == [(valid, "rap-native-smoke")] * 4
        for side in fields.values():
            np.testing.assert_array_equal(side["smoke"], np.float64(37.))
