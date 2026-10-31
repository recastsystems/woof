"""A prepared cache binds what a preparation ran, not what it measured (A138).

A CUDA preparation recorded ``selection.device_fit.free_bytes`` (the card's
free memory when the preparation was priced) inside every domain's cache
identity, so two CUDA preparations of the same inputs differed in content
digest by that reading alone.  The receipt (the proof) keeps every
measurement; the cache binds :func:`preprocess_identity` of it.
"""

from __future__ import annotations

import copy

import pytest

from woof.ingest.preprocess_backend import (
    PREPROCESS_RECEIPT_MEASUREMENTS,
    ParallelCpuPreprocessBackend,
    preprocess_identity,
    preprocess_identity_matches,
    preprocess_measurements,
    preprocess_reports_identity,
    preprocess_selection_identity,
)


def _cuda_receipt(*, free: int, utilization: float, host_workers):
    """A CUDA receipt shaped as CudaPreprocessBackend.receipt() writes it
    after ``auto`` chose the card and admit_preparation priced it."""

    return {
        "schema": "gpuwm-preprocess-implementation-v2",
        "backend": "cuda",
        "implementation": "cupy-fp32",
        "workers": 1,
        "cupy_version": "14.0.0",
        "cuda_runtime_version": 13000,
        "contracts": {"vertical": "wrf-real"},
        "implementation_tree": {"sha256": "a" * 64},
        "vertical_interpolation": [
            {"source_levels": 50, "backend": "cuda",
             "kernel_level_tier": 64, "reason": "fits the 64-level tier"}],
        "masked_surface_chain": {
            "implementation": "wps-masked-chain",
            "workers": host_workers,
            "bridge": {"name": "libgpuwm_cpu.so", "sha256": "b" * 64},
        },
        "selection": {
            "requested": "auto",
            "backend": "cuda",
            "reason": "cupy 14.0.0 on CUDA runtime 13000 is certified",
            "device_load": {
                "probe": "device_memory_probe_subprocess",
                "free_bytes": free, "total_bytes": 16 * 2**30,
                "utilization_gpu_percent": utilization,
                "busy_utilization_threshold_percent": 50,
            },
            "device_fit": {
                "route": "mapped", "need_bytes": 3 * 2**30,
                "free_bytes": free, "total_bytes": 16 * 2**30,
                "fits": True,
            },
        },
    }


def test_two_cuda_preparations_bind_the_same_receipt_whatever_the_card_read():
    first = _cuda_receipt(free=15_000_000_000, utilization=0.0,
                          host_workers="auto")
    second = _cuda_receipt(free=12_345_678_912, utilization=37.0,
                           host_workers=24)
    assert first != second
    assert preprocess_identity(first) == preprocess_identity(second)


def test_the_bound_receipt_drops_exactly_the_named_measurements():
    receipt = _cuda_receipt(free=1, utilization=0.0, host_workers=8)
    before = copy.deepcopy(receipt)
    bound = preprocess_identity(receipt)

    assert receipt == before, "the proof's receipt must stay whole"
    assert bound["selection"] == {"requested": "auto", "backend": "cuda"}
    assert "workers" not in bound
    assert "workers" not in bound["masked_surface_chain"]
    assert bound["masked_surface_chain"]["bridge"] == \
        receipt["masked_surface_chain"]["bridge"]
    for key in ("schema", "backend", "implementation", "cupy_version",
                "cuda_runtime_version", "contracts", "implementation_tree",
                "vertical_interpolation"):
        assert bound[key] == receipt[key]
    # The route record a backend appends to after the receipt is taken
    # reads the same through the bound copy.
    assert bound["vertical_interpolation"] is receipt["vertical_interpolation"]
    # Every name in the table is a path into a receipt, one level deep at most.
    assert all(path.count(".") <= 1 for path in PREPROCESS_RECEIPT_MEASUREMENTS)



def test_the_measurements_are_exactly_what_the_identity_drops():
    receipt = _cuda_receipt(free=7, utilization=12.0, host_workers=8)
    measured = preprocess_measurements(receipt)
    assert set(measured) == {"workers", "masked_surface_chain", "selection"}
    rebuilt = copy.deepcopy(preprocess_identity(receipt))
    for key, value in measured.items():
        if isinstance(value, dict):
            rebuilt[key] = {**rebuilt[key], **value}
        else:
            rebuilt[key] = value
    assert rebuilt == receipt
    assert preprocess_measurements(None) == {}
    assert preprocess_selection_identity(receipt["selection"]) ==         preprocess_identity(receipt)["selection"]
    assert preprocess_selection_identity(None) is None

@pytest.mark.parametrize("change", [
    ("backend", "cpu"),
    ("implementation", "other"),
    ("cupy_version", "13.6.0"),
    ("implementation_tree", {"sha256": "c" * 64}),
])
def test_what_shapes_the_arrays_stays_bound(change):
    key, value = change
    receipt = _cuda_receipt(free=1, utilization=0.0, host_workers=8)
    other = dict(receipt, **{key: value})
    assert preprocess_identity(other) != preprocess_identity(receipt)


def test_a_changed_selector_stays_bound():
    receipt = _cuda_receipt(free=1, utilization=0.0, host_workers=8)
    other = copy.deepcopy(receipt)
    other["selection"]["requested"] = "cuda"
    assert preprocess_identity(other) != preprocess_identity(receipt)


def test_a_cache_binds_new_or_legacy_and_nothing_else():
    receipt = _cuda_receipt(free=1, utilization=0.0, host_workers=8)
    # Written since A138.
    assert preprocess_identity_matches(preprocess_identity(receipt), receipt)
    # Written before it: the whole receipt, still this preparation exactly.
    assert preprocess_identity_matches(copy.deepcopy(receipt), receipt)
    # Another backend's preparation is not this proof's.
    other = dict(receipt, backend="cpu")
    assert not preprocess_identity_matches(preprocess_identity(other), receipt)
    assert not preprocess_identity_matches(other, receipt)
    assert preprocess_identity_matches(None, None)


def test_reports_carry_their_backend_receipts_bound():
    reports = {
        "f00": {"target": "d01", "preprocess_backend": _cuda_receipt(
            free=1, utilization=0.0, host_workers=8)},
        "f01": {"sides": {"west": {"preprocess_backend": _cuda_receipt(
            free=2, utilization=5.0, host_workers=4)}}},
    }
    bound = preprocess_reports_identity(reports)
    assert bound["f00"]["target"] == "d01"
    assert bound["f00"]["preprocess_backend"] == preprocess_identity(
        reports["f00"]["preprocess_backend"])
    assert bound["f01"]["sides"]["west"]["preprocess_backend"] == \
        bound["f00"]["preprocess_backend"]


def test_cpu_receipts_at_two_worker_counts_bind_the_same():
    try:
        backend = ParallelCpuPreprocessBackend(workers=24)
    except (FileNotFoundError, OSError) as exc:
        pytest.skip(f"native CPU bridge is not built: {exc}")
    backend.selection = {"requested": "cpu", "backend": "cpu",
                         "reason": "named by the caller"}
    slot = backend.at_workers(3)
    assert backend.receipt() != slot.receipt()
    assert preprocess_identity(backend.receipt()) == \
        preprocess_identity(slot.receipt())
    assert "host_cpu_count" not in preprocess_identity(backend.receipt())


def test_a_case_store_binds_no_memory_reading(tmp_path, monkeypatch):
    """The host store's cache left the free host and card bytes of its
    admissions in its hashed metadata; they ride the initialization
    receipt's ``memory`` block instead."""

    from dataclasses import dataclass
    from datetime import datetime

    import numpy as np

    import woof.ingest.prepared_cache as prepared_cache
    import woof.native_wrf_contract as native_wrf_contract
    from woof.ingest.case_store import (
        CaseStoreRequest, write_case_store_input)

    written = []

    def capture(path, **kwargs):
        written.append(kwargs)
        return {"content_sha256": "0" * 64}

    monkeypatch.setattr(prepared_cache, "write_prepared_cache", capture)
    monkeypatch.setattr(native_wrf_contract, "canonical_noah_surface",
                        lambda soil: {})

    @dataclass
    class Run:
        nx: int = 2
        ny: int = 2

    @dataclass
    class Vertical:
        eta: tuple = (1.0, 0.0)

    reading = {"device_budget_bytes": 12_345_678_912,
               "host_available_bytes": 98_765_432_100}
    request = CaseStoreRequest(tmp_path / "store", admissions=[reading])
    inputs = write_case_store_input(
        request, cfg=Run(), vertical=Vertical(),
        times=[datetime(2026, 9, 30)], initial_result=object(), met=object(),
        soil=type("Soil", (), {"tsk": np.zeros((1, 1))})(),
        soil_fields={}, reconciled_soil_type=np.zeros((1, 1)),
        boundaries=None, landuse_attrs={}, trace_gas_overrides=None,
        radiation_column_chunk=1, constant_glw_wm2=None)

    (kwargs,) = written
    assert "12345678912" not in repr(kwargs.get("metadata"))
    assert "98765432100" not in repr(kwargs.get("metadata"))
    assert "12345678912" not in repr(kwargs["identity"])
    assert inputs.admissions == (reading,)


def _cpu_receipt_leaving_a_busy_card(*, free: int, utilization: float,
                                     cpus: int, requested: str = "auto"):
    """A CPU receipt shaped as ParallelCpuPreprocessBackend.receipt()
    writes it after ``auto`` left a busy card."""

    return {
        "schema": "gpuwm-preprocess-implementation-v2",
        "backend": "cpu",
        "implementation": "rust-native",
        "workers": "auto",
        "host_cpu_count": cpus,
        "bridge": {"name": "libgpuwm_cpu.so", "sha256": "b" * 64,
                   "abi_version": 9, "required_abi_version": 9},
        "contracts": {"vertical": "wrf-real"},
        "implementation_tree": {"sha256": "a" * 64},
        "vertical_interpolation": [],
        "masked_surface_chain": {"implementation": "wps-masked-chain",
                                 "workers": "auto"},
        "selection": {
            "requested": requested,
            "backend": "cpu",
            "reason": (f"GPU utilization {utilization:.0f}% meets the busy "
                       "threshold of 50%"),
            "device_load": {
                "probe": "device_memory_probe_subprocess",
                "free_bytes": free, "total_bytes": 16 * 2**30,
                "utilization_gpu_percent": utilization,
                "busy_utilization_threshold_percent": 50,
            },
            "device_fit": {
                "route": "met_em", "need_bytes": 3 * 2**30,
                "free_bytes": free, "total_bytes": 16 * 2**30,
                "fits": True,
            },
        },
    }


def _prepare_met_em(tmp_path, monkeypatch, receipt, name):
    """``prepare_metem_run`` with ``receipt`` as its backend's, through
    its documents and metgrid-import.json; no domain is built."""

    import tomllib
    from types import SimpleNamespace

    import numpy as np

    from test_namelist_import import _pair
    from woof import metem_door, metem_forecast
    from woof.experiment import build_experiment
    from woof.ingest import preprocess_backend
    from woof.namelist_import import import_namelists, parse_namelist_text
    from woof.static import projection

    source = tmp_path / "source"
    source.mkdir(exist_ok=True)
    namelist, wps = _pair(source)
    text, report = import_namelists(namelist, wps,
                                    metgrid_initialization=True)
    exp = build_experiment(tomllib.loads(text), source="met_em fixture")
    paths = {}
    for domain in exp.domains:
        met = source / f"met_em.d{domain.grid_id:02d}.2026-05-17_18_00_00.nc"
        met.write_bytes(b"input")
        paths[domain.grid_id] = (met,)
    run = SimpleNamespace(
        toml_text=text, experiment=exp, namelist_input=namelist, paths=paths,
        interval_seconds=3600., coverage_seconds=exp.run_seconds,
        controls=parse_namelist_text(namelist.read_text()),
        substitution_report=report)
    monkeypatch.setattr(metem_forecast, "resolve_metem_vertical",
                        lambda run, text, **kw: (text, "explicit", None))
    monkeypatch.setattr(metem_door, "metgrid_memory_admission",
                        lambda *a, **kw: {})
    monkeypatch.setattr(
        preprocess_backend, "resolve_preprocess_backend",
        lambda *a, **kw: SimpleNamespace(
            name=receipt["backend"],
            receipt=lambda: copy.deepcopy(receipt)))
    monkeypatch.setattr(projection, "grids_from_projection_config",
                        lambda *a: ())
    monkeypatch.setattr(metem_forecast, "read_met_em_terrain",
                        lambda path: np.zeros((2, 2)))
    monkeypatch.setattr(metem_forecast, "adapt_experiment_vertical",
                        lambda exp, fields, **kw: (exp, None))
    monkeypatch.setattr(metem_forecast, "_vertical_coordinate_receipt",
                        lambda exp, adaptation: {})
    return metem_forecast.prepare_metem_run(run, tmp_path / name)


def test_a_met_em_receipt_keeps_what_its_implementation_document_drops(
        tmp_path, monkeypatch):
    """preprocess.json is hashed into every met_em domain's cache identity
    and binds no measurement; metgrid-import.json, the preparation's
    receipt, still says why the preparation left the card and what it
    measured."""

    import json

    receipt = _cpu_receipt_leaving_a_busy_card(
        free=5 * 2**30, utilization=73.0, cpus=24)
    inputs = _prepare_met_em(tmp_path, monkeypatch, receipt, "prepared")

    imported = json.loads(
        inputs.artifact_paths["preparation_receipt"].read_text())
    assert imported["preprocess_backend_selection"] == receipt["selection"]
    assert "73% meets the busy threshold" in \
        imported["preprocess_backend_selection"]["reason"]
    assert imported["preprocess_backend_measurements"] == {
        "host_cpu_count": 24, "workers": "auto",
        "masked_surface_chain": {"workers": "auto"}}

    implementation = json.loads(
        inputs.artifact_paths["preprocess_implementation"].read_text())
    for measured in ("selection", "host_cpu_count", "workers"):
        assert measured not in implementation
    assert implementation["masked_surface_chain"] == {
        "implementation": "wps-masked-chain"}
    assert implementation["backend"] == "cpu"
    assert implementation["bridge"] == receipt["bridge"]


def test_two_met_em_preparations_bind_one_identity_whatever_the_machine_read(
        tmp_path, monkeypatch):
    """Two preparations of the same met_em files that measured a
    different card and CPU count hash the same implementation document
    (a member of every domain's cache identity) and bind the same
    checkpoint identity; their receipts keep what each measured."""

    from woof.metem_forecast import MetemInitialization

    first = _prepare_met_em(tmp_path, monkeypatch,
        _cpu_receipt_leaving_a_busy_card(
            free=5 * 2**30, utilization=73.0, cpus=24), "first")
    second = _prepare_met_em(tmp_path, monkeypatch,
        _cpu_receipt_leaving_a_busy_card(
            free=2 * 2**30, utilization=91.0, cpus=64), "second")
    # CONTROL: the same measurements, but the caller named the CPU.
    named = _prepare_met_em(tmp_path, monkeypatch,
        _cpu_receipt_leaving_a_busy_card(
            free=5 * 2**30, utilization=73.0, cpus=24, requested="cpu"),
        "named")

    for name in ("preprocess_implementation", "experiment_config"):
        assert first.authority_sha256[name] == second.authority_sha256[name]
    assert first.authority_sha256["preparation_receipt"] != \
        second.authority_sha256["preparation_receipt"]
    assert MetemInitialization(first).preparation_receipt_sha256() == \
        MetemInitialization(second).preparation_receipt_sha256()
    assert MetemInitialization(named).preparation_receipt_sha256() != \
        MetemInitialization(first).preparation_receipt_sha256()


def test_a_reused_met_em_preparation_keeps_what_its_original_measured(
        tmp_path, monkeypatch):
    """A second preparation of the same met_em files on another machine
    reuses the first (their implementation documents are one) and its
    receipt stays the first's, byte for byte: the retained authority is
    compared, never rewritten with today's readings."""

    import json

    first = _cpu_receipt_leaving_a_busy_card(
        free=5 * 2**30, utilization=73.0, cpus=24)
    inputs = _prepare_met_em(tmp_path, monkeypatch, first, "prepared")
    # The fixture builds no domain; these placeholders complete the
    # inventory a reuse is decided on.  No cache reader is exercised.
    for domain in inputs.experiment.domains:
        (inputs.prepared_root / f"static-d{domain.grid_id:02d}.npz"
         ).write_bytes(b"static marker")
        cache = inputs.prepared_root / f"d{domain.grid_id:02d}"
        cache.mkdir()
        (cache / "header.json").write_text("{}")
    receipt_path = inputs.artifact_paths["preparation_receipt"]
    before = receipt_path.read_bytes()

    again = _prepare_met_em(tmp_path, monkeypatch,
        _cpu_receipt_leaving_a_busy_card(
            free=2 * 2**30, utilization=91.0, cpus=64), "prepared")

    assert again.prepared_root == inputs.prepared_root
    assert receipt_path.read_bytes() == before
    kept = json.loads(before)
    assert kept["preprocess_backend_selection"] == first["selection"]
    assert kept["preprocess_backend_measurements"]["host_cpu_count"] == 24


@pytest.mark.parametrize('source', [
    'woof/core/nest_interp.py', 'woof/core/kernels/nest.cu',
    'woof/ingest/nest_init.py', 'woof/numerics.py',
])
def test_changed_nest_preparation_arithmetic_invalidates_cached_inputs(monkeypatch, source):
    from woof.ingest import preprocess_backend as backend
    before = backend._implementation_tree('cuda')
    original = backend._sha256
    monkeypatch.setattr(backend, '_sha256', lambda path:
                        '0' * 64 if path.as_posix().endswith('/' + source)
                        else original(path))
    after = backend._implementation_tree('cuda')
    assert before['sha256'] != after['sha256'], source


def test_prepared_cache_binds_column_eos_implementation():
    from woof.ingest.preprocess_backend import _implementation_tree
    for backend in ("cpu", "cuda"):
        tree = _implementation_tree(backend)
        assert "woof/core/diagnostics.py" in tree["files"]
        if backend == "cuda":
            assert "woof/core/kernels/diagnostics.cu" in tree["files"]
