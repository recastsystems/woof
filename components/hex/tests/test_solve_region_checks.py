"""The host path of a forecast step: checks on the device, read once.

The profile of one step on the 43,884-cell point mesh
(the 2026-09-13 performance profile) put a fifth of every step
in the host health gate (31 strided CuPy reductions, 28 stream drains), 36
event synchronizes and a second launch of every recovery kernel in the timed
launch path, 59 ``cudaGetDeviceProperties`` calls at 1.05 ms each from the
translation units' admission helpers, and 3,328 ``cupy_fill`` launches from
the garbage discipline.  These tests hold the shape of the remedy:

* an admission is measured once per process and a refusal never memoised;
* the step path recovers state untimed, one launch per kernel;
* the solve-region check unit is its own translation unit, is left alone by
  the garbage discipline, and its kernels compare and select without
  arithmetic on the values;
* the health gate, the density validation and the recovered-state
  validation read nothing per field.

The device halves (marked ``gpu`` by conftest because they import cupy) prove
the bits: the fused envelope equals ``cp.min``/``cp.max``/``cp.argmax`` on the
trimmed views term for term, the validators see only the solve region, the
fused restoration writes the same bytes the per-column fills wrote, and an
untimed launch is exactly one launch.
"""

from __future__ import annotations

from pathlib import Path
import re
import sys
import types

import numpy as np
import pytest

from woof.hex.cuda_regional_forecast_v841 import SELF_MANAGED_GARBAGE_MODULES
from woof.hex.cuda_solve_region_v841 import (
    MODULE_KEY,
    SOLVE_REGION_CUDA_SOURCE,
    SOLVE_REGION_KERNELS,
)
from _layout import PACKAGE_DIR

ROOT = Path(__file__).resolve().parents[1]
SRC = PACKAGE_DIR


def _between(source: str, start: str, end: str) -> str:
    return source.split(start, 1)[1].split(end, 1)[0]


# ---------------------------------------------------------------------------
# the admission is measured once
# ---------------------------------------------------------------------------


def _fake_cupy(monkeypatch, *, major: int, minor: int, counter: dict[str, int]):
    def properties(device_id):
        counter["properties"] += 1
        return {
            "major": major,
            "minor": minor,
            "name": "Fake GPU",
            "totalGlobalMem": 10_240 * 1024 * 1024,
            "multiProcessorCount": 68,
        }

    runtime = types.SimpleNamespace(
        getDeviceCount=lambda: 1,
        getDevice=lambda: 0,
        getDeviceProperties=properties,
        runtimeGetVersion=lambda: 13_000,
        driverGetVersion=lambda: 13_030,
    )

    class _Device:
        def __init__(self, device_id):
            self.device_id = device_id

        def use(self):
            return None

    nvrtc = types.ModuleType("cupy.cuda.nvrtc")
    nvrtc.getSupportedArchs = lambda: (75, 80, 86, 89, 90, 100, 120, 121)
    nvrtc.getVersion = lambda: (13, 0)
    cuda = types.ModuleType("cupy.cuda")
    cuda.runtime = runtime
    cuda.Device = _Device
    cuda.nvrtc = nvrtc
    cupy = types.ModuleType("cupy")
    cupy.cuda = cuda
    cupy.__version__ = "fake-for-admission-tests"
    monkeypatch.setitem(sys.modules, "cupy", cupy)
    monkeypatch.setitem(sys.modules, "cupy.cuda", cuda)
    monkeypatch.setitem(sys.modules, "cupy.cuda.nvrtc", nvrtc)
    return cupy


def test_an_admission_is_measured_once_per_process(monkeypatch, tmp_path):
    from woof.hex.cuda_backend import runtime as runtime_module

    runtime_module.forget_admissions()
    counter = {"properties": 0}
    _fake_cupy(monkeypatch, major=12, minor=0, counter=counter)
    first = runtime_module.require_cuda(min_compute=(12, 0), cache_dir=str(tmp_path))
    assert counter["properties"] == 1
    for _ in range(58):
        again = runtime_module.require_cuda(min_compute=(12, 0), cache_dir=str(tmp_path))
        assert again is first
    assert counter["properties"] == 1, "59 admissions must cost one measurement"
    # A different question is a different measurement.
    runtime_module.require_cuda(
        min_compute=(12, 0), required_compute=(12, 0), cache_dir=str(tmp_path)
    )
    assert counter["properties"] == 2
    runtime_module.forget_admissions()


def test_a_refusal_is_never_memoised(monkeypatch, tmp_path):
    from woof.hex.cuda_backend import arch_admission
    from woof.hex.cuda_backend import runtime as runtime_module

    runtime_module.forget_admissions()
    monkeypatch.setattr(arch_admission, "ADMITTED_BELOW_FLOOR", {})
    counter = {"properties": 0}
    # sm_61: the one architecture refusal left is a card NVRTC cannot
    # compile for (the fake lists compute_75 and newer).
    _fake_cupy(monkeypatch, major=6, minor=1, counter=counter)
    for _ in range(3):
        with pytest.raises(runtime_module.CudaRefusal):
            runtime_module.require_cuda(min_compute=(12, 0), cache_dir=str(tmp_path))
    assert counter["properties"] == 3
    runtime_module.forget_admissions()


def test_a_stubbed_cupy_is_never_answered_from_another_module_s_admission(
    monkeypatch, tmp_path
):
    """The memo keeps the module it measured, so a test double cannot inherit
    a real card's answer or a previous double's."""

    from woof.hex.cuda_backend import runtime as runtime_module

    runtime_module.forget_admissions()
    first_counter = {"properties": 0}
    _fake_cupy(monkeypatch, major=12, minor=0, counter=first_counter)
    first = runtime_module.require_cuda(min_compute=(12, 0), cache_dir=str(tmp_path))
    second_counter = {"properties": 0}
    _fake_cupy(monkeypatch, major=12, minor=1, counter=second_counter)
    second = runtime_module.require_cuda(min_compute=(12, 0), cache_dir=str(tmp_path))
    assert second is not first
    assert second.compute == (12, 1)
    assert (first_counter["properties"], second_counter["properties"]) == (1, 1)
    runtime_module.forget_admissions()


# ---------------------------------------------------------------------------
# the step path recovers untimed
# ---------------------------------------------------------------------------


def test_recover_state_is_untimed_by_default_and_on_both_driver_sites():
    recovery = (SRC / "cuda_backend" / "recovery.py").read_text(encoding="utf-8")
    assert "timing_repeats: int = 0," in recovery
    assert "launch_checked(kernel, grid, block, args)" in recovery
    driver = (SRC / "cuda_driver.py").read_text(encoding="utf-8")
    calls = re.findall(r"recover_state\((.*?)\)\n", driver, flags=re.DOTALL)
    assert len(calls) == 2, "the driver recovers on the commit rebuild and per RK stage"
    for call in calls:
        assert "timing_repeats=0" in call
        assert "timing_repeats=1" not in call


def test_launch_checked_records_no_event_and_synchronizes_nothing():
    runtime = (SRC / "cuda_backend" / "runtime.py").read_text(encoding="utf-8")
    body = _between(runtime, "def launch_checked(", "def launch_timed(")
    code = body.split('"""')[2]  # after the docstring
    assert "cp.cuda.Event(" not in code
    assert ".synchronize(" not in code
    assert "_check_launch_error(cp)" in code


# ---------------------------------------------------------------------------
# the check unit
# ---------------------------------------------------------------------------


def test_the_check_unit_is_its_own_translation_unit():
    assert MODULE_KEY == "hexcore.cuda_solve_region_v841"
    defined = re.findall(r"__global__ void (\w+)", SOLVE_REGION_CUDA_SOURCE)
    assert tuple(defined) == SOLVE_REGION_KERNELS


def test_the_garbage_discipline_leaves_the_check_unit_alone():
    assert MODULE_KEY in SELF_MANAGED_GARBAGE_MODULES


def test_the_check_kernels_do_no_arithmetic_on_the_values():
    """They compare, select and store what they were handed.

    The only intrinsics in the source are the comparing and selecting ones
    and the bit casts; none of the FTZ arithmetic helpers the dycore
    kernels compute with (``mpas_mul``, ``mpas_add``, ``mpas_div``, ...)
    appears, and the unit does not include ``CUDA_FTZ_HELPERS`` at all.
    Every float a kernel stores is a float it loaded or a constant bit
    pattern, so no value can move by a rounding.
    """

    intrinsics = set(re.findall(r"\b(f\w+f|__\w+|isfinite|isnan)\(", SOLVE_REGION_CUDA_SOURCE))
    assert intrinsics <= {
        "fminf", "fmaxf", "fabsf", "isfinite", "__float_as_int", "__int_as_float",
        "__syncthreads",
    }, intrinsics
    assert "mpas_" not in SOLVE_REGION_CUDA_SOURCE
    assert "CUDA_FTZ_HELPERS" not in (SRC / "cuda_solve_region_v841.py").read_text(encoding="utf-8")


def test_the_health_gate_reads_once():
    source = (PACKAGE_DIR / "drivers" / "run_cuda_v841_forecast.py").read_text(encoding="utf-8")
    body = _between(source, "def step_health_gate(", "\nclass BoundaryFingerprintWriter")
    for reduction in ("cp.min(", "cp.max(", "cp.argmax(", "cp.abs("):
        assert reduction not in body, reduction
    assert "SolveRegionKernels.for_cache(stack[\"driver\"].cache).envelope(fields)" in body


def test_the_regional_validators_set_the_flag_on_the_device():
    source = (SRC / "cuda_regional_forecast_v841.py").read_text(encoding="utf-8")
    recovered = _between(source, "    def validate_recovered(", "    def validate_density(")
    density = _between(source, "    def validate_density(", "    def history_slice(")
    for body in (recovered, density):
        assert "self.solve_region.validate(" in body
        assert "bool(cp.all" not in body
        assert "flag[0] = 1" not in body


def test_the_discipline_batches_its_restoration_when_nothing_watches():
    source = (SRC / "cuda_regional_forecast_v841.py").read_text(encoding="utf-8")
    body = _between(source, "    def scrub(", "    def receipt(")
    assert "fused = None if (self.measure or self.audit is not None) else self.fused" in body
    assert "fused.scrub_columns(batch)" in body
    assert "column[...] = pool" in body, "the measured and audited forms keep the per-column fill"


# ---------------------------------------------------------------------------
# the device halves
# ---------------------------------------------------------------------------


def _kernel_cache():
    from woof.hex.cuda_backend import KernelCache, require_cuda

    try:
        capability = require_cuda(min_compute=(12, 0))
    except Exception as error:  # pragma: no cover - no admitted device
        pytest.skip(f"no admitted CUDA device: {error}")
    return KernelCache(capability=capability)


def test_the_envelope_equals_the_cupy_reductions_on_the_trimmed_views():
    import cupy as cp

    from woof.hex.cuda_solve_region_v841 import SolveRegionKernels

    kernels = SolveRegionKernels.for_cache(_kernel_cache())
    rng = np.random.default_rng(7)
    nlev, cells, edges = 7, 1_030, 3_101
    fields = {}
    rho = rng.uniform(0.5, 1.5, (nlev, cells + 1)).astype(np.float32)
    rho[:, cells] = np.nan  # the garbage column: must never be read
    fields["rho"] = (rho, cells)
    w = rng.normal(0.0, 3.0, (nlev + 1, cells + 1)).astype(np.float32)
    w[3, 10] = 41.0
    w[5, 400] = -41.0  # a tie in |w| at a later flat index: the first wins
    w[:, cells] = 1e30
    fields["w"] = (w, cells)
    u = rng.normal(0.0, 10.0, (nlev, edges + 1)).astype(np.float32)
    u[2, 77] = np.inf  # +inf inside the solve region is carried, not NaN
    u[:, edges] = np.nan
    fields["u"] = (u, edges)
    bad = rng.normal(0.0, 1.0, (nlev, cells + 1)).astype(np.float32)
    bad[4, 900] = np.nan  # a NaN inside the solve region: min and max are NaN
    fields["bad"] = (bad, cells)
    scalars = rng.uniform(0.0, 1e-3, (6, nlev, cells + 1)).astype(np.float32)
    scalars[2, 1, 5] = -2.5e-4
    scalars[:, :, cells] = -7.0
    fields["scalars"] = (scalars, cells)
    fields["scalars[1:]"] = (scalars[1:], cells)
    fields["whole"] = (rng.normal(size=(nlev, 33)).astype(np.float32), None)

    device = {name: cp.asarray(array) for name, (array, _n) in fields.items()}
    device["scalars[1:]"] = device["scalars"][1:]
    measured = kernels.envelope(
        [(name, device[name], fields[name][1]) for name in fields]
    )
    for name, (array, n_solve) in fields.items():
        view = device[name] if n_solve is None else device[name][..., :n_solve]
        low, high, has_nan, argmax = measured[name]
        expected_low = float(cp.min(view))
        expected_high = float(cp.max(view))
        if np.isnan(expected_low):
            assert np.isnan(low) and np.isnan(high) and has_nan, name
        else:
            assert (low, high) == (expected_low, expected_high), name
            assert not has_nan, name
            assert argmax == int(cp.argmax(cp.abs(view))), name
    # The second call hits the table cache: same arrays, same geometry.
    uploads = kernels.table_uploads
    kernels.envelope([(name, device[name], fields[name][1]) for name in fields])
    assert kernels.table_uploads == uploads


def test_the_validators_see_only_the_solve_region():
    import cupy as cp

    from woof.hex.cuda_solve_region_v841 import SolveRegionKernels

    kernels = SolveRegionKernels.for_cache(_kernel_cache())
    nlev, cells = 5, 257
    density = cp.ones((nlev, cells + 1), dtype=cp.float32)
    density[:, cells] = 0.0  # the pool zero in the garbage column
    flag = cp.zeros((1,), dtype=cp.int32)
    kernels.validate(flag, positive=((density, cells),))
    assert int(flag.get()[0]) == 0
    density[2, 100] = 0.0
    kernels.validate(flag, positive=((density, cells),))
    assert int(flag.get()[0]) == 1
    flag[0] = 0
    kernels.validate(flag, finite=((density, cells),))
    assert int(flag.get()[0]) == 0, "zero is finite"
    density[2, 100] = cp.nan
    kernels.validate(flag, finite=((density, cells),))
    assert int(flag.get()[0]) == 1
    flag[0] = 0
    density[2, 100] = 1.0
    density[0, cells] = cp.nan
    kernels.validate(flag, positive=((density, cells),), finite=((density, cells),))
    assert int(flag.get()[0]) == 0, "a NaN in the garbage column is nobody's business"


def test_the_fused_restoration_writes_the_bytes_the_fills_wrote():
    import cupy as cp

    from woof.hex.cuda_solve_region_v841 import SolveRegionKernels

    kernels = SolveRegionKernels.for_cache(_kernel_cache())
    rng = np.random.default_rng(11)
    nlev, cells, edges = 6, 411, 1_233
    arrays = [
        rng.normal(size=(nlev, cells + 1)).astype(np.float32),
        rng.normal(size=(nlev + 1, cells + 1)).astype(np.float32),
        rng.normal(size=(4, nlev, cells + 1)).astype(np.float32),
        rng.normal(size=(nlev, edges + 1)).astype(np.float32),
    ]
    pools = [1.0, 0.0, 0.0, 0.0]
    solves = [cells, cells, cells, edges]
    by_fill = [cp.asarray(a) for a in arrays]
    by_kernel = [cp.asarray(a) for a in arrays]
    for array, pool, solve in zip(by_fill, pools, solves):
        array[..., solve][...] = np.float32(pool)
    kernels.scrub_columns(
        [
            (int(array.data.ptr), int(array.size // array.shape[-1]), int(array.shape[-1]), solve, pool)
            for array, pool, solve in zip(by_kernel, pools, solves)
        ]
    )
    for left, right in zip(by_fill, by_kernel):
        assert left.get().tobytes() == right.get().tobytes()


def test_an_untimed_launch_is_exactly_one_launch():
    import cupy as cp

    from woof.hex.cuda_backend import launch_checked, launch_timed

    cache = _kernel_cache()
    kernel = cache.raw_kernel(
        "count_launches",
        r"""
        extern "C" __global__ void count_launches(int* counter) {
            if (threadIdx.x == 0 && blockIdx.x == 0) atomicAdd(counter, 1);
        }
        """,
        module_key="hexcore.tests.count_launches",
    )
    counter = cp.zeros((1,), dtype=cp.int32)
    launch_checked(kernel, (1,), (32,), (counter,))
    assert int(counter.get()[0]) == 1
    timing = launch_timed(kernel, (1,), (32,), (counter,), warmup=0, repeats=1)
    assert timing.repeats == 1
    assert int(counter.get()[0]) == 3, "the timed path launches twice for repeats=1"


def _synthetic_atmosphere(cp, *, terrain: bool):
    """A small consistent atmosphere for recover_state: 64 cells, 96 edges,
    4 levels, every array resident, shapes as the kernels index them."""

    rng = np.random.default_rng(5)
    cells, edges, nlev, max_edges = 64, 96, 4, 6
    f32 = lambda shape: cp.asarray(rng.uniform(0.5, 1.5, shape).astype(np.float32))  # noqa: E731
    cells_on_edge = cp.asarray(rng.integers(0, cells, (edges, 2)).astype(np.int32))
    edges_on_cell = cp.asarray((np.arange(cells * max_edges) % edges).reshape(cells, max_edges).astype(np.int32))
    mesh = types.SimpleNamespace(
        n_cells=cells, n_edges=edges, max_edges=max_edges, dtype=np.float32, index_dtype=np.dtype(np.int32),
        cells_on_edge=cells_on_edge, edges_on_cell=edges_on_cell,
        n_edges_on_cell=cp.full((cells,), 2, dtype=cp.int32),
        edge_sign_on_cell=f32((cells, max_edges)),
    )
    vertical = types.SimpleNamespace(
        n_vert_levels=nlev, dtype=np.float32, zz=f32((nlev, cells)),
        fzm=f32((nlev,)), fzp=f32((nlev,)), cf1=1.5, cf2=-0.5, cf3=0.0,
    )
    reference = types.SimpleNamespace(
        dtype=np.float32, rho_base=f32((nlev, cells)), rho_theta_base=f32((nlev, cells)), exner_base=f32((nlev, cells)),
    )
    state = types.SimpleNamespace(
        dtype=np.float32, rho=f32((nlev, cells)), rho_theta=f32((nlev, cells)) * 300.0,
        rho_u=f32((nlev, edges)), rho_w=f32((nlev + 1, cells)),
    )
    saved = types.SimpleNamespace(dtype=np.float32)
    terrain_metrics = (
        types.SimpleNamespace(zb_cell=f32((nlev, cells, max_edges)), zb3_cell=f32((nlev, cells, max_edges)))
        if terrain else None
    )
    return types.SimpleNamespace(
        mesh=mesh, vertical=vertical, reference=reference, state=state, saved=saved, terrain=terrain_metrics,
    )


@pytest.mark.parametrize("terrain", (True, False))
def test_recover_state_launches_once_on_both_routes_and_matches_the_timed_path(terrain):
    import cupy as cp

    from woof.hex.cuda_backend.recovery import recover_state

    cache = _kernel_cache()
    atmosphere = _synthetic_atmosphere(cp, terrain=terrain)
    untimed = recover_state(atmosphere, cache=cache)
    assert set(untimed.timings) == {"pressure", "normal_velocity", "vertical_velocity"}
    assert all(value is None for value in untimed.timings.values())
    velocities_only = recover_state(atmosphere, cache=cache, include_pressure=False)
    assert velocities_only.theta_m is None and velocities_only.pressure is None
    assert set(velocities_only.timings) == {"normal_velocity", "vertical_velocity"}
    timed = recover_state(atmosphere, cache=cache, warmup=0, timing_repeats=1)
    assert all(timing.repeats == 1 for timing in timed.timings.values())
    for name in ("theta_m", "exner", "pressure", "density_perturbation", "rho_theta_perturbation",
                 "pressure_perturbation", "normal_velocity", "vertical_velocity"):
        left = getattr(untimed, name).get().tobytes()
        assert left == getattr(timed, name).get().tobytes(), name
        assert np.isfinite(getattr(untimed, name).get()).all(), name
    for name in ("normal_velocity", "vertical_velocity"):
        assert getattr(velocities_only, name).get().tobytes() == getattr(untimed, name).get().tobytes(), name
