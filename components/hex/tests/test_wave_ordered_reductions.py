"""Wave steps are bitwise repeatable: the same step from the same state gives the same words.

THE BREAKAGE THIS CHECKS: woof's wave kernels summed with float atomicAdd (the DIA quartet scatter, the basin
sums and the partition spectra), so threads added in whatever order they arrived. Two identical 1-card hex runs
(x1.40962 120 km, waves on, 3 h) differed in 4,863,958 of 44,807,348 wave words, 22,315 of them at t=0, and
with roughness feedback on the atmosphere of 2 rank processes vs 2 rank threads differed in 386,196,279 words.
On a tree without woof.hex.wave_ordered the solver runs the engine's atomic kernels and the repeat differs.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _cupy():
    if os.environ.get("GPUWM_NO_LOCAL_GPU") == "1":
        pytest.skip("GPU tests are disabled for this run")
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA card")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("no CUDA driver")
    pytest.importorskip("woof.ocean.waves")
    return cp


def _solver(cp, *, ordered=True):
    """A small open-ocean solver with a broad, uneven sea and a varied wind: every quartet and basin is busy."""
    from woof.ocean.waves import create_solver
    shape = (48, 64)
    latitude, longitude = cp.meshgrid(cp.linspace(-50, 50, shape[0], dtype=cp.float32),
                                      cp.linspace(-170, 170, shape[1], dtype=cp.float32), indexing="ij")
    grid = SimpleNamespace(ny=shape[0], nx=shape[1], latitude=latitude, longitude=longitude)
    wet = cp.ones(shape, cp.float32)
    solver = create_solver(grid, SimpleNamespace(depth=cp.full(shape, 1000, cp.float32), wet=wet), config={})
    if ordered:
        try:
            from woof.hex.wave_ordered import install
        except ImportError:      # a tree without the fix: the solver keeps the engine's atomic kernels
            install = None
        if install is not None:
            install(solver)
    rng = np.random.default_rng(11)
    frequency = solver.frequency[None, :, None]
    peak = cp.asarray(rng.uniform(.06, .2, (solver.nc, 1, 1)), cp.float32)
    phase = cp.asarray(rng.uniform(0, 2 * math.pi, (solver.nc, 1, 1)), cp.float32)
    angle = cp.arange(solver.nd, dtype=cp.float32)[None, None, :] * np.float32(2 * math.pi / solver.nd)
    solver.state[...] = (.002 * cp.exp(-((frequency - peak) / .04) ** 2) * (1.05 + cp.cos(angle - phase))
                         * cp.asarray(rng.uniform(.5, 1.5, solver.state.shape), cp.float32))
    wind = cp.asarray(rng.uniform(3, 20, shape), cp.float32)
    solver.set_forcing(SimpleNamespace(u10=wind, v10=.3 * wind, rho_air=cp.full(shape, 1.225, cp.float32)))
    return solver


def _different(cp, a, b):
    words = a.view(cp.uint8).reshape(a.size, -1) != b.view(cp.uint8).reshape(b.size, -1)
    return int(cp.count_nonzero(cp.any(words, axis=1)))


def test_wave_step_is_bitwise_repeatable():
    cp = _cupy()
    solver = _solver(cp)
    start = {name: array.copy() for name, array in solver.checkpoint_arrays().items()}
    results = []
    for _ in range(3):
        for name, array in solver.checkpoint_arrays().items():
            cp.copyto(array, start[name])
        solver.step(300.)
        results.append({name: array.copy() for name, array in solver.checkpoint_arrays().items()})
    assert float(cp.max(results[0]["bulk_outputs"][0])) > .1          # a real sea, not an empty field
    differing = {name: [_different(cp, results[0][name], other[name]) for other in results[1:]]
                 for name in results[0]}
    assert all(count == 0 for counts in differing.values() for count in counts), differing


def test_ordered_dia_is_the_scatter_dia_to_float_rounding():
    """Same equations, only the order of the sums changed: the gather agrees with the scatter to rounding and
    keeps the DIA's discrete energy balance."""
    cp = _cupy()
    solver = _solver(cp, ordered=False)
    scatter = solver.kernels.get("wave_dia_rates")
    if scatter is None or getattr(solver, "ordered_reductions", None):
        pytest.skip("this woof has no scatter DIA left to compare with")
    from woof.hex.wave_ordered import OrderedDia, compile_module
    import json
    from woof.ocean.waves import action
    packs = json.loads((action.PACKAGE / "source_packs.json").read_text())
    _, _, module = compile_module(nf=solver.nf, nd=solver.nd,
                                  dia_coefficient=packs[solver.source_pack]["dia_coefficient"])
    logq = math.log(solver.spectral.ratio)
    dia = OrderedDia(module, solver.frequency_ext, solver.df_ext, solver.nd, logq, diagonal=True)
    batch = solver._batches[0]
    solver._moments()
    solver._launch("wave_dia_moments", batch.nc, (batch.moments, solver.legacy_moments, np.int32(batch.nc)))
    solver._launch("wave_extend_dia", batch.nc * solver.nfx * solver.nd, (batch.state, solver.frequency_ext,
                   solver.dia_energy, np.int32(batch.nc), np.int32(solver.nfx)))
    outputs = []
    for run in ("scatter", "gather", "gather"):
        solver._zero(solver.dia_rate)
        solver._zero(solver.dia_diagonal)
        if run == "scatter":
            solver._launch("wave_dia_rates", batch.nc * solver.nfx * solver.nd, (
                solver.dia_energy, solver.dia_rate, solver.dia_diagonal, solver.frequency_ext, solver.df_ext,
                batch.depth, solver.legacy_moments, np.int32(batch.nc), np.int32(solver.nfx), np.int32(solver.nd),
                np.float32(logq)))
        else:
            dia.launch(solver.dia_energy, solver.dia_rate, solver.dia_diagonal, solver.frequency_ext,
                       solver.df_ext, batch.depth, solver.legacy_moments, batch.nc)
        outputs.append((solver.dia_rate.copy(), solver.dia_diagonal.copy()))
    (old_rate, old_diagonal), (rate, diagonal), (again_rate, again_diagonal) = outputs
    assert _different(cp, rate, again_rate) == 0 and _different(cp, diagonal, again_diagonal) == 0
    for new, old in ((rate, old_rate), (diagonal, old_diagonal)):
        scale = float(cp.max(cp.abs(old)))
        assert scale > 0 and float(cp.max(cp.abs(new - old))) <= 1e-6 * scale
    energy = rate[:, :solver.nf] * solver.df[None, :, None]
    assert abs(float(cp.sum(energy))) <= 2e-6 * float(cp.sum(cp.abs(energy)))


def test_install_leaves_a_solver_whose_engine_already_orders_its_sums():
    from woof.hex.wave_ordered import install
    kernels = {"wave_dia_rates": object()}
    solver = SimpleNamespace(ordered_reductions=True, kernels=dict(kernels))
    assert install(solver) == "woof"
    assert solver.kernels == kernels
