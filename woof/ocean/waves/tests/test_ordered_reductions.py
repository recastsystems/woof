"""Wave steps are bitwise repeatable: the same step from the same state gives the same words.

THE BREAKAGE THIS CHECKS: the DIA quartet scatter and the basin/partition sums used float atomicAdd, so threads
added in whatever order they arrived. Two identical 1-card woof hex runs (x1.40962 120 km, waves on, 3 h)
differed in 4,863,958 of 44,807,348 wave words, 22,315 of them at t=0, and with roughness feedback on the
atmosphere of 2 rank processes vs 2 rank threads differed in 386,196,279 words. On the atomic kernels both
tests below fail (the repeat differs in tens of thousands of words).
"""
import math
import os
from types import SimpleNamespace

import numpy as np
import pytest


def device():
    if os.environ.get("GPUWM_NO_LOCAL_GPU") == "1":
        pytest.skip("GPU tests are disabled for this run")
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA card")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("no CUDA driver")
    return cp


def _different(cp, a, b):
    words = a.view(cp.uint8).reshape(a.size, -1) != b.view(cp.uint8).reshape(b.size, -1)
    return int(cp.count_nonzero(cp.any(words, axis=1)))


def _sea(cp, frequency, shape, nd, seed):
    """A broad, uneven sea: every quartet, basin and partition sum has many terms."""
    rng = np.random.default_rng(seed)
    cells = int(np.prod(shape))
    peak = cp.asarray(rng.uniform(.06, .2, (cells, 1, 1)), cp.float32)
    phase = cp.asarray(rng.uniform(0, 2 * math.pi, (cells, 1, 1)), cp.float32)
    angle = cp.arange(nd, dtype=cp.float32)[None, None, :] * np.float32(2 * math.pi / nd)
    sea = (.002 * cp.exp(-((frequency[None, :, None] - peak) / .04) ** 2) * (1.05 + cp.cos(angle - phase))
           * cp.asarray(rng.uniform(.5, 1.5, (cells, frequency.size, nd)), cp.float32))
    return sea, cp.asarray(rng.uniform(3, 20, shape), cp.float32)


def test_shared_solver_step_is_bitwise_repeatable():
    from woof.ocean.waves import create_solver
    cp = device()
    shape = (48, 64)
    latitude, longitude = cp.meshgrid(cp.linspace(-50, 50, shape[0], dtype=cp.float32),
                                      cp.linspace(-170, 170, shape[1], dtype=cp.float32), indexing="ij")
    grid = SimpleNamespace(ny=shape[0], nx=shape[1], latitude=latitude, longitude=longitude)
    solver = create_solver(grid, SimpleNamespace(depth=cp.full(shape, 1000, cp.float32),
                                                 wet=cp.ones(shape, cp.float32)), config={})
    sea, wind = _sea(cp, solver.frequency, shape, solver.nd, 11)
    solver.state[...] = sea
    solver.set_forcing(SimpleNamespace(u10=wind, v10=.3 * wind, rho_air=cp.full(shape, 1.225, cp.float32)))
    start = {name: array.copy() for name, array in solver.checkpoint_arrays().items()}
    results = []
    for _ in range(3):
        for name, array in solver.checkpoint_arrays().items():
            cp.copyto(array, start[name])
        solver.step(300.)
        results.append({name: array.copy() for name, array in solver.checkpoint_arrays().items()})
    assert float(cp.max(results[0]["bulk_outputs"][0])) > .1
    differing = {name: [_different(cp, results[0][name], other[name]) for other in results[1:]]
                 for name in results[0]}
    assert all(count == 0 for counts in differing.values() for count in counts), differing


def test_legacy_model_dia_is_bitwise_repeatable_and_conservative():
    from woof.ocean.waves.model import WaveModel
    cp = device()
    m = WaveModel(cp.full((40, 48), 1000, cp.float32), cp.linspace(20, 39.5, 40, dtype=cp.float32),
                  cp.linspace(-96, -72.5, 48, dtype=cp.float32))
    sea, wind = _sea(cp, m.frequency, (40, 48), m.nd, 5)
    m.spectrum[...] = sea.reshape(m.spectrum.shape)
    m._moments(wind, wind * 0, wind, wind * 0, 0)
    rates = []
    for limited in (True, True, False, False):
        m.rate.fill(0)
        m.dia.launch(m.spectrum, m.rate, None, m.frequency, m.df, m.depth, m.moments, m.nc,
                     dt=30, limited=limited)
        rates.append(m.rate.copy())
    assert _different(cp, rates[0], rates[1]) == 0 and _different(cp, rates[2], rates[3]) == 0
    energy = rates[2] * m.df[None, None, :, None]
    assert float(cp.sum(cp.abs(energy))) > 0
    assert abs(float(cp.sum(energy))) <= 2e-6 * float(cp.sum(cp.abs(energy)))
