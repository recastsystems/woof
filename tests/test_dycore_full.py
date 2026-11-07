import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import requires_gpu
from woof.config import RunConfig

pytestmark = pytest.mark.gpu


@requires_gpu
def test_stability_cfl_is_co_located_and_threshold_continuous():
    import cupy as cp
    from woof.core import constants as c
    from woof.core.dycore import stability_gate_failed, stability_report

    class SyntheticState:
        def __init__(self, w, z):
            nz = len(z) - 1
            self.u = cp.zeros((nz, 1, 2), dtype=cp.float32)
            self.w = cp.asarray(w, dtype=cp.float32).reshape(nz + 1, 1, 1)
            self.thp = cp.zeros((nz, 1, 1), dtype=cp.float32)
            self.php = (
                cp.asarray(z, dtype=cp.float32) * cp.float32(c.G)
            ).reshape(nz + 1, 1, 1)
            self.phb = cp.zeros(nz + 1, dtype=cp.float32)
            self.buffers = {}

        def scratch(self, shape, slot):
            return self.buffers.setdefault(
                slot, cp.zeros(shape, dtype=cp.float32))

    run = SimpleNamespace(dt=10.0, dx=1000.0)

    # Strong updraft aloft, over its own thick layer: old global extrema
    # produced CFL=100; the co-located vertical term is exactly 1.
    aloft = stability_report(
        SyntheticState([0.0, 0.0, 100.0], [0.0, 10.0, 1010.0]), run)
    assert aloft["vertical_cfl"] == pytest.approx(1.0)
    assert not stability_gate_failed(
        aloft, max_cfl=10.0, max_w_ms=150.0)

    # A genuine first-layer violation remains fatal.
    surface = stability_report(
        SyntheticState([0.0, 11.0, 0.0], [0.0, 10.0, 1010.0]), run)
    assert surface["vertical_cfl"] == pytest.approx(11.0)
    assert stability_gate_failed(
        surface, max_cfl=10.0, max_w_ms=150.0)

    # The existing strict-greater-than threshold is continuous.
    at_limit = stability_report(
        SyntheticState([0.0, 10.0, 0.0], [0.0, 10.0, 1010.0]), run)
    assert at_limit["cfl"] == pytest.approx(10.0)
    assert not stability_gate_failed(
        at_limit, max_cfl=10.0, max_w_ms=150.0)
    above = stability_report(
        SyntheticState(
            [0.0, np.nextafter(np.float32(10.0), np.float32(np.inf)), 0.0],
            [0.0, 10.0, 1010.0]),
        run)
    assert above["cfl"] > 10.0
    assert stability_gate_failed(
        above, max_cfl=10.0, max_w_ms=150.0)


def _theta_centroid_height(s, b, vc):
    """theta'-weighted mean of the base-state half-level heights (m)."""
    import cupy as cp
    thp = np.clip(cp.asnumpy(s.thp).astype(np.float64), 0.0, None)
    z = (0.5 * (b.phb[:-1] + b.phb[1:]) / 9.81)[:, None, None]
    return float((thp * z).sum() / thp.sum())


def _rest_setup(nx=64, nz=32, N=0.01):
    # constant Brunt-Vaisala frequency N: theta(z) = 300*exp(N^2 z / g)
    from woof.core.grid import make_vertical_coord, make_base_state
    cfg = RunConfig(nx=nx, ny=1, nz=nz, dx=1000.0, dy=1000.0, ztop=10000.0,
                    dt=6.0, run_seconds=0.0)
    vc = make_vertical_coord(nz)
    th = lambda z: 300.0 * np.exp(N * N * np.asarray(z, float) / 9.81)
    b = make_base_state(vc, th, p_surf=cfg.p_surf, ztop=cfg.ztop)
    return cfg, vc, b


@requires_gpu
def test_at_rest_stays_at_rest():
    import cupy as cp
    from woof.core.state import init_at_rest
    from woof.core.dycore import run_steps, stability_report
    cfg, vc, b = _rest_setup()
    s = init_at_rest(cfg, vc, b)
    run_steps(s, cfg, n=500)                     # 3000 s
    r = stability_report(s, cfg)
    assert not r["nan"]
    assert r["cfl"] is not None and r["cfl"] < 0.5   # real CFL value exercised
    assert r["w_max"] < 1e-3                     # FP32 round-off scale only
    assert float(cp.abs(s.thp).max()) < 1e-2


@requires_gpu
def test_warm_bubble_rises():
    import cupy as cp
    from woof.core.state import init_theta_perturbation
    from woof.core.dycore import run_steps
    cfg, vc, b = _rest_setup(nx=100, nz=50)
    cfg = dataclasses.replace(cfg, **{"dx": 200.0, "dy": 200.0, "dt": 1.0})
    def bubble(x, z):
        L = np.sqrt((x[None, None, :] / 2000.0) ** 2
                    + ((z[:, None, None] - 2000.0) / 2000.0) ** 2)
        return np.where(L < 1.0, 2.0 * np.cos(np.pi * L / 2) ** 2, 0.0) \
                 * np.ones((cfg.nz, cfg.ny, cfg.nx))
    s = init_theta_perturbation(cfg, vc, b, bubble)
    zc0 = _theta_centroid_height(s, b, vc)
    run_steps(s, cfg, n=600)                     # 600 s
    zc1 = _theta_centroid_height(s, b, vc)
    assert zc1 > zc0 + 500.0                     # bubble clearly rising
    from woof.core.dycore import stability_report
    assert not stability_report(s)["nan"]
