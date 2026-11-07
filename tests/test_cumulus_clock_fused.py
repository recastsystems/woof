"""The fused cumulus clock writes exactly what the array expressions wrote.

``PhysicsDriver._advance_cumulus_clock`` and ``finish_step`` take one launch
each on CuPy arrays (``woof.core.cumulus_clock``) and keep their CuPy array
expressions for any other carrier.  This runs the driver's own two paths on
the same inputs and compares every written array as uint32, over inputs built
to reach each branch: negative, signed-zero, below-normal-range, infinite and
NaN rain rates, NCA values on and next to the expiry and hold boundaries, and
held volumes whose columns do and do not expire.
"""
from __future__ import annotations

from types import SimpleNamespace
from _run_config_fake import run_config_fake

import numpy as np
import pytest

from conftest import requires_gpu

_RATES = ("rthcuten", "rqvcuten", "rqccuten", "rqicuten", "rqrcuten",
          "rqscuten")
_COUPLED = ("rtheta", "rqv", "rqc", "rqi", "rqr", "rqs")


def _inputs(rng, ny, nx, nz, dt):
    f32 = np.float32
    n = ny * nx
    special = np.array(
        [0.0, -0.0, 1.0e-40, -1.0e-40, 3.0e-4, -3.0e-4, np.inf, -np.inf,
         np.nan, 1.0e30, -1.0e30, 1.0e-3], dtype=f32)
    pratec = rng.standard_normal(n).astype(f32) * f32(1.0e-3)
    pratec[:special.size] = special
    step = f32(dt)
    edges = []
    for k in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
        centre = f32(k) * step
        edges += [np.nextafter(centre, f32(-np.inf)), centre,
                  np.nextafter(centre, f32(np.inf))]
    nca_special = np.array(
        [-100.0, 0.0, -0.0, 1.0e-40, np.nan, np.inf, -np.inf, 1.0e30]
        + edges, dtype=f32)
    nca = rng.uniform(-200.0, 5.0 * dt, n).astype(f32)
    nca[:nca_special.size] = nca_special
    nca[n // 2:n // 2 + nca_special.size] = nca_special
    surface = {name: rng.standard_normal(n).astype(f32) * f32(10.0)
               for name in ("rainc", "rainbl", "raincv")}
    for array in surface.values():
        array[3] = f32(-0.0)
    volumes = {name: rng.standard_normal((nz, n)).astype(f32)
               for name in _RATES + _COUPLED}
    for array in volumes.values():
        array[:, 5] = f32(-0.0)
    shape2 = (ny, nx)
    return (pratec.reshape(shape2), nca.reshape(shape2),
            {k: v.reshape(shape2) for k, v in surface.items()},
            {k: v.reshape(nz, ny, nx) for k, v in volumes.items()})


def _driver(cp, pratec, nca, surface, volumes, *, land, coupled):
    from woof.core.physics import PhysicsDriver

    driver = object.__new__(PhysicsDriver)
    driver.ruc_params = None
    driver.noahmp_params = object() if land else None
    driver.fields = {"surface_raincv": cp.asarray(surface["raincv"])}
    driver.rainc = cp.asarray(surface["rainc"])
    driver._pending_rainbl = cp.asarray(surface["rainbl"])
    driver.cu_pratec = cp.asarray(pratec)
    driver.cu_nca = cp.asarray(nca)
    driver.cu_expiring = cp.zeros(nca.shape, dtype=cp.float32)
    driver._cu_expiry_pending = False
    driver.cu_rates = {name: cp.asarray(volumes[name]) for name in _RATES}
    driver.cumulus_tendencies = SimpleNamespace(**{
        name: (cp.asarray(volumes[name]) if name in coupled else None)
        for name in _COUPLED})
    return driver


def _written(cp, driver):
    out = {"rainc": driver.rainc, "rainbl": driver._pending_rainbl,
           "raincv": driver.fields["surface_raincv"], "nca": driver.cu_nca,
           "expiring": driver.cu_expiring}
    out.update({f"rate_{k}": v for k, v in driver.cu_rates.items()})
    out.update({f"coupled_{k}": getattr(driver.cumulus_tendencies, k)
                for k in _COUPLED
                if getattr(driver.cumulus_tendencies, k) is not None})
    return {k: cp.asnumpy(v).view(np.uint32) for k, v in out.items()}


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("scheme", [1, 99])
@pytest.mark.parametrize("land", [False, True])
@pytest.mark.parametrize("coupled", [_COUPLED, ("rtheta", "rqv", "rqc"), ()])
@pytest.mark.parametrize("dt", [15.0, 24.0, 60.0])
def test_fused_clock_is_bitwise_the_array_expressions(monkeypatch, land,
                                                       coupled, dt, scheme):
    cp = pytest.importorskip("cupy")
    from woof.core import cumulus_clock

    rng = np.random.default_rng(20260930)
    inputs = _inputs(rng, 37, 53, 7, dt)
    cfg = run_config_fake(dt=dt, clock_dt=0.0, cu_physics=scheme, cudt_minutes=5.0)
    state = SimpleNamespace(elapsed_seconds=0.0)

    fused = _driver(cp, *inputs, land=land, coupled=coupled)
    launches = []
    real_advance, real_clear = cumulus_clock.advance, cumulus_clock.clear
    monkeypatch.setattr(cumulus_clock, "advance",
                        lambda **k: (launches.append("advance"),
                                     real_advance(**k))[1])
    monkeypatch.setattr(cumulus_clock, "clear",
                        lambda *a: (launches.append("clear"),
                                    real_clear(*a))[1])
    for _ in range(3):
        fused._advance_cumulus_clock(state, cfg)
    assert launches == ["advance", "clear"] * 3
    assert fused._cu_expiry_pending is False

    monkeypatch.setattr(cumulus_clock, "device_arrays", lambda *a: False)
    reference = _driver(cp, *inputs, land=land, coupled=coupled)
    for _ in range(3):
        reference._advance_cumulus_clock(state, cfg)
    assert reference._cu_expiry_pending is False

    got, want = _written(cp, fused), _written(cp, reference)
    assert sorted(got) == sorted(want)
    for name in want:
        differing = int(np.count_nonzero(got[name] != want[name]))
        assert differing == 0, f"{name}: {differing} words differ"
    # The inputs reach both branches: some columns expired and were cleared,
    # and some held columns kept their rates.
    expired = cp.asnumpy(reference.cu_rates["rthcuten"])[0] == 0.0
    assert expired.any() and not expired.all()
