"""Rain a cumulus scheme feeds two-moment Thompson gets a rain number.

Kain-Fritsch hands its fed-back precipitation to the grid as rain mass with
no number; classic Thompson then clamps the drops at 2.5 mm, and a few
mg/kg of shallow-cumulus drizzle read 20-30 dBZ (WOOF Mexico City 29 Oct
12Z: echo layers 0-1.2 km deep, 4-10 mg/kg of rain, straight strips along
the parent's edges and a frame around the nest).
:func:`woof.core.microphysics.seed_cumulus_rain_number` seeds each step's
fed rain with real.exe's ``make_RainNumber``.
"""
from types import SimpleNamespace

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
try:
    cp.cuda.runtime.getDeviceCount()
except Exception:  # pragma: no cover - CPU-only host
    pytest.skip("needs a CUDA device", allow_module_level=True)

import woof.wrf_exact
from woof.core import constants as c
from woof.core import microphysics as mp
from woof.core.thompson_entry import make_rain_number

DT = 15.0
#: Pressure (Pa) and full theta (K) of four levels: warm, the 0 to -2 C
#: intercept ramp, and cold.
P = np.array([90000.0, 70000.0, 70000.0, 60000.0], np.float32)
THETA = np.array([300.0, 301.3, 300.0, 300.0], np.float32)


def _state(rqr, nr0=0.0):
    nz = len(P)
    shape = (nz, 2, 3)
    thb = np.full(nz, 290.0, np.float32)
    thp = np.broadcast_to((THETA - thb)[:, None, None], shape).copy()
    p = np.broadcast_to(P[:, None, None], shape).copy()
    state = SimpleNamespace(
        thb=cp.asarray(thb), thp=cp.asarray(thp), p=cp.asarray(p),
        nr=cp.full(shape, np.float32(nr0)),
        physics=SimpleNamespace(cu_rates={"rqrcuten": cp.asarray(rqr)}))
    return state, p


def _rates():
    rqr = np.zeros((len(P), 2, 3), np.float32)
    rqr[:, 0, 0] = 4.0e-8          # shallow Kain-Fritsch drizzle feed
    rqr[:, 0, 1] = 1.0e-6
    rqr[:, 1, 0] = 1.0e-11         # under Morrison's 1e-10 threshold
    return rqr


def _expected(rqr, p, nr0=0.0):
    temp = (np.broadcast_to(THETA[:, None, None], rqr.shape)
            * np.power(p / np.float32(c.P0), np.float32(c.RCP))).astype(np.float32)
    rho = (p / (np.float32(c.RD) * temp)).astype(np.float32)
    fed = rqr >= np.float32(1.0e-10)
    q = (np.where(fed, rqr, np.float32(1.0)) * np.float32(DT)).astype(np.float32) * rho
    number = np.asarray(make_rain_number(q, temp), np.float32) / rho
    return np.where(fed, np.float32(nr0) + number,
                    np.float32(nr0)).astype(np.float32), rho


def test_fed_rain_takes_make_rain_number():
    rqr = _rates()
    state, p = _state(rqr, nr0=5.0)
    cfg = SimpleNamespace(mp_physics=8, cu_physics=1)
    assert mp.seed_cumulus_rain_number(state, cfg, DT) is True
    expected, _ = _expected(rqr, p, nr0=5.0)
    got = cp.asnumpy(state.nr)
    np.testing.assert_allclose(got, expected, rtol=2e-5)
    assert (got[:, 1, 0] == 5.0).all() and (got[:, 1, 1:] == 5.0).all()


def test_fed_drizzle_is_no_longer_2_5_mm_drops():
    rqr = _rates()
    state, p = _state(rqr)
    mp.seed_cumulus_rain_number(state, SimpleNamespace(mp_physics=28, cu_physics=1), DT)
    _, rho = _expected(rqr, p)
    mass = rqr[0, 0, 0] * DT * rho[0, 0, 0]                 # kg m-3
    number = float(state.nr[0, 0, 0]) * rho[0, 0, 0]        # m-3
    d_mv = (6.0 * mass / (np.pi * 1000.0 * number)) ** (1.0 / 3.0)
    # Unseeded, Thompson takes this mass as 2.5 mm drops (its clamp).
    assert d_mv < 0.25e-3


@pytest.mark.parametrize("mp_physics, cu_physics", [(10, 1), (6, 1), (8, 0)])
def test_other_schemes_and_no_cumulus_untouched(mp_physics, cu_physics):
    state, _ = _state(_rates(), nr0=3.0)
    cfg = SimpleNamespace(mp_physics=mp_physics, cu_physics=cu_physics)
    assert mp.seed_cumulus_rain_number(state, cfg, DT) is False
    assert (cp.asnumpy(state.nr) == 3.0).all()


def test_wrf_exact_keeps_wrf(monkeypatch):
    monkeypatch.setattr(woof.wrf_exact, "ENABLED", True)
    state, _ = _state(_rates(), nr0=3.0)
    assert mp.seed_cumulus_rain_number(
        state, SimpleNamespace(mp_physics=8, cu_physics=1), DT) is False
    assert (cp.asnumpy(state.nr) == 3.0).all()


def test_one_level_slabs_seed_the_same(monkeypatch):
    rqr = _rates()
    monkeypatch.setattr(mp, "_SEED_SLAB_CELLS", 1)
    state, p = _state(rqr, nr0=5.0)
    mp.seed_cumulus_rain_number(state, SimpleNamespace(mp_physics=8, cu_physics=1), DT)
    expected, _ = _expected(rqr, p, nr0=5.0)
    np.testing.assert_allclose(cp.asnumpy(state.nr), expected, rtol=2e-5)
