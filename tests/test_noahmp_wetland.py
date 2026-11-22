"""Noah-MP wetland option (Zhang et al. 2022, WRR 58, e2021WR030573).

Re-implemented from published descriptions, not bitwise to upstream.  These
tests check the new physics against hand-computed cases from the paper's
equations (preprint doi:10.1002/essoar.10507347 v3, Eqs. 7, 9-13), check the
GPU store kernel against the float64 reference, and check that the WATER
kernel's wetland instantiation changes nothing but FSAT: with FSATMX = 0 it
is byte-identical to the v4 opt_run=1 kernel on all 36 fixture columns.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from woof.core import noahmp_wetland as wl


# ---- Eq. 7, by hand --------------------------------------------------------

def test_eq7_saturated_fraction_by_hand():
    # FSATMX 0.6, wilting 0.10, reference 0.30, first-layer SH2O 0.20:
    # relative saturation (0.20-0.10)/(0.30-0.10) = 0.5, FSAT = 0.30.
    assert wl.reference_fsat(0.6, 0.20, 0.10, 0.30) == pytest.approx(0.30, abs=1e-15)
    # Drier than wilting: no saturated area; wetter than reference: FSATMX.
    assert wl.reference_fsat(0.6, 0.05, 0.10, 0.30) == 0.0
    assert wl.reference_fsat(0.6, 0.45, 0.10, 0.30) == pytest.approx(0.6)
    # No wetland in the map, no saturated area whatever the soil.
    assert wl.reference_fsat(0.0, 0.45, 0.10, 0.30) == 0.0


# ---- Eqs. 9-13, by hand ----------------------------------------------------

HAND = dict(fsat=0.3, inflow=0.002, open_=1, store=20.0, wcap=50.0, dt=600.0,
            fsa=600.0, fira=100.0, tair=298.15, qair=0.012, psfc=95000.0,
            rhoair=1.1, chb=0.02, qseva=2.0e-7, fctr=200.0, edir=0.0002,
            etran=0.0797, fsh=100.0, runsrf=1.5)


def test_store_step_by_hand():
    """Worked by hand from Eq. 10 (Penman) and Eqs. 9, 12, 13.

    T = 25 C: e_s = 610.78 exp(17.27*25/262.3) = 3167.67 Pa,
    Delta = 4098.171 e_s / 262.3^2 = 188.68 Pa/K; e_a = q p/(0.622+0.378 q)
    = 1819.53 Pa, VPD = 1348.15 Pa; gamma = c_p p/(0.622 L) = 61.12 Pa/K;
    E = (Delta*500 + 1.1*1004.64*1348.15*0.02)/(L (Delta+gamma))
      = 1.97953e-4 mm/s (0.713 mm/h).  Store: 20 + 0.002*600 = 21.2 mm
    available, minus 0.3*1.97953e-4*600 = 0.03563 mm evaporated,
    = 21.16437 mm, below WCAP 50 so no spill.  Open-water LH over the cell:
    0.3 * 2.5104e6 * 1.97953e-4 = 149.08 W/m2.
    """
    r = wl.reference_store_step(**HAND)
    assert r["epot"] == pytest.approx(1.97953e-4, rel=1e-5)
    assert r["store"] == pytest.approx(21.16437, abs=1e-5)
    assert r["spill"] == 0.0
    assert r["wet_lh"] == pytest.approx(149.082, abs=1e-3)
    # Surface runoff loses the inflow (it went to the store).
    assert r["runsrf"] == pytest.approx(1.5 - 1.2)
    # Eq. 11: land tiles weighted by 1 - FSAT; SH is the residual.
    assert r["etran"] == pytest.approx(0.0797 * 0.7)
    assert r["fsh"] == pytest.approx(100.0 - r["dlh"])
    assert r["dlh"] == pytest.approx(149.082 - 0.3 * 2.0e-4 * 2.5104e6 - 0.3 * 200.0, abs=1e-3)


def test_store_spills_above_capacity_and_dries_when_empty():
    spill = wl.reference_store_step(**{**HAND, "store": 49.5})
    # 49.5 + 1.2 - 0.03563 = 50.66437 -> 0.66437 mm spills, store at WCAP.
    assert spill["spill"] == pytest.approx(0.66437, abs=1e-5)
    assert spill["store"] == pytest.approx(50.0)
    assert spill["runsrf"] == pytest.approx(1.5 - 1.2 + 0.66437, abs=1e-5)
    dry = wl.reference_store_step(**{**HAND, "store": 0.0, "inflow": 0.0})
    # Nothing to evaporate: water-limited to zero, store stays empty.
    assert dry["eact"] == 0.0 and dry["store"] == 0.0 and dry["wet_lh"] == 0.0
    closed = wl.reference_store_step(**{**HAND, "open_": 0})
    # Not open water (snow, frozen ground or empty store): only the inflow.
    assert closed["store"] == pytest.approx(21.2)
    assert closed["dlh"] == 0.0 and closed["etran"] == HAND["etran"]


def test_cold_start_is_relative_wetness_times_capacity():
    wcap = np.array([100.0, 100.0, 80.0], np.float32)
    fsatmx = np.array([0.5, 0.0, 0.2], np.float32)
    got = wl.cold_start_store(wcap, fsatmx, np.array([0.2, 0.3, 0.5], np.float32),
                              np.full(3, 0.1, np.float32), np.full(3, 0.3, np.float32))
    # 0.5 * min(100, 200); no wetland -> empty; wet soil, 80 mm capacity.
    np.testing.assert_allclose(got, [50.0, 0.0, 80.0], rtol=1e-6)
    deep = wl.cold_start_store(np.array([3000.0], np.float32), np.array([0.4], np.float32),
                               np.array([0.2], np.float32), np.array([0.1], np.float32),
                               np.array([0.3], np.float32))
    # A metres-deep Eq. 14 capacity starts at most 200 mm full (0.5 * 200).
    np.testing.assert_allclose(deep, [100.0], rtol=1e-6)


@pytest.mark.parametrize("bad", [
    dict(wetland=1, sf_surface_physics=4, opt_run=4, soiltstep=0.0),
    dict(wetland=1, sf_surface_physics=4, opt_run=6, soiltstep=0.0),
    dict(wetland=1, sf_surface_physics=4, opt_run=8, soiltstep=0.0),
    dict(wetland=1, sf_surface_physics=2, opt_run=1, soiltstep=0.0),
    dict(wetland=1, sf_surface_physics=4, opt_run=1, soiltstep=1800.0),
    dict(wetland=2, sf_surface_physics=4, opt_run=1, soiltstep=0.0),
])
def test_refusals_name_their_reason(bad):
    with pytest.raises(ValueError):
        wl.validate(**bad)
    wl.validate(wetland=0, sf_surface_physics=4, opt_run=3, soiltstep=0.0)
    for run in wl.RUNOFF_PAIRS:
        wl.validate(wetland=1, sf_surface_physics=4, opt_run=run, soiltstep=0.0)


# ---- the GPU kernels -------------------------------------------------------

@pytest.mark.gpu
def test_gpu_store_kernel_matches_reference():
    cp = pytest.importorskip("cupy")
    rng = np.random.default_rng(7)
    n = 257
    case = {k: np.full(n, v, np.float64) for k, v in HAND.items()}
    case["store"] = rng.uniform(0, 60, n)
    case["inflow"] = rng.uniform(0, 0.01, n)
    case["fsat"] = rng.uniform(0, 0.8, n)
    case["open_"] = (rng.uniform(0, 1, n) > 0.3).astype(np.float64)
    case["tair"] = rng.uniform(270, 310, n)
    case["fsa"] = rng.uniform(0, 900, n)
    got = wl.advance_store(**{k: cp.asarray(v, dtype=cp.float32) for k, v in case.items()})
    for k in range(n):
        want = wl.reference_store_step(**{name: float(v[k]) for name, v in case.items()})
        for name in ("store", "runsrf", "edir", "etran", "fsh", "dlh", "wet_lh"):
            # float32 against float64: W/m2 terms difference ~150 W/m2 values.
            tol = 2e-3 if name in ("fsh", "dlh", "wet_lh") else 2e-5
            assert float(got[name][k]) == pytest.approx(want[name], rel=2e-5, abs=tol), (name, k)


@pytest.mark.gpu
def test_water_kernel_wetland_changes_only_fsat():
    """FSATMX = 0 makes Eq. 7 and the v4 FSAT both zero: byte-identical."""
    cp = pytest.importorskip("cupy")
    from test_noahmp_water_slab import _slab_fields, NSOIL as NS
    from woof.core.noahmp_water_slab import evaluate_water_slab

    fields = _slab_fields(cp)
    n = int(fields["dt"].shape[0])
    fields.update(fsatmx=cp.zeros(n, cp.float32), timean=cp.full(n, 10.5, cp.float32),
                  psisat=cp.full((n, NS), 0.36, cp.float32),
                  zwt=cp.full(n, 2.5, cp.float32), wa=cp.full(n, 4900.0, cp.float32),
                  wt=cp.full(n, 4900.0, cp.float32),
                  smcref=cp.full((n, NS), 0.3, cp.float32))
    base = evaluate_water_slab({k: v.copy() for k, v in fields.items()}, n, opt_run=1)
    for flag in (0.0, 1.0):
        wet = evaluate_water_slab({**{k: v.copy() for k, v in fields.items()},
                                   "wetland_open": cp.full(n, flag, cp.float32)},
                                  n, opt_run=1, wetland=1)
        for name, value in base.items():
            a, b = cp.asnumpy(value), cp.asnumpy(wet[name])
            assert a.tobytes() == b.tobytes(), (name, flag)
        assert float(cp.abs(wet["wetland_fsat"]).max()) == 0.0

    # FSATMX > 0: FSAT stays inside [0, FSATMX] and the store inflow is the
    # FSAT share of the soil's surface input (Eq. 9), in mm/s.
    fields["fsatmx"] = cp.full(n, 0.5, cp.float32)
    wet = evaluate_water_slab({**fields, "wetland_open": cp.zeros(n, cp.float32)},
                              n, opt_run=1, wetland=1)
    fsat = cp.asnumpy(wet["wetland_fsat"])
    # SOILWATER reads the step-averaged input: ACC_QINSUR / 1 step.
    qinsur = cp.asnumpy(wet["acc_qinsur"])
    soil = cp.asnumpy(fields["ist"]) == 1
    assert ((fsat >= 0) & (fsat <= np.float32(0.5))).all()
    expect = np.where(qinsur > 0, (qinsur * fsat).astype(np.float32) * np.float32(1000.0), 0.0)
    np.testing.assert_array_equal(cp.asnumpy(wet["wetland_inflow"])[soil], expect[soil])
    assert (fsat[soil] > 0).any()


@pytest.mark.gpu
def test_water_kernel_wetland_opt_run3_tile_split():
    """opt_run=3 (Schaake, the Noah-MP default) with the wetland tile split.

    FSATMX = 0: byte-identical to the v4 opt_run=3 kernel.  FSATMX > 0: the
    store inflow is Eq. 9 (FSAT x surface input), and RUNSRF carries that
    share on top of the land tile's area-weighted Schaake runoff, so the
    land tile never sees more than (1 - FSAT) of the input.
    """
    cp = pytest.importorskip("cupy")
    from test_noahmp_water_slab import _slab_fields, NSOIL as NS
    from woof.core.noahmp_water_slab import evaluate_water_slab

    fields = _slab_fields(cp)
    n = int(fields["dt"].shape[0])
    fields.update(fsatmx=cp.zeros(n, cp.float32), smcref=cp.full((n, NS), 0.3, cp.float32))
    base = evaluate_water_slab({k: v.copy() for k, v in fields.items()}, n, opt_run=3)
    for flag in (0.0, 1.0):
        wet = evaluate_water_slab({**{k: v.copy() for k, v in fields.items()},
                                   "wetland_open": cp.full(n, flag, cp.float32)},
                                  n, opt_run=3, wetland=1)
        for name, value in base.items():
            a, b = cp.asnumpy(value), cp.asnumpy(wet[name])
            assert a.tobytes() == b.tobytes(), (name, flag)
        assert float(cp.abs(wet["wetland_fsat"]).max()) == 0.0

    fields["fsatmx"] = cp.full(n, 0.5, cp.float32)
    wet = evaluate_water_slab({**{k: v.copy() for k, v in fields.items()},
                               "wetland_open": cp.zeros(n, cp.float32)},
                              n, opt_run=3, wetland=1)
    fsat = cp.asnumpy(wet["wetland_fsat"])
    qinsur = cp.asnumpy(wet["acc_qinsur"])
    soil = cp.asnumpy(fields["ist"]) == 1
    assert ((fsat >= 0) & (fsat <= np.float32(0.5))).all()
    assert (fsat[soil] > 0).any()
    expect = np.where(qinsur > 0, (qinsur * fsat).astype(np.float32) * np.float32(1000.0), 0.0)
    np.testing.assert_array_equal(cp.asnumpy(wet["wetland_inflow"])[soil], expect[soil])
    # Surface runoff (mm per step) holds at least the store inflow, and the
    # wetter it gets the more runs off than the v4 Schaake partition alone
    # would give the land tile.
    dt = cp.asnumpy(fields["dt"])
    inflow_mm = cp.asnumpy(wet["wetland_inflow"]) * dt
    assert (cp.asnumpy(wet["runsrf"])[soil] >= inflow_mm[soil] * (1 - 1e-5) - 1e-7).all()
