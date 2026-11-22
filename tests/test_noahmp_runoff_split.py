"""Separate surface/subsurface runoff options (opt_runsrf / opt_runsub).

Re-implemented from published descriptions, not bitwise to upstream.

1. The split kernel at a stock pair (N, N) is bit-identical to the stock
   selected kernel at opt_run=N, for every N and both opt_inf, on the 36
   heterogeneous WATER fixture columns.  This proves the generator moved
   gates only: with SRF == SUB every gate is the original one.
2. A mixed pair's subsurface half reproduces a hand computation written
   from the published equations: the equilibrium water table (He et al.
   2023 tech note Eqs. 3.7.322-3.7.327) and the TOPMODEL baseflow
   (Eq. 3.7.328) under VIC surface runoff.
3. Pair resolution keeps stock namelists' meaning, and every refusal names
   its breakage.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from conftest import requires_gpu

from woof.core import noahmp_runoff_split as split

REPO = Path(__file__).resolve().parents[1]
NSOIL = 4


def test_generated_split_kernel_is_current():
    subprocess.run([sys.executable, "-I", str(REPO / "tools/noahmp_runoff_split/generate.py"), "--check"],
                   check=True)


@pytest.mark.parametrize("opt_run", split.RUNOFF_OPTIONS)
def test_a_stock_namelist_keeps_its_meaning(opt_run):
    assert split.resolve(opt_run) == (opt_run, opt_run)
    assert split.canonical(*split.resolve(opt_run)) == (opt_run, 0)
    assert split.resolve(opt_run, opt_run, opt_run) == (opt_run, opt_run)
    assert split.carrier_run(opt_run, 0) == opt_run


def test_wrf48_pair_resolution():
    assert split.resolve(3, 6, 1) == (6, 1)
    assert split.resolve(1, 6, 0) == (6, 1)       # a missing half inherits opt_run
    assert split.canonical(6, 3) == (6, 0)        # subsurface 3/6/7/8 is one drainage scheme
    assert split.canonical(3, 7) == (3, 0)
    assert split.canonical(4, 3) == (4, 3)        # BATS drainage differs (Eq. 3.7.255)
    assert split.canonical(6, 1) == (6, 1)
    assert split.carrier_run(6, 1) == 1 and split.carrier_run(3, 5) == 5 and split.carrier_run(2, 4) == 2
    assert split.carrier_run(4, 3) == 4
    assert split.transferred_names(6, 2) == ("BVIC", "FSATMX", "TIMEAN")
    assert split.transferred_names(3, 5) == ("FSATMX",)
    with pytest.raises(split.RunoffSplitRefused):
        split.resolve(3, 9, 0)


@pytest.mark.parametrize("pair,word", [((1, 3), "cold-start ZWT"), ((5, 3), "MMF water table"),
                                       ((2, 5), "wrong sign"), ((1, 5), "Niu 2007")])
def test_refusals_name_their_breakage(pair, word):
    with pytest.raises(split.RunoffSplitRefused, match=word):
        split.validate(*pair)


def test_tile_drainage_and_dvic_refusals():
    with pytest.raises(split.RunoffSplitRefused, match="silently not drain"):
        split.validate(6, 1, opt_tdrn=1)
    with pytest.raises(split.RunoffSplitRefused, match="FSUR"):
        split.validate(8, 1, opt_infdv=0)
    split.validate(8, 1, opt_infdv=2)
    split.validate(3, 3, opt_tdrn=2)              # stock pairs never refuse here


# ---------------------------------------------------------------------------
# GPU: the fixture columns plus every option's parameters and carriers.
# ---------------------------------------------------------------------------

def _fields(cp, *, mmf=False):
    from test_noahmp_water import CASES, TABLE
    from test_noahmp_water_slab import _slab_fields

    f = _slab_fields(cp)
    n = len(CASES)
    p = [TABLE[(c, "param")] for c in CASES]
    col = np.arange(n, dtype=np.float32)
    one = lambda v: cp.asarray(np.asarray(v, np.float32).reshape(n))
    lay = lambda v: cp.asarray(np.asarray(v, np.float32).reshape(n, NSOIL))
    f.update(
        fsatmx=one([r[("fsatmx", 0)] for r in p]), timean=one([r[("timean", 0)] for r in p]),
        # Typical MPTABLE magnitudes; column-varying so a mis-addressed tail
        # cannot agree by accident.
        psisat=lay(np.outer(0.05 + 0.01 * col, np.ones(NSOIL)) + np.array([0.0, 0.01, 0.02, 0.03])),
        bvic=one(0.05 + 0.005 * col), axaj=one(0.009 + 0.0005 * col), bxaj=one(0.2 + 0.01 * col),
        xxaj=one(0.15 + 0.002 * col), bbvic=one(1.0 + 0.02 * col), bdvic=one(0.2 + 0.003 * col),
        gdvic=one(0.18 + 0.002 * col),
        smcref=lay(cp.asnumpy(f["smcmax"]) * np.float32(0.7)),
        zwt=one((-2.0 - 0.05 * col) if mmf else (2.5 + 0.05 * col)),
        wa=one(4800.0 + col), wt=one(4850.0 + col),
        smcwtd=one(0.3 + 0.001 * col), deeprech=one(0.0 * col), rech=one(0.0 * col),
        smceq=lay(cp.asnumpy(f["smcmax"]) * np.float32(0.8)))
    return f, n


def _bits(a):
    a = np.ascontiguousarray(np.asarray(a))
    return a.view(np.uint32 if a.dtype.itemsize == 4 else np.uint8)


@requires_gpu
@pytest.mark.parametrize("opt_inf", (1, 2))
@pytest.mark.parametrize("opt_run", split.RUNOFF_OPTIONS)
def test_stock_pair_through_the_split_kernel_is_bit_identical(opt_run, opt_inf):
    import cupy as cp
    from woof.core import noahmp_water_slab as slab

    f, n = _fields(cp, mmf=opt_run == 5)
    infdv = 1 if opt_run == 8 else 0
    # Force the stock selected kernel even at opt_run=3 (the default image is
    # a different, separately measured unit).
    want = slab.evaluate_water_slab(dict(f), n, opt_inf=opt_inf, opt_run=opt_run, opt_infdv=infdv,
                                    soil_update_steps=2 if opt_run == 3 else 1)
    got = slab._evaluate_split(dict(f), n, opt_inf=opt_inf, opt_run=opt_run, opt_runsub=opt_run,
                               opt_infdv=infdv, soil_update_steps=2 if opt_run == 3 else 1,
                               calculate_soil=True, opt_irr=0, opt_tdrn=0)
    assert set(got) == set(want)
    bad = [k for k in want if not np.array_equal(_bits(cp.asnumpy(got[k])), _bits(cp.asnumpy(want[k])))]
    assert not bad, f"opt_run={opt_run} opt_inf={opt_inf}: split kernel differs in {bad}"


def _hand_equilibrium(smcmax1, bexp1, psisat1, zsoil, dz, sh2o):
    """Tech note Eqs. 3.7.322-3.7.327, written from the description, float32."""
    f32 = np.float32
    deficit = f32(0.0)
    for k in range(NSOIL):
        deficit = f32(deficit + f32(f32(smcmax1 - sh2o[k]) * dz[k]))
    fine = f32(f32(f32(3.0) * f32(-zsoil[-1])) / f32(100.0))
    zwt = f32(f32(f32(-3.0) * zsoil[-1]) - f32(0.001))
    acc = f32(0.0)
    for k in range(1, 101):
        depth = f32(f32(k) * fine)
        temp = f32(f32(1.0) + f32(f32(zwt - depth) / psisat1))
        acc = f32(acc + f32(f32(smcmax1 * f32(f32(1.0) - f32(np.power(temp, f32(f32(-1.0) / bexp1))))) * fine))
        if abs(f32(acc - deficit)) <= f32(0.01):
            return depth
    return zwt


@requires_gpu
@pytest.mark.parametrize("srf", (3, 6))
def test_topmodel_baseflow_under_another_surface_scheme_matches_the_paper(srf):
    """Mixed (srf, 2): RUNSUB = (1-FCRMAX) * 4 * exp(-TIMEAN) * exp(-2 ZWT_eq), Eq. 3.7.328.

    Columns are kept where nothing else moves RUNSUB: no snowpack (SNOFLOW
    is zero), soil water well above the WATMIN floor (Eq. 3.7.331-3.7.335
    adds nothing), one soil step.  The equilibrium ZWT comes from the
    soil liquid water after the saturation-excess clip, Eq. 3.7.191.
    """
    import cupy as cp
    from woof.core import noahmp_water_slab as slab

    f, n = _fields(cp)
    host = {k: cp.asnumpy(v) for k, v in f.items()}
    got = slab._evaluate_split(dict(f), n, opt_inf=1, opt_run=srf, opt_runsub=2, opt_infdv=0,
                               soil_update_steps=1, calculate_soil=True, opt_irr=0, opt_tdrn=0)
    runsub = cp.asnumpy(got["runsub"])
    zwt = cp.asnumpy(got["zwt"])
    f32 = np.float32
    checked = 0
    for c in range(n):
        if (host["isnow"][c] != 0 or host["ist"][c] != 1 or host["sneqv"][c] > 0
                or host["frozen_ground"][c]):
            continue
        sh2o, sice = host["sh2o"][c].astype(f32), host["sice"][c].astype(f32)
        smcmax = host["smcmax"][c].astype(f32)
        zsoil = host["zsoil"][c].astype(f32)
        dz = np.diff(np.concatenate(([f32(0)], -zsoil))).astype(f32)
        if np.min(sh2o * dz * 1000) < 5.0:     # far from the 0.01 mm floor
            continue
        epore = np.maximum(f32(1e-4), smcmax - sice)
        clipped = np.minimum(epore, sh2o)
        fice = np.minimum(f32(1.0), sice / smcmax)
        fcr = np.maximum(f32(0), np.exp(-f32(4.0) * (f32(1) - fice)) - np.exp(f32(-4.0))) / (f32(1) - np.exp(f32(-4.0)))
        fcrmax = f32(np.max(fcr))
        zeq = _hand_equilibrium(smcmax[0], host["bexp"][c][0].astype(f32), host["psisat"][c][0].astype(f32),
                                zsoil, dz, clipped)
        rsub = f32(f32(f32(f32(1) - fcrmax) * f32(4.0)) * np.exp(-host["timean"][c].astype(f32))) * np.exp(f32(-2.0) * zeq)
        dt = host["dt"][c].astype(f32)
        assert zwt[c] == pytest.approx(zeq, rel=1e-6, abs=1e-6), f"column {c}: ZWT"
        assert runsub[c] == pytest.approx(rsub * dt, rel=2e-5), f"column {c}: RUNSUB"
        checked += 1
    assert checked >= 5, f"only {checked} columns qualify"
