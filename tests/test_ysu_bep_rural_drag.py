"""The declared rural-drag divergence of YSU's flag_bep arm (sf_urban_physics 2/3).

THE WRF DEFECT.  Under BEP or BEP+BEM, WRF v4.7.1's YSU
(``phys/physics_mmm/bl_ysu.F90:1311-1314``) keeps ``(1-frc)*fric``, the rural
share of its own surface drag, on the first-level diagonal and removes only
the urban share.  The BEP couple (``module_sf_noahdrv.F:1708-1711``,
``module_sf_noahmpdrv.F:3718-3721``) has already folded the same rural drag,
``(1-frc)*(-ust*ust)/dz8w/|U|``, into ``a_u_bep``/``a_v_bep`` at level 1, and
``bl_ysu.F90:1359-1368`` subtracts ``a_u*dt2`` from that same diagonal.  So
the rural surface drag is applied twice, on every column that is not wholly
urban: the ocean, farmland, parks.  Heat and moisture are counted once (YSU
drops its own flux with ``(1-bepswitch)*hfx`` and takes the rural flux from
``b_t_bep``), and WRF's MYJ path under BEP (``myjurb``'s VDIFV) takes the
surface drag only through ``a_u``, so neither has the defect.

WHAT IT DID TO A FORECAST.  On the 2026-09-29 08Z HRRR cycle at 750 m, BEP
under YSU slowed the 10 m wind over the open Pacific, where there are no
buildings, by 1.25 m/s within two hours off San Francisco (0.34 m/s off Los
Angeles), and rural land winds by 0.3 to 1.8 m/s; the MYJ arm moved the
ocean wind by +0.03 m/s.

WHAT GPUWM DOES.  ``kernels/ysu.cu`` removes the WHOLE of YSU's own
first-level drag under BEP (WRF's line without the ``frc_urb1d`` factor), so
the surface drag enters once, through ``a_u_bep``, the way heat already
enters through ``b_t_bep``.  One expression under ``if constexpr (BEP)``:
``ysu_column`` (urban off) compiles from unchanged statements.  WRF with
exactly that one-line change is the oracle ``tests/test_ysu_bep_wrf471_
parity.py`` grades the port against (``build_ysu_bep_fix.sh``).  Reverting
is one line in each place.

WHAT THIS FILE PINS.  The fixture's rural columns (frc = 0: cases 6, 18 and
24; case 12 has ust = hfx = qfx = 0, so with ust = 0 it drags nothing
either way and is left out) run through the port twice: under BEP with the forcing
the couple hands YSU, and with urban off.  The measure is the column's
momentum budget, ``sum_k du_k * delp_k / g``: the vertical mixing conserves
column momentum, so this is minus the surface stress the solve applied,
whatever the countergradient and diffusion terms did inside the column.

* The fixed arm applies the urban-off stress times the ratio of the two
  spellings of one drag: the couple writes ``ust**2/dz8w/|U|``, YSU writes
  ``ust**2/wspd1*rho*g/delp*(wspd1/wspd)**2``, and on these columns
  (``wspd = |U|``) they differ by ``delp/(rho*g*dz8w)``, YSU's own density
  against the column's layer thickness.  The synthetic fixture's pressure
  column is not built from that density, so the factor is 1.031 to 1.038
  here; in a model state it is the hydrostatic residual.  After dividing it
  out the fixed arm is the urban-off column to within 0.2 percent.
* Stock WRF applies about twice it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu

import test_ysu_bep_wrf471_parity as parity

REPO = Path(__file__).resolve().parents[1]

#: The fixture columns that are wholly rural (frc = 0) and carry a surface
#: stress: run_ysu_bep.F90's build_bep cycles frc by mod(ic, 6) and
#: case 12 has ust = 0.
RURAL_CASES = (6, 18, 24)

#: Measured, see the module docstring.  Column surface stress relative to the
#: same column with urban off, cases 6, 18, 24, u then v, 2026-09-30 on
#: test host 2 (RTX 5090, sm_120, NVRTC 13.4.92):
#:   woof's arm / urban off / spelling ratio:
#:       0.99856 0.99931 0.99954 0.99854 0.99925 0.99950
#:   stock WRF flag_bep / urban off (the defect):
#:       1.96084 2.00044 1.99994 1.96035 1.99863 1.99863
#: The bands hold those numbers with room for a card's rounding and nothing
#: that could hide a second drag: a doubled drag is a ratio of about 2.
RATIO_FIXED_BAND = (0.998, 1.001)
RATIO_STOCK_BAND = (1.95, 2.01)


def _column_stress(fx, du, dv):
    """sum_k du_k * delp_k / g per case, (u, v), in the fixture's own units."""
    p_int = fx["inputs"]["p_interface"][:, 0, :]
    delp = p_int[:-1] - p_int[1:]
    g = np.float64(9.81)
    su = (du[:, 0, :].astype(np.float64) * delp).sum(axis=0) / g
    sv = (dv[:, 0, :].astype(np.float64) * delp).sum(axis=0) / g
    return su, sv


def _spelling_ratio(fx):
    """delp / (rho * g * dz8w) at level 1 per case, rho as ysu.cu forms it
    (psfc / (RD * T1 * (1 + ep1 * qv1)), ep1 = RV / RD - 1 in float32)."""
    f4 = np.float32
    p_int = fx["inputs"]["p_interface"][:, 0, :]
    delp = (p_int[0] - p_int[1]).astype(np.float64)
    dz = fx["inputs"]["dz"][0, 0, :].astype(np.float64)
    temp = (fx["inputs"]["theta"][0, 0, :] * fx["inputs"]["exner"][0, 0, :]).astype(np.float64)
    qv = fx["inputs"]["qv"][0, 0, :].astype(np.float64)
    ep1 = np.float64(f4(f4(461.6) / f4(287.0)) - f4(1.0))
    rho = fx["inputs"]["psfc"][0].astype(np.float64) / (287.0 * temp * (1.0 + ep1 * qv))
    return delp / (rho * 9.81 * dz)


def _rural_index(fx):
    frc = fx["frc"].reshape(-1)
    idx = np.asarray([c - 1 for c in RURAL_CASES])
    assert (frc[idx] == 0.0).all(), frc[idx]
    assert (fx["inputs"]["ust"].reshape(-1)[idx] > 0).all()
    return idx


@pytest.mark.parametrize("module", ("ysu", "ysu_480"))
def test_the_kernel_removes_the_whole_of_its_own_drag_under_bep(module):
    """The one expression, spelled where it lives, with WRF's stock line
    recorded beside it so the revert is one line."""
    from woof.core.kernels import _unit_text
    text = _unit_text(module)
    # Since development branch the BEP entry point takes WRF's ctopo drag arm
    # (ctopo = 1, as WRF's driver passes it), so the removal is spelled on
    # WRF's own bracket, fric*vconvlim + ctopo*fric*(1-vconvlim).
    arm = text[text.index("real ctopo = topo.ctopo ? topo.ctopo[col] : 1.0f;"):]
    arm = arm[:arm.index("rhs[0] = u0;")]
    assert "DECLARED DIVERGENCE FROM WRF v4.7.1" in arm
    assert "diag[0] = __fsub_rn(diag[0], __fadd_rn(drag_v, drag_c));" in arm
    assert ("`__fsub_rn(diag[0], __fmul_rn(bep.frc[col], "
            "__fadd_rn(drag_v, drag_c)))`") in arm
    assert "ysu_column_body<true, true>" in text
    physics = (REPO / "docs" / "public" / "PHYSICS.md").read_text(encoding="utf-8")
    assert "surface drag twice" in physics


def test_stock_wrf_doubles_the_rural_stress_against_the_fixed_build():
    """WRF against WRF, no card: on each rural column the stock build's
    column stress is about twice the fixed build's, the defect measured on
    WRF's own arithmetic.  Measured 1.89, 1.93 and 1.94 (u) for cases 6, 18
    and 24: short of 2 because the implicit solve's stress is the drag
    coefficient times the NEW wind, which the doubled drag also slows."""
    fixed, stock = parity._fixture(), parity._fixture(parity.STOCK_FIXTURE)
    idx = _rural_index(fixed)
    fu, fv = _column_stress(fixed, fixed["ref"]["ctopo_utnp"], fixed["ref"]["ctopo_vtnp"])
    su, sv = _column_stress(stock, stock["ref"]["ctopo_utnp"], stock["ref"]["ctopo_vtnp"])
    for a, b in ((su, fu), (sv, fv)):
        ratio = a[idx] / b[idx]
        print("stock / fixed WRF column stress", ratio)
        assert ((ratio > 1.85) & (ratio < 2.0)).all(), ratio


@requires_gpu
def test_a_rural_column_under_bep_feels_the_urban_off_surface_stress():
    """The column test: a non-urban column under BEP (the port's fixed arm)
    against the SAME column with urban off.  Stock WRF's flag_bep result on
    the same columns is the defect's record."""
    import cupy  # parity._port opens the device through launch_ysu

    assert cupy.cuda.runtime.getDeviceCount() >= 1
    fx = parity._fixture()
    idx = _rural_index(fx)
    off = parity._port(fx, bep=False)
    bep = parity._port(fx, bep=True)
    stock = parity._fixture(parity.STOCK_FIXTURE)
    ou, ov = _column_stress(fx, off["du"], off["dv"])
    bu, bv = _column_stress(fx, bep["du"], bep["dv"])
    su, sv = _column_stress(fx, stock["ref"]["ctopo_utnp"], stock["ref"]["ctopo_vtnp"])
    spelling = np.concatenate([_spelling_ratio(fx)[idx]] * 2)
    fixed = np.concatenate([bu[idx] / ou[idx], bv[idx] / ov[idx]]) / spelling
    doubled = np.concatenate([su[idx] / ou[idx], sv[idx] / ov[idx]])
    print("spelling ratio delp/(rho g dz8w)", spelling)
    print("fixed arm / urban off / spelling", fixed)
    print("stock WRF / urban off", doubled)
    lo, hi = RATIO_FIXED_BAND
    assert ((fixed > lo) & (fixed < hi)).all(), fixed
    lo, hi = RATIO_STOCK_BAND
    assert ((doubled > lo) & (doubled < hi)).all(), doubled


@requires_gpu
def test_the_urban_off_column_is_untouched_by_the_divergence():
    """bep=None launches ysu_column, whose statements the divergence does
    not reach: the plain port on the rural columns equals the plain port on
    the same inputs with the BEP forcing never built (a guard against a
    shared-workspace leak between the two entry points in one process)."""
    import cupy  # parity._port opens the device through launch_ysu

    assert cupy.cuda.runtime.getDeviceCount() >= 1
    fx = parity._fixture()
    first = parity._port(fx, bep=False)
    parity._port(fx, bep=True)
    again = parity._port(fx, bep=False)
    for name in ("du", "dv", "dtheta", "dqv", "exch_m", "hpbl"):
        assert np.array_equal(np.asarray(first[name]).view(np.uint32),
                              np.asarray(again[name]).view(np.uint32)), name


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q", "-s"]))
