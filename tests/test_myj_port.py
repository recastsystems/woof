"""MYJ PBL (bl_pbl_physics=2) and its Eta surface layer (sf_sfclay=2).

WHAT THIS FILE IS EVIDENCE FOR, stated before the first assertion so nobody
has to infer it from a passing run:

* the float32 CPU authority (``woof/verify/myj_ref.py``) runs both column
  functions on land and water soundings and produces finite, bounded,
  physically ordered output;
* the CUDA translation units agree with that authority within a stated
  tolerance, on the same columns;
* the SHIPPED seams reach the scheme -- ``initialize_physics`` allocates
  MYJ's own fields and ``PhysicsDriver.compute`` dispatches to
  ``_run_myj_sfclay`` and ``_run_myj_pbl`` -- and a forecast advances;
* the pairing law refuses a half-suite in both directions, at load;
* the registry, the namelist importer and the restart identity table all
  name the scheme.

WHAT IT IS **NOT** EVIDENCE FOR.  There is NO ORACLE COMPARISON AGAINST THE
WRF FORTRAN anywhere in this file.  No gfortran replay of
``phys/module_bl_myjpbl.F`` or ``phys/module_sf_myjsfc.F`` has been run,
no fixture of WRF words exists, and no ULP table is asserted.  Everything
below is self-consistency, physical sanity and integration; the number that
would say "this is WRF" has not been measured.  That campaign is the
declared next stage, as it was for Shin-Hong and Grell-Freitas.

Every physical-sanity assertion carries a MUTATION CONTROL, and the
controls are REAL: each one replaces a ported routine inside
``woof.verify.myj_ref`` and re-runs the SAME assertion helper the shipped
test calls, requiring it to go red.  Nothing here asserts on a locally
built stub dictionary -- delete a routine and this file fails.  The
per-routine table (``_ROUTINE_MUTATIONS``) records which bars each stub
breaks, measured on this tree, so a bar that quietly loses its teeth
changes the table rather than passing anyway.  Two coverage holes are
DECLARED there rather than papered over: stubbing ``_vdifq`` breaks
nothing in this file, and neutering the similarity-table lookup leaves
every surface bar green.

The Eta surface layer's WRF oracle lives in its own file:
tests/test_myjsfc_wrf461_parity.py grades the CUDA kernel and the host
similarity tables word for word against the byte-unmodified
phys/module_sf_myjsfc.F (tools/myjsfc_wrf461_oracle).  Nothing in THIS file
depends on it.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from conftest import requires_gpu

from woof.config import (MYJ_PBL_SCHEME, MYJ_SFCLAY_SCHEME, PBL_SCHEMES,
                          SURFACE_LAYER_SCHEMES, RunConfig,
                          validate_myj_pairing)
from woof.verify import myj_ref
from woof.verify.myj_ref import EPSQ2, myjsfc_psi_tables

F = np.float32
_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _sounding(nz=30, dtheta_dz=0.004, wind=8.0):
    """A dry-ish 3-km column: constant dz, linear theta, uniform wind."""
    dz = np.full(nz, 100.0, F)
    z = (np.cumsum(dz) - F(50.0)).astype(F)
    th = (F(300.0) + F(dtheta_dz) * z).astype(F)
    p = (F(1.0e5) * np.exp(-z / F(8500.0))).astype(F)
    exner = ((p / F(1.0e5)) ** F(287.0 / 1004.5)).astype(F)
    t = (th * exner).astype(F)
    # A moisture GRADIENT, not a well-mixed profile.  A uniform qv makes
    # every moisture tendency an exact cancellation, so the CPU-vs-CUDA
    # comparison below would be comparing one roundoff against another and
    # would fail (or pass) for reasons that have nothing to do with the
    # port.  0.012 at the surface falling to about 0.006 at 3 km is an
    # ordinary summer sounding.
    qv = (F(0.012) - F(2.0e-4) * np.arange(nz, dtype=F)).astype(F)
    return {
        "dz": dz, "th": th, "p": p, "exner": exner, "t": t,
        "qv": np.maximum(qv, F(1.0e-4)).astype(F),
        "qc": np.zeros(nz, F), "qi": np.zeros(nz, F),
        "u": np.full(nz, F(wind), F), "v": np.full(nz, 2.0, F),
        "tke": np.full(nz, 0.1, F),
    }


def _sfc_column(col, *, xland, tsk, itimestep=1, ust=0.1):
    return myj_ref.np_myjsfc_column(
        dz=col["dz"], tke=col["tke"], u1=col["u"][0], v1=col["v"][0],
        t1=col["t"][0], th1=col["th"][0], qv1=col["qv"][0],
        qc1=col["qc"][0], p1=col["p"][0], psfc=F(1.0e5), tsk=F(tsk),
        qsfc=F(0.012), thz0=F(300.0), qz0=F(0.012), uz0=F(0.0), vz0=F(0.0),
        ust=F(ust), znt=F(0.1), z0base=F(0.1), akhs=F(0.01), akms=F(0.01),
        xland=F(xland), mavail=F(1.0), itimestep=itimestep,
        tables=myjsfc_psi_tables())


def _pbl_column(col, sfc, *, xland, tsk, dtturbl=60.0, qi=None):
    return myj_ref.np_myjpbl_column(
        dz=col["dz"], u=col["u"], v=col["v"], t=col["t"], th=col["th"],
        exner=col["exner"], qv=col["qv"], qc=col["qc"],
        qi=col["qi"] if qi is None else qi, p=col["p"], psfc=F(1.0e5),
        tke=col["tke"], ust=sfc["ust"], tsk=F(tsk), qsfc=sfc["qsfc"],
        chklowq=F(1.0), thz0=sfc["thz0"], qz0=sfc["qz0"], uz0=sfc["uz0"],
        vz0=sfc["vz0"], xland=F(xland), sice=F(0.0), snow=F(0.0),
        akhs=sfc["akhs"], akms=sfc["akms"], elflx=F(50.0), ct=sfc["ct"],
        dtturbl=F(dtturbl))


# ---------------------------------------------------------------------------
# The similarity tables MYJSFCINIT builds.
# ---------------------------------------------------------------------------

def test_the_similarity_tables_are_the_accumulated_fortran_ones():
    """ZTMAX is the ACCUMULATED endpoint minus EPS, not the literal 1.0.

    module_sf_myjsfc.F:1288-1299 records ZETA at K=KZTM and then subtracts
    EPS=1e-6.  ZETA is a float32 running sum of 10000 DZETA steps, so the
    endpoint overshoots the nominal ZTMAX1=1.0 the source assigns at :1185.
    A port that used the literal 1.0 would clamp zeta to a different value
    on every stable column, so the accumulation is pinned here.
    """
    tables = myjsfc_psi_tables()
    assert tables["ztmin1"] == F(-5.0) and tables["ztmin2"] == F(-5.0)
    assert tables["fh01"] == F(1.0) and tables["fh02"] == F(1.0)
    for name in ("ztmax1", "ztmax2"):
        # Above the nominal 1.0 despite the -1e-6: the float32 sum drifts up.
        assert tables[name] > F(1.0), name
        assert tables[name] < F(1.001), name
    for name in ("psim1", "psih1", "psim2", "psih2"):
        psi = tables[name]
        assert psi.shape == (10001,) and psi.dtype == F
        assert np.all(np.isfinite(psi))
    # PSIH is monotone-increasing in zeta over the whole tabulated range
    # (Paulson unstable, Holtslag-de Bruin stable): the integral of a
    # positive stability function.
    assert np.all(np.diff(tables["psih1"]) > 0.0)
    # The v4.6.1 ranges make the sea and land tables identical.  They are
    # built independently anyway (the Fortran does), so proving equality is
    # proving the transcription did not diverge, not proving a shortcut.
    assert np.array_equal(tables["psim1"], tables["psim2"])
    assert np.array_equal(tables["psih1"], tables["psih2"])


# ---------------------------------------------------------------------------
# CPU authority: physical sanity, with mutation controls.
# ---------------------------------------------------------------------------

def _assert_surface_layer_is_physical(out, col, xland, tag):
    """THE surface-layer bars, in one function.

    Both the real test and its mutation control call this, so the control
    exercises the SHIPPED assertions rather than a copy of them: if a bar
    is weakened here, the control that requires it to fail on a stub goes
    red in the same edit.
    """
    for name, value in out.items():
        assert np.isfinite(value), f"{tag}: {name} is not finite"
    # Friction velocity above the scheme's own floor and inside the range a
    # 8 m s-1 lowest level can produce.
    assert F(1.0e-9) < out["ust"] < F(2.0), tag
    # Exchange coefficients are positive and at least the EXCM/ZSL floor
    # SFCDIF applies (module_sf_myjsfc.F:618-619, :790-791): EXCML=1e-4 over
    # a 50 m half-layer is 2e-6.
    assert out["akhs"] >= F(2.0e-6) and out["akms"] >= F(2.0e-6), tag
    # 2 m and 10 m diagnostics are bracketed by the surface and the lowest
    # model level, which is what SFCDIF's own bracketing guard enforces
    # (:964-974).
    lo, hi = sorted((float(out["thz0"]), float(col["th"][0])))
    assert lo - 1e-3 <= float(out["tshltr"]) <= hi + 1e-3, tag
    assert lo - 1e-3 <= float(out["th10"]) <= hi + 1e-3, tag
    # The shelter humidity is a specific humidity in [0,1) and its
    # mixing-ratio publication is positive.
    assert F(0.0) <= out["qshltr"] < F(1.0), tag
    assert out["q2"] > F(0.0), tag
    # PSHLTR is a pressure a hair below PSFC (:978-979).
    assert F(0.99e5) < out["pshltr"] < F(1.0e5), tag
    # CT is identically zero in ARW: MYJSFC zeroes it and SFCDIF's
    # countergradient block is commented out (:206-211, :816-825).
    assert out["ct"] == F(0.0), tag
    # The sign of the surface heat flux follows the surface-air contrast.
    if xland < 1.5:
        assert out["hfx"] > F(0.0), "warm land should heat the column"
    else:
        assert out["hfx"] < F(0.0), "cool water should cool the column"


@pytest.mark.parametrize("xland,tsk,tag", [(1.0, 305.0, "land"),
                                           (2.0, 295.0, "water")])
def test_the_surface_layer_column_is_finite_and_physical(xland, tsk, tag):
    col = _sounding()
    out = _sfc_column(col, xland=xland, tsk=tsk)
    _assert_surface_layer_is_physical(out, col, xland, tag)


def test_the_surface_layer_bars_fail_when_the_ported_routine_is_stubbed(
        monkeypatch):
    """MUTATION CONTROL for the test above -- a REAL one.

    The scheme under test is replaced in the module the fixture calls
    through (``woof.verify.myj_ref.np_myjsfc_column``), the SAME helper
    that carries the shipped bars is re-run on the result, and the test
    requires it to raise.  Nothing here asserts on a locally built dict:
    delete the port and this test fails, which is the whole point of a
    mutation control.
    """
    col = _sounding()
    real = _sfc_column(col, xland=1.0, tsk=305.0)
    _assert_surface_layer_is_physical(real, col, 1.0, "land")

    # (a) the scheme returns zeros: every finiteness check still passes and
    #     the physical ones do not.
    monkeypatch.setattr(
        myj_ref, "np_myjsfc_column",
        lambda **kwargs: {name: F(0.0) for name in real})
    zeroed = _sfc_column(col, xland=1.0, tsk=305.0)
    assert zeroed["ust"] == F(0.0), "the stub did not reach the fixture"
    with pytest.raises(AssertionError):
        _assert_surface_layer_is_physical(zeroed, col, 1.0, "land")

    # WHAT THESE BARS DO NOT CATCH, measured rather than assumed: replacing
    # the similarity-function table lookup (module_sf_myjsfc.F:583-588) with
    # a constant leaves every bar above green, on all four variants tried
    # (0.0, psi[0], psi[-1], and the interpolation dropped).  The bars bound
    # the SHAPE of the answer, not the stability functions inside it; the
    # tables are pinned separately by
    # test_the_similarity_tables_are_the_accumulated_fortran_ones, and only
    # an oracle can bound the values themselves.
    monkeypatch.undo()
    monkeypatch.setattr(myj_ref, "_table",
                        lambda psi, zeta, ztmin, dzeta: F(0.0))
    _assert_surface_layer_is_physical(
        _sfc_column(col, xland=1.0, tsk=305.0), col, 1.0, "land")


def _assert_pbl_column_is_physical(out, col, tag):
    """THE PBL-column bars, in one function, so the mutation controls below
    re-run the shipped assertions instead of a copy of them."""
    nz = col["dz"].shape[0]
    for name in ("rublten", "rvblten", "rthblten", "rqvblten", "rqcblten",
                 "rqiblten", "tke", "el", "exch_h"):
        assert np.all(np.isfinite(out[name])), f"{tag}: {name}"
        assert out[name].shape == (nz,), f"{tag}: {name}"
    for name in ("pblh", "mixht", "thz0", "qz0", "qsfc", "ct"):
        assert np.isfinite(out[name]), f"{tag}: {name}"
    # PRODQ2's floor is EPSQ2 on q2, so TKE = 0.5*q2 never drops below it
    # (module_bl_myjpbl.F:461).
    assert np.all(out["tke"] >= F(0.5) * EPSQ2 - F(1e-9)), tag
    # The mixing length is non-negative everywhere and is deliberately zero
    # at the LOWEST model level.  That is WRF's own flip, not an omission:
    # EL is dimensioned KTS:KTE-1 in MYJ's top-down layout and EL_MYJ(K) is
    # written only when KFLIP=KTE+1-K is below KTE, so the ground-adjacent
    # level -- MYJ's KTE, "EL IS NOT DEFINED AT KTE (ground surface)" --
    # keeps the zero MYJPBL initialised it to (module_bl_myjpbl.F:341,:463).
    assert np.all(out["el"] >= F(0.0)), tag
    assert out["el"][0] == F(0.0), tag
    assert out["el"][-1] > F(0.0), tag
    assert np.all(out["exch_h"] >= F(0.0)), tag
    # KPBL is a one-based model level inside the column, and MIXHT/PBLH are
    # heights above ground within it.
    assert 1 <= int(out["kpbl"]) <= nz, tag
    assert F(0.0) <= out["pblh"] <= F(nz * 100.0), tag
    assert F(0.0) <= out["mixht"] <= F(nz * 100.0), tag
    # The scheme actually did something: a sheared, heated column must
    # produce nonzero momentum and heat tendencies.
    assert np.max(np.abs(out["rublten"])) > F(0.0), tag
    assert np.max(np.abs(out["rthblten"])) > F(0.0), tag


@pytest.mark.parametrize("xland,tsk,tag", [(1.0, 305.0, "land"),
                                           (2.0, 295.0, "water")])
def test_the_pbl_column_is_finite_bounded_and_mixes(xland, tsk, tag):
    col = _sounding()
    sfc = _sfc_column(col, xland=xland, tsk=tsk)
    _assert_pbl_column_is_physical(
        _pbl_column(col, sfc, xland=xland, tsk=tsk), col, tag)


def _sealed_column(col, sfc):
    """The same closed column both the conservation test and its control run.

    CHKLOWQ=0 seals the surface moisture row: it multiplies CLOW(2) and
    therefore RKSS for that row (module_bl_myjpbl.F:576, :1505), leaving
    VDIFH a conservative implicit solve with no source at all.
    """
    return myj_ref.np_myjpbl_column(
        dz=col["dz"], u=col["u"], v=col["v"], t=col["t"], th=col["th"],
        exner=col["exner"], qv=col["qv"], qc=col["qc"], qi=col["qi"],
        p=col["p"], psfc=F(1.0e5), tke=col["tke"], ust=sfc["ust"],
        tsk=F(305.0), qsfc=sfc["qsfc"], chklowq=F(0.0), thz0=sfc["thz0"],
        qz0=sfc["qz0"], uz0=sfc["uz0"], vz0=sfc["vz0"], xland=F(1.0),
        sice=F(0.0), snow=F(0.0), akhs=sfc["akhs"], akms=sfc["akms"],
        elflx=F(0.0), ct=sfc["ct"], dtturbl=F(60.0))


def _assert_the_sealed_column_conserves_vapour(sealed, col):
    """THE conservation bar, in one function, called by both tests below."""
    # rho*dz weights: uniform dz, so the density profile is the whole weight.
    rho = (col["p"] / (F(287.0) * col["t"])).astype(np.float64)
    weight = rho * col["dz"].astype(np.float64)
    net = float(np.sum(weight * sealed["rqvblten"].astype(np.float64)))
    scale = float(np.sum(weight * np.abs(
        sealed["rqvblten"].astype(np.float64))))
    # A closed column: the net is roundoff against the gross exchange.  The
    # bar is loose because the sum is float32 arithmetic reduced in float64;
    # it is three orders tighter than any real surface flux.
    assert scale > 0.0, "the sealed column exchanged nothing to measure"
    assert abs(net) < 1.0e-3 * scale, (net, scale)


def test_vertical_diffusion_conserves_the_column_it_should():
    """Mass-weighted vapour is conserved to the surface flux, not created.

    VDIFH is a conservative implicit solve whose only source is the lower
    boundary (module_bl_myjpbl.F:1504-1509), so with the surface row's
    exchange switched off the column integral of the vapour tendency must
    vanish to roundoff.
    """
    col = _sounding()          # already carries the moisture gradient
    sfc = _sfc_column(col, xland=1.0, tsk=305.0)
    _assert_the_sealed_column_conserves_vapour(_sealed_column(col, sfc), col)


def test_the_conservation_bar_fails_when_the_solve_is_stubbed(monkeypatch):
    """MUTATION CONTROL for the conservation test -- a REAL one.

    The ported routine (``woof.verify.myj_ref._vdifh``) is replaced and the
    SAME bar the test above uses is re-run on the result, through the real
    driver.  Two mutations, because the bar has two halves:

    (a) a solve that does nothing -- what a dropped call or an unwritten
        output looks like -- makes every vapour tendency zero, and the
        non-vacuity half ("the sealed column exchanged nothing to measure")
        is what catches it;
    (b) the real solve followed by a uniform gain in every row -- what a
        sign error or a dropped back-substitution row looks like -- leaves
        the exchange intact and breaks the conservation inequality itself.

    Neither asserts on a locally built array: delete ``_vdifh`` and this
    test stops passing.
    """
    col = _sounding()
    sfc = _sfc_column(col, xland=1.0, tsk=305.0)
    _assert_the_sealed_column_conserves_vapour(_sealed_column(col, sfc), col)
    real_vdifh = myj_ref._vdifh

    monkeypatch.setattr(myj_ref, "_vdifh", lambda *args, **kwargs: None)
    dead = _sealed_column(col, sfc)
    assert not np.any(dead["rqvblten"]), "the stub did not reach the driver"
    with pytest.raises(AssertionError):
        _assert_the_sealed_column_conserves_vapour(dead, col)

    def _leaky(dtdif, lmh, lpbl, sz0, rkhs, clow, cts, species, nspec,
               rkh, zh, rho):
        real_vdifh(dtdif, lmh, lpbl, sz0, rkhs, clow, cts, species, nspec,
                   rkh, zh, rho)
        species += F(1.0e-6)          # a source in every row, from nowhere
    monkeypatch.setattr(myj_ref, "_vdifh", _leaky)
    leaky = _sealed_column(col, sfc)
    with pytest.raises(AssertionError):
        _assert_the_sealed_column_conserves_vapour(leaky, col)


def test_the_ice_arm_changes_the_answer_and_the_no_ice_arm_is_zero():
    """WRF's PRESENT(QCI) arm is a real branch, not a spelling.

    With ice the species stack is four rows and RQIBLTEN is produced;
    without it three, and the row is a published zero
    (module_bl_myjpbl.F:264-273, :505-509, :647).
    """
    col = _sounding()
    sfc = _sfc_column(col, xland=1.0, tsk=305.0)
    dry = _pbl_column(col, sfc, xland=1.0, tsk=305.0, qi=None)
    assert np.all(dry["rqiblten"] == F(0.0))
    iced = np.zeros_like(col["qi"])
    iced[10:16] = F(2.0e-5)
    wet = myj_ref.np_myjpbl_column(
        dz=col["dz"], u=col["u"], v=col["v"], t=col["t"], th=col["th"],
        exner=col["exner"], qv=col["qv"], qc=col["qc"], qi=iced,
        p=col["p"], psfc=F(1.0e5), tke=col["tke"], ust=sfc["ust"],
        tsk=F(305.0), qsfc=sfc["qsfc"], chklowq=F(1.0), thz0=sfc["thz0"],
        qz0=sfc["qz0"], uz0=sfc["uz0"], vz0=sfc["vz0"], xland=F(1.0),
        sice=F(0.0), snow=F(0.0), akhs=sfc["akhs"], akms=sfc["akms"],
        elflx=F(50.0), ct=sfc["ct"], dtturbl=F(60.0))
    assert np.max(np.abs(wet["rqiblten"])) > F(0.0)
    # The ice loading also enters CWM and therefore THE, so the heat
    # tendency must move: an unread QCI would leave it bitwise identical.
    assert not np.array_equal(wet["rthblten"], dry["rthblten"])


def test_the_cold_start_seed_is_wrfs_epsq2_and_it_decides_the_first_step_pbl():
    """MYJPBLINIT seeds TKE_MYJ at EPSQ2, and the seed is not cosmetic.

    ``module_bl_myjpbl.F:1725`` is ``TKE_MYJ(I,K,J)=EPSQ2`` with epsq2=0.2
    (share/module_model_constants.F:92, "initial TKE"); only the four
    tendencies and EXCH_H are zeroed there, and EL_MYJ is not even an
    argument of MYJPBLINIT (:1694-1697) -- it is INTENT(OUT) on MYJPBL and
    rewritten from zero at the top of every call (:341).

    The seed decides MIXLEN's LPBL scan on step one.  q2 = 2*TKE and the
    threshold is EPSQ2*FH = 0.202, so WRF's seed (q2 = 0.4) falls through
    to LPBL=1 and the Blackadar EL0 branch runs the whole column, while a
    zero seed trips the scan at the first level tested.  This test pins
    BOTH sides of that fork, and it is red under a zero cold start -- which
    is what woof shipped before this was measured.
    """
    from woof.core.physics import MYJ_TKE_COLD_START

    # The shipped constant IS the authority's EPSQ2.  core carries a literal
    # because it must not import the verification tree; this is the gate
    # that keeps the two from drifting (the Shin-Hong precedent).
    assert F(MYJ_TKE_COLD_START) == EPSQ2

    col = _sounding()
    nz = col["dz"].shape[0]
    seeded = dict(col, tke=np.full(nz, float(MYJ_TKE_COLD_START), F))
    zeroed = dict(col, tke=np.zeros(nz, F))

    warm_sfc = _sfc_column(seeded, xland=1.0, tsk=305.0)
    warm = _pbl_column(seeded, warm_sfc, xland=1.0, tsk=305.0)
    # The zero seed makes MIXLEN's SQ (a sum of sqrt(q2)) exactly zero, so
    # EL0=MIN(ALPH*SZQ*0.5/SQ,EL0MAX) is a 0/0 WRF never reaches.  It is
    # laundered back to EPSL by PRODQ2 and nothing non-finite escapes, but
    # the invalid operation is real and is silenced here rather than hidden.
    with np.errstate(invalid="ignore"):
        cold_sfc = _sfc_column(zeroed, xland=1.0, tsk=305.0)
        cold = _pbl_column(zeroed, cold_sfc, xland=1.0, tsk=305.0)

    # WRF's seed: the scan falls through, so the whole 3 km column is inside
    # the PBL on step one.  PBLH = Z(LPBL+1)-Z(LMH+1) with LPBL=1 is the
    # column depth less the lowest layer (module_bl_myjpbl.F:797-806, :860).
    assert int(warm["kpbl"]) == nz
    assert float(warm["pblh"]) == float(F((nz - 1) * 100.0))
    # A zero seed: the scan trips immediately and pins LPBL = LMH-1, one
    # layer deep.  This is the value woof produced before the fix.
    assert int(cold["kpbl"]) == 2
    assert float(cold["pblh"]) == float(F(100.0))
    # And it is not a diagnostic-only difference.  The surface layer reads
    # the step-1 PBLH through BTGH -> WSTAR2 -> u* (module_sf_myjsfc.F:279,
    # :430-447), so the friction velocity and the surface heat flux move
    # with the seed, and so does the TKE the next step starts from.
    assert float(warm_sfc["ust"]) > float(cold_sfc["ust"])
    assert float(warm_sfc["hfx"]) > float(cold_sfc["hfx"])
    assert float(np.max(warm["tke"])) > float(np.max(cold["tke"]))


def test_the_dropped_terrain_height_cancels_in_float32():
    """The DECLARED divergence, measured instead of asserted.

    Since the MYJPBL oracle campaign it is a divergence of the SURFACE
    LAYER only: the PBL kernel seeds ZINT(KTE+1)=HT as WRF does and is
    graded bit-identical (tests/test_myjpbl_wrf461_parity.py).  This test
    still drives both CPU-authority halves, so it keeps measuring how far
    the ground-relative column moves the words.

    WRF seeds the interface-height column with the terrain height
    (``ZINT(KTE+1)=HT``, module_bl_myjpbl.F:312, module_sf_myjsfc.F:162);
    woof seeds 0 and carries heights above ground
    (``woof/verify/myj_ref.py::_interface_heights``).  HT cancels exactly
    in real arithmetic.  In float32 it does NOT cancel exactly, and the
    docstring's old claim that it did was the thing that needed measuring.

    Measured on this tree, over five columns (four stretched so that no dz
    and no interface height is an exactly representable float32, plus one
    uniform 100 m column) x three terrain heights (1523.7314, 2987.3129 and
    4411.0837 m), land and water:

    * KPBL is IDENTICAL in every case -- no branch decision moves;
    * non-tendency fields differ by at most 69 ULP.  The field that
      attains it is ``lh`` over water, where the move is 1.645e-05
      W m-2;
    * in RELATIVE terms the worst non-tendency move is 5.75e-06, and it
      is attained on ``qfx``, not on ``lh``.  ULP and relative are
      measured separately here because they land on different fields:
      the earlier claim of 9.3e-07 relative was this file's own
      1.645e-05 W m-2 water-case absolute divided by the LAND ``lh``,
      a pairing that measures nothing.  Both numbers below come out of
      the same loop that produces the bounds;
    * tendency rows differ by at most 2.05 quanta, where a quantum is the
      source field's float32 ULP over dt -- the resolution limit of a
      float32 tendency, so this is the smallest difference expressible.

    The bounds below carry headroom over those measurements; the test's job
    is to catch the divergence GROWING, and to fail if a future edit makes
    the column exactly HT-invariant by accident, which would mean it is no
    longer measuring anything.
    """
    dt = 60.0
    tend_source = {"rublten": "u", "rvblten": "v", "rthblten": "th",
                   "rqvblten": "qv", "rqcblten": "qc", "rqiblten": "qi"}

    def stretched(nz, dz0, dtheta_dz, wind):
        dz = (F(dz0) * np.array([1.07 ** i for i in range(nz)], F)
              * F(0.4371)).astype(F)
        z = (np.cumsum(dz) - dz / F(2.0)).astype(F)
        th = (F(300.0) + F(dtheta_dz) * z).astype(F)
        p = (F(1.0e5) * np.exp(-z / F(8500.0))).astype(F)
        exner = ((p / F(1.0e5)) ** F(287.0 / 1004.5)).astype(F)
        return {"dz": dz, "th": th, "p": p, "exner": exner,
                "t": (th * exner).astype(F),
                "qv": (F(0.012) - F(0.006) * (z / z[-1])).astype(F),
                "qc": np.zeros(nz, F), "qi": np.zeros(nz, F),
                "u": np.full(nz, F(wind), F), "v": np.zeros(nz, F),
                "tke": np.full(nz, float(EPSQ2), F)}

    def both(col, ht, xland, tsk):
        sfc = myj_ref.np_myjsfc_column(
            dz=col["dz"], tke=col["tke"], u1=col["u"][0], v1=col["v"][0],
            t1=col["t"][0], th1=col["th"][0], qv1=col["qv"][0],
            qc1=col["qc"][0], p1=col["p"][0], psfc=F(1.0e5), tsk=F(tsk),
            qsfc=F(0.012), thz0=F(tsk), qz0=F(0.012), uz0=F(0.0),
            vz0=F(0.0), ust=F(0.3), znt=F(0.1), z0base=F(0.1),
            akhs=F(0.0), akms=F(0.0), xland=F(xland), mavail=F(1.0),
            itimestep=7, tables=myjsfc_psi_tables(), ht=F(ht))
        pbl = myj_ref.np_myjpbl_column(
            dz=col["dz"], u=col["u"], v=col["v"], t=col["t"], th=col["th"],
            exner=col["exner"], qv=col["qv"], qc=col["qc"], qi=col["qi"],
            p=col["p"], psfc=F(1.0e5), tke=col["tke"], ust=sfc["ust"],
            tsk=F(tsk), qsfc=sfc["qsfc"], chklowq=F(1.0), thz0=sfc["thz0"],
            qz0=sfc["qz0"], uz0=sfc["uz0"], vz0=sfc["vz0"], xland=F(xland),
            sice=F(0.0), snow=F(0.0), akhs=sfc["akhs"], akms=sfc["akms"],
            elflx=F(50.0), ct=sfc["ct"], dtturbl=F(dt), ht=F(ht))
        return sfc, pbl

    cases = [(stretched(30, 100.0, 0.004, 8.0), 1.0, 305.0),
             (stretched(30, 100.0, 0.004, 8.0), 2.0, 295.0),
             (stretched(60, 50.0, 0.012, 3.0), 1.0, 290.0),
             (stretched(40, 150.0, 0.001, 14.0), 1.0, 310.0),
             (_sounding(), 1.0, 305.0)]
    worst_tend = 0.0
    worst_other = 0.0
    worst_other_at = ("", 0.0)
    worst_rel = 0.0
    worst_rel_at = ""
    for col, xland, tsk in cases:
        base = both(col, 0.0, xland, tsk)
        for ht in (1523.7314, 2987.3129, 4411.0837):
            moved = both(col, ht, xland, tsk)
            for ref, alt in zip(base, moved):
                for name, want in ref.items():
                    got = alt[name]
                    if name == "kpbl":
                        assert int(got) == int(want), (name, ht)
                        continue
                    want = np.asarray(want, F)
                    got = np.asarray(got, F)
                    delta = np.abs(got.astype(np.float64)
                                   - want.astype(np.float64))
                    if name in tend_source:
                        quantum = np.spacing(
                            np.abs(np.asarray(col[tend_source[name]], F))) / dt
                        worst_tend = max(worst_tend, float(
                            np.max(delta / np.maximum(quantum, 1e-45))))
                    else:
                        quantum = np.spacing(
                            np.maximum(np.abs(want), np.abs(got)))
                        ulps = np.ravel(delta / np.maximum(quantum, 1e-45))
                        j = int(np.argmax(ulps))
                        if float(ulps[j]) > worst_other:
                            worst_other = float(ulps[j])
                            worst_other_at = (name,
                                              float(np.ravel(delta)[j]))
                        # RELATIVE is tracked separately: it peaks on a
                        # different field than ULP does, so neither number
                        # may be derived from the other.
                        scale = np.maximum(np.abs(want.astype(np.float64)),
                                           np.abs(got.astype(np.float64)))
                        rel = np.ravel(delta / np.maximum(scale, 1e-30))
                        k = int(np.argmax(rel))
                        if float(rel[k]) > worst_rel:
                            worst_rel = float(rel[k])
                            worst_rel_at = name
    assert worst_tend <= 4.0, (
        f"tendency rows moved {worst_tend:.2f} quanta with terrain, "
        "measured at 2.05 when the divergence was declared")
    assert worst_other <= 128.0, (
        f"diagnostic fields moved {worst_other:.1f} ULP with terrain "
        f"(worst on {worst_other_at[0]}, {worst_other_at[1]:.4g} in field "
        "units), measured at 69 ULP / 1.645e-05 W m-2 on lh over water "
        "when the divergence was declared")
    assert worst_rel <= 1.0e-05, (
        f"diagnostic fields moved {worst_rel:.3g} relative with terrain "
        f"(worst on {worst_rel_at}), measured at 5.75e-06 on qfx when the "
        "divergence was declared")
    # Non-vacuity: the divergence is real, so a test that measured zero
    # would be measuring an exactly representable grid, not cancellation.
    assert worst_other > 0.0 and worst_rel > 0.0, (
        "no field moved at all -- this column cannot detect the divergence")


def _noop(*args, **kwargs):
    return None


def _difcof_zeros(lmh, gm, gh, el, q2, zh):
    return np.zeros(lmh, F), np.zeros(lmh, F)


def _mixlen_zeros(lmh, u, v, t, the, q, cwm, q2, zh, ct):
    return (np.zeros(lmh, F), np.zeros(lmh, F), np.zeros(lmh, F),
            F(0.0), 1, 1, ct, F(0.0))


#: MEASURED mutation-control table.  Each row stubs ONE ported routine in
#: ``woof.verify.myj_ref`` and names the shipped bars that go red because of
#: it -- measured on this tree, not assumed.  The point of the table is that
#: the bars have teeth against the SCHEME, not against a hand-built array:
#: delete any of these routines and the suite fails.
_ROUTINE_MUTATIONS = [
    # VDIFH is the species solve: stubbing it leaves every tendency zero, so
    # the "did something" bars and the conservation non-vacuity bar fire.
    ("_vdifh", _noop, ("pbl", "conservation")),
    # VDIFV is the momentum solve: only the momentum-tendency bar sees it.
    ("_vdifv", _noop, ("pbl",)),
    # PRODQ2 is the TKE closure.  It does NOT trip the column bars, because
    # q2 is floored at EPSQ2 either way -- what it trips is the physical
    # ORDERING bar, which is the one that says stability matters at all.
    ("_prodq2", _noop, ("ordering",)),
    # DIFCOF turns (GM,GH,EL,Q2) into the exchange coefficients: zeros stop
    # the column exchanging anything, which the conservation bar's
    # non-vacuity half catches, and flatten the stable/unstable ordering.
    ("_difcof", _difcof_zeros, ("conservation", "ordering")),
    # MIXLEN is the master-length scale the whole scheme hangs off.
    ("_mixlen", _mixlen_zeros, ("conservation", "ordering")),
]


@pytest.mark.parametrize(
    "victim,stub,expect", _ROUTINE_MUTATIONS,
    ids=[row[0] for row in _ROUTINE_MUTATIONS])
def test_stubbing_a_ported_routine_turns_a_shipped_bar_red(
        monkeypatch, victim, stub, expect):
    """MUTATION CONTROL, one row per ported routine.

    Every bar in this file is re-run against a tree in which exactly one of
    MYJ's routines has been replaced, and the row asserts WHICH bars must go
    red.  ``expect`` is measured, so the row also fails if a stub starts
    breaking MORE than it used to -- a bar that grew a new sensitivity is as
    much a change as one that lost it.

    DECLARED GAP, measured the same way: stubbing ``_vdifq`` (the TKE
    tridiagonal, module_bl_myjpbl.F:1330-1406) breaks NOTHING in this file.
    Its output is floored at EPSQ2 and published as TKE, and no bar here
    resolves the difference between a diffused q2 profile and an undiffused
    one.  That is a real hole in the sanity net, it is why the row is absent
    from the table above rather than silently green, and only the oracle
    campaign closes it.
    """
    col = _sounding()

    def _bars():
        red = set()
        try:
            sfc = _sfc_column(col, xland=1.0, tsk=305.0)
            _assert_surface_layer_is_physical(sfc, col, 1.0, "land")
        except AssertionError:
            red.add("sfc")
        for tag, xland, tsk in (("land", 1.0, 305.0), ("water", 2.0, 295.0)):
            try:
                sfc = _sfc_column(col, xland=xland, tsk=tsk)
                _assert_pbl_column_is_physical(
                    _pbl_column(col, sfc, xland=xland, tsk=tsk), col, tag)
            except AssertionError:
                red.add("pbl")
        try:
            sfc = _sfc_column(col, xland=1.0, tsk=305.0)
            _assert_the_sealed_column_conserves_vapour(
                _sealed_column(col, sfc), col)
        except AssertionError:
            red.add("conservation")
        try:
            _assert_the_stable_column_mixes_less()
        except AssertionError:
            red.add("ordering")
        return red

    assert _bars() == set(), "the unmutated tree must be green first"
    monkeypatch.setattr(myj_ref, victim, stub)
    assert _bars() == set(expect), (
        f"stubbing {victim} did not break exactly the measured bars")


def test_a_stably_stratified_calm_column_barely_mixes():
    """A physical ordering, not a magnitude: strong inversion, no shear.

    MYJ's mixing length collapses to EPSL where the flux Richardson number
    passes the forbidden-area threshold, so the exchange coefficients of a
    cold, calm, strongly stable column must be far smaller than those of a
    warm, sheared one.
    """
    _assert_the_stable_column_mixes_less()


def _assert_the_stable_column_mixes_less():
    """THE physical-ordering bar, in one function (see the test above)."""
    stable = _sounding(dtheta_dz=0.02, wind=0.2)
    calm = _sfc_column(stable, xland=1.0, tsk=280.0)
    quiet = _pbl_column(stable, calm, xland=1.0, tsk=280.0)
    active = _sounding(dtheta_dz=0.001, wind=12.0)
    windy = _sfc_column(active, xland=1.0, tsk=310.0)
    mixed = _pbl_column(active, windy, xland=1.0, tsk=310.0)
    assert float(np.max(quiet["exch_h"])) < float(np.max(mixed["exch_h"]))
    assert float(np.max(quiet["tke"])) < float(np.max(mixed["tke"]))


# ---------------------------------------------------------------------------
# The pairing law.
# ---------------------------------------------------------------------------

def _cfg(**kwargs):
    base = dict(nx=8, ny=8, nz=20, dx=3000.0, dy=3000.0, dt=12.0,
                ztop=15000.0, run_seconds=120.0)
    base.update(kwargs)
    return RunConfig(**base)


def test_the_schemes_are_in_the_config_schema():
    assert MYJ_SFCLAY_SCHEME == 2 and MYJ_PBL_SCHEME == 2
    assert MYJ_SFCLAY_SCHEME in SURFACE_LAYER_SCHEMES
    assert MYJ_PBL_SCHEME in PBL_SCHEMES


@pytest.mark.parametrize("sfclay,pbl", [(2, 0), (2, 1), (2, 5), (2, 11),
                                        (0, 2), (1, 2), (5, 2), (91, 2)])
def test_a_half_myj_suite_is_refused_at_load(sfclay, pbl):
    """WRF's own fatal, plus WOOF's stated reverse.

    module_physics_init.F:3770-3772 refuses bl_pbl_physics=2 without
    isfc=2; the reverse direction is WOOF's, because the Eta layer
    publishes no MOL/ZOL/PSIM/PSIH and every other ported PBL reads one.
    """
    with pytest.raises(ValueError) as excinfo:
        validate_myj_pairing(_cfg(sf_sfclay_physics=sfclay,
                                  bl_pbl_physics=pbl))
    message = str(excinfo.value)
    assert "MYJ" in message and "Eta similarity" in message
    # The refusal has to say what to do, not just that it refused.
    assert "sf_sfclay_physics=" in message and "bl_pbl_physics=" in message


#: The ported PBL slot and the surface layer each scheme is paired with.
#: SASE (900) is not here: it states its own moist requirement, for its own
#: reason (saturated-N2 stability and condensate rows), at plan review and
#: again at the driver, and it is its own audit item.
_DRY_PBL_SLOT = ((1, 1), (2, 2), (5, 5), (9, 1), (11, 1))


def test_a_dry_pbl_config_is_admitted_by_both_config_doors():
    """The dry refusal is retired at plan review, and the reason it was wrong.

    It read ``PRESENT(qv_curr) .AND. PRESENT(qc_curr)``
    (module_pbl_driver.F:1500-1513) as MYJ's own law.  That is the
    OPTIONAL-argument plumbing every scheme arm in WRF's PBL driver
    carries, and in WRF the presence of qv/qc is decided by the mp_physics
    package, not by the PBL selector.

    THIS TEST MEASURES THE CONFIG DOORS ONLY -- ``validate_myj_pairing``
    and ``validate_run_config``.  Admission is not a run: the test that
    says a dry PBL configuration STARTS and MIXES is
    :func:`test_a_dry_pbl_run_reaches_the_driver_and_mixes` below, and
    until this candidate it would have failed for every scheme in the slot
    because ``initialize_physics`` carried a blanket dry refusal of its
    own.  Both are needed; neither substitutes for the other.

    The property here is symmetry for the existing dry-capable PBL slot.
    CCPP requires moist buoyancy inputs, so its admission gate is outside
    this dry capability claim.
    """
    from woof.config import (CCPP_TKE_EDMF_PBL_SCHEME, PBL_SCHEMES,
                              validate_run_config)

    validate_myj_pairing(_cfg(sf_sfclay_physics=2, bl_pbl_physics=2,
                              moist=False))
    for pbl in PBL_SCHEMES:
        # Preserve the existing SASE exclusion and the CCPP moist-input gate.
        if pbl in (900, CCPP_TKE_EDMF_PBL_SCHEME):
            continue
        sfclay = 2 if pbl == 2 else 1
        validate_run_config(_cfg(sf_sfclay_physics=sfclay,
                                 bl_pbl_physics=pbl, moist=False))
    assert {pbl for pbl, _ in _DRY_PBL_SLOT} == {
        pbl for pbl in PBL_SCHEMES
        if pbl not in (0, 900, CCPP_TKE_EDMF_PBL_SCHEME)}, (
        "a scheme joined the PBL slot without joining the dry run below")


@requires_gpu
@pytest.mark.parametrize("pbl,sfclay", _DRY_PBL_SLOT)
def test_a_dry_pbl_run_reaches_the_driver_and_mixes(pbl, sfclay):
    """A dry PBL configuration STARTS, stays finite, and actually mixes.

    The refusal this replaces lived at ``initialize_physics`` and read
    ``bl_pbl_physics and state.qv is None`` -- every scheme in the slot,
    after the config door had already admitted the run, naming no breakage
    and no way out.  A dry state is not an absent state:
    ``_atmosphere_for_physics`` hands every seam the persistent zero
    ``physics_dry_qv``/``physics_dry_qc`` planes, the moisture rows solve
    against zero surface boundary values so their tendencies are exactly
    zero, and with no moist scalar advance nothing consumes them.

    The bars are the ones that separate "called" from "coupled": the
    diffusivity grows away from the cold state and the PBL top is a real
    depth, on a state whose ``qv`` is ``None``.
    """
    import cupy as cp

    from woof.core.dycore import run_steps
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.physics import (
        DECLARED_CONSTANT_GLW_WM2, initialize_physics)
    from woof.core.state import init_at_rest

    cfg = _cfg(nx=6, ny=5, nz=24, dt=6.0, run_seconds=60.0, moist=False,
               sf_sfclay_physics=sfclay, bl_pbl_physics=pbl,
               sf_surface_physics=2, num_soil_layers=4, bldt=0.0, radt=0.0,
               cu_physics=0, mp_physics=0, km_opt=4, c_s=0.25)
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(
        coord, lambda z: 300.0 + 0.003 * np.asarray(z, np.float64),
        p_surf=cfg.p_surf, ztop=cfg.ztop)
    state = init_at_rest(cfg, coord, base)
    assert state.qv is None, "the fixture is not dry"
    state.u[...] = cp.asarray(
        np.full(tuple(state.u.shape), 6.0), dtype=state.u.dtype)
    # A sunlit surface (swdown 900 W m-2), so the column is heated from
    # below and every scheme in the slot has turbulence to produce.  Under
    # swdown = 0 Noah cools the skin at once, the column is stable and
    # shear-free, and the diffusivity bar below measured only YSU's 0.01
    # m2 s-1 background floor; the UW scheme (bl_pbl_physics = 9) has no
    # such floor and returns WRF's exact zero there, so a night fixture
    # could not tell coupling from a floor for any of them.
    driver = initialize_physics(state, cfg, landmask=1.0, tsk=305.0,
                                glw=DECLARED_CONSTANT_GLW_WM2, swdown=900.0)

    run_steps(state, cfg, 10)

    assert driver.call_counts["ysu"] > 0        # the PBL-slot counter
    assert driver.call_counts["sfclay"] > 0
    for name in ("exch_h", "pblh", "hfx", "ust"):
        values = cp.asnumpy(driver.fields[name])
        assert np.all(np.isfinite(values)), name
    # Coupled, not merely called: the diffusivity left the cold state and
    # the PBL top is a depth rather than the first interface.
    assert float(np.max(cp.asnumpy(driver.fields["exch_h"]))) > 0.0
    assert float(np.max(cp.asnumpy(driver.fields["pblh"]))) > 0.0
    assert np.all(np.isfinite(cp.asnumpy(state.thp)))


@requires_gpu
def test_myj_cold_start_writes_wrfs_initial_ust():
    """MYJSFCINIT sets USTAR = 0.1 at every point on a cold start, so
    wrf.exe's t=0 history frame writes UST = 0.1; 2.9.0 wrote the MM5 cold
    start 1e-4 there (every UST word of NCAR's CONUS benchmark differed from
    WRF at t=0)."""
    import cupy as cp

    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.physics import (
        DECLARED_CONSTANT_GLW_WM2, initialize_physics)
    from woof.core.state import init_at_rest

    cfg = _cfg(nx=6, ny=5, nz=24, dt=6.0, run_seconds=60.0, moist=False,
               sf_sfclay_physics=2, bl_pbl_physics=2,
               sf_surface_physics=2, num_soil_layers=4, bldt=0.0, radt=0.0,
               cu_physics=0, mp_physics=0, km_opt=4, c_s=0.25)
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(
        coord, lambda z: 300.0 + 0.003 * np.asarray(z, np.float64),
        p_surf=cfg.p_surf, ztop=cfg.ztop)
    state = init_at_rest(cfg, coord, base)
    driver = initialize_physics(state, cfg, landmask=1.0, tsk=290.0,
                                glw=DECLARED_CONSTANT_GLW_WM2, swdown=0.0)
    ust = cp.asnumpy(driver.fields["ust"])
    assert np.all(ust == np.float32(0.1)), np.unique(ust)


def test_the_matched_pair_is_admitted():
    validate_myj_pairing(_cfg(sf_sfclay_physics=2, bl_pbl_physics=2,
                              moist=True))
    # And the whole loader accepts the runnable suite.
    from woof.config import validate_run_config
    validate_run_config(_cfg(sf_sfclay_physics=2, bl_pbl_physics=2,
                             moist=True, sf_surface_physics=2,
                             num_soil_layers=4))


def test_the_mm5_only_roughness_knobs_stay_refused_under_myj():
    """isftcflx/iz0tlnd would do nothing here, so they are refused.

    WRF passes IVGTYP/ISURBAN/IZ0TLND into MYJSFC and then never reads
    them: the Chen-Zhang CZIL block is commented out and CZIL=0.1 is
    hard-coded (module_sf_myjsfc.F:689-697).  Accepting the knobs would let
    a user believe a thermal-roughness option was in effect.
    """
    from woof.config import validate_run_config
    with pytest.raises(ValueError, match="isftcflx/iz0tlnd"):
        validate_run_config(_cfg(sf_sfclay_physics=2, bl_pbl_physics=2,
                                 moist=True, sf_surface_physics=2,
                                 num_soil_layers=4, iz0tlnd=1))


# ---------------------------------------------------------------------------
# The rest of the integration surface.
# ---------------------------------------------------------------------------

def test_the_driver_routes_each_half_to_its_own_runner():
    from woof.core.physics import (PHYSICS_SLOT_DISPATCH,
                                    resolve_physics_slot)
    assert PHYSICS_SLOT_DISPATCH["sf_sfclay_physics"][2] == "_run_myj_sfclay"
    assert PHYSICS_SLOT_DISPATCH["bl_pbl_physics"][2] == "_run_myj_pbl"
    assert resolve_physics_slot("bl_pbl_physics", 2) == "_run_myj_pbl"
    assert resolve_physics_slot("sf_sfclay_physics", 2) == "_run_myj_sfclay"


def test_the_registry_rows_are_accurate_about_the_evidence():
    registry = json.loads(
        (_ROOT / "woof" / "physics_registry_v2.json").read_text("utf-8"))
    myj = registry["components"]["pbl"]["options"]["myj"]
    eta = registry["components"]["surface_layer"]["options"]["eta-similarity"]
    for option, selector, value in ((myj, "bl_pbl_physics", 2),
                                    (eta, "sf_sfclay_physics", 2)):
        assert option["implemented"] is True
        assert option["maturity"] == "implemented-unverified"
        assert option["reachability"] == {"state": "template"}
        assert option["selectors"] == {selector: value}
        assert option["scientific_evidence"] == "none"
    # The PBL's evidence string must name the oracle that grades it bitwise,
    # and must no longer carry the ground-relative divergence it retired.
    myj_text = " ".join(myj["warnings"])
    assert "BIT-IDENTICAL" in myj_text
    assert "tests/test_myjpbl_wrf461_parity.py" in myj_text
    assert "tools/myjpbl_wrf461_oracle" in myj_text
    assert "NO ORACLE COMPARISON" not in myj_text
    assert "GROUND-relative" not in myj_text
    # So must the surface layer's.
    eta_text = " ".join(eta["warnings"])
    assert "BIT-IDENTICAL" in eta_text
    assert "tests/test_myjsfc_wrf461_parity.py" in eta_text
    assert "tools/myjsfc_wrf461_oracle" in eta_text
    assert "NO ORACLE COMPARISON" not in eta_text
    assert "GROUND-relative" not in eta_text
    # Audit R-067 gave the pair its first named suite, so reachability is
    # 'template' now and the old "no template may select either half" is
    # retired with it.  What survives is the property that actually
    # protects the port: a template selects the pair or neither half,
    # because MYJ's own constraint requires the Eta surface layer and the
    # Eta surface layer's requires MYJ.
    paired = 0
    for template in registry["templates"].values():
        components = template.get("components", {})
        selects = (components.get("pbl") == "myj",
                   components.get("surface_layer") == "eta-similarity")
        assert selects[0] == selects[1], template.get("label")
        paired += int(selects[0])
    assert paired == 1, (
        "the pair had no named suite at all until R-067 and has exactly "
        f"one now; the registry declares {paired}")


def test_the_pair_is_selectable_per_domain_and_only_as_a_pair():
    registry = json.loads(
        (_ROOT / "woof" / "physics_registry_v2.json").read_text("utf-8"))
    allowed = registry["runner_routes"][
        "tools.prepared_domain_tree_forecast"]["allowed_component_options"]
    assert "myj" in allowed["pbl"]
    assert "eta-similarity" in allowed["surface_layer"]
    myj = registry["components"]["pbl"]["options"]["myj"]
    eta = registry["components"]["surface_layer"]["options"]["eta-similarity"]
    assert myj["constraints"]["requires_components"]["surface_layer"] == [
        "eta-similarity"]
    assert eta["constraints"]["requires_components"]["pbl"] == ["myj"]


def test_a_namelist_naming_myj_imports_it_natively():
    from woof.namelist_import import _BL_MAP, _SFCLAY_ALLOWED
    wrf_value, wrf_label, gpuwm_label = _BL_MAP[2]
    assert wrf_value == 2, "MYJ must not map onto another scheme"
    assert wrf_label == gpuwm_label == "MYJ", (
        "a differing pair of labels is how a SUBSTITUTION is spelled in "
        "this table; MYJ is a native import")
    assert 2 in _SFCLAY_ALLOWED


def test_the_restart_identities_name_both_halves():
    from woof.io.restart import (PBL_ALGORITHM_IDENTITIES,
                                  SURFACE_LAYER_ALGORITHM_IDENTITIES)
    assert "myj" in PBL_ALGORITHM_IDENTITIES[2]
    assert "v4.6.1" in PBL_ALGORITHM_IDENTITIES[2]
    assert "eta-similarity" in SURFACE_LAYER_ALGORITHM_IDENTITIES[2]
    # Every identity is distinct: a resume must not silently cross schemes.
    assert len(set(PBL_ALGORITHM_IDENTITIES.values())) == len(
        PBL_ALGORITHM_IDENTITIES)


def test_preflights_myj_output_roster_matches_the_launchers():
    """The memory-budget roster, gated to the launcher that allocates it.

    ``woof/core/preflight.py`` prices the MYJ per-call output bundle from
    its OWN ``_MYJ_3D``/``_MYJ_2D`` literals, because preflight must stay
    importable without CuPy and ``woof.core.myjpbl`` imports it at module
    scope.  A duplicated roster is only safe if something fails when the
    two drift, which is the ``_SHINHONG_3D`` idiom preflight's comment
    cites (tests/test_shinhong_runtime.py::
    test_preflight_roster_and_cold_start_match_the_authorities) -- so MYJ
    gets the same gate rather than the same comment.

    The Eta surface layer's empty row is asserted too: ``myjsfc.py``
    allocates nothing per call, so the budget must price nothing, and a
    future per-call buffer there has to come here to be admitted.
    """
    import dataclasses

    from woof.core import preflight as pf
    from woof.core.myjpbl import (MYJ_PBL_COLUMN_OUTPUTS,
                                   MYJ_PBL_SURFACE_OUTPUTS)

    assert pf._MYJ_3D == MYJ_PBL_COLUMN_OUTPUTS
    assert pf._MYJ_2D == MYJ_PBL_SURFACE_OUTPUTS

    off = _cfg(sf_sfclay_physics=1, bl_pbl_physics=1)
    assert pf.myj_output_transient_shapes(off) == {}
    on = dataclasses.replace(off, sf_sfclay_physics=2, bl_pbl_physics=2)
    shapes = pf.myj_output_transient_shapes(on)
    assert set(shapes) == {
        f"myj_output/{name}"
        for name in MYJ_PBL_COLUMN_OUTPUTS + MYJ_PBL_SURFACE_OUTPUTS}
    assert all(shapes[f"myj_output/{n}"] == (on.nz, on.ny, on.nx)
               for n in MYJ_PBL_COLUMN_OUTPUTS)
    assert all(shapes[f"myj_output/{n}"] == (on.ny, on.nx)
               for n in MYJ_PBL_SURFACE_OUTPUTS)
    # ``tke`` is CARRIED state, allocated once by initialize_physics and
    # already priced as a driver field: pricing it here would double-count.
    assert "myj_output/tke" not in shapes


def test_the_documented_column_ceiling_matches_the_kernel_define():
    from woof.core.myjpbl import MYJ_MAX_COLUMN_LEVELS
    source = (_ROOT / "woof" / "core" / "kernels" / "myjpbl.cu").read_text(
        "utf-8")
    line = next(ln for ln in source.splitlines()
                if ln.startswith("#define MYJ_KMAX"))
    assert int(line.split()[-1]) == MYJ_MAX_COLUMN_LEVELS


def test_the_surface_kernels_pblh_accumulator_is_column_local():
    """``ZINT(I,K,J)=ZINT(I,K+1,J)+DZ(I,KFLIP,J)`` (:177-184) stays in ``I``.

    Every ``dz`` the PBLH scan touches must carry this thread's column, and
    the CPU authority cannot say so: ``np_myjsfc_column`` takes one ``(nz,)``
    column and ``_interface_heights`` has no column index to get wrong.  The
    device gate below is the numerical statement and it needs a card; this
    is the same claim on the shipped bytes, and it runs everywhere.

    ``dz_a[0]`` -- layer 0 of column 0 -- was the accumulator's seed for
    every thread until 2.6.5, so ``PBLH`` was wrong by
    ``dz[0][0] - dz[0][col]`` on every column but the first, and ``PBLH``
    re-enters ``SFCDIF`` as ``BTGH`` (:431-435) rather than only being
    published.
    """
    source = (_ROOT / "woof" / "core" / "kernels" / "myjsfc.cu").read_text(
        "utf-8")
    code = [ln.split("//")[0].strip() for ln in source.splitlines()]
    assert [ln for ln in code if "dz_a[" in ln] == [
        "real zlow = zground + dz_a[col];",
        "zcum = zcum + dz_a[(size_t)iz * st + col];",
    ]
    # ZINT(KTE+1)=HT (module_sf_myjsfc.F:165) is this thread's own HT too.
    assert [ln for ln in code if "ht_a[" in ln] == [
        "real zground = ht_a[col];",
    ]


def test_physics_md_carries_the_pair_in_its_tables():
    text = (_ROOT / "docs" / "public" / "PHYSICS.md").read_text("utf-8")
    assert "| MYJ " in text and "Eta similarity" in text
    # The doc row must repeat the accurate tier, not just the label.
    myj_rows = [ln for ln in text.splitlines()
                if ln.startswith("| MYJ ") or "Eta similarity |" in ln]
    assert myj_rows, "no MYJ row in PHYSICS.md"
    for row in myj_rows:
        assert "implemented-unverified" in row, row


# ---------------------------------------------------------------------------
# The CUDA mirrors.
# ---------------------------------------------------------------------------

def _to_device(cp, array):
    """Contiguous float32 device copy.

    ``cp`` is passed IN rather than imported here on purpose: a
    module-level or helper-level ``import cupy`` makes tests/conftest.py
    mark this whole file ``gpu`` (``_cupy_scope``), which would hide every
    CPU-authority test above behind a device.
    """
    return cp.ascontiguousarray(cp.asarray(array, dtype=np.float32))


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("xland,tsk,tag", [(1.0, 305.0, "land"),
                                           (2.0, 295.0, "water")])
def test_the_surface_kernel_agrees_with_the_cpu_authority(xland, tsk, tag):
    """CPU-vs-CUDA, on the same column, word for word.

    This used to be a tolerance check (rel 2e-4, abs 1e-6) because neither
    half pinned its libm.  Both are now graded bit-identical to WRF v4.6.1's
    MYJSFC on the column oracle (tests/test_myjsfc_wrf461_parity.py: the
    kernel on GPU, the CPU authority on CPU), and the tolerance had become
    the wrong instrument: on the water column below, a cold-start QZ0 that
    equals QLOW to within one ULP makes QFX and LH a pure rounding residue
    (WRF writes LH = -1.7143344e-05 W m-2, the kernel the same word, and
    the CPU authority wrote -0.0 while it still used NumPy's log/exp/pow,
    one ULP off on QZ0).  With both halves pinned the residue is the same
    word on both sides, so the bar is bitwise.
    """
    import cupy as cp

    from woof.core.myjsfc import (MYJ_SFCLAY_INOUT, MYJ_SFCLAY_OUTPUTS,
                                   launch_myj_sfclay)

    col = _sounding()
    reference = _sfc_column(col, xland=xland, tsk=tsk)
    shape = (1, 1)
    nz = col["dz"].shape[0]
    columns = {
        "dz": _to_device(cp, col["dz"].reshape(nz, 1, 1)),
        "tke": _to_device(cp, col["tke"].reshape(nz, 1, 1)),
    }
    surface = {
        "u1": _to_device(cp, np.full(shape, col["u"][0])),
        "v1": _to_device(cp, np.full(shape, col["v"][0])),
        "t1": _to_device(cp, np.full(shape, col["t"][0])),
        "th1": _to_device(cp, np.full(shape, col["th"][0])),
        "qv1": _to_device(cp, np.full(shape, col["qv"][0])),
        "qc1": _to_device(cp, np.full(shape, col["qc"][0])),
        "p1": _to_device(cp, np.full(shape, col["p"][0])),
        "psfc": _to_device(cp, np.full(shape, 1.0e5)),
        "tsk": _to_device(cp, np.full(shape, tsk)),
        "xland": _to_device(cp, np.full(shape, xland)),
        "mavail": _to_device(cp, np.ones(shape)),
        "z0base": _to_device(cp, np.full(shape, 0.1)),
        "ht": _to_device(cp, np.zeros(shape)),
    }
    seed = {"ust": 0.1, "znt": 0.1, "thz0": 300.0, "qz0": 0.012,
            "uz0": 0.0, "vz0": 0.0, "qsfc": 0.012, "akhs": 0.01,
            "akms": 0.01}
    state = {name: _to_device(cp, np.full(shape, seed[name]))
             for name in MYJ_SFCLAY_INOUT}
    outputs = {name: cp.zeros(shape, dtype=np.float32)
               for name in MYJ_SFCLAY_OUTPUTS}
    launch_myj_sfclay(columns, surface, state, outputs, itimestep=1)
    for name in (*MYJ_SFCLAY_INOUT, *MYJ_SFCLAY_OUTPUTS):
        source = state if name in state else outputs
        device = float(cp.asnumpy(source[name]).ravel()[0])
        host = float(reference[name])
        assert np.isfinite(device), f"{tag}: {name} non-finite on device"
        assert (np.float32(device).tobytes()
                == np.float32(host).tobytes()), (
            f"{tag}: {name} CPU {host!r} vs CUDA {device!r}")


@pytest.mark.gpu
@requires_gpu
def test_the_surface_kernel_pblh_uses_each_columns_own_dz():
    """Two columns, two different lowest-layer depths, one launch.

    The conformance gate above runs at ``shape = (1, 1)``, so ``n == 1``,
    every thread is ``col == 0`` and a column-0 index is indistinguishable
    from a per-column one; ``_sounding()`` is uniform ``dz`` besides, so
    even widening it would not separate them.  Two columns with DIFFERENT
    ``dz[0]`` is the minimum fixture that does, and it is what a
    terrain-following coordinate hands the kernel on any real domain -- the
    lowest layer thins over high ground.

    ``PBLH`` is not diagnostic-only: ``module_sf_myjsfc.F:431-435`` feeds it
    back as ``BTGH = BTGX*PBLH`` above 1000 m, which sets ``WSTAR2`` and
    therefore ``USTAR``, ``AKMS`` and ``AKHS``.
    """
    import cupy as cp

    from woof.core.myjsfc import (MYJ_SFCLAY_INOUT, MYJ_SFCLAY_OUTPUTS,
                                   launch_myj_sfclay)

    xland, tsk = 1.0, 305.0
    thin = _sounding()
    thin["dz"] = thin["dz"].copy()
    thin["dz"][0] = F(60.0)                  # this column sits on a hill
    thick = _sounding()                      # uniform 100 m
    hosts = [_sfc_column(c, xland=xland, tsk=tsk) for c in (thick, thin)]

    nz = thick["dz"].shape[0]
    shape = (1, 2)
    stacked = lambda name: _to_device(cp, np.stack(
        [thick[name], thin[name]], axis=-1).reshape(nz, 1, 2))
    columns = {"dz": stacked("dz"), "tke": stacked("tke")}
    surface = {name: _to_device(cp, np.full(shape, value)) for name, value in
               (("u1", thick["u"][0]), ("v1", thick["v"][0]),
                ("t1", thick["t"][0]), ("th1", thick["th"][0]),
                ("qv1", thick["qv"][0]), ("qc1", thick["qc"][0]),
                ("p1", thick["p"][0]), ("psfc", 1.0e5), ("tsk", tsk),
                ("xland", xland), ("mavail", 1.0), ("z0base", 0.1),
                ("ht", 0.0))}
    seed = {"ust": 0.1, "znt": 0.1, "thz0": 300.0, "qz0": 0.012,
            "uz0": 0.0, "vz0": 0.0, "qsfc": 0.012, "akhs": 0.01,
            "akms": 0.01}
    state = {name: _to_device(cp, np.full(shape, seed[name]))
             for name in MYJ_SFCLAY_INOUT}
    outputs = {name: cp.zeros(shape, dtype=np.float32)
               for name in MYJ_SFCLAY_OUTPUTS}
    launch_myj_sfclay(columns, surface, state, outputs, itimestep=1)

    pblh = cp.asnumpy(outputs["pblh"]).ravel()
    for col, host in enumerate(hosts):
        assert pblh[col] == pytest.approx(float(host["pblh"]),
                                          rel=2.0e-4, abs=1.0e-6), (
            f"column {col}: CPU {float(host['pblh'])!r} vs CUDA "
            f"{pblh[col]!r}")
    # Stated explicitly so the gate cannot pass by both columns agreeing on
    # column 0's answer: the two differ by exactly the lowest-layer spread.
    assert float(pblh[0] - pblh[1]) == pytest.approx(40.0, abs=1.0e-3)


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("xland,tsk,tag", [(1.0, 305.0, "land"),
                                           (2.0, 295.0, "water")])
def test_the_pbl_kernel_agrees_with_the_cpu_authority(xland, tsk, tag):
    """The same conformance statement for the PBL translation unit."""
    import cupy as cp

    from woof.core.myjpbl import MYJ_PBL_INOUT, myj_pbl_step

    col = _sounding()
    sfc = _sfc_column(col, xland=xland, tsk=tsk)
    reference = _pbl_column(col, sfc, xland=xland, tsk=tsk)
    nz = col["dz"].shape[0]
    shape = (1, 1)

    def column(name):
        return _to_device(cp, np.asarray(col[name], np.float32).reshape(nz, 1, 1))

    columns = {name: column(name) for name in
               ("dz", "u", "v", "th", "exner", "qv", "qc", "qi", "p")}
    columns["t"] = column("t")
    tke = column("tke")
    surface = {
        "psfc": _to_device(cp, np.full(shape, 1.0e5)),
        "ust": _to_device(cp, np.full(shape, float(sfc["ust"]))),
        "tsk": _to_device(cp, np.full(shape, tsk)),
        "chklowq": _to_device(cp, np.ones(shape)),
        "xland": _to_device(cp, np.full(shape, xland)),
        "sice": _to_device(cp, np.zeros(shape)),
        "snow": _to_device(cp, np.zeros(shape)),
        "akhs": _to_device(cp, np.full(shape, float(sfc["akhs"]))),
        "akms": _to_device(cp, np.full(shape, float(sfc["akms"]))),
        "elflx": _to_device(cp, np.full(shape, 50.0)),
        "uz0": _to_device(cp, np.full(shape, float(sfc["uz0"]))),
        "vz0": _to_device(cp, np.full(shape, float(sfc["vz0"]))),
        # ZINT(KTE+1)=HT; the CPU authority below runs its default ht=0.
        "ht": _to_device(cp, np.zeros(shape)),
    }
    seed = {"thz0": float(sfc["thz0"]), "qz0": float(sfc["qz0"]),
            "qsfc": float(sfc["qsfc"]), "ct": float(sfc["ct"])}
    state = {name: _to_device(cp, np.full(shape, seed[name]))
             for name in MYJ_PBL_INOUT}
    out = myj_pbl_step(columns, surface, state, tke,
                       dtturbl=60.0, flqi=True)
    # THE RESOLUTION LIMIT, stated instead of assumed.  A tendency is
    # (new - old)/dt of a float32 field, so it can only take values that
    # are integer multiples of that field's ULP divided by dt.  Over water
    # with AKHS ~ 6e-3 the moisture tendency IS one such quantum: the
    # solved q moves by a single last place in 60 s.  Comparing that
    # against a relative tolerance compares two roundoffs, so each
    # tendency row is compared against BOTH a relative bar and its own
    # quantum, and passes on whichever is looser.  A difference of a few
    # quanta is a last-place difference in the solved state; a difference
    # of many is a difference in the scheme.
    quantum = {
        "rublten": float(np.max(np.spacing(col["u"]))) / 60.0,
        "rvblten": float(np.max(np.spacing(col["v"]))) / 60.0,
        "rthblten": float(np.max(np.spacing(col["th"]))) / 60.0,
        "rqvblten": float(np.max(np.spacing(col["qv"]))) / 60.0,
        "rqcblten": float(np.max(np.spacing(col["qv"]))) / 60.0,
    }
    for name in ("rublten", "rvblten", "rthblten", "rqvblten", "rqcblten",
                 "el_myj", "exch_h"):
        device = cp.asnumpy(out[name]).reshape(nz)
        host = np.asarray(reference[
            {"el_myj": "el"}.get(name, name)], np.float64)
        scale = max(float(np.max(np.abs(host))), 1.0e-30)
        bar = max(5.0e-3 * scale, 4.0 * quantum.get(name, 0.0))
        assert np.all(np.isfinite(device)), f"{tag}: {name}"
        worst = float(np.max(np.abs(device - host)))
        assert worst <= bar, (
            f"{tag}: {name} CPU-vs-CUDA {worst:.3e} beyond the stated bar "
            f"{bar:.3e} (relative 5e-3 of {scale:.3e}; four quanta of "
            f"{quantum.get(name, 0.0):.3e})")
    assert float(cp.asnumpy(out["pblh"]).ravel()[0]) == pytest.approx(
        float(reference["pblh"]), rel=1.0e-5)
    assert int(cp.asnumpy(out["kpbl"]).ravel()[0]) == int(reference["kpbl"])
    tke_new = cp.asnumpy(tke).reshape(nz)
    assert np.max(np.abs(tke_new - reference["tke"])) <= 5.0e-3 * max(
        float(np.max(np.abs(reference["tke"]))), 1.0e-30)
    # And the pair really is comparing something: at least one row of the
    # reference has to be above its own quantum, or the whole comparison
    # above would be vacuous.
    assert float(np.max(np.abs(reference["rthblten"]))) > (
        10.0 * quantum["rthblten"]), (
        "the fixture produced no heat tendency above its resolution limit; "
        "the CPU-vs-CUDA comparison would be vacuous")


@pytest.mark.gpu
@requires_gpu
def test_a_myj_forecast_advances_through_the_shipped_seams():
    """initialize_physics + PhysicsDriver.compute, not a launcher call.

    This is the test that would catch a half-registered port: a scheme can
    be transcribed perfectly and still be unreachable because a field is
    unallocated, a dispatch row is missing, or the surface layer never runs
    before the PBL that reads its output.
    """
    import cupy as cp

    from woof.core.dycore import run_steps
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.physics import (
        DECLARED_CONSTANT_GLW_WM2, initialize_physics)
    from woof.core.state import init_at_rest

    cfg = _cfg(nx=6, ny=5, nz=24, dt=6.0, run_seconds=60.0, moist=True,
               sf_sfclay_physics=2, bl_pbl_physics=2, sf_surface_physics=2,
               num_soil_layers=4, bldt=0.0, radt=0.0, cu_physics=0,
               mp_physics=0, km_opt=4, c_s=0.25)
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(
        coord, lambda z: 300.0 + 0.003 * np.asarray(z, np.float64),
        p_surf=cfg.p_surf, ztop=cfg.ztop)
    state = init_at_rest(cfg, coord, base)
    state.u[...] = cp.asarray(
        np.full(tuple(state.u.shape), 6.0), dtype=state.u.dtype)
    # radt=0.0 with Noah on: this column runs no radiation scheme, so it
    # declares its downward longwave AND its downward shortwave instead of
    # letting either be invented.  That is the second remedy the carrier
    # contract names, and it is the right one here because the subject is
    # the MYJ PBL and surface layer advancing through the shipped seams,
    # not the sky.  swdown=0.0 is the value the buffer always held; the
    # declaration moves no number, it names the number's origin.
    driver = initialize_physics(state, cfg, landmask=1.0, tsk=305.0,
                                glw=DECLARED_CONSTANT_GLW_WM2, swdown=0.0)

    # The scheme's own fields exist under its own selectors, and nothing
    # else's do.
    for name in ("thz0", "qz0", "uz0", "vz0", "akhs", "akms",
                 "tshltr", "qshltr", "pshltr", "u10e", "v10e", "mixht"):
        assert name in driver.fields, name
    assert driver.fields["tke_myj"].shape == (cfg.nz, cfg.ny, cfg.nx)
    assert driver.fields["el_myj"].shape == (cfg.nz, cfg.ny, cfg.nx)
    # The SHIPPED cold start, before a single step: MYJPBLINIT's
    # TKE_MYJ=EPSQ2 over the whole column (module_bl_myjpbl.F:1725), and
    # EL_MYJ at the zero MYJPBL rewrites every call anyway (:341).  A zero
    # TKE seed here is the defect the CPU test above pins the consequences
    # of, so it is checked on the real allocation too.
    from woof.core.physics import MYJ_TKE_COLD_START
    seed = cp.asnumpy(driver.fields["tke_myj"])
    assert seed.min() == seed.max() == np.float32(MYJ_TKE_COLD_START)
    assert not np.any(cp.asnumpy(driver.fields["el_myj"]))

    run_steps(state, cfg, 10)

    assert driver.call_counts["sfclay"] > 0
    assert driver.call_counts["ysu"] > 0        # the PBL-slot counter
    tke = cp.asnumpy(driver.fields["tke_myj"])
    assert np.all(np.isfinite(tke))
    assert np.all(tke >= 0.5 * float(EPSQ2) - 1e-6)
    # RE-DERIVED BAR.  The old one ("max TKE above 0.5*EPSQ2") stopped being
    # a bar the moment the cold start became EPSQ2: the seed alone satisfies
    # it.  What actually proves the scheme ran is PRODQ2's lower boundary,
    # q2(LMH)=AMAX1((B1**(2/3)*USTAR)*USTAR, EPSQ2) (module_bl_myjpbl.F:1164)
    # -- VDIFQ never writes that row (:1330-1406 solves KTS..LMH-1) and the
    # publication loop only floors it, so the ground-adjacent TKE is that
    # expression exactly, computed from the driver's own u*.
    b1_23 = float(np.float32(11.87799326209552761)
                  ** np.float32(np.float32(2.0) / np.float32(3.0)))
    ust = cp.asnumpy(driver.fields["ust"])
    np.testing.assert_allclose(
        tke[0], 0.5 * np.maximum(b1_23 * ust ** 2, float(EPSQ2)),
        rtol=1e-5, err_msg="the surface TKE source did not set q2(LMH)")
    # And the column is no longer the uniform cold-start seed: the scheme
    # wrote it, rather than the allocation being all that is on show.
    assert float(tke.min()) != float(tke.max())
    for name in ("akhs", "akms", "pblh", "hfx", "u10", "t2", "exch_h"):
        values = cp.asnumpy(driver.fields[name])
        assert np.all(np.isfinite(values)), name
    assert np.all(cp.asnumpy(driver.fields["akhs"]) > 0.0)
    for name in ("u", "v", "thp", "qv"):
        assert np.all(np.isfinite(cp.asnumpy(getattr(state, name)))), name
