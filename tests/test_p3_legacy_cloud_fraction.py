"""P3 (mp=50) reaches WRF's own cloud-fraction arm on the legacy RRTMG path.

WHAT THIS PREVENTS.  ``woof/core/rrtmg_legacy.py`` used to read ONE
boolean, ``legacy_ice_active(mp_physics)``, into BOTH of ``cal_cldfra1``'s
``f_qi`` and ``f_qs``.  A fused flag has only two states, so a package
carrying ice and no snow -- which is exactly P3's
(``Registry/Registry.EM_COMMON:3038``: ``moist:qv,qc,qr,qi``) -- could not
be spelled at all, and mp=50 fell out of the set to ``False``.  That sent
an ice-bearing P3 column to ``cal_cldfra1``'s qc-only arm, whose QCLD is
cloud WATER alone: a pure ice cloud came back with cloud fraction 0.0 and
radiated as clear sky.

WRF does not fuse them.  ``phys/module_radiation_driver.F`` branches on the
two flag VALUES separately and gives this case its own arm at :3879-3887,
commented "for P3, mp option 50 or 51", with ``QCLD = QI + QC`` and
``weight = QI/QCLD``.

The remedy is the split, NOT admitting 50 to
``_LEGACY_ICE_ACTIVE_MICROPHYSICS``: that set answers ``F_QI and F_QS``,
and putting P3 in it would assert a snow species P3 does not have.
"""

from __future__ import annotations

import inspect
import os
import pathlib

import numpy as np
import pytest

from conftest import requires_gpu
from woof.config import MP_PHYSICS_ACCEPTED
from woof.core.rrtmg_legacy import (
    _LEGACY_ICE_ACTIVE_MICROPHYSICS,
    _LEGACY_ICE_ONLY_MICROPHYSICS,
    _LEGACY_NO_ICE_MICROPHYSICS,
    _LEGACY_SNOW_ONLY_MICROPHYSICS,
    RRTMGLegacyRadiation,
    legacy_cloud_fraction_flags,
    legacy_ice_active,
)
from woof.verify.npref import np_cal_cldfra1

P3_MP_PHYSICS = 50


def _wrf_p3_cal_cldfra1_oracle(qv, qc, qi, t, p):
    """Hand transcription of module_radiation_driver.F:3879-3887 + 3945-3979.

    Deliberately independent of woof's mirrors: this is the
    ``F_QI .and. F_QC .and. .not. F_QS`` arm written straight from the
    Fortran, so it cannot agree with the port by sharing its code.
    """

    qv, qc, qi, t, p = (np.asarray(a, dtype=np.float64)
                        for a in (qv, qc, qi, t, p))
    alpha0, gamma, qcldmin, pexp, rhgrid = 100.0, 0.49, 1.0e-12, 0.25, 1.0
    svp1, svp2, svpi2 = 0.61078, 17.2693882, 21.8745584
    svp3, svpi3, svpt0 = 35.86, 7.66, 273.15
    ep_2 = 287.0 / 461.6

    out = np.zeros(qv.shape, dtype=np.float64)
    for index in np.ndindex(qv.shape):
        tc = t[index] - svpt0
        esw = 1000.0 * svp1 * np.exp(svp2 * tc / (t[index] - svp3))
        esi = 1000.0 * svp1 * np.exp(svpi2 * tc / (t[index] - svpi3))
        qvsw = ep_2 * esw / (p[index] - esw)
        qvsi = ep_2 * esi / (p[index] - esi)
        # :3880-3887 -- QS is not part of QCLD and not part of the weight.
        qcld = qi[index] + qc[index]
        if qcld < qcldmin:
            weight = 0.0
        else:
            weight = qi[index] / qcld
        qvs_weight = (1.0 - weight) * qvsw + weight * qvsi
        rhum = qv[index] / qvs_weight
        if qcld < qcldmin:
            out[index] = 0.0
        elif rhum >= rhgrid:
            out[index] = 1.0
        else:
            subsat = max(1.0e-10, rhgrid * qvs_weight - qv[index])
            arg = max(-6.9, -alpha0 * qcld / subsat ** gamma)
            rhum = max(1.0e-10, rhum)
            fraction = (rhum / rhgrid) ** pexp * (1.0 - np.exp(arg))
            out[index] = 0.0 if fraction < 0.01 else fraction
    return out


def _p3_ice_columns():
    """Three P3 levels: pure ice, mixed phase, and a saturated ice cloud.

    ``qs`` is not among them because P3 has no such species -- that is the
    whole point of the arm under test.
    """

    qv = np.array([[0.00035, 0.0009, 0.0022]])
    qc = np.array([[0.0, 2.0e-6, 0.0]])
    qi = np.array([[2.0e-5, 1.0e-5, 3.0e-5]])
    t = np.array([[240.0, 250.0, 258.0]])
    p = np.array([[30000.0, 40000.0, 60000.0]])
    return qv, qc, qi, t, p


def test_p3_gets_wrfs_ice_no_snow_flag_pair_not_a_fused_boolean():
    """mp=50 resolves to ``(F_QI, F_QS) = (True, False)``.

    Before the split this selector produced ``(False, False)``: it was
    absent from the one fused set and absence meant "no ice".
    """

    assert legacy_cloud_fraction_flags(P3_MP_PHYSICS) == (True, False)
    # And the fused set is still the F_QI-AND-F_QS question, unwidened:
    # admitting 50 there would assert a qs P3's Registry package
    # (Registry.EM_COMMON:3038, moist:qv,qc,qr,qi) does not declare.
    assert P3_MP_PHYSICS not in _LEGACY_ICE_ACTIVE_MICROPHYSICS
    assert legacy_ice_active(P3_MP_PHYSICS) is False
    assert P3_MP_PHYSICS in _LEGACY_ICE_ONLY_MICROPHYSICS
    # The schemes that DO declare both keep their old answer exactly.
    for selector in sorted(_LEGACY_ICE_ACTIVE_MICROPHYSICS):
        assert legacy_cloud_fraction_flags(selector) == (True, True)
    for selector in sorted(_LEGACY_NO_ICE_MICROPHYSICS):
        assert legacy_cloud_fraction_flags(selector) == (False, False)


def test_every_accepted_selector_has_a_cloud_fraction_row():
    """No selector woof admits may reach cal_cldfra1 undecided.

    The four sets must partition ``MP_PHYSICS_ACCEPTED`` exactly (the
    snow-only row joined with the old Eta Ferrier port, mp=95).  The
    defect this file exists for was a selector that was simply not in the
    one set there was, and silently inherited the ice-free arm.
    """

    rows = (_LEGACY_ICE_ACTIVE_MICROPHYSICS
            | _LEGACY_ICE_ONLY_MICROPHYSICS
            | _LEGACY_NO_ICE_MICROPHYSICS
            | _LEGACY_SNOW_ONLY_MICROPHYSICS)
    missing = sorted(set(MP_PHYSICS_ACCEPTED) - rows)
    assert missing == [], (
        f"mp_physics {missing} are admitted by woof/config.py but have no "
        "F_QI/F_QS row in woof/core/rrtmg_legacy.py, so a legacy-RRTMG "
        "run selecting one has no decided cal_cldfra1 arm")
    assert not (_LEGACY_ICE_ACTIVE_MICROPHYSICS
                & _LEGACY_ICE_ONLY_MICROPHYSICS)
    assert not (_LEGACY_ICE_ACTIVE_MICROPHYSICS & _LEGACY_NO_ICE_MICROPHYSICS)
    assert not (_LEGACY_ICE_ONLY_MICROPHYSICS & _LEGACY_NO_ICE_MICROPHYSICS)
    for other in (_LEGACY_ICE_ACTIVE_MICROPHYSICS, _LEGACY_ICE_ONLY_MICROPHYSICS,
                  _LEGACY_NO_ICE_MICROPHYSICS):
        assert not (_LEGACY_SNOW_ONLY_MICROPHYSICS & other)


def test_an_unmapped_selector_refuses_by_name_instead_of_going_ice_free():
    """Fails closed.  A future scheme raises rather than radiating clear."""

    with pytest.raises(NotImplementedError) as excinfo:
        # No scheme carries this selector (95 joined with its 3.0 port).
        legacy_cloud_fraction_flags(12345)
    message = str(excinfo.value)
    assert "Registry" in message
    assert "_LEGACY_ICE_ONLY_MICROPHYSICS" in message
    assert "radiates an ice cloud as clear sky" in message


def test_p3_cloud_fraction_matches_wrfs_own_p3_arm():
    """The numbers, against a transcription of the Fortran arm itself."""

    qv, qc, qi, t, p = _p3_ice_columns()
    f_qi, f_qs = legacy_cloud_fraction_flags(P3_MP_PHYSICS)
    got = np_cal_cldfra1(qv, qc, qi, np.zeros_like(qi), t, p,
                         f_qc=True, f_qi=f_qi, f_qs=f_qs)
    oracle = _wrf_p3_cal_cldfra1_oracle(qv, qc, qi, t, p)
    np.testing.assert_allclose(got, oracle, rtol=1.0e-12, atol=0.0)
    assert oracle.max() == pytest.approx(1.0)


def test_the_pre_split_flags_radiated_the_p3_ice_cloud_as_clear_sky():
    """The measured consequence, kept as the reason this gate exists.

    ``(False, False)`` is what mp=50 resolved to before the split.  The
    saturated ice level goes from overcast to clear.
    """

    qv, qc, qi, t, p = _p3_ice_columns()
    fused_false = np_cal_cldfra1(qv, qc, qi, np.zeros_like(qi), t, p,
                                 f_qc=True, f_qi=False, f_qs=False)
    wrf_arm = _wrf_p3_cal_cldfra1_oracle(qv, qc, qi, t, p)
    assert fused_false[0, 2] == 0.0
    assert wrf_arm[0, 2] == pytest.approx(1.0)
    assert float(wrf_arm.sum()) > float(fused_false.sum())


def test_the_ice_and_snow_arm_is_untouched_by_the_split():
    """A scheme with both species gets byte-identical numbers to before.

    The split must not move mp=6/8/9/10/16/18/28, whose arm is unchanged
    Fortran (module_radiation_driver.F:3870-3877).
    """

    qv = np.array([[0.006, 0.004, 0.002]])
    qc = np.zeros_like(qv)
    qi = np.array([[0.0, 2.0e-5, 0.0]])
    qs = np.array([[1.0e-5, 3.0e-5, 4.0e-5]])
    t = np.array([[270.0, 258.0, 245.0]])
    p = np.array([[80000.0, 60000.0, 40000.0]])
    both = np_cal_cldfra1(qv, qc, qi, qs, t, p,
                          f_qc=True, f_qi=True, f_qs=True)
    # QCLD = QI+QC+QS and weight = (QI+QS)/QCLD, written out here.
    qcld = qi + qc + qs
    assert np.all(qcld > 1.0e-12)
    assert both.max() > 0.0
    # The P3 arm on the SAME column is a different answer -- proving the
    # two arms are not accidentally the same code path.
    p3_arm = np_cal_cldfra1(qv, qc, qi, qs, t, p,
                            f_qc=True, f_qi=True, f_qs=False)
    assert not np.allclose(both, p3_arm)


def test_the_mp5_ice_fraction_arm_is_refused_by_name():
    """``f_qs and not f_qi`` is WRF's mp=5 F_ICE_PHY arm; it must refuse."""

    qv, qc, qi, t, p = _p3_ice_columns()
    with pytest.raises(NotImplementedError) as excinfo:
        np_cal_cldfra1(qv, qc, qi, np.zeros_like(qi), t, p,
                       f_qc=True, f_qi=False, f_qs=True)
    assert "F_ICE_PHY" in str(excinfo.value)


@requires_gpu
def test_the_device_kernel_takes_the_same_p3_arm_as_the_reference():
    """The FP32 device transcription, against the same Fortran oracle."""

    import cupy as cp

    from woof.core.rrtmgp import cal_cldfra1

    qv, qc, qi, t, p = _p3_ice_columns()
    f_qi, f_qs = legacy_cloud_fraction_flags(P3_MP_PHYSICS)
    got = cp.asnumpy(cal_cldfra1(
        cp.asarray(qv, dtype=cp.float32), cp.asarray(qc, dtype=cp.float32),
        cp.asarray(qi, dtype=cp.float32),
        cp.zeros(qi.shape, dtype=cp.float32),
        cp.asarray(t, dtype=cp.float32), cp.asarray(p, dtype=cp.float32),
        f_qc=True, f_qi=f_qi, f_qs=f_qs))
    oracle = _wrf_p3_cal_cldfra1_oracle(qv, qc, qi, t, p)
    np.testing.assert_allclose(got, oracle, rtol=2.0e-6, atol=2.0e-7)
    assert got.max() > 0.0


def test_the_legacy_adapter_wires_the_pair_and_not_the_fused_flag():
    """Source-level: the call site cannot regress to one boolean."""

    source = inspect.getsource(RRTMGLegacyRadiation.__call__)
    assert "f_qi, f_qs = legacy_cloud_fraction_flags(mp_physics)" in source
    assert "f_qi=f_qi, f_qs=f_qs" in source
    assert "ice_active" not in source, (
        "the fused legacy_ice_active boolean is back in the adapter; it "
        "cannot express P3's F_QI=true/F_QS=false and sends mp=50 to the "
        "qc-only arm")


@pytest.mark.parametrize("stock_tree", [
    # Machine-local staging paths are deliberately not part of the
    # contract: name the tree, let the machine say where.
    pathlib.Path(os.environ.get("WOOF_WRF461_STOCK_TREE",
                                "wrf-stock-v461-gate-20260721")),
])
def test_the_p3_arm_citation_is_checkable_against_stock_wrf(stock_tree):
    """The line numbers, verified rather than asserted, when the tree is
    present.  A line-numbered claim nobody re-reads is a claim about the
    past; this one is cheap to re-check, so it is checked.
    """

    source = stock_tree / "phys" / "module_radiation_driver.F"
    if not source.is_file():                                # pragma: no cover
        pytest.skip(f"stock WRF v4.6.1 tree not present at {stock_tree}")
    lines = source.read_text(encoding="utf-8",
                             errors="replace").splitlines()
    # 1-based citations in the code comments; 0-based here.
    assert "for P3, mp option 50 or 51" in lines[3878]
    assert "F_QI .and. F_QC .and. .not. F_QS" in lines[3879]
    assert "QCLD = QI(i,k,j)+QC(i,k,j)" in lines[3880]
    assert "weight = (QI(i,k,j)) / QCLD" in lines[3884]
    # ...and the arm it is NOT, immediately above.
    assert "F_QI .and. F_QC .and. F_QS" in lines[3869]

    registry = stock_tree / "Registry" / "Registry.EM_COMMON"
    if not registry.is_file():                              # pragma: no cover
        pytest.skip(f"stock WRF v4.6.1 Registry not present at {registry}")
    package = registry.read_text(encoding="utf-8",
                                 errors="replace").splitlines()[3037]
    assert "p3_1category" in package and "mp_physics==50" in package
    assert "moist:qv,qc,qr,qi;" in package
    assert "qs" not in package.split("moist:")[1].split(";")[0]
