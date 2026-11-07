"""WRF's KF-to-Morrison mixed-phase interface.

WRF v4.6.1 keeps KF cloud water, cloud ice, rain, and snow tendencies
independent (``module_cu_kfeta.F:2311-2382,2599-2634``).  Morrison consumes
the raw rain/snow/ice rates to seed the matching number moments before its
entry cleanup and PSD reconstruction
(``module_mp_morr_two_moment.F:1327-1343``).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from _run_config_fake import run_config_fake


class _NumpyCupyShim:
    """CPU-only subset used to exercise the driver seam without a GPU."""

    ndarray = np.ndarray
    asarray = staticmethod(np.asarray)
    ascontiguousarray = staticmethod(np.ascontiguousarray)
    isfinite = staticmethod(np.isfinite)
    zeros = staticmethod(np.zeros)
    maximum = staticmethod(np.maximum)
    floor = staticmethod(np.floor)
    where = staticmethod(np.where)


_KF_ORACLE = Path(__file__).parent / "data" / "oracles" / "kf"


def test_wrf_returns_all_four_mixed_phase_tendencies_independently():
    """The interface Morrison relies on, read off WRF itself.

    With F_QS and F_QI, KF_eta_CPS returns cloud water, cloud ice, rain and
    snow as four separate rates (module_cu_kfeta.F:2622-2634).  Deep
    convection rains out through RAINCV (FBFRC = 0), so its QR/QS rates are
    zero and its QC/QI rates carry the detrained condensate; shallow
    convection (FBFRC = 1) feeds its fallout back as QR/QS too, so on the
    ERA5 stable column all four are nonzero at once -- the category
    structure Morrison's number seed consumes.  kernels/kf.cu reproduces
    these words exactly (tests/test_kf_wrf461_oracle.py), which is what
    replaced the float64 mirror this test used to read.
    """
    import json

    from tools.kf_wrf461_oracle import kf_columns

    outs = kf_columns.read_outputs(_KF_ORACLE / "oracle-real74-O0.out")
    meta = json.loads((_KF_ORACLE / "columns-real74.json").read_text(
        encoding="utf-8"))
    deep = [o for m, o in zip(meta, outs)
            if m["regime"] == "real74_unstable" and m["branch"] == "qs_qi"]
    assert deep and all(int(o["shall"]) == 0 for o in deep)
    for o in deep:
        assert np.any(o["rqccuten"] != 0.0) and np.any(o["rqicuten"] != 0.0)
        np.testing.assert_array_equal(o["rqrcuten"], 0.0)
        np.testing.assert_array_equal(o["rqscuten"], 0.0)
        assert o["pratec"] > 0.0
    shallow = [o for m, o in zip(meta, outs)
               if m["regime"] == "real74_stable" and m["branch"] == "qs_qi"
               and int(o["shall"]) == 1]
    four = [o for o in shallow
            if all(np.any(o[name] != 0.0) for name in
                   ("rqccuten", "rqicuten", "rqrcuten", "rqscuten"))]
    assert len(four) == 6
    for o in shallow:
        assert o["pratec"] == 0.0           # FBFRC = 1: nothing to RAINCV
        for name in ("rqccuten", "rqicuten", "rqrcuten", "rqscuten"):
            assert np.isfinite(o[name]).all()
    # Without F_QS the frozen categories fold into QC/QR and WRF writes no
    # QI/QS tendency at all (the -999 sentinel survives the call).
    folded = [o for m, o in zip(meta, outs)
              if m["regime"] == "real74_unstable" and m["branch"] == "no_qs"]
    for o in folded:
        np.testing.assert_array_equal(o["rqicuten"], np.float32(-999.0))
        np.testing.assert_array_equal(o["rqscuten"], np.float32(-999.0))
        assert np.any(o["rqccuten"] != 0.0)    # QL+QI, folded


def test_kf_phase_mode_mapping_keeps_kessler_and_no_mp_distinct():
    from woof.core.kf import KFPhaseMode, kf_phase_mode_for_microphysics

    assert kf_phase_mode_for_microphysics(0) == \
        KFPhaseMode.NO_SEPARATE_SNOW
    assert kf_phase_mode_for_microphysics(1) == KFPhaseMode.WARM_RAIN
    assert kf_phase_mode_for_microphysics(10) == \
        KFPhaseMode.SEPARATE_ICE_SNOW
    with pytest.raises(ValueError, match="no verified phase-output contract"):
        kf_phase_mode_for_microphysics(99)


def test_morrison_cumulus_number_seed_matches_wrf_formulas_and_threshold():
    from woof.verify.npref import _np_morrison_seed_cumulus_numbers

    rho = np.array([1.0, 0.8], dtype=np.float64)
    qrcu = np.array([1.0e-4, 0.999e-10], dtype=np.float64)
    qscu = np.array([2.0e-4, -1.0e-5], dtype=np.float64)
    qicu = np.array([3.0e-8, 0.0], dtype=np.float64)
    nr0 = np.array([11.0, 12.0], dtype=np.float64)
    ns0 = np.array([21.0, 22.0], dtype=np.float64)
    ni0 = np.array([31.0, 32.0], dtype=np.float64)
    dt = 60.0

    nr, ns, ni = _np_morrison_seed_cumulus_numbers(
        nr0, ns0, ni0, qrcu, qscu, qicu, rho, dt)

    expected_nr = nr0.copy()
    expected_ns = ns0.copy()
    expected_ni = ni0.copy()
    expected_nr[0] += 1.8e5 * (
        qrcu[0] * dt / (np.pi * 997.0 * rho[0] ** 3)) ** 0.25
    expected_ns[0] += 3.0e5 * (
        qscu[0] * dt / (100.0 * np.pi * rho[0] ** 3)) ** 0.25
    expected_ni[0] += qicu[0] * dt / (
        (500.0 * np.pi / 6.0) * (80.0e-6) ** 3)

    np.testing.assert_allclose(nr, expected_nr, rtol=2.0e-15, atol=0.0)
    np.testing.assert_allclose(ns, expected_ns, rtol=2.0e-15, atol=0.0)
    np.testing.assert_allclose(ni, expected_ni, rtol=2.0e-15, atol=0.0)
    # The comparison is >= 1e-10 in WRF.  Sub-threshold and negative rates
    # leave their corresponding moments exactly unchanged.
    assert nr[1] == nr0[1]
    assert ns[1] == ns0[1]
    assert ni[1] == ni0[1]


def test_custom_warm_rain_attachment_requires_energy_defined_phase_closure(
        monkeypatch):
    """The driver cannot infer latent heating from already diagnosed rates."""
    from woof.core import physics
    from woof.core.physics import CumulusResult, PhysicsDriver

    monkeypatch.setattr(physics, "cp", _NumpyCupyShim)
    shape = (2, 2, 3)
    surface = shape[1:]
    state = SimpleNamespace(
        p=np.zeros(shape, dtype=np.float32),
        c1h=np.ones(shape[0], dtype=np.float32),
        c2h=np.zeros(shape[0], dtype=np.float32),
        has_msf=False,
        total_mu=lambda: np.full(surface, 2.0, dtype=np.float32),
    )
    # Deliberately do not define state.qi: that is the real mp=0/1 contract.
    rates = {
        "rthcuten": np.full(shape, 1.0, dtype=np.float32),
        "rqvcuten": np.full(shape, 2.0, dtype=np.float32),
        "rqccuten": np.full(shape, 3.0, dtype=np.float32),
        "rqicuten": np.full(shape, 5.0, dtype=np.float32),
        "rqrcuten": np.full(shape, 7.0, dtype=np.float32),
        "rqscuten": np.full(shape, 11.0, dtype=np.float32),
    }
    driver = object.__new__(PhysicsDriver)
    # This focused clock fixture bypasses __init__; optional LSM parameter
    # bundles were added later and are absent for this Morrison-only path.
    driver.ruc_params = None
    driver.noahmp_params = None
    driver.fields = {}
    driver.cu_rates = {name: np.zeros(shape, dtype=np.float32)
                       for name in rates}
    driver.rainc = np.zeros(surface, dtype=np.float32)
    driver._pending_rainbl = np.zeros(surface, dtype=np.float32)
    # SEAM A widened the coupler ring-mask predicate to
    # `specified or nested`; this stub models the same non-LBC,
    # non-nested domain it always did, now with the attribute the
    # production RunConfig has always carried.
    cfg = SimpleNamespace(
        specified=False, nested=False, cu_physics=1, mp_physics=1)

    from woof.core.physics import CUMULUS_HISTORY_RATES
    for history_name in CUMULUS_HISTORY_RATES:
        driver.fields.setdefault(history_name, np.zeros(shape, dtype=np.float32))
    driver.cumulus_callable = lambda **_kwargs: CumulusResult(**rates)
    with pytest.raises(ValueError, match="latent-energy-consistent"):
        driver._run_cumulus({}, state, cfg)

    # A custom scheme may instead return its own energy-consistent, already
    # folded warm-rain contract and omit the unsupported frozen categories.
    folded = {**rates, "rqccuten": rates["rqccuten"] + rates["rqicuten"],
              "rqrcuten": rates["rqrcuten"] + rates["rqscuten"]}
    folded.pop("rqicuten")
    folded.pop("rqscuten")
    driver.cumulus_callable = lambda **_kwargs: CumulusResult(**folded)
    from woof.core.physics import CUMULUS_HISTORY_RATES
    for history_name in CUMULUS_HISTORY_RATES:
        driver.fields.setdefault(history_name, np.zeros(shape, dtype=np.float32))
    driver._run_cumulus({}, state, cfg)

    np.testing.assert_array_equal(driver.cu_rates["rqccuten"], 8.0)
    np.testing.assert_array_equal(driver.cu_rates["rqrcuten"], 18.0)
    np.testing.assert_array_equal(driver.cu_rates["rqicuten"], 0.0)
    np.testing.assert_array_equal(driver.cu_rates["rqscuten"], 0.0)
    # The held scalar stack receives dry-mass coupling after phase folding.
    np.testing.assert_array_equal(driver.cumulus_tendencies.rqc, 16.0)
    np.testing.assert_array_equal(driver.cumulus_tendencies.rqr, 36.0)
    assert driver.cumulus_tendencies.rqi is None
    assert driver.cumulus_tendencies.rqs is None


def test_nca_expiry_preserves_current_rk_copy_but_zeros_morrison_rates(
        monkeypatch):
    """advance_ppt precedes Morrison while this step's RK copy survives."""
    from woof.core import physics
    from woof.core.physics import PhysicsDriver

    monkeypatch.setattr(physics, "cp", _NumpyCupyShim)
    shape = (2, 1, 1)
    surface = (1, 1)
    driver = object.__new__(PhysicsDriver)
    driver.ruc_params = None
    driver.noahmp_params = None
    # No land-surface rain carriers and no CUPPT diagnostic allocated.
    driver.fields = {}
    driver.cu_pratec = np.zeros(surface, dtype=np.float32)
    driver.rainc = np.zeros(surface, dtype=np.float32)
    driver._pending_rainbl = np.zeros(surface, dtype=np.float32)
    driver.cu_nca = np.full(surface, 60.0, dtype=np.float32)
    driver.cu_expiring = np.zeros(surface, dtype=np.float32)
    names = ("rthcuten", "rqvcuten", "rqccuten", "rqicuten",
             "rqrcuten", "rqscuten")
    driver.cu_rates = {
        name: np.full(shape, index + 1.0, dtype=np.float32)
        for index, name in enumerate(names)
    }
    scalar_names = ("rtheta", "rqv", "rqc", "rqi", "rqr", "rqs")
    driver.cumulus_tendencies = SimpleNamespace(**{
        name: np.full(shape, index + 11.0, dtype=np.float32)
        for index, name in enumerate(scalar_names)
    })
    # _compose_tendencies has already copied these values into the target
    # consumed by this RK step; it is deliberately distinct from the held
    # component that advance_ppt clears for the next step.
    current_rk = SimpleNamespace(**{
        name: getattr(driver.cumulus_tendencies, name).copy()
        for name in scalar_names
    })
    driver.tendencies = current_rk

    driver._advance_cumulus_clock(
        SimpleNamespace(elapsed_seconds=0.0),
        run_config_fake(dt=60.0, clock_dt=0.0, cu_physics=1))

    for name in scalar_names:
        np.testing.assert_array_equal(
            getattr(driver.tendencies, name),
            np.full(shape, scalar_names.index(name) + 11.0,
                    dtype=np.float32))
        np.testing.assert_array_equal(
            getattr(driver.cumulus_tendencies, name), 0.0)
    # These are exactly the three arrays apply_morrison forwards to the
    # number-moment seed interface later in the step.
    for name in ("rqrcuten", "rqicuten", "rqscuten"):
        np.testing.assert_array_equal(driver.cu_rates[name], 0.0)
    np.testing.assert_array_equal(driver.cu_nca, 0.0)
    np.testing.assert_array_equal(driver.cu_expiring, 0.0)
    assert driver._cu_expiry_pending is False


def test_cuda_sources_keep_four_rates_and_wrf_number_seed_statement_order():
    root = Path(__file__).parents[1]
    kf = (root / "woof" / "core" / "kernels" / "kf.cu").read_text(
        encoding="utf-8")
    morrison = (
        root / "woof" / "core" / "kernels" / "morrison.cu").read_text(
            encoding="utf-8")

    assert "int phase_mode" in kf
    assert "KF_PHASE_WARM_RAIN" in kf
    assert "KF_PHASE_NO_SEPARATE_SNOW" in kf
    assert "KF_PHASE_SEPARATE_SNOW" in kf
    # KF_eta_PARA's feedback (module_cu_kfeta.F:2599-2640): the F_QS arm
    # keeps four rates, and the folding arms apply latent fusion to TG
    # BEFORE DTDT is formed and divided by PI.
    assert "dqidt = (qig - 0.0f) / timec;" in kf
    assert "dqsdt = (qsg - 0.0f) / timec;" in kf
    assert "dqrdt = (qrg - 0.0f) / timec;" in kf
    assert "dqcdt = (qlg - 0.0f) / timec;" in kf
    latent = kf.index("tgk = tgk - ((qig + qsg) * rlf) / cpm;")
    theta_rate = kf.index("rthcuten[i3] = dtdt / pi3(kk);")
    assert latent < kf.index("dtdt = (tgk - t0(kk)) / timec;") < theta_rate

    # WRF :1332-1343, in order, before the small-amount evaporation
    # (:1348) and every process rate; the transcription spells WRF's own
    # expressions (RHO**3 through libgcc's __powisf2 chain, (80E-6)**DI
    # through powf).
    rain = morrison.index("if (qrcu1d >= 1.E-10f)")
    snow = morrison.index("if (qscu1d >= 1.E-10f)")
    ice = morrison.index("if (qicu1d >= 1.E-10f)")
    evaporate = morrison.index("if (qvqvs < 0.9f)")
    assert rain < snow < ice < evaporate
    assert "(MO_RHOW * MO_PI) * mo_powi3(rho)" in morrison
    assert "mo_powi3(rho) * MO_CONS1" in morrison
    assert "gfk_pow(80.E-6f, MO_DI) * MO_CI" in morrison
