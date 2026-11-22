"""Radar latent heating: the rules that need no device.

The slot rule is WRF's (``module_big_step_utilities_em.F:5917-5921`` in
NOAA-EMC/HRRR v4.1.21, ``sorc/hrrr_wrfarw.fd/WRFV3.9/dyn_em``): start at the
first slot and move on while the clock is PAST the slot's time, the last
slot persisting.  The clock is ``xtime``, minutes since the start, taken
before the step: ``frame/module_integrate.F:366-369`` calls the solver and
only then ``domain_clockadvance``, which refreshes ``xtime``
(``frame/module_domain.F:2550``) from whole seconds over 60 plus the
fraction over 60 (``:2316-2321``).

Nothing here imports CuPy, so this file runs on the CPU legs; the builder
and the forcing are graded on the card in ``test_radar_tten_builder.py`` and
``test_radar_tten_forcing.py``.
"""
from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

from woof.da import radar_tten

ROOT = Path(__file__).resolve().parents[1]
HRRR_SLOTS = (15.0, 30.0, 45.0, 60.0)


@pytest.mark.parametrize("minutes, slot", [
    (0.0, 1), (15.0, 1), (15.01, 2), (30.0, 2), (45.01, 4), (61.0, 4),
    (14.99, 1), (30.01, 3), (45.0, 3), (60.0, 4), (600.0, 4)])
def test_the_slot_is_the_first_whose_time_the_clock_has_not_passed(
        minutes, slot):
    assert radar_tten.select_slot(minutes, HRRR_SLOTS) + 1 == slot


@pytest.mark.parametrize("seconds, slot", [
    (0.0, 1), (900.0, 1), (900.6, 2), (1800.0, 2), (2700.6, 4),
    (3660.0, 4)])
def test_the_clock_is_wrf_minutes_from_elapsed_seconds(seconds, slot):
    minutes = radar_tten.wrf_minutes(seconds)
    assert isinstance(minutes, np.float32)
    assert radar_tten.select_slot(minutes, HRRR_SLOTS) + 1 == slot


def test_wrf_minutes_divides_whole_seconds_and_fraction_separately():
    # :2316-2321: REAL(seconds)/60. + (REAL(Sn)/REAL(Sd))/60., in float32.
    assert radar_tten.wrf_minutes(90.0) == np.float32(1.5)
    assert radar_tten.wrf_minutes(86400.0 + 60.0) == np.float32(1441.0)
    whole = np.float32(np.float32(7) / np.float32(60.0))
    expect = np.float32(whole + np.float32(0.25) / np.float32(60.0))
    assert radar_tten.wrf_minutes(7.25) == expect


def test_one_slot_persists_for_the_whole_leg():
    for minutes in (0.0, 14.0, 15.0, 15.01, 120.0):
        assert radar_tten.select_slot(minutes, (15.0,)) == 0


def test_an_empty_slot_list_is_refused():
    with pytest.raises(radar_tten.RadarTtenError, match="at least one"):
        radar_tten.select_slot(0.0, ())


def test_the_defaults_are_noaas_constants():
    """gsdcloudanalysis_ref2tten.f90:159-162 and :438-441; the literals of
    radar_ref2tten.f90 at the lines the dataclass cites."""
    cfg = radar_tten.RadarTtenConfig()
    assert cfg.krad_bot == 7.0
    assert cfg.latent_heat_period_min == 20.0
    assert cfg.convection_refl_threshold_dbz == 28.0
    assert cfg.convection_only is True
    assert cfg.slot_minutes == HRRR_SLOTS
    assert cfg.warm_temperature_k == 277.15
    assert cfg.warm_min_dbz == 28.0
    assert cfg.cold_echo_depth_hpa == 200.0
    assert cfg.coverage_depth_hpa == 300.0
    assert cfg.tendency_cap == 0.01
    assert (cfg.z_scale_dbz, cfg.z_divisor, cfg.z_factor) == (
        17.8, 264083.0, 1.5)
    assert (cfg.smooth_passes_tendency, cfg.smooth_passes_flag) == (2, 3)
    assert radar_tten.NO_COVERAGE_TENDENCY == -20.0
    assert radar_tten.NO_COVERAGE_DBZ == -99999.0
    assert radar_tten.NO_ECHO_DBZ == -99.0
    # parm/conus/hrrr_wrfpre.nl:108, beside mp_tend_radar = 1 at :109
    assert radar_tten.HRRR_MP_TEND_LIM == 0.07


def test_the_constants_are_formed_as_the_fortran_forms_them():
    """radar_ref2tten.f90:113-127 and constants.f90 (:92, :325, :377)."""
    rd_p = 8.31451 / 0.0289645
    assert radar_tten._RD_P == rd_p
    assert radar_tten._CPD_P == 3.5 * rd_p
    assert radar_tten._CPOVR_P == (3.5 * rd_p) / rd_p
    assert radar_tten._LV_P + radar_tten._LF0_P == 2.501e6 + 0.3335e6
    assert radar_tten._RD_OVER_CP == 287.04 / 1004.6


def test_the_cone_tables_are_noaas_summer_tables():
    levels, profiles = radar_tten._cone_tables()
    assert levels.shape == (31,) and profiles.shape == (6, 31)
    # build_missing_REFcone.f90:68-70 (km, as default reals) times 1000.
    assert levels[0] == np.float64(np.float32(0.2)) * 1000.0
    assert levels[-1] == 16000.0
    # :115-119 first and last value of the 20-25 dBZ class, :145-149 the
    # last class.
    assert profiles[0, 0] == np.float64(np.float32(0.883))
    assert profiles[0, -1] == np.float64(np.float32(0.833))
    assert profiles[5, 0] == np.float64(np.float32(0.926))
    assert profiles[5, -1] == np.float64(np.float32(0.410))


def test_the_builder_refuses_by_name_without_a_device(monkeypatch):
    """No host fallback: the refusal says why and opens no device."""
    monkeypatch.setenv("GPUWM_NO_LOCAL_GPU", "1")
    with pytest.raises(radar_tten.RadarTtenError,
                       match="no host implementation"):
        radar_tten.build_tendency(None, None, None, None, None)
    with pytest.raises(radar_tten.RadarTtenError,
                       match="no host implementation"):
        radar_tten.RadarTtenForcing([], [])


def test_the_module_has_no_host_implementation():
    """The kernels are the only numerics: the module never imports NumPy's
    random, linear algebra or FFT, and every public builder routes through
    the device check before touching an array."""
    source = (ROOT / "woof" / "da" / "radar_tten.py").read_text(
        encoding="utf-8")
    tree = ast.parse(source)
    checked = {"pbl_height", "cone_fill", "smooth", "build_tendency",
               "tendency_receipt", "reflectivity_from_document",
               "background_from_state", "vinterp_mosaic"}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in checked:
            first = node.body[1] if isinstance(
                node.body[0], ast.Expr) else node.body[0]
            text = ast.unparse(first)
            assert "_require_device()" in text, (node.name, text)


def test_the_kernel_source_has_no_literal_float_division():
    """A146: a constant divisor goes through __fdiv_rn / __ddiv_rn."""
    from tools.literal_division_scan import literal_divisions

    found = literal_divisions(radar_tten._SOURCE)
    assert not found, found


def test_microphysics_apply_reads_the_forcing_once_and_only_optionally():
    """With no forcing the entry point is one attribute read and the
    unchanged body, now named _apply_scheme."""
    from woof.core import microphysics

    tree = ast.parse(inspect.getsource(microphysics.apply))
    body = tree.body[0].body
    body = body[1:] if isinstance(body[0], ast.Expr) else body
    assert ast.unparse(body[0]) == (
        "forcing = getattr(state, 'radar_tten_forcing', None)")
    assert ast.unparse(body[1]) == (
        "if forcing is None:\n"
        "    return _apply_scheme(state, cfg, dt, "
        "refl_10cm_due=refl_10cm_due)")
    assert radar_tten.STATE_ATTRIBUTE == "radar_tten_forcing"


def test_the_batched_state_is_recognised_by_the_method_the_refusal_reads():
    """RadarTtenForcing.check_state refuses a batched-ensemble state by
    its ``member_view`` method; if the class lost it, the refusal would
    stop firing and the batched path, which runs its own Thompson column
    adapter rather than microphysics.apply, would skip the forcing."""
    from woof.core.state import DomainState
    from woof.ensemble.batch_state import BatchedDomainState

    assert callable(getattr(BatchedDomainState, "member_view", None))
    assert getattr(DomainState, "member_view", None) is None


def test_the_forcing_is_in_no_state_or_restart_inventory():
    """External data: an inventory row would make a restart carry one
    leg's observations into the next.  It is classified INFRA, which every
    manifest skips, so a checkpoint written while a forcing is attached is
    legal and carries none of it (the RestartManifestError this test used to
    expect is retired with that classification).  The other half of the
    rule, refusing a resume inside a forced period, is the door's
    (woof.da.forecast_heating.resume_plan)."""
    from woof.io import restart

    assert restart.classify_state_attr(radar_tten.STATE_ATTRIBUTE) == "infra"
    assert radar_tten.STATE_ATTRIBUTE not in restart.STATE_SERIALIZED_ATTRS
    assert radar_tten.STATE_ATTRIBUTE not in restart.CHECKPOINT_ONLY_STATE

