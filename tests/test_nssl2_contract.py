"""CPU-only gates for the staged WRF NSSL ``mp_physics=18`` port."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from woof.config import RunConfig, validate_run_config
from woof.core.nssl2_contract import (
    CONTRACT_ID,
    DEFAULT_MODE,
    DEFAULT_RESTART_FIELDS,
    RADIATION_EFFECTIVE_RADIUS_FIELDS,
    WRF_NAMELIST_DEFAULTS,
    WRF_REFERENCE_COMMIT,
    resolve_nssl2_mode,
)
from woof.core.preflight import nest_field_kinds, state_array_shapes
from woof.io.restart import STATE_REBUILT_ATTRS, STATE_SERIALIZED_ATTRS


_ORACLE = (Path(__file__).parents[1] / "woof" / "data" / "nssl2" /
           "oracle" / "effective-radius.csv")
_INITIAL_ORACLE = _ORACLE.with_name("initial-state.csv")
_RAIN_SEDIMENT_ORACLE = _ORACLE.with_name("rain-sedimentation.csv")
_RAIN_ACCRETION_ORACLE = _ORACLE.with_name("rain-cloud-accretion.csv")
_RAIN_EVAPORATION_ORACLE = _ORACLE.with_name("rain-evaporation.csv")
_CLEAR_AIR_ACTIVATION_ORACLE = _ORACLE.with_name(
    "clear-air-activation.csv")


def test_native_mp18_defaults_resolve_to_full_two_moment_hail_mode():
    assert CONTRACT_ID == (
        "wrf-v4.6.1-nssl-mp18-two-moment-hail-ccn-density-v1")
    assert WRF_REFERENCE_COMMIT == (
        "d66e442fccc04111067e29274c9f9eaccc3cef28")
    assert DEFAULT_MODE.two_moment
    assert DEFAULT_MODE.hail
    assert DEFAULT_MODE.predicted_ccn
    assert DEFAULT_MODE.density_moments == 2
    assert DEFAULT_MODE.sixth_moments == 0


def test_default_restart_and_radiation_state_is_exactly_pinned():
    assert DEFAULT_RESTART_FIELDS == (
        "qv", "qc", "qr", "qi", "qs", "qg", "qh",
        "qndrop", "qnr", "qni", "qns", "qng", "qnh",
        "qnn", "qvolg", "qvolh",
    )
    assert DEFAULT_MODE.radiation_fields == RADIATION_EFFECTIVE_RADIUS_FIELDS
    assert RADIATION_EFFECTIVE_RADIUS_FIELDS == (
        "re_cloud", "re_ice", "re_snow")


def test_hail_off_and_three_moment_packages_follow_wrf_resolution():
    no_hail = resolve_nssl2_mode(
        nssl_hail_on=0, nssl_ccn_on=0, nssl_density_on=-1)
    assert no_hail.density_moments == 1
    assert "qh" not in no_hail.transported_fields
    assert "qvolg" in no_hail.transported_fields
    assert "qvolh" not in no_hail.transported_fields

    three = resolve_nssl2_mode(nssl_3moment=1)
    assert three.sixth_moments == 2
    assert three.transported_fields[-3:] == ("qzr", "qzg", "qzh")


def test_inconsistent_nssl_selectors_fail_closed():
    with pytest.raises(ValueError, match="requires mp_physics=18"):
        resolve_nssl2_mode(mp_physics=17)
    with pytest.raises(ValueError, match="hail volume"):
        resolve_nssl2_mode(nssl_hail_on=0, nssl_density_on=2)
    with pytest.raises(ValueError, match="unallocated hail-number"):
        resolve_nssl2_mode(nssl_2moment_on=0, nssl_3moment=1)
    with pytest.raises(TypeError, match="integer selector"):
        resolve_nssl2_mode(nssl_ccn_on=True)


def test_v461_registry_defaults_override_stale_readme_hail_density():
    assert WRF_NAMELIST_DEFAULTS["nssl_rho_qhl"] == 900.0
    assert WRF_NAMELIST_DEFAULTS["nssl_icdx"] == 6
    assert WRF_NAMELIST_DEFAULTS["nssl_icdxhl"] == 6
    assert WRF_NAMELIST_DEFAULTS["nssl_cccn"] == 0.5e9


def test_mp18_run_config_and_state_inventory_are_admitted():
    cfg = RunConfig(
        nx=8, ny=6, nz=4, dx=1000.0, dy=1000.0, ztop=10000.0,
        dt=1.0, run_seconds=10.0, moist=True, mp_physics=18)
    validate_run_config(cfg)
    current = set(DEFAULT_RESTART_FIELDS) - {"qv", "qc", "qr"}
    copies = {name + "0" for name in current}
    shapes = state_array_shapes(cfg)
    assert current | copies <= shapes.keys()
    assert current <= set(STATE_SERIALIZED_ATTRS)
    assert copies <= STATE_REBUILT_ATTRS
    assert nest_field_kinds(cfg)[-10:] == (
        "qh", "qndrop", "qnr", "qni", "qns", "qng", "qnh", "qnn",
        "qvolg", "qvolh")


def test_mp18_without_moist_state_fails_at_config_validation():
    cfg = RunConfig(
        nx=8, ny=6, nz=4, dx=1000.0, dy=1000.0, ztop=10000.0,
        dt=1.0, run_seconds=10.0, moist=False, mp_physics=18)
    with pytest.raises(ValueError, match="mp_physics=18 requires moist=true"):
        validate_run_config(cfg)


def test_the_low_temperature_cnuc_hack_reads_the_raw_staggered_w():
    """``module_mp_nssl_2mom.F:10122`` reads ``w``, not ``wvel``.

    NUCOND uses vertical velocity in two different shapes and the
    difference is essential::

        :10122   IF ( qx(mgs,lc) > 10.*qxmin(lc) .and.
                      w(igs(mgs),jgs,kgs(mgs)) > 2.0 ) THEN
        :10381   wvel(mgs) = (0.5)*(w(igs(mgs),jgs,kp1)
                                   +w(igs(mgs),jgs,kgs(mgs)))

    The first is the RAW staggered element at the point's own ``k`` -- the
    cell's bottom face.  The second is the mass-level average, and it is
    the ONLY assignment to ``wvel`` in the whole subroutine (``NUCOND``
    opens at ``:9611``), so at ``:10122`` no averaged ``w`` exists yet and
    the raw read cannot be an oversight to smooth over.

    Named breakage: a cell at ``T < 265 K`` with bottom face ``w = 1.8``
    under top face ``w = 2.6`` averages to 2.2.  WRF keeps
    ``cnuc = 0.1*ccnc``; an averaging port sets ``cnuc = 0``.  That is a
    10x-to-zero difference in the nucleation pool, not a rounding residue.
    Reached only with ``nssl_ccn_on = 0`` (which drives ``renucfrac`` to
    1.0 and arms ``:10120``), so it is latent in the shipped default rather
    than absent.

    The gate is two-sided on purpose: it requires the raw element in the
    cnuc block AND requires the mass-level average to survive at the
    ``wvel`` site, so it pins WRF's own distinction rather than banning
    one spelling outright.
    """
    kernel = (Path(__file__).parents[1] / "woof" / "core" / "kernels"
              / "nssl2_nucond.cu").read_text(encoding="utf-8")
    average = ("0.5f * (w_interface[(kp1 - 1) * horizontal_size + col]\n"
               "                                       + w_interface[idx])")

    start = kernel.index("// cnuc (:10113-10129)")
    block = kernel[start:kernel.index("// w and ssfilt neighbours", start)]
    assert "w_interface[idx] > 2.0f" in block, (
        ":10122's raw w(igs,jgs,kgs) read is gone from the cnuc block")
    assert "horizontal_size" not in block, (
        "the cnuc block reads a second interface, so it is averaging to "
        "the mass level where WRF reads the cell's own bottom face")

    # The averaged form is correct at :10381 and must not be swept away.
    assert kernel.count(average) == 1, (
        "wvel's mass-level average (:10381) is gone; this gate would then "
        "pass for the wrong reason")


def test_effective_radius_oracle_fixture_is_content_addressed():
    assert hashlib.sha256(_ORACLE.read_bytes()).hexdigest() == (
        "06e4f75711c751f1066292990063a28e3d25e5219640121513ac6b2c5c8dc3aa")
    assert hashlib.sha256(_INITIAL_ORACLE.read_bytes()).hexdigest() == (
        "51c216b974634740d8f6e85c8ba828704be882010a93e31f87aea9aec9f2c35c")
    assert hashlib.sha256(_RAIN_SEDIMENT_ORACLE.read_bytes()).hexdigest() == (
        "31ec833b02c519ac71a17d7d9fc772339a1f53897551199d624e775c0ce53c83")
    assert hashlib.sha256(_RAIN_ACCRETION_ORACLE.read_bytes()).hexdigest() == (
        "91e5bc4aa5d20d70437de08d821fc21a69ddd695fce5d63186d43a8b0f8fccf2")
    assert hashlib.sha256(_RAIN_EVAPORATION_ORACLE.read_bytes()).hexdigest() == (
        "5517bb039765a93768468d1c4223f1ff9216e952a7412abb89643fa8e7602e43")
    assert hashlib.sha256(
        _CLEAR_AIR_ACTIVATION_ORACLE.read_bytes()).hexdigest() == (
        "8f7a46688095450330ef1a789bad726289a061c868729a40ead25066aa90356f")
