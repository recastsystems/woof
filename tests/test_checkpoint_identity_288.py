"""2.8.8 moved the default arithmetic of these schemes, so their 2.8.7
checkpoints are refused before restore; unchanged ones still resume.

The breakage this guards (woof/checkpoint_identity.py's contract): a
restart records the identity of every scheme that integrated it so that a
resume onto a different implementation fails BEFORE it restores, rather
than continuing a trajectory that is not the one it claims.  2.8.8 changed
the default-build arithmetic or tendencies of YSU, Shin-Hong, aerosol-aware
Thompson, the four surface layers, RUC and the dycore's mixing operators
under unchanged selectors, and moved the shared microphysics finish clamp to
WRF's REAL product mp_tend_lim*dt under every scheme that finishes through
it, so the configuration fingerprint alone would let
a 2.8.7 checkpoint of any of them resume and splice two implementations
into one forecast.  Each identity below is the exact string 2.8.7 wrote.
"""
from __future__ import annotations

import numpy as np
import pytest

from woof import checkpoint_identity
from woof.io import restart
from woof.physics_registry import physics_registry
from test_restart import _NumpyCupyShim, _cfg, _fill_setup, _shim_state

#: The refusal restart._require_current_algorithm_identities makes BEFORE
#: the configuration walk, naming each scheme with its selector and both
#: identities (it used to be the physics gate's "these components differ:
#: algorithms, configuration_sha256", which named no scheme).
REFUSAL = "was integrated by scheme implementations this build does not run: "

#: The physics gate's own sentence, asked directly by the RUC test.
GATE_REFUSAL = ("was written under a different physics setup; these "
                "components differ: .*algorithms")

#: The selector each table is chosen by, as the refusal names it.
_SELECTOR = {
    "MICROPHYSICS_ALGORITHM_IDENTITIES": "mp_physics",
    "SURFACE_LAYER_ALGORITHM_IDENTITIES": "sf_sfclay_physics",
    "LAND_SURFACE_ALGORITHM_IDENTITIES": "sf_surface_physics",
    "PBL_ALGORITHM_IDENTITIES": "bl_pbl_physics",
}

#: (table, selector value) -> the identity string 2.8.7 wrote.
RELEASED_287 = {
    ("MICROPHYSICS_ALGORITHM_IDENTITIES", 1): "kessler-warm-rain-v1",
    ("MICROPHYSICS_ALGORITHM_IDENTITIES", 6): (
        "wsm6-single-moment-six-class-wrf-v4.6.1-v1"),
    ("MICROPHYSICS_ALGORITHM_IDENTITIES", 9): (
        "milbrandt-yau-wrf-v4.6.1-v1-six-category-2mom-ccntype2-"
        "meyers-contact-nucl-nonspherical-snow-full-sedimentation"),
    ("MICROPHYSICS_ALGORITHM_IDENTITIES", 10): (
        "morrison-two-moment-v3-kf-number-seeding-finite-freezing-"
        "final-vapor-in-range-number"),
    ("MICROPHYSICS_ALGORITHM_IDENTITIES", 16): (
        "wdm6-double-moment-warm-rain-wrf-v4.6.1-v4-prognostic-nc-nr-ccn-"
        "gamma-mu1-rain-ccn-activation-xland-autoconversion-ccn-conc-init-"
        "conservative-rain-interface-flux-bounded-transport-time-"
        "zero-rate-rain-no-evaporation"),
    ("MICROPHYSICS_ALGORITHM_IDENTITIES", 18): (
        "nssl-two-moment-state-transport-v1-process-boundary-fail-loud"),
    ("MICROPHYSICS_ALGORITHM_IDENTITIES", 50): (
        "p3-one-category-wrf-v4.6.1-v1-2mom-ice-specified-nc-"
        "diagnosed-ssat-rime-mass-volume-transported"),
    ("MICROPHYSICS_ALGORITHM_IDENTITIES", 28): (
        "thompson-aerosol-aware-wrf-v4.6.1-v1-prognostic-nc-nwfa-nifa-"
        "ccn-activate-table-demott-koop-scavenging-surface-emission-"
        "synthetic-aerosol-init"),
    ("SURFACE_LAYER_ALGORITHM_IDENTITIES", 1): "revised-mm5-surface-layer-v1",
    ("SURFACE_LAYER_ALGORITHM_IDENTITIES", 2): (
        "eta-similarity-surface-layer-wrf-v4.6.1-v1-myjsfcinit-tables"),
    ("SURFACE_LAYER_ALGORITHM_IDENTITIES", 5): (
        "mynn-surface-layer-wrf-v4.6.1-v1"),
    ("SURFACE_LAYER_ALGORITHM_IDENTITIES", 91): "classic-mm5-surface-layer-v1",
    ("LAND_SURFACE_ALGORITHM_IDENTITIES", 3): (
        "ruc-lsm-wrf-v4.6.1-v6-default-selection"),
    ("PBL_ALGORITHM_IDENTITIES", 1): "ysu-v1",
    ("PBL_ALGORITHM_IDENTITIES", 5): (
        "mynn-edmf-pbl-wrf-v4.6.1-v2-rounded-mixing-length"),
    ("PBL_ALGORITHM_IDENTITIES", 11): "shinhong-pbl-wrf-v4.6.1-v1",
}

#: The registry component whose consumer row publishes each table.
_COMPONENT = {
    "MICROPHYSICS_ALGORITHM_IDENTITIES": "microphysics",
    "SURFACE_LAYER_ALGORITHM_IDENTITIES": "surface_layer",
    "LAND_SURFACE_ALGORITHM_IDENTITIES": "land_surface",
    "PBL_ALGORITHM_IDENTITIES": "pbl",
}

#: A configuration that integrates each bumped scheme with no driver.
#: NSSL (mp=18) is not one: its checkpoint writer requires an attached
#: MP18 PhysicsDriver (restart.RestartManifestError), so its row is held
#: by the registry-publication test above alone.
_DRIVERLESS = {
    ("MICROPHYSICS_ALGORITHM_IDENTITIES", 1): dict(moist=True, mp_physics=1),
    ("MICROPHYSICS_ALGORITHM_IDENTITIES", 6): dict(moist=True, mp_physics=6),
    ("MICROPHYSICS_ALGORITHM_IDENTITIES", 9): dict(moist=True, mp_physics=9),
    ("MICROPHYSICS_ALGORITHM_IDENTITIES", 10): dict(moist=True,
                                                    mp_physics=10),
    ("MICROPHYSICS_ALGORITHM_IDENTITIES", 16): dict(moist=True,
                                                    mp_physics=16),
    ("MICROPHYSICS_ALGORITHM_IDENTITIES", 50): dict(moist=True,
                                                    mp_physics=50),
    ("MICROPHYSICS_ALGORITHM_IDENTITIES", 28): dict(moist=True, mp_physics=28),
    ("SURFACE_LAYER_ALGORITHM_IDENTITIES", 1): dict(sf_sfclay_physics=1),
    ("SURFACE_LAYER_ALGORITHM_IDENTITIES", 2): dict(sf_sfclay_physics=2),
    ("SURFACE_LAYER_ALGORITHM_IDENTITIES", 5): dict(sf_sfclay_physics=5),
    ("SURFACE_LAYER_ALGORITHM_IDENTITIES", 91): dict(sf_sfclay_physics=91),
    # The bumped 2.8.8 YSU is the wrf_461 reproduction arm; the WRF 4.8
    # default (ysu_version = "wrf_480") is a separate identity.
    ("PBL_ALGORITHM_IDENTITIES", 1): dict(moist=True, bl_pbl_physics=1,
                                          ysu_version="wrf_461"),
    ("PBL_ALGORITHM_IDENTITIES", 5): dict(moist=True, bl_pbl_physics=5),
    # Likewise Shin-Hong: the bumped identity is the wrf_461 arm; WRF 4.8.0's
    # rewrite (shinhong_version = "wrf_480", the default) is a separate one.
    ("PBL_ALGORITHM_IDENTITIES", 11): dict(moist=True, bl_pbl_physics=11,
                                           shinhong_version="wrf_461"),
}


def _refused_before_restore(cfg, monkeypatch, tmp_path, write_as_287):
    """Write under ``write_as_287``, then require the resume to refuse with
    the existing sentence and leave the live state untouched."""
    source = _shim_state(cfg, monkeypatch)
    _fill_setup(source)
    with monkeypatch.context() as old:
        write_as_287(old)
        path = restart.write_restart(tmp_path / "released-287.npz",
                                     source, cfg)
    fresh = _shim_state(cfg, monkeypatch)
    _fill_setup(fresh)
    marker = fresh.thp if fresh.qv is None else fresh.qv
    marker.fill(np.float32(.0123))
    before = marker.tobytes()
    with pytest.raises(restart.RestartMismatchError, match=REFUSAL) as caught:
        restart.restore_restart(path, fresh, cfg)
    assert marker.tobytes() == before
    return str(caught.value)


@pytest.mark.parametrize("row", sorted(RELEASED_287), ids=lambda row: (
    f"{row[0].split('_')[0].lower()}-{row[1]}"))
def test_every_bumped_identity_moved_and_the_registry_publishes_it(row):
    table, key = row
    current = getattr(checkpoint_identity, table)[key]
    assert current != RELEASED_287[row]
    component = physics_registry()["components"][_COMPONENT[table]]
    published = [option["consumers"]["restart_algorithm_identity"]
                 for option in component["options"].values()
                 if option.get("implemented") is True
                 and current == option.get("consumers", {}).get(
                     "restart_algorithm_identity")]
    assert published, (table, key, current)


@pytest.mark.parametrize("row", sorted(_DRIVERLESS), ids=lambda row: (
    f"{row[0].split('_')[0].lower()}-{row[1]}"))
def test_a_287_checkpoint_of_a_bumped_scheme_is_refused_before_restore(
        row, tmp_path, monkeypatch):
    table, key = row
    message = _refused_before_restore(
        _cfg(**_DRIVERLESS[row]), monkeypatch, tmp_path,
        lambda old: old.setitem(getattr(restart, table), key,
                                RELEASED_287[row]))
    # The scheme, its selector and both identities are named, so the user
    # learns which scheme moved and that no configuration change helps.
    current = getattr(checkpoint_identity, table)[key]
    assert (f"{_SELECTOR[table]}={key} "
            f"({_COMPONENT[table].replace('_', ' ')}): "
            f"{RELEASED_287[row]} -> {current}") in message, message
    assert "no configuration change resumes it" in message


def test_a_287_ruc_checkpoint_is_refused_by_the_restart_gate(monkeypatch):
    """RUC needs a driver to be named; the gate itself is asked directly,
    as tests/test_trace_gas_overrides_gpu.py asks it."""
    import woof.core.physics as physics

    cfg = _cfg(sf_surface_physics=3, sf_sfclay_physics=1, num_soil_layers=9)
    state = _shim_state(cfg, monkeypatch)
    monkeypatch.setattr(physics, "cp", _NumpyCupyShim)
    # The driver needs a RUC bundle to exist; its bytes are not what moved,
    # so the bundle identity is pinned below and only the algorithm row
    # differs between the two sides.
    state.physics = physics.PhysicsDriver(
        state, cfg, fields={}, sfclay_result=None, noah_params=None,
        ruc_params=object())
    _fill_setup(state)
    monkeypatch.setattr(restart, "_land_surface_parameters_identity",
                        lambda cfg, driver: {"bundle": "fixture"})
    row = ("LAND_SURFACE_ALGORITHM_IDENTITIES", 3)
    with monkeypatch.context() as old:
        old.setitem(restart.LAND_SURFACE_ALGORITHM_IDENTITIES, 3,
                    RELEASED_287[row])
        stored = restart.physics_setup_identity(state, cfg)
    header = {"physics_setup": stored,
              "physics_setup_fingerprint": restart._json_sha256(stored)}
    with pytest.raises(restart.RestartMismatchError,
                       match=GATE_REFUSAL) as caught:
        restart._require_physics_setup_match(header, state, cfg, "released")
    # The gate's own sentence names the moved identity as well.
    assert (f"sf_surface_physics=3 (land surface): {RELEASED_287[row]} -> "
            f"{restart.LAND_SURFACE_ALGORITHM_IDENTITIES[3]}") in str(
                caught.value)
    current = restart.physics_setup_identity(state, cfg)
    header = {"physics_setup": current,
              "physics_setup_fingerprint": restart._json_sha256(current)}
    restart._require_physics_setup_match(header, state, cfg, "current")


@pytest.mark.parametrize("overrides", [
    dict(km_opt=2),
    dict(km_opt=3),
    dict(km_opt=4),
    dict(diff_6th_opt=2),
    dict(km_opt=1, khdif=100.0, bl_pbl_physics=1, moist=True),
], ids=["km2", "km3", "km4", "sixth-order", "constant-k"])
def test_a_287_checkpoint_of_a_mixing_run_is_refused_before_restore(
        overrides, tmp_path, monkeypatch):
    """2.8.7 recorded no mixing identity; a run that mixes now records it."""
    cfg = _cfg(**overrides)
    assert (checkpoint_identity.dycore_mixing_identity(cfg)
            == checkpoint_identity.DYCORE_MIXING_ALGORITHM_IDENTITY)
    message = _refused_before_restore(
        cfg, monkeypatch, tmp_path,
        lambda old: old.setattr(restart, "dycore_mixing_identity",
                                lambda cfg: None))
    assert (f"(dycore mixing): none recorded -> "
            f"{checkpoint_identity.DYCORE_MIXING_ALGORITHM_IDENTITY}"
            ) in message, message


def test_a_mixing_checkpoint_of_this_release_round_trips(tmp_path,
                                                         monkeypatch):
    cfg = _cfg(km_opt=4, diff_6th_opt=2)
    source = _shim_state(cfg, monkeypatch)
    _fill_setup(source)
    source.thp.fill(np.float32(.25))
    path = restart.write_restart(tmp_path / "current.npz", source, cfg)
    header = restart.read_restart_header(path)
    assert header["physics_setup"]["algorithms"]["dycore_mixing"] == \
        checkpoint_identity.DYCORE_MIXING_ALGORITHM_IDENTITY
    fresh = _shim_state(cfg, monkeypatch)
    _fill_setup(fresh)
    restart.restore_restart(path, fresh, cfg)
    np.testing.assert_array_equal(fresh.thp, source.thp)


def test_a_287_checkpoint_of_a_damped_run_is_refused_before_restore(
        tmp_path, monkeypatch):
    """2.8.7 formed damp_opt = 3's dampmag as a double product rounded once
    and recorded no damping identity; a damped run now records it."""
    cfg = _cfg(damp_opt=3, zdamp=5000.0, dampcoef=0.2)
    assert (checkpoint_identity.upper_damping_identity(cfg)
            == checkpoint_identity.UPPER_DAMPING_ALGORITHM_IDENTITY)
    message = _refused_before_restore(
        cfg, monkeypatch, tmp_path,
        lambda old: old.setattr(restart, "upper_damping_identity",
                                lambda cfg: None))
    assert (f"damp_opt=3 (upper damping): none recorded -> "
            f"{checkpoint_identity.UPPER_DAMPING_ALGORITHM_IDENTITY}"
            ) in message, message


def test_a_damped_checkpoint_of_this_release_round_trips(tmp_path,
                                                         monkeypatch):
    cfg = _cfg(damp_opt=3, zdamp=5000.0, dampcoef=0.2)
    source = _shim_state(cfg, monkeypatch)
    _fill_setup(source)
    source.thp.fill(np.float32(.25))
    path = restart.write_restart(tmp_path / "current.npz", source, cfg)
    header = restart.read_restart_header(path)
    assert header["physics_setup"]["algorithms"]["upper_damping"] ==         checkpoint_identity.UPPER_DAMPING_ALGORITHM_IDENTITY
    fresh = _shim_state(cfg, monkeypatch)
    _fill_setup(fresh)
    restart.restore_restart(path, fresh, cfg)
    np.testing.assert_array_equal(fresh.thp, source.thp)


def test_an_unchanged_schemes_287_checkpoint_still_resumes(tmp_path,
                                                           monkeypatch):
    """MYJ did not move, a moist run with microphysics off never reaches the
    finish clamp, and a km_opt = 1 run without khdif or kvdif under a PBL
    scheme builds no mixing tendency: the header 2.8.7 wrote for it is the
    header this build writes, so it resumes.  (Kessler served here until
    2.8.8 moved its finish clamp to WRF's REAL product.)"""
    cfg = _cfg(moist=True, mp_physics=0, bl_pbl_physics=2)
    assert restart.MICROPHYSICS_ALGORITHM_IDENTITIES[0] == "disabled"
    assert restart.PBL_ALGORITHM_IDENTITIES[2] == \
        "myj-pbl-wrf-v4.6.1-v1-mellor-yamada-2.5-janjic"
    assert checkpoint_identity.dycore_mixing_identity(cfg) is None
    source = _shim_state(cfg, monkeypatch)
    _fill_setup(source)
    source.qv.fill(np.float32(.0042))
    path = restart.write_restart(tmp_path / "released-287.npz", source, cfg)
    algorithms = restart.read_restart_header(path)["physics_setup"][
        "algorithms"]
    assert "dycore_mixing" not in algorithms
    assert "upper_damping" not in algorithms
    assert algorithms["microphysics"] == "disabled"
    fresh = _shim_state(cfg, monkeypatch)
    _fill_setup(fresh)
    restart.restore_restart(path, fresh, cfg)
    np.testing.assert_array_equal(fresh.qv, source.qv)
