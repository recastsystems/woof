"""Chem state rides the engine's own checkpoint (DESIGN 2.4).

CPU only, on a NumPy-backed DomainState (the shim tests/test_restart.py
uses).  Proven here:

* a chem-off state's object graph gains nothing (no ``chem`` attribute, no
  ``chem_*``/``chem0_*``/``chemdiag_*`` member), so its restart manifest is
  exactly the fixed inventory it was;
* every attribute a chem state carries is classified by the restart
  manifest, the species fields and ledger vectors are serialized, the time
  copies are rebuilt;
* write_restart then restore_restart reproduces every chem array byte for
  byte, including adversarial bit patterns;
* a chem checkpoint is refused by a chem-off configuration (and the
  reverse), by name, instead of resuming with missing species.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from woof.config import CHEM_RUN_FIELDS, RunConfig
from woof.io import restart
from woof.state_serialization_contract import (CHEM_DIAG_PREFIX,
                                                CHEM_STATE_PREFIX,
                                                CHEM_TIME_PREFIX,
                                                serialized_state_attrs)

from test_restart import _NumpyCupyShim, _fill_serialized, _fill_setup


def _cfg(**overrides) -> RunConfig:
    values = dict(nx=6, ny=4, nz=5, dx=2000.0, dy=2000.0, ztop=8000.0,
                  dt=10.0, run_seconds=0.0, moist=True, mp_physics=0)
    values.update(overrides)
    return RunConfig(**values)


def _state(cfg, monkeypatch):
    import woof.core.state as state_module
    import woof.core.dycore  # noqa: F401  (binds the real cp first; see test_restart._shim_state)

    monkeypatch.setattr(state_module, "cp", _NumpyCupyShim)
    return state_module.DomainState(cfg)


def _chem_members(state):
    return sorted(name for name in vars(state)
                  if name == "chem" or name.startswith(
                      (CHEM_STATE_PREFIX, CHEM_TIME_PREFIX, CHEM_DIAG_PREFIX,
                       "chemwork_")))


# These are serialization-contract values, including controls which the
# normal configuration door refuses until their process exists. Writing a
# constructed configuration here proves checkpoint identity, not admission
# or numerical execution of those processes.
CHEM_CHANGED_VALUES = {
    "chem_sets": "",
    "chem_sources": "rave-3km",
    "chem_adv_opt": 2,
    "chem_mix2_off": True,
    "chem_mix6_off": True,
    "chemdt": 1.0,
    "kemit": 17,
    "biomass_burn_opt": 1,
    "plumerisefire_frq": 60,
    "dust_opt": 3,
    "seas_opt": 1,
    "dmsemis_opt": 1,
    "wetscav_onoff": -1,
    "chem_conv_tr": 0,
    "vertmix_onoff": 0,
    "aer_ra_feedback": 1,
    "aer_op_opt": 2,
    "dust_alpha": 2.0,
    "dust_gamma": 2.0,
    "dust_smtune": 2.0,
    "dust_ustune": 2.0,
    "mynn_chem_vertmx": True,
    "fire_emission_mode": "observed_hourly",
    "plume_fire_properties": "landuse",
    "aerosol_mp_coupling": "emission",
    # The HRRR v4.1.21 smoke fork's controls (fc1e2df360 and its
    # simple_dir_fdb predecessor), appended to the chem block.
    "simple_dir_fdb": 1,
    "enh_vermix": True,
    "flam_part": 0.5,
    "aer_drydep_opt": 111,
}


def _seed_contract_state(state, seed):
    _fill_setup(state)
    _fill_serialized(state, seed=seed)
    rng = np.random.default_rng(seed)
    for name in serialized_state_attrs(state):
        if not name.startswith((CHEM_STATE_PREFIX, CHEM_DIAG_PREFIX)):
            continue
        value = getattr(state, name, None)
        if value is None:
            continue
        value[...] = rng.standard_normal(value.shape).astype(value.dtype)
        if value.dtype == np.float32 and value.size:
            value.reshape(-1)[0] = np.float32(-0.0)
            value.reshape(-1)[-1] = np.float32(1.0e-42)


def _state_words(state):
    return {name: value.tobytes() for name, value in vars(state).items()
            if isinstance(value, np.ndarray)}


def test_every_chem_field_has_a_checkpoint_contract_witness():
    assert tuple(CHEM_CHANGED_VALUES) == CHEM_RUN_FIELDS


@pytest.mark.parametrize("name", CHEM_RUN_FIELDS)
@pytest.mark.parametrize("changed", [False, True], ids=["current", "changed"])
def test_each_active_chem_control_matches_its_written_checkpoint(
        name, changed, monkeypatch, tmp_path):
    cfg = _cfg(chem_sets="tracer_test")
    if changed:
        cfg = replace(cfg, **{name: CHEM_CHANGED_VALUES[name]})
    source = _state(cfg, monkeypatch)
    _seed_contract_state(source, 20261004)
    expected = {field: getattr(source, field).tobytes()
                for field in serialized_state_attrs(source)
                if getattr(source, field, None) is not None}
    path = restart.write_restart(tmp_path / "matching-chem.npz", source, cfg)
    echo = restart.read_restart_header(path)["config"]
    if cfg.chem_sets:
        # The smoke fork's keys are absent at their defaults, so a chem
        # checkpoint written before them keeps its digest; absent reads the
        # default on resume.
        fork = restart.CHEM_FORK_RUN_DEFAULTS
        assert {key for key in CHEM_RUN_FIELDS
                if key not in fork or getattr(cfg, key) != fork[key]
                } <= echo.keys()
        assert echo.get(name, fork.get(name)) == getattr(cfg, name)
    else:
        assert not set(CHEM_RUN_FIELDS) & echo.keys()
    fresh = _state(cfg, monkeypatch)
    _seed_contract_state(fresh, 20261005)
    restart.restore_restart(path, fresh, cfg)
    for field, words in expected.items():
        assert getattr(fresh, field).tobytes() == words, field


@pytest.mark.parametrize("name", CHEM_RUN_FIELDS)
@pytest.mark.parametrize("stored_changed", [False, True],
                         ids=["current-to-changed", "changed-to-current"])
def test_each_active_chem_control_change_refuses_before_state_mutation(
        name, stored_changed, monkeypatch, tmp_path):
    current = _cfg(chem_sets="tracer_test")
    changed = replace(current, **{name: CHEM_CHANGED_VALUES[name]})
    stored, live = ((changed, current) if stored_changed
                    else (current, changed))
    source = _state(stored, monkeypatch)
    _seed_contract_state(source, 20261004)
    path = restart.write_restart(tmp_path / "changed-chem.npz", source, stored)
    destination = _state(live, monkeypatch)
    _seed_contract_state(destination, 20261005)
    before = _state_words(destination)
    with pytest.raises(restart.RestartMismatchError, match=name):
        restart.restore_restart(path, destination, live)
    assert _state_words(destination) == before


@pytest.mark.parametrize("name", [name for name in CHEM_RUN_FIELDS
                                  if name != "chem_sets"])
def test_each_inactive_chem_control_is_omitted_and_restores_exactly(
        name, monkeypatch, tmp_path):
    stored = replace(_cfg(), **{name: CHEM_CHANGED_VALUES[name]})
    source = _state(stored, monkeypatch)
    _seed_contract_state(source, 20261004)
    path = restart.write_restart(tmp_path / "inactive-chem.npz", source, stored)
    assert not set(CHEM_RUN_FIELDS) & restart.read_restart_header(path)["config"].keys()
    fresh = _state(_cfg(), monkeypatch)
    _seed_contract_state(fresh, 20261005)
    restart.restore_restart(path, fresh, _cfg())
    for field in serialized_state_attrs(source):
        if getattr(source, field, None) is not None:
            assert getattr(fresh, field).tobytes() == getattr(source, field).tobytes(), field


def test_chem_off_state_object_graph_is_unchanged(monkeypatch):
    state = _state(_cfg(), monkeypatch)
    assert _chem_members(state) == []
    assert serialized_state_attrs(state) == restart.STATE_SERIALIZED_ATTRS
    manifest = restart.state_manifest(state)
    assert not any("chem" in key for key in manifest)


def test_chem_state_members_are_all_classified(monkeypatch):
    state = _state(_cfg(chem_sets="tracer_test"), monkeypatch)
    members = _chem_members(state)
    assert "chem" in members
    assert "chem_passive_1" in members and "chem0_passive_1" in members
    for name in vars(state):
        restart.classify_state_attr(name)
    assert restart.classify_state_attr("chem_passive_1") == "serialize"
    assert restart.classify_state_attr("chem0_passive_1") == "rebuild"
    assert restart.classify_state_attr("chem") == "infra"
    assert restart.classify_state_attr(
        "chemdiag_ledger_transport") == "serialize"
    manifest = restart.state_manifest(state)
    assert "state/chem_passive_1" in manifest
    assert "state/chemdiag_ledger_initial" in manifest
    assert "state/chem0_passive_1" not in manifest
    # The fields are views into the one arena.
    chem = state.chem
    assert chem.stacked(state) is chem.arena
    assert np.shares_memory(state.chem_passive_1, chem.arena)


def test_rebinding_a_species_drops_the_stacked_fast_path(monkeypatch):
    state = _state(_cfg(chem_sets="tracer_test"), monkeypatch)
    state.chem_passive_1 = state.chem_passive_1.copy()
    assert state.chem.stacked(state) is None


def test_chem_restart_roundtrips_bit_exactly(monkeypatch, tmp_path):
    cfg = _cfg(chem_sets="tracer_test")
    state = _state(cfg, monkeypatch)
    _fill_setup(state)
    _fill_serialized(state, seed=20260930)
    rng = np.random.default_rng(11)
    chem_names = [name for name in serialized_state_attrs(state)
                  if name.startswith((CHEM_STATE_PREFIX, CHEM_DIAG_PREFIX))]
    assert chem_names
    for name in chem_names:
        array = getattr(state, name)
        if array.dtype == np.float32:
            array[...] = rng.standard_normal(array.shape).astype(np.float32)
            flat = array.reshape(-1)
            flat[0] = np.float32(-0.0)
            flat[-1] = np.float32(1.0e-42)
        elif array.dtype == np.float64:
            array[...] = rng.standard_normal(array.shape)
        else:
            array[...] = 1
    state.elapsed_seconds = 360.0
    path = restart.write_restart(tmp_path / "chem.npz", state, cfg)

    fresh = _state(cfg, monkeypatch)
    _fill_setup(fresh)
    restart.restore_restart(path, fresh, cfg)
    for name in chem_names:
        assert (getattr(fresh, name).tobytes()
                == getattr(state, name).tobytes()), name
    # The restored species is still a view of the fresh state's arena.
    assert fresh.chem.stacked(fresh) is fresh.chem.arena
