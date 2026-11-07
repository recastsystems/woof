"""Stock 4.7.1 MP21 input and the pinned 4.6.1 passive CCN carrier."""
from dataclasses import replace
from types import SimpleNamespace

import netCDF4
import numpy as np
import pytest

from woof.config import RunConfig
from woof.ingest import wrfinput as wi
from wrf_input_fixtures import _small_wrfinput


def _cfg(**changes):
    return RunConfig(**(dict(nx=5, ny=4, nz=2, dx=30000., dy=30000.,
                            dt=60., ztop=20000., run_seconds=7200., moist=True,
                            mp_physics=18, nssl_2moment_on=0,
                            nssl_hail_on=0, nssl_density_on=0) | changes))


def _state(cfg):
    arrays = {}
    for name in wi.active_moisture_map(cfg).values():
        arrays[name] = np.full((cfg.nz, cfg.ny, cfg.nx), -17., np.float32)
        arrays[name + "0"] = np.full_like(arrays[name], -23.)
    return SimpleNamespace(**arrays)


@pytest.mark.parametrize("cccn", [0.5e9, 1.e7, 0., -1.e7])
def test_missing_mp21_ccn_gets_native_cold_fill_and_rk_copy(tmp_path, cccn):
    cfg = _cfg(nssl_cccn=cccn)
    required, allowed = wi.active_moisture_inventory(cfg)
    assert "QNCCN" not in required and "QNCCN" in allowed
    path = tmp_path / "wrfinput_d01"
    _small_wrfinput(path, moisture_names=tuple(allowed - {"QNCCN"}))
    with netCDF4.Dataset(path, "a") as dataset:
        dataset.createDimension("soil_layers_stag", 4)
        dataset.setncatts(dict(MP_PHYSICS=18,
                              SF_SURFACE_PHYSICS=cfg.sf_surface_physics,
                              BL_PBL_PHYSICS=cfg.bl_pbl_physics))
    dimensions = dict(west_east=5, west_east_stag=6, south_north=4,
                      south_north_stag=5, bottom_top=2, bottom_top_stag=3,
                      soil_layers_stag=4)
    restored = wi.read_wrfinput(path, cfg=cfg, expected_dimensions=dimensions,
                               require_complete=False)
    assert "QNCCN" not in restored.raw
    assert "QNCCN" not in wi.missing_required_wrfinput(restored.raw, cfg)
    state = _state(cfg)
    wi._restore_active_moisture(state, restored.raw, cfg, np)
    native = np.float32(cccn) / np.float32(1.225)
    assert np.all(state.qnn.view("u4") == native.view("u4"))
    assert state.qnn.tobytes() == state.qnn0.tobytes()
    # A missing evolved restart carrier must never be manufactured.
    with pytest.raises(ValueError, match="lacks active.*QNCCN"):
        wi._restore_active_moisture(_state(cfg), restored.raw, cfg, np, warm=True)


@pytest.mark.parametrize("warm", [False, True])
def test_supplied_mp21_ccn_words_are_preserved(warm):
    cfg = _cfg()
    raw = {wrf: np.zeros((2, 4, 5), np.float32)
           for wrf in wi.active_moisture_map(cfg)}
    raw["QNCCN"] = np.arange(40, dtype=np.float32).reshape(2, 4, 5) * 1.25
    state = _state(cfg)
    wi._restore_active_moisture(state, raw, cfg, np, warm=warm)
    assert state.qnn.tobytes() == raw["QNCCN"].tobytes() == state.qnn0.tobytes()


@pytest.mark.parametrize("changes", [dict(nssl_2moment_on=1, nssl_density_on=1),
                                     dict(nssl_hail_on=2, nssl_density_on=1)])
def test_mp18_and_mp19_ccn_remains_required(changes):
    cfg = replace(_cfg(), **changes)
    required, allowed = wi.active_moisture_inventory(cfg)
    assert "QNCCN" in required and "QNCCN" in allowed
    raw = {name: np.zeros((2, 4, 5), np.float32) for name in allowed - {"QNCCN"}}
    assert "QNCCN" in wi.missing_required_wrfinput(raw, cfg)
    with pytest.raises(ValueError, match="lacks active.*QNCCN"):
        wi._restore_active_moisture(_state(cfg), raw, cfg, np)


def test_mp21_still_requires_every_active_mass():
    cfg = _cfg()
    required, _ = wi.active_moisture_inventory(cfg)
    assert required == frozenset(("QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP"))
    assert set(wi.missing_required_wrfinput({}, cfg)) >= required


def test_inventory_without_optional_selectors_keeps_default_mp18_contract():
    cfg = SimpleNamespace(mp_physics=18, moist=True)
    required, allowed = wi.active_moisture_inventory(cfg)
    assert "QNCCN" in required and "QNCCN" in allowed


def test_legacy_verification_adapter_uses_the_same_mp21_cold_fill():
    from woof.verify.cases import real74_n5s

    cfg = _cfg()
    _, allowed = wi.active_moisture_inventory(cfg)
    raw = {name: np.zeros((2, 4, 5), np.float32)
           for name in allowed - {"QNCCN"}}
    state = _state(cfg)
    real74_n5s._restore_active_moisture(state, raw, cfg, np)
    native = np.float32(cfg.nssl_cccn) / np.float32(1.225)
    assert np.all(state.qnn.view("u4") == native.view("u4"))
    assert state.qnn.tobytes() == state.qnn0.tobytes()
