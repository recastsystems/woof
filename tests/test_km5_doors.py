"""SMS-3DTKE admission and real namelist-to-domain settings."""
from dataclasses import replace
import tomllib
import pytest
from woof.config import RunConfig, validate_km_opt, ConfigurationRefused
from woof.experiment import build_experiment
from woof.namelist_import import import_namelists


def config(**kw):
    return RunConfig(nx=30, ny=30, nz=20, dx=3000., dy=3000.,
                     ztop=15000., dt=18., run_seconds=3600.,
                     km_opt=5, diff_opt=2, bl_pbl_physics=0, **kw)


@pytest.mark.parametrize('surface', [0, 1, 5, 91])
def test_native_surface_interfaces(surface):
    validate_km_opt(config(sf_sfclay_physics=surface))


@pytest.mark.parametrize('change,reason', [
    ({'diff_opt':1}, 'diff_opt'), ({'bl_pbl_physics':1}, 'bl_pbl_physics'),
    ({'sf_sfclay_physics':2}, 'sf_sfclay_physics'), ({'cu_physics':11}, 'cu_physics'),
])
def test_native_pairing_rules(change,reason):
    with pytest.raises(ConfigurationRefused,match=reason):
        validate_km_opt(replace(config(), **change))


def test_namelist_keeps_domain_tke_filter_controls(tmp_path):
    from test_namelist_import import _pair, INPUT_TEXT
    text=INPUT_TEXT.replace('km_opt = 4, 4,','km_opt = 5, 5,').replace(
        'bl_pbl_physics = 11, 11,','bl_pbl_physics = 0, 0,')
    text=text.replace('&dynamics', '&dynamics\n tke_mix2_off=.true.,.false.,\n tke_mix6_off=.false.,.true.,\n tke_adv_opt=1,1,\n sfs_opt=0,0,\n m_opt=0,0,')
    emitted,_=import_namelists(*_pair(tmp_path,inp=text))
    exp=build_experiment(tomllib.loads(emitted),source='SMS native settings')
    assert [(d.run.km_opt,d.run.tke_mix2_off,d.run.tke_mix6_off) for d in exp.domains] == [(5,True,False),(5,False,True)]


def test_km5_shared_workspace_matches_priced_backings(monkeypatch):
    import types
    import numpy as np
    from woof.core import state as state_mod
    from woof.core.preflight import shared_dycore_state_workspace_bytes

    monkeypatch.setattr(state_mod, "cp", np)
    small = replace(config(use_theta_m=1, do_avgflx_em=1), nx=9, ny=7, nz=5)
    large = replace(small, nx=12, ny=10, nz=8)
    domains = tuple(types.SimpleNamespace(run=c) for c in (small, large))
    workspace = state_mod.build_shared_dycore_state_workspace(domains)
    states = tuple(state_mod.DomainState(c, dycore_state_workspace=workspace)
                   for c in (small, large))
    assert workspace.nbytes == shared_dycore_state_workspace_bytes(domains)
    for name in ("nlflux", "dlk", "l_diss", "elmin", "xkmv_meso"):
        for state in states:
            assert np.shares_memory(getattr(state, name), workspace.backing(name))
    for name in ("tke", "gamu", "gamv", "thp0"):
        assert not np.shares_memory(getattr(states[0], name), getattr(states[1], name))
    for state in states:
        assert state.existing_scratch("avgflx_clock").dtype == np.int32
    assert not np.shares_memory(states[0].existing_scratch("avgflx_clock"),
                                states[1].existing_scratch("avgflx_clock"))
