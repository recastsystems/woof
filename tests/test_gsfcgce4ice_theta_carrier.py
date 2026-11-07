"""MP7's production wrapper composes dry theta for either WRF carrier."""
from types import SimpleNamespace
import numpy as np
import pytest


@pytest.mark.parametrize('moist_theta', [0, 1])
def test_mp7_column_receives_dry_theta_words(monkeypatch, moist_theta):
    from woof.core import microphysics as mp, gsfcgce4ice as g4, constants as c
    from woof.core.state import DomainState
    from woof.physics_registry import column_microphysics_binding

    monkeypatch.setattr(mp, 'cp', np)
    shape = (3, 1, 2)
    qv = np.array([.001, .01, .005, .02, .003, .015], np.float32).reshape(shape)
    dry = np.array([299., 302., 306., 308., 310., 315.], np.float32).reshape(shape)
    stored = dry * (np.float32(1) + np.float32(c.RVOVRD) * qv) if moist_theta else dry
    thb = np.full(3, 290., np.float32)
    state = SimpleNamespace(p=np.full(shape, 90000., np.float32), thb=thb,
        thp=stored-thb[:, None, None], qv=qv, use_theta_m=moist_theta,
        phb=np.arange(4, dtype=np.float32)*np.float32(c.G*100),
        php=np.zeros((4, 1, 2), np.float32), alt=np.ones(shape, np.float32),
        w=np.zeros((4, 1, 2), np.float32), ht=np.zeros((1, 2), np.float32),
        physics=SimpleNamespace(fields={'xland': np.ones((1, 2), np.float32)}, microphysics_updates=0))
    scratch = {}
    state.scratch = lambda size, name: scratch.setdefault(name, np.empty(size, np.float32))
    state.prognostic_theta = lambda: DomainState.prognostic_theta(state)
    state.total_theta = lambda: DomainState.total_theta(state)
    binding = column_microphysics_binding(7)
    names = set(binding['species'].values())
    for attrs in binding['groups'].values():
        names.update(attrs)
    names.update(binding.get('radiation_radii', {}))
    names.update(binding.get('sum_outputs', {}))
    for attrs in binding.get('sum_outputs', {}).values():
        names.update(attrs)
    for name in names - {'qv'}:
        setattr(state, name, np.zeros(shape, np.float32))
    expected = state.total_theta().copy()
    if moist_theta:
        assert np.any(expected.view(np.uint32) != state.prognostic_theta().view(np.uint32))
    observed = []
    finished = []
    monkeypatch.setattr(g4, binding['launcher'], lambda **kw: observed.append(kw['th'].copy()))
    monkeypatch.setattr(mp, 'save_pre_mp_theta', lambda state: None)
    monkeypatch.setattr(mp, 'moist_physics_finish', lambda state, cfg, theta, dt: finished.append(theta.copy()))
    mp._apply_column_kernel(state, SimpleNamespace(mp_physics=7, dx=30000.), 60., binding, refl_10cm_due=False)
    for words in (observed[0], finished[0]):
        np.testing.assert_array_equal(words.view(np.uint32), expected.view(np.uint32))
