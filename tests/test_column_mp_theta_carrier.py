"""Column adapters must pass dry theta to native WRF microphysics."""
from types import SimpleNamespace
import sys
import numpy as np
import pytest


@pytest.mark.parametrize('moist_theta', [0, 1])
@pytest.mark.parametrize('adapter', ['sbu_ylin', 'gsfcgce'])
def test_native_column_receives_dry_theta_words(monkeypatch, moist_theta, adapter):
    from woof.core.state import DomainState
    from woof.core import constants as c
    from woof.core import microphysics
    from woof.core import column_microphysics as sbu
    from woof.core import mp_column_schemes as gce

    # Exercise the production preparation with arrays on the CPU. The real
    # state's conversion is used, and the launch captures its argument.
    monkeypatch.setitem(sys.modules, 'cupy', np)
    shape = (3, 1, 2)
    qv = np.array([.001, .01, .005, .02, .003, .015], np.float32).reshape(shape)
    full = np.array([299., 302., 306., 308., 310., 315.], np.float32).reshape(shape)
    prognostic = full * (np.float32(1) + np.float32(c.RVOVRD) * qv) if moist_theta else full
    thb = np.full(3, 290., np.float32)
    state = SimpleNamespace(p=np.full(shape, 90000., np.float32), thb=thb,
        thp=prognostic-thb[:,None,None], qv=qv, use_theta_m=moist_theta,
        phb=np.arange(4,dtype=np.float32)*np.float32(c.G*100),
        php=np.zeros((4,1,2),np.float32), alt=np.ones(shape,np.float32),
        ht=np.zeros((1,2),np.float32))
    scratch = {}
    state.scratch = lambda size, name: scratch.setdefault(name,np.empty(size,np.float32))
    state.prognostic_theta = lambda: DomainState.prognostic_theta(state)
    state.total_theta = lambda: DomainState.total_theta(state)
    expected = state.total_theta().copy()
    if moist_theta:
        assert np.any(expected.view(np.uint32) != state.prognostic_theta().view(np.uint32))
    if adapter == 'sbu_ylin':
        observed, _ = sbu.prepare_inputs(state)
    else:
        for name in ('qc', 'qr', 'qi', 'qs', 'qg'):
            setattr(state,name,np.zeros(shape,np.float32))
        captured = []
        monkeypatch.setattr(gce,'launch_column',lambda row,theta,*a,**kw: captured.append(theta.copy()))
        monkeypatch.setattr(microphysics,'save_pre_mp_theta',lambda state: None)
        monkeypatch.setattr(microphysics,'moist_physics_finish',lambda *args: None)
        gce.apply(state,SimpleNamespace(mp_physics=97,gsfcgce_hail=0,gsfcgce_2ice=0),1.)
        observed = captured[0]
    np.testing.assert_array_equal(observed.view(np.uint32),expected.view(np.uint32))
