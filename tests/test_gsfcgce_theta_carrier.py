"""Column adapters must pass dry theta to native WRF microphysics."""
from types import SimpleNamespace
import sys
import numpy as np
import pytest


@pytest.mark.parametrize('moist_theta', [0, 1])
def test_native_column_receives_dry_theta_words(monkeypatch, moist_theta):
    from woof.core import constants as c
    from woof.core import microphysics
    from woof.core import mp_column_schemes as gce

    # Exercise production preparation with CPU arrays and the dry-theta
    # conversion protocol; the launch captures its actual argument.
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
    # This branch's DomainState integrates dry theta. The settings union
    # supplies the same total_theta protocol for a moist prognostic carrier.
    state.prognostic_theta = lambda: thb[:, None, None] + state.thp
    state.total_theta = lambda: (state.prognostic_theta() / (np.float32(1.) + np.float32(c.RVOVRD) * state.qv)
                                if moist_theta else state.prognostic_theta())
    expected = state.total_theta().copy()
    if moist_theta:
        assert np.any(expected.view(np.uint32) != state.prognostic_theta().view(np.uint32))
    for name in ('qc', 'qr', 'qi', 'qs', 'qg'):
        setattr(state,name,np.zeros(shape,np.float32))
    captured = []
    monkeypatch.setattr(gce,'launch_column',lambda row,theta,*a,**kw: captured.append(theta.copy()))
    monkeypatch.setattr(microphysics,'save_pre_mp_theta',lambda state: None)
    monkeypatch.setattr(microphysics,'moist_physics_finish',lambda *args: None)
    gce.apply(state,SimpleNamespace(mp_physics=97,gsfcgce_hail=0,gsfcgce_2ice=0),1.)
    np.testing.assert_array_equal(captured[0].view(np.uint32),expected.view(np.uint32))
