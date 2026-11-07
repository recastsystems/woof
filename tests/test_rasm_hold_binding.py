"""The fast mean must retain the time-t producer while the solve advances."""
from types import SimpleNamespace
import numpy as np
import pytest
from woof.core import rasm


@pytest.mark.parametrize('nest_hold', [False, True])
def test_rasm_hold_survives_current_theta_update(monkeypatch, nest_hold):
    monkeypatch.setattr(rasm, 'WRF_EXACT', False)
    current=np.full((3,2,2), 7, np.float32)
    saved=np.full_like(current, 5)
    state=SimpleNamespace(thp=current,thp0=saved,
                          existing_scratch=lambda _:pytest.fail('no extra hold allocation'))
    if nest_hold:
        state._wrf_th_phy_m_t0=np.full_like(current,3)
        expected=state._wrf_th_phy_m_t0
    else:expected=saved
    cfg=SimpleNamespace(mean_diag=1,diurnal_diag=0,dt=12.)
    rasm.prepare(state,cfg)
    current.fill(11)
    assert np.shares_memory(state._rasm_theta_bound,expected)
    np.testing.assert_array_equal(state._rasm_theta_bound,expected[0])
    # A completed solve must checkpoint its owner without serializing the view.
    from woof.io.restart import state_manifest
    carried=SimpleNamespace(thp=current,thp0=saved,
                            _rasm_theta_bound=state._rasm_theta_bound,
                            _rasm_bound_calls=(saved,current))
    manifest=state_manifest(carried)
    assert set(manifest)=={'state/thp'}
    assert manifest['state/thp'] is current


def test_fractional_diurnal_keeps_native_finish_check(monkeypatch):
    monkeypatch.setattr(rasm,'WRF_EXACT',False)
    assert not rasm._bound_clock(SimpleNamespace(diurnal_diag=1,dt=12.5))
    assert not rasm._bound_clock(SimpleNamespace(diurnal_diag=1,dt=13.))
    assert rasm._bound_clock(SimpleNamespace(diurnal_diag=1,dt=12.))
