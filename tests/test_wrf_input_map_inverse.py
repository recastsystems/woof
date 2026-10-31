"""Regression for WRF's independently stored, noncanonical MF_VX_INV words."""
import numpy as np
import pytest
from conftest import requires_gpu
from woof.core.state import DomainState
from woof.config import RunConfig


def test_input_inverse_is_independent_setup_state():
    cfg = RunConfig(nx=8, ny=8, nz=8, dx=3000., dy=3000., ztop=16000., dt=18., run_seconds=0.)
    state = DomainState(cfg, array_module=np)
    factor = np.full(state.msfv.shape, np.float32(1.03))
    stored = np.nextafter(np.float32(1.) / factor, np.float32(0.))
    state.set_map_coriolis(msfv=factor, msfv_inv=stored)
    np.testing.assert_array_equal(state.msfv_inv.view('u4'), stored.view('u4'))
    assert np.all(state.msfv_inv != np.float32(1.) / state.msfv)
    from woof.core.device_inventory import state_array_shapes
    from woof.state_serialization_contract import STATE_SETUP_ARRAYS
    assert state_array_shapes(cfg)['msfv_inv'] == state.msfv_inv.shape
    assert 'msfv_inv' in STATE_SETUP_ARRAYS
    state.set_map_coriolis(msfv=factor)
    np.testing.assert_array_equal(state.msfv_inv, np.float32(1.) / factor)
    with pytest.raises(ValueError, match='msfv_inv must have shape'):
        state.set_map_coriolis(msfv_inv=np.ones((2, 2), np.float32))


@requires_gpu
@pytest.mark.gpu
def test_native_v_flux_uses_stored_inverse_words():
    import cupy as cp
    from woof.core.dycore import _couple_momentum_kernel
    # Native couple_momentum, module_big_step_utilities_em.F:383:
    # rv = v * (c1h * muv + c2h) * msfvx_inv, rounded at each operator.
    factor = np.array([1.03, 1.04, 1.05], np.float32)
    inverse = np.nextafter(np.float32(1.) / factor, np.float32(0.))
    wind = np.array([[1.234, -2.345, 3.456]], np.float32)
    mass = np.array([90000., 90500., 91000.], np.float32)
    c1 = np.array([.75], np.float32); c2 = np.array([25000.], np.float32)
    expected = (wind * (c1[:,None] * mass + c2[:,None])) * inverse
    rebuilt = (wind * (c1[:,None] * mass + c2[:,None])) * (np.float32(1.) / factor)
    assert np.any(expected.view('u4') != rebuilt.view('u4'))
    out = cp.empty_like(cp.asarray(wind))
    _couple_momentum_kernel(True, True)(cp.asarray(wind), cp.asarray(c1), cp.asarray(c2),
        cp.asarray(mass), cp.asarray(inverse), np.int32(3), out)
    np.testing.assert_array_equal(cp.asnumpy(out).view('u4'), expected.view('u4'))
