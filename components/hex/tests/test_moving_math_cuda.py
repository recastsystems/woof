import os
import numpy as np
import pytest


@pytest.mark.skipif(os.environ.get('HEX_TEST_CUDA') != '1',reason='requires an owned CUDA card')
def test_device_powers_and_evaporation_preserve_numpy_words(monkeypatch):
    import cupy as cp
    from woof.hex.moving_math_cuda import exner_pressure,balance_pressure
    from woof.hex.moving_balance import DRY_RGAS,DRY_CP
    from woof.hex.moving_saturation_cuda import evaporation
    rng=np.random.default_rng(3131)
    zz=rng.uniform(.3,1.2,(55,4097))
    heat=rng.uniform(20.,500.,zz.shape)
    dz,dh=cp.asarray(zz),cp.asarray(heat)
    exner,pressure=exner_pressure(dz,dh)
    # The device ports glibc pow. np.float_power is the platform libm pow on every CPU; `**` takes
    # numpy's AVX-512 SIMD pow where present and differs in the last place (woof.hex.libm_identity).
    expected=np.float_power(zz*(DRY_RGAS/1.e5)*heat,DRY_RGAS/(DRY_CP-DRY_RGAS))
    assert cp.asnumpy(exner).tobytes()==expected.tobytes()
    assert cp.asnumpy(pressure).tobytes()==(1.e5*np.float_power(expected,DRY_CP/DRY_RGAS)).tobytes()
    balance_pressure(dz,dh,pressure)
    assert cp.asnumpy(pressure).tobytes()==(1.e5*np.float_power(287*zz*heat/1.e5,1.4)).tobytes()
    rho=rng.uniform(.2,1.2,zz.shape)
    vapor=rng.uniform(0.,.015,zz.shape)
    cloud=rng.uniform(0.,.002,zz.shape)
    monkeypatch.setenv('WOOF_HEX_MOVING_POW_GPU','0')
    expected=evaporation(rho,heat,zz,vapor,cloud)
    monkeypatch.setenv('WOOF_HEX_MOVING_POW_GPU','1')
    actual=evaporation(rho,heat,zz,vapor,cloud)
    assert all(a.tobytes()==b.tobytes() for a,b in zip(actual,expected))
