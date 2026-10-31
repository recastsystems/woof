import os
import numpy as np
import pytest
from woof.hex import moving_saturation_cuda as cuda
from woof.hex.moving_balance import evaporation


@pytest.mark.skipif(os.environ.get('HEX_TEST_CUDA') != '1', reason='requires an owned CUDA card')
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('fmad', ['false', 'true'])
def test_device_saturation_keeps_numpy_words_with_fast_flags(monkeypatch, dtype, fmad):
    monkeypatch.setattr(cuda, 'OPTIONS', (f'--fmad={fmad}', '--ftz=true'))
    cuda._kernel.cache_clear()
    rng = np.random.default_rng(10301)
    shape = 55, 37
    rho = rng.uniform(.2, 1.4, shape).astype(dtype)
    heat = (rho*rng.uniform(250., 350., shape)).astype(dtype)
    zz = rng.uniform(.85, 1.15, shape).astype(dtype)
    vapor = rng.uniform(0., .03, shape).astype(dtype)
    cloud = rng.uniform(0., .002, shape).astype(dtype)
    cloud[:, 0] = -0.
    cloud[:, 1] = 1.e-40
    try:
        expected = evaporation(rho, heat, zz, vapor, cloud)
        actual = evaporation(rho, heat, zz, vapor, cloud, backend='cuda')
        for a, b in zip(actual, expected):
            assert a.tobytes() == b.tobytes()
    finally:
        cuda._kernel.cache_clear()
