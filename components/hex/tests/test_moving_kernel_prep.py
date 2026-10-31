import os
from types import SimpleNamespace
import numpy as np
import pytest


@pytest.mark.skipif(os.environ.get('HEX_TEST_CUDA') != '1', reason='requires an owned CUDA card')
def test_moving_kernels_can_all_compile_without_carried_fields():
    from woof.hex.moving_kernel_prep import prepare
    rows = prepare(SimpleNamespace(target_dzw=np.ones(55),source_dzw=np.ones(55),source_area=np.ones(17)))
    assert {r['group'] for r in rows} == {'mass','rates','momentum','balance','powers','velocity','saturation','surface','class-word-gathers','rate-staging'}
    assert sum(r['kernels'] for r in rows) >= 14
