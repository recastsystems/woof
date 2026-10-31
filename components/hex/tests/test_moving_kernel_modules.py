import os
import numpy as np
import pytest
from woof.hex import moving_kernel_modules as bank


def test_resident_modules_are_bounded_and_keep_compile_evidence_private(monkeypatch):
    monkeypatch.setattr(bank, 'MAX_MODULES', 2)
    bank.clear()
    try:
        module = object()
        evidence = dict(options=['a'])
        bank.put(('device', 'source'), module, evidence, 'aot')
        evidence['options'].append('external change')
        saved = bank.get(('device', 'source'))
        assert saved[0] is module and saved[1] == dict(options=['a'])
        saved[1]['options'].append('caller change')
        assert bank.get(('device', 'source'))[1] == dict(options=['a'])
        assert bank.get(('other device', 'source')) is None
        bank.put('second', object(), {}, 'jit')
        bank.put('third', object(), {}, 'jit')
        assert bank.get(('device', 'source')) is None
    finally:
        bank.clear()


@pytest.mark.skipif(os.environ.get('HEX_TEST_CUDA') != '1', reason='requires an owned CUDA card')
@pytest.mark.parametrize('physics', [False, True])
def test_class_caches_share_bare_modules_but_never_launch_observers(monkeypatch, physics):
    import cupy as cp
    from woof.hex.cuda_backend import KernelCache
    if physics:
        from woof.hex.mpasphys.compile_v842 import MpasPhysicsKernelCache as KernelCache
    monkeypatch.setenv('WOOF_HEX_MOVING_RESIDENT', '1')
    monkeypatch.setenv('WOOF_HEX_MOVING_KERNEL_MODULES', '1')
    bank.clear()
    source = 'extern "C" __global__ void moving_test(float *x){x[threadIdx.x]=threadIdx.x;}'
    first, second = KernelCache(), KernelCache()
    events = []
    first.post_launch = lambda *a, **k: events.append('first')
    second.post_launch = lambda *a, **k: events.append('second')
    try:
        a = first.raw_kernel('moving_test', source, module_key='hexcore.moving_test')
        b = second.raw_kernel('moving_test', source, module_key='hexcore.moving_test')
        assert list(first._modules.values())[0] is list(second._modules.values())[0]
        assert a is not b
        out = cp.empty(32, dtype=cp.float32)
        a((1,), (32,), (out,))
        b((1,), (32,), (out,))
        assert events == ['first', 'second']
        np.testing.assert_array_equal(out.get(), np.arange(32, dtype=np.float32))
        if not physics:
            assert second._load_timings[-1]['cache_status'] == 'moving_resident_module'
        assert bank.stats()['hits'] >= 1
        assert first.compile_manifest() == second.compile_manifest()
    finally:
        bank.clear()
