import sys
from types import SimpleNamespace

from woof.hex.mpasphys.gf_mpas_v842 import _blocks_per_processor
from woof.hex.mpasphys.gf_lanes import LANES


def test_cuda_keeps_release_register_and_thread_slab_sizing(monkeypatch):
    calls = []
    def registers(*args):
        calls.append(args)
        return 65
    def occupancy(*args):
        raise AssertionError("CUDA slab sizing must retain the release calculation")
    driver = SimpleNamespace(funcGetAttribute=registers, CU_FUNC_ATTRIBUTE_NUM_REGS=4,
                             occupancyMaxActiveBlocksPerMultiprocessor=occupancy)
    cuda = SimpleNamespace(driver=driver, runtime=SimpleNamespace(is_hip=False))
    monkeypatch.setitem(sys.modules, 'cupy', SimpleNamespace(cuda=cuda))
    monkeypatch.setitem(sys.modules, 'cupy.cuda', cuda)
    props = dict(regsPerMultiprocessor=65536, maxThreadsPerMultiProcessor=2048)
    assert _blocks_per_processor(SimpleNamespace(ptr=123), props) == 7
    assert calls == [(4, 123)]


def test_amd_uses_native_occupancy_without_nvidia_register_properties(monkeypatch):
    from woof import hip_runtime
    calls = []
    def occupancy(*args):
        calls.append(args)
        return 7
    monkeypatch.setattr(hip_runtime, 'active_blocks_per_multiprocessor', occupancy)
    cuda = SimpleNamespace(runtime=SimpleNamespace(is_hip=True))
    monkeypatch.setitem(sys.modules, 'cupy', SimpleNamespace(cuda=cuda))
    assert _blocks_per_processor(SimpleNamespace(ptr=123), {}) == 7
    assert calls == [(123, LANES, 0)]
