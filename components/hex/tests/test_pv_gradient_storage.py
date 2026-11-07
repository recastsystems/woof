"""APVM keeps the same PV words when unused gradient outputs are omitted."""
import numpy as np
import pytest
from woof.hex.cuda_horizontal_v841 import _CUDA_SOURCE


@pytest.mark.gpu
def test_null_gradient_outputs_preserve_pv_words():
    cp = pytest.importorskip('cupy')
    if not cp.cuda.runtime.getDeviceCount():
        pytest.skip('CUDA required')
    kernel = cp.RawModule(code=_CUDA_SOURCE,
        options=('--fmad=false', '--ftz=true', '--std=c++17')).get_function('pv_apvm_v841_f32')
    rng = np.random.default_rng(842)
    nz, nc, ne, nv = 7, 257, 771, 514
    def a(shape):
        return cp.asarray(rng.uniform(-1, 1, shape).astype(np.float32))
    normal, tangent = a((nz, ne)), a((nz, ne))
    vertices, cells = a((nz, nv)), a((nz, nc))
    words = np.array([0, 0x80000000, 1, 0x007fffff], np.uint32).view(np.float32)
    normal[0, :4] = cp.asarray(words)
    args = (normal, tangent, vertices, cells,
        cp.asarray(rng.integers(0, nv, (ne, 2), dtype=np.int32)),
        cp.asarray(rng.integers(0, nc, (ne, 2), dtype=np.int32)),
        a(ne), a(ne), np.float32(.01), np.int32(nz), np.int32(nc),
        np.int32(ne), np.int32(nv))
    initial = a((nz, ne))
    expected, actual = initial.copy(), initial.copy()
    kernel(((nz*ne+127)//128,), (128,), (*args, expected,
        cp.empty_like(initial), cp.empty_like(initial)))
    kernel(((nz*ne+127)//128,), (128,), (*args, actual, np.uint64(0), np.uint64(0)))
    np.testing.assert_array_equal(cp.asnumpy(actual).view(np.uint32),
                                  cp.asnumpy(expected).view(np.uint32))
