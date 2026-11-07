"""The fast linear MATRIX factorization solves WRF's full bordered operator."""
import numpy as np
import pytest
from conftest import requires_gpu

pytestmark = [pytest.mark.gpu, requires_gpu]


@pytest.mark.parametrize('levels', [4, 40, 60, 257])
def test_fast_bordered_solver_preserves_every_matrix_term(levels):
    import cupy as cp
    from woof.core.acm2pbl import module_source
    from woof.core.kernels import compile_no_ftz
    from woof.numerics import is_strict
    if is_strict():
        pytest.skip('strict mode exercises the native MATRIX operation order')
    source = module_source() + r'''
extern "C" __global__ void bordered_probe(float *p, int kl) {
    int lane=threadIdx.x; long long pitch=(kl+2)*ACM2_BLOCK;
    AcmCol a={p+lane}, b={p+pitch+lane}, c={p+2*pitch+lane}, e={p+3*pitch+lane};
    AcmCol d[4], x[4];
    for(int v=0;v<4;++v) {d[v]={p+(4+v)*pitch+lane};x[v]={p+(8+v)*pitch+lane};}
    AcmCol z={p+12*pitch+lane}, inv={p+13*pitch+lane};
    acm2_bordered<4>(a,b,c,d,e,x,z,inv,kl);
}
'''
    module = compile_no_ftz(source, module_key='tests:acm2-bordered',
                            filename='acm2_bordered_probe.cu', device=cp.cuda.Device().id)
    rng = np.random.default_rng(3000 + levels)
    values = np.zeros((14, levels+2, 32), np.float32)
    matrices, rhs = [], []
    for lane in range(32):
        scale = 10.0 ** (-4 + lane * 8 / 31)
        a, c, e = (-rng.uniform(.01, 1, levels).astype(np.float32) * scale
                   for _ in range(3))
        a[0] = c[0] = e[-1] = 0
        # Native MATRIX treats A(2), rather than C(2), as the first-column term.
        matrix = np.diag(np.ones(levels, np.float64))
        for k in range(levels):
            if k >= 1: matrix[k, 0] += a[k]
            if k >= 2: matrix[k, k-1] += c[k]
            if k < levels-1: matrix[k, k+1] += e[k]
            matrix[k, k] = 1 + sum(abs(matrix[k, j]) for j in range(levels) if j != k)
        b = matrix.diagonal().astype(np.float32)
        matrix[np.diag_indices(levels)] = b
        values[0, 1:levels+1, lane] = a
        values[1, 1:levels+1, lane] = b
        values[2, 1:levels+1, lane] = c
        values[3, 1:levels+1, lane] = e
        d = rng.normal(size=(4, levels)).astype(np.float32) * np.array([300, .01, 5, 1],np.float32)[:,None]
        values[4:8, 1:levels+1, lane] = d
        matrices.append(matrix);rhs.append(d.astype(np.float64))
    device = cp.asarray(values)
    module.get_function('bordered_probe')((1,), (32,), (device, np.int32(levels)))
    actual = device.get()[8:12, 1:levels+1].astype(np.float64)
    errors = []
    for lane, (a, d) in enumerate(zip(matrices, rhs)):
        x = actual[:,:,lane].T
        residual = np.abs(a @ x - d.T)
        scale = np.abs(a) @ np.abs(x) + np.abs(d.T)
        errors.append(float(np.max(residual / np.maximum(scale, np.finfo(np.float64).tiny))))
    # Componentwise backward error, including every border/sub/superdiagonal term.
    assert max(errors) < 16 * np.finfo(np.float32).eps, (levels, max(errors))
