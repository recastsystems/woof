"""Cooperative XA integration agrees with the original serial quadrature."""
import numpy as np
import pytest


def test_cooperative_xa_quadrature_matches_serial_native_nodes():
    cp = pytest.importorskip('cupy')
    from woof.wrf_exact import ENABLED
    if ENABLED:
        pytest.skip('cooperative XA is the fast-numerics route')
    from woof.core.kernels import module_source, module_options
    source = module_source('wind_mav') + r'''
extern "C" __global__ void compare_xa(const float* p,float* out){
 int i=blockIdx.x,lane=threadIdx.x&31;
 float a=p[5*i],b=p[5*i+1],r=p[5*i+2],sx=p[5*i+3],sy=p[5*i+4];
 float value=mav_gaussian_warp(a,b,r,sx,sy);
 if(lane==0){out[2*i]=mav_gaussian(a,b,r,sx,sy);out[2*i+1]=value;}
}
'''
    inputs = np.array([
        [0, 0, 63, 90, 75], [30, 0, 63, 90, 75],
        [0, 40, 63, 90, 75], [80, -30, 63, 90, 75],
        [-50, 90, 40, 30, 45], [200, 0, 50, 35, 30],
        [0, 0, 63, 2000, 1400], [150, 110, 63, 220, 180],
    ], dtype=np.float32)
    fn = cp.RawModule(code=source, options=module_options('wind_mav')).get_function('compare_xa')
    result = cp.empty((len(inputs), 2), cp.float32)
    fn((len(inputs),), (32,), (cp.asarray(inputs), result))
    actual = cp.asnumpy(result)
    np.testing.assert_allclose(actual[:, 1], actual[:, 0], rtol=3e-6, atol=1e-7)
