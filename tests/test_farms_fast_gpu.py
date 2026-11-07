"""Parallel FARMS reduction covers a partial tile and mixed night/radius flags."""
import numpy as np
import pytest

pytestmark = pytest.mark.gpu


def test_fast_farms_partial_tile_matches_ordered_reduction():
    cp = pytest.importorskip('cupy')
    from woof.wrf_exact import ENABLED
    if ENABLED:
        pytest.skip('fast entry is excluded from strict compilation')
    from woof.core.farms import farms_surface_columns
    from woof.core.kernels import get_kernel
    nz, nc = 39, 33
    shape = (nz, nc)
    full = lambda value: cp.full(shape, value, cp.float32)
    p, rho, dz = full(90000), full(0.8), full(250)
    qv, qi, qs, qc, cf = (full(x) for x in (.008, .00001, .00002, .00003, .6))
    rc, ri, rs = full(1.e-6), full(24.e-6), full(2.e-6)
    surf = lambda value: cp.full(nc, value, cp.float32)
    alb, ssa, asy, aod, ang, cz = (surf(x) for x in (.2, .9, .7, .1, 1.3, .5))
    cz[::3] = 0
    got = farms_surface_columns(p, rho, dz, alb, ssa, asy, aod, ang, cz,
        qv, qi, qs, qc, rc, ri, rs, cf, julian=24, has_reqc=1, has_reqs=1)
    ref = cp.empty_like(got)
    args = (np.int32(nc), np.int32(nz-1), np.int32(1), np.int32(0),
        np.int32(1), np.float32(24), p, rho, dz, alb, ssa, asy, aod, ang,
        cz, qv, qi, qs, qc, rc, ri, rs, cf, ref)
    get_kernel('farms', 'farms_column')((1,), (64,), args)
    cp.testing.assert_allclose(got, ref, rtol=2.e-6, atol=.002)
    cp.testing.assert_array_equal(got[:, ::3], cp.zeros_like(got[:, ::3]))
    assert bool(cp.isfinite(got).all())


@pytest.mark.parametrize('fi,fs,ferrier', [(1,1,0),(1,0,0),(1,0,1),(0,0,0)])
def test_fused_cloud_fraction_keeps_shared_composition_words(fi, fs, ferrier):
    """Keep clear, sub/supersaturated, freezing and partial-block branches."""
    cp = pytest.importorskip('cupy')
    from woof.core.kernels import get_kernel
    from woof.core.rrtmgp import cal_cldfra1
    rng = np.random.default_rng(471)
    shape = (39, 137)
    fields = [rng.uniform(lo,hi,shape).astype('float32') for lo,hi in
              ((0,.025),(0,.0002),(0,.0002),(0,.0002),(210,305),(20000,100000))]
    fields[1][:,::7] = 0
    fields[2][:,::7] = 0
    fields[3][:,::7] = 0
    qv,qc,qi,qs,t,p = map(cp.asarray,fields)
    expected = cal_cldfra1(qv,qc,qi,qs,t,p,f_qc=True,f_qi=bool(fi),
                          f_qs=bool(fs),mp_physics=5 if ferrier else 6)
    actual = cp.empty_like(expected)
    kernel = get_kernel('farms','farms_cloud_fraction')
    kernel(((actual.size+127)//128,),(128,),
           (qv,qc,qi,qs,t,p,actual,np.int32(actual.size),np.int32(1),
            np.int32(fi),np.int32(fs),np.int32(ferrier)))
    cp.testing.assert_array_equal(actual.view(cp.uint32),expected.view(cp.uint32))
    kernel(((actual.size+127)//128,),(128,),
           (qv,qc,qi,qs,t,p,actual,np.int32(actual.size),np.int32(0),
            np.int32(fi),np.int32(fs),np.int32(ferrier)))
    cp.testing.assert_array_equal(actual,cp.zeros_like(actual))
