"""Preserve CuPy preparation bits, including flushed subnormal inputs.

theta and dz keep the CuPy ufunc roundings; the Exner function is WRF's
phy_prep (p/p1000mb)**rcp with glibc's powf (morrison.cu mo_exner) since
development branch, graded against the host glibc on normal pressures.
"""
import numpy as np
import pytest

@pytest.mark.parametrize("thb_full,phb_full", [(False,False),(False,True),(True,False),(True,True)])
def test_prepare_bits(thb_full,phb_full):
    cp = pytest.importorskip("cupy")
    try:
        if not cp.cuda.runtime.getDeviceCount(): pytest.skip("CUDA unavailable")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA unavailable")
    from woof.core.morrison import _prepare_fields
    from woof.core import constants as c
    shape=(3,2,5); ncol=10
    rng=np.random.default_rng(412)
    thb=cp.asarray(rng.uniform(280,310,shape if thb_full else (3,)).astype("f4"))
    phb=cp.asarray(rng.uniform(0,30000,(4,2,5) if phb_full else (4,)).astype("f4"))
    thp=cp.asarray(rng.uniform(-10,10,shape).astype("f4"))
    php=cp.asarray(rng.uniform(-100,100,(4,2,5)).astype("f4"))
    pressure=cp.asarray(rng.uniform(20000,100000,shape).astype("f4"))
    # Exercise the ufunc FTZ behavior without replacing a reference result.
    thb.flat[0]=np.float32(1e-40); thp.flat[0]=np.float32(1e-40)
    phb.flat[0]=np.float32(1e-40); php.flat[0]=np.float32(1e-40)
    pressure.flat[0]=np.float32(1e-40)
    tiny = np.float32(np.finfo(np.float32).tiny)
    pressure.flat[1] = tiny
    thb.flat[ncol if thb_full else 1] = tiny
    thp.flat[ncol] = -np.nextafter(tiny, np.float32(np.inf))
    phb.flat[ncol if phb_full else 1] = tiny
    phb.flat[2*ncol if phb_full else 2] = np.float32(2)*tiny
    php.flat[ncol] = np.float32(0)
    php.flat[2*ncol] = np.float32(0)
    theta=cp.empty(shape,dtype="f4"); dz=cp.empty_like(theta)
    tb=thb if thb_full else thb[:,None,None]
    pb=phb if phb_full else phb[:,None,None]
    expected_theta=tb+thp
    z=(pb+php)/np.float32(c.G)
    expected_dz=z[1:]-z[:-1]
    _prepare_fields(thb,phb,thp,php,np.int32(ncol),np.int32(thb_full),np.int32(phb_full),
                    np.float32(c.G),theta,dz)
    for actual,expected in zip((theta,dz),(expected_theta,expected_dz)):
        np.testing.assert_array_equal(cp.asnumpy(actual).view("u4"),cp.asnumpy(expected).view("u4"))
