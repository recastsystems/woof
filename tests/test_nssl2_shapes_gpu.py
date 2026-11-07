"""Distribution-dependent mixed-nest closure on receiving-domain coefficients."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from conftest import requires_gpu


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize('source_mp',[8,6])
def test_shape_seed_in_both_coupling_directions(source_mp):
    import cupy as cp
    from woof.config import RunConfig
    from woof.core.microphysics_transition import (
        resolve_microphysics_transition,resolve_reverse_microphysics_transition,
        launch_microphysics_edge_field)
    base=RunConfig(nx=6,ny=6,nz=4,dx=1000.,dy=1000.,ztop=10000.,dt=3.,run_seconds=9.,moist=True)
    target=replace(base,mp_physics=18,nssl_alphah=2.5,nssl_alphahl=0.)
    other=replace(base,mp_physics=source_mp)
    shape=(4,6,6);size=np.prod(shape)
    alt=np.linspace(.7,1.5,size,dtype=np.float32).reshape(shape)
    qg=np.resize(np.array([0.,1e-9,2e-8,1e-6,1e-5,1e-4],np.float32),shape)
    zero=lambda:cp.zeros(shape,cp.float32)
    state=SimpleNamespace(alt=cp.asarray(alt),qv=zero(),qc=zero(),qr=zero(),qi=zero(),qs=zero(),qg=cp.asarray(qg),
        mub2d=cp.full(shape[1:],90000.,cp.float32),mup=cp.zeros(shape[1:],cp.float32),
        c1h=cp.ones(4,cp.float32),c2h=cp.zeros(4,cp.float32))
    rho=np.float32(1.)/alt;active=qg>np.float32(1e-8)
    a=np.float32(2.5)
    factor=np.float32(((np.float32(6)+a)*(np.float32(5)+a)*(np.float32(4)+a))/((np.float32(3)+a)*(np.float32(2)+a)*(np.float32(1)+a)))
    lam=np.power(rho.astype(np.float64)*qg.astype(np.float64)*np.float64(np.float32(2.2736419413860176e-9)),.25)
    intercept=((lam*np.float64(200000.))*np.float64(factor))/np.float64(20.)
    maximum=rho.astype(np.float64)*qg.astype(np.float64)/np.float64(np.float32(9.8960235561662557e-9))
    number=np.minimum(intercept,maximum).astype(np.float32)
    expected=(number.astype(np.float64)*(1./rho.astype(np.float64))).astype(np.float32)
    expected[~active]=0.
    for contract in (resolve_microphysics_transition(other,target),resolve_reverse_microphysics_transition(target,other)):
        assert contract.target_nssl_shapes==(2.5,0.)
        assert contract.receipt()['nssl2_shapes']==[2.5,0.]
        out=cp.empty(shape,cp.float32)
        launch_microphysics_edge_field(contract,state,'qng',out=out,coupled=False)
        np.testing.assert_array_equal(cp.asnumpy(out).view(np.uint32),expected.view(np.uint32))
