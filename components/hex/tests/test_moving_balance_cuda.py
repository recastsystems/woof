"""The moving Newton algebra must retain the reference's rounded words."""
import os
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(os.environ.get('HEX_TEST_CUDA') != '1', reason='CUDA lease required')


@pytest.mark.parametrize('nl,nc', [(20,7),(55,19),(55,1),(2,3),(55,129),(129,1),(257,1)])
@pytest.mark.parametrize('fmad', ['false','true'])
def test_newton_words_match_host(nl,nc,fmad,monkeypatch):
    from woof.hex import moving_balance_cuda as cuda
    from woof.hex.moving_balance import project_columns, source_pressure_acceleration
    monkeypatch.setattr(cuda,'NEWTON_OPTIONS',('--fmad='+fmad,))
    cuda._kernel.cache_clear()
    z=np.arange(nl)[:,None]*400.+200.
    r=np.broadcast_to(1.2*np.exp(-z/8500),(nl,nc)).copy()
    r*=1.+np.arange(nc)[None,:]*0.0001
    h=r*(300.+.004*z)
    ref=SimpleNamespace(rho_base=.98*r,rho_theta_base=.99*h,
        exner_base=(287*.99*h/1e5)**.4)
    zz=np.ones_like(r) if nc!=129 else 1.+.01*np.sin(z/3000.+np.arange(nc)[None,:])
    v=SimpleNamespace(zz=zz,dzw=np.full(nl,400.),
        rdzu=np.full(nl,1/400.),fzm=np.full(nl,.5),fzp=np.full(nl,.5))
    q=.01*np.exp(-z/3000.)*np.ones_like(r)
    pb=zz*287*ref.rho_theta_base*ref.exner_base
    acc=source_pressure_acceleration(dict(rho=r,
        pressure_perturbation=1e5*(287*zz*h/1e5)**1.4-pb,
        density_perturbation=r-ref.rho_base,scalars=q[None]),v,[0])
    noisy_r=r*(1+.002*np.sin(z/2000.))
    noisy_r*=(v.dzw@r)/(v.dzw@noisy_r)
    noisy_h=h*(1+.002*np.cos(z/2000.))
    noisy_h*=(v.dzw@h)/(v.dzw@noisy_h)
    args=(noisy_r,noisy_h,r*q,acc,v,ref)
    expected=project_columns(*args)
    actual=project_columns(*args,backend='cuda')
    for got,want in zip(actual[:2],expected[:2]):
        np.testing.assert_array_equal(got.view(np.uint64),want.view(np.uint64))
    assert actual[2] == expected[2]
