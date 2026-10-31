from types import SimpleNamespace as NS
import numpy as np
import pytest
from woof.hex.mpasphys.iau_witness import IauWitness

@pytest.mark.parametrize('factor, passed', [(1,True),(0,False),(2,False)])
def test_witness_detects_missing_and_doubled_native_addition(factor, passed):
    # One real cell and edge plus an allocation column. Dry theta=3,
    # rho_zz=2, zz=1.5; dry density increment=.3, theta increment=.4,
    # wind increment=.5. Zero vapour increment keeps theta_m=theta.
    shape=(1,2)
    state=NS(rho=np.full(shape,2,np.float32), scalars=np.zeros((1,1,2),np.float32))
    saved=NS(theta_m=np.full(shape,3,np.float32), normal_velocity=np.full(shape,4,np.float32))
    diag=NS(h_edge=np.full(shape,2,np.float32))
    inc={'rho':np.array([[.3,0]],np.float32), 'theta':np.array([[.4,0]],np.float32),
         'u':np.array([[.5,0]],np.float32),'scalars':np.zeros((1,1,2),np.float32)}
    sources=NS(rho=np.full(shape,.01,np.float32),rho_theta=np.full(shape,.02,np.float32),rho_u=np.full(shape,.03,np.float32),scalars=np.zeros((1,1,2),np.float32))
    witness=IauWitness(np)
    for _ in range(2):
        token=witness.before(sources,state,saved,diag,inc,np.full(shape,1.5,np.float32),
            np.array([[0,0],[-1,-1]]),{'index_qv':1},(1,1),.5,1,True)
        # rho source .1, modified theta source .7, edge momentum .9.
        sources.rho[0,0]+=np.float32(.1*factor)
        sources.rho_theta[0,0]+=np.float32(.7*factor)
        sources.rho_u[0,0]+=np.float32(.9*factor)
        witness.after(token,sources)
    receipt=witness.receipt()
    assert receipt['passed'] is passed
    assert receipt['fields']['rho']['expected_sum']==pytest.approx(.2)
    assert receipt['fields']['theta_m']['expected_sum']==pytest.approx(1.4)
    assert receipt['fields']['u']['expected_sum']==pytest.approx(1.8)
    assert all(r['words']==1 for r in receipt['fields'].values())
    assert sources.rho[0,1]==np.float32(.01)
