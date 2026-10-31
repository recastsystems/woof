import os
import numpy as np
import pytest


@pytest.mark.skipif(os.environ.get('HEX_TEST_CUDA') != '1',reason='requires an owned CUDA card')
def test_device_phase_transfer_and_receipts_keep_host_words(monkeypatch):
    from woof.hex.moving_balance import saturation_adjustment
    rng=np.random.default_rng(9931)
    shape=(9,129)
    rho=rng.uniform(.5,1.3,shape)
    heat=rho*rng.uniform(270.,320.,shape)
    zz=rng.uniform(.8,1.1,shape)
    vapor=rho*rng.uniform(0.,.015,shape)
    cloud=rho*rng.uniform(0.,.002,shape)
    donor=rng.uniform(0.,1.e-4,shape)
    for water,threshold in [(cloud,1.e-9),(np.zeros_like(cloud),1.e-9),(cloud,-1.e-9)]:
        total=lambda a:np.sum(a)
        expected=saturation_adjustment(vapor,water,rho,heat,zz,donor,total,threshold=threshold,backend='cpu')
        actual=saturation_adjustment(vapor,water,rho,heat,zz,donor,total,threshold=threshold,backend='cuda')
        for a,b in zip(actual[:3],expected[:3]):
            assert a.tobytes()==b.tobytes()
        assert actual[3]==expected[3]
