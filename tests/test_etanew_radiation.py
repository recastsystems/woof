"""WRF 4.6.1 older Eta radiation membership and frozen condensate mapping."""
import numpy as np
import pytest
from woof.core.rrtmg_legacy import legacy_cloud_fraction_flags
from woof.core.rrtmg_legacy_prep import _moist_prep, _moist_prep_b

def test_eta_registry_flags():
    assert legacy_cloud_fraction_flags(95) == (False, True)

@pytest.mark.parametrize('icloud', [0, 1])
def test_eta_wrapper_phase_mapping(icloud):
    # WRF wrappers :12115-12125 (LW), :10716-10726 (SW).
    t=np.array([250., 280., 260.], np.float32)
    qc=np.array([2e-4, 3e-4, -1e-5], np.float32)
    qr=np.array([1e-4, 2e-4, 3e-4], np.float32)
    qs=np.array([6e-4, 4e-4, -2e-5], np.float32)
    z=np.zeros(3, np.float32)
    args=(icloud,False,True,True,False,True,False,t,z,qc,qr,z,qs,z,z)
    scalar=_moist_prep(3,*args,eta_ferrier=True)
    batch=_moist_prep_b(np,*args,eta_ferrier=True)
    for a,b in zip(scalar,batch):
        assert a.tobytes()==b.tobytes()
    expected=(np.maximum(0,qc),np.maximum(0,np.float32(.1)*qs),np.float32(.9)*qs)
    for i,e in zip((1,3,4),expected):
        np.testing.assert_array_equal(scalar[i], e if icloud else z)
    if icloud:
        np.testing.assert_array_equal(scalar[2], np.where(t<np.float32(273.15),0,qr))

@pytest.mark.gpu
def test_eta_cloud_fraction_uses_carried_phase():
    cp=pytest.importorskip('cupy')
    from woof.core.rrtmgp import cal_cldfra1
    qv=cp.full((2,3),3e-4,cp.float32)
    qc=cp.zeros_like(qv); qs=cp.full_like(qv,2e-5)
    t=cp.full_like(qv,250); p=cp.full_like(qv,40000)
    def run(ice):
        return cal_cldfra1(qv,qc,qc,qs,t,p,f_qi=False,f_qs=True,f_ice_phy=ice)
    liquid=run(cp.zeros_like(qv)); ice=run(cp.ones_like(qv))
    assert bool(cp.all(ice>liquid)) and bool(cp.all(liquid>0))
    with pytest.raises(ValueError, match='F_ICE_PHY'):
        run(None)

@pytest.mark.gpu
def test_eta_cloud_fraction_matches_unmodified_wrf_routine(tmp_path):
    from pathlib import Path
    import json
    cp=pytest.importorskip('cupy')
    from woof.core.rrtmgp import cal_cldfra1
    path=Path(__file__).parent/'data/oracles/etanew/radiation.csv'
    words=np.array([[int(x,16) for x in line.split(',')] for line in path.read_text().splitlines()[1:]],np.uint32)
    values=words.view(np.float32)
    qv,qc,qs,t,p,phase,ref=[cp.asarray(values[:,i:i+1].copy()) for i in range(7)]
    got=cal_cldfra1(qv,qc,cp.zeros_like(qv),qs,t,p,f_qi=False,f_qs=True,f_ice_phy=phase)
    host=cp.asnumpy(got).ravel(); reference=values[:,6]
    max_abs=float(np.max(np.abs(host-reference)))
    max_ulp=int(np.max(np.abs(host.view(np.uint32).astype(np.int64)-words[:,6].astype(np.int64))))
    (tmp_path/'radiation-comparison.json').write_text(json.dumps(dict(columns=12,max_abs=max_abs,max_ulp=max_ulp,tolerance_abs=0)))
    np.testing.assert_array_equal(host.view(np.uint32),words[:,6])
    print(f'ETA radiation cloud fraction: max_abs={max_abs} max_ulp={max_ulp}')
