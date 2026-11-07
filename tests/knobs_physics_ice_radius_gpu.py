"""Real legacy LW/SW prep tests for tuned ice bounds and untouched fields."""
import argparse,json,os
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--value',type=float,required=True)
p.add_argument('--out',type=Path,required=True);a=p.parse_args()
assert os.environ.get('GPU_MUTEX_LANE')=='port-tuning-knobs'
assert os.environ.get('CUDA_VISIBLE_DEVICES')
from woof.knobs import physics
physics.activate({'physics.rrtmg.radius_ice':a.value})
import cupy as cp
from test_rrtmg_legacy_prep_device import synthetic,side_kwargs,device_kwargs
from woof.core import rrtmg_legacy_device as dev,rrtmg_legacy_prep as ref
ref._PERFWAVE_DEVICE_XP=None
kw=synthetic(5,nz=3)
radius=np.array([10,50,100,120,140],np.float32)
kw['re_ice']=np.broadcast_to((radius*np.float32(1e-6))[:,None],(5,3)).copy()
kw['cldfra3d'].fill(1)
result=[]
for sw in [False,True]:
    args=side_kwargs(kw,sw)
    reference=(ref.swrad_prep_batch if sw else ref.lwrad_prep_batch)(**args)
    call=dev.swrad_prep_batch_device if sw else dev.lwrad_prep_batch_device
    actual=call(**device_kwargs(args))
    ice=cp.asnumpy(actual['reicmcl'])[:,:3]
    expected=np.minimum(radius*np.float32(1+a.value),np.float32(140))
    np.testing.assert_allclose(ice,np.broadcast_to(expected[:,None],ice.shape),rtol=2e-7,atol=3e-5)
    assert np.all((ice>=5)&(ice<=140))
    for name in ['relqmcl','resnmcl','pdel','play','tlay']:
        np.testing.assert_array_equal(cp.asnumpy(actual[name]).view(np.uint32),reference[name].view(np.uint32))
    nanargs=device_kwargs(args);nanargs['re_ice'][0,0]=cp.nan
    assert bool(cp.isnan(call(**nanargs)['reicmcl'][0,0]))
    result.append({'side':'SW' if sw else 'LW','radius_um':ice[:,0].tolist(),
                   'untouched_fields_equal':True,'nan_propagates':True})
a.out.write_text(json.dumps({'value':a.value,'status':'PASS','results':result},indent=2)+'\n')
print('ICE_RADIUS_PASS',a.value,flush=True)
