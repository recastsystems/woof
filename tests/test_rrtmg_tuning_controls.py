"""Exercise the formerly licence-held RRTMG controls on the actual device.

Fresh processes are required: production constants are immutable after compile.
The overlap oracle uses the WRF-gated NumPy generator's two physical endpoints.
"""
import os
from pathlib import Path
import subprocess
import sys
import pytest

cp = pytest.importorskip('cupy')
ROOT = Path(__file__).resolve().parents[1]

CHECK = r'''
import sys,json
import numpy as np
import cupy as cp
from woof import physics_params as pp
from woof.knobs import physics
from woof.core import rrtmg_mcica as m
from test_rrtmg_mcica import _load,_fixture_kwargs,FIXTURE_DIR
fraction=float(sys.argv[1])
physics.activate({'physics.rrtmg.overlap_break_fraction':fraction})
words=0
for side in ('lw','sw'):
 data=_load(FIXTURE_DIR / ('mcica_'+side+'01.npz'))
 _,kw=_fixture_kwargs(data)
 host=getattr(m,'generate_'+side+'_subcolumns')
 gpu=getattr(m,'gpu_generate_'+side+'_subcolumns')
 maximum=host(**kw)
 random=host(**(kw|{'icld':1}))
 # Selection is constant along height; use Python integers to model the
 # documented modulo-2**32 selector independently of CUDA arithmetic.
 pick=np.empty((getattr(m,'NGPT'+side.upper()),kw['ncol'],1),bool)
 for g in range(pick.shape[0]):
  for c in range(pick.shape[1]):
   x=((c*0x9e3779b9)^(g*0x85ebca6b))&0xffffffff
   x=((x^(x>>16))*0x7feb352d)&0xffffffff
   x=((x^(x>>15))*0x846ca68b)&0xffffffff
   x=x^(x>>16)
   pick[g,c,0]=(x>>8)/16777216 < fraction
 for layout,fused,chunk in [('gpoint',False,3),('column',True,3),('column',True,None)]:
  got=gpu(**kw,layout=layout,fused=fused,ncol_chunk=chunk)
  for name,a in got.items():
   want=np.where(pick,random[name],maximum[name]) if a.ndim==3 else maximum[name]
   if a.ndim==3 and layout=='column':want=want.transpose(1,2,0)
   actual=cp.asnumpy(a)
   assert np.array_equal(actual.view(np.uint32),want.view(np.uint32)),(side,fraction,layout,name,int(np.sum(actual!=want)))
   words+=actual.size
print(json.dumps({'fraction':fraction,'compared_words':words,'differing_words':0}))
'''

@pytest.mark.parametrize('fraction',[0,.25,1])
def test_overlap_matches_native_gated_endpoints_in_both_layouts(fraction):
 env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'tests')+os.pathsep+env.get('PYTHONPATH','')
 result=subprocess.run([sys.executable,'-B','-c',CHECK,str(fraction)],env=env,capture_output=True,text=True)
 assert result.returncode==0,result.stdout+result.stderr
 print(result.stdout.strip())

RADIUS_CHECK = r'''
import sys,json
import numpy as np
import cupy as cp
from woof.knobs import physics
from woof.core import rrtmg_legacy_device as dev
from woof.core import rrtmg_legacy_prep as ref
from test_rrtmg_legacy_prep_device import synthetic,side_kwargs,device_kwargs
control=sys.argv[1];fraction=float(sys.argv[2])
physics.activate({'physics.rrtmg.radius_'+control:fraction})
# Independently rescale physical input radii for the WRF-gated CPU wrapper.
# Its multiply-before-unit-conversion order can differ by rounding from
# the production control's multiply-after-conversion order, hence 2e-6.
kw=synthetic(7,nz=12)
kw['re_cloud'].fill(np.float32(12e-6));kw['re_ice'].fill(np.float32(50e-6));kw['re_snow'].fill(np.float32(80e-6))
# Exercise explicit radii and the ice-to-snow fallback separately.
words=0;max_relative=0
for fallback in (False,True):
 if fallback:kw['has_reqs']=0
 for sw in (False,True):
  base=side_kwargs(kw,sw);manual=base.copy()
  name={'liquid':'re_cloud','ice':'re_ice','snow':'re_snow'}[control]
  if fallback and control=='snow':name='re_ice'
  # Ice tuning does not tune the snow fallback, which has its own control.
  if not (fallback and control=='ice'):
   manual[name]=(manual[name]*np.float32(1+fraction)).astype(np.float32)
  expected=(ref.swrad_prep_batch if sw else ref.lwrad_prep_batch)(**manual)
  got=(dev.swrad_prep_batch_device if sw else dev.lwrad_prep_batch_device)(**device_kwargs(base))
  for name in ('reicmcl','relqmcl','resnmcl','ciwpmcl','clwpmcl','cswpmcl'):
   want=expected[name]
   if want.ndim==3:want=want.transpose(1,2,0)
   a=cp.asnumpy(got[name]);np.testing.assert_allclose(a,want,rtol=2e-6,atol=0,err_msg=(control,str(fraction),str(fallback),name))
   words+=a.size
   max_relative=max(max_relative,float(np.max(np.abs(a-want)/np.maximum(np.abs(want),np.float32(1e-30)))))
print(json.dumps({'control':control,'fraction':fraction,'compared_words':words,'maximum_relative_error':max_relative}))
'''

@pytest.mark.parametrize('control',['liquid','ice','snow'])
@pytest.mark.parametrize('fraction',[-.2,.01,.2])
def test_radius_controls_match_rescaled_native_gated_wrapper(control,fraction):
 env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'tests')+os.pathsep+env.get('PYTHONPATH','')
 result=subprocess.run([sys.executable,'-B','-c',RADIUS_CHECK,control,str(fraction)],env=env,capture_output=True,text=True)
 assert result.returncode==0,result.stdout+result.stderr
 print(result.stdout.strip())
