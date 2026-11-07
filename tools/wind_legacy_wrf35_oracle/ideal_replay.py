#!/usr/bin/env python3
"""Check native ideal initialization and drag using a real forecast capture."""
from pathlib import Path
from types import SimpleNamespace
import argparse,hashlib,json,subprocess,shutil
import numpy as np
from replay import load_capture
p=argparse.ArgumentParser(description=__doc__);p.add_argument('capture',type=Path);p.add_argument('cache',type=Path);p.add_argument('scratch',type=Path)
a=p.parse_args();here=Path(__file__).resolve().parent;pin=json.loads((here/'native-pin.json').read_text())
for r in pin['sources']:
 assert hashlib.sha256((a.cache/r['path']).read_bytes()).hexdigest()==r['sha256'],r['path']
scratch=a.scratch.resolve();scratch.mkdir(exist_ok=False)
(scratch/'constants.f90').write_text('module module_driver_constants\ninteger,parameter::max_domains=21\nend module\nmodule module_model_constants\nreal,parameter::piconst=3.14159265358979323846,g=9.81\nend module\n')
shutil.copy2(a.capture/'legacy-native.in',scratch/'legacy-native.in')
cmd=['gfortran','-cpp','-O2','-fno-fast-math','-ffp-contract=off','-fno-tree-vectorize','-fconvert=big-endian','-ffree-form','-ffree-line-length-none','constants.f90',str((a.cache/'phys/module_wind_generic.F').resolve()),str((a.cache/'phys/module_wind_fitch.F').resolve()),str(here/'ideal_driver.f90'),'-o','ideal-driver']
subprocess.run(cmd,cwd=scratch,check=True);subprocess.run(['./ideal-driver'],cwd=scratch,check=True)
import cupy as cp
from woof.core.wind_legacy import LegacyFarm
from woof.wind_legacy import DEFAULTS
from woof.wrf_exact import ENABLED
assert ENABLED,'strict proof requires GPUWM_WRF_EXACT=1'
x=load_capture(a.capture);nz,ny,nx=x['shape'];cfg=SimpleNamespace(windturbines_spec='ideal',grid_id=1,dx=float(x['meta'][0]),dt=float(x['meta'][1]),bl_mynn_tkeadvect=True,**(DEFAULTS|dict(td_turbgridid=1,td_ewfx=2,td_ewfy=2,td_pwfx=18,td_pwfy=22,td_turbpercell=2.5)))
f=LegacyFarm(cfg,x['shape'],phb=x['phb']);gpu=lambda v:cp.asarray(np.ascontiguousarray(v));fields={k:gpu(x[k]) for k in ('qke','qke_adv')};rates={k:gpu(x[k]) for k in ('du','dv')}
f.apply_mynn({k:gpu(x[k]) for k in ('u','v','dz')},fields,rates)
out=(scratch/'ideal-native.out').read_bytes();words=np.frombuffer(out,dtype='>f4').astype('f4');outputs={}
for i,k in enumerate(('du','dv','qke','qke_adv')):
 expected=words[i*nz*ny*nx:(i+1)*nz*ny*nx].reshape(ny,nz,nx).transpose(1,0,2).copy();actual=cp.asnumpy((rates|fields)[k]);outputs[k]=dict(words=actual.size,different=int(np.count_nonzero(actual.view('u4')!=expected.view('u4'))),maximum_absolute=float(np.max(abs(actual-expected))))
raw=(scratch/'ideal-native.geometry').read_bytes();nt=int(np.frombuffer(raw,dtype='>i4',count=1)[0]);rows=np.frombuffer(raw[4:],dtype='>f4').astype('f4').reshape(nt,11);cols=((rows[:,4]-1)*nx+rows[:,3]-1).astype('i4');geometry_differences=int(np.count_nonzero(cols!=cp.asnumpy(f.col)));parameter_differences=int(np.count_nonzero(rows[:,5:].copy().view('u4')!=cp.asnumpy(f.params).view('u4')))
r=dict(schema='wind-legacy-ideal-replay.v1',command=cmd,numerics='strict',weather_input_sha256=x['input_sha256'],native_output_sha256=hashlib.sha256(out).hexdigest(),native_turbines=nt,coordinate_words=nt,coordinate_differences=geometry_differences,parameter_words=nt*6,parameter_differences=parameter_differences,outputs=outputs,production_receipt=f.receipt())
print(json.dumps(r,indent=2));assert not geometry_differences and not parameter_differences and all(v['different']==0 for v in outputs.values())
