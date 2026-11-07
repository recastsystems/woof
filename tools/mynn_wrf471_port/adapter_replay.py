#!/usr/bin/env python3
"""Grade the shipped Python pack/unpack against kept native outputs."""
import argparse
import json
from pathlib import Path
import numpy as np
import cupy as cp
from woof.core.mynn471_gpu import driver, schema

p=argparse.ArgumentParser()
p.add_argument('captures',type=Path)
p.add_argument('references',type=Path)
p.add_argument('--set',action='append',default=[])
p.add_argument('--out',type=Path,required=True)
p.add_argument('--all-slabs',action='store_true',help='Also compare every native output in the shipped work slabs')
a=p.parse_args();meta=schema(49)
fields={n:f for n,f in meta['arguments'].items() if f['type']!='character'}
def read(path):
 raw=path.read_bytes();offset=0;result={}
 for name,f in fields.items():
  count=f['count']
  if name=='chem':count=49*int(result['nchem'][0])
  if name=='vdep':count=int(result['ndvel'][0])
  dtype='>f4' if f['slab']=='f32' else '>i4'
  result[name]=np.frombuffer(raw,dtype=dtype,count=count,offset=offset).astype(dtype[1:])
  offset+=4*count
 assert offset==len(raw),(path,offset,len(raw))
 return result
paths=sorted(a.captures.glob('*.input.bin'));inputs=[read(p) for p in paths]
refs=[read(a.references/(p.stem+'.native.bin')) for p in paths]
overrides=dict(v.split('=',1) for v in a.set)
for row in inputs:
 for name,v in overrides.items():
  row[name][0]=(v.lower() in ('1','true','.true.') if fields[name]['type']=='logical' else float(v))
values={name:cp.asarray(np.stack([r[name][:f['count']] for r in inputs]))
        for name,f in fields.items()}
for name,f in fields.items():
 if not f['dims']:values[name]=values[name][:,0]
values['nchem'].fill(1);values['ndvel'].fill(1)
values['dz']=values['dz1']
values['rmol']=cp.zeros(len(paths),dtype=cp.float32)
class RetainedScratch:
 def __init__(self):self.arrays={}
 def one(self,name,shape):
  self.arrays[name]=cp.empty(shape,dtype=cp.float32)
  return self.arrays[name]
scratch=RetainedScratch() if a.all_slabs else None
out=driver(values,initflag=1,delt=12.,tke_budget=1,bl_mynn_output=1,scratch=scratch)
mapping={name:name+'1' for name in ('qke','qke_adv','el','sh','sm','tsq','qsq','cov','qc_bl','qi_bl','cldfra_bl','sqv')}
mapping.update(exch_h='kh1',exch_m='km1')
for n,k in [('du','rublten'),('dv','rvblten'),('dth','rthblten'),('dqv','rqvblten'),('dqc','rqcblten'),('dqi','rqiblten'),('dqs','rqsblten'),('dqnc','rqncblten'),('dqni','rqniblten'),('dqnwfa','rqnwfablten'),('dqnifa','rqnifablten'),('dqnbca','rqnbcablten'),('dozone','dozone')]:mapping[k]=n+'1'
for name in ('pblh','kpbl','maxwidth','maxmf','ztop_plume'):mapping[name]=name
for name in ('qshear','qbuoy','qdiss','qwt','dqke','edmf_a','edmf_w','edmf_thl','edmf_qt','edmf_ent','edmf_qc','sub_thl3d','sub_sqv3d','det_thl3d','det_sqv3d'):mapping[name]=name.replace('3d','')+'1'
report=dict(columns=len(paths),compared=0,differing=0,fields={})
for name,native in mapping.items():
 got=cp.asnumpy(out[name]).reshape(len(paths),-1)
 expected=np.stack([r[native] for r in refs])
 diff=got.view('u4')!=expected.view('u4')
 report['compared']+=diff.size;report['differing']+=int(diff.sum())
 report['fields'][name]=dict(words=diff.size,differing=int(diff.sum()))
 if diff.any():
  i=tuple(np.argwhere(diff)[0]);report['fields'][name]['first']=[int(x) for x in i]+[float(got[i]),float(expected[i])]
if a.all_slabs:
 raw=dict(compared=0,differing=0,fields={})
 for name,f in fields.items():
  if f['intent']=='in':continue
  slab=cp.asnumpy(scratch.arrays['mynn471_'+f['slab']])
  if f['slab']=='i32':slab=slab.view(np.int32)
  slab=slab.reshape(meta['slab_fields'][f['slab']],len(paths))
  got=np.ascontiguousarray(slab[f['offset']:f['offset']+f['count']].T)
  expected=np.stack([r[name] for r in refs])
  if name=='chem' and expected.shape[1]!=f['count']:
   assert all(not r['mix_chem'][0] for r in inputs)
   complete=np.stack([r[name] for r in inputs]);complete[:,:f['count']]=got;got=complete
  diff=got.view('u4')!=expected.view('u4')
  raw['compared']+=diff.size;raw['differing']+=int(diff.sum())
  raw['fields'][name]=dict(words=diff.size,differing=int(diff.sum()))
  if diff.any():
   i=tuple(np.argwhere(diff)[0]);raw['fields'][name]['first']=[int(x) for x in i]+[float(got[i]),float(expected[i])]
 report['all_slabs']=raw
a.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
raise SystemExit(bool(report['differing'] or report.get('all_slabs',{}).get('differing',0)))
