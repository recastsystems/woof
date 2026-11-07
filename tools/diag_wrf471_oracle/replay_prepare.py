#!/usr/bin/env python3
"""Strict device replay of bounded phy_prep captures from native WRF."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
def pack_below_400hpa(p8w):
 """Native p8w (nz+1, ny, nx) as the resident leaf's bit words ([word][column])."""
 bits=np.asarray(p8w)>np.float32(40000.);words=np.zeros(((bits.shape[0]+31)//32,*bits.shape[1:]),dtype='u4')
 for k in range(bits.shape[0]):words[k>>5]|=bits[k].astype('u4')<<np.uint32(k&31)
 return words
def main():
 p=argparse.ArgumentParser();p.add_argument('captures',nargs='+',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--print-surface',action='store_true');a=p.parse_args()
 import cupy as cp
 from woof.wrf_exact import ENABLED
 from woof.core.kernels import get_kernel,module_source
 if not ENABLED:raise ValueError('requires GPUWM_WRF_EXACT=1')
 rows=[]
 for path in a.captures:
  raw=path.read_bytes();h=np.frombuffer(raw,dtype='<i4',count=4);magic,nx,ny,nz=map(int,h)
  if magic!=47103:raise ValueError('bad capture magic')
  offset=16
  def take(shape):
   nonlocal offset
   n=int(np.prod(shape));b=np.frombuffer(raw,dtype='<f4',count=n,offset=offset).reshape(shape,order='F');offset+=n*4
   return np.ascontiguousarray(b.transpose(1,2,0) if len(shape)==3 else b)
  data={key:take((nx,nz,ny)) for key in ('p','pb','alt','qv')}
  data.update({key:take((nx,nz+1,ny)) for key in ('ph','phb')})
  data.update({key:take((nz+1,)) for key in ('fnm','fnp')})
  expected={'rho':take((nx,nz,ny)),'p8w':take((nx,nz+1,ny))}
  if offset!=len(raw):raise ValueError('trailing bytes')
  d={k:cp.asarray(v) for k,v in data.items()};d['p']+=d['pb']
  outputs={k:cp.empty_like(cp.asarray(v)) for k,v in expected.items()};phb=cp.empty_like(d['phb'])
  get_kernel('nwp_windows','nwp_prepare')(((nx*ny+127)//128,),(128,),
    (*(d[k] for k in ('p','alt','qv','ph','phb','fnm','fnp')),outputs['rho'],outputs['p8w'],phb,np.int32(nz),np.int32(nx*ny),np.int32(1)))
  if a.print_surface:
   outputs={'print_surface':cp.empty((ny,nx),dtype='f4')}
   expected={'print_surface':expected['p8w'][0]}
   get_kernel('misc_diag_print','misc_print_surface_pressure')(((nx*ny+255)//256,),(256,),
    (d['p'],d['ph'],d['phb'],outputs['print_surface'],np.int32(nx*ny),np.int32(1)))
  if not a.print_surface:
   # The resident driver's leaf on the same inputs: density, the windows'
   # "p8w > 400 hPa" bits and the surface interface plane, against native.
   words=(nz+1+31)//32;rr=cp.empty_like(outputs['rho']);below=cp.empty((words,ny,nx),dtype='u4');surface=cp.empty((ny,nx),dtype='f4')
   get_kernel('nwp_windows','nwp_prepare_resident')(((nx*ny+127)//128,),(128,),
     (d['p'],d['alt'],d['qv'],np.int32(1),d['ph'],d['phb'],np.int32(1),d['fnm'],d['fnp'],rr,below,surface,np.int32(1),np.int32(nz),np.int32(nx*ny)))
   outputs.update(resident_rho=rr,resident_below_400hpa=below,resident_surface_pw=surface)
   expected.update(resident_rho=expected['rho'],resident_below_400hpa=pack_below_400hpa(expected['p8w']),resident_surface_pw=expected['p8w'][0])
  fields={}
  for key,want in expected.items():
   got=cp.asnumpy(outputs[key]);diff=got.view('u4')!=want.view('u4');fields[key]={'words':int(want.size),'differing_words':int(diff.sum())}
   if diff.any():
    ix=tuple(map(int,np.argwhere(diff)[0]));fields[key]['first_difference']={'index':ix,'native':float(want[ix]),'cuda':float(got[ix]),'native_word':int(want.view('u4')[ix]),'cuda_word':int(got.view('u4')[ix])}
  rows.append({'capture':str(path),'capture_sha256':hashlib.sha256(raw).hexdigest(),'fields':fields})
 result={'kind':'strict native phy_prep real-input module replay','module_sha256':hashlib.sha256(module_source('misc_diag_print' if a.print_surface else 'nwp_windows').encode()).hexdigest(),'results':rows}
 a.output.write_text(json.dumps(result,indent=2)+'\n');words=sum(f['words'] for r in rows for f in r['fields'].values());diff=sum(f['differing_words'] for r in rows for f in r['fields'].values());print(json.dumps({'words':words,'differing_words':diff}))
 if diff:raise SystemExit(1)
if __name__=='__main__':main()
