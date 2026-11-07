from pathlib import Path
import argparse
parser=argparse.ArgumentParser(description='Replay unchanged native cloud producers on real captured profiles')
parser.add_argument('--capture',type=Path,required=True)
parser.add_argument('--library',type=Path,required=True)
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args()
import ctypes as ct,json,hashlib
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import cupy as cp
import numpy as np
from woof.core.kernels import get_kernel
from woof.wrf_exact import ENABLED
lib=ct.CDLL(str(args.library.resolve()));fn=lib.run
ptr=np.ctypeslib.ndpointer(np.float32,flags='C_CONTIGUOUS')
fn.argtypes=[ct.c_int,*([ptr]*7),ct.c_int,ct.c_int,ct.c_int,ct.c_int,ct.c_int]
# These are actual full native columns at the first active lightning call.
p=args.capture;raw=p.read_bytes();magic,nx,ny,nz,*_=np.frombuffer(raw,'<i4',7);off=28+32+8+4
for k in range(5):off+=nx*ny*4
f={}
for k in ('t','z','w','refl','dz','theta','pi','qv','qc','qr','qi','qs','qg'):
 f[k]=np.ascontiguousarray(np.frombuffer(raw,'<f4',nx*ny*nz,off).reshape(ny,nz,nx)[:-1,:-1,:-1]).reshape(-1);off+=nx*ny*nz*4
# Native pi is Exner; turn it into a physical pressure for this leaf call.
f['p']=np.float32(100000)*np.power(f['pi'],np.float32(1004./287.)).astype('f4')
rows=[]
for fi,fs,mp,fc,simple in ((1,1,8,1,0),(1,0,50,1,0),(1,0,5,1,0),(0,0,1,1,0),(1,1,8,1,1),(0,0,1,1,1),(0,0,0,0,1)):
 arrays=[np.ascontiguousarray(f[k],'f4') for k in ('qv','qc','qi','qs','t','p')]
 native=np.zeros_like(arrays[0]);fn(native.size,*arrays,native,fi,fs,mp,fc,simple)
 dev=[cp.asarray(v) for v in arrays];out=cp.zeros_like(dev[0]);n=out.size
 get_kernel('native_surface_diagnostics','solar_cloud_fraction')(((n+127)//128,),(128,),(*dev,out,np.int32(n),np.int32(fc),np.int32(fi),np.int32(fs),np.int32(simple),np.int32(mp in (5,15))))
 actual=cp.asnumpy(out);bad=int(np.count_nonzero(actual.view('u4')!=native.view('u4')))
 rows.append(dict(mp=mp,cloud_active=fc,simple=simple,words=n,different_words=bad,native_nonzero=int(np.count_nonzero(native)),max_abs=float(abs(actual-native).max())))
s=dict(strict=ENABLED,source='public WRF module_radiation_driver.F cal_cldfra1/cal_cldfra2 unchanged; actual native T/q profiles and pressure derived from captured Exner',library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),capture_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),words=sum(x['words'] for x in rows),different_words=sum(x['different_words'] for x in rows),rows=rows)
args.out.write_text(json.dumps(s,indent=2)+'\n');print(s);assert ENABLED and not s['different_words']
