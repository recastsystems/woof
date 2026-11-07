"""Replay full native-domain captures, including split domain-reduction launches."""
from pathlib import Path
import json,hashlib,sys,argparse
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import cupy as cp
from woof.core.lightning import launch_lightning,next_activation
from woof.wrf_exact import ENABLED
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('captures',nargs='+',type=Path)
parser.add_argument('--out',type=Path,required=True)
parser.add_argument("--month",type=int,default=1,help="native call month (unused by admitted methods 0..3)")
args=parser.parse_args();receipts=[]
assert ENABLED
for p in sorted(args.captures):
 raw=p.read_bytes();off=0
 def read(dtype,shape=()):
  global off
  count=int(np.prod(shape));a=np.frombuffer(raw,dtype=dtype,count=count,offset=off).copy().reshape(shape);off+=count*np.dtype(dtype).itemsize;return a
 magic,nx,ny,nz,opt,method,cell=map(int,read('<i4',(7,)));assert magic in (47103,47104)
 if method==4 and opt!=3:raise ValueError("native Registry monthly IC:CG allocation is 2D; method 4 reads outside it")
 dt,dx,dy,interval,factor,num,den,adjust=map(float,read('<f4',(8,)))
 seconds=float(read('<f8'));activation=float(read('<f4'))
 fields={k:read('<i4' if k=='top' else '<f4',(ny,nx)) for k in ('lat','lon','land','ht','top')}
 for k in ('t','z','w','refl','dz','theta','pi','qv','qc','qr','qi','qs','qg'):
  fields[k]=np.ascontiguousarray(read('<f4',(ny,nz,nx)).transpose(1,0,2))
 for k in ('inputnum','inputden'):
  a=read('<f4',(12,ny,nx) if magic==47103 else (ny,nx))
  fields[k]=a if magic==47103 else np.broadcast_to(a,(12,ny,nx)).copy()
 names=('ic_flashrate','cg_flashrate','ic_flashcount','cg_flashcount','lpi')
 before={k:read('<f4',(ny,nx)) for k in names};after={k:read('<f4',(ny,nx)) for k in names};post=float(read('<f4'));assert off==len(raw)
 actual={k:cp.asarray(v) for k,v in before.items()}
 gpu={k:cp.asarray(v) for k,v in fields.items()}
 tiles=None if cell!=2 else [(0,nx//2,0,ny),(nx//2,nx,0,ny)]
 launch_lightning(gpu,actual,option=opt,method=method,cellcount_method=cell,dx=dx,dy=dy,interval=interval,factor=factor,num=num,den=den,adjust=adjust,month=args.month,tiles=tiles)
 rows={k:{'words':v.size,'different_words':int(np.count_nonzero(cp.asnumpy(actual[k]).view('u4')!=v.view('u4'))),'native_nonzero':int(np.count_nonzero(v)),'maximum':float(np.nanmax(v))} for k,v in after.items()}
 clock=next_activation(activation,seconds,dt,interval);assert np.float32(clock).view('u4')==np.float32(post).view('u4')
 receipts.append({'file':p.name,'option':opt,'shape':[nz,ny,nx],'seconds':seconds,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'cellcount_method':cell,'gpu_tiles':tiles,'words':sum(x['words'] for x in rows.values())+1,'different_words':sum(x['different_words'] for x in rows.values()),'fields':rows,'clock':post,'top_range':[int(fields['top'].min()),int(fields['top'].max())]})
 print(p.name,receipts[-1]['different_words'],flush=True)
summary={'strict':True,'captures':len(receipts),'words':sum(x['words'] for x in receipts),'different_words':sum(x['different_words'] for x in receipts),'rows':receipts}
args.out.write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='rows'}));assert summary['captures'] and not summary['different_words']
