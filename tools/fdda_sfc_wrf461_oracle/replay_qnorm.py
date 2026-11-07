from pathlib import Path
import json,hashlib,argparse
parser=argparse.ArgumentParser(description="Replay native YSU QNORM into the production CUDA kernel.")
parser.add_argument("capture",type=Path);parser.add_argument("--report",type=Path,required=True)
args=parser.parse_args()
import numpy as np
import cupy as cp
r=Path(__file__).resolve().parents[2];p=args.capture;raw=p.read_bytes();assert raw[:8]==b'QNORM001'
nx,ny=np.frombuffer(raw,'<i4',count=2,offset=8);dt=np.frombuffer(raw,'<f4',count=1,offset=16)[0];assert len(raw)==20+int(nx)*int(ny)*12
v=np.frombuffer(raw,'<f4',offset=20).reshape(3,int(ny),int(nx));dq,qv=map(cp.asarray,v[:2]);out=cp.empty_like(qv)
s=(r/'woof/spectral_nudging/surface_forecast.cu').read_text();m=cp.RawModule(code=s,options=('--std=c++17','--fmad=false'));n=int(nx)*int(ny)
m.get_function('sf_qnorm')(((n+127)//128,),(128,),(dq,qv,out,np.float32(dt),np.int32(n)))
got=out.get();d=int(np.count_nonzero(got.view('u4')!=v[2].view('u4')))
a=dict(words=n,differing_words=d,nonzero=int(np.count_nonzero(v[2])),dt=float(dt),capture_sha256=hashlib.sha256(raw).hexdigest(),shape=[int(ny),int(nx)],min=float(got.min()),max=float(got.max()))
a['differing_words_exact']=[{'index':idx.tolist(),'native':hex(int(v[2].view('u4')[tuple(idx)])),'cuda':hex(int(got.view('u4')[tuple(idx)]))} for idx in np.argwhere(got.view('u4')!=v[2].view('u4'))]
args.report.write_text(json.dumps(a,indent=2)+'\n');print(json.dumps(a));assert d==0
