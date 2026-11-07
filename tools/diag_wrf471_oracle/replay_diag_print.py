from pathlib import Path
import sys,json,hashlib,re
import numpy as np,cupy as cp
from woof.core.kernels import get_kernel
from woof import wrf_exact
assert wrf_exact.ENABLED, "Native replay requires GPUWM_WRF_EXACT=1"
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--captures',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--native-rsl',type=Path);cli=parser.parse_args()
r=cli.captures.resolve().parent.parent;rows=[];global_mu={}
for p in sorted(cli.captures.resolve().glob('print-*.bin')):
 raw=p.read_bytes();magic,nx,ny,mode,step=np.frombuffer(raw,dtype='<i4',count=5);assert magic==47116
 dt,xtime=np.frombuffer(raw,dtype='<f4',count=2,offset=20);n=int(nx*ny)
 a=np.frombuffer(raw,dtype='<f4',count=11*n,offset=28).copy().reshape(11,n);expected=np.frombuffer(raw,dtype='<f4',offset=28+44*n)
 d=cp.asarray(a);dps=cp.empty(n,dtype='f4');dmu=cp.empty(n,dtype='f4');out=cp.empty(14,dtype='f4')
 get_kernel('misc_diag_print','misc_print_fields')(((n+255)//256,),(256,),
  (d[0],d[2],d[1],d[3],dps,dmu,dt,np.int32(xtime==0),np.int32(n)))
 if xtime==0:continue
 assert expected.size==13
 get_kernel('misc_diag_print','misc_print_native_order')((1,),(32,),
  (dps,dmu,*[d[i] for i in range(4,11)],out,dt,np.int32(n)))
 all_actual=cp.asnumpy(out);actual=all_actual[:13];global_mu.setdefault(float(xtime),[]).append(float(all_actual[13]));bad=actual.view('u4')!=expected.view('u4')
 row=dict(path=str(p.relative_to(r)),sha256=hashlib.sha256(raw).hexdigest(),step=int(step),xtime=float(xtime),words=13,differing_words=int(bad.sum()),differences=[dict(index=int(i),gpu=float(actual[i]),native=float(expected[i])) for i in np.flatnonzero(bad)])
 rows.append(row);print(row,flush=True)
 result=dict(status='PASS' if rows and all(x['differing_words']==0 for x in rows) else 'FAIL',words=sum(x['words'] for x in rows),captures=rows,scope='All thirteen actual native local diagnostic-output reductions; strict oracle preserves native FP32 association, default forecast uses GPU parallel trees')
if cli.native_rsl:
 native={};time=None
 for line in cli.native_rsl.read_text().splitlines():
  if 'Domain average of dpsdt, dmudt (mb/3h):' in line:time=float(line.split(':',1)[1].split()[0])
  elif 'Max mu change time step:' in line and time is not None:native[time]=float(line.split(':',1)[1].split()[-1])
 global_rows=[]
 for time,values in sorted(global_mu.items()):
  key=min(native,key=lambda x:abs(time-x));assert abs(key-time)<0.001,(time,key)
  actual=np.float32(max(values));expected=np.float32(native[key]);diff=int(actual.view('u4')!=expected.view('u4'))
  global_rows.append(dict(xtime=float(time),patches=len(values),gpu=float(actual),native=float(expected),differing_words=diff))
 result['global_mu_max']=global_rows;result['words']+=len(global_rows)
 result['native_rsl']=dict(path=str(cli.native_rsl),sha256=hashlib.sha256(cli.native_rsl.read_bytes()).hexdigest())
 if any(x['differing_words'] for x in global_rows):result['status']='FAIL'
 result['scope']+='; global maximum mass-change word reconstructed from nine-significant-digit native output and all actual patch maxima'
cli.output.write_text(json.dumps(result,indent=2)+'\n');assert result['status']=='PASS'
