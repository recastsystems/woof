from woof.wrf_exact import ENABLED
if not ENABLED:
 raise RuntimeError("replay requires GPUWM_WRF_EXACT=1 before importing woof")
import sys
from pathlib import Path
import json,hashlib,numpy as np,cupy as cp
from woof.core.kernels import get_kernel,module_source
root=Path(sys.argv[1]).resolve();rows=[]
k=get_kernel('solar_eclipse','solar_eclipse')
for f in sorted((root/'capture-eclipse').glob('*.bin')):
 data=f.read_bytes();n=int(np.frombuffer(data[:4],'<i4')[0]);v=np.frombuffer(data[4:],'<f4');assert len(v)==34+4*n,(f,len(v),n)
 h=v[:34];row=np.r_[2024,99,h[2:32]].astype('f4');lat=cp.asarray(v[34:34+n]);lon=cp.asarray(v[34+n:34+2*n]);out=cp.empty(n,cp.float32);mask=cp.empty(n,cp.int32);track=cp.empty(2,cp.float32)
 k(((n+255)//256,),(256,),(np.int32(n),cp.asarray(row),np.float32(h[0]),lat,lon,out,mask,track[:1],track[1:]))
 want=(v[34+2*n:34+3*n],v[34+3*n:].view('i4'),h[32:34]);actual=(out.get(),mask.get(),track.get());result={}
 for name,a,b in zip(('obscuration','mask','track'),actual,want):
  d=a.view('u4')!=b.view('u4');ix=np.flatnonzero(d);result[name]={'words':a.size,'differing_words':int(d.sum()),'examples':[{'index':int(i),'gpu':float(a[i]),'native':float(b[i]),'ulp':int(abs(int(a.view('u4')[i])-int(b.view('u4')[i])))} for i in ix[:4]]}
 rows.append({'file':f.name,'sha256':hashlib.sha256(data).hexdigest(),'hour':float(h[0]),'nonzero_obscuration':int(np.count_nonzero(want[0])),'results':result})
result={'source_sha256':hashlib.sha256(module_source('solar_eclipse').encode()).hexdigest(),'captures':rows,'words':sum(z['words'] for r in rows for z in r['results'].values()),'differing_words':sum(z['differing_words'] for r in rows for z in r['results'].values())}
(root/'evidence/resume/eclipse-replay.json').write_text(json.dumps(result,indent=2));print({k:v for k,v in result.items() if k!='captures'});print([r for r in rows if any(z['differing_words'] for z in r['results'].values())][:2]);assert rows and result['differing_words']==0
