from woof.wrf_exact import ENABLED
if not ENABLED:
 raise RuntimeError("replay requires GPUWM_WRF_EXACT=1 before importing woof")
import sys
"""Replay private cloud merges captured around real WRF driver calls."""
from pathlib import Path
import hashlib,json
import numpy as np
import cupy as cp
from woof.core.kernels import get_kernel,module_source
root=Path(sys.argv[1]).resolve()
rows=[]
for scheme in (1,3):
 total=diff=active=inputs_changed=0;digests=[];columns=0
 for path in sorted((root/f'capture-cu{scheme}').glob('*.bin')):
  data=path.read_bytes(); mode,n=np.frombuffer(data[:8],dtype='<i4');v=np.frombuffer(data[8:],dtype='<f4')
  if v.size!=10*n: raise ValueError((path,v.size,n))
  a=v[:7*n].reshape(7,n).copy();expected=v[7*n:].reshape(3,n)
  before=a.copy();dev=cp.asarray(a);out=dev[[0,1,6]].copy();cfcu=cp.empty(n,cp.float32)
  get_kernel('cu_radiation','cu_radiation_merge')(((int(n)+127)//128,),(128,),
   (np.int64(n),np.int32(mode),np.int32(1),np.int32(1),dev[2],dev[3],dev[4],dev[5],out[0],out[1],out[2],cfcu))
  actual=out.get();diff+=int(np.count_nonzero(actual.view('u4')!=expected.view('u4')));total+=actual.size
  inputs_changed+=int(np.count_nonzero(dev.get().view('u4')!=before.view('u4')))
  active+=int(np.count_nonzero(a[2:6]));columns+=1;digests.append({'file':path.name,'sha256':hashlib.sha256(data).hexdigest()})
 rows.append(dict(cu_physics=scheme,icloud_cu=int(mode),columns=columns,levels=int(n),output_words=total,differing_words=diff,input_words_changed=inputs_changed,nonzero_feedback_input_words=active,inputs=digests))
 if diff or inputs_changed or active==0: raise RuntimeError(rows[-1])
result=dict(numerics='strict fmad=false ftz=false; rounded float32 operations in native statement order',module_sha256=hashlib.sha256(module_source('cu_radiation').encode()).hexdigest(),rows=rows)
(root/'evidence/resume/cu-replay.json').write_text(json.dumps(result,indent=2));print(json.dumps({**result,'rows':[{k:v for k,v in x.items() if k!='inputs'} for x in rows]},indent=2))
