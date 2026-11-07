from woof.wrf_exact import ENABLED
if not ENABLED:
 raise RuntimeError("replay requires GPUWM_WRF_EXACT=1 before importing woof")
import sys
from pathlib import Path
import hashlib,json
import numpy as np
import cupy as cp
from woof.core.wrf_stock_aer3 import optics_device
from woof.core.kernels import module_source, module_options
from woof.wrf_exact import effective_options
root=Path(sys.argv[1]).resolve();files=sorted((root/'capture-aer3').glob('*.bin'))
assert files
p=[]; inputs=[[] for _ in range(5)];want=[[] for _ in range(5)];hashes=[]
for f in files:
 b=f.read_bytes();n=int(np.frombuffer(b[:4],'<i4')[0]);v=np.frombuffer(b[4:],'<f4');assert len(v)==49*n+5
 p.append(v[:n+1]);j=n+1
 for a in inputs:a.append(v[j:j+n]);j+=n
 want[0].append(v[j:j+n]);j+=n
 for a in want[1:4]:a.append(v[j:j+14*n].reshape(14,n));j+=14*n
 want[4].append(v[j:j+4]);hashes.append({'file':f.name,'sha256':hashlib.sha256(b).hexdigest()})
pressure=cp.asarray(np.array(p));args=[cp.asarray(np.array(x)) for x in inputs]
opt,aod,diag=optics_device(pressure[:,:-1],*args,pressure_halo=pressure[:,-1]);actual=[aod.get(),*(a[:,:,:n].get() for a in opt),diag.get().T]
# Native order tau,asy,ssa and surface SSA,ASY,TAOD,ANG.
names=['TAOD5503D','TAUAER','ASYAER','SSAAER','diagnostics'];rows={}
for name,got,w in zip(names,actual,want):
 expected=np.array(w);diff=got.view('u4')!=expected.view('u4');ix=np.argwhere(diff)
 rows[name]={'words':got.size,'differing_words':int(diff.sum()),'max_abs':float(np.nanmax(abs(got-expected))),'examples':[{'index':i.tolist(),'gpu':float(got[tuple(i)]),'native':float(expected[tuple(i)])} for i in ix[:5]]}
result={'columns':len(files),'levels':n,'pressure_halo_words':sorted(set(float(x[-1]) for x in p)), 'results':rows,'inputs':hashes,
 'source_sha256':hashlib.sha256(module_source('wrf_stock_aer3').encode()).hexdigest(),
 'effective_options':effective_options(module_options('wrf_stock_aer3'))}
(root/'evidence/resume/aer3-replay.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='inputs'},indent=2))
assert all(x['differing_words']==0 for x in rows.values())
