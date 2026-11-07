"""Identify scalar math divergences on operands reached by the column chain."""
import json,sys
from pathlib import Path
import cupy as cp
import numpy as np
from woof.core.kernels import load_module
rows=np.load(Path(__file__).with_name('math-operands.npz'))['rows']
a=cp.asarray(rows); b=cp.empty(rows.shape[0],dtype=cp.float32)
load_module('gsfcsw').get_function('gsfcsw_math_probe')(((len(rows)+127)//128,),(128,),(a,b,np.int32(len(rows))))
b=cp.asnumpy(b); result={}
for k,name in enumerate(('exp','log','pow','log10')):
    sel=rows[:,0]==k; bad=sel&(b.view('u4')!=rows[:,3].copy().view('u4'))
    inds=np.flatnonzero(bad)
    result[name]={'operands':int(sel.sum()),'mismatches':len(inds),'first':[{'a':float(rows[i,1]),'b':float(rows[i,2]),'expected':float(rows[i,3]),'actual':float(b[i]),'a_word':int(rows[i,1].view('u4')),'expected_word':int(rows[i,3].view('u4')),'actual_word':int(b[i].view('u4'))} for i in inds[:20]]}
Path(sys.argv[1]).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
