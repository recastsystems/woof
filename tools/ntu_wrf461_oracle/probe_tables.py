"""Read NTU's initialized device tables as bit words for the oracle check."""
import json,sys
from pathlib import Path
import cupy as cp
import numpy as np
from woof.core import ntu
fields=[('IBAER','<i4',2),('TBLRC','<f8',25),('TBLXF','<f8',50)]+[(n,'<f4',1) for n in ('DNC0','DNC1','DNC2','DNC3')]+[(n,'<f4',2) for n in ('SENS','ASH','CMAS','DNAS','AVAN','BETA1','RXMIN')]+[(n,'<f4',6) for n in ('ZCCN','CNMOD','CNSTD','WMAS','RFACT')]
result={}
for cc in range(1,5):
 raw=cp.asnumpy(ntu.aero_constants(cp.cuda.Device().id,cc)).tobytes(); off=0; rows={}
 for n,d,c in fields:
  value=np.frombuffer(raw,dtype=d,count=c,offset=off); off+=value.nbytes; rows[n]={'dtype':d,'bits':[int(x) for x in value.view('<u8' if d=='<f8' else '<u4')]}
 result[cc]=rows
Path(sys.argv[1]).write_text(json.dumps(result,indent=2))
frames={}
for n in ('ntu_aero_const_kernel','ntu_init_kernel','ntu_mp_kernel','ntu_refl_kernel'):
 fn=ntu._kernel(n,64); frames[n]={'local_bytes':cp.cuda.driver.funcGetAttribute(3,fn.ptr),'registers':cp.cuda.driver.funcGetAttribute(4,fn.ptr)}
Path(sys.argv[1]+'.frames.json').write_text(json.dumps(frames,indent=2))
print(frames)
