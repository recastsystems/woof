"""Measure all oracle fields and mutation sensitivity on the assigned card."""
import argparse
import hashlib
import json
import struct
from pathlib import Path
import numpy as np
from tests.test_fogdes_wrf461_parity import CASES, _case, _ulp, run_port
from woof.core import fog_settling as fog
from woof.certify.compile_platform import compile_platform_fingerprint
from woof.certify.kernel_manifest import kernel_manifest
from woof.wrf_exact import ENABLED

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True); args=p.parse_args()
receipt={'strict':bool(ENABLED),'platform':compile_platform_fingerprint(),'cases':{},'mutations':{},'tolerance':{'vdfg':0,'fgdp':0,'dfgdp':0,'rqcblten':0}}
for name in CASES:
 c=_case(name); out=run_port(c); fields={}
 for field,value in out.items():
  ref=c[field] if field!='rqcblten' else c[field][:,:int(c['kte']),:]
  delta=_ulp(value,ref)
  fields[field]={'max_ulp':int(delta.max()),'different_words':int((delta>0).sum()),'max_abs':float(np.max(np.abs(value-ref))),'sha256':hashlib.sha256(value.tobytes()).hexdigest(),'wrf_sha256':hashlib.sha256(np.ascontiguousarray(ref).tobytes()).hexdigest(),'words':int(value.size)}
  assert fields[field]['max_ulp']==0,(name,field,fields[field])
 receipt['cases'][name]={'columns':int(c['vdfg'].shape[0]),'steps':int(c['vdfg'].shape[1]),'fields':fields}
for constant in ('17.03f','0.0164f','1.6666666f'):
 value=float(constant[:-1]); bits=struct.unpack('<I',struct.pack('<f',value))[0]^1; new=struct.unpack('<f',struct.pack('<I',bits))[0]
 module=fog.compile_module(fog.module_source().replace(constant,float(new).hex()+'f'))
 misses=0
 for name in CASES:
  c=_case(name)
  for field,value in run_port(c,module=module).items():
   ref=c[field] if field!='rqcblten' else c[field][:,:int(c['kte']),:]
   misses+=int((_ulp(value,ref)>0).sum())
 assert misses>0,constant
 receipt['mutations'][constant]={'bit_xor':1,'different_words':misses}
receipt['kernel_manifest']=kernel_manifest()
args.output.write_text(json.dumps(receipt,indent=2,default=str)+'\n')
print('PASS',sum(c['columns'] for c in receipt['cases'].values()),'columns; 0 ULP in all fields; mutations caught')
