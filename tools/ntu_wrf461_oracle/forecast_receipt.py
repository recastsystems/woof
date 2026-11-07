"""Read forecast fields with the Rust NetCDF bridge and keep only receipts."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from woof.netcdf_bridge import open_dataset
from woof.core.ntu import NTU_SPECIES
from woof.io.history_layout import live_state_history_fields
from types import SimpleNamespace

def field_names():
 fields=live_state_history_fields(SimpleNamespace(**{name:object() for name in NTU_SPECIES}))
 return ("T","QVAPOR","QCLOUD","QRAIN")+tuple(fields)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);ap.add_argument('files',type=Path,nargs='+');a=ap.parse_args()
 names=field_names()
 result={'decoder':'woof.netcdf_bridge / rw_netcdf','fields':{}}
 for path in a.files:
  fields={}
  with open_dataset(path) as ds:
   for name in names:
    var=ds.variables[name];var.set_auto_mask(False);value=np.ascontiguousarray(var[:],dtype='<f4')
    finite=np.isfinite(value);words=value.view('<u4')
    fields[name]={'dimensions':var.dimensions,'shape':value.shape,'stored_dtype':str(var.dtype),'sha256_f32_words':hashlib.sha256(value.tobytes()).hexdigest(),'words':value.size,'nonfinite':int(value.size-np.count_nonzero(finite)),'nonzero_words':int(np.count_nonzero(words&np.uint32(0x7fffffff))),'minimum':float(value[finite].min()) if finite.any() else None,'maximum':float(value[finite].max()) if finite.any() else None,'units':getattr(var,'units','')}
  result['fields'][path.name]=fields
 a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 print(json.dumps({'frames':len(result['fields']),'fields_per_frame':len(names),'nonfinite':sum(v['nonfinite'] for f in result['fields'].values() for v in f.values())}))
if __name__=='__main__':main()
