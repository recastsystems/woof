"""Small, full-field numeric receipts through the Rust NetCDF reader."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from woof.netcdf_bridge import Dataset

def collect(root, pattern='wrfout_d??_*'):
 result={}
 for path in sorted(root.glob(pattern)):
  if not path.is_file():continue
  fields={}
  with Dataset(path) as ds:
   ds.prefetch(list(ds.variables))
   for name,var in ds.variables.items():
    value=np.asarray(var[:])
    wire=np.ascontiguousarray(value if var.is_character else value.astype('<f4'))
    entry=dict(sha256=hashlib.sha256(wire.tobytes()).hexdigest(),shape=list(wire.shape),words=int(wire.size),dtype=str(wire.dtype))
    if not var.is_character:
     finite=np.isfinite(wire)
     entry.update(nonfinite=int(np.count_nonzero(~finite)),nonzero=int(np.count_nonzero(wire)),minimum=float(np.nanmin(wire)),maximum=float(np.nanmax(wire)))
    fields[name]=entry
  result[path.name]=fields
 if not result:raise ValueError(f'no forecast frames at {root}')
 return result

def main():
 ap=argparse.ArgumentParser();ap.add_argument('candidate',type=Path);ap.add_argument('out',type=Path);ap.add_argument('--reference',type=Path);ap.add_argument('--frames',default='wrfout_d??_*',help='explicit filename glob, including domains and valid time');a=ap.parse_args()
 result=dict(candidate=str(a.candidate),pattern=a.frames,hash_definition='all decoded numeric fields normalized to float32, all character bytes unchanged',frames=collect(a.candidate,a.frames))
 if a.reference:
  expected=collect(a.reference,a.frames);bad=[];words=0
  if set(expected)!=set(result['frames']):bad.append(dict(error='frame inventory differs'))
  for frame,fields in expected.items():
   received=result['frames'].get(frame,{})
   if set(fields)!=set(received):bad.append(dict(frame=frame,error='field inventory differs'))
   for name,row in fields.items():
    words+=row['words']
    if name not in received or received[name]['sha256']!=row['sha256'] or received[name]['shape']!=row['shape']:bad.append(dict(frame=frame,field=name))
  result.update(reference=str(a.reference),checked_words=words,mismatches=bad,bit_identical=not bad)
 a.out.write_text(json.dumps(result,indent=1));print('frames',len(result['frames']),'mismatches',len(result.get('mismatches',[])))
 if result.get('mismatches'):raise SystemExit(1)
if __name__=='__main__':main()
