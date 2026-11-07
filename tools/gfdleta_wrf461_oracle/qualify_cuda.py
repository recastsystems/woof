"""Emit actual initialized CUDA columns for the original full-port gate."""
from pathlib import Path
from datetime import datetime
import argparse,json,csv
import numpy as np
import cupy as cp
from woof.core.gfdleta import GFDLTables,_LAYOUT,_OFFSETS
from woof.core.kernels import get_kernel
from validate_gfdleta import TABLE_FIELDS,check_coverage,compare_words,read_words
ROOT=Path(__file__).parents[2]
p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args()
ref=a.output/'reference';got=a.output/'candidate';ref.mkdir(parents=True,exist_ok=True);got.mkdir(parents=True,exist_ok=True)
desc={x['name'].upper():x for x in _LAYOUT['fields']}
KEYS={'RTHRATEN':'THRATEN','RTHRATENLW':'THRATENLW','RTHRATENSW':'THRATENSW','SWDOWN':'RSWIN','SWDOWNC':'RSWINC'}
def tables_csv(path,words,n,solar):
 with path.open('w',newline='') as stream:
  w=csv.writer(stream);w.writerow(['field','index','bits'])
  for name in sorted(TABLE_FIELDS):
   if name=='SOLARD':v=solar
   elif name=='TABLE_AUX':v=np.array([words[_OFFSETS[k][0]] for k in ['AB15WD','SKC1R','SKO3R','SKO2D']],np.uint32)
   else:
    off,count=_OFFSETS[name];v=words[off:off+count]
    shape=_LAYOUT['tables'][name][1]
    if shape=='82,82':v=v.reshape(82,82)[:n+1,:n+1].reshape(-1)
    elif shape=='82':v=v[:n+1]
    elif shape=='81' and name!='PRGFDL':v=v[:n]
   for k,bits in enumerate(v):w.writerow([name,k+1,f'{int(bits):08X}'])
checks={}
for n in [12,24,42,60]:
 with np.load(ROOT/f'tests/fixtures/gfdleta/column/init-{n}.npz') as iz,np.load(ROOT/f'tests/fixtures/gfdleta/column/columns-{n}.npz') as z:
  for arm in [0,1]:
   name=f'n{n}-a{arm}'
   table=GFDLTables(cp.asarray(iz['sigma_full'][:n+1]),cp.asarray(iz['sigma_half'][:n]),1000.,co2tf=arm,transmission=iz['transmission'],start_time=datetime(2001,3,21,12))
   f=cp.asarray(z['floats']);ints=cp.asarray(z['integers']);work=cp.empty((8,_LAYOUT['scratch_bytes']),cp.uint8)
   get_kernel('gfdleta_column','gd_radiation_columns')((1,),(32,),(table.words,f,ints,work,np.int32(8)))
   actual=cp.asnumpy(f).view(np.uint32)
   for folder,values in [(got,actual[z['columns'],z['indices']]),(ref,z['reference'])]:
    with (folder/f'{name}-fields.csv').open('w',newline='') as stream:
     w=csv.writer(stream);w.writerow(['column','level','field','bits'])
     for field,col,ix,bits in zip(z['fields'],z['columns'],z['indices'],values):
      d=desc[KEYS.get(str(field),str(field))];level=int(ix)-d['offset']+1 if d['count']>1 else 0
      w.writerow([int(col)+1,level,str(field),f'{int(bits):08X}'])
   tables_csv(got/f'{name}-tables.csv',cp.asnumpy(table.words),n,cp.asnumpy(table.solar_distance).view(np.uint32))
   tables_csv(ref/f'{name}-tables.csv',iz['table'],n,iz['solar_distance'])
   for kind in ['fields','tables']:
    file=f'{name}-{kind}.csv';checks[file]=compare_words(read_words(ref/file),read_words(got/file))
   print(name,'pass',all(checks[f'{name}-{kind}.csv']['pass'] for kind in ['fields','tables']),flush=True)
coverage=check_coverage(got);check_coverage(ref)
result={'pass':all(v['pass'] for v in checks.values()),'coverage':coverage,'checks':checks}
(a.output/'qualification.json').write_text(json.dumps(result,indent=2)+'\n')
raise SystemExit(0 if result['pass'] else 1)
