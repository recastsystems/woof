"""All run-level GFDLETAINIT table words on four vertical grids."""
import os
from pathlib import Path
import numpy as np
import pytest
pytestmark=pytest.mark.skipif(os.environ.get('GPUWM_NO_LOCAL_GPU')=='1',reason='CPU-only check')
@pytest.mark.parametrize('n',[12,24,42,60])
@pytest.mark.parametrize('co2tf',[0,1])
def test_full_initialization(n,co2tf):
 import cupy as cp
 from woof.core.gfdleta import GFDLTables,_OFFSETS,_LAYOUT
 with np.load(Path(__file__).parent/f'fixtures/gfdleta/column/init-{n}.npz') as z:
  tables=GFDLTables(cp.asarray(z['sigma_full'][:n+1]),cp.asarray(z['sigma_half'][:n]),1000.,co2tf=co2tf,transmission=z['transmission'])
  got=cp.asnumpy(tables.words);want=z['table'];stats=[]
  for name,(off,count) in _OFFSETS.items():
   bad=np.flatnonzero(got[off:off+count]!=want[off:off+count])
   if len(bad):stats.append((name,len(bad),[(int(k),f'{got[off+k]:08X}',f'{want[off+k]:08X}') for k in bad[:3]]))
  assert not stats,stats

def test_conrad_override_cannot_redefine_the_native_table_pin(tmp_path):
 import cupy as cp
 import json,hashlib,shutil
 from woof.core.gfdleta import GFDLTables
 from woof.table_assets import physics_table_root
 from woof.table_assets import TableAssetError
 root=physics_table_root('gfdleta')
 for p in root.iterdir():shutil.copyfile(p,tmp_path/p.name)
 with np.load(Path(__file__).parent/'fixtures/gfdleta/column/init-12.npz') as z:
  target=tmp_path/'co2-trans-12.npy';np.save(target,z['transmission'])
  manifest=json.loads((tmp_path/'MANIFEST.json').read_text());manifest['binary_levels']=12
  data=target.read_bytes();manifest['assets'][target.name]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
  (tmp_path/'MANIFEST.json').write_text(json.dumps(manifest))
  with pytest.raises(TableAssetError,match='manifest differs from the package pin'):
   GFDLTables(cp.asarray(z['sigma_full'][:13]),cp.asarray(z['sigma_half'][:12]),1000.,co2tf=0,table_root=tmp_path)
