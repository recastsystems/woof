"""Compare every stored NetCDF word at every common history time."""
from pathlib import Path
import hashlib
import json
import subprocess

base = Path('/workspace/runs/port-dyn-options')
reader = '/workspace/src/engine/tools/rustwx/target/release/rw_netcdf'
before = base / 'default-before/run-20261102-101752Z_i202405251800Z/wrfout'
after = base / 'default-after/run-20261102-102157Z_i202405251800Z/wrfout'
receipt = dict(commits={arm:(base/f'default-{arm}.commit').read_text().strip()
                       for arm in ('before','after')}, frames=[])
for old in sorted(before.glob('wrfout_d01_*')):
    new = after / old.name
    inventories = [json.loads(subprocess.check_output([reader,'inventory',str(p)]))
                   for p in (old,new)]
    names = [[v['name'] for v in i['variables']] for i in inventories]
    assert names[0] == names[1], (old,names)
    row = dict(file=old.name, variables=len(names[0]), differing_bytes=0, bytes=0,
               file_identical=old.read_bytes()==new.read_bytes())
    for arm, path in (('before',old),('after',new)):
        dest=base/'default-words'/arm/old.name
        dest.mkdir(parents=True, exist_ok=True)
        subprocess.run([reader,'stored-words',str(path),str(dest),*names[0]],check=True,
                       stdout=subprocess.DEVNULL)
    root=base/'default-words'
    row['fields']={}
    for p in sorted((root/'before'/old.name).iterdir()):
        if p.name=='metadata.json': continue
        a=p.read_bytes();b=(root/'after'/old.name/p.name).read_bytes()
        assert len(a)==len(b)
        different=sum(x!=y for x,y in zip(a,b))
        row['bytes']+=len(a);row['differing_bytes']+=different
        row['fields'][p.name]=dict(bytes=len(a),different=different,
                                  sha256=hashlib.sha256(a).hexdigest())
    receipt['frames'].append(row)
    print(old.name,row['differing_bytes'],row['bytes'],row['file_identical'],flush=True)
(base/'default-identity.json').write_text(json.dumps(receipt,indent=2)+'\n')
assert all(r['differing_bytes']==0 for r in receipt['frames'])
