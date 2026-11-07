"""Package compiler-extracted FLG tables in a deterministic, pinned order."""
import csv
import hashlib
import json
from pathlib import Path
import sys

def package(build,source,dest):
    build=Path(build); dest=Path(dest); dest.mkdir(parents=True,exist_ok=True)
    raw=(build/'tables.bin').read_bytes(); chunks={}; shapes={}
    for row in csv.DictReader((build/'tables-index.csv').open()):
        name=row['name']; offset=int(row['offset']); count=int(row['count'])
        if name in chunks: raise ValueError('duplicate table '+name)
        chunks[name]=raw[offset:offset+4*count]; shapes[name]=[int(x) for x in row['shape'].split('x') if x]
        if len(chunks[name])!=4*count: raise ValueError('truncated table '+name)
    payload=b''; entries={}
    for name in sorted(chunks):
        chunk=chunks[name]
        entries[name]={'offset':len(payload),'count':len(chunk)//4,'shape':shapes[name],'sha256':hashlib.sha256(chunk).hexdigest()}
        if name=='ice6.dps':
            if chunk!=bytes(96): raise ValueError('the pinned WRF compiler did not expose zero ice6.dps words')
            entries[name]['initialization']='No DATA initializer in WRF; the pinned gfortran object exposes zero BSS words.'
        payload+=chunk
    if len(payload)!=len(raw): raise ValueError('index does not cover the extracted table file')
    manifest={'format':'flg-fp32-tables-v1','dtype':'<f4','order':'F','bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest(),
              'wrf_commit':'d66e442fccc04111067e29274c9f9eaccc3cef28','source':'phys/module_ra_flg.F',
              'source_sha256':hashlib.sha256(Path(source).read_bytes()).hexdigest(),'tables':entries}
    (dest/'tables.bin').write_bytes(payload); (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__': package(*sys.argv[1:])
