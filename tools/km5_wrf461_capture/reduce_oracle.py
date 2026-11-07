"""Keep real captured columns and their unchanged one-cell neighbours."""
import argparse
from pathlib import Path
import hashlib
import json
import numpy as np
from replay import read_capture

p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('out',type=Path);p.add_argument('--theta',type=int,required=True);args=p.parse_args()
args.out.mkdir(parents=True,exist_ok=True)
manifest=[]
for src in sorted(args.source.glob('*_in_*_i0001_j0051.bin')):
    for side in ('in','out'):
        path=Path(str(src).replace('_in_','_'+side+'_'))
        raw=read_capture(path)
        # This includes the moist-theta rounding witness at [k=0,j=23,i=78].
        i0,j0,n=74,20,9
        for k,v in tuple(raw.items()):
            if isinstance(v,np.ndarray) and v.ndim>=2:
                raw[k]=np.ascontiguousarray(v[...,j0:j0+n,i0:i0+n])
        for k,v in dict(ims=0,ime=n-1,jms=0,jme=n-1,its=1,ite=n-2,jts=1,jte=n-2,
                        ids=-100,ide=1000,jds=-100,jde=1000,_interior_only=1).items():
            raw[k]=np.int32(v)
        out=args.out/(path.stem+f'_theta{args.theta}.npz')
        np.savez_compressed(out,**raw)
        manifest.append(dict(file=out.name,source=path.name,source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),crop=dict(i0=i0,j0=j0,width=n),sha256=hashlib.sha256(out.read_bytes()).hexdigest()))
(args.out/f'manifest-theta{args.theta}.json').write_text(json.dumps(manifest,indent=2)+'\n')
