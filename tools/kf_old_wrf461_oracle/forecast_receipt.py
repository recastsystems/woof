"""Small all-field receipts from a forecast, read by the Rust NetCDF bridge."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from woof.netcdf_bridge import open_dataset
from woof.io.wrfout import iter_wrfout_files

def record(root):
    result = {}
    for path in iter_wrfout_files(Path(root)):
        fields = {}
        with open_dataset(path) as ds:
            for name, variable in sorted(ds.variables.items()):
                a = np.ascontiguousarray(variable[:])
                entry = {'shape': list(a.shape), 'dtype': str(a.dtype),
                         'sha256': hashlib.sha256(a.tobytes()).hexdigest()}
                if np.issubdtype(a.dtype, np.number):
                    entry.update(finite=int(np.count_nonzero(np.isfinite(a))),
                                 words=int(a.size))
                    if a.size and np.isfinite(a).all():
                        entry.update(min=float(a.min()), max=float(a.max()))
                fields[name] = entry
        result[path.name] = fields
    if not result:
        raise ValueError(f'no completed forecast frames in {root}')
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('receipt',type=Path)
    p.add_argument('--compare',type=Path)
    args=p.parse_args()
    current=record(args.root)
    output={'root':str(args.root),'frames':current}
    if args.compare:
        baseline=json.loads(args.compare.read_text())['frames']
        differences=[]
        for frame in sorted(set(current)|set(baseline)):
            now=current.get(frame,{})
            old=baseline.get(frame,{})
            for name in sorted(set(now)|set(old)):
                if now.get(name) != old.get(name):
                    differences.append({'frame':frame,'field':name,
                                        'have':now.get(name),'want':old.get(name)})
        output['comparison']={'baseline':str(args.compare),
                              'differing_fields':differences}
    args.receipt.write_text(json.dumps(output,indent=1)+'\n')
    print('frames',len(current),'fields',sum(map(len,current.values())))
    if args.compare:
        print('different fields',len(output['comparison']['differing_fields']))
        return bool(output['comparison']['differing_fields'])
    return 0

if __name__=='__main__':
    raise SystemExit(main())
