"""Compare stored forecast words through the Rust NetCDF reader."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import tempfile
import numpy as np


def compare(left, right, reader):
    def inventory(path):
        return json.loads(subprocess.check_output([reader, 'inventory', str(path)]))
    li, ri = inventory(left), inventory(right)
    def names(value):
        rows = value['variables']
        return set(rows) if isinstance(rows, dict) else {v['name'] for v in rows}
    ln, rn = names(li), names(ri)
    common = sorted(ln & rn)
    with tempfile.TemporaryDirectory() as temp:
        dirs = [Path(temp)/'left', Path(temp)/'right']
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            list(pool.map(lambda args: subprocess.run(args, check=True),
                [[reader,'stored-words',str(src),str(dst),*common]
                 for src,dst in zip((left,right),dirs)]))
        metadata = [json.loads((p/'metadata.json').read_text()) for p in dirs]
        rows=[]
        for a,b in zip(*(m['variables'] for m in metadata)):
            assert a['name'] == b['name'] and a['dtype'] == b['dtype'] and a['shape'] == b['shape']
            dtype=np.dtype(a['dtype']); width=dtype.itemsize
            av=(dirs[0]/a['filename']).read_bytes();bv=(dirs[1]/b['filename']).read_bytes()
            assert len(av)==len(bv)
            mask=np.frombuffer(av,dtype=f'u{width}')!=np.frombuffer(bv,dtype=f'u{width}')
            rows.append(dict(name=a['name'],words=len(mask),differing=int(mask.sum())))
    return dict(left=str(left),right=str(right),left_only=sorted(ln-rn),right_only=sorted(rn-ln),
                variables=rows,words=sum(r['words'] for r in rows),differing=sum(r['differing'] for r in rows))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('left',type=Path);p.add_argument('right',type=Path)
    p.add_argument('--reader',default=os.environ.get('WOOF_RW_NETCDF','rw_netcdf'))
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=compare(a.left,a.right,a.reader)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='variables'}))
    raise SystemExit(bool(result['differing'] or result['left_only'] or result['right_only']))
