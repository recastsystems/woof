"""Complete-case array hashes permit retiring histories before the base replay."""
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime,timedelta
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np
from netCDF4 import Dataset

def hash_frame(path):
    path=Path(path);rows=[]
    with Dataset(path) as f:
        f.set_auto_maskandscale(False)
        for name in sorted(f.variables):
            x=f[name];h=hashlib.sha256()
            for k in (range(x.shape[0]) if x.ndim else [None]):
                a=np.ascontiguousarray(x[...] if k is None else x[k])
                h.update(a.tobytes())
            rows.append(dict(variable=name,shape=list(x.shape),dtype=str(x.dtype),sha256=h.hexdigest()))
    return dict(frame=path.name,variables=rows)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--start',required=True);p.add_argument('--hours',type=int,required=True);p.add_argument('--domains',type=int,nargs='+',default=[1,2]);p.add_argument('--workers',type=int,default=6);p.add_argument('--reference',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args();t=time.perf_counter()
    initial=datetime.fromisoformat(a.start.replace('Z','+00:00'))
    expected=sorted(f'wrfout_d{d:02d}_{initial+timedelta(hours=h):%Y-%m-%d_%H_%M_%S}' for d in a.domains for h in range(a.hours+1))
    found=sorted(x.name for x in a.root.glob('wrfout_d??_*') if x.is_file())
    if found!=expected:raise ValueError(f'Incomplete inventory: missing {sorted(set(expected)-set(found))}; extra {sorted(set(found)-set(expected))}')
    with ProcessPoolExecutor(max_workers=a.workers) as pool:rows=list(pool.map(hash_frame,[a.root/n for n in expected]))
    result=dict(schema='tc-hafs-init.whole-history-sha256/v1',start=a.start,hours=a.hours,domains=a.domains,frames=rows,seconds=time.perf_counter()-t,complete=True)
    if a.reference:
        ref=json.loads(a.reference.read_text())
        for key in ('schema','start','hours','domains','complete'):
            if ref[key]!=result[key]:raise ValueError(f'Reference contract differs: {key}')
        result['reference']=str(a.reference);result['all_array_hashes_equal']=ref['frames']==rows
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
    if a.reference and not result['all_array_hashes_equal']:raise SystemExit('Array hashes differ; no OFF bitwise claim')
if __name__=='__main__':main()
