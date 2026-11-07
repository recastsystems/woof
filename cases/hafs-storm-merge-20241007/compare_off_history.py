"""Compare every netCDF history array over a complete real forecast.

This is a numerical-array check, not a container-byte check: timestamps and
compression metadata may differ. Missing frames or variables are failures.
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from datetime import datetime,timedelta
import argparse,hashlib,json,time
import numpy as np
from netCDF4 import Dataset

def compare_pair(task):
    left,right=map(Path,task);rows=[]
    with Dataset(left) as a,Dataset(right) as b:
        a.set_auto_maskandscale(False);b.set_auto_maskandscale(False)
        if set(a.variables)!=set(b.variables):raise ValueError(f'{left.name}: variable inventory differs')
        for name in sorted(a.variables):
            x,y=a[name],b[name]
            if x.shape!=y.shape or x.dtype!=y.dtype:raise ValueError(f'{left.name}:{name}: shape/dtype differs')
            h1,h2=hashlib.sha256(),hashlib.sha256();same=True
            # Bound memory for atmospheric variables without omitting levels.
            pieces=range(x.shape[0]) if x.ndim else [None]
            for k in pieces:
                v1=np.ascontiguousarray(x[...] if k is None else x[k]);v2=np.ascontiguousarray(y[...] if k is None else y[k])
                p1,p2=v1.tobytes(),v2.tobytes();h1.update(p1);h2.update(p2);same=same and p1==p2
            rows.append(dict(variable=name,shape=x.shape,dtype=str(x.dtype),bitwise_equal=same,left_sha256=h1.hexdigest(),right_sha256=h2.hexdigest()))
    return dict(frame=left.name,variables=rows,bitwise_equal=all(r['bitwise_equal'] for r in rows))

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--off',type=Path);p.add_argument('--start',required=True);p.add_argument('--hours',type=int,required=True);p.add_argument('--interval-minutes',type=int,default=60);p.add_argument('--domains',type=int,nargs='+',default=[1,2]);p.add_argument('--workers',type=int,default=6);p.add_argument('--out',type=Path,required=True);a=p.parse_args();start=time.perf_counter()
    if a.off is None:
        raise ValueError('--off required for direct pair comparison; preserve both histories until the check completes')
    if a.interval_minutes<=0 or a.hours*60%a.interval_minutes:raise ValueError('interval must divide requested duration')
    moment=datetime.fromisoformat(a.start.replace('Z','+00:00'));expected=[]
    for minutes in range(0,a.hours*60+1,a.interval_minutes):
        for domain in a.domains:expected.append(f'wrfout_d{domain:02d}_{moment+timedelta(minutes=minutes):%Y-%m-%d_%H_%M_%S}')
    for directory in (a.base,a.off):
        found={p.name for p in directory.glob('wrfout_d??_*') if p.is_file()}
        if found!=set(expected):raise ValueError(f'{directory}: missing {sorted(set(expected)-found)}, extra {sorted(found-set(expected))}')
    with ProcessPoolExecutor(max_workers=a.workers) as pool:rows=list(pool.map(compare_pair,[(str(a.base/n),str(a.off/n)) for n in expected]))
    result=dict(schema='tc-hafs-init.whole-off-arrays/v1',start=a.start,hours=a.hours,domains=a.domains,expected_frames=len(expected),compared_frames=len(rows),bitwise_equal=all(r['bitwise_equal'] for r in rows),frames=rows,seconds=time.perf_counter()-start)
    a.out.parent.mkdir(exist_ok=True,parents=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
    if not result['bitwise_equal']:raise SystemExit('OFF array comparison failed; see receipt')
if __name__=='__main__':main()
