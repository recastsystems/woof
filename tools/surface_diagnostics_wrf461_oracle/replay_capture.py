"""Offline replay of passive captures from the strict, whole native run."""
from pathlib import Path
from datetime import datetime,timedelta
import sys,argparse
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from woof.core.rasm import alarm
import json, hashlib
import numpy as np
import cupy as cp
from woof.core.native_surface_diagnostics import solar,rasm_mean,rasm_diurnal,SURFACE,MEANS
from woof.wrf_exact import ENABLED

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('captures',nargs='+',type=Path)
parser.add_argument('--start',required=True,help='simulation start ISO date/time')
parser.add_argument('--schema',type=Path,required=True)
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args()
schema=json.loads(args.schema.read_text())
receipts=[]
for path in sorted(args.captures):
    kind=path.name.split('-')[0]
    if kind not in schema:continue
    data=path.read_bytes(); header=np.frombuffer(data,dtype='<i4',count=8)
    magic,nx,ny,nz,first,last,kms,kde=map(int,header)
    assert magic==47102
    off=32; rows={}
    for row in schema[kind]['rows']:
        name,dim,phase=row['name'],row['shape'],row['phase']
        shape={'s':(),'2':(ny,nx),'3':(ny,nz,nx),'d':(ny,8,nx),'u':(ny,nz,nx+1),'v':(ny+1,nz,nx)}[dim]
        count=int(np.prod(shape));dtype='<f4' if dim!='s' or name in ('dt','xtime') else '<i4'
        a=np.frombuffer(data,dtype=dtype,count=count,offset=off).copy().reshape(shape);off+=count*4
        if dim in ('3','d','u','v'):a=np.ascontiguousarray(a.transpose(1,0,2))
        rows[phase,name]=a
    assert off==len(data),(kind,off,len(data))
    def get(phase,name):return rows[phase,name]
    def gpu(phase,names):return {k:cp.asarray(get(phase,k)) for k in names}
    before={k:cp.asarray(v) for (ph,k),v in rows.items() if ph=='before' and v.ndim}
    dt=float(get('scalar','dt')); first-=kms;last-=kms
    xt=float(get('scalar','xtime')) if kind!='solar_diag' else 0.
    now=datetime.fromisoformat(args.start)+timedelta(seconds=xt*60)
    if kind=='solar_diag':
        from woof.core.native_surface_diagnostics import SOLAR_PROFILES,SOLAR_PLANES
        present=[int(get('scalar','f_qv'))]
        for q in ('qc','qi','qs'):
            present.append(int(bool(get('scalar','f_'+q)) and get('scalar','p_'+q)>get('scalar','param_first_scalar')))
        solar(gpu('input',SOLAR_PROFILES),gpu('input',SOLAR_PLANES),before,dt=dt,
              first=first,last=last,present=present,radii=tuple(int(get('scalar',k)) for k in ('has_reqc','has_reqi','has_reqs')))
    elif kind=='mean_output_calc':
        xt=float(get('scalar','xtime'));freq=int(get('scalar','output_freq'));interval=int(get('scalar','stats_interval'))*(1,1,60,3600,86400,1)[freq]
        step=int(np.floor(np.float32(np.float32(xt)*60)/np.float32(dt)+.5))
        period=int(np.floor(np.float32(interval)/np.float32(dt)+.5))
        counter=cp.full((ny,nx),get('before','nsteps'),dtype=cp.int32)
        at={k:cp.asarray(get('input',k)[0] if k!='ht' else get('input',k)) for k in ('t','p','pb','moist','ht')}
        rasm_mean(gpu('input',SURFACE),at,{k:before[k+'_mean'] for k in MEANS},counter,reset=alarm(xt*60,dt,interval,monthly=freq==5,now=now)[0],finish=alarm(xt*60,dt,interval,monthly=freq==5,now=now)[1])
        before['nsteps']=counter
    elif kind=='diurnalcycle_output_calc':
        xt=float(get('scalar','xtime'));step=int(np.floor(np.float32(np.float32(xt)*60)/np.float32(dt)+.5));period=int(np.floor(10800/dt+.5))
        counter=cp.full((ny,nx),get('before','avg_nsteps'),dtype=cp.int32)
        days=cp.full((ny,nx),get('before','diurnal_nsteps'),dtype=cp.int32)
        rasm_diurnal(gpu('input',SURFACE),{k:before[k+'_dtmp'] for k in SURFACE},
            {k:before[k+'_diurn'] for k in SURFACE},counter,days,reset=step%period==0,
            reset_month=alarm(xt*60,dt,1,monthly=True,now=now)[0],finish=alarm(xt*60,dt,10800)[1],finish_month=alarm(xt*60,dt,1,monthly=True,now=now)[1],
            cycle=(now+timedelta(seconds=int(dt))).hour//3 or 8)
        before.update(avg_nsteps=counter,diurnal_nsteps=days)
    words=diff=0; fields={}
    for (phase,name),expected in rows.items():
        if phase!='after':continue
        actual=cp.asnumpy(before[name]); n=expected.size
        if expected.ndim==0:
            bad=int(np.any(actual!=expected))
        else:bad=int(np.count_nonzero(actual.view('u4')!=expected.view('u4')))
        words+=n;diff+=bad
        if bad:
            fields[name]={'different_words':bad,'max_abs':float(np.nanmax(abs(actual-expected))),
                          'actual0':float(actual.flat[0]),'expected0':float(expected.flat[0])}
    receipts.append({'file':str(path),'sha256':hashlib.sha256(data).hexdigest(),'kind':kind,
        'words':words,'different_words':diff,'fields':fields})
summary={'strict':ENABLED,'captures':len(receipts),'words':sum(r['words'] for r in receipts),
 'different_words':sum(r['different_words'] for r in receipts),'rows':receipts}
args.out.write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='rows'}))
for r in receipts:
 if r['different_words']:print(json.dumps(r))
assert ENABLED and summary['captures'] and summary['different_words']==0
