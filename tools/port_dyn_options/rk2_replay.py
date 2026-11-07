"""Replay real native RK2 scalar updates through production bookkeeping/update kernels."""
from pathlib import Path
from types import SimpleNamespace
import hashlib,json,os,subprocess,sys
import numpy as np


def read(path):
    with path.open('rb') as f:
        h=np.fromfile(f,'<i4',26); dt=np.fromfile(f,'<f4',1)[0]
        ims,ime,jms,jme,kms,kme=map(int,h[8:14]);ns=int(h[21]-h[20]+1)
        ni,nj,nk=ime-ims+1,jme-jms+1,kme-kms+1
        def a3():return np.fromfile(f,'<f4',ni*nj*nk).reshape(nj,nk,ni).transpose(1,0,2).copy()
        def a4():return np.fromfile(f,'<f4',ns*ni*nj*nk).reshape(ns,nj,nk,ni).transpose(0,2,1,3).copy()
        def a2():return np.fromfile(f,'<f4',ni*nj).reshape(nj,ni).copy()
        a={}
        if h[1]==1:
            a.update({k:a4() for k in ('scalar_1','scalar_2','sc_tend')})
            a['advect_tend']=a3();a['msftx']=a2();a['msfty']=a2()
        else:a.update({k:a4() for k in ('scalar','sc_tend')})
        a.update({k:a2() for k in ('mu_old','mu_new','mu_base')})
        a.update({k:np.fromfile(f,'<f4',nk) for k in ('c1','c2')})
        a.update({k+'_reference':a4() for k in (('scalar_1','scalar_2','sc_tend') if h[1]==1 else ('scalar','sc_tend'))})
        assert not f.read(),path
    return h,dt,a


def assemble(paths):
    tiles=[read(p) for p in paths];h,dt,a=tiles[0]
    nx,ny,nz=int(h[3]-h[2]),int(h[5]-h[4]),int(h[7]-h[6])
    out={k:np.full(v.shape[:-3]+(nz,ny,nx) if v.ndim>=3 else (ny,nx),np.nan,'f4') for k,v in a.items() if v.ndim>=2}
    masks={k:np.zeros(v.shape,bool) for k,v in out.items()}
    for th,td,ta in tiles:
        assert td==dt and th[22]==h[22]
        ims,ime,jms,jme,kms,kme,its,ite,jts,jte=map(int,th[8:18])
        x0,x1=max(its,int(h[2])),min(ite+1,int(h[3]));y0,y1=max(jts,int(h[4])),min(jte+1,int(h[5]))
        for key,dest in out.items():
            target=(slice(y0-int(h[4]),y1-int(h[4])),slice(x0-int(h[2]),x1-int(h[2])))
            source=(slice(y0-jms,y1-jms),slice(x0-ims,x1-ims))
            if dest.ndim>=3:target=(slice(0,nz),)+target;source=(slice(int(h[6])-kms,int(h[6])-kms+nz),)+source
            if dest.ndim==4:target=(slice(None),)+target;source=(slice(None),)+source
            dest[target]=ta[key][source];masks[key][target]=True
    assert all(m.all() for m in masks.values())
    for k in ('c1','c2'):out[k]=a[k][int(h[6])-int(h[12]):int(h[7])-int(h[12])].copy()
    return h,dt,out


def main():
    import cupy as cp
    from woof.core import dycore,moist
    from woof.verify.advect_oracle import measure_words
    root,dest=map(Path,sys.argv[1:3]);dest.mkdir(parents=True,exist_ok=True)
    records={}
    for stem in ('rk_update_scalar-1','rk_update_scalar-2','rk_update_scalar_pd-2'):
        h,dt,a=assemble(sorted(root.glob(stem+'-*.bin')));d={k:cp.asarray(v) for k,v in a.items() if not k.endswith('_reference')}
        mu0=d['mu_old']+d['mu_base'];mu=d['mu_new']+d['mu_base'];outputs={}
        if h[1]==1:
            q0,q=d['scalar_1'],d['scalar_2']
            if h[22]==1:
                launch=dycore._prepare_bookkeeping(SimpleNamespace(),tuple((q[i],q0[i]) for i in range(q.shape[0])))
                if launch is None:q0[...]=q
                else:launch()
                outputs['scalar_1']=q0
            for i in range(q.shape[0]):
                tend=d['advect_tend'].copy()
                if h[25]&3:moist._exclude_specified_ring_advection(tend,int(h[24]))
                moist._update_scalar_in_place(q[i],q0[i],tend,d['c1'],d['c2'],mu0.reshape(-1),mu.reshape(-1),float(dt),msft=d['msfty'].reshape(-1),physics=d['sc_tend'][i],clamp=False)
            outputs['scalar_2']=q
        else:
            assert np.array_equal(a['mu_old'],a['mu_new']), 'native pre-PD source fold must use one mass'
            q=d['scalar'];chm0=d['c1'][:,None,None]*mu0[None]+d['c2'][:,None,None]
            for i in range(q.shape[0]):moist._fold_wrf_pd_sources(q[i],chm0,d['sc_tend'][i],float(dt),q[i])
            outputs['scalar']=q
        rows={}
        for key,value in outputs.items():
            got=cp.asnumpy(value);reference=a[key+'_reference'];rows[key]=measure_words(got,reference)
            np.savez_compressed(dest/(stem+'-'+key+'-words.npz'),got=got,reference=reference,**{'input_'+k:v for k,v in a.items() if not k.endswith('_reference')})
        records[stem]=dict(dt=float(dt),rk_step=int(h[22]),rk_ord=int(h[23]),input_nonzero_words=int(np.count_nonzero(a['scalar_2' if h[1]==1 else 'scalar'])),rows=rows)
        print(stem,json.dumps(records[stem]),flush=True)
    from woof import __file__ as engine_file
    engine=Path(engine_file).resolve().parent.parent
    from woof.config import RunConfig
    import re
    schedule=[]
    for line in (root.parent/'rsl.error.0000').read_text().splitlines():
        if 'DYN_RK2_STAGE' in line:
            v=line.split('DYN_RK2_STAGE',1)[1].split();schedule.append([int(v[0]),int(v[1]),float(v[2]),float(v[3]),int(v[4])])
    cfg=RunConfig(nx=150,ny=150,nz=49,dx=3000,dy=3000,ztop=20000,dt=12,run_seconds=60,rk_ord=2,time_step_sound=4)
    expected=[[i+1,2,n*dt,dt,n] for i,(n,dt) in enumerate(dycore._rk_acoustic_stages(cfg))]
    assert schedule and all(row==expected[i%2] for i,row in enumerate(schedule))
    receipt=dict(native_schedule=schedule,production_schedule=expected,command=sys.argv,engine_commit=subprocess.check_output(['git','-C',str(engine),'rev-parse','HEAD'],text=True).strip(),strict=os.getenv('GPUWM_WRF_EXACT'),rows=records)
    (dest/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    assert all(v['different_words']==0 for r in records.values() for v in r['rows'].values())

if __name__=='__main__':main()
