"""Replay full-domain native transport captures through production launchers."""
from pathlib import Path
from types import SimpleNamespace
import json
import os
import sys
import hashlib
import numpy as np


def read(path):
    with path.open('rb') as f:
        hdr = np.fromfile(f,dtype='<i4',count=27)
        _,mode,ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,its,ite,jts,jte,kts,kte,*_ = hdr
        rdx,rdy,dt = np.fromfile(f,dtype='<f4',count=3)
        shape=(jme-jms+1,kme-kms+1,ime-ims+1)
        def a3():
            return np.fromfile(f,dtype='<f4',count=int(np.prod(shape))).reshape(shape).transpose(1,0,2).copy()
        def a2():
            return np.fromfile(f,dtype='<f4',count=(jme-jms+1)*(ime-ims+1)).reshape(jme-jms+1,ime-ims+1)
        a={k:a3() for k in ('q','old','tend','ru','rv','rw','romi')}
        a.update({k:a2() for k in ('mut','mub','mu_old','msfux','msfuy','msfvx','msfvy','msftx','msfty')})
        a.update({k:np.fromfile(f,dtype='<f4',count=kme-kms+1) for k in ('c1h','c2h','fnm','fnp','rdnw','rdn')})
        a['reference']=a3()
        if mode==5:
            a['h_reference']=a3();a['z_reference']=a3()
        assert not f.read(),path
    return dict(header=hdr,a=a,rdx=rdx,rdy=rdy,dt=dt)


def assemble(paths):
    rows=[read(p) for p in paths]
    h=rows[0]['header']; mode=int(h[1]); ids,ide,jds,jde,kds,kde=map(int,h[2:8])
    nx,ny,nz=ide-ids,jde-jds,kde-kds
    fieldshape=(nz+(mode in (4,9)),ny+(mode in (3,8)),nx+(mode in (2,7)))
    shapes={k:fieldshape for k in ('q','old','tend','reference','h_reference','z_reference')}
    shapes.update(ru=(nz,ny,nx+1),rv=(nz,ny+1,nx),rw=(nz+1,ny,nx),romi=(nz+1,ny,nx),
                  mut=(ny,nx),mub=(ny,nx),mu_old=(ny,nx),msftx=(ny,nx),msfty=(ny,nx),
                  msfux=(ny,nx+1),msfuy=(ny,nx+1),msfvx=(ny+1,nx),msfvy=(ny+1,nx))
    out={k:np.full(s,np.nan,dtype=np.float32) for k,s in shapes.items() if k in rows[0]['a']}
    filled={k:np.zeros(s,dtype=bool) for k,s in shapes.items() if k in out}
    for row in rows:
        hdr,a=row['header'],row['a']
        ims,ime,jms,jme,kms,kme,its,ite,jts,jte,kts,kte=map(int,hdr[8:20])
        for key,dest in out.items():
            sx,sy=dest.shape[-1],dest.shape[-2]
            # Each rank owns its tile; the global redundant staggered face
            # belongs to the last tile, not a periodic copy of the first.
            x0=max(its,ids);x1=min(ite+1,ids+sx)
            y0=max(jts,jds);y1=min(jte+1,jds+sy)
            target=(slice(y0-jds,y1-jds),slice(x0-ids,x1-ids))
            source=(slice(y0-jms,y1-jms),slice(x0-ims,x1-ims))
            if dest.ndim==3:
                target=(slice(0,dest.shape[0]),)+target
                source=(slice(kds-kms,kds-kms+dest.shape[0]),)+source
            dest[target]=a[key][source]
            filled[key][target]=True
    for key,mask in filled.items():
        if not mask.all():
            raise ValueError(f'{key}: {np.count_nonzero(~mask)} uncaptured words')
    for key in ('c1h','c2h','fnm','fnp','rdnw','rdn'):
        out[key]=rows[0]['a'][key][kds-int(h[12]):kde-int(h[12])+1].copy()
    return rows[0],out


def main():
    import cupy as cp
    from woof.core.advection import launch_flux_div_scalar,launch_flux_div_u,launch_flux_div_v,launch_flux_div_w
    from woof.core.moist import launch_pd_fluxes,launch_pd_renorm_apply
    from woof.verify.advect_oracle import measure_words
    directory=Path(sys.argv[1]);receipt=Path(sys.argv[2])
    records={}
    for name in ('scalar','u','v','w','scalar_pd','weno_u','weno_v','weno_w'):
        paths=sorted(directory.glob(name+'-*.bin'))
        if not paths: continue
        meta,a=assemble(paths);h=meta['header'];mode=int(h[1])
        dev={k:cp.asarray(v) for k,v in a.items() if not k.endswith('reference')}
        coord=SimpleNamespace(**dev, mub2d=dev['mub'], mup0=dev['mu_old'])
        flags=int(h[25]);spec=bool(flags&12)
        common=dict(msf=dev['msftx' if mode in(1,4,5,9) else 'msfux' if mode in(2,7) else 'msfvy'],
                    has_msf=True,open_x=spec or bool(flags&48),open_y=spec or bool(flags&192))
        nz,ny,nx=int(h[7]-h[6]),int(h[5]-h[4]),int(h[3]-h[2])
        vorder=int(h[24] if mode in(1,4,5,9) else h[22])
        if mode in (1,4,5): common['horder']=int(h[23])
        dx=float(np.float32(1.0)/meta['rdx']);dy=float(np.float32(1.0)/meta['rdy'])
        if mode==5:
            fluxes=tuple(cp.zeros(s,dtype=cp.float32) for s in
                         ((nz,ny,nx+1),)*2+((nz,ny+1,nx),)*2+((nz+1,ny,nx),)*2)
            c={k:v for k,v in common.items() if k!='msf'}
            c['msft']=common['msf']
            launch_pd_fluxes(dev['q'],dev['old'],dev['ru'],dev['rv'],dev['rw'],dev['mut'],
                             coord,dx,dy,float(meta['dt']),*fluxes,vorder=vorder,**c)
            c.pop('horder',None)
            launch_pd_renorm_apply(dev['old'],dev['mu_old'],*fluxes,dev['tend'],coord,
                                  dx,dy,float(meta['dt']),**c)
        else:
            launcher={1:launch_flux_div_scalar,2:launch_flux_div_u,3:launch_flux_div_v,4:launch_flux_div_w,
                      7:launch_flux_div_u,8:launch_flux_div_v,9:launch_flux_div_w}[mode]
            if mode in (7,8,9): common['weno']=True
            if mode in (2,3): common['horder']=int(h[21])
            launcher(dev['q'],dev['ru'],dev['rv'],dev['rw'],dev['tend'],coord,dx,dy,
                     spec=spec,vorder=vorder,**common)
        got=cp.asnumpy(dev['tend'])
        row=measure_words(got,a['reference'])
        if '--undefined-witness' in sys.argv and mode == 5:
            witness=Path(sys.argv[sys.argv.index('--undefined-witness')+1])
            assert int(h[23]) == 2 and spec, 'only witnessed specified H2 PD flux is undefined'
            with np.load(witness) as f: mask=f['undefined']
            expected=np.zeros_like(got,dtype=bool)
            expected[:,(0,ny-1),1:nx-1]=True
            assert np.array_equal(mask,expected), 'witness must name exactly the native unwritten band'
            raw=row
            row=measure_words(got[~mask],a['reference'][~mask])
            row.update(undefined_words=int(mask.sum()),total_words=int(got.size),
                       undefined_witness=str(witness),witness_sha256=hashlib.sha256(witness.read_bytes()).hexdigest(),
                       unmasked_comparison=raw)
        row['input_nonzero_words']=int(np.count_nonzero(a['q']))
        records[name]=row
        np.savez_compressed(receipt.parent/(name+'-words.npz'),got=got,reference=a['reference'])
        print(name,json.dumps(row),flush=True)
    import subprocess
    from woof import __file__ as engine_file
    engine_root=Path(engine_file).resolve().parent.parent
    receipt.write_text(json.dumps(dict(engine_commit=subprocess.check_output(['git','-C',str(engine_root),'rev-parse','HEAD'],text=True).strip(), gpu=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(), command=sys.argv,strict=os.environ.get('GPUWM_WRF_EXACT'),
                                      rows=records),indent=2)+'\n')


if __name__=='__main__':main()
