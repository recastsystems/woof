"""Replay native SMS-3DTKE call captures and count float32 words.

GPUWM_WRF_EXACT=1 python tools/km5_wrf461_capture/replay.py CAPDIR --json receipt.json
The capture reader preserves halo words and transposes WRF (j,k,i) storage
into the GPU's (k,j,i) storage without arithmetic.
"""
from pathlib import Path
import argparse
import json
import struct
import numpy as np


def read_capture(path):
    if Path(path).suffix == '.npz':
        with np.load(path, allow_pickle=False) as data:
            return {k: data[k][()] if data[k].ndim == 0 else data[k]
                    for k in data.files}
    result = {}
    with Path(path).open('rb') as stream:
        while header := stream.read(44):
            if len(header) != 44: raise ValueError(f'truncated header: {path}')
            name, kind, i0, i1, k0, k1, j0, j1 = struct.unpack('<16s7i', header)
            ni,nk,nj=i1-i0+1,k1-k0+1,j1-j0+1
            dtype='<i4' if kind in (4,5) else '<f4'
            count=ni*nk*nj
            data=stream.read(count*4)
            if len(data)!=count*4: raise ValueError(f'truncated field: {path}')
            a=np.frombuffer(data,dtype=dtype).copy()
            name=name.decode().strip()
            if kind in (1,4):a=a[0]
            elif kind in (2,5):a=a.reshape(nj,ni)
            elif ni==nj==1:a=a.reshape(nk)
            else:a=np.ascontiguousarray(a.reshape(nj,nk,ni).transpose(1,0,2))
            result[name]=a
    return result


def replay(path):
    import cupy as cp
    from woof.core.kernels import get_kernel
    raw=read_capture(path)
    expected=read_capture(str(path).replace('_in_','_out_'))
    a={k:cp.asarray(v) if isinstance(v,np.ndarray) else v for k,v in raw.items()}
    nz=int(a['kde'])-1; nx=int(a['ime']-a['ims']+1); ny=int(a['jme']-a['jms']+1)
    dummy=cp.zeros((nz+1,ny,nx),dtype=cp.float32)
    a['qv']=a.get('qv',a.get(f'moist_{int(a["p_qv"]):02d}',dummy))
    def bounds(stag=''):
        ib=int(a['its']); ie=min(int(a['ite']),int(a['ide'])-(stag!='u'))
        jb=int(a['jts']); je=min(int(a['jte']),int(a['jde'])-(stag!='v'))
        for axis in 'xy':
            lo,hi=(ib,ie) if axis=='x' else (jb,je)
            start,end=('ids','ide') if axis=='x' else ('jds','jde')
            stagger=(axis=='x' and stag=='u') or (axis=='y' and stag=='v')
            if a['open_'+axis+'s'] or a['specified'] or a['nested']:lo=max(lo,int(a[start])+1)
            if a['open_'+axis+'e'] or a['specified'] or a['nested']:hi=min(hi,int(a[end])-(1 if stagger else 2))
            if axis=='x' and a['periodic_x']:lo=int(a['its']);hi=min(int(a['ite']),int(a['ide'])-(stag!='u'))
            if axis=='x':ib,ie=lo,hi
            else:jb,je=lo,hi
        return tuple(np.int32(v) for v in (nz,ny,nx,ib-int(a['ims']),ie-int(a['ims'])+1,jb-int(a['jms']),je-int(a['jms'])+1))
    def call(name,args,*,cells=False,stag=''):
        args=[a[x] if isinstance(x,str) else x for x in args]
        get_kernel('km5',name)(((nx+127)//128,ny,nz if cells else 1),(128,1,1),tuple(args)+bounds(stag))
    def f(x):return np.float32(x)
    def i(x):return np.int32(x)
    a['zfull']=cp.zeros_like(dummy);a['grad']=cp.zeros_like(dummy)
    routine=path.name.split('_in_')[0]
    theta='th_phy' if routine=='nonlocal_flux' else 'theta'
    if routine in ('nonlocal_flux','free_atmos_length','meso_length_scale','tke_km'):
        call('km5_heights_gradient',['rdzw',a.get('rdz',dummy),theta,a.get('p8w',dummy),a.get('t8w',dummy),'zfull','grad'])
    if routine=='nonlocal_flux':
        call('km5_nonlocal',['zfull',theta,'u_phy','v_phy','qv','rho','ust','hfx','br','msftx','msfty','wspd','nlflux','gamu','gamv','hpbl','kpbl','dx','dy','ep1','karman','isfflx','tke_heat_flux'])
    elif routine=='free_atmos_length':
        call('km5_free_length',['zfull','rdzw',theta,'tke','elmin'],cells=True)
    elif routine=='meso_length_scale':
        call('km5_meso_length',['zfull','rdzw','grad','tke','qv','rho','rmol','hpbl','elmin','hfx','dlk','isfflx','tke_heat_flux',f(0.4)])
    elif routine=='tke_km':
        seed=f(1e-6 if a['isfflx']==0 and a['tke_drag_coeffic']<1e-10 and a['tke_heat_flux']<1e-10 else 0)
        call('km5_coefficients',[theta,'tke','grad','rdzw','defor11','defor22','defor12','zx','zy','msftx','msfty','hpbl','dlk','xkmh','xkhh','xkmv','xkhv','xkmv_meso','dx','dy','c_s','c_k',f(1/3),seed,i(nx),i(ny),i(nx),i(ny)],cells=True)
    elif routine=='tke_rhs':
        call('km5_tke_source',[theta,'tke','bn2','defor11','defor22','defor33','defor12','defor13','defor23','u','v','xkmh','xkmv','xkhv','mu','c1','c2','rdzw','msftx','msfty','ust','hfx','qv','rho','nlflux','hpbl','dlk','l_diss','tendency','dx','dy','dt','c_k','tke_drag_coeffic','tke_heat_flux','isfflx',i(nx),i(ny),i(nx),i(ny)],cells=True)
    elif routine=='vertical_diffusion_implicit':
        a['solveq']=cp.zeros_like(dummy);a['solved']=cp.zeros_like(dummy)
        for component,stag in enumerate(('u','v')):
            call('km5_implicit_momentum',['u_2','v_2','r'+stag+'_tendf','xkmv','xkmv_meso','rho','rdz','mu','ust','gam'+stag,'fnm','fnp','dnw','c1h','c2h','solveq','solved','dt',i(component),i(nx),i(ny),i(nx),i(ny)],stag=stag)
        call('km5_implicit_w',['w_2','rw_tendf','xkmh','rho','rdzw','dn','mu','c1f','c2f','solveq','solved','dt'])
        rows=[('thp','rt_tendf',2,'t_base'),('tke','tke_tendf',3,None)]
        for family in ('moist','scalar','chem','tracer'):
            for sp in range(int(a['param_first_scal']),int(a['n_'+family])+1):
                if family=='scalar' and sp in [int(a[x]) for x in ('p_qns','p_qnr','p_qng','p_qnh','p_qt','p_qvolg')]:continue
                kind=1 if family=='moist' and sp==int(a['p_qv']) else 0
                rows.append((f'{family}_{sp:02d}',f'{family}_tendf_{sp:02d}',kind,'qv_base' if kind==1 else None))
        for var,tend,kind,base in rows:
            call('km5_implicit_scalar',[var,tend,a[base] if base else dummy,'xkmv' if kind==3 else 'xkhv','rho','mu','c1h','c2h','fnm','fnp','dnw','rdz','nlflux','l_diss','qv','theta','hfx','qfx','solveq','solved','dt',f(461.6)/f(287.0),i(kind),i(base is not None and not a['mix_full_fields']),'use_theta_m'])
    else:raise ValueError(routine)
    cp.cuda.get_current_stream().synchronize()
    fields={}
    for name,expected_array in expected.items():
        if not isinstance(expected_array,np.ndarray):continue
        actual=cp.asnumpy(a[name])
        differing=actual.view('u4')!=expected_array.view('u4')
        compared_words=int(actual.size)
        if raw.get('_interior_only', 0) and actual.ndim >= 2:
            _,_,_,ib,ie,jb,je=bounds()
            region=np.zeros(actual.shape, bool)
            region[...,jb:je,ib:ie]=True
            differing &= region
            compared_words=int(np.count_nonzero(region))
        n=int(np.count_nonzero(differing))
        info={'different':n,'words':compared_words}
        if n:
            idx=tuple(np.argwhere(differing)[0]);info.update(first_index=[int(x) for x in idx],actual=float(actual[idx]),expected=float(expected_array[idx]))
        fields[name]=info
    return {'capture':str(path),'fields':fields,'different':sum(x['different'] for x in fields.values())}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--json',type=Path,required=True);p.add_argument('--routine');args=p.parse_args()
    receipts=[]
    for path in sorted(p for p in args.directory.glob((args.routine or '*')+'_in_*')
                       if p.suffix in ('.bin', '.npz')):
        row=replay(path);receipts.append(row)
        print(path.name,row['different'],{k:v for k,v in row['fields'].items() if v['different']},flush=True)
        args.json.write_text(json.dumps(receipts,indent=2)+'\n')
    if not receipts:raise SystemExit('no captures')
    raise SystemExit(any(x['different'] for x in receipts))
