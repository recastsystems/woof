"""Grade RAD_FLG packing and carriers against its unmodified Fortran entry.

Synthetic driver regression. The separate capture replay is the real-input proof.
"""
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime
import argparse,ctypes as ct,json,hashlib
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('library');p.add_argument('receipt');p.add_argument('--icloud',type=int,default=1);a=p.parse_args()
    import cupy as cp
    from woof.config import RunConfig
    from woof.core.flg_radiation import FLGRadiation
    from woof.core.rrtm_lw import _t8w_columns
    from woof.core import constants as c
    from woof.core.swint import calendar_scalars
    n=8;nz=24
    pressure=np.tile(np.geomspace(100000,1500,nz+1,dtype=np.float32),(n,1))
    temp=np.tile(np.linspace(294,215,nz,dtype=np.float32),(n,1))
    height=np.tile(np.linspace(0,20000,nz+1,dtype=np.float32),(n,1))
    qv=np.tile(np.geomspace(.012,1e-6,nz,dtype=np.float32),(n,1))
    zero=np.zeros((n,nz),np.float32);qc=zero.copy();qi=zero.copy()
    qc[1::2,4:8]=1.e-4;qi[2::3,16:20]=2.e-5
    lat=np.array([-70,-40,-20,20,40,60,70,45],np.float32);lon=np.linspace(-140,140,n,dtype=np.float32)
    def model(value):return cp.asarray(value.T.reshape(value.shape[1],1,n))
    exner=(.5*(pressure[:,:-1]+pressure[:,1:])/np.float32(100000))**np.float32(c.RD/c.CP)
    rho=np.ones_like(temp)
    atmosphere=dict(pressure=model(.5*(pressure[:,:-1]+pressure[:,1:])),temperature=model(temp),
                    p_interface=model(pressure),z_interface=model(height),exner=model(exner),rho=model(rho))
    state=SimpleNamespace(**{key:model(value) for key,value in dict(qv=qv,qc=qc,qr=zero,qi=qi,qs=zero,qg=zero).items()},
        fnm=cp.full(nz+1,.5,cp.float32),fnp=cp.full(nz+1,.5,cp.float32),elapsed_seconds=36000.)
    fields={name:cp.full((1,n),value,cp.float32) for name,value in (('albedo',.15),('emiss',.98),('tsk',295.))}
    cfg=RunConfig(nx=n,ny=1,nz=nz,dx=30000,dy=30000,ztop=20000,dt=180,run_seconds=10800,mp_physics=6,
                  ra_lw_physics=7,ra_sw_physics=7,radt_minutes=30,icloud=a.icloud)
    start=datetime(2001,6,15)
    adapter=FLGRadiation(start,lat,lon)
    # Warm compilation and the CUDA context before the no-readback tripwire.
    got=adapter(atmosphere=atmosphere,fields=fields,state=state,cfg=cfg)
    cp.cuda.get_current_stream().synchronize()
    original=cp.asnumpy
    def forbidden(*args,**kwargs):raise AssertionError('host readback in FLG forecast adapter')
    cp.asnumpy=forbidden
    try:got=adapter(atmosphere=atmosphere,fields=fields,state=state,cfg=cfg)
    finally:cp.asnumpy=original
    adapter.column.check_errors()
    # Native explicit-shape arrays use i,k,j Fortran order, including a zero
    # top humidity ghost. Native RAD_FLG floors that ghost to 1.e-20.
    keep=[]
    def arr(v):
        b=np.asfortranarray(v,dtype=np.float32);keep.append(b);return b.ctypes.data_as(ct.POINTER(ct.c_float))
    def field(v,top=0.):return arr(np.concatenate([v,np.full((n,1),top,np.float32)],axis=1))
    def real(v):b=ct.c_float(float(v));keep.append(b);return ct.byref(b)
    def integer(v):b=ct.c_int(int(v));keep.append(b);return ct.byref(b)
    lib=ct.CDLL(str(Path(a.library).resolve()));native=lib.__module_ra_flg_MOD_rad_flg;native.restype=None
    tw=cp.asnumpy(_t8w_columns(cp,cp.asarray(temp),cp.asarray(height),state.fnm,state.fnp))
    julian,xtime,gmt=calendar_scalars(start,state.elapsed_seconds)
    scalar=cp.asnumpy(adapter.solar)
    from woof.core.rrtmg_legacy import _DEGRAD
    args=[arr(pressure),field(zero),arr(tw),field(temp),field(exner,1.),field(zero),real(c.G),real(c.CP),
          arr(np.full(n,.15,np.float32)),arr(np.full(n,295.,np.float32)),field(qv),field(qi),field(qc),field(zero),field(zero),field(zero)]
    args += [integer(1) for _ in range(6)]+[integer(0),field(zero),arr(np.full(n,.98,np.float32)),field(rho),field(np.ones_like(temp)),
            real(scalar[1]),real(scalar[0]),real(xtime),arr(lon),arr(lat),integer(int(julian)+1),real(gmt),real(30),real(_DEGRAD),real(180)]
    args += [integer(v) for v in (1,n+1,1,2,1,nz+1,1,n,1,1,1,nz+1,1,n,1,1,1,nz)]
    ref={key:np.zeros((n,nz+1) if key in ('total','rthratensw','rthratenlw') else n,dtype=np.float32,order='F')
         for key in ('native_uswtop','olr','gsw','glw','swdown','total','rthratensw','rthratenlw','swddir','swddif','swddni')}
    args += [arr(v) for v in ref.values()]
    native(*args)
    # arr preserves existing F-contiguous output buffers.
    receipt={'kind':'synthetic native RAD_FLG forecast-adapter regression','fields':{},'no_host_readback':True,
             'native_uswtop':'excluded: WRF module_ra_flg.F:8960 assigns the whole array once per column; CUDA publishes each column fus(1)'}
    for key in ('olr','gsw','glw','swdown','rthratensw','rthratenlw','swddir','swddif','swddni'):
        value=cp.asnumpy(got.flux_diagnostics['SWDDNI'] if key=='swddni' else getattr(got,key))
        actual=value.reshape((-1,n)).T if key.startswith('rth') else value.ravel()
        expected=ref[key][:,:nz] if key.startswith('rth') else ref[key]
        misses=np.argwhere(actual.view(np.uint32)!=expected.view(np.uint32))
        receipt['fields'][key]={'words':expected.size,'differing_words':len(misses),'first_mismatch':misses[:1].tolist(),
            'max_absolute_difference':float(np.max(abs(actual-expected))),'native_sha256':hashlib.sha256(expected.tobytes()).hexdigest(),
            'cuda_sha256':hashlib.sha256(actual.tobytes()).hexdigest()}
    Path(a.receipt).write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
    if any(v['differing_words'] for v in receipt['fields'].values()):raise SystemExit(1)

if __name__=='__main__':main()
