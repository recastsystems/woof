"""WRF 4.7.1 Deng driver: resident prognostic cloud state and CUDA columns."""
from pathlib import Path
import hashlib
import cupy as cp
import numpy as np
from woof.core.deng_contract import ARGUMENTS, arena_slots


def step(driver, a, state, cfg):
    from woof.core.cam5_runtime import workspace
    from woof.core.physics import couple_column_tendencies
    from woof.core.kernels import load_module
    f=driver.fields;owner=workspace(driver)
    nz,ny,nx=state.p.shape;ncol=ny*nx
    table_path=Path(__file__).parents[1]/'data/deng_tables.bin'
    if not hasattr(driver,'_deng_tables'):
        payload=table_path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != 'dd7d5e24bb57816816f68b790b46c4c7341107edf0c2f22155df601ac5bf301a':
            raise ValueError('Deng source-bound saturation table hash differs')
        driver._deng_tables=cp.asarray(np.frombuffer(payload,dtype='<f4').copy())
    from woof.core.swint import calendar_scalars
    _,xtime,gmt=calendar_scalars(driver.tmn_start_time,state.elapsed_seconds)
    elapsed=float(state.elapsed_seconds)
    constants=dict(xlv=2.5e6,xls=2.85e6,xlv0=3.15e6,xlv1=2370.,xls0=2.905e6,xls1=259.532,
                   cp=1004.5,r=287.,g=9.81,svp1=.6112,svp2=17.67,svp3=29.65,svpt0=273.15)
    inputs={name:np.float32(value) for name,value in constants.items()}
    inputs.update(dt=np.float32(cfg.dt),ktau=np.int32(round(elapsed/cfg.dt)+1),dx=np.float32(cfg.dx),
                  xtime=np.float32(xtime),gmt=np.float32(gmt),adapt_step_flag=np.bool_(False))
    inputs.update(xlong=f['deng_xlong'],ht=a['z_interface'][0],pblh=f['pblh'],u=a['u'],v=a['v'],w=state.w,
                  th=a['theta'],t=a['temperature'],qv=a['qv'],qc=a['qc'],qr=state.qr,
                  dz8w=a['dz'],pcps=a['pressure'],rho=a['rho'],z_at_w=a['z_interface'],pi=a['exner'],
                  tke=f['tke_myj'],kth=f['exch_h'],bbls=f['el_myj'],
                  ten_radl=driver.rthratenlw,ten_rads=driver.rthratensw)
    dsigma=owner.array('deng/dsigma',(nz+1,),cp.float32)
    dsigma[:nz]=state.dnw;dsigma[nz]=np.float32(0.)
    inputs['dsigma']=dsigma
    slots=arena_slots(nz)
    arena=owner.array('deng/arena',(slots,ncol),cp.float32)
    status=owner.array('deng/status',(1,),cp.int32,clear=True)
    args=tuple(inputs[n] if n in inputs else f[n] for n in ARGUMENTS)
    load_module('deng').get_function('deng_column')(((ncol+63)//64,),(64,),
        (np.int32(nz),np.int32(ncol),arena,np.int32(slots),driver._deng_tables,*args,status))
    from woof.core import health_ledger
    def refuse(line):
        raise RuntimeError(f'Deng native STOP or column workspace overflow: source line {line}')
    error=health_ledger.read_status(status,site='deng',describe=refuse)
    if error:
        refuse(error)
    driver.shallow_tendencies=couple_column_tendencies(state,cfg,ru=f['rushten'],rv=f['rvshten'],
        rtheta=f['rthshten'],rqv=f['rqvshten'],rqc=f['rqcshten'],rqr=f['rqrshten'])
    increment=cp.maximum(f['rainsh'],np.float32(0.))
    driver._pending_rainbl+=increment
    if driver.ruc_params is not None or driver.noahmp_params is not None:f['surface_raincv']+=increment
    driver.call_counts['shallow_cumulus']=driver.call_counts.get('shallow_cumulus',0)+1
