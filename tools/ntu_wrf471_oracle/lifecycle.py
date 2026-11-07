#!/usr/bin/env python3
"""Exercise checkpoint continuation and independent ensemble slabs on real inputs.

This is a module/lifecycle qualification, not a second weather forecast. The
read-only native thermodynamic inputs are held for three MP_NTU calls.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from replay import read_capture, ENGINE_NAMES


def exercise(capture, scratch, receipt):
    import cupy as cp
    from woof.config import RunConfig
    from woof.core import ntu
    from woof.core.state import DomainState
    from woof.core.device_inventory import state_array_shapes
    from woof.io import restart
    from woof.ensemble.batch_state import PreparedHostMember, BatchedDomainState
    from woof.ensemble.batch_dycore import member_domain_view

    meta, host, surface_host = read_capture(capture)
    cfg = RunConfig(nx=meta['nx'], ny=meta['ny'], nz=meta['nz'], dx=30000,
                    dy=30000, ztop=20000, dt=meta['dt'], run_seconds=0,
                    moist=True, mp_physics=56, ntu_ccnty=meta['ccnty'])
    shape=(cfg.nz,cfg.ny,cfg.nx)
    names=('thp','qv','qc','qr')+ntu.NTU_SPECIES
    fixed={name:cp.asarray(host[key]) for name,key in
           (('p','P'),('dz','DZ'),('pii','PII'),('w','W'))}
    slots=dict(zip(ntu._FIELD2,ntu.PRECIPITATION_SLOTS))
    def setup(state):
        for i,name in enumerate(restart.STATE_SETUP_ARRAYS):
            getattr(state,name).fill(np.float32(i+1))
        state.thb.fill(0)
    def initial(state):
        setup(state)
        state.thp.set(host['TH'])
        for key,value in host.items():
            name=ENGINE_NAMES.get(key,key.lower())
            if name in names:getattr(state,name).set(value)
        for key,value in surface_host.items():
            name=ENGINE_NAMES.get(key,key.lower())
            state.scratch((cfg.ny,cfg.nx),slots[name]).set(value)
    def step(state):
        fields=dict(fixed,theta=state.thp)
        fields.update({name:getattr(state,name) for name in names[1:]})
        surface={name:state.scratch((cfg.ny,cfg.nx),slot) for name,slot in slots.items()}
        ntu.launch(fields,surface,cfg.dt,ccnty=cfg.ntu_ccnty,
                   workspace=state.scratch((ntu.WORK_VOLUMES,*shape),'ntu_workspace'))
    def words(state):
        values={n:getattr(state,n).get() for n in names}
        values.update({n:state.scratch((cfg.ny,cfg.nx),slot).get() for n,slot in slots.items()})
        return values
    def compare(left,right):
        a,b=words(left),words(right)
        return dict(words=sum(v.size for v in a.values()),
                    differing_words=sum(int(np.count_nonzero(a[n].view('u4')!=b[n].view('u4'))) for n in a),
                    nonfinite=sum(int((~np.isfinite(v)).sum()) for v in b.values()),
                    sha256={n:hashlib.sha256(v.tobytes()).hexdigest() for n,v in b.items()})
    source=DomainState(cfg);initial(source)
    before=words(source)
    step(source)
    evolved=words(source)
    changed=[n for n in names if before[n].tobytes()!=evolved[n].tobytes()]
    scratch.mkdir(parents=True,exist_ok=True)
    checkpoint=restart.write_restart(scratch/'real-ntu.npz',source,cfg)
    resumed=DomainState(cfg);setup(resumed)
    restart.restore_restart(checkpoint,resumed,cfg)
    restore=compare(source,resumed)
    for _ in range(2):
        step(source);step(resumed)
    continuation=compare(source,resumed)
    # Pack each member from the same real, evolved state, then launch actual
    # native physics on its independent GPU member slab.
    inventory=state_array_shapes(cfg)
    arrays={n:getattr(resumed,n).get() for n in inventory}
    excluded={'physics','lateral_boundaries','_scratch','_scratch_arena',
              '_host_setup_state','_phb_host'}
    scalars={n:v for n,v in vars(resumed).items() if n not in inventory and n not in excluded}
    clock=dict(ticks=0,step_ticks=int(cfg.dt),tick_den=1,run_ticks=0,step_count=0,
               dt_fp32=np.float32(cfg.dt),dtbc_fp32=np.float32(0))
    prepared=PreparedHostMember(cfg,arrays,scalars,clock,
             scratch={slot:resumed._scratch[slot].get() for slot in ntu.PRECIPITATION_SLOTS})
    batch=BatchedDomainState.from_prepared((prepared,prepared),array_module=cp,
                    available_bytes=cp.cuda.runtime.memGetInfo()[0])
    views=[member_domain_view(batch,i) for i in range(2)]
    untouched=words(views[1])
    step(views[0])
    isolation=all(v.tobytes()==words(views[1])[n].tobytes() for n,v in untouched.items())
    step(views[1]);step(resumed)
    members=[compare(resumed,v) for v in views]
    result=dict(input=str(capture),input_sha256=hashlib.sha256(capture.read_bytes()).hexdigest(),
                metadata=meta,device=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
                changed_prognostics=changed,restore=restore,continuation=continuation,
                ensemble_members=members,ensemble_isolation=isolation,
                scope='real-capture module continuation; fixed native p/dz/pii/w; no dycore forecast',
                passed=restore['differing_words']==0 and continuation['differing_words']==0
                       and all(m['differing_words']==0 and m['nonfinite']==0 for m in members) and isolation)
    receipt.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('changed_prognostics','ensemble_isolation','passed')},indent=2))
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('capture',type=Path);ap.add_argument('scratch',type=Path);ap.add_argument('receipt',type=Path)
    args=ap.parse_args()
    raise SystemExit(0 if exercise(args.capture,args.scratch,args.receipt)['passed'] else 1)
