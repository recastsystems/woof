"""Measure production GPU adapters on profiles captured from a real WRF run.

Run through the shared GPU helper. Tiling repeats observed columns only
to match the forecast's column count; it creates no new science cases.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np


def main():
    import cupy as cp
    from woof.core.cloud_fraction import cal_cldfra3
    from woof.core.farms import farms_surface_columns
    from woof.core.rrtmgp import cal_cldfra1
    from woof.core.kernels import get_kernel, module_source
    from woof.core.health_ledger import HealthLedger, deferring
    from woof.wrf_exact import ENABLED

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    parser.add_argument('receipt', type=Path)
    parser.add_argument('--columns', type=int, default=4380)
    args = parser.parse_args()
    if ENABLED:
        raise ValueError('speed qualification uses default fast numerics')
    n = args.columns
    records = [p.read_bytes() for p in sorted(args.capture.glob('cloud-*.bin'))]
    nz = int(np.frombuffer(records[0][:4], '<i4')[0])
    cloud = np.stack([np.frombuffer(r[4:], '<f4') for r in records])
    idx = np.arange(n) % len(records)
    values = cloud[idx]
    profiles = [cp.asarray(np.ascontiguousarray(values[:,i*nz:(i+1)*nz]))
                for i in range(7)]
    land, grid = [cp.asarray(np.ascontiguousarray(values[:,7*nz+i])) for i in (0,1)]
    qv,qc,qi,qs,dz,p,t = profiles
    cf3 = lambda: cal_cldfra3(*profiles,land,grid)
    cf1 = lambda args=(qv,qc,qi,qs,t,p): cal_cldfra1(*args,mp_physics=6)

    records = [p.read_bytes() for p in sorted(args.capture.glob('farms-*.bin'))]
    nz,hc,hi,hs = np.frombuffer(records[0][:16], '<i4')
    farms = np.stack([np.frombuffer(r[16:], '<f4') for r in records])
    farms = farms[farms[:,11*nz+6] == farms[0,11*nz+6]]
    idx = np.arange(n) % len(farms)
    values = farms[idx]
    volumes = [cp.asarray(np.ascontiguousarray(values[:,i*nz:(i+1)*nz].T))
               for i in range(11)]
    surfaces = [cp.asarray(np.ascontiguousarray(values[:,11*nz+i])) for i in range(6)]
    p,rho,dz,qv,qi,qs,qc,rc,ri,rs,cf = volumes
    alb,ssa,asy,aod,ang,cz = surfaces
    out = cp.empty((6,n),cp.float32)
    julian = values[0,11*nz+6]
    # All recorded Julian values belong to this run's first Solar_step.
    assert np.all(values[:,11*nz+6] == julian)
    far = lambda: farms_surface_columns(p,rho,dz,alb,ssa,asy,aod,ang,cz,
        qv,qi,qs,qc,rc,ri,rs,cf,julian=julian,
        has_reqc=int(hc),has_reqi=int(hi),has_reqs=int(hs),out=out)
    interp = get_kernel('swint','swint_interp')
    one = cp.ones(n,cp.float32)
    zero = cp.zeros(n,cp.float32)
    interp_out = [cp.empty(n,cp.float32) for _ in range(5)]
    # Existing interpolation's first-guess branch on the same observed sun.
    sw1 = lambda: interp(((n+127)//128,), (128,),
        (np.int64(n),cz,cz,out[1],zero,one,out[0],zero,one,alb,*interp_out))

    def measure(call):
        for _ in range(3): call()
        start,end = cp.cuda.Event(),cp.cuda.Event()
        start.record()
        for _ in range(100): call()
        end.record()
        end.synchronize()
        return float(cp.cuda.get_elapsed_time(start,end)/100)

    with deferring(HealthLedger(label='cloud/FARMS timing')):
        timings = {name: measure(call) for name,call in
                   (('icloud1',cf1),('icloud3',cf3),('farms',far),('swint1_interp',sw1))}
    result = {'columns':n,'levels':int(nz),'strict':False,
              'device':cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
              'milliseconds':timings,
              'cloud_ratio':timings['icloud3']/timings['icloud1'],
              'farms_ratio':timings['farms']/timings['swint1_interp'],
              'scope':'GPU events, production module adapters; cosine and driver packing excluded from both shortwave alternatives',
              'cloud_scope':'cloud wrappers include device packing/status work; no per-call host read',
              'source_sha256':{name:hashlib.sha256(module_source(name).encode()).hexdigest()
                               for name in ('cldfra3','farms','swint')}}
    args.receipt.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
