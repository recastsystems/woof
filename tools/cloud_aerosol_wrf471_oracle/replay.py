"""Replay bounded binary records captured inside an actual strict WRF run.

Run through the GPU pool with GPUWM_WRF_EXACT=1. Input records are native
little-endian float32, written by the lane's instrument_native.py. No
synthetic columns are included in this replay denominator.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import numpy as np


def metrics(actual, expected):
    a, b = np.ascontiguousarray(actual), np.ascontiguousarray(expected)
    assert a.dtype == b.dtype == np.float32 and a.shape == b.shape
    different = a.view(np.uint32) != b.view(np.uint32)
    return {"words": int(a.size), "different": int(different.sum()),
            "max_abs": float(np.max(np.abs(a.astype('f8')-b))),
            "actual_sha256": hashlib.sha256(a.tobytes()).hexdigest(),
            "native_sha256": hashlib.sha256(b.tobytes()).hexdigest()}


def timed(call):
    import cupy as cp
    call()
    start, end = cp.cuda.Event(), cp.cuda.Event()
    start.record()
    for _ in range(40):
        call()
    end.record()
    end.synchronize()
    return float(cp.cuda.get_elapsed_time(start, end) / 40)


def cloud(paths):
    import cupy as cp
    from woof.core.cloud_fraction import cal_cldfra3
    records = []
    for path in paths:
        raw = path.read_bytes()
        nz = int(np.frombuffer(raw[:4], '<i4')[0])
        values = np.frombuffer(raw[4:], '<f4').copy()
        assert values.size == 11 * nz + 4, path
        records.append((nz, values))
    assert len({r[0] for r in records}) == 1
    nz = records[0][0]
    inputs = np.stack([r[1][:7*nz].reshape(7,nz) for r in records])
    knobs = np.stack([r[1][7*nz:7*nz+4] for r in records])
    assert len(set(map(tuple, knobs[:,2:]))) == 1
    args = [cp.asarray(np.ascontiguousarray(inputs[:,i])) for i in range(7)]
    land, dx = (cp.asarray(np.ascontiguousarray(knobs[:,i])) for i in (0,1))
    expected = np.stack([r[1][7*nz+4:].reshape(4,nz) for r in records])
    def run():
        return cal_cldfra3(*args,land,dx,modify_qvapor=bool(knobs[0,2]),max_relh=float(knobs[0,3]))
    got = np.stack([cp.asnumpy(a) for a in run()],axis=1)
    return {"columns": len(records), "levels": nz,
            "fields": {name:metrics(got[:,i],expected[:,i])
                       for i,name in enumerate(('cldfra','qv','qc','qi'))},
            "input_buffers":metrics(np.stack([cp.asnumpy(a) for a in args],axis=1),inputs),
            "module_ms":timed(run)}


def farms(paths):
    import cupy as cp
    from woof.core.farms import farms_surface_columns, FARMS_FIELDS
    records = []
    for path in paths:
        raw = path.read_bytes()
        header = np.frombuffer(raw[:16], '<i4').copy()
        nz = int(header[0])
        values = np.frombuffer(raw[16:], '<f4').copy()
        assert values.size == 11*nz + 13, path
        records.append((header, values))
    assert len({tuple(r[0]) for r in records}) == 1
    nz, hc, hi, hs = records[0][0]
    values = np.stack([r[1] for r in records])
    volume = [cp.asarray(np.ascontiguousarray(values[:,i*nz:(i+1)*nz].T)) for i in range(11)]
    surface = [cp.asarray(np.ascontiguousarray(values[:,11*nz+i])) for i in range(6)]
    julian = values[:,11*nz+6]
    p,rho,dz,qv,qi,qs,qc,rc,ri,rs,cf = volume
    alb,ssa,asy,aod,ang,cz = surface
    groups = []
    for day in np.unique(julian):
        indices = np.flatnonzero(julian == day)
        v = [cp.ascontiguousarray(a[:,indices]) for a in volume]
        s = [cp.ascontiguousarray(a[indices]) for a in surface]
        groups.append((indices,day,v,s,cp.empty((6,len(indices)),dtype=cp.float32)))
    def run():
        outputs = []
        for indices,day,v,s,out in groups:
            p,rho,dz,qv,qi,qs,qc,rc,ri,rs,cf = v
            alb,ssa,asy,aod,ang,cz = s
            outputs.append(farms_surface_columns(p,rho,dz,alb,ssa,asy,aod,ang,cz,
                qv,qi,qs,qc,rc,ri,rs,cf,julian=day,has_reqc=int(hc),
                has_reqi=int(hi),has_reqs=int(hs),out=out))
        return outputs
    got = np.empty((6,len(records)),dtype=np.float32)
    for group,output in zip(groups,run()):
        got[:,group[0]] = cp.asnumpy(output)
    expected = np.ascontiguousarray(values[:,-6:].T)
    return {"columns":len(records),"levels":int(nz),"integrated_levels":int(nz-1),
            "daylight_columns":int(np.count_nonzero(cp.asnumpy(cz)>0)),
            "fields":{name:metrics(got[i],expected[i]) for i,name in enumerate(FARMS_FIELDS)},
            "module_ms":timed(run)}


def main():
    import cupy as cp
    from woof.core.kernels import module_source, module_options
    from woof.wrf_exact import ENABLED, effective_options, compile_receipt
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture',type=Path)
    parser.add_argument('receipt',type=Path)
    args=parser.parse_args()
    if not ENABLED:
        parser.error('strict replay requires GPUWM_WRF_EXACT=1')
    result={"strict":os.environ.get('GPUWM_WRF_EXACT')=='1',
            "device":cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
            "capture_files":{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted(args.capture.glob('*.bin'))},
            "schemes":{},"sources":{}}
    for scheme, replay in (('cloud',cloud),('farms',farms)):
        paths=sorted(args.capture.glob(scheme+'-*.bin'))
        if paths:
            result['schemes'][scheme]=replay(paths)
    for unit in ('cldfra3','farms'):
        result['sources'][unit]={"sha256":hashlib.sha256(module_source(unit).encode()).hexdigest(),
                                 "options":effective_options(module_options(unit)) if ENABLED else module_options(unit)}
    result['compile_receipt']=compile_receipt()
    result['passed']=set(result['schemes']) == {'cloud', 'farms'} and all(
        field['different']==0 for scheme in result['schemes'].values()
        for field in scheme['fields'].values()) and (
            result['schemes']['cloud']['input_buffers']['different'] == 0)
    args.receipt.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['schemes'],indent=2))
    return 0 if result['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
