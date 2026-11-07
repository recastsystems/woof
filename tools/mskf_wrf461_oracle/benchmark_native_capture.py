"""Time MSKF and native-order KF on the same real WRF column capture.

This measures active column calls, including workspace/device launch overhead,
with CUDA events. Input and INOUT restoration is outside each timed interval.
The five observations are a timing sample, not a forecast campaign.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import statistics
import numpy as np
from replay_native_capture import read_capture


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('capture', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--columns', type=int, default=8192)
    args = ap.parse_args()
    import cupy as cp
    from woof.core import mskf, kf
    from woof.core.health_ledger import HealthLedger, deferring
    from woof.numerics import mode
    h, sc, coords, values, initial, expected = read_capture(args.capture)
    nz = int(h[0]); ncol = args.columns
    pick = np.arange(ncol) % len(values)
    def dev3(x):
        return cp.asarray(np.ascontiguousarray(x[pick].T.reshape(nz, 1, ncol)))
    inputs = {name: dev3(values[:, j*nz:(j+1)*nz])
              for j, name in enumerate(mskf.INPUT_3D)}
    for j, name in enumerate(mskf.INPUT_2D):
        inputs[name] = cp.asarray(np.ascontiguousarray(values[pick, 10*nz+1+j].reshape(1,ncol)))
    outputs = {name: dev3(initial[:, j*nz:(j+1)*nz])
               for j, name in enumerate(mskf.OUTPUT_3D)}
    for j, name in enumerate(mskf.OUTPUT_2D):
        outputs[name] = cp.asarray(np.ascontiguousarray(initial[pick,26*nz+j].reshape(1,ncol)))
    outputs['cu_act'] = cp.zeros((1,ncol),dtype=cp.int32)
    # This comparison prices the full physics, not the NCA timer's held calls.
    outputs['nca'].fill(0)
    baseline = {key:value.copy() for key,value in outputs.items()}
    ledger = HealthLedger(label='MSKF/KF stage timing on real WRF inputs')
    def ms():
        mskf.raise_on_status(mskf.launch_mskf(inputs,outputs,dt=float(sc[0]),dx=float(sc[1]),
            cudt_minutes=float(sc[2]),kf_edrates=int(h[10]),phase=mskf.MskfPhase(*map(bool,h[5:9]))))
    phase = (kf.KFPhaseMode.WARM_RAIN if h[5] else
             kf.KFPhaseMode.SEPARATE_ICE_SNOW if h[7] and h[8] else
             kf.KFPhaseMode.SEPARATE_SNOW if h[8] else kf.KFPhaseMode.NO_SEPARATE_SNOW)
    def nearest():
        return kf.launch_kf(*(inputs[key] for key in ('u','v','t','qv','p','rho','pi','dz8w','w0avg')),
             dt=float(sc[0]),dx=float(sc[1]),cudt_minutes=float(sc[2]),
             phase_mode=phase,f_qr=bool(h[6]),cloud_diagnostics=True)
    timings = {}
    for label, launch in [('mskf',ms),('kf',nearest)]:
        with deferring(ledger): launch()
        ledger.drain()
        samples = []
        for _ in range(5):
            for key,value in baseline.items(): cp.copyto(outputs[key],value)
            begin,end = cp.cuda.Event(),cp.cuda.Event()
            with deferring(ledger):
                begin.record(); result=launch(); end.record()
            end.synchronize(); samples.append(float(cp.cuda.get_elapsed_time(begin,end)))
            ledger.drain()
        timings[label] = {'samples_ms':samples,'median_ms':statistics.median(samples)}
    receipt = {'numerics':mode(),'columns':ncol,'nz':nz,'capture':args.capture.name,
               'capture_columns':len(coords),'capture_precipitating_columns':int((expected[:,26*nz]>0).sum()),
               'nca_forced_active':True,'w0avg_updated':False,'timings':timings,
               'mskf_over_kf':timings['mskf']['median_ms']/timings['kf']['median_ms'],
               'gpu':cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}
    args.out.write_text(json.dumps(receipt,indent=2)+'\n'); print(json.dumps(receipt))

if __name__ == '__main__': main()
