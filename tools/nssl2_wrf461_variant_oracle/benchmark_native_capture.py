"""Paired stage timing on replicated columns captured from real native WRF.

This is a microphysics stage benchmark, not a model or a substitute forecast.
CUDA events bound production launches. Setup, input tiling and output copies
are outside each stage; the first two samples only warm code and device tables.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import functools
import json
from pathlib import Path
import statistics
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from replay_native_capture import read_capture
import cases
from run_gpu import admit, run_mode


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture',type=Path)
    parser.add_argument('receipt',type=Path)
    parser.add_argument('--columns',type=int,default=90000)
    args=parser.parse_args()
    meta,(before,_)=read_capture(args.capture)
    if args.columns<6 or args.columns%meta['nx']:
        parser.error('columns must be a positive multiple of captured columns')
    import cupy as cp
    import woof.core.nssl2_production_coordinator as coordinator
    import woof.core.nssl2_fused_gs as gs
    import woof.core.nssl2_nucond as nucond
    records=[]
    def timed(name,original):
        @functools.wraps(original)
        def call(*a,**kw):
            start,end=cp.cuda.Event(),cp.cuda.Event()
            start.record()
            result=original(*a,**kw)
            end.record()
            records.append((name,start,end))
            return result
        return call
    for module,name in ((coordinator,'gather_initialize_and_sediment'),
                        (coordinator,'reduce_nssl2_precipitation'),
                        (coordinator,'launch_radardd02'),
                        (coordinator,'launch_effective_radius_concentration'),
                        (coordinator,'scatter_nssl2_driver_workspace'),
                        (gs,'launch_fused_gs'),(nucond,'launch_nucond')):
        setattr(module,name,timed(name,getattr(module,name)))
    repeats=args.columns//meta['nx']
    cells={name:np.tile(a,(repeats,1)) for name,a in before[0].items()}
    surfaces={name:np.tile(a,repeats) for name,a in before[1].items()}
    state={n:cells[n] for n in cases.STATE_FIELDS}
    env={n:cells[n] for n in ('pressure','exner','rho','dz','w')}
    coefficients={'nearest_default_mp18':0.5e9,'candidate_cccn':meta['params'][0]}
    samples={label:[] for label in coefficients}
    # Alternate order so drift in shared-card contention is visible.
    for pair in range(7):
        order=list(coefficients) if pair%2==0 else list(reversed(coefficients))
        for label in order:
            cccn=coefficients[label]
            mode=replace(cases.MODE_BY_LABEL['cccn-low'],cccn=cccn,
                         selectors={'nssl_cccn':cccn})
            resolved=admit(mode)
            records.clear()
            run_mode(mode,resolved,1,meta['dt'],meta['itimestep'],
                     input_case=(state,env),surface_initial=surfaces,
                     diagnostic_initial=cells)
            sample={name:float(cp.cuda.get_elapsed_time(start,end)) for name,start,end in records}
            sample['total_stages']=sum(sample.values())
            samples[label].append(sample)
    stages=samples['candidate_cccn'][-1].keys()
    medians={label:{name:statistics.median(s[name] for s in data[2:]) for name in stages}
             for label,data in samples.items()}
    ratios={name:medians['candidate_cccn'][name]/medians['nearest_default_mp18'][name]
            for name in stages}
    from woof.certify.kernel_manifest import kernel_manifest
    from woof.wrf_exact import ENABLED
    receipt={'source_capture':meta,'columns':args.columns,'levels':meta['nz'],
             'method':'CUDA events around production stage launches; same replicated real columns; first two paired samples excluded',
             'strict_numerics':ENABLED,'coefficients':coefficients,'samples_ms':samples,
             'median_ms':medians,'candidate_over_nearest':ratios,
             'kernel_manifest':kernel_manifest(),
             'device':cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)['name'].decode()}
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'median_ms':medians,'candidate_over_nearest':ratios},indent=2))


if __name__=='__main__': main()
