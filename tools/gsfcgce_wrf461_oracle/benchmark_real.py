"""Warm stage timings on one fixed batch of captured real columns.

Captured columns are repeated spatially (200 rows by default) to time a
mesh-sized launch. Both schemes receive the same original inputs, restored outside the
event interval before every sample. This is a stage comparison, not another
forecast or a numerical proof campaign.
"""
from pathlib import Path
import argparse,hashlib,json,statistics
import numpy as np
from replay_real import decode,IN3,IN2,OUT3,OUT2


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture',type=Path)
    parser.add_argument('receipt',type=Path)
    parser.add_argument('--hail',type=int,choices=(0,1),default=0)
    parser.add_argument('--ice',type=int,choices=(0,1,2,3),default=0)
    parser.add_argument('--copies',type=int,default=200)
    args=parser.parse_args()
    import cupy as cp
    from woof.core.mp_column_schemes import MP_COLUMN_SCHEMES,launch_column,launch_refl
    from woof.core.wsm6 import launch_wsm6
    from woof.core.refl import launch_refl10cm_wsm6
    from woof import wrf_exact
    prefix=args.capture.with_suffix('')
    meta=prefix.with_suffix('.meta').read_text().split()
    nx,nz=map(int,meta[:2]);dt=float(meta[2])
    initial=decode(prefix.with_suffix('.in'),IN3,IN2,nx,nz)
    initial.update(decode(prefix.with_suffix('.pre'),OUT3,OUT2,nx,nz))
    copies=args.copies
    controls=(args.hail,args.ice)
    original={k:cp.asarray(np.repeat(v,copies,axis=-2)) for k,v in initial.items()}
    state={k:v.copy() for k,v in original.items()}
    radii={k:cp.empty_like(state['th']) for k in ('effc','effi','effs')}
    sediment_vt=cp.empty_like(state['th'])
    temperature=original['th']*original['pii']
    def reset():
        for k,v in state.items():cp.copyto(v,original[k])
    def gce():
        launch_column(MP_COLUMN_SCHEMES[97],state['th'],state,state['rho'],state['pii'],
            state['p'],state['z'],state['dz8w'],state['ht'],
            *(state[k] for k in OUT2),np.float32(dt),controls,sediment_vt=sediment_vt)
    def wsm6():
        launch_wsm6(*(state[k] for k in ['th','qv','qc','qr','qi','qs','qg','rho','pii','p','dz8w']),
                     *(state[k] for k in OUT2),dt,**radii)
    def radar():
        launch_refl(MP_COLUMN_SCHEMES[97],*(state[k] for k in ['th','pii','p','qv','qr','qs','qg','refl']),controls)
    def wsm6_radar():
        launch_refl10cm_wsm6(state['qv'],state['qr'],state['qs'],state['qg'],
                            temperature,state['p'],state['refl'])
    results={}
    for name,func in [('gce_column',gce),('wsm6_column',wsm6),
                      ('gce_reflectivity',radar),('wsm6_reflectivity',wsm6_radar)]:
        reset();func();cp.cuda.get_current_stream().synchronize()
        samples=[]
        for _ in range(5):
            reset()
            start,end=cp.cuda.Event(),cp.cuda.Event()
            start.record();func();end.record();end.synchronize()
            samples.append(cp.cuda.get_elapsed_time(start,end))
        results[name]={'cuda_event_ms':samples,'median_ms':statistics.median(samples)}
    result={'strict':wrf_exact.ENABLED,'gpu':cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
            'shape':[nz,copies,nx],'controls':controls,'dt':dt,'capture':str(args.capture),
            'capture_sha256':hashlib.sha256(prefix.with_suffix('.in').read_bytes()).hexdigest(),
            'scope':'fixed captured inputs repeated spatially, one warmup and five restored stage samples; WSM6 temperature prepared outside events; shared GPU, no exclusive lease',
            'stages':results,'column_ratio':results['gce_column']['median_ms']/results['wsm6_column']['median_ms'],
            'reflectivity_ratio':results['gce_reflectivity']['median_ms']/results['wsm6_reflectivity']['median_ms']}
    args.receipt.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
