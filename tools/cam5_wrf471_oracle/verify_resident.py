"""Prove warmed CAM driver calls capture without host status readbacks."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .capture import layouts, read
from .diagnose import compare

def main():
    ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path)
    ap.add_argument('--out',required=True,type=Path);args=ap.parse_args()
    import cupy as cp
    from woof.core.cam5_workspace import CamWorkspace
    from woof.core.cammgmp import cammgmp_step,CAMMGMP_INOUT_3D,CAMMGMP_INOUT_2D
    from woof.core.camzm import camzm_step
    from woof.core.uwshcu import uwshcu_step
    from woof.core.health_ledger import HealthLedger,deferring
    from woof import numerics
    report=dict(numerics=numerics.receipt(),device=cp.cuda.Device().id,schemes={})
    stream=cp.cuda.Stream(non_blocking=True)
    for key,schema in layouts().items():
        p=next(iter(sorted(args.directory.glob(key+'*-in.bin'))))
        meta,data=read(p,schema['in'])
        _,expected=read(p.with_name(p.name.replace('-in.bin','-out.bin')),schema['out'])
        workspace=CamWorkspace();ledger=HealthLedger(label=key)
        with stream:
            seed={n:cp.asarray(v) for n,v in data.items()}
            dev={n:a.copy() for n,a in seed.items()}
            def reset():
                for name,a in dev.items():cp.copyto(a,seed[name])
            def launch():
                if key=='cu7':
                    return camzm_step(dev,dt=meta['dt'],dx=meta['dx'],cudt=meta['cudt'],
                        itimestep=meta['step'],bl_pbl_physics=meta['pbl'],
                        sf_sfclay_physics=meta['sfc'],diagnostics=True,workspace=workspace)
                if key=='shcu2':
                    out=uwshcu_step(dev,{'cush':dev['cush']},dt=meta['dt'],
                        has_qnc=meta['p_qnc']>=meta['first_scalar'],
                        has_qni=meta['p_qni']>=meta['first_scalar'],workspace=workspace)
                    out['cush']=dev['cush'];return out
                out={n:dev[n] for n in CAMMGMP_INOUT_3D+CAMMGMP_INOUT_2D}
                cammgmp_step(dev,out,dt=meta['dt'],itimestep=meta['step'],
                    cu_coupled=meta['cu']==7,shcu_coupled=meta['shcu']==2,
                    accum_mode=meta['accum'],aitken_mode=meta['aitken'],
                    coarse_mode=meta['coarse'],workspace=workspace)
                return out
            with deferring(ledger):
                # Warm compilation, allocations, pointer tables and ledger kernel.
                out=launch()
            stream.synchronize();ledger.drain()
            warm={n:cp.asnumpy(a) for n,a in out.items()}
            ptrs={n:a.data.ptr for n,a in workspace.arrays.items()}
            with deferring(ledger):
                stream.begin_capture();reset();out=launch();graph=stream.end_capture()
            graph.launch(stream);stream.synchronize();ledger.drain()
            captured={n:cp.asnumpy(a) for n,a in out.items()}
            identity=compare(captured,warm)
            assert identity['differ']==0,identity
            assert ptrs=={n:a.data.ptr for n,a in workspace.arrays.items()}
            # One warmed CUDA-event observation. Includes device input reset.
            start,stop=cp.cuda.Event(),cp.cuda.Event()
            start.record(stream);graph.launch(stream);stop.record(stream)
            stop.synchronize();ledger.drain()
            grade=compare(captured,expected)
        report['schemes'][key]=dict(graph_capture=True,graph_identity=identity,
            native_grade=grade,input_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
            milliseconds_including_device_reset=cp.cuda.get_elapsed_time(start,stop),
            workspace_bytes=sum(a.nbytes for a in workspace.arrays.values()),
            allocations_reused=True,health_records=ledger.records,health_drains=ledger.drains)
        args.out.write_text(json.dumps(report,indent=2)+'\n')
        print(key,'capture passed',identity['words32'],'words',flush=True)
    report['passed']=True;args.out.write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
