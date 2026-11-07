#!/usr/bin/env python3
"""Exercise DeviceDfiState graphs against the real complete-reset captures."""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path

from replay_reset import read, ROOT


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    import numpy as np
    import cupy as cp
    from woof.core.dfi import DeviceDfiState, NativeLayout, DYNAMICS, COORDS
    from woof.core.kernels import module_source
    args.output.mkdir(parents=True,exist_ok=True)
    captures=[read(x,np) for x in sorted(args.capture.glob('lane-reset-*.bin'))]
    assert captures
    driver=C.CDLL('libcuda.so.1')
    launch=driver.cuGraphLaunch;launch.argtypes=[C.c_void_p,C.c_void_p];launch.restype=C.c_int
    receipt=dict(scope='DeviceDfiState real native save/reset graphs',forecast_qualified=False,
                 strict_words=0,strict_differing_words=0,cases=[],
                 source={str(x.relative_to(ROOT)):hashlib.sha256(x.read_bytes()).hexdigest()
                    for x in (ROOT/'woof/core/dfi.py',ROOT/'woof/core/kernels/dfi.cu',
                              ROOT/'woof/core/kernels/dfi_reset.cu')},
                 assembled_source={n:hashlib.sha256(module_source(n).encode()).hexdigest()
                    for n in ('dfi','dfi_reset')})
    for strict in (True,False):
        for i,cap in enumerate(captures):
            h=cap['h']; mode=int(h[4])
            if mode not in (4,5,9):continue
            assert h[6]==6, 'QC/QI inventory in this captured package is WSM6'
            layout=NativeLayout(columns=int(h[16]*h[17]),levels=int(h[12]),
                moist_species=int(h[7]),qv=int(h[14])-1,qc=int(h[14]),qi=int(h[14])+2,
                first_scalar=int(h[15])-1,theta_m=bool(h[10]),hypsometric=int(h[11]),
                p_top=float(cap['p_top']),surface_physics=int(h[9]),
                microphysics_dfi=int(h[6]),save_hydrometeors=bool(h[5]))
            live={n:cp.asarray(a) for n,a in cap['fields'].items() if not n.startswith('dfi_')}
            state=DeviceDfiState(live,layout,strict=strict)
            for n,a in state.saved.items():
                a[...] = cp.asarray(cap['fields'][n])
            state.total[...] = cap['total']
            stream=cp.cuda.Stream(non_blocking=True)
            cp.cuda.get_current_stream().synchronize()
            if mode==4:
                graph=state.capture_graph(lambda:state.save(backward=h[2]==1),stream=stream)
                expected=cap
                names=['dfi_'+n for n in state.soil+('snow','snowh','snowc','canwat')]
                if h[5]: names+=['dfi_moist','dfi_scalar']
                if h[2]==1:names+=['dfi_rh']
                outputs=state.saved
            elif mode==5:
                graph=state.capture_graph(lambda:state.reset(backward=h[2]==1),stream=stream)
                expected=next(x for x in captures[i+1:] if x['h'][4]==6)
                names=[n for n in live if n not in COORDS]
                outputs=live
            else:
                graph=state.capture_graph(state.clear,stream=stream)
                expected=captures[i+1]
                assert expected['h'][4]==10
                names=[n for n in cap['fields'] if n not in COORDS]
                outputs=dict(live,**state.saved)
            start,stop=cp.cuda.Event(),cp.cuda.Event()
            start.record(stream)
            assert launch(C.c_void_p(graph.graphExec),C.c_void_p(stream.ptr))==0
            stop.record(stream);stop.synchronize()
            result=dict(strict=strict,mode=mode,stage=int(h[2]),input=cap['path'],
                expected=expected['path'],words=0,differing_words=0,
                graph_ms=cp.cuda.get_elapsed_time(start,stop),fields=[])
            if mode==9:
                assert state.total.get().view(np.uint32)[0]==expected['total'].view(np.uint32)
                assert int(state.cursor.get()[0])==0 and int(state.error.get()[0])==0
            for n in names:
                got=cp.asnumpy(outputs[n]).reshape(-1);want=expected['fields'][n]
                bad=np.flatnonzero(got.view(np.uint32)!=want.view(np.uint32))
                row=dict(name=n,words=got.size,differing_words=bad.size)
                if bad.size:
                    at=int(bad[0]);row['first']=dict(index=at,actual=float(got[at]),expected=float(want[at]))
                result['words']+=got.size;result['differing_words']+=bad.size;result['fields'].append(row)
            if strict:
                receipt['strict_words']+=result['words'];receipt['strict_differing_words']+=result['differing_words']
            receipt['cases'].append(result)
    (args.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:receipt[k] for k in ('strict_words','strict_differing_words')}))
    if receipt['strict_differing_words']:raise SystemExit(1)


if __name__=='__main__':main()
