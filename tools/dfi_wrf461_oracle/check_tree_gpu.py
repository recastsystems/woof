#!/usr/bin/env python3
"""Real CUDA-driver test of signed Rust nest scheduling, not weather dynamics."""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2]

class DomainGraphs(C.Structure):
    _fields_=[('parent',C.c_int32),('ratio',C.c_uint32),('step_ticks',C.c_int64),
              ('model_graph',C.c_void_p),('force_graph',C.c_void_p),
              ('accumulate_graph',C.c_void_p),('advance_graph',C.c_void_p)]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    import cupy as cp
    import numpy as np
    libpath=a.output/'libarwen_dfi.so'
    command=['rustc','--edition=2021','--crate-type=cdylib','-C','opt-level=3',
             str(ROOT/'tools/dfi-runtime/src/lib.rs'),'-o',str(libpath)]
    subprocess.run(command,check=True)
    lib=C.CDLL(str(libpath));driver=C.CDLL('libcuda.so.1')
    call=lib.aw_dfi_submit_tree
    call.argtypes=[C.c_void_p,C.POINTER(DomainGraphs),C.c_size_t,C.c_size_t,C.c_int64,C.c_void_p]
    call.restype=C.c_int
    module=cp.RawModule(code=r'''
extern "C" __global__ void event(long long* ticks, unsigned int* cursor,
 unsigned int* count, long long* records, int node, int kind, long long dt) {
 if(blockIdx.x||threadIdx.x)return;
 unsigned int row=(*count)++;
 records[4*row]=node;records[4*row+1]=kind;
 records[4*row+2]=ticks[node];records[4*row+3]=cursor[node];
 if(kind==0)ticks[node]+=dt;
 if(kind==3)cursor[node]++;
}
''',options=('-std=c++11',))
    kernel=module.get_function('event');kernel.compile()
    topology=[(-1,1,6),(0,2,3),(1,3,1),(0,3,2)]
    receipt=dict(scope='CUDA graph launch and signed integer nest-clock contract; no weather dynamics',
        forecast_qualified=False,rust_command=command,source_sha256=hashlib.sha256(
            (ROOT/'tools/dfi-runtime/src/lib.rs').read_bytes()).hexdigest(),cases=[])
    for sign in (-1,1):
        ticks=cp.zeros(4,cp.int64);cursor=cp.zeros(4,cp.uint32)
        count=cp.zeros(1,cp.uint32);records=cp.full((256,4),-999,cp.int64)
        stream=cp.cuda.Stream(non_blocking=True);graphs=[];nodes=(DomainGraphs*4)()
        cp.cuda.get_current_stream().synchronize()
        for i,(parent,ratio,step) in enumerate(topology):
            handles=[]
            for kind in range(4):
                with stream:
                    stream.begin_capture()
                    kernel((1,),(1,),(ticks,cursor,count,records,np.int32(i),
                        np.int32(kind),np.int64(step*sign)))
                    graph=stream.end_capture()
                graphs.append(graph);handles.append(graph.graphExec)
            nodes[i]=DomainGraphs(parent,ratio,step*sign,*handles)
        # Model=0, force=1, accumulate=2, cursor advance=3. The ABI order
        # uses model, force, accumulate, advance, matching handles above.
        expected=[];host_ticks=[0]*4;host_cursor=[0]*4
        def event(i,kind):
            expected.append([i,kind,host_ticks[i],host_cursor[i]])
            if kind==0:host_ticks[i]+=topology[i][2]*sign
            if kind==3:host_cursor[i]+=1
        def interval(i,steps):
            children=[j for j,n in enumerate(topology) if n[0]==i]
            for _ in range(steps):
                event(i,2);event(i,0);event(i,3)
                for j in children:event(j,1)
                for j in children:interval(j,topology[j][1])
            event(i,2)
        interval(0,2)
        status=call(C.cast(driver.cuGraphLaunch,C.c_void_p),nodes,4,0,12*sign,stream.ptr)
        assert status==0,status
        # Sole completion/read boundary, after the entire submitted tree.
        stream.synchronize()
        actual=cp.asnumpy(records)[:int(cp.asnumpy(count)[0])]
        want=np.asarray(expected,np.int64)
        assert np.array_equal(actual,want),(actual,want)
        assert cp.asnumpy(ticks).tolist()==[12*sign]*4
        assert cp.asnumpy(cursor).tolist()==[2,4,12,6]
        # A bad signed ratio must refuse before even the first graph launch.
        previous_count=int(cp.asnumpy(count)[0]);nodes[2].step_ticks*=-1
        invalid=call(C.cast(driver.cuGraphLaunch,C.c_void_p),nodes,4,0,12*sign,stream.ptr)
        assert invalid==-4 and int(cp.asnumpy(count)[0])==previous_count
        receipt['cases'].append(dict(sign=sign,events=actual.tolist(),event_count=len(actual),
            final_ticks=host_ticks,final_cursors=host_cursor,invalid_clock_status=invalid,
            exact_native_order=True))
    (a.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(cases=len(receipt['cases']),events=sum(x['event_count'] for x in receipt['cases']),differences=0)))


if __name__=='__main__':main()
