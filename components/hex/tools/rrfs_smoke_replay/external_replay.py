"""Replay real GNU ABI captures of the complete pinned RRFS smoke wrapper.

Only captured PRE words feed CUDA. Every present POST argument is compared,
including unchanged words. The native ordinary/observed history comparison
must pass before this tool starts. SAVE state persists across consecutive calls.
"""
import argparse,ctypes as ct,hashlib,json,os,time
from pathlib import Path
import numpy as np


def read(lib,path):
    h=(ct.c_int32*21)(); data=ct.POINTER(ct.c_uint32)()
    handle=lib.nssl_record_open(os.fsencode(path),h,ct.byref(data))
    if not handle:raise ValueError('Cannot decode native record: '+str(path))
    try:
        shape=tuple(h[3:3+h[2]]);size=int(np.prod(shape)) if shape else 1
        words=np.ctypeslib.as_array(data,shape=(size,)).copy()
        return words.view('f4' if h[0]==1 else 'i4').reshape(shape,order='F'),tuple(h[8:8+h[2]])
    finally:lib.nssl_record_close(handle)


def run(a):
    import cupy as cp
    import importlib
    from woof.hex import numerics
    from woof.hex.mpasphys.rrfs import smoke_resident as r
    # The graph of the capture's SMOKE pin: 9652bbf (MPAS-Model-rrfs 1d6e370e) or 5d2c71e2 (RRFSx 44c0849).
    m=importlib.import_module('woof.hex.mpasphys.rrfs.smoke_'+a.pin)
    assert json.loads(a.passivity.read_text())['all_identical']
    numerics.set_mode('strict')
    from woof import numerics as engine_numerics
    engine_numerics.select('strict')
    lib=ct.CDLL(str(a.reader.resolve()))
    lib.nssl_record_open.argtypes=[ct.c_char_p,ct.POINTER(ct.c_int32),ct.POINTER(ct.POINTER(ct.c_uint32))]
    lib.nssl_record_open.restype=ct.c_void_p;lib.nssl_record_close.argtypes=[ct.c_void_p]
    rows=[json.loads(x) for x in (a.capture/'rank0-calls.jsonl').read_text().splitlines()]
    pre=[x for x in rows if x['phase']=='pre'];post=[x for x in rows if x['phase']=='post']
    assert len(pre)>=3 and [x['call'] for x in pre]==list(range(len(pre)))
    assert [x['call'] for x in post]==[x['call'] for x in pre]
    if a.arena_highwater:
        # Measurement only: record the most scratch words any column holds at once (atomicMax over every column-mode
        # arena allocation), to size the per-column scratch from the real calls rather than a floor.
        target=getattr(m,'graph',m);original=target.cuda_source
        def probed(*args,**kwargs):
            text=original(*args,**kwargs)
            hook=('__device__ int sd_arena_high;\n'
                  '__device__ void sd_highwater(bool column,int used){if(column)atomicMax(&sd_arena_high,used);}\n'
                  'struct Arena {')
            assert text.count('struct Arena {')==1 and text.count('used+=words;return r;}')==2
            return text.replace('struct Arena {',hook,1).replace('used+=words;return r;}','used+=words;sd_highwater(column_mode,used);return r;}')
        target.cuda_source=probed
    t=time.monotonic();state=r.new_state(m)
    result=dict(compile_seconds=time.monotonic()-t,numerics='strict',rows=[],passed=False,
                source_sha256=hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest(),smoke_pin=m.SMOKE_PIN,fork_pin=m.FORK_PIN,
                cuda_source_sha256=hashlib.sha256(m.cuda_source(r.math_source(m),columns=True).encode()).hexdigest(),
                active_fire_proven=False,scope='Complete native mpas_smoke_driver ABI; real captured inputs',
                compile_metadata=state.compile_metadata,
                engine_numerics=engine_numerics.receipt(),
                effective_options=engine_numerics.effective_options(numerics.options(m.cuda_options(columns=True))))
    for row in pre:
        args={};lower={};wanted={};prefix=row['prefix']
        for name in m.ROUTINES['mpas_smoke_driver']['args']:
            present,_=read(lib,a.capture/(prefix+'.pre.'+name+'.present.bin'))
            if not present.item():args[name]=None;continue
            v,b=read(lib,a.capture/(prefix+'.pre.'+name+'.bin'))
            w,_=read(lib,a.capture/(prefix+'.post.'+name+'.bin'));wanted[name]=w
            args[name]=cp.asarray(v,order='F') if v.ndim else v.item();lower[name]=b
        fire_in=cp.asnumpy(args['frp_in'])
        emitted=wanted['e_bb_out']
        coverage=dict(active_fire_columns=int(np.count_nonzero(fire_in>0)),
                      positive_emission_words=int(np.count_nonzero(emitted>0)),
                      plume_top_max=float(np.max(wanted['max_bb_plume'])),
                      wet_deposition_words=int(np.count_nonzero(wanted['wetdep_resolved'])),
                      dry_deposition_words=int(np.count_nonzero(wanted['drydep_flux'])))
        start,end=cp.cuda.Event(),cp.cuda.Event();start.record()
        descriptor=state.call('mpas_smoke_driver',args,lower_bounds=lower)
        end.record();end.synchronize();returned=cp.asnumpy(descriptor)
        report=dict(call=row['call'],ktau=int(args['ktau']),milliseconds=float(cp.cuda.get_elapsed_time(start,end)),words=0,differing_words=0,fields=[],coverage=coverage)
        for i,name in enumerate(m.ROUTINES['mpas_smoke_driver']['args']):
            if name not in wanted:continue
            w=wanted[name]
            if w.ndim:v=cp.asnumpy(args[name])
            else:v=np.array(returned['fvalue' if w.dtype==np.float32 else 'ivalue'][i],dtype=w.dtype)
            left=v.ravel(order='F').view('u4');right=w.ravel(order='F').view('u4');different=np.flatnonzero(left!=right)
            report['words']+=left.size;report['differing_words']+=different.size
            if different.size:report['fields'].append(dict(name=name,differing_words=int(different.size),first=[dict(index=int(j),cuda=hex(int(left[j])),native=hex(int(right[j]))) for j in different[:4]]))
        result['rows'].append(report);a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(report),flush=True)
    if a.arena_highwater:
        import ctypes
        # cupy.cuda.function.Module.get_global_var returns the symbol's device address as an int.
        address=int(state._module.get_global_var('sd_arena_high'));high=np.zeros(1,np.int32)
        cp.cuda.runtime.memcpy(high.ctypes.data,address,4,cp.cuda.runtime.memcpyDeviceToHost)
        result['arena_highwater_words']=int(high[0]);result['workspace_words_per_column']=int(state._workspace.size//state._tile)
        print(json.dumps({'arena_highwater_words':int(high[0]),'workspace_words_per_column':result['workspace_words_per_column'],
                          'tile':state._tile,'state_bytes':state._state_size,'plume_bytes':state._plume_size}),flush=True)
    result['passed']=all(x['differing_words']==0 for x in result['rows'])
    result['active_fire_proven']=result['passed'] and any(x['coverage']['active_fire_columns'] and x['coverage']['positive_emission_words'] for x in result['rows'])
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    return 0 if result['passed'] else 1

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('capture','passivity','reader','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--pin',choices=('9652bbf','5d2c71e2'),default='9652bbf')
    p.add_argument('--arena-highwater',action='store_true')
    raise SystemExit(run(p.parse_args()))
