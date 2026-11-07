"""Real-call word replay. Used by the lane launcher, never by production."""
from pathlib import Path
from functools import wraps
import hashlib
import importlib.util
import json
import os
import re
import statistics
import sys
import numpy as np


def signature(source, name):
    args = re.search(r'void\s+' + name + r'\s*\((.*?)\)\s*\{', source, re.S)[1].split(',')
    return ([re.search(r'(\w+)\s*$', arg)[1] for arg in args],
            [i for i, arg in enumerate(args) if '*' in arg and 'const' not in arg])


def emit(row):
    with open(os.environ['SCALAR_REPORT'], 'a') as f:
        f.write(json.dumps(row) + '\n')
    print('SCALAR_AB ' + json.dumps(row), flush=True)


def timed_pair(cp, launches, restore):
    stream = cp.cuda.Stream(non_blocking=True)
    cp.cuda.get_current_stream().synchronize()
    with stream:
        graphs = {}
        for label, launch in launches.items():
            restore(); launch(); stream.synchronize()
            stream.begin_capture(); launch(); graphs[label] = stream.end_capture()
        samples = {label: [] for label in launches}
        for iteration in range(9):
            order = list(launches)
            if iteration % 2: order.reverse()
            for label in order:
                restore()
                start, end = cp.cuda.Event(), cp.cuda.Event()
                start.record(); graphs[label].launch(stream); end.record(); end.synchronize()
                samples[label].append(float(cp.cuda.get_elapsed_time(start, end)))
    return {label: dict(median_ms=statistics.median(values), samples_ms=values)
            for label, values in samples.items()}


def modules():
    from woof.hex import cuda_transport_v841 as current, numerics
    from cupy.cuda import function
    root = Path(os.environ['LANE']) / 'evidence'
    spec = importlib.util.spec_from_file_location('woof.hex._scalar_base', root/'cuda_transport_v841_base.py')
    base = importlib.util.module_from_spec(spec); sys.modules[spec.name] = base; spec.loader.exec_module(base)
    compiled = {}
    manifest = json.loads((root/'compile.json').read_text())
    for family in ('transport', 'driver'):
        compiled[family] = {}
        for label in ('base','tip'):
            row = next(r for r in manifest if (r['family'],r['label'],r['mode']) == (family,label,numerics.mode()))
            blob = (root/f'{family}-{label}-{numerics.mode()}.cubin').read_bytes()
            assert hashlib.sha256(blob).hexdigest() == row['image_sha256']
            if family == 'transport':
                assert hashlib.sha256((base if label=='base' else current)._CUDA_SOURCE.encode()).hexdigest() == row['source_sha256']
            module = function.Module(); module.load(blob); compiled[family][label] = module
    return base,current,compiled


def install():
    import cupy as cp
    from woof.hex import numerics
    from woof.hex.cuda_driver import CudaDryDycoreDriver
    base, current, images = modules()
    # These real x1 calls use generic kernels. Large shape-specialized kernels
    # are separately compiled from the captured dimensions for the tile replay.
    for label, module in [('base',base),('tip',current)]:
        module._kernel = lambda name, cache=None, dimensions=(), label=label: images['transport'][label].get_function(name)
    mode = numerics.mode(); step = 0; seen = {}
    original_step = CudaDryDycoreDriver._step_device_v841
    @wraps(original_step)
    def advance(driver, *args, **kwargs):
        nonlocal step
        step += 1
        return original_step(driver, *args, **kwargs)
    CudaDryDycoreDriver._step_device_v841 = advance
    tip_launch = current._launch
    def check(name, args, launches, names, outputs, count):
        key = (step,name); call=seen.get(key,0); seen[key]=call+1
        initial = {i: args[i].copy() for i in outputs}
        def restore():
            for i,value in initial.items(): args[i][...] = value
        launches['base'](); expected={i: args[i].copy() for i in outputs}
        restore(); launches['tip']()
        fields={names[i]:dict(words=int(args[i].size),different=int(cp.count_nonzero(args[i].view(cp.uint32)!=expected[i].view(cp.uint32)).item())) for i in outputs}
        row=dict(mode=mode,route=os.environ['SCALAR_ROUTE'],step=step,kernel=name,call=call,count=int(count),fields=fields,words=sum(v['words'] for v in fields.values()),different=sum(v['different'] for v in fields.values()))
        emit(row)
        root=Path(os.environ['SCRATCH'])/'capture'/os.environ['SCALAR_ROUTE']/mode
        root.mkdir(parents=True,exist_ok=True)
        values={f'arg{i}':cp.asnumpy(initial[i] if i in initial else value) if isinstance(value,cp.ndarray) else value for i,value in enumerate(args)}
        values['count']=np.int64(count)
        np.savez(root/f'{step}-{name}-{call}.npz',**values)
        assert row['different']==0,row
        if step==24:
            emit(dict(mode=mode,route=os.environ['SCALAR_ROUTE'],step=step,kernel=name,call=call,timing=timed_pair(cp,launches,restore)))
            restore();launches['tip']()
    def launch(name,count,args,cache):
        if step not in (1,12,24) or name.startswith('validate_'):
            return tip_launch(name,count,args,cache)
        names,outputs=signature(current._CUDA_SOURCE,name)
        check(name,args,{'base':lambda:base._launch(name,count,args,cache),'tip':lambda:tip_launch(name,count,args,cache)},names,outputs,count)
    current._launch=launch
    original_pair=CudaDryDycoreDriver._add_pair
    original_kernel=CudaDryDycoreDriver._kernel
    def kernel(driver,name):
        if name=='add_pair_inplace_f32':return images['driver']['tip'].get_function(name)
        return original_kernel(driver,name)
    CudaDryDycoreDriver._kernel=kernel
    def pair(driver,target,first,second):
        if step not in (1,12,24):return original_pair(driver,target,first,second)
        nl,no=map(int,target.shape); args=(np.int32(nl),np.int32(no),first,second,target)
        add=images['driver']['base'].get_function('add_inplace_f32')
        def old():
            for source in (first,second):add(((nl*no+127)//128,),(128,),(args[0],args[1],source,target))
        check('add_pair_inplace_f32',args,{'base':old,'tip':lambda:original_pair(driver,target,first,second)},['nlevels','nowners','first','second','target'],[4],nl*no)
    CudaDryDycoreDriver._add_pair=pair
    from woof.hex import cuda_driver
    original_transport=cuda_driver.advance_scalar_transport_cuda_v841
    @wraps(original_transport)
    def whole(*args, **kwargs):
        actual=original_transport(*args, **kwargs)
        if step in (1,12,24):
            baseline_kwargs=dict(kwargs)
            baseline_kwargs.pop('bind_density_pool',None)
            expected=base.advance_scalar_transport_cuda_v841(*args,**baseline_kwargs)
            fields={}
            for name in ('scalars','density'):
                a,b=getattr(actual,name),getattr(expected,name)
                fields[name]=dict(words=int(a.size),different=int(cp.count_nonzero(a.view(cp.uint32)!=b.view(cp.uint32)).item()))
            row=dict(mode=mode,route=os.environ['SCALAR_ROUTE'],step=step,kernel='whole_transport',rk_step=kwargs['rk_step'],fields=fields,words=sum(v['words'] for v in fields.values()),different=sum(v['different'] for v in fields.values()),scalar_count=int(actual.scalars.shape[0]))
            emit(row);assert row['different']==0,row
        return actual
    cuda_driver.advance_scalar_transport_cuda_v841=whole


def test_real_replay_receipts():
    import pytest
    path=os.environ.get('SCALAR_DEFAULT_RECEIPTS')
    if path is None:pytest.skip('requires real GPU replay receipts')
    for route in ('hrrrv5','cp'):
        for mode in ('fast','strict'):
            rows=[json.loads(s) for s in (Path(path)/f'{route}-{mode}.jsonl').read_text().splitlines()]
            for step in (1,12,24):
                calls=[r for r in rows if r.get('step')==step and 'words' in r]
                assert calls and all(r['different']==0 for r in calls)
                assert {'transport_edge_values','transport_standard_finish_v841','transport_edge_values_mono_v841','fct_finish_v841','add_pair_inplace_f32'} <= {r['kernel'] for r in calls}

                whole=[r for r in calls if r['kernel']=='whole_transport']
                assert len(whole)==3 and {r['rk_step'] for r in whole}=={1,2,3}
                assert all(r['scalar_count']==(13 if route=='hrrrv5' else 8) for r in whole)
                assert all(set(r['fields'])=={'scalars','density'} for r in whole)
                assert all(all(f['words']>0 and f['different']==0 for f in r['fields'].values()) for r in calls)
