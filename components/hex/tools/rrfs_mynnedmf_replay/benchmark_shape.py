"""Time packet kernels on real captured columns and their exact column tiling.

No compilation takes place here. Each sample restores every captured input; CUDA
 events cover only MYNN kernel launches. All buffers and error flags are checked
 against an untiled reference call, including unchanged words and repeated runs.
"""
import argparse
import gc
import json
import os
import statistics
from pathlib import Path

import numpy as np
from replay_mynn import load_module


class TimedKernel:
    def __init__(self, kernel, cp, events):
        self.kernel, self.cp, self.events = kernel, cp, events
        self.ptr = kernel.ptr

    def __call__(self, *args, **kwargs):
        start, stop = self.cp.cuda.Event(), self.cp.cuda.Event()
        start.record()
        self.kernel(*args, **kwargs)
        stop.record()
        self.events.append((start, stop))


def capture(folder):
    manifest = json.loads((folder / 'manifest.json').read_text())
    hosts = {i: np.load(folder / f'b{i:03d}.npy') for i in range(manifest['buffers'])}
    return manifest, hosts


def device_inputs(manifest, hosts, factor, cp):
    small = manifest['scalars']['ite']
    assert manifest['scalars']['its'] == manifest['scalars']['ims'] == 1
    assert manifest['scalars']['ime'] == small
    assert manifest['scalars']['jts'] == manifest['scalars']['jte'] == 1
    buffers = {}
    originals = {}
    for i, a in hosts.items():
        assert a.shape[0] == small
        originals[i] = cp.asarray(a, order='F')
        buffers[i] = cp.empty((small*factor, *a.shape[1:]), dtype=a.dtype, order='F')
    arguments = {r['name']: buffers[r['buffer']] for r in manifest['arrays']}
    arguments.update(manifest['scalars'])
    arguments.update(ime=small*factor, ite=small*factor, ide=small*factor+1)

    def restore():
        for i, a in buffers.items():
            a.reshape((small, factor, *a.shape[1:]), order='F')[...] = originals[i][:, None, ...]

    return arguments, buffers, restore


def compare(buffers, expected, small, factor, cp):
    total = different = 0
    for i, a in buffers.items():
        dtype = {4: cp.uint32, 8: cp.uint64}[a.dtype.itemsize]
        want = expected[i].view(dtype)
        actual = a.view(dtype).reshape((small, factor, *a.shape[1:]), order='F')
        total += a.size
        different += int(cp.count_nonzero(actual != want[:, None, ...]).item())
    return total, different


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference', required=True, type=Path)
    ap.add_argument('--reference-image', required=True, type=Path)
    ap.add_argument('--variant', action='append', nargs=3, metavar=('NAME', 'MODULE', 'IMAGE'), default=[])
    ap.add_argument('--capture', required=True, type=Path)
    ap.add_argument('--factors', default='1,40')
    ap.add_argument('--widths', default='1,4,16')
    ap.add_argument('--samples', default=9, type=int)
    ap.add_argument('--batch-columns', type=int)
    ap.add_argument('--numerics', choices=('fast','strict'), default='fast')
    ap.add_argument('--out', required=True, type=Path)
    a = ap.parse_args()
    assert a.samples >= 9
    from woof.hex import numerics
    numerics.set_mode(a.numerics, source='benchmark_shape')
    import cupy as cp
    from cupy.cuda import driver
    manifest, hosts = capture(a.capture)
    small = manifest['scalars']['ite']
    module = load_module('reference', a.reference)
    ref = module.MynnEdmfCuda(compiled_image=a.reference_image.read_bytes())
    args, buffers, restore = device_inputs(manifest, hosts, 1, cp)
    restore()
    result = ref.run(args)
    expected = {i: cp.array(b, order='F') for i,b in buffers.items()}
    expected['errflg'] = cp.array(result['errflg'])
    del args, buffers, result, restore, ref
    gc.collect(); cp.get_default_memory_pool().free_all_blocks()
    report = {'capture': str(a.capture), 'numerics':a.numerics, 'samples':a.samples,
              'device': cp.cuda.runtime.getDeviceProperties(0)['name'].decode(), 'rows': []}
    variants = [('base', str(a.reference), str(a.reference_image)), *a.variant]
    for name, path, image in variants:
        m = load_module(name, path)
        backend = m.MynnEdmfCuda(compiled_image=Path(image).read_bytes(), batch_columns=a.batch_columns)
        backend._compile()
        attrs = {w: {k:driver.funcGetAttribute(attr, f.ptr) for k,attr in
                 [('registers',4), ('local_bytes',3)]} for w,f in backend._kernels.items()}
        events = []
        backend._kernels = {w:TimedKernel(k, cp, events) for w,k in backend._kernels.items()}
        for factor in map(int,a.factors.split(',')):
            args, buffers, restore = device_inputs(manifest, hosts, factor, cp)
            for width in map(int,a.widths.split(',')):
                if width:
                    os.environ['WOOF_HEX_MYNNEDMF_LANES'] = str(width)
                else:
                    os.environ.pop('WOOF_HEX_MYNNEDMF_LANES', None)
                backend._lane_widths.clear()
                times = []
                diffs = []
                for rep in range(a.samples+1):
                    restore(); events.clear()
                    result = backend.run(args)
                    cp.cuda.get_current_stream().synchronize()
                    ms = sum(cp.cuda.get_elapsed_time(s,e) for s,e in events)
                    if rep:
                        times.append(ms)
                    if rep in (0,a.samples):
                        words,diff = compare({**buffers, 'errflg': result['errflg']}, expected, small, factor, cp)
                        diffs.append(diff)
                row = {'variant':name,'columns':small*factor,'width':backend.last_lane_width,
                       'requested_width':width,'batch_columns':backend.last_batch_columns,
                       'kernel_launches':len(events), 'workspace_stride':m.workspace_bytes(args),
                       'kernel_median_ms':statistics.median(times), 'kernel_samples_ms':times,
                       'ms_per_100000_columns':statistics.median(times)*100000/(small*factor),
                       'compared_words':words,'different_words':diffs, **attrs[backend.last_lane_width]}
                report['rows'].append(row)
                print(json.dumps(row),flush=True)
                a.out.write_text(json.dumps(report,indent=2)+'\n')
            del args, buffers, restore, result
            gc.collect(); cp.get_default_memory_pool().free_all_blocks()
        del backend
        gc.collect(); cp.get_default_memory_pool().free_all_blocks()
    assert all(not any(r['different_words']) for r in report['rows'])


if __name__ == '__main__':
    main()
