"""Real-call, all-word A/B for the RRFS monotonic launch mapping.

Run as a forecast launcher with MONO_BASE pointing to the unmodified module
and MONO_REPORT to the JSON receipt. Calls 1, 12 and 24 are replayed in process.
Each changed kernel is checked against the merged base on identical operands,
including every scalar's private scratch and untouched allocation words.
"""
from __future__ import annotations

import importlib.util
import hashlib
import json
import os
from pathlib import Path
import re
import statistics

import numpy as np


# Private pointers, in the CUDA signature's spelling. All other pointers are
# shared across species. This table is independent of the production launcher.
PRIVATE = {
    'mono_source_rrfs': (), 'mono_density_rrfs': (),
    'mono_carriers_rrfs': ('o', 'n'),
    'mono_vertical_rrfs': ('o', 'n', 'mn', 'mx', 'w'),
    'mono_high_edge_rrfs': ('q', 'flux'),
    'mono_low_vertical_rrfs': ('o', 'n', 'w', 'scale'),
    'mono_low_edge_rrfs': ('o', 'flux', 'upwind', 'residual'),
    'mono_scale_rrfs': ('upwind', 'residual', 'mn', 'mx', 'n', 'scale'),
    'mono_limit_edge_rrfs': ('o', 'scale', 'flux'),
    'mono_finish_rrfs': ('scale', 'flux', 'w', 'n'),
}


# Layout changes only; both sides retain the accepted level/species launch grid.
TRANSPOSED = {
    'mono_density_rrfs': ('edges', 'sign'),
    'mono_high_edge_rrfs': ('cells', 'coef', 'coef3'),
    'mono_scale_rrfs': ('edges', 'sign'),
    'mono_finish_rrfs': ('edges', 'sign'),
}


def load_base(path):
    spec = importlib.util.spec_from_file_location('woof.hex._mono_lane_base', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def signature(source, name):
    match = re.search(r'void\s+' + name + r'\((.*?)\)\{', source, re.S)
    declarations = match[1].split(',')
    names = [re.search(r'(\w+)\s*$', arg)[1] for arg in declarations]
    outputs = [i for i, arg in enumerate(declarations) if '*' in arg and 'const' not in arg]
    return names, outputs


def compare(cp, expected, actual):
    assert expected.shape == actual.shape and expected.dtype == actual.dtype
    words = expected.size * expected.dtype.itemsize // 4
    different = int(cp.count_nonzero(expected.view(cp.uint32) != actual.view(cp.uint32)).item())
    return {'words': int(words), 'different': different}


def snapshot(cp, a):
    return {key: value.copy() if isinstance(value, cp.ndarray) else value for key, value in a.items()}


def timed(cp, fn, reset, repeats=9):
    # Capture launches after warm-up so CUDA events measure device work without
    # Python launch gaps. Restores are outside the timed graph.
    stream = cp.cuda.Stream(non_blocking=True)
    cp.cuda.get_current_stream().synchronize()
    with stream:
        reset()
        fn()
        reset()
        stream.synchronize()
        stream.begin_capture()
        fn()
        graph = stream.end_capture()
        samples = []
        for _ in range(repeats):
            reset()
            start, end = cp.cuda.Event(), cp.cuda.Event()
            start.record()
            graph.launch(stream)
            end.record()
            end.synchronize()
            samples.append(float(cp.cuda.get_elapsed_time(start, end)))
    return {'median_ms': statistics.median(samples), 'samples_ms': samples}


def replay(cp, base, current, entry, config_apply_lbcs, capture_dir=None):
    """Compare each kernel and the complete public work boundary."""
    before = snapshot(cp, entry)
    expected = snapshot(cp, entry)
    actual = snapshot(cp, entry)
    records = []
    old_kernel = base.kernel
    new_kernel = current.kernel

    def checked(source, module_key, name, cache=None):
        candidate = new_kernel(source, module_key, name, cache)
        reference = old_kernel(base._CUDA_SOURCE, 'woof.hex._mono_lane_base', name)
        names, outputs = signature(base._CUDA_SOURCE, name)

        def run(grid, block, args):
            args = tuple(args)
            reference_args = [arg.copy() if i in outputs else arg for i, arg in enumerate(args)]
            species = grid[2] if len(grid) > 2 else 1
            for i, argname in enumerate(names):
                if argname in TRANSPOSED.get(name, ()):
                    reference_args[i] = cp.ascontiguousarray(args[i].T)
            initial = {i: args[i].copy() for i in outputs}
            reference(grid, block, tuple(reference_args))
            candidate(grid, block, args)
            fields = {names[i]: compare(cp, reference_args[i], args[i]) for i in outputs}
            row = {'kernel': name, 'species': species, 'fields': fields,
                   'words': sum(v['words'] for v in fields.values()),
                   'different': sum(v['different'] for v in fields.values())}
            if name in TRANSPOSED:
                timings = {}
                for label, fn, operands in (('base', reference, tuple(reference_args)),
                                            ('tip', candidate, args)):
                    def reset():
                        for i in outputs:
                            operands[i][...] = initial[i]
                    timings[label] = timed(cp, lambda: fn(grid, block, operands), reset)
                row['timings'] = timings
                if capture_dir is not None:
                    # One species is sufficient for the large-mesh sequential
                    # scalar schedule. Keep the actual real-call operands.
                    saved = {}
                    for i, argname in enumerate(names):
                        value = initial[i] if i in outputs else reference_args[i]
                        if isinstance(value, cp.ndarray):
                            if argname in PRIVATE[name] and value.ndim == (4 if argname == 'scale' else 3):
                                value = value[0]
                            if argname == 'stage':
                                value = value[int(args[names.index('s')]):][:1]
                            saved[argname] = cp.asnumpy(value)
                        else:
                            saved[argname] = np.int32(0) if argname == 's' else value
                    np.savez(Path(capture_dir)/f'{name}.npz', **saved)
            records.append(row)
            if row['different']:
                raise AssertionError(json.dumps(row))
        return run

    base.advance_monotonic_work_cuda_rrfs(expected, config_apply_lbcs=config_apply_lbcs)
    current.kernel = checked
    try:
        current.advance_monotonic_work_cuda_rrfs(actual, config_apply_lbcs=config_apply_lbcs)
    finally:
        current.kernel = new_kernel
    fields = {name: compare(cp, expected[name], value) for name, value in actual.items()
              if isinstance(value, cp.ndarray)}
    assert not sum(row['different'] for row in fields.values()), fields
    # Additional whole-boundary proof with batching disabled covers the large
    # mesh and halo scheduling launch mapping on the very same real operands.
    unbatched = snapshot(cp, before)
    cap = current._BATCH_BYTES
    current._BATCH_BYTES = 0
    try:
        current.advance_monotonic_work_cuda_rrfs(unbatched, config_apply_lbcs=config_apply_lbcs)
    finally:
        current._BATCH_BYTES = cap
    serial_fields = {name: compare(cp, expected[name], value) for name, value in unbatched.items()
                     if isinstance(value, cp.ndarray)}
    assert not sum(row['different'] for row in serial_fields.values()), serial_fields
    timings = {}
    for label, module in (('base', base), ('tip', current)):
        state = snapshot(cp, before)
        def reset():
            for key, value in before.items():
                if isinstance(value, cp.ndarray):
                    state[key][...] = value
        timings[label] = timed(cp, lambda: module.advance_monotonic_work_cuda_rrfs(
            state, config_apply_lbcs=config_apply_lbcs), reset)
    return {'shape': list(entry['scalars_old'].shape), 'kernels': records,
            'fields': fields, 'unbatched_fields': serial_fields, 'timings': timings}


def install(base_path, report_path, calls=(1, 12, 24)):
    import cupy as cp
    from woof.hex import cuda_monotonic_rrfs as current, numerics
    base = load_base(base_path)
    # Optional CPU-built images avoid spending a GPU lease on NVRTC. The
    # forecast still uses its normal cache and compile manifest.
    image_dir = os.environ.get('MONO_IMAGES')
    if image_dir:
        from cupy.cuda import function
        pins = json.loads((Path(image_dir)/'compile.json').read_text())
        for label, module in (('base', base), ('tip', current)):
            image = (Path(image_dir)/f'mono-{label}-{numerics.mode()}.cubin').read_bytes()
            pin = next(row for row in pins if row['variant'] == label and row['numerics'] == numerics.mode())
            assert pin['source_sha256'] == hashlib.sha256(module._CUDA_SOURCE.encode()).hexdigest()
            assert pin['image_sha256'] == hashlib.sha256(image).hexdigest()
            compiled = function.Module()
            compiled.load(image)
            fallback = module.kernel
            def precompiled(source, module_key, name, cache=None, compiled=compiled, fallback=fallback):
                return (compiled.get_function(name) if cache is None
                        else fallback(source, module_key, name, cache))
            module.kernel = precompiled
    original = current.advance_monotonic_work_cuda_rrfs
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    count = 0

    def wrapped(a, **kw):
        nonlocal count
        count += 1
        if count in calls:
            # Detach the hook during replay, then return to the actual forecast.
            current.advance_monotonic_work_cuda_rrfs = original
            try:
                row = replay(cp, base, current, a, kw['config_apply_lbcs'],
                             os.environ.get('MONO_CAPTURE') if count == 24 else None)
            finally:
                current.advance_monotonic_work_cuda_rrfs = wrapped
            row['call'] = count
            records.append(row)
            receipt = {'numerics': numerics.mode(), 'expected_calls': list(calls),
                       'records': records, 'complete': [r['call'] for r in records] == list(calls)}
            report_path.write_text(json.dumps(receipt, indent=2) + '\n')
            print('MONO_AB ' + json.dumps({'call': count, 'shape': row['shape'],
                  'different': sum(v['different'] for v in row['fields'].values()),
                  'timings': row['timings']}), flush=True)
        return original(a, **kw)

    current.advance_monotonic_work_cuda_rrfs = wrapped
    return records


def test_batch_memory_bound_and_halo_schedule():
    from woof.hex.cuda_monotonic_rrfs import _species_per_pass, _BATCH_BYTES
    assert _species_per_pass(13, 10 * 1024**2, None) == 13
    assert _species_per_pass(13, _BATCH_BYTES + 1, None) == 1
    assert _species_per_pass(13, 10 * 1024**2, object()) == 1
    assert _species_per_pass(13, _BATCH_BYTES // 3, None) == 3


def test_every_kernel_has_explicit_output_contract():
    from woof.hex.cuda_monotonic_rrfs import _CUDA_SOURCE
    names = set(re.findall(r'__global__ void (mono_\w+)', _CUDA_SOURCE))
    assert names == set(PRIVATE)
    for name in names:
        arguments, outputs = signature(_CUDA_SOURCE, name)
        assert outputs
        assert set(PRIVATE[name]) <= set(arguments)



def test_stencil_cache_padding_streams_and_lifetime(monkeypatch):
    # CPU stand-in exercises the cache contract without touching a GPU.
    import gc
    import sys
    from types import SimpleNamespace
    from woof.hex import cuda_monotonic_rrfs as mono
    stream = SimpleNamespace(ptr=7)
    fake = SimpleNamespace(zeros=np.zeros, cuda=SimpleNamespace(
        Device=lambda: SimpleNamespace(id=0), get_current_stream=lambda: stream))
    monkeypatch.setitem(sys.modules, 'cupy', fake)
    monkeypatch.setattr(mono, '_MONO_STENCILS', {})
    arrays = (np.arange(12, dtype=np.int32).reshape(4, 3).copy(),
              np.array([[0.0, -0.0], [np.nan, 1.0]], dtype=np.float32))
    result = mono._stencil_layout(arrays, (5, 2))
    assert result[0].flags.c_contiguous and result[1].flags.c_contiguous
    np.testing.assert_array_equal(result[0][:, :-1], arrays[0].T)
    assert not result[0][:, -1].any()
    assert result[1].tobytes() == arrays[1].T.copy().tobytes()
    assert mono._stencil_layout(arrays, (5, 2)) is result
    stream.ptr = 8
    assert mono._stencil_layout(arrays, (5, 2)) is not result
    assert len(mono._MONO_STENCILS) == 2
    del arrays
    gc.collect()
    assert not mono._MONO_STENCILS


if __name__ == '__main__':
    import sys
    from woof.hex import cli
    args = cli.build_parser().parse_args(sys.argv[1:])
    cli._select_numerics(args)
    records = install(os.environ['MONO_BASE'], os.environ['MONO_REPORT'])
    result = int(args.handler(args))
    assert [row['call'] for row in records] == [1, 12, 24], 'incomplete real-step coverage'
    sys.exit(result)
