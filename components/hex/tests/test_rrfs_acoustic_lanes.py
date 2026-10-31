"""Bitwise acoustic lane proof, also usable as the real forecast launcher.

Run this file via the standard hrrrv5 recipe's LAUNCHER override. Set
RRFS_ACOUSTIC_BASE to the source exported from the base commit and
RRFS_ACOUSTIC_RECEIPT to the JSON receipt path. The production driver is
instrumented only here, never in product code. Each selected outer step
checks every acoustic substep, including all RK stages.
"""
from __future__ import annotations

from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import types


def _declarations(source, name):
    signature = source.split('extern "C" __global__ void ' + name + '(', 1)[1].split(')\n{', 1)[0]
    return [part.strip() for part in signature.split(',')]


def _outputs(source, name):
    return [(i, declaration.split('*')[-1].strip())
            for i, declaration in enumerate(_declarations(source, name))
            if '*' in declaration and not declaration.startswith('const ')]


def test_level_lane_mapping():
    from woof.hex import cuda_acoustic_rrfs as acoustic
    for name in ('ru', 'prepare', 'rs_ts', 'coefficients', 'matrix',
                 'column_pre', 'column_post_w', 'column_post_rt'):
        body = acoustic._CUDA_SOURCE.split('void acoustic_' + name + '_rrfs(', 1)[1].split('\n}\n', 1)[0]
        assert 'const int k = element /' in body
        assert 'for (int k' not in body
    for name in ('column_solve', 'factor'):
        body = acoustic._CUDA_SOURCE.split('void acoustic_' + name + '_rrfs(', 1)[1].split('\n}\n', 1)[0]
        assert 'for (int k = 1; k < nlev;' in body


def _stencil_cpu_backend(monkeypatch):
    import numpy as np
    from woof.hex import cuda_acoustic_rrfs as acoustic
    device = types.SimpleNamespace(id=0)
    stream = types.SimpleNamespace(ptr=7)
    fake = types.SimpleNamespace(zeros=np.zeros, ascontiguousarray=np.ascontiguousarray,
                                 cuda=types.SimpleNamespace(Device=lambda: device,
                                     get_current_stream=lambda: stream))
    monkeypatch.setattr(acoustic, '_cupy', lambda: fake)
    monkeypatch.setattr(acoustic, '_CELL_STENCILS', {})
    return acoustic, device, stream


def test_cell_stencil_missing_dummy_row_is_padded(monkeypatch):
    import numpy as np
    acoustic, device, stream = _stencil_cpu_backend(monkeypatch)
    edges = np.arange(12, dtype=np.int32).reshape(4, 3).copy()
    signs = np.array([[1., -1., -0.]] * 4, dtype=np.float32)
    layout = acoustic._cell_stencil_layout(edges, signs, 5)
    for source, table in zip((edges, signs), layout):
        assert table.shape == (3, 5) and table.flags.c_contiguous
        assert table[:, :-1].tobytes() == source.T.copy().tobytes()
        assert table[:, -1].tobytes() == np.zeros(3, dtype=source.dtype).tobytes()
    assert acoustic._cell_stencil_layout(edges, signs, 5) is layout
    exact = acoustic._cell_stencil_layout(edges, signs, 4)
    assert exact is not layout
    for source, table in zip((edges, signs), exact):
        assert table.tobytes() == source.T.copy().tobytes()
    device.id = 1
    assert acoustic._cell_stencil_layout(edges, signs, 5) is not layout
    device.id = 0
    stream.ptr = 8
    assert acoustic._cell_stencil_layout(edges, signs, 5) is not layout


def test_cell_stencil_wrong_extent_raises(monkeypatch):
    import numpy as np
    import pytest
    acoustic, _, _ = _stencil_cpu_backend(monkeypatch)
    edges = np.arange(12, dtype=np.int32).reshape(4, 3).copy()
    signs = np.ones((4, 3), dtype=np.float32)
    acoustic._cell_stencil_layout(edges, signs, 4)
    for extent in (3, 6):
        with pytest.raises(ValueError, match='native stencil extent'):
            acoustic._cell_stencil_layout(edges, signs, extent)
    with pytest.raises(ValueError, match='native stencil extent'):
        acoustic._cell_stencil_layout(edges, signs[:2], 4)


class Replay:
    def __init__(self, baseline, receipt):
        import cupy as cp
        from woof.hex import cuda_acoustic_rrfs as new
        from woof.hex.cuda_compile_rrfs import module
        self.cp, self.new = cp, new
        self.path = Path(receipt)
        self.step = 0
        self.substep = None
        self.checked = []
        self.timings = []
        self.timed = set()
        self.plan = []
        self.old = types.ModuleType('woof.hex._acoustic_base')
        self.old.__package__ = 'hexcore'
        exec(compile(Path(baseline).read_text(), str(baseline), 'exec'), self.old.__dict__)
        self.old_module = module(self.old._CUDA_SOURCE, 'hexcore.acoustic_base_replay')
        self.old._kernel = lambda name, kernel_cache=None: self.old_module.get_function(name)
        self.original_kernel = new._kernel
        self.original_advance = new.advance_acoustic_step_cuda_rrfs
        self.receipt = dict(base_source_sha256=hashlib.sha256(self.old._CUDA_SOURCE.encode()).hexdigest(),
                            new_source_sha256=hashlib.sha256(new._CUDA_SOURCE.encode()).hexdigest(),
                            comparisons=self.checked, timings=self.timings)

    def active(self):
        return self.step in (1, 12, 24)

    def clone(self, arguments):
        # Keep alias relationships when an argument is supplied more than once.
        copies = {}
        return tuple(copies.setdefault(id(a), a.copy()) if isinstance(a, self.cp.ndarray) else a
                     for a in arguments)

    def compare(self, label, got, want):
        cp = self.cp
        different = int(cp.count_nonzero(got.view(cp.uint32) != want.view(cp.uint32)).item())
        self.checked.append(dict(step=self.step, substep=self.substep, field=label,
                                 words=int(got.size), different_words=different))
        if different:
            self.save()
            raise AssertionError(f'{label}: {different} words differ at step {self.step}, substep {self.substep}')

    def time_plan(self, label, old_plan, new_plan, args):
        if os.environ.get('RRFS_ACOUSTIC_TIME', '1') == '0':
            return
        cp = self.cp
        timing = dict(step=self.step, substep=self.substep, kernel=label, repeats=9)
        for variant, plan in (('base', old_plan), ('lanes', new_plan)):
            work = self.clone(args)
            samples = []
            for _ in range(10):
                # Identical input state on every iteration, outside the events.
                for a, b in zip(work, args):
                    if isinstance(a, cp.ndarray):
                        cp.copyto(a, b)
                begin, end = cp.cuda.Event(), cp.cuda.Event()
                begin.record()
                for kernel, grid, block in plan:
                    kernel(grid, block, work)
                end.record()
                end.synchronize()
                samples.append(float(cp.cuda.get_elapsed_time(begin, end)))
            timing[variant + '_ms'] = statistics.median(samples[1:])
        self.timings.append(timing)

    def kernel(self, name, kernel_cache=None):
        new_kernel = self.original_kernel(name, kernel_cache)
        if not self.active():
            return new_kernel
        if name.startswith('acoustic_column_'):
            def column_pass(grid, block, args, **kwargs):
                if name == 'acoustic_column_pre_rrfs':
                    self.column_inputs = self.clone(args)
                    self.plan = []
                self.plan.append((new_kernel, grid, block))
                new_kernel(grid, block, args, **kwargs)
                if name == 'acoustic_column_post_rt_rrfs':
                    count = int(args[1])
                    old_kernel = self.old_module.get_function('acoustic_column_solve_rrfs')
                    old_grid, old_block = ((count + 31) // 32,), (32,)
                    old_args = self.clone(self.column_inputs)
                    old_kernel(old_grid, old_block, old_args)
                    for index, field in _outputs(self.old._CUDA_SOURCE, 'acoustic_column_solve_rrfs'):
                        self.compare('acoustic_column_solve_rrfs.' + field, args[index], old_args[index])
                    key = (self.step, 'column', self.substep == 1)
                    if key not in self.timed:
                        self.timed.add(key)
                        self.time_plan('acoustic_column_solve_rrfs_all_passes',
                                       [(old_kernel, old_grid, old_block)], self.plan, self.column_inputs)
                    self.plan = []
                    del self.column_inputs
            return column_pass
        if name != 'acoustic_rs_ts_rrfs':
            return new_kernel
        old_kernel = self.old_module.get_function(name)
        outputs = _outputs(self.old._CUDA_SOURCE, name)
        def checked(grid, block, args, **kwargs):
            before = self.clone(args)
            # Production receives cached slot-major geometry. The merged base
            # consumes cell-major geometry, with exactly the same entries.
            old_args = list(self.clone(before))
            for index in (7, 9):
                old_args[index] = self.cp.ascontiguousarray(old_args[index].T)
            old_args = tuple(old_args)
            old_kernel(grid, block, old_args, **kwargs)
            new_kernel(grid, block, args, **kwargs)
            for index, field in outputs:
                self.compare(name + '.' + field, args[index], old_args[index])
            key = (self.step, name, self.substep == 1)
            if key not in self.timed:
                self.timed.add(key)
                def old_layout_kernel(g, b, a):
                    values = list(a)
                    values[7], values[9] = old_args[7], old_args[9]
                    old_kernel(g, b, tuple(values))
                self.time_plan(name, [(old_layout_kernel, grid, block)],
                               [(new_kernel, grid, block)], before)
            return None
        return checked

    def advance(self, *args, **kwargs):
        self.substep = int(kwargs['small_step'])
        return self.original_advance(*args, **kwargs)

    def validate(self):
        if {c['step'] for c in self.checked} != {1, 12, 24}:
            raise AssertionError('real-input proof requires outer steps 1, 12 and 24')
        for step in (1, 12, 24):
            counts = [sum(c['step'] == step and c['field'] == field for c in self.checked)
                      for field in ('acoustic_rs_ts_rrfs.rs', 'acoustic_column_solve_rrfs.rw_p')]
            if counts != [24] * 2:
                raise AssertionError(f'outer step {step}: incomplete acoustic substeps {counts}')
        if any(c['different_words'] for c in self.checked):
            raise AssertionError('acoustic replay contains different words')

    def save(self):
        from woof.hex import numerics
        self.receipt.update(numerics=numerics.mode(), outer_steps=self.step,
                            compared_words=sum(c['words'] for c in self.checked),
                            different_words=sum(c['different_words'] for c in self.checked),
                            checked_steps=sorted({c['step'] for c in self.checked}))
        self.path.write_text(json.dumps(self.receipt, indent=2) + '\n')

    def install(self):
        from woof.hex import cuda_driver_rrfs as driver
        self.new._kernel = self.kernel
        driver.advance_acoustic_step_cuda_rrfs = self.advance
        original_step = driver.CudaRRFSDycoreDriver._step_device_v841
        replay = self
        @wraps(original_step)
        def step(instance, *args, **kwargs):
            replay.step += 1
            replay.substep = None
            result = original_step(instance, *args, **kwargs)
            replay.save()
            print(f'ACOUSTIC_REPLAY step={replay.step} comparisons={len(replay.checked)}', flush=True)
            return result
        driver.CudaRRFSDycoreDriver._step_device_v841 = step


def main():
    # Offline-compiled images keep acoustic NVRTC work outside the GPU lease.
    image_dir = os.environ.get('RRFS_ACOUSTIC_IMAGES')
    if image_dir:
        from cupy.cuda import compiler
        original_compile = compiler._compile_using_nvrtc_no_warning
        def compile_cached(source, options=(), *args, **kwargs):
            key = hashlib.sha256((source + repr(tuple(options))).encode()).hexdigest()
            image = Path(image_dir) / (key + '.cubin')
            if image.exists():
                return image.read_bytes(), None
            return original_compile(source, options, *args, **kwargs)
        compiler._compile_using_nvrtc_no_warning = compile_cached
    from woof.hex import cli
    args = cli.build_parser().parse_args(sys.argv[1:])
    cli._select_numerics(args)
    if os.environ.get('RRFS_ACOUSTIC_SMOKE') == '1':
        return int(args.handler(args))
    replay = Replay(os.environ['RRFS_ACOUSTIC_BASE'], os.environ['RRFS_ACOUSTIC_RECEIPT'])
    replay.install()
    try:
        status = int(args.handler(args))
        if status == 0:
            replay.validate()
        return status
    finally:
        replay.save()


if __name__ == '__main__':
    sys.exit(main())
