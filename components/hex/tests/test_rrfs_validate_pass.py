"""Real-call replay helpers and corruption coverage for level-parallel checks.

The lane launcher installs KernelReplay around the two production launchers.
GPU tests run only under the pool with RRFS_VALIDATE_BASE_SRC naming the base
hexcore directory. No captured weather fields are embedded in this test.
"""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path

import numpy as np


def baseline_source(module: str, root: Path) -> str:
    from woof.hex.cuda_fp32 import CUDA_FTZ_HELPERS
    tree = ast.parse((root / f"{module}.py").read_text())
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "_CUDA_SOURCE"
                              for t in node.targets))
    return eval(compile(ast.Expression(assignment.value), str(root), "eval"),
                {"CUDA_FTZ_HELPERS": CUDA_FTZ_HELPERS})


class KernelReplay:
    """Restore identical inputs, compare every array argument, leave new output."""

    def __init__(self, baseline: Path, receipt: Path):
        import cupy as cp
        self.cp = cp
        self.baseline = baseline
        self.receipt = receipt
        self.step = 0
        self.calls = {}
        self.sources = {}
        self.kernels = {}
        self.baseline_caches = {}

    def run(self, module, name, count, args, cache, launch):
        if self.step not in (1, 12, 24):
            return launch()
        cp = self.cp
        key = (self.step, name)
        self.calls[key] = self.calls.get(key, 0) + 1
        if module not in self.sources:
            self.sources[module] = baseline_source(module, self.baseline)
        source = self.sources[module]
        kernel_key = (id(cache), name)
        if kernel_key not in self.kernels:
            # Compiler dispatch is module-key-sensitive (carrier GNU trig).
            # Keep the canonical key in an independent cache of the same class.
            if id(cache) not in self.baseline_caches:
                self.baseline_caches[id(cache)] = type(cache)(
                    capability=cache.capability, cache_dir=cache.cache_directory,
                    base_options=cache.base_options)
            self.kernels[kernel_key] = self.baseline_caches[id(cache)].raw_kernel(
                name, source, module_key=f"hexcore.{module}")
        old = self.kernels[kernel_key]
        old_count = max(int(args[1]), int(args[2])) if name.startswith("validate_") else int(args[1])
        arrays = {a.data.ptr: a for a in args if isinstance(a, cp.ndarray)}
        snapshots = {ptr: a.copy() for ptr, a in arrays.items()}

        def restore():
            for ptr, a in arrays.items():
                cp.copyto(a, snapshots[ptr])

        def before():
            old(((old_count + 127) // 128,), (128,), args)

        before()
        expected = {ptr: cp.asnumpy(a).view(np.uint32) for ptr, a in arrays.items()}
        restore()
        launch()
        words = sum(a.size for a in arrays.values())
        different = sum(int(np.count_nonzero(cp.asnumpy(a).view(np.uint32) != expected[ptr]))
                        for ptr, a in arrays.items())
        outputs = args[-1:] if name.startswith("validate_") else args[-6:]
        output_words = sum(a.size for a in outputs)
        output_different = sum(int(np.count_nonzero(cp.asnumpy(a).view(np.uint32) != expected[a.data.ptr]))
                               for a in outputs)
        row = dict(step=self.step, kernel=name, call=self.calls[key], words=words,
                   different=different, output_words=output_words,
                   output_different=output_different, old_count=old_count, new_count=count)
        if self.calls[key] == 1:
            import importlib
            current_module = importlib.import_module(f"hexcore.{module}")
            current = (current_module._kernel(name, cache) if name.startswith("validate_")
                       else current_module._kernel(cache, name))
            def after():
                current(((count + 127) // 128,), (128,), args)
            # CUDA graphs remove Python launch gaps from the kernel-only interval.
            # Each invocation fully overwrites winds or ORs a sticky validation flag.
            timings = [[], []]
            cp.cuda.get_current_stream().synchronize()
            stream = cp.cuda.Stream(non_blocking=True)
            graphs = []
            with stream:
                for function in (before, after):
                    stream.begin_capture()
                    for _ in range(32):
                        function()
                    graphs.append(stream.end_capture())
                for iteration in range(11):
                    for which in (iteration % 2, 1 - iteration % 2):
                        restore()
                        start, end = cp.cuda.Event(), cp.cuda.Event()
                        start.record()
                        graphs[which].launch(stream)
                        end.record()
                        end.synchronize()
                        timings[which].append(float(cp.cuda.get_elapsed_time(start, end))/32)
            row.update(old_ms=float(np.median(timings[0])), new_ms=float(np.median(timings[1])),
                       samples_ms=timings, graph_calls_per_sample=32)
            restore()
            launch()
        with self.receipt.open("a") as output:
            output.write(json.dumps(row, sort_keys=True) + "\n")
        assert different == 0, row


def test_validate_corruption_all_fields():
    """NaN, infinity and nonpositive density, including final interface/edge."""
    import pytest
    root = os.environ.get("RRFS_VALIDATE_BASE_SRC")
    if not root:
        pytest.skip("pool-only GPU proof requires RRFS_VALIDATE_BASE_SRC")
    import cupy as cp
    from woof.hex import cuda_dynamics_v841 as dynamics
    from woof.hex.cuda_backend import KernelCache
    cache = KernelCache()
    old = cache.raw_kernel("validate_recovered_v841_f32",
                           baseline_source("cuda_dynamics_v841", Path(root)),
                           module_key="hexcore.cuda_dynamics_v841.validate_base")
    new = dynamics._kernel("validate_recovered_v841_f32", cache)
    nl, nc, ne = 3, 129, 137  # tail lanes and different cell/edge strides
    sizes = [nl*nc, nl*nc, nl*ne, (nl+1)*nc, nl*nc, nl*nc,
             nl*nc, nl*nc, nl*nc, nl*ne, (nl+1)*nc]
    fields = [cp.ones(n, dtype=cp.float32) for n in sizes]
    flag = cp.zeros(1, dtype=cp.int32)
    args = (np.int32(nl), np.int32(nc), np.int32(ne), *fields, flag)

    def pair(expected):
        for kernel, total in ((old, max(nc, ne)), (new, max((nl+1)*nc, nl*ne))):
            flag.fill(0)
            kernel(((total+127)//128,), (128,), args)
            assert int(flag.get()[0]) == expected
    pair(0)
    for field in fields:
        for index in (0, field.size-1):
            for value in (np.nan, np.inf, -np.inf):
                field[index] = value
                pair(1)
            field[index] = 1
    for index in (0, 1, 4, 5):
        for value in (0, -1):
            fields[index][-1] = value
            pair(1)
        fields[index][-1] = 1
    # A later valid recovery must never clear an earlier failure.
    flag.fill(1)
    new(((max((nl+1)*nc, nl*ne)+127)//128,), (128,), args)
    assert int(flag.get()[0]) == 1


def test_reconstruct_direct_cell_count_and_refusals():
    """Resident callers pass cells; all levels and sealed-geometry checks survive."""
    import pytest
    root = os.environ.get("RRFS_VALIDATE_BASE_SRC")
    if not root:
        pytest.skip("pool-only GPU proof requires RRFS_VALIDATE_BASE_SRC")
    import cupy as cp
    from woof.hex import cuda_physics_prep_v841 as prep
    from woof.hex.mpasphys.compile_v842 import MpasPhysicsKernelCache
    cache = MpasPhysicsKernelCache()
    old_cache = MpasPhysicsKernelCache(capability=cache.capability, cache_dir=cache.cache_directory)
    old = old_cache.raw_kernel("reconstruct_winds_v841_f32",
                           baseline_source("cuda_physics_prep_v841", Path(root)),
                           module_key="hexcore.cuda_physics_prep_v841")
    nl, nc, ne, me = 3, 129, 137, 3
    edges = cp.asarray((np.arange(nc*me) % ne).astype(np.int32))
    counts = cp.full(nc, me, dtype=cp.int32)
    coeffs = cp.asarray(np.random.default_rng(41).normal(size=nc*me*3).astype(np.float32))
    lat = cp.linspace(-1, 1, nc, dtype=cp.float32)
    lon = cp.linspace(-3, 3, nc, dtype=cp.float32)
    velocity = cp.asarray(np.random.default_rng(42).normal(size=(nl, ne)).astype(np.float32))
    geometry = [edges, counts, coeffs, lat, lon]
    sealed = [a.copy() for a in geometry]
    outputs = [cp.full((nl, nc), -999, dtype=cp.float32) for _ in range(5)]
    flag = cp.zeros(1, dtype=cp.int32)
    args = (np.int32(nl), np.int32(nc), np.int32(ne), np.int32(me), velocity,
            *geometry, *sealed, *outputs, flag)

    def pair(expected_flag):
        for a in outputs: a.fill(-999)
        flag.fill(0)
        old(((nc+127)//128,), (128,), args)
        expected = [a.get().view(np.uint32) for a in (*outputs, flag)]
        for a in outputs: a.fill(-999)
        flag.fill(0)
        prep._launch(cache, "reconstruct_winds_v841_f32", nc, args)
        for a, b in zip((*outputs, flag), expected):
            np.testing.assert_array_equal(a.get().view(np.uint32), b)
        assert int(flag.get()[0]) == expected_flag
    pair(0)
    # Changed seal: no output write in the affected column, including upper levels.
    coeffs[-1] += 1
    pair(1)
    coeffs[-1] = sealed[2][-1]
    # Invalid count, invalid edge, nonfinite coefficient and upper-level wind.
    counts[-1] = sealed[1][-1] = me+1
    pair(1)
    counts[-1] = sealed[1][-1] = me
    edges[-1] = sealed[0][-1] = ne
    pair(1)
    edges[-1] = sealed[0][-1] = 0
    coeffs[-1] = sealed[2][-1] = np.nan
    pair(1)
    coeffs[-1] = sealed[2][-1] = 1
    velocity[-1,0] = np.inf
    pair(1)
