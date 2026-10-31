"""Scalar transport word replay on real forecast calls, plus stencil corner cases.

Run this file as the forecast LAUNCHER with SCALAR_AB_BASE pointing to the
baseline cuda_transport_rrfs.py and SCALAR_AB_OUT naming a JSONL receipt.
SCALAR_AB_IMAGES supplies CPU-precompiled baseline/candidate images. No host
instrumentation is installed in the production module.
"""
from __future__ import annotations

from functools import wraps
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys

OUTPUTS = {
    "scalar_edge_rrfs": (-1,),
    "scalar_finish_rrfs": (-2, -1),
    "gsl_scalar_laplacian_rrfs": (-2, -1),
    "gsl_scalar_filter4_rrfs": (-1,),
}


def baseline():
    spec = importlib.util.spec_from_file_location(
        "woof.hex._scalar_edge_baseline", os.environ["SCALAR_AB_BASE"])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_image(label):
    from cupy.cuda import function
    from woof.hex import numerics
    manifest = json.loads((Path(os.environ["SCALAR_AB_IMAGES"]) / "manifest.json").read_text())
    if label in ("base", "candidate"):
        from woof.hex import cuda_transport_rrfs
        source = baseline()._CUDA_SOURCE if label == "base" else cuda_transport_rrfs._CUDA_SOURCE
        row = next(row for row in manifest if row["label"] == label and row["mode"] == numerics.mode())
        assert row["source_sha256"] == hashlib.sha256(source.encode()).hexdigest(), "stale proof image"
    module = function.Module()
    module.load((Path(os.environ["SCALAR_AB_IMAGES"]) /
                 f"{label}-{numerics.mode()}.cubin").read_bytes())
    return module


def emit(row):
    from woof.hex import numerics
    row["numerics"] = numerics.mode()
    with open(os.environ["SCALAR_AB_OUT"], "a") as stream:
        stream.write(json.dumps(row) + "\n")
    print("SCALAR_AB " + json.dumps(row), flush=True)


def compare(name, args, before, after, step):
    import cupy as cp
    rows = []
    for index, previous, current in zip(OUTPUTS[name], before, after):
        words = previous.view(cp.uint32)
        diff = words != current.view(cp.uint32)
        # Include all allocation words, including untouched mask/halo slots.
        rows.append(dict(argument=index, shape=list(current.shape), words=words.size,
                         different=int(cp.count_nonzero(diff)),
                         scalar_differences=cp.asnumpy(diff.reshape(int(args[0]), -1).sum(axis=1)).tolist()
                         if current.ndim == 3 else None))
    row = dict(kernel=name, step=step, outputs=rows,
               words=sum(x["words"] for x in rows),
               different=sum(x["different"] for x in rows))
    emit(row)
    assert row["different"] == 0, row


def median_ms(launch, restore, repetitions=11):
    import cupy as cp
    times = []
    for _ in range(repetitions):
        restore()
        start, end = cp.cuda.Event(), cp.cuda.Event()
        start.record()
        launch()
        end.record()
        end.synchronize()
        times.append(float(cp.cuda.get_elapsed_time(start, end)))
    return statistics.median(times)


class Replay:
    def __init__(self):
        self.layout_guard = os.environ.get("SCALAR_AB_LAYOUT_GUARD") == "1"
        self.layouts = {}
        self.finish_only = os.environ.get("SCALAR_AB_FINISH_ONLY") == "1"
        self.old = load_image("base")
        self.new = load_image("candidate")
        self.mpas = None if self.finish_only or self.layout_guard else load_image("mpas")
        self.step = 0
        self.seen = set()
        self.timed = set()

    def kernel(self, source, module_key, name, cache=None):
        new = self.new.get_function(name)
        if self.finish_only and name != "scalar_finish_rrfs":
            return new
        if self.layout_guard and name not in ("scalar_edge_rrfs", "scalar_finish_rrfs"):
            return new
        if self.step not in (1, 12, 24):
            return new
        old = self.old.get_function(name)

        def launch(grid, block, args, **kw):
            import cupy as cp
            snapshots = [args[i].copy() for i in OUTPUTS[name]]

            def restore():
                for i, snapshot in zip(OUTPUTS[name], snapshots):
                    args[i][...] = snapshot

            old_args = args
            if self.layout_guard:
                old_args = tuple(self.layouts[id(x)][1] if id(x) in self.layouts else x for x in args)
            elif name == "scalar_finish_rrfs":
                old_args = (*args[:8], *(cp.ascontiguousarray(x.T) for x in args[8:10]), *args[10:])
            elif name == "scalar_edge_rrfs":
                old_args = (*args[:14], *(cp.ascontiguousarray(x.T) for x in args[14:17]), *args[17:])
            old_grid = (((int(args[5]) + 127) // 128),) if name.startswith("gsl_") else (grid[0],)
            old(old_grid, block, old_args, **kw)
            previous = [args[i].copy() for i in OUTPUTS[name]]
            restore()
            new(grid, block, args, **kw)
            current = [args[i].copy() for i in OUTPUTS[name]]
            try:
                compare(name, args, previous, current, self.step)
            except AssertionError:
                import numpy as np
                capture = Path(os.environ["SCALAR_AB_OUT"]).with_suffix(".failure.npz")
                np.savez(capture, **{f"arg{i}": cp.asnumpy(value) if isinstance(value, cp.ndarray) else value
                                    for i, value in enumerate(args)},
                         **{f"output{i}": cp.asnumpy(value) for i, value in enumerate(snapshots)})
                raise
            if self.finish_only and self.step == 24 and (24, name) not in self.seen:
                import numpy as np
                from woof.hex import numerics
                capture = Path(os.environ["SCALAR_AB_CAPTURE"]) / f"finish-{numerics.mode()}.npz"
                capture.parent.mkdir(parents=True, exist_ok=True)
                np.savez(capture, **{f"arg{i}": cp.asnumpy(value) if isinstance(value, cp.ndarray) else value
                                    for i, value in enumerate(old_args)})
            self.seen.add((self.step, name))
            if not self.layout_guard and name not in self.timed:
                old_ms = median_ms(lambda: old(old_grid, block, old_args, **kw), restore)
                new_ms = median_ms(lambda: new(grid, block, args, **kw), restore)
                from cupy_backends.cuda.api import driver
                regs = lambda f: driver.funcGetAttribute(driver.CU_FUNC_ATTRIBUTE_NUM_REGS, f.ptr)
                local = lambda f: driver.funcGetAttribute(driver.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, f.ptr)
                timing = dict(kernel=name, step=self.step, repetitions=11,
                              baseline_ms=old_ms, candidate_ms=new_ms,
                              scalars=int(args[0]), old_registers=regs(old),
                              new_registers=regs(new), old_local_bytes=local(old),
                              new_local_bytes=local(new))
                if name == "scalar_edge_rrfs":
                    # Same card, same real stencil/state, eight tracers as in
                    # convection_permitting. This is a kernel comparison, not
                    # a claim of an MPAS-A full forecast profile.
                    import numpy as np
                    ns, nl, nc, ne, width = map(int, args[:5])
                    nmpas = min(8, ns)
                    out = cp.empty((nmpas, nl, ne), dtype=cp.float32)
                    ma = (np.int32(nmpas), *args[1:5], np.float32(1),
                          old_args[11], old_args[12], old_args[15], old_args[16], old_args[13], old_args[14], out)
                    mpas = self.mpas.get_function("transport_edge_values")
                    mpas_launch = lambda: mpas(((ne+127)//128,), (128,), ma)
                    mpas_launch()
                    timing["mpas_ms"] = median_ms(mpas_launch, lambda: None)
                    timing["mpas_scalars"] = nmpas
                    timing["mpas_registers"] = regs(mpas)
                emit(timing)
                self.timed.add(name)
            # Timings restore mutating outputs before every sample. Publish
            # exactly the checked candidate result into the forecast.
            for i, value in zip(OUTPUTS[name], current):
                args[i][...] = value
        return launch


def install():
    from woof.hex import cuda_transport_rrfs as transport
    from woof.hex.cuda_driver_rrfs import CudaRRFSDycoreDriver
    replay = Replay()
    if replay.layout_guard:
        old_transport = baseline()
        original_layout = transport._slot_major_layout

        @wraps(original_layout)
        def layout(arrays, extent):
            current = original_layout(arrays, extent)
            previous = old_transport._slot_major_layout(arrays)
            # Both helpers receive the original mesh arrays independently.
            # Keep the candidate alive so its identity cannot be recycled.
            for new, old in zip(current, previous):
                replay.layouts[id(new)] = (new, old)
            return current

        transport._slot_major_layout = layout
        emit(dict(layout_guard=True, baseline_host_sha256=hashlib.sha256(
            Path(os.environ["SCALAR_AB_BASE"]).read_bytes()).hexdigest(),
            candidate_host_sha256=hashlib.sha256(Path(transport.__file__).read_bytes()).hexdigest()))
    original = CudaRRFSDycoreDriver._step_device_v841

    @wraps(original)
    def step(driver, *args, **kwargs):
        replay.step += 1
        return original(driver, *args, **kwargs)

    emit(dict(shadow_mixing=not (replay.finish_only or replay.layout_guard), filter_coefficient=1.0, writes="private scratch only"))
    original_transport = CudaRRFSDycoreDriver._rrfs_scalar_transport

    @wraps(original_transport)
    def scalar_transport(driver, *args, **kwargs):
        if replay.step in (1, 12, 24) and kwargs["rk_step"] == 1 and not driver.config.config_scalar_eddy_mix:
            # The default match deck disables scalar mixing. Exercise both
            # optional kernels on its REAL state/geometry, writing only private
            # scratch. A unit filter coefficient keeps the filter path live.
            import cupy as cp
            transport.apply_gsl_scalar_mixing_cuda_rrfs(
                args[0], args[1], args[2], cp.zeros_like(args[2]),
                kdiff=driver._rrfs_kdiff, rho_edge=driver._rrfs_scalar_rho_edge,
                mesh_scaling_del2=driver.horizontal.rrfs_scaling_del2,
                mesh_scaling_del4=driver.horizontal.rrfs_scaling_del4,
                h_scalar_filter4=1.0, kernel_cache=driver.cache)
        return original_transport(driver, *args, **kwargs)

    CudaRRFSDycoreDriver._step_device_v841 = step
    if not (replay.finish_only or replay.layout_guard):
        CudaRRFSDycoreDriver._rrfs_scalar_transport = scalar_transport
    transport.kernel = replay.kernel
    return replay


def test_edge_boundary_and_stencil_words():
    """Counts 0/9/10/11, masks 0..7, interval tails and 1/14 species."""
    if "SCALAR_AB_IMAGES" not in os.environ:
        import pytest
        pytest.skip("requires precompiled real baseline and a GPU pool lease")
    import numpy as np
    import cupy as cp
    from woof.hex import cuda_transport_rrfs as transport
    replay = Replay()
    replay.step = 1
    rng = np.random.default_rng(389)
    nl, nc, ne, width = 5, 37, 53, 11
    ints = lambda x: cp.asarray(x, dtype=cp.int32)
    floats = lambda shape: cp.asarray(rng.normal(size=shape), dtype=cp.float32)
    for ns in (1, 14):
        for regional in (0, 1):
            edge = floats((ns, nl, ne))
            args = (*map(np.int32, (ns, nl, nc, ne, width, regional, 3, ne-2)),
                    ints(np.arange(ne) % 8), ints(rng.integers(nc, size=(ne, 2))),
                    floats(ne), floats((ns, nl, nc)), floats((nl, ne)),
                    ints(np.resize([0, 9, 10, 11], ne)), ints(rng.integers(nc, size=(ne, width))),
                    floats((ne, width)), floats((ne, width)), edge)
            args = (*args[:14], *(cp.ascontiguousarray(x.T) for x in args[14:17]), *args[17:])
            replay.kernel(transport._CUDA_SOURCE, "hexcore.cuda_transport_rrfs", "scalar_edge_rrfs")(
                ((nl*ne+127)//128,), (128,), args)


def test_stencil_layout_cache_lifetime_and_streams():
    if "SCALAR_AB_IMAGES" not in os.environ:
        import pytest
        pytest.skip("requires a GPU pool lease")
    import gc
    from types import SimpleNamespace
    import cupy as cp
    from woof.hex import cuda_transport_rrfs as transport
    coefficients = SimpleNamespace(
        adv_cells_for_edge=cp.arange(30, dtype=cp.int32).reshape(6, 5),
        adv_coefs=cp.arange(30, dtype=cp.float32).reshape(6, 5),
        adv_coefs_3rd=cp.arange(30, dtype=cp.float32).reshape(6, 5))
    initial = set(transport._EDGE_STENCILS)
    layout = transport._edge_stencil_layout(coefficients, 6)
    assert transport._edge_stencil_layout(coefficients, 6) is layout
    cp.cuda.get_current_stream().synchronize()
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        other = transport._edge_stencil_layout(coefficients, 6)
        assert other is not layout
        assert transport._edge_stencil_layout(coefficients, 6) is other
        for value, original in zip(other, vars(coefficients).values()):
            assert value.flags.c_contiguous
            assert bool(cp.all(value == original.T))
    stream.synchronize()
    del original, coefficients
    gc.collect()
    assert set(transport._EDGE_STENCILS) == initial



def _cpu_layout_backend(monkeypatch):
    """Exercise host shape/cache logic without allocating on a GPU."""
    import numpy as np
    from types import SimpleNamespace
    from woof.hex import cuda_transport_rrfs as transport
    backend = SimpleNamespace(
        zeros=np.zeros, ascontiguousarray=np.ascontiguousarray,
        cuda=SimpleNamespace(Device=lambda: SimpleNamespace(id=0),
                             get_current_stream=lambda: SimpleNamespace(ptr=0)))
    monkeypatch.setitem(sys.modules, "cupy", backend)
    monkeypatch.setattr(transport, "_EDGE_STENCILS", {})
    return transport


def test_slot_layout_pads_dummy_row(monkeypatch):
    import numpy as np
    transport = _cpu_layout_backend(monkeypatch)
    arrays = (np.arange(12, dtype=np.int32).reshape(4, 3),
              np.arange(12, dtype=np.float32).reshape(4, 3))
    exact = transport._slot_major_layout(arrays, 4)
    padded = transport._slot_major_layout(arrays, 5)
    assert padded is transport._slot_major_layout(arrays, 5)
    assert exact is not padded  # Extent participates in cache identity.
    for source, full, dummy in zip(arrays, exact, padded):
        np.testing.assert_array_equal(full, source.T)
        assert dummy.shape == (3, 5)
        assert dummy.dtype == source.dtype
        assert dummy.flags.c_contiguous
        np.testing.assert_array_equal(dummy[:, :-1], source.T)
        np.testing.assert_array_equal(dummy[:, -1], 0)


def test_slot_layout_rejects_wrong_extent(monkeypatch):
    import numpy as np
    import pytest
    transport = _cpu_layout_backend(monkeypatch)
    arrays = (np.arange(12, dtype=np.int32).reshape(4, 3),)
    for extent in (3, 6):
        with pytest.raises(ValueError, match="native stencil extent differs from physical plus dummy"):
            transport._slot_major_layout(arrays, extent)
    assert not transport._EDGE_STENCILS


def finish_tiled():
    """Repeat independent finish columns, retaining their real edge references.

    This kernel reads no neighbouring cell fields. Repeating each cell-axis
    input therefore preserves every output word without tiling the edge pool.
    The experiment measures the requested column count, not a new mesh.
    """
    import numpy as np
    import cupy as cp
    from woof.hex import numerics
    from woof.hex import cuda_transport_rrfs as transport
    from cupy_backends.cuda.api import driver
    path = Path(os.environ["SCALAR_AB_CAPTURE"]) / f"finish-{numerics.mode()}.npz"
    with np.load(path) as data:
        small = [cp.asarray(data[f"arg{i}"]) if data[f"arg{i}"].ndim else data[f"arg{i}"][()]
                 for i in range(27)]
    ns, nl, nc = map(int, small[:3])
    assert int(small[5]) == nc, "this tiled proof requires a fully solved source mesh"
    repeats = (200000 + nc - 1) // nc
    cells = repeats * nc
    # Scalars/levels remain ordered. No neighbour-cell references are present.
    cell_axes = {6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 13: 1,
                 17: 2, 18: 2, 19: 1, 20: 1, 21: 2}
    large = list(small)
    large[2] = np.int32(cells)
    large[5] = np.int32(cells)
    for index, axis in cell_axes.items():
        shape = [1] * small[index].ndim
        shape[axis] = repeats
        large[index] = cp.tile(small[index], shape)
    large[25] = cp.empty((ns, nl, cells), dtype=cp.float32)
    large[26] = cp.empty((nl, cells), dtype=cp.float32)
    old_image, new_image = load_image("base"), load_image("candidate")
    old = old_image.get_function("scalar_finish_rrfs")
    new = new_image.get_function("scalar_finish_rrfs")
    grid, block = ((nl*cells+127)//128,), (128,)
    old(grid, block, tuple(large))
    baseline_outputs = [large[i].copy() for i in (25, 26)]
    new_args = (*large[:8], *transport._slot_major_layout(tuple(large[8:10]), cells), *large[10:])
    new(grid, block, new_args)
    compare("scalar_finish_rrfs", new_args, baseline_outputs, large[25:], 24)
    differences = []
    for index in (25, 26):
        tiled = large[index].reshape(*small[index].shape[:-1], repeats, nc).view(cp.uint32)
        expected = small[index][..., None, :].view(cp.uint32)
        differences.append(int(cp.count_nonzero(tiled != expected)))
    assert not any(differences), differences
    regs = lambda f: driver.funcGetAttribute(driver.CU_FUNC_ATTRIBUTE_NUM_REGS, f.ptr)
    emit(dict(kernel="scalar_finish_rrfs", tiled_cells=cells, source_cells=nc,
              scalars=ns, levels=nl, repeats=repeats, tiled_differences=differences,
              words=sum(x.size for x in large[25:]), repetitions=11,
              baseline_ms=median_ms(lambda: old(grid, block, tuple(large)), lambda: None),
              candidate_ms=median_ms(lambda: new(grid, block, new_args), lambda: None),
              baseline_registers=regs(old), candidate_registers=regs(new),
              memory_pool_used=cp.get_default_memory_pool().used_bytes(),
              memory_pool_total=cp.get_default_memory_pool().total_bytes()))


if __name__ == "__main__":
    if sys.argv[1:] == ["--finish-tiled"]:
        finish_tiled()
    elif sys.argv[1:] == ["--synthetic"]:
        test_edge_boundary_and_stencil_words()
    else:
        from woof.hex import cli
        args = cli.build_parser().parse_args(sys.argv[1:])
        cli._select_numerics(args)
        replay = install()
        result = int(args.handler(args))
        names = ("scalar_finish_rrfs",) if replay.finish_only else OUTPUTS
        if replay.layout_guard:
            names = ("scalar_edge_rrfs", "scalar_finish_rrfs")
        required = {(step, name) for step in (1, 12, 24) for name in names}
        assert required <= replay.seen, (required - replay.seen)
        sys.exit(result)
