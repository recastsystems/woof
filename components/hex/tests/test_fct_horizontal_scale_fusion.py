"""The fused FCT row and its existing LTS interception point."""
from types import SimpleNamespace

import numpy as np
import pytest


def test_fused_low_order_keeps_the_lts_hook_before_the_accumulation(monkeypatch):
    from woof.hex import cuda_fullstep_lts_scalars as lts
    from woof.hex import cuda_transport_v841 as transport

    name = "fct_horizontal_low_order_scale_elem_v841"
    assert name in lts._LOW_ORDER
    cache = object()
    events = []
    arguments = tuple(np.int32(v) for v in (2, 55, 129, 389, 11)) + tuple(
        object() for _ in range(12))
    hook = object.__new__(lts._ClassScalars)
    hook.owner = SimpleNamespace(ntr=2, nlev=55, cp=None, reg_scalar=np.zeros(2), hold=np.zeros(2))
    hook.amounts = np.zeros(6)  # the pass's rows are sliced up front since lts-share 7b2ac42a
    hook.klass = SimpleNamespace(cell_stride=129)
    hook.vertical = (None,) * 6
    hook.species = None     # every tracer in one pass, as _ClassScalars.__init__ sets it (lts-share 7b2ac42a)
    hook.has_coarse_side = hook.has_fine_side = False

    def before(args):
        assert args is arguments
        lts._ClassScalars.on_low_order(hook, args)
        events.append("hook")

    def kernel(grid, block, args):
        assert hook.vertical is None
        assert args is arguments
        events.append("kernel")

    monkeypatch.setattr(transport, "_KERNELS", {(cache, name): kernel})
    lts.tap_kernel(transport, cache, name, before=before)
    transport._launch(name, 2 * 55 * 129, arguments, cache)
    assert events == ["hook", "kernel"]



@pytest.mark.parametrize("name", ("fct_horizontal_low_order_scale_elem_v841", "fct_finish_elem_v841"))
def test_lts_hook_reads_native_stencils_and_the_kernel_slot_major(monkeypatch, name):
    """rrfs/s6-scalar-default 04b17ec0 relaid the stencils in _launch, above the tap, so
    lts_donor_guard (and the finish audit) indexed the slot-major tables as (cell, slot)."""
    from woof.hex import cuda_fullstep_lts_scalars as lts
    from woof.hex import cuda_transport_v841 as transport

    cp = SimpleNamespace(cuda=SimpleNamespace(Device=lambda: SimpleNamespace(id=0),
                                              get_current_stream=lambda: SimpleNamespace(ptr=1)),
                         zeros=np.zeros)
    monkeypatch.setattr(transport, "_cupy", lambda: cp)
    monkeypatch.setattr(transport, "_EDGE_STENCILS", {})
    monkeypatch.setattr(transport, "_KERNELS", {})
    ncells, max_edges = 7, 6
    edges = np.arange(ncells * max_edges, dtype=np.int32).reshape(ncells, max_edges)
    signs = np.linspace(-1, 1, ncells * max_edges, dtype=np.float32).reshape(ncells, max_edges)
    arguments = (np.int32(2), np.int32(3), np.int32(ncells), np.int32(20), np.int32(max_edges),
                 np.full(ncells, max_edges, np.int32), edges, signs) + tuple(object() for _ in range(9))
    seen = {}

    class Cache:
        def raw_kernel(self, kernel_name, source, *, module_key):
            def kernel(grid, block, args):
                seen["kernel"] = args
            return kernel

    cache = Cache()
    lts.tap_kernel(transport, cache, name, before=lambda args: seen.setdefault("hook", args))
    transport._launch(name, 2 * 3 * ncells, arguments, cache)
    assert seen["hook"] is arguments
    assert seen["hook"][6] is edges and seen["hook"][7] is signs
    kernel_args = seen["kernel"]
    np.testing.assert_array_equal(kernel_args[6], edges.T)
    np.testing.assert_array_equal(kernel_args[7], signs.T)
    assert kernel_args[6].flags.c_contiguous and kernel_args[7].flags.c_contiguous
    assert all(a is b for i, (a, b) in enumerate(zip(kernel_args, arguments)) if i not in (6, 7))

@pytest.mark.gpu
def test_fused_horizontal_scale_matches_separate_kernels_word_for_word():
    cp = pytest.importorskip("cupy")
    from woof.hex.cuda_transport_v841 import _launch

    ntracers, nlev, ncells, nedges, width = 2, 55, 129, 389, 11
    rng = np.random.default_rng(641)
    shape = (ntracers, nlev, ncells)

    def field(shape):
        values = rng.uniform(-1, 1, shape).astype("f4")
        values.ravel()[:10] = np.array([
            0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x807fffff,
            0x00800000, 0x80800000, 0x3f800000, 0xbf800000,
        ], dtype="u4").view("f4")
        return cp.asarray(values)

    counts = np.resize(np.array([0, 5, 6, 10, 11], dtype="i4"), ncells)
    inputs = tuple(np.int32(v) for v in (ntracers, nlev, ncells, nedges, width)) + (
        cp.asarray(counts),
        cp.asarray(rng.integers(0, nedges, (ncells, width), dtype="i4")),
        cp.asarray(rng.choice([-1, 1], (ncells, width)).astype("f4")),
        cp.asarray(rng.uniform(.1, 2, ncells).astype("f4")),
        field((ntracers, nlev, nedges)), field((ntracers, nlev, nedges)),
    )
    mass, incoming, outgoing = (field(shape) for _ in range(3))
    minimum = -cp.abs(field(shape))
    maximum = cp.abs(field(shape))
    density = cp.asarray(rng.uniform(.1, 2, (nlev, ncells)).astype("f4"))
    reference = tuple(value.copy() for value in (mass, incoming, outgoing))
    total = ntracers * nlev * ncells
    _launch("fct_horizontal_low_order_scale_elem_v841", total,
            (*inputs, mass, incoming, outgoing, minimum, maximum, density), None)
    _launch("fct_horizontal_low_order_elem_v841", total, (*inputs, *reference), None)
    _launch("fct_scale_elem_v841", total,
            (*inputs[:3], minimum, maximum, density, *reference), None)
    for actual, expected in zip((mass, incoming, outgoing), reference):
        np.testing.assert_array_equal(cp.asnumpy(actual).view("u4"),
                                      cp.asnumpy(expected).view("u4"))
    # Both clipping endpoints and a nontrivial limiter must be exercised.
    factors = cp.concatenate((incoming.ravel(), outgoing.ravel()))
    assert bool(cp.any(factors == 0))
    assert bool(cp.any(factors == 1))
    assert bool(cp.any((factors > 0) & (factors < 1)))


@pytest.mark.parametrize("source", ("flag", "namelist"))
def test_replay_rejects_a_different_forecast_mode_before_cuda(
        monkeypatch, tmp_path, capsys, source):
    from pathlib import Path
    import runpy
    import sys

    from woof.hex import cuda_transport_v841 as transport, numerics

    monkeypatch.setattr(numerics, "_mode", None)
    monkeypatch.setattr(numerics, "_bound", False)
    monkeypatch.setenv("WOOF_HEX_NUMERICS", "fast")
    monkeypatch.setenv("GPUWM_NUMERICS", "fast")
    monkeypatch.setattr(transport, "_cupy", lambda: pytest.fail("CUDA started before mode validation"))
    forecast = ["forecast"]
    if source == "flag":
        forecast += ["--numerics", "fast"]
    else:
        namelist = tmp_path / "physics.nml"
        namelist.write_text("&config\nconfig_numerics = 'fast'\n/\n")
        forecast += ["--physics-namelist", str(namelist)]
    script = Path(__file__).resolve().parents[1] / "tools/replay_fct_horizontal_scale.py"
    monkeypatch.setattr(sys, "argv", [str(script), "--numerics", "strict", "--shape", "H60",
                                   "--receipt", str(tmp_path / "receipt.json"), "--", *forecast])
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(script))["main"]()
    assert error.value.code == 2
    assert "forecast selects fast numerics; replay requests strict" in capsys.readouterr().err
    assert not (tmp_path / "receipt.json").exists()
