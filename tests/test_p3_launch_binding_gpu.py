"""Persistent domains reuse launch metadata and safely bind replaced arrays."""
import numpy as np

from conftest import requires_gpu


def _setup():
    import cupy as cp
    from woof.core import p3_device as pd
    from test_p3_cuda_gpu import _tiny_column_case

    host, nk, ncol = _tiny_column_case()
    fields = {n: cp.asarray(v) for n, v in host.items()}
    fields["qzi"] = 6 * fields["qi"] / cp.float32(np.pi * 300)
    fields["nc"].fill(1.e8)
    diag = {n: cp.zeros((nk, ncol), cp.float32) for n in pd.DIAG_SLOTS}
    surf = {n: cp.zeros(ncol, cp.float32) for n in pd.SURF_SLOTS}
    workspace = pd.make_workspace(ncol, nk, three_moment=True)
    tables = pd.device_tables(three_moment=True)
    return fields, diag, surf, workspace, tables


def _step(fields, diag, surf, workspace, tables, it):
    from woof.core import p3_device as pd
    pd.run_p3_device(fields, diag, surf, workspace=workspace, tables=tables,
                     dt=20., it=it, arm="sedlevels",
                     log_predictNc=True, log_3momentIce=True)


@requires_gpu
def test_repeated_step_has_no_host_array_upload(monkeypatch):
    import cupy as cp

    fields, diag, surf, ws, tables = _setup()
    _step(fields, diag, surf, ws, tables, 1)
    binding = ws.launch_binding
    def reject_upload(*args, **kwargs):
        raise AssertionError("a resident three-moment step uploaded a host array")
    with monkeypatch.context() as patch:
        patch.setattr(cp, "asarray", reject_upload)
        _step(fields, diag, surf, ws, tables, 2)
    assert ws.launch_binding is binding
    assert bool(cp.isfinite(fields["qzi"]).all())


@requires_gpu
def test_replaced_arrays_rebind_and_match_fresh_workspace_bitwise():
    import cupy as cp
    from woof.core import p3_device as pd

    fields, diag, surf, ws, tables = _setup()
    _step(fields, diag, surf, ws, tables, 1)
    old_binding = ws.launch_binding
    # Replace all three argument groups, as a fresh state after restore can.
    groups = [{n: v.copy() for n, v in group.items()}
              for group in (fields, diag, surf)]
    reference = [{n: v.copy() for n, v in group.items()} for group in groups]
    fresh = pd.make_workspace(ws.ncol, ws.nk, three_moment=True)
    _step(*groups, ws, tables, 2)
    _step(*reference, fresh, tables, 2)
    assert ws.launch_binding is not old_binding
    for actual, expected in zip(groups, reference):
        for name in actual:
            np.testing.assert_array_equal(cp.asnumpy(actual[name]).view(np.uint32),
                                          cp.asnumpy(expected[name]).view(np.uint32))
