"""Carried physics uses the same restriction in resident and bounded stores."""
from types import SimpleNamespace

import numpy as np
import pytest

from woof.core.nest_physics_feedback import feedback_physics
from woof.core.nest_interp import register_nest


@pytest.mark.parametrize("ratio", [2, 3])
def test_bounded_physics_matches_resident_restriction(ratio):
    cp = pytest.importorskip("cupy")
    rng = np.random.default_rng(309)
    nx, ny, cx, cy = 24, 23, 12 * ratio, 11 * ratio
    reg = register_nest(nri=ratio, nrj=ratio, i_parent_start=5,
                        j_parent_start=5, child_nx=cx, child_ny=cy,
                        parent_nx=nx, parent_ny=ny)
    parent = {"t2": rng.normal(size=(ny, nx)).astype("f4"),
              "smois": rng.random((4, ny, nx), dtype="f4"),
              "ivgtyp": rng.integers(1, 25, (ny, nx), dtype="i4")}
    child = {"t2": rng.normal(size=(cy, cx)).astype("f4"),
             "smois": rng.random((4, cy, cx), dtype="f4"),
             "ivgtyp": rng.integers(1, 25, (cy, cx), dtype="i4")}

    def run(bounded):
        p = {k: (v.copy() if bounded else cp.asarray(v)) for k, v in parent.items()}
        c = {k: (v.copy() if bounded else cp.asarray(v)) for k, v in child.items()}

        def state(fields):
            if bounded:
                return SimpleNamespace(_streamed_domain=SimpleNamespace(
                    store={"fields/" + k: v for k, v in fields.items()}))
            return SimpleNamespace(physics=SimpleNamespace(fields=fields))

        node = SimpleNamespace(state=state(c), parent=SimpleNamespace(state=state(p)),
                               cfg=SimpleNamespace(run=SimpleNamespace(spec_zone=1)))
        coupler = SimpleNamespace(registrations={"m": reg}, feedback_sync_bytes=0,
                                  _transaction_chunk_shape=lambda: (8, 9))
        feedback_physics(coupler, node)
        return {k: v if bounded else cp.asnumpy(v) for k, v in p.items()}, coupler

    resident, _ = run(False)
    bounded, coupler = run(True)
    for name in parent:
        np.testing.assert_array_equal(resident[name].view("u4"), bounded[name].view("u4"))
        assert np.any(resident[name] != parent[name])
        np.testing.assert_array_equal(resident[name][..., :3, :], parent[name][..., :3, :])
    assert coupler.feedback_sync_bytes > 0
