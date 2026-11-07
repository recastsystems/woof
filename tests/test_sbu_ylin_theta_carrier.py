"""SBU columns consume dry theta under either thermodynamic carrier."""

from types import SimpleNamespace
import sys

import numpy as np
import pytest


@pytest.mark.parametrize("carrier", [None, 0, 1])
@pytest.mark.parametrize("base_3d", [False, True])
def test_preparation_consumes_selected_dry_theta(monkeypatch, carrier, base_3d):
    from woof.core import column_microphysics as column
    from woof.core import constants as c

    # CPU arrays exercise the production preparation without opening a card.
    # The state accessor owns the moist-to-dry conversion in the settings port.
    monkeypatch.setitem(sys.modules, "cupy", np)
    shape = (3, 1, 2)
    qv = np.array([.001, .01, .005, .02, .003, .015], np.float32).reshape(shape)
    dry = np.array([299., 302., 306., 308., 310., 315.], np.float32).reshape(shape)
    factor = np.float32(1) + np.float32(c.RVOVRD) * qv
    full = dry * factor if carrier == 1 else dry
    base = np.full(3, 290., np.float32)
    thb = np.broadcast_to(base[:, None, None], shape).copy() if base_3d else base
    state = SimpleNamespace(
        p=np.full(shape, 90000., np.float32), thb=thb,
        thp=full - base[:, None, None], qv=qv,
        phb=np.arange(4, dtype=np.float32) * np.float32(c.G * 100),
        php=np.zeros((4, 1, 2), np.float32), alt=np.ones(shape, np.float32),
        ht=np.zeros((1, 2), np.float32))
    if carrier is not None:
        state.use_theta_m = carrier
    slots = {}
    state.scratch = lambda size, name: slots.setdefault(name, np.empty(size, np.float32))
    represented = base[:, None, None] + state.thp
    expected = represented / factor if carrier == 1 else represented
    calls = []

    def total_theta():
        assert carrier == 1, "default preparation must retain its original arithmetic"
        calls.append(True)
        return represented / factor

    state.total_theta = total_theta
    observed, _ = column.prepare_inputs(state)
    np.testing.assert_array_equal(observed.view(np.uint32), expected.view(np.uint32))
    assert len(calls) == (1 if carrier == 1 else 0)
    if carrier == 1:
        assert np.any(expected.view(np.uint32) != represented.view(np.uint32))
