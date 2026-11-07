"""A completed status read must not retain a scheme's atmospheric inputs."""
import sys
from contextlib import nullcontext
from types import SimpleNamespace
import weakref

import numpy as np
import pytest

from woof.core.health_ledger import HealthLedger


@pytest.fixture
def ledger(monkeypatch):
    # Exercise the real OR/read/clear protocol without opening a CUDA context.
    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace(
        zeros=np.zeros, uint32=np.uint32, bitwise_or=np.bitwise_or,
        asnumpy=lambda value: value.copy(),
        cuda=SimpleNamespace(runtime=SimpleNamespace(getDevice=lambda: 0),
            Device=lambda _: nullcontext(), get_current_stream=nullcontext)))
    return HealthLedger()


def test_successful_drain_releases_inputs_and_reuses_slot(ledger):
    value = np.ones(16, dtype=np.float32)
    ref = weakref.ref(value)
    ledger.record("ysu", np.array([0], dtype=np.uint32),
                  lambda flags, inputs=value: pytest.fail(str(inputs)))
    del value
    assert ref() is not None
    ledger.drain()
    assert ref() is None
    assert ledger._describe == {}
    assert ledger._index == {"ysu": 0}
    seen = []

    def refuse(flags):
        seen.append(flags)
        raise FloatingPointError("replacement inputs")

    ledger.record("ysu", np.array([2], dtype=np.uint32), refuse)
    ledger.record("ysu", np.array([4], dtype=np.uint32), refuse)
    with pytest.raises(FloatingPointError, match="replacement inputs"):
        ledger.drain()
    assert seen == [6]
    assert ledger.records == 3
    assert ledger.drains == 2
    assert not ledger.slots.any()
    assert ledger._describe["ysu"] is refuse


def test_no_record_after_successful_drain_needs_no_old_description(ledger):
    ledger.record("ysu", np.array([0], dtype=np.uint32),
                  lambda flags: pytest.fail(str(flags)))
    ledger.drain()
    ledger.drain()
    assert ledger.drains == 1  # No new descriptions means no second read.
    assert not ledger.slots.any()
    assert ledger._describe == {}


def test_failed_drain_keeps_diagnostics_and_original_site_order(ledger):
    values = [np.ones(16), np.ones(16)]
    refs = [weakref.ref(value) for value in values]

    def description(name, value):
        def refuse(flags):
            assert value.size == 16
            raise FloatingPointError(f"{name}: {flags}")
        return refuse

    for name, value in zip(("ysu", "kf"), values):
        ledger.record(name, np.array([1], dtype=np.uint32),
                      description(name, value))
    del value, values
    with pytest.raises(FloatingPointError, match="ysu: 1"):
        ledger.drain()
    assert all(ref() is not None for ref in refs)
    assert not ledger.slots.any()
    ledger.drain()  # Already consumed flags cannot repeat a historical error.
    assert all(ref() is None for ref in refs)
