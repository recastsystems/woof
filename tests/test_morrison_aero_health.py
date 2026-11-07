"""Positive MP40 must retain activation faults without draining each step."""
import numpy as np
import pytest
from conftest import requires_gpu

@requires_gpu
@pytest.mark.parametrize('deferred', [False, True])
def test_activation_health_preserves_transient_fault(deferred):
    import cupy as cp
    from woof.core.health_ledger import HealthLedger, deferring
    from woof.core.morrison_aero import _check_activation_status

    flags = cp.zeros((5, 3, 7), dtype=cp.float32)
    flags[2, 1, 4] = np.float32(1)
    if deferred:
        ledger = HealthLedger()
        # The producer overwrites its scratch on the next call. The ledger
        # must keep the earlier invalid ice count until publication/drain.
        with deferring(ledger):
            _check_activation_status(flags)
            flags.fill(0)
            _check_activation_status(flags)
        with pytest.raises(ValueError, match='prescribed ice nucleation'):
            ledger.drain()
        ledger.drain()
    else:
        with pytest.raises(ValueError, match='prescribed ice nucleation'):
            _check_activation_status(flags)

@requires_gpu
def test_activation_health_can_be_captured_without_host_read():
    import cupy as cp
    from woof.core.health_ledger import HealthLedger, deferring
    from woof.core.morrison_aero import _check_activation_status

    flags = cp.zeros((5, 3, 7), dtype=cp.float32)
    ledger = HealthLedger()
    with deferring(ledger):
        _check_activation_status(flags)
    ledger.drain()
    stream = cp.cuda.Stream(non_blocking=True)
    cp.cuda.get_current_stream().synchronize()
    with stream, deferring(ledger):
        stream.begin_capture()
        _check_activation_status(flags)
        graph = stream.end_capture()
        flags.fill(1)
        graph.launch(stream)
    with pytest.raises(ValueError, match='prescribed ice nucleation'):
        ledger.drain()
