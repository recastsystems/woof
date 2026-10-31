"""Writer refusal ordering, including an unfinished device-check snapshot."""
from contextlib import nullcontext
from datetime import timedelta
import importlib.util
from pathlib import Path
import os
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tests'))
from test_mpasphys_streams_output_v842 import session, stream
from woof.hex.mpasphys.streams_output_v842 import PhysicsStreams


def collector():
    source = os.environ.get('OUTPUT_CHECKS_SOURCE')
    if not source:
        from woof.hex.mpasphys.deferred_checks_v842 import DeferredChecks
        if not hasattr(DeferredChecks, 'output_validator'):
            pytest.skip('requires the companion collector patch')
    else:
        spec = importlib.util.spec_from_file_location('proposal_checks', source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        DeferredChecks = module.DeferredChecks
    checks = DeferredChecks(SimpleNamespace(cuda=SimpleNamespace(Device=nullcontext)))
    checks.step = 5
    return checks


class PendingEvent:
    def __init__(self):
        self.released = threading.Event()
        self.waiting = threading.Event()
        self.threads = []

    @property
    def done(self):
        return self.released.is_set()

    def synchronize(self):
        self.threads.append(threading.current_thread().name)
        self.waiting.set()
        assert self.released.wait(3), 'test did not release the simulated GPU check'


def batch(event, value, step=5):
    return event, (None, None, np.array([value], dtype=np.int64)), [('bad word {value}', ValueError)], step


@pytest.mark.parametrize('value', [0, 7])
def test_frame_validation_waits_on_writer_and_refuses_before_encoding(tmp_path, value):
    checks = collector()
    event = PendingEvent()
    checks._inflight.append(batch(event, value))
    state, _, _, start = session(resident=True)
    writer = PhysicsStreams(state, (stream(),), tmp_path, final_time=start+timedelta(minutes=2))
    writer.output_validator = checks.output_validator
    encoded = []
    writer._encode = lambda *args: encoded.append(threading.current_thread().name) or []
    try:
        writer.write(step=0, dt=60, receipt={})
        assert event.waiting.wait(2)
        assert encoded == []
        assert event.threads == ['mpas-output_0']
        event.released.set()
        if value:
            with pytest.raises(ValueError, match='bad word 7.*step 5'):
                writer.close()
            assert encoded == []
        else:
            writer.close()
            assert encoded == ['mpas-output_0']
    finally:
        event.released.set()
        if not writer._closed:
            writer.close()
    assert not list(tmp_path.glob('*.nc'))


def test_validator_owns_its_frame_batches_not_later_checks():
    checks = collector()
    first = PendingEvent()
    checks._inflight.append(batch(first, 0))
    validate = checks.output_validator()
    # Main collector can retire its list and enqueue a later refusal while the
    # first frame still owns its immutable buffers.
    first.released.set()
    checks.poll(block=False)
    later = PendingEvent()
    later.released.set()
    checks._inflight.append(batch(later, 9, step=6))
    validate()
    with pytest.raises(ValueError, match='bad word 9.*step 6'):
        checks.poll(block=True)
