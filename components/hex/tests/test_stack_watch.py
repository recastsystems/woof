"""Exercise periodic stack snapshots while interpreter worker frames retire."""
import io
import threading
import time

import pytest

from woof.hex.stack_watch import dump_stacks, start


def test_snapshot_includes_live_worker_frame():
    ready = threading.Event()
    release = threading.Event()

    def held_worker():
        ready.set()
        release.wait()

    worker = threading.Thread(target=held_worker)
    worker.start()
    try:
        assert ready.wait(5)
        output = io.StringIO()
        dump_stacks(output)
        assert 'held_worker' in output.getvalue()
        assert f'Thread 0x{worker.ident:x}' in output.getvalue()
    finally:
        release.set()
        worker.join(5)


def test_periodic_sampling_survives_retiring_frames():
    output = io.StringIO()
    stop, sampler = start(0.001, stream=output)
    try:
        for _ in range(100):
            workers = [threading.Thread(target=lambda: sum(range(100))) for _ in range(8)]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join()
        deadline = time.monotonic() + 5
        while not output.getvalue() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert sampler.is_alive()
        assert 'Periodic Python stack snapshot:' in output.getvalue()
    finally:
        stop.set()
        sampler.join(5)
    assert not sampler.is_alive()


def test_interval_must_be_positive():
    with pytest.raises(ValueError, match='positive'):
        start(0)
