"""The MPAS route's frame download leaves the forecast thread (PhysicsStreams._capture).

CPU stand-ins for the device array, events and the copy stream: the copy stream's downloads run only when the
writer waits on its event, so a forecast that mutates its pools right after ``write`` must not change the frame.
Synthetic words exercise ordering only; they are not forecast identity proofs.
"""
from datetime import timedelta
import ctypes
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from woof.hex.mpasphys.streams_output_v842 import PhysicsStreams

from test_mpasphys_streams_output_v842 import session, stream


@pytest.fixture(autouse=True)
def legacy_frame_policy(monkeypatch):
    # The overlap tests exercise the explicitly selected staged-frame path.
    monkeypatch.setenv('WOOF_HEX_MEMBER_MEMORY', 'single')


class _Device(np.ndarray):
    """A device array stand-in: ``data.ptr`` like CuPy's, pinned-style get."""

    @property
    def data(self):
        return SimpleNamespace(ptr=self.ctypes.data)

    def get(self, order='C', *, out=None, blocking=True):
        if out is not None:
            np.copyto(out, np.asarray(self))
            return out
        return np.array(self, copy=True, order=order, subok=False)


class _FakeCuda:
    """Events and a copy stream whose queued downloads run only when the writer waits."""

    def __init__(self):
        self.queued = []
        self.log = []
        self.release = threading.Event()   # the copy stream "finishes" only when the test says so
        fake = self

        class Event:
            def __init__(self, disable_timing=True):
                pass

            def record(self, stream=None):
                return None

            def synchronize(self):
                fake.log.append(('wait', threading.current_thread().name))
                assert fake.release.wait(30), 'the test never released the copy stream'
                for dst, src, size, _ in fake.queued:
                    ctypes.memmove(dst, src, size)
                fake.queued.clear()

        class Stream:
            def __init__(self, non_blocking=False):
                self.ptr = 1
                fake.streams.append(self)

            def wait_event(self, event):
                fake.log.append(('stream waits step event', None))

        self.streams = []
        self.current = SimpleNamespace(ptr=0)
        self.Event = Event
        self.Stream = Stream
        self.get_current_stream = lambda: self.current
        self.alloc_pinned_memory = lambda size: bytearray(size + 128)
        self.memory = SimpleNamespace(OutOfMemoryError=MemoryError)
        self.free = 64 << 30   # device bytes free (memGetInfo)

        def memcpy(dst, src, size, kind, stream):
            fake.queued.append((dst, src, size, stream))

        self.runtime = SimpleNamespace(memcpyAsync=memcpy, memcpyDeviceToHost=2,
                                       memGetInfo=lambda: (fake.free, 80 << 30))


def _fake_cp():
    cuda = _FakeCuda()

    def empty(shape, dtype):
        return np.empty(shape, dtype=dtype).view(_Device)

    return SimpleNamespace(cuda=cuda, ndarray=_Device, empty=empty, uint8=np.uint8, uint32=np.uint32,
                           ascontiguousarray=lambda a: np.ascontiguousarray(a).view(_Device),
                           get_default_memory_pool=lambda: SimpleNamespace(free_bytes=lambda: 0)), cuda


def test_frame_is_the_words_at_write_even_when_the_pools_change_before_the_download(tmp_path):
    state, fields, _, start = session(resident=True)
    cp, cuda = _fake_cp()
    state.device_pools.cp = cp
    state.device_pools.field = lambda pool, name: fields[pool, name].view(_Device)
    before = fields['state', 'scalars'][..., :3].copy()
    written = []
    receipt = {}
    with PhysicsStreams(state, (stream(),), tmp_path, final_time=start + timedelta(minutes=2)) as writer:
        def encode(jobs, snapshot, when, step, extras=None):
            written.append({name: np.array(item.values, copy=True) for name, item in snapshot.items()
                            if hasattr(item, 'values')})
            return []

        writer._encode = encode
        writer.write(step=0, dt=60, receipt=receipt)
        # The forecast mutates its pools at once; the download has not run yet.
        assert cuda.queued, 'the frame download must be left queued'
        assert {entry[3] for entry in cuda.queued} == {1}, 'staged downloads go on the copy stream'
        fields['state', 'scalars'][...] = 7
        cuda.release.set()
        writer._drain()
        assert writer.capture_receipt['staged_frames'] == 1
        assert writer.capture_receipt['staging_bytes_max'] >= before.nbytes
    np.testing.assert_array_equal(written[0]['qv'].view(np.uint32), before[0].view(np.uint32))
    np.testing.assert_array_equal(written[0]['qc'].view(np.uint32), before[1].view(np.uint32))
    # The NaN payload word (0x7fc12345) is carried untouched.
    assert 0x7fc12345 in written[0]['qv'].view(np.uint32)
    waits = [entry for entry in cuda.log if entry[0] == 'wait']
    assert waits and all(name.startswith('mpas-output') for _, name in waits), \
        'only the writer thread waits for the download'
    assert receipt['output_capture'][0]['download'].startswith('copy stream')


def test_held_first_frame_and_snapshot_overrides_keep_the_synchronous_capture(tmp_path):
    state, fields, _, start = session(resident=True)
    cp, cuda = _fake_cp()
    cuda.release.set()
    state.device_pools.cp = cp
    state.device_pools.field = lambda pool, name: fields[pool, name].view(_Device)

    class Override(PhysicsStreams):
        def _snapshot(self, selected):
            return super()._snapshot(selected)

    receipt = {}
    with Override(state, (stream(),), tmp_path, final_time=start + timedelta(minutes=2)) as writer:
        writer._encode = lambda *args, **kwargs: []
        writer.write(step=0, dt=60, receipt=receipt)
        # The subclass's capture completed on the forecast thread before write returned.
        assert not cuda.queued
        assert [name for kind, name in cuda.log if kind == 'wait'] == [threading.current_thread().name]
    assert receipt['output_capture'][0]['download'] == 'forecast thread'


@pytest.mark.parametrize('policy,free', [('single', 1 << 30), ('compact', 64 << 30)])
def test_bounded_frame_download_uses_the_step_stream(tmp_path, monkeypatch, policy, free):
    monkeypatch.setenv('WOOF_HEX_MEMBER_MEMORY', policy)
    state, fields, _, start = session(resident=True)
    cp, cuda = _fake_cp()
    cuda.free = free
    state.device_pools.cp = cp
    state.device_pools.field = lambda pool, name: fields[pool, name].view(_Device)
    before = fields['state', 'scalars'][..., :3].copy()
    written = []
    with PhysicsStreams(state, (stream(),), tmp_path, final_time=start + timedelta(minutes=2)) as writer:
        writer._encode = lambda jobs, snapshot, when, step, extras=None: written.append(
            np.array(snapshot['qv'].values, copy=True)) or []
        writer.write(step=0, dt=60, receipt={})
        # Queued on the step stream (ptr 0): later kernels wait for it on the device, the host does not.
        assert cuda.queued and {entry[3] for entry in cuda.queued} == {0}
        cuda.release.set()
        writer._drain()
        assert writer.capture_receipt['in_stream_frames'] == 1
        assert writer.capture_receipt['staged_frames'] == 0
    np.testing.assert_array_equal(written[0].view(np.uint32), before[0].view(np.uint32))


def test_a_second_frame_written_while_the_first_encodes_takes_its_own_host_storage(tmp_path):
    state, fields, _, start = session(resident=True)
    cp, cuda = _fake_cp()
    cuda.release.set()
    state.device_pools.cp = cp
    state.device_pools.field = lambda pool, name: fields[pool, name].view(_Device)
    first = fields['state', 'scalars'][..., :3].copy()
    gate, written = threading.Event(), []

    def encode(jobs, snapshot, when, step, extras=None):
        if step == 0:
            assert gate.wait(30)
        written.append((step, np.array(snapshot['qv'].values, copy=True)))
        return []

    receipt = {}
    with PhysicsStreams(state, (stream(),), tmp_path, final_time=start + timedelta(minutes=2)) as writer:
        writer._encode = encode
        writer.write(step=0, dt=60, receipt=receipt)
        fields['state', 'scalars'][...] = 7
        writer.write(step=1, dt=60, receipt=receipt)   # frame 0 still encoding: no wait, second storage set
        assert receipt['output_capture'][1]['host_storage_sets'] == 2
        gate.set()
    assert [step for step, _ in written] == [0, 1]
    np.testing.assert_array_equal(written[0][1].view(np.uint32), first[0].view(np.uint32))
    np.testing.assert_array_equal(written[1][1], 7)
