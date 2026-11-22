"""The forward-speed words (woof.hex.speed_words): on by default, each turned off by 0, and each one's host-side law.

The card-side halves (the pinned table uploads, the per-peer receive streams, the fast publication) are checked on a
card when one is visible, word for word against the route they replace; the whole-run proof (two ranks, a forecast
and a cycle hour, every output word) was run on cards.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

from woof.hex import speed_words as W


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for name in W.WORDS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("WOOF_HEX_LIMITER_SPECIES_PER_PASS", raising=False)


def test_every_word_is_on_by_default_and_each_turns_off():
    assert W.limiter_batch() == W.LIMITER_BATCH_DEFAULT == 7
    for name in W.WORDS:
        assert W.on(name)
    for off in ("0", "off", "false", "no", "OFF"):
        assert not W.on(W.PUBLISH_FAST, {W.PUBLISH_FAST: off})
    for on in ("1", "", "on"):
        assert W.on(W.PUBLISH_FAST, {W.PUBLISH_FAST: on})
    assert W.limiter_batch({W.LIMITER_BATCH: "0"}) == 1


def test_limiter_batch_reads_a_whole_number():
    assert W.limiter_batch({W.LIMITER_BATCH: "7"}) == 7
    assert W.limiter_batch({W.LIMITER_BATCH: "1"}) == 1
    assert W.limiter_batch({W.LIMITER_BATCH: "off"}) == 1
    with pytest.raises(ValueError):
        W.limiter_batch({W.LIMITER_BATCH: "-2"})
    with pytest.raises(ValueError):
        W.limiter_batch({W.LIMITER_BATCH: "seven"})


def test_limiter_batch_widens_only_the_multi_rank_schedule(monkeypatch):
    from woof.hex.cuda_transport_v841 import limiter_species_per_pass
    from woof.hex.cuda_backend.regional_admission import MINTED_KERNEL_SET_SHA256, kernel_set_sha256
    # Off: one species a pass on the schedule (one round E a species), as before.
    monkeypatch.setenv(W.LIMITER_BATCH, "0")
    assert W.schedule_species_per_pass(13) == limiter_species_per_pass(13) == 1
    monkeypatch.delenv(W.LIMITER_BATCH)   # the default: 7
    # The schedule: ceil(13/7) = 2 round-E parts.
    width = W.schedule_species_per_pass(13)
    assert width == 7 and -(-13 // width) == 2
    assert W.schedule_species_per_pass(5) == 5          # never wider than the species
    assert W.schedule_species_per_pass(0) == 1
    # The transport's own law is untouched: a rank's exchanger declares the parts the schedule made.
    assert limiter_species_per_pass(13) == 1
    assert limiter_species_per_pass(13, SimpleNamespace(fct_scale_parts=2)) == 7
    assert limiter_species_per_pass(13, SimpleNamespace(fct_scale_parts=13)) == 1
    # The existing explicit override still wins.
    monkeypatch.setenv("WOOF_HEX_LIMITER_SPECIES_PER_PASS", "3")
    assert W.schedule_species_per_pass(13) == 3
    # No regional translation unit moved: the minted limited-area classes still hold.
    assert kernel_set_sha256() == MINTED_KERNEL_SET_SHA256


class _Route:
    def __init__(self, *, export=None, due=False, control=None, frame=0.0):
        self.export_path, self._due, self.control, self.frame_seconds = export, due, control, frame
        self.offset_seconds = 0.0

    def checkpoint_due(self, step, dt):
        return self._due


def test_publish_lean_keeps_every_reader_of_the_last_step(monkeypatch):
    from woof.hex.mpasphys.runtime_v842 import moving_publication_due as due
    plain = _Route()
    # Off: the last step always publishes.
    monkeypatch.setenv(W.PUBLISH_LEAN, "0")
    assert due(plain, step=10, steps=10, dt=20.0)
    monkeypatch.delenv(W.PUBLISH_LEAN)   # the default: on
    assert not due(plain, step=10, steps=10, dt=20.0)
    assert due(_Route(export="/x"), step=10, steps=10, dt=20.0)
    assert due(_Route(due=True), step=10, steps=10, dt=20.0)
    assert due(_Route(control=object(), frame=900.0), step=10, steps=10, dt=20.0)
    # The first step and the middle of the run are unchanged by the word.
    assert due(plain, step=0, steps=10, dt=20.0)
    assert not due(plain, step=5, steps=10, dt=20.0)
    assert due(_Route(due=True), step=5, steps=10, dt=20.0)
    # A distributed route's export is on its route.
    assert due(SimpleNamespace(route=_Route(export="/x"), checkpoint_due=lambda s, d: False, control=None,
                               frame_seconds=0.0), step=10, steps=10, dt=20.0)


def test_restart_release_lets_go_of_the_global_copies_only():
    from woof.hex.mpasphys.samehost_v842 import _release_restored
    words = np.zeros((4, 1 << 18), dtype=np.float32)
    state = dict(route="worker", control=None, restored={("state", "theta", 1): words}, accumulators={"a": words},
                 window={"w": words}, ocean_pwp=None, restore_receipt={"ok": 1}, layout="L", mesh="M",
                 wave_state="held")
    moving = SimpleNamespace(carrier={"physics": {}})
    mib = _release_restored(moving, state)
    assert mib == pytest.approx(3 * words.nbytes / 1048576.0, rel=1e-6)
    assert state["restored"] == {} and state["accumulators"] is None and state["window"] is None
    assert moving.carrier is None
    # What the ranks still read stays.
    assert (state["route"], state["restore_receipt"], state["layout"], state["mesh"], state["wave_state"]) == \
        ("worker", {"ok": 1}, "L", "M", "held")


# ---------------------------------------------------------------------------------------------- on a card
def _cupy():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no card")
    except Exception:
        pytest.skip("no card")
    return cp


def test_pinned_table_upload_is_the_same_table():
    cp = _cupy()
    host = np.arange(45, dtype=np.int64).reshape(5, 9) * 1_000_003
    device, pinned = W.pinned_upload_int64(cp, host)
    cp.cuda.get_current_stream().synchronize()
    assert pinned is not None and np.array_equal(device.get(), host)


def test_table_cache_word_gives_the_same_tables(monkeypatch):
    cp = _cupy()
    from woof.hex.cuda_solve_region_v841 import SolveRegionKernels, _DESCRIPTOR_WORDS
    rows = [tuple(range(i, i + _DESCRIPTOR_WORDS)) for i in range(3)]
    monkeypatch.setenv(W.HALO_TABLE_CACHE, "0")
    plain = SolveRegionKernels(kernel_cache=None).table(rows).get()
    monkeypatch.setenv(W.HALO_TABLE_CACHE, "1")
    kernels = SolveRegionKernels(kernel_cache=None)
    first = kernels.table(rows)
    again = kernels.table(rows)
    cp.cuda.get_current_stream().synchronize()
    assert again is first and kernels.table_uploads == 1
    assert np.array_equal(first.get(), plain)


class _Native:
    def __init__(self):
        self.put = []
        self._fields = {}

    def put_field(self, pool, name, host, *, level, nlevels):
        self.put.append((pool, name, level, np.array(host, copy=True)))


def _pools(cp, rng, word):
    from woof.hex.mpasphys.device_pools_v842 import NativeDevicePoolViews
    views = NativeDevicePoolViews.__new__(NativeDevicePoolViews)
    views.cp, views.native = cp, _Native()
    views.transfers = {"publication_download_bytes": 0}
    views.resident_buffers = {}
    shapes = [((3, 7), np.float32), ((59, 1000), np.float32), ((2, 59, 300_000), np.float32), ((1000,), np.int32),
              ((40, 50), np.float64), ((0, 5), np.float32), ((5, 4, 3), np.int8)]
    for index, (shape, dtype) in enumerate(shapes):
        words = (rng.integers(0, 2**31, size=shape) if np.dtype(dtype).kind in "iu"
                 else rng.standard_normal(shape)).astype(dtype)
        host = np.zeros((*shape[:-1], shape[-1] + 2) if shape else shape, dtype=dtype, order="F")
        selection = (..., slice(0, shape[-1])) if shape else ()
        views.native._fields[("state", f"f{index}")] = (None, None, 1)
        views.resident_buffers[("state", f"f{index}", 1)] = (host, cp.asarray(words), selection)
    views.resident_buffers[("prepared", "skip", 1)] = (np.zeros(3, np.float32), cp.zeros(3, cp.float32), ())
    return views


def test_fast_publish_puts_the_same_words_in_the_same_order(monkeypatch):
    cp = _cupy()
    monkeypatch.setenv(W.PUBLISH_FAST, "0")
    plain = _pools(cp, np.random.default_rng(7), False)
    plain.publish()
    monkeypatch.setenv(W.PUBLISH_FAST, "1")
    fast = _pools(cp, np.random.default_rng(7), True)
    fast.FAST_PUBLISH_BYTES = 1 << 20     # force both the shared block and the slab stream
    fast.publish()
    assert [p[:3] for p in fast.native.put] == [p[:3] for p in plain.native.put]
    for (_, _, _, a), (_, _, _, b) in zip(fast.native.put, plain.native.put):
        assert a.dtype == b.dtype and a.shape == b.shape
        assert np.array_equal(np.ascontiguousarray(a).view(np.uint8), np.ascontiguousarray(b).view(np.uint8))
    assert fast.transfers == plain.transfers


def test_samehost_transport_word_only_fills_an_auto_door(monkeypatch):
    from woof.hex.mpasphys.samehost_v842 import requested_transport
    monkeypatch.delenv("WOOF_HEX_SAMEHOST_TRANSPORT", raising=False)
    assert requested_transport(SimpleNamespace(samehost_transport=None)) == "auto"
    monkeypatch.setenv("WOOF_HEX_SAMEHOST_TRANSPORT", "nccl")
    assert requested_transport(SimpleNamespace(samehost_transport=None)) == "nccl"
    assert requested_transport(SimpleNamespace(samehost_transport="auto")) == "nccl"
    assert requested_transport(SimpleNamespace(samehost_transport="staged")) == "staged"   # the door flag wins
    monkeypatch.setenv("WOOF_HEX_SAMEHOST_TRANSPORT", "carrier-pigeon")
    with pytest.raises(ValueError):
        requested_transport(SimpleNamespace(samehost_transport=None))
