"""A tile buffer starts from interval zero while later forcing is unposted."""
from collections.abc import Sequence
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from woof.core import streaming
from woof.ingest import lateral_bc


class PostedIntervals(Sequence):
    def __init__(self, intervals):
        self.intervals = tuple(intervals)
        self.bounds = tuple((row.start_seconds, row.end_seconds) for row in intervals)
        self.ready = 1
        self.read = []
        self.checked = []
        self.validate = None

    def __len__(self):
        return len(self.intervals)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(self[k] for k in range(*index.indices(len(self))))
        if not 0 <= index < len(self):
            raise IndexError(index)
        assert index < self.ready, f'buffer startup read unposted interval {index}'
        self.read.append(index)
        interval = self.intervals[index]
        if index > 0 and self.validate is not None:
            self.validate(interval)
            self.checked.append(index)
        return interval


def _boundaries():
    values = np.arange(3 * 10 * 12, dtype=np.float64).reshape(3, 10, 12)
    snapshots = [{name: values + offset for name in ('u', 'v', 'theta', 'phi', 'mu')}
                 for offset in (0.125, 0.375, 0.875)]
    return lateral_bc.build_lateral_boundaries(snapshots, [0., 3600., 7200.])


def _factory(monkeypatch, boundaries):
    from woof.io import restart
    from tilestream import harness

    def make_state(*args, **kwargs):
        pool = {}
        def scratch(shape, slot):
            if slot not in pool:
                pool[slot] = np.zeros(shape, dtype=np.float32)
            return pool[slot]
        return SimpleNamespace(_scratch=pool, scratch=scratch, existing_scratch=pool.get,
                               _host_setup_state=True), None

    monkeypatch.setattr(streaming, 'domain_vertical_coord', lambda *args: object())
    monkeypatch.setattr(streaming, '_impose_domain_setup', lambda *args: 0)
    monkeypatch.setattr(streaming, 'prime_lazy_carriers', lambda *args: ())
    monkeypatch.setattr(restart, 'lifecycle_window_slots', lambda state: ())
    monkeypatch.setattr(harness, 'neutral_geography', lambda cfg: SimpleNamespace())
    monkeypatch.setattr(harness, 'make_physics_state', make_state)
    # Actual make() and actual eager/lazy boundary attachment run on host
    # arrays. Only the physics buffer construction is replaced here.
    return streaming.prepared_tile_state_factory(
        SimpleNamespace(physics=None, existing_scratch=lambda *_: None), object(), tables0=boundaries)


def _assert_interval_bytes(actual, expected):
    assert set(actual.fields) == set(expected.fields)
    for name, fields in expected.fields.items():
        for side in ('west', 'east', 'south', 'north'):
            target, source = getattr(actual.fields[name], side), getattr(fields, side)
            for key in ('value', 'tendency'):
                np.testing.assert_array_equal(getattr(target, key),
                                              getattr(source, key).astype(np.float32))


def test_first_buffer_uses_only_interval_zero_and_loads_next_at_the_seam(monkeypatch):
    eager = _boundaries()
    posted = PostedIntervals(eager.intervals)
    boundaries = replace(eager, intervals=posted)
    tile = _factory(monkeypatch, boundaries)(object())
    mirror = tile._lateral_boundary_device
    assert mirror.streaming_external
    assert len(mirror.intervals) == 1
    assert set(posted.read) == {0}
    assert posted.checked == []
    _assert_interval_bytes(mirror.intervals[0], eager.intervals[0])

    # The same buffer reloads its one packed allocation at the model seam.
    packed = mirror.packed_forcing
    posted.ready = 2
    interval = tile.lateral_boundaries.interval_at(3600.)
    loaded = lateral_bc._resident_interval(tile, interval)
    assert posted.checked == [1]
    assert mirror.packed_forcing is packed
    assert mirror.external_reload_count == 2
    _assert_interval_bytes(loaded, eager.intervals[1])


def test_sealed_buffer_reloads_one_interval_without_changing_boundary_words(monkeypatch):
    boundaries = _boundaries()
    tile = _factory(monkeypatch, boundaries)(object())
    mirror = tile._lateral_boundary_device
    assert mirror.streaming_external
    assert len(mirror.intervals) == 1
    assert len(boundaries.intervals) == 2
    packed = mirror.packed_forcing
    for index, interval in enumerate(boundaries.intervals):
        _assert_interval_bytes(lateral_bc._resident_interval(tile, interval), interval)
        assert mirror.packed_forcing is packed
        assert len(mirror.intervals) == 1
    assert mirror.external_reload_count == 2


def test_lazy_factory_retains_next_interval_geometry_validation(monkeypatch):
    eager = _boundaries()
    posted = PostedIntervals(eager.intervals)
    boundaries = replace(eager, intervals=posted)
    tile = _factory(monkeypatch, boundaries)(object())
    posted.ready = 2
    next_interval = posted.intervals[1]
    # A different inventory is refused when the interval is reached.
    posted.intervals = (posted.intervals[0], replace(
        next_interval, fields={key: value for key, value in next_interval.fields.items()
                               if key != 'mu'}))
    with pytest.raises(ValueError, match='missing.*mu'):
        tile.lateral_boundaries.interval_at(3600.)


def test_prepared_buffer_skips_the_land_surface_cold_start(monkeypatch):
    """A buffer whose carriers are gathered from the store asks
    ``initialize_physics`` for no LSM cold start.

    On RUC that cold start is a host column loop over the whole rank; the
    2-card HRRR profile (development branch) spent 84-92 s of idle cards
    in it, rank after rank, before step 1, every launch.
    """
    from woof.io import restart
    from tilestream import harness

    seen = {}

    def make_state(cfg, seed, **kwargs):
        seen.update(kwargs)
        pool = {}
        return SimpleNamespace(_scratch=pool, existing_scratch=pool.get, _host_setup_state=True,
                               scratch=lambda shape, slot: pool.setdefault(
                                   slot, np.zeros(shape, dtype=np.float32))), None

    monkeypatch.setattr(streaming, 'domain_vertical_coord', lambda *args: object())
    monkeypatch.setattr(streaming, '_impose_domain_setup', lambda *args: 0)
    monkeypatch.setattr(streaming, 'prime_lazy_carriers', lambda *args: ())
    monkeypatch.setattr(restart, 'lifecycle_window_slots', lambda state: ())
    monkeypatch.setattr(harness, 'neutral_geography', lambda cfg: SimpleNamespace())
    monkeypatch.setattr(harness, 'make_physics_state', make_state)
    factory = streaming.prepared_tile_state_factory(SimpleNamespace(physics=None, existing_scratch=lambda *_: None),
                                                    object(), tables0=None)
    factory(object())
    assert seen.get('lsm_cold_start') is False
    assert seen.get('poison') == 'constant'
