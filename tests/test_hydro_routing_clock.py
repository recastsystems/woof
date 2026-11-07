"""Clock orchestration checks; resident numerical proof is in the CUDA test."""
import copy
import json
import math
from fractions import Fraction
from types import SimpleNamespace

import pytest

from woof.hydro.driver import ResidentHydroDriver, Stage
from woof.hydro import door, provider


def intervals(dt, duration=1001.0):
    result, elapsed = [], 0.0
    while elapsed < duration:
        step = dt[len(result) % len(dt)] if isinstance(dt, list) else dt
        step = min(step, duration - elapsed)
        result.append(step)
        elapsed += step
    return result


def clock_driver(dt, accumulated, duration, components):
    return ResidentHydroDriver(device_id=0, interval_seconds=dt, identity={},
        after=[Stage(name, component, component.advance)
               for name, component in components.items()],
        routing_clock={'period_seconds': 300., 'duration_seconds': duration,
                       'accumulated_parent': accumulated})


@pytest.mark.parametrize('dt', [11.25, 128.571, [15., 18., 24., 401.]])
@pytest.mark.parametrize('accumulated', [True, False])
def test_clock_carries_time_and_water_and_publishes_tail(dt, accumulated):
    water = SimpleNamespace(depth=0., produced=0., bucket=0., received=[], samples=[])
    class Component:
        arrays, restart_identity = {}, {}
        def __init__(self, callback):
            self.advance = callback
    def exchange(span, **kw):
        water.samples.append(span)
        if accumulated:
            water.depth = water.produced - getattr(water, 'previous', 0.)
            water.previous = water.produced
    def runoff(span, fraction, reset, **kw):
        water.bucket = (0. if reset else water.bucket) + water.depth * fraction
    def routing(span, **kw):
        assert span == 300.
        water.received.append(water.bucket)
    frames = []
    components = {name: Component(callback) for name, callback in
                  [('exchange', exchange), ('runoff', runoff), ('routing', routing),
                   ('history', lambda span, **kw: frames.append(kw))]}
    steps = intervals(dt)
    driver = clock_driver(steps[0], accumulated, sum(steps), components)
    driver.allow_variable_interval()
    for i, step in enumerate(steps):
        water.depth = step * (i + 1) * .003
        water.produced += water.depth
        driver.before_lsm()
        driver.advance(step)
    assert sum(water.received) == pytest.approx(water.produced, rel=2e-15)
    assert [f['time_seconds'] for f in frames] == pytest.approx([300., 600., 900., 1001.], abs=1e-10)
    assert frames[-1]['final']
    assert len(water.samples) <= 4 if accumulated else len(water.samples) == len(steps)
    saved = copy.deepcopy(driver.checkpoint_metadata())
    fresh = clock_driver(steps[0], accumulated, sum(steps), components)
    fresh.allow_variable_interval()
    fresh.validate_checkpoint(saved, {})
    fresh.restore_metadata(saved)
    assert fresh.checkpoint_metadata() == saved


class CountedStage:
    arrays, restart_identity = {}, {}
    def __init__(self, callback):
        self.advance = callback


def counted_clock(steps, duration):
    """A routing clock with counting stages over parent steps of exact 1/300 s ticks, summed as the driver sums them."""
    frames, routing = [], []
    components = {name: CountedStage(callback) for name, callback in
                  [('exchange', lambda span, **kw: None), ('runoff', lambda span, **kw: None),
                   ('routing', lambda span, **kw: routing.append(span)),
                   ('history', lambda span, **kw: frames.append(kw))]}
    driver = clock_driver(13.5, True, duration, components)
    driver.allow_variable_interval()
    elapsed = 0.
    for ticks in steps:
        dt = float(Fraction(ticks, 300))
        elapsed += dt
        driver.before_lsm()
        driver.advance(dt)
    return driver, frames, routing, elapsed


# The two ways 48 h of adaptive steps miss 172,800 s in floating point, from real forecasts' repeating 180 s cycles.
@pytest.mark.parametrize('cycle,past_the_end', [([4044] * 13 + [1428], True), ([3708] * 14 + [2088], False)])
def test_clock_roundoff_opens_no_routing_step_past_the_forecast_end(cycle, past_the_end):
    # Breakage: every 48 h run that summed past the end took a 577th routing step for 1.8e-8 s, and its final capture
    # fell outside the 48-frame history: "history buffer exhausted", no hydro output at all.
    duration = 172800.
    driver, frames, routing, elapsed = counted_clock(cycle * 960, duration)
    assert (elapsed > duration) is past_the_end and abs(elapsed - duration) > 16 * math.ulp(duration)
    assert len(routing) == len(frames) == 576 == driver._clock_routed
    assert [f['final'] for f in frames] == [False] * 575 + [True]
    assert frames[-1]['time_seconds'] == duration  # the writer refuses a final time past its frame (series.rs)
    assert driver._clock_finalized
    # provider.composition's hourly history: 576 routing steps hold 48 frames, and every capture lands inside them.
    cadence, capacity = 12, (576 + 12 - 1) // 12
    assert capacity == 48 and all((tick + cadence - 1) // cadence - 1 < capacity for tick in range(1, len(frames) + 1))
    driver.finish_clock()  # publication after a finished forecast adds nothing
    assert len(frames) == 576


def test_a_stop_on_a_routing_boundary_recaptures_it_whatever_the_roundoff():
    # A forecast stopped on an hour boundary whose sum landed just past it: no extra routing step, no time past the frame.
    driver, frames, routing, elapsed = counted_clock(([4044] * 13 + [1428]) * 20, 172800.)
    assert elapsed != 3600. and abs(elapsed - 3600.) < 1e-6
    assert len(routing) == 12
    driver.finish_clock()
    assert len(routing) == 12 and driver._clock_routed == 12
    assert frames[-1]['final'] and frames[-1]['recapture'] and frames[-1]['time_seconds'] == 3600.


@pytest.mark.parametrize('key,value', [('interval_seconds', 0), ('duration_seconds', -1),
                                       ('interval_seconds', float('inf')), ('duration_seconds', None)])
def test_invalid_clock_gap_precedes_any_availability_fetch(key, value):
    row = json.loads(door.TABLE.read_text())['providers'][0]
    geometry = door.geometry_from_points([40.], [-96.], center_lon=-96., parent_lsm=2,
                                          grid_key='test', interval_seconds=11.25,
                                          duration_seconds=1001.)
    geometry[key] = value
    verdict = door.evaluate(row, geometry, probe_row=lambda row: pytest.fail('tool/source fetch'))
    assert any('coupling_interval_invalid' in why for why in verdict['reasons'])


@pytest.mark.parametrize('dt', [0., float('nan'), 37.])
def test_invalid_routing_or_hourly_spinup_clock_precedes_fetch(dt):
    row = next(r for r in json.loads(door.TABLE.read_text())['providers'] if r.get('preparation'))
    row['preparation']['routing_interval_seconds'] = dt
    geometry = door.geometry_from_points([40.], [-96.], center_lon=-96., parent_lsm=2,
        grid_key='test', interval_seconds=128.571, duration_seconds=101.)
    verdict = door.evaluate(row, geometry, probe_row=lambda row: pytest.fail('tool/source fetch'))
    assert any('coupling_' in why for why in verdict['reasons'])


@pytest.mark.parametrize('dt', [11.25, 128.571, 401.])
def test_composition_has_no_whole_parent_or_duration_requirement(tmp_path, dt):
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'parent_grid_contract': 'parent',
        'routing': {'dt_seconds': 300., 'initial_state_array': 'state'},
        'input': {'sha256': 'fabric'},
        'files': {'exchange_weights': {'sha256': 'weights'}, 'node_ids': {'shape': [2]}}}))
    result = provider.composition(manifest=manifest,
        footprint={'parent_grid_contract': 'parent', 'interval_seconds': dt,
                   'duration_seconds': 101., 'time_origin': '2024-04-26T17:00:00Z'},
        realization='parent-noah-routed')
    assert result['routing_clock']['accumulated_parent'] is True
    history = result['stages'][-1]['options']
    assert history['capacity_steps'] == 1 and history['routing_clock']
    # Exercise the default provider's actual TOML/process hand-off and table
    # expansion: a correct direct driver is insufficient if the door drops it.
    from woof.hydro.runtime import read_settings
    settings = tmp_path / 'hydro-door.toml'
    result.update(enabled=True, realization='parent-noah-routed',
                  settings_bindings={'fabric_manifest': str(manifest)})
    settings.write_text(door.to_toml(result))
    resolved = read_settings(settings)
    assert resolved['routing_clock'] == result['routing_clock']
    assert resolved['stages'] == result['stages']
