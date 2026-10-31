"""The MPAS-physics route's active-LTS timeline (woof.hex.mpasphys.lts_v842), on the CPU.

THE BREAKAGE THESE PREVENT (HEX-SHAPES-1CARD 2026-10-29): every LTS class thread wrote the run's one
``native-runtime-timings.json``, so the file held whichever class finished last.  On the 96 -> 3 km zoom globe that
was the top class's 30 outer steps out of 450 class steps, 65-90 s of a 101 s loop sat in no labelled stage, and a
receipt tool that took a step median times 3600 / dt reported 36 s per forecast hour (99x real time).  The run's
timeline now carries every class step, the coordinator's rows and the loop's own measured rate; the gate's step
figure reads that rate on an LTS run.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from woof.hex.mpasphys import lts_v842

TREE = Path(__file__).resolve().parents[1]


def _class_file(out: Path, index: int, steps: int) -> None:
    rows = [dict(stage='native-runtime', wall_seconds=10.0), dict(stage='dycore-build', wall_seconds=1.0)]
    for _ in range(steps):
        rows += [dict(stage='outer-physics', wall_seconds=0.01), dict(stage='dycore-and-post-transport',
                                                                         wall_seconds=0.2)]
    (out / f'native-runtime-timings.class{index}.json').write_text(json.dumps({'stages': rows}))


def test_every_class_step_is_in_the_run_timeline(tmp_path):
    units = [SimpleNamespace(rate=1), SimpleNamespace(rate=2), SimpleNamespace(rate=4)]
    _class_file(tmp_path, 0, 8)
    _class_file(tmp_path, 1, 4)        # class 2 left no file (died before its first row)
    clock = lts_v842._Clock()
    coordinator = [clock.row('native-runtime', clock.perf, clock.perf + 30.0, {'device': 'GPU'})]
    stages = lts_v842.write_lts_timeline(tmp_path, coordinator, units, 15.0)
    on_disk = json.loads((tmp_path / 'native-runtime-timings.json').read_text())['stages']
    assert on_disk == json.loads(json.dumps(stages))
    assert [s['stage'] for s in on_disk].count('native-runtime') == 1          # the run's, the coordinator's
    # perf + 30.0 - perf is 30.0 only to the clock's precision (29.999999999999886 on a CI runner, 2026-10-30).
    assert on_disk[0]['wall_seconds'] == pytest.approx(30.0)
    steps = [s for s in on_disk if s['stage'] == 'dycore-and-post-transport']
    assert [s['lts_class'] for s in steps] == [0] * 8 + [1] * 4
    assert {s['lts_rate'] for s in steps if s['lts_class'] == 1} == {2}
    assert {s['class_step_model_seconds'] for s in steps if s['lts_class'] == 1} == {30.0}
    assert [s for s in on_disk if s['stage'] == 'lts-class-runtime' and s.get('status') == 'missing'] == [
        dict(stage='lts-class-runtime', status='missing', lts_class=2, lts_rate=4, class_step_model_seconds=60.0)]
    assert not list(tmp_path.glob('native-runtime-timings.class*.json'))


def _macros(clock, n, *, seconds, wait):
    rows, mark = [], clock.perf
    for index in range(n):
        rows.append(dict(macro=index + 1, base_steps=8, seconds=seconds, _began=mark, boundary_wait_seconds=wait,
                         synchronize_seconds=0.001, coordinator_seconds=0.3,
                         class_steps=[dict(rate=1, steps=8, seconds=1.6), dict(rate=8, steps=1, seconds=0.15)]))
        mark += seconds + wait
    return rows, mark


def test_the_loop_rate_is_measured_not_a_step_median_times_3600_over_dt(tmp_path):
    clock = lts_v842._Clock()
    units = [SimpleNamespace(rate=1), SimpleNamespace(rate=8)]
    macros, ended = _macros(clock, 30, seconds=2.05, wait=1.0)        # 30 macros of 8 x 15 s = 1 h
    row = lts_v842.stepping_row(clock, macros, began=clock.perf, ended=ended, top=8, base_dt=15.0, units=units,
                                multi_card=False)
    assert row['model_seconds'] == 3600.0
    assert row['seconds_per_forecast_hour'] == pytest.approx(30 * 3.05, rel=1e-6)   # boundary waits included
    assert row['x_real_time'] == pytest.approx(3600.0 / (30 * 3.05), rel=1e-6)
    assert row['warm_seconds_per_forecast_hour'] == pytest.approx(3.05 * 3600.0 / 120.0, rel=1e-6)
    # The old receipt arithmetic on the top class's 150 ms step: 0.15 x 3600 / 15 = 36 s per hour.  Not this.
    assert row['seconds_per_forecast_hour'] > 2 * (0.15 * 3600.0 / 15.0)
    assert row['class_steps'] == [
        dict(**{'class': 0}, rate=1, steps=240, seconds=pytest.approx(48.0), seconds_per_step=pytest.approx(0.2)),
        dict(**{'class': 1}, rate=8, steps=30, seconds=pytest.approx(4.5), seconds_per_step=pytest.approx(0.15))]
    assert row['coordinator_seconds_total'] == pytest.approx(9.0, rel=1e-6)
    assert row['boundary_wait_seconds_total'] == pytest.approx(30.0, rel=1e-6)


def test_device_spans_attribute_each_class_its_own_device_time(tmp_path):
    """WOOF_HEX_LTS_DEVICE_SPANS=1 rows: a class's device seconds, not the host wall its launches waited.

    THE BREAKAGE THIS PREVENTS (HEX-SHAPES-1CARD HB18 remeasure, 2026-10-29): in stream order the 955-cell rate-2
    ring's host wall (176 ms per step) carried the queued device work of the 64,832-cell rate-1 steps before it, and
    the receipt and its picture called that the ring's cost (18 s of the 66 s hour).
    """
    clock = lts_v842._Clock()
    units = [SimpleNamespace(rate=1), SimpleNamespace(rate=8)]
    macros, ended = _macros(clock, 30, seconds=2.05, wait=1.0)
    for row in macros:
        row['class_steps'][0]['device_seconds'] = 1.2
        row['class_steps'][1]['device_seconds'] = 0.6
        row.update(device_seconds=2.0, coordinator_device_seconds=0.2)
    row = lts_v842.stepping_row(clock, macros, began=clock.perf, ended=ended, top=8, base_dt=15.0, units=units,
                                multi_card=False)
    assert [(c['device_seconds'], c['device_seconds_per_step']) for c in row['class_steps']] == [
        (pytest.approx(36.0), pytest.approx(0.15)), (pytest.approx(18.0), pytest.approx(0.6))]
    assert row['seconds_per_forecast_hour'] == pytest.approx(30 * 3.05, rel=1e-6)   # the measured loop, unchanged
    assert row['device_seconds_total'] == pytest.approx(60.0)
    assert row['coordinator_device_seconds_total'] == pytest.approx(6.0)
    assert 'CUDA events' in row['device_basis']
    # A run without the spans carries no device fields (the default receipt is unchanged).
    plain, _ = _macros(clock, 2, seconds=2.05, wait=1.0)
    assert 'device_seconds_total' not in lts_v842.stepping_row(
        clock, plain, began=clock.perf, ended=ended, top=8, base_dt=15.0, units=units, multi_card=False)


def _gate():
    spec = importlib.util.spec_from_file_location("both_routes_gate", TREE / "tools" / "both_routes_gate.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["both_routes_gate"] = module
    spec.loader.exec_module(module)
    return module


def test_the_gate_step_figure_reads_the_lts_loop_rate(tmp_path):
    clock = lts_v842._Clock()
    units = [SimpleNamespace(rate=1), SimpleNamespace(rate=8)]
    macros, ended = _macros(clock, 30, seconds=2.05, wait=1.0)
    stepping = lts_v842.stepping_row(clock, macros, began=clock.perf, ended=ended, top=8, base_dt=15.0, units=units,
                                     multi_card=False)
    for index in range(2):
        _class_file(tmp_path, index, 3)
    lts_v842.write_lts_timeline(tmp_path, [stepping], units, 15.0)
    # 3.05 s per 120 s macro = 0.381 s per 15 s base step, never the class rows' 0.21 s median step
    assert _gate().step_seconds(tmp_path) == round(3.05 * 3600.0 / 120.0 * 15.0 / 3600.0, 3)
