"""Observation scheduling contracts, separate from native numerical checks."""
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from tools import da_recent_observations as obs


def _fake_native_thread_budget(monkeypatch):
    """Answer rw-superob's thread_budget the way the library does, without it.

    The slot-ordering cases below fake every other native call; this was the
    one that reached the library.  On the Windows cpu leg, which builds no
    natives, both refused with SuperobBridgeMissing before any slot drained
    (E3 a1388eae7, emulated on test host 4).  The real operation is checked by
    tests/test_obs_parallel_native.py and rw-superob's own cargo test.
    """
    from woof.obs import superob_bridge
    requested = []

    def budget(threads):
        requested.append(int(threads))
        return {'ok': True, 'threads': int(threads),
                'requested_threads': int(threads), 'initialized': True}
    monkeypatch.setattr(superob_bridge, 'configure_threads', budget)
    return requested


def test_default_worker_count_follows_site_count_and_ram():
    assert obs.observation_workers(11, available_bytes=100 * 2**30)['workers'] == 11
    assert obs.observation_workers(150, available_bytes=10 * 2**30)['workers'] == 4
    assert obs.observation_workers(11, requested=3, available_bytes=100 * 2**30)['workers'] == 3
    plan = obs.observation_workers(10, 3, available_bytes=40 * 2**30,
                                   retained_slot_bytes=10 * 2**30)
    assert plan['slot_workers'] == 2 and plan['workers'] == 3 and plan['total_workers'] == 6
    assert plan['total_workers'] * plan['worker_scratch_bytes'] + plan['slot_workers'] * plan['retained_slot_bytes'] <= plan['budget_bytes']


def test_ram_too_small_names_incomplete_output_breakage():
    with pytest.raises(ValueError, match='OOM and incomplete observations'):
        obs.observation_workers(10, available_bytes=2**30, retained_slot_bytes=2**30)
    with pytest.raises(ValueError, match='suppress observations'):
        obs.observation_workers(10, requested=0)


def test_native_reference_is_not_pooled(monkeypatch):
    from woof.obs import superob_bridge
    monkeypatch.setattr(superob_bridge, 'python_reference_requested', lambda: True)
    with pytest.raises(ValueError, match='Python reference cannot be pooled'):
        obs.require_native_observations()


def test_conus_prepare_runs_surface_beside_radar(tmp_path, monkeypatch):
    from tools import conus_da_observations as conus
    rendezvous = threading.Barrier(2)
    order = []
    def stage(args):
        rendezvous.wait(timeout=5)
        order.append(threading.get_ident())
    monkeypatch.setattr(conus, 'radar', stage)
    monkeypatch.setenv('WOOF_TIME_LEDGER', str(tmp_path / 'ledger.jsonl'))
    monkeypatch.setattr(conus, '_surface', stage)
    conus.prepare(SimpleNamespace(out=str(tmp_path)))
    assert len(set(order)) == 2
    rows = [json.loads(line) for line in (tmp_path / 'ledger.jsonl').read_text().splitlines()]
    assert [(row['stage'], row['phase']) for row in rows] == [
        ('prep.obs.surface', 'begin'), ('prep.obs.surface', 'end')]


def test_conus_prepare_import_graph_is_ready_in_a_fresh_process():
    import subprocess
    import sys
    script = '''
from threading import Barrier
from types import SimpleNamespace
from tools import conus_da_observations as obs
barrier = Barrier(2)
def radar(args):
    barrier.wait(timeout=10)
    from woof.obs.dealias import DealiasParams
def surface(args):
    barrier.wait(timeout=10)
    from woof.obs.frontdoor import ASOS
obs.radar, obs.surface = radar, surface
obs.prepare(SimpleNamespace())
'''
    done = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=30)
    assert done.returncode == 0, done.stdout + done.stderr


def test_stage_events_pair_slot_and_site_across_threads(tmp_path):
    def stage(slot):
        with obs.observation_stage('prep.obs.radar', tmp_path, slot=slot, site='AAAA'):
            pass
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(stage, [0, 1]))
    events = [json.loads(line) for line in (tmp_path / 'timeline.jsonl').read_text().splitlines()]
    assert len(events) == 4
    for slot in (0, 1):
        rows = [e for e in events if e['attrs']['slot'] == slot]
        assert [e['phase'] for e in rows] == ['begin', 'end']
        assert ({k: v for k, v in rows[0]['attrs'].items() if k != 'cpu_s'}
                == {k: v for k, v in rows[1]['attrs'].items() if k != 'cpu_s'})
        assert all(e['product'] == 'da' for e in rows)


def test_stage_events_survive_eight_workers_emitting_together(tmp_path):
    """Every span of a busy worker pool reaches timeline.jsonl whole.

    The Windows CRT emulated O_APPEND as seek-then-write, so workers that
    sought together overwrote each other: begin/end events vanished and
    spliced lines made the timeline report refuse the file as malformed
    timing JSON (E4 5e0e1ba59, real Windows).  The two-thread case above
    failed 55 of 500 repeats there; eight workers on one ledger lost more
    than one record in ten, so this case failed every time.
    """
    go = threading.Barrier(8)

    def worker(slot):
        go.wait(timeout=10)
        for site in range(25):
            with obs.observation_stage('prep.obs.radar', tmp_path, slot=slot, site=f'S{site:03d}'):
                pass
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(worker, range(8)))
    lines = (tmp_path / 'timeline.jsonl').read_text().splitlines()
    events = [json.loads(line) for line in lines]
    assert len(events) == 8 * 25 * 2
    for slot in range(8):
        rows = [e for e in events if e['attrs']['slot'] == slot]
        assert [e['attrs']['site'] for e in rows][::2] == [f'S{site:03d}' for site in range(25)]
        assert [e['phase'] for e in rows] == ['begin', 'end'] * 25


def test_native_worker_defaults_on_both_clis():
    from tools import obs_radar_grid_build_all as allradar
    args = allradar.build_parser().parse_args(['--valid-time', '2026-10-07T12:00:00Z',
        '--grid-wrfout', 'grid', '--out', 'obs', '--work-dir', 'work'])
    assert args.workers is None and args.compute_workers is None


def test_scorer_join_waits_for_complete_hashed_truth(tmp_path):
    obs._truth_status(tmp_path, 'RUNNING')
    with ThreadPoolExecutor(max_workers=1) as pool:
        joined = pool.submit(obs.join_truth, tmp_path, timeout_seconds=5)
        time.sleep(.15)
        assert not joined.done()
        obs.save(tmp_path / 'truth.json', {'schema': 'truth-test', 'value': 1})
        # A manifest without the completion marker is still partial.
        time.sleep(.15)
        assert not joined.done()
        obs._truth_status(tmp_path, 'COMPLETE', truth_sha256=obs.sha(tmp_path / 'truth.json'))
        assert joined.result(timeout=5)['value'] == 1


def test_failed_truth_start_does_not_fail_forecast_preparation(tmp_path, monkeypatch):
    def failed(*args, **kwargs):
        raise OSError('native truth launch unavailable')
    monkeypatch.setattr(obs.subprocess, 'Popen', failed)
    job = obs.start_truth({'case': {}}, {'objects': []}, tmp_path, 0, 0)
    assert job['out'] == str(tmp_path.resolve())
    status = json.loads((tmp_path / 'truth-job-status.json').read_text())
    assert status['status'] == 'FAILED'
    with pytest.raises(RuntimeError, match='verification unavailable'):
        obs.join_truth(tmp_path)


def test_completed_truth_tampering_is_refused(tmp_path):
    obs.save(tmp_path / 'truth.json', {'value': 1})
    obs._truth_status(tmp_path, 'COMPLETE', truth_sha256=obs.sha(tmp_path / 'truth.json'))
    obs.save(tmp_path / 'truth.json', {'value': 2})
    with pytest.raises(ValueError, match='partial or changed truth'):
        obs.join_truth(tmp_path)


def test_truth_job_cannot_reuse_other_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(obs.subprocess, 'Popen', lambda *a, **k: SimpleNamespace(pid=123))
    obs.start_truth({'version': 1}, {}, tmp_path, 0, 0)
    with pytest.raises(ValueError, match='different inputs'):
        obs.start_truth({'version': 2}, {}, tmp_path, 0, 0)


def test_concurrent_truth_claims_start_one_writer(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(obs.subprocess, 'Popen', lambda *a, **k: calls.append(a) or SimpleNamespace(pid=123))
    original = obs.json.dump
    claim_open = threading.Event()
    def delayed_dump(value, stream, **kwargs):
        claim_open.set()
        time.sleep(.15)
        return original(value, stream, **kwargs)
    monkeypatch.setattr(obs.json, 'dump', delayed_dump)
    with ThreadPoolExecutor(max_workers=2) as pool:
        creator = pool.submit(obs.start_truth, {'version': 1}, {}, tmp_path, 0, 0)
        assert claim_open.wait(timeout=5)
        follower = pool.submit(obs.start_truth, {'version': 1}, {}, tmp_path, 0, 0)
        assert creator.result(timeout=5) == follower.result(timeout=5)
    assert len(calls) == 1


def test_recent_slots_overlap_surface_and_keep_slot_and_site_order(tmp_path, monkeypatch):
    from test_da_recent_observations import case, obj, contribution, surface
    from woof.obs import coverage, frontdoor, nexrad, target_grid, radar_grid, superob
    import datetime as dt
    config = case()
    config['truth_padding_provenance'] = {'model_grid_definition': {'ref_lon': -89.25, 'ref_lat': 31.25}}
    analyses = config['analysis_times_utc']
    inv = {'case': config, 'scope': 'full-case', 'publication_policy': 'test replay',
           'objects': [], 'slots': [{'analysis_time': stamp,
              'radar_candidates': {site: [] for site in config['radar_sites']}} for stamp in analyses]}
    fetched = {'objects': []}
    binary = tmp_path / 'native'; binary.write_bytes(b'native test')
    grid_path = tmp_path / 'grid'; grid_path.write_bytes(b'grid test')
    monkeypatch.setattr(obs, 'validate_fetched', lambda *args: None)
    monkeypatch.setattr(obs, 'require_native_observations', lambda: None)
    truth_centers = []
    monkeypatch.setattr(obs, 'start_truth', lambda *args, **kwargs: truth_centers.append(args[3:5]) or {'status': 'started'})
    monkeypatch.setattr(obs, 'available_memory_bytes', lambda: 32 * 2**30)
    monkeypatch.setattr(obs, 'observation_memory', lambda *args: 0)
    monkeypatch.setattr(nexrad, 'find_nexrad_bin', lambda: binary)
    monkeypatch.setattr(frontdoor.FrontDoor, 'find', lambda self: binary)
    from woof.obs import dealias
    monkeypatch.setattr(dealias, 'engine_unavailable_reason', lambda engine: None)
    monkeypatch.setattr(target_grid.TargetGrid, 'from_wrfout', lambda path: SimpleNamespace())
    monkeypatch.setattr(coverage, 'read_site_table', lambda binary: dict.fromkeys(config['radar_sites']))
    all_surfaces_started = threading.Event()
    station_started = threading.Event()
    first_radar_started = threading.Event()
    surface_count = [0]
    lock = threading.Lock()
    def native(command, log):
        if command[1] == 'stations':
            station_started.set()
            assert first_radar_started.wait(timeout=5), 'native ASOS table fetch was serialized before radar'
        if command[1] == 'fetch':
            with lock:
                surface_count[0] += 1
                if surface_count[0] == 3:
                    all_surfaces_started.set()
        if '--out' in command:
            path = command[command.index('--out') + 1]
            if command[1] == 'decode':
                stamp = command[command.index('--end') + 1]
                observed = obs.iso(obs.seam_utc(stamp) - dt.timedelta(minutes=1))
                obs.save(path, surface(stamp=stamp, observed=observed))
            else:
                from pathlib import Path
                Path(path).write_bytes(b'native test')
        return {'status': 'PASS'}, .001
    monkeypatch.setattr(obs, 'run_json', native)
    def site_work(site, keys, raw, case, analysis, *args):
        if site == case['radar_sites'][0] and obs.iso(analysis) == analyses[0]:
            first_radar_started.set()
            assert station_started.wait(timeout=5)
        else:
            assert all_surfaces_started.wait(timeout=5), 'slots or ASOS were serialized behind radar'
        return contribution(), {'site': site, 'measured_end_utc': obs.iso(analysis)}, []
    monkeypatch.setattr(obs, '_site_volume', site_work)
    monkeypatch.setattr(superob, 'merge_contributions', lambda *a, **k: None)
    def write(path, *args, **kw):
        path.write_bytes(kw['valid_time'].encode())
        return {'sha256': obs.sha(path)}
    monkeypatch.setattr(radar_grid, 'write_radar_grid', write)
    budgets = _fake_native_thread_budget(monkeypatch)
    result = obs.prepare(inv, fetched, grid_path, tmp_path / 'out')
    assert len(budgets) == 1
    assert truth_centers == [(-89.25, 31.25)]
    assert [slot['analysis_time'] for slot in result['slots']] == analyses
    for slot in result['slots']:
        assert [row['site'] for row in slot['radar_sources']] == config['radar_sites']
    events = [json.loads(line) for line in (tmp_path / 'out/timeline.jsonl').read_text().splitlines()]
    slots = [row for row in events if row['stage'] == 'prep.obs.slot']
    assert [row['phase'] for row in slots[:3]] == ['begin'] * 3


def test_conus_drains_parallel_slots_in_order_without_repeating_calibration(tmp_path, monkeypatch):
    import contextlib
    from tools import conus_da_observations as conus
    from woof.obs import coverage, dealias, nexrad, target_grid, radar_grid, superob, superob_bridge
    names = ['KAAA', 'KBBB']
    grid_path = tmp_path / 'grid'; grid_path.write_bytes(b'grid')
    binary = tmp_path / 'native'; binary.write_bytes(b'native')
    grid = SimpleNamespace(identity_sha256=lambda: 'identity')
    monkeypatch.setattr(obs, 'require_native_observations', lambda: None)
    monkeypatch.setattr(obs, 'observation_memory', lambda *a: 0)
    monkeypatch.setattr(obs, 'available_memory_bytes', lambda: 32 * 2**30)
    monkeypatch.setattr(coverage, 'read_site_table', lambda *a: dict.fromkeys(names))
    monkeypatch.setattr(coverage, 'sites_covering', lambda *a, **k: [SimpleNamespace(
        site=SimpleNamespace(id=name), to_payload=lambda: {}) for name in names])
    monkeypatch.setattr(nexrad, 'find_nexrad_bin', lambda: binary)
    monkeypatch.setattr(dealias, 'engine_unavailable_reason', lambda *a: None)
    monkeypatch.setattr(superob_bridge, 'unavailable_reason', lambda: None)
    monkeypatch.setattr(target_grid.TargetGrid, 'from_wrfout', lambda *a: grid)
    monkeypatch.setattr(conus, 'list_objects', lambda *a: [])
    rendezvous = threading.Barrier(3)
    original_stage = obs.observation_stage
    @contextlib.contextmanager
    def stages(stage, out, **attrs):
        with original_stage(stage, out, **attrs):
            if stage == 'prep.obs.slot':
                rendezvous.wait(timeout=5)
            yield
    monkeypatch.setattr(obs, 'observation_stage', stages)
    calls = []
    def native(task):
        calls.append((task[0], task[1]))
        return object(), {'site': task[0], 'status': 'contributed', 'seconds': {}, 'refusals': []}
    monkeypatch.setattr(conus, 'site_cycle', native)
    merged_rosters = []
    monkeypatch.setattr(superob, 'merge_contributions', lambda rows, *a, **k: merged_rosters.append(len(rows)))
    def write(path, *a, **kw):
        assert [r['site'] for r in kw['provenance']['sources']] == names
        path.write_bytes(kw['valid_time'].encode())
        return {'sha256': obs.sha(path), 'bytes': path.stat().st_size}
    monkeypatch.setattr(radar_grid, 'write_radar_grid', write)
    analyses = ['2026-10-03T22:00:00Z', '2026-10-03T23:00:00Z', '2026-10-04T00:00:00Z']
    args = SimpleNamespace(out=str(tmp_path / 'out'), grid_wrfout=str(grid_path), cycles=analyses,
        workers=None, limit_sites=0, raw_cache=None, publication_gate='measured-end')
    budgets = _fake_native_thread_budget(monkeypatch)
    conus.radar(args)
    assert len(budgets) == 1
    doc = json.loads((tmp_path / 'out/radar-slots.json').read_text())
    assert [slot['analysis_time'] for slot in doc['slots']] == analyses
    assert len(calls) == 6 and len(set(calls)) == 6
    assert merged_rosters == [2, 2, 2]
    assert doc['workers'] == len(names)
