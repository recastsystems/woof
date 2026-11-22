"""Real hex forecast on every card of the box, at the door's own defaults.

Adapted from the speed lane's run_speed.py.  Differences, each for a reason:
- no --dt, --time-integration-order, --acoustic-substeps or --radiation-seconds
  unless asked: the door's defaults are MPAS-A v8.4.2's (RK2, 4 acoustic
  substeps, 3 splits, monotonic, radiation 1800 s, 6 s per km), and a run that
  passes nothing is the proof that a bare run gets them;
- history every --history-every-minutes (default 360), written off the step
  path by the samehost writer thread (WOOF_HEX_ASYNC_HISTORY default 1);
- convection from the door (auto) unless --convection is given;
- per-hour progress lines and partial-speed.json while it runs.

usage: run_allcards.py CASE.json ARM --hours H [--cards N] [--history-every-minutes M]
                  [--dt S] [--convection auto|gf|ntiedtke|off] [--extra "..."]
"""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from woof.hex import card_queue
from pathlib import Path
import subprocess
import sys
import threading
import time


def card_label():
    r = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                       capture_output=True, text=True)
    names = sorted({x.strip() for x in r.stdout.splitlines() if x.strip()})
    return '-'.join(names[0].replace('NVIDIA ', '').split()) if names else 'unknown'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('case', type=Path)
    p.add_argument('arm')
    p.add_argument('--hours', type=float, required=True)
    p.add_argument('--cards', type=int, default=None)
    p.add_argument('--history-every-minutes', type=float, default=360)
    p.add_argument('--dt', type=float, default=None)
    p.add_argument('--nominal-dt', type=float, default=180.0,
                   help='only for progress accounting when --dt is not passed')
    p.add_argument('--convection', default=None)
    p.add_argument('--extra', default='')
    p.add_argument('--door', action='store_true',
                   help='call the forecast door (woof.hex.cli forecast) directly instead of the '
                        'check_samehost measurement harness; no boundary fingerprints unless asked')
    p.add_argument('--fingerprint-every', type=int, default=None)
    p.add_argument('--case-label', default='ops')
    a = p.parse_args()
    cards = a.cards or len([c for c in card_queue.granted() if c])
    if cards < 1:
        raise SystemExit('no cards: run through the host card queue with --cards N')
    profile = json.loads(a.case.read_text())
    layout_root = Path(profile['layout_roots'][str(cards)])
    entries = list(layout_root.glob('*.K3.json'))
    if len(entries) != 1:
        raise ValueError(f'exactly one verified {cards}-rank layout is required in {layout_root}')
    layout = json.loads(entries[0].read_text())
    root = Path(os.environ['L'])
    run = root / 'runs' / (profile['case'] + '-' + Path(profile['init']).parent.name.split('-')[-1] + '-' + a.arm)
    run.mkdir(parents=True, exist_ok=False)
    tree = Path(os.environ['HEX_CAPABILITY_CHECKOUT'])
    env = dict(os.environ, WOOF_HEX_MESH_ROWS=profile['mesh_rows'], GPUWM_WRF_EXACT='0',
               WOOF_HEX_PREP_CACHE=str(root / 'hostprep-cache'))
    dt = a.dt or a.nominal_dt
    expected = int(round(a.hours * 3600 / dt))
    flags = [f for f in profile['forecast_flags']]
    if '--convection' in flags:
        i = flags.index('--convection')
        del flags[i:i + 2]
    if a.convection and a.convection != 'auto':
        flags += ['--convection', a.convection]
    if a.door:
        fp = 0 if a.fingerprint_every is None else a.fingerprint_every
        cmd = [sys.executable, '-m', 'woof.hex.cli']
    else:
        fp = expected if a.fingerprint_every is None else a.fingerprint_every
        cmd = [sys.executable, str(tree / 'tools/check_samehost_forecast_v841.py'),
               '--card', card_label(), '--fingerprint-every', str(max(1, fp)),
               '--report', str(run / 'measurement.json')]
        cmd += ['--']
    cmd += ['forecast', '--mesh', profile['mesh'], '--grid', profile['grid'],
            '--static', profile['static'], '--init', profile['init'], '--hours', repr(a.hours),
            '--start-time', profile['start'], '--history-every-minutes', str(a.history_every_minutes),
            '--init-source', profile['init_source'], '--out', str(run / 'fc'),
            '--scratch', str(run / 'scratch'), '--case-label', a.case_label,
            '--gpuwm-checkout', os.environ['E'],
            '--samehost-devices', ','.join(map(str, range(cards))),
            '--samehost-layouts', layout['layouts'], '--samehost-part', layout['part'],
            '--samehost-fingerprint-every', str(fp), *flags]
    if a.dt:
        cmd += ['--dt', repr(a.dt)]
    cmd += [x for x in a.extra.split() if x]
    (run / 'command.json').write_text(json.dumps({
        'command': cmd, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'gil_enabled': sys._is_gil_enabled(), 'cards': card_queue.get(card_queue.CARDS, None, env),
        'CUDA_VISIBLE_DEVICES': env.get('CUDA_VISIBLE_DEVICES'), 'hours': a.hours,
        'history_every_minutes': a.history_every_minutes,
        'schedule': 'door defaults unless the command names a value',
        'tree': str(tree), 'source_case': profile}, indent=2))
    samples = []
    stop = threading.Event()
    sample_interval = 0.5

    def monitor():
        while not stop.is_set():
            r = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                                '--format=csv,noheader,nounits'], capture_output=True, text=True)
            if r.returncode == 0:
                rows = [x.split(',') for x in r.stdout.splitlines()]
                samples.append({'t': time.monotonic(), 'MiB': {int(x[0]): int(x[1]) for x in rows},
                                'util': {int(x[0]): int(x[2]) for x in rows}})
            stop.wait(sample_interval)
    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    started = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    times = []
    first_step_wall = None
    seen_dt = None
    try:
        with (run / 'forecast.log').open('w') as f:
            child = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, bufsize=1)
            for line in child.stdout:
                f.write(line)
                f.flush()
                if line.startswith('{"step":'):
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue
                    if row.get('samehost_ranks') == cards and 'seconds' in row:
                        if first_step_wall is None:
                            first_step_wall = time.monotonic() - started
                        times.append(float(row['seconds']))
                        if seen_dt is None and row.get('time_seconds') and row.get('step'):
                            seen_dt = float(row['time_seconds']) / float(row['step'])
                        step_dt = seen_dt or dt
                        hour_step = max(1, int(round(3600 / step_dt)))
                        if row['step'] % hour_step == 0:
                            h = row['step'] / hour_step
                            partial = {'completed_hours': h, 'step': row['step'], 'dt_seconds': step_dt,
                                       'integration_seconds': sum(times),
                                       'seconds_per_forecast_hour_integration': sum(times) / h,
                                       'wall_seconds_since_start': time.monotonic() - started,
                                       'wall_seconds_since_first_step': time.monotonic() - started - first_step_wall,
                                       'max_abs_w': row.get('vertical_velocity_abs_max')}
                            (run / 'partial-speed.json').write_text(json.dumps(partial, indent=2))
                            print('HOUR ' + json.dumps(partial), flush=True)
            rc = child.wait()
    finally:
        stop.set()
        watcher.join(timeout=5)
    wall = time.monotonic() - started
    peaks = {}
    for s in samples:
        for k, v in s['MiB'].items():
            peaks[k] = max(peaks.get(k, 0), v)
    (run / 'gpu-samples.json').write_text(json.dumps({'interval_s': sample_interval,
                                                     'peak_definition': 'maximum sampled nvidia-smi memory.used, including context and allocator reserve', 'peak_MiB': peaks,
                                                     'samples': samples}))
    receipt_path = run / 'fc/cuda-v841-forecast-receipt.json'
    doc = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    fc = doc.get('forecast', {})
    health = fc.get('health', [])
    timeline = run / 'fc/timeline.jsonl'
    startup = None
    if timeline.exists():
        for line in timeline.read_text().splitlines():
            event = json.loads(line)
            if event.get('stage') == 'forecast.first_step' and event.get('phase') == 'begin':
                startup = float(event['t_mono']) - started
                break
    summary = {
        'case': profile['case'], 'arm': a.arm, 'rc': rc, 'cards': cards, 'hours': a.hours,
        'started_utc': started_utc, 'finished_utc': datetime.now(timezone.utc).isoformat(),
        'door_wall_seconds': wall, 'startup_seconds_to_first_step': first_step_wall,
        'startup_seconds_before_first_step': startup,
        'startup_definition': 'runner start through first-step begin in monotonic timeline; excludes first step',
        'steps_executed': fc.get('steps_executed'), 'step_seconds_sum': sum(times),
        'seconds_per_forecast_hour_integration': sum(times) / a.hours if times else None,
        'seconds_per_forecast_hour_all_in': wall / a.hours,
        'minutes_per_forecast_day_all_in': wall / a.hours * 24 / 60,
        'minutes_per_forecast_day_integration': (sum(times) / a.hours * 24 / 60) if times else None,
        'walls': fc.get('walls'), 'refusal': fc.get('refusal'), 'driver_status': doc.get('status'),
        'finite_every_step_every_rank': bool(health) and all(
            all(item.get('finite') is True for item in rank) for rank in health),
        'max_abs_w_m_s': max((item.get('vertical_velocity_abs_max', 0) for rank in health for item in rank),
                             default=None),
        'peak_MiB_by_card': peaks, 'forecast_receipt': str(receipt_path),
    }
    (run / 'summary.json').write_text(json.dumps(summary, indent=2))
    print('SUMMARY ' + json.dumps(summary), flush=True)
    return rc


if __name__ == '__main__':
    sys.exit(main())
