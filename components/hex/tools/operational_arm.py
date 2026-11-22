#!/usr/bin/env python3
"""Run the good-config real-door measurement and receipt check on a queue grant.

Prepared case paths are references to the shared cache. No model data is decoded
here. Invoke only through the host's card queue with a timing tag and --cards N.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from woof.hex import card_queue
from pathlib import Path
import subprocess
import time


def forecast_command(case, run, cards, hours):
    root = Path(case['hex_tree'])
    dt = float(case['dt_seconds'])
    steps = round(hours * 3600 / dt)
    history = case.get('history_minutes', 60)
    stride = round(history * 60 / dt)
    if (steps < 1 or stride < 1 or abs(steps * dt - hours * 3600) > 1e-7
            or abs(stride * dt - history * 60) > 1e-7 or steps % stride):
        raise ValueError('requested forecast must contain complete steps and history intervals')
    cadence = case.get('fingerprint_every', max(1, round(hours * 3600 / case['dt_seconds'])))
    if not isinstance(cadence, int) or cadence < 1:
        raise ValueError('real-door measurement requires a positive fingerprint cadence')
    args = ['forecast', '--mesh', case['mesh'], '--grid', case['grid'], '--static', case['static'],
            '--init', case['init'], '--lbc-dir', case['lbc'], '--start-time', case['start'],
            '--hours', str(hours), '--history-every-minutes', str(history),
            '--init-source', case['init_source'], '--out', str(run / 'fc'), '--scratch', str(run / 'scratch'),
            '--gpuwm-checkout', case['engine'], '--dt', repr(dt), *case.get('forecast_flags', [])]
    if cards > 1:
        layout = case['layouts'][str(cards)]
        args += ['--samehost-devices', ','.join(map(str, range(cards))), '--samehost-layouts', layout['layouts'],
                 '--samehost-part', layout['part'], '--samehost-fingerprint-every', str(cadence)]
    command = [os.sys.executable, str(root / 'tools' / 'check_samehost_forecast_v841.py'),
               '--report', str(run / 'measurement.json'), '--fingerprint-every', str(cadence)]
    # The case's own dt reaches the forecast as --dt above (the door would
    # otherwise take MPAS-A's 6 s per km, and this tool's step and history
    # arithmetic is done at the case's dt).  --measurement-mint only labels
    # the receipt since 2026-10-10: there is no anchor gate for it to open.
    if case.get('measurement_mint'):
        command += ['--measurement-mint', '--card', case['card_name']]
    return command + ['--', *args]


def stage(name, command, run):
    start = time.monotonic()
    with (run / (name + '.log')).open('x') as log:
        process = subprocess.run(command, cwd=run, stdout=log, stderr=subprocess.STDOUT, check=False)
    return {'command': command, 'rc': process.returncode, 'wall_seconds': time.monotonic() - start}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', type=Path)
    parser.add_argument('arm')
    parser.add_argument('--cards', type=int, choices=(1, 4, 8), required=True)
    parser.add_argument('--hours', type=float, required=True)
    parser.add_argument('--arithmetic', choices=('default', 'strict'), default='default')
    parser.add_argument('--root', type=Path, default=Path(os.environ.get('HEX_OPERATIONAL_ROOT', 'operational-arm')))
    args = parser.parse_args()
    grant = [v for v in card_queue.granted() if v]
    if len(grant) != args.cards:
        raise ValueError('timed forecast must run under an exact whole-card queue grant')
    case = json.loads(args.case.read_text())
    if case.get('mesh_rows'):
        os.environ['WOOF_HEX_MESH_ROWS'] = case['mesh_rows']
    if args.hours > case['boundary_hours']:
        raise ValueError('forecast exceeds prepared boundary coverage')
    if args.arithmetic == 'strict':
        os.environ['GPUWM_WRF_EXACT'] = '1'
    else:
        os.environ.pop('GPUWM_WRF_EXACT', None)
    os.environ['PYTHON_GIL'] = '0'
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run = args.root / 'runs' / f'{args.arm}-n{args.cards}-{args.arithmetic}-{stamp}'
    with (args.root / 'MANIFEST.txt').open('a') as manifest:
        manifest.write(str(run) + '\n')
    run.mkdir(parents=True, exist_ok=False)
    (run / 'case.json').write_text(json.dumps(case, indent=2))
    manifest = {'created_utc': stamp, 'hours': args.hours, 'dt_seconds': case['dt_seconds'],
                'cards': args.cards, 'physical_grant': grant, 'arithmetic': args.arithmetic,
                'prepared_case': str(args.case), 'init_source': case['init_source'],
                'whole_box_usd_per_hour': case.get('whole_box_usd_per_hour'),
                'host_load_average': os.getloadavg() if hasattr(os, 'getloadavg') else None}
    result = stage('forecast', forecast_command(case, run, args.cards, args.hours), run)
    manifest.update(forecast_rc=result['rc'], forecast_wall_seconds=result['wall_seconds'])
    (run / 'stages.json').write_text(json.dumps({'forecast': result}, indent=2))
    (run / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    summary = stage('proof', [os.sys.executable, str(Path(case['hex_tree']) / 'tools' / 'operational_proof.py'), str(run)], run)
    with (args.root / 'MANIFEST.txt').open('a') as owned:
        for path in sorted(run.rglob('*')):
            owned.write(str(path) + '\n')
    print(json.dumps({'run': str(run), 'forecast': result, 'proof': summary}))
    return result['rc'] or summary['rc']


if __name__ == '__main__':
    raise SystemExit(main())
