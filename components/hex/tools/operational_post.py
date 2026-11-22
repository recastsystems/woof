#!/usr/bin/env python3
"""Render verified forecast frames with Rust after releasing the card grant."""
import argparse
import json
import os
from woof.hex import card_queue
from pathlib import Path
import subprocess
import time


def render_command(case, run, histories):
    return [os.sys.executable, '-m', 'woof.hex.cli', 'render', '--history', *map(str, histories),
            '--mesh', case['static'], '--init', case['init'], '--out', str(run / 'png'),
            '--simulation-start', case['start'], '--scratch', str(run / 'render-scratch'),
            '--products', case.get('products', 'direct'), '--window', 'mesh', '--size', '1200x900']


def selected_histories(run, proof, endpoints_only=False):
    names = sorted(proof['history_files'])
    if not names:
        raise ValueError('verified forecast has no published histories')
    if endpoints_only:
        names = list(dict.fromkeys((names[0], names[-1])))
    return [run / 'fc' / name for name in names]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--endpoints-only', action='store_true',
                        help='render the initial and final published frames')
    args = parser.parse_args()
    run = args.run.resolve()
    case = json.loads((run / 'case.json').read_text())
    proof = json.loads((run / 'proof.log').read_text())
    if proof.get('status') != 'PASS':
        raise ValueError('cannot show a forecast without complete forecast and sanity evidence')
    histories = selected_histories(run, proof, args.endpoints_only)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    card_queue.drop(env, card_queue.CARDS)
    command = render_command(case, run, histories)
    started = time.monotonic()
    with (run / 'render.log').open('x') as log:
        rc = subprocess.run(command, env=env, cwd=run, stdout=log, stderr=subprocess.STDOUT, check=False).returncode
    pngs = sorted((run / 'png').rglob('*.png'))
    from operational_proof import file_hash
    result = {'command': command, 'rc': rc, 'wall_seconds': time.monotonic() - started,
              'history_selection': 'endpoints' if args.endpoints_only else 'all',
              'histories': [str(p.relative_to(run)) for p in histories],
              'png_count': len(pngs), 'maps': {str(p.relative_to(run)): file_hash(p) for p in pngs},
              'status': 'PASS' if rc == 0 and pngs else 'INCOMPLETE'}
    (run / 'render-receipt.json').write_text(json.dumps(result, indent=2))
    stages = json.loads((run / 'stages.json').read_text())
    stages['render'] = result
    stages['forecast_and_render_wall_seconds'] = stages['forecast']['wall_seconds'] + result['wall_seconds']
    (run / 'stages.json').write_text(json.dumps(stages, indent=2))
    with (run.parents[1] / 'MANIFEST.txt').open('a') as owned:
        for path in sorted(run.rglob('*')):
            owned.write(str(path) + '\n')
    print(json.dumps(result, indent=2))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
