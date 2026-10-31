"""Runtime regional classes earned by two independent dry forecast processes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

from .errors import MpasPortError

SCHEMA = 'gpuwm-hex.regional-class-mint/v1'


def class_id(key):
    payload = json.dumps(key.as_dict(), sort_keys=True).encode()
    return 'runtime-' + hashlib.sha256(payload).hexdigest()[:24]


def checked_pair(runs, key):
    from .drivers.run_cuda_regional_mint import FIELDS, mint_steps

    if len(runs) != 2 or runs[0].get('pid') == runs[1].get('pid'):
        raise ValueError('regional mint needs two independent processes')
    for run in runs:
        try:
            steps = mint_steps(run.get('start_time'), run.get('stop_time'), key.dt_ms / 1000)
        except (TypeError, ValueError) as error:
            raise ValueError(f'regional mint has an invalid forecast window: {error}') from error
        if (run.get('schema') != 'gpuwm-hex.regional-mint-run/v1'
                or run.get('instrument') != 'run_cuda_regional_mint'
                or run.get('minted_without_anchor_gate') is not True
                or run.get('numerics') != 'strict' or run.get('key') != key.as_dict()
                or type(run.get('steps')) is not int or run['steps'] != steps):
            raise ValueError('regional mint identity, numerics or step count differs')
        frames = run.get('frames') or []
        if [frame.get('step') for frame in frames] != list(range(steps + 1)):
            raise ValueError('regional mint is missing forecast steps')
        for frame in frames:
            hashes = frame.get('field_sha256') or {}
            if (set(hashes) != set(FIELDS) or set(frame.get('finite') or {}) != set(FIELDS)
                    or any(frame['finite'][name] is not True for name in FIELDS)
                    or any(len(hashes[name]) != 64 or any(c not in '0123456789abcdef' for c in hashes[name])
                           for name in FIELDS)):
                raise ValueError('regional mint fields are missing, non-finite or not digested')
    for name in ('start_time', 'stop_time', 'config_sha256', 'driver_sha256',
                 'n_cells', 'bdy_mask_sha256', 'frames'):
        if not runs[0].get(name) or runs[0][name] != runs[1].get(name):
            raise ValueError(f'regional mint pair differs in {name}')
    return runs[0]


def read_class(path):
    from .cuda_backend.regional_admission import RegionalClass, RegionalClassKey, _DRY_MINT_CAVEAT
    from .drivers import run_cuda_regional_mint

    document = json.loads(path.read_text())
    if document.get('schema') != SCHEMA:
        raise ValueError('not a regional class mint')
    raw = document['key']
    key = RegionalClassKey.build(boundary_zone_width=raw['boundary_zone_width'],
                                n_vert_levels=raw['n_vert_levels'], finest_edge_m=raw['finest_edge_m'],
                                dt_seconds=raw['dt_seconds'], kernel_set=raw['kernel_set_sha256'])
    if document['class_id'] != class_id(key):
        raise ValueError('regional class name differs from its measured key')
    names = [item['file'] for item in document['runs']]
    if any(not isinstance(name, str) or Path(name).name != name for name in names):
        raise ValueError('regional mint run receipts must be adjacent files')
    paths = [path.parent / name for name in names]
    if len(paths) != 2 or len(set(paths)) != 2:
        raise ValueError('regional mint needs two distinct run receipts')
    runs = []
    for run_path, item in zip(paths, document['runs']):
        content = run_path.read_bytes()
        if hashlib.sha256(content).hexdigest() != item['sha256']:
            raise ValueError('regional mint run receipt changed')
        runs.append(json.loads(content))
    first = checked_pair(runs, key)
    if first['driver_sha256'] != hashlib.sha256(Path(run_cuda_regional_mint.__file__).read_bytes()).hexdigest():
        raise ValueError('regional mint instrument changed')
    return RegionalClass(
        class_id=document['class_id'], key=key, parent=document['parent'], card=first['card'],
        admitted_on=document['generated_utc'], mint_receipts=tuple(str(item) for item in paths),
        minted_on_geometries=(document['mesh_row'],),
        basis=(f"Runtime mint: two independent strict dry regional forecasts, {first['steps']} steps each; "
               'all six published fields finite and byte-identical at every step. ' + _DRY_MINT_CAVEAT))


def presented_classes():
    from .cuda_backend.regional_admission import _ledger_directories

    for directory in _ledger_directories():
        for path in sorted(directory.glob('*.mint.json')):
            try:
                yield read_class(path)
            except (OSError, ValueError, KeyError, TypeError):
                # Invalid evidence never contributes an admitted class.
                continue


def mint(key, generated, value, status):
    from .drivers.run_cuda_regional_mint import mint_steps
    from .run_status import utc_now

    try:
        steps = mint_steps(value['start_time'], value['stop_time'], key.dt_ms / 1000)
    except ValueError as error:
        raise MpasPortError(f'regional class mint refused: {error}') from error
    root = Path(value['paths']['contract'])
    root.mkdir(parents=True, exist_ok=True)
    identifier = class_id(key)
    path = root / (identifier + '.mint.json')
    try:
        held = read_class(path)
        if held.key.matches(key):
            return held
    except (OSError, ValueError, KeyError, TypeError):
        pass
    cull = generated['cull']
    runs = []
    for number in (1, 2):
        receipt = root / f'{identifier}-run{number}.json'
        command = [sys.executable, '-m', 'woof.hex.drivers.run_cuda_regional_mint',
                   '--grid', cull['grid'], '--static', cull['static'], '--init', value['paths']['init'],
                   '--lbc-dir', cull['lbc_dir'], '--start-time', value['start_time'],
                   '--stop-time', value['stop_time'],
                   '--dt', str(key.dt_ms / 1000), '--len-disp', str(value['fine_dx_m']),
                   '--out', str(receipt)]
        status.note(' '.join(command))
        log = root / f'{identifier}-run{number}.log'
        with log.open('w') as handle:
            result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
        if result.returncode != 0:
            raise MpasPortError(f'regional class mint run {number} exited {result.returncode}; log {log}')
        runs.append(json.loads(receipt.read_text()))
    try:
        checked_pair(runs, key)
    except ValueError as error:
        raise MpasPortError(f'regional class mint failed: {error}; receipts in {root}') from error
    document = dict(schema=SCHEMA, class_id=identifier, key=key.as_dict(),
                    parent=generated['parent']['row'], mesh_row=cull['row'], generated_utc=utc_now(),
                    runs=[dict(file=f'{identifier}-run{number}.json', sha256=hashlib.sha256(
                        (root / f'{identifier}-run{number}.json').read_bytes()).hexdigest()) for number in (1, 2)])
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(document, indent=2) + '\n')
    temporary.replace(path)
    status.note(f'MINT class={identifier} dual_run=True steps={steps} fields=6 receipt={path}')
    return read_class(path)
