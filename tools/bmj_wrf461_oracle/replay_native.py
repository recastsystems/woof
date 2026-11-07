"""Replay BMJDRV records captured from an instrumented real WRF forecast.

The capture procedure and record layout are documented in README.md. This
checks float32 words, including signed zero, rather than an ULP tolerance.
Set GPUWM_NUMERICS=strict before running the proof. The default production
arithmetic can be checked with the same command in a separate process.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from bmj_oracle_io import LEVEL_OUTPUTS, SURFACE_OUTPUTS, read_arm


def replay(directory: Path) -> dict:
    import cupy as cp
    from woof.core.column_cumulus import column_kernel_spec, column_tables, run_columns
    from woof.numerics import mode

    spec = column_kernel_spec(2)
    tables = column_tables(spec)
    device_tables = cp.asnumpy(tables).view(np.uint32)
    records = []
    for capture in sorted(directory.glob('bmj-native-*.bin')):
        meta = Path(str(capture) + '.txt').read_text().split()
        ncol, nz = map(int, meta[:2])
        dt, stepcu, feedback = float(meta[2]), int(meta[3]), bool(int(meta[4]))
        source = read_arm(capture, ncol, nz, nstep=1)[0]
        cldefi = cp.asarray(source['in_cldefi'])
        out = run_columns(
            spec, level={key: source['in_' + key] for key in
                         ('t', 'qv', 'pmid', 'pi', 'rho', 'dz8w')},
            interface={'pint': source['in_pint']},
            column={'xland': source['in_xland']}, carried={'cldefi': cldefi},
            dt=dt, stepcu=stepcu, rad_feedback=feedback, tables=tables)
        got = {key: cp.asnumpy(value) for key, value in out.items()}
        got.update(cutop=got['htop'], cubot=got['hbot'], cldefi=cp.asnumpy(cldefi))
        fields = {}
        for key in LEVEL_OUTPUTS + SURFACE_OUTPUTS:
            actual, expected = got[key].view(np.uint32), source[key].view(np.uint32)
            indices = np.argwhere(actual != expected)
            fields[key] = {'words': int(expected.size), 'differing_words': len(indices)}
            if len(indices):
                at = tuple(int(v) for v in indices[0])
                fields[key]['first_difference'] = {
                    'index': at, 'actual_hex': f'{actual[at]:08x}',
                    'native_hex': f'{expected[at]:08x}',
                    'actual': float(got[key][at]), 'native': float(source[key][at])}
        native_tables = np.fromfile(str(capture) + '.tables', dtype='<u4')
        if native_tables.shape != device_tables.shape:
            raise ValueError(f'{capture}: native and device table lengths differ')
        fields['bmjinit_tables'] = {
            'words': int(native_tables.size),
            'differing_words': int(np.count_nonzero(native_tables != device_tables))}
        # Native BMJDRV sets this flag true for every column. The engine
        # column route applies every returned column without an activity mask.
        active = source['cu_act_flag']
        if not np.all(active == 1):
            raise ValueError(f'{capture}: native BMJDRV did not mark every column active')
        records.append({
            'capture': capture.name, 'capture_bytes': capture.stat().st_size,
            'capture_sha256': hashlib.sha256(capture.read_bytes()).hexdigest(),
            'columns': ncol, 'levels': nz, 'dt_seconds': dt, 'stepcu': stepcu,
            'bmj_rad_feedback': feedback, 'fields': fields,
            'native_active_columns': int(active.sum()),
            'gpu_path_counts': {str(int(path)): int(count) for path, count in
                                zip(*np.unique(got['bmj_path'], return_counts=True))},
            'precipitating_columns': int(np.count_nonzero(source['raincv'] > 0)),
            'words': sum(field['words'] for field in fields.values()),
            'differing_words': sum(field['differing_words'] for field in fields.values())})
    if not records:
        raise ValueError(f'{directory}: no bmj-native-*.bin capture records')
    return {'schema': 'gpuwm.bmj-native-replay.v1', 'numerics': mode(),
            'device': cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
            'records': records, 'words': sum(row['words'] for row in records),
            'differing_words': sum(row['differing_words'] for row in records)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    receipt = replay(args.directory)
    args.receipt.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({key: receipt[key] for key in ('numerics', 'words', 'differing_words')}))
    return int(receipt['differing_words'] != 0)


if __name__ == '__main__':
    raise SystemExit(main())
