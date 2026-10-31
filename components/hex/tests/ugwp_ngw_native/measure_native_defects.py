"""Measure pinned full-native climate defects at fixed ranks; never acceptance.

Run once on each host with the SAME published strict binary and MPI libraries.
The supplied 20-minute deck must write only diag_ugwp, after the physics call:
the general diagnostics writer crashes on NaN dycore state before that stream.
The Rust fixture changes DAYS only. Rust decodes every diagnostic array.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys

import numpy as np

from compare_real import Transport


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('build', 'deck', 'case', 'tables', 'mpi', 'climate',
                'transport', 'fixture', 'work', 'receipt'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--pin-set-hash', required=True)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=False)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    host = socket.gethostname()
    runs = []
    runner = Path(__file__).resolve().parents[2]/'tools/native_physics_reference/run_forecast.py'
    env = dict(os.environ)
    env.pop('MPAS_ORACLE_DIR', None)
    provenance = {
        'schema': 'p15.native-time-defect/v2', 'acceptance': False,
        'host': host, 'pin_set_hash': args.pin_set_hash,
        'binary_sha256': digest(args.build/'atmosphere_model'),
        'namelist_sha256': digest(args.deck/'namelist.atmosphere'),
        'streams_sha256': digest(args.deck/'streams.atmosphere'),
        'source_climate_sha256': digest(args.climate),
        'fixture_sha256': digest(args.fixture),
        'transport_sha256': digest(args.transport), 'runs': runs,
    }
    for mode in ('before-first', 'after-last'):
        case = args.work/mode
        case.mkdir()
        for source in args.case.iterdir():
            if source.is_file() and (source.suffix == '.nc' or '.graph.info' in source.name):
                (case/source.name).symlink_to(source.resolve())
        climate = case/'ugwp_limb_tau.nc'
        if climate.exists():
            raise ValueError('case must not already contain the climate fixture')
        subprocess.run([str(args.fixture.resolve()), str(args.climate.resolve()),
                        str(climate.resolve()), mode], check=True)
        for ranks in (1, 2):
            for repeat in (0, 1):
                out = args.work/f'{mode}-rank{ranks}-repeat{repeat}'
                command = [sys.executable, str(runner), '--build', str(args.build.resolve()),
                           '--deck', str(args.deck.resolve()), '--case', str(case.resolve()),
                           '--out', str(out.resolve()), '--tables', str(args.tables.resolve()),
                           '--mpi', str(args.mpi.resolve()), '--ranks', str(ranks),
                           '--pin-set-hash', args.pin_set_hash]
                with (args.work/(out.name+'.log')).open('w') as log:
                    result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                record = {'mode': mode, 'process_count': ranks, 'repeat': repeat,
                          'exit_code': result.returncode, 'command': command,
                          'climate_sha256': digest(climate), 'fields': []}
                for name in ('native.stdout.log', 'log.atmosphere.0000.out'):
                    path = out/name
                    record[name] = path.read_text()[-12000:] if path.exists() else ''
                transport = Transport(args.transport)
                try:
                    for path in sorted(out.glob('diag_ugwp.*.nc')):
                        for name in ('dudt_ngw', 'dvdt_ngw', 'dtdt_ngw'):
                            array = transport.netcdf(path, name)
                            nonfinite = array.view('u4')[~np.isfinite(array)]
                            values, counts = np.unique(nonfinite, return_counts=True)
                            record['fields'].append({
                                'file': path.name, 'field': name, 'shape': list(array.shape),
                                'sha256': hashlib.sha256(array.tobytes()).hexdigest(),
                                'nonfinite_words': int(nonfinite.size),
                                'nonfinite_bits': {hex(int(v)): int(c) for v, c in zip(values, counts)},
                            })
                finally:
                    transport.close()
                # Initial zeros never witness the scheme's affected words.
                if result.returncode == 0:
                    assert any(f['file'].endswith('_12.20.00.nc') for f in record['fields'])
                runs.append(record)
                args.receipt.write_text(json.dumps(provenance, indent=2)+'\n')
                print(host, mode, ranks, repeat, result.returncode, flush=True)
                # Fresh output belongs to this invocation; hashes precede deletion.
                shutil.rmtree(out)


if __name__ == '__main__':
    main()
