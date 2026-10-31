#!/usr/bin/env python3
"""Run the audited native microphysics table utilities and retain raw bytes.

Thompson authority: utils/build_tables.F:10-19 and
utils/atmphys_build_tables_thompson.F:37-114 at fork 1d6e370e.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from publish_cache import publish, sha


UTILITIES = {
    'tempo_v3': ('build_tables_tempo_v3', ('tempo_v3_tables.o', 'tempo_v3_program.o'),
                 ('freeze_water_data_tempo_v3', 'qr_acr_qs_data_tempo_v3', 'qr_acr_qg_data_tempo_v3')),
    'thompson': ('build_tables', ('build_tables.o', 'atmphys_build_tables_thompson.o'),
                 ('MP_THOMPSON_QRacrQG_DATA.DBL', 'MP_THOMPSON_QRacrQS_DATA.DBL',
                  'MP_THOMPSON_freezeH2O_DATA.DBL', 'MP_THOMPSON_QIautQS_DATA.DBL')),
}


def run_tables(build, out, mpi, ranks, destination, ready, kind='tempo_v3'):
    utility, required_objects, names = UTILITIES[kind]
    build = build.resolve(strict=True)
    audit = json.loads((build / 'flag-audit.json').read_text())
    if not audit['all_ok'] or audit['core'] != 'atmosphere' or audit['recipe'] not in ('gnu-strict', 'gnu-o3'):
        raise ValueError('audited atmosphere build required')
    objects = {r['object']: r for r in audit['objects']}
    for name in required_objects:
        if not objects.get('src/core_atmosphere/utils/' + name, {}).get('flags_ok'):
            raise ValueError('native table utility object not audited: ' + name)
    manifest = json.loads((build / 'MANIFEST.json').read_text())
    if manifest.get('reference', 'primary') != 'primary':
        raise ValueError('primary table publication requires the primary fork build')
    if manifest['flags_sha256'] != sha(build / 'flag-audit.json'):
        raise ValueError('audit differs from published build receipt')
    if ranks < 1:
        raise ValueError('positive rank count required')
    binary = build / utility
    if manifest['utility_sha256'][utility] != sha(binary):
        raise ValueError('native table utility differs from published build receipt')
    out.mkdir(parents=True, exist_ok=False)
    argv = [str(mpi.resolve(strict=True)), '-np', str(ranks), str(binary)]
    with (out / 'native.stdout.log').open('w') as log:
        status = subprocess.run(argv, cwd=out, stdout=log, stderr=subprocess.STDOUT,
                                env=dict(os.environ, OMP_NUM_THREADS='1')).returncode
    outputs = {name: sha(out / name) for name in names
               if (out / name).is_file() and (out / name).stat().st_size > 0}
    receipt = dict(schema='mpas.rrfs-'+('tempo' if kind == 'tempo_v3' else kind)+'-native-tables/v1',
                   kind=kind, pin_set_hash=manifest['pin_set_hash'],
                   recipe=audit['recipe'], binary_sha256=sha(binary),
                   audit_sha256=sha(build / 'flag-audit.json'),
                   toolchain_sha256=sha(build / 'toolchain.txt'), argv=argv, ranks=ranks,
                   cpu_affinity=sorted(os.sched_getaffinity(0)), host=os.uname().nodename, outputs=outputs,
                   exit_status=status, completed=status == 0 and len(outputs) == len(names))
    (out / 'MANIFEST.json').write_text(json.dumps(receipt, indent=2) + '\n')
    if not receipt['completed']:
        raise RuntimeError('native table build failed; see ' + str(out / 'MANIFEST.json'))
    if destination is None:
        return receipt
    return publish(out, destination, ['MANIFEST.json', 'native.stdout.log', *names], ready,
                   f'Primary fork {kind} tables rebuilt by {audit["recipe"]} single-precision native utility; '
                   'manifest records toolchain, pin set, utility/object audit and raw table hashes; '
                   'link the recorded original table basenames for atmosphere runs')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('build', 'out', 'mpi'):
        p.add_argument('--' + key, type=Path, required=True)
    for key in ('destination', 'ready'):
        p.add_argument('--' + key, type=Path)
    p.add_argument('--kind', choices=UTILITIES, default='tempo_v3')
    p.add_argument('--stage-only', action='store_true',
                   help='retain private products for authenticated transfer to the archive host')
    p.add_argument('--ranks', type=int, required=True)
    a = p.parse_args()
    if not a.stage_only and (a.destination is None or a.ready is None):
        p.error('--destination and --ready are required unless --stage-only is selected')
    if a.stage_only and (a.destination is not None or a.ready is not None):
        p.error('--stage-only cannot publish a cache')
    print(json.dumps(run_tables(a.build, a.out, a.mpi, a.ranks,
                               a.destination, a.ready, a.kind)))
