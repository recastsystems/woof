#!/usr/bin/env python3
"""Publish an audited RRFS executable using f0's immutable cache publisher."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from publish_cache import publish, sha


def publish_build(build, destination, core, recipe):
    audit = json.loads((build / 'flag-audit.json').read_text())
    if not audit['all_ok'] or audit['core'] != core or audit['recipe'] != recipe:
        raise ValueError('audit does not prove this build')
    binary = build / (core + '_model')
    pins = json.loads((build / 'SOURCE-PINS.json').read_text())
    if pins.get('diagnostic') or pins.get('source_correction'):
        raise ValueError('diagnostic binaries cannot be published as reference builds')
    pin_set = {key: pins[key] for key in ('fork', 'workflow', 'submodules', 'mpas_data', 'precision', 'musica')}
    pin_hash = hashlib.sha256(json.dumps(pin_set, sort_keys=True).encode()).hexdigest()
    receipt = dict(schema='mpas.rrfs-native-build/v1', pin_set_hash=pin_hash,
                   pin_set=pin_set, core=core, recipe=recipe,
                   reference=pins.get('reference', 'primary'),
                   inactive_submodules=pins.get('inactive_submodules', {}),
                   binary_sha256=sha(binary), toolchain_sha256=sha(build / 'toolchain.txt'),
                   flags_sha256=sha(build / 'flag-audit.json'),
                   source_diff_sha256=sha(build / 'build-source.diff'),
                   objects=len(audit['objects']), groups=audit['groups'], forecast_identity='pending')
    if core == 'atmosphere':
        receipt['utility_sha256'] = {name: sha(build / name) for name in ('build_tables', 'build_tables_tempo_v3')}
    key = hashlib.sha256(json.dumps(receipt, sort_keys=True).encode()).hexdigest()
    (build / 'MANIFEST.json').write_text(json.dumps(receipt, indent=2) + '\n')
    include = [binary.name, 'MANIFEST.json', 'SOURCE-PINS.json', 'toolchain.txt', 'build-source.diff',
               'executable.sha256', 'flag-audit.json', 'compile-commands.jsonl', 'build.log',
               'namelist.*', 'streams.*', 'stream_list.*']
    if core == 'atmosphere':
        include += ['build_tables', 'build_tables_tempo_v3', 'src/core_atmosphere/physics/physics_wrf/files/*',
                    'src/core_atmosphere/physics/physics_noahmp/parameters/NoahmpTable.TBL',
                    'src/core_atmosphere/physics/physics_noaa/TEMPO/tables/*']
    return publish(build, destination / 'native' / key, include, destination / 'READY.md',
                   f'RRFS {pins.get("reference", "primary")} {pins["fork"][:8]} {core} {recipe}, single; all {len(audit["objects"])} objects audited; '
                   'run with MPICH from toolchain.txt, taskset -c 8-23; forecast identity pending')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('build', type=Path)
    p.add_argument('destination', type=Path)
    p.add_argument('--core', choices=('atmosphere', 'init_atmosphere'), required=True)
    p.add_argument('--recipe', choices=('gnu-strict', 'gnu-o3'), required=True)
    a = p.parse_args()
    print(json.dumps(publish_build(a.build.resolve(), a.destination.resolve(), a.core, a.recipe)))
