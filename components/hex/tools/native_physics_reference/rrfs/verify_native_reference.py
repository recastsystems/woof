#!/usr/bin/env python3
"""Authenticate pinned native binaries, captured compiler audits and native tables.

This consumes existing immutable publications. It neither rebuilds objects nor
replays physics, and success does not establish forecast or lane acceptance.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from audit_flags import STRICT
from verify_diagnostic_contract import publication
from transport_inputs import sha


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fingerprint(command):
    return json.dumps(command, sort_keys=True)


def compiled_object(command):
    args = command['argv']
    sources = [a for a in args if Path(a).suffix.lower() in ('.f', '.f90', '.c', '.cc', '.cpp')]
    if len(sources) != 1:
        return None
    output = args[args.index('-o') + 1] if '-o' in args else Path(sources[0]).stem + '.o'
    # Historical compiler paths are metadata; transport can relocate a sealed
    # publication. Do not reinterpret them through today's filesystem links.
    return os.path.normpath(str(Path(command['cwd']) / output))


def verify_commands(audit, manifest, records):
    require(audit['all_ok'] is True and not audit['errors'] and not audit.get('diagnostic'),
            'requires complete pristine compiler audit')
    require(audit['recipe'] == manifest['recipe'] and audit['core'] == manifest['core'],
            'audit recipe/core differs')
    objects = {r['object']: r for r in audit['objects']}
    require(objects and len(objects) == len(audit['objects']) == manifest['objects'],
            'object inventory empty, duplicated or incomplete')
    roots = set()
    for name, row in objects.items():
        if not row.get('archive_copy_origins') and len(row['commands']) == 1:
            target = Path(compiled_object(row['commands'][0]))
            parts = Path(name).parts
            require(target.parts[-len(parts):] == parts, 'compiler output path differs: ' + name)
            roots.add(str(target.parents[len(parts) - 1]))
    require(len(roots) == 1, 'compiler build roots differ or are absent')
    source = roots.pop()
    last = {}
    for record in records:
        target = compiled_object(record)
        if target is not None:
            last[target] = record
    strict = manifest['recipe'] == 'gnu-strict'
    required = ('-O3',) + (STRICT if strict else ())
    forbidden = ('-fdefault-real-8', '-fdefault-double-8', '-ffast-math', '-Ofast', '-fopenmp')
    if strict:
        forbidden += ('-ffp-contract=fast', '-ffp-contract=on', '-ftree-vectorize',
                      '-funsafe-math-optimizations')
    for name, row in objects.items():
        path = Path(name)
        require(not path.is_absolute() and '..' not in path.parts and row['flags_ok'] is True,
                'unsafe or rejected object: ' + name)
        origins = row.get('archive_copy_origins', [])
        commands = row['commands']
        require(commands and len(commands) == (len(origins) if origins else 1),
                'object compiler provenance incomplete: ' + name)
        targets = origins or [name]
        for origin, command in zip(targets, commands):
            require(origin in objects and objects[origin]['sha256'] == row['sha256'],
                    'archive object copy differs: ' + name)
            target = os.path.normpath(str(Path(source) / origin))
            require(target in last and fingerprint(last[target]) == fingerprint(command),
                    'object command is not final captured invocation: ' + name)
            args = command['argv']
            require(command['status'] == 0 and all(flag in args for flag in required)
                    and not any(flag in args for flag in forbidden),
                    'compiler recipe flags differ: ' + name)
            require(command['compiler'] not in ('mpif90', 'gfortran') or '-DSINGLE_PRECISION' in args,
                    'Fortran single precision missing: ' + name)
            require(not any(a.startswith(('-finit-real=', '-finit-integer=', '-fcheck=')) for a in args),
                    'diagnostic flags in pristine authority: ' + name)
    groups = {group: sum('/' + group + '/' in name for name in objects)
              for group in audit['groups']}
    require(groups == audit['groups'] == manifest['groups'] and all(groups.values()),
            'physics group coverage differs')
    if manifest['core'] == 'atmosphere':
        required_groups = {'physics_mmm', 'physics_wrf', 'physics_noahmp'}
        required_groups.update('physics_noaa/' + n for n in
                              ('TEMPO', 'MYNN-EDMF', 'MYNN-SFC', 'RUCLSM', 'GFL', 'SMOKE', 'UGWP'))
        require(set(groups) == required_groups, 'native physics group missing')
    return dict(objects=len(objects), groups=groups, historical_compiler_root=source,
                reaudited_captured_commands=True,
                raw_objects_rehashed=False)


def verify(document):
    pins = json.loads(Path(document).read_text())['rrfs']
    pin_hash = hashlib.sha256(json.dumps(pins['primary'], sort_keys=True).encode()).hexdigest()
    require(pin_hash == pins['pin_set_hash'] and pins['primary']['precision'] == 'single'
            and pins['primary']['musica'] is False, 'primary pin set differs')
    required_builds = {'primary/' + recipe + '/' + core for recipe in ('gnu-strict', 'gnu-o3')
                       for core in ('atmosphere', 'init_atmosphere')}
    required_builds.add('secondary/gnu-strict/atmosphere')
    require(set(pins['native_builds']) == required_builds, 'native build matrix incomplete or ambiguous')
    builds = {}
    for key, entry in pins['native_builds'].items():
        directory = Path(entry['path'])
        authority = publication(directory / 'PUBLICATION.json', entry['publication_sha256'])
        members = authority['files']
        require(all(name in members for name in ('MANIFEST.json', 'SOURCE-PINS.json',
                    'flag-audit.json', 'compile-commands.jsonl', 'toolchain.txt', 'build-source.diff')),
                'native build authority missing members')
        manifest = json.loads((directory / 'MANIFEST.json').read_text())
        source_pins = json.loads((directory / 'SOURCE-PINS.json').read_text())
        require(manifest['schema'] == 'mpas.rrfs-native-build/v1'
                and key == '/'.join(manifest[k] for k in ('reference', 'recipe', 'core')),
                'build selection differs from manifest')
        require(not source_pins.get('diagnostic') and all(
                    source_pins.get(k) == v for k, v in manifest['pin_set'].items()),
                'native build source pins differ or are diagnostic')
        actual_pin = hashlib.sha256(json.dumps(manifest['pin_set'], sort_keys=True).encode()).hexdigest()
        require(actual_pin == manifest['pin_set_hash'], 'native build pin digest differs')
        require(manifest['pin_set']['precision'] == 'single'
                and manifest['pin_set']['musica'] is False, 'native build precision/chemistry differs')
        if manifest['reference'] == 'primary':
            require(manifest['pin_set'] == pins['primary'], 'primary build source differs')
        else:
            require(manifest['pin_set']['fork'] == pins['secondary_fork'], 'secondary fork differs')
        bindings = {'binary_sha256': manifest['core'] + '_model',
                    'toolchain_sha256': 'toolchain.txt', 'flags_sha256': 'flag-audit.json'}
        for field, name in bindings.items():
            require(members.get(name) == entry[field] == manifest[field],
                    'native build member binding differs: ' + field)
        require(members['build-source.diff'] == manifest['source_diff_sha256']
                and manifest.get('utility_sha256', {}) == entry['utility_sha256']
                and all(members.get(n) == v for n, v in entry['utility_sha256'].items()),
                'source diff or table utility binding differs')
        audit = json.loads((directory / 'flag-audit.json').read_text())
        records = [json.loads(line) for line in (directory / 'compile-commands.jsonl').read_text().splitlines()]
        checked = verify_commands(audit, manifest, records)
        builds[key] = dict(path=str(directory), publication_sha256=entry['publication_sha256'],
                           binary_sha256=manifest['binary_sha256'], **checked)
    tables = {}
    coverage = set()
    utility_by_schema = {'mpas.rrfs-tempo-native-tables/v1': 'build_tables_tempo_v3',
                         'mpas.rrfs-thompson-native-tables/v1': 'build_tables'}
    for name, entry in pins['native_tables'].items():
        directory = Path(entry['path'])
        authority = publication(directory / 'PUBLICATION.json', entry['publication_sha256'])
        require('MANIFEST.json' in authority['files'], 'native table manifest not published')
        manifest = json.loads((directory / 'MANIFEST.json').read_text())
        require(manifest['schema'] in utility_by_schema and manifest['completed'] is True
                and manifest['exit_status'] == 0 and manifest['ranks'] > 0,
                'native table execution incomplete')
        build = pins['native_builds'].get('primary/' + manifest['recipe'] + '/atmosphere')
        require(build is not None and manifest['pin_set_hash'] == pin_hash
                and manifest['binary_sha256'] == build['utility_sha256'][utility_by_schema[manifest['schema']]]
                and manifest['audit_sha256'] == build['flags_sha256']
                and manifest['toolchain_sha256'] == build['toolchain_sha256'],
                'native table recipe/build authority differs')
        require(manifest['outputs'] and manifest['outputs'] == entry['table_sha256']
                and all(authority['files'].get(n) == v for n, v in manifest['outputs'].items()),
                'native table output binding differs')
        identity = (manifest['schema'], manifest['recipe'])
        require(identity not in coverage, 'duplicate native table recipe')
        coverage.add(identity)
        tables[name] = dict(publication_sha256=entry['publication_sha256'],
                           files=len(manifest['outputs']), ranks=manifest['ranks'], host=manifest['host'])
    require(coverage == {(schema, recipe) for schema in utility_by_schema for recipe in ('gnu-strict', 'gnu-o3')},
            'native table recipe matrix incomplete')
    return dict(schema='mpas.rrfs-native-reference-authority-verification/v1', verified=True,
                pins_document_sha256=sha(document), pin_set_hash=pin_hash, builds=builds, tables=tables,
                objects=sum(r['objects'] for r in builds.values()),
                table_files=sum(r['files'] for r in tables.values()), weather_execution=False,
                physics_replay=False, forecast_identity=False, lane_acceptance=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pins-document', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.pins_document)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
