#!/usr/bin/env python3
"""Verify approved reference provenance; this does not replay native physics."""
import argparse
import json
from pathlib import Path
from ruled_reference import validate_authority, validate_init_case, validate_init_configuration
from verify_diagnostic_contract import publication
from verify_native_reference import verify_commands


def verify_build(entry):
    directory = Path(entry['path'])
    members = publication(directory / 'PUBLICATION.json', entry['publication_sha256'])['files']
    manifest = json.loads((directory / 'MANIFEST.json').read_text())
    value = validate_authority(manifest['source_correction'], manifest['binary_sha256'])
    pins = json.loads((directory / 'SOURCE-PINS.json').read_text())
    if pins.get('diagnostic') or pins.get('source_correction') != value:
        raise ValueError('approved build source authority differs')
    core = 'init_atmosphere' if value['kind'] == 'init' else 'atmosphere'
    if manifest['core'] != core or manifest['recipe'] != 'gnu-strict':
        raise ValueError('approved build core/recipe differs')
    bindings = {core + '_model':manifest['binary_sha256'],
                'flag-audit.json':manifest['flags_sha256'],
                'toolchain.txt':manifest['toolchain_sha256'],
                'APPROVED.patch':value['patch_sha256'],
                'DREW-RULING.json':value['ruling_sha256']}
    if any(members.get(name) != digest for name, digest in bindings.items()):
        raise ValueError('approved build member binding differs')
    records = [json.loads(line) for line in (directory / 'compile-commands.jsonl').read_text().splitlines()]
    audit = json.loads((directory / 'flag-audit.json').read_text())
    commands = verify_commands(audit, manifest, records)
    return dict(kind=value['kind'], authority=value, publication_sha256=entry['publication_sha256'],
                compiler_audit=commands)


def verify_case(entry):
    directory = Path(entry['path'])
    members = publication(directory / 'PUBLICATION.json', entry['publication_sha256'])['files']
    manifest = json.loads((directory / 'MANIFEST.json').read_text())
    value = validate_init_case(manifest)
    if not value or not manifest['completed'] or manifest['exit_status']:
        raise ValueError('approved canonical case did not complete')
    validate_init_configuration((directory / 'namelist.init_atmosphere').read_text())
    if (not manifest['outputs'] or any(members.get(name) != digest
            for group in ('outputs', 'configuration') for name, digest in manifest[group].items())):
        raise ValueError('approved case output/configuration binding differs')
    if value['label'] != manifest['reference_label']:
        raise ValueError('approved case label differs')
    return dict(path=str(directory), publication_sha256=entry['publication_sha256'],
                authority=value, case=manifest['row']['variables']['init_case'],
                ranks=manifest['ranks'], host=manifest['host'], outputs=manifest['outputs'])


def verify(request):
    builds = [verify_build(row) for row in request.get('builds', [])]
    cases = [verify_case(row) for row in request.get('cases', [])]
    if not builds and not cases:
        raise ValueError('explicit approved reference products required')
    return dict(schema='mpas.rrfs-approved-reference-provenance-verification/v1',
                builds=builds, cases=cases, verified=True, weather_execution=False,
                module_replay=False, forecast_identity=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('request', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(json.loads(args.request.read_text())), indent=2))
