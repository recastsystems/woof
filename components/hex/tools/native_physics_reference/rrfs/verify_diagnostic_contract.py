#!/usr/bin/env python3
"""Verify a sealed diagnostic ingress contract, without executing native weather.

This checks unchanged bytes and publication membership only. It proves no
physics replay, affected-word set, source-defect exclusion or forecast identity.
"""
import argparse
import json
from pathlib import Path

from transport_inputs import sha


def publication(path, expected):
    path = Path(path)
    if sha(path) != expected:
        raise ValueError('publication digest changed: ' + str(path))
    data = json.loads(path.read_text())
    root = path.parent.resolve(strict=True)
    for name, digest in data['files'].items():
        member = (root / name).resolve(strict=True)
        if not member.is_relative_to(root) or name == 'PUBLICATION.json':
            raise ValueError('publication member escapes its directory')
        if sha(member) != digest:
            raise ValueError('published member changed: ' + name)
    return data


def verify(path, expected, runtime=None, contract_id=None, ranks=None):
    path = Path(path)
    original = publication(path, expected)
    if 'INDEX.json' not in original['files']:
        raise ValueError('contract index is not a published member')
    index = json.loads((path.parent / 'INDEX.json').read_text())
    if index['schema'] != 'mpas.rrfs-original-measured-gfl-checked-ingress-contract/v1':
        raise ValueError('unsupported diagnostic ingress contract')
    checked = Path(index['checked_build'])
    checked_pub = publication(checked / 'PUBLICATION.json', index['checked_publication_sha256'])
    for member in ('DIAGNOSTIC-MANIFEST.json', 'atmosphere_model'):
        if member not in checked_pub['files']:
            raise ValueError('checked build is missing a published member')
    diagnostic = json.loads((checked / 'DIAGNOSTIC-MANIFEST.json').read_text())
    if (diagnostic['reference_binary'] or not diagnostic['native_source_unchanged']
            or diagnostic['precision'] != 'single'
            or diagnostic['binary_sha256'] != index['checked_binary_sha256']
            or checked_pub['files']['atmosphere_model'] != index['checked_binary_sha256']):
        raise ValueError('requires labelled source-unchanged checked diagnostic')
    if not {'-fcheck=all', '-finit-real=snan'} <= set(diagnostic['checked_fortran_flags']):
        raise ValueError('checked diagnostic flags are absent')
    shared = {}
    published = {}
    ids = set()
    for row in index['contracts']:
        if row['id'] in ids:
            raise ValueError('duplicate diagnostic contract ID')
        ids.add(row['id'])
        if original['files'].get(row['generated_namelist']) != row['namelist_sha256']:
            raise ValueError('generated namelist is not a matching published member')
        partition = row['mpi2_partition']
        if original['files'].get(partition['path']) != partition['sha256']:
            raise ValueError('MPI2 partition is not a matching published member')
        for item in [row['original_base_namelist'], *row['ingress'].values()]:
            body = Path(item['path']).resolve(strict=True)
            authority = Path(item['publication'])
            key = (str(authority), item['publication_sha256'])
            if key not in published:
                # Authenticate the original catalog. Hash only required bodies;
                # its other files are outside this selected ingress contract.
                if sha(authority) != item['publication_sha256']:
                    raise ValueError('shared publication digest changed')
                published[key] = json.loads(authority.read_text())
            if not body.is_relative_to(authority.parent.resolve(strict=True)):
                raise ValueError('shared ingress escapes publication')
            member = str(body.relative_to(authority.parent.resolve(strict=True)))
            if published[key]['files'].get(member) != item['sha256']:
                raise ValueError('shared ingress is not a published member')
            if str(body) not in shared:
                if body.stat().st_size != item['bytes'] or sha(body) != item['sha256']:
                    raise ValueError('shared ingress bytes changed')
                shared[str(body)] = item['sha256']
            elif shared[str(body)] != item['sha256']:
                raise ValueError('conflicting shared ingress digest')
    checked_runtime = None
    if runtime is not None:
        if contract_id not in ids or ranks not in (1, 2):
            raise ValueError('runtime requires an existing contract and explicit rank count')
        row = next(row for row in index['contracts'] if row['id'] == contract_id)
        required = {name: item['sha256'] for name, item in row['ingress'].items()}
        required['namelist.atmosphere'] = row['namelist_sha256']
        if ranks == 2:
            required[row['mpi2_partition']['path']] = row['mpi2_partition']['sha256']
        root = Path(runtime).resolve(strict=True)
        for name, digest in required.items():
            target = root / name
            if not target.is_file() or sha(target) != digest:
                raise ValueError('runtime ingress differs from measured contract: ' + name)
        checked_runtime = dict(contract=contract_id, ranks=ranks, files=len(required))
    elif contract_id is not None or ranks is not None:
        raise ValueError('runtime options require a runtime directory')
    return dict(schema='mpas.rrfs-diagnostic-ingress-byte-verification/v1',
        publication_sha256=expected, contracts=len(ids), shared_files=len(shared),
        checked_binary_sha256=index['checked_binary_sha256'], runtime=checked_runtime,
        original_runner_matches_recorded_commit=index['recorded_runner_matches_commit'],
        verified=True, weather_execution=False, affected_word_set_proven=False,
        native_defect_policy_complete=False, accepted_exclusions=False, hex_identity=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('publication', type=Path)
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--runtime', type=Path)
    parser.add_argument('--contract')
    parser.add_argument('--ranks', type=int, choices=(1, 2))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.publication, args.expected_sha256, args.runtime, args.contract, args.ranks)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
