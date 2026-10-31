#!/usr/bin/env python3
"""Authenticate a pristine native case transfer on the permanent archive host."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

spec=importlib.util.spec_from_file_location('rrfs_received_case',Path(__file__).with_name('cache_cases.py'))
cache=importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache)


def retire_transferred_outputs(source, destination, owned_root, publication):
    """Retire only received outputs authenticated against the sealed archive."""
    owned_root = owned_root.resolve(strict=True)
    if not source.is_relative_to(owned_root) or source == owned_root or source.is_symlink():
        raise ValueError('received output retirement requires owned transfer scratch')
    if cache.sha(destination / 'PUBLICATION.json') != publication['publication_sha256']:
        raise ValueError('received output archive publication changed')
    manifest = json.loads((source / 'MANIFEST.json').read_text())
    sealed = json.loads((destination / 'PUBLICATION.json').read_text())
    rows = []
    for name, digest in manifest['outputs'].items():
        path = source / name
        if (path.is_symlink() or sealed['files'].get(name) != digest
                or cache.sha(destination / name) != digest or cache.sha(path) != digest):
            raise ValueError('received output differs from immutable archive')
        rows.append(dict(file=name, sha256=digest, bytes=path.stat().st_size))
    # Authenticate the complete inventory before deleting any member.
    for row in rows:
        (source / row['file']).unlink()
    receipt = dict(publication=publication, retired=rows)
    (source / 'TRANSFER-OUTPUT-RETIREMENT.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


def receive(source, destination, ready, build, expected, retire_output_root=None):
    source=source.resolve(strict=True)
    build=build.resolve(strict=True)
    manifest=source/'MANIFEST.json'
    if cache.sha(manifest)!=expected['manifest_sha256']:
        raise ValueError('received native case manifest changed')
    case=json.loads(manifest.read_text())
    audit=json.loads((build/'flag-audit.json').read_text())
    pins=json.loads((build/'SOURCE-PINS.json').read_text())
    if (pins.get('diagnostic') or case['source_pins']!=pins
            or not audit['all_ok'] or audit['recipe']!='gnu-strict'
            or audit['core']!='init_atmosphere'
            or case['binary_sha256']!=cache.sha(build/'init_atmosphere_model')
            or case['audit_sha256']!=cache.sha(build/'flag-audit.json')):
        raise ValueError('received native case requires its audited strict initializer authority')
    if any(p.is_symlink() or not p.is_file() for p in source.iterdir()):
        raise ValueError('received native case cannot retain scratch links or directories')
    publication = cache.publish_case(source,destination,ready,expected['description'])
    if retire_output_root is not None:
        retirement = retire_transferred_outputs(source, destination, retire_output_root, publication)
        return dict(publication, retired_transfer_bytes=sum(row['bytes'] for row in retirement['retired']))
    return publication


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','destination','ready','build'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--retire-output-root',type=Path)
    args=parser.parse_args()
    print(json.dumps(receive(args.source,args.destination,args.ready,args.build,json.load(sys.stdin),args.retire_output_root)))
