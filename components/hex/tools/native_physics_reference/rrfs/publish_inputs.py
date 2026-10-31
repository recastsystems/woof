#!/usr/bin/env python3
"""Seal verified public raw inputs only after copied bytes match their witnesses."""
import argparse
import fcntl
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from publish_cache import publish, sha


def publish_inputs(source, destination, ready, staging_ready):
    manifest = json.loads((source / 'MANIFEST.json').read_text())
    rows = manifest['files']
    if not manifest['completed'] or not rows or len({r['name'] for r in rows}) != len(rows):
        raise ValueError('complete unique verified input inventory required')
    for row in rows:
        if Path(row['name']).name != row['name'] or not row['verified']:
            raise ValueError('unsafe or unverified public input')
        if json.loads((source / (row['name'] + '.VERIFIED.json')).read_text()) != row:
            raise ValueError('input checksum witness differs from inventory')
    if destination.exists():
        raise ValueError('refusing existing immutable input cache')
    stage = destination.with_name(destination.name + '.checksum-staging')
    selection = ['MANIFEST.json', *[r['name'] for r in rows], *[r['name'] + '.VERIFIED.json' for r in rows]]
    description = f'{len(rows)} public raw inputs, complete S3 whole-object MD5/size and SHA256 witnesses; native ungrib decodes these unchanged bytes'
    result = publish(source, stage, selection, staging_ready, description)
    copied = json.loads((stage / 'PUBLICATION.json').read_text())
    for row in rows:
        if copied['files'][row['name']] != row['sha256'] or (stage / row['name']).stat().st_size != row['bytes']:
            raise ValueError('copied public input differs from verified S3 bytes: ' + row['name'])
    stage.rename(destination)
    with ready.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.write(f'{description}: `{destination}`; PUBLICATION.json sha256 `{result["publication_sha256"]}`.\n')
    result['path'] = str(destination)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('source', 'destination', 'ready', 'staging-ready', 'receipt'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    result = publish_inputs(args.source.resolve(), args.destination, args.ready, args.staging_ready)
    args.receipt.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
