#!/usr/bin/env python3
"""Seal native compressed geography images on the permanent reference host."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from publish_cache import publish, sha


def receive(source, destination, ready, expected):
    source = source.resolve(strict=True)
    manifest = source / 'MANIFEST.json'
    if sha(manifest) != expected['manifest_sha256']:
        raise ValueError('native geography byte-verification receipt changed')
    receipt = json.loads(manifest.read_text())
    if not receipt['completed'] or receipt['storage'] != 'native-read-only-compressed-filesystem':
        raise ValueError('native geography verification is incomplete')
    archives = {row['sha256']: row for row in receipt['archives']}
    products = {row['source_sha256']: row for row in receipt['products']}
    if (len(archives) != len(receipt['archives']) or len(products) != len(receipt['products'])
            or set(archives) != set(products) or set(archives) != set(expected['archive_paths'])):
        raise ValueError('native geography source/image inventory differs')
    names = ['MANIFEST.json']
    for digest, product in products.items():
        archive = Path(expected['archive_paths'][digest]).resolve(strict=True)
        image = source / product['image']
        if (archive.stat().st_size != archives[digest]['bytes'] or sha(archive) != digest
                or Path(product['image']).name != product['image'] or image.is_symlink()
                or not product['all_regular_members_native_cmp_identical']
                or image.stat().st_size != product['bytes']
                or sha(image) != product['image_sha256']):
            raise ValueError('native geography original archive or verified image changed')
        names.append(product['image'])
    if len(set(names)) != len(names):
        raise ValueError('native geography image names are not unique')
    return publish(source, destination, names, ready, expected['description'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'destination', 'ready'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(receive(args.source, args.destination, args.ready, json.load(sys.stdin))))
