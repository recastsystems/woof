"""Verified, relocatable HIP code objects for private prebuilt wheel kits.

The canonical and translated source hashes, ISA, ordered options, compiler
binary/header digest and every external include-tree digest must all match.
Changing a header or a numeric option falls back to normal compilation.
"""
import functools
import hashlib
import json
import os
from pathlib import Path
import zipfile

SCHEMA = 'gpuwm.hip-prebuilt/v1'


def binding(source_sha256, translated_sha256, arch, options, headers, translator):
    # Absolute include roots are installation locations. The ordered content
    # digests in headers bind their complete contents independently of paths.
    normalized = []
    separate = False
    for option in options:
        if separate:
            normalized.append('@include-root'); separate = False
        elif option in ('-I', '--include-path'):
            normalized.append(option); separate = True
        elif option.startswith('-I') or option.startswith('--include-path='):
            normalized.append('-I@include-root')
        else:
            normalized.append(option)
    return dict(schema=SCHEMA, source_sha256=source_sha256,
        translated_sha256=translated_sha256, arch=arch, options=normalized,
        headers=headers, translator=translator)


def key(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@functools.lru_cache(maxsize=4)
def index(path):
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read('MANIFEST.json'))
    if manifest['schema'] != SCHEMA:
        raise ValueError('Unsupported HIP prebuilt manifest: '+path)
    return manifest['objects']


def lookup(value):
    identity = key(value)
    for path in filter(None, os.environ.get('WOOF_HIP_PREBUILT', '').split(os.pathsep)):
        record = index(path).get(identity)
        if record is None:
            continue
        if record['binding'] != value:
            raise ValueError('HIP prebuilt binding mismatch: '+path)
        member = record['image']
        if member != 'images/'+record['receipt']['image_sha256']+'.hsaco':
            raise ValueError('Invalid HIP prebuilt image name: '+path)
        with zipfile.ZipFile(path) as archive:
            blob = archive.read(member)
        if blob[:4] != b'\x7fELF' or hashlib.sha256(blob).hexdigest() != record['receipt']['image_sha256']:
            raise ValueError('Corrupt HIP prebuilt image: '+path)
        return blob, dict(record['receipt'], prebuilt_origin=dict(
            archive=str(Path(path).resolve()), binding_sha256=identity,
            build_cache_key=record['receipt']['key']))
    return None
