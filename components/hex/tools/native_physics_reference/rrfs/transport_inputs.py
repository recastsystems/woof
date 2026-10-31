#!/usr/bin/env python3
"""Transport unchanged native inputs through the remote runner's bounded work mirror.

Chunks are raw bytes, never decoded weather fields. Both ends verify the
published source digest and every chunk before atomically exposing a file.
Destinations remain under the lane's own work directory.
"""
import argparse
import hashlib
import json
from pathlib import Path
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def pack(sources, destination, chunk_bytes=128 * 1024**2):
    from rrfs.ruled_reference import input_authorities
    if not 0 < chunk_bytes < 200 * 1024**2:
        raise ValueError('chunks must fit the remote work mirror')
    destination.mkdir(parents=True, exist_ok=False)
    rows = {}
    for source, expected in sources.items():
        source = Path(source)
        if sha(source) != expected:
            raise ValueError('source differs from its published digest: ' + str(source))
        directory = destination / expected
        directory.mkdir(exist_ok=True)
        chunks = []
        with source.open('rb') as stream:
            number = 0
            while data := stream.read(chunk_bytes):
                path = directory / str(number)
                if path.exists():
                    if sha(path) != hashlib.sha256(data).hexdigest():
                        raise ValueError('different chunk for the same content address')
                else:
                    path.write_bytes(data)
                chunks.append(dict(file=str(path.relative_to(destination)), sha256=sha(path), bytes=len(data)))
                number += 1
        rows[str(source)] = dict(sha256=expected, bytes=source.stat().st_size, chunks=chunks,
                                init_derivative_authorities=input_authorities({'files': {source.name: str(source)}}))
    manifest = dict(schema='mpas.rrfs-raw-byte-transport/v1', files=rows)
    (destination / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def unpack(source, destination):
    from rrfs.ruled_reference import validate_init_authorities
    source = source.resolve(strict=True)
    manifest = json.loads((source / 'MANIFEST.json').read_text())
    destination.mkdir(parents=True, exist_ok=True)
    mapping = {}
    sidecar = destination/'INPUT-AUTHORITIES.json'
    authorities = json.loads(sidecar.read_text()) if sidecar.exists() else dict(
        schema='mpas.rrfs-transported-input-authority/v1', files={})
    if authorities.get('schema') != 'mpas.rrfs-transported-input-authority/v1':
        raise ValueError('transported input authority schema differs')
    for original, row in manifest['files'].items():
        digest = row['sha256']
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('invalid content address')
        inherited = validate_init_authorities(row.get('init_derivative_authorities', []))
        previous = authorities['files'].get(digest)
        if previous and previous['sha256'] != digest:
            raise ValueError('existing transported authority content address differs')
        found = {v['binary_sha256']: v for v in validate_init_authorities(
            previous['init_derivative_authorities'] if previous else [])}
        found.update({v['binary_sha256']: v for v in inherited})
        target = destination / digest
        if target.exists():
            if target.stat().st_size != row['bytes'] or sha(target) != digest:
                raise ValueError('existing transported file differs from receipt')
        else:
            temporary = destination / (digest + '.part')
            with temporary.open('wb') as output:
                for chunk in row['chunks']:
                    path = (source / chunk['file']).resolve(strict=True)
                    if not path.is_relative_to(source):
                        raise ValueError('unsafe transport chunk path')
                    if path.stat().st_size != chunk['bytes'] or sha(path) != chunk['sha256']:
                        raise ValueError('transport chunk differs from its receipt')
                    with path.open('rb') as stream:
                        while data := stream.read(1024**2):
                            output.write(data)
                output.flush()
                os.fsync(output.fileno())
            if temporary.stat().st_size != row['bytes'] or sha(temporary) != digest:
                temporary.unlink()
                raise ValueError('assembled native input differs from publication')
            temporary.rename(target)
        mapping[original] = str(target.resolve())
        authorities['files'][digest] = dict(sha256=digest,
            init_derivative_authorities=[found[k] for k in sorted(found)])
    temporary = destination/'INPUT-AUTHORITIES.json.part'
    temporary.write_text(json.dumps(authorities, indent=2)+'\n')
    temporary.replace(sidecar)
    return mapping


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['pack', 'unpack'])
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    result = (pack(json.loads(args.source.read_text()), args.destination) if args.mode == 'pack'
              else unpack(args.source, args.destination))
    print(json.dumps(result, indent=2))
