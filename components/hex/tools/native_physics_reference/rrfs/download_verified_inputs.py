#!/usr/bin/env python3
"""Extend f0's downloader with S3 whole-object ETag and size verification.

This copies and hashes public bytes. It neither interprets nor repairs GRIB.
One controller owns the directory lock, including resumable partial files.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from download_inputs import download


def digests(path):
    md5, sha = hashlib.md5(), hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            md5.update(block)
            sha.update(block)
    return md5.hexdigest(), sha.hexdigest()


def verified(row, directory):
    name = row['name']
    if Path(name).name != name or not row['url'].startswith('https://'):
        raise ValueError('simple filename and HTTPS URL required')
    headers = directory / (name + '.verification-headers')
    subprocess.run(['curl', '--silent', '--show-error', '--fail', '--location', '--head',
                    '--retry', '5', '--retry-all-errors', '--connect-timeout', '30', '--max-time', '120',
                    '--output', str(headers), row['url']], check=True)
    text = headers.read_text()
    etags = re.findall(r'^etag:\s*"([0-9a-f]{32})"\s*$', text, re.I | re.M)
    sizes = re.findall(r'^content-length:\s*(\d+)\s*$', text, re.I | re.M)
    if not etags or not sizes:
        raise ValueError('whole-object MD5 ETag and Content-Length required: ' + row['url'])
    etag, size = etags[-1].lower(), int(sizes[-1])
    path = directory / name
    repaired = None
    if path.exists():
        actual_md5, actual_sha = digests(path)
        if path.stat().st_size == size and actual_md5 == etag:
            result = dict(row, bytes=size, sha256=actual_sha, s3_etag=etag, verified=True)
            save_verified(directory, result)
            return result
        repaired = dict(bytes=path.stat().st_size, md5=actual_md5, sha256=actual_sha)
        # Only this controller's explicit input directory is modified. A link's
        # previous target is retained; publication/scratch retention is separate.
        path.unlink()
        partial = directory / (name + '.partial')
        if partial.exists():
            partial.unlink()
    elif (directory / (name + '.partial')).exists():
        # A legacy partial has no single-controller witness. Start this object
        # again, so append races cannot survive with a coincidental valid size.
        (directory / (name + '.partial')).unlink()
    for attempt in range(3):
        try:
            result = download(row, directory)
            break
        except subprocess.CalledProcessError:
            if attempt == 2:
                raise
    actual_md5, actual_sha = digests(path)
    if path.stat().st_size != size or actual_md5 != etag:
        raise ValueError('download differs from public S3 whole-object bytes: ' + name)
    result.update(sha256=actual_sha, s3_etag=etag, verified=True)
    if repaired:
        result['replaced_invalid_input'] = repaired
    save_verified(directory, result)
    return result


def save_verified(directory, result):
    path = directory / (result['name'] + '.VERIFIED.json')
    partial = path.with_name(path.name + '.partial')
    partial.write_text(json.dumps(result, indent=2) + '\n')
    partial.replace(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('index', type=Path)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=4, choices=range(1, 5))
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    rows = json.loads(args.index.read_text())['files']
    if not rows or len({row['name'] for row in rows}) != len(rows):
        raise ValueError('nonempty unique input rows required')
    with (args.directory / '.download.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            results = list(pool.map(lambda row: verified(row, args.directory), rows))
        temporary = args.directory / 'MANIFEST.json.partial'
        temporary.write_text(json.dumps(dict(schema='mpas.rrfs-verified-public-input/v1',
            completed=True, files=results), indent=2) + '\n')
        temporary.replace(args.directory / 'MANIFEST.json')
        print(json.dumps(dict(verified=len(results))), flush=True)
