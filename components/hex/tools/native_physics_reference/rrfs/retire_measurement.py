#!/usr/bin/env python3
"""Hash and retire closed measurement outputs while preserving explicit windows.

This is disposable scratch orchestration, never reference archive retirement.
The source-version runner holds the same lock for native-exit/output receipts.
"""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import time


def retire(root, request, receipt):
    root = root.resolve(strict=True)
    labels, keep = request['runs'], set(request['keep'])
    if (not labels or not keep or len(set(labels)) != len(labels)
            or any(Path(name).name != name for name in [*labels, *keep])
            or '/' in request['pattern'] or '..' in request['pattern']):
        raise ValueError('explicit safe measurement runs and retained filenames required')
    retired = {}
    while True:
        finished = []
        for label in labels:
            directory = root/label
            if not directory.is_dir():
                finished.append(False)
                continue
            with (directory/'native-output-receipt.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                done = (directory/'NATIVE-EXIT.json').exists()
                finished.append(done)
                if done:
                    continue
                # In the native time-per-file stream, the newest two timestamps
                # are protected; earlier outputs have already been closed.
                for path in sorted(directory.glob(request['pattern']))[:-2]:
                    if path.name in keep:
                        continue
                    if path.is_symlink() or not path.is_file():
                        raise ValueError('measurement retirement requires own regular outputs')
                    with path.open('rb') as stream:
                        digest = hashlib.file_digest(stream,'sha256').hexdigest()
                    retired[str(path.relative_to(root))] = dict(
                        sha256=digest, bytes=path.stat().st_size)
                    result = dict(schema='mpas.rrfs-dry-disposable-output/v1',
                        request=request, retired=retired, completed=False)
                    receipt.write_text(json.dumps(result,indent=2)+'\n')
                    path.unlink()
        if all(finished):
            result = dict(schema='mpas.rrfs-dry-disposable-output/v1',
                request=request, retired=retired, completed=True)
            receipt.write_text(json.dumps(result,indent=2)+'\n')
            return result
        time.sleep(5)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','request','receipt'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    result=retire(args.root,json.loads(args.request.read_text()),args.receipt)
    print(json.dumps(dict(completed=result['completed'],retired_files=len(result['retired']))))
