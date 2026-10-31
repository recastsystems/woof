#!/usr/bin/env python3
"""Hash a completed native reference and advertise it immediately."""
import datetime
import hashlib
import json
from pathlib import Path
import sys

ref = Path(sys.argv[1]).resolve()
name = sys.argv[2]
run = ref / 'runs' / name
def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()
assert (run / 'exit-code').read_text().strip() == '0'
assert 'Critical error messages =            0' in (run / 'log.atmosphere.0000.out').read_text()
files = {p.name: dict(sha256=sha(p), bytes=p.stat().st_size) for p in run.iterdir() if p.is_file() and not p.is_symlink() and p.name != 'MANIFEST.json'}
receipt = dict(label='Pristine strict atmosphere with two-NULLIFY init derivative, confirmation pending',
               command=f'bash hex/tree/tools/rrfs_44c0849_reference/run_forecast.sh {ref} {name}',
               inputs_sha256=sha(run / 'INPUTS.json'),
               init_manifest_sha256=sha(ref / 'native/init-two-nullify-derivative/MANIFEST.json'), files=files,
               rank_count=1, strict=True, exit_code=0, critical_errors=0)
(run / 'MANIFEST.json').write_text(json.dumps(receipt, indent=2) + '\n')
line = f'{name} native forecast | {run}/MANIFEST.json | sha256 {sha(run / "MANIFEST.json")} | read history/diag/restart with hashes in manifest; derivative init.'
with (ref / 'READY.md').open('a') as f:
    f.write(line + '\n')
now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
inbox = __import__('os').environ.get('RRFS_REPIN_INBOX')
if inbox:
    with Path(inbox).open('a') as f:
        f.write(f'{now} [native-ref] RESULT: {line}\n')
print(line)
