#!/usr/bin/env python3
"""Publish the labelled, carried two-NULLIFY derivative and its case receipts."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

ref, first, repeat = map(lambda x: Path(x).resolve(), sys.argv[1:4])
here = Path(__file__).resolve().parent
def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()
def write(p, data):
    p.write_text(json.dumps(data, indent=2) + '\n')
binary = 'init_atmosphere_model'
assert sha(first / binary) == sha(repeat / binary)
assert sha(first / 'source-correction.diff') == sha(repeat / 'source-correction.diff')
for b in (first, repeat):
    assert json.loads((b / 'flag-audit.json').read_text())['all_ok']
pub = ref / 'native/init-two-nullify-derivative'
pub.mkdir(parents=True, exist_ok=True)
for name in (binary, 'SOURCE-PINS.json', 'build.log', 'toolchain.txt', 'compile-commands.jsonl', 'flag-audit.json', 'executable.sha256', 'build-source.diff'):
    shutil.copy2(first / name, pub / name)
shutil.copy2(first / 'source-correction.diff', pub / 'APPROVED.patch')
shutil.copy2(repeat / 'flag-audit.json', pub / 'rebuild-flag-audit.json')
old = Path(__import__('os').environ['MPAS_LANES']) / 'ref/rrfs/native/incoming-approved-init-r0-v1'
ruling = json.loads((old / 'DREW-RULING.json').read_text())
ruling.update(carried=True, confirmation='pending', target_fork='44c084944e17f7098668bef6d1e353d12e3c2d1a',
              carried_by='GUIDE-A1A4 INBOX 2026-10-28T21:53:48Z', carried_patch_sha256=sha(pub / 'APPROVED.patch'))
write(pub / 'DREW-RULING.json', ruling)
recipe = pub / 'recipe'
shutil.copytree(ref / 'native/init_atmosphere/recipe', recipe, dirs_exist_ok=True)
for name in ('prepare.py', 'build.sh', 'rebuild.sh'):
    shutil.copy2(here / name, recipe / 'rrfs_44c0849_reference' / name)
(pub / 'rebuild.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\nHERE=$(dirname "$(realpath "$0")")\nexport INIT_POINTER_WITNESS=1\nexec bash "$HERE/recipe/rrfs_44c0849_reference/rebuild.sh" "' + str(ref) + '" "${1:?new scratch directory}" init_atmosphere\n')
(pub / 'rebuild.sh').chmod(0o755)
label = 'init derivative (carried two-NULLIFY ruling 2026-10-14 07:45Z; confirmation pending)'
write(pub / 'REBUILD.json', dict(first_sha256=sha(first / binary), rebuild_sha256=sha(repeat / binary), same=True,
      first=str(first), rebuild=str(repeat), command='INIT_POINTER_WITNESS=1 bash recipe/rrfs_44c0849_reference/rebuild.sh REF NEW_WORK init_atmosphere'))
write(pub / 'MANIFEST.json', dict(label=label, pristine=False, scope='Only case 7/9 inputs; every derived forecast binds this receipt.',
      model_commit=ruling['target_fork'], files={p.name: sha(p) for p in pub.iterdir() if p.is_file() and p.name != 'MANIFEST.json'}))
ready = [f'init two-NULLIFY derivative | {pub}/MANIFEST.json | sha256 {sha(pub / "MANIFEST.json")} | {pub}/rebuild.sh NEW_SCRATCH; binary {sha(pub / binary)}; source-corrected, confirmation pending.']
for mesh, case in [('x1.2562', 7), ('lam7k', 7), ('lam7k', 9)]:
    directory = ref / 'cases' / mesh / f'case{case}-pointer-witness'
    log = directory / 'log.init_atmosphere.0000.out'
    assert 'Critical error messages =            0' in log.read_text()
    outputs = [directory / 'init.nc'] if case == 7 else sorted(directory.glob('lbc.*.nc'))
    assert outputs and all(p.is_file() for p in outputs)
    files = {p.name: dict(sha256=sha(p), bytes=p.stat().st_size) for p in directory.iterdir() if p.is_file()}
    write(directory / 'MANIFEST.json', dict(label=label, pristine=False, init_build_manifest=str(pub / 'MANIFEST.json'),
          init_build_manifest_sha256=sha(pub / 'MANIFEST.json'), command='OMP_NUM_THREADS=1 timeout 1800 nice -n 10 taskset -c 8-23 ' + str(pub / binary),
          cwd=str(directory), outputs=[p.name for p in outputs], files=files, critical_errors=0))
    ready.append(f'{mesh} case {case} init derivative | {directory}/MANIFEST.json | sha256 {sha(directory / "MANIFEST.json")} | replay inputs and outputs listed in receipt; not pristine.')
with (ref / 'READY.md').open('a') as f:
    f.write('\n'.join(ready) + '\n')
print('\n'.join(ready))
