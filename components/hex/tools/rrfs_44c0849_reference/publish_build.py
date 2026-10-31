#!/usr/bin/env python3
"""Publish an audited build only after a clean-clone rebuild repeats its hash."""
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import sys

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here.parent / 'native_physics_reference'))
from rrfs.publish_build import publish_build

ref, first, repeat = map(lambda x: Path(x).resolve(), sys.argv[1:4])
core = sys.argv[4]
def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()
assert sha(first / (core + '_model')) == sha(repeat / (core + '_model'))
for build in (first, repeat):
    assert json.loads((build / 'flag-audit.json').read_text())['all_ok']
existing = list((ref / 'native').glob('*/MANIFEST.json')) if (ref / 'native').exists() else []
matching = [p for p in existing if json.loads(p.read_text()).get('binary_sha256') == sha(first / (core + '_model'))]
if not matching:
    publish_build(first, ref, core, 'gnu-strict')
manifest = json.loads((first / 'MANIFEST.json').read_text())
key = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
pub = ref / 'native' / key
pub.chmod(0o755)
recipe = pub / 'recipe'
for name in ('prepare.py', 'build.sh', 'rebuild.sh'):
    target = recipe / 'rrfs_44c0849_reference' / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(here / name, target)
for name in ('rrfs/prepare_build.py', 'rrfs/pins.json', 'rrfs/audit_flags.py', 'rrfs/diagnostic_flags.py', 'rrfs/compiler_capture.py', 'audit_flags.py'):
    target = recipe / 'native_physics_reference' / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(here.parent / 'native_physics_reference' / name, target)
target = recipe / 'native_dry_reference/edit_makefile_flags.py'
target.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(here.parent / 'native_dry_reference/edit_makefile_flags.py', target)
shutil.copy2(repeat / 'executable.sha256', pub / 'rebuild-executable.sha256')
shutil.copy2(repeat / 'flag-audit.json', pub / 'rebuild-flag-audit.json')
(pub / 'REBUILD.json').write_text(json.dumps(dict(first_sha256=sha(first / (core + '_model')), rebuild_sha256=sha(repeat / (core + '_model')), same=True, first=str(first), rebuild=str(repeat), recipe={str(p.relative_to(recipe)): sha(p) for p in recipe.rglob('*') if p.is_file()}), indent=2) + '\n')
(pub / 'rebuild.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\nHERE=$(dirname "$(realpath "$0")")\nexec bash "$HERE/recipe/rrfs_44c0849_reference/rebuild.sh" ' + shlex.quote(str(ref)) + ' "${1:?new scratch directory}" ' + shlex.quote(core) + '\n')
(pub / 'rebuild.sh').chmod(0o755)
named = ref / 'native' / core
named.symlink_to(key)
with (ref / 'READY.md').open('a') as f:
    f.write(f'{core} clean-clone rebuild identity | {pub}/REBUILD.json | sha256 {sha(pub / "REBUILD.json")} | {pub}/rebuild.sh NEW_SCRATCH; executable {manifest["binary_sha256"]}; flags audited in both builds.\n')
print(pub)
