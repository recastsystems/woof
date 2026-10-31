#!/usr/bin/env python3
"""Refresh a native capture source manifest from authenticated local git pins."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ref, output = map(Path, sys.argv[1:3])
src = ref / 'src/MPAS-Model'
pins = json.loads((ref / 'SOURCE-PINS.json').read_text())
base = Path(__file__).resolve().parents[2] / 'src/hexcore/data/mpasphys-native/mpas-source-rrfs.json'
manifest = json.loads(base.read_text())
manifest['pin_set'] = 'rrfs-44c0849'
manifest['pins']['rrfs-model'] = pins['model_commit']
manifest['submodule_commits'] = {m['path']: m['commit'] for m in pins['submodules']}
origins = {'rrfs-model': dict(prefix='', url=f'https://raw.githubusercontent.com/RRFSx/MPAS-Model/{pins["model_commit"]}/{{path}}')}
repos = [('rrfs-model', '', src)]
for m in pins['submodules']:
    origin = Path(m['path']).name.lower()
    if origin == 'physics_mmm':
        origin = 'mmm-physics'
    manifest['pins'][origin] = m['commit']
    repository = m['url'].removesuffix('.git').removeprefix('https://github.com/')
    origins[origin] = dict(prefix=m['path'] + '/', url=f'https://raw.githubusercontent.com/{repository}/{m["commit"]}/{{path}}')
    repos.append((origin, m['path'] + '/', src / m['path']))
files = {}
for origin, prefix, root in repos:
    paths = subprocess.check_output(['git', '-C', str(root), 'ls-files', 'src' if origin == 'rrfs-model' else '.'], text=True, timeout=60).splitlines()
    for relative in paths:
        path = root / relative
        if not path.is_file() or '.git' in path.parts:
            continue
        if path.suffix.lower() not in {'.f', '.f90', '.c', '.h', '.inc', '.xml'} and not path.name.startswith('Makefile'):
            continue
        files[prefix + relative] = dict(origin=origin, path=relative, bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
# Keep non-source fixtures from the old manifest only where still present.
for relative, row in manifest['files'].items():
    path = src / relative
    if relative not in files and path.is_file():
        files[relative] = dict(row, bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
manifest['files'] = dict(sorted(files.items()))
manifest['origins'] = origins
output.write_text(json.dumps(manifest, sort_keys=True, indent=1) + '\n')
print(f'{len(files)} files; {output}')
