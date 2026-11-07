#!/usr/bin/env python3
"""Verify public-source pins before instrumenting an owned WRF source copy.

The cache layout is <cache>/<ref>/<path>, matching native-pin.json. Downloads
are deliberately separate: inspect the licence files before obtaining code.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess

here = Path(__file__).resolve().parent
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('wrf', type=Path, help='owned, uninstrumented WRF v4.7.1 source')
ap.add_argument('cache', type=Path, help='already obtained and licence-reviewed pins')
args = ap.parse_args()
pin = json.loads((here / 'native-pin.json').read_text())
for item in pin['sources']:
    source = args.cache / item['ref'] / item['path']
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != item['sha256']:
        raise SystemExit(f'pin mismatch: {source}: {digest}')
for name in ('phys/module_wind_mav.F', 'LICENSE.txt', 'share/module_llxy.F'):
    if (args.wrf / name).read_bytes() != (args.cache / 'v4.7.1' / name).read_bytes():
        raise SystemExit(f'WRF v4.7.1 base mismatch: {name}')
# This newer public module adds induction. The capture patch only adapts the
# extra THRUST formal to the older driver and records it locally instead.
new_fitch = args.cache / pin['induction_commit'] / 'phys/module_wind_fitch.F'
(args.wrf / 'phys/module_wind_fitch.F').write_bytes(new_fitch.read_bytes())
subprocess.run(['patch', '--batch', '--forward', '-p1', '-i', str(here / 'capture.patch')],
               cwd=args.wrf, check=True)
registry = args.wrf / 'Registry/Registry.EM_COMMON'
text = registry.read_text()
if 'rconfig integer windfarm_induction ' in text:
    raise SystemExit('induction registry row already exists; use a clean owned source copy')
lines = text.splitlines(True)
index = next(i for i, line in enumerate(lines) if line.startswith('rconfig') and ' windfarm_opt ' in line)
lines.insert(index, 'rconfig integer windfarm_induction namelist,physics 1 0 rh "windfarm_induction" "" ""\n')
registry.write_text(''.join(lines))
print(json.dumps({'base': pin['base_version'], 'induction': pin['induction_commit'],
                  'verified_files': len(pin['sources']), 'capture_patch_sha256':
                  hashlib.sha256((here / 'capture.patch').read_bytes()).hexdigest()}, indent=2))
