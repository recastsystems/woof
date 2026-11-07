#!/usr/bin/env python3
"""Verify public pins and adapt the unchanged V3.5 physics to a WRF 4.7.1 driver."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess

here=Path(__file__).resolve().parent
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('wrf',type=Path,help='owned uninstrumented WRF v4.7.1 source copy')
ap.add_argument('cache',type=Path,help='licence-reviewed V3.5 source cache, paths from native-pin.json')
args=ap.parse_args();pin=json.loads((here/'native-pin.json').read_text())
for item in pin['sources']:
    source=args.cache/item['path']
    assert hashlib.sha256(source.read_bytes()).hexdigest()==item['sha256'],source
for path,sha in pin['base_sources'].items():
    assert hashlib.sha256((args.wrf/path).read_bytes()).hexdigest()==sha,path
subprocess.run(['patch','--batch','--forward','-p1','-i',str(here/'capture.patch')],cwd=args.wrf,check=True)
# WRF's make dependency tracks Registry/Registry, not included fragments.
# Touch its parent to regenerate all native namelist accessors as well.
(args.wrf/'Registry/Registry.EM').touch()
print(json.dumps(dict(verified_sources=len(pin['sources']),
    capture_patch_sha256=hashlib.sha256((here/'capture.patch').read_bytes()).hexdigest()),indent=2))
