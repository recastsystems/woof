#!/usr/bin/env python3
"""Read source-bound SNICAR frames through the production CUDA loader."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--out', type=Path, required=True)
args = ap.parse_args()
import cupy as cp
from woof.certify.compile_platform import compile_platform_fingerprint
from woof.core.noahmp_kernel_sources import exported_kernels
from woof.core.snow_papers import _module, _lap_module
if cp.cuda.runtime.getDeviceCount() != 1:
    raise SystemExit('DEFECT: SNICAR frame measurement requires one admitted CUDA device')
kernels = root / 'woof/core/kernels'
source = (kernels / 'snow_papers.cu').read_text()
rows = {}
for name, loader, code in (
    ('snow_papers', _module, source),
    ('snow_lap', _lap_module, source + '\n' + (kernels / 'snow_lap.cu').read_text()),
):
    print(f'compiling {name}', flush=True)
    start = time.monotonic()
    module = loader(cp.cuda.Device().id)
    attrs = {s: module.get_function(s).attributes for s in exported_kernels(code)}
    rows[name] = dict(sha256=hashlib.sha256(code.encode()).hexdigest(),
                      functions=attrs, seconds=time.monotonic()-start)
    print(json.dumps({name: rows[name]}), flush=True)
args.out.write_text(json.dumps(dict(platform=compile_platform_fingerprint(),
    strict=os.environ.get('GPUWM_WRF_EXACT'), numerics=os.environ.get('GPUWM_NUMERICS'),
    rows=rows), indent=2) + '\n')
