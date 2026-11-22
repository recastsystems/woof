#!/usr/bin/env python3
"""Word-for-word comparison of native outputs from runs on different rank counts.

usage: native_rank_identity.py OUT.json REFERENCE_RUN_DIR OTHER_RUN_DIR... 
Compares every float variable of every tracer.*.nc present in all runs.
"""
import os as _hex_os
_hex_os.environ["WOOF_HEX_NUMERICS"] = "strict"  # a bitwise proof against native MPAS runs strict numerics (fast is the default, strict the opt-in)
import hashlib, json, sys
from pathlib import Path
import numpy as np
from netCDF4 import Dataset

out = Path(sys.argv[1]); runs = [Path(p) for p in sys.argv[2:]]
names = sorted(set.intersection(*[{f.name for f in r.glob('tracer.*.nc')} for r in runs]))
report = {'schema': 'gpuwm-hex.native-rank-identity/v1', 'runs': [str(r) for r in runs], 'files': {}}
ok = True
for name in names:
    ref = Dataset(runs[0] / name); ref.set_auto_mask(False)
    entry = {}
    for other in runs[1:]:
        d = Dataset(other / name); d.set_auto_mask(False)
        diffs = {}
        for v in ref.variables:
            a = np.asarray(ref[v][:]); b = np.asarray(d[v][:])
            if a.dtype.kind != 'f':
                continue
            n = int(np.count_nonzero(a.view(np.uint32 if a.dtype == np.float32 else np.uint64) != b.view(np.uint32 if b.dtype == np.float32 else np.uint64)))
            if n:
                diffs[v] = n
        entry[other.name] = {'identical': not diffs, 'differing_words': diffs}
        ok &= not diffs
        d.close()
    ref.close()
    report['files'][name] = entry
report['all_identical'] = ok
out.write_text(json.dumps(report, indent=1) + '\n')
print(json.dumps({'all_identical': ok, 'files': len(names), 'runs': [r.name for r in runs]}))
