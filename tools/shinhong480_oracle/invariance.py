#!/usr/bin/env python3
"""Assembled CUDA source + NVRTC options of every kernel module, fast or strict.

    PYTHONPATH=ENGINE python invariance.py fast|strict OUT.json

Run on the tip and on the branch; every module both trees share must carry
the same source digest and options (the 4.7.1 option's kernels are today's).
"""
import hashlib
import json
import os
import sys
from pathlib import Path

mode = sys.argv[1]
os.environ.pop("GPUWM_WRF_EXACT", None)
if mode == "strict":
    os.environ["GPUWM_WRF_EXACT"] = "1"
import woof.core.kernels as K  # noqa: E402

names = sorted(p.stem for p in Path(K.__file__).parent.glob("*.cu"))
out = {}
for n in names:
    try:
        src = K.module_source(n)
        opts = K.module_options(n)
        out[n] = [hashlib.sha256(src.encode()).hexdigest(), list(opts)]
    except Exception as e:  # noqa: BLE001
        out[n] = ["ERROR " + type(e).__name__ + ": " + str(e)[:120], None]
Path(sys.argv[2]).write_text(json.dumps(out, indent=0, sort_keys=True))
print(mode, len(out), "modules")
