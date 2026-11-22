"""Crop replay captures to small decks and record the native referee's words.

    python make_oracle.py --native ./native.sh --out DIR SRC.npz:NAME:J0:I0:N ...

Writes NAME.capture.npz (the cropped actual arguments) and NAME.native.npz
(every N2 field after the native call), the pair tests/test_mynnsfc_b66187f.py
replays on the GPU.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import replay as R  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--native", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("specs", nargs="+")
args = ap.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
for spec in args.specs:
    src, name, j0, i0, n = spec.split(":")
    j0, i0, n = int(j0), int(i0), int(n)
    cap = dict(np.load(src))
    c = {}
    for k, v in cap.items():
        v = np.asarray(v)
        if v.ndim == 2:
            c[k] = np.ascontiguousarray(v[j0:j0 + n, i0:i0 + n])
        elif v.ndim == 3:
            c[k] = np.ascontiguousarray(v[:, j0:j0 + n, i0:i0 + n])
        else:
            c[k] = v
    c["ny"] = np.int64(n)
    c["nx"] = np.int64(n)
    native = R.run_native(args.native, c)
    np.savez_compressed(out / f"{name}.capture.npz", **c)
    np.savez_compressed(out / f"{name}.native.npz", **native)
    print(name, n * n, "columns")
