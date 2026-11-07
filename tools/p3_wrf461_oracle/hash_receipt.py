"""Small full-array and per-frame hash receipts, with state tendencies."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from compare import read, OUT3D, OUT2D


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a, dtype="<f4").tobytes()).hexdigest()


def receipt(fix, root):
    index = json.loads((fix / "index.json").read_text())
    out3d = index.get("out3d", OUT3D)
    result = {"cases": {}}
    for case in index["cases"]:
        name = case["name"]
        data = read(root / (name + ".ref"), case["ncol"], case["nk"],
                    case["nsteps"], out3d)
        fields = {}
        for field, a in data.items():
            value = a[1:]
            # The wrapper writes the post-call state. These are the net
            # tendencies implied by successive post-call states, in FP32.
            tendency = (a[1:] - a[:-1]) / np.float32(case["dt"])
            fields[field] = dict(sha256=digest(value), words=int(value.size),
                                 tendency_sha256=digest(tendency),
                                 frames=[digest(v) for v in value],
                                 nan_words=int(np.count_nonzero(np.isnan(value))))
        result["cases"][name] = dict(in_claim=case.get("in_claim", True),
                                     fields=fields)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fix", type=Path)
    ap.add_argument("root", type=Path)
    ap.add_argument("out", type=Path)
    a = ap.parse_args()
    a.out.write_text(json.dumps(receipt(a.fix, a.root), indent=1))


if __name__ == "__main__":
    main()
