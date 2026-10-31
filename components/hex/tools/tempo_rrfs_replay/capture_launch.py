"""Measurement launcher (not product code): the hex CLI, with the TEMPO slot's inputs captured at chosen calls.

TEMPO_CAPTURE_CALLS="1,12,24" picks 1-based launch_mpas calls; TEMPO_CAPTURE_DIR receives one folder per call:
every device array reachable from the call's pool dict (aliases kept: one file per distinct buffer), the
launch keywords and the scalars, so tempo_replay.py can rerun the same call on any module version.
"""
import json
import os
import sys

import numpy as np

CALLS = {int(x) for x in os.environ.get("TEMPO_CAPTURE_CALLS", "1").split(",") if x}
OUT = os.environ["TEMPO_CAPTURE_DIR"]
SKIP = {"resources", "_native_workspace"}


def _walk(node, path, arrays, scalars):
    import cupy as cp
    if isinstance(node, dict):
        for key, value in node.items():
            if path == () and key in SKIP:
                continue
            _walk(value, path + (key,), arrays, scalars)
    elif isinstance(node, cp.ndarray):
        arrays.append((path, node))
    elif node is None or isinstance(node, (bool, int, float, str, np.generic)):
        scalars.append((path, node.item() if isinstance(node, np.generic) else node))
    else:
        raise TypeError(f"capture: unhandled {type(node).__name__} at {path}")


def _keyword(value):
    """A launch keyword as JSON: tuples as lists; the device do_diag_dbz word (hourly) as the host bool it means."""
    import cupy as cp
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, cp.ndarray):
        return bool(int(cp.asnumpy(value).reshape(-1)[0]) == 1)
    return value


def save(p, kw, folder):
    import cupy as cp
    os.makedirs(folder, exist_ok=False)
    arrays, scalars = [], []
    _walk(p, (), arrays, scalars)
    buffers, rows = {}, []
    for path, a in arrays:
        key = (a.data.ptr, a.shape, a.strides, a.dtype.str)
        if key not in buffers:
            buffers[key] = len(buffers)
            np.save(os.path.join(folder, f"b{buffers[key]:03d}.npy"), cp.asnumpy(a))
        rows.append({"path": list(path), "buffer": buffers[key]})
    spans = sorted((k[0], k[0] + int(np.prod(k[1])) * np.dtype(k[3]).itemsize, b) for k, b in buffers.items())
    overlaps = [(a[2], b[2]) for a, b in zip(spans, spans[1:]) if b[0] < a[1]]
    manifest = {"arrays": rows, "scalars": [{"path": list(pth), "value": v} for pth, v in scalars],
                "keywords": {k: _keyword(v) for k, v in kw.items()},
                "buffers": len(buffers), "overlapping_buffers": overlaps}
    with open(os.path.join(folder, "manifest.json"), "w") as stream:
        json.dump(manifest, stream, indent=1)
    print(f"TEMPO capture {folder}: {len(rows)} arrays in {len(buffers)} buffers, overlaps {overlaps}", flush=True)


def install():
    from woof.hex.mpasphys.rrfs import tempo_v303 as t
    original = t.launch_mpas
    count = [0]

    def captured(p, **kw):
        count[0] += 1
        if count[0] in CALLS:
            save(p, kw, os.path.join(OUT, f"call{count[0]:03d}"))
        return original(p, **kw)

    t.launch_mpas = captured


if __name__ == "__main__":
    install()
    from woof.hex import cli
    a = cli.build_parser().parse_args(sys.argv[1:])
    cli._select_numerics(a)
    sys.exit(int(a.handler(a)))
