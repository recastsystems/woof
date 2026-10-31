"""Measurement launcher (not product code): the hex CLI, with MYNN-EDMF's kernel-level inputs captured at chosen calls.

MYNN_CAPTURE_CALLS="1,12,24" picks 1-based MynnEdmfCuda.run calls; MYNN_CAPTURE_DIR receives one folder per call:
every device array in the call's argument mapping (aliases kept: one file per distinct buffer view), its scalars,
so replay_mynn.py can rerun the same call on any module version. The call itself then runs unchanged.
"""
import json
import os
import sys

import numpy as np

CALLS = {int(x) for x in os.environ.get("MYNN_CAPTURE_CALLS", "1").split(",") if x}
OUT = os.environ["MYNN_CAPTURE_DIR"]


def save(arguments, folder):
    import cupy as cp
    os.makedirs(folder, exist_ok=False)
    buffers, rows, scalars = {}, [], {}
    for name, value in arguments.items():
        if isinstance(value, cp.ndarray):
            key = (value.data.ptr, value.shape, value.strides, value.dtype.str)
            if key not in buffers:
                buffers[key] = len(buffers)
                host = cp.asnumpy(value)
                np.save(os.path.join(folder, f"b{buffers[key]:03d}.npy"), host)
            rows.append({"name": name, "buffer": buffers[key], "order": "F" if value.flags.f_contiguous and not
                         value.flags.c_contiguous else "C"})
        elif value is None or isinstance(value, (bool, int, float, str, np.generic)):
            scalars[name] = value.item() if isinstance(value, np.generic) else value
        else:
            raise TypeError(f"capture: unhandled {type(value).__name__} for {name}")
    spans = sorted((k[0], k[0] + int(np.prod(k[1])) * np.dtype(k[3]).itemsize, b) for k, b in buffers.items())
    overlaps = [(a[2], b[2]) for a, b in zip(spans, spans[1:]) if b[0] < a[1]]
    with open(os.path.join(folder, "manifest.json"), "w") as stream:
        json.dump({"arrays": rows, "scalars": scalars, "buffers": len(buffers), "overlapping_buffers": overlaps},
                  stream, indent=1)
    print(f"MYNN capture {folder}: {len(rows)} arrays in {len(buffers)} buffers, overlaps {overlaps}", flush=True)


def install():
    from woof.hex.mpasphys.rrfs import bl_mynnedmf_c942418 as m
    original = m.MynnEdmfCuda.run
    count = [0]

    def captured(self, arguments, **kw):
        count[0] += 1
        if count[0] in CALLS:
            import cupy as cp
            cp.cuda.get_current_stream().synchronize()
            save(arguments, os.path.join(OUT, f"call{count[0]:03d}"))
        return original(self, arguments, **kw)

    m.MynnEdmfCuda.run = captured


if __name__ == "__main__":
    install()
    from woof.hex import cli
    a = cli.build_parser().parse_args(sys.argv[1:])
    cli._select_numerics(a)
    sys.exit(int(a.handler(a)))
