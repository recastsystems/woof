#!/usr/bin/env python3
"""Replay bounded NWP inputs against outputs witnessed in a real native run.

All layout conversion and transfers are offline proof work. The CUDA leaves
remain asynchronous device launches. A successful receipt proves these module
calls, not option admission or whole-forecast integration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decode(path):
    raw = path.read_bytes()
    h = np.frombuffer(raw, dtype="<i4", count=26)
    if len(h) != 26 or h[0] != 47102:
        raise ValueError("invalid native NWP capture header")
    nx, ny, nz = map(int, h[1:4])
    if not (0 < nx <= 4 and 0 < ny <= 4 and nz >= 2):
        raise ValueError("invalid bounded capture dimensions")
    scalar = np.frombuffer(raw, dtype="<f4", count=9, offset=104)
    offset = 140
    arrays = {}

    def take(shape):
        nonlocal offset
        count = int(np.prod(shape))
        a = np.frombuffer(raw, dtype="<f4", count=count, offset=offset).reshape(shape, order="F")
        offset += count * 4
        return np.ascontiguousarray(a.transpose(1, 2, 0) if len(shape) == 3
                                   else a.T if len(shape) == 2 else a)

    for name in ("w", "p8w", "qg", "rho", "ph", "phb", "refl"):
        arrays[name] = take((nx, nz+1 if name in ("w", "p8w", "ph", "phb") else nz, ny))
    arrays["znw"] = take((nz+1,))
    for name in ("u10", "v10"):
        arrays[name] = take((nx, ny))
    for flag, name in zip(h[14:20], ("qh", "ng", "nh", "volume", "qr", "nr")):
        if flag:
            arrays[name] = take((nx, nz, ny))
    before = np.stack([take((nx, ny)) for _ in range(12)])
    after = np.stack([take((nx, ny)) for _ in range(12)])
    if offset != len(raw):
        raise ValueError("unexpected trailing native NWP capture bytes")
    return h, scalar, arrays, before, after


def main():
    p = argparse.ArgumentParser()
    p.add_argument("captures", type=Path, nargs="+")
    p.add_argument("--run-receipt", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    run = json.loads(args.run_receipt.read_text())
    if not run.get("native_run_completed") or not run.get("strict_numerics"):
        raise ValueError("a witnessed completed strict native run receipt is required")
    import cupy as cp
    from woof import wrf_exact
    from woof.core import nwp_windows, nwp_hail
    from woof.core.kernels import get_kernel, module_source
    if not wrf_exact.ENABLED:
        raise ValueError("start with GPUWM_WRF_EXACT=1")
    rows = []
    for path in args.captures:
        h, scalar, arrays, before, expected = decode(path)
        if scalar[2].view("u4") != np.float32(9.81).view("u4"):
            raise ValueError("native gravity differs from the pinned CUDA constant")
        d = {k: cp.asarray(v) for k, v in arrays.items()}
        out = cp.asarray(before)
        nwp_windows.update(*(d[k] for k in ("w", "p8w", "znw", "qg", "rho", "ph", "phb",
                                            "u10", "v10", "refl")), out,
                           dt=float(scalar[0]), xtime=float(scalar[1]), history_minutes=int(h[5]),
                           step=int(h[6]), adaptive=bool(h[7]), max_time_step=int(h[8]))
        kw = {k: d[k] for k in ("qh", "ng", "nh", "volume", "qr", "nr") if k in d}
        if h[4] == 18 and h[12] == 0:
            kw["nssl_single_moment"] = nwp_hail.NsslSingleMoment(*map(float, scalar[3:]), int(h[13]))
        nwp_hail.update(d["qg"], d["rho"], out[7], out[8], mp_physics=int(h[4]),
                        gsfcgce_hail=int(h[9]), gsfcgce_2ice=int(h[10]), mpuse_hail=int(h[11]), **kw)
        actual = cp.asnumpy(out)
        diff = expected.view("u4") != actual.view("u4")
        # The resident driver's leaf on the same inputs: the windows read the
        # interface pressure through its "p8w > 400 hPa" bits and write the
        # nine history planes (the three column outputs have no reader there).
        from replay_prepare import pack_below_400hpa
        nz, ny, nx = arrays["qg"].shape
        planes = [cp.asarray(before[i]) for i in range(9)]
        get_kernel("nwp_windows", "nwp_window_resident")(((ny * nx + 127) // 128,), (128,),
            (d["w"], cp.asarray(pack_below_400hpa(arrays["p8w"])), d["znw"], d["qg"], np.int32(1),
             d["rho"], d["ph"], d["phb"], np.int32(1), d["u10"], d["v10"], d["refl"], *planes,
             np.int32(nz), np.int32(ny), np.int32(nx), np.float32(scalar[0]), np.float32(scalar[1]),
             np.int32(h[5]), np.int32(h[6]), np.int32(h[7]), np.int32(h[8])))
        nwp_hail.update(d["qg"], d["rho"], planes[7], planes[8], mp_physics=int(h[4]),
                        gsfcgce_hail=int(h[9]), gsfcgce_2ice=int(h[10]), mpuse_hail=int(h[11]), **kw)
        resident = np.stack([cp.asnumpy(plane) for plane in planes])
        resident_diff = expected[:9].view("u4") != resident.view("u4")
        row = dict(capture=str(path), capture_sha256=sha(path), capture_bytes=path.stat().st_size,
                   header=h.tolist(), scalars=scalar.tolist(), words=int(expected.size),
                   differing_words=int(diff.sum()),
                   resident_words=int(resident_diff.size),
                   resident_differing_words=int(resident_diff.sum()), fields={})
        for i, name in enumerate(nwp_windows.FIELDS):
            finite = np.isfinite(expected[i])
            row["fields"][name] = dict(words=int(expected[i].size), differing_words=int(diff[i].sum()),
                                       native_nonfinite=int((~finite).sum()))
            if finite.any():
                row["fields"][name].update(native_min=float(expected[i][finite].min()),
                                           native_max=float(expected[i][finite].max()))
        if diff.any():
            ix = tuple(map(int, np.argwhere(diff)[0]))
            row["first_difference"] = dict(index=ix, field=nwp_windows.FIELDS[ix[0]],
                                            native=float(expected[ix]), cuda=float(actual[ix]),
                                            native_word=int(expected.view("u4")[ix]),
                                            cuda_word=int(actual.view("u4")[ix]))
        rows.append(row)
    receipt = dict(schema="wrf471-real-nwp-module-replay-v1", native_run=run,
                   native_run_receipt_sha256=sha(args.run_receipt), replay_sha256=sha(Path(__file__)),
                   qualification="module calls only; whole option remains unqualified",
                   modules={name: hashlib.sha256(module_source(name).encode()).hexdigest()
                            for name in ("nwp_windows", "nwp_hail")}, results=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    differing = sum(r["differing_words"] + r["resident_differing_words"] for r in rows)
    print(json.dumps(dict(words=sum(r["words"] for r in rows),
                          differing_words=sum(r["differing_words"] for r in rows),
                          resident_words=sum(r["resident_words"] for r in rows),
                          resident_differing_words=sum(r["resident_differing_words"] for r in rows),
                          receipt=str(args.output))))
    if differing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
