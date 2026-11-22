"""Replay native-captured Fitch inputs against the WRF v4.8.0 strict referee.

The capture directory holds ``fitch-native.in`` (tools/wind_mav_wrf471_oracle
capture.patch layout: real inputs at the first dragforce call of a whole native
forecast) and ``fitch-native.out`` written by the referee's own compiled
module_wind_fitch.o (native/fitch_replay.F90 linked against the pinned strict
build's libwrflib.a). ``--induction`` selects the WRF 4.8.0 branch:

* 1 -> wind_fitch_480.fitch_induction_480 (``--kernel wrf_471`` replays
  wind_mav.fitch_induction, the previous pin's kernel, on the same words)
* 0 -> wind_fitch_480.fitch_original_480 (4.8.0 original branch + THRUST)

Every output word is compared as raw float32 bits. No generated weather columns.
"""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import cupy as cp

from woof.wrf_exact import ENABLED, compile_receipt
from woof.core.kernels import get_kernel

ap = argparse.ArgumentParser()
ap.add_argument("capture", type=Path)
ap.add_argument("--induction", choices=("0", "1"), required=True)
ap.add_argument("--allow-differences", action="store_true", help="diagnostic only")
ap.add_argument("--kernel", choices=("wrf_480", "wrf_471"), default="wrf_480")
ap.add_argument("--timing", type=int, default=0, help="also time N launches (fast numerics only)")
args = ap.parse_args()
p = args.capture
raw = (p / "fitch-native.in").read_bytes()
at = 0


def read(n, kind="f4"):
    global at
    a = np.frombuffer(raw, dtype=">" + kind, count=n, offset=at).astype(kind)
    at += n * 4
    return a


nx, ny, nz, nt, captured_induction = map(int, read(5, "i4"))
nc = nx * ny
meta = read(3)
diam, hub = [read(nt) for _ in range(2)]
ix, jy, nkind = [read(nt, "i4") for _ in range(3)]
stc, stc2, cutin, cutout, npower = [read(nt) for _ in range(5)]
curves = [read(10000).reshape(100, 100) for _ in range(3)]


def vol():
    return read(nx * nz * ny).reshape(ny, nz, nx).transpose(1, 0, 2).copy()


u, v, dz = [vol() for _ in range(3)]
zw = read(nx * (nz + 1) * ny).reshape(ny, nz + 1, nx).transpose(1, 0, 2).copy()
du, dv, qke = [vol() for _ in range(3)]
assert at == len(raw), (at, len(raw))
out = np.frombuffer((p / "fitch-native.out").read_bytes(), dtype=">f4").astype("f4")
expected = {}
for i, name in enumerate(["du", "dv", "qke"]):
    expected[name] = out[i * nz * nc:(i + 1) * nz * nc].reshape(ny, nz, nx).transpose(1, 0, 2).copy()
for i, name in enumerate(["POWER", "THRUST"]):
    expected[name] = out[3 * nz * nc + i * nc:3 * nz * nc + (i + 1) * nc].reshape(ny, nx)
gpu = lambda a: cp.asarray(np.ascontiguousarray(a))
col = (jy - 1) * nx + ix - 1
ids = np.argsort(col, kind="stable").astype("i4")
counts = np.bincount(col, minlength=nc).astype("i4")
ptr = np.r_[np.int32(0), np.cumsum(counts, dtype="i4")]
cols = np.flatnonzero(counts).astype("i4")
ws, tc, pw = [np.stack([c[:, k - 1] for k in nkind], axis=1).copy() for c in curves]
nv = np.asarray([np.flatnonzero(ws[:, t] == cutout[t])[0] + 1 for t in range(nt)], "i4")
if args.induction == "0":
    assert args.kernel == "wrf_480", "the wrf_471 pin has no native original-branch kernel"
    module, func = "wind_fitch_480", "fitch_original_480"
else:
    module, func = (("wind_fitch_480", "fitch_induction_480") if args.kernel == "wrf_480"
                    else ("wind_mav", "fitch_induction"))
kernel = get_kernel(module, func)
block = max(32, 1 << (nz - 1).bit_length())
static = (gpu(cols), gpu(ptr), gpu(ids), gpu(diam), gpu(hub), gpu(nv), gpu(ws), gpu(tc), gpu(pw),
          gpu(stc), gpu(stc2), gpu(u), gpu(v), gpu(dz), gpu(zw))


def launch():
    gdu, gdv, gq = [gpu(a) for a in (du, dv, qke)]
    adv = gq.copy()
    power = cp.zeros((ny, nx), "f4")
    thrust = cp.zeros_like(power)
    rate, inc = cp.zeros_like(gq), cp.zeros_like(gq)
    hout, tp = cp.zeros((nt, 10), "f4"), cp.zeros(nt, "f4")
    kernel((len(cols),), (block,), static + (gdu, gdv, gq, adv, power, thrust, rate, inc, hout, tp,
           np.int32(nc), np.int32(nz), np.int32(nt), *meta), shared_mem=nz * 4)
    return dict(du=gdu, dv=gdv, qke=gq, POWER=power, THRUST=thrust)


res_dev = launch()
cp.cuda.get_current_stream().synchronize()
actual = {k: cp.asnumpy(a) for k, a in res_dev.items()}
res = {}
for name, b in expected.items():
    a = actual[name]
    neq = a.view("u4") != b.view("u4")
    idx = np.argwhere(neq)
    res[name] = dict(words=int(a.size), different=int(neq.sum()),
                     max_absolute=float(np.nanmax(abs(a - b))),
                     first=[dict(index=i.tolist(), native=float(b[tuple(i)]), gpu=float(a[tuple(i)])) for i in idx[:8]])
timing = None
if args.timing:
    # CUDA events around N kernel launches on the same captured columns; inputs
    # are reset outside the timed region so every launch does the native work.
    gdu, gdv, gq = [gpu(a) for a in (du, dv, qke)]
    bufs = (gdu, gdv, gq, gq.copy(), cp.zeros((ny, nx), "f4"), cp.zeros((ny, nx), "f4"), cp.zeros_like(gq),
            cp.zeros_like(gq), cp.zeros((nt, 10), "f4"), cp.zeros(nt, "f4"))
    tail = (np.int32(nc), np.int32(nz), np.int32(nt), *meta)
    for _ in range(20):
        kernel((len(cols),), (block,), static + bufs + tail, shared_mem=nz * 4)
    samples = []
    for _ in range(7):
        s, e = cp.cuda.Event(), cp.cuda.Event()
        s.record()
        for _ in range(args.timing):
            kernel((len(cols),), (block,), static + bufs + tail, shared_mem=nz * 4)
        e.record()
        e.synchronize()
        samples.append(cp.cuda.get_elapsed_time(s, e) / args.timing)
    timing = dict(launches_per_sample=args.timing, samples_ms=samples, median_ms=float(np.median(samples)),
                  device=cp.cuda.runtime.getDeviceProperties(0)["name"].decode())
print(json.dumps(dict(shape=[nz, ny, nx], turbines=nt, captured_induction=captured_induction,
                      replayed_induction=int(args.induction), kernel=f"{module}.{func}",
                      capture_sha256=hashlib.sha256(raw).hexdigest(),
                      native_out_sha256=hashlib.sha256((p / "fitch-native.out").read_bytes()).hexdigest(),
                      numerics="strict" if ENABLED else "fast", compiler=compile_receipt(),
                      outputs=res, timing=timing), indent=2))
assert args.allow_differences or all(v["different"] == 0 for v in res.values()), "native module replay differs"
