"""Production-path replay and kernel timing of the Fitch pins on a native capture.

Builds each farm through woof.core.wind_farm.attach_wind_farm (the production
routing) from native WRF turbine files, applies it once to the captured real
inputs and compares every word with the native referee's own outputs; then
times the column kernel with CUDA events (launch only, same arguments).

usage: bench_fitch_480.py NATIVE_DIR OUT_JSON [--timing N]
NATIVE_DIR holds fitch-native.in, n480-ind1.out, n480-ind0.out, n471.out and the
two native turbine files.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse, hashlib, json, os, shutil, tempfile
import numpy as np
import cupy as cp

from woof.wrf_exact import ENABLED
from woof.core.wind_farm import attach_wind_farm

ap = argparse.ArgumentParser()
ap.add_argument("native", type=Path)
ap.add_argument("out", type=Path)
ap.add_argument("--timing", type=int, default=0)
ap.add_argument("--dump", type=Path, help="write every output array of every arm to this .npz")
ap.add_argument("--today", action="store_true",
                help="run on the pre-port tree: only the wrf_471 arms, with its default windfarm_induction=0 for -1")
args = ap.parse_args()
raw = (args.native / "fitch-native.in").read_bytes()
at = 0


def read(n, kind="f4"):
    global at
    a = np.frombuffer(raw, dtype=">" + kind, count=n, offset=at).astype(kind)
    at += n * 4
    return a


nx, ny, nz, nt, _ = map(int, read(5, "i4"))
nc = nx * ny
dx, dt, factor = map(float, read(3))
read(2 * nt); read(3 * nt, "i4"); read(5 * nt); read(30000)


def vol():
    return read(nx * nz * ny).reshape(ny, nz, nx).transpose(1, 0, 2).copy()


u, v, dz = vol(), vol(), vol()
zw = read(nx * (nz + 1) * ny).reshape(ny, nz + 1, nx).transpose(1, 0, 2).copy()
du0, dv0, qke0 = vol(), vol(), vol()
assert at == len(raw)


def native(name):
    out = np.frombuffer((args.native / name).read_bytes(), dtype=">f4").astype("f4")
    r = {}
    for i, k in enumerate(["du", "dv", "qke"]):
        r[k] = out[i * nz * nc:(i + 1) * nz * nc].reshape(ny, nz, nx).transpose(1, 0, 2)
    for i, k in enumerate(["POWER", "THRUST"]):
        r[k] = out[3 * nz * nc + i * nc:3 * nz * nc + (i + 1) * nc].reshape(ny, nx)
    return r


tables = Path(tempfile.mkdtemp(prefix="fitch480-"))
for f in ("windturbines-ij.txt", "wind-turbine-1.tbl"):
    shutil.copy(args.native / f, tables / f)
shape = (nz, ny, nx)
arms = [
    ("wrf_471", -1, "n471.out"),    # today's default Fitch (ArWen's resident scheme)
    ("wrf_471", 1, "n480-ind1.out"),  # today's induction kernel
    ("wrf_480", -1, "n480-ind1.out"),  # 4.8.0 default: induction on
    ("wrf_480", 0, "n480-ind0.out"),  # 4.8.0 original branch (4.7.1 module words + THRUST)
]
results = []
dumps = {}
if args.today:
    arms = [a for a in arms if a[0] == "wrf_471"]
for version, induction, ref in arms:
    cfg_induction = 0 if (args.today and induction == -1) else induction
    cfg = SimpleNamespace(wind_farm_table=str(tables / "windturbines-ij.txt"), wind_farm_scheme="fitch",
                          windfarm_ij=1, dx=dx, dy=dx, dt=dt, windfarm_tke_factor=factor,
                          windfarm_induction=cfg_induction, windfarm_fitch_version=version,
                          wind_farm_max_dx_m=3000.0, map_proj=1)
    farm, detection = attach_wind_farm(cfg, shape)
    atm = dict(u=cp.asarray(u), v=cp.asarray(v), dz=cp.asarray(dz), z_interface=cp.asarray(zw),
               rho=cp.full(shape, 1.23, cp.float32))
    fields = dict(qke=cp.asarray(qke0), exch_m=cp.zeros(shape, cp.float32))
    fields["qke_adv"] = fields["qke"]
    rates = dict(du=cp.asarray(du0), dv=cp.asarray(dv0))
    farm.apply_mynn(atm, fields, rates)
    cp.cuda.get_current_stream().synchronize()
    hist = farm.history_fields()
    got = dict(du=cp.asnumpy(rates["du"]), dv=cp.asnumpy(rates["dv"]), qke=cp.asnumpy(fields["qke"]),
               POWER=cp.asnumpy(hist["POWER"]))
    if "THRUST" in hist:
        got["THRUST"] = cp.asnumpy(hist["THRUST"])
    exp = native(ref)
    cmp = {k: dict(words=int(a.size), different=int((a.view("u4") != exp[k].view("u4")).sum()))
           for k, a in got.items()}
    row = dict(windfarm_fitch_version=version, windfarm_induction=induction, farm=type(farm).__name__,
               kernel=("wind_fitch_480.fitch_induction_480" if getattr(farm, "induction", None) == 1 else
                       "wind_fitch_480.fitch_original_480" if getattr(farm, "induction", None) == 0 else
                       "wind_mav.fitch_induction" if hasattr(farm, "native_kernel") else
                       "wind_farm.wind_fitch_sparse" if getattr(farm, "sparse", False) else "wind_farm.legacy"),
               native_reference=ref, history_fields=sorted(hist), compare=cmp,
               total_words=sum(c["words"] for c in cmp.values()),
               total_different=sum(c["different"] for c in cmp.values()))
    if args.timing:
        class Capture:
            def __call__(self, *a, **k):
                self.args, self.kwargs = a, k
        cap = Capture()
        attr = "native_kernel" if hasattr(farm, "native_kernel") else "fused"
        real = getattr(farm, attr)
        setattr(farm, attr, cap)
        farm.apply_mynn(atm, fields, rates)
        setattr(farm, attr, real)
        for _ in range(20):
            real(*cap.args, **cap.kwargs)
        samples = []
        for _ in range(9):
            s, e = cp.cuda.Event(), cp.cuda.Event()
            s.record()
            for _ in range(args.timing):
                real(*cap.args, **cap.kwargs)
            e.record(); e.synchronize()
            samples.append(cp.cuda.get_elapsed_time(s, e) / args.timing * 1000.0)
        row["kernel_us_samples"] = samples
        row["kernel_us_median"] = float(np.median(samples))
    if args.dump is not None:
        dumps.update({f"{version}_{induction}_{k}": a for k, a in got.items()})
    results.append(row)
    print(json.dumps({k: row[k] for k in row if k not in ("compare", "kernel_us_samples")}), flush=True)
shutil.rmtree(tables)
if args.dump is not None:
    np.savez(args.dump, **dumps)
args.out.write_text(json.dumps(dict(numerics="strict" if ENABLED else "fast",
                                    device=cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
                                    capture_sha256=hashlib.sha256(raw).hexdigest(), arms=results), indent=1))
