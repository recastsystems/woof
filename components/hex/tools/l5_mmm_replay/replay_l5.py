#!/usr/bin/env python3
"""L5 module replay (referee tool, never on a run path): hex CUDA YSU / revised MM5 vs standalone native MMM-physics.

Inputs are the captured hex forecast calls (capture/sitecustomize.py). For each captured call and each MMM version
(a4baf7f, 550b5b4) the hex CUDA kernel and the standalone native build of that version
(native/build-<v>/libysu.so, libmm5.so; pristine MMM-physics source, MPAS strict GNU flags) run the same inputs; every
output word is compared bit for bit. YSU runs through MPAS v8.4.2's physics_wrf wrapper on both sides. The MM5
replay also runs the KIM form-drag option (550b5b4 only) with the captured var2d.

Usage: replay_l5.py CAPTURE_DIR NATIVE_DIR OUT.json [--tofd-factor 0.003]
"""
import ctypes as ct
import glob
import json
import os
import sys
import time

import numpy as np
import cupy as cp

from woof.hex.mpasphys import ysu_v842 as ysu
from woof.hex.mpasphys import sfclay_mm5_v842 as mm5

PTR = ct.POINTER(ct.c_float)
IPTR = ct.POINTER(ct.c_int)
VERSIONS = ("a4baf7f", "550b5b4")
FTZ = None  # --native-ftz: native YSU columns run with x86 flush-to-zero (diagnostic only; the bar is IEEE native)


class NativeYSU:
    def __init__(self, path):
        self.lib = ct.CDLL(path)
        self.wrapper = self.lib.run_wrapper
        self.wrapper.argtypes = [ct.POINTER(PTR), PTR, IPTR]
        status = np.full(2, -1, np.int32)
        self.lib.native_lifecycle.argtypes = [IPTR]
        self.lib.native_lifecycle(status.ctypes.data_as(IPTR))
        assert not np.any(status), status

    def run(self, arrays, flags, values):
        arrays = {a: None if v is None else v.copy() for a, v in arrays.items()}
        # tests/mpas_ysu_v842/oracle.py Oracle.run(wrapper=True): WRF dimensions carry one unused top level.
        for a in ysu._REQUIRED_LEVEL + ysu._OUTPUT_LEVEL + ysu._OPTIONAL_LEVEL:
            if arrays.get(a) is not None:
                arrays[a] = np.concatenate((arrays[a], np.zeros(1, arrays[a].dtype)))
        ptrs = (PTR * len(ysu.ARRAY_ARGUMENTS))(*[
            PTR() if arrays.get(a) is None else arrays[a].ctypes.data_as(PTR) for a in ysu.ARRAY_ARGUMENTS])
        scalars = np.array([values[a] for a in ysu.FLOAT_ARGUMENTS], np.float32)
        integers = np.array([flags[a] for a in ysu.INTEGER_ARGUMENTS], np.int32)
        self.wrapper(ptrs, scalars.ctypes.data_as(PTR), integers.ctypes.data_as(IPTR))
        for a in ysu._REQUIRED_LEVEL + ysu._OUTPUT_LEVEL + ysu._OPTIONAL_LEVEL:
            if arrays.get(a) is not None:
                arrays[a] = arrays[a][:-1]
        return arrays


def words(a):
    a = np.ascontiguousarray(a)
    return a.view(np.uint32) if a.dtype.itemsize == 4 else a.view(np.uint64)


def ysu_case(path, meta, natives, header, flip_pblmix):
    data = dict(np.load(path))
    kw = meta["kwargs"]
    values = dict(meta["constants"])
    values["dt"] = float(kw["dt"])
    topdown = 1 if flip_pblmix else int(kw["ysu_topdown_pblmix"])
    flags = dict(f_qc=bool(kw["flag_qc"]), f_qi=bool(kw["flag_qi"]), flag_bep=False)
    # ysu(): the CUDA wrapper's own name translation, applied identically to the native per-column arrays.
    state = {name: value for name, value in data.items() if name in ysu.ARRAY_ARGUMENTS}
    state.update({ysu._WRAPPER_FIELDS[name]: value for name, value in data.items() if name in ysu._WRAPPER_FIELDS})
    nz, ncol = state["tx"].shape
    rows = []
    for version in VERSIONS:
        device = {k: cp.asarray(v) for k, v in data.items()}
        cp.cuda.Device().synchronize()
        out = ysu.ysu(device, {k: np.float32(v) for k, v in meta["constants"].items()}, dt=np.float32(kw["dt"]),
                      flag_qc=flags["f_qc"], flag_qi=flags["f_qi"], ysu_topdown_pblmix=topdown,
                      device_math_header=header, flag_bep=False, mmm_version=version)
        gpu = {k: cp.asnumpy(v) for k, v in out.items()}
        native = {name: np.empty_like(gpu[name]) for name in gpu}
        lib = natives[version]
        if FTZ is not None:
            saved_csr = FTZ.l5_csr_get(); FTZ.l5_csr_set(saved_csr | 0x8000)
        nflags = dict(its=1, ite=1, kte=nz, kme=nz + 1, nmix=0, f_qc=int(flags["f_qc"]), f_qi=int(flags["f_qi"]),
                      flag_bep=0, ysu_topdown_pblmix=topdown)
        for col in range(ncol):
            arrays = {name: None for name in ysu.ARRAY_ARGUMENTS}
            for name, value in state.items():
                if name in arrays:
                    arrays[name] = np.ascontiguousarray(value[:, col] if value.ndim == 2 else value[col:col + 1])
            for name in ysu._OUTPUT_LEVEL + ysu._OUTPUT_SURFACE:
                if arrays[name] is None:
                    arrays[name] = np.zeros(nz if name in ysu._OUTPUT_LEVEL else 1,
                                            np.int32 if name == "kpbl1d" else np.float32)
            arrays["qmix"] = arrays["qmixtnp"] = np.empty(0, np.float32)
            res = lib.run(arrays, nflags, values)
            for name, raw in ysu._WRAPPER_OUTPUTS.items():
                native[name][..., col] = res[raw]
            native["rthblten"][:, col] = res["ttnp"]
        if FTZ is not None:
            FTZ.l5_csr_set(saved_csr)
        fields = []
        for name in sorted(gpu):
            g, n = words(gpu[name]), words(native[name])
            fields.append(dict(name=name, words=int(g.size), different=int(np.count_nonzero(g != n))))
        rows.append(dict(version=version, ysu_topdown_pblmix=topdown, columns=int(ncol), levels=int(nz),
                         words=sum(f["words"] for f in fields), different=sum(f["different"] for f in fields),
                         fields=fields, outputs=gpu))
    a, b = rows[0].pop("outputs"), rows[1].pop("outputs")
    changed = {name: int(np.count_nonzero(words(a[name]) != words(b[name]))) for name in a}
    return dict(capture=os.path.basename(path), call=meta["call"], rows=rows,
                versions_differ_words=sum(changed.values()), versions_differ_fields={k: v for k, v in changed.items() if v})


def mm5_case(path, meta, natives, factor):
    data = dict(np.load(path))
    packed, constants = data["packed"], data["constants"]
    n = packed.shape[1]
    var2d = data.get("var2d")
    scheme = meta["kwargs"].get("scheme", "sf_monin_obukhov_rev")
    rows = []
    arms = [("a4baf7f", False), ("550b5b4", False)]
    if var2d is not None and var2d.shape == (n,) and scheme == mm5.SCHEMES[0]:
        arms.append(("550b5b4", True))
    outs = {}
    for version, tofd in arms:
        device = cp.asarray(packed)
        mm5.launch_columns(device, cp.asarray(constants), scheme=scheme,
                           tofd=(cp.asarray(var2d), factor) if tofd else None)
        gpu = cp.asnumpy(device)
        lib = natives[version]
        a = np.ascontiguousarray(packed.copy())
        lib.l5_mm5_run(a.ctypes.data_as(PTR), np.ascontiguousarray(constants, np.float32).ctypes.data_as(PTR),
                       ct.c_int(n), np.ascontiguousarray(var2d if var2d is not None else np.zeros(n, np.float32),
                                                         np.float32).ctypes.data_as(PTR),
                       ct.c_int(int(tofd)), ct.c_float(factor))
        g, w = words(gpu), words(a)
        per = [int(np.count_nonzero(g[f] != w[f])) for f in range(g.shape[0])]
        rows.append(dict(version=version, kim_tofd=tofd, columns=int(n), words=int(g.size),
                         different=int(sum(per)), different_by_row={i: d for i, d in enumerate(per) if d}))
        outs[(version, tofd)] = gpu
    diff = lambda x, y: int(np.count_nonzero(words(outs[x]) != words(outs[y])))
    extra = dict(pin_vs_550b5b4_off_words=diff(("a4baf7f", False), ("550b5b4", False)))
    if ("550b5b4", True) in outs:
        extra["tofd_on_vs_off_words"] = diff(("550b5b4", False), ("550b5b4", True))
        extra["var2d_gt_5m_columns"] = int(np.count_nonzero(var2d > 5.0))
    return dict(capture=os.path.basename(path), call=meta["call"], scheme=scheme, rows=rows, **extra)


def main():
    capture, native_dir, out = sys.argv[1:4]
    factor = float(sys.argv[sys.argv.index("--tofd-factor") + 1]) if "--tofd-factor" in sys.argv else 0.003
    t0 = time.time()
    global FTZ
    if "--native-ftz" in sys.argv:
        FTZ = ct.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ftz", "l5ftz.so"))
        FTZ.l5_csr_get.restype = ct.c_uint
        FTZ.l5_csr_set.argtypes = [ct.c_uint]
    header = ysu.shared_device_math_header()
    ysu_libs = {v: NativeYSU(os.path.join(native_dir, f"build-{v}", "libysu.so")) for v in VERSIONS}
    mm5_libs = {}
    for v in VERSIONS:
        lib = ct.CDLL(os.path.join(native_dir, f"build-{v}", "libmm5.so"))
        lib.l5_mm5_init()
        lib.l5_mm5_run.argtypes = [PTR, PTR, ct.c_int, PTR, ct.c_int, ct.c_float]
        mm5_libs[v] = lib
    report = dict(schema="l5-hex-mmm-replay-v1", capture_dir=capture, native_dir=native_dir, ysu=[], mm5=[],
                  native_ysu_ftz=FTZ is not None, numerics=os.environ.get("WOOF_HEX_NUMERICS", "fast(default)"))
    for path in sorted(glob.glob(os.path.join(capture, "ysu-*.npz"))):
        meta = json.load(open(path[:-4] + ".json"))
        report["ysu"].append(ysu_case(path, meta, ysu_libs, header, flip_pblmix=False))
        print(json.dumps({k: v for k, v in report["ysu"][-1].items() if k != "rows"}),
              [(r["version"], r["words"], r["different"]) for r in report["ysu"][-1]["rows"]], flush=True)
    for path in sorted(glob.glob(os.path.join(capture, "mm5-*.npz"))):
        meta = json.load(open(path[:-4] + ".json"))
        report["mm5"].append(mm5_case(path, meta, mm5_libs, factor))
        print(json.dumps({k: v for k, v in report["mm5"][-1].items() if k != "rows"}),
              [(r["version"], r["kim_tofd"], r["words"], r["different"]) for r in report["mm5"][-1]["rows"]], flush=True)
    total = [r for case in report["ysu"] + report["mm5"] for r in case["rows"]]
    report["words"] = sum(r["words"] for r in total)
    report["different"] = sum(r["different"] for r in total)
    report["seconds"] = round(time.time() - t0, 1)
    json.dump(report, open(out, "w"), indent=1)
    print("TOTAL words", report["words"], "different", report["different"], "seconds", report["seconds"])


if __name__ == "__main__":
    main()
