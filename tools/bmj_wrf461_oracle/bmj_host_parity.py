"""Grade the BMJ kernel source against the WRF column oracle on a CPU.

Builds bmj_host_harness.cpp around the exact string nvrtc receives
(``woof.core.kernels.module_source("bmj")``), runs BMJINIT and every arm
and step of the oracle through it, and reports the ULP distance per field
against both oracle arms (stock gamma, correctly rounded gamma).

    python tools/bmj_wrf461_oracle/bmj_host_parity.py ORACLE_DIR BUILD_DIR
        [--plant OLD=NEW] [--json OUT]

``--plant`` edits the assembled source once (a literal-for-literal swap) to
prove the comparison can fail: a one-bit change in one kernel constant must
move some output word.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bmj_oracle_io as io  # noqa: E402

TABLE_FLOATS = 87972


def build(build_dir: Path, plant: str | None) -> ctypes.CDLL:
    from woof.core.kernels import module_source

    src = module_source("bmj")
    if plant:
        old, new = plant.split("=", 1)
        if src.count(old) < 1:
            raise SystemExit(f"--plant: {old!r} not in the assembled source")
        src = src.replace(old, new, 1)
    build_dir.mkdir(parents=True, exist_ok=True)
    (build_dir / "bmj_assembled.cu").write_text(src, encoding="utf-8")
    lib = build_dir / ("libbmj_host_planted.so" if plant else "libbmj_host.so")
    subprocess.run(
        ["g++", "-std=c++17", "-O2", "-ffp-contract=off", "-fno-fast-math",
         "-fPIC", "-shared", "-I", str(build_dir),
         "-o", str(lib), str(HERE / "bmj_host_harness.cpp")], check=True)
    return ctypes.CDLL(str(lib))


def _p(a):
    return a.ctypes.data_as(ctypes.c_void_p)


def run(lib, oracle: Path) -> dict:
    f32 = np.float32
    ncol, nz = io.read_cases(oracle / "cases.bin")
    report = {"ncol": ncol, "nz": nz}
    tab = np.zeros(TABLE_FLOATS, dtype=f32)
    lib.host_bmj_init_tables(_p(tab), ctypes.c_float(1004.5),
                             ctypes.c_float(287.0))
    otab = np.fromfile(oracle / "tables.bin", dtype="<f4")[:TABLE_FLOATS]
    report["tables_max_ulp"] = int(io.ulp_distance(tab, otab).max())
    report["tables_words_differing"] = int((io.ulp_distance(tab, otab) > 0).sum())
    arms = {}
    for arm, (fb, dt, stepcu) in io.ARMS.items():
        for ref in ("crgamma", "stock"):
            steps = io.read_arm(oracle / f"{ref}-arm{arm}.bin", ncol, nz)
            cld = steps[0]["in_cldefi"].astype(f32).copy()
            worst = {}
            paths: dict[int, int] = {}
            for st in steps:
                lev = {k: np.zeros((nz, ncol), f32) for k in io.LEVEL_OUTPUTS}
                sur = {k: np.zeros(ncol, f32) for k in
                       ("raincv", "pratec", "cutop", "cubot", "convcld")}
                path = np.zeros(ncol, np.int32)
                ins = [np.ascontiguousarray(st["in_" + k], dtype=f32)
                       for k in ("t", "qv", "pmid", "pi", "rho", "dz8w")]
                pint = np.ascontiguousarray(st["in_pint"], dtype=f32)
                xland = np.ascontiguousarray(st["in_xland"], dtype=f32)
                lib.host_bmj_column(
                    *[_p(a) for a in ins], _p(pint), _p(xland), _p(cld),
                    _p(tab), *[_p(lev[k]) for k in io.LEVEL_OUTPUTS],
                    *[_p(sur[k]) for k in ("raincv", "pratec", "cutop",
                                           "cubot", "convcld")],
                    _p(path), ctypes.c_float(dt), ctypes.c_int(stepcu),
                    ctypes.c_int(int(fb)), ctypes.c_int(ncol),
                    ctypes.c_int(nz))
                got = dict(lev)
                got.update(sur)
                got["cldefi"] = cld
                for k in io.LEVEL_OUTPUTS + io.SURFACE_OUTPUTS:
                    worst[k] = max(worst.get(k, 0),
                                   int(io.ulp_distance(got[k], st[k]).max()))
                for p in path.tolist():
                    paths[p] = paths.get(p, 0) + 1
            arms[f"arm{arm}-{ref}"] = {"max_ulp": worst,
                                       "paths": dict(sorted(paths.items()))}
    report["arms"] = arms
    return report


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("oracle", type=Path)
    ap.add_argument("build", type=Path)
    ap.add_argument("--plant")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)
    lib = build(a.build, a.plant)
    rep = run(lib, a.oracle)
    rep["plant"] = a.plant
    text = json.dumps(rep, indent=1)
    print(text)
    if a.json:
        a.json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
