#!/usr/bin/env python3
"""Measure WOOF's SBU-YLin column (mp_physics=13) against its WRF oracle.

The fixture (``woof/data/sbu_ylin/oracle``) holds the case file and the two
reference outputs build.sh wrote: ``pristine`` from the byte-unmodified WRF
v4.6.1 module and ``declared`` from the same module with WOOF's declared
fixes D1-D3 applied.  Every column is launched through the production
launcher (:func:`woof.core.column_microphysics.launch_column_kernel` with
the registry's own ``column_kernel`` row), grouped by time step because
``DT_IN`` is a scalar of the call, and every output word -- th, qv, qc, qr,
qi, qs, rimi per level and RAINNC, RAINNCV per column -- is compared.

Run under ``GPUWM_WRF_EXACT=1`` for the strict arm (the 0-ULP proof against
``pristine``) and without it for the default arm.  ``--plant`` recompiles
the kernel with one literal moved by one ULP (the sedimentation CFL factor
0.9f -> 0.90000004f) and must report differences: a comparator that cannot
see a one-ULP change in the source proves nothing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from woof.core.fp32_ulp import fp32_ulp_distance  # noqa: E402

DEFAULT_FIXTURE_DIR = REPO_ROOT / "woof" / "data" / "sbu_ylin" / "oracle"
INPUT_FIELDS = ("th", "qv", "ql", "qr", "qi", "qs", "rho", "pii", "p", "z",
                "dz8w")
LEVEL_OUTPUTS = ("th", "qv", "qc", "qr", "qi", "qs", "rimi")
SURFACE_OUTPUTS = ("rainnc", "rainncv")
PLANT_OLD = "__fdiv_rn(0.9f * dzk, v)"
PLANT_NEW = "__fdiv_rn(0.90000004f * dzk, v)"


def _words(line: str) -> np.ndarray:
    return np.array([int(x, 16) for x in line.split()],
                    dtype=np.uint32).view(np.float32)


def read_cases(path: Path) -> dict:
    lines = path.read_text(encoding="ascii").splitlines()
    ncol, nz = (int(x) for x in lines[0].split())
    head = np.empty((ncol, 3), np.float32)
    levels = np.empty((ncol, nz, len(INPUT_FIELDS)), np.float32)
    i = 1
    for c in range(ncol):
        head[c] = _words(lines[i])
        i += 1
        for k in range(nz):
            levels[c, k] = _words(lines[i])
            i += 1
    return {"ncol": ncol, "nz": nz, "dt": head[:, 0], "ht": head[:, 1],
            "rainnc": head[:, 2], "levels": levels}


def read_outputs(path: Path) -> dict:
    lines = path.read_text(encoding="ascii").splitlines()
    ncol, nz = (int(x) for x in lines[0].split())
    surface = np.empty((ncol, 2), np.float32)
    levels = np.empty((ncol, nz, len(LEVEL_OUTPUTS)), np.float32)
    i = 1
    for c in range(ncol):
        surface[c] = _words(lines[i])
        i += 1
        for k in range(nz):
            levels[c, k] = _words(lines[i])
            i += 1
    return {"surface": surface, "levels": levels}


def run_port(cases: dict, *, plant: bool = False) -> dict:
    """Launch every column through the production door; return its outputs."""
    import cupy as cp

    from woof.core.column_microphysics import (
        column_kernel_row, launch_column_kernel)

    row = column_kernel_row(13)
    if row is None:
        raise SystemExit("the registry publishes no column_kernel row for 13")
    ncol, nz = cases["ncol"], cases["nz"]
    surface = np.empty((ncol, 2), np.float32)
    levels = np.empty((ncol, nz, len(LEVEL_OUTPUTS)), np.float32)
    planted = _planted_kernel() if plant else None
    for dt in np.unique(cases["dt"]):
        cols = np.flatnonzero(cases["dt"] == dt)
        lv = cases["levels"][cols]                       # (n, nz, field)

        def vol(j):
            return cp.asarray(np.ascontiguousarray(
                lv[:, :, j].T[:, None, :], dtype=np.float32))

        def plane(a):
            return cp.asarray(np.ascontiguousarray(
                a[cols][None, :], dtype=np.float32))

        theta = vol(0)
        species = {"qv": vol(1), "qc": vol(2), "qr": vol(3), "qi": vol(4),
                   "qs": vol(5)}
        state = {"rimi": cp.full_like(theta, -999.0)}
        prep = {"rho": vol(6), "pii": vol(7), "p": vol(8), "z": vol(9),
                "dz8w": vol(10), "ht": plane(cases["ht"])}
        surf = {"mp_rainnc": plane(cases["rainnc"]),
                "mp_rainncv": cp.full_like(plane(cases["rainnc"]), -999.0),
                "mp_sr": cp.zeros_like(plane(cases["rainnc"]))}
        if planted is None:
            launch_column_kernel(row, theta, species, state, prep, surf,
                                 float(dt))
        else:
            n = len(cols)
            args = ((theta,) + tuple(species[s] for s in row.species)
                    + (state["rimi"],)
                    + tuple(prep[s] for s in ("rho", "pii", "p", "z", "dz8w"))
                    + (prep["ht"],)
                    + tuple(surf[s] for s in row.surface_slots)
                    + (np.float32(dt), np.int32(nz), np.int32(1),
                       np.int32(n)))
            planted(((n + 63) // 64,), (64,), args)
        outs = (theta, species["qv"], species["qc"], species["qr"],
                species["qi"], species["qs"], state["rimi"])
        for j, arr in enumerate(outs):
            levels[cols, :, j] = cp.asnumpy(arr)[:, 0, :].T
        surface[cols, 0] = cp.asnumpy(surf["mp_rainnc"])[0]
        surface[cols, 1] = cp.asnumpy(surf["mp_rainncv"])[0]
    return {"surface": surface, "levels": levels}


def _planted_kernel():
    import cupy as cp

    from woof.core.kernels import module_source

    src = module_source("sbu_ylin")
    if src.count(PLANT_OLD) != 1:
        raise SystemExit("plant site not found exactly once in sbu_ylin.cu")
    from woof.core.kernels import compile_noftz_module
    mod = compile_noftz_module("sbu_ylin", src.replace(PLANT_OLD, PLANT_NEW),
                              "sbu_ylin:planted")
    return mod.get_function("sbu_ylin_column")


def compare(got: dict, want: dict) -> dict:
    report = {}
    pairs = [(name, got["levels"][:, :, j], want["levels"][:, :, j])
             for j, name in enumerate(LEVEL_OUTPUTS)]
    pairs += [(name, got["surface"][:, j], want["surface"][:, j])
              for j, name in enumerate(SURFACE_OUTPUTS)]
    for name, g, w in pairs:
        g = np.ascontiguousarray(g, np.float32)
        w = np.ascontiguousarray(w, np.float32)
        ulp = fp32_ulp_distance(g, w)
        words = int(np.count_nonzero(g.view(np.uint32) != w.view(np.uint32)))
        worst = int(np.argmax(ulp))
        report[name] = {
            "max_ulp": int(ulp.max()),
            "words_differing": words,
            "words": int(g.size),
            "worst_index": [int(i) for i in np.unravel_index(worst, g.shape)],
        }
    report["_total_words_differing"] = sum(
        v["words_differing"] for k, v in report.items() if not k.startswith("_"))
    report["_max_ulp"] = max(
        v["max_ulp"] for k, v in report.items() if not k.startswith("_"))
    return report


def measure(fixture_dir: Path = DEFAULT_FIXTURE_DIR, *,
            plant: bool = False) -> dict:
    cases = read_cases(fixture_dir / "sbu-ylin-cases.txt")
    got = run_port(cases, plant=plant)
    from woof.wrf_exact import ENABLED
    result = {"strict": bool(ENABLED), "planted": plant,
              "columns": cases["ncol"], "levels": cases["nz"]}
    for ref in ("declared", "pristine"):
        result[ref] = compare(
            got, read_outputs(fixture_dir / f"sbu-ylin-{ref}.txt"))
    # Historical formula changes remain negative controls, never the target.
    pristine = read_outputs(fixture_dir / "sbu-ylin-pristine.txt")
    declared = read_outputs(fixture_dir / "sbu-ylin-declared.txt")
    result["declared_vs_pristine"] = compare(declared, pristine)
    names = (fixture_dir / "sbu-ylin-case-names.txt").read_text(
        encoding="ascii").split()
    diff_cols = sorted({names[c] for c in np.flatnonzero(
        (got["levels"].view(np.uint32) != declared["levels"].view(np.uint32))
        .any(axis=(1, 2)) | (got["surface"].view(np.uint32)
                             != declared["surface"].view(np.uint32)).any(axis=1))})
    result["columns_differing_from_declared"] = diff_cols
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE_DIR)
    parser.add_argument("--plant", action="store_true")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    result = measure(args.fixture, plant=args.plant)
    arm = "strict" if result["strict"] else "default"
    print(f"SBU-YLin column oracle, {arm} arm"
          + (", PLANTED one-ULP literal" if args.plant else "")
          + f": {result['columns']} columns x {result['levels']} levels")
    for ref in ("declared", "pristine"):
        rep = result[ref]
        print(f"  vs {ref}: max ULP {rep['_max_ulp']}, "
              f"{rep['_total_words_differing']} words differ")
        for name in LEVEL_OUTPUTS + SURFACE_OUTPUTS:
            r = rep[name]
            print(f"    {name:8s} max_ulp {r['max_ulp']:>10d}  "
                  f"words {r['words_differing']:>5d}/{r['words']}")
    print("  columns differing from declared:",
          ", ".join(result["columns_differing_from_declared"]) or "none")
    if args.json:
        args.json.write_text(json.dumps(result, indent=1) + "\n",
                             encoding="utf-8")
    return int(result["pristine"]["_total_words_differing"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())
