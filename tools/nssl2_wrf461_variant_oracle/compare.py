"""Compare GPUWM NSSL variant outputs against the WRF v4.6.1 oracle.

For every field this reports the maximum ULP distance (float32 ordered-int
metric, sign-aware), the count of differing cells, and the maximum relative
and absolute differences.  The strict arm's acceptance is 0 ULP on every
field; the default arm is judged against the declared tolerance table below
(the composed-replay thresholds already admitted for the shipped mp18 lane,
declared before any variant result was inspected).

Usage:
    python compare.py ORACLE_DIR GPU_DIR REPORT.json [--arm strict|default]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

FIELDS = (
    "theta", "qv", "qc", "qr", "qi", "qs", "qg", "qh",
    "qndrop", "qnr", "qni", "qns", "qng", "qnh", "qnn",
    "qvolg", "qvolh", "qzr", "qzg", "qzh", "refl_10cm",
    "effc_m", "effi_m", "effs_m",
    "rainnc", "rainncv", "snownc", "snowncv", "graupelnc",
    "graupelncv", "hailnc", "hailncv", "sr",
)

MASS = {"qv", "qc", "qr", "qi", "qs", "qg", "qh"}
NUMBER = {"qndrop", "qnr", "qni", "qns", "qng", "qnh", "qnn"}
VOLUME = {"qvolg", "qvolh"}
SIXTH = {"qzr", "qzg", "qzh"}
RADIUS = {"effc_m", "effi_m", "effs_m"}
PRECIP = {"rainnc", "rainncv", "snownc", "snowncv", "graupelnc",
          "graupelncv", "hailnc", "hailncv", "sr"}
STATE_FIELDS = FIELDS[:20]
FIELDS += tuple(f"tendency_{name}" for name in STATE_FIELDS)


def tolerance(name: str) -> tuple[float, float]:
    """(rtol, atol) for the default arm, from compare_replay.py."""
    if name.startswith("tendency_"):
        # Net per-step changes, in state units per second. This is a stated
        # test budget, not evidence that a mode attains it.
        return tolerance(name.removeprefix("tendency_"))
    if name == "theta":
        return 3.0e-6, 5.0e-5
    if name in MASS:
        return 3.0e-5, 3.0e-9
    if name in NUMBER:
        return 3.0e-5, (5.0e-2 if name == "qni" else 64.0)
    if name in VOLUME:
        return 3.0e-5, 3.0e-12
    if name in SIXTH:
        return 3.0e-5, 1.0e-24
    if name == "refl_10cm":
        return 1.0e-4, 5.0e-2
    if name in RADIUS:
        return 3.0e-5, 3.0e-9
    if name in PRECIP:
        return 5.0e-5, 1.0e-6
    raise KeyError(name)


def _ordered(values: np.ndarray) -> np.ndarray:
    bits = values.astype(np.float32).view(np.int32).astype(np.int64)
    return np.where(bits < 0, -(bits & 0x7FFFFFFF), bits)


def ulp_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(_ordered(a) - _ordered(b))


def compare_mode(wrf: dict, gpu: dict, arm: str) -> dict:
    if arm not in ("strict", "default"):
        raise ValueError(f"unknown comparison arm {arm!r}")
    fields = {}
    worst = 0
    tol_ok = True
    bitwise_ok = True
    for name in FIELDS:
        a = np.asarray(wrf[name])
        b = np.asarray(gpu[name])
        if a.dtype != np.float32 or b.dtype != np.float32:
            raise TypeError(f"{name}: comparison requires float32 fields")
        if a.shape != b.shape:
            raise ValueError(f"{name}: shape {a.shape} vs {b.shape}")
        if not a.size:
            raise ValueError(f"{name}: empty field")
        finite = np.isfinite(a) & np.isfinite(b)
        nonfinite = int(np.sum(~finite))
        nan_mismatch = int(np.sum(np.isnan(a) != np.isnan(b)))
        # ULP alone equates signed zeros. Strict means identical float images.
        images_differ = a.view(np.uint32) != b.view(np.uint32)
        ulp = ulp_distance(np.where(finite, a, 0), np.where(finite, b, 0))
        diff = np.zeros(a.shape, np.float64)
        np.subtract(a.astype(np.float64), b.astype(np.float64),
                    out=diff, where=finite)
        diff = np.abs(diff)
        scale = np.maximum(np.abs(np.where(finite, a.astype(np.float64), 0)), 1e-300)
        rtol, atol = tolerance(name)
        within = finite & (diff <= atol + rtol * np.abs(a.astype(np.float64)))
        entry = {
            "max_ulp": int(ulp.max()) if ulp.size else 0,
            "cells_differing": int(np.sum(images_differ)),
            "cells": int(ulp.size),
            "max_abs": float(diff.max()) if diff.size else 0.0,
            "max_rel": float((diff / scale).max()) if diff.size else 0.0,
            "within_default_tolerance": bool(within.all()) and nan_mismatch == 0,
            "nan_mismatch": nan_mismatch,
            "nonfinite_cells": nonfinite,
        }
        if entry["cells_differing"]:
            idx = np.unravel_index(int(np.argmax(ulp)), ulp.shape)
            entry["worst_index"] = [int(i) for i in idx]
            entry["worst_wrf"] = float(a[idx]) if finite[idx] else str(a[idx])
            entry["worst_gpu"] = float(b[idx]) if finite[idx] else str(b[idx])
        worst = max(worst, entry["max_ulp"])
        tol_ok = tol_ok and entry["within_default_tolerance"]
        bitwise_ok = bitwise_ok and nonfinite == 0 and not images_differ.any()
        fields[name] = entry
    passed = bitwise_ok if arm == "strict" else tol_ok
    return {"arm": arm, "max_ulp": worst, "within_default_tolerance": tol_ok,
            "passed": passed, "fields": fields}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("oracle_dir")
    parser.add_argument("gpu_dir")
    parser.add_argument("report")
    parser.add_argument("--arm", choices=("strict", "default"),
                        default="strict")
    parser.add_argument("--modes", default="",
                        help="Explicit subset; default requires every declared mode")
    args = parser.parse_args()
    report = {}
    status = 0
    from cases import MODE_BY_LABEL
    labels = args.modes.split(",") if args.modes else list(MODE_BY_LABEL)
    if not labels or any(label not in MODE_BY_LABEL for label in labels):
        raise ValueError("unknown or empty mode selection")
    expected = set(labels)
    actual = {p.stem for p in Path(args.gpu_dir).glob("*.npz")}
    for label in sorted(expected - actual):
        report[label] = {"passed": False, "error": "missing GPU output"}
        print(f"{label:28s} FAIL missing GPU output")
        status = 1
    for label in sorted(actual - expected):
        report[label] = {"passed": False, "error": "unexpected GPU output"}
        print(f"{label:28s} FAIL unexpected GPU output")
        status = 1
    for gpu_path in sorted(Path(args.gpu_dir).glob("*.npz")):
        label = gpu_path.stem
        if label not in expected:
            continue
        oracle_path = Path(args.oracle_dir) / f"{label}.npz"
        if not oracle_path.is_file():
            report[label] = {"passed": False, "error": "missing WRF output"}
            print(f"{label:28s} FAIL missing WRF output")
            status = 1
            continue
        try:
            with np.load(oracle_path, allow_pickle=False) as o, np.load(gpu_path, allow_pickle=False) as g:
                if "refused" in g.files:
                    report[label] = {"refused": str(g["refused"]), "passed": False}
                    print(f"{label:28s} REFUSED: {g['refused']}")
                    status = 1
                    continue
                if str(o['mode_label']) != label or str(g['mode_label']) != label:
                    raise ValueError('mode label differs from filename')
                if not np.array_equal(o['settings'], g['settings']):
                    raise ValueError('different run settings')
                if o['dt'].tobytes() != g['dt'].tobytes():
                    raise ValueError('different time steps')
                from cases import REGIMES, NZ
                steps = int(o['settings'][0])
                expected_inputs = {
                    n.removeprefix('in_'): hashlib.sha256(o[n].tobytes()).hexdigest()
                    for n in o.files if n.startswith('in_')}
                expected_inputs.update({
                    n.removeprefix('env_'): hashlib.sha256(o[n].tobytes()).hexdigest()
                    for n in o.files if n.startswith('env_')})
                if expected_inputs != json.loads(str(g['input_sha256'])):
                    raise ValueError('WRF and GPU inputs differ')
                wrf = {n: o[f"wrf_{n}"] for n in FIELDS}
                gpu = {n: g[f"gpu_{n}"] for n in FIELDS}
                for name in FIELDS:
                    shape = (steps, len(REGIMES)) if name in PRECIP else (steps, len(REGIMES), NZ)
                    if wrf[name].shape != shape or gpu[name].shape != shape:
                        raise ValueError(f'{name}: incomplete trajectory shape, expected {shape}')
        except (KeyError, ValueError, TypeError, OSError) as exc:
            report[label] = {"passed": False, "error": str(exc)}
            print(f"{label:28s} FAIL invalid artifact: {exc}")
            status = 1
            continue
        result = compare_mode(wrf, gpu, args.arm)
        report[label] = result
        bad = [n for n, e in result["fields"].items() if e["max_ulp"]]
        print(f"{label:28s} {'PASS' if result['passed'] else 'FAIL'} "
              f"max_ulp={result['max_ulp']} tol_ok="
              f"{result['within_default_tolerance']} differing={bad}")
        status |= 0 if result["passed"] else 1
    Path(args.report).write_text(json.dumps(report, indent=1, sort_keys=True))
    return status


if __name__ == "__main__":
    sys.exit(main())
