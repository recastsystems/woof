"""Run product BouLac columns and write field metrics and full-field hashes."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT))
from test_boulac_wrf461_parity import (
    COMPARED, DECLARED_COLUMN, DIVERGENT_UV, mismatches, run_grid)
from woof.verify.boulac_oracle import grids
from woof.core.kernels import _load_module_no_ftz, get_kernel, module_options, module_source
from woof.certify.compile_platform import compile_platform_fingerprint
from woof.certify.kernel_manifest import kernel_manifest
from woof.physics_params import registry_sha256
from woof.core.fp32_ulp import monotone_fp32_key

def ordered(a):
    return monotone_fp32_key(a)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = dict(strict=os.getenv("GPUWM_WRF_EXACT", "0") == "1",
                  tolerance_ulp=0, fields={}, grids=[], total_mismatches=0,
                  platform=compile_platform_fingerprint(),
                  constant_manifest_sha256=registry_sha256())
    for grid in grids():
        name, labels = grid["grid"], grid["labels"]
        inp, rec, results = run_grid(name)
        bad = mismatches(name, rec, results, labels)
        report["total_mismatches"] += len(bad)
        row = dict(grid=name, columns=inp["ncol"], steps=len(results),
                   mismatched_words=len(bad), fields={})
        for key in COMPARED + ("dlg_bep", "pblh"):
            got = np.concatenate([r[key].reshape(-1) for r in results])
            want = np.concatenate([r["out"][key].reshape(-1) for r in rec["steps"]])
            metric_mask = np.ones(got.shape, bool)
            if DECLARED_COLUMN in labels and key in DIVERGENT_UV:
                metric_mask &= np.isfinite(want)
            gg, ww = got[metric_mask], want[metric_mask]
            ulp = int(np.max(np.abs(ordered(gg)-ordered(ww))))
            delta = float(np.max(np.abs(gg.astype(np.float64)-ww)))
            words = int(gg.size)
            row["fields"][key] = dict(
                checked_words=words, excluded_wrf_nan_words=int((~metric_mask).sum()),
                max_ulp=ulp, max_abs=delta,
                product_sha256=hashlib.sha256(got.tobytes()).hexdigest(),
                wrf_sha256=hashlib.sha256(want.tobytes()).hexdigest())
            summary = report["fields"].setdefault(key, dict(checked_words=0,max_ulp=0,max_abs=0.))
            summary["checked_words"] += words
            summary["max_ulp"] = max(summary["max_ulp"], ulp)
            summary["max_abs"] = max(summary["max_abs"], delta)
        report["grids"].append(row)
    src = module_source("boulac")
    original = "#define BL_CK_B 0.40000000596046448f"
    assert src.count(original) == 1
    planted = src.replace(original, "#define BL_CK_B 0.40000003576278687f")
    mod = _load_module_no_ftz("boulac-planted", planted, module_options("boulac"))
    grid = next(g for g in grids() if g["grid"] == "rural40")
    inp, rec, results = run_grid("rural40", kernel=mod.get_function("boulac_column"))
    bad = mismatches("rural40", rec, results, grid["labels"])
    report["one_bit_change"] = dict(mismatched_words=len(bad),
                                    fields=sorted({b[1] for b in bad}))
    from cupy.cuda import driver
    k = get_kernel("boulac", "boulac_column")
    report["local_size_bytes"] = int(driver.funcGetAttribute(
        driver.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, k.ptr))
    report["kernel_manifest"] = kernel_manifest()
    report["passed"] = report["total_mismatches"] == 0 and len(bad) > 1000
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k:report[k] for k in ("strict","total_mismatches","one_bit_change","local_size_bytes","passed")}))
    return 0 if report["passed"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
