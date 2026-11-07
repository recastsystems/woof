"""Record the surface replay's measured per-field differences."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
from test_temf_sfclay_wrf461_parity import replay, _ulp, _dry_cells, OUTPUT_NAMES
from temf_sfclay_oracle import load, dims
from woof.core.kernels import module_options, module_source

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    got = replay()
    dry = _dry_cells()
    report = {"mode": "strict" if os.environ.get("GPUWM_WRF_EXACT") == "1" else "default",
              "dimensions": dict(zip(("steps","columns"), dims())),
              "source_sha256": hashlib.sha256(module_source("temf_sfclay").encode()).hexdigest(),
              "compile_options": list(module_options("temf_sfclay")),
              "default_tolerance_ulp_against_crlog10": 0,
              "reference_arms": {}, "field_sha256": {},
              "declared_divergences": ["correctly rounded LOG10", "MAVAIL=0 finite EXCH_TEMF/CHS limits"],
              "pbl_feed": "recorded formulas, not a PBL kernel",
              "forecast": "see separate CLI receipt",
              "mol": "not written by TEMFSFCLAY; driver test preserves its bytes",
              "znt": "WRF input only; checked byte-identical",
              "constant_table_sha256": __import__("woof.core.temf_constants",fromlist=["TEMF_CONSTANTS_SHA256"]).TEMF_CONSTANTS_SHA256}
    for name,a in got.items():
        report["field_sha256"][name] = hashlib.sha256(a.tobytes()).hexdigest()
    for arm in ("wrf","wrf-crlog10"):
        want = load(arm)
        fields = {}
        for name,a in got.items():
            b = want["out:"+name]
            keep = ~dry if name in ("exch","chs") else np.ones_like(dry)
            dist = _ulp(a[keep],b[keep])
            finite = keep & np.isfinite(a) & np.isfinite(b)
            absolute = np.abs(a[finite].astype(np.float64)-b[finite].astype(np.float64))
            fields[name] = {"words":int(keep.sum()),
                "different_words":int((a.view(np.uint32)[keep]!=b.view(np.uint32)[keep]).sum()),
                "max_ulp":int(dist.max()),
                "max_absolute_error":float(absolute.max()) if absolute.size else 0.0,
                "declared_dry_cells":int((~keep).sum())}
        report["reference_arms"][arm] = fields
    args.out.write_text(json.dumps(report,indent=2)+"\n")
    bad = {name:v for name,v in report["reference_arms"]["wrf-crlog10"].items() if v["different_words"]}
    print(json.dumps({"mode":report["mode"],"different_fields_crlog10":bad,"out":str(args.out)}))
    return bool(bad)
if __name__ == "__main__":
    raise SystemExit(main())
