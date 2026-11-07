#!/usr/bin/env python3
"""Read the MSKF oracle fixture and report what it covers.

usage: validate_mskf_oracle.py BUILD_DIR

Prints one line per case (regime reached, cloud base/top, hold, rain) and a
coverage summary, and writes coverage.csv beside the fixture.  The regime is
read from WRF's own outputs: cldfra_dp > 0 somewhere means the deep branch
(module_cu_mskf.F:6323-6333), cldfra_sh > 0 the shallow branch
(:6311-6321), an untouched sentinel NCA the held-column skip (:3843), and
a zero CUTOP/CUBOT pair with zero tendencies a column that never convected.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

FIELDS_3D = ("rthcuten", "rqvcuten", "rqccuten", "rqrcuten", "rqicuten",
             "rqscuten", "rucuten", "rvcuten", "cldfra_dp", "cldfra_sh",
             "qc_cu", "qi_cu", "qr_cu", "qs_cu", "nc_cu", "ni_cu", "nr_cu",
             "ns_cu", "ccn_cu", "efcs", "efis", "efss", "udr_kf", "ddr_kf",
             "uer_kf", "der_kf", "w0avg")
FIELDS_2D = ("raincv", "pratec", "nca", "cutop", "cubot", "cu_act_flag",
             "ainc_frac", "timec_kf")
SENTINEL = np.float32(-7777.0)


def load(build: Path):
    meta = json.loads((build / "cases.json").read_text())
    nz, n = meta["nz"], meta["ncase"]
    raw = np.fromfile(build / "out_f32.bin", "<f4")
    width = len(FIELDS_3D) * nz + len(FIELDS_2D)
    raw = raw.reshape(n, width)
    out = []
    for c in range(n):
        rec = {}
        for i, name in enumerate(FIELDS_3D):
            rec[name] = raw[c, i * nz:(i + 1) * nz]
        base = len(FIELDS_3D) * nz
        for i, name in enumerate(FIELDS_2D):
            rec[name] = raw[c, base + i]
        out.append(rec)
    return meta, out


def regime(rec) -> str:
    if rec["cu_act_flag"] == 0.0:
        return "held"
    dp = rec["cldfra_dp"]
    sh = rec["cldfra_sh"]
    if np.any((dp > 0) & (dp != SENTINEL)):
        return "deep"
    if np.any((sh > 0) & (sh != SENTINEL)):
        return "shallow"
    if np.any(rec["rthcuten"] != 0.0):
        return "tendency-only"
    return "none"


def main(build: Path) -> None:
    meta, out = load(build)
    rows = []
    for c, (case, rec) in enumerate(zip(meta["cases"], out), start=1):
        rows.append({
            "case": c, "sounding": case["sounding"], "dx": case["dx"],
            "phase": case["phase"], "aercu_opt": case["aercu_opt"],
            "aerosol": case["aerosol"], "regime": regime(rec),
            "cubot": float(rec["cubot"]), "cutop": float(rec["cutop"]),
            "nca": float(rec["nca"]), "pratec": float(rec["pratec"]),
            "max_abs_rthcuten": float(np.max(np.abs(rec["rthcuten"]))),
            "max_abs_rucuten": float(np.max(np.abs(rec["rucuten"]))),
            "ainc_frac": float(rec["ainc_frac"]),
        })
    with open(build / "coverage.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]),
                           lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["regime"]] = counts.get(r["regime"], 0) + 1
        print(f"{r['case']:4d} {r['sounding']:17s} dx={r['dx']:8.0f} "
              f"{r['phase']:9s} aercu={r['aercu_opt']} {r['aerosol']:8s} "
              f"{r['regime']:8s} base={r['cubot']:5.0f} top={r['cutop']:5.0f}"
              f" nca={r['nca']:8.1f} pratec={r['pratec']:.3e}")
    print("regimes:", counts)
    rain = sum(1 for r in rows if r["pratec"] > 0)
    cmt = sum(1 for r in rows if r["max_abs_rucuten"] > 0
              and r["max_abs_rucuten"] != 7777.0)
    print(f"raining columns {rain}, columns with momentum transport {cmt}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
