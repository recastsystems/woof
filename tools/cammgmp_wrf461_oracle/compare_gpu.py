"""Replay the CAMMGMP oracle deck through the GPU port and grade it.

Runs every group of the deck as a free-running chain on the device (the
port's own outputs carry from step to step; only the deck's float32 forcing
and t_phy = th*pi_phy are applied between steps, exactly as
run_cammgmp.F90 does) and compares each step's fields with the oracle,
word for word.  Prints one row per field: differing words, columns
touched and the largest float32 ULP distance.

The default is the unmodified WRF output and stock kernel. Pass
--corrected-ndrop only for the altered-source attribution instrument.
No generated sounding in this legacy deck is a forecast capture.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tools.cammgmp_wrf461_oracle.make_cases import OUT_2D, OUT_3D
from tools.cammgmp_wrf461_oracle.oracle_io import read_deck, read_outputs
from woof.core.fp32_ulp import monotone_fp32_key

#: oracle field name -> launcher field name
OUT_NAME = {"nc3d": "nc", "ni3d": "ni", "ns3d": "ns", "nr3d": "nr"}


def ulp_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """float32 total-order ULP distance (NaN-free inputs)."""
    return np.abs(monotone_fp32_key(a) - monotone_fp32_key(b))


def run_group(group, *, reproduce_defect=True):
    import cupy as cp
    from woof.core.cammgmp import cammgmp_step
    f3, f2 = group["fields3d"], group["fields2d"]
    nz, ncol = group["nz"], group["ncol"]
    dt = np.float32(group["dt"])

    def dev(a, nlev=None):
        a = a if nlev is None else a[:nlev]
        return cp.asarray(np.ascontiguousarray(a).reshape(a.shape[0], 1, ncol)
                          if a.ndim == 2 else a.reshape(1, ncol))

    inputs = {
        "p_hyd": dev(f3["p_hyd"], nz), "pi_phy": dev(f3["pi_phy"], nz),
        "z": dev(f3["z_sea_level"], nz), "alt": dev(f3["alt"], nz),
        "icwmrsh": dev(f3["icwmrsh3d"], nz), "icwmrdp": dev(f3["icwmrdp3d"], nz),
        "shfrc": dev(f3["shfrc3d"], nz),
        "p8w": dev(f3["p8w"]), "tke": dev(f3["tke_pbl"]), "kvh": dev(f3["kvh3d"]),
        "cmfmc": dev(f3["cmfmc3d"]), "cmfmc2": dev(f3["cmfmc2_3d"]),
        "ht": dev(f2["ht"]), "xland": dev(f2["xland"]), "snowh": dev(f2["snowh"]),
    }
    deck_name = {"nc": "nc3d", "ni": "ni3d", "ns": "ns3d", "nr": "nr3d"}
    from woof.core.cammgmp import CAMMGMP_INOUT_2D, CAMMGMP_INOUT_3D
    inout = {name: dev(f3[deck_name.get(name, name)], nz) for name in CAMMGMP_INOUT_3D}
    inout.update({name: dev(f2[name]) for name in CAMMGMP_INOUT_2D})
    t_phy = f3["t_phy"][:nz].copy()
    dthdt = f3["dthdt"][:nz]
    dqvdt = f3["dqvdt"][:nz]
    pi_phy = f3["pi_phy"][:nz]
    steps = []
    for istep in range(group["nstep"]):
        inputs["t_phy"] = dev(t_phy)
        cammgmp_step(inputs, inout, dt=float(dt), itimestep=group["it0"] + istep,
                     cu_coupled=group["cu"] == 7, shcu_coupled=group["shcu"] == 2,
                     accum_mode=float(group["accum"]),
                     aitken_mode=float(group["aitken"]),
                     coarse_mode=float(group["coarse"]),
                     _reproduce_ndrop_defect=reproduce_defect)
        out = {name: cp.asnumpy(a).reshape(a.shape[0], ncol) if a.ndim == 3
               else cp.asnumpy(a).reshape(ncol) for name, a in inout.items()}
        steps.append(out)
        # run_cammgmp.F90's between-step forcing, float32 op for op
        th = out["th"] + dthdt * dt
        qv = out["qv"] + dqvdt * dt
        t_phy = (th * pi_phy).astype(np.float32)
        inout["th"] = dev(th.astype(np.float32))
        inout["qv"] = dev(qv.astype(np.float32))
    return steps


def grade(oracle_groups, gpu_groups):
    if len(oracle_groups) != len(gpu_groups) or not oracle_groups:
        raise ValueError("oracle/GPU group count mismatch or empty deck")
    rows = {}
    for gi, (og, gg) in enumerate(zip(oracle_groups, gpu_groups)):
        if len(og["steps"]) != len(gg) or not gg:
            raise ValueError(f"group {gi}: oracle/GPU step count mismatch")
        for si, (os_, gs) in enumerate(zip(og["steps"], gg)):
            for name in OUT_3D + OUT_2D:
                ref = os_[name]
                got = gs[OUT_NAME.get(name, name)]
                if ref.shape != got.shape or ref.dtype != np.float32 or got.dtype != np.float32:
                    raise ValueError(f"group {gi} step {si} {name}: shape/dtype mismatch")
                if not np.isfinite(ref).all() or not np.isfinite(got).all():
                    raise ValueError(f"group {gi} step {si} {name}: non-finite value")
                bad = ref.view(np.int32) != got.view(np.int32)
                row = rows.setdefault(name, {"words": 0, "differ": 0, "cols": set(),
                                             "max_ulp": 0, "first": None})
                row["words"] += ref.size
                if bad.any():
                    row["differ"] += int(bad.sum())
                    cols = np.nonzero(bad.any(axis=0) if bad.ndim == 2 else bad)[0]
                    row["cols"].update((gi, int(c)) for c in cols)
                    u = int(ulp_distance(ref, got)[bad].max())
                    row["max_ulp"] = max(row["max_ulp"], u)
                    if row["first"] is None:
                        idx = np.argwhere(bad)[0]
                        row["first"] = dict(group=gi, step=si, index=idx.tolist(),
                                            ref=float(ref[tuple(idx)]),
                                            got=float(got[tuple(idx)]))
    for row in rows.values():
        row["cols"] = len(row["cols"])
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("build_dir")
    parser.add_argument("--corrected-ndrop", action="store_true",
                        help="altered-source attribution instrument only")
    parser.add_argument("--json")
    args = parser.parse_args()
    build = Path(args.build_dir)
    deck = read_deck(build / "cases.bin")
    oracle = read_outputs(build / ("fixed-out.bin" if args.corrected_ndrop else "pristine-out.bin"))
    gpu = [run_group(g, reproduce_defect=not args.corrected_ndrop) for g in deck]
    rows = grade(oracle, gpu)
    total = sum(r["differ"] for r in rows.values())
    words = sum(r["words"] for r in rows.values())
    for name, r in rows.items():
        print(f"{name:14s} differ {r['differ']:7d}/{r['words']:7d}  cols {r['cols']:3d}"
              f"  max_ulp {r['max_ulp']}  first {r['first']}")
    print(f"TOTAL differ {total} of {words} words")
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"referee": "fixed-instrument" if args.corrected_ndrop else "pristine",
             "stock_wrf_indexing": not args.corrected_ndrop, "rows": rows,
             "differ": total, "words": words}, indent=1) + "\n")
    raise SystemExit(1 if total else 0)


if __name__ == "__main__":
    main()
