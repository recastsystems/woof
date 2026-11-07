"""Bitwise comparison of WOOF's WDM5 against the WRF v4.6.1 oracle.

usage: python3 compare.py FIXTURE_DIR REFERENCE_DIR WOOF_DIR [--json OUT]
                          [--columns-only REGEX]

For every configuration ``<name>.in`` in FIXTURE_DIR, reads the reference
``<name>.out`` (written by run_wdm5_stock) and WOOF's
``<name>.out`` and reports, for every output field of every step, how many
values differ in their bits, the largest FP32 ULP distance (the repository's
one certified definition, woof.verify.state_equiv.fp32_signed_ulp) and the
largest absolute difference.  A match means identical bits everywhere.

Effective radii: WRF publishes metres and WRF's RRTMG forms microns as
``re*1.e6`` in REAL (module_ra_rrtmg_lw.F, module_ra_rrtmg_sw.F); WOOF stores
that micron value.  The reference column is therefore converted with exactly
that one float32 multiply before the comparison, never a float64 one.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from oracle_io import (OUTPUT_LEVEL_FIELDS, OUTPUT_SURFACE_FIELDS,  # noqa: E402
                       read_input, read_output)

_RADII = ("re_cloud", "re_ice", "re_snow")


def _ulp(candidate, reference):
    from woof.verify.state_equiv import fp32_signed_ulp
    return fp32_signed_ulp(candidate, reference)


def compare_arrays(woof, ref):
    woof = np.asarray(woof, np.float32)
    ref = np.asarray(ref, np.float32)
    differ = woof.view(np.uint32) != ref.view(np.uint32)
    ulp = np.abs(_ulp(woof, ref))
    with np.errstate(invalid="ignore", over="ignore"):
        absdiff = np.abs(woof.astype(np.float64) - ref.astype(np.float64))
    absdiff = np.where(differ, absdiff, 0.0)
    return differ, ulp, absdiff


def compare_config(fixture: Path, refdir: Path, woofdir: Path, name: str,
                   column_names: list[str], column_mask=None,
                   skip=(), fortran_pair: bool = False) -> dict:
    case = read_input(fixture / f"{name}.in")
    ref = read_output(refdir / f"{name}.out", case.ncol, case.nz, case.nsteps)
    woof = read_output(woofdir / f"{name}.out", case.ncol, case.nz, case.nsteps)
    mask = (np.ones(case.ncol, bool) if column_mask is None else column_mask)
    fields = {}
    columns_differing = np.zeros(case.ncol, bool)
    first_step_of_column = np.full(case.ncol, -1)
    total_values = 0
    total_differ = 0
    for step, ((rl, rs), (wl, ws)) in enumerate(zip(ref, woof)):
        for fname in OUTPUT_LEVEL_FIELDS + OUTPUT_SURFACE_FIELDS:
            if fname in skip:
                continue
            level = fname in OUTPUT_LEVEL_FIELDS
            r = (rl if level else rs)[fname]
            w = (wl if level else ws)[fname]
            if fname in _RADII and not fortran_pair:
                r = (r.astype(np.float32) * np.float32(1.0e6)).astype(np.float32)
            differ, ulp, absdiff = compare_arrays(w, r)
            if level:
                differ = differ & mask[None, :]
                colhit = differ.any(axis=0)
            else:
                differ = differ & mask
                colhit = differ
            ulp = np.where(differ, ulp, 0)
            absdiff = np.where(differ, absdiff, 0.0)
            newly = colhit & (first_step_of_column < 0)
            first_step_of_column[newly] = step
            columns_differing |= colhit
            entry = fields.setdefault(fname, {"values": 0, "differ": 0,
                                              "max_ulp": 0, "max_abs": 0.0,
                                              "worst": None})
            n = int(mask.sum()) * (case.nz if level else 1)
            entry["values"] += n
            entry["differ"] += int(differ.sum())
            total_values += n
            total_differ += int(differ.sum())
            if differ.any():
                idx = np.unravel_index(int(np.argmax(ulp)), ulp.shape)
                mu = int(ulp[idx])
                if mu > entry["max_ulp"]:
                    col = idx[-1]
                    entry["max_ulp"] = mu
                    entry["worst"] = {
                        "step": step, "column": int(col),
                        "column_name": column_names[col],
                        "level": int(idx[0]) if level else None,
                        "woof": float(w[idx]), "wrf": float(r[idx])}
                entry["max_abs"] = max(entry["max_abs"], float(absdiff.max()))
    return {"config": name, "ncol": int(mask.sum()), "nz": case.nz,
            "nsteps": case.nsteps,
            "delt": float(case.delt), "values": total_values,
            "differ": total_differ,
            "max_ulp": max(f["max_ulp"] for f in fields.values()),
            "columns_differing": [column_names[i] for i in
                                  np.flatnonzero(columns_differing)],
            "fields": fields}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("fixture", type=Path)
    ap.add_argument("reference", type=Path)
    ap.add_argument("woof", type=Path)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--columns-only", default=None,
                    help="regex over column names to restrict the comparison")
    ap.add_argument("--fortran-pair", action="store_true",
                    help="both directories are Fortran arms (radii in "
                         "metres on both sides)")
    ap.add_argument("--skip", action="append", default=[],
                    help="an output field to leave out (the host mirror "
                         "writes no reflectivity)")
    args = ap.parse_args(argv)
    names = [line.split()[1] for line in
             (args.fixture / "columns.txt").read_text().splitlines()]
    mask = None
    if args.columns_only:
        rx = re.compile(args.columns_only)
        mask = np.asarray([bool(rx.search(n)) for n in names])
    results = []
    for inp in sorted(args.fixture.glob("*.in")):
        res = compare_config(args.fixture, args.reference, args.woof,
                             inp.stem, names, mask, tuple(args.skip),
                             args.fortran_pair)
        results.append(res)
        print(f"{res['config']:24s} ncol={res['ncol']:3d} steps={res['nsteps']:2d} "
              f"values={res['values']:8d} differ={res['differ']:7d} "
              f"max_ulp={res['max_ulp']} columns_differing="
              f"{len(res['columns_differing'])}")
        for fname, f in res["fields"].items():
            if f["differ"]:
                w = f["worst"]
                print(f"    {fname:11s} differ={f['differ']:6d} "
                      f"max_ulp={f['max_ulp']:<11d} max_abs={f['max_abs']:.3e} "
                      f"worst={w['column_name']} k={w['level']} "
                      f"step={w['step']} woof={w['woof']:.9g} "
                      f"wrf={w['wrf']:.9g}")
    if args.json:
        args.json.write_text(json.dumps(results, indent=1))
    return 0 if all(r["differ"] == 0 for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
