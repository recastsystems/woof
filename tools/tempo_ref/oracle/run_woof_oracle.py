"""TEMPO column oracle: the WOOF side and the bitwise comparison.

Runs every column set of columns_in.bin through the engine's GPU TEMPO
runtime (woof.core.tempo_runtime.TempoRuntime, the code a forecast runs:
NVRTC module, pack, column kernel, unpack) and compares every output of
every step with TEMPO's Fortran (ref_out.bin from tempo_oracle_driver).

    python run_woof_oracle.py columns_in.bin columns_catalog.json ref_out.bin \
        --tables <TEMPO table dir> | --blob <tempo_blob.bin> \
        --out summary.json [--mode free|replay|both]

Modes:
  free    the GPU state evolves on its own for every step (the forecast case);
          a 0 ULP result here implies 0 ULP for each single step as well.
  replay  before every step the GPU state is reset to the Fortran state of the
          previous step, so each step's difference is measured on its own
          (no growth through feedback).

The arithmetic mode is whatever the process was started with:
GPUWM_WRF_EXACT=1 (strict build) or unset (default arithmetic).  The summary
records it together with the compile options CuPy was given.

Distance: ULP distance between float64 words on the ordered-integer line
(+0 and -0 are 0 apart), plus absolute and relative difference.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

STATE_IN = ("t", "p", "w", "dz", "qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr", "nc",
            "nwfa", "nifa", "ng", "qb")
UPDATED = ("t", "qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr", "nc", "nwfa", "nifa",
           "ng", "qb")
SURFACE = ("rain_precip", "ice_liquid_equiv_precip", "snow_liquid_equiv_precip",
           "graupel_liquid_equiv_precip", "frz_rain_precip", "frozen_fraction",
           "max_hail_diameter_sfc", "max_hail_diameter_column")
COLUMN = ("refl10cm", "re_cloud", "re_ice", "re_snow")
OUTPUTS = UPDATED + SURFACE + COLUMN


def read_columns(path):
    raw = np.fromfile(path, dtype="<f8")
    nsets = int(raw[:1].view("<i8")[0])
    i = 1
    sets = []
    for _ in range(nsets):
        ncol, nz, nsteps = (int(x) for x in raw[i:i + 3].view("<i8"))
        dt = float(raw[i + 3])
        i += 4
        fields = {}
        for name in STATE_IN:
            fields[name] = raw[i:i + ncol * nz].reshape(ncol, nz).copy()
            i += ncol * nz
        sets.append(dict(ncol=ncol, nz=nz, nsteps=nsteps, dt=dt, fields=fields))
    if i != raw.size:
        raise ValueError(f"{path}: {raw.size - i} trailing values")
    return sets


def read_reference(path, sets, key="ref"):
    raw = np.fromfile(path, dtype="<f8")
    i = 0
    for s in sets:
        ncol, nz = s["ncol"], s["nz"]
        steps = []
        for _ in range(s["nsteps"]):
            out = {}
            for name in UPDATED:
                out[name] = raw[i:i + ncol * nz].reshape(ncol, nz)
                i += ncol * nz
            for name in SURFACE:
                out[name] = raw[i:i + ncol].reshape(ncol, 1)
                i += ncol
            for name in COLUMN:
                out[name] = raw[i:i + ncol * nz].reshape(ncol, nz)
                i += ncol * nz
            steps.append(out)
        s[key] = steps
    if i != raw.size:
        raise ValueError(f"{path}: {raw.size - i} trailing values (layout mismatch)")


# TEMPO defect, documented and not copied: tempo_main sets every precipitation
# diagnostic to zero on entry but not frozen_fraction, and returns early
# (module_mp_tempo_main.F90:386, "if (.not. do_micro .and. .not.
# supersaturated) return") before frozen_fraction is assigned (line 747), so
# tempo_run copies an undefined value for such a column (line 381).  WOOF
# returns 0, which is what TEMPO's own formula gives for zero precipitation:
# (0 + 0 + 0) / (0 + 0 + 0 + 0 + r1) is exactly 0.  A nonzero Fortran value
# beside four precipitation totals that are all exactly zero therefore cannot
# come from the formula; such a cell is counted as "tempo_undefined" (WOOF
# must hold +0 there) instead of being compared.
PRECIP4 = ("rain_precip", "ice_liquid_equiv_precip", "snow_liquid_equiv_precip",
           "graupel_liquid_equiv_precip")


def undefined_frozen_fraction(ref_step, got_ff):
    """(mask of cells excused, count) for one step; WOOF must be +0 there."""
    zero = np.ones_like(ref_step["frozen_fraction"], dtype=bool)
    for n in PRECIP4:
        zero &= ref_step[n] == 0.0
    ff = ref_step["frozen_fraction"]
    mask = zero & (ff.view(np.int64) != 0) & (np.asarray(got_ff).view(np.int64) == 0)
    return mask


def ordered(a):
    i = np.ascontiguousarray(a, dtype=np.float64).view(np.int64)
    return np.where(i < 0, np.int64(-0x8000000000000000) - i, i)


def ulp_distance(a, b):
    oa, ob = ordered(a), ordered(b)
    same = (oa >= 0) == (ob >= 0)
    d_same = np.abs(oa - np.where(same, ob, oa)).astype(np.uint64)
    d_cross = np.abs(oa).astype(np.uint64) + np.abs(ob).astype(np.uint64)
    return np.where(same, d_same, d_cross)


class Stats:
    """Per (field) and per (regime, field) accumulators."""

    def __init__(self):
        self.d = {}

    def add(self, key, got, want):
        e = self.d.setdefault(key, dict(n=0, n_bitwise=0, max_ulp=0, max_abs=0.0,
                                        max_rel=0.0, nan_mismatch=0, first_miss=None))
        g = np.asarray(got, dtype=np.float64)
        w = np.asarray(want, dtype=np.float64)
        eq = g.view(np.int64) == w.view(np.int64)
        e["n"] += int(eq.size)
        e["n_bitwise"] += int(eq.sum())
        if eq.all():
            return None
        nan = np.isnan(g) | np.isnan(w)
        e["nan_mismatch"] += int((nan & ~eq).sum())
        ok = ~nan
        u = ulp_distance(np.where(ok, g, 0.0), np.where(ok, w, 0.0))
        e["max_ulp"] = max(e["max_ulp"], int(u.max()))
        with np.errstate(invalid="ignore", over="ignore"):
            ad = np.where(ok, np.abs(g - w), 0.0)
            rel = np.where(ok & (w != 0.0), ad / np.abs(np.where(w == 0.0, 1.0, w)),
                           np.where(ok & (ad > 0.0), np.inf, 0.0))
        e["max_abs"] = max(e["max_abs"], float(ad.max()))
        e["max_rel"] = max(e["max_rel"], float(rel.max()))
        return ~eq

    def table(self):
        return self.d


def run(args):
    if args.strict and os.environ.get("GPUWM_WRF_EXACT") != "1":
        raise SystemExit("--strict given but GPUWM_WRF_EXACT is not 1")
    if args.got_file:
        return compare_file(args)
    import cupy as cp
    import woof  # noqa: F401  (installs the strict compiler hook when selected)
    from woof import wrf_exact
    from woof.core.tempo_runtime import TempoConfig, TempoRuntime

    sets = read_columns(args.columns)
    read_reference(args.reference, sets)
    catalog = json.loads(Path(args.catalog).read_text())
    regime_of = {(r["set"], r["column"]): r["regime"] for r in catalog}
    variant_of = {(r["set"], r["column"]): r["variant"] for r in catalog}

    def make_runtime():
        if args.blob:
            return TempoRuntime(args.blob, nz_max=64, version=args.version)
        return TempoRuntime(table_dir=args.tables, nz_max=64, version=args.version)

    modes = ["free", "replay"] if args.mode == "both" else [args.mode]
    result = dict(
        strict=os.environ.get("GPUWM_WRF_EXACT") == "1",
        wrf_exact_enabled=bool(wrf_exact.ENABLED),
        strict_options=list(wrf_exact.STRICT_OPTIONS) if wrf_exact.ENABLED else [],
        tempo_version=args.version,
        blob_source=("fortran dump " + str(args.blob)) if args.blob else
                    ("rust tempo_init from " + str(args.tables)),
        device=cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        nvrtc=".".join(str(x) for x in cp.cuda.nvrtc.getVersion()),
        columns=sum(s["ncol"] for s in sets),
        sets=[dict(dt=s["dt"], ncol=s["ncol"], nz=s["nz"], nsteps=s["nsteps"]) for s in sets],
        values_per_mode=sum(s["nsteps"] * s["ncol"] * (len(UPDATED) * s["nz"] + len(SURFACE)
                                                       + len(COLUMN) * s["nz"]) for s in sets),
        modes={},
    )
    for mode in modes:
        t0 = time.time()
        rt = make_runtime()
        if mode == "free":
            result["compile_options"] = list(getattr(rt.module, "options", ()) or ())
        per_field, per_regime = Stats(), Stats()
        tempo_undefined = 0
        miss_columns = {}
        first_miss = None
        for si, s in enumerate(sets, start=1):
            ncol, nz = s["ncol"], s["nz"]
            regimes = [regime_of[(si, c + 1)] for c in range(ncol)]
            # (nz, 1, ncol): TempoRuntime packs (nz, ny, nx) fields column by column
            f = {n: cp.ascontiguousarray(cp.asarray(s["fields"][n].T[:, None, :]))
                 for n in STATE_IN}
            for it in range(s["nsteps"]):
                if mode == "replay" and it > 0:
                    prev = s["ref"][it - 1]
                    for n in UPDATED:
                        f[n][...] = cp.asarray(prev[n].T[:, None, :])
                if args.version != "v3.0.3":
                    # v3.2.5 tempo_run: is_first_step = (itimestep == 1); the
                    # Fortran oracle restarts itimestep at 1 for every set.
                    rt.first_call = it == 0
                out = rt.step(f, s["dt"], TempoConfig())
                ref = s["ref"][it]
                got = {n: f[n][:, 0, :].get().T for n in UPDATED}
                for n in SURFACE:
                    got[n] = out["surface"][n][0, :].get().reshape(ncol, 1)
                for n in COLUMN:
                    got[n] = out["column"][n][:, 0, :].get().T
                for n in OUTPUTS:
                    if n == "frozen_fraction":
                        excused = undefined_frozen_fraction(ref, got[n])
                        if excused.any():
                            tempo_undefined += int(excused.sum())
                            got[n] = np.where(excused, ref[n], got[n])
                    miss = per_field.add(n, got[n], ref[n])
                    if miss is None:
                        for reg in set(regimes):
                            per_regime.add((reg, n), np.zeros(1), np.zeros(1))
                        continue
                    cols = np.nonzero(miss.any(axis=1))[0]
                    for c in cols:
                        key = f"{si}:{c + 1}:{variant_of[(si, c + 1)]}"
                        miss_columns.setdefault(key, set()).add(n)
                    if first_miss is None:
                        c = int(cols[0])
                        k = int(np.nonzero(miss[c])[0][0])
                        first_miss = dict(set=si, step=it + 1, column=c + 1,
                                          variant=variant_of[(si, c + 1)], field=n, level=k + 1,
                                          got=float(got[n][c, k]), want=float(ref[n][c, k]),
                                          got_hex=float(got[n][c, k]).hex(),
                                          want_hex=float(ref[n][c, k]).hex())
                    for reg in set(regimes):
                        idx = [c for c in range(ncol) if regimes[c] == reg]
                        per_regime.add((reg, n), got[n][idx], ref[n][idx])
        fields = per_field.table()
        total = sum(v["n"] for v in fields.values())
        bitwise = sum(v["n_bitwise"] for v in fields.values())
        regime_tab = {}
        for (reg, n), v in per_regime.table().items():
            r = regime_tab.setdefault(reg, dict(max_ulp=0, max_abs=0.0, max_rel=0.0,
                                                fields_with_misses=[]))
            r["max_ulp"] = max(r["max_ulp"], v["max_ulp"])
            r["max_abs"] = max(r["max_abs"], v["max_abs"])
            r["max_rel"] = max(r["max_rel"], v["max_rel"])
            if v["max_ulp"] or v["nan_mismatch"]:
                r["fields_with_misses"].append(n)
        result["modes"][mode] = dict(
            values=total, bitwise_equal=bitwise, mismatched=total - bitwise,
            max_ulp=max(v["max_ulp"] for v in fields.values()),
            max_abs=max(v["max_abs"] for v in fields.values()),
            max_rel=max(v["max_rel"] for v in fields.values()),
            nan_mismatch=sum(v["nan_mismatch"] for v in fields.values()),
            per_field=fields, per_regime=regime_tab,
            columns_with_misses=len(miss_columns),
            columns_with_misses_list={k: sorted(v) for k, v in sorted(miss_columns.items())[:200]},
            first_miss=first_miss, seconds=round(time.time() - t0, 1),
            tempo_undefined_frozen_fraction_cells=tempo_undefined)
        m = result["modes"][mode]
        print(f"[{mode}] strict={result['strict']} values={total} bitwise={bitwise} "
              f"mismatched={total - bitwise} max_ulp={m['max_ulp']} max_abs={m['max_abs']:.3e} "
              f"max_rel={m['max_rel']:.3e} columns_with_misses={len(miss_columns)} "
              f"tempo_undefined_frozen_fraction={tempo_undefined}")
        if first_miss:
            print(f"[{mode}] first miss: {first_miss}")
    receipt = wrf_exact.compile_receipt()
    result["compile_receipt"] = dict(
        enabled=receipt["enabled"], strict_options=receipt["strict_options"],
        nvrtc_compiles=[r for r in receipt["records"] if r["kind"] == "nvrtc_compile"])
    Path(args.out).write_text(json.dumps(result, indent=1, default=str) + "\n")
    return result


def compare_file(args):
    """Compare a precomputed output file (ref_out.bin layout), for example
    the port's host build (tempo_cpu_oracle.cpp), with the reference."""
    sets = read_columns(args.columns)
    read_reference(args.reference, sets)
    read_reference(args.got_file, sets, key="got")
    catalog = json.loads(Path(args.catalog).read_text())
    variant_of = {(r["set"], r["column"]): r["variant"] for r in catalog}
    per_field = Stats()
    miss_columns = {}
    first = {}
    tempo_undefined = 0
    for si, s in enumerate(sets, start=1):
        for it in range(s["nsteps"]):
            for n in OUTPUTS:
                g = s["got"][it][n]
                if n == "frozen_fraction":
                    excused = undefined_frozen_fraction(s["ref"][it], g)
                    if excused.any():
                        tempo_undefined += int(excused.sum())
                        g = np.where(excused, s["ref"][it][n], g)
                miss = per_field.add(n, g, s["ref"][it][n])
                if miss is None:
                    continue
                for c in np.nonzero(miss.any(axis=1))[0]:
                    key = f"{si}:{c + 1}:{variant_of[(si, c + 1)]}"
                    miss_columns.setdefault(key, set()).add(n)
                    if key not in first:
                        k = int(np.nonzero(miss[c])[0][0])
                        first[key] = dict(step=it + 1, field=n, level=k + 1,
                                          got=float(s["got"][it][n][c, k]),
                                          want=float(s["ref"][it][n][c, k]))
    fields = per_field.table()
    total = sum(v["n"] for v in fields.values())
    bitwise = sum(v["n_bitwise"] for v in fields.values())
    result = dict(got_file=str(args.got_file), values=total, bitwise_equal=bitwise,
                  mismatched=total - bitwise,
                  max_ulp=max(v["max_ulp"] for v in fields.values()),
                  tempo_undefined_frozen_fraction_cells=tempo_undefined,
                  per_field=fields, columns_with_misses={k: sorted(v) for k, v in miss_columns.items()},
                  first_miss_per_column=first)
    print(f"[file] values={total} bitwise={bitwise} mismatched={total - bitwise} "
          f"max_ulp={result['max_ulp']} columns_with_misses={len(miss_columns)} "
          f"tempo_undefined_frozen_fraction={tempo_undefined}")
    for k in list(first)[:40]:
        print(f"  {k}: {first[k]}")
    Path(args.out).write_text(json.dumps(result, indent=1, default=str) + "\n")
    result["modes"] = {"file": result}
    return result


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("columns")
    ap.add_argument("catalog")
    ap.add_argument("reference")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--tables")
    src.add_argument("--blob")
    src.add_argument("--got-file", help="compare this precomputed output file instead of "
                                        "running the GPU")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="both", choices=("free", "replay", "both"))
    ap.add_argument("--version", default="v3.0.3", choices=("v3.0.3", "v3.2.5"),
                    help="TEMPO generation of the runtime and of the reference files")
    ap.add_argument("--strict", action="store_true",
                    help="refuse to run unless GPUWM_WRF_EXACT=1 is set")
    return ap.parse_args(argv)


def main(argv=None):
    res = run(parse_args(argv))
    worst = max(m["mismatched"] for m in res["modes"].values())
    sys.exit(0 if worst == 0 else 1)


if __name__ == "__main__":
    main()
