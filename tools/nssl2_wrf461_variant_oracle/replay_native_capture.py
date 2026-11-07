"""Replay an output-only capture from an actual native WRF driver call.

Strict comparisons count every float32 image, including signed zero. The
receipt distinguishes absent/unported paths from executed, differing paths.
No synthetic values replace captured meteorology or prognostics.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cases
from run_gpu import admit, run_mode
from run_oracle import OUTPUT_FIELDS, SURFACE_FIELDS

CELL_FIELDS = (*cases.STATE_FIELDS, "pressure", "exner", "rho", "dz", "w",
               "refl_10cm", "effc_m", "effi_m", "effs_m")


def read_capture(path):
    raw = Path(path).read_bytes()
    version, nx, nz, it0 = np.frombuffer(raw, "<i4", count=4)
    if version not in (1, 2) or nx < 1 or (version == 1 and nx > 6) or nz < 2:
        raise ValueError("invalid native capture header")
    dt = float(np.frombuffer(raw, "<f4", count=1, offset=16)[0])
    ipconc, hail, density, lccn, icdx, icdxhl = map(int, np.frombuffer(raw, "<i4", count=6, offset=20))
    params = np.frombuffer(raw, "<f4", count=10, offset=44).copy()
    coords = np.frombuffer(raw, "<i4", count=nx*2, offset=84).reshape(nx, 2).copy()
    if version == 2 and (not np.all(coords[:,1] == coords[0,1]) or
                         not np.array_equal(coords[:,0], np.arange(coords[0,0], coords[0,0]+nx))):
        raise ValueError("complete native row must preserve contiguous X ordering")
    offset = 84 + nx*8
    count = nx*nz*29 + nx*9
    if len(raw) != offset + count*4*2:
        raise ValueError("native capture is truncated or has trailing words")
    snapshots = []
    for _ in range(2):
        cells = np.frombuffer(raw, "<f4", count=nx*nz*29, offset=offset).reshape(29, nx, nz).copy()
        offset += cells.nbytes
        surfaces = np.frombuffer(raw, "<f4", count=nx*9, offset=offset).reshape(9, nx).copy()
        offset += surfaces.nbytes
        snapshots.append(({n: cells[i] for i,n in enumerate(CELL_FIELDS)},
                          {n: surfaces[i] for i,n in enumerate(SURFACE_FIELDS)}))
    return {"capture_version": int(version), "complete_native_row": bool(version == 2), "source_sha256": hashlib.sha256(raw).hexdigest(), "nx": int(nx), "nz": int(nz),
            "itimestep": int(it0), "dt": dt, "ipconc": ipconc, "hail": hail,
            "density": density, "lccn": lccn, "icdx": icdx, "icdxhl": icdxhl,
            "params": params.tolist(), "coords": coords.tolist()}, snapshots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()
    meta, (before, after) = read_capture(args.capture)
    mode = replace(cases.MODE_BY_LABEL["cccn-low"], cccn=meta["params"][0],
                   ipconc=meta["ipconc"], hail_on=meta["hail"],
                   ccn_on=int(meta["lccn"] > 1),
                   density_on=meta["density"], icdx=meta["icdx"], icdxhl=meta["icdxhl"],
                   alphah=meta["params"][1], alphahl=meta["params"][2])
    mode = replace(mode, selectors={
        "nssl_2moment_on": int(meta["ipconc"] >= 5),
        "nssl_hail_on": (1 if meta["ipconc"] >= 5 else 2) if meta["hail"] else 0,
        "nssl_density_on": meta["density"], "nssl_cccn": meta["params"][0],
        "nssl_ccn_on": int(meta["lccn"] > 1),
        "nssl_3moment": int(meta["ipconc"] >= 6),
        "nssl_alphah": meta["params"][1], "nssl_alphahl": meta["params"][2],
        "nssl_icdx": meta["icdx"], "nssl_icdxhl": meta["icdxhl"],
        **dict(zip(("nssl_cnoh", "nssl_cnohl", "nssl_cnor", "nssl_cnos",
                    "nssl_rho_qh", "nssl_rho_qhl", "nssl_rho_qs"), meta["params"][3:])),
    })
    record = {"capture": meta, "real_native_driver_call": True,
              "strict_numerics": os.environ.get("GPUWM_WRF_EXACT") == "1",
              "finite_input": all(np.isfinite(a).all() for a in (*before[0].values(), *before[1].values())),
              "finite_native_output": all(np.isfinite(a).all() for a in (*after[0].values(), *after[1].values()))}
    record["native_output_sha256"] = {
        n: hashlib.sha256(a.tobytes()).hexdigest() for n,a in {**after[0], **after[1]}.items()}
    record["input_sha256"] = {
        n: hashlib.sha256(a.tobytes()).hexdigest() for n,a in {**before[0], **before[1]}.items()}
    status = 0
    if not args.inspect_only:
        try:
            resolved = admit(mode)
        except (TypeError, ValueError) as exc:
            record.update(refused=f"{type(exc).__name__}: {exc}", compared_words=0, passed=False)
            status = 2
        else:
            state = {n: before[0][n] for n in cases.STATE_FIELDS}
            env = {n: before[0][n] for n in ("pressure", "exner", "rho", "dz", "w")}
            out, _ = run_mode(mode, resolved, 1, meta["dt"], meta["itimestep"],
                              input_case=(state, env), surface_initial=before[1],
                              diagnostic_initial=before[0])
            details = {}
            # Inactive Registry slots are represented by zeros in the file
            # to avoid reading their null scalar actual arguments. Those
            # placeholders are neither captured native words nor proof.
            active = {"theta", *resolved.transported_fields,
                      "refl_10cm", "effc_m", "effi_m", "effs_m"}
            if meta['ipconc'] == 0:
                # Native physics_init.F:985-1023 leaves has_req*=0 for
                # one-moment NSSL. Driver :3170-3208 consequently never
                # writes these PRESENT diagnostics. Preserve a witness,
                # excluding neither an active prognostic nor a written output.
                unwritten = {}
                for name in ('effc_m', 'effi_m', 'effs_m'):
                    a, b = before[0][name], after[0][name]
                    unwritten[name] = {
                        'words': int(a.size),
                        'nonfinite_words': int((~np.isfinite(b)).sum()),
                        'unchanged_native_bytes': a.tobytes() == b.tobytes(),
                        'gpu_preserved_bytes': out[name][0].tobytes() == b.tobytes(),
                        'sha256': hashlib.sha256(b.tobytes()).hexdigest(),
                    }
                    active.remove(name)
                record['native_unwritten_diagnostics'] = unwritten
                inputs = {'theta', *resolved.transported_fields,
                          'pressure', 'exner', 'rho', 'dz', 'w'}
                record['finite_input'] = all(np.isfinite(before[0][n]).all() for n in inputs)
                record['finite_input'] &= all(np.isfinite(a).all() for a in before[1].values())
                record['finite_native_output'] = all(np.isfinite(after[0][n]).all() for n in active)
                record['finite_native_output'] &= all(np.isfinite(a).all() for a in after[1].values())
            record["inactive_placeholders_excluded"] = [
                name for name in OUTPUT_FIELDS if name not in active]
            for name in (*(name for name in OUTPUT_FIELDS if name in active), *SURFACE_FIELDS):
                native = after[1][name] if name in SURFACE_FIELDS else after[0][name]
                gpu = out[name][0]
                different = native.view(np.uint32) != gpu.view(np.uint32)
                finite_pair = np.isfinite(native) & np.isfinite(gpu)
                max_abs = (float(np.max(np.abs(native[finite_pair].astype(np.float64)-gpu[finite_pair])))
                           if finite_pair.any() else None)
                details[name] = {"words": int(native.size), "differing_words": int(different.sum()),
                                 "max_abs": max_abs,
                                 "native_nonfinite_words": int((~np.isfinite(native)).sum()),
                                 "finite": bool(np.isfinite(gpu).all()),
                                 "gpu_sha256": hashlib.sha256(gpu.tobytes()).hexdigest()}
            record["fields"] = details
            record["compared_words"] = sum(v["words"] for v in details.values())
            record["differing_words"] = sum(v["differing_words"] for v in details.values())
            record["passed"] = (record["strict_numerics"] and record["finite_input"]
                                and record["finite_native_output"]
                                and record["differing_words"] == 0
                                and all(v['unchanged_native_bytes'] and v['gpu_preserved_bytes']
                                        for v in record.get('native_unwritten_diagnostics', {}).values())
                                and all(v["finite"] for v in details.values()))
            status = 0 if record["passed"] else 1
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(record, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in record.items() if k in ("passed", "refused", "compared_words", "differing_words")}))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
