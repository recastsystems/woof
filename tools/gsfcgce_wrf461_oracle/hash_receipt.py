"""Full-field and per-step increment hashes, without transferring arrays.
The legacy "tendency" receipt key holds float32 state increments. Equal
increments and the fixture's identical dt prove equal derived tendencies.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
try:
    from .compare import OUT3D, OUT2D, read
except ImportError:
    from compare import OUT3D, OUT2D, read

DIAGNOSTIC_BIN_DBZ = 0.01
GRID_PHASES = 4

def diagnostic_grid(values):
    # Each block is one complete column at one step. Equal integer-bin
    # hashes on any phase prove every value in that block differs by
    # less than the bin width. Multiple phases avoid bin-edge false failures.
    out = []
    for frame in values:
        for column in frame.T:
            out.append([hashlib.sha256(
                np.floor(column.astype(np.float64) / DIAGNOSTIC_BIN_DBZ + phase / GRID_PHASES)
                  .astype("<i8").tobytes()).hexdigest()
                for phase in range(GRID_PHASES)])
    return out

def receipt(fixdir, dumpdir):
    index = json.loads((Path(fixdir) / "index.json").read_text())
    result = {"schema": "gce-full-field-hashes-v1", "cases": {}}
    for case in index["cases"]:
        path = Path(dumpdir) / (case["name"] + ".ref")
        if not path.exists():
            continue
        data = read(path, case["ncol"], case["nk"], case["nsteps"])
        fields = {}
        for name in OUT3D + OUT2D:
            values = data[name]
            states = [hashlib.sha256(v.astype("<f4").tobytes()).hexdigest()
                      for v in values[1:]]
            tendencies = np.subtract(values[1:], values[:-1], dtype=np.float32)
            fields[name] = {
                "state": states,
                "tendency": [hashlib.sha256(v.astype("<f4").tobytes()).hexdigest()
                             for v in tendencies],
                "elements": int(values[1:].size),
                "nonfinite": int(np.count_nonzero(~np.isfinite(values[1:])))}
        result["cases"][case["name"]] = fields
        pii = np.fromfile(Path(fixdir) / case["input"], dtype="<f4")
        count = case["ncol"] * case["nk"]
        pii = pii[8*count:9*count].reshape(case["nk"], case["ncol"])
        temperature = np.multiply(data["th"][1:], pii, dtype=np.float32)
        wet = np.zeros(temperature.shape, dtype=bool)
        use_snow, use_dense = case["ice2"] in (0,1), case["ice2"] in (0,2)
        for step in range(case["nsteps"]):
            for col in range(case["ncol"]):
                rain = data["qr"][step+1,:,col] > np.float32(1.e-9)
                snow = (data["qs"][step+1,:,col] > np.float32(1.e-9)) & use_snow
                dense = (data["qg"][step+1,:,col] > np.float32(1.e-9)) & use_dense
                for k in range(case["nk"]-2, -1, -1):
                    if temperature[step,k,col] > np.float32(273.15) and rain[k] and (snow[k+1] or dense[k+1]):
                        wet[step,:k+1,col] = (snow[:k+1] & snow[k+1]) | (dense[:k+1] & dense[k+1])
                        break
        diagnostic = data["refl"][1:]
        fields["refl"]["unmelted_sha256"] = hashlib.sha256(diagnostic[~wet].astype("<f4").tobytes()).hexdigest()
        fields["refl"]["melting_sha256"] = hashlib.sha256(diagnostic[wet].astype("<f4").tobytes()).hexdigest()
        fields["refl"]["melting_elements"] = int(np.count_nonzero(wet))
        fields["refl"]["grid_bin_dbz"] = DIAGNOSTIC_BIN_DBZ
        fields["refl"]["grid"] = diagnostic_grid(data["refl"][1:])
    return result

def grade(reference, candidate):
    mismatches = []
    for case, fields in reference["cases"].items():
        for name, expected in fields.items():
            actual = candidate["cases"].get(case, {}).get(name)
            if actual is None:
                mismatches.append({"case": case, "field": name, "kind": "missing"})
                continue
            for kind in ("state", "tendency"):
                if actual[kind] != expected[kind]:
                    steps = [i + 1 for i, (a,b) in enumerate(zip(actual[kind], expected[kind])) if a != b]
                    mismatches.append({"case": case, "field": name, "kind": kind, "steps": steps})
            if actual["nonfinite"]:
                mismatches.append({"case": case, "field": name, "kind": "nonfinite",
                                   "count": actual["nonfinite"]})
    state_mismatches = [m for m in mismatches if m["field"] != "refl"]
    failed_blocks = []
    for case, fields in reference["cases"].items():
        a = fields["refl"]
        b = candidate["cases"].get(case, {}).get("refl", {})
        if b.get("grid_bin_dbz") != a.get("grid_bin_dbz") or len(b.get("grid", [])) != len(a.get("grid", [])):
            failed_blocks.append({"case": case, "block": "missing"})
            continue
        for i, (ref, cand) in enumerate(zip(a["grid"], b["grid"])):
            if not any(x == y for x,y in zip(ref,cand)):
                failed_blocks.append({"case": case, "block": i})
    qualified = (bool(reference["cases"]) and not state_mismatches and
                 not failed_blocks and all(
                     candidate["cases"].get(case,{}).get("refl",{}).get("unmelted_sha256") == fields["refl"]["unmelted_sha256"]
                     and not candidate["cases"].get(case,{}).get("refl",{}).get("nonfinite",1)
                     for case,fields in reference["cases"].items()))
    return {"passed": qualified,
            "all_outputs_bit_identical": bool(reference["cases"]) and not mismatches,
            "state_bit_identical": bool(reference["cases"]) and not state_mismatches,
            "unmelted_bit_identical": all(
                candidate["cases"].get(case,{}).get("refl",{}).get("unmelted_sha256") == fields["refl"]["unmelted_sha256"]
                for case,fields in reference["cases"].items()),
            "diagnostic_abs_bound_dbz": DIAGNOSTIC_BIN_DBZ,
            "diagnostic_bound_proved": not failed_blocks and bool(reference["cases"]),
            "diagnostic_failed_blocks": failed_blocks,
            "reference_cases": len(reference["cases"]),
            "candidate_cases": len(candidate["cases"]), "mismatches": mismatches}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("fixdir"); p.add_argument("dumpdir"); p.add_argument("out")
    p.add_argument("--reference"); p.add_argument("--grade-out")
    a = p.parse_args()
    result = receipt(a.fixdir, a.dumpdir)
    Path(a.out).write_text(json.dumps(result, separators=(",", ":")))
    if a.reference:
        verdict = grade(json.loads(Path(a.reference).read_text()), result)
        Path(a.grade_out).write_text(json.dumps(verdict, indent=1))
        print(json.dumps({k: v for k,v in verdict.items() if k != "mismatches"}))
        return 0 if verdict["passed"] else 1
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
