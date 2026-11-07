"""Every-field, every-step SHA-256 receipts for the WDM7 column oracle."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from oracle_io import OUTPUT_LEVEL_FIELDS, OUTPUT_SURFACE_FIELDS, read_input, read_output

def receipt(fixture, outputs, fortran=False):
    result = {}
    for inp in sorted(Path(fixture).glob("*.in")):
        c = read_input(inp)
        out = Path(outputs) / (inp.stem + ".out")
        steps = read_output(out, c.ncol, c.nz, c.nsteps)
        fields = {}
        for step, (level, surface) in enumerate(steps):
            for name in OUTPUT_LEVEL_FIELDS + OUTPUT_SURFACE_FIELDS:
                a = np.asarray((level if name in OUTPUT_LEVEL_FIELDS else surface)[name], dtype="<f4")
                if fortran and name in ("re_cloud", "re_ice", "re_snow"):
                    a = a * np.float32(1.0e6)
                fields[f"{step}:{name}"] = {
                    "sha256": hashlib.sha256(a.tobytes()).hexdigest(), "words": int(a.size),
                    "finite": bool(np.isfinite(a).all()), "min": float(a.min()) if np.isfinite(a).all() else None, "max": float(a.max()) if np.isfinite(a).all() else None}
        result[inp.stem] = {"input_sha256": hashlib.sha256(inp.read_bytes()).hexdigest(),
                            "ncol": c.ncol, "nz": c.nz, "nsteps": c.nsteps, "fields": fields}
    return result

def main():
    p = argparse.ArgumentParser()
    p.add_argument("fixture", type=Path)
    p.add_argument("outputs", type=Path)
    p.add_argument("--fortran", action="store_true")
    p.add_argument("--reference", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    got = receipt(args.fixture, args.outputs, args.fortran)
    report = {"configurations": got, "matched": None}
    if args.reference:
        ref = json.loads(args.reference.read_text())["configurations"]
        differences = []
        for cfg, want in ref.items():
            have = got[cfg]
            assert have["input_sha256"] == want["input_sha256"], cfg
            for field, word in want["fields"].items():
                if have["fields"][field] != word:
                    differences.append(cfg + "/" + field)
        assert set(got) == set(ref)
        report["matched"] = not differences
        report["differing_fields"] = differences
    args.out.write_text(json.dumps(report, indent=1, allow_nan=False) + "\n")
    print("matched", report["matched"], "field records",
          sum(len(c["fields"]) for c in got.values()))
    return 1 if report["matched"] is False else 0

if __name__ == "__main__":
    raise SystemExit(main())
