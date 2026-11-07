#!/usr/bin/env python3
"""Replay unchanged clphy1d_ylin calls captured from a real native forecast.

Run capture_native.py before rebuilding native WRF. This command requires
all four capture times, so a partly completed forecast cannot pass as proof.
"""
import argparse
import hashlib
import json
from pathlib import Path

from validate_sbu_ylin_oracle import read_cases, read_outputs, run_port, compare


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("captures", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    inputs = sorted(args.captures.glob("replay-*-in.txt"))
    calls = {int(f.name.split("-")[1]) for f in inputs}
    if calls != {1, 241, 481, 720} or len(inputs) != 144:
        raise SystemExit(f"Need 144 native columns at calls 1/241/481/720; got {len(inputs)}, {calls}")
    nz = int(inputs[0].read_text().splitlines()[0].split()[1])
    source_words, reference_words, manifest = [], [], []
    for f in inputs:
        o = f.with_name(f.name.replace("-in.txt", "-out.txt"))
        for path, target in ((f, source_words), (o, reference_words)):
            lines = path.read_text().splitlines()
            if tuple(map(int, lines[0].split())) != (1, nz):
                raise SystemExit(f"Unexpected shape in {path}")
            target.extend(lines[1:])
            manifest.append({"file": path.name, "bytes": path.stat().st_size,
                             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    cases_path = args.out.parent / "native-cases.txt"
    outputs_path = args.out.parent / "native-reference.txt"
    for path, words in ((cases_path, source_words), (outputs_path, reference_words)):
        path.write_text(f"{len(inputs)} {nz}\n" + "\n".join(words) + "\n")
    from woof.wrf_exact import ENABLED
    if not ENABLED:
        raise SystemExit("Native proof requires GPUWM_WRF_EXACT=1")
    cases = read_cases(cases_path)
    result = compare(run_port(cases), read_outputs(outputs_path))
    receipt = {"schema": "gpuwm.sbu-ylin.native-replay.v1", "columns": len(inputs),
               "levels": nz, "calls": sorted(calls), "strict": True,
               "comparison": result, "captures": manifest,
               "words": sum(v["words"] for k, v in result.items() if not k.startswith("_")),
               "input_species_positive": {name: int((cases["levels"][:, :, j] > 0).sum())
                   for name, j in (("qv", 1), ("qc", 2), ("qr", 3), ("qi", 4), ("qs", 5))}}
    args.out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "captures"}, indent=2))
    return int(result["_total_words_differing"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())
