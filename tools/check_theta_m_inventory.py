#!/usr/bin/env python3
"""Check the moist-theta source inventory against a WRF 4.6.1 source tree.

This checks source coverage, not implementation correctness or forecast output.
Only the Python standard library is needed. Generated .f90 files are excluded.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wrf_root", type=Path)
    args = parser.parse_args()
    receipt = Path(__file__).resolve().parents[1] / "proof/theta-m-port/wrf461-sites.json"
    inventory = json.loads(receipt.read_text(encoding="utf-8"))
    expected = {
        (row["file"], row["line"]): row for row in inventory["sites"]
    }
    if len(expected) != len(inventory["sites"]):
        parser.error("inventory contains duplicate source locations")
    found = {}
    for directory in ("dyn_em", "share", "phys", "Registry", "main"):
        root = args.wrf_root / directory
        if not root.is_dir():
            parser.error(f"missing WRF source directory: {root}")
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix != ".F" and path.name != "Registry.EM_COMMON":
                continue
            content = path.read_bytes()
            for number, line in enumerate(content.decode("utf-8").splitlines(), 1):
                if "use_theta_m" in line.lower():
                    key = (path.relative_to(args.wrf_root).as_posix(), number)
                    found[key] = (line.strip(), hashlib.sha256(content).hexdigest())
    failures = []
    for file, line in sorted(found.keys() - expected.keys()):
        failures.append(f"unlisted site: {file}:{line}")
    for file, line in sorted(expected.keys() - found.keys()):
        failures.append(f"missing site: {file}:{line}")
    for key in sorted(found.keys() & expected.keys()):
        actual_line, digest = found[key]
        row = expected[key]
        if actual_line != row["source"]:
            failures.append(f"source line changed: {key[0]}:{key[1]}")
        if digest != row["sha256"]:
            failures.append(f"source file hash changed: {key[0]}")
    if failures:
        for failure in dict.fromkeys(failures):
            print(failure, file=sys.stderr)
        return 1
    counts = Counter(file for file, _ in found)
    print(json.dumps({
        "status": "source-inventory-matches",
        "wrf_commit_recorded": inventory["commit"],
        "sites": len(found),
        "files": len(counts),
        "counts": dict(sorted(counts.items())),
        "forecast_proof": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
