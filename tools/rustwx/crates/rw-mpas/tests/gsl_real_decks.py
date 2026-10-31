"""Replay published native GSL files through the real Rust executable.

Orchestration only: Rust decodes the mesh/geography and compares every word.
Native boundary: mpas_init_atm_gwd_gsl.F:49-203 (MPAS-A v8.4.2).
Invoke with --engine BIN --geog DIRECTORY --scratch OWNED_DIRECTORY
--receipts DIRECTORY --deck GRID.nc NATIVE_ORO.nc (repeat --deck).
GRID must contain geometry from the same initialization checkpoint as native:
full init uses post-static geometry (mpas_init_atm_static.F:288-307); GSL-only
init may use the raw unit sphere (mpas_init_atm_gwd_gsl.F:122-165). Scaling
edges before averaging and scaling the average have different RKIND rounding.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


def sha256(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("engine", "geog", "scratch", "receipts"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--deck", type=Path, nargs=2, action="append", required=True)
    args = parser.parse_args()
    args.scratch.mkdir(parents=True, exist_ok=True)
    args.receipts.mkdir(parents=True, exist_ok=True)
    rows = []
    cached = {}
    with tempfile.TemporaryDirectory(prefix="gsl-real-", dir=args.scratch) as scratch:
        for index, (grid, native) in enumerate(args.deck):
            grid_hash = sha256(grid)
            if grid_hash not in cached:
                output = Path(scratch) / f"{index}.nc"
                generation = args.receipts / f"generation-{index}.json"
                command = [str(args.engine), "--grid", str(grid), "--geog", str(args.geog),
                           "--out", str(output), "--receipt", str(generation)]
                with (args.receipts / f"generation-{index}.log").open("w") as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
                cached[grid_hash] = (output, generation, command)
            output, generation, command = cached[grid_hash]
            compare = args.receipts / f"compare-{index}.json"
            comparison = [str(args.engine), "--compare", str(native), str(output),
                          "--receipt", str(compare)]
            with (args.receipts / f"compare-{index}.log").open("w") as log:
                subprocess.run(comparison, stdout=log, stderr=subprocess.STDOUT, check=True)
            result = json.loads(compare.read_text())
            assert result["words"] > 0 and result["unequal_words"] == 0
            rows.append({"grid": str(grid), "native": str(native),
                         "grid_sha256": grid_hash, "native_sha256": sha256(native),
                         "generation_receipt": generation.name, "compare_receipt": compare.name,
                         "generation_command": command, "compare_command": comparison,
                         "words": result["words"], "unequal_words": result["unequal_words"]})
            print(f"deck {index}: {result['words']} words, 0 unequal", flush=True)
    evidence = {"schema": "gsl-published-native-replay/v1",
                "engine_sha256": sha256(args.engine), "decks": rows,
                "words": sum(row["words"] for row in rows), "unequal_words": 0,
                "generated_files_deleted": True,
                "hex_front_door_proven": False, "forecast_proven": False}
    (args.receipts / "replay.json").write_text(json.dumps(evidence, indent=2) + "\n")


if __name__ == "__main__":
    main()
