#!/usr/bin/env python3
"""Cut the committed Eta Ferrier oracle fixtures from a build directory.

usage: make_fixtures.py BUILD_DIR FIXTURE_DIR

Copies ``columns_inputs.csv``, keeps steps 1, 2 and 10 of
``O0/levels.csv`` (every level of every column), all of
``O0/surface.csv``, the ``mp_restart_state`` rows of ``O0/init.csv`` and
the SHA-256 of each 7501-entry saturation table (``tables.csv``), then
writes ``SHA256SUMS`` over the fixture files.  The full per-step files
stay on the build host; the fixtures are what the focused tests grade.
"""

from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path

KEEP_STEPS = {"1", "2", "10"}


def main(build: Path, fixture: Path) -> None:
    fixture.mkdir(parents=True, exist_ok=True)
    ref = build / "O0"
    shutil.copyfile(build / "columns_inputs.csv",
                    fixture / "columns_inputs.csv")
    with open(ref / "levels.csv") as src, \
            open(fixture / "levels.csv", "w", newline="\n") as dst:
        dst.write(src.readline())
        for line in src:
            if line.split(",", 2)[1] in KEEP_STEPS:
                dst.write(line)
    shutil.copyfile(ref / "surface.csv", fixture / "surface.csv")
    tables: dict[tuple[str, str], list[str]] = {}
    with open(ref / "init.csv") as src, \
            open(fixture / "init.csv", "w", newline="\n") as dst:
        dst.write(src.readline())
        for line in src:
            col, name, idx, val = line.strip().split(",")
            if name == "mp_restart_state":
                dst.write(line)
            else:
                tables.setdefault((col, name), []).append(val)
    with open(fixture / "tables.csv", "w", newline="\n") as dst:
        dst.write("col,name,words,sha256_of_hex_lines\n")
        for (col, name), vals in sorted(tables.items(),
                                        key=lambda kv: (int(kv[0][0]),
                                                        kv[0][1])):
            digest = hashlib.sha256(("\n".join(vals) + "\n").encode()) \
                .hexdigest()
            dst.write(f"{col},{name},{len(vals)},{digest}\n")
    names = ("columns_inputs.csv", "levels.csv", "surface.csv", "init.csv",
             "tables.csv")
    with open(fixture / "SHA256SUMS", "w", newline="\n") as dst:
        for name in names:
            digest = hashlib.sha256((fixture / name).read_bytes()).hexdigest()
            dst.write(f"{digest}  {name}\n")
    print(f"fixtures written to {fixture}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(Path(sys.argv[1]), Path(sys.argv[2]))
