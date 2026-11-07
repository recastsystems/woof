"""Install the SAS oracle outputs as test fixtures.

    python tools/sas_wrf461_oracle/install_fixtures.py BUILD_DIR [DEST]

Copies the scalar (referee) tree's records from BUILD_DIR (build.sh's
output) into DEST (default tests/data/oracles/sas), gzip'd with a fixed
mtime so the bytes are reproducible, and replaces the 758 KB tables.bin by
the SHA-256 of each table section (the GPU tables are compared by digest).
Refuses unless the strict and scalar trees agreed on every file and both
harness self-checks (chained, row) reported zero.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path

SECTIONS = {  # name: (offset, words) in tables.bin, as woof.core.sas
    "pvs": (0, 2 + 7501),
    "tdp": (2 + 7501, 2 + 5001),
    "the": (2 + 7501 + 2 + 5001, 4 + 241 * 151),
    "ma": (2 + 7501 + 2 + 5001 + 4 + 241 * 151, 4 + 2 * 151 * 121),
    "tlcl": (2 + 7501 + 2 + 5001 + 4 + 241 * 151 + 4 + 2 * 151 * 121, 4 + 151 * 61),
}


def main(argv) -> int:
    build = Path(argv[0])
    dest = Path(argv[1]) if len(argv) > 1 else (
        Path(__file__).resolve().parents[2] / "tests" / "data" / "oracles" / "sas")
    verdict = (build / "strict-vs-scalar.txt").read_text()
    if "!=" in verdict or verdict.count("==") != 5:
        raise SystemExit(f"strict and scalar trees disagree:\n{verdict}")
    for tree in ("strict", "scalar"):
        log = (build / tree / "run.log").read_text()
        if "chained-mismatch 0 row-mismatch 0" not in log:
            raise SystemExit(f"{tree} harness self-check failed:\n{log}")
    dest.mkdir(parents=True, exist_ok=True)
    src = build / "scalar"
    for name in ("cases.bin", "stage1.bin", "stage2.bin", "stage3.bin", "stage3_shaloff.bin"):
        data = (src / name).read_bytes()
        with open(dest / (name + ".gz"), "wb") as fh:
            with gzip.GzipFile(filename=name, mode="wb", fileobj=fh, mtime=0) as gz:
                gz.write(data)
    tb = (src / "tables.bin").read_bytes()
    total = sum(n for _, n in SECTIONS.values())
    if len(tb) != 8 * total:
        raise SystemExit(f"tables.bin is {len(tb)} bytes, expected {8 * total}")
    digests = {name: hashlib.sha256(tb[8 * off: 8 * (off + n)]).hexdigest()
               for name, (off, n) in SECTIONS.items()}
    (dest / "tables.json").write_text(json.dumps(digests, indent=1) + "\n")
    prov = [
        "WRF v4.6.1 (d66e442) SAS column oracle, tools/sas_wrf461_oracle/build.sh",
        "linked against the built strict and scalar trees' phys objects;",
        "the records are the scalar tree's (identical to strict, see below).",
        "",
        verdict.strip(),
        "",
        "strict " + (build / "strict" / "flags.txt").read_text().strip(),
        "scalar " + (build / "scalar" / "flags.txt").read_text().strip(),
        (src / "compiler.txt").read_text().strip(),
        (src / "run.log").read_text().strip(),
    ]
    (dest / "PROVENANCE.txt").write_text("\n".join(prov) + "\n")
    print(f"installed {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
