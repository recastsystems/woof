"""Pack the PX land-use tables from the compiled module's own words.

Reads pxlsm-tables.txt (written by dump_tables.F90 from the gfortran object
of WRF v4.6.1 phys/module_sf_pxlsm_data.F) and writes
woof/data/pxlsm/pxlsm_tables.json: per dataset, the nine parameter rows
VEGELAND reads, each value the shortest decimal that reads back to the
same binary32 word.  tests/test_pxlsm_tables.py re-reads the JSON and holds
every word equal to the dump, and woof/core/pxlsm.py pins the file's
SHA-256, so the packaged bytes cannot move without both noticing.

Usage: python pack_tables.py pxlsm-tables.txt SOURCE_SHA256 OUT.json
"""

from __future__ import annotations

import json
import sys

import numpy as np

ROWS = ("RSMIN", "Z00", "VEG0", "VEGMN0", "LAI0", "LAIMN0", "SNUP0", "ALBF",
        "SNOALB")


def main() -> None:
    dump, source_sha, out = sys.argv[1:4]
    datasets: dict[str, dict] = {}
    for line in open(dump):
        parts = line.split()
        if not parts:
            continue
        name, row, count = parts[0], parts[1], int(parts[2])
        words = np.array([int(w, 16) for w in parts[3:]], dtype=np.uint32)
        if words.size != count:
            raise SystemExit(f"{name} {row}: {words.size} words, header {count}")
        values = words.view(np.float32)
        entry = datasets.setdefault(name, {"categories": count})
        if entry["categories"] != count:
            raise SystemExit(f"{name}: rows disagree on the category count")
        text = [np.format_float_positional(v, unique=True, trim="-")
                for v in values]
        back = np.array([float(t) for t in text], dtype=np.float32)
        if not np.array_equal(back.view(np.uint32), words):
            raise SystemExit(f"{name} {row}: decimal text does not round-trip")
        entry[row] = [float(t) for t in text]
    for name, entry in datasets.items():
        missing = [row for row in ROWS if row not in entry]
        if missing:
            raise SystemExit(f"{name}: missing rows {missing}")
    document = {
        "schema": "gpuwm-pxlsm-tables-v1",
        "source": "WRF v4.6.1 phys/module_sf_pxlsm_data.F",
        "source_sha256": source_sha,
        "rows": list(ROWS),
        "datasets": {name: {"categories": entry["categories"],
                            **{row: entry[row] for row in ROWS}}
                     for name, entry in sorted(datasets.items())},
    }
    with open(out, "w", newline="\n") as f:
        json.dump(document, f, indent=1)
        f.write("\n")


if __name__ == "__main__":
    main()
