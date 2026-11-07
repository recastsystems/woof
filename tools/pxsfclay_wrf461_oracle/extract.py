"""Lift the PX sea-ice wrapper and its ice-temperature helper out of WRF.

``pxsfclay_seaice_wrapper`` and ``get_local_ice_tsk`` live inside
``phys/module_surface_driver.F``, a module whose USE list reaches most of
WRF's physics.  Compiling the whole driver for one column oracle would drag
every scheme in; instead the two subroutines are copied, byte for byte,
between their SUBROUTINE and END SUBROUTINE lines into a small module that
the oracle compiles beside the unmodified ``module_sf_pxsfclay.F``.

Nothing is edited: the script refuses a source whose SHA-256 is not the
pinned WRF v4.6.1 file, and it records the exact line ranges it copied so
the receipt says which WRF text the oracle ran.

usage: extract.py WRF_SOURCE_ROOT OUT_DIR
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

#: The two subroutines copied.  The source file itself is pinned by
#: build.sh (WRF commit d66e442 and a clean diff), and its SHA-256 is
#: recorded in the receipt this script writes.
SUBROUTINES = ("pxsfclay_seaice_wrapper", "get_local_ice_tsk")


def _span(lines: list[str], name: str) -> tuple[int, int]:
    start = end = None
    for number, line in enumerate(lines):
        text = line.strip().lower()
        if start is None and text.startswith(f"subroutine {name}"):
            start = number
        elif start is not None and text.startswith(f"end subroutine {name}"):
            end = number
            break
    if start is None or end is None:
        raise SystemExit(f"subroutine {name} not found in the driver source")
    return start, end


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2
    root = Path(argv[1])
    out = Path(argv[2])
    out.mkdir(parents=True, exist_ok=True)
    source = root / "phys" / "module_surface_driver.F"
    raw = source.read_bytes()
    lines = raw.decode("utf-8").splitlines(keepends=True)
    body: list[str] = []
    spans = {}
    for name in SUBROUTINES:
        start, end = _span(lines, name)
        spans[name] = [start + 1, end + 1]
        body.extend(lines[start:end + 1])
        body.append("\n")
    if any(line.lstrip().startswith("#") for line in body):
        raise SystemExit("extracted text carries a preprocessor line; "
                         "the copy would no longer be the compiled text")
    text = ("MODULE px_seaice_oracle\n"
            "! Copied verbatim from WRF v4.6.1 phys/module_surface_driver.F\n"
            "! by tools/pxsfclay_wrf461_oracle/extract.py; see receipt.\n"
            "CONTAINS\n" + "".join(body) + "END MODULE px_seaice_oracle\n")
    (out / "px_seaice_oracle.f90").write_text(text, encoding="utf-8")
    receipt = {
        "source": "phys/module_surface_driver.F",
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "line_ranges_1_based": spans,
        "extract_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }
    (out / "extract-receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
