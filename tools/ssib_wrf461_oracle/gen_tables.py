"""Extract SSiB's vegetation DATA tables from WRF v4.6.1 module_sf_ssib.F.

Writes woof/data/ssib/ssib_veg_tables.json: every table as float32 bit
patterns (hex), in Fortran (column-major) element order, plus the three
category tables SSIB keeps as local DATA (IVUSGS, IVMODIS, TD_DEPTH).

Each decimal literal is rounded to binary32 exactly (round to nearest,
ties to even), which is what gfortran does with a REAL literal in a DATA
statement.  build.sh cross-checks the module tables against the bits a
gfortran program USEing the compiled module dumps (dump_tables.F90).

    python3 gen_tables.py WRF_SOURCE_ROOT OUT.json
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
import sys
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import numpy as np

#: Module tables, their Fortran shapes (module_sf_ssib.F:41-47), in the
#: order woof/core/kernels/ssib.cu lays them out in ssib_tab.
MODULE_TABLES = (
    ("tran0", (13, 2, 3, 2)), ("ref0", (13, 2, 3, 2)),
    ("green0", (13, 12, 2)), ("vcover0", (13, 12, 2)), ("zlt0", (13, 12, 2)),
    ("rstpar0", (13, 2, 3)),
    ("z000", (13, 12)), ("d0", (13, 12)), ("z20", (13, 12)), ("z10", (13, 12)),
    ("rdc0", (13, 12)), ("rbc0", (13, 12)),
    ("depth0", (13, 3)), ("soref0", (13, 3)),
    ("chil0", (13, 2)), ("topt0", (13, 2)), ("tl0", (13, 2)), ("tu0", (13, 2)),
    ("defac0", (13, 2)), ("ph10", (13, 2)), ("ph20", (13, 2)),
    ("rootd0", (13, 2)),
    ("bee0", (13,)), ("phsat0", (13,)), ("poros0", (13,)), ("satco0", (13,)),
    ("slope0", (13,)),
)
#: SSIB's local DATA tables (module_sf_ssib.F:996-1007).
LOCAL_TABLES = (("ivusgs", (24,)), ("ivmodis", (20,)), ("td_depth", (13,)))
TABLES = MODULE_TABLES + LOCAL_TABLES


def f32_round(text: str) -> np.float32:
    """Nearest binary32 to the decimal ``text`` (ties to even)."""
    exact = Fraction(Decimal(text.replace("d", "e").replace("D", "e")))
    guess = np.float32(float(exact))
    best = None
    for cand in (np.nextafter(guess, np.float32(-np.inf)), guess,
                 np.nextafter(guess, np.float32(np.inf))):
        err = abs(Fraction(float(cand)) - exact)
        key = (err, int(np.frombuffer(cand.tobytes(), "<u4")[0]) & 1)
        if best is None or key < best[0]:
            best = (key, cand)
    return best[1]


def _data_block(text: str, name: str) -> list[str]:
    """The literal list of ``data name/ ... /`` (case-insensitive)."""
    m = re.search(rf"(?im)^\s*data\s+{re.escape(name)}\s*/(.*?)/", text, re.S)
    if m is None:
        raise SystemExit(f"no DATA statement for {name}")
    body = m.group(1).replace("&", " ")
    return [tok for tok in re.split(r"[\s,]+", body) if tok]


def _expand(tokens: list[str]) -> list[str]:
    out = []
    for tok in tokens:
        if "*" in tok:                       # repeat count r*value
            r, v = tok.split("*", 1)
            out.extend([v] * int(r))
        else:
            out.append(tok)
    return out


def extract(source: Path) -> dict:
    raw = source.read_bytes()
    text = raw.decode("latin-1")
    # Strip Fortran comments (whole-line '!' and trailing '!').
    lines = []
    for line in text.splitlines():
        cut = line.split("!", 1)[0]
        lines.append(cut)
    text = "\n".join(lines)
    tables = {}
    for name, shape in TABLES:
        vals = _expand(_data_block(text, name))
        n = int(np.prod(shape))
        if len(vals) != n:
            raise SystemExit(f"{name}: {len(vals)} values, shape {shape} needs {n}")
        f32 = [f32_round(v) for v in vals]
        tables[name] = {
            "shape": list(shape),
            "order": "fortran",
            "f32_hex": [f"{int(np.frombuffer(v.tobytes(), '<u4')[0]):08x}" for v in f32],
        }
    return {
        "source": "WRF v4.6.1 phys/module_sf_ssib.F (d66e442)",
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "layout": [name for name, _ in TABLES],
        "tables": tables,
    }


def flat(doc: dict) -> np.ndarray:
    """The ssib_tab constant array, in kernel layout order."""
    parts = []
    for name in doc["layout"]:
        words = [int(h, 16) for h in doc["tables"][name]["f32_hex"]]
        parts.append(np.array(words, "<u4").view("<f4"))
    return np.concatenate(parts)


if __name__ == "__main__":
    src = Path(sys.argv[1]) / "phys" / "module_sf_ssib.F"
    out = Path(sys.argv[2])
    doc = extract(src)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    print(f"{len(flat(doc))} words -> {out}")
