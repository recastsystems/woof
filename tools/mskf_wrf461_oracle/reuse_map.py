"""Record every source-line difference between the two pinned WRF KF files."""
from __future__ import annotations
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import re


def routines(lines):
    result = {}
    start = None
    name = None
    for i, line in enumerate(lines):
        found = re.match(r"\s*subroutine\s+(\w+)", line, re.I)
        if found:
            start, name = i, found[1].upper()
        if re.match(r"\s*end\s*subroutine", line, re.I) and start is not None:
            result[name] = (start, i + 1)
            start = name = None
    return result


def build(source, output):
    left = source / "phys/module_cu_kfeta.F"
    right = source / "phys/module_cu_mskf.F"
    a, b = left.read_text().splitlines(True), right.read_text().splitlines(True)
    ra, rb = routines(a), routines(b)
    aliases = {"MSKF_CPS": "KF_ETA_CPS", "MSKF_ETA_PARA": "KF_ETA_PARA",
               "MSKF_INIT": "KF_ETA_INIT", "MSKF_LUTAB": "KF_LUTAB"}
    rows = []
    for name, (start, end) in rb.items():
        other = aliases.get(name, name)
        if other not in ra:
            rows.append({"mskf": name, "mskf_lines": [start + 1, end],
                         "kf": None, "status": "new"})
            continue
        lo, hi = ra[other]
        same = a[lo:hi] == b[start:end]
        def code(lines):
            return [re.sub(r"\s+", "", x.split("!", 1)[0]).lower().replace("&", "")
                    for x in lines if x.split("!", 1)[0].strip()]
        ca, cb = code(a[lo:hi]), code(b[start:end])
        # Names and the extra TPMIX2 QSu output are retained as real differences.
        code_same = ca == cb

        # Include all non-equal operations, with absolute 1-based line ranges.
        edits = [{"operation": op, "kf_lines": [lo+i+1, lo+j],
                  "mskf_lines": [start+k+1, start+l]}
                 for op, i, j, k, l in difflib.SequenceMatcher(
                     None, a[lo:hi], b[start:end], autojunk=False).get_opcodes()
                 if op != "equal"]
        rows.append({"mskf": name, "mskf_lines": [start+1, end], "kf": other,
                     "kf_lines": [lo+1, hi], "status": "identical" if same else "differs",
                     "edits": edits, "executable_text_identical": code_same,
                     "executable_diff": list(difflib.unified_diff(ca, cb, lineterm=""))})
    result = {"sources": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (left, right)}, "routines": rows}
    output.mkdir(parents=True, exist_ok=True)
    (output / "cu1-cu11-reuse.json").write_text(json.dumps(result, indent=2)+"\n")
    (output / "cu1-cu11-source.diff").write_text("".join(difflib.unified_diff(
        a, b, fromfile="WRF-4.6.1/phys/module_cu_kfeta.F",
        tofile="WRF-4.6.1/phys/module_cu_mskf.F")))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = build(args.source, args.output)
    for row in result["routines"]:
        print(row["mskf"], row["status"], row["mskf_lines"])
