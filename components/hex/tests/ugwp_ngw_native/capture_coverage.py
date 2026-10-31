"""Record native gcov branch gaps and validated MPAS v8.4.2 source citations."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


def capture(reference: Path, work: Path, receipts: Path) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from woof.hex.mpasphys import ugwp_ngw_v842 as port

    receipts.mkdir(parents=True, exist_ok=True)
    records = {}
    for name, digest in port.SOURCE_PINS.items():
        native = work / "native" / name
        assert hashlib.sha256(native.read_bytes()).hexdigest() == digest, name
        result = subprocess.run(["gcov", "-b", "-c", "-j", f"liboracle.so-{native.stem}.gcno"],
                                cwd=native.parent, text=True, capture_output=True, check=True)
        (receipts / f"coverage-{native.stem}.log").write_text(result.stdout + result.stderr)
        packed = native.parent / f"liboracle.so-{native.stem}.gcov.json.gz"
        (receipts / packed.name).write_bytes(packed.read_bytes())
        data = json.loads(gzip.decompress(packed.read_bytes()))
        for source in data["files"]:
            lines = source["lines"]
            branches = [branch for line in lines for branch in line.get("branches", [])]
            executed = re.search(r"Branches executed:([0-9.]+)% of (\d+)", result.stdout)
            # gcov's CFG execution statistic differs from the count of branches
            # attached to an executed source line, especially for array reductions.
            assert executed and int(executed[2]) == len(branches), name
            records[source["file"]] = {
                "source_sha256": digest,
                "lines": len(lines), "executed_lines": sum(line["count"] > 0 for line in lines),
                "branches": len(branches),
                "executed_branches": round(float(executed[1]) * len(branches) / 100),
                "taken_branches": sum(branch["count"] > 0 for branch in branches),
                "gcov_summary": result.stdout,
                "branch_gaps": [{"line": line["line_number"], "line_count": line["count"],
                                 "branch_counts": [b["count"] for b in line.get("branches", [])]}
                                for line in lines if any(b["count"] == 0 for b in line.get("branches", []))],
            }
    (receipts / "coverage-summary.json").write_text(json.dumps({
        "schema": "p15.ugwp-ngw-coverage/v2", "acceptance": False,
        "compiler": "gfortran 15.2 GNU strict --coverage", "files": records,
    }, indent=2) + "\n")

    module = Path(port.__file__)
    rows = []
    for line_number, line in enumerate(module.read_text().splitlines(), 1):
        for match in re.finditer(r"([A-Za-z_0-9]+\.F):(\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*)", line):
            name, ranges = match.groups()
            paths = list(reference.rglob(name))
            assert len(paths) == 1, (name, paths)
            source_lines = paths[0].read_bytes().splitlines(keepends=True)
            for span in ranges.split(","):
                parts = span.split("-")
                first, last = int(parts[0]), int(parts[-1])
                assert 1 <= first <= last <= len(source_lines), (name, span)
                rows.append({"citation": f"{name}:{span}",
                             "source_path": str(paths[0].relative_to(reference)),
                             "first_line": first, "last_line": last,
                             "window_sha256": hashlib.sha256(b"".join(source_lines[first-1:last])).hexdigest(),
                             "python_line": line_number})
    (receipts / "mpas-citations-v842.json").write_text(json.dumps({
        "schema": "p15.mpas-v842-source-windows/v1", "source_release": "v8.4.2",
        "module_sha256": hashlib.sha256(module.read_bytes()).hexdigest(), "rows": rows,
    }, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--receipts", type=Path, required=True)
    args = parser.parse_args()
    capture(args.reference.resolve(), args.work.resolve(), args.receipts.resolve())
