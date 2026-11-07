"""Digest a TEMF oracle output file, per grid, field and step.

    python3 digest_outputs.py <OUT.json> <arm> grid=cases.bin:out.bin ...

The parity test compares WOOF's words with WRF's through these digests: two
words agree exactly when every step's SHA-256 over a field's words agrees,
so the fixture carries 41 short digests per step instead of the words.  The
full output files stay on the oracle box for word-level diagnosis
(compare_arms.py).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from woof.verify.temf_oracle import (ALL_FIELDS, read_cases,  # noqa: E402
                                      read_outputs, step_digests)


def main() -> int:
    out_path, arm = Path(sys.argv[1]), sys.argv[2]
    doc = json.loads(out_path.read_text()) if out_path.exists() else {}
    doc.setdefault("fields", list(ALL_FIELDS))
    arms = doc.setdefault("arms", {})
    for spec in sys.argv[3:]:
        grid, files = spec.split("=", 1)
        cases_path, out_file = files.split(":", 1)
        cases = read_cases(cases_path)
        arms.setdefault(arm, {})[grid] = step_digests(
            read_outputs(out_file, cases))
    out_path.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
