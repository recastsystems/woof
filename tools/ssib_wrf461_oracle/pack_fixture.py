"""Pack an oracle build into the committed test fixture.

    python3 pack_fixture.py BUILD_DIR tests/data/oracles/ssib/ssib-wrf461-columns.npz

Carries the exact input file the oracle ran (columns.bin), the reference
output (out-ref.bin, -O0 -finit-local-zero) and the column names, plus the
SHA-256 of each so the test can say which bytes it graded against.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def main(build: Path, out: Path) -> None:
    cols = (build / "columns.bin").read_bytes()
    ref = (build / "out-ref.bin").read_bytes()
    meta = json.loads((build / "columns.json").read_text())
    meta["columns_sha256"] = hashlib.sha256(cols).hexdigest()
    meta["out_ref_sha256"] = hashlib.sha256(ref).hexdigest()
    default = (build / "out-default-units.bin").read_bytes()
    meta["out_default_units_sha256"] = hashlib.sha256(default).hexdigest()
    meta["oracle"] = ("WRF v4.6.1 phys/module_sf_ssib.F (d66e442) at gfortran "
                      "-O0 -finit-local-zero, driven by "
                      "tools/ssib_wrf461_oracle/ssib_oracle.F90")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, columns=np.frombuffer(cols, np.uint8),
                        out_ref=np.frombuffer(ref, np.uint8),
                        out_default_units=np.frombuffer(default, np.uint8),
                        meta=np.frombuffer(json.dumps(meta).encode(), np.uint8))
    print(f"{out}: {out.stat().st_size} bytes")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
