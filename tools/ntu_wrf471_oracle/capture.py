#!/usr/bin/env python3
"""Add observation-only MP_NTU stream dumps to the pinned native source.

The physics statements remain verbatim. Each tile writes its real module
arguments before and after native timesteps 1 and 60, in little-endian form.
Run WRF with GFORTRAN_CONVERT_UNIT=little_endian:63 (unit 63 is reserved here).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from build import verify

FIELDS = "TH P DZ PII QV QC QR QI QS QG QH NC NR NI NS NG NH QDCN QTCN QCCN QRCN QNIN FI FS VI VS VG AI AS AG AH I3M".split()
OUTPUT = ["TH", *FIELDS[4:]]
SURFACE = "RAINNC RAINNCV SNOWNC SNOWNCV GRAPNC GRAPNCV HAILNC HAILNCV SR".split()


def block(output: bool) -> str:
    names = OUTPUT if output else FIELDS
    suffix = "out" if output else "in"
    lines = [
        "      IF (ITIMESTEP.EQ.1.OR.ITIMESTEP.EQ.60) THEN",
        f"         WRITE(lane_path,'(A,I6.6,A,I5.5,A,I5.5,A)') &",
        f"              'ntu-step',ITIMESTEP,'-i',ITS,'-j',JTS,'.{suffix}'",
        "         OPEN(UNIT=63,FILE=TRIM(lane_path),ACCESS='STREAM', &",
        "              FORM='UNFORMATTED',STATUS='REPLACE',ACTION='WRITE')",
        "         WRITE(63) INT(Z'4E545532'),MIN(ITE,IDE-1)-ITS+1, &",
        "              MIN(JTE,JDE-1)-JTS+1,KTE-KTS+1,ITIMESTEP,ITS,JTS,lane_ccnty",
        "         WRITE(63) DT_MP",
    ]
    for name in names:
        lines.append(f"         WRITE(63) ((({name}(lane_i,lane_k,lane_j), &")
        lines.append("              lane_i=ITS,MIN(ITE,IDE-1)), &")
        lines.append("              lane_j=JTS,MIN(JTE,JDE-1)),lane_k=KTS,KTE)")
    if not output:
        lines.extend([
            "         WRITE(63) (((W(lane_i,lane_k,lane_j), &",
            "              lane_i=ITS,MIN(ITE,IDE-1)), &",
            "              lane_j=JTS,MIN(JTE,JDE-1)),lane_k=KTS,KTE+1)",
        ])
    for name in SURFACE:
        lines.extend([
            f"         WRITE(63) (({name}(lane_i,lane_j), &",
            "              lane_i=ITS,MIN(ITE,IDE-1)),lane_j=JTS,MIN(JTE,JDE-1))",
        ])
    lines.extend(["         CLOSE(63)", "      ENDIF", ""])
    return "\n".join(lines)


def instrument(source: Path, licence: Path, destination: Path) -> dict:
    verify(source, licence)
    original = source.read_text()
    text = original.replace(
        "      CONTAINS\n",
        "      INTEGER, PRIVATE, SAVE :: lane_ccnty = 2\n      CONTAINS\n", 1)
    # Record the actual aerosol selector, without changing AERO_CONST.
    marker = "      CALL AERO_CONST(CCNTY)"
    if text.count(marker) != 1:
        raise ValueError("NTU_INIT AERO_CONST marker changed")
    text = text.replace(marker, "      lane_ccnty = CCNTY\n" + marker)
    start = text.index("      SUBROUTINE MP_NTU(")
    end = text.index("      END SUBROUTINE MP_NTU", start)
    part = text[start:end]
    part = part.replace("      DT = DT_MP\n", "      INTEGER :: lane_i,lane_j,lane_k\n      CHARACTER(LEN=128) :: lane_path\n\n" + block(False) + "      DT = DT_MP\n", 1)
    text = text[:start] + part + block(True) + text[end:]
    destination.write_text(text)
    receipt = {
        "source_sha256": hashlib.sha256(original.encode()).hexdigest(),
        "instrumented_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "scope": "module-entry and module-return real forecast arguments; no physics edits",
        "capture_timesteps": [1, 60], "unit": 63,
        "layout": {"header": "int32 magic,nx,ny,nz,step,its,jts,ccnty; float32 dt", "input_3d": FIELDS + ["W (nz+1 levels)"], "output_3d": OUTPUT, "surface": SURFACE, "array_order": "k,j,i; i fastest", "byte_order": "little endian"},
    }
    destination.with_suffix(".capture.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "licence", "destination"):
        ap.add_argument(name, type=Path)
    args = ap.parse_args()
    print(json.dumps(instrument(args.source, args.licence, args.destination), indent=2))
