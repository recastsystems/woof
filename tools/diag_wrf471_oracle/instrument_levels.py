#!/usr/bin/env python3
"""Add a bounded raw-input capture to an owned, pinned native WRF checkout.

Only the native ZLD entry is instrumented. No expression in the original
routine changes. Two calls per tile are captured, at most four columns on
each horizontal axis, with every vertical level and the required wind faces.
The offline decoder is separate from WRF's integration loop.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MARKER = "      !  Silly, but transfer the small namelist.input array into the grid structure for output purposes."
DECLARATIONS = """      ! Input-only capture; the original routine below is unchanged.
      INTEGER, SAVE :: gpuwm_capture_count = 0
      INTEGER :: gpuwm_unit, gpuwm_status, gpuwm_ie, gpuwm_je
      CHARACTER(LEN=2048) :: gpuwm_directory, gpuwm_filename

"""
CAPTURE = """      CALL GET_ENVIRONMENT_VARIABLE('WOOF_DIAG_CAPTURE_DIR', gpuwm_directory, STATUS=gpuwm_status)
      IF (gpuwm_status == 0 .AND. LEN_TRIM(gpuwm_directory) > 0 .AND. gpuwm_capture_count < 2) THEN
         gpuwm_capture_count = gpuwm_capture_count + 1
         gpuwm_ie = MIN(its+3,ite,ide-1)
         gpuwm_je = MIN(jts+3,jte,jde-1)
         WRITE(gpuwm_filename,'(A,"/zld-",I6.6,"-",I6.6,"-",I2.2,".bin")') &
             TRIM(gpuwm_directory),its,jts,gpuwm_capture_count
         OPEN(NEWUNIT=gpuwm_unit,FILE=TRIM(gpuwm_filename),ACCESS='STREAM', &
              FORM='UNFORMATTED',STATUS='NEW',CONVERT='LITTLE_ENDIAN')
         WRITE(gpuwm_unit) 47101, gpuwm_ie-its+1, gpuwm_je-jts+1, kte-kts, num_z_levels
         WRITE(gpuwm_unit) its,gpuwm_ie,jts,gpuwm_je,kts,kte
         WRITE(gpuwm_unit) u(its:gpuwm_ie+1,kts:kte-1,jts:gpuwm_je)
         WRITE(gpuwm_unit) v(its:gpuwm_ie,kts:kte-1,jts:gpuwm_je+1)
         WRITE(gpuwm_unit) w(its:gpuwm_ie,kts:kte,jts:gpuwm_je)
         WRITE(gpuwm_unit) t(its:gpuwm_ie,kts:kte-1,jts:gpuwm_je)
         WRITE(gpuwm_unit) qv(its:gpuwm_ie,kts:kte-1,jts:gpuwm_je)
         WRITE(gpuwm_unit) zp(its:gpuwm_ie,kts:kte,jts:gpuwm_je)
         WRITE(gpuwm_unit) zb(its:gpuwm_ie,kts:kte,jts:gpuwm_je)
         WRITE(gpuwm_unit) pp(its:gpuwm_ie,kts:kte-1,jts:gpuwm_je)
         WRITE(gpuwm_unit) pb(its:gpuwm_ie,kts:kte-1,jts:gpuwm_je)
         WRITE(gpuwm_unit) p(its:gpuwm_ie,kts:kte-1,jts:gpuwm_je)
         WRITE(gpuwm_unit) pw(its:gpuwm_ie,kts:kte,jts:gpuwm_je)
         WRITE(gpuwm_unit) ht(its:gpuwm_ie,jts:gpuwm_je)
         WRITE(gpuwm_unit) z_levels(1:num_z_levels)
         CLOSE(gpuwm_unit)
      END IF

"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkout", type=Path)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    target = args.checkout.resolve() / "phys/module_diag_zld.F"
    manifest = json.loads((HERE / "sources.json").read_text())
    row = next(r for r in manifest["sources"] if r["path"] == "phys/module_diag_zld.F")
    original = target.read_bytes()
    if hashlib.sha256(original).hexdigest() != row["sha256"]:
        raise ValueError("refusing to instrument a source outside the native pin")
    text = original.decode()
    if text.count(MARKER) != 1:
        raise ValueError("native capture insertion point is not unique")
    changed = text.replace(MARKER, DECLARATIONS + CAPTURE + MARKER).encode()
    target.write_bytes(changed)
    args.receipt.write_text(json.dumps({
        "wrf": manifest["wrf"], "original_sha256": row["sha256"],
        "instrumented_sha256": hashlib.sha256(changed).hexdigest(),
        "instrumenter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source": str(target), "kind": "raw native ZLD entry; no arithmetic changes",
        "limit": "two calls per tile; at most 4x4 columns; all levels",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
