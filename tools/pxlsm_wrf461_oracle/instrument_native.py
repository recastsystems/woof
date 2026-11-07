"""Add bounded entry/exit snapshots to a private, pinned native WRF copy.

No numerical statement is replaced. Run one MPI rank with no OpenMP tiles.
The capture path is enabled only by WOOF_PXLSM_CAPTURE_DIR. Steps 1, 61 and
121 each write a single-call case and the native result in schema.py order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import schema


def instrument(source: bytes) -> bytes:
    pin = json.loads((Path(__file__).with_name("source-pin.json")).read_text())
    entry = pin["files"]["phys/module_sf_pxlsm.F"]
    if hashlib.sha256(source).hexdigest() != entry["sha256"]:
        raise ValueError("native capture requires the pinned PXLSM source")
    text = source.decode()
    marker = "\n      ALN10  = ALOG(10.0)"
    declarations = """
! GPUWM capture declarations. Only private native proof builds use this.
      INTEGER :: capture_unit, capture_status, capture_nx, capture_ny
      INTEGER :: capture_hi(16)
      REAL :: capture_hf(8)
      LOGICAL :: capture_this_call
      CHARACTER(LEN=1024) :: capture_dir, capture_file
"""
    setup = """
! GPUWM capture entry. The unmodified science follows this snapshot.
      CALL GET_ENVIRONMENT_VARIABLE('WOOF_PXLSM_CAPTURE_DIR', &
                                    capture_dir, STATUS=capture_status)
      capture_this_call = capture_status == 0 .AND. &
                          LEN_TRIM(capture_dir) > 0 .AND. &
                          (ITIMESTEP == 1 .OR. ITIMESTEP == 61 .OR. ITIMESTEP == 121)
      IF (capture_this_call) THEN
         IF (ids /= 1 .OR. jds /= 1 .OR. its /= ids .OR. jts /= jds .OR. &
             MIN(ite,ide-1) /= ide-1 .OR. MIN(jte,jde-1) /= jde-1 .OR. &
             kts /= 1 .OR. NSOIL /= 2 .OR. NSCAT /= 16) THEN
            CALL wrf_error_fatal('PX capture requires one whole-domain tile, two soil layers and 16 soil categories')
         END IF
         capture_nx = ide-ids
         capture_ny = jde-jds
         capture_hi = 0
         capture_hi(1:10) = (/ capture_nx, capture_ny, NLCAT, NSCAT, 1, &
                              PXLSM_SMOIS_INIT, PXLSM_SOIL_NUDGE, &
                              pxlsm_modis_veg, ANAL_INTERVAL, ITIMESTEP /)
         capture_hf = 0.0
         capture_hf(1:7) = (/ DT, DTBL, curr_secs, DZS(1), DZS(2), ZS(1), ZS(2) /)
         WRITE(capture_file,'(A,A,I6.6,A)') TRIM(capture_dir), &
                                          '/case-native-step', ITIMESTEP, '.bin'
         OPEN(NEWUNIT=capture_unit, FILE=TRIM(capture_file), &
              ACCESS='stream', FORM='unformatted', CONVERT='little_endian', &
              STATUS='new', ACTION='write')
         WRITE(capture_unit) capture_hi, capture_hf
"""

    def write_planes(names, level=None):
        middle = f",{level}," if level else ","
        return "".join(
            f"         WRITE(capture_unit) {name.upper()}(ids:ide-1{middle}jds:jde-1)\n"
            for name in names)

    setup += write_planes(schema.IN_2D + schema.INOUT_2D + schema.OUT_2D)
    setup += write_planes(("isltyp",))
    setup += write_planes(schema.ATM_3D + schema.SOIL_3D, "1:2")
    setup += write_planes(("landusef",), "1:NLCAT")
    setup += write_planes(("soilctop", "soilcbot"), "1:NSCAT")
    setup += "         CLOSE(capture_unit)\n      END IF\n"
    finish = """
! GPUWM capture exit. Preserve every native output word, including residues.
      IF (capture_this_call) THEN
         WRITE(capture_file,'(A,A,I6.6,A)') TRIM(capture_dir), &
                                          '/out-native-step', ITIMESTEP, '.bin'
         OPEN(NEWUNIT=capture_unit, FILE=TRIM(capture_file), &
              ACCESS='stream', FORM='unformatted', CONVERT='little_endian', &
              STATUS='new', ACTION='write')
"""
    finish += write_planes(schema.INOUT_2D + schema.OUT_2D + ("isltyp",))
    finish += write_planes(schema.SOIL_3D, "1:2")
    finish += "         CLOSE(capture_unit)\n      END IF\n"
    if text.count(marker) != 1 or text.count("   END SUBROUTINE pxlsm") != 1:
        raise ValueError("pinned native PXLSM capture anchors are not unique")
    return text.replace(marker, declarations + setup + marker).replace(
        "   END SUBROUTINE pxlsm", finish + "   END SUBROUTINE pxlsm").encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        parser.error("write a separate instrumented file, not the source pin")
    source = args.source.read_bytes()
    patched = instrument(source)
    with args.output.open("xb") as stream:
        stream.write(patched)
    print(json.dumps({"source": str(args.source), "output": str(args.output),
                      "source_sha256": hashlib.sha256(source).hexdigest(),
                      "instrumented_sha256": hashlib.sha256(patched).hexdigest(),
                      "steps": [1, 61, 121], "numerical_statements_replaced": 0},
                     indent=2))


if __name__ == "__main__":
    main()
