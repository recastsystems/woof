"""Add one-shot binary taps to a private strict WRF 4.7.1 build.

The expressions and loop bounds stay byte-for-byte native. Set WOOF_HS_CAPTURE
to an existing directory for a one-rank em_heldsuarez run. HSRAD call 2 and
momentum call 7 capture evolved, native-produced inputs, rather than profiles
made by the oracle. Output is little-endian, Fortran array order.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

PINS = {
    "phys/module_ra_hs.F": "7a83582205dedd31071a8fbb3307e227be7e244f282db278c84368e6ae232e3e",
    "dyn_em/module_damping_em.F": "933628eae4ef09513d338317529d252933ef602c8659972b99ad67d70583c98f",
}

DECL = """
    INTEGER, SAVE :: hs_capture_calls = 0
    INTEGER :: hs_capture_unit, hs_capture_status
    LOGICAL :: hs_capture_active
    CHARACTER(LEN=1024) :: hs_capture_directory
"""


def before(kind: int, call: int, inputs: str) -> str:
    return f"""
    hs_capture_calls = hs_capture_calls + 1
    hs_capture_active = .false.
    CALL get_environment_variable('WOOF_HS_CAPTURE', hs_capture_directory, STATUS=hs_capture_status)
    IF (hs_capture_status == 0 .AND. hs_capture_calls == {call}) THEN
      OPEN(NEWUNIT=hs_capture_unit, FILE=TRIM(hs_capture_directory)//'/capture-{kind}.bin', &
           ACCESS='stream', FORM='unformatted', STATUS='replace', CONVERT='little_endian')
      WRITE(hs_capture_unit) {kind}, hs_capture_calls, &
        ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,its,ite,jts,jte,kts,kte
      WRITE(hs_capture_unit) {inputs}
      hs_capture_active = .true.
    END IF
"""


def instrument(root: Path) -> None:
    for relative, expected in PINS.items():
        path = root / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"{path}: expected pristine {expected}, got {actual}")
    for relative, start, end, anchor, kind, call, inputs, outputs in (
        ("phys/module_ra_hs.F", "   SUBROUTINE HSRAD(", "   END SUBROUTINE HSRAD",
         "   delty   = 60.0", 1, 2,
         "degrad,R_d,CP,xlat,p8w,p_phy,pi_phy,t_phy,RTHRATEN", "RTHRATEN"),
        ("dyn_em/module_damping_em.F", "  SUBROUTINE held_suarez_damp(",
         "  END SUBROUTINE held_suarez_damp", "    sigb=0.7", 2, 7,
         "p,pb,ru,rv,ru_tend,rv_tend", "ru_tend,rv_tend"),
    ):
        path = root / relative
        source = path.read_text()
        begin = source.index(start)
        finish = source.index(end, begin)
        body = source[begin:finish]
        statement = body.index(anchor)
        # Declarations must precede the first executable statement. Comments
        # between the declarations and this statement need no modification.
        body = body[:statement] + DECL + before(kind, call, inputs) + body[statement:]
        body += (f"    IF (hs_capture_active) THEN\n"
                 f"      WRITE(hs_capture_unit) {outputs}\n"
                 "      CLOSE(hs_capture_unit)\n    END IF\n")
        path.write_text(source[:begin] + body + source[finish:])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("private_native_copy", type=Path)
    instrument(parser.parse_args().private_native_copy)
