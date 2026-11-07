#!/usr/bin/env python3
"""Add receipt-only capture to a clean, pinned CU_OSAS source copy.

Compile with the strict native flags in README.md, then run one MPI rank
with OMP_NUM_THREADS=1. The native XKT2 draw and physics are unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

_HOOK = """
     if (.not. osas_capture_active) then
       if (any(RTHCUTEN(its:ite,kts:kte,j) /= 0.) .or. &
           any(RQVCUTEN(its:ite,kts:kte,j) /= 0.) .or. &
           any(RAINCV(its:ite,j) /= 0.)) osas_capture_active = .true.
       if (osas_capture_active .or. (ITIMESTEP == 1 .and. J == (jts+jte)/2)) then
       open(newunit=osas_capture_unit,file='osas-native-capture.bin', &
            access='stream',form='unformatted',status='replace',convert='little_endian')
       write(osas_capture_unit) IM, 1, KX, 1
       write(osas_capture_unit) DT, STEPCU, ITIMESTEP, 0
       write(osas_capture_unit) T3D(its:ite,kts:kte,j), QV3D(its:ite,kts:kte,j), &
            QC3D(its:ite,kts:kte,j), QI3D(its:ite,kts:kte,j), &
            U3D(its:ite,kts:kte,j), V3D(its:ite,kts:kte,j), &
            PCPS(its:ite,kts:kte,j), PI3D(its:ite,kts:kte,j), &
            RHO3D(its:ite,kts:kte,j), DZ8W(its:ite,kts:kte,j), &
            W(its:ite,kts:kte+1,j), P8W(its:ite,kts:kte+1,j), XLAND(its:ite,j), XKT2
       close(osas_capture_unit)
       open(newunit=osas_capture_unit,file='osas-native-output.bin', &
            access='stream',form='unformatted',status='replace',convert='little_endian')
       write(osas_capture_unit) RTHCUTEN(its:ite,kts:kte,j), RQVCUTEN(its:ite,kts:kte,j), &
            RQCCUTEN(its:ite,kts:kte,j), RQICUTEN(its:ite,kts:kte,j), &
            RAINCV(its:ite,j), PRATEC(its:ite,j), HTOP(its:ite,j), HBOT(its:ite,j)
       close(osas_capture_unit)
       open(newunit=osas_capture_unit,file='osas-native-selection.txt',status='replace')
       write(osas_capture_unit,*) ITIMESTEP, J, osas_capture_active
       close(osas_capture_unit)
       endif
     endif
"""


def instrument(root: Path) -> dict:
    source = root / "phys/module_cu_osas.F"
    if source.is_symlink():
        raise ValueError("capture requires a writable source copy, not a symlink")
    raw = source.read_bytes()
    pins = json.loads(Path(__file__).with_name("source-pins.json").read_text())
    pin = next(row for row in pins["files"] if row["file"] == "phys/module_cu_osas.F")
    digest = hashlib.sha256(raw).hexdigest()
    if digest != pin["sha256"]:
        raise ValueError(f"CU_OSAS source hash {digest} differs from pin {pin['sha256']}; use a clean copy")
    text = raw.decode("utf-8")
    declaration = "      INTEGER ::                        ICLDCK"
    end = "   ENDDO    ! Outer most J loop"
    if text.count(declaration) != 1 or text.count(end) != 1:
        raise ValueError("pinned CU_OSAS capture insertion points are absent or ambiguous")
    text = text.replace(declaration,
        "      INTEGER :: osas_capture_unit\n"
        "      LOGICAL, SAVE :: osas_capture_active = .false.\n" + declaration, 1)
    text = text.replace(end, _HOOK + end, 1)
    updated = text.encode("utf-8")
    source.write_bytes(updated)
    return dict(source=str(source), original_sha256=digest,
                instrumented_sha256=hashlib.sha256(updated).hexdigest(),
                bytes=len(updated), receipt_only=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("wrf_source", type=Path, help="own clean source copy, never a shared build")
    args = p.parse_args()
    print(json.dumps(instrument(args.wrf_source), indent=2))


if __name__ == "__main__":
    main()
