#!/usr/bin/env python3
"""Add observation-only DFI tile captures to a PRIVATE strict native build.

The original arithmetic is unchanged. The tile is 8 x 4 x 8 interior words
per three-dimensional carrier. Each pre/post pair includes the actual model
inputs and accumulators, not a generated atmospheric fixture. Whole forecast
qualification is separate from this module capture.
"""
from pathlib import Path
import argparse
import hashlib
import json

PIN = "c7b94989ccdf637ebe756b9dcdff26d5b9ba798852ec74c3415a853a32a4ae0f"
FIELDS = [
    ("mu_2", "dfi_mu", 2), ("u_2", "dfi_u", 3),
    ("v_2", "dfi_v", 3), ("w_2", "dfi_w", 3),
    ("ww", "dfi_ww", 3), ("t_2", "dfi_t", 3),
    ("phb", "dfi_phb", 3), ("ph0", "dfi_ph0", 3),
    ("php", "dfi_php", 3), ("p", "dfi_p", 3),
    ("ph_2", "dfi_ph", 3), ("tke_2", "dfi_tke", 3),
    ("al", "dfi_al", 3), ("alt", "dfi_alt", 3),
    ("pb", "dfi_pb", 3), ("moist", "dfi_moist", 4),
    ("scalar", "dfi_scalar", 4),
]
RADII = ["re_cloud", "re_ice", "re_snow"]
GSFC = [f"re_{x}_gsfc" for x in ("cloud", "rain", "ice", "snow", "graupel", "hail")]


def capture_source():
    lines = ["""
SUBROUTINE lane_dfi_capture(grid,mode)
 USE module_domain, ONLY: domain
 USE module_state_description
 IMPLICIT NONE
 TYPE(domain) :: grid
 INTEGER, INTENT(IN) :: mode
 INTEGER :: unit,i0,i1,j0,j1,k0,k1,n,ios
 CHARACTER(160) :: filename
 ! A single rank owns this fixed global interior tile. No arithmetic changes.
 IF (grid%id /= 1) RETURN
 IF (grid%sp31 > 31 .OR. grid%ep31 < 38) RETURN
 IF (grid%sp33 > 31 .OR. grid%ep33 < 38) RETURN
 i0=31; i1=38; j0=31; j1=38; k0=2; k1=5
 WRITE(filename,'(A,I1,A,I1,A,I5.5,A,I1,A)') &
  'lane-dfi-o',grid%dfi_opt,'-s',grid%dfi_stage,'-t',grid%itimestep,'-m',mode,'.bin'
 OPEN(newunit=unit,file=TRIM(filename),access='stream',form='unformatted', &
      status='replace',action='write',iostat=ios)
 IF (ios /= 0) CALL wrf_error_fatal('DFI capture open failed')
 WRITE(unit) 1145456945,grid%id,grid%dfi_opt,grid%dfi_stage,grid%itimestep,mode, &
             grid%dfi_savehydmeteors,grid%mp_physics_dfi,num_moist,num_scalar
 WRITE(unit) grid%hcoeff(grid%itimestep+1),grid%hcoeff_tot
"""
    ]
    def emit(field, acc, dims):
        extent = "i0:i1,j0:j1" if dims == 2 else "i0:i1,k0:k1,j0:j1"
        if dims == 4:
            extent += ",:"
        x, a = f"grid%{field}({extent})", f"grid%{acc}({extent})"
        lines.extend([f" n=SIZE({x})", " WRITE(unit) n", f" WRITE(unit) {x},{a}"])
    for row in FIELDS:
        emit(*row)
    lines.append(" IF (grid%mp_physics_dfi /= NUWRF4ICESCHEME_DFI) THEN")
    for f in RADII:
        emit(f, f"dfi_{f}", 3)
    lines.append(" ELSE")
    for f in GSFC:
        emit(f, f"dfi_{f}", 3)
    lines.extend([" ENDIF", " CLOSE(unit)", "END SUBROUTINE lane_dfi_capture\n"])
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build", type=Path, required=True)
    p.add_argument("--receipt", type=Path, required=True)
    args = p.parse_args()
    path = args.build / "share/dfi.F"
    before = path.read_bytes()
    if hashlib.sha256(before).hexdigest() != PIN:
        raise SystemExit("private build's original dfi.F does not match WRF pin")
    text = before.decode()
    anchor = "      hn = grid%hcoeff(grid%itimestep+1)"
    assert text.count(anchor) == 1
    text = text.replace(anchor, "      CALL lane_dfi_capture(grid,0)\n" + anchor)
    anchor = "      grid%hcoeff_tot = grid%hcoeff_tot + hn"
    assert text.count(anchor) == 1
    text = text.replace(anchor, anchor + "\n      CALL lane_dfi_capture(grid,1)")
    # Mode 2/3 brackets only the original normalization block, before the
    # separate surface restoration and optional rebalance in this routine.
    anchor = "      grid%MU_2(:,:)    = grid%dfi_mu(:,:)      / grid%hcoeff_tot"
    assert text.count(anchor) == 1
    text = text.replace(anchor, "      CALL lane_dfi_capture(grid,2)\n" + anchor)
    anchor = "     if ( grid%sf_surface_physics .EQ. RUCLSMSCHEME ) then\n!      grid%qvg(:,:)"
    assert text.count(anchor) == 1
    text = text.replace(anchor, "      CALL lane_dfi_capture(grid,3)\n" + anchor)
    restored = text
    for mode in range(4):
        restored = restored.replace(f"      CALL lane_dfi_capture(grid,{mode})\n", "")
    assert restored.encode() == before, 'instrumentation changed original native statements'
    text += capture_source()
    path.write_text(text)
    args.receipt.write_text(json.dumps({
        "path": str(path), "before_sha256": PIN,
        "after_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "capture_scope": "interior tile of real native DFI accumulation and normalization",
        "tile": {"i": [31, 38], "j": [31, 38], "k": [2, 5]},
        "native_arithmetic_changed": False,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
