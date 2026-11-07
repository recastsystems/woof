"""Instrument a private WRF copy to capture one real mp=55 wrapper call.

The pinned module is identical at v4.6.1 and v4.7.1. Numerical statements
are retained verbatim. Set WOOF_CAPTURE_MP55_PREFIX and an explicit
WOOF_CAPTURE_MP55_STEP for the native forecast. Only the monitor's first
tile at that step is captured, one complete west-east row of columns.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

PIN = "4cefa1a2f28c2bc36061100bd4bef10371643ccf1e80da43be96f8ba31843326"
INPUTS = ("p", "dz", "th", "qv", "qc", "qr", "nr", "qi1", "ni1", "ai1", "ci1",
          "qi2", "ni2", "ai2", "ci2", "qi3", "ni3", "ai3", "ci3")
OUTPUTS = INPUTS[2:] + ("diag_effc3d", "diag_effi3d", "diag_dbz3d",) + tuple(
    f"diag_{name}_" + str(cat) for cat in (1, 2, 3) for name in ("vmi3d", "di3d", "rhopo3d", "phii3d")) + (
    "diag_itype_1", "diag_itype_2", "diag_itype_3")


def instrument(source: bytes) -> bytes:
    if hashlib.sha256(source).hexdigest() != PIN:
        raise ValueError("native mp=55 module differs from the WRF v4.7.1 pin")
    text = source.decode("utf-8")
    start = text.index("  subroutine  MP_JENSEN_ISHMAEL(")
    end = text.lower().index("  end subroutine mp_jensen_ishmael", start)
    wrapper = text[start:end].replace("  subroutine  MP_JENSEN_ISHMAEL(",
        "  recursive subroutine MP_JENSEN_ISHMAEL(", 1)
    decl = """
    ! Setup-only native capture, no numerical expressions are changed.
    logical, save :: cap_claimed = .false.
    logical :: cap_call
    logical, external :: wrf_dm_on_monitor
    character(len=1024) :: cap_prefix, cap_step_text
    integer :: cap_status, cap_step, cap_unit, cap_nc, cap_nz, cap_j
"""
    anchor = "  !.. Accumulated precipitation in one timestep (zero out before microphysics)"
    if wrapper.count(anchor) != 1:
        raise ValueError("capture insertion anchor is not unique")
    setup = """
    cap_call = .false.
    !$omp critical (gpuwm_mp55_capture)
    if (.not.cap_claimed .and. wrf_dm_on_monitor()) then
      call get_environment_variable('WOOF_CAPTURE_MP55_PREFIX', cap_prefix, status=cap_status)
      if (cap_status == -1) error stop 'mp55 capture prefix too long'
      if (cap_status == 0 .and. len_trim(cap_prefix) > 0) then
        call get_environment_variable('WOOF_CAPTURE_MP55_STEP', cap_step_text, status=cap_status)
        if (cap_status /= 0) error stop 'mp55 capture requires explicit step'
        read(cap_step_text, *, iostat=cap_status) cap_step
        if (cap_status /= 0 .or. cap_step < 1) error stop 'invalid mp55 capture step'
        if (itimestep == cap_step .and. ite >= its .and. kte >= kts .and. jte >= jts) then
          cap_call = .true.
          cap_claimed = .true.
        endif
      endif
    endif
    !$omp end critical (gpuwm_mp55_capture)
    if (cap_call) then
      if (storage_size(1.0) /= 32) error stop 'mp55 capture requires REAL32'
      cap_nc = ite-its+1
      cap_nz = kte-kts+1
      cap_j = (jts+jte)/2
      open(newunit=cap_unit, file=trim(cap_prefix)//'-inputs.bin', status='new', &
           action='write', access='stream', form='unformatted', convert='little_endian')
      write(cap_unit) 1229146957, cap_nc, cap_nz
"""
    for name in INPUTS:
        setup += f"      write(cap_unit) {name}(its:ite,kts:kte,cap_j)\n"
    setup += """      write(cap_unit) rainnc(its:ite,cap_j), snownc(its:ite,cap_j)
      close(cap_unit)
      open(newunit=cap_unit, file=trim(cap_prefix)//'-want.bin', status='new', &
           action='write', access='stream', form='unformatted', convert='little_endian')
      write(cap_unit) 1229146959, cap_nc, cap_nz, 1, dt_in
    endif
"""
    # Declarations must precede the routine's first executable statement.
    wrapper = wrapper.replace(anchor, decl + "\n" + anchor + setup, 1)
    trailer = "\n    if (cap_call) then\n"
    for name in OUTPUTS:
        trailer += f"      write(cap_unit) {name}(its:ite,kts:kte,cap_j)\n"
    trailer += """      write(cap_unit) rainnc(its:ite,cap_j), rainncv(its:ite,cap_j), &
                      snownc(its:ite,cap_j), snowncv(its:ite,cap_j)
      close(cap_unit)
    endif
"""
    return (text[:start] + wrapper + trailer + text[end:]).encode("utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.read_bytes()
    result = instrument(source)
    # Refuse overwrite so an accidental repeat cannot corrupt the authority.
    with args.output.open("xb") as stream:
        stream.write(result)
    args.receipt.write_text(json.dumps({
        "source": str(args.source), "source_sha256": PIN,
        "instrumented": str(args.output), "instrumented_sha256": hashlib.sha256(result).hexdigest(),
        "capture": "first monitor tile, central west-east row, explicit model step",
        "capture_executed": False,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
