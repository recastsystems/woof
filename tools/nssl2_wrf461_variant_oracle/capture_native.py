"""Add output-only, first-call NSSL capture to the pinned native WRF module.

The capture contains six wettest columns, or a complete native X row, selected
before the call, at every level. It records both sides of the actual driver
call. No input, tendency, selector or executable physics statement changes.
Reversing the four marked insertions recovers the pinned source exactly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

SOURCE_SHA = "1eb1b138b75ff3b0cfe33c23779f4ec9b72e57a5455a53ef11c9e55ae0f42722"
STATE = ("th", "qv", "qc", "qr", "qi", "qs", "qh", "qhl", "ccw",
         "crw", "cci", "csw", "chw", "chl", "cn", "vhw", "vhl",
         "zrw", "zhw", "zhl")
ENV = ("p", "pii", "dn", "dz", "w")
DIAGNOSTIC = ("dbz", "re_cloud", "re_ice", "re_snow")
SURFACE = ("rainnc", "rainncv", "snownc", "snowncv", "grplnc",
           "grplncv", "hailnc", "hailncv", "sr")
REQUIRED = {"qv", "qc", "qr", "qs", "qh", "p", "pii", "dn", "dz", "w",
            "rainnc", "rainncv"}
# WRF passes null Registry scalar slots as PRESENT optional arrays. Never
# inspect an inactive package: use the same category guards as its gather.
ACTIVE = {
    "qhl": "lhl > 1", "ccw": "ipconc >= 5", "crw": "ipconc >= 5",
    "cci": "ipconc >= 5", "csw": "ipconc >= 5", "chw": "ipconc >= 5",
    "chl": "ipconc >= 5 .and. lhl > 1", "cn": "lccn > 1 .and. flag_ccn",
    "vhw": "lvh > 0", "vhl": "lvhl > 0", "zrw": "lzr > 0",
    "zhw": "lzh > 0", "zhl": "lzhl > 0",
    # module_physics_init.F:985-1023 disables has_req* for one-moment
    # NSSL. PRESENT alone does not make these Registry actuals defined.
    "re_cloud": "ipconc >= 5", "re_ice": "ipconc >= 5",
    "re_snow": "ipconc >= 5",
}


def marked(label, body):
    return f"\n! NSSL_CAPTURE_BEGIN {label}\n{body}\n! NSSL_CAPTURE_END {label}\n"


def instrument(source, *, wet_threshold=0.0, complete_row=False):
    if not math.isfinite(wet_threshold) or wet_threshold < 0:
        raise ValueError("wet threshold must be finite and nonnegative")
    if hashlib.sha256(source.encode()).hexdigest() != SOURCE_SHA:
        raise ValueError("capture requires the unchanged pinned WRF 4.6.1 module")
    start = source.index("SUBROUTINE nssl_2mom_driver(")
    end = source.index("END SUBROUTINE nssl_2mom_driver", start)
    original = source[start:end]
    declaration = """      integer :: cap_unit, cap_n, cap_i, cap_j, cap_c
      integer :: cap_loc(2), cap_xy(2,6)
      real :: cap_score(its:ite,jts:jte)
      real :: cap_cells(kte-kts+1,6,29), cap_surface(6,9)
      logical, save :: cap_done = .false.
      logical :: cap_this
      logical, external :: wrf_dm_on_monitor
      real :: cap_params(10)"""
    before = """      cap_this = .not. cap_done .and. wrf_dm_on_monitor()
      if (cap_this) then
        do cap_j=jts,jte
          do cap_i=its,ite
            cap_score(cap_i,cap_j) = sum(qc(cap_i,kts:kte,cap_j)) &
              + sum(qr(cap_i,kts:kte,cap_j)) + sum(qs(cap_i,kts:kte,cap_j)) &
              + sum(qh(cap_i,kts:kte,cap_j))
            if (present(qi)) cap_score(cap_i,cap_j) = cap_score(cap_i,cap_j) &
              + sum(qi(cap_i,kts:kte,cap_j))
          enddo
        enddo
      endif
      CAPTURE_WET_TEST
      if (cap_this) then
        cap_done = .true.
        cap_n = min(6,(ite-its+1)*(jte-jts+1))
        do cap_c=1,cap_n
          cap_loc = maxloc(cap_score)
          cap_xy(:,cap_c) = cap_loc + [its-1,jts-1]
          cap_score(cap_xy(1,cap_c),cap_xy(2,cap_c)) = -huge(1.0)
        enddo
        open(newunit=cap_unit,file='nssl-native-capture.bin',status='new', &
             access='stream',form='unformatted',convert='little_endian')
        cap_params = [ccn,alphah,alphahl,cnoh,cnohl,cnor,cnos,rho_qh,rho_qhl,rho_qs]
        write(cap_unit) 1,cap_n,kte-kts+1,itimestep,dtp, &
             ipconc,merge(1,0,lhl>1),merge(2,merge(1,0,lvh>0),lvhl>0), &
             lccn,icdx,icdxhl,cap_params,cap_xy(:,1:cap_n)
        call capture_snapshot()
      endif"""
    if complete_row:
        declaration = declaration.replace("cap_xy(2,6)", "cap_xy(2,ite-its+1)").replace(
            "cap_cells(kte-kts+1,6,29), cap_surface(6,9)",
            "cap_cells(kte-kts+1,ite-its+1,29), cap_surface(ite-its+1,9)")
        first=before.index("        cap_n = min(6,")
        last=before.index("        open(newunit",first)
        before=before[:first]+"""        cap_n = ite-its+1
        cap_loc = maxloc(cap_score)
        do cap_c=1,cap_n
          cap_xy(:,cap_c) = [its+cap_c-1, jts+cap_loc(2)-1]
        enddo
"""+before[last:]
        before=before.replace("write(cap_unit) 1,cap_n", "write(cap_unit) 2,cap_n")
    wet_test = ("if (cap_this) cap_this = itimestep > 1 .and. "
                f"maxval(cap_score) > {wet_threshold:.9e}") if wet_threshold else ""
    before = before.replace("CAPTURE_WET_TEST", wet_test)
    after = """      if (cap_this) then
        call capture_snapshot()
        close(cap_unit)
      endif"""
    writer = ["  contains", "    subroutine capture_snapshot()",
              "      cap_cells = 0.0", "      cap_surface = 0.0",
              "      cap_cells(:,:,26:29) = -999.0",
              "      do cap_c=1,cap_n", "        cap_i=cap_xy(1,cap_c)",
              "        cap_j=cap_xy(2,cap_c)"]
    for i, name in enumerate((*STATE, *ENV, *DIAGNOSTIC), 1):
        assignment = f"cap_cells(:,cap_c,{i}) = {name}(cap_i,kts:kte,cap_j)"
        guards = []
        if name not in REQUIRED:
            guards.append(f"present({name})")
        if name in ACTIVE:
            guards.append(ACTIVE[name])
        for guard in guards:
            writer.append(f"        if ({guard}) then")
        writer.append("          " + assignment)
        writer.extend("        endif" for _ in guards)
    for i, name in enumerate(SURFACE, 1):
        assignment = f"cap_surface(cap_c,{i}) = {name}(cap_i,cap_j)"
        if name not in REQUIRED:
            assignment = f"if (present({name})) " + assignment
        writer.append("        " + assignment)
    writer += ["      enddo", "      write(cap_unit) cap_cells(:,1:cap_n,:),cap_surface(1:cap_n,:)",
               "    end subroutine capture_snapshot"]
    pieces = [marked("declarations", declaration), marked("before", before),
              marked("after", after), marked("writer", "\n".join(writer))]
    changed = original.replace("      rdt = 1.0/dtp", pieces[0] + "      rdt = 1.0/dtp", 1)
    anchor = "     IF ( .not. flag_qi .and. ffrzs < 1.0 )"
    changed = changed.replace(anchor, pieces[1] + anchor, 1)
    changed = changed.rsplit("  RETURN", 1)
    changed = changed[0] + pieces[2] + "  RETURN" + changed[1] + pieces[3]
    result = source[:start] + changed + source[end:]
    recovered = result
    for piece in pieces:
        recovered = recovered.replace(piece, "", 1)
    assert recovered == source
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--wet-threshold", type=float, default=0.0,
                        help="capture the first post-initial call above this column condensate sum")
    parser.add_argument("--complete-row", action="store_true",
                        help="capture all X columns of one native Y row, preserving gathered-64 ordering")
    args = parser.parse_args()
    source = args.source.read_text()
    result = instrument(source, wet_threshold=args.wet_threshold, complete_row=args.complete_row)
    args.destination.write_text(result)
    args.destination.with_suffix(".capture.json").write_text(json.dumps({
        "source_sha256": SOURCE_SHA,
        "instrumented_sha256": hashlib.sha256(result.encode()).hexdigest(),
        "reversible_output_only": True, "cell_fields": [*STATE, *ENV, *DIAGNOSTIC],
        "wet_threshold": args.wet_threshold, "complete_native_row": args.complete_row,
        "surface_fields": SURFACE,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
