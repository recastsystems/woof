"""Capture actual CLWRF calls from a private strict native WRF forecast.

Instrument before building. The original public module is hash checked.
The first three monitor-rank calls record inputs, prior carrier words and
native outputs. No model value or clock is changed.
"""
from pathlib import Path
import argparse
import difflib
import hashlib
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from woof.core.clwrf import HISTORY, INPUTS

SOURCE_SHA256 = "2ec1e964a9ae94244d513fac38e0ce9a33212b21ca232fcc01de890afd152a54"


def instrument(root, patch):
    path = root / "phys/module_diag_cl.F"
    original = path.read_bytes()
    actual = hashlib.sha256(original).hexdigest()
    if actual != SOURCE_SHA256:
        raise ValueError(f"{path}: expected public source {SOURCE_SHA256}, got {actual}")
    source = original.decode()
    declarations = """   INTEGER, SAVE :: lane_call = 0
   INTEGER :: lane_u, lane_i, lane_j, lane_i1, lane_j1, lane_loc(2)
   LOGICAL :: lane_capture
   CHARACTER(LEN=32) :: lane_path
"""
    region = "(lane_i:lane_i1,lane_j:lane_j1)"
    slots = [name.removeprefix("clwrf_") for name in HISTORY]
    fields = ["skintemp" if name == "tsk" else name for name in INPUTS]

    def arrays(names):
        return "   WRITE(lane_u) " + ", &\n    ".join(name + region for name in names) + "\n"

    before = """   lane_capture = lane_call < 3 .and. wrf_dm_on_monitor()
   IF (lane_capture) THEN
     lane_call=lane_call+1
     lane_loc=maxloc(rainncv(i_start(1):i_end(1),j_start(1):j_end(1)))
     lane_i=max(i_start(1),min(i_end(1)-15,i_start(1)+lane_loc(1)-8))
     lane_j=max(j_start(1),min(j_end(1)-15,j_start(1)+lane_loc(2)-8))
     lane_i1=min(i_end(1),lane_i+15);lane_j1=min(j_end(1),lane_j+15)
     WRITE(lane_path,'(A,I3.3,A)') 'lane-real-clwrf-',lane_call,'.bin'
     OPEN(newunit=lane_u,file=trim(lane_path),form='unformatted',access='stream', &
       status='replace',convert='little_endian')
     WRITE(lane_u) 'CLWRF001',lane_i1-lane_i+1,lane_j1-lane_j+1,nsteps,clwrfH
     IF (is_restart) THEN
       WRITE(lane_u) 1
     ELSE
       WRITE(lane_u) 0
     ENDIF
     WRITE(lane_u) dt,xtime,curr_secs2
""" + arrays(fields) + arrays(slots) + "   ENDIF\n"
    after = "   IF (lane_capture) THEN\n" + arrays(slots) + "   WRITE(lane_u) nsteps\n   CLOSE(lane_u)\n   ENDIF\n"
    for needle, replacement in (
        ("   REAL                                      :: t273\n", declarations),
        ("   t273  = 273.\n", before),
        ("   END SUBROUTINE clwrf_output_calc", after),
    ):
        if source.count(needle) != 1:
            raise ValueError(f"{path}: native anchor is not unique: {needle!r}")
        source = (source.replace(needle, needle + replacement, 1)
                  if ":: t273" in needle else source.replace(needle, replacement + needle, 1))
    patch.write_text("".join(difflib.unified_diff(original.decode().splitlines(True),
        source.splitlines(True), fromfile="a/phys/module_diag_cl.F", tofile="b/phys/module_diag_cl.F")))
    path.write_text(source)
    print(json.dumps({"source": str(path), "source_sha256": actual,
        "instrumented_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "patch": str(patch)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wrf-source", type=Path, required=True)
    parser.add_argument("--patch", type=Path, required=True)
    args = parser.parse_args()
    instrument(args.wrf_source, args.patch)
