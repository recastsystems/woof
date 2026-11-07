"""Instrument the pinned public WRF module for real-input module replay.

Run on a private native build copy, then rebuild module_bl_eepsilon.o and
wrf.exe using the same strict flags as build.sh. The captured interior strip
must fit entirely in one MPI tile. It records calls 1, 301, ..., 1501, suitable
for dt=12 s hourly captures, in replay.py's stream layout. The sole arithmetic
change is the separately witnessed initialization of native undefined RADSUM.
"""
from pathlib import Path
import argparse
import difflib
import hashlib
import json

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", type=Path, required=True, help="unmodified pinned module")
parser.add_argument("--output", type=Path, required=True, help="module in a private native build")
parser.add_argument("--receipt", type=Path, required=True)
args = parser.parse_args()
raw = args.source.read_bytes()
if hashlib.sha256(raw).hexdigest() != "b1ea6db55866af15cb3c6962b1a15fa512eda803078f7304e6a133462d7e12de":
    parser.error("source is not the pinned public WRF v4.6.1/v4.7.1 EEPS module")
source = raw.decode().replace('\r\n', '\n')
modified = source.replace('      integer ::  i,j,k,zz,im,kx,pp', '''      integer ::  i,j,k,zz,im,kx,pp
      integer :: capture_unit
      logical :: capture_call
      character(len=80) :: capture_path''', 1)
inputs3 = 'u3d v3d t3d qv3d qc3d qi3d qr3d qs3d qg3d p3d pi3d rho3d rthraten dz8w'.split()
inputs2 = 'psfc qsfc tsk ust rmol wspd xland hfx qfx'.split()
carried = 'pek pep pek_adv pep_adv'.split()
outputs3 = 'rublten rvblten rthblten rqvblten rqcblten rqiblten pek pep pek_adv pep_adv'.split()
entry = '''      capture_call = its <= 40 .and. ite >= 103 .and. &
                     jts <= 75 .and. jte >= 75 .and. &
                     mod(itimestep-1,300) == 0 .and. itimestep <= 1501
      if (capture_call) then
         write(capture_path,'(A,I6.6,A)') 'eeps-real-',itimestep,'.bin'
         open(newunit=capture_unit,file=trim(capture_path), &
              access='stream',form='unformatted',status='replace', &
              convert='little_endian')
         write(capture_unit) 64, kte-kts+1, 1, itimestep, 1
         write(capture_unit) dt
'''
for field in inputs3:
    if field in ('qr3d', 'qs3d', 'qg3d'):
        entry += f'         if (present({field})) then\n'
        entry += f'            write(capture_unit) {field}(40:103,kts:kte,75)\n'
        entry += '         else\n            write(capture_unit) ((0.,i=40,103),k=kts,kte)\n         end if\n'
    else:
        entry += f'         write(capture_unit) {field}(40:103,kts:kte,75)\n'
for field in inputs2:
    entry += f'         write(capture_unit) {field}(40:103,75)\n'
for field in carried:
    entry += f'         write(capture_unit) {field}(40:103,kts:kte,75)\n'
entry += '         write(capture_unit) itimestep\n      end if\n'
modified = modified.replace('      im=ite-its+1', entry + '\n      im=ite-its+1', 1)
exit_code = '      if (capture_call) then\n'
for field in outputs3:
    exit_code += f'         write(capture_unit) {field}(40:103,kts:kte,75)\n'
for field in ('exch_h', 'exch_m'):
    exit_code += f'         write(capture_unit) {field}(40:103,kts:kte+1,75)\n'
for field in ('pblh', 'kpbl'):
    exit_code += f'         write(capture_unit) {field}(40:103,75)\n'
exit_code += '         close(capture_unit)\n      end if\n'
modified = modified.replace('   end subroutine eeps\n', exit_code + '   end subroutine eeps\n', 1)
assert modified.count('         minrad(j)  = 100.') == 1
modified = modified.replace('         minrad(j)  = 100.', '         minrad(j)  = 100.\n         radsum(j)  = 0.', 1)
args.output.write_text(modified)
args.receipt.parent.mkdir(parents=True, exist_ok=True)
args.receipt.with_suffix(".diff").write_text("".join(difflib.unified_diff(
    source.splitlines(True), modified.splitlines(True),
    fromfile="WRF-v4.7.1/module_bl_eepsilon.F", tofile="native-capture/module_bl_eepsilon.F")))
args.receipt.write_text(json.dumps({
    "stock_sha256": hashlib.sha256(raw).hexdigest(),
    "instrumented_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    "columns": [40,103], "row":75, "steps":[1,301,601,901,1201,1501],
    "arithmetic_change":"RADSUM initialized to zero; native undefined accumulator witness in synthetic oracle receipt"
}, indent=2)+"\n")
print(args.receipt)
