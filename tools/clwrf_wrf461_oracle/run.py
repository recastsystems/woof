"""Compile the unchanged public-domain stock WRF CLWRF module and record calls.

Usage: python tools/clwrf_wrf461_oracle/run.py SOURCE_PHYS OUTPUT_DIR
Constructed inputs exercise window boundaries; these are not forecast captures.
"""
from pathlib import Path
import hashlib, json, re, subprocess, sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from woof.core.clwrf import HISTORY, INPUTS

source, out = map(Path, sys.argv[1:])
out.mkdir(parents=True, exist_ok=True)
original = (source / "module_diag_cl.F").read_bytes()
(out / "module_diag_cl.F").write_bytes(original)
text = original.decode()
signature = text[text.index("   SUBROUTINE clwrf_output_calc"):text.index("!----------------------------------------------------------------------")]
formals = re.sub(r"!.*", "", signature).replace("&", "").split("(", 1)[1].split(")", 1)[0]
formals = formals.replace("\n", "").replace(" ", "").lower().split(",")
slots = [slot.removeprefix("clwrf_") for slot in HISTORY]
mapping = dict(skintemp="tsk", clwrfh="interval")
arguments = []
for name in formals:
    if name in (*INPUTS, *slots, "nsteps", "dt", "xtime", "curr_secs2", "is_restart"):
        value = name
    elif name in mapping:
        value = mapping[name]
    else:
        value = {"ide": "4", "jde": "3", "ime": "3", "jme": "2", "ipe": "3", "jpe": "2",
                 "i_start": "[1]", "i_end": "[3]", "j_start": "[1]", "j_end": "[2]"}.get(name, "1")
    arguments.append(name + "=" + value)
stubs = '''module module_dm
contains
real function wrf_dm_sum_real(x)
real :: x
wrf_dm_sum_real=x
end function
subroutine wrf_dm_maxval
end subroutine
end module
module module_configure
end module
logical function wrf_dm_on_monitor()
wrf_dm_on_monitor=.false.
end function
subroutine wrf_debug(level,message)
integer :: level
character(*) :: message
end subroutine
'''
(out / "stubs.f90").write_text(stubs)
program = '''program oracle
use module_diag_cl
implicit none
integer :: s,n,flag,interval,nsteps
real :: dt,xtime,curr_secs2
logical :: is_restart
'''
program += "\n".join("real :: " + name + "(3,2)" for name in (*INPUTS, *slots)) + "\n"
program += "\n".join(name + "=-123.25" for name in slots) + "\nnsteps=0\n"
program += "read(*,*) n\ndo s=1,n\nread(*,*) dt,xtime,curr_secs2,flag,interval\nis_restart=flag/=0\n"
program += "\n".join("read(*,*) " + name for name in INPUTS) + "\n"
program += "call clwrf_output_calc(&\n" + ", &\n".join(arguments) + ")\n"
program += "\n".join("print '(6(Z8.8,1X))',transfer(" + name + ",[0,0,0,0,0,0])" for name in slots)
program += "\nprint '(Z8.8)',nsteps\nenddo\nend program\n"
(out / "main.f90").write_text(program)
times = [(20., float(s * 20), 0, 1) for s in range(15)]
times += [(7., float(s * 7), 0, 1) for s in range(18)]
times += [(30., 0., 1, 1), (30., 30., 1, 1), (30., 60., 1, 1), (30., 90., 1, 1),
          (60., 0., 0, 1), (60., 60., 0, 1)]
times += [(1., 16777215., 0, 1), (1., 16777216., 0, 1)]
data, clock, lines = [], [], [str(len(times))]
cell = np.arange(6, dtype=np.float32).reshape(2, 3)
for s, (dt, seconds, restart, interval) in enumerate(times):
    phase = (s % 7) - 3
    values = np.array([280. + cell + phase, .01 + cell * .001 + phase * .0001,
        290. + cell + phase, 3. + cell + phase, -4. + cell - phase,
        .1 + cell * .02 + (s % 4) * .01, .2 + cell * .03 + (s % 3) * .02], dtype=np.float32)
    row = np.array([dt, np.float32(seconds / 60), seconds, restart, interval], dtype=np.float32)
    data.append(values); clock.append(row)
    lines.append(" ".join(format(float(x), ".9g") for x in row))
    lines.extend(" ".join(format(float(x), ".9g") for x in field.ravel()) for field in values)
(out / "inputs.txt").write_text("\n".join(lines) + "\n")
command = ["gfortran", "-cpp", "-DNMM_CORE=0", "-ffree-form", "-ffree-line-length-none", "-O0",
           "-ffp-contract=off", "-fno-tree-vectorize", "-fcheck=bounds", "stubs.f90", "module_diag_cl.F", "main.f90", "-o", "oracle"]
subprocess.run(command, cwd=out, check=True)
with (out / "inputs.txt").open() as inp:
    result = subprocess.run([str(out.resolve() / "oracle")], cwd=out, stdin=inp, capture_output=True, text=True, check=True)
(out / "words.txt").write_text(result.stdout)
words = np.array([int(word, 16) for word in result.stdout.split()], dtype=np.uint32).reshape(len(times), -1)
np.savez_compressed(out / "vectors.npz", inputs=np.array(data), clock=np.array(clock),
    expected=words[:, :-1].reshape(len(times), len(slots), 2, 3), counters=words[:, -1], slots=np.array(list(HISTORY)))
receipt = dict(source_sha256=hashlib.sha256(original).hexdigest(), authority="WRF v4.6.1/v4.7.1 public-domain module_diag_cl.F",
    kind="constructed inputs, not a real-run capture", command=command, calls=len(times), fields=len(slots), words=int(words.size),
    compiler=subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0])
(out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt))
