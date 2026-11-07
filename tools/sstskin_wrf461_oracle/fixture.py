"""Generate word captures from the unmodified WRF column routine."""
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import numpy as np

def generate(build):
    build = Path(build)
    n, ns = 72, 48
    k = np.arange(n)
    f = lambda a: np.asarray(a, dtype=np.float32)
    initial = {
        "xland": f([2.] * 64 + [1., 1.5, 1.4999, 1.5001, 1., 2., 1.5, 2.]),
        "tsk": f(272. + (k % 4) * 14.),
        "ust": f(np.asarray([0., .015, .2, 1.])[k // 4 % 4]),
        "emiss": f(.96 + (k % 3) * .01),
        "sst": f(273. + (k % 4) * 14.),
        "dtw": f(np.asarray([0., .05, .5, 4.])[k // 16 % 4]),
    }
    initial["sstsk"] = initial["tsk"].copy()
    initial["tslb"] = f(initial["tsk"] - 3.)
    flux = []
    for step in range(ns):
        flux.append({
            "glw": f(np.asarray([100.,350.,500.,850.])[k // 16 % 4]),
            "gsw": f(np.full(n, max(0., np.sin(2. * np.pi * step / ns)) * 1100.)),
            "hfx": f(np.asarray([-150.,0.,100.,500.])[k // 4 % 4]),
            "qfx": f(np.asarray([0.,.00002,.0002,-.00001])[k % 4]),
        })
    arms = []
    for update in (0,1):
        inp, out = build / "input.bin", build / "output.bin"
        with inp.open("wb") as h:
            h.write(struct.pack("<iiiff",n,ns,update,300.,5.67051e-8))
            for a in initial.values(): h.write(a.tobytes())
            for step in flux:
                for a in step.values(): h.write(a.tobytes())
        subprocess.run([str(build / "oracle"),str(inp),str(out)],check=True)
        expected = np.fromfile(out,dtype=np.float32).reshape(ns,4,n)
        arms.append({"sst_update":update,"expected_words":expected.view(np.uint32).tolist()})
    return {
        "source_sha256":"214328c30a6f542aed0e6e77eeeea4152174738977e7b6460eff3a7001fb9630",
        "compiler":(build / "compiler.txt").read_text().strip(),
        "oracle_sha256":hashlib.sha256((build / "oracle").read_bytes()).hexdigest(),
        "columns":n,"water_columns":64,"steps":ns,"dt":300.,"stbolt":float(f(5.67051e-8)),
        "initial":{key:value.view(np.uint32).tolist() for key,value in initial.items()},
        "flux":[{key:value.view(np.uint32).tolist() for key,value in step.items()} for step in flux],
        "arms":arms,
    }

if __name__ == "__main__":
    Path(sys.argv[2]).write_text(json.dumps(generate(sys.argv[1]),separators=(",",":"))+"\n")
