"""Driver-seam checks on WRF's own captured words.

1. CHKLOWQ: woof's PhysicsDriver._noah_myj_chklowq on WRF's operands (QV at
   the lowest level, MYJSFC's QGH and QZ0 of the step) against the CHKLOWQ
   WRF's Noah handed MYJPBL in the same step.
2. Z0BASE: WRF's MYJSFC Z0 argument per step against the step-1 value (the
   cold-start ZNT copy woof used to bind).
"""
import json
import sys
from types import SimpleNamespace

import numpy as np
import cupy as cp

from myj_replay import assemble, words
from woof.core.physics import PhysicsDriver

capdir, steps = sys.argv[1], [int(s) for s in sys.argv[2].split(",")]
out = {"chklowq": {}, "z0base": {}}
z0_first = None
for step in steps:
    sin, _ = assemble(capdir, "myjsfc", "in", step)
    sout, _ = assemble(capdir, "myjsfc", "out", step)
    pin, _ = assemble(capdir, "myjpbl", "in", step)
    fake = SimpleNamespace(fields={"qgh": cp.asarray(sout["qgh"]), "qz0": cp.asarray(sout["qz0"]),
                                   "chklowq": cp.full(sout["qgh"].shape, 7.0, np.float32)})
    PhysicsDriver._noah_myj_chklowq(fake, {"qv": cp.asarray(sin["qv_curr"][:1])})
    got = cp.asnumpy(fake.fields["chklowq"])
    d, _ = words(got, pin["chklowq"])
    zeros = int((pin["chklowq"] == 0).sum())
    old = int((pin["chklowq"] != 1).sum())   # what the always-1 Noah kernel got wrong
    out["chklowq"][step] = {"words_different": d, "wrf_zero_columns": zeros,
                            "always_one_words_different": old, "n": int(got.size)}
    if z0_first is None:
        z0_first = sin["z0"].copy()
    land = sin["xland"] < 1.5
    dz0 = sin["z0"].view(np.int32) != z0_first.view(np.int32)
    out["z0base"][step] = {"land_columns": int(land.sum()),
                           "z0_differs_from_step1": int(dz0.sum()),
                           "on_land": int((dz0 & land).sum())}
    print(step, out["chklowq"][step], out["z0base"][step])
json.dump(out, open(sys.argv[3], "w"), indent=1)
