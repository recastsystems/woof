"""Measurement launcher (not product code): the hex CLI with the RRFS convective driver's inputs captured.

CONV_CAPTURE_DIR receives compute-NNN/ and update-NNN/ folders (1-based call counts): every array argument
(numpy copy of the device words), the outputs before the call, the call's scalars and the driver geometry,
so replay_current.py can rerun the same call on resident module versions.
The layout=pool metadata distinguishes these from the historical cell-major capture.
"""
import json
import os
import sys

import numpy as np

OUT = os.environ["CONV_CAPTURE_DIR"]
COMPUTE = {int(x) for x in os.environ.get("CONV_CAPTURE_COMPUTE", "1,2").split(",") if x}
UPDATE = {int(x) for x in os.environ.get("CONV_CAPTURE_UPDATE", "1,3,5,6").split(",") if x}


def save(kind, n, driver, arrays, scalars):
    import cupy as cp
    folder = os.path.join(OUT, f"{kind}-{n:03d}")
    os.makedirs(folder, exist_ok=False)
    np.save(os.path.join(folder, "outputs_before.npy"), cp.asnumpy(cp.stack(driver._rows)))
    names = []
    for name, a in arrays.items():
        np.save(os.path.join(folder, f"in_{name}.npy"), cp.asnumpy(a))
        names.append(name)
    meta = dict(scalars, arrays=names, n_cells=driver.n_cells, n_solve=driver.n_solve, n_levels=driver.n_levels,
                update_mask=int(driver.update_mask), layout="pool")
    with open(os.path.join(folder, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"CONV capture {folder}: {len(names)} arrays", flush=True)


def install():
    from woof.hex.mpasdiag import convective_rrfs as m
    D = m.ConvectiveDiagnostics
    oc, ou = D.compute, D.update
    count = {"compute": 0, "update": 0}

    def compute(self, arrays, fields_will_be_written, **kw):
        count["compute"] += 1
        names = list(fields_will_be_written)
        if count["compute"] in COMPUTE:
            save("compute", count["compute"], self, arrays, dict(fields=names, **{k: float(v) for k, v in kw.items()}))
        r = oc(self, arrays, names, **kw)
        if count["compute"] in COMPUTE:
            import cupy as cp
            np.save(os.path.join(OUT, f"compute-{count['compute']:03d}", "outputs_after.npy"), cp.asnumpy(cp.stack(self._rows)))
        return r

    def update(self, arrays, *, nsteps, convection, n_mesh_cells=None):
        count["update"] += 1
        n = count["update"]
        if n in UPDATE:
            save("update", n, self, arrays, dict(nsteps=int(nsteps), convection=bool(convection),
                                                 n_mesh_cells=None if n_mesh_cells is None else int(n_mesh_cells)))
        r = ou(self, arrays, nsteps=nsteps, convection=convection, n_mesh_cells=n_mesh_cells)
        if n in UPDATE:
            import cupy as cp
            np.save(os.path.join(OUT, f"update-{n:03d}", "outputs_after.npy"), cp.asnumpy(cp.stack(self._rows)))
        return r

    D.compute, D.update = compute, update


if __name__ == "__main__":
    install()
    from woof.hex import cli
    a = cli.build_parser().parse_args(sys.argv[1:])
    cli._select_numerics(a)
    sys.exit(int(a.handler(a)))
