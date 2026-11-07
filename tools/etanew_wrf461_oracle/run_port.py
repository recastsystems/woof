#!/usr/bin/env python3
"""Run WOOF's Eta Ferrier port on the oracle columns, writing oracle CSVs.

usage: run_port.py COLUMNS_INPUTS_CSV OUT_DIR [--plant LITERAL=REPLACEMENT]

Reads the canonical ``columns_inputs.csv`` (binary32 hex) that
``make_columns.py`` wrote, and for each distinct time step runs WOOF's
``etanew_init`` and then ``etanew_column`` for every column with that
step, NSTEPS times, carrying theta, qv, qt, the three fractions and RAINNC
exactly as ``run_etanew.F90`` does.  Writes ``levels.csv``, ``surface.csv``
and ``init.csv`` in the Fortran driver's format, so ``compare_etanew.py``
grades the two directly.

The arithmetic mode is the process's: ``GPUWM_WRF_EXACT=1`` for strict.

``--plant`` replaces one literal in the compiled kernel source (the
planted-bit control: a one-bit change to one kernel constant must make
the comparison fail).
"""

from __future__ import annotations

import hashlib
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

FIELDS_IN = ("p", "pi", "th", "qv", "qt", "dz8w", "rho",
             "f_ice", "f_rain", "f_rimef")


def _f(h: str) -> np.float32:
    return np.frombuffer(bytes.fromhex(h), dtype=">f4")[0].astype(np.float32)


def _hex(a: np.ndarray) -> list[str]:
    return [f"{int(x):08X}" for x in np.asarray(a, np.float32).view(np.uint32)]


def read_inputs(path: Path):
    cols = OrderedDict()
    levels = {}
    for line in open(path):
        if line.startswith("#"):
            continue
        parts = line.strip().split(",")
        if parts[0] == "C":
            ic = int(parts[1])
            cols[ic] = {"regime": parts[2], "nsteps": int(parts[3]),
                        "dt": _f(parts[4]), "dx": _f(parts[5]),
                        "rainnc": _f(parts[6])}
        elif parts[0] == "L":
            ic, k = int(parts[1]), int(parts[2])
            levels.setdefault(ic, {})[k] = [_f(h) for h in parts[3:]]
    out = OrderedDict()
    for ic, meta in cols.items():
        nz = len(levels[ic])
        arr = np.array([levels[ic][k] for k in range(1, nz + 1)],
                       dtype=np.float32)          # (nz, nfield)
        meta = dict(meta)
        for j, name in enumerate(FIELDS_IN):
            meta[name] = arr[:, j].copy()
        out[ic] = meta
    return out


def main(argv):
    plant = None
    if "--plant" in argv:
        i = argv.index("--plant")
        plant = argv[i + 1].split("=", 1)
        argv = argv[:i] + argv[i + 2:]
    if len(argv) != 3:
        sys.exit(__doc__)
    inputs, out = Path(argv[1]), Path(argv[2])
    out.mkdir(parents=True, exist_ok=True)

    import woof.core.etanew as eta
    import cupy as cp

    if plant is not None:
        original = eta.module_source

        def planted():
            src = original()
            if src.count(plant[0]) != 1:
                raise SystemExit(f"--plant literal {plant[0]!r} occurs "
                                 f"{src.count(plant[0])} times")
            return src.replace(plant[0], plant[1])
        eta.module_source = planted

    cols = read_inputs(inputs)
    by_dt = OrderedDict()
    for ic, meta in cols.items():
        by_dt.setdefault(float(meta["dt"]), []).append(ic)

    level_rows, surf_rows, init_rows, table_rows = [], [], [], []
    for dt, ids in by_dt.items():
        setup = eta.etanew_init(np.float32(dt))
        first = ids[0]
        rs = cp.asnumpy(setup.restart_state)
        for n, v in enumerate(_hex(rs), 1):
            init_rows.append(f"{first},mp_restart_state,{n},{v}")
        for name, arr in (("tbpvs", setup.tbpvs), ("tbpvs0", setup.tbpvs0)):
            values = _hex(cp.asnumpy(arr))
            digest = hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()
            table_rows.append((first, name, len(values), digest))
            for n, v in enumerate(values, 1):
                init_rows.append(f"{first},{name},{n},{v}")
        nz = cols[ids[0]]["p"].size
        ncol = len(ids)

        def stack(name):
            return cp.asarray(np.stack([cols[i][name] for i in ids], axis=1),
                              dtype=cp.float32)        # (nz, ncol)
        p, pii, th, qv, qt = (stack(n) for n in ("p", "pi", "th", "qv", "qt"))
        dz8w, rho = stack("dz8w"), stack("rho")
        f_ice, f_rain, f_rimef = (stack(n) for n in
                                  ("f_ice", "f_rain", "f_rimef"))
        qc = cp.zeros((nz, ncol), cp.float32)
        qr = cp.zeros_like(qc)
        qs = cp.zeros_like(qc)
        rainnc = cp.asarray(np.array([cols[i]["rainnc"] for i in ids],
                                     dtype=np.float32))
        rainncv = cp.zeros(ncol, cp.float32)
        sr = cp.zeros(ncol, cp.float32)
        nsteps = cols[ids[0]]["nsteps"]
        for step in range(1, nsteps + 1):
            eta.launch_column(setup, th, qv, qt, qc, qr, qs, f_ice, f_rain,
                              f_rimef, p, pii, rho, dz8w, rainnc, rainncv,
                              sr, np.float32(dt))
            host = {n: cp.asnumpy(a) for n, a in (
                ("th", th), ("qv", qv), ("qt", qt), ("qc", qc), ("qr", qr),
                ("qs", qs), ("f_ice", f_ice), ("f_rain", f_rain),
                ("f_rimef", f_rimef))}
            s_nc, s_ncv, s_sr = (cp.asnumpy(a) for a in (rainnc, rainncv, sr))
            for j, ic in enumerate(ids):
                for k in range(nz):
                    vals = [host[n][k, j] for n in
                            ("th", "qv", "qt", "qc", "qr", "qs", "f_ice",
                             "f_rain", "f_rimef")]
                    level_rows.append((ic, step, k + 1, _hex(np.array(vals))))
                surf_rows.append((ic, step, _hex(np.array(
                    [s_nc[j], s_ncv[j], s_sr[j]]))))

    with open(out / "levels.csv", "w", newline="\n") as f:
        f.write("col,step,k,th,qv,qt,qc,qr,qs,f_ice,f_rain,f_rimef\n")
        for ic, step, k, vals in sorted(level_rows, key=lambda r: r[:3]):
            f.write(f"{ic},{step},{k}," + ",".join(vals) + "\n")
    with open(out / "surface.csv", "w", newline="\n") as f:
        f.write("col,step,rainnc,rainncv,sr\n")
        for ic, step, vals in sorted(surf_rows, key=lambda r: r[:2]):
            f.write(f"{ic},{step}," + ",".join(vals) + "\n")
    with open(out / "init.csv", "w", newline="\n") as f:
        f.write("col,name,index,value\n")
        f.write("\n".join(init_rows) + "\n")
    with open(out / "tables.csv", "w", newline="\n") as f:
        f.write("col,name,words,sha256_of_hex_lines\n")
        for row in sorted(table_rows):
            f.write(",".join(map(str, row)) + "\n")
    print(f"wrote {len(level_rows)} level rows for {len(cols)} columns "
          f"to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
