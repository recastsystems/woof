"""Drive WOOF's ISHMAEL kernel over the oracle's columns; compare to WRF.

    run_gpu.py INPUTS.bin OUT.bin NSTEPS DT [--want WRF.bin] [--mutate ID]

Runs woof.core.ishmael.run_column_step (the production launcher, the
production compile site) over exactly the bytes make_columns.py gave the
Fortran driver, writes the step file in run_ishmael.F90's layout, and with
``--want`` grades it word for word against the oracle's file.

``--mutate`` plants a deliberate one-bit change in the assembled source
(MUTATIONS below) so a gate can prove it would catch one.  Run it on a card
through the queue; it needs CuPy and a device.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import compare as cmp_mod          # noqa: E402
import make_columns as mc          # noqa: E402

#: id -> (exact text in ishmael.cu, replacement): each flips the last bit
#: of one constant in a habit or deposition rate.
MUTATIONS = {
    # The vapour-growth deposition density cap (vaporgrow :3220).
    "deposition-density-cap": ("rhodep = ish_min(rhodep, 700.f);",
                               "rhodep = ish_min(rhodep, __int_as_float(0x442f0001));"),
    # One inherent-growth-ratio datum (-6 C, habit selection).
    "igr-minus-6C": ("2.32423f, 2.08818f", "2.324229717254638671875f, 2.08818f"),
}


def _host_step(fields, diag, surf):
    out = {}
    for n in cmp_mod.OUTPUT_FIELDS:
        if n in ("th", "qv", "qc", "qr", "nr") or n[:2] in ("qi", "ni", "ai", "ci"):
            out[n] = fields[n].get()
        elif n in ("effc", "effi", "dbz"):
            out[n] = diag[n].get()
        elif n.startswith("itype"):
            out[n] = diag[n].get()
        else:
            out[n] = diag[n].get()
    for n in cmp_mod.SURFACE_FIELDS:
        out[n] = surf[n].get()
    return out


def run(inputs: Path, nsteps: int, dt: float, *, source: str | None = None):
    import cupy as cp
    from woof.core import ishmael

    arrays, rainnc, snownc = mc.read(inputs)
    nz, ncol = arrays["p"].shape
    fields = {n: cp.asarray(np.ascontiguousarray(arrays[n], np.float32))
              for n in ishmael.FIELD_SLOTS}
    diag = {n: cp.zeros((nz, ncol), cp.float32) for n in ishmael.DIAG_SLOTS}
    scratch = {n: cp.zeros((nz, ncol), cp.float32) for n in ishmael.SCRATCH_SLOTS}
    surf = {"rainnc": cp.asarray(rainnc), "rainncv": cp.zeros(ncol, cp.float32),
            "snownc": cp.asarray(snownc), "snowncv": cp.zeros(ncol, cp.float32)}
    steps = []
    for _ in range(nsteps):
        ishmael.run_column_step(fields, diag, surf, scratch, dt=dt, source=source)
        steps.append(_host_step(fields, diag, surf))
    return steps


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("inputs", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("nsteps", type=int)
    ap.add_argument("dt", type=float)
    ap.add_argument("--want", type=Path)
    ap.add_argument("--mutate", choices=sorted(MUTATIONS))
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    from woof.wrf_exact import ENABLED
    if args.want and not ENABLED:
        raise SystemExit("native replay requires GPUWM_WRF_EXACT=1 before import")
    source = None
    if args.mutate:
        from woof.core.kernels import module_source
        source = module_source("ishmael")
        old, new = MUTATIONS[args.mutate]
        if source.count(old) != 1:
            raise SystemExit(f"mutation {args.mutate}: anchor not unique")
        source = source.replace(old, new)
    steps = run(args.inputs, args.nsteps, args.dt, source=source)
    cmp_mod.write_steps(args.out, steps, dt=args.dt)
    if args.want:
        report = cmp_mod.compare(cmp_mod.read_steps(args.out),
                                 cmp_mod.read_steps(args.want))
        from woof.core.kernels import module_source, module_options
        report.update(
            strict_numerics=ENABLED,
            input_sha256=hashlib.sha256(args.inputs.read_bytes()).hexdigest(),
            expected_sha256=hashlib.sha256(args.want.read_bytes()).hexdigest(),
            output_sha256=hashlib.sha256(args.out.read_bytes()).hexdigest(),
            kernel_source_sha256=hashlib.sha256(
                (source or module_source("ishmael")).encode()).hexdigest(),
            kernel_options=module_options("ishmael"),
        )
        for row in report["differences"][:60]:
            print(f"step {row['step']:3d} {row['field']:8s} words {row['words']:6d} "
                  f"(+-0 only {row['signed_zero_only']}) max_ulp {row['max_ulp']} "
                  f"cols {row['columns']}")
        print(f"total differing words: {report['total_words']} over "
              f"{report['nsteps']} steps")
        if args.json:
            args.json.write_text(json.dumps(report, indent=1) + "\n")
        return 0 if report["passed"] else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
