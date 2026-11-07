#!/usr/bin/env python3
"""Run an NTU oracle batch through WOOF's own mp_physics=56 launchers.

Reads a make_columns.py batch, lays it out as the engine does
((nz, ny, nx) with ny = 1, nx = ncol), runs woof.core.ntu.launch_init and
woof.core.ntu.launch, and writes the oracle's output layout.

usage: run_gpu.py BATCH.bin OUT.bin
"""
import sys

import numpy as np

MASS = ("TH P DZ PII ALT QV QC QR QI QS QG QH NC NR NI NS NG NH "
        "QDCN QTCN QCCN QRCN QNIN FI FS VI VS VG AI AS AG AH I3M").split()
ENGINE = dict(TH="theta", P="p", DZ="dz", PII="pii", QV="qv", QC="qc",
              QR="qr", QI="qi", QS="qs", QG="qg", QH="qh", NC="nc", NR="nr",
              NI="ni", NS="ns", NG="ng", NH="nh", QDCN="qdcn", QTCN="qtcn",
              QCCN="qccn", QRCN="qrcn", QNIN="qnin", FI="fi", FS="fs",
              VI="vi", VS="vs", VG="vg", AI="ai", AS="as_", AG="ag", AH="ah",
              I3M="i3m")
OUT = ["TH"] + MASS[5:]


def main(pin, pout):
    import cupy as cp
    from woof.core import ntu
    if "--mutate-one-bit" in sys.argv[3:]:
        original = ntu.module_source
        def mutated(*args, **kwargs):
            source = original(*args, **kwargs)
            old = "TH[at] = TK1D[k] / PII[at];"
            if source.count(old) != 1:
                raise RuntimeError("NTU one-bit mutation site did not match exactly once")
            return source.replace(old, old + " TH[at] = ntu_bits_f(ntu_f2u(TH[at]) ^ 1u);")
        ntu.module_source = mutated
    raw = np.fromfile(pin, dtype="<i4", count=8)
    _, ncol, nz, nsteps, ccnty, restart = (int(x) for x in raw[:6])
    with open(pin, "rb") as f:
        f.seek(32)
        dt = float(np.fromfile(f, dtype="<f4", count=1)[0])
        host = {n: np.fromfile(f, dtype="<f4", count=nz * ncol).reshape(nz, 1, ncol)
                for n in MASS}
        for n in ("W", "PHB", "PH"):
            host[n] = np.fromfile(f, dtype="<f4", count=(nz + 1) * ncol).reshape(nz + 1, 1, ncol)
        for n in ("RAINNC", "SNOWNC", "GRAPNC", "HAILNC", "XLAND"):
            host[n] = np.fromfile(f, dtype="<f4", count=ncol).reshape(1, ncol)
    dev = {n: cp.asarray(np.ascontiguousarray(v)) for n, v in host.items()}
    zeros3 = cp.zeros((nz, 1, ncol), dtype=cp.float32)
    ntu.launch_init(dev["PHB"], dev["PH"], dev["P"], zeros3, dev["ALT"],
                    dev["QV"], [dev[n] for n in ("QDCN", "QTCN", "QCCN", "QRCN", "QNIN")],
                    ccnty=ccnty, restart=bool(restart))
    fields = {ENGINE[n]: dev[n] for n in ENGINE}
    fields["w"] = dev["W"]
    z2 = lambda: cp.zeros((1, ncol), dtype=cp.float32)
    surface = dict(rainnc=dev["RAINNC"], rainncv=z2(), snownc=dev["SNOWNC"],
                   snowncv=z2(), graupelnc=dev["GRAPNC"], graupelncv=z2(),
                   hailnc=dev["HAILNC"], hailncv=z2(), sr=z2())
    with open(pout, "wb") as fo:
        for n in ("QDCN", "QTCN", "QCCN", "QRCN", "QNIN"):
            cp.asnumpy(dev[n]).astype("<f4").tofile(fo)
        for _ in range(nsteps):
            ntu.launch(fields, surface, dt, ccnty=ccnty)
            for n in OUT:
                cp.asnumpy(dev[n]).astype("<f4").tofile(fo)
            for n in ("rainnc", "rainncv", "snownc", "snowncv", "graupelnc",
                      "graupelncv", "hailnc", "hailncv", "sr"):
                cp.asnumpy(surface[n]).astype("<f4").tofile(fo)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
