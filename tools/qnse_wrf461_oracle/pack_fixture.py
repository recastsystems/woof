"""Read the QNSE oracle's stream records and pack them for the tests.

    python pack_fixture.py RUN_DIR OUT_DIR

reads ``RUN_DIR/<arm>-static.bin`` and ``RUN_DIR/<arm>-s<n>.bin`` (the
records run_qnse.F90 writes: name, rank, dims, kind, data) and writes one
``qnse-<arm>.npz`` per arm with every array in C order: a Fortran
``(ncol, nz)`` array comes back ``(nz, ncol)``, the layout woof's kernels
use.  Step records are keyed ``s<n>/<name>``.

    python pack_fixture.py --diff RUN_A RUN_B

prints, per arm, step and field, how many words differ and the largest
float32 ULP distance between two runs of the oracle.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def read_records(path: Path) -> dict[str, np.ndarray]:
    raw = path.read_bytes()
    pos = 0
    out: dict[str, np.ndarray] = {}
    if path.name.endswith("-static.bin"):
        hdr = np.frombuffer(raw, dtype=np.int32, count=5, offset=0)
        out["meta_ncol"], out["meta_nz"], out["meta_nstep"] = hdr[0], hdr[1], hdr[2]
        out["meta_mfshconv"], out["meta_stepbl"] = hdr[3], hdr[4]
        out["meta_dt"] = np.frombuffer(raw, dtype=np.float32, count=1, offset=20)[0]
        pos = 24
    while pos < len(raw):
        name = raw[pos:pos + 16].decode("ascii").strip()
        pos += 16
        rank = int(np.frombuffer(raw, np.int32, 1, pos)[0])
        pos += 4
        dims = tuple(int(d) for d in np.frombuffer(raw, np.int32, rank, pos))
        pos += 4 * rank
        kind = int(np.frombuffer(raw, np.int32, 1, pos)[0])
        pos += 4
        count = int(np.prod(dims))
        dtype = np.float32 if kind == 0 else np.int32
        data = np.frombuffer(raw, dtype, count, pos).copy()
        pos += 4 * count
        # Fortran order (ncol, nz) -> C order (nz, ncol).
        out[name] = data.reshape(dims[::-1])
    return out


def load_arm(run: Path, arm: str) -> dict[str, np.ndarray]:
    fx = {k: np.asarray(v) for k, v in read_records(run / f"{arm}-static.bin").items()}
    step = 1
    while (run / f"{arm}-s{step}.bin").exists():
        for k, v in read_records(run / f"{arm}-s{step}.bin").items():
            fx[f"s{step}/{k}"] = v
        step += 1
    return fx


def ulp(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ia = a.astype(np.float32).view(np.int32).astype(np.int64)
    ib = b.astype(np.float32).view(np.int32).astype(np.int64)
    ia = np.where(ia < 0, -(ia & 0x7FFFFFFF), ia)
    ib = np.where(ib < 0, -(ib & 0x7FFFFFFF), ib)
    return np.abs(ia - ib)


def main(argv: list[str]) -> int:
    if argv[1] == "--diff":
        ra, rb = Path(argv[2]), Path(argv[3])
        for arm in ("mf0", "mf1"):
            a, b = load_arm(ra, arm), load_arm(rb, arm)
            for k in sorted(a):
                if k.startswith("meta"):
                    continue
                x, y = a[k], b[k]
                if x.dtype == np.int32:
                    n = int((x != y).sum())
                    if n:
                        print(f"{arm} {k}: {n} ints differ")
                    continue
                same = (x.view(np.int32) == y.view(np.int32))
                if not same.all():
                    d = ulp(x, y)
                    cols = np.unique(np.nonzero(~same)[-1])
                    print(f"{arm} {k}: {int((~same).sum())} words, max ulp "
                          f"{int(d.max())}, columns {cols.tolist()[:12]}")
        return 0
    run, out = Path(argv[1]), Path(argv[2])
    out.mkdir(parents=True, exist_ok=True)
    for arm in ("mf0", "mf1"):
        fx = load_arm(run, arm)
        np.savez_compressed(out / f"qnse-{arm}.npz", **fx)
        print(f"{arm}: {len(fx)} arrays")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
