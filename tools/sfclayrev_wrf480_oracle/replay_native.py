"""Replay native WRF v4.8.0 SFCLAYREV captures (kim_tofd) through kernels/sfclay.cu.

    python replay_native.py CASE_DIR [--json OUT] [--export-hex HEX --every N --step S]

CASE_DIR/captures holds sfclayrev-<itimestep>-<its>-<jts>-{in,out}.bin pairs
written by instrument_sfclayrev480.py around the real CALL SFCLAYREV of a
native 4.8.0 strict run (header: nx ny itimestep isfflx isftcflx iz0tlnd
kim_tofd as int32, tofd_factor as float32; then 22 input and 34 output
planes, Fortran (i,j) order).  Each "in" capture is run through the
production entry point woof.core.sfclay.launch_sfclay with WRF's own dx2d
plane and var2d, and its 34 outputs are compared word for word with "out".

--export-hex writes every N-th column of capture step S as the 22-word input
format of run_sfclayrev480.F90, for the twelve-arm referee-object oracle.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

IN = ("u", "v", "t", "qv", "p", "dz8w", "psfc", "tsk", "pblh", "mavail", "xland",
      "lakemask", "dx", "znt", "ust", "ustm", "mol", "hfx", "qfx", "qsfc", "zol", "varf")


def read(path: Path, nplanes: int):
    raw = path.read_bytes()
    head = np.frombuffer(raw, "<i4", 7, 0).copy()
    factor = np.frombuffer(raw, "<f4", 1, 28)[0]
    nx, ny = int(head[0]), int(head[1])
    planes = np.frombuffer(raw, "<f4", offset=32).reshape(nplanes, ny, nx)
    assert planes.size * 4 + 32 == len(raw), path
    return head, factor, planes


def replay(case: Path):
    import cupy as cp
    from woof.core.physics_inventory import SFCLAY_OUTPUTS
    from woof.core.sfclay import SFClayResult, launch_sfclay
    rows, total, differ_total = [], 0, 0
    per_field = {n: 0 for n in SFCLAY_OUTPUTS}
    for path in sorted((case / "captures").glob("sfclayrev-*-in.bin")):
        head, factor, planes = read(path, len(IN))
        ohead, ofactor, want = read(path.with_name(path.name.replace("-in.bin", "-out.bin")),
                                    len(SFCLAY_OUTPUTS))
        assert np.array_equal(head, ohead) and factor == ofactor
        nx, ny, step, isfflx, isftcflx, iz0tlnd, tofd = map(int, head)
        f = {n: cp.ascontiguousarray(cp.asarray(planes[i])) for i, n in enumerate(IN)}
        res = {n: cp.zeros((ny, nx), np.float32) for n in SFCLAY_OUTPUTS}
        for n in ("znt", "ust", "ustm", "mol", "hfx", "qfx", "qsfc", "zol"):
            res[n][...] = f[n]
        result = SFClayResult(**res)
        launch_sfclay(f["u"], f["v"], f["t"], f["qv"], f["p"], f["dz8w"], f["psfc"],
                      f["tsk"], f["pblh"], f["mavail"], f["xland"], f["lakemask"], result,
                      option=1, dx=float(planes[IN.index("dx")].flat[0]), isfflx=bool(isfflx),
                      isftcflx=isftcflx, iz0tlnd=iz0tlnd, dx_field=f["dx"],
                      **({"varf": f["varf"], "tofd_factor": float(factor)} if tofd else {}))
        cp.cuda.Device().synchronize()
        got = np.stack([cp.asnumpy(getattr(result, n)) for n in SFCLAY_OUTPUTS])
        d = want.view(np.uint32) != got.view(np.uint32)
        if not isfflx:
            d[SFCLAY_OUTPUTS.index("lh")] = False
        for i, n in enumerate(SFCLAY_OUTPUTS):
            per_field[n] += int(d[i].sum())
        cd_i = SFCLAY_OUTPUTS.index("cd")
        drag = int((planes[IN.index("varf")] > 5.0).sum())
        rows.append({"capture": path.name, "itimestep": step, "kim_tofd": bool(tofd),
                     "tofd_factor": float(factor), "columns": nx * ny,
                     "columns_varf_gt_5": drag, "compared_words": int(d.size),
                     "different_words": int(d.sum()),
                     "max_cd": float(want[cd_i].max())})
        total += int(d.size); differ_total += int(d.sum())
    return {"captures": rows, "compared_words": total, "different_words": differ_total,
            "per_field_different": per_field}


def export_hex(case: Path, out: Path, every: int, step: int):
    rows = []
    for path in sorted((case / "captures").glob(f"sfclayrev-{step}-*-in.bin")):
        head, _, planes = read(path, len(IN))
        flat = planes.reshape(len(IN), -1)
        for c in range(0, flat.shape[1], every):
            words = " ".join(f"{int(w):08X}" for w in flat[:, c].view(np.uint32))
            rows.append(f"{c} {words}")
    out.write_text("# case(column index) " + " ".join(IN) + " (float32 hex)\n" + "\n".join(rows) + "\n")
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case", type=Path)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--export-hex", type=Path)
    ap.add_argument("--every", type=int, default=7)
    ap.add_argument("--step", type=int, default=30)
    a = ap.parse_args()
    if a.export_hex:
        print("exported", export_hex(a.case, a.export_hex, a.every, a.step), "columns")
        return
    from woof import wrf_exact
    report = replay(a.case)
    report["strict_numerics"] = bool(wrf_exact.ENABLED)
    text = json.dumps(report, indent=1)
    if a.json:
        a.json.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
