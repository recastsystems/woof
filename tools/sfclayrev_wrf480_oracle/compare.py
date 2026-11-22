"""Grade kernels/sfclay.cu (option 1) against the WRF v4.8.0 referee oracle.

    python compare.py INPUTS_HEX ORACLE_OUTPUTS_HEX [--tofd-factor F] [--steps N] [--json OUT]

Runs the production entry points (woof.core.sfclay.sfclay / launch_sfclay)
on every column, arm and chained step the oracle ran: arms 1-6 without varf
(the unchanged sfclay unit, i.e. kim_tofd = .false.), arms 7-12 with varf
(the sfclay_tofd unit).  Counts differing float32 words per field; the one
WRF-undefined word (LH with isfflx = 0, arms 6 and 12) is not compared.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

INPUT_FIELDS = ("u", "v", "t", "qv", "p", "dz8w", "psfc", "tsk", "pblh",
                "mavail", "xland", "lakemask", "dx", "znt", "ust", "ustm",
                "mol", "hfx", "qfx", "qsfc", "zol", "varf")
SWITCHES = ((1, 0, 0), (1, 1, 0), (1, 2, 0), (1, 0, 1), (1, 0, 2), (0, 0, 0))
ARMS = tuple((s, False) for s in SWITCHES) + tuple((s, True) for s in SWITCHES)


def _words(tokens):
    return np.array([int(t, 16) for t in tokens], dtype=np.uint32).view(np.float32)


def load(inputs_path: Path, outputs_path: Path, steps: int):
    from woof.core.physics_inventory import SFCLAY_OUTPUTS
    rows = [l.split() for l in inputs_path.read_text().splitlines() if l and not l.startswith("#")]
    cases = np.array([int(r[0]) for r in rows])
    data = np.stack([_words(r[1:]) for r in rows])
    assert data.shape[1] == len(INPUT_FIELDS), data.shape
    inputs = {n: np.ascontiguousarray(data[:, j]) for j, n in enumerate(INPUT_FIELDS)}
    index = {int(c): i for i, c in enumerate(cases)}
    out = np.full((len(ARMS), steps, len(cases), len(SFCLAY_OUTPUTS)), np.nan, np.float32)
    seen = 0
    for line in outputs_path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        tok = line.split()
        arm = int(tok[0]); sw = tuple(map(int, tok[1:4])); tofd = bool(int(tok[4]))
        step = int(tok[5]); case = int(tok[6])
        assert (sw, tofd) == ARMS[arm - 1], (arm, sw, tofd)
        out[arm - 1, step - 1, index[case]] = _words(tok[7:])
        seen += 1
    assert seen == out.shape[0] * out.shape[1] * out.shape[2], seen
    return cases, inputs, out


def port(inputs, shape, steps, tofd_factor):
    import cupy as cp
    from woof.core.physics_inventory import SFCLAY_OUTPUTS
    from woof.core.sfclay import launch_sfclay, sfclay
    out = np.full(shape, np.nan, np.float32)
    for a, ((isfflx, isftcflx, iz0tlnd), tofd) in enumerate(ARMS):
        for dx in np.unique(inputs["dx"]):
            sel = np.flatnonzero(inputs["dx"] == dx)
            g = {k: cp.asarray(v[sel][None, :]) for k, v in inputs.items()}
            atm = (g["u"], g["v"], g["t"], g["qv"], g["p"], g["dz8w"], g["psfc"], g["tsk"])
            extra = {"varf": g["varf"], "tofd_factor": tofd_factor} if tofd else {}
            result = sfclay(*atm, g["znt"], g["pblh"], g["mavail"], g["xland"], option=1,
                            qsfc=g["qsfc"], zol=g["zol"], ust=g["ust"], ustm=g["ustm"],
                            mol=g["mol"], hfx=g["hfx"], qfx=g["qfx"], lakemask=g["lakemask"],
                            dx=float(dx), isfflx=bool(isfflx), isftcflx=isftcflx,
                            iz0tlnd=iz0tlnd, **extra)
            for step in range(steps):
                if step:
                    launch_sfclay(*atm, g["pblh"], g["mavail"], g["xland"], g["lakemask"],
                                  result, option=1, dx=float(dx), isfflx=bool(isfflx),
                                  isftcflx=isftcflx, iz0tlnd=iz0tlnd, **extra)
                cp.cuda.Device().synchronize()
                for f, name in enumerate(SFCLAY_OUTPUTS):
                    out[a, step, sel, f] = cp.asnumpy(getattr(result, name))[0]
    return out


def grade(cases, want, got):
    from woof.core.physics_inventory import SFCLAY_OUTPUTS
    w = want.view(np.uint32); p = got.view(np.uint32)
    differ = w != p
    lh = SFCLAY_OUTPUTS.index("lh")
    excluded = 0
    for a, ((isfflx, _, _), _) in enumerate(ARMS):
        if not isfflx:
            excluded += differ[a, :, :, lh].size
            differ[a, :, :, lh] = False
    compared = differ.size - excluded
    fields = {name: int(differ[..., f].sum()) for f, name in enumerate(SFCLAY_OUTPUTS)}
    arms = {f"arm{a + 1}{'-tofd' if ARMS[a][1] else ''}": int(differ[a].sum()) for a in range(len(ARMS))}
    first = [(int(a) + 1, int(s) + 1, int(cases[c]), SFCLAY_OUTPUTS[f],
              f"{int(w[a, s, c, f]):08X}", f"{int(p[a, s, c, f]):08X}")
             for a, s, c, f in list(zip(*np.nonzero(differ)))[:20]]
    tofd = sum(int(differ[a].sum()) for a in range(6, 12))
    cd = SFCLAY_OUTPUTS.index("cd")
    drag_cols = int((want[6:, :, :, cd] != want[:6, :, :, cd]).any(axis=(0, 1)).sum())
    return {"compared_words": int(compared), "different_words": int(differ.sum()),
            "tofd_arm_different_words": tofd, "columns": int(len(cases)),
            "columns_where_tofd_changes_cd": drag_cols,
            "excluded_wrf_undefined_lh_words": int(excluded),
            "per_field": fields, "per_arm": arms, "first_differences": first}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", type=Path)
    ap.add_argument("oracle", type=Path)
    ap.add_argument("--tofd-factor", type=float, default=0.003)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    from woof import wrf_exact
    cases, inputs, want = load(a.inputs, a.oracle, a.steps)
    got = port(inputs, want.shape, a.steps, a.tofd_factor)
    report = grade(cases, want, got)
    report["strict_numerics"] = bool(wrf_exact.ENABLED)
    text = json.dumps(report, indent=1)
    if a.json:
        a.json.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
