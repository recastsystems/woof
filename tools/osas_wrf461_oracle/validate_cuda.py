"""Write full-word column receipts from the production OSAS launcher.

Run only inside the sprint GPU card queue:
  GPUWM_WRF_EXACT=1 python tools/osas_wrf461_oracle/validate_cuda.py OUT.json

Strict arithmetic with the corrected backout is also compared with WRF
given the identical counter draw. Fast arithmetic is a separate bounded check.
Only backed-out condensate columns may differ. Their absolute tolerance is
1e-4 kg kg-1 s-1; the correction is also checked against its exact restored
word in the parity tests. Every other field has a zero-word tolerance.
"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
from woof.core import column_cumulus_r8 as cc
from woof.core.kernels import module_source
from woof.certify.compile_platform import compile_platform_fingerprint
from test_cu_osas_wrf461_parity import (
    _fixture, _call, _run, _stage_view, _bits, _bad_columns, _backout_columns)

def statistics(got, want):
    g, w = np.asarray(got), np.asarray(want, dtype=got.dtype)
    words_g, words_w = _bits(g), _bits(w)
    sign = np.uint64(1 << (g.dtype.itemsize * 8 - 1))
    a, b = words_g.astype(np.uint64), words_w.astype(np.uint64)
    mask = np.uint64((1 << (g.dtype.itemsize * 8)) - 1)
    a = np.where(a & sign, (~a) & mask, a | sign)
    b = np.where(b & sign, (~b) & mask, b | sign)
    diff = g.astype(np.float64) - w.astype(np.float64)
    return dict(words=int(g.size), unequal_words=int(np.count_nonzero(words_g != words_w)),
                max_ulp=int((np.maximum(a,b)-np.minimum(a,b)).max(initial=0)),
                max_abs=float(np.abs(diff).max(initial=0)),
                rmse=float(np.sqrt(np.mean(diff * diff))),
                got_sha256=hashlib.sha256(g.tobytes()).hexdigest(),
                wrf_sha256=hashlib.sha256(w.tobytes()).hexdigest())

def main(path):
    from woof import wrf_exact
    if not wrf_exact.ENABLED:
        raise RuntimeError("word receipts require GPUWM_WRF_EXACT=1")
    report = dict(schema="osas-column-receipt-v1",
                  platform=compile_platform_fingerprint(),
                  arithmetic="strict",
                  source_sha256=hashlib.sha256(module_source("cu_osas").encode()).hexdigest(),
                  table_sha256=cc.GFS_FUNCPHYS_SHA256,
                  strict=[], corrected_backout=[], strict_unequal_words=0)
    for grid in ("k44", "k29"):
        fx = _fixture(grid)
        for n in range(int(fx["ncall"])):
            dt, stepcu, itimestep, inputs, want, stage = _call(fx, n)
            got = _run(inputs, dt, stepcu, flags=None)
            sg = _stage_view(got["stages"], int(fx["kx"]))
            fields = {name: statistics(got[name], w) for name, w in want.items()}
            stages = {name: statistics(sg[name], w) for name, w in stage.items()}
            # KUO is the old module's convection flag (KCNV in newer SAS).
            errors = sum(v["unequal_words"] for v in (*fields.values(), *stages.values()))
            report["strict_unequal_words"] += errors
            report["strict"].append(dict(grid=grid, call=n, dt=dt, stepcu=stepcu,
                itimestep=itimestep, ncol=inputs["xland"].size, fields=fields,
                stages=stages, deep_columns=int(np.count_nonzero(stage["kuo"])),
                shallow_columns=int(np.count_nonzero(
                    (stage["kuo"] == 0) & np.any(
                        _bits(stage["t1_shallow"]) != _bits(stage["t1"]), axis=0)))))
            assert errors == 0, (grid, n, fields, stages)
        fx = _fixture(grid + "-default")
        for n in range(int(fx["ncall"])):
            dt, stepcu, itimestep, inputs, want, stage = _call(fx, n)
            got = _run(inputs, dt, stepcu, flags=cc.COLUMN_CUMULUS_SCHEMES[95].flags,
                       stages=False)
            backed = set(_backout_columns(inputs, stage).tolist())
            bad = _bad_columns(got, want)
            assert all(name in ("rqccuten", "rqicuten") and set(cols) <= backed
                       for name, cols in bad.items()), bad
            fields = {name: statistics(got[name], w) for name, w in want.items()}
            for name, stats in fields.items():
                tolerance = 1e-4 if name in ("rqccuten", "rqicuten") else 0.0
                stats["absolute_tolerance"] = tolerance
                assert stats["max_abs"] <= tolerance, (grid, n, name, stats)
            report["corrected_backout"].append(dict(grid=grid, call=n, dt=dt, stepcu=stepcu,
                itimestep=itimestep, fields=fields, corrected_columns=sorted(backed)))
    Path(path).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(strict_unequal_words=report["strict_unequal_words"],
                          strict_calls=len(report["strict"]), corrected_backout_calls=len(report["corrected_backout"]))))
if __name__ == "__main__":
    main(sys.argv[1])
