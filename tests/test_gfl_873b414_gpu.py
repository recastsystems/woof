"""GPU module replay of WRF v4.8.0 GFL on 72 real columns captured from native WRF v4.8.0 strict.

The breakage it prevents: a transcription, support-header or launcher change
that moves any word of the scheme's output. The oracle
(tests/data/oracles/gfl_wrf480_strict239, PROVENANCE.txt) holds a captured
cu_grell_freitas_li call (NCAR CONUS 12 km benchmark input, step 3) for 72
columns (48 raining, 24 of them whose words move with the gamma function) and
two POST files from the referee's own objects: glibc 2.39 (the referee as
built) and the same objects linked with a correctly rounded tgammaf.

* Strict proof mode against a glibc 2.39 referee (GPUWM_WRF_EXACT=1, the
  default GPUWM_WRF_GLIBC): every word equals the referee's.
* Every other mode uses the engine's correctly rounded gamma: every word equals
  the correctly rounded referee's, and the glibc 2.39 words differ only in the
  columns where the two native POST files differ.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ORACLE = ROOT / "tests/data/oracles/gfl_wrf480_strict239"


@pytest.mark.gpu
def test_gfl_replays_the_native_wrf480_columns_word_for_word():
    cp = pytest.importorskip("cupy")
    sys.path.insert(0, str(ROOT / "tools/gfl_873b414"))
    from replay_gfl import read_pre, read_post, LEVEL_IN, SURF_IN, SURF_INOUT, INT_INOUT, LEVEL_INOUT
    from woof import wrf_exact
    from woof.core import gfl_873b414 as gfl

    d, a, shape = read_pre(ORACLE / "pre.bin")
    ni, nk, nj = shape
    want_cr = read_post(ORACLE / "post_native_crgamma.bin", shape)
    want_glibc = read_post(ORACLE / "post_native_glibc239.bin", shape)
    ncol = ni

    def lev(x):
        return cp.ascontiguousarray(cp.asarray(x[0].T))

    def srf(x):
        return cp.ascontiguousarray(cp.asarray(x[0]))

    f = {n: lev(a[n]) for n in LEVEL_IN + ("rqvften", "rthften", "rthblten", "rqvblten", "rthraten",
                                          "gdc", "gdc2") + LEVEL_INOUT}
    f.update({n: srf(a[n]) for n in SURF_IN + SURF_INOUT + ("kpbl",) + INT_INOUT})
    for n, typ in (("itimestep", cp.int32), ("ichoice_deep", cp.int32), ("ishallow_g3", cp.int32),
                   ("dt", cp.float32), ("dxcell", cp.float32), ("xlv", cp.float32), ("cp", cp.float32),
                   ("g", cp.float32), ("r_v", cp.float32)):
        f[n] = cp.asarray([d[n]], dtype=typ)
    gfl.launch_columns(f, ncol=ncol, nz=d["kte"],
                       vertical=(d["kms"], d["kme"], d["kds"], d["kde"], d["kts"], d["kte"]))
    got = {n: cp.asnumpy(f[n]).T[None] for n in ("gdc", "gdc2") + LEVEL_INOUT}
    got.update({n: cp.asnumpy(f[n])[None] for n in SURF_INOUT + INT_INOUT})
    columns = json.loads((ORACLE / "columns.json").read_text())
    assert columns["columns"] == ncol
    referee_gamma = wrf_exact.ENABLED and not wrf_exact.referee_correctly_rounded_trig()
    want = want_glibc if referee_gamma else want_cr
    different = {n: int(np.count_nonzero(got[n].view(np.uint32) != want[n].view(np.uint32))) for n in got}
    assert sum(different.values()) == 0, different
    if referee_gamma:
        return
    gamma_columns = np.zeros(ncol, bool)
    for n in got:
        x, y = want_cr[n].view(np.uint32), want_glibc[n].view(np.uint32)
        gamma_columns |= (x != y).reshape(-1, ncol).any(axis=0) if x.ndim == 3 else (x != y)[0]
    for n in got:
        moved = (got[n].view(np.uint32) != want_glibc[n].view(np.uint32))
        moved = moved.reshape(-1, ncol).any(axis=0) if moved.ndim == 3 else moved[0]
        assert not (moved & ~gamma_columns).any(), n
    assert gamma_columns.sum() == columns["gamma_sensitive"]
