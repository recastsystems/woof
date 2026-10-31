"""The production aerosol passes against the serial mpas_sw_aerosol kernel, word for word.

MPASShortwave.aerosol_optics runs mpas_sw_aerosol (module_ra_rrtmg_sw_aerosols.F:70-451) as a column pass, a
(column, level) pass and a finish pass. Each output word must be the serial kernel's: every option combination
the source defines (aod 1-2, exponent/ssa/asy 1-3, 2-D or 3-D optical depth, every diagnostic mask), both
nl = nz + 1 and nl = nz, under the process's numerics mode (run once with WOOF_HEX_NUMERICS=fast and once
with =strict). THE BREAKAGE THIS PREVENTS: a level-parallel aerosol pass that is faster but no longer the
audited serial arithmetic would silently move every aerosol-aware SW word (hrrrv5, mp_thompson_aerosols).

Needs CUDA and the SW tables (the kernels share the table-bound compilation unit): MPAS_RRTMG_SW_PACKET names a
physics-resources packet (woof hex physics-resources) holding an "rrtmg_sw" entry.
"""
import itertools
import json
import os
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    packet = os.environ.get("MPAS_RRTMG_SW_PACKET")
    if not packet:
        pytest.skip("MPAS_RRTMG_SW_PACKET must name a physics-resources packet with an rrtmg_sw entry")
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip("no CUDA device")
    from woof.hex.mpasphys.rrtmg_sw_v842 import load_shortwave_resources
    entry = json.loads(Path(packet).read_text())
    entry = dict(entry.get("rrtmg_sw", entry))
    entry["cache_root"] = str(tmp_path_factory.mktemp("rrtmg-sw"))
    return load_shortwave_resources(entry, 0)


def _inputs(cp, nc, nz, seed):
    rng = np.random.default_rng(seed)
    top = rng.uniform(1.0e3, 5.0e3, nc)
    surface = rng.uniform(5.0e4, 1.04e5, nc)
    frac = np.linspace(0.0, 1.0, nz)
    p = (surface[:, None] - (surface - top)[:, None] * frac[None, :] ** 0.8).astype(np.float32)
    t = (rng.uniform(270.0, 305.0, nc)[:, None] - 70.0 * frac[None, :]).astype(np.float32)
    qv = (rng.uniform(0.0, 0.02, (nc, nz)) * (1.0 - frac[None, :]) ** 2).astype(np.float32)
    qv[rng.random((nc, nz)) < 0.02] = -1.0e-6          # the source clips negative vapour (rv=max(0,qv))
    dz = rng.uniform(20.0, 900.0, (nc, nz)).astype(np.float32)
    ht = rng.uniform(-50.0, 4000.0, nc).astype(np.float32)
    kind = rng.integers(1, 4, nc).astype(np.int32)
    aod = rng.uniform(0.0, 2.0, nc).astype(np.float32)
    aod[: max(1, nc // 50)] = 0.0
    aod3 = rng.uniform(1.0e-6, 0.05, (nc, nz)).astype(np.float32)
    ang = rng.uniform(-0.5, 3.0, nc).astype(np.float32)
    ang[: max(1, nc // 40)] = 1.0                       # the exponent's log branch (|exponent - 1| <= 0.001)
    ssa = rng.uniform(0.0, 1.0, nc).astype(np.float32)
    asy = rng.uniform(-1.0, 1.0, nc).astype(np.float32)
    up = cp.asarray
    return ([up(p), up(t), up(qv), up(dz)], up(ht), up(kind), up(aod), up(aod3), up(ang), up(ssa), up(asy))


def _run(engine, cp, inputs, nz, nl, options, serial):
    arrays, ht, kind, aod, aod3, ang, ssa2, asy2 = inputs
    has3d, aod_opt, ang_opt, ssa_opt, asy_opt, diag = options
    nc = int(ht.size)
    # Words a combination leaves unwritten (3-D tau with exponent option 1/2) keep this sentinel in both paths.
    tau, ssa, asy = (cp.full((nc, 14, nl), np.float32(-7.25), dtype=cp.float32) for _ in range(3))
    column = [a.copy() for a in (aod, ang, ssa2, asy2)]
    engine._aerosol_passes(nc, nz, nl, has3d, aod_opt, ang_opt, ssa_opt, asy_opt, .12, 1.3, .85, .9, diag,
                           arrays, ht, kind, column[0], aod3 if has3d else cp.empty(0, cp.float32),
                           column[1], column[2], column[3], tau, ssa, asy, serial=serial)
    return [tau, ssa, asy, *column]


def test_every_option_combination_is_word_identical(engine):
    import cupy as cp
    nc, nz = 1029, 59
    inputs = _inputs(cp, nc, nz, 20261028)
    compared = differing = 0
    rows = []
    for nl in (nz + 1, nz):
        for options in itertools.product((False, True), (1, 2), (1, 2, 3), (1, 2, 3), (1, 2, 3), range(8)):
            serial = _run(engine, cp, inputs, nz, nl, options, True)
            passes = _run(engine, cp, inputs, nz, nl, options, False)
            for name, a, b in zip(("tau", "ssa", "asy", "aod2", "ang2", "ssa2", "asy2"), serial, passes):
                bad = int(cp.count_nonzero(a.view(cp.uint32) != b.view(cp.uint32)).get())
                compared += int(a.size)
                differing += bad
                if bad:
                    rows.append((nl, options, name, bad))
    assert compared > 0
    assert differing == 0, rows[:20]


def test_aerosol_optics_defaults_match_the_serial_kernel(engine):
    """The hrrrv5 / mp_thompson_aerosols call: 3-D optical depth, aod option 2, every table option 3."""
    import cupy as cp
    nc, nz = 2562, 59
    arrays, ht, kind, aod, aod3, *_ = _inputs(cp, nc, nz, 7)
    out = engine.aerosol_optics(*arrays, ht, kind, aod5502d=aod.copy(), aod5503d=aod3)
    tau, ssa, asy = (cp.empty((nc, 14, nz + 1), dtype=cp.float32) for _ in range(3))
    column = [aod.copy(), cp.empty(nc, cp.float32), cp.empty(nc, cp.float32), cp.empty(nc, cp.float32)]
    engine._aerosol_passes(nc, nz, nz + 1, True, 2, 3, 3, 3, .12, 1.3, .85, .9, 0, arrays, ht, kind,
                           column[0], aod3, column[1], column[2], column[3], tau, ssa, asy, serial=True)
    for name, expected in (("tauaer", tau), ("ssaaer", ssa), ("asyaer", asy), ("aod5502d", column[0])):
        assert bool(cp.all(out[name].view(cp.uint32) == expected.view(cp.uint32))), name


def test_out_of_range_aod_still_refuses(engine):
    import cupy as cp
    nc, nz = 64, 59
    arrays, ht, kind, aod, aod3, *_ = _inputs(cp, nc, nz, 11)
    aod[5] = -1.0
    with pytest.raises(ValueError, match=r"native aod550 outside \[0,inf\]"):
        engine.aerosol_optics(*arrays, ht, kind, aod5502d=aod, aod5503d=aod3)
