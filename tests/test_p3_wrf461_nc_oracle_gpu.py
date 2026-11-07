"""P3 with prognostic droplet number (mp_physics=51) against WRF 4.6.1.

The column oracle (tools/p3_wrf461_oracle, README there) runs WRF's
unmodified ``module_mp_p3.F`` (sha256 716950a3...f6a3c6) through
``mp_p3_wrapper_wrf`` with the mp_physics=51 call shape -- ``nc_3d``
passed, so ``log_predictNc`` is on (module_mp_p3.F:819-820) -- compiled
``-O0 -ffp-contract=off`` and linked with correctly rounded ``tgammaf`` and
``log10f`` (arm CR).  These gates replay committed slices of it through
``run_p3_device`` with ``log_predictNc=True`` -- the call
``woof.core.p3.apply`` makes for mp=51 -- and require every float32 word
of every frame, the droplet number included, to be EQUAL.

Breakage each prevents, measured before the port this file landed with:

* the device draft evaluated the activation's ``derf`` as CUDA's double
  ``erf``; WRF's ``DERF`` is P3's own float32 piecewise polynomial
  (module_mp_p3.F:6090-6204), so activated droplet number differed from
  WRF in every supersaturated cell at step 1 (``NA03``, ``NS30``);
* a planted one-bit change in the activation path (the surface-tension
  constant ``sigvl``, :3318) must fail this gate -- the planted-defect test
  below shows the gate can fail, so its passes mean something.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "data" / "oracles" / "p3-nc"
INDEX = json.loads((DATA / "index.json").read_text())
IN3D = INDEX["in3d"]
IN2D = INDEX["in2d"]
OUT3D = INDEX["out3d"]
OUT2D = INDEX["out2d"]
CASES = [(case, DATA / f"{case['name']}.bin", DATA / f"{case['name']}.ref")
         for case in INDEX["cases"]]

#: The planted defect: one bit of the activation's surface-tension constant
#: (0.0761 -> its float32 successor), exactly one occurrence in p3.cu.
PLANT_FROM = "float sigvl = 0.0761f - 1.55e-4f * (t - 273.15f);"
PLANT_TO = "float sigvl = 0x1.37b4a4p-4f - 1.55e-4f * (t - 273.15f);"


def _run(case, fixture, arm, module=None):
    import cupy as cp

    from woof.core import p3_device as PD

    ncol, nk = int(case["ncol"]), int(case["nk"])
    nsteps, dt = int(case["nsteps"]), float(case["dt"])
    buf = np.fromfile(fixture, dtype="<f4")
    assert buf.size == len(IN3D) * ncol * nk + len(IN2D) * ncol
    host, off = {}, 0
    for name in IN3D:      # (ncol, nk) Fortran order == (nk, ncol) C order
        host[name] = buf[off:off + ncol * nk].reshape((nk, ncol)).copy()
        off += ncol * nk
    fields = {name: cp.asarray(host[name]) for name in
              ("qc", "nc", "qr", "nr", "qi", "qir", "ni", "qib", "th", "qv",
               "th_old", "qv_old")}
    fields["pres"] = cp.asarray(host["p"])
    fields["dz"] = cp.asarray(host["dz"])
    fields["ssat"] = cp.zeros((nk, ncol), dtype=cp.float32)
    diag = {name: cp.full((nk, ncol), -8.888e30, dtype=cp.float32)
            for name in PD.DIAG_SLOTS}
    surf = {"prt_liq": cp.zeros(ncol, dtype=cp.float32),
            "prt_sol": cp.zeros(ncol, dtype=cp.float32)}
    for name in IN2D:
        surf[name] = cp.asarray(buf[off:off + ncol].copy())
        off += ncol
    workspace = PD.make_workspace(ncol, nk)
    frames = []

    def snap():
        parts = [cp.asnumpy(fields[n] if n in fields else diag[n]).ravel()
                 for n in OUT3D]
        parts += [cp.asnumpy(surf[n]).ravel() for n in OUT2D]
        frames.append(np.concatenate(parts).astype("<f4"))

    snap()
    for it in range(1, nsteps + 1):
        PD.run_p3_device(fields, diag, surf, workspace=workspace, dt=dt,
                         it=it, arm=arm, log_predictNc=True, module=module)
        snap()
    return np.concatenate(frames)


def _first_difference(case, got, want):
    differ = np.flatnonzero(got.view(np.uint32) != want.view(np.uint32))
    if not differ.size:
        return None
    ncol, nk = int(case["ncol"]), int(case["nk"])
    per = len(OUT3D) * ncol * nk + len(OUT2D) * ncol
    step, at = divmod(int(differ[0]), per)
    field = (OUT3D[at // (ncol * nk)] if at < len(OUT3D) * ncol * nk
             else OUT2D[(at - len(OUT3D) * ncol * nk) // ncol])
    return (f"{differ.size} of {got.size} words differ from WRF 4.6.1; "
            f"first at step {step}, field {field}: WRF {want[differ[0]]!r}, "
            f"WOOF {got[differ[0]]!r}")


@pytest.mark.parametrize("arm", ["sedlevels", "unfused", "fused", "levels"])
@pytest.mark.parametrize("case,fixture,reference", CASES,
                         ids=[row[0]["name"] for row in CASES])
def test_every_word_of_every_frame_is_wrfs(case, fixture, reference, arm):
    got = _run(case, fixture, arm)
    want = np.fromfile(reference, dtype="<f4")
    assert got.size == want.size
    bad = _first_difference(case, got, want)
    if bad:
        pytest.fail(f"{case['name']} [{arm}]: {bad}")


def test_the_droplet_number_actually_moves():
    """Not vacuous: WRF's own dump shows activation and the nc sinks at
    work -- nc changes in most cloudy cells over the activation fixture."""
    case, _fixture, reference = next(
        row for row in CASES if row[0]["name"].startswith("NA03"))
    ncol, nk = int(case["ncol"]), int(case["nk"])
    per = len(OUT3D) * ncol * nk + len(OUT2D) * ncol
    want = np.fromfile(reference, dtype="<f4").reshape(-1, per)
    k = OUT3D.index("nc")
    nc0 = want[0, k * ncol * nk:(k + 1) * ncol * nk]
    nc1 = want[1, k * ncol * nk:(k + 1) * ncol * nk]
    assert np.count_nonzero(nc1 != nc0) > ncol * nk // 4


def test_a_planted_one_bit_activation_change_fails_the_gate():
    from woof.core import p3_device as PD

    source = PD.p3_source()
    assert source.count(PLANT_FROM) == 1, "plant site moved; update PLANT_*"
    planted = PD.p3_module(PD.DEFAULT_OPTIONS,
                           source=source.replace(PLANT_FROM, PLANT_TO))
    case, fixture, reference = next(
        row for row in CASES if row[0]["name"].startswith("NA03"))
    want = np.fromfile(reference, dtype="<f4")
    assert _first_difference(case, _run(case, fixture, "sedlevels"),
                             want) is None
    caught = _first_difference(
        case, _run(case, fixture, "sedlevels", module=planted), want)
    assert caught is not None, "the gate did not see a one-bit activation change"
