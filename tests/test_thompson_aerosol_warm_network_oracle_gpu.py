"""mp=28's warm network against unmodified WRF v4.6.1, bit for bit.

THE BREAKAGE THIS PREVENTS
--------------------------
Each column in ``tests/data/mp28_warm_network_oracle.npz`` is one whose
warm-network outputs (vapour, theta, cloud water and droplet number, rain
water and rain number, the two aerosol numbers, surface rain) the strict
GPU build gives EQUAL to WRF v4.6.1's own Fortran, and did not before the
transcription repairs of development branch.  Undoing any one of
them moves a committed column off WRF:

* ``pnr_wau``'s divisor ``am_r*nu_c*10.*D0r*D0r*D0r`` grouped
  ``(...*D0r)*(D0r*D0r)`` instead of left to right (:2191);
* the rain mean volume diameter divided by the DOUBLE literal 3.672 where
  WRF divides the REAL sum ``3.0 + mu_r + 0.672`` (:1889, :2149), which is
  what decides the 1950 micron break-up in ``pnr_rcr``, and the entry
  clamp's ``lamr**bm_r`` as a product chain where gfortran calls ``pow``;
* the rain collection rates read the :2147 ``lamr`` where WRF reads
  ``lamr = 1./ilamr(k)`` (:2198, :2213), the sub-freezing copy multiplied
  ``N0_r`` into the power before the REAL prefix, and its number caps were
  DOUBLE products where WRF's ``nc*odts`` is REAL;
* the cloud, rain, snow, graupel and vapour conservation limiters and the
  paired rain/graupel transfer carried ``sump``/``rate_max``/``ratio`` in
  DOUBLE where WRF declares them REAL (:1615, :2862-2950), and the warm
  rain number tendency summed its sinks in a different order (:3064);
* the source networks, the condensation and the rain evaporation added
  their increments to the running vapour and temperature, where WRF adds
  the rates to the REAL ``qvten``/``tten`` (:2982, :3164-3179, :3479-3483,
  :3563-3566) and re-forms ``qv1d + DT*qvten`` and ``t1d + DT*tten``
  (:3189, :3488-3489, :3569-3571).  That moved the post-adjustment
  ``ssatw`` whose sign opens the rain evaporation (:3501) at levels the
  adjustment had just saturated.

The committed answers are WRF's, from
``tools/thompson_aerosol_column_oracle`` (run_oracle.sh, then
make_warm_gate.py against the run before the repairs).

RE-CUT 2026-10-07 by development branch.  The first cut was taken on a tree
that still reproduced WRF's out-of-bounds rain-graupel table read; once
WOOF read the slab the tables hold (the declared rain-graupel divergence,
module_mp_thompson.F:465, :607-615, :2527-2545), the columns where rain
meets graupel left bit identity with WRF, among them all three real
"rain evaporation at saturation", both "rain break-up diameter" and both
"warm rain" columns.  The gate is now cut from the merged tree's 157-column
run (the four residue-cell edge columns included) against the e5302f46e
run, graded with WRF's answers there: 52 columns, 42 newly bit-identical in
every warm field at 20 s and 45 at 5 s.  The rain evaporation and break-up
repairs in rain-and-graupel columns are measured on the measurement copy
that reproduces WRF's read (tools/thompson_aerosol_column_oracle/
make_racg_read_copy.sh), where every column of the oracle is bit-identical;
tests/test_thompson_aerosol_column_oracle_gpu.py holds all 23 words of the
109 columns outside the divergence.  WRF ran on the
Exner function the adapter formed; this test checks the adapter forms the
same one again before it compares anything.

Strict build only (``GPUWM_WRF_EXACT=1``): that is the build the 0 ULP
claim is made for.  The GPU run is a subprocess of
``tools/thompson_aerosol_column_oracle/gpu_run.py`` so the strict compiler
hook is chosen before anything imports a kernel.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.gpu

_ROOT = Path(__file__).resolve().parents[1]
_GATE = _ROOT / "tests" / "data" / "mp28_warm_network_oracle.npz"
_GPU_RUN = _ROOT / "tools" / "thompson_aerosol_column_oracle" / "gpu_run.py"
_INPUTS = ("p", "th", "geop", "w", "qv", "qc", "qr", "qi", "qs", "qg", "ni",
           "nr", "nc", "nwfa", "nifa", "nwfa2d", "nifa2d")
_FIELDS = ("qv", "th", "qc", "nc", "qr", "nr", "nwfa", "nifa", "rainnc")


def _gate():
    z = np.load(_GATE, allow_pickle=False)
    return {k: z[k] for k in z.files}


def test_the_gate_covers_the_repaired_regimes():
    """The committed columns include the regimes the repairs act in."""
    gate = _gate()
    labels = [str(x) for x in gate["labels"]]
    for regime in ("rain emptied at the source stage", "cold snow riming",
                   "cloud drained by heavy rain", "polluted cumulus",
                   "freezing rain", "stable night fog"):
        assert any(regime in label for label in labels), regime
    for dt in gate["dts"]:
        assert int(gate[f"check_dt{dt}"].sum()) >= 40, dt


@pytest.mark.parametrize("dt", [20, 5])
def test_warm_network_outputs_equal_wrf_bit_for_bit(tmp_path, dt):
    pytest.importorskip("cupy")
    gate = _gate()
    columns = tmp_path / "columns.npz"
    np.savez(columns, labels=gate["labels"],
             regime=np.asarray(["gate"] * len(gate["labels"])),
             **{k: gate[k] for k in _INPUTS})
    out = tmp_path / f"gpu-dt{dt}.npz"
    env = dict(os.environ, GPUWM_WRF_EXACT="1")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_ROOT), env.get("PYTHONPATH", "")])
    subprocess.run([sys.executable, str(_GPU_RUN), str(columns), str(out),
                    "--dt", str(dt)], check=True, env=env, cwd=tmp_path)
    got = np.load(out)
    check = gate[f"check_dt{dt}"]

    def words(a):
        return np.asarray(a, np.float32).view(np.int32)

    pii_same = (words(got["in_pii"]) == words(gate[f"in_pii_dt{dt}"])).all()
    assert pii_same, "the adapter formed a different Exner function"
    lines = []
    for name in _FIELDS:
        same = words(got[name]) == words(gate[f"wrf_{name}_dt{dt}"])
        same = same if same.ndim == 1 else same.all(axis=1)
        bad = np.flatnonzero(check & ~same)
        if bad.size:
            lines.append(f"{name}: {bad.size} columns differ from WRF, "
                         f"first {str(gate['labels'][bad[0]])!r}")
    assert not lines, "; ".join(lines)
