"""mp=28 against unmodified WRF v4.6.1 on the whole column oracle, bit for bit.

THE BREAKAGE THIS PREVENTS
--------------------------
``tests/data/mp28_column_oracle_wrf461.npz`` carries the 157 columns of the
mp=28 column oracle (``tools/thompson_aerosol_column_oracle``: 42 real
convective columns, eight synthetic regimes raw and after WRF's own
spin-up, 19 edge cases) with WRF v4.6.1's answers for all 23 compared words
(the eleven moments, theta, 10 cm reflectivity, three effective radii and
seven surface accumulations) at 20 s and 5 s.  On every column where rain
does not meet graupel, WOOF's mp=28 must give those words exactly.

That covers every repair that made it so, each of which, undone, moves a
committed column: the source networks in WRF's arithmetic order
(ThompsonAaLevel, the warm and cold lanes), WRF's REAL accumulators
applied once (qc, qr, nr, qi, ni, qv, T, and since development branch qs, qg
and the graupel number), the phase cleanup's latent heat on WRF's own
ocp(k)/lvap(k), the snow and graupel
fallout, surface totals and echo in WRF's form, and WOOF's own math
routines, which reproduce the host libm's results bit for bit.

The remaining columns are the declared rain-graupel divergence: WRF reads
its rain-graupel collision tables out of bounds when the scheme is not
hail aware (module_mp_thompson.F:465, :607-615, :2527-2545) and WOOF reads
the slab the tables hold.  The cutter
(``tools/thompson_aerosol_column_oracle/make_oracle_gate.py --wrfread``)
refused to write the gate unless every one of those columns was
bit-identical on the measurement copy that reproduces WRF's read.

Both arithmetic builds are checked: the strict one (``GPUWM_WRF_EXACT=1``)
the 0 ULP claim is made for, and the default one, which compiles the same
pinned operations and must give the same words.  The GPU run is a
subprocess of ``tools/thompson_aerosol_column_oracle/gpu_run.py`` so the
compiler hook is chosen before anything imports a kernel.

The 48 declared columns are pinned to WOOF's strict words on all 23 fields.
The exact-freezing fixture below separately holds the melting-level repair,
including a real legacy-mask run that must differ from WRF's answers.
"""

from __future__ import annotations

import os
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.gpu

_ROOT = Path(__file__).resolve().parents[1]
_GATE = _ROOT / "tests" / "data" / "mp28_column_oracle_wrf461.npz"
_GPU_RUN = _ROOT / "tools" / "thompson_aerosol_column_oracle" / "gpu_run.py"
_INPUTS = ("p", "th", "geop", "w", "qv", "qc", "qr", "qi", "qs", "qg", "ni",
           "nr", "nc", "nwfa", "nifa", "nwfa2d", "nifa2d")


def _gate():
    z = np.load(_GATE, allow_pickle=False)
    return {k: z[k] for k in z.files}


def _words(a):
    return np.asarray(a, np.float32).view(np.int32)


def test_the_gate_is_the_whole_oracle_set():
    """157 columns, all 23 words, both steps; the declared columns are a
    minority and every one of them carries rain and graupel at entry or is
    a WRF spin-up of a column that does."""
    gate = _gate()
    labels = [str(x) for x in gate["labels"]]
    assert len(labels) == 157
    assert len(gate["fields"]) == 23
    assert sorted(int(x) for x in gate["dts"]) == [5, 20]
    checks = json.loads(str(gate["measurement_copy_checks"]))
    present = ((gate["qr"] > 1.0e-12) & (gate["qg"] > 1.0e-12)).any(axis=1)
    for dt in gate["dts"]:
        assert checks[str(dt)]["columns_checked"] == 157
        assert checks[str(dt)]["columns_identical"] == 157
        same = gate[f"identical_dt{dt}"]
        assert int(same.sum()) >= 109, (dt, int(same.sum()))
        # The declared divergence can only act where rain and graupel share
        # a level (:2527-2545 runs under L_qr and rg >= r_g(1)).
        stray = [labels[i] for i in np.flatnonzero(~same & ~present)]
        assert not stray, stray
    for regime in ("real", "edge", "convective", "snow/ice",
                   "stable night fog", "freezing rain"):
        assert any(regime in label for label in labels), regime


@pytest.mark.parametrize("mode", ["strict", "default"])
@pytest.mark.parametrize("dt", [20, 5])
def test_every_column_outside_the_declared_divergence_equals_wrf(
        tmp_path, dt, mode):
    pytest.importorskip("cupy")
    gate = _gate()
    columns = tmp_path / "columns.npz"
    np.savez(columns, labels=gate["labels"],
             regime=np.asarray(["gate"] * len(gate["labels"])),
             **{k: gate[k] for k in _INPUTS})
    out = tmp_path / f"gpu-{mode}-dt{dt}.npz"
    env = dict(os.environ)
    if mode == "strict":
        env["GPUWM_WRF_EXACT"] = "1"
    else:
        env.pop("GPUWM_WRF_EXACT", None)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_ROOT), env.get("PYTHONPATH", "")])
    subprocess.run([sys.executable, str(_GPU_RUN), str(columns), str(out),
                    "--dt", str(dt)], check=True, env=env, cwd=tmp_path)
    got = np.load(out)
    assert (_words(got["in_pii"]) == _words(gate[f"in_pii_dt{dt}"])).all(), (
        "the adapter formed a different Exner function")
    expected = gate[f"identical_dt{dt}"]
    lines = []
    for name in (str(x) for x in gate["fields"]):
        target = np.array(gate[f"wrf_{name}_dt{dt}"], copy=True)
        target[~expected] = gate[f"woof_{name}_dt{dt}"][~expected]
        same = _words(got[name]) == _words(target)
        same = same if same.ndim == 1 else same.all(axis=1)
        bad = np.flatnonzero(~same)
        if bad.size:
            lines.append(f"{name}: {bad.size} columns differ from WRF, "
                         f"first {str(gate['labels'][bad[0]])!r}")
    assert not lines, "; ".join(lines)


@pytest.mark.parametrize("mode", ["strict", "default", "legacy-mask"])
@pytest.mark.parametrize("dt", [20, 5])
def test_exact_freezing_columns_equal_wrf_and_reject_legacy_mask(tmp_path, mode, dt):
    pytest.importorskip("cupy")
    fixture = np.load(_ROOT / "tests/data/mp28_melting_level_wrf461.npz")
    columns = tmp_path / "columns.npz"
    np.savez(columns, labels=fixture["labels"], regime=fixture["regime"],
             **{k: fixture[k] for k in _INPUTS})
    out = tmp_path / "out.npz"
    env = dict(os.environ)
    if mode == "default":
        env.pop("GPUWM_WRF_EXACT", None)
    else:
        env["GPUWM_WRF_EXACT"] = "1"
    cmd = [sys.executable, str(_GPU_RUN), str(columns), str(out), "--dt", str(dt)]
    if mode == "legacy-mask":
        cmd.append("--legacy-warm-mask")
    subprocess.run(cmd, check=True, env=env, cwd=tmp_path)
    got = np.load(out)
    fields = [str(x) for x in _gate()["fields"]]
    same = {k: np.array_equal(_words(got[k]), _words(fixture[f"wrf_{k}_dt{dt}"]))
            for k in fields}
    if mode == "legacy-mask":
        assert not same["nc"], "the real legacy adapter must expose the melting defect"
    else:
        assert all(same.values()), [k for k, equal in same.items() if not equal]
