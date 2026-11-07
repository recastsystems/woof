"""The CUDA WSM6 (mp_physics = 6) against unmodified WRF v4.6.1, word for word.

THE BREAKAGE THIS PREVENTS
--------------------------
Until development branch the CUDA WSM6 kernel was a restructured
transcription graded only within a committed ULP table: up to 596 ULP on
real-valued fields, 7.7e8 ULP on near-zero cells, reflectivity off by up
to 15,421 ULP, because it re-associated WRF's sums, used CUDA's expf/
logf/powf, recomputed mp_wsm6_init's Gamma-function constants in float64
(WRF's are float32: pvtr 2500.0637, not 2500.0682), sedimented cloud ice
with a different fall-speed law and formed reflectivity in float64 where
WRF mixes float32 and float64.  The registry called that "supported".
This file holds the rewritten kernel to WRF's own words, so any edit that
moves one word away from WRF goes red here.

THE ORACLE (tests/data/oracles/wsm6, built by
tools/wsm6_wrf461_column_oracle/build.sh on W1: gfortran 13.3.0 -O0,
glibc 2.39, WRF v4.6.1 at d66e442f, no source edited) drives WRF's own
entry point ``module_mp_wsm6::wsm6`` -- the wrapper the microphysics
driver calls -- with WRF's own constants module, the radiation effective
radii on and REFL_10CM requested, over 65 columns x 45 levels: deep
convection, stratiform rain through a melting layer, freezing rain, stable
clear nights, fog, marine stratocumulus, all-snow winter columns, high
terrain, cirrus, supercooled cloud below -40 C, and edge columns sitting
exactly on qcrmin, qc0, qmin, t0c and the triple point, with negative,
negative-zero and subnormal inputs, condensate only at the top or bottom,
thin layers that force the sedimentation CFL limiter, and a supersaturated
clear column.  Six runs: hail_opt 0 and 1 at dt 30 s
(12 free-running steps), 150 s (3 steps, one minor loop longer than
dtcldcr) and 300 s (4 steps, three minor loops).  Every output: th, qv,
qc, qi, qr, qs, qg, re_cloud/re_ice/re_snow (as woof's micron words,
re*1.0e6 in float32), REFL_10CM, th*pii, and rain, rainncv, snow, snowncv,
graupel, graupelncv, sr.

THE CLAIMS
----------
* strict build (GPUWM_WRF_EXACT=1: no FMA, no FTZ): every word of every
  step of every run is WRF's -- SHA-256 per field per step, and the final
  step word by word -- at both WSM6_KMAX tiers the launcher compiles;
* default build (CuPy's -ftz=true): the same, on every column except the
  two whose WRF trajectory carries subnormal words (``edge-negative``
  feeds 1e-40-class hydrometeors, ``edge-dry-tiny`` produces 1.8e-44
  snow), which a flushing build cannot reproduce by construction.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from conftest import requires_gpu

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "data" / "oracles" / "wsm6"
#: Columns whose WRF trajectory holds subnormal words.  The default build
#: compiles with -ftz=true (CuPy appends it to every module), so it flushes
#: them; the strict build does not exclude them.
FTZ_COLUMNS = ("edge-negative", "edge-dry-tiny")


def _dump_words() -> dict:
    """``wsm6-init-f32.txt`` as {hail: {name: [hex words in order]}}."""
    out: dict = {}
    hail = None
    for line in (FIXTURE / "wsm6-init-f32.txt").read_text().splitlines():
        parts = line.split()
        if parts[0] == "hail_opt":
            hail = int(parts[1])
            out[hail] = {}
            continue
        out[hail].setdefault(parts[0], []).append(int(parts[1], 16))
    return out


def test_frozen_init_and_radar_words_are_the_words_wrf_computed():
    """The kernel's constants are mp_wsm6_init's and radar_init's own words.

    Breakage prevented: a constant recomputed on the host (float64 Gamma,
    exact 720 for WGAMMA(7.)) instead of copied from WRF's float32 init.
    """
    from woof.core.wsm6_constants import (WRF_INIT_F32, WRF_RADAR_F32,
                                           WRF_RADAR_F64)
    dump = _dump_words()
    for hail in (0, 1):
        for name, bits in WRF_INIT_F32[hail]:
            assert dump[hail][name] == [bits], (hail, name)
        indexed = {"xcre1": ("xcre", 0), "xcre4": ("xcre", 3),
                   "xcrg3": ("xcrg", 2), "xcrg4": ("xcrg", 3),
                   "xcse1": ("xcse", 0), "xcse4": ("xcse", 3),
                   "xcsg3": ("xcsg", 2), "xcsg4": ("xcsg", 3),
                   "xcge1": ("xcge", 0), "xcge4": ("xcge", 3),
                   "xcgg3": ("xcgg", 2), "xcgg4": ("xcgg", 3)}
        for name, bits in WRF_RADAR_F32[hail]:
            key, i = indexed.get(name, (name, 0))
            assert dump[hail][key][i] == bits, (hail, name)
        f64 = [bits for _, bits in WRF_RADAR_F64]
        scalars = [dump[hail][n][0] for n in ("k_w", "pi5", "lamda4", "m_w_0_re",
                                             "m_w_0_im", "m_i_0_re", "m_i_0_im")]
        bins = [dump[hail][n] for n in ("xxds", "xdts", "xxdg", "xdtg", "simpson")]
        assert f64 == scalars + [w for group in bins for w in group], hail


def test_the_column_set_is_the_oracle_input_and_spans_the_regimes():
    """Breakage prevented: a fixture regenerated from a different column set
    than the one WRF ran, or one that lost a regime."""
    import hashlib

    meta = json.loads((FIXTURE / "columns.json").read_text())
    digests = json.loads((FIXTURE / "wrf461-step-digests.json").read_text())
    assert hashlib.sha256((FIXTURE / "columns.bin").read_bytes()).hexdigest() \
        == digests["columns_bin_sha256"]
    assert meta["ncol"] >= 64 and meta["nz"] == 45
    regimes = {c["regime"] for c in meta["columns"]}
    for needed in ("deep convection", "stratiform melting layer",
                   "freezing rain", "stable clear night", "radiation fog",
                   "ocean stratocumulus", "winter snow", "high terrain",
                   "cirrus", "supercooled below -40C"):
        assert needed in regimes, needed
    assert sum(r.startswith("edge:") for r in regimes) >= 10
    for name in FTZ_COLUMNS:
        assert name in {c["name"] for c in meta["columns"]}, name


def test_the_reference_was_built_without_vector_libm():
    """The -O0 reference pulled no glibc vector-math symbol (build.sh fails
    otherwise), and the positive control proves the grep can see one."""
    report = (FIXTURE / "libmvec-report.txt").read_text()
    o0 = report.split("# -O0 (THE REFERENCE)")[1].split("# -O2")[0]
    assert "_ZGV" not in o0
    assert "_ZGVbN4v_expf" in report.split("# positive control:")[1]


def _check(extra_env: dict, *args) -> dict:
    env = dict(os.environ)
    env.pop("GPUWM_WRF_EXACT", None)
    env.update(extra_env)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), str(ROOT / "tests"), env.get("PYTHONPATH", "")])
    proc = subprocess.run(
        [sys.executable, "-m", "tools.wsm6_wrf461_column_oracle.run_gpu",
         "check-fixture", str(FIXTURE), *args],
        cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=1800)
    lines = proc.stdout.strip().splitlines()
    assert lines, proc.stderr[-4000:]
    return json.loads(lines[-1])


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("tier", (64, 80))
def test_strict_build_is_wrf461_on_every_word_of_every_step(tier):
    """Strict arithmetic: zero differing words, every column, every step."""
    result = _check({"GPUWM_WRF_EXACT": "1"}, "--tier", str(tier))
    assert result["strict_build"] is True
    assert result["total_mismatch"] == 0, result


@pytest.mark.gpu
@requires_gpu
def test_default_build_is_wrf461_outside_the_subnormal_columns():
    """The arithmetic users run: bitwise wherever WRF's words are normal."""
    args = []
    for name in FTZ_COLUMNS:
        args += ["--exclude", name]
    result = _check({}, *args)
    assert result["strict_build"] is False
    assert result["total_mismatch"] == 0, result
