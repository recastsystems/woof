"""MYJ seams found by replaying WRF 4.6.1's own wrapper words (2026-10-29).

The WRF-JAX CONUS-suite case (500x500x44, Thompson/RRTMG/MYJ/Eta/Noah/
Tiedtke) was run through a capture build of the pinned WRF 4.6.1 (glibc
2.43) that writes every MYJSFC and MYJPBL wrapper argument before and after
the call (tools/myj_wrf461_capture).  Replaying those words through the
CUDA kernels left three seams, each pinned here:

1. MYJSFCINIT's PSIM table calls the host's atanf, and glibc 2.41+ ship a
   correctly rounded one: 816 of 10001 PSIM words differ between a glibc
   2.39 referee and a 2.43 one.  ``GPUWM_WRF_GLIBC`` names the referee.
2. WRF's Noah writes CHKLOWQ = 0 under the Eta layer when the lowest level
   is near saturation and below QZ0 (module_sf_noahdrv.F:803-814); the Noah
   kernel writes 1 only.
3. MYJSFC's Z0BASE is the Registry Z0 Noah rewrites each call, not a frozen
   copy of the cold-start ZNT (22,141 of 31,166 land columns differed from
   step 2 on).
"""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from conftest import requires_gpu
from woof.core.myjsfc_tables import build_psi_tables
from woof.core.noahmp_libm import atanf, atanf_correctly_rounded

#: SHA-256 of the little-endian float32 words MYJSFCINIT wrote in the
#: pinned WRF 4.6.1 build on Ubuntu GLIBC 2.43-2ubuntu2.4 (a development machine,
#: tools/myj_wrf461_capture/run_myjsfc on the captured case, 2026-10-29).
REFEREE_243_PSIM = "404d6c605938057af623d99b6856b7131a9c5dc00e0dd27faa0c8f1094bf6f93"
REFEREE_PSIH = "2a60d4f997dc3b5a0a63907a8fd236ea0fdc7814c46b4934a482a230c3f5a28b"
#: The glibc 2.39 generation (the committed W1 column oracle's tables).
GLIBC_239_PSIM = "1b4b769bb05499b73f91775008331a2f93b2aa0d534a8134eeaaad3b61b0d4ea"


def _sha(words) -> str:
    return hashlib.sha256(np.asarray(words, "<f4").tobytes()).hexdigest()


def test_psim_table_follows_the_referee_glibc():
    old, new = build_psi_tables(False), build_psi_tables(True)
    for name in ("psim1", "psim2"):
        assert _sha(old[name]) == GLIBC_239_PSIM
        assert _sha(new[name]) == REFEREE_243_PSIM
    for name in ("psih1", "psih2"):
        assert _sha(old[name]) == _sha(new[name]) == REFEREE_PSIH
    differ = np.asarray(old["psim1"]).view(np.int32) != np.asarray(new["psim1"]).view(np.int32)
    assert int(differ.sum()) == 816


def test_correctly_rounded_atanf_edges():
    assert atanf_correctly_rounded(np.float32(0.0)) == 0.0
    assert np.signbit(atanf_correctly_rounded(np.float32(-0.0)))
    half_pi = atanf_correctly_rounded(np.float32(np.inf))
    assert half_pi == np.float32(np.pi / 2)
    assert atanf_correctly_rounded(np.float32(-np.inf)) == -half_pi
    assert np.isnan(atanf_correctly_rounded(np.float32(np.nan)))
    for x in (1e-30, 1e-5, 0.3, 1.0, 1.7, 2.9, 1e3, 3e7):
        x = np.float32(x)
        assert atanf_correctly_rounded(-x) == -atanf_correctly_rounded(x)
        # Both generations agree to within the old routine's error.
        assert abs(float(atanf_correctly_rounded(x)) - float(atanf(x))) <= 2 * np.spacing(np.float32(atanf(x)))


def test_referee_glibc_parsing(monkeypatch):
    from woof import wrf_exact
    monkeypatch.delenv(wrf_exact.GLIBC_VARIABLE, raising=False)
    assert wrf_exact.referee_glibc() == (2, 39)
    assert not wrf_exact.referee_correctly_rounded_trig()
    for text, expect in (("2.43", True), ("glibc-2.41", True), ("2.40", False), ("glibc 2.39", False)):
        monkeypatch.setenv(wrf_exact.GLIBC_VARIABLE, text)
        assert wrf_exact.referee_correctly_rounded_trig() is expect
    monkeypatch.setenv(wrf_exact.GLIBC_VARIABLE, "new")
    with pytest.raises(ValueError, match="GPUWM_WRF_GLIBC"):
        wrf_exact.referee_glibc()


@pytest.mark.gpu
@requires_gpu
def test_noah_myj_chklowq_arm():
    import cupy as cp
    from woof.core.noah import noah_chklowq_myj
    qv = cp.asarray([[0.010, 0.010, 0.010, 0.0099, 0.020]], dtype=cp.float32)
    qgh = cp.asarray([[0.0104, 0.0104, 0.0200, 0.0104, 0.0204]], dtype=cp.float32)
    qz0 = cp.asarray([[0.0110, 0.0090, 0.0110, 0.0110, 0.0300]], dtype=cp.float32)
    fields = {"qv1": qv, "qgh": qgh, "qz0": qz0,
              "chklowq": cp.ones(qv.shape, cp.float32)}
    noah_chklowq_myj(fields)
    q2k = qv / (1 + qv)
    q2sat = qgh / (1 + qgh)
    want = cp.where((q2k >= q2sat * np.float32(0.95)) & (q2k < qz0), 0.0, 1.0)
    assert cp.asnumpy(fields["chklowq"]).tolist() == cp.asnumpy(want).tolist()
    # near-saturated and below QZ0 -> 0; above QZ0, unsaturated -> 1
    assert cp.asnumpy(fields["chklowq"])[0, 0] == 0.0
    assert cp.asnumpy(fields["chklowq"])[0, 1] == 1.0
    assert cp.asnumpy(fields["chklowq"])[0, 2] == 1.0


def test_myjsfc_z0base_is_the_live_z0():
    from pathlib import Path
    import woof
    from woof.core.physics_inventory import MYJ_SFCLAY_FIELDS_2D
    assert "z0base" not in MYJ_SFCLAY_FIELDS_2D
    source = (Path(woof.__file__).parent / "core" / "physics.py").read_text(encoding="utf-8")
    body = source.split("def _run_myj_sfclay", 1)[1].split("    def ", 1)[0]
    assert '"z0base": f["z0"]' in body


def test_noah_myj_selector_is_rebuilt_from_configuration():
    from woof.io.restart import DRIVER_REBUILT_ATTRS
    assert "noah_myj" in DRIVER_REBUILT_ATTRS
