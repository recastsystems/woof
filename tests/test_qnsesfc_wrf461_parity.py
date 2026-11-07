"""The QNSE surface layer against WRF v4.6.1, word for word.

The fixture (tests/data/oracles/qnsesfc/qnsesfc-wrf461.npz) is the output of
tools/qnsesfc_wrf461_oracle/build.sh: the byte-unmodified
phys/module_sf_qnsesfc.F compiled by gfortran 13.3 on glibc 2.39 with WRF's
own optimised phys flags, driven through QNSESFCINIT and successive QNSESFC
calls over 240 columns in three sets (cold start at ITIMESTEP 1-4, warm
seeded state at ITIMESTEP 8-11 across the three sea viscous regimes, and the
two-level minimum column), with a stand-in land-surface HFX/QFX written over
the land columns before every call because QNSE's land branch reads the
incoming HFX.  Its compiler.txt, o2-equality.txt, o0-difference.txt and
coverage-qnsesfc.txt beside it are the build receipts: no FMA, no libmvec,
stock configure's flags write the reference's bytes, and every executable
line of QNSESFC and SFCDIF runs but one debug PRINT.

What is asserted:

* CPU, everywhere: the PSIM/PSIH tables WOOF builds on the host are
  QNSESFCINIT's, all 4 x 10,001 words and the scalars.
* GPU: every INOUT and output field of every call is bit-identical to
  QNSESFC, both replaying WRF's pre-call state and free-running on the
  kernel's own.  Strict mode disables contraction and retains scalar WRF math. Production
  uses CUDA math and contraction and is measured separately.
* A planted one-bit change in one kernel constant (PRT0) is caught.
* The Eta (MYJ) kernel run on the same columns is NOT QNSE: the grade tells
  the two schemes apart, so a dispatch that sent sf_sfclay_physics=4 to the
  MYJ kernel would fail here.
"""

from __future__ import annotations

import os
import numpy as np
import pytest

from conftest import requires_gpu
from woof.verify import qnsesfc_oracle


@pytest.fixture(scope="module")
def fixture():
    return qnsesfc_oracle.load()


def _zsl(fixture, name):
    return fixture[f"{name}/in/dz"][0] * np.float32(0.5)


def test_the_fixture_covers_the_regimes_it_claims(fixture):
    names = qnsesfc_oracle.set_names(fixture)
    assert names == ["cold", "nz2", "warm"]
    columns = sum(int(fixture[f"{s}/meta"][0]) for s in names)
    assert columns >= 64
    fields = [str(f) for f in fixture["fields"]]
    for name in names:
        xland = fixture[f"{name}/in/xland"]
        assert np.any(xland > 1.5) and np.any(xland < 1.5), name
        # Multiple calls per set, so USTAR and Z0 carry.
        assert int(fixture[f"{name}/meta"][2]) >= 3, name
    # Terrain above the TOPOTERM floor (TOPOFAC*HT**2 > 3 needs HT > 577 m)
    # in every set, and above 4 km somewhere.
    for name in names:
        assert float(fixture[f"{name}/in/ht"].max()) > 1000.0, name
    assert max(float(fixture[f"{s}/in/ht"].max()) for s in names) > 4000.0
    warm_ust = fixture["warm/in/ustar"]
    assert np.any(warm_ust < 0.225)
    assert np.any((warm_ust >= 0.225) & (warm_ust < 0.7))
    assert np.any(warm_ust >= 0.7)
    # Stable, neutral and convective surface layers, and both ends of the
    # ZETA table: z/L at the lowest half-level beyond -5 and beyond 2.3.
    zeta = []
    rib = []
    for name in names:
        out = fixture[f"{name}/out"]
        zeta.append(out[:, fields.index("rmol")] * _zsl(fixture, name))
        rib.append(out[:, fields.index("rib")])
    zeta = np.concatenate([z.ravel() for z in zeta])
    rib = np.concatenate([r.ravel() for r in rib])
    assert np.any(zeta < -5.0) and np.any(zeta > 2.3)
    assert np.any(rib < -0.1) and np.any(rib > 0.1)
    assert np.any(np.abs(rib) < 0.01)
    pblh = fixture["cold/out"][:, fields.index("pblh")]
    assert np.any(pblh > 1000.0) and np.any(pblh < 1000.0)
    # The land branch reads a non-zero incoming HFX on warm calls.
    assert np.any(fixture["warm/pre"][:, list(fixture["pre_fields"]).index("hfx")]
                  != 0.0)


def test_the_host_tables_are_qnsesfcinits_word_for_word(fixture):
    report = qnsesfc_oracle.compare_tables(fixture)
    for name, stats in report.items():
        assert stats["mismatch"] == 0, (name, stats)


@pytest.mark.gpu
@requires_gpu
@pytest.mark.skipif(os.environ.get("GPUWM_WRF_EXACT") != "1",
                    reason="word-grade proof selects GPUWM_WRF_EXACT=1")
def test_every_field_of_every_call_is_wrfs(fixture):
    report = qnsesfc_oracle.measure(fixture)
    assert qnsesfc_oracle.failing_fields(report) == {
        "replay": {}, "free_run": {}}


@pytest.mark.gpu
@requires_gpu
@pytest.mark.skipif(os.environ.get("GPUWM_WRF_EXACT") != "1",
                    reason="word-grade proof selects GPUWM_WRF_EXACT=1")
def test_a_planted_one_bit_constant_change_is_caught(fixture, monkeypatch):
    """PRT0 moved by one ULP in the kernel source: the oracle sees it."""
    import cupy as cp
    from woof.core import kernels, myjsfc

    source = kernels.module_source("qnsesfc")
    line = "#define QNSE_PRT0     0.72f"
    assert source.count(line) == 1
    bumped = np.nextafter(np.float32(0.72), np.float32(1.0))
    planted = source.replace(
        line, f"#define QNSE_PRT0     {float(bumped):.9g}f")
    assert planted != source
    from woof import nvrtc_ptx_cache as compiler
    image, _ = compiler.compile_using_nvrtc(
        planted, kernels.module_options("qnsesfc"), None,
        "qnsesfc_one_bit.cu")
    module = cp.cuda.function.Module()
    module.load(image.encode() if isinstance(image, str) else image)
    function = module.get_function("qnsesfc_column")
    real_get_kernel = myjsfc.get_kernel
    monkeypatch.setattr(
        myjsfc, "get_kernel",
        lambda name, entry: function if name == "qnsesfc"
        else real_get_kernel(name, entry))
    report = qnsesfc_oracle.measure(fixture)
    bad = qnsesfc_oracle.failing_fields(report)
    assert bad["replay"], "a one-ULP PRT0 change left every word unchanged"
    assert "chs" in bad["replay"] and "th2" in bad["replay"]


@pytest.mark.gpu
@requires_gpu
def test_the_eta_kernel_is_not_qnse(fixture):
    """The MYJ row run on QNSE's columns fails the grade."""
    report = qnsesfc_oracle.measure(fixture, sf_sfclay_physics=2)
    bad = qnsesfc_oracle.failing_fields(report)
    assert {"pblh", "akhs", "chs", "th2"} <= set(bad["replay"])
