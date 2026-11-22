"""The whole GF chain on the device against the pinned WRF boundary.

The oracle is gf-levels.csv / gf-surface.csv -- GFDRV itself, 18 cases x 6
grid spacings x 2 ishallow arms = 216 columns.  gf_gfdrv_stage runs the
driver's mixed-precision preparation, CUP_gf_sh, neg_check, cup_gf,
neg_check and the output algebra per column, one thread per column, with
the WRF-faithful k22 flag set, since the reference being graded against is
WRF.

fzu IS PINNED HERE, and that changed at 2.7.0.  Through 2.6.5 gfk_tgamma was
WOOF-ArWen's earlier gamma, which returned glibc's tgammaf words, and this gate
could compute fzu on the device.  That gamma is replaced;
WOOF-ArWen's gamma is CORRECTLY ROUNDED and glibc's is not (39.44 per
cent of [0.25, 36], worst 6 ULP), so the two no longer agree and neither
does fzu.  ``docs/gf_gamma_known_delta.md`` is the record.  This suite now
does exactly what tests/test_gf_driver_parity.py has always done: it feeds
WRF's captured up_fzu/dn_fzu/sh_fzu back through the kernel's
``fzu_override`` slots (DINS_fzu_up/dn/sh, built by
gf_field_lists.drv_scalar_inputs) so that the ~4,000 transcribed lines of
GFDRV either side of gamma stay graded bitwise against WRF.  What is graded
here is the port; what is graded against a 113-bit oracle instead of against
WRF is gamma, in tests/test_gf_gamma_correctly_rounded.py.

There is no driver residual.  Until 2026-10-07 this suite excused 8 of the
216 columns as "GFDRV's own mixed precision"; the cause was the kernel's
own spelling of the module_gfs_physcons constants as accurate doubles where
WRF computes with float64(float32(9.80665)).  gf.cu now carries WRF's words
and every column is graded bitwise against GFDRV.

The corrected-k22 half of the shallow ledger lives here too: with the
shipped default (corrected indexing) every word of every GFDRV output is
identical to the faithful run over the whole fixture, because the three
columns the correction touches are rejected under both modes.
"""
# Replaced in 2.9.1 (the WRF exact chain): test_on_those_eight_the_residual_is_amplitude_not_structure -> test_no_column_is_inexact.
# Replaced in 2.9.1 (the WRF exact chain): test_the_eight_inexact_columns_are_the_fixtures_own -> test_no_column_is_inexact.

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TOOLS = os.path.join(_ROOT, "tools", "gf_wrf461_oracle")
for _p in (_ROOT, _TOOLS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from woof.core.gf import gf_workspace_floats                 # noqa: E402
from woof.core.kernels import load_module                    # noqa: E402
from woof.verify.gf_oracle import (                          # noqa: E402
    GF_NZ, load_gf_oracle,
)
from gf_field_lists import (                                  # noqa: E402
    DRV_IN_LEV, DRV_ISCA_FIELDS, DRV_LEV_FIELDS, DRV_SCA_FIELDS,
    captured_fzu, drv_scalar_inputs,
)

NZ = GF_NZ

_WRF_LEV = ["rthcuten", "rqvcuten", "rqccuten", "rqicuten", "dudt_phy",
            "dvdt_phy", "gdc", "gdc2"]
_SEAM_LEV = ["outt", "outq", "outqc", "outu", "outv", "outts", "outqs",
             "outqcs"]


@pytest.fixture(scope="module")
def module():
    return load_module("gf")


@pytest.fixture(scope="module")
def fixture():
    return load_gf_oracle()


def _launch(module, fixture, k22_wrf_faithful, pin_fzu=True):
    gl = fixture.levels
    gs = fixture.surface
    n = fixture.ncol
    lvin = np.zeros((n, len(DRV_IN_LEV), NZ), dtype=np.float32)
    scin = drv_scalar_inputs(fixture, pin_fzu)
    iin = np.zeros((n, 3), dtype=np.int32)
    for j, name in enumerate(DRV_IN_LEV):
        lvin[:, j, :] = gl[name]
    iin[:, 0] = gs["kpbl"].astype(np.int32)
    iin[:, 1] = gs["ishallow"].astype(np.int32)
    iin[:, 2] = gs["ichoice"].astype(np.int32)
    d_lv = cp.asarray(np.ascontiguousarray(lvin))
    d_sc = cp.asarray(np.ascontiguousarray(scin))
    d_ii = cp.asarray(np.ascontiguousarray(iin))
    d_lev = cp.zeros((n, len(DRV_LEV_FIELDS), NZ), dtype=cp.float32)
    d_sca = cp.zeros((n, len(DRV_SCA_FIELDS)), dtype=cp.float32)
    d_isc = cp.zeros((n, len(DRV_ISCA_FIELDS)), dtype=cp.int32)
    fn = module.get_function("gf_gfdrv_stage")
    d_ws = cp.empty(gf_workspace_floats(NZ, n), dtype=cp.float32)
    fn(((n + 63) // 64,), (64,),
       (d_lv, d_sc, d_ii, d_lev, d_sca, d_isc, d_ws,
        np.int32(k22_wrf_faithful), np.int32(n), np.int32(NZ)))
    cp.cuda.Stream.null.synchronize()
    return dict(lev=cp.asnumpy(d_lev), sca=cp.asnumpy(d_sca),
                isc=cp.asnumpy(d_isc))


@pytest.fixture(scope="module")
def drv(module, fixture):
    return _launch(module, fixture, 1)


@pytest.fixture(scope="module")
def exact(fixture):
    return np.ones(fixture.ncol, dtype=bool)


@pytest.mark.gpu
def test_strict_driver_computes_stock_normalization_without_pins():
    """Actual scalar-build factors are computed without runtime pins."""
    from woof.wrf_exact import ENABLED
    if not ENABLED:
        pytest.skip("stock operation sequence is the strict arithmetic arm")
    from woof.core.kernels import module_source
    source = module_source("gf") + '''
extern "C" __global__ void normalization_probe(const float *a, const float *b, float *r) {
    int i=threadIdx.x; if(i<2) r[i]=gfd_pdf_normalizer(a[i],b[i]);
}
'''
    probe = cp.RawModule(code=source, options=("-std=c++17",))
    a = cp.asarray(np.array([0x406cccc7, 0x40a67a98], np.uint32).view(np.float32))
    b = cp.asarray(np.array([0x3fa66666, 0x40800000], np.uint32).view(np.float32))
    r = cp.zeros(2, cp.float32)
    probe.get_function("normalization_probe")((1,), (2,), (a,b,r))
    np.testing.assert_array_equal(cp.asnumpy(r).view(np.uint32),
                                  np.array([0x40cd2e4b, 0x439edc76], np.uint32))


def _lev(drv, name):
    return drv["lev"][:, DRV_LEV_FIELDS.index(name), :]


def _sca(drv, name):
    return drv["sca"][:, DRV_SCA_FIELDS.index(name)]


def _assert_bit_exact(got, wantw, what):
    gb = np.ascontiguousarray(got, dtype=np.float32).view(np.uint32)
    wb = np.ascontiguousarray(wantw, dtype=np.float32).view(np.uint32)
    bad = np.flatnonzero(gb.ravel() != wb.ravel())
    assert bad.size == 0, (
        f"{what}: {bad.size}/{gb.size} lanes differ; first at flat "
        f"{int(bad[0])}: got 0x{gb.ravel()[int(bad[0])]:08X} "
        f"want 0x{wb.ravel()[int(bad[0])]:08X}")


# ==========================================================================
# the tendencies GFDRV writes, on every column
# ==========================================================================
@pytest.mark.gpu
@pytest.mark.parametrize("field", _WRF_LEV)
def test_gfdrv_level_output_bitwise(drv, fixture, exact, field):
    _assert_bit_exact(_lev(drv, field)[exact], fixture.levels[field][exact],
                      f"gfdrv.{field}")


@pytest.mark.gpu
@pytest.mark.parametrize("field", ["raincv", "pratec"])
def test_gfdrv_precip_bitwise(drv, fixture, exact, field):
    _assert_bit_exact(_sca(drv, field)[exact],
                      fixture.surface[field][exact], f"gfdrv.{field}")


@pytest.mark.gpu
@pytest.mark.parametrize("field", ["htop", "hbot", "xmb_shallow"])
def test_gfdrv_xmb_free_surface_output_bitwise_everywhere(drv, fixture,
                                                          field):
    """No deep xmb in these three, so all 216 columns are exact."""
    _assert_bit_exact(_sca(drv, field), fixture.surface[field],
                      f"gfdrv.{field}")


@pytest.mark.gpu
@pytest.mark.parametrize(
    "field", ["ktop_deep", "k22_shallow", "kbcon_shallow", "ktop_shallow"])
def test_gfdrv_surface_indices_exact(drv, fixture, field):
    got = drv["isc"][:, DRV_ISCA_FIELDS.index(field)].astype(np.int64)
    want = fixture.surface[field].astype(np.int64)
    bad = np.flatnonzero(got != want)
    assert bad.size == 0, f"{field}: columns {bad[:8].tolist()}"


# ==========================================================================
# the stage seam, on ALL 216 columns
# ==========================================================================
@pytest.mark.gpu
@pytest.mark.parametrize("field", _SEAM_LEV)
def test_the_port_reproduces_the_stage_path_on_every_column(drv, fixture,
                                                            field):
    _assert_bit_exact(_lev(drv, field), fixture.stage_levels[field],
                      f"seam.{field}")


@pytest.mark.gpu
@pytest.mark.parametrize("field", ["pret", "prets"])
def test_the_stage_precipitation_seam(drv, fixture, field):
    ss = fixture.stage_surface
    idx = {}
    for i in range(ss["case"].shape[0]):
        idx[(int(ss["case"][i]), int(ss["idx"][i]), int(ss["arm"][i]))] = i
    order = [idx[tuple(int(v) for v in k)] for k in fixture.key]
    _assert_bit_exact(_sca(drv, field), ss[field][order], f"seam.{field}")


# ==========================================================================
# no inexact column (the retired 8-column residual)
# ==========================================================================
@pytest.mark.gpu
def test_no_column_is_inexact(drv, fixture):
    """Every GFDRV column bitwise.  Replaces the retired gate that accepted
    8 columns at up to 34 ULP: the cause was the kernel's accurate-double
    spelling of WRF's float32-widened GFS constants."""
    got = _lev(drv, "rthcuten")
    want = fixture.levels["rthcuten"]
    inexact = ~np.all(got.view(np.uint32) == want.view(np.uint32), axis=1)
    assert int(inexact.sum()) == 0, np.flatnonzero(inexact).tolist()


# ==========================================================================
# the gating
# ==========================================================================
@pytest.mark.gpu
def test_the_cuten_gate_and_the_shallow_switch(drv, fixture):
    pret = _sca(drv, "pret")
    cuten = _sca(drv, "cuten")
    assert np.array_equal(cuten, (pret > 0).astype(np.float32))
    arm = np.array([int(k[2]) for k in fixture.key])
    cutens = _sca(drv, "cutens")
    assert np.all(cutens[arm == 0] == 0.0)
    on = arm == 1
    xmbs = _sca(drv, "xmb_shallow")[on]
    assert np.array_equal(cutens[on], (xmbs > 0).astype(np.float32))


@pytest.mark.gpu
def test_every_column_is_covered(drv):
    assert drv["isc"].shape[0] == 216


# ==========================================================================
# the corrected-k22 ledger, driver half
# ==========================================================================
@pytest.mark.gpu
def test_the_shipped_default_is_bitwise_at_the_wrf_boundary(module, fixture,
                                                            drv):
    """The SHIPPED kernel (corrected k22, flag 0) against the faithful run:
    every output word identical over the whole fixture, because the three
    columns the corrected indexing touches are rejected under both modes
    (tests/test_gf_shallow_cuda.py::test_the_corrected_k22_ledger_entry
    holds the scheme-level half with the per-case numbers)."""
    shipped = _launch(module, fixture, 0)
    for j, name in enumerate(DRV_LEV_FIELDS):
        a = drv["lev"][:, j, :].view(np.uint32)
        b = shipped["lev"][:, j, :].view(np.uint32)
        assert np.array_equal(a, b), name
    for j, name in enumerate(DRV_SCA_FIELDS):
        a = drv["sca"][:, j].view(np.uint32)
        b = shipped["sca"][:, j].view(np.uint32)
        assert np.array_equal(a, b), name
    for j, name in enumerate(DRV_ISCA_FIELDS):
        assert np.array_equal(drv["isc"][:, j], shipped["isc"][:, j]), name


# ==========================================================================
# the gamma divergence, at the driver boundary -- the control that keeps the
# pin above accurate
# ==========================================================================
@pytest.mark.gpu
def test_the_unpinned_run_is_the_documented_divergence_not_a_regression(
        module, fixture, drv):
    """With fzu COMPUTED -- the shipped forecast path -- GFDRV's output is
    NOT bitwise WRF's, and that is the known delta, not a fault.

    This is the negative control for the pin every other test in this file
    uses.  Three things must hold and none is free.

    * The unpinned run DIFFERS somewhere.  That proves the pin is
      essential and that gamma has not quietly gone back to reproducing
      glibc's tgammaf.  If this stops firing, read
      docs/gf_gamma_known_delta.md before believing anything else here.
    * A column with NO pinned fzu cannot move.  WRF captures 0 for a profile
      whose PDF never ran and 0 means "compute", so those columns get
      identical inputs in both runs and must produce identical words.  This
      is exact, not a heuristic: MEASURED on the host harness, 24 of the 216
      columns have all three fzu at 0 and 0 of them move.
    * No integer index moves.  MEASURED, all six index fields -- ktop,
      kbcon, ktop_deep, k22_shallow, kbcon_shallow, ktop_shallow -- are
      identical on all 216 columns, so the divergence displaces the answer
      and does not flip a branch.  A branch flip would be a different and
      much larger claim than the one docs/gf_gamma_known_delta.md makes.

    The magnitude is documented rather than asserted, because it is a
    property of this fixture and not of the port.  MEASURED on the host
    harness over these 216 columns: RAINCV and PRATEC move on all 84 of
    their nonzero columns, worst 7.273 per cent, median 1.613; the shallow
    mass flux moves at worst 0.839 per cent, median 0.366; and every
    tendency field moves by at most 2.12 per cent of its own maximum.
    docs/gf_gamma_known_delta.md section 3 carries the whole table.
    """
    from woof.wrf_exact import ENABLED
    if ENABLED:
        pytest.skip("default normalization control; strict uses the scalar build sequence")
    free = _launch(module, fixture, 1, pin_fzu=False)
    up, dn, sh = captured_fzu(fixture)
    pinned = (up > 0) | (dn > 0) | (sh > 0)
    assert int(pinned.sum()) == 192, int(pinned.sum())

    moved = np.zeros(fixture.ncol, dtype=bool)
    for j in range(len(DRV_LEV_FIELDS)):
        a = drv["lev"][:, j, :].copy().view(np.uint32)
        b = free["lev"][:, j, :].copy().view(np.uint32)
        moved |= (a != b).any(axis=1)
    for j in range(len(DRV_SCA_FIELDS)):
        a = drv["sca"][:, j].copy().view(np.uint32)
        b = free["sca"][:, j].copy().view(np.uint32)
        moved |= (a != b)

    assert moved.any(), (
        "computing fzu instead of pinning it changed nothing -- either the "
        "override stopped being read or gamma is reproducing glibc again")

    stray = np.flatnonzero(moved & ~pinned)
    assert stray.size == 0, (
        f"{stray.size} columns with no pinned fzu moved: "
        f"{stray[:8].tolist()}.  Those columns get identical inputs in both "
        f"runs, so the difference is NOT the gamma delta and must be found")

    for j, name in enumerate(DRV_ISCA_FIELDS):
        assert np.array_equal(drv["isc"][:, j], free["isc"][:, j]), (
            f"{name} moved when fzu was computed -- the gamma delta flipped "
            f"a branch, which is a larger claim than the known-delta note "
            f"makes and must be re-measured before it is accepted")
