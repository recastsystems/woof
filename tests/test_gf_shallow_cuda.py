"""Device acceptance gate for the Grell-Freitas shallow scheme kernel.

Same bar as tests/test_gf_shallow_parity.py: bitwise identity with the WRF
v4.6.1 per-case capture (gf-shallow-levels.csv / gf-shallow-surface.csv, 18
cases) with fzu PINNED from that capture, exactly as the CPU suite pins it.

WHY IT IS PINNED, AND WHAT CHANGED AT 2.7.0.  Through 2.6.5 gfk_tgamma was
WOOF-ArWen's earlier gamma, which returned glibc 2.39's tgammaf words, so the
kernel and this gate could run the chain unpinned, the notch past the CPU
suite.  That gamma is replaced, and gfk_tgamma is now WOOF-ArWen's own
CORRECTLY ROUNDED gamma,
which glibc's is not on 39.44 per cent of [0.25, 36].  The shallow arm is
the worst-exposed one: beta = 2.5 makes all three gamma calls live, and the
xkshal = (xaa0-aa1)/mbdt cancellation turns a last-bit fzu change into
8.4e-3 relative in the shallow mass flux.  So fzu is pinned here, the
divergence is recorded in docs/gf_gamma_known_delta.md, and gamma itself is
graded against a 113-bit oracle in
tests/test_gf_gamma_correctly_rounded.py.

OWNER RULING, and the ledger entry this file carries
----------------------------------------------------
WRF's k22 trigger (module_cu_gf_sh.F:373) is
``k22(i) = maxloc(HEO_CUP(i,2:kbmax(i)), 1)`` -- a MAXLOC over an array
SECTION whose result WRF uses as an absolute level index without adding
the section offset, leaving k22 one level BELOW the argmax wherever the
argmax sits above level 2.  The kernel's SHIPPED default is the CORRECTED
indexing (k22 IS the argmax); the WRF-faithful off-by-one lives behind the
``k22_wrf_faithful`` launch argument, which this parity suite sets to 1.

The measured delta between the two modes over the committed fixture
(test_the_corrected_k22_ledger_entry):

* k22_0 moves on exactly 3 of 18 cases -- case 6 (15 -> 16), case 13
  (8 -> 9, the witness the CPU suite pins), case 16 (9 -> 10);
* all three columns are REJECTED under both modes with identical ierr
  (3, 5, 3), so ZERO tendency words differ at the scheme boundary and
  zero words at the GFDRV boundary (tests/test_gf_gfdrv_cuda.py holds the
  driver half of the same claim);
* every converged column has its heo_cup argmax at level 2, where the two
  indexings coincide through the max(2, k22) floor.

So on this fixture the correction is diagnostically visible and
numerically free; the flag exists for the soundings where it would not be.
"""

from __future__ import annotations

import csv
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
    GF_NCASE, GF_NZ, GF_ORACLE_DIR, load_gf_oracle,
)
from gf_field_lists import (                                  # noqa: E402
    SH_IN_LEV, SH_IN_SCA, SH_ISCA_FIELDS, SH_LEV_FIELDS, SH_SCA_FIELDS,
    captured_fzu,
)

NZ = GF_NZ


@pytest.fixture(scope="module")
def module():
    return load_module("gf")


@pytest.fixture(scope="module")
def fixture():
    return load_gf_oracle()


@pytest.fixture(scope="module")
def want(fixture):
    with (GF_ORACLE_DIR / "gf-shallow-surface.csv").open(
            newline="", encoding="ascii") as fh:
        rows = list(csv.DictReader(fh))
    want_s = {k: [None] * GF_NCASE for k in rows[0]}
    for r in rows:
        for k, v in r.items():
            want_s[k][int(r["case"]) - 1] = v
    return want_s, fixture.shallow_levels


def _launch(module, fixture, k22_wrf_faithful, pin_fzu=True):
    lv = fixture.stage_levels
    sf = fixture.stage_surface
    _, _, fzu_sh = captured_fzu(fixture)
    col_of_case = {}
    for ci, (case, idx, arm) in enumerate(fixture.key):
        col_of_case.setdefault(int(case), ci)
    n = GF_NCASE
    lvin = np.zeros((n, len(SH_IN_LEV), NZ), dtype=np.float32)
    scin = np.zeros((n, len(SH_IN_SCA)), dtype=np.float32)
    iin = np.zeros(n, dtype=np.int32)
    for case in range(1, n + 1):
        ci = col_of_case[case]
        for j, name in enumerate(SH_IN_LEV):
            lvin[case - 1, j, :] = lv[name][ci]
        for j, name in enumerate(SH_IN_SCA):
            if name == "fzu_sh":
                scin[case - 1, j] = fzu_sh[ci] if pin_fzu else 0.0
            else:
                scin[case - 1, j] = sf[name][ci]
        iin[case - 1] = int(sf["kpbli"][ci])
    d_lv = cp.asarray(np.ascontiguousarray(lvin))
    d_sc = cp.asarray(np.ascontiguousarray(scin))
    d_ii = cp.asarray(np.ascontiguousarray(iin))
    d_lev = cp.zeros((n, len(SH_LEV_FIELDS), NZ), dtype=cp.float32)
    d_sca = cp.zeros((n, len(SH_SCA_FIELDS)), dtype=cp.float32)
    d_isc = cp.zeros((n, len(SH_ISCA_FIELDS)), dtype=cp.int32)
    fn = module.get_function("gf_shallow_stage")
    d_ws = cp.empty(gf_workspace_floats(NZ, n), dtype=cp.float32)
    fn(((n + 63) // 64,), (64,),
       (d_lv, d_sc, d_ii, d_lev, d_sca, d_isc, d_ws,
        np.int32(k22_wrf_faithful), np.int32(n), np.int32(NZ)))
    cp.cuda.Stream.null.synchronize()
    return dict(lev=cp.asnumpy(d_lev), sca=cp.asnumpy(d_sca),
                isc=cp.asnumpy(d_isc))


@pytest.fixture(scope="module")
def stage(module, fixture):
    """WRF-faithful k22, fzu pinned from the WRF capture."""
    return _launch(module, fixture, 1)


def _assert_bit_exact(got, wantw, what):
    gb = np.ascontiguousarray(got, dtype=np.float32).view(np.uint32)
    wb = np.ascontiguousarray(wantw, dtype=np.float32).view(np.uint32)
    bad = np.flatnonzero(gb.ravel() != wb.ravel())
    assert bad.size == 0, (
        f"{what}: {bad.size}/{gb.size} lanes differ; first at flat "
        f"{int(bad[0])}: got 0x{gb.ravel()[int(bad[0])]:08X} "
        f"want 0x{wb.ravel()[int(bad[0])]:08X}")


@pytest.mark.gpu
@pytest.mark.parametrize("field", SH_LEV_FIELDS)
def test_shallow_level_field_bitwise(stage, want, field):
    j = SH_LEV_FIELDS.index(field)
    _assert_bit_exact(stage["lev"][:, j, :], want[1][field],
                      f"gf_shallow_stage.{field}")


@pytest.mark.gpu
@pytest.mark.parametrize("field", SH_SCA_FIELDS)
def test_shallow_scalar_bitwise(stage, want, field):
    j = SH_SCA_FIELDS.index(field)
    wantf = np.array([np.float32(v) for v in want[0][field]],
                     dtype=np.float32)
    _assert_bit_exact(stage["sca"][:, j], wantf, f"gf_shallow_stage.{field}")


@pytest.mark.gpu
@pytest.mark.parametrize("field", SH_ISCA_FIELDS)
def test_shallow_int_field_exact(stage, want, field):
    j = SH_ISCA_FIELDS.index(field)
    got = stage["isc"][:, j].astype(np.int64)
    wantf = np.array([int(v) for v in want[0][field]], dtype=np.int64)
    bad = np.flatnonzero(got != wantf)
    assert bad.size == 0, (
        f"{field}: differs on cases {(bad + 1).tolist()}, got "
        f"{got[bad].tolist()} want {wantf[bad].tolist()}")


@pytest.mark.gpu
def test_the_sh_fzu_pin_was_honoured(stage, want):
    """The override is READ, so every other assertion here is not standing
    on a silent no-op.

    This replaces ``test_sh_fzu_needed_no_pin``, which asserted that the
    device COMPUTED glibc's own sh_fzu with no override.  That was true only
    while the earlier gfk_tgamma returned glibc's tgammaf words; it is
    false now, deliberately, and docs/gf_gamma_known_delta.md is the record.
    What is asserted instead is that the kernel reports back the word it was
    handed, which is what makes the 18-case bitwise result above mean
    something."""
    j = SH_SCA_FIELDS.index("sh_fzu")
    wantf = np.array([np.float32(v) for v in want[0]["sh_fzu"]],
                     dtype=np.float32)
    _assert_bit_exact(stage["sca"][:, j], wantf, "sh_fzu")


@pytest.mark.gpu
def test_the_unpinned_shallow_fzu_is_the_documented_divergence(module,
                                                               fixture,
                                                               want):
    """Negative control for the pin: computing sh_fzu instead of pinning it
    moves it away from WRF, which is the known delta, not a fault.

    If this ever stops firing, gamma has gone back to reproducing glibc's
    tgammaf and has silently lost its correct rounding; read
    docs/gf_gamma_known_delta.md before believing it.  The bound is asserted
    too: the shallow arm's alpha and beta stay inside the interval gamma is
    proven correctly rounded over, so no case may move by more than the
    4-ULP budget the CPU suite already carries for fzu."""
    free = _launch(module, fixture, 1, pin_fzu=False)
    j = SH_SCA_FIELDS.index("sh_fzu")
    got = free["sca"][:, j].copy().view(np.uint32)
    wantf = np.array([np.float32(v) for v in want[0]["sh_fzu"]],
                     dtype=np.float32)
    wantw = wantf.copy().view(np.uint32)
    live = wantf != np.float32(0.0)     # 0 means the PDF never ran on that case
    assert int(live.sum()) == 16, int(live.sum())
    d = np.abs(got[live].astype(np.int64) - wantw[live].astype(np.int64))
    # MEASURED on the host harness (tools/gf_wrf461_oracle/gf_host_harness.cpp
    # built as C++, no GPU): 12 of the 16 live cases move, worst 4 ULP, and
    # the shallow mass flux moves at worst 0.839 per cent, median 0.366.
    assert int((d != 0).sum()) == 12, int((d != 0).sum())
    assert int(d.max()) <= 4, (
        f"sh_fzu moved by {int(d.max())} ULP, past the 4-ULP budget "
        f"tests/test_gf_shallow_parity.py carries; re-measure before "
        f"accepting it")


@pytest.mark.gpu
def test_shallow_coverage(stage):
    """5 converged, ierr 3/5/21 exercised -- the fixture's own census."""
    j = SH_ISCA_FIELDS.index("ierr")
    ierr = stage["isc"][:, j]
    counts = {int(v): int((ierr == v).sum()) for v in np.unique(ierr)}
    assert counts == {0: 5, 3: 2, 5: 9, 21: 2}


@pytest.mark.gpu
def test_the_corrected_k22_ledger_entry(module, fixture, stage):
    """The divergence-ledger entry for the owner's k22 ruling, MEASURED.

    Shipped default (k22_wrf_faithful = 0) against the parity mode this
    suite grades, over the whole committed fixture.  See the module
    docstring for the three cases and why the delta is zero words on every
    output.  If a future fixture makes the corrected mode change a
    CONVERGED column, this test fails and the ledger entry gets a new
    number -- that is a finding, not a regression.
    """
    corrected = _launch(module, fixture, 0)
    jk = SH_ISCA_FIELDS.index
    f_isc = stage["isc"]
    c_isc = corrected["isc"]

    moved = np.flatnonzero(f_isc[:, jk("k22_0")] != c_isc[:, jk("k22_0")])
    assert (moved + 1).tolist() == [6, 13, 16]
    assert f_isc[moved, jk("k22_0")].tolist() == [15, 8, 9]
    assert c_isc[moved, jk("k22_0")].tolist() == [16, 9, 10]
    # case 13 is the CPU suite's witness for the faithful reading
    assert int(f_isc[12, jk("k22_0")]) == 8

    # every moved case is rejected under BOTH modes, with the same reason
    assert np.array_equal(f_isc[:, jk("ierr")], c_isc[:, jk("ierr")])
    assert f_isc[moved, jk("ierr")].tolist() == [3, 5, 3]

    # and the correction reaches NO output word on this fixture
    for name in ("outt", "outq", "outqc", "zuo"):
        j = SH_LEV_FIELDS.index(name)
        a = stage["lev"][:, j, :].view(np.uint32)
        b = corrected["lev"][:, j, :].view(np.uint32)
        assert np.array_equal(a, b), name
    for name in ("xmb", "xmb_out", "pre"):
        j = SH_SCA_FIELDS.index(name)
        a = stage["sca"][:, j].view(np.uint32)
        b = corrected["sca"][:, j].view(np.uint32)
        assert np.array_equal(a, b), name

    # the converged columns coincide because their argmax is at level 2,
    # where max(2, k22) folds the two indexings together -- assert the
    # reason, not just the outcome
    ok = f_isc[:, jk("ierr")] == 0
    assert np.array_equal(f_isc[ok, jk("k22_0")],
                          c_isc[ok, jk("k22_0")])
    assert np.all(f_isc[ok, jk("k22_0")] == 2)
