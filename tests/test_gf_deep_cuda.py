"""Device acceptance gate for the Grell-Freitas deep cloud model kernel.

Same bar as the CPU side: bitwise identity with the WRF v4.6.1 per-stage
capture over all 216 committed columns, with fzu PINNED from the oracle
exactly as tests/test_gf_deep_parity.py pins it.

WHY IT IS PINNED, AND WHAT CHANGED AT 2.7.0.  Through 2.6.5 gf.cu carried
WOOF-ArWen's earlier gamma, with its lgammaf, exp2f and expm1f helpers, which
returned glibc 2.39's tgammaf words, so the kernel's fzu WAS glibc's word
and this gate could run the chain with no override anywhere, the one claim
the CPU reference could not make.  That earlier gamma was this project's
own work, as its replacement is; it is replaced and its helpers are gone
with it, because nothing else called them.  gfk_tgamma is
now WOOF-ArWen's own CORRECTLY ROUNDED gamma: measured against a 113-bit oracle
over all 59,768,833 float32 of [0.25, 36] it is right on every one and
glibc 2.39 is wrong on 39.44 per cent of them, worst 6 ULP.  So the kernel
and WRF no longer agree on fzu, deliberately, and
``docs/gf_gamma_known_delta.md`` is the record.

A 1-ULP fzu error moves xmb by up to 7.3 per cent through the
xk = (xaa0-aa1)/mbdt cancellation, so this is not a tolerance question and
nothing here is loosened.  Pinning the oracle's own fzu words puts the
divergence back to zero at that ONE seam and leaves the other ~4,000
transcribed lines graded bitwise, which is what the CPU suite has done since
the port landed.  gamma itself is graded -- harder than before -- against the
113-bit reference in tests/test_gf_gamma_correctly_rounded.py.

Same story for powf: the CPU reference computes the correctly rounded power
and carries a measured 10-lane / 1-ULP zu divergence where glibc's powf
lands on the far side of a rounding boundary.  The kernel's gfk_pow is
glibc's, so ``zu_pdf`` is asserted bitwise on ALL lanes here -- the CPU
suite's ``test_the_powf_residual_is_glibcs_and_reaches_no_output`` documents
the same 10 lanes as a divergence.  One scheme, two implementations, and
the GPU is the one with the shorter divergence ledger.

What this file proves, in order:

1. The 123-word __constant__ scheme table survived ptxas bit for bit,
   checked against words re-derived from woof.verify.gf_deep_ref at test
   time (the ESAT/noahmp precedent: ptxas 12.x's folder mis-rounds literal
   FP32 arrays, so the table lives in __constant__ memory and this test is
   what shows the precaution worked).
2. gfk_tgamma is the CORRECTLY ROUNDED gamma on this device, graded
   against gf-crgamma-tgammaf.csv (65,638 arguments from a 113-bit
   tgammaq), and gfk_pow reproduces every powf row of the committed answer
   sheet including ppowhard's rounding-boundary case.  gfk_lgamma_pos,
   gfk_expm1 and gfk_exp2 and their gf-libm sweeps are gone: their only
   caller was the earlier gamma, so they graded dead code.
3. CUDA's builtin tgammaf and powf are DIFFERENT functions again -- neither
   glibc's nor correctly rounded -- negative controls that fire, so
   gfk_tgamma and gfk_pow are provably essential, not decoration.
4. fzu = gamma(a+b)/(gamma(a)*gamma(b)) is bitwise against the correctly
   rounded reference gf-crgamma-fzu.csv over the whole probe grid.
5. gf_deep_stage reproduces every graded field of gf-deep-levels.csv /
   gf-deep-surface.csv bit for bit -- 83 level fields, 69 float scalars, 39
   integer fields -- with the same three masks the CPU gate uses (WRF leaves
   hkb/pmin_lev/x-state lanes undefined on rejected columns).  cupclw and
   cnvwt are graded against gf-stage-levels.csv, the same cross-fixture
   seam the CPU gate checks.

FTZ note: sm_120 flushes FP32 subnormals in ALL arithmetic and CuPy appends
-ftz=true.  The bitwise gates over the full fixture are the subnormal
detector: any flushed intermediate that reaches any of the ~750k graded
words fails a max_ulp-0 assertion.  The transcriptions' double->float
conversions go through gfk_d2f_rn (the proven rrtmg_sw/noahmp
countermeasure), which the libm sweeps exercise directly.
"""

from __future__ import annotations

import csv
import os
import re
import struct
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
from woof.core.kernels import load_module, module_source     # noqa: E402
from woof.verify.gf_oracle import (                          # noqa: E402
    GF_NZ, GF_ORACLE_DIR, load_gf_oracle,
)
# The capture layout, shared with the no-GPU crosscheck
# (tools/gf_wrf461_oracle/gf_host_parity.py) so the two graders cannot
# drift; must match the enums in woof/core/kernels/gf.cu.
from gf_field_lists import (                                  # noqa: E402
    IN_LEV, IN_SCA, ISCA_FIELDS, LEV_FIELDS, SCA_FIELDS,
    captured_fzu, reference_constant_words,
)

NZ = GF_NZ


def _uh(word: str) -> np.float32:
    return np.frombuffer(struct.pack("<I", int(word, 16)),
                         dtype=np.float32)[0]


def _load_word_csv(name):
    xs, ys = [], []
    with (GF_ORACLE_DIR / name).open(encoding="ascii") as fh:
        for line in fh:
            a, b = line.strip().split(",")
            xs.append(int(a, 16))
            ys.append(int(b, 16))
    return (np.array(xs, dtype=np.uint32).view(np.float32),
            np.array(ys, dtype=np.uint32))


@pytest.fixture(scope="module")
def module():
    return load_module("gf")


@pytest.fixture(scope="module")
def fixture():
    return load_gf_oracle()


@pytest.fixture(scope="module")
def want(fixture):
    """gf-deep-surface.csv / gf-deep-levels.csv in fixture.key order."""
    order = [tuple(int(v) for v in k) for k in fixture.key]
    with (GF_ORACLE_DIR / "gf-deep-surface.csv").open(
            newline="", encoding="ascii") as fh:
        rows = list(csv.DictReader(fh))
    key = {(int(r["case"]), int(r["idx"]), int(r["arm"])): r for r in rows}
    want_s = {k: [] for k in rows[0]}
    for trip in order:
        for k, v in key[trip].items():
            want_s[k].append(v)

    with (GF_ORACLE_DIR / "gf-deep-levels.csv").open(
            newline="", encoding="ascii") as fh:
        lrows = list(csv.DictReader(fh))
    lkey = {(int(r["case"]), int(r["idx"]), int(r["arm"]), int(r["k"])): r
            for r in lrows}
    names_l = [n for n in lrows[0] if n not in ("case", "idx", "arm", "k")]
    want_l = {n: np.zeros((len(order), NZ), dtype=np.float32)
              for n in names_l}
    for ci, trip in enumerate(order):
        for k in range(1, NZ + 1):
            r = lkey[trip + (k,)]
            for n in names_l:
                want_l[n][ci, k - 1] = np.float32(r[n])
    return want_s, want_l


@pytest.fixture(scope="module")
def stage(module, fixture):
    """One launch of gf_deep_stage over all 216 columns, fzu PINNED.

    The two fzu slots carry WRF's captured up_fzu/dn_fzu.  See this module's
    docstring and docs/gf_gamma_known_delta.md: gamma is a deliberate
    divergence, and the pin is what keeps everything downstream of it graded
    at max_ulp 0 against WRF.  ``test_the_fzu_pin_was_honoured`` below proves
    the override is actually read, so these 191 field assertions cannot be
    standing on a silent no-op.
    """
    up_fzu, dn_fzu, _ = captured_fzu(fixture)
    lv = fixture.stage_levels
    sf = fixture.stage_surface
    gs = fixture.surface
    n = fixture.ncol
    lvin = np.zeros((n, len(IN_LEV), NZ), dtype=np.float32)
    for j, name in enumerate(IN_LEV):
        lvin[:, j, :] = lv[name]
    scin = np.zeros((n, len(IN_SCA)), dtype=np.float32)
    for j, name in enumerate(IN_SCA):
        if name == "dx":
            scin[:, j] = gs["dx"].astype(np.float32)
        elif name == "fzu_up":
            scin[:, j] = up_fzu       # > 0 means: use WRF's captured word
        elif name == "fzu_dn":
            scin[:, j] = dn_fzu
        else:
            scin[:, j] = sf[name].astype(np.float32)
    iin = sf["kpbli"].astype(np.int32)

    d_lv = cp.asarray(np.ascontiguousarray(lvin))
    d_sc = cp.asarray(np.ascontiguousarray(scin))
    d_ii = cp.asarray(np.ascontiguousarray(iin))
    d_lev = cp.zeros((n, len(LEV_FIELDS), NZ), dtype=cp.float32)
    d_sca = cp.zeros((n, len(SCA_FIELDS)), dtype=cp.float32)
    d_isc = cp.zeros((n, len(ISCA_FIELDS)), dtype=cp.int32)

    fn = module.get_function("gf_deep_stage")
    d_ws = cp.empty(gf_workspace_floats(NZ, n), dtype=cp.float32)
    fn(((n + 63) // 64,), (64,),
       (d_lv, d_sc, d_ii, d_lev, d_sca, d_isc, d_ws,
        np.int32(n), np.int32(NZ)))
    cp.cuda.Stream.null.synchronize()
    return dict(lev=cp.asnumpy(d_lev), sca=cp.asnumpy(d_sca),
                isc=cp.asnumpy(d_isc))


def _ws(want, name):
    return np.array([np.float32(v) for v in want[0][name]], dtype=np.float32)


def _wi(want, name):
    return np.array([int(v) for v in want[0][name]], dtype=np.int64)


def _ierr6_mask(want):
    return _wi(want, "ierr_6") == 0


def _pmin_mask(want):
    ok = _wi(want, "ierr_1") == 0
    shut = (_ws(want, "frh_kb") >= np.float32(0.97)) & (
        _ws(want, "sig") <= _ws(want, "sig_thresh"))
    return ok & ~shut


def _assert_bit_exact(got, wantw, what):
    gb = np.ascontiguousarray(got, dtype=np.float32).view(np.uint32)
    wb = np.ascontiguousarray(wantw, dtype=np.float32).view(np.uint32)
    bad = np.flatnonzero(gb.ravel() != wb.ravel())
    assert bad.size == 0, (
        f"{what}: {bad.size}/{gb.size} lanes differ; first at flat index "
        f"{int(bad[0])}: got 0x{gb.ravel()[int(bad[0])]:08X} "
        f"want 0x{wb.ravel()[int(bad[0])]:08X}")


# ==========================================================================
# 1. the constant table survived ptxas
# ==========================================================================
def _reference_constant_words():
    return reference_constant_words()


def test_constant_table_source_matches_the_reference():
    """The GFC words in the .cu source ARE the CPU reference's constants.

    Source-level, no GPU needed: a retyped word cannot hide.  The device
    half of the same claim is the next test.
    """
    want = _reference_constant_words()
    src = module_source("gf")
    m = re.search(r"__constant__ unsigned int GFC\[GF_NCONST\] = \{(.*?)\};",
                  src, re.S)
    assert m is not None
    got = np.array([int(w, 16) for w in
                    re.findall(r"0x([0-9A-Fa-f]{8})u", m.group(1))],
                   dtype=np.uint32)
    assert got.size == want.size
    bad = np.flatnonzero(got != want)
    assert bad.size == 0, f"GFC[{bad.tolist()}] wrong in source"


@pytest.mark.gpu
def test_constant_table_survived_ptxas(module):
    want = _reference_constant_words()
    n = want.size
    d_out = cp.zeros(n, dtype=cp.uint32)
    fn = module.get_function("gf_deep_const_dump")
    fn(((n + 63) // 64,), (64,), (d_out,))
    got = cp.asnumpy(d_out)
    bad = np.flatnonzero(got != want)
    assert bad.size == 0, (
        f"GFC[{bad.tolist()}]: device words "
        f"{[hex(int(got[i])) for i in bad[:8]]} vs "
        f"{[hex(int(want[i])) for i in bad[:8]]}")


# ==========================================================================
# 2. the libm surface: gamma against the 113-bit reference, powf against the
#    committed answer sheet
# ==========================================================================
#: gf_libm_unary_probe is 4 slots since 2.7.0 -- gfk_tgamma, CUDA's builtin
#: tgammaf (negative control), gfk_exp, gfk_log.  Slots 2/3/4 held
#: gfk_lgamma_pos / gfk_expm1 / gfk_exp2, whose only caller was the earlier
#: gamma; all four are deleted.
_UNARY_SLOTS = 4


def _run_unary(module, x):
    n = x.size
    d_x = cp.asarray(np.ascontiguousarray(x))
    d_out = cp.zeros(_UNARY_SLOTS * n, dtype=cp.float32)
    fn = module.get_function("gf_libm_unary_probe")
    fn(((n + 255) // 256,), (256,), (d_x, d_out, np.int32(n)))
    return cp.asnumpy(d_out).reshape(n, _UNARY_SLOTS)


@pytest.mark.gpu
def test_the_device_gamma_is_correct_and_the_builtin_is_not(module):
    """The one-line precondition for the 191 field assertions below.

    Everything downstream of fzu is graded against WRF with fzu PINNED, so
    this file no longer proves anything about gamma by proving those.  It is
    proved directly instead, and harder than the old
    ``test_device_libm_matches_live_glibc[tgammaf]`` did: slot 0 must equal
    the CORRECTLY ROUNDED answer over all 65,638 sweep arguments, not merely
    equal what one binary returns.  glibc 2.39 fails that check on 25,713 of
    these same rows (tests/test_gf_gamma_correctly_rounded.py measures it).

    The second half is the negative control, and it must FIRE: CUDA's
    builtin tgammaf is a third function again, so if it ever agrees with the
    reference everywhere the control is stale -- re-measure before believing
    anything else in this file, because the 7.3 per cent xmb stake means a
    silent gamma change is a silent physics change.
    """
    x, wantw = _load_word_csv("gf-crgamma-tgammaf.csv")
    out = _run_unary(module, x)
    gb = out[:, 0].copy().view(np.uint32)
    bad = np.flatnonzero(gb != wantw)
    assert bad.size == 0, (
        f"gfk_tgamma: {bad.size}/{x.size} words differ from the correctly "
        f"rounded reference; first x=0x{int(x.view(np.uint32)[bad[0]]):08X} "
        f"got=0x{int(gb[bad[0]]):08X} want=0x{int(wantw[bad[0]]):08X}")
    ndiff = int(np.count_nonzero(out[:, 1].copy().view(np.uint32) != wantw))
    assert ndiff > 0, (
        "CUDA's builtin tgammaf matched the correctly rounded reference on "
        "every sweep argument -- the negative control no longer fires; "
        "re-measure before trusting either function")


def _pow_rows():
    xs, ys, wants = [], [], []
    with (GF_ORACLE_DIR / "gf-pow-probe.txt").open(encoding="ascii") as fh:
        for line in fh:
            p = line.split()
            if line.startswith("pbeta "):
                kr, al, be, ka, kb_ = p[1], p[2], p[3], p[4], p[5]
                kw = _uh(kr)
                xs.append(kw)
                ys.append(np.float32(_uh(al) - np.float32(1.0)))
                wants.append(int(ka, 16))
                xs.append(np.float32(np.float32(1.0) - kw))
                ys.append(np.float32(_uh(be) - np.float32(1.0)))
                wants.append(int(kb_, 16))
            elif line.startswith("ppowhard "):
                xs.append(_uh(p[1])); ys.append(_uh(p[2]))
                wants.append(int(p[3], 16))
            elif line.startswith("p3333 "):
                xs.append(_uh(p[1])); ys.append(_uh(p[5]))
                wants.append(int(p[2], 16))
    return (np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32),
            np.array(wants, dtype=np.uint32))


@pytest.mark.gpu
def test_device_powf_matches_the_answer_sheet(module):
    """Every powf row of the committed sheet, including ppowhard's
    rounding-boundary case -- the one that separates glibc's powf from the
    correctly rounded answer and cost the CPU reference 10 zu lanes."""
    x, y, wantw = _pow_rows()
    n = x.size
    d_x = cp.asarray(x)
    d_y = cp.asarray(y)
    d_out = cp.zeros(2 * n, dtype=cp.float32)
    fn = module.get_function("gf_libm_pow_probe")
    fn(((n + 255) // 256,), (256,), (d_x, d_y, d_out, np.int32(n)))
    got = cp.asnumpy(d_out).reshape(n, 2)
    gb = got[:, 0].copy().view(np.uint32)
    bad = np.flatnonzero(gb != wantw)
    assert bad.size == 0, (
        f"gfk_pow: {bad.size}/{n} sheet rows differ; first "
        f"x=0x{int(x.view(np.uint32)[bad[0]]):08X} "
        f"y=0x{int(y.view(np.uint32)[bad[0]]):08X} "
        f"got=0x{int(gb[bad[0]]):08X} want=0x{int(wantw[bad[0]]):08X}")
    # the builtin is a different function -- the control must fire here too
    ndiff = int(np.count_nonzero(got[:, 1].copy().view(np.uint32) != wantw))
    assert ndiff > 0, "CUDA builtin powf negative control no longer fires"


@pytest.mark.gpu
def test_fzu_bitwise_on_the_pgamma_grid(module):
    """fzu = gamma(a+b)/(gamma(a)*gamma(b)) against the CORRECTLY ROUNDED
    reference over the whole probe grid.

    The oracle used to be the ``pgamma`` rows of gf-pow-probe.txt, i.e.
    WRF's own captured fzu.  It is gf-crgamma-fzu.csv since 2.7.0: the same
    (alpha, beta) pairs and 26 more, recomputed from a 113-bit gamma with one
    float32 rounding per operation exactly as get_zu_zd_pdf_fim spells it.
    MEASURED: on the 100 pairs the two fixtures share, this reference differs
    from WRF's captured fzu on 68 (68.0 per cent), worst 5 ULP; against fzu
    composed from glibc's own tgammaf it differs on 89 of all 126 rows (70.6
    per cent), worst 5.  That IS the known delta --
    docs/gf_gamma_known_delta.md.  Grading against WRF's words here would be
    grading against a gamma that is wrong on 39 per cent of its domain.
    """
    a_l, b_l, w_l = [], [], []
    with (GF_ORACLE_DIR / "gf-crgamma-fzu.csv").open(encoding="ascii") as fh:
        for line in fh:
            aw, bw, fw = line.strip().split(",")
            a_l.append(_uh(aw))
            b_l.append(_uh(bw))
            w_l.append(int(fw, 16))
    a = np.array(a_l, dtype=np.float32)
    b = np.array(b_l, dtype=np.float32)
    wantw = np.array(w_l, dtype=np.uint32)
    n = a.size
    d_a = cp.asarray(a)
    d_b = cp.asarray(b)
    d_out = cp.zeros(n, dtype=cp.float32)
    fn = module.get_function("gf_fzu_probe")
    fn(((n + 255) // 256,), (256,), (d_a, d_b, d_out, np.int32(n)))
    got = cp.asnumpy(d_out).view(np.uint32)
    bad = np.flatnonzero(got != wantw)
    assert bad.size == 0, (
        f"fzu: {bad.size}/{n} correctly-rounded rows differ; first alpha="
        f"0x{int(a.view(np.uint32)[bad[0]]):08X}")


# ==========================================================================
# 5. the deep scheme, bitwise, fzu pinned
# ==========================================================================
@pytest.mark.gpu
@pytest.mark.parametrize("field", LEV_FIELDS)
def test_deep_level_field_bitwise(stage, fixture, want, field):
    j = LEV_FIELDS.index(field)
    got = stage["lev"][:, j, :]
    if field in ("cupclw", "cnvwt"):
        wantf = fixture.stage_levels[field]
    else:
        wantf = want[1][field]
    if field in ("xhe", "xq", "xt", "xqes", "xhes", "dtempdz"):
        m = _ierr6_mask(want)
        got, wantf = got[m], wantf[m]
    _assert_bit_exact(got, wantf, f"gf_deep_stage.{field}")


@pytest.mark.gpu
@pytest.mark.parametrize("field", SCA_FIELDS)
def test_deep_scalar_bitwise(stage, want, field):
    j = SCA_FIELDS.index(field)
    got = stage["sca"][:, j]
    wantf = _ws(want, field)
    if field in ("hkb0", "hkbo0"):
        m = _wi(want, "k22_0") > 0
        got, wantf = got[m], wantf[m]
    elif field in ("xhkb", "pr7"):
        m = _ierr6_mask(want)
        got, wantf = got[m], wantf[m]
    _assert_bit_exact(got, wantf, f"gf_deep_stage.{field}")


@pytest.mark.gpu
@pytest.mark.parametrize("field", ISCA_FIELDS)
def test_deep_int_field_exact(stage, want, field):
    j = ISCA_FIELDS.index(field)
    got = stage["isc"][:, j].astype(np.int64)
    wantf = _wi(want, field)
    if field in ("pmin_lev", "start_level"):
        m = _pmin_mask(want)
        got, wantf = got[m], wantf[m]
    bad = np.flatnonzero(got != wantf)
    assert bad.size == 0, (
        f"{field}: differs on {bad.size} columns, first col {int(bad[0])} "
        f"got {int(got[bad[0]])} want {int(wantf[bad[0]])}")


@pytest.mark.gpu
def test_the_fzu_pin_was_honoured(stage, want, fixture):
    """The override is READ, not ignored -- so the 191 assertions above are
    not standing on a silent no-op.

    This replaces ``test_fzu_needed_no_pin``, which asserted the opposite
    claim -- that with fzu COMPUTED on the device, up_fzu and dn_fzu were
    bitwise WRF's on every column.  That claim was true only because the
    earlier gfk_tgamma returned glibc's tgammaf words.  It is
    false now, on purpose, and docs/gf_gamma_known_delta.md is the record.

    What is asserted instead is that the kernel reports back exactly the
    words it was handed on every column that HAS a pin (WRF captures 0 for a
    column whose PDF never ran, and 0 means "compute" -- those columns are
    excluded here and carry no mass flux to any graded output anyway).  If
    the override were ever dropped from the kernel or the fixture, this
    fails loudly instead of the 191 field tests failing obscurely.
    """
    up, dn, _ = captured_fzu(fixture)
    for name, pin in (("up_fzu", up), ("dn_fzu", dn)):
        j = SCA_FIELDS.index(name)
        m = pin > 0
        assert int(m.sum()) > 0, f"no column pins {name}"
        _assert_bit_exact(stage["sca"][:, j][m], pin[m], f"{name} (pinned)")
        # and the pin IS WRF's own word, not something re-derived here
        _assert_bit_exact(pin[m], _ws(want, name)[m], f"{name} (capture)")


@pytest.mark.gpu
def test_every_column_is_covered(stage, want):
    assert stage["isc"].shape[0] == 216
    ierr = _wi(want, "ierr")
    assert int((ierr == 0).sum()) == 60
    got_ierr = stage["isc"][:, ISCA_FIELDS.index("ierr")]
    assert int((got_ierr == 0).sum()) == 60


@pytest.mark.gpu
def test_the_inversion_clamp_is_documented_not_exercised(stage):
    """WRF's t_cup(kend+8) out-of-bounds read: the kernel clamps kend to
    ktf-8 exactly as the CPU reference and the oracle capture do, and the
    clamp count on the committed fixture is zero -- the divergence stays a
    ledger entry, not a live branch."""
    j = ISCA_FIELDS.index("kinv_clamped")
    assert int(stage["isc"][:, j].sum()) == 0
