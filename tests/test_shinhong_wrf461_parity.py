"""Shin-Hong against the byte-unmodified WRF v4.6.1 module, for the first time.

``bl_pbl_physics=11`` is the gray-zone PBL, and until this file existed no WRF
number had ever been produced for it in this repository.
``tools/shinhong_wrf461_oracle/`` drives the pinned
``phys/module_bl_shinhong.F`` (sha256 ``99f44dbe...``) through its own 3-D
``shinhong`` entry on gfortran 15.2.0 -O0 / glibc 2.43 and dumps every input
and output; ``woof.verify.shinhong_ref.np_shinhong_column`` is the float32
NumPy authority, and this module measures the authority against the dump and
pins what it measures.

**The numbers below are a measurement, not a target.**  They record what the
authority actually does, asserted for *equality*, not as an upper bound: an
authority that got better silently is as much a drift as one that got worse.
Every number was produced by
``python tools/shinhong_wrf461_oracle/validate_shinhong_oracle.py`` on
2026-08-03 (Windows 11 x64, CPython 3.13.7, numpy 2.2.6) against the packaged
fixture, and the result is *bitwise* -- max ULP 0 and zero differing bit
patterns, signed zeros included -- on every field of both oracle arms over
every lane WRF's arithmetic defines.  The float64-round-once surrogates for
glibc's correctly rounded ``powf``/``expf``/``tanhf`` never cost a ULP on this
fixture (``pow-probe.txt`` pins 26/26 pow forms bitwise).

Two lane families are excluded from that mask, each asserted separately in
its own test below because it is a documented divergence, not arithmetic:

* case 13's NaN lanes (``rthblten`` all 40 levels, ``exch_h`` k=2,
  ``tke_out`` k>=2, x6 dx).  WRF's own ``prfac2`` 0/0
  (module_bl_shinhong.F:1010: ``15.9*wstar3/ust3`` with both cubes exactly
  zero) writes the column; the authority reproduces the identical NaN
  *geometry*, but the fixture's ES24.16E3 format wrote the sign-less token
  ``NaN``, so the CSV cannot carry the payload or sign (gfortran x86 0/0 is
  0xFFC00000; the parsed text is 0x7FC00000) and bit identity is unknowable
  from the receipt.  Geometry and counts are pinned instead.
The first cut of the fixture accidentally ran case 22 (``kpbl == kte``)
with ``tke_diag = 1``, which makes WRF read ``q2xk(kte+1)`` one past the end
of a ``kts:kte`` array at :1557, and its ``tke_out`` therefore embedded
stack garbage.  The fixture was regenerated with the documented
``tke_diag = 0`` for case 22 (the generator fix is in the same lane), so
that arm is now deliberately fixture-less -- see the oracle README's "Not
covered, deliberately".  The authority still guards the read (``kpbl < kte``
leaves ``efxpbl`` at 0.0, the defined behaviour);
``test_the_oob_tke_arm_is_guarded_and_fixture_less`` pins both the guard and
the reason there is no reference to compare it to.
"""

from __future__ import annotations

import functools
import hashlib

import numpy as np

from woof.certify.compile_platform import (
    compile_platform_fingerprint,
    nvrtc_build,
)
from woof.core.fp32_ulp import fp32_ulp_distance
from woof.verify import shinhong_ref
from woof.verify.shinhong_oracle import (
    ARM_A,
    ARM_B,
    LEVEL_FIELD_MAP,
    SHINHONG_ORACLE_DIR,
    SURFACE_FIELD_MAP,
    TOPO_LEVEL_FIELD_MAP,
    TOPO_SURFACE_FIELD_MAP,
    load_shinhong_oracle,
    shinhong_ref_outputs,
)
from woof.verify.shinhong_ref import np_pq, np_pthl, np_pthnl, np_ptke, np_pu

#: The case whose NaN lanes are geometry-pinned rather than bit-compared
#: (WRF's own 0/0; the CSV text cannot carry NaN bits).
NAN_CASE = 13

#: The kpbl == kte case.  Its tke_diag=1 arm is deliberately fixture-less
#: (WRF's :1557 reads q2xk(kte+1) there); the fixture runs it with
#: tke_diag=0 and the authority's guarded arm is pinned separately.
GUARDED_OOB_CASE = 22

#: Worst ULP distance from ``np_shinhong_column`` to the word the pinned WRF
#: module wrote, over the arithmetic mask (only case 13's NaN lanes out,
#: geometry-pinned in their own test).  All zeros, and asserted bitwise on
#: top (zero differing bit patterns -- ``max_ulp == 0`` alone would still
#: pass -0.0 against +0.0).  Measured 2026-08-03 against the regenerated
#: fixture (case 22 at its documented tke_diag=0),
#: validate_shinhong_oracle.py.
BASELINE_MAX_ULP = {
    "rublten": 0,
    "rvblten": 0,
    "rthblten": 0,
    "rqvblten": 0,
    "rqcblten": 0,
    "rqiblten": 0,
    "exch_h": 0,
    "tke": 0,
    "el": 0,
    "hpbl": 0,
    "wstar": 0,
    "delta": 0,
    "u10": 0,
    "v10": 0,
}

#: Same measurement for arm B (ctopo=0.85, ctopo2=0.7) against the _tw
#: columns.  Also all zeros, also asserted bitwise.
TOPO_BASELINE_MAX_ULP = {
    "rublten": 0,
    "rvblten": 0,
    "exch_h": 0,
    "hpbl": 0,
    "wstar": 0,
    "delta": 0,
    "u10": 0,
    "v10": 0,
}

#: NaN lanes per arm-A field, all of them case 13 (6 dx columns): rthblten
#: is the whole heat column (6*40), exch_h is k=2 only (xkzh(1) is the one
#: NaN coefficient; kpbl there is 2), tke_out is k>=2 (6*39).  Counted from
#: the fixture AND required of the authority, so neither the 0/0 nor the
#: fixture's ability to reach it can quietly disappear.
NAN_LANES = {"rthblten": 240, "exch_h": 6, "tke": 234}
TOPO_NAN_LANES = {"exch_h": 6}

#: WRF against WRF, no port involved: the ctopo/ctopo2 gap between the two
#: oracle arms.  ctopo enters exactly one word -- the k=1 momentum diagonal
#: (:1387) -- and ctopo2 exactly the u10/v10 blend (:1472-1473), so the two
#: arms are required to be *bitwise identical* everywhere else (exch_h_tw,
#: hpbl_tw, wstar_tw, delta_tw pin that), while the momentum tendencies and
#: the blend differ by these measured maxima (the tendencies cross zero
#: between the arms, hence the enormous ULP distances -- they are tiny
#: absolute numbers of opposite sign).  Measured 2026-08-03.
WRF_TOPO_GAP_MAX_ULP = {
    "rublten": 1899259495,
    "rvblten": 1809108173,
    "u10_out": 1063706951,
    "v10_out": 1058623456,
}


@functools.cache
def _fixture():
    return load_shinhong_oracle()


@functools.cache
def _port(arm_name: str):
    return shinhong_ref_outputs(_fixture(),
                                arm=ARM_A if arm_name == "A" else ARM_B)


def _bit_diff(got, want):
    return got.view(np.uint32) != want.view(np.uint32)


def _measure_field(got, want):
    """(max_ulp, differing-bit-count) over the lanes where neither is NaN."""
    nan = np.isnan(got) | np.isnan(want)
    distance = np.where(~nan, fp32_ulp_distance(got, want), np.int64(0))
    differing = int((_bit_diff(got, want) & ~nan).sum())
    return int(distance.max()) if distance.size else 0, differing


# --------------------------------------------------------------------------
# The fixture is what it says it is.
# --------------------------------------------------------------------------

def test_fixture_is_the_pinned_wrf_module_and_its_own_receipts():
    """The CSVs must hash to what build.sh recorded next to them."""
    receipts = (SHINHONG_ORACLE_DIR / "oracle-sha256sums.txt").read_text(
        encoding="ascii").splitlines()
    recorded = {}
    for line in receipts:
        digest, path = line.split(maxsplit=1)
        recorded[path.strip().rsplit("/", 1)[-1]] = digest
    for name in ("shinhong-levels.csv", "shinhong-surface.csv",
                 "shinhong-partition.csv"):
        blob = (SHINHONG_ORACLE_DIR / name).read_bytes()
        assert hashlib.sha256(blob).hexdigest() == recorded[name], name
    # The WRF source the fixture came from is named in the same receipt, so a
    # fixture regenerated from a different tree cannot pass silently.
    assert "module_bl_shinhong.F" in " ".join(receipts)


def test_libmvec_receipt_shows_the_reference_is_scalar_libm():
    """gfortran's vector libm must not be in the reference object, and the
    grep for it must be capable of firing -- the report carries its own
    positive control (a plain expf loop that DOES vectorise).  This module is
    not YSU: at WRF's own -O2 the scheme pulls in _ZGVbN4vv_powf, so the
    scalar -O0 reference is a choice the receipt has to keep proving."""
    report = (SHINHONG_ORACLE_DIR / "libmvec-report.txt").read_text(
        encoding="ascii")
    reference, _, rest = report.partition("# -O2")
    assert "_ZGV" not in reference, report
    assert "_ZGVbN4v_expf" in rest, (
        "the positive control produced no vector symbol, so the absence of"
        " one in the -O0 reference proves nothing")
    for symbol in ("U expf", "U powf", "U tanhf"):
        assert symbol in reference, (symbol, report)


def test_fixture_covers_the_branches_that_matter():
    fixture = _fixture()
    assert fixture.nz == 40 and fixture.ncol == 180
    assert len(fixture.case_ids) == 30
    # -- the br.gt.0 sfcflg compare: both signed zeros, both signed smallest
    # subnormals, and the smallest normal.
    br = fixture.inputs["br"].reshape(-1)
    smallest = np.float32(np.finfo(np.float32).smallest_subnormal)
    plus_zero = (br == 0) & ~np.signbit(br)
    minus_zero = (br == 0) & np.signbit(br)
    assert plus_zero.any() and minus_zero.any()
    assert (br == smallest).any() and (br == -smallest).any()
    assert (br == np.float32(np.finfo(np.float32).smallest_normal)).any()
    # -- ust exactly zero (WRF runs the scheme; there is no short circuit to
    # hide behind) and ust subnormal (case 13, the 0/0 feeder).
    ust = fixture.inputs["ust"].reshape(-1)
    assert (ust == 0).any(), "the ust==0 probe disappeared"
    assert (ust == smallest).any(), "the subnormal-ust probe disappeared"
    # -- WRF's in-cloud test is `.gt. 0.01e-3`; the fixture straddles it
    # exactly (cases 16/17).
    qc = fixture.inputs["qc"]
    assert (qc == np.float32(0.01e-3)).any()
    assert (qc == np.nextafter(np.float32(0.01e-3), np.float32(1))).any()
    # -- kpbl reaches both extremes: a one-layer PBL and the full column.
    kpbl = fixture.kpbl_reference.reshape(-1)
    assert kpbl.min() <= 2 and kpbl.max() == fixture.nz
    # -- both tke_diag values.  The OFF columns are exactly cases 22 and 27:
    # 27 is the deliberate OFF/ON pair for case 1, and 22 (kpbl == kte) MUST
    # be off because tke_diag=1 there makes WRF read q2xk(kte+1) out of
    # bounds at :1557 (see test_the_oob_tke_arm_is_guarded_and_fixture_less).
    td = fixture.tke_diag.reshape(-1)
    off = {fixture.columns[i] for i in range(td.size) if td[i] == 0}
    assert off == {(c, d) for c in (22, 27) for d in range(1, 7)}, off
    # -- dx is an axis: 6 spacings per case, and case 28 pins dy = 4*dx so
    # sqrt(dx*dy) cannot silently become dx.
    dx = fixture.inputs["dx"].reshape(-1)
    dy = fixture.inputs["dy"].reshape(-1)
    aniso = dy != dx
    assert aniso.any(), "the dy != dx probe disappeared"
    assert np.all(dy[aniso] == np.float32(4.0) * dx[aniso])
    # -- dt is an axis too (45 s and the 6 s acoustic step).
    assert set(fixture.inputs["dt"].reshape(-1).tolist()) == {45.0, 6.0}
    # -- the csfac tanh ramp (:844-851): uwstx = -80*|ust/wstar-0.5|+14 must
    # be sampled deep in the floor (tanhf returns exactly -1), on the live
    # ramp, and on the upper limb, else cslen -- the h that four of the five
    # partition functions see -- degenerates to hpbl and the tanh is dead.
    wstar = fixture.surface_reference["wstar"].reshape(-1)
    conv = wstar > 0
    assert conv.any()
    uwstx = (np.float32(-80.0)
             * np.abs(ust[conv] / wstar[conv] - np.float32(0.5))
             + np.float32(14.0))
    assert (uwstx <= -14).any(), "the csfac floor is no longer sampled"
    assert ((uwstx > -9) & (uwstx < 0)).any(), (
        "the live tanh ramp is no longer sampled")
    assert (uwstx > 0).any(), "the csfac upper limb is no longer sampled"
    # -- corf: negative (mixlen's f = max(corf,eps1) clamp) and exactly zero
    # (uniform-wind case 26, the rigsmax branch).
    corf = fixture.inputs["corf"].reshape(-1)
    assert (corf < 0).any() and (corf == 0).any()
    # -- ocean columns for brcr_sbro, one with subnormal u10 and -0.0 v10.
    xland = fixture.inputs["xland"].reshape(-1)
    assert (xland == np.float32(2.0)).any()
    u10 = fixture.inputs["u10_in"].reshape(-1)
    v10 = fixture.inputs["v10_in"].reshape(-1)
    assert (u10 == smallest).any()
    assert ((v10 == 0) & np.signbit(v10)).any()
    # -- the wspd guard probe: one column sits below any plausible port-side
    # wspd floor, so a guard's effect would be measured, not assumed.
    assert (fixture.inputs["wspd_in"].reshape(-1) == np.float32(1e-10)).any()
    # -- wspd is inout in WRF and never written; the fixture proves it.
    assert np.array_equal(
        fixture.surface_reference["wspd_out"].view(np.uint32),
        fixture.inputs["wspd_in"].view(np.uint32))


def test_partition_probe_reaches_the_clamps():
    """Measured facts of shinhong-partition.csv: pthnl/pthl reach exactly 0.0
    and exactly 1.0 on h > 0 rows (both clamps live); pu/pq/ptke reach
    exactly 1.0 (for pu/ptke only via the h == 0 early-out -- their rational
    forms approach 1 from below for every h > 0 -- and pq also saturates to
    1.0 at large d/h).  Counts are asserted > 0 so a regenerated probe grid
    cannot silently stop exercising a clamp."""
    p = _fixture().partition
    h0 = p["h"] == 0
    assert int(h0.sum()) == 14, "the h == 0 early-out rows disappeared"
    for name in ("pu", "pq", "pthnl", "pthl", "ptke"):
        assert np.all(p[name][h0] == np.float32(1.0)), (
            f"{name}: an h == 0 row is not exactly 1.0; the early-out broke")
    for name in ("pthnl", "pthl"):
        assert int(((~h0) & (p[name] == 0)).sum()) > 0, (
            f"{name} no longer reaches its pmin clamp")
        assert int(((~h0) & (p[name] == 1)).sum()) > 0, (
            f"{name} no longer reaches its pmax clamp")
    for name in ("pu", "pq", "ptke"):
        assert int((p[name] == 1).sum()) > 0, (
            f"{name} no longer reaches 1.0 anywhere")


# --------------------------------------------------------------------------
# The partition functions, directly against the probe.
# --------------------------------------------------------------------------

def test_partition_functions_are_bitwise_against_the_direct_probe():
    p = _fixture().partition
    d, h = p["d"], p["h"]
    for name, fn in (("pu", np_pu), ("pq", np_pq), ("pthnl", np_pthnl),
                     ("pthl", np_pthl), ("ptke", np_ptke)):
        got = np.asarray([fn(d[i], h[i]) for i in range(d.size)], np.float32)
        assert not _bit_diff(got, p[name]).any(), (
            f"{name}: max_ulp="
            f"{int(fp32_ulp_distance(got, p[name]).max())}")


def test_partition_probe_discriminates_the_coefficients():
    """Negative control: one ULP on one coefficient must move the curve.

    The perturbed curve below is np_pthnl's own spelling with a7 nudged by a
    single ULP.  If the probe grid could not see that, the bitwise test above
    would be vacuously satisfiable by a mistranscribed coefficient.
    """
    p = _fixture().partition
    d, h = p["d"], p["h"]
    f32 = np.float32
    a7 = np.nextafter(f32(0.243), f32(1.0))
    a1, a2, a3 = f32(1.000), f32(0.936), f32(-1.110)
    a4, a5, a6 = f32(1.000), f32(0.312), f32(0.329)

    def perturbed(dv, hv):
        if hv != 0:
            doh = dv / hv
            t2 = doh * doh
            tb = shinhong_ref._powf(doh, f32(0.875))
            value = (a7 * (a1 * t2 + a2 * tb + a3)
                     / (a4 * t2 + a5 * tb + a6) + (f32(1.0) - a7))
        else:
            value = f32(1.0)
        return f32(np.minimum(np.maximum(value, f32(0.0)), f32(1.0)))

    got = np.asarray([perturbed(d[i], h[i]) for i in range(d.size)],
                     np.float32)
    moved = int(_bit_diff(got, p["pthnl"]).sum())
    assert moved > 0, (
        "a one-ULP perturbation of pthnl's a7 is invisible to the probe grid,"
        " so the bitwise partition test cannot discriminate the coefficients"
        " and has stopped proving anything")


# --------------------------------------------------------------------------
# The measurement itself.
# --------------------------------------------------------------------------

def test_the_authority_is_bitwise_wherever_wrf_arithmetic_is_defined():
    fixture = _fixture()
    port = _port("A")
    measured_ulp = {}
    measured_bits = {}
    for name, column in LEVEL_FIELD_MAP.items():
        got = port[name]
        want = fixture.level_reference[column]
        measured_ulp[name], measured_bits[name] = _measure_field(got, want)
    for name, column in SURFACE_FIELD_MAP.items():
        measured_ulp[name], measured_bits[name] = _measure_field(
            port[name], fixture.surface_reference[column])
    assert measured_ulp == BASELINE_MAX_ULP, (
        "the authority's distance from the unmodified WRF module changed.\n"
        f"  measured {measured_ulp}\n  recorded {BASELINE_MAX_ULP}\n"
        "If a field got worse, something regressed.  If it got better --"
        " impossible here, the table is zero -- say so in a commit.")
    assert all(count == 0 for count in measured_bits.values()), (
        f"bit-pattern differences on the arithmetic mask: {measured_bits}"
        " (max_ulp==0 with differing bits means a signed-zero flip)")
    # kpbl is part of the same claim: the 1-based Fortran level index, exact
    # on every one of the 180 columns, both extremes included (2 and nz).
    np.testing.assert_array_equal(port["kpbl"], fixture.kpbl_reference)


def test_arm_b_is_bitwise_too():
    """ctopo = 0.85, ctopo2 = 0.7 -- the topo_wind arm, same zero table."""
    fixture = _fixture()
    port = _port("B")
    measured_ulp = {}
    measured_bits = {}
    for name, column in TOPO_LEVEL_FIELD_MAP.items():
        measured_ulp[name], measured_bits[name] = _measure_field(
            port[name], fixture.level_reference[column])
    for name, column in TOPO_SURFACE_FIELD_MAP.items():
        measured_ulp[name], measured_bits[name] = _measure_field(
            port[name], fixture.surface_reference[column])
    assert measured_ulp == TOPO_BASELINE_MAX_ULP, (
        f"measured {measured_ulp}\n  recorded {TOPO_BASELINE_MAX_ULP}")
    assert all(count == 0 for count in measured_bits.values()), measured_bits
    np.testing.assert_array_equal(port["kpbl"], fixture.kpbl_tw_reference)


def test_the_topo_wind_arm_is_real_and_this_far_from_arm_a():
    """A WRF-against-WRF number: no port, no tolerance to argue about.

    ctopo touches exactly one word (the k=1 momentum diagonal, :1387) and
    ctopo2 exactly the u10/v10 blend (:1472-1473).  So the two arms must be
    bitwise identical on exch_h/hpbl/wstar/delta -- pinned at 0 -- while the
    momentum tendencies and the blend differ by the pinned maxima.  The gap
    is huge in ULP because the tendencies are near-cancellations that cross
    zero between the arms.
    """
    fixture = _fixture()
    for plain, tw in (("exch_h", "exch_h_tw"),):
        got = fixture.level_reference[plain]
        want = fixture.level_reference[tw]
        nan = np.isnan(got) | np.isnan(want)
        assert not (_bit_diff(got, want) & ~nan).any(), (
            f"{plain}: the arms were bitwise identical here on 2026-08-03")
    for plain, tw in (("hpbl", "hpbl_tw"), ("wstar", "wstar_tw"),
                      ("delta", "delta_tw")):
        assert not _bit_diff(fixture.surface_reference[plain],
                             fixture.surface_reference[tw]).any(), plain
    for plain, tw in (("rublten", "rublten_tw"), ("rvblten", "rvblten_tw")):
        got = fixture.level_reference[plain]
        want = fixture.level_reference[tw]
        gap, moved = _measure_field(got, want)
        assert gap == WRF_TOPO_GAP_MAX_ULP[plain], (
            f"{plain}: ctopo gap moved to {gap}")
        assert moved > 0, (
            "if the two arms agree the fixture stopped reaching the ctopo"
            " diagonal and this test has become vacuous")
    for plain, tw in (("u10_out", "u10_tw"), ("v10_out", "v10_tw")):
        gap, moved = _measure_field(fixture.surface_reference[plain],
                                    fixture.surface_reference[tw])
        assert gap == WRF_TOPO_GAP_MAX_ULP[plain], (plain, gap)
        assert moved > 0, (
            "the u10/v10 blend no longer distinguishes ctopo2=0.7 from 1;"
            " the blend probe has become vacuous")


def test_scale_awareness_is_alive():
    """Negative control: the deep-convective case must MOVE with dx.

    Shin-Hong's entire reason to exist is that pu/pq/pthnl/pthl/ptke scale
    the transport with sqrt(dx*dy)/h.  If the six dx columns of case 1 agree
    on rthblten, the scheme has stopped being scale-aware and every
    dx-dependent test in this file is comparing a constant to itself.
    Asserted on the fixture (WRF's own words) and on the authority.
    """
    fixture = _fixture()
    port = _port("A")
    mask = fixture.case_mask(1)
    assert int(mask.sum()) == 6
    for label, rth in (("WRF", fixture.level_reference["rthblten"]),
                       ("authority", port["rthblten"])):
        columns = rth[:, 0, mask]
        distinct = {columns[:, j].tobytes() for j in range(columns.shape[1])}
        assert len(distinct) > 1, (
            f"{label}: case 1's six dx columns produced identical rthblten --"
            " the partition functions have degenerated to constants")


def test_tke_diag_off_leaves_tke_alone_and_still_zeroes_el():
    """Negative control pair: case 27 is case 1 with the diagnostics OFF.

    OFF must leave tke bitwise equal to the input while el is still zeroed
    (el_pbl is zeroed unconditionally at :673); case 1 -- same inputs, diag
    ON -- must have moved tke, which proves the OFF-path assertion is capable
    of failing.  Both directions asserted on fixture and authority.
    """
    fixture = _fixture()
    port = _port("A")
    off = fixture.case_mask(27)
    on = fixture.case_mask(1)
    tke_in = fixture.inputs["tke_in"]
    for label, tke, el in (
            ("WRF", fixture.level_reference["tke_out"],
             fixture.level_reference["el_out"]),
            ("authority", port["tke"], port["el"])):
        assert np.array_equal(tke[:, :, off].view(np.uint32),
                              tke_in[:, :, off].view(np.uint32)), (
            f"{label}: tke_diag=0 touched tke")
        assert not el[:, :, off].any(), (
            f"{label}: tke_diag=0 failed to zero el_pbl (:673)")
        moved = int((tke[:, :, on].view(np.uint32)
                     != tke_in[:, :, on].view(np.uint32)).sum())
        assert moved > 0, (
            f"{label}: case 1 (diag ON, same inputs as 27) left tke bitwise"
            " untouched, so the OFF-path assertion above is vacuous")


def test_case13_nan_column_is_wrfs_own_0_over_0():
    """The held-out NaN lanes, stated as what they are rather than as ULP.

    WRF's ``prfac2 = 15.9*wstar3/ust3/(1.+4.*karman*wstar3/ust3)`` at :1010
    divides 0 by 0 on case 13 (ust is the smallest subnormal so ust**3.
    underflows to zero; hfx = +0.0 and qfx = -0.0 make wstar3 zero while
    sfcflg stays true) and the NaN fills the heat column through the
    tridiagonal solve.  The authority must reproduce the identical NaN
    GEOMETRY.  The lanes stay out of the bitwise table for one reason only:
    the fixture's ES24.16E3 wrote the sign-less token ``NaN``, so the receipt
    cannot say which of the 2^23 payloads WRF held (x86 0/0 is 0xFFC00000;
    parsing the text yields 0x7FC00000) and bit identity is unknowable.  If
    the NaN column ever disappears from the fixture, the 0/0 was the point --
    this test failing is the intended way to find out.
    """
    fixture = _fixture()
    port = _port("A")
    mask13 = fixture.case_mask(NAN_CASE)
    for name, column in LEVEL_FIELD_MAP.items():
        want_nan = np.isnan(fixture.level_reference[column])
        got_nan = np.isnan(port[name])
        assert np.array_equal(got_nan, want_nan), (
            f"{name}: the authority's NaN geometry is not WRF's")
        assert int(want_nan.sum()) == NAN_LANES.get(name, 0), (
            f"{name}: NaN lane count moved; WRF's 0/0 column changed shape")
        assert not want_nan[:, :, ~mask13].any(), (
            f"{name}: a NaN outside case 13 is a new artifact, not the"
            " documented one")
    # The specific claim, in full: every level of case 13's heat tendency.
    rth = fixture.level_reference["rthblten"][:, :, mask13]
    assert int(np.isnan(rth).sum()) == rth.size, (
        "WRF's NaN heat column disappeared; the 0/0 in prfac2 was the point")
    # And the receipt for why the lanes are geometry-pinned: the CSV text
    # carries NaN with no sign, so the file itself lost the bit pattern.
    text = (SHINHONG_ORACLE_DIR / "shinhong-levels.csv").read_text(
        encoding="ascii")
    assert "NaN" in text and "-NaN" not in text, (
        "the fixture now records signed NaN tokens; the geometry-pinning"
        " rationale above no longer holds and these lanes should be"
        " re-examined for full bit comparison")
    # Arm B shares the same divergence through its exch_h_tw lanes.
    port_b = _port("B")
    want_nan = np.isnan(fixture.level_reference["exch_h_tw"])
    assert np.array_equal(np.isnan(port_b["exch_h"]), want_nan)
    assert int(want_nan.sum()) == TOPO_NAN_LANES["exch_h"]


def test_the_oob_tke_arm_is_guarded_and_fixture_less():
    """The documented divergence: tke_diag = 1 with kpbl == kte.

    WRF reads ``q2xk(kpbl+1) == q2xk(kte+1)`` -- one past the end of a
    ``kts:kte`` array -- at :1557 whenever the TKE diagnostic runs on a PBL
    that fills the column.  There is deliberately NO fixture row for that
    arm (a row would carry whatever the oracle's stack held; the first cut
    of this fixture proved it by accident and was regenerated), so the
    defined behaviour is pinned reference-free:

    * the preconditions still exist in the fixture: case 22 reaches
      ``kpbl == kte`` and runs with ``tke_diag = 0``, and under that flag its
      every field sits inside the zero table -- so the case did not quietly
      stop filling the column, which would make this whole test moot;
    * the authority's guarded arm (``kpbl < kte`` leaves ``efxpbl`` at 0.0)
      runs case 22's inputs WITH the diagnostic on and yields finite tke and
      el everywhere, still at ``kpbl == kte`` -- the guard engaged rather
      than the column having shrunk;
    * the guard is real, not vacuous: on case 25 (``kpbl < kte``, and the
      one column whose TKE input still carries a q2 gradient across the PBL
      top -- every parabolic profile dies below zfrac 0.95 and leaves
      ``dex == 0``) the efxpbl path is active: the entrainment TKE flux is
      nonzero, so the guard is a branch that discriminates, not dead code.
    """
    fixture = _fixture()
    mask22 = fixture.case_mask(GUARDED_OOB_CASE)
    assert int(mask22.sum()) == 6
    assert np.all(fixture.tke_diag[0, mask22] == 0), (
        "case 22 runs tke_diag=1 again: its tke_out would embed WRF's"
        " out-of-bounds read at :1557; regenerate with the documented flag")
    assert np.all(fixture.kpbl_reference[0, mask22] == fixture.nz), (
        "case 22's PBL no longer fills the column, so :1557 would be in"
        " bounds and this test is measuring nothing")
    # The guarded arm, reference-free by design: same inputs, diagnostic ON.
    x = fixture.inputs
    j = int(np.nonzero(mask22.reshape(-1))[0][0])
    column = shinhong_ref.np_shinhong_column(
        x["u"][:, 0, j], x["v"][:, 0, j], x["t"][:, 0, j],
        x["qv"][:, 0, j], x["qc"][:, 0, j], x["qi"][:, 0, j],
        x["p2d"][:, 0, j], x["pi2d"][:, 0, j],
        x["p_interface"][:, 0, j], x["dz8w"][:, 0, j], x["tke_in"][:, 0, j],
        psfc=x["psfc"][0, j], znt=x["znt"][0, j], ust=x["ust"][0, j],
        hfx=x["hfx"][0, j], qfx=x["qfx"][0, j], wspd=x["wspd_in"][0, j],
        br=x["br"][0, j], psim=x["psim"][0, j], psih=x["psih"][0, j],
        xland=x["xland"][0, j], corf=x["corf"][0, j],
        u10=x["u10_in"][0, j], v10=x["v10_in"][0, j],
        dt=x["dt"][0, j], dx=x["dx"][0, j], dy=x["dy"][0, j],
        tke_diag=1, ctopo=ARM_A[0], ctopo2=ARM_A[1])
    assert int(column["kpbl"]) == fixture.nz, (
        "the diagnostic-on rerun no longer fills the column; the guard was"
        " never exercised")
    assert np.isfinite(column["tke"]).all() and np.isfinite(column["el"]).all(), (
        "the guarded tke_diag=1 arm produced a non-finite value where WRF"
        " reads out of bounds; the guard is not the defined behaviour")
    assert (column["tke"].view(np.uint32)
            != x["tke_in"][:, 0, j].view(np.uint32)).any(), (
        "the diagnostic-on rerun left tke bitwise untouched, so this arm"
        " proved nothing about the guarded chain")
    # And the guard discriminates: on the kpbl < kte column built to keep a
    # q2 gradient at the PBL top (case 25's linear TKE ramp), the efxpbl
    # path it protects is ACTIVE (nonzero entrainment TKE flux).
    on = fixture.case_mask(25)
    j1 = int(np.nonzero(on.reshape(-1))[0][0])
    assert int(fixture.kpbl_reference[0, j1]) < fixture.nz
    column1 = shinhong_ref.np_shinhong_column(
        x["u"][:, 0, j1], x["v"][:, 0, j1], x["t"][:, 0, j1],
        x["qv"][:, 0, j1], x["qc"][:, 0, j1], x["qi"][:, 0, j1],
        x["p2d"][:, 0, j1], x["pi2d"][:, 0, j1],
        x["p_interface"][:, 0, j1], x["dz8w"][:, 0, j1],
        x["tke_in"][:, 0, j1],
        psfc=x["psfc"][0, j1], znt=x["znt"][0, j1], ust=x["ust"][0, j1],
        hfx=x["hfx"][0, j1], qfx=x["qfx"][0, j1], wspd=x["wspd_in"][0, j1],
        br=x["br"][0, j1], psim=x["psim"][0, j1], psih=x["psih"][0, j1],
        xland=x["xland"][0, j1], corf=x["corf"][0, j1],
        u10=x["u10_in"][0, j1], v10=x["v10_in"][0, j1],
        dt=x["dt"][0, j1], dx=x["dx"][0, j1], dy=x["dy"][0, j1],
        tke_diag=1, ctopo=ARM_A[0], ctopo2=ARM_A[1])
    assert column1["efxpbl"] != np.float32(0.0), (
        "case 1's entrainment TKE flux is zero, so the kpbl<kte guard no"
        " longer discriminates anything and its test is vacuous")
    assert column["efxpbl"] == np.float32(0.0), (
        "the guard left efxpbl nonzero at kpbl == kte; that is not the"
        " defined behaviour the port documents")


# ==========================================================================
# GPU: kernels/shinhong.cu against the same fixture.
#
# Everything below runs the CUDA transcription (woof/core/shinhong.py ->
# kernels/shinhong.cu, arm A) over all 180 fixture columns and pins its
# measured distance from the words the pinned WRF module wrote.  cupy
# appears only inside the GPU test bodies (conftest marks exactly those
# tests ``gpu``), and the launcher import lives inside the helper, so the
# CPU gates above collect and run on a machine with no CUDA at all.
# ==========================================================================

import pytest  # noqa: E402  (GPU section; the CPU gates above use bare asserts)

from conftest import requires_gpu  # noqa: E402

#: kernel output name -> oracle column name (the kernel speaks the engine's
#: du/dv/dtheta dialect; the oracle speaks WRF's).
GPU_LEVEL_FIELD_MAP = {
    "du": "rublten",
    "dv": "rvblten",
    "dtheta": "rthblten",
    "dqv": "rqvblten",
    "dqc": "rqcblten",
    "dqi": "rqiblten",
    "exch_h": "exch_h",
    "tke": "tke_out",
    "el": "el_out",
}
GPU_SURFACE_FIELD_MAP = {"hpbl": "hpbl", "wstar": "wstar", "delta": "delta"}

#: Worst ULP distance from ``kernels/shinhong.cu`` to the word the pinned WRF
#: module wrote, per field, over ALL 180 columns (NaN lanes excluded and
#: geometry-pinned in their own test; there are NO held-out branch cases --
#: the one branch sm_120's DAZ flips, case 7's subnormal ``br``, is closed by
#: the sh_f2d double-compare countermeasure and proven closed below).
#:
#: **A measurement, not a target.**  Asserted for *equality*: a kernel that
#: got better silently is as much a drift as one that got worse; if a field
#: improves, update the table in the same commit with the attribution.
#: Measured 2026-08-03 on the local RTX 5090 (Windows 11, driver 610.74 /
#: CUDA 13.3 UMD, cupy 14.0.1, CPython 3.13.7, numpy 2.2.6).  One card only
#: -- the second-GPU cross-check the YSU table had is replaced here by the
#: determinism pair below, which is this no-ECC card's corruption screen.
#:
#: What the numbers are made of, established by single-site experiments in
#: the kernel (see shinhong.cu's header):
#:
#:   dtheta 0            The heat tendency is BITWISE WRF's on every lane of
#:                       every column, NaN lanes aside -- zero differing bit
#:                       patterns, through both tridiagonal solves and the
#:                       wrapper's own ttnp*pi/pi round-trip, which the
#:                       kernel reproduces spelling for spelling.
#:   hpbl/wstar/delta 1  The PBL diagnosis chain is 1 ULP from WRF at its
#:                       worst column (case 5, dxi 1).
#:   exch_h 8, el 14     Device expf/powf/cbrtf against glibc's correctly
#:   tke 2022            rounded libm, through the K-profile and the QNSE
#:                       mixing-length chain.
#:   du 93207, dv 46603  Near-total cancellations amplified in ULP terms:
#:   dqv 349526          the worst du lane is 3.4966e-06 vs 3.4756e-06 m/s2.
#:                       The partition functions (powf b2 curves + FMA
#:                       contraction, pinned in isolation below) multiply
#:                       au/al inside the solves.
#:   dqc 155602          NOT arithmetic: the maxima ARE the -ftz flush lanes
#:   dqi 3055548         (kernel 0.0 vs WRF's subnormal tendency, e.g.
#:                       4.28e-39 for dqi), pinned as lane counts in the
#:                       subnormal test below.
#:
#: **The table is keyed by the kernel compiler, and that is not bookkeeping.**
#: Every number above is a property of the compiled image, so it is only
#: meaningful next to the NVRTC build that produced it.  Two rows are recorded
#: because this project has measured two, one ULP table each, from a
#: byte-identical ``shinhong.cu``:
#:
#:   13.0.48   The system CUDA v13.0 toolkit, which CuPy resolved on this box
#:             until 2026-08-04.  The certification row: it is what
#:             ``feat(shinhong): CUDA mirror on the RTX 5090`` (58dfd599c)
#:             measured, and what shipped through ArWen 1.6.3.
#:   12.9.86   The pip ``nvidia-cuda-nvrtc-cu12`` wheel, which landed in
#:             site-packages on 2026-08-04 21:47 and which CuPy resolves in
#:             preference to the toolkit from then on.  Measured 2026-08-06.
#:
#: What separates them is ONE site: the ``wscalek`` cube root at
#: shinhong.cu:981.  Under 13.0.48 spelling it ``cbrtf`` instead of
#: ``powf(x, h1)`` moved dv 1491308 -> 46603, tke 2005 -> 2022 and exch_h
#: 9 -> 8.  Under 12.9.86 the same spelling still buys exch_h 9 -> 8 and
#: nothing else: the dv and tke halves of that measurement were properties of
#: the 13.0 code generator, not of the source.  The control is in the
#: evidence: the ``powf`` spelling reads 1491308 / 9 / 2005 under BOTH
#: compilers, so the toolchain did not move generally -- only the one
#: intrinsic did.  Every other field in the table is compiler-independent,
#: including the two ``-ftz`` flush maxima, which are lane counts rather than
#: arithmetic and could not move.
#:
#: 12.8.61, 12.8.93 and 12.9.41 were measured too and all read as 12.9.86
#: does; only the CUDA 13 line produces the certification row.  A build this
#: table has never been measured under is a hard failure with its own message,
#: NOT a silent pass and not an approximate match to the nearest row.
#:
#: **The table is keyed by the card as well.**  Measured 2026-09-18 on
#: test host 1's RTX 4090 (compute capability 8.9, driver 610.57.04, cupy 14.2.0)
#: at the the 2.7.6 verification change tip 221f66f14, with the same kernel, fixture
#: and oracle: the 4090 reads ``el`` 13 where the 5090 reads 14, and every
#: other field the same as the certification row.  That is the CARD and not
#: the compiler, because the 4090 reads the identical table under NVRTC
#: 13.0.48 (build CL-36260728, library sha256 fd8dab02...8559, the
#: certification row's own compiler) and under 13.3.33 (CL-37862127,
#: e51d197b...6750a), two processes each; and the two compilers, which are
#: one site apart on the 5090, produce the same table on the 4090.  So a
#: row belongs to one (build, card) pair: an unmeasured compiler still fails
#: naming itself, and a measured compiler on a card with no row SKIPS naming
#: both, because a gate that fails for the card and not the code prevents
#: no breakage; the release card stage names that skip in its receipt.
#: Receipts: tests/data/receipts/pin-gates/shinhong-ulp-nvrtc13048-*.json,
#: shinhong-ulp-nvrtc13333-*.json and extra-pin-readings-221f66f14-run*.json.
#: A146 (2026-10-01) overturned the attribution: the el 14 was sm_120's
#: reciprocal multiply for the kernel's constant divisions, not the card;
#: with them spelled __fdiv_rn the 5090 reads el 13 under all three
#: compilers and no field differs between cards (GPU_CARD_SENSITIVE_FIELDS).
RTX_PRO_6000 = "NVIDIA RTX PRO 6000 Blackwell Server Edition"

#: The kernel's distance from the pinned WRF words since development branch
#: (sweep row 20): every field is WRF's word on every lane except the 72
#: flushed subnormal tendencies (GPU_SUBNORMAL_LANES), which are the only
#: non-zero dqv/dqc/dqi maxima here.  The unit now takes glibc's powf and
#: expf (gfk_pow/gfk_exp, glibc_flt32.cuh) instead of CUDA's, WRF's ``**h1``
#: powf at the wscalek site that used to be the one cbrtf, tanh in double
#: rounded once, and compiles with --fmad=false
#: (kernels/__init__.py _NO_FMAD_MODULES).  The rows this table carried before
#: (du 93207, dv 46603 or 1491308 by compiler, tke 2022 or 2005, el 13,
#: exch_h 8, hpbl/wstar/delta 1, on the RTX 5090, 4090 and 5070 Ti) were CUDA
#: libm and contraction; they are history now, in the commit that retired them.
#:
#: Measured 2026-10-05 on box K's RTX PRO 6000 (sm_120), one process per
#: compiler, NVRTC 13.4.92 (cupy-cuda13x 14.2.0) and 12.9.86 (cupy-cuda12x
#: 14.2.0): identical.  The compiler split that justified per-build rows is
#: gone with the device libm, so both builds carry the same row; a build or
#: card with no row is still refused or skipped, naming itself, because nobody
#: has read it under the new kernel yet.
_GPU_ROW_GLIBC = {
    "du": 0,
    "dv": 0,
    "dtheta": 0,
    "dqv": 15069,
    "dqc": 155602,
    "dqi": 3055548,
    "exch_h": 0,
    "tke": 0,
    "el": 0,
    "hpbl": 0,
    "wstar": 0,
    "delta": 0,
}

GPU_BASELINE_MAX_ULP_BY_NVRTC_BUILD = {
    # Published 2.9.11 and candidate match; attributed receipt in pin-gates.
    "13.4.59": {"NVIDIA GeForce RTX 5090": _GPU_ROW_GLIBC},
    "13.4.92": {RTX_PRO_6000: _GPU_ROW_GLIBC},
    "12.9.86": {RTX_PRO_6000: _GPU_ROW_GLIBC}, "13.2.51": {RTX_PRO_6000: _GPU_ROW_GLIBC},  # pin-gates/nvrtc-13.2.51-reading.json
}

#: The certification row.
GPU_BASELINE_MAX_ULP = _GPU_ROW_GLIBC

#: Fields allowed to differ between compiler rows on one card: NONE since the
#: device libm left the unit (the ``wscalek`` cube root used to move dv and
#: tke between the CUDA 12 and 13 lines).
GPU_COMPILER_SENSITIVE_FIELDS: tuple = ()

#: Fields allowed to differ between cards under one compiler: NONE (A146).
GPU_CARD_SENSITIVE_FIELDS: tuple = ()


def _device_name() -> str:
    import cupy

    name = cupy.cuda.runtime.getDeviceProperties(0)["name"]
    return name.decode() if isinstance(name, bytes) else str(name)


def _recorded_gpu_baseline() -> dict:
    """The row for the NVRTC build compiling this process's kernels, on
    this card.

    An unmeasured compiler fails HERE, naming itself, instead of falling
    through to a bare number mismatch against some other build's row.  That
    is the whole point of the split: the 2026-08-04 drift cost a day of
    forensics because the failure said only that two dicts differed.  A
    measured compiler on a card with no row skips, naming both: the
    2026-09-18 reading moved ``el`` with the card alone.
    """
    build = nvrtc_build()
    rows = GPU_BASELINE_MAX_ULP_BY_NVRTC_BUILD.get(build)
    assert rows is not None, (
        "no recorded ULP row for the NVRTC build compiling these kernels.\n"
        f"  kernel compiler: NVRTC {build}\n"
        f"  rows recorded:   {sorted(GPU_BASELINE_MAX_ULP_BY_NVRTC_BUILD)}\n"
        f"  fingerprint:     {compile_platform_fingerprint()}\n"
        "This table is a measurement of the compiled image, so a compiler it"
        " has never been measured under has no baseline to be compared"
        " against.  Measure this build and add its row with the attribution,"
        " or compile with a recorded one.  Do NOT relax the comparison.")
    card = _device_name()
    if card not in rows:
        pytest.skip(
            "the Shin-Hong ULP table is per NVRTC build and card, and no row is"
            f" recorded for {card!r} under NVRTC {build}; cards recorded for"
            f" this build: {sorted(rows)}.  Measure this card under this build"
            " (two processes) and add its row with the attribution.")
    return rows[card]

#: Branch divergences between the kernel and WRF, held out of the table
#: above.  EMPTY, and kept present so the day one appears it has a named
#: place to land: the plain-float32 kernel had one (case 7, the DAZ-flushed
#: subnormal br -- measured wstar 0.2362 vs WRF 0.0 before the sh_f2d
#: countermeasure went in), and
#: test_shinhong_cuda_br_countermeasure_holds_wrfs_branch proves it stays
#: closed rather than silently reopening.
GPU_BRANCH_DIVERGENCE_CASES: tuple = ()

#: Every lane where the pinned WRF module wrote a subnormal tendency.  CuPy
#: appends ``-ftz=true`` unconditionally (the FTZ receipt records this route
#: flushing on every measured mechanism), so the kernel writes exactly zero in
#: all of them.  Pinned as counts so that neither the flush nor the fixture's
#: ability to reach it can quietly disappear.
GPU_SUBNORMAL_LANES = {"dqv": 30, "dqc": 24, "dqi": 18}

#: NaN lanes per kernel field -- the same case-13 prfac2 0/0 geometry the
#: CPU gate pins (NAN_LANES), in the kernel's field names.
GPU_NAN_LANES = {"dtheta": 240, "exch_h": 6, "tke": 234}

#: Device partition curves against the oracle's direct d x h probe
#: (shinhong-partition.csv), pinned in isolation from the column dynamics.
#: All bitwise since development branch (were pu 2, pq 2, pthl 1, ptke 2:
#: pq's 2 was pure --fmad=true contraction, the others CUDA powf).
#: Measured on the RTX PRO 6000 under NVRTC 13.4.92 and 12.9.86.
GPU_PARTITION_MAX_ULP = {"pu": 0, "pq": 0, "pthnl": 0, "pthl": 0, "ptke": 0}

_GPU_PORT_CACHE: dict = {}


def _shinhong_gpu_outputs(cp, fixture, cache: bool = True):
    """Run ``launch_shinhong`` over the fixture and stitch the columns.

    ``cp`` is the cupy module, passed in by the calling GPU test (which
    imports it in its own body -- keeping this helper free of cupy imports
    is what lets conftest mark ONLY the GPU tests instead of the whole
    module).  ``dt``/``dx``/``dy``/``tke_diag`` are launch-wide scalars in
    the engine while the fixture varies them per column, so the kernel is
    launched once per (dt, dx, dy, tke_diag) group and the columns stitched
    back by group membership, exactly like ``ysu_port_outputs`` does for
    ``ysu_topdown_pblmix``.  This fixture stores WRF's ``t`` and not the
    model theta it came from, so ``theta`` is reconstructed as ``t/pi2d``.
    The kernel forms ``tx = theta*exner`` and ``thx = tx/exner`` as WRF does
    (phy_prep, :559); on all 7,200 levels of this fixture
    ``(t/pi2d)*pi2d == t``, so both words are WRF's.  The model-side
    contract (the model owns theta, WRF hands the scheme ``t = th*pi``) is
    graded on 903 columns by tests/test_shinhong_column_oracle.py.
    """
    if cache and "A" in _GPU_PORT_CACHE:
        return _GPU_PORT_CACHE["A"]
    from woof.core.shinhong import launch_shinhong

    x = fixture.inputs
    theta = np.ascontiguousarray((x["t"] / x["pi2d"]).astype(np.float32))
    dev3 = {name: cp.asarray(value) for name, value in (
        ("u", x["u"]), ("v", x["v"]), ("theta", theta), ("qv", x["qv"]),
        ("qc", x["qc"]), ("qi", x["qi"]), ("p", x["p2d"]),
        ("p_interface", x["p_interface"]), ("exner", x["pi2d"]),
        ("dz", x["dz8w"]), ("tke", x["tke_in"]))}
    dev2 = {name: cp.asarray(value) for name, value in (
        ("psfc", x["psfc"]), ("znt", x["znt"]), ("ust", x["ust"]),
        ("hfx", x["hfx"]), ("qfx", x["qfx"]), ("wspd", x["wspd_in"]),
        ("br", x["br"]), ("psim", x["psim"]), ("psih", x["psih"]),
        ("xland", x["xland"]), ("u10", x["u10_in"]), ("v10", x["v10_in"]),
        ("corf", x["corf"]))}
    dt = x["dt"].reshape(-1)
    dx = x["dx"].reshape(-1)
    dy = x["dy"].reshape(-1)
    td = fixture.tke_diag.reshape(-1)
    groups: dict = {}
    for j in range(fixture.ncol):
        key = (float(dt[j]), float(dx[j]), float(dy[j]), int(td[j]))
        groups.setdefault(key, []).append(j)
    merged: dict = {}
    for (gdt, gdx, gdy, gtd), cols in sorted(groups.items()):
        out = launch_shinhong(
            dev3["u"], dev3["v"], dev3["theta"], dev3["qv"], dev3["qc"],
            dev3["qi"], dev3["p"], dev3["p_interface"], dev3["exner"],
            dev3["dz"], dev3["tke"],
            psfc=dev2["psfc"], znt=dev2["znt"], ust=dev2["ust"],
            hfx=dev2["hfx"], qfx=dev2["qfx"], wspd=dev2["wspd"],
            br=dev2["br"], psim=dev2["psim"], psih=dev2["psih"],
            xland=dev2["xland"], u10=dev2["u10"], v10=dev2["v10"],
            corf=dev2["corf"],
            dt=gdt, dx=gdx, dy=gdy, tke_diag=gtd)
        host = {name: cp.asnumpy(value) for name, value in out.items()}
        take = np.zeros(fixture.ncol, dtype=bool)
        take[cols] = True
        for name, value in host.items():
            if name not in merged:
                merged[name] = np.zeros_like(value)
            merged[name][..., take] = value[..., take]
    if cache:
        _GPU_PORT_CACHE["A"] = merged
    return merged


def test_the_rows_only_differ_where_the_compiler_or_the_card_reaches():
    """CPU-only: across compilers on one card the rows are one site apart,
    across cards under one compiler they are identical (``el`` differed
    until A146 showed that split was the reciprocal multiply), and both
    must stay so.

    A compiler row is allowed to re-baseline ``dv`` and ``tke`` -- the two
    fields the ``wscalek`` cube root reaches -- and nothing else; a card row
    may re-baseline only GPU_CARD_SENSITIVE_FIELDS, which is empty.
    Without this, a future "the compiler moved" or "the card moved"
    commit could quietly relax a field neither ever touched, which is
    exactly the failure mode the split would otherwise create.
    """
    rows = GPU_BASELINE_MAX_ULP_BY_NVRTC_BUILD
    assert len(rows) >= 2, "two compilers must be measured, or a compiler move is unguarded"
    every = [row for by_card in rows.values() for row in by_card.values()]
    fields = {name for row in every for name in row}
    assert all(set(row) == fields for row in every), (
        "every row must carry every field")
    assert fields == set(GPU_LEVEL_FIELD_MAP) | set(GPU_SURFACE_FIELD_MAP)
    cards = {card for by_card in rows.values() for card in by_card}
    compiler_moved = set()
    for card in cards:
        on_card = [by_card[card] for by_card in rows.values() if card in by_card]
        compiler_moved |= {name for name in fields
                           if len({row[name] for row in on_card}) > 1}
    assert compiler_moved == set(GPU_COMPILER_SENSITIVE_FIELDS), (
        "the compiler rows differ on fields the wscalek site does not reach:"
        f" {sorted(compiler_moved ^ set(GPU_COMPILER_SENSITIVE_FIELDS))}.  If"
        " a new compiler really moves another field, say which site carries"
        " it in the same commit that widens GPU_COMPILER_SENSITIVE_FIELDS.")
    card_moved = set()
    for by_card in rows.values():
        card_moved |= {name for name in fields
                       if len({row[name] for row in by_card.values()}) > 1}
    assert card_moved == set(GPU_CARD_SENSITIVE_FIELDS), (
        "the card rows differ outside GPU_CARD_SENSITIVE_FIELDS:"
        f" {sorted(card_moved ^ set(GPU_CARD_SENSITIVE_FIELDS))}.  If a new"
        " card really moves another field, record the reading in the same"
        " commit that widens GPU_CARD_SENSITIVE_FIELDS.")
    # Since development branch no field differs between compilers or
    # cards (GPU_COMPILER_SENSITIVE_FIELDS and GPU_CARD_SENSITIVE_FIELDS are
    # empty), and the equalities above hold every row to that.


def test_the_kernel_compiler_identifies_itself_to_four_parts():
    """CPU-only: the fingerprint resolves, and resolves finely enough.

    ``nvrtc.getVersion()`` reports ``(12, 9)`` for both 12.9.41 and 12.9.86,
    which do not generate the same code -- the published pin row was measured
    at that resolution and could not name the 2026-08-04 drift.  Four parts
    is the resolution the row key needs, so a fingerprint that silently fell
    back to a coarser one has to fail here rather than at some ULP table.
    """
    fingerprint = compile_platform_fingerprint()
    build = fingerprint["nvrtc_build"]
    assert build != "unavailable", (
        "NVRTC would not identify itself; the ULP rows below cannot be keyed"
        f" without it.  fingerprint: {fingerprint}")
    assert len(build.split(".")) == 3, f"not a four-part build: {build!r}"
    assert build == nvrtc_build()
    assert fingerprint["nvrtc_build_id"].startswith("CL-"), fingerprint


@pytest.mark.gpu
@requires_gpu
def test_shinhong_cuda_column_holds_its_measured_distance_from_wrf():
    import cupy  # noqa: F401  (marks this test for -m "not gpu")

    fixture = _fixture()
    port = _shinhong_gpu_outputs(cupy, fixture)
    measured = {}
    for name, column in GPU_LEVEL_FIELD_MAP.items():
        measured[name], _ = _measure_field(
            np.ascontiguousarray(port[name], np.float32),
            fixture.level_reference[column])
    for name, column in GPU_SURFACE_FIELD_MAP.items():
        measured[name], _ = _measure_field(
            np.ascontiguousarray(port[name], np.float32),
            fixture.surface_reference[column])
    from woof import wrf_exact

    if wrf_exact.ENABLED:
        # The strict build (GPUWM_WRF_EXACT=1: no FMA, no FTZ, IEEE division
        # and square root) carries no flush lanes, so every field of every
        # column is 0 ULP.  Measured 2026-10-07 on the test host's RTX PRO 6000 under
        # NVRTC 12.9.86 (development branch).  The default build keeps its
        # per-compiler rows below.
        assert measured == {name: 0 for name in measured}, measured
        return
    recorded = _recorded_gpu_baseline()
    assert measured == recorded, (
        "the CUDA kernel's distance from the unmodified WRF module changed.\n"
        f"  measured {measured}\n  recorded {recorded}\n"
        f"  kernel compiler: NVRTC {nvrtc_build()}, card {_device_name()}\n"
        "The compiler and card row above matched, so this is NOT the"
        " 2026-08-04 NVRTC swap and not the card: the source or the fixture"
        " moved.  If a field got worse, something regressed.  If it got"
        " better, say so: update the row in the same commit as the"
        " improvement, with the attribution, so the residual stays"
        " documented.")
    # The table above must not be vacuously zero: device libm and FMA
    # contraction are real, and the fields that carry them must show it.
    # (dtheta IS zero -- bitwise -- and that is the exception, not the rule.)
    assert any(value > 0 for value in measured.values()), (
        "every field measured 0 ULP: either the kernel became bitwise"
        " (update this guard and celebrate in the commit message) or the"
        " comparison is no longer measuring the kernel")


@pytest.mark.gpu
@requires_gpu
def test_shinhong_cuda_kpbl_matches_wrf_exactly():
    """The 1-based Fortran level index, exact on every one of the 180
    columns -- there are no held-out cases (GPU_BRANCH_DIVERGENCE_CASES is
    empty, and this test is part of what keeps it accurate)."""
    import cupy  # noqa: F401

    fixture = _fixture()
    port = _shinhong_gpu_outputs(cupy, fixture)
    assert GPU_BRANCH_DIVERGENCE_CASES == ()
    np.testing.assert_array_equal(
        np.asarray(port["kpbl"], dtype=np.int32), fixture.kpbl_reference)


@pytest.mark.gpu
@requires_gpu
def test_shinhong_cuda_nan_geometry_is_wrfs_own_0_over_0():
    """Case 13's NaN lanes: geometry asserted, bits never compared.

    WRF's prfac2 0/0 (:1010) writes the NaN; the kernel's 0.0f/0.0f writes
    one too, with a device payload the CSV text could not carry even if the
    x86 payload were known (the fixture wrote the sign-less token ``NaN``).
    So the claim is the CPU gate's claim: the kernel goes NaN exactly where
    WRF goes NaN, in exactly the pinned counts, and nowhere else.
    """
    import cupy  # noqa: F401

    fixture = _fixture()
    port = _shinhong_gpu_outputs(cupy, fixture)
    mask13 = fixture.case_mask(NAN_CASE)
    for name, column in GPU_LEVEL_FIELD_MAP.items():
        want_nan = np.isnan(fixture.level_reference[column])
        got_nan = np.isnan(port[name])
        assert np.array_equal(got_nan, want_nan), (
            f"{name}: the kernel's NaN geometry is not WRF's")
        assert int(got_nan.sum()) == GPU_NAN_LANES.get(name, 0), (
            f"{name}: NaN lane count moved")
        assert not got_nan[:, :, ~mask13].any(), (
            f"{name}: a NaN outside case 13 is a new artifact, not the"
            " documented one")
    for name in GPU_SURFACE_FIELD_MAP:
        assert not np.isnan(port[name]).any(), (
            f"{name}: WRF writes no NaN surface fields on this fixture")


@pytest.mark.gpu
@requires_gpu
def test_shinhong_cuda_br_countermeasure_holds_wrfs_branch():
    """The one branch the route's flush flips, proven closed and non-vacuous.

    Case 7 puts the smallest positive subnormal on ``br``; WRF's
    ``br .gt. 0`` is true and the column is STABLE (wstar = delta = 0).
    Every FP32 mechanism the receipt measures on this compile route flushes
    that subnormal to zero, the float compare among them; and this test's
    own measurement, not a receipt cell, is that ``(double)brv > 0.0`` read
    identical to the plain float32 compare --
    so shinhong.cu decodes the bits (sh_f2d, rrtmg_sw.cu's rsw_f2d) before
    comparing.  Before the countermeasure the kernel measured wstar
    0.2362 / delta 18.49 on case 7 (the convective arm); these assertions
    fail loudly if it ever regresses.  Case 8 (-1.4e-45) needs no
    countermeasure -- both sides take the false arm -- and case 7's
    membership in the equality table above is the rest of the proof.
    """
    import cupy  # noqa: F401

    fixture = _fixture()
    port = _shinhong_gpu_outputs(cupy, fixture)
    br = fixture.inputs["br"].reshape(-1)
    smallest = np.float32(np.finfo(np.float32).smallest_subnormal)
    mask7 = fixture.case_mask(7)
    # the probe is still the probe (vacuity guard on the fixture side)
    assert np.all(br[mask7] == smallest), (
        "case 7 no longer carries +1.4e-45 on br; this test measures nothing")
    assert (br == -smallest).any(), "the case-8 negative probe disappeared"
    # WRF's stable arm on the fixture side (second vacuity guard)
    assert np.all(fixture.surface_reference["wstar"].reshape(-1)[mask7] == 0.0)
    # ... and the kernel takes it too
    got_wstar = np.asarray(port["wstar"]).reshape(-1)
    got_delta = np.asarray(port["delta"]).reshape(-1)
    assert np.all(got_wstar[mask7] == 0.0), (
        "the kernel went convective on case 7: the sh_f2d br countermeasure"
        " has regressed to a flushing compare")
    assert np.all(got_delta[mask7] == 0.0)
    # and the stable arm is not trivially universal
    assert (got_wstar > 0.0).any(), (
        "no column is convective at all; wstar == 0 on case 7 proves nothing")


@pytest.mark.gpu
@requires_gpu
def test_shinhong_cuda_ftz_flushes_every_subnormal_tendency_wrf_wrote():
    """``-ftz=true`` is appended by CuPy and cannot be turned off from here
    (the FTZ receipt records this route flushing on every measured mechanism).

    Asserted as an equality in both directions: the fixture must still reach
    subnormal tendencies at all, and the kernel must still be writing exactly
    zero for all of them.  If either changes, the flush has been fixed or the
    fixture has stopped exercising it, and both deserve a commit message.
    """
    import cupy  # noqa: F401

    from woof import wrf_exact

    strict = wrf_exact.ENABLED
    fixture = _fixture()
    port = _shinhong_gpu_outputs(cupy, fixture)
    smallest_normal = np.float32(np.finfo(np.float32).smallest_normal)
    found = {}
    for name, column in GPU_LEVEL_FIELD_MAP.items():
        want = fixture.level_reference[column]
        subnormal = (np.abs(want) > 0) & (np.abs(want) < smallest_normal)
        if not subnormal.any():
            continue
        found[name] = int(subnormal.sum())
        got = np.ascontiguousarray(port[name], np.float32)
        if strict:
            # GPUWM_WRF_EXACT=1 compiles with --ftz=false: every subnormal
            # WRF wrote comes back as WRF's own word.
            assert np.array_equal(got[subnormal].view(np.uint32),
                                  want[subnormal].view(np.uint32)), name
            continue
        assert np.all(got[subnormal] == 0), (
            f"{name}: a subnormal lane survived the flush, which would be"
            " news")
    assert found == GPU_SUBNORMAL_LANES, found


@pytest.mark.gpu
@requires_gpu
def test_shinhong_cuda_partition_curves_hold_their_distance_from_glibc():
    """The five device partition curves against the oracle's direct probe.

    This is the GPU section's guaranteed-firing negative control: device
    powf is not glibc's correctly rounded powf and --fmad=true contraction
    is real, so SOME probe row must differ -- if none does, either device
    libm became correctly rounded (update the table and say so) or the probe
    stopped reaching the curves.  pq isolates the contraction alone: its
    rational form has no transcendental, and its 2 ULP is the measured cost
    of get_kernel's default compile options on this scheme's simplest curve.
    """
    import cupy

    fixture = _fixture()
    from woof.core.kernels import get_kernel

    p = fixture.partition
    n = int(p["d"].size)
    d = cupy.asarray(p["d"])
    h = cupy.asarray(p["h"])
    out = {name: cupy.empty(n, dtype=cupy.float32)
           for name in ("pu", "pq", "pthnl", "pthl", "ptke")}
    kernel = get_kernel("shinhong", "shinhong_partition_probe")
    blocks = (n + 255) // 256
    kernel((blocks,), (256,),
           (d, h, out["pu"], out["pq"], out["pthnl"], out["pthl"],
            out["ptke"], np.int64(n)))
    measured = {}
    differing_total = 0
    h0 = p["h"] == 0
    for name in GPU_PARTITION_MAX_ULP:
        got = cupy.asnumpy(out[name])
        measured[name] = int(fp32_ulp_distance(got, p[name]).max())
        differing_total += int(_bit_diff(got, p[name]).sum())
        # the h == 0 early-out must stay exact on device too
        assert np.all(got[h0] == np.float32(1.0)), (
            f"{name}: an h == 0 row is not exactly 1.0 on device")
    assert measured == GPU_PARTITION_MAX_ULP, (
        f"measured {measured}\n  recorded {GPU_PARTITION_MAX_ULP}")
    # Bitwise since the unit took glibc's powf and dropped contraction.
    assert differing_total == 0, differing_total


@pytest.mark.gpu
@requires_gpu
def test_shinhong_cuda_determinism_pair_is_bit_identical():
    """Two full-fixture runs must agree bit for bit -- the no-ECC screen.

    This card has no ECC, so a silent VRAM flip's observable symptom is a
    wrong number, not an exception; dual-run byte comparison is the
    project's standing corruption detector.  The kernel is deterministic by
    construction (no atomics on the numeric path, one thread per column),
    so ANY pair disagreement is corruption or a scheduler-visible defect,
    and either is worth a loud stop.  Cheap here: each run is 24 launches
    over 180 columns.
    """
    import cupy  # noqa: F401

    fixture = _fixture()
    first = _shinhong_gpu_outputs(cupy, fixture, cache=False)
    second = _shinhong_gpu_outputs(cupy, fixture, cache=False)
    assert set(first) == set(second)
    for name in first:
        a, b = first[name], second[name]
        if a.dtype == np.int32:
            same = np.array_equal(a, b)
        else:
            same = np.array_equal(a.view(np.uint32), b.view(np.uint32))
        assert same, (
            f"{name}: the determinism pair disagrees -- on this card treat"
            " it as corruption until proven otherwise")


@pytest.mark.gpu
@requires_gpu
def test_shinhong_cuda_scale_awareness_is_alive_on_device():
    """Negative control: case 1's six dx columns must MOVE on device too.

    Shin-Hong exists to scale transport with sqrt(dx*dy)/h; if the six
    spacings agree on the device heat tendency, the partition functions have
    degenerated to constants inside the kernel and every dx-dependent number
    above is comparing a constant to itself.
    """
    import cupy  # noqa: F401

    fixture = _fixture()
    port = _shinhong_gpu_outputs(cupy, fixture)
    mask = fixture.case_mask(1)
    assert int(mask.sum()) == 6
    columns = port["dtheta"][:, 0, mask]
    distinct = {columns[:, j].tobytes() for j in range(columns.shape[1])}
    assert len(distinct) > 1, (
        "case 1's six dx columns produced identical device rthblten -- the"
        " kernel's partition functions have degenerated to constants")


@pytest.mark.gpu
@requires_gpu
def test_shinhong_cuda_validator_reports_wrfs_nan_and_stays_quiet_otherwise():
    """The separate-TU validator, exercised in both directions.

    ``shinhong_validation.cu`` lives in its own NVRTC module so that adding
    or changing the diagnostic cannot affect compilation of the numerical
    kernel; this test is what keeps the TU itself from rotting.  A launch
    that includes case 13 must be flagged, and the first-invalid name must
    be ``dtheta`` -- the lowest bit set in launcher order, since WRF's
    prfac2 0/0 fills the heat column while du/dv stay finite.  The same
    validator over a launch with case 13's columns removed must return
    ``None``: a validator that cries wolf on WRF-clean bundles would train
    the driver to ignore it.
    """
    import cupy

    from woof.core.shinhong import launch_shinhong, validate_shinhong_outputs

    fixture = _fixture()
    x = fixture.inputs
    theta = np.ascontiguousarray((x["t"] / x["pi2d"]).astype(np.float32))
    host3 = {"u": x["u"], "v": x["v"], "theta": theta, "qv": x["qv"],
             "qc": x["qc"], "qi": x["qi"], "p": x["p2d"],
             "p_interface": x["p_interface"], "exner": x["pi2d"],
             "dz": x["dz8w"], "tke": x["tke_in"]}
    host2 = {"psfc": x["psfc"], "znt": x["znt"], "ust": x["ust"],
             "hfx": x["hfx"], "qfx": x["qfx"], "wspd": x["wspd_in"],
             "br": x["br"], "psim": x["psim"], "psih": x["psih"],
             "xland": x["xland"], "u10": x["u10_in"], "v10": x["v10_in"],
             "corf": x["corf"]}
    status = cupy.zeros((1,), dtype=cupy.uint32)

    def run(column_mask):
        d3 = {name: cupy.asarray(np.ascontiguousarray(value[..., column_mask]))
              for name, value in host3.items()}
        d2 = {name: cupy.asarray(np.ascontiguousarray(value[..., column_mask]))
              for name, value in host2.items()}
        # The scalars are arbitrary valid values here: the NaN mechanism is
        # case 13's own surface inputs (ust**3. == 0 with wstar3 == 0), and
        # neither it nor its absence depends on dt/dx/dy.
        return launch_shinhong(
            d3["u"], d3["v"], d3["theta"], d3["qv"], d3["qc"], d3["qi"],
            d3["p"], d3["p_interface"], d3["exner"], d3["dz"], d3["tke"],
            psfc=d2["psfc"], znt=d2["znt"], ust=d2["ust"], hfx=d2["hfx"],
            qfx=d2["qfx"], wspd=d2["wspd"], br=d2["br"], psim=d2["psim"],
            psih=d2["psih"], xland=d2["xland"], u10=d2["u10"],
            v10=d2["v10"], corf=d2["corf"],
            dt=45.0, dx=1000.0, dy=1000.0, tke_diag=1)

    everything = np.ones(fixture.ncol, dtype=bool)
    flagged = validate_shinhong_outputs(run(everything), status)
    assert flagged == "dtheta", (
        f"expected the case-13 NaN heat column to be flagged first, got"
        f" {flagged!r}")
    clean = validate_shinhong_outputs(run(~fixture.case_mask(NAN_CASE)),
                                      status)
    assert clean is None, (
        f"the validator flagged {clean!r} on a bundle WRF itself keeps"
        " finite")
