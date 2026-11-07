"""Noah against the byte-unmodified WRF v4.6.1 driver, for the first time.

``sf_surface_physics=2`` is the land surface in every registry template and
every production config in this tree, and until this file existed no WRF
number had ever been produced for it.  ``kernels/noah.cu`` was checked only
against ``woof.verify.npref.np_noah_column`` -- a float64 mirror of the same
transcription -- plus energy closure and physics plausibility, all of which a
transcription error present in both copies passes.  ``tools/noah_wrf461_oracle/``
now drives ``lsm`` in the pinned tree over 80 columns and dumps every
driver input and every driver output; this module measures the kernel against
that dump and pins what it measures.

**The numbers below are a measurement, not a target.**  They record what the
shipped kernel actually does, so it cannot change without someone saying so in
a commit.  They are asserted for *equality*: a kernel that got better silently
is as much a drift as one that got worse.  Every baseline was produced twice
on the local RTX 5090 in separate processes and was bit-identical.

What the oracle found on its first run
--------------------------------------

1. ``chklowq`` was 0.0 against WRF's 1.0 on the open-water and sea-ice
   columns.  ``module_sf_noahdrv.F`` writes CHKLOWQ for every column at
   :809-814, *before* the XLAND branch; the kernel returned from its three
   skip paths without writing it.  Fixed in its own commit, and the mirror had
   it right all along -- which is precisely why 37 self-consistency tests
   could not see it.
2. Case 18 -- ``SNOW`` is the smallest positive subnormal.  Fixed in 2.8.6;
   see "Case 18 and the land-ice columns" below.
3. The frozen-ground infiltration family (cases 6, 7, 42) and the fp64 energy
   residual, both described beside their baselines.  The first of those was
   filed as libm noise for a release and was not: ``SRT`` was being reached
   with ``REDPRM``'s ``FRZFACT`` where WRF passes its ``FRZX``, so ``ACRT``
   ran 1/FRZK = 6.67x large and the frozen-ground infiltration limiter did
   not limit.  Both the mirror and the kernel had made the same substitution,
   so only this fixture could see it -- and only once something graded the
   *mirror* against WRF too, which nothing did until
   :func:`test_the_mirror_reproduces_wrfs_frozen_ground_infiltration`.

This gate has been observed to fail
----------------------------------

A gate that has never fired is not evidence, so it was made to fire twice
before it was committed, by patching ``kernels/noah.cu`` and re-running:

* deleting the one-line ``chklowq`` fix -- the defect this oracle found --
  moves ``chklowq`` from 0 to 1065353216 and the table assertion fails;
* respelling ``q2k = qv1/(1+qv1)`` as the algebraically identical
  ``1 - 1/(1+qv1)`` moves ``snopcx`` from 1125 to 1243 and it fails again.

One mutation that did *not* fire is worth recording too: perturbing ``val =
1 - expf(-kdt*dt1)`` by a relative 1e-7 changed nothing, because ``kdt*dt1``
is small enough that the perturbation falls below float32 resolution there.
The table is sensitive to the physics, not to every keystroke in the file.

The two CPU gates below were made to fire the same way, on the tree as it
stood before the FRZX fix: the mirror gate reports case 6 SFCRUNOFF
8.477e-05 against WRF's 4.935e-03, and the source gate reports ``noah_smflx``
called with ``frzfact``.  nvcc is a third witness -- it emitted ``variable
"frzx" was declared but never referenced`` for ``noah.cu`` until the fix.

Case 18 and the land-ice columns
--------------------------------

Both used to be held out of the table, and both were defects, fixed in 2.8.6.

* **case 18** -- ``SNOW`` is the smallest positive float32 subnormal.  WRF's
  ``IF(SNOW(I,J).GT.0.0)`` is TRUE, so the column takes the ice-saturation
  ``Q2SAT``/``DQSDT2`` blend.  CuPy appends ``-ftz=true`` unconditionally, so
  the kernel's ``snow_a[idx] > 0.0f`` was FALSE and the column took the
  snow-free arm, bit-identical to its own case 1 (TSK 0.083 K, HFX 1.95 W/m2,
  LH 2.87 W/m2 from WRF).  The compare now decodes the bits (``noah_f2d``, the
  Shin-Hong ``sh_f2d`` pattern), case 18 matches WRF bitwise in TSK, HFX and
  LH, and it is back in the table without moving any row of it.
* **land ice, cases 28 and 43-55** -- ``ivgtyp == isice``.  WRF calls
  ``SFLX_GLACIAL`` (``module_sf_noahlsm_glacial_only.F``); ``noah_column``
  returned without touching the column (case 28: TSK 255.0 against 273.15 K,
  HFX 0 against 388.9 W/m2, LH 0 against -319.1 W/m2, SH2O 0.28 against 1.0).
  ``noah_glacial_column`` now runs it with gfortran's rounding and glibc's
  expf/powf.  The fixture gained thirteen land-ice columns, one per
  SFLX_GLACIAL branch, and all fourteen are a population of their own,
  pinned bitwise in :data:`GLACIAL_MAX_ULP`.

Held-out column
---------------

One fixture column is excluded from the ULP table and asserted individually,
because it does not measure arithmetic:

* **case 25** -- ``CHS``/``CHS2``/``CQS2`` are all the smallest positive
  subnormal.  Traced on W1 with print statements in a scratch copy of
  ``module_sf_noahlsm.F``: PENMAN's ``RR`` overflows to +Inf (the
  ``/(SFCPRS*CH)`` term), ``RAD = FNET/RCH`` is +Inf, ``EPSCA`` is Inf/Inf =
  NaN and ``ETP`` is NaN.  NOPAC then takes neither of its arms for ``ETA``:
  ``ETP > 0`` is false, so EVAPO never sets it, and ``ETP <= 0`` is false, so
  ``ETA = ETP`` never runs.  ``ETA`` is an INTENT(OUT) dummy that nothing
  assigned, and WRF writes it out as ``QFX``: the 147.7 in the fixture is
  whatever that stack word held, a WRF uninitialised read and not a result.
  The port's ``eta`` is initialised, so its QFX is 0; that is not copied.
  ``PRCP1 = PRCP1 + DEW`` carries the NaN into SMFLX in both, and WRF's soil
  goes NaN through ``MAX(MIN(STOT,SMCMAX),0.02)`` in SSTEP, which gfortran
  builds so that the NaN operand wins; CUDA's fminf/fmaxf return the
  number, so the kernel keeps a finite soil state.  Which operand wins a NaN
  in a gfortran MIN/MAX depends on how fold orders the operands at that
  call site (measured; see the lane commit that withdrew noah_max), so no
  one device spelling reproduces it.  The input is unrunnable in both and
  the row is pinned as such.
"""
# Replaced in 2.9.1 (the WRF exact chain): test_a_land_ice_column_is_no_longer_left_untouched -> test_land_ice_columns_are_bitwise_against_wrf.
# Replaced in 2.9.1 (the WRF exact chain): test_case_18_is_the_ftz_flush_and_nothing_else -> test_case_18_takes_wrfs_snow_branch_on_a_subnormal_snow.
# Replaced in 2.9.1 (the WRF exact chain): test_noah_cuda_land_ice_columns_hold_their_measured_distance_from_wrf -> test_land_ice_columns_are_bitwise_against_wrf.

from __future__ import annotations

import hashlib
import re

import numpy as np
import pytest

from conftest import requires_gpu
from woof.core.fp32_ulp import fp32_ulp_distance
from woof.core.kernels import module_source
from woof.core.noah import load_tables, pack_params
from woof.verify.noah_oracle import (
    FIXTURE_DT,
    FIXTURE_DZS,
    FIXTURE_ISICE,
    FIXTURE_ISURBAN,
    FIXTURE_XICE_THRESHOLD,
    INPUT_COLUMNS,
    NOAH_GLACIAL_ORACLE_FILES,
    NOAH_ORACLE_DIR,
    NOAH_ORACLE_FILES,
    OUTPUT_COLUMNS,
    SOIL_OUTPUT_COLUMNS,
    SFCDIAGS_COLUMNS,
    load_noah_oracle,
    noah_port_outputs,
    noah_sfcdiags_outputs,
)
from woof.verify.npref import np_noah_column

#: Columns whose disagreement with WRF is not a rounding.  Asserted one at a
#: time below instead of being folded into a ULP maximum.  18 (the -ftz flush)
#: and 28 (SFLX_GLACIAL) left this tuple when both were fixed in 2.8.6.
HELD_OUT_CASES = (25,)

#: The land-ice columns (``ivgtyp == isice``) of every fixture: 28 and 43-55.
GLACIAL_CASES = (28, *range(43, 56))

#: SRT's own frozen-ground predicate, ``IF (DICE > 1.E-2)``
#: (``module_sf_noahlsm.F:3794``): below this the infiltration limiter is
#: inert and a column says nothing about FRZX.
FROZEN_DICE_M = 1.0e-2

#: The columns, in every switch fixture, where SRT's frozen-ground limiter is
#: live (DICE > 0.01) and WRF runs water off: 6, 7 and 42 from the original
#: 42, and 61 and 67-69, 71 from the regime columns added for the 0 ULP
#: verification (development branch).
FROZEN_RUNOFF_CASES = (6, 7, 42, 61, 67, 68, 69, 71)

#: How far the float64 mirror may sit from WRF's float32 SFCRUNOFF on the
#: columns where it IS live.  ``RUNOFF1 = PCPDRP - INFMAX``
#: (``module_sf_noahlsm.F:3830``) subtracts two nearly equal quantities, so
#: the precision gap is amplified; measured worst is 7.8e-06 over all four
#: fixtures, against 9.9e-01 when SRT is handed FRZFACT instead of FRZX.
FROZEN_RUNOFF_REL = 1.0e-4

#: Worst ULP distance from ``kernels/noah.cu``'s ``noah_column`` to the word
#: ``lsm`` wrote, over every column of all four switch fixtures except the
#: land-ice population (:data:`GLACIAL_MAX_ULP`) and the held-out case 25:
#: ZERO IN EVERY FIELD.  MEASURED 2026-10-07 on test host (RTX PRO 6000
#: Blackwell, sm_120, NVRTC 12.9.86, cupy-cuda12x 14.2.0) at
#: development branch, under GPUWM_WRF_EXACT=1 and under default
#: arithmetic alike, over the 80-column fixtures (65 graded per fixture:
#: ordinary land and open water).
#:
#: How the table got from the numbers it held before to zero, in the commits
#: of development branch (the previous readings are in git history):
#:
#:   noahres   164754 -> 0   summed in float32 in WRF's order
#:                           (module_sf_noahdrv.F:1308) instead of float64
#:                           rounded once; T1**4 is an INTEGER power, which
#:                           gfortran -O0 hands to libgcc's binary powering
#:                           (noah_powi).
#:   sfcrunoff   2812 -> 0   SRT's frozen-ground FCR series ACRT**(CVFRZ-J) is
#:   smstav         2 -> 0   an INTEGER power too (noah_powi), and its EXP is
#:   smstot         1 -> 0   WOOF's own float32 expf (gfk_exp), not CUDA's.
#:   hfx, grdflx, qfx, lh, potevp, qsfc, canwat, snowc, albedo, tsk, lai,
#:   udrunoff, acsnom       -> 0   nvcc's FMA contraction (noah and noah_init
#:                           now compile with --fmad=false, as gfortran -O0
#:                           forms no FMA) and CUDA's expf/logf/powf/log10f
#:                           (now gfk_exp/gfk_log/gfk_pow and noah_log10f).
#:   snopcx      1141 -> 0   the last 1 ULP was SNOMLT*1000*XLF/DT where WRF
#:                           divides by FDTLIW = DT/(1.E3*XLF)
#:                           (module_sf_noahdrv.F:730,1595).
#:
#: chs, chs2 and cqs2 joined the table with the 80-column fixtures: lsm's
#: CHS2 = CQS2 (module_sf_noahdrv.F:1275) is now discriminated (case 76).
BASELINE_MAX_ULP = {name: 0 for name in (*OUTPUT_COLUMNS, *SOIL_OUTPUT_COLUMNS)}

#: Per-architecture and per-compiler rows.  The rows sm_86/sm_89 and
#: (sm_120, NVRTC 12.9.86) carried before existed because nvcc contracted
#: and NVRTC turned divisions by constants into reciprocal multiplies
#: differently per card; with --fmad=false and every division spelled
#: __fdiv_rn neither choice is left to the compiler.  Only sm_120 under
#: NVRTC 12.9.86 has been read at zero (W1, 2026-10-07).  A card that reads
#: anything else fails the equality below, which is the point: it is then
#: measured and recorded here with its numbers.
BASELINE_MAX_ULP_BY_ARCHITECTURE: dict[str, dict[str, int]] = {}
BASELINE_MAX_ULP_BY_ARCH_AND_BUILD: dict[tuple[str, str], dict[str, int]] = {}

#: The zero table above is the strict build's (GPUWM_WRF_EXACT=1: no FMA
#: contraction, no flush).  A default process compiles noah.cu with CuPy's
#: -ftz=true and NVRTC's contraction, and lands here instead.  MEASURED
#: 2026-10-16 on test host 2's RTX 5090 (sm_120, NVRTC 13.4.92) at the 2.9.1
#: exact-chain intake, over the same columns.  For scale, 2.9.1's default
#: kernel (before the intake) read hfx 260781, lh 192330, potevp 266755 and
#: tsk 2784 ULP on these fixtures.  A card or compiler not listed fails
#: here with its numbers, to be measured and recorded.
#: 2.9.1 integration: noah also compiles without FMA contraction in a default
#: process (kernels/__init__.py _NO_FMAD_MODULES), as the lane that brought
#: these words measured it, and the default column reads ZERO in every field
#: here too (an RTX 5090, sm_120, NVRTC 13.4.92, 2026-10-16), down from
#: hfx 802, grdflx 2089, snopcx 545 and noahres 606208 ULP with contraction.
DEFAULT_ARITHMETIC_MAX_ULP_BY_ARCH_AND_BUILD: dict[tuple[str, str], dict[str, int]] = {
    # Restored pinned tables: all 34 fields retain zero ULP.
    ("120", "13.4.59"): dict(BASELINE_MAX_ULP),
    ("120", "13.4.92"): dict(BASELINE_MAX_ULP), ("120", "13.2.51"): dict(BASELINE_MAX_ULP),  # pin-gates/nvrtc-13.2.51-reading.json
}

#: SFCDIAGS on the port's words, default build, same measurement.
SFCDIAGS_DEFAULT_MAX_ULP_BY_ARCH_AND_BUILD: dict[tuple[str, str], dict[str, int]] = {
    ("120", "13.4.92"): {"t2": 1, "th2": 1, "q2": 4},
}
#: Rows an (architecture, build) row may move from its architecture's row.
ARCH_AND_BUILD_SENSITIVE_FIELDS = ("snow", "snowh")

#: ``noah_glacial_column`` against SFLX_GLACIAL on the fourteen land-ice
#: columns of all four fixtures: bitwise in every field.  MEASURED 2026-10-05
#: on an RTX 5090 (sm_120, NVRTC 13.4.92).  The arm spells every operation
#: with an __f*_rn intrinsic and calls glibc's own expf/powf words, so no
#: contraction or libm choice is left to the compiler.  Before 2.8.6 the
#: column was not computed at all: case 28 read TSK 255.0 against 273.15 K and
#: HFX 0 against 388.9 W/m2.
GLACIAL_MAX_ULP = {name: 0 for name in (*OUTPUT_COLUMNS, *SOIL_OUTPUT_COLUMNS)}

#: The same arm on the 24-column land-ice fixtures, which carry every output
#: column but CHS, CHS2 and CQS2 (they predate that echo).
GLACIAL_FIXTURE_MAX_ULP = {name: 0 for name in (*OUTPUT_COLUMNS, *SOIL_OUTPUT_COLUMNS)
                           if name not in ("chs", "chs2", "cqs2")}

#: The float64 mirror's SFLX_GLACIAL against WRF's float32 words: relative.
GLACIAL_MIRROR_REL = 1.0e-4


def _fixtures():
    return [load_noah_oracle(name) for name in NOAH_ORACLE_FILES]


def _glacial_fixtures():
    return [load_noah_oracle(name) for name in NOAH_GLACIAL_ORACLE_FILES]


def _arithmetic_mask(fixture) -> np.ndarray:
    return np.asarray([c not in HELD_OUT_CASES and c not in GLACIAL_CASES
                       for c in fixture.cases])


def _glacial_mask(fixture) -> np.ndarray:
    return np.asarray([c in GLACIAL_CASES for c in fixture.cases])


def _mirror_column(fixture, i) -> dict:
    """One fixture column as the plain dict ``np_noah_column`` takes."""
    col = {field: float(fixture.inputs[field].reshape(-1)[i])
           for field in INPUT_COLUMNS}
    col["ivgtyp"] = int(fixture.ivgtyp.reshape(-1)[i])
    col["isltyp"] = int(fixture.isltyp.reshape(-1)[i])
    for field in ("smois", "tslb", "sh2o"):
        col[field] = np.asarray(
            fixture.inputs[field].reshape(4, -1)[:, i], np.float64)
    return col


def _srt_dice(fixture, i) -> float:
    """SRT's ``DICE`` for one column, in m: total column soil-ice depth.

    ``module_sf_noahlsm.F:3754`` seeds it with ``-ZSOIL(1)*SICE(1)`` and
    ``:3765`` accumulates ``(ZSOIL(KS-1)-ZSOIL(KS))*SICE(KS)``.
    """
    smois = fixture.inputs["smois"].reshape(4, -1)[:, i]
    sh2o = fixture.inputs["sh2o"].reshape(4, -1)[:, i]
    sice = np.asarray(smois - sh2o, np.float64)
    zsoil = -np.cumsum(np.asarray(FIXTURE_DZS, np.float64))
    dice = -zsoil[0] * sice[0]
    for k in range(1, len(sice)):
        dice += (zsoil[k - 1] - zsoil[k]) * sice[k]
    return float(dice)


def _call_argument_lists(source: str, callee: str) -> list[list[str]]:
    """Every parenthesised argument list ``callee`` appears with, in order.

    The first is the definition, whose parameter list names the positions;
    the rest are the call sites.
    """
    out: list[list[str]] = []
    for match in re.finditer(rf"\b{re.escape(callee)}\s*(?:<[^<>]+>\s*)?\(", source):
        depth, j = 1, match.end()
        while depth:
            depth += {"(": 1, ")": -1}.get(source[j], 0)
            j += 1
        out.append([a.strip() for a in source[match.end():j - 1].split(",")])
    return out


def _measure(fixture, port, mask) -> dict[str, int]:
    out: dict[str, int] = {}
    for name in OUTPUT_COLUMNS:
        if name not in fixture.reference:
            continue
        got = np.ascontiguousarray(port[name][:, mask], np.float32)
        want = np.ascontiguousarray(fixture.reference[name][:, mask])
        out[name] = int(fp32_ulp_distance(got, want).max())
    for name in SOIL_OUTPUT_COLUMNS:
        got = np.ascontiguousarray(port[name][:, :, mask], np.float32)
        want = np.ascontiguousarray(fixture.soil_reference[name][:, :, mask])
        out[name] = int(fp32_ulp_distance(got, want).max())
    return out


# --------------------------------------------------------------------------
# CPU-only: the fixture is what it says it is.
# --------------------------------------------------------------------------

def test_fixture_is_the_pinned_wrf_driver_and_its_own_receipts():
    """The CSVs must hash to what build.sh recorded next to them."""
    receipts = (NOAH_ORACLE_DIR / "oracle-sha256sums.txt").read_text(
        encoding="ascii").splitlines()
    recorded = {}
    for line in receipts:
        digest, path = line.split(maxsplit=1)
        recorded[path.strip().rsplit("/", 1)[-1]] = digest
    for name in (*NOAH_ORACLE_FILES, *NOAH_GLACIAL_ORACLE_FILES):
        blob = (NOAH_ORACLE_DIR / name).read_bytes()
        assert hashlib.sha256(blob).hexdigest() == recorded[name], name
    # The WRF sources the fixture came from are named in the same receipt, so a
    # fixture regenerated from a different tree cannot pass silently.  So are
    # the three parameter tables, which is what makes this fixture a check on
    # woof's SOIL_VEG_GEN_PARM transcription and not just on the physics.
    joined = " ".join(receipts)
    for expected in ("module_sf_noahdrv.F", "module_sf_noahlsm.F",
                     "module_sf_noahlsm_glacial_only.F", "module_sf_sfcdiags.F",
                     "module_model_constants.F", "module_wrf_error.F",
                     "run_lsm.F90", "run_lsm_glacial.F90",
                     "VEGPARM.TBL", "SOILPARM.TBL", "GENPARM.TBL"):
        assert expected in joined, expected


def test_libmvec_receipt_shows_the_reference_is_scalar_libm():
    """gfortran's vector libm must not be in the reference, and the grep for it
    must be capable of firing -- the report carries its own positive control."""
    report = (NOAH_ORACLE_DIR / "libmvec-report.txt").read_text(encoding="ascii")
    reference, _, rest = report.partition("# -O2")
    assert "_ZGV" not in reference, report
    assert "_ZGVbN4v_expf" in rest, (
        "the positive control produced no vector symbol, so the absence of one"
        " in the -O0 reference proves nothing")
    for symbol in ("U expf", "U powf", "U logf", "U log10f", "U atanf"):
        assert symbol in reference, report


def test_the_four_switch_fixtures_are_not_the_same_file():
    """A switch the fixture cannot discriminate is a switch nothing checks.

    ``noah-lsm-thcnd2.csv`` really was byte-identical to ``noah-lsm.csv``
    until soil types 3 and 4 were added: TDFCND's ``opt_thcnd == 2`` arm
    (``module_sf_noahlsm.F:4173``) is reachable for no other soil type.
    """
    blobs = {name: (NOAH_ORACLE_DIR / name).read_bytes()
             for name in NOAH_ORACLE_FILES}
    base = blobs["noah-lsm.csv"]
    for name, blob in blobs.items():
        if name == "noah-lsm.csv":
            continue
        assert blob != base, (
            f"{name} is byte-identical to noah-lsm.csv: no fixture column"
            " reaches what that switch changes")


def test_fixture_covers_the_branches_that_matter():
    fixture = load_noah_oracle()
    assert fixture.ncase == 80
    # the land-ice population is exactly the columns lsm sends to SFLX_GLACIAL
    assert tuple(int(c) for c in np.asarray(fixture.cases)[fixture.glacial]) \
        == GLACIAL_CASES
    smallest = np.float32(np.finfo(np.float32).smallest_subnormal)
    inputs = {k: v.reshape(-1) for k, v in fixture.inputs.items()
              if v.ndim == 2}
    # signed zeros and subnormals on five different sign compares
    assert (inputs["rainbl"] == smallest).any()
    assert (np.signbit(inputs["rainbl"]) & (inputs["rainbl"] == 0)).any()
    assert (inputs["snow"] == smallest).any()
    assert (inputs["qv1"] == smallest).any()
    assert (np.signbit(inputs["qv1"]) & (inputs["qv1"] == 0)).any()
    assert (np.signbit(inputs["vegfra"]) & (inputs["vegfra"] == 0)).any()
    assert (inputs["canwat"] == smallest).any()
    assert (inputs["chs"] == smallest).any()
    # the three .GT. boundaries in the driver's snow block, straddled exactly
    assert (inputs["sfctmp"] == np.float32(273.15)).any()
    assert (inputs["sfctmp"]
            == np.nextafter(np.float32(273.15), np.float32(1e9))).any()
    assert (inputs["tsk"] == np.float32(273.14)).any()
    assert (inputs["tsk"] == np.float32(273.0)).any()
    assert (inputs["swdown"] == np.float32(10.0)).any()
    # the three columns the driver skips or hands to another routine
    assert (inputs["xland"] >= 1.5).any()
    assert (inputs["xice"] >= 0.5).any()
    assert fixture.glacial.any()
    # soil type 14 at a land point, which WRF rewrites to 7
    isltyp = fixture.isltyp.reshape(-1)
    xice = inputs["xice"]
    assert ((isltyp == 14) & (xice == 0.0)).any()
    # and the only two soil types opt_thcnd can change
    assert (isltyp == 3).any() and (isltyp == 4).any()
    # lsm's CHS2 = CQS2 write moves a word only where they differ on entry
    land = inputs["xland"] < 1.5
    assert (land & (inputs["chs2"] != inputs["cqs2"])).any()
    # SFCDIAGS's CQS2 .LT. 1.E-5 test, straddled
    assert (inputs["cqs2"] == np.float32(1.0e-5)).any()
    assert ((inputs["cqs2"] < np.float32(1.0e-5))
            & (inputs["cqs2"] > np.float32(1.0e-30))).any()


def test_the_baseline_table_names_every_measured_field():
    """A field silently dropped from the table would silently stop being
    gated.  The table and the oracle's field list must be the same set."""
    assert set(BASELINE_MAX_ULP) == set(OUTPUT_COLUMNS) | set(
        SOIL_OUTPUT_COLUMNS)


def test_a_compiler_row_moves_only_what_the_compiler_reaches():
    """A167: an (architecture, build) row may differ from its
    architecture's row only in ARCH_AND_BUILD_SENSITIVE_FIELDS, so a
    compiler row cannot quietly re-baseline a field no compiler moved."""
    for (capability, build), row in BASELINE_MAX_ULP_BY_ARCH_AND_BUILD.items():
        base = BASELINE_MAX_ULP_BY_ARCHITECTURE.get(capability,
                                                    BASELINE_MAX_ULP)
        assert set(row) == set(base), (capability, build)
        moved = {name for name in row if row[name] != base[name]}
        assert moved and moved <= set(ARCH_AND_BUILD_SENSITIVE_FIELDS), (
            capability, build, sorted(moved))


def test_the_mirror_reproduces_wrfs_frozen_ground_infiltration():
    """SRT must spend REDPRM's ``FRZX``, and only WRF can say whether it does.

    ``REDPRM`` builds ``FRZFACT = (SMCMAX/SMCREF)*(0.412/0.468)`` and
    ``FRZX = FRZK*FRZFACT`` (``module_sf_noahlsm.F:2477-2478``), and ``SFLX``
    passes *FRZX* down (``:769``, ``:784``).  The receiving dummy is merely
    *named* ``FRZFACT`` in NOPAC (``:1909``), SNOPAC (``:3015``) and SMFLX
    (``:2670``), and is renamed back to ``FRZX`` in SRT (``:3655``), which
    spends it as ``ACRT = CVFRZ*FRZX/DICE`` (``:3795``).  Handing that chain
    the REDPRM local ``FRZFACT`` instead scales ACRT by ``1/FRZK`` = 6.67,
    ``FCR`` saturates at 1 and the frozen-ground infiltration limiter stops
    limiting.  No self-consistency gate in this tree can see that, because
    ``npref`` and ``noah.cu`` made the same substitution -- which is why this
    is measured against WRF's own word and nothing else.
    """
    params = pack_params(load_tables())
    measured = []
    for name in NOAH_ORACLE_FILES:
        fixture = load_noah_oracle(name)
        for i, case in enumerate(fixture.cases):
            want = float(fixture.reference["sfcrunoff"].reshape(-1)[i])
            # The mirror has no SFLX_GLACIAL; land ice is graded against WRF
            # directly by test_land_ice_columns_are_bitwise_against_wrf.
            if case in HELD_OUT_CASES or case in GLACIAL_CASES or want <= 0.0:
                continue
            if _srt_dice(fixture, i) <= FROZEN_DICE_M:
                continue
            out = np_noah_column(_mirror_column(fixture, i), params,
                                 FIXTURE_DT, FIXTURE_DZS,
                                 isurban=FIXTURE_ISURBAN,
                                 isice=FIXTURE_ISICE,
                                 xice_threshold=FIXTURE_XICE_THRESHOLD,
                                 **fixture.switches)
            measured.append((name, case, out["sfcrunoff"], want))
    for name in NOAH_ORACLE_FILES:
        assert tuple(case for fixture, case, _, _ in measured
                     if fixture == name) == FROZEN_RUNOFF_CASES, (
            f"{name} no longer carries the frozen columns that run off, so"
            " this test cannot see the frozen-ground limiter at all")
    for name, case, got, want in measured:
        assert got == pytest.approx(want, rel=FROZEN_RUNOFF_REL), (
            f"{name} case {case}: mirror SFCRUNOFF {got!r} against WRF's"
            f" {want!r}.  SRT's ACRT is not being formed from REDPRM's FRZX.")


def test_the_kernel_hands_smflx_the_same_word_the_mirror_does():
    """``noah.cu`` inlines NOPAC and SNOPAC, so its ``noah_smflx`` call sites
    ARE ``SFLX``'s ``:769``/``:784``, and the word they pass must be ``frzx``.

    The mirror is graded against WRF above; the kernel cannot be, without a
    device.  This is the part of that claim a source can carry: the argument
    standing in ``noah_smflx``'s ``frzfact`` slot -- named after the Fortran
    dummy, as in WRF -- is the ``frzx`` REDPRM built, at every call site.
    nvcc says the same thing more bluntly: before this was true it reported
    ``variable "frzx" was declared but never referenced``.
    """
    lists = _call_argument_lists(module_source("noah"), "noah_smflx")
    signature, calls = lists[0], lists[1:]
    slot = [p.split()[-1].lstrip("*&") for p in signature].index("frzfact")
    assert len(calls) == 3, (
        f"noah.cu calls noah_smflx {len(calls)} times, not 3 (WRF: NOPAC"
        " :2048 and :2111, SNOPAC :3367); read the new one before repinning")
    # SNOPAC uses the declared zero irrigation rate; NOPAC supplies it.
    # Omitted arguments must be trailing defaults, never a shifted FRZX.
    required = next((i for i, arg in enumerate(signature) if "=" in arg),
                    len(signature))
    for args in calls:
        assert required <= len(args) <= len(signature)
        assert args[slot] == "frzx", (
            f"noah_smflx is called with {args[slot]!r} where WRF's SFLX"
            " passes FRZX (module_sf_noahlsm.F:769, :784)")


# --------------------------------------------------------------------------
# GPU: the measurement itself.
# --------------------------------------------------------------------------

@pytest.mark.gpu
@requires_gpu
def test_noah_cuda_column_holds_its_measured_distance_from_wrf():
    import cupy  # noqa: F401  (marks this test for -m "not gpu")

    measured: dict[str, int] = {}
    for fixture in _fixtures():
        port = noah_port_outputs(fixture)
        one = _measure(fixture, port, _arithmetic_mask(fixture))
        for name, value in one.items():
            measured[name] = max(measured.get(name, 0), value)
    from woof.certify.compile_platform import nvrtc_build

    from woof import wrf_exact

    capability = str(cupy.cuda.Device().compute_capability)
    build = nvrtc_build()
    if wrf_exact.ENABLED:
        recorded = BASELINE_MAX_ULP_BY_ARCH_AND_BUILD.get(
            (capability, build),
            BASELINE_MAX_ULP_BY_ARCHITECTURE.get(capability, BASELINE_MAX_ULP))
    else:
        recorded = DEFAULT_ARITHMETIC_MAX_ULP_BY_ARCH_AND_BUILD.get(
            (capability, build), {})
    assert measured == recorded, (
        "Noah's distance from the unmodified WRF driver changed.\n"
        f"  measured {measured}\n  recorded {recorded}\n"
        f"  on sm_{capability} under NVRTC {build}\n"
        "If a field got worse, something regressed.  If it got better, say so:"
        " update the table in the same commit as the improvement, with the"
        " expression that changed.")


@pytest.mark.gpu
@requires_gpu
def test_case_18_takes_wrfs_snow_branch_on_a_subnormal_snow():
    """A subnormal SNOW selects the ice-saturation blend, as in WRF.

    Under -ftz the kernel's case 18 used to be bit-identical to its own
    snow-free case 1.  WRF's two columns differ, so the port's must differ the
    same way: case 18 is asserted bitwise against WRF in the three fields the
    flush cost, and the WRF half stops this passing on a fixture that lost
    its probe.
    """
    import cupy  # noqa: F401

    fixture = load_noah_oracle()
    port = noah_port_outputs(fixture)
    i1 = fixture.cases.index(1)
    i18 = fixture.cases.index(18)
    assert float(fixture.inputs["snow"].reshape(-1)[i18]) == float(
        np.finfo(np.float32).smallest_subnormal)
    tsk = fixture.reference["tsk"].reshape(-1)
    assert abs(float(tsk[i18]) - float(tsk[i1])) > 0.05
    for field in ("tsk", "hfx", "lh"):
        got = np.ascontiguousarray(port[field], np.float32).reshape(-1)
        want = fixture.reference[field].reshape(-1)
        assert got[i18].tobytes() == want[i18].tobytes(), (
            f"{field}: case 18 {got[i18]!r} against WRF's {want[i18]!r}; the"
            " subnormal SNOW compare is flushing again")
        assert got[i18].tobytes() != got[i1].tobytes(), field


@pytest.mark.gpu
@requires_gpu
def test_case_25_is_unusable_in_wrf_too():
    """A subnormal exchange coefficient destroys the column in WRF as well.

    This is not a port defect and must not be recorded as one, but the two
    NaN patterns are not the same and that is worth pinning: WRF loses the
    soil state, the kernel keeps it, and WRF's QFX is an uninitialised read
    (see "Held-out column" above).  QSFC is NaN in WRF; the port's is NaN
    under default arithmetic, where the flush to zero makes RCH zero, and
    finite under GPUWM_WRF_EXACT=1, so it is not asserted.
    """
    import cupy  # noqa: F401

    fixture = load_noah_oracle()
    port = noah_port_outputs(fixture)
    i25 = fixture.cases.index(25)
    for field in ("tsk", "hfx", "lh", "grdflx", "potevp"):
        assert not np.isfinite(fixture.reference[field].reshape(-1)[i25]), field
        assert not np.isfinite(
            np.ascontiguousarray(port[field], np.float32).reshape(-1)[i25]), field
    # WRF's soil state is NaN at this case and the port's is finite.  The two
    # asserts pin that difference as measured; neither column is claimed right.
    assert not np.isfinite(fixture.soil_reference["smois"][0].reshape(-1)[i25])
    assert np.isfinite(
        np.ascontiguousarray(port["smois"], np.float32)[0].reshape(-1)[i25])


@pytest.mark.gpu
@requires_gpu
def test_land_ice_columns_are_bitwise_against_wrf():
    """SFLX_GLACIAL, measured on its own population: every field, every
    land-ice column, every switch fixture, to the word lsm wrote."""
    import cupy  # noqa: F401

    measured: dict[str, int] = {}
    for fixture in _fixtures():
        mask = _glacial_mask(fixture)
        assert int(mask.sum()) == len(GLACIAL_CASES), fixture.name
        port = noah_port_outputs(fixture)
        for name, value in _measure(fixture, port, mask).items():
            measured[name] = max(measured.get(name, 0), value)
    assert measured == GLACIAL_MAX_ULP, (
        "noah_glacial_column moved from WRF's SFLX_GLACIAL.\n"
        f"  measured {measured}")


@pytest.mark.gpu
@requires_gpu
def test_sfcdiags_after_the_column_is_bitwise_against_wrf():
    """T2, TH2 and Q2: WRF's SFCDIAGS on lsm's words against the driver's
    noah_sfcdiags kernel on the port's words, on every graded column.

    Q2 is compared where WRF's Q2 is positive.  Where it is not (case 78:
    CQS2 exactly 1.E-5 under daytime evaporation, WRF Q2 = -6.46 kg/kg), the
    port publishes qv(k=1) by design -- the bound
    ``PhysicsDriver._refresh_surface_diagnostics`` documents -- so that column
    must carry qv(k=1) exactly, and there must be one, or this cannot see the
    bound at all.  tools/noah_wrf461_oracle/sfcdiags_sweep.py grades the same
    kernel on every float32 surface pressure from 20 to 110 kPa.
    """
    import cupy  # noqa: F401

    from woof import wrf_exact
    from woof.certify.compile_platform import nvrtc_build

    if wrf_exact.ENABLED:
        allowed = {name: 0 for name in SFCDIAGS_COLUMNS}
    else:
        key = (str(cupy.cuda.Device().compute_capability), nvrtc_build())
        allowed = SFCDIAGS_DEFAULT_MAX_ULP_BY_ARCH_AND_BUILD.get(key, {})
    floors = 0
    for fixture in _fixtures():
        port = noah_port_outputs(fixture)
        diags = noah_sfcdiags_outputs(fixture, port)
        keep = np.asarray([c not in HELD_OUT_CASES for c in fixture.cases])
        want_q2 = fixture.sfcdiags_reference["q2"].reshape(-1)
        floor = keep & (want_q2 <= 0.0)
        floors += int(floor.sum())
        for name in SFCDIAGS_COLUMNS:
            got = np.ascontiguousarray(diags[name], np.float32).reshape(-1)
            want = fixture.sfcdiags_reference[name].reshape(-1)
            graded = keep & ~floor if name == "q2" else keep
            distance = fp32_ulp_distance(got[graded], want[graded])
            assert int(distance.max()) <= allowed.get(name, 0), (
                f"{fixture.name} {name}: max {int(distance.max())} ULP "
                f"(recorded {allowed.get(name)})")
        qv1 = fixture.inputs["qv1"].reshape(-1)
        got_q2 = np.ascontiguousarray(diags["q2"], np.float32).reshape(-1)
        assert got_q2[floor].tobytes() == qv1[floor].tobytes(), fixture.name
    assert floors == len(NOAH_ORACLE_FILES), floors


# --------------------------------------------------------------------------
# The 24-column land-ice fixtures (run_lsm_glacial.F90).
# --------------------------------------------------------------------------

def test_the_land_ice_and_urban_categories_follow_the_dataset():
    """ISURBAN and ISICE are the land-use dataset's, not MODIS's for everyone.

    WRF hands its Noah driver the input file's ISURBAN/ISICE
    (share/input_wrf.F:849-862).  The driver here used to pass 13 and 15 for
    every dataset, so under USGS -- where 13 is evergreen broadleaf forest,
    15 is mixed forest, 1 is urban and 24 is snow or ice -- the mixed forest
    was skipped as land ice and the real ice ran ordinary SFLX.  Measured on
    the float64 mirror with the USGS tables: category 15 runs SFLX (skip 0)
    and category 24 runs SFLX_GLACIAL (skip 3); with the MODIS tables the
    same two categories do the opposite.
    """
    from woof.core.noah import NOAH_LANDUSE_CATEGORY_IDS, noah_category_ids

    assert noah_category_ids(pack_params(load_tables())) == (13, 15)
    usgs = pack_params(load_tables(mminlu="USGS"))
    assert noah_category_ids(usgs) == (1, 24)
    fixture = load_noah_oracle()
    base = _mirror_column(fixture, fixture.cases.index(1))
    expect = {"USGS": {15: 0, 24: 3}, "MODIFIED_IGBP_MODIS_NOAH": {15: 3}}
    for dataset, cases in expect.items():
        params = pack_params(load_tables(mminlu=dataset))
        for category, skip in cases.items():
            out = np_noah_column(dict(base, ivgtyp=category), params,
                                 FIXTURE_DT, FIXTURE_DZS)
            assert out["skip"] == skip, (dataset, category, out["skip"])
            assert out["tsk"] != base["tsk"], (
                dataset, category, "the column was left untouched")
    # every dataset VEGPARM.TBL carries has an identity, and an unknown one
    # is refused by name rather than run with somebody else's categories
    for dataset in ("USGS", "MODIFIED_IGBP_MODIS_NOAH", "NLCD40", "USGS-RUC",
                    "MODI-RUC"):
        assert dataset in NOAH_LANDUSE_CATEGORY_IDS
    # (the two -RUC blocks carry RUC's column layout and are not Noah's to
    # load; the three below are)
    for dataset in ("USGS", "MODIFIED_IGBP_MODIS_NOAH", "NLCD40"):
        noah_category_ids(pack_params(load_tables(mminlu=dataset)))
    import dataclasses
    stranger = dataclasses.replace(usgs, lutype="NOT-A-DATASET")
    with pytest.raises(ValueError, match="ISURBAN/ISICE"):
        noah_category_ids(stranger)


def test_the_glacial_fixtures_reach_the_branches_they_claim():
    """Every column is land ice, the two passes differ, and WRF's own words
    show the branches were taken: melt (SFCRUNOFF > 0), snowfall (SNOW up),
    deposition (LH < 0), sublimation (LH > 0), the entry reset (SNOW set to
    100 mm from less), snow aging (ALBEDO below the fresh-snow value), the
    SNOWZ0 burial arm (ZNT above Z0S) and a skin pinned at TFREEZ."""
    plain, frpcpn = _glacial_fixtures()
    assert (NOAH_ORACLE_DIR / "noah-lsm-glacial.csv").read_bytes() != (
        NOAH_ORACLE_DIR / "noah-lsm-glacial-frpcpn.csv").read_bytes()
    for fixture in (plain, frpcpn):
        assert fixture.ncase == 24 and fixture.glacial.all()
        ref = {k: v.reshape(-1) for k, v in fixture.reference.items()}
        snow_in = fixture.inputs["snow"].reshape(-1)
        assert (ref["sfcrunoff"] > 0).sum() >= 3
        assert (ref["snow"] > np.maximum(snow_in, 100.0)).sum() >= 3
        assert (ref["lh"] < 0).sum() >= 3 and (ref["lh"] > 0).sum() >= 10
        assert ((snow_in < 100.0) & (ref["snow"] == np.float32(100.0))).sum() >= 2
        assert (ref["albedo"] < ref["albedo"].max()).sum() >= 2
        assert (ref["znt"] > np.float32(0.001)).any()
        assert (ref["tsk"] == np.float32(273.15)).sum() >= 3
    # SR = 0.3 at case 6 and SR = 0.8 at case 23 send the frpcpn pass down
    # the other precipitation arm
    for case in (6, 23):
        i = plain.cases.index(case)
        assert plain.reference["snow"].reshape(-1)[i] != \
            frpcpn.reference["snow"].reshape(-1)[i], case


def test_the_mirror_reproduces_wrfs_land_ice_columns():
    """The float64 mirror's SFLX_GLACIAL against WRF's float32 words, on every
    land-ice column of every fixture -- the CPU half of the port's grade."""
    params = pack_params(load_tables())
    checked = 0
    for fixture in (*_fixtures(), *_glacial_fixtures()):
        for i in np.flatnonzero(fixture.glacial.reshape(-1)):
            out = np_noah_column(_mirror_column(fixture, i), params,
                                 FIXTURE_DT, FIXTURE_DZS,
                                 isurban=FIXTURE_ISURBAN,
                                 isice=FIXTURE_ISICE,
                                 xice_threshold=FIXTURE_XICE_THRESHOLD,
                                 **fixture.switches)
            assert out["skip"] == 3, "the mirror skipped a land-ice column"
            for field in OUTPUT_COLUMNS:
                if field == "noahres" or field not in fixture.reference:
                    continue
                want = float(fixture.reference[field].reshape(-1)[i])
                got = float(out[field])
                assert abs(got - want) <= GLACIAL_MIRROR_REL * max(
                    abs(want), 1.0e-3), (fixture.name, fixture.cases[i],
                                         field, got, want)
            for field in SOIL_OUTPUT_COLUMNS:
                want = fixture.soil_reference[field].reshape(4, -1)[:, i]
                np.testing.assert_allclose(
                    np.asarray(out[field], np.float64), want,
                    rtol=GLACIAL_MIRROR_REL, atol=0.0,
                    err_msg=f"{fixture.name} case {fixture.cases[i]} {field}")
            checked += 1
    assert checked == len(NOAH_ORACLE_FILES) * len(GLACIAL_CASES) + 2 * 24


@pytest.mark.gpu
@requires_gpu
def test_noah_cuda_glacial_fixtures_are_bitwise_against_wrf():
    """``noah_glacial_column`` on the 24-column land-ice fixtures of
    ``run_lsm_glacial.F90`` (snowfall at both density laws, freezing rain,
    melt, the dew adjustment's three arms, sublimation, snow aging, SNOWZ0
    both sides of its burial test, both SNOWPACK clamps), run with and
    without ``frpcpn``: every field these fixtures carry, to the word lsm
    wrote."""
    import cupy  # noqa: F401

    measured: dict[str, int] = {}
    for fixture in _glacial_fixtures():
        port = noah_port_outputs(fixture)
        one = _measure(fixture, port, np.ones(fixture.ncase, dtype=bool))
        for name, value in one.items():
            measured[name] = max(measured.get(name, 0), value)
    assert measured == GLACIAL_FIXTURE_MAX_ULP, (
        "noah_glacial_column moved from WRF's SFLX_GLACIAL on the land-ice "
        f"fixtures.\n  measured {measured}")
