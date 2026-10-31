"""The mp_physics=8 freeze gate -- WP-00 of the mp_physics=28 port.

This was the merge criterion for every other package on the
``feature/mp28-thompson-aerosol-aware`` branch.  The port is landed and
shipped; what the file guards from now on is stated under RE-FROZEN
2026-09-17 below, and ``tools/release/precut_gpu_gate.py`` runs it on the
release node before every cut.

WHAT IT PROVES, AND WHY THAT IS ENOUGH
--------------------------------------
The mp=8 non-regression argument is a MECHANISM, not a measurement.
``woof/core/kernels/__init__.py::load_module`` builds exactly one
``cupy.RawModule`` per ``.cu`` file from a single source string.  If that
string is byte-identical then the PTX is identical, therefore the register
allocation and FP contraction are identical, therefore every mp=8 result is
bit-identical.  No PTX-diff tooling is needed, and none should be relied on:
a PTX gate is a test that fails late and can be "fixed" by relaxing it.

So the primary assertion here is a digest of the *assembled* source string
-- captured by driving the REAL loader with a recording ``RawModule``, not
by re-implementing it -- for every ``.cu`` translation unit that existed at
the frozen commit.  That catches an edit to ``thompson.cu``, an edit to
``common.cuh``, a change to ``CUDA_DEFINES``, a change to ``_preamble()``,
and any loader change that is not perfectly inert.  WP-02's
``_EXTRA_HEADERS`` allow-list is exactly such a change, and this gate is
what holds it to its inertness claim.

Around the mechanism sit six receipts (R1..R6), each a real failure mode
that would otherwise produce a model that runs, stays stable and is
silently wrong -- the specific hazard of this port, where a half-converted
prognostic ``nc`` never raises anything.

R1  source identity of thompson.cu / thompson.py / every frozen module.
R2  the classic table contract (CCN_ACTIVATE.BIN must never appear in it).
R3  ``extra_moist_species`` -- Morrison's deliberate ``nc`` exclusion.
R4  the preflight allocation and scratch-arena surface.
R5  the nest-transition edge field codes (28 APPENDED, never inserted).
R6  ``acoustic`` n_mass selection plus the recorded ``_apply_thompson``
    launcher call graph, argument by argument.

Plus two fixture receipts: F1 freezes the 92 committed mp=8 oracle CSVs,
and F2 pins the four-file clean-rebuild exception together with a hermetic
witness for its cause (see :mod:`tools.mp8_freeze_receipt`).

New mp=28 files are expected and permitted everywhere: the gate pins what
existed and ignores additions.  It never passes because something was
deleted -- every pinned name must still be present.

RE-FROZEN 2026-09-17 (lane/2.7.6-pin-gates), AND WHAT A RED DIGEST MEANS
------------------------------------------------------------------------
The mp=28 port is landed and shipped, so "mp=28 has not touched mp=8" is
no longer a merge criterion anyone waits on.  What the digests still
catch is an edit to an mp=8 unit that ships without a reading of what it
did to mp=8 results.  4ae7913df (fix(mp8): preserve rain concentration
and condensation history, in 2.7.4) is the case in point: it edited
thompson.cu, thompson.py and the adapter's launcher arguments, shipped
its reading (quoted below), and did NOT re-freeze here, so seven
assertions in this file were red on the 2.7.4 tip 7417342a8 and the
2.7.5 tip c36f4c1f1 while the release contract set, which runs no GPU
pin, never ran them.  Confirmed on test host A at fc639c51f: 7 failed, 14
passed, 1 skipped.  tools/release/precut_gpu_gate.py now runs this file
before a cut.

Decision per assertion, under the gate law (a gate names the concrete
breakage it prevents) and the retire-its-guards law:

* KEPT, re-frozen at 4ae7913df with its reading: the thompson.cu file
  digest, the assembled compile-string digest, the 65-module digest
  table, the thompson.py digest and launch inventory, the constant-Nt_c
  site inventory (all R1) and the recorded launcher call graph (R6).
  The breakage each prevents from now on: an mp=8 result change shipped
  with no reading.  A red digest therefore means "a frozen unit was
  edited by a commit that shipped no reading of its own": the fixer
  records one (a fixture, a test and a page, as 4ae7913df did) and
  re-freezes here citing it.  Each failure text says so.
* KEPT unchanged, their breakage still exists and is named in each
  docstring: the preamble / common.cuh / CUDA_DEFINES pins (a constant
  change moves every scheme silently), the loader inertness check, R2
  the classic table contract, R3 extra_moist_species, R4 the allocation
  surface and the scratch arena aliasing contract, R5 the edge field
  codes, R6 n_mass, F1 the 92 oracle CSVs, F2 the four-file rebuild
  exception and its inverted witness.
* RETIRED: none.  Every assertion names a breakage still possible.

THE READING 4ae7913df SHIPPED, quoted from its fixture, test and page.
tests/fixtures/thompson-active-collision.json holds three complete
classic column calls (saturated, subsaturated, supersaturated: 274.15 K,
80000 Pa, qc 0.001, qr 0.0003, qs 0.0002, qg 0.0002 kg/kg, nr 30000/kg,
dt 10 s) generated through the pinned WRF v4.6.1 driver at
d66e442fccc04111067e29274c9f9eaccc3cef28 by
tools/thompson_wrf461_oracle/active_collision_fixture.py.  After one
call the saturated column's lowest level goes qr 3.000e-04 to 3.331e-04,
nr 30000 to 30450.35, qc 1.000e-03 to 9.525e-04, theta 292.1977 to
292.1955 K, with surface rainncv 0.025264 mm, snowncv 0.004562 mm and
graupelncv 0.011440 mm; the subsaturated column, where rain evaporation
runs, goes qr 3.000e-04 to 2.968e-04, nr 30000 to 29332.79, qc 1.000e-03
to 4.002e-04, theta 292.1977 to 290.7510 K, rainncv 0.023742 mm; the
supersaturated column goes qr to 3.331e-04, nr to 30450.46, qc to
1.218e-03, theta to 292.8959 K, rainncv 0.025276 mm.
tests/test_thompson_active_collision.py holds the CUDA adapter to those
columns at rtol 1e-5 and atol 3e-11 on the mass fields, rtol 1e-5 and
atol 0.03 on the number fields, 2 ULP of the field on theta and rtol
1e-5, atol 5e-8 on the surface totals, on both the output-due and the
ordinary route, and pins the two mechanisms directly: the rain density
formed before cloud adjustment is refreshed only when evaporation runs
(rho against 0.99 rho at rtol 3e-7), and a positive condensation marker
leaves rain, number, temperature and vapour bit-identical through the
same-call evaporation.  docs/thompson-active-collision-accounting.md
records the source-stage ledger residual of about -9.13 and -10.27 J/kg
for the two ten-second source cases, present in the reference too, and
that warm and cold collision arithmetic and the canonical tables are
unchanged.  The CHANGELOG 2.7.4 entry is the one-line form.

RE-FROZEN 2026-09-24 by the WRF v4.6.1 real-column repairs, and the
reading they shipped.  thompson.cu, thompson.py and _apply_thompson moved
in thirteen commits (ce303c3e5, 217e84e18, 3c1b51317, 08f1f9373,
6bd61312c, c4f3fcc70, 4d6e551ef, e39adb262, 8a504a23a, a9b054c2a,
9298324cc, 3b57369fe, 7727fda3c), each a WRF v4.6.1 rule the classic
port did not follow, cited to module_mp_thompson.F in the kernel.  The
reading is tools/thompson_real_column_parity --mp 8, which runs WRF's own
Fortran beside the port's kernels compiled for the host on 137,200
columns of seven saved real-data states: before the repairs 54,390
(forecast) and 35,290 (analysis) process-rate cells and 100,744 and
58,594 final-state cells differed from WRF beyond 1e-2 unexplained by
rounding, and the echo by up to 43.9 dB; after them no rate does except
the two rounding decides, 1 and 6 final-state cells do (rounding
residues), and the echo is within 0.045 dB.  The committed classic
fixture tests/data/thompson_real_columns_wrf461_mp8.npz holds that in
tests/test_thompson_real_column_host_parity.py (no rate and no
final-state quantity beyond 1e-2 unexplained, echo within 0.05 dB, exit
temperature within 1e-5).  The 92 classic oracle CSVs (F1) are unchanged;
tools/thompson_real_column_parity/README.md and
docs/public/wrf-comparison/mp28-column-evidence.md say what closed and what
remains.

The source-capture fixture opens no device. It uses the actual loader with
recording compiler boundaries and a synthetic CUDA cache identity.
"""
# Replaced in 2.9.1 (the WRF exact chain): test_clean_oracle_rebuild_matches_except_the_four_documented_files -> test_clean_oracle_rebuild_reproduces_the_committed_oracle.
# Replaced in 2.9.1 (the WRF exact chain): test_rebuild_exception_list_is_exactly_four_named_files -> test_rebuild_exception_list_is_retired_and_the_libc_allowance_is_one_file.

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from conftest import requires_cupy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import mp8_freeze_receipt as freeze          # noqa: E402

# ==========================================================================
# PINS.  RE-ANCHORED at the 1.4.1 merge, and the rule is unchanged: nothing
# on this branch may move a value below, and a diff that has to edit this
# file has by definition changed mp=8.
#
# The pins were captured from an orphan snapshot of ArWen 1.3.1
# (789f61181fb0b198ace10775f3ea184eb5e786a3), the last tree before any
# mp_physics=28 work.  That tree is 371 commits behind the release line the
# port now sits on, and ten of the pinned kernels moved on that line while
# the port was off it: eight noahmp_* translation units, rrtmg_sw, and
# thompson itself, the last by 5e4af4e3 ("the rain MVD bound belongs to
# TAU+1") and cb765336 ("the rain-presence gate is a mass concentration").
# SCRATCH_SLOT_REGISTRY_MP8 gained physics_validation_status the same way.
#
# Re-pinning a freeze is exactly how a freeze gets defeated, so the new
# values were not taken on trust.  Every one of the ten was verified to be
# BYTE-IDENTICAL to integration/release-1.4.1 -- `git show
# integration/release-1.4.1:woof/core/kernels/<name>.cu | sha256sum`
# equals `git show HEAD:...` for all ten -- before its digest was written
# here.  The compiled-string digests are safe for the same reason at one
# remove: PREAMBLE_SHA256 and COMMON_CUH_SHA256 below did NOT move, and
# kernels/__init__._EXTRA_HEADERS lists only the six mp=28 translation
# units, so _extra_header_text() returns "" for every frozen module and
# their assembled sources are still exactly _preamble() + source.
#
# The freeze therefore still asserts what it always asserted: mp=28 has not
# touched mp=8.  What it no longer asserts is that mp=8 never changes --
# it does, on its own line, in its own commits, and this branch inherits
# those by merging rather than by editing.
# ==========================================================================

#: The release-line commit these pins were re-captured against.  The
#: original capture point, 789f61181fb0b198ace10775f3ea184eb5e786a3, is an
#: orphan snapshot with no parent and no descendant; it cannot be diffed
#: against the current line, which is why the anchor moved.
FROZEN_COMMIT = "b15a2558a569c661e148c0db3cfc99896f3af91a"
FROZEN_COMMIT_ORIGINAL = "789f61181fb0b198ace10775f3ea184eb5e786a3"

# -- R1 --------------------------------------------------------------------

#: RE-FROZEN 2026-09-17 at 4ae7913df's thompson.cu (fix(mp8): preserve
#: rain concentration and condensation history), whose reading is
#: tests/fixtures/thompson-active-collision.json, tests/test_thompson_
#: active_collision.py and docs/thompson-active-collision-accounting.md;
#: the docstring's "RE-FROZEN" section quotes it.  Previously
#: 3ca6b7e9/8cb23f0a (340875 chars) from the 1.4.1 re-anchor.
#: RE-FROZEN 2026-09-24 at 7727fda3c's thompson.cu, the last of the
#: thirteen real-column repairs the docstring's second reading lists.
#: Previously 938bf573/f9b8547f (343097 chars) from 4ae7913df.  The
#: compile string is the loader's preamble plus the file (thompson takes no
#: extra header): measured without cupy by the receipt's reconstruction,
#: which at 4ae7913df's bytes gives the loader-captured f9b8547f exactly.
#: RE-FROZEN 2026-09-30 by the speed change that adds Thompson's exact
#: shortcuts (an empty column's fallout and an empty cell's radii take a
#: short path with the full path's writes) and the level-parallel fallout
#: kernels.  No answer moved: the kernels' bit tests
#: (tests/test_thompson_speed_shortcuts.py) and three 1 h forecasts, mp=8
#: product and convective and mp=28 convective, are byte-identical to
#: d6929cb8d on an RTX 4090 (the product forecast and the oracle also on an
#: RTX 5090).  Previously d77977dc/e2ea3185.
#: RE-FROZEN 2026-09-30 by the fused classic adapter launches
#: (thompson_adapter_prepare / _entry / _masks / _finish), which share
#: thompson_level_has_microphysics and thompson_classic_graupel_number with
#: the two kernels they replace in the adapter.  No answer moved:
#: tests/test_thompson_speed_glue.py and the lane's three 1 h forecasts are
#: byte-identical to d6929cb8d on an RTX 4090, and the oracle and the
#: product forecast on an RTX 5090.  Previously d0997611/fd8a053a.
#: RE-FROZEN for A146 on top of those: thompson.cu's constant-divisor float
#: divisions spelled __fdiv_rn, because NVRTC compiles x / C as a multiply by
#: the rounded reciprocal on Blackwell targets. Previously a7955418.
#: Re-frozen again by the A146 review repair: seven more divisions by a
#: local constant (am_r, am_i, am_g) in the speed lanes' fused kernels,
#: which the compute_89 census found and the source scan could not see.
#: RE-FROZEN 2026-10-01 for the cold network register bound and smaller
#: cold/warm launch blocks. tests/test_thompson_speed_blocks.py compares
#: raw output words against the unbounded source and former launch widths.
#: Direct mixed and dense sweeps on RTX 4090 and RTX 5090 are byte-identical
#: for the admitted bound. More aggressive bounds failed the RTX 5090 check.
#: RE-FROZEN 2026-10-08 (gfix/288-thompson-classic): every classic
#: rain-graupel table read goes through thompson_racg_index, the one slab WRF
#: builds, instead of WRF v4.6.1's idx_bg1=5 subscript (+4*37*37 words, and
#: zero rates where that left the table): the declared divergence mp=28
#: already carries.  Reading: tests/test_thompson_real_column_host_parity.py
#: grades the 42 real columns on WRF with only that index corrected
#: (thompson_real_columns_wrf461_mp8_corrected_racg.npz) with no cell beyond
#: the gate, and the unchanged stock fixture now shows exactly the
#: rain-graupel footprint (74 prr_rcg, 80 pnr_rcg, 74 prg_rcg, 55 png_rcg,
#: 7 prw_vcd cells, one final qg); on sm_89 only the six reading kernels'
#: SASS moved.  Previously 1ab8d031/b3bbf88b (428981 chars).
THOMPSON_CU_SHA256 = (
    "43f76b17febd32acdfe96e9cca4d9c932ce2d72c5a5378ed73b5f3eac33998d8")
#: sha256 of ``_preamble() + thompson.cu`` -- the exact string nvrtc sees.
#: THIS is the mp=8 numerics guarantee.
THOMPSON_COMPILED_SOURCE_SHA256 = (
    "35ac29fc801c66fc1eb64506267ded1efa2374f0998b0f80eea72f2afd5aa34a")
THOMPSON_COMPILED_SOURCE_LEN = 428156

COMMON_CUH_SHA256 = (
    "c78b17cb02ef67a2ad24d19e06e1129d7d5bcda74b972b38470fd33a6e58ff43")
PREAMBLE_SHA256 = (
    "1888bcf077e4251398b3baf7cca53b7c15ba993ab791992b7e278c3e32bb75e8")
PREAMBLE_LEN = 768

CUDA_DEFINES_PIN = {
    "G": 9.81, "RD": 287.0, "RV": 461.6, "CP": 1004.5, "CV": 717.5,
    "P0": 100000.0, "T0": 300.0, "GAMMA": 1.4,
    "RCP": 0.2857142857142857, "RCV": 0.4,
    "RVOVRD": 1.6083624362945557, "RERADIUS": 1.5698587127158556e-07,
    "XLV": 2500000.0, "SVP1": 0.6112, "SVP2": 17.67, "SVP3": 29.65,
    "SVPT0": 273.15, "RHOWATER": 1000.0, "EP2": 0.6217504332755632,
}

#: RE-FROZEN 2026-09-17 at 4ae7913df's thompson.py (launch_rain_evaporation
#: gained source_density and condensation_marker, launch_cloud_saturation_
#: adjust gained condensation_marker; __all__ is unchanged).  Previously
#: ede422fe.
#: RE-FROZEN 2026-09-24 at 7727fda3c's thompson.py: 217e84e18 added
#: launch_microphysics_columns (WRF's per-column no_micro flag, :2020) to
#: __all__, and the other moving commits gave existing launchers new
#: keyword inputs (cloud_presence, density_carries_rain_presence,
#: melt_rain_density, micro_columns).  Previously 6e446a46 from 4ae7913df.
#: RE-FROZEN 2026-09-30: the fallout launchers send columns of at most 64
#: levels to the level-parallel kernels and the cold source launches 128
#: threads a block; __all__ is unchanged and no answer moved.  Previously
#: b952306f.
#: RE-FROZEN 2026-09-30: launch_adapter_prepare, _entry, _masks and
#: _finish join __all__ (the fused classic adapter launches); no answer
#: moved.  Previously 9e004667.
#: RE-FROZEN 2026-10-01: the cold network uses 64-thread blocks; the warm
#: network retains 256 on device architecture majors 9 and 10, and 64 on
#: the others. tests/test_thompson_speed_blocks.py checks geometry and bits.
#: RE-FROZEN 2026-10-08 (gfix/288-thompson-classic): launch_adapter_finish
#: clamps at DTYPE(mp_tend_lim) * DTYPE(dt), the REAL product WRF forms
#: (module_big_step_utilities_em.F:5706), not the rounded double product;
#: a clamped cell's heating moved one float32 unit to WRF's word
#: (tests/test_thompson_slab1_oracle_gpu.py), and launch_rain_graupel_
#: collection's docstring names the one-slab read.  __all__ is unchanged.
#: Previously 8b091e96.
#: RE-PINNED for WOOF 1.0.0: the rename rewrote the package name on 7
#: import lines of woof/core/thompson.py and nothing else, so no launcher and no
#: keyword moved.  As the engine froze it: 00f661f0.
THOMPSON_PY_SHA256 = (
    "eff18d896df4bae1cfe1523255a4ccf67bf8f0810c4edfed4181fe69d3a37dae")

#: ``woof/core/thompson.py::__all__`` verbatim, in declaration order.
#: mp=28 launchers live in the new ``thompson_aerosol_*.py`` modules; not
#: one name may be added here.
THOMPSON_PY_ALL = (
    # RE-PINNED 2026-09-30: the four fused classic adapter launches.
    "launch_adapter_entry",
    "launch_adapter_finish",
    "launch_adapter_masks",
    "launch_adapter_prepare",
    "launch_cloud_freezing",
    "launch_cloud_saturation_adjust",
    "launch_cloud_sedimentation",
    "launch_classic_graupel_number_init",
    "launch_classic_graupel_number_finalize",
    "launch_cold_cloud_source_network",
    "launch_cold_rain_source_network",
    "launch_cold_rain_snow_graupel_network",
    "launch_effective_radius",
    "launch_final_phase_cleanup",
    "launch_frozen_vapor_network",
    "launch_frozen_vapor_network_from_owner",
    "launch_graupel_cloud_riming",
    "launch_graupel_fallout_column_mask",
    "launch_hydrometeor_column_mask",
    "launch_microphysics_columns",
    "launch_graupel_sedimentation",
    "launch_graupel_melting",
    "launch_graupel_sublimation",
    "launch_ice_autoconversion",
    "launch_ice_deposition",
    "launch_ice_nucleation",
    "launch_ice_sedimentation",
    "launch_rain_evaporation",
    "launch_rain_freezing",
    "launch_rain_graupel_collection",
    "launch_rain_ice_collection",
    "launch_rain_snow_collection",
    "launch_rain_sedimentation",
    "launch_rain_self_collection",
    "launch_snow_sublimation",
    "launch_snow_cloud_riming",
    "launch_snow_ice_collection",
    "launch_snow_melting",
    "launch_snow_rime_conversion",
    "launch_snow_sedimentation",
    "launch_snow_vapor_exchange",
    "launch_warm_autoconversion",
    "launch_warm_process_network",
    "launch_warm_frozen_source_network",
    "launch_warm_frozen_source_network_from_owner",
    "launch_warm_rain_collection",
    "launch_warm_saturation_adjust",
)

#: The constant-Nt_c inventory, MEASURED on the frozen tree.  Every line
#: here hardcodes what mp=28 must make prognostic.  Recorded so that a
#: reader can see the port never "fixed" one of them in place, and so
#: that the claim in the port spec is checkable rather than asserted.
#:
#: NOTE, and this corrects the spec's summary paragraph: the measured
#: counts are 13 / 6 / 2 / 3, not 12 / 7 / 3.  ``thompson.cu:343`` is the
#: one CORRECT 2730.0f -- ``calc_effectRad`` genuinely uses WRF's integer
#: ``g_ratio`` PARAMETER there -- and the 272.0f site is line 1007, not
#: 1006.
#:
#: RE-FROZEN 2026-09-17: 4ae7913df added lines to thompson.cu and every
#: site below moved down by 22 (through old line 2094) or 49 (from old
#: line 2903); the counts are still 13 / 6 / 2 / 3 and no site was
#: edited.  Previously 343, 895, 1012, 2094, 2903, 2970, 3187, 3793, 4018,
#: 4141, 4273, 4693, 6945 / 343, 895, 1012, 4018, 4141, 4693 / 901, 1020 /
#: 3946, 4359, 7018.
#:
#: RE-FROZEN 2026-09-24: the real-column repairs added lines to
#: thompson.cu above every site; the counts are still 13 / 6 / 2 / 3 and
#: no site was edited.  Previously 365, 917, 1034, 2116, 2952, 3019, 3236,
#: 3842, 4067, 4190, 4322, 4742, 6994 / 365, 917, 1034, 4067, 4190, 4742 /
#: 923, 1042 / 3995, 4408, 7067.
#: RE-FROZEN 2026-09-30: the level-parallel held-density cloud fallout
#: appended to thompson.cu repeats its column kernel's 100.0e6f, 2730.0f
#: and 272.0f line, and the exact shortcuts added lines above most sites;
#: the counts are 14 / 7 / 3 / 3 and no existing site was edited.
#: RE-FROZEN 2026-09-30: the fused adapter kernels added lines above the
#: level-parallel fallout; counts 14 / 7 / 3 / 3, no site edited.
#: RE-FROZEN 2026-10-01: the network bound adds seven lines above later
#: sites. Literal counts and numerical values are unchanged.
#: RE-FROZEN 2026-10-08: thompson_racg_index added 23 lines above the
#: later sites and the six one-slab reads removed lines below them; counts
#: 14 / 7 / 3 / 3 and no site edited.
THOMPSON_CU_LITERAL_SITES = {
    '100.0e6f': [416, 1136, 1265, 2572, 3446, 3513, 3762, 4437, 4717, 4840, 4972, 5392, 7658, 9631],
    '2730.0f': [416, 1136, 1265, 4717, 4840, 5392, 9631],
    '272.0f': [1142, 1273, 9639],
    'cloud_number_bin = 65': [4645, 5058, 7740],
}

#: Every ``.cu`` translation unit present at the frozen commit, as
#: ``name -> (file sha256, assembled-compile-string sha256)``.  Modules
#: added later (the mp=28 ones) are ignored by the gate; a pinned name
#: that disappears is a failure.
#: Re-ratified 2026-10-01 for the opt-in WRF verification additions below.
#: The default-off six-hour INTEGRATION-928e60 run matches FIX-CAP-ALBBCK:
#: all 155,877,414 stored values and all seven history files are identical.
#: Retained readings: WRF-EXACT-2026-10-01/evidence/INTEGRATION-928e60/
#: stored-words-vs-FIX-CAP-ALBBCK.json and
#: file-byte-identity-vs-FIX-CAP-ALBBCK.json; source 928e60e7c.
#: This ratifies source text, not an aggregate R1 fixture replay.
FROZEN_MODULE_DIGESTS = {
    # Eight dycore units below were re-anchored to the byte-identical
    # integrated c2946a54e artifacts on 2026-10-01. Their raw and assembled
    # source hashes were read separately through module_source(). This is
    # source identity metadata, not a new numerical or forecast result:
    # the named compiled-WRF and arithmetic-control tests retain that
    # authority. No kernel arithmetic moved in this re-anchoring.
    # CONUS strict MF_VX_INV repair: imported inverse replaces a rebuilt reciprocal.
    # Full native history at steps1/2/3: 0/265734210 words each, no missing rows;
    # receipts conus-suite-exact/mp-rad-lsm/receipts/sol-continuation.
    # Re-pinned for the WOOF 1.0.0 text scrub: comments only (punctuation, host
    # labels), proven equal with comments stripped; the preprocessor drops
    # comments, so the compiled binary does not move.
    'acoustic': (
        # Re-pinned for the WPHI_MAX_LEV tier ladder (LES program P2): the
        # unconditional `#define WPHI_MAX_LEV 129` became an `#ifndef`
        # guard around the same literal, so the launcher can compile the
        # implicit w''-phi'' solve deeper than 128 levels.  acoustic is not
        # an mp=8 translation unit and the mp=8 numerics guarantee is
        # untouched.  The guard is a preprocessor no-op at the shipped
        # tier -- proven against a real host preprocessor, with a negative
        # control, in tests/test_acoustic_nz_tiers.py -- so this pin moves
        # while the compiled binary does not.
        # Re-pinned for A146 (a98f2482e): every float division by a compile-time
        # constant is spelled __fdiv_rn, because NVRTC compiles x / C as a
        # multiply by the rounded reciprocal on Blackwell targets.  Reading:
        # sm_89 unchanged (the compute_89 PTX with __fdiv_rn spelled '/' is
        # identical to the base's); Blackwell cards now round these quotients
        # IEEE-correctly.  Previously 76628563/9c8836bf.
        # Re-pinned for the compiled WRF v4.7.1 small-step oracle (46b0a09fe,
        # merged e30268fc9): the w-damping pi/2 literal was 3FC90FDA where
        # WRF's is 3FC90FDB.  Reading: compiled-WRF controls reproduce every
        # W/PH word except a damping-sine residue of at most 1 ULP
        # (tests/test_smallstep_vertical_wrf471_parity.py).  Previously 1ec40899/a7054926.
        # Re-pinned for the opt-in strict WRF arithmetic (lane/282-wrf-exact,
        # merged 1efb5a415): 85c636a13 (feat(verify): reproduce WRF acoustic
        # arithmetic order) adds `#if GPUWM_WRF_EXACT` branches and an
        # exact-frame kernel inside them, which the preprocessor keeps only when
        # a GPUWM_WRF_EXACT selector defines the macro.  Reading: with no
        # selector, module_source('acoustic') at f3e4ca716 and at c2946a54e
        # compiles to byte-identical PTX, the whole text and 14 of 14 .entry
        # kernels, for compute_89, compute_90 and compute_120 under the loader's
        # options and under the RawModule options CuPy compiles with, with NVRTC
        # 13.4 and with 12.9 (tools/kernel_ptx_identity/receipts/,
        # wrf-exact-default-2.8.2-nvrtc13.4.json and -nvrtc12.9.json), so the
        # default build does not move.  Previously fa154649/d790562d.
        # Re-frozen for bandwidth scheduling and column workspace reuse.
        # Recorded native vertical words are reproduced on their original
        # RTX 5090 card; complete default forecast
        # histories and canonical state remain byte-identical.
        # The small-step vertical and big-step momentum receipts
        # record the current assembled source and card evidence.
        # Re-pinned for lane/verify-base4 (combo sweep round 3): strict-only
        # operation order inside `#if GPUWM_WRF_EXACT` -- calc_coef_w's top row with rdnw(kde-1)**2 first, advance_uv's external-mode term inside the u/v statement, spec_bdyupdate_ph's (field*num)/den and WRF's muts carrier on the specified ring.
        # Reading: with no selector the default acoustic PTX is byte-identical to
        # bc2ce86b/2356c7c8 for compute_89/90/120 under both option sets
        # (tools/kernel_ptx_identity/receipts/wrf-exact-round3-default.json); the
        # strict words are graded in tests/test_wrf_exact_round3_gpu.py and against
        # WRF's own dumped words (tools/wrf_exact_localize).  Previously
        # bc2ce86b/2356c7c8.
        # Re-pinned for the CONUS strict MF_VX_INV repair (paper/conus-suite-exact 0e21ecf29a, folded in
        # 2.9.11): advance_uv's and apply_emdiv's GPUWM_WRF_EXACT branches and wrf_total_divergence take the
        # imported MF_VX_INV instead of a rebuilt reciprocal.  The row moved the file pin and left the
        # assembled-source pin at 22704608; this is the assembled digest of the same bytes.  Reading: the
        # 2.9.11 assembled-legacy replay (base 2.9.10 b8165c59fe vs f5e393fb8c, RTX 5070 Ti) reproduces all
        # 132 published acoustic arrays / 305,632 words and the coordinate and driver families with 0
        # differing words on the default route; strict words are the row's own CONUS receipts.
        '14f8d637ecfa54b7b8c19d80fc2bbf5ad5566f6fdc57b9671d0cfa8566f4331b',
        '5c7205005dfa7457f8e27bc1f636a751a297f78710b03bfd5920256de3d2d57c'),
    # Re-pinned for the WOOF 1.0.0 text scrub: comments only (punctuation, host
    # labels), proven equal with comments stripped; the preprocessor drops
    # comments, so the compiled binary does not move.
    'advection': (
        # Re-pinned for A146 (a98f2482e): every float division by a compile-time
        # constant is spelled __fdiv_rn, because NVRTC compiles x / C as a
        # multiply by the rounded reciprocal on Blackwell targets.  Reading:
        # sm_89 unchanged (the compute_89 PTX with __fdiv_rn spelled '/' is
        # identical to the base's); Blackwell cards now round these quotients
        # IEEE-correctly.  Previously 8a88c2fc/3449e3bc.
        # Re-pinned for the compiled WRF v4.7.1 advection oracle (dc226e820,
        # merged dde6ea687): w advection at the lid carries the top-level flux
        # compiled WRF has, and the mapped open-boundary radiation kernels include
        # the map-factor divisor.  Reading: 4 of 6 routines ULP-bounded on 8
        # fixtures on the 4090 and H100 (tests/test_advect_wrf471_parity.py).
        # Previously 3c64071b/10dcae55.
        # Re-pinned for the opt-in strict WRF arithmetic (lane/282-wrf-exact,
        # merged 1efb5a415): 2029c5726 (feat(verify): reproduce active WRF
        # advection and limiter order) adds `#if GPUWM_WRF_EXACT_C_ADVECTION`
        # branches, which the preprocessor keeps only when a GPUWM_WRF_EXACT
        # selector defines the macro.  Reading: with no selector,
        # module_source('advection') at f3e4ca716 and at c2946a54e compiles to
        # byte-identical PTX, the whole text and 4 of 4 .entry kernels, for
        # compute_89, compute_90 and compute_120 under the loader's options and
        # under the RawModule options CuPy compiles with, with NVRTC 13.4 and
        # with 12.9 (tools/kernel_ptx_identity/receipts/,
        # wrf-exact-default-2.8.2-nvrtc13.4.json and -nvrtc12.9.json), so the
        # default build does not move.  Previously 963e5870/50cc068e.
        # Re-pinned for the bandwidth tuning (lane/282-bw-advection 8c69d26ab,
        # 48cfb1008; merged 7486d241c): in the default build flux_div_scalar
        # and flux_div_v wrap periodic indices with one signed remainder
        # instead of two; the WRF-exact build keeps PERIODIC.  Index arithmetic
        # only, no floating expression, reduction order or compile option
        # moved, but the PTX of those two entries does on compute_89/90/120
        # (tools/kernel_ptx_identity/receipts/, bw-advection-2.8.2-
        # nvrtc13.4.json and -nvrtc12.9.json).  Reading: the product default
        # suite (8 steps, 64x64x32) is byte-identical, canonical state digest
        # and history file, on test host D's RTX 5070 Ti at 9eee49ce0 against
        # 7486d241c; the lane's report (DYCORE-SPEED-282/advection-RESULT.md)
        # records forecast identity on B200, H100 and RTX PRO 6000 (3 km
        # CONUS) and on the RTX 4090 and RTX 5090 (default suite); and the
        # compiled-WRF advect oracle reproduces every recorded word on the
        # RTX 4090 and on sm_120 (tools/advect_wrf471_oracle/README.md).
        # Previously 6facc4f8/66e08b7a.
        # Re-pinned for lane/286-vadv5 (WRF v_sca_adv_order / v_mom_adv_order):
        # the four flux-divergence entries take a trailing int vorder, a
        # vertical flux5v (WRF flux5 with -vel) joins flux3, zface_half and
        # advect_w's five vertical-face blocks run WRF's vert_order == 5
        # ladder under vorder 5 and the unchanged vert_order 3 ladder under
        # vorder 3, which every launcher defaults to.  Reading: the compiled
        # WRF 4.7.1 advect oracle reproduces every recorded production word
        # on sm_120 at vorder 3 (tests/data/wrf471_advect receipts recaptured
        # with only the kernel digests moved), and at vorder 5 the HRRR-fork
        # oracle fixture (tests/data/wrf_legacy_advect, NOAA-EMC/HRRR
        # 40ee6058c WRFV3.9) holds the four explicit routines bitwise under
        # the exact-C build (tests/test_advect_wrf_legacy_parity.py).
        # Previously 85062bec/e3fa6135.
        '1f9cd387e5e20469ce7905065a11b367110ec635adb0efe331dce3ae450a98bf',
        'ba500e338d41859b88351ca8014921381c2a33a78b9d56ec66d351a3ccdc073c'),
    # Re-pinned for the WOOF 1.0.0 text scrub: comments only (punctuation, host
    # labels), proven equal with comments stripped; the preprocessor drops
    # comments, so the compiled binary does not move.
    'coriolis_map': (
        # Re-pinned for the opt-in strict WRF arithmetic (lane/282-wrf-exact,
        # merged 1efb5a415): d1252eb2a (feat(verify): restore WRF big-step and
        # boundary arithmetic) adds `#if GPUWM_WRF_EXACT_C_BIGSTEP` branches,
        # which the preprocessor keeps only when a GPUWM_WRF_EXACT selector
        # defines the macro.  Reading: with no selector,
        # module_source('coriolis_map') at f3e4ca716 and at c2946a54e compiles
        # to byte-identical PTX, the whole text and 1 of 1 .entry kernels, for
        # compute_89, compute_90 and compute_120 under the loader's options and
        # under the RawModule options CuPy compiles with, with NVRTC 13.4 and
        # with 12.9 (tools/kernel_ptx_identity/receipts/,
        # wrf-exact-default-2.8.2-nvrtc13.4.json and -nvrtc12.9.json), so the
        # default build does not move.  Previously 53fc3739/79813843.
        # Re-frozen for bandwidth scheduling and column-workspace reuse.
        # Recorded native words are reproduced on their original
        # RTX 4090 and RTX 5090 cards; complete default forecast
        # histories and canonical state remain byte-identical.
        # The small-step vertical and big-step momentum receipts
        # record the current assembled source and card evidence.
        'd49df6c23a90ed645e260c5eeba5d6d8569acd2e2f10a8d378effaea51548ea7',
        '30a7c0778f9df73efee7e8014d9fdce650bd7e4c3b955ef70ccbd0b9b53d5b1b'),
    'diagnostics': (
        # Re-pinned for the two-way feedback landing (88fdf60b9,
        # "bitwise-gated" by its own suite).  Not an mp=8 unit.
        # Re-pinned again for the EOS spelling change: calc_p_alpha stopped
        # forming the layer geopotential thickness by differencing two
        # ~2.4e4 J/kg totals and stopped writing opt 2's log ratio as
        # log(pfd/pfu).  No physics moved -- the same two quantities are
        # computed by a spelling that does not cancel.  Measured against
        # the float64 mirror, relative error in p on a random 1 K state:
        # opt 1, 2.06e-6 -> 3.64e-7 at nz=16 and 2.89e-5 -> 4.13e-7 at
        # nz=160; opt 2, 4.25e-6 -> 4.54e-7 and 1.15e-4 -> 5.49e-7.  The
        # error stopped tracking 1/dz, which is what
        # tests/test_diagnostics.py::
        # test_eos_error_does_not_scale_with_vertical_resolution pins.
        # Re-pinned for the opt-in strict WRF arithmetic (lane/282-wrf-exact,
        # merged 1efb5a415): 270038669 (feat(verify): retain native WRF
        # diagnostic and pressure words) adds `#ifdef
        # GPUWM_WRF_EXACT_D_DIAGNOSTICS` branches, the base pressure and
        # perturbation-pressure arguments among them, which the preprocessor
        # keeps only when a GPUWM_WRF_EXACT selector defines the macro.
        # Reading: with no selector, module_source('diagnostics') at f3e4ca716
        # and at c2946a54e compiles to byte-identical PTX, the whole text and 1
        # of 1 .entry kernels, for compute_89, compute_90 and compute_120 under
        # the loader's options and under the RawModule options CuPy compiles
        # with, with NVRTC 13.4 and with 12.9
        # (tools/kernel_ptx_identity/receipts/,
        # wrf-exact-default-2.8.2-nvrtc13.4.json and -nvrtc12.9.json), so the
        # default build does not move.  Previously 384bf67c/eb691c93.
        # Re-pinned for lane/verify-base4 (combo sweep round 3): strict-only
        # operation order inside `#if GPUWM_WRF_EXACT` -- calc_p_rho_phi with the acoustic loop's grid%muts as an argument (a helper and a calc_p_alpha_carried_muts entry).
        # Reading: with no selector the default diagnostics PTX is byte-identical to
        # 7b348c87/f3da804e for compute_89/90/120 under both option sets
        # (tools/kernel_ptx_identity/receipts/wrf-exact-round3-default.json); the
        # strict words are graded in tests/test_wrf_exact_round3_gpu.py and against
        # WRF's own dumped words (tools/wrf_exact_localize).  Previously
        # 7b348c87/f3da804e.
        # Re-pinned for c61736da9 (Pin nest preparation EOS to host FP32
        # arithmetic, folded into 2.9.10 as eced9efd6c): calc_p_alpha's log1pf
        # and powf are spelled EOS_LOG1P/EOS_POW, which become the pinned
        # plm_log1pf/gfk_pow only under GPUWM_PREPARATION_EOS (the nest
        # preparation loader) and stay log1pf/powf everywhere else.  Reading:
        # the 2.9.10 GPU suite at a613224e3a (this edit included) kept all six
        # default-forecast identity arms identical to the f0195d80ac1b
        # reference, and CPU and CUDA nest preparation agree in 0 of
        # 18,170,276 values on the 00Z ICON-EU moving-nest case.  Previously
        # 38ee5c02/4418fe6f.
        'f611da1651af581955fcb8bf3e0b2b12cd8c8f11058e555194c091bbb478b0b7',
        '40e9e6cb687a54a8f1489aa6d92aa496ed6a646e3a506b4a34030dcbd5d00a16'),
    'diff6': (
        # Re-pinned for the compiled WRF v4.7.1 diffusion oracle (83fde6032,
        # merged dd4908a0f): sixth-order filtering supplies each field's
        # projection map factor, couples hybrid mass before face averaging and
        # keeps WRF's REAL staging in the main and seam kernels.  Reading: zero
        # differing words over 376,160 against compiled WRF on the 4090 and 5090
        # (tests/test_diff6_wrf471_parity.py).  Previously 7dbcfb2d/563febbc.
        '0e0de46ac99d264eef27a50ac74e6e92fd68374ebd42698aceb4e1a4808a7467',
        'be41f8092a63933c6165f67e59f03afaa89bacbd6ffbf1a5ddcef367f38e4bcc'),
    'diff6_seam': (
        # Re-pinned for the compiled WRF v4.7.1 diffusion oracle (83fde6032,
        # merged dd4908a0f): sixth-order filtering supplies each field's
        # projection map factor, couples hybrid mass before face averaging and
        # keeps WRF's REAL staging in the main and seam kernels.  Reading: zero
        # differing words over 376,160 against compiled WRF on the 4090 and 5090
        # (tests/test_diff6_wrf471_parity.py).  Previously 776ed705/7af0e3dd.
        'b693000a64ccfece2e2cdd45756091c7d200f55e1d136f5a79ab548fe22706e9',
        '90c48b98a08fa87fc1cabfe41cb0ba7c6ef52e74817f4502c08d494ecacfb905'),
    'diffusion': (
        '00fb2e5d5550680fef154b4f67c7e282ad7ca1b170df59abdea89f888dad91ef',
        'f4958de3298bfcd764a5fd848aedfbb13938043ab91132db419408951c0a061e'),
    # CONUS strict MF_VX_INV repair: imported inverse replaces a rebuilt reciprocal.
    # Full native history at steps1/2/3: 0/265734210 words each, no missing rows;
    # receipts conus-suite-exact/mp-rad-lsm/receipts/sol-continuation.
    'dycore': (
        # Re-pinned for the small-step lateral-boundary fix:
        # small_step_init_uv / small_step_finish_uv took WRF calc_mu_uv's
        # periodic_x branch unconditionally, so a specified / nested / open
        # domain drew its west boundary face mass from the EAST-most mass
        # column.  They now take boundary_x/boundary_y and clamp when the
        # axis is not periodic.  A PERIODIC run is bit-identical -- the two
        # branches differ only at i = 0 / i = nx and j = 0 / j = ny -- and
        # that is measured, not argued: tilestream/test_gate.py's whole
        # output (51 physics cases, dry through full physics + MYNN +
        # Noah-MP, plus every negative control) is byte-identical across the
        # change, as is the whole-inventory SHA-256 after 1 and 8 steps on
        # three periodic rungs.  mp=8 is periodic-agnostic and dycore is not
        # an mp=8 translation unit, so the mp=8 numerics guarantee is
        # untouched; see tests/test_small_step_lateral_wrap.py.
        # Compiled WRF v4.7.1 rhs_ph oracle proves that outer open and
        # specified rows skip the entire normal-direction term. Both GPU
        # and mirror previously retained an interior half-face term.
        # test_rhs_ph_specified_corner_has_no_horizontal_advection is
        # red on that former behavior; periodic corpus words are unchanged.
        # Re-pinned for the opt-in strict WRF arithmetic (lane/282-wrf-exact,
        # merged 1efb5a415): 85c636a13, 1df1d50e3 and d1252eb2a
        # (feat/refactor(verify): WRF acoustic order, default acoustic tokens,
        # big-step arithmetic) add `#if GPUWM_WRF_EXACT`,
        # `GPUWM_WRF_EXACT_C_BIGSTEP` and `GPUWM_WRF_EXACT_D_DIAGNOSTICS`
        # branches, which the preprocessor keeps only when a GPUWM_WRF_EXACT
        # selector defines the macro.  Reading: with no selector,
        # module_source('dycore') at f3e4ca716 and at c2946a54e compiles to
        # byte-identical PTX, the whole text and 9 of 9 .entry kernels, for
        # compute_89, compute_90 and compute_120 under the loader's options and
        # under the RawModule options CuPy compiles with, with NVRTC 13.4 and
        # with 12.9 (tools/kernel_ptx_identity/receipts/,
        # wrf-exact-default-2.8.2-nvrtc13.4.json and -nvrtc12.9.json), so the
        # default build does not move.  Previously 450bb509/e318baa1.
        # Re-frozen for bandwidth scheduling and column-workspace reuse.
        # Recorded native words are reproduced on their original
        # RTX 4090 and RTX 5090 cards; complete default forecast
        # histories and canonical state remain byte-identical.
        # The small-step vertical and big-step momentum receipts
        # record the current assembled source and card evidence.
        # The SFIRE port (lane/ec-sfire, from lane/283-sfire 2d1de25d6)
        # adds the compiled open-boundary geopotential kernel
        # slow_geopotential_open, launched only for open, unforced lateral
        # boundaries (woof/core/dycore.py _launch_open_geopotential); every
        # other entry is unchanged. Native RHS controls and the coupled ideal
        # timeline receipts are in tools/sfire_coupled_ideal/rhs_ph_open/receipts.
        # Re-pinned for lane/sfire-do1-287 (2.8.7): slow_geopotential_open
        # follows unmodified WRF v4.7.1 rhs_ph at the west top row, where WRF
        # reads FNM/FNP(kde) and U(kde), which hold their zero allocation
        # value at run time; the former min(k,kde-1) correction added a
        # nonzero upwind term WRF never applies.  Reading: 0 differing words
        # of 43,008 against compiled unmodified WRF (no allowed-difference
        # mask), default and strict, RTX 5070 Ti; the previous kernel misses
        # 192 (the west top row of the 16 open-x fixtures).  Previously
        # a9567a3d/3d47bada.
        # Re-pinned for lane/verify-base3 (combo sweep round 2): strict-only
        # operation order inside `#if GPUWM_WRF_EXACT` -- calc_php's
        # left-to-right half-level geopotential, pg_buoy_w's (1./msfty)*g
        # scale, and small_step_finish's muus from muts = (mub+mu)+mu''.
        # Reading: with no selector the default dycore PTX is byte-identical
        # to dc89c74d/2b6f1337 for compute_89/90/120 under both option sets
        # (tools/kernel_ptx_identity/receipts/
        # wrf-exact-round2-dycore-default.json); the strict words are graded
        # in tests/test_wrf_exact_operation_order_gpu.py.  Previously
        # dc89c74d/2b6f1337.
        # Re-pinned for lane/verify-base4 (combo sweep round 3): strict-only
        # operation order inside `#if GPUWM_WRF_EXACT` -- calc_mu_uv's face mass in rhs_ph and horizontal_pressure_gradient, calc_cq's species-pair qtot in pg_buoy_w, and small_step_finish reading the muts carrier.
        # Reading: with no selector the default dycore PTX is byte-identical to
        # 6fac2471/af786cd2 for compute_89/90/120 under both option sets
        # (tools/kernel_ptx_identity/receipts/wrf-exact-round3-default.json); the
        # strict words are graded in tests/test_wrf_exact_round3_gpu.py and against
        # WRF's own dumped words (tools/wrf_exact_localize).  Previously
        # 6fac2471/af786cd2.
        # Assembled pin re-read for 2.9.11 (paper/conus-suite-exact 0e21ecf29a): the row's strict MF_VX_INV
        # edit sits inside `#if GPUWM_WRF_EXACT`; it moved the file pin above and left this one stale.
        # Default route unchanged: the 2.9.11 assembled-legacy replay, 0 differing words in all three families.
        '1983f89c3d44c723b8c322e8cccb611fade5ec3830c39d9002be60eda6ea8223',
        '5d66c9b385d0fae7429db2786f9316322cf4993a2d260ca33b7a7cf849bc70b9'),
    'health': (
        # The full-state scanner now carries specified-rim geometry. It
        # changes a strict boundary residual's verdict only, never a model
        # word. NumPy/native/CUDA corruption twins keep interior floors,
        # finiteness and ceilings; NCAR default files stay byte-identical.
        # RECOMPUTED at the tilestream port, over the MERGED health.cu that
        # carries both re-pins below.  The pin that arrived with the port
        # was the integration side's alone and described a source that does
        # not exist in this tree, so it was red by construction and a pass
        # would have been coincidence; both rationales are kept because both
        # edits are present in the merged source.  The two values are the
        # file's own sha256 and the sha256 of the assembled compile string,
        # both computed from text: no card and no nvrtc were involved, which
        # is why this recompute closed on the CPU shard.
        #
        # --- from v1.8.7 ---
        # Re-pinned for the validation-gate launch geometry: blockIdx.y now
        # selects the descriptor and blockIdx.x a chunk within it, so the
        # full-state scan is no longer one block per field.  health is not
        # an mp=8 translation unit -- it computes no model state at all, it
        # only reads state and writes the compact health record -- so the
        # mp=8 numerics guarantee is untouched.  The record itself is
        # unchanged by construction: the epilogue is the same atomicOr over
        # status bits and the same atomicMin over (field << 48) | index,
        # both order-independent, so which block visits an element cannot
        # change what is reported (tests/test_health.py).
        #
        # --- from integration ---
        # Re-pinned for the STREAMED SAFETY FOLD (defect2-observer-fold).
        # health.cu gained one kernel, health_partial_tile: the per-tile half
        # of stability_report, emitted after a tile's step and before its
        # interior is scattered, so that under [tiles] with a host store
        # the run loop's nan / w_max / CFL gate reduces over the STORE rather
        # than over a DomainState the sweep never writes.  ADDITIVE: nothing
        # that was in this translation unit changed, health_partial and
        # health_final are byte-identical, and mp=8 does not compile health
        # at all, so the mp=8 numerics guarantee is untouched.  The fold's
        # own proof is tilestream/test_obsfold.py; the whole-gate evidence is
        # that tilestream/test_gate.py passes 233/233 across the change,
        # negative controls included.
        '7a49b886f883cc95453ac30ee4d4ebf30ef347968df5317991fadf141054da9c',
        'a512b39fd3d468d2fbecdbbabc92a263e403e2fb0e826723ca15dea5e4bff7c7'),
    # Re-pinned for the WOOF 1.0.0 text scrub: comments only (punctuation, host
    # labels), proven equal with comments stripped; the preprocessor drops
    # comments, so the compiled binary does not move.
    'kessler': (
        # Re-pinned for A146 (a98f2482e): every float division by a compile-time
        # constant is spelled __fdiv_rn, because NVRTC compiles x / C as a
        # multiply by the rounded reciprocal on Blackwell targets.  Reading:
        # sm_89 unchanged (the compute_89 PTX with __fdiv_rn spelled '/' is
        # identical to the base's); Blackwell cards now round these quotients
        # IEEE-correctly.  Previously fecf2e80/530faef7.
        #
        # RE-PINNED by lane/verify-kessler-mp1: the six REAL**REAL powers
        # and the EXP call WOOF's own gfk_pow/gfk_exp (glibc_flt32.cuh now
        # prepended by _EXTRA_HEADERS, which moves the compiled string), the
        # unit compiles --fmad=false, and qr = (qr + prod - qr) is evaluated
        # as the Fortran writes it.  Reading: bitwise against the
        # byte-unmodified WRF v4.6.1 kessler on 104 columns x 25 calls,
        # tests/test_kessler_wrf461_parity.py.  Not an mp=8 unit.
        # Previously 80856aca/f6f6aa8d.
        #
        # RE-PINNED by lane/verify-kessler-mp1-refute: es is capped at
        # 0.99 p before qvs (WOOF's guard for the WRF defect where es >= p
        # makes qvs negative and the top layers heat without bound).
        # Reading: bitwise against WRF v4.6.1 on every word where
        # es <= 0.98 p across the lane's fixture and five new column sets,
        # and bounded (WRF: 4,435 K) where it engages,
        # tests/test_kessler_refute_columns.py.  Previously 864f27a2/dc6cd6a4.
        '437e750df30338848cc44b6c4a386839ba910551a2093c356228702a87017a7e',
        'cfa356099db905a10a2080a1dd9841c663ca1a4642fc8257898464d5d530df04'),
    # Re-pinned for the WOOF 1.0.0 text scrub: comments only (punctuation, host
    # labels), proven equal with comments stripped; the preprocessor drops
    # comments, so the compiled binary does not move.
    'kf': (
        # Re-pinned for the column-workspace move (48ff6b813), measured
        # on test host A with its two declared bit-moving placements named in
        # the commit.  Not an mp=8 unit.
        #
        # RE-PINNED by the WRF-parity shallow-TIMEC repair (PAR-CU-KF-05).
        # module_cu_kfeta.F:1598-1600 sets TIMEC=2400. for the shallow arm
        # and then ROUNDS it to FLOAT(NINT(TIMEC/DT))*DT; :2571-2573 sets
        # TIMEC = 2400. again immediately before the feedback loop, and the
        # six tendencies at :2603-2640 divide by that un-rounded value.  The
        # kernel divided all six by the rounded one.  Everything WRF keeps
        # rounded stays rounded here: the closure, AINCMX/AINC, DTIME/DTT/
        # NSTEP, the TADVEC comparison (:2569) and TIMEC_KF (:2387).  This
        # MOVES ANSWERS on shallow KF columns whenever DT does not divide
        # 2400 -- the pinned case is DT=90, where the rounded TIMEC is 2430
        # and every shallow tendency was 1.23% low (1 - 2400/2430).  kf is
        # not an mp=8 translation unit and thompson.cu is byte-unchanged.
        # Measured against the float64 mirror by tests/test_kf.py::
        # test_shallow_feedback_tendencies_divide_by_an_unrounded_2400.
        #
        # RE-PINNED 2026-09-30 by the default-pieces speed lane (08a1eed93):
        # KF launches eight-warp blocks and orders each block's columns so
        # the ones its trigger test predicts to convect share warps; order
        # only, every column runs the whole scheme on its own workspace
        # lane, and the output zero-fills the kernel overwrites are gone.
        # No answer moved: the lane's A/B dumps (sm_120, KF every step) and
        # a 1 h real HRRR default-suite forecast (sm_89) are byte-identical
        # to d6929cb8d.  Not an mp=8 unit.
        # Re-pinned for A146 on top of that: every float division by a
        # compile-time constant is spelled __fdiv_rn, because NVRTC compiles x
        # / C as a multiply by the rounded reciprocal on Blackwell targets.
        # Reading: sm_89 unchanged; Blackwell cards now round these quotients
        # IEEE-correctly. Previously 30308b1b/7d81bfc9.
        #
        # RE-PINNED 2026-10-07 (lane/verify-kain-fritsch): kf.cu is now a
        # line-for-line port of WRF v4.6.1 KF_eta_PARA and compiles on its
        # own --fmad=false --ftz=false route with WOOF's own float32 libm
        # (the assembled source gains glibc_flt32.cuh).  This MOVES ANSWERS
        # on every convecting KF column: the re-derivation it replaces
        # matched WRF's Fortran on none of 1,793 convecting columns; the
        # port matches it on every output word of 4,326 columns (structured,
        # sweep and ERA5 real74 sets), strict and default builds,
        # tests/test_kf_wrf461_oracle.py and tools/kf_wrf461_oracle.  Not an
        # mp=8 unit.  Previously 705dded6/0ef80d84.
        #
        # RE-PINNED 2026-10-12 (golden/parity): the port keeps d547dc8c3's
        # floor on PPTFLX (fmaxf(TRPPT-TDER, 0)), the one statement where it
        # leaves WRF on purpose.  Reading: no answer moves on any WRF oracle
        # column -- the committed structured/real74 sets, the hex
        # capped-downdraft set and the 4,096/32,768-column seeded sweeps have
        # no negative PRATEC in WRF's own Fortran, and the GPU grade of the
        # committed sets stays 0 differing words in both builds
        # (tests/test_kf_wrf461_oracle.py, tests/test_kf.py).
        # Previously 9eb33736/5e418a4f; then 78cf5577/0a8949d9 until one
        # word of the header comment changed for the banned-word census
        # (no statement moved).
        'be4fadd73ce14d639552bc3584c82367a0e1207a82a27d98a0a5946c9a83e38e',
        'f65f52d6b7464baefead86116ef88e269b347c9a4bd54771e8331e9316e26084'),
    'lbc_flow': (
        # 4febd041f supplies resolved WDM6/NSSL inflow concentrations.
        # docs/dev/qnn-specified-inflow.md records 4 CPU + 10 GPU
        # edge/corner, velocity and scalar transport controls.
        '68a743950e30e308676fada38f96ea3139d283447028edfeb85d2d64c36441fa',
        '233391c535605271d70b32dc5ce85321cb0ed633340dae3fe2cd3ebcf08ab6ba'),
    # CONUS strict MF_VX_INV repair: imported inverse replaces a rebuilt reciprocal.
    # Full native history at steps1/2/3: 0/265734210 words each, no missing rows;
    # receipts conus-suite-exact/mp-rad-lsm/receipts/sol-continuation.
    'lbc_state': (
        # Re-pinned for parent-window and child-frame coupling enumeration.
        # Full-field entry points and per-element expressions stay intact.
        # Re-pinned for the per-side relaxation mask:
        # state_specified_relaxation takes relax_sides, and
        # a streamed tile clears the bit of each interior seam so a
        # relaxation zone sized in parent cells cannot reach owned cells.
        # A whole domain passes 15, where every branch condition reduces
        # to the previous one and the two new j-neighbour clamps are
        # inert (j >= 2 and j <= ny - 3 there), so no whole-domain bit
        # moves.  lbc_state is not an mp=8 translation unit.
        #
        # RE-PINNED: frame_point and frame_offset list the middle ring of
        # an odd side once.  A streamed tile window narrower than two
        # relaxation zones listed that line twice and two threads raced on
        # each of its cells.  A whole domain never has such a ring
        # (_validate_frame_domain), so every ring there keeps two rows and
        # two columns, the enumeration is the previous one entry for entry,
        # and no whole-domain bit moves.
        # Re-pinned for the opt-in strict WRF arithmetic (lane/282-wrf-exact,
        # merged 1efb5a415): d1252eb2a (feat(verify): restore WRF big-step and
        # boundary arithmetic) adds `#if GPUWM_WRF_EXACT_C_BIGSTEP` branches
        # whose `#if !` arms keep the default lines, which the preprocessor
        # keeps only when a GPUWM_WRF_EXACT selector defines the macro.
        # Reading: with no selector, module_source('lbc_state') at f3e4ca716 and
        # at c2946a54e compiles to byte-identical PTX, the whole text and 7 of 7
        # .entry kernels, for compute_89, compute_90 and compute_120 under the
        # loader's options and under the RawModule options CuPy compiles with,
        # with NVRTC 13.4 and with 12.9 (tools/kernel_ptx_identity/receipts/,
        # wrf-exact-default-2.8.2-nvrtc13.4.json and -nvrtc12.9.json), so the
        # default build does not move.  Previously 138ac171/67fec8de.
        # Re-pinned for the bandwidth tuning (lane/282-bw-advection 8c69d26ab,
        # 48cfb1008; merged 7486d241c): in the default build
        # finalize_state_field reads the installed boundary value first and
        # couples the old target only where no boundary value replaces it,
        # so a specified cell no longer loads and couples a value that is
        # then overwritten; the WRF-exact build keeps the original order.
        # No floating expression, reduction order or compile option moved,
        # but the PTX of finalize_state_field does on compute_89/90/120 (the
        # other six entries are identical; tools/kernel_ptx_identity/
        # receipts/, bw-advection-2.8.2-nvrtc13.4.json and -nvrtc12.9.json).
        # Reading: the product default suite (8 steps, 64x64x32) is
        # byte-identical, canonical state digest and history file, on
        # test host D's RTX 5070 Ti at 9eee49ce0 against 7486d241c, and the lane's
        # report (DYCORE-SPEED-282/advection-RESULT.md) records forecast
        # identity on B200, H100 and RTX PRO 6000 (3 km CONUS) and on the
        # RTX 4090 and RTX 5090 (default suite).  Previously 05f074d8/c6c21f26.
        # Re-pinned for lane/verify-base4 (combo sweep round 3): strict-only
        # operation order inside `#if GPUWM_WRF_EXACT` -- relax_bdytend_core's (tend+fcx*f0)-gcx*(...) order, calc_mu_uv/couple_momentum coupling of ru/rv, and spec_bdy_final on the specified rows only from grid%muts.
        # Reading: with no selector the default lbc_state PTX is byte-identical to
        # 27ad19be/e2e4f809 for compute_89/90/120 under both option sets
        # (tools/kernel_ptx_identity/receipts/wrf-exact-round3-default.json); the
        # strict words are graded in tests/test_wrf_exact_round3_gpu.py and against
        # WRF's own dumped words (tools/wrf_exact_localize).  Previously
        # 27ad19be/e2e4f809.
        # Assembled pin re-read for 2.9.11 (paper/conus-suite-exact 0e21ecf29a): the row's strict MF_VX_INV
        # edit sits inside `#if GPUWM_WRF_EXACT`; it moved the file pin above and left this one stale.
        # Default route unchanged: the 2.9.11 assembled-legacy replay, 0 differing words in all three families.
        'b35120f7012a89b7e8b3a0da7e30ce434ae3867e1ff35fea4665e3f645a16949',
        '1730a72093ba5b64c04aada3d8ebc94fc7c0a40ace6b1685e5fd3a04b405cdaf'),
    # Re-pinned for the WOOF 1.0.0 text scrub: comments only (punctuation, host
    # labels), proven equal with comments stripped; the preprocessor drops
    # comments, so the compiled binary does not move.
    'morrison': (
        # MOVER: the deposition-freezing cold-trap bound, carried to the
        # engine line from lane/level5-owner 0c54221d2.  UNLIKE the two
        # sedimentation rearrangements this pin previously recorded, this
        # one DOES change FP results, and it is a DECLARED DIVERGENCE from
        # WRF, not a transcription repair: WRF F:2902-2905 sets MNUCCD
        # with no vapor-availability test, and the F:3009-3015 FUDGEF
        # rescale tests only the two matching sign pairs, so a positive
        # MNUCCD over a subsaturated state is never rescaled.  Below the
        # 159.4887 K POLYSVP crossover measured on this transcription the
        # extrapolated liquid curve falls under the ice curve, F:1315
        # clamps QVI==QVS, and the 0.999*QVS trigger fires at or below ice
        # saturation: the unfixed kernel drove qv to -1.87e-4 kg/kg out of
        # 5.55e-8 kg/kg available at 156 K / 250 Pa, dt-independent at
        # dt = 10/50/300/900 s.  Neither WRF nor a p_top-limited regional
        # domain reaches that state.  The bound ALSO engages where WRF can
        # reach -- 44 of 227 MNUCCD executions over the 28-column WRF
        # oracle fixture, 189.88-199.96 K, number moment scaled 0.197-0.898
        # -- moving case 13's ni on 16 levels (67% relative), qi on 23
        # levels (1.4e-3) and qv at 1.8e-16.  On the sm_86 device NO pinned
        # per-field max_ulp moves and the overall 1,709,094,255 / sr is
        # unchanged; the fixture mismatch count rises 3,512 -> 3,554 of
        # 10,948.  Recorded in the morrison-mp10 registry warnings.
        #
        # RE-PINNED by the WRF-parity PGAM density repair (PAR-MP-MORR-10).
        # module_mp_morr_two_moment.F:3918-3920 builds the PGAM reference
        # density as DUM = PRES(K)/(287.15*T3D(K)) from the CURRENT T3D, and
        # by that line T3D has moved: the tendency apply (:3710), the
        # sedimentation evaporation (:3735-3758) and the ice melt and both
        # homogeneous freezings (:3805-3843) all precede it.  RHO(K) is
        # built once at :1325 and is NOT that density.  morr_bound scaled
        # the stale RHO by RD/287.15 instead, so PGAM -- and through it the
        # cloud-droplet spectral shape and EFFC -- came from the entry-time
        # temperature.  The sibling site :1558 is downstream of a T3D change
        # too (:1504, :1511), and morr_process_level applies the same melt
        # to *temp before its call, so the one repair corrects both call
        # sites.  This MOVES ANSWERS wherever a level's temperature changed
        # within the step: EFFC feeds RRTMG, so radiation moves with it.
        # morrison is not an mp=8 translation unit and thompson.cu is
        # byte-unchanged.  Measured by tests/test_morr_rimed_ice.py::
        # test_pgam_reference_density_is_rebuilt_from_the_current_temperature,
        # which compiles module_source("morrison") and drives morr_bound
        # through ctypes.
        # The finite-transfer correction changes Morrison's rain/cloud
        # freezing range, final vapor store and in-range number behavior.
        # Its device contracts cover each change; continuation identity v3
        # distinguishes it. Morrison is not an mp=8 translation unit, and
        # the other source digests remain unchanged. These are source
        # identities, not regenerated physical reference outputs.
        # RE-PINNED by 68d2b1fec (speed, no reading moves): sedimentation
        # runs one thread per column and category and computes each
        # category's fall speeds once; the substep count is the same fmaxf
        # maximum.  A 1 h Morrison default-suite forecast (220 x 176 x 49,
        # 240 steps) writes every wrfout frame byte-identical to d6929cb8d
        # on an RTX 5090 and an RTX 4090.  Then the clear-air finalize path
        # (speed, no reading moves): a level with exactly zero hydrometeor
        # mass and positive vapour writes the values the full path computes
        # for it without the conversions that are all zero there; the same
        # forecasts stay byte-identical.
        # Re-pinned for A146 on top of that: every float division by a
        # compile-time constant is spelled __fdiv_rn, because NVRTC compiles x
        # / C as a multiply by the rounded reciprocal on Blackwell targets.
        # Reading: sm_89 unchanged; Blackwell cards now round these quotients
        # IEEE-correctly. Previously b0b9f1d2/506849c6.
        # Re-pinned again by the A146 review repair: cons15, a constant
        # numerator of powf calls over 4 * 720, the first pass missed.
        # REPLACED by lane/parity-mp-libm (sweep row 19): morrison.cu is now
        # the literal transcription of WRF v4.6.1 module_mp_morr_two_moment
        # (driver, MICRO, POLYSVP, GAMMA, refl10cm_hm) with WOOF's own
        # libm, WRF init's own words (morrison_constants.cuh) and
        # --fmad=false.  Carried to lane/verify-morrison-mp10 off the
        # verify base, comments reworded (WOOF's own libm); measured there
        # by tools/morrison_wrf461_oracle (fixture, stress sets and the
        # regime columns, regime_columns.py) on an RTX PRO 6000: with
        # MORR_ROW37_OFF every returned word equals WRF's; the production
        # build keeps the owner-ruled row-37 corrections (the cold-trap
        # bound above and the wide non-finite freezing rates).
        # Previously 9d4aad5a/96bf1249.
        # RE-PINNED by lane/verify-morrison-mp10: mo_process, the sediment
        # pair and mo_finalize take a column range (col0, ccols) and index
        # WORK by the call's own cells, so a forecast step runs the scheme
        # in ceil(ncol / NWORK)-column calls and WORK is one volume, not
        # NWORK (+186 MB on the d01 itemization fixture otherwise).
        # Indexing only; no arithmetic line moved.  Reading: the regime
        # columns, whole-domain, forecast-chunked and one column a call,
        # bitwise against WRF with MORR_ROW37_OFF (W1, RTX PRO 6000).
        # Previously b602a938/3f94f09d.
        # RE-PINNED by 04897aa077b (2.9.3, the WoFS ensemble lane):
        # mo_sediment_impl bounds the per-level mass after its tendency
        # (dmass = max(dmass, 0)) inside the production row-37 block
        # (#ifndef MORR_ROW37_OFF), because a source tendency can cancel
        # its category to a negative FP32 residue and the inherited fall
        # speed then exported negative water (measured -1.54e-14 mm in a
        # regional forecast; the negative SR floor it needed is retired).
        # The MORR_ROW37_OFF oracle build is unchanged line for line, so
        # the WRF bitwise reading above stands for it.
        # Previously 5b518642/e6166236.
        # RE-PINNED by fix/morrison-kf-speed (2026-10-23): mo_sediment_impl
        # skips the still levels above KTOP (mo_still_level: +0 speeds,
        # finite sign-clear moments, positive DZ/RHO, number tendency not
        # -0, so every split step there adds or subtracts +0 and changes no
        # word) and evaluates (RHOSU/RHO)**0.54 and **0.35 only for the
        # categories and levels that read them.  No arithmetic line moved.
        # Reading: captured real sedimentation inputs (150x150x49, step 60)
        # 0 of 37,665,000 argument words differ, 0.91 -> 0.21 ms a call
        # (an RTX 4090, shared card, 50 back-to-back launches); the 15-min
        # whole forecast's wrfout 0 of 31,808,221 words differ.
        # Previously 741d71ee/09b1d5a5.
        '3fd84d25ccae3b3616be2f99e5f4ddc594256c2ecbb457aa0381f724db481054',
        '7655b4e64f892082b0a0eecfee2db62fb566cf2bb24393976c23163ad822553c'),

    'mynn_pbl': (
        # RE-PINNED by the level-major MYNN layout (speed lane, 2.8.1):
        # every per-column kernel addresses level k of column c at
        # k * ncol + c instead of c * nz + k, so a warp's loads coalesce.
        # Addressing only; no arithmetic line moved.  Outputs are bitwise
        # identical to d6929cb8d on the RTX 4090 and RTX 5090: the WRF
        # oracle tests (test_mynn_pbl_gpu, test_mynn_pbl_driver_gpu), and
        # every returned tendency and carried field of a captured real
        # 288x288x59 call replayed carried, cold start, at odd and whole-
        # domain chunk widths, tiled, and with bl_mynn_mixscalars=1.  The
        # DMP sibling carries the same edit and its own re-pin.  The pin
        # before it was the rounded ordinary mixing length (continuation
        # identity v2), which this change does not touch.
        # Re-pinned for A146 (a98f2482e): every float division by a compile-time
        # constant is spelled __fdiv_rn, because NVRTC compiles x / C as a
        # multiply by the rounded reciprocal on Blackwell targets.  Reading:
        # sm_89 unchanged (the compute_89 PTX with __fdiv_rn spelled '/' is
        # identical to the base's); Blackwell cards now round these quotients
        # IEEE-correctly.  Previously 5c5a7278/0258a93f.
        # Option 2 now reaches WRF's local mixing-length law in cold and
        # carried calls. The option-1 kernel body is unchanged; Fortran
        # column and assembled-driver checks live in test_mynn_mixlength2.
        # Previously b03be073/8771aada.
        # Re-pinned by lane/286-fork-mynn: the GSD MYNN v4.1 rows
        # (bl_mynn_version = "gsd_41") sit behind #if defined(MYNN_GSD41)
        # and compile only through the integer-define loader.  Reading: the
        # default build's binary is byte-identical to the previous source's
        # (NVRTC on the RTX 4090, sha256 c0e92423...), so every wrf_461 run
        # is the same machine code; tests/test_mynn_gsd41.py holds the rows.
        # Moved again by the gsd_41 cloud block (stratus condensation,
        # shallow-cumulus cloud, decay memory, in-cloud QC_BL write-back),
        # all behind MYNN_GSD41; the default binary is still c0e92423.
        # Previously f4706c64/6923ecda.
        # Moved again by the gsd_41 surface TKE source and 1/L and its
        # dissipative heating, behind MYNN_GSD41; default binary c0e92423.
        # Previously ab421d81/3f08b100.
        # Moved again by the gsd_41 PBL height input (theta-v of the
        # liquid-water theta) and KPBL blend; default binary c0e92423.
        # Previously 52121cc9/1fc6c827.
        # Moved again by the gsd_41 stability-function limits; default
        # binary c0e92423.  Previously b61e6fa2/d07dbe63.
        # Plume saturation and its iteration stop now take the fork form
        # under MYNN_GSD41; the leaf matches 24 unmodified Fortran states
        # bit for bit. The new PBL-height divisions use __fdiv_rn. Default
        # PTX remains c0e92423. Previously dec5ec9e/04bb84ea.
        # GSD mass-flux population, trigger, ascent, scale awareness and
        # density-free transport are conditional. Default PTX remains
        # identical on the RTX 5090 (3dbf9aead85e, 1234799 bytes).
        # P20 uses the fork's Neumann top row, 1e-4 floor, no TKE cap and
        # 1e-12 initialization production floor under MYNN_GSD41 only.
        # All 96 predictor TKE words match unmodified source. Default PTX
        # is identical to staging cdf5a7dea at compute89 and compute120.
        # Previously 3243ddcb/ecf106fc.
        # P23 (lane/ec-mynn-p23): the fork's closure level 2.5 diagnoses
        # tsq, qsq and cov in mym_predict under MYNN_GSD41 only; eight
        # columns match the unmodified predictor as FP32 bits
        # (variance-gsd41.csv).  Default PTX is byte-identical to 534c6dea3
        # at compute_89 (c9058bb3) and compute_120 (d5028fa7).
        # RE-PINNED by lane/mynn-exact (the YSU recipe on MYNN), MOVES ANSWERS
        # for bl_pbl_physics = 5: --fmad=false, glibc 2.39 expf/log10f and the
        # FMA-ifunc logf/powf/expf in the unit, fdlibm tanhf at every site;
        # under MYNN_GSD41 the fork's a2den algebra and 0.608 thetav.  Reading
        # (RTX 5090 sm_120, NVRTC 12.9): every MYNN CUDA leaf and both drivers 0
        # ULP against WRF v4.6.1, six column families x 12 steps free-running 0
        # ULP (tests/test_mynn_wrf461_exact_gpu.py); gsd_41 0 ULP against the
        # fork's driver (tests/test_mynn_gsd41_driver_exact_gpu.py). Previously
        # 9d6b1bcc.
        # Review: fork CASE(1) and its buoyancy flux now follow the fork.
        # The stock preprocessed source is unchanged, checked by the sibling test.
        # Wave3: export arm folded from sibling; default tokens unchanged.
        '88d6909612543927aee364c6f0cbd905eabd808da2a9fe3f94b57c9fab743b49',
        '1f5e6f64a206f602e4281d0d38bc8295b897a13050826f62d487ff5f0e61d871'),
    'mynn_surface': (
        # Re-pinned for A146 (a98f2482e): every float division by a compile-time
        # constant is spelled __fdiv_rn, because NVRTC compiles x / C as a
        # multiply by the rounded reciprocal on Blackwell targets.  Reading:
        # sm_89 unchanged (the compute_89 PTX with __fdiv_rn spelled '/' is
        # identical to the base's); Blackwell cards now round these quotients
        # IEEE-correctly.  Previously a94de3ff/891ec5d5.
        # The selectable GSL surface variant and explicit table rounding
        # coexist. The ordinary table retains the native Blackwell operation
        # order; the selected variant keeps its own stability solver.
        # RE-PINNED by lane/cap-waves-couple (8cd59d74c15): over water the
        # wave-age momentum roughness replaces z0 before exchange, compiled
        # in only under OCEAN_WAVE_ROUGHNESS (woof/core/mynn_sfclay.py
        # defines it only with a coupled wave model); without it the
        # preprocessed unit is the previous one.  Previously b8b5b9b5/2c93b3ba.
        'f1a1b6c98704fa00ad35f4a3c81183f706509adf42bfba9022b2a882c395126a',
        '5b294157fb4224d0aa2ce11ce4856dafeb9e9f0a0e8145694dd6e0c48acfe64f'),
    'nest': (
        # Re-pinned for a combined four-side boundary launch. The original
        # single-side entry point and SINT expression tree remain unchanged.
        # e1bdd7741 + d17f1d08b preserve global terrain coordinates and
        # sequential donor arithmetic through bounded child operands.
        # Measured 8d317e5ec: 34 focused + 29 resident CUDA controls;
        # 295d6ec0a: public moving/restart, 137 exact arrays per domain.
        # Re-pinned with diagnostics for 88fdf60b9's parent smoothers.
        # RE-PINNED by bf05966444a (2.9.3): relocation_nearest_donor, a
        # new kernel appended after the existing entry points (one block
        # per exposed cell, integer squared distances, first donor on
        # ties) so a moving nest's relocation carry stays on CUDA; its
        # reference is the NumPy argmin it replaced and no existing
        # kernel line moved.  Previously de9a5a8f/06055661.
        '6219f7071e8e54c8b491da4b394bdf5f23e70f18e53e8cb5861e2d4517953d9b',
        '3171f8ea4367510c073206d9cdad2708881a6c05d225a62d33b90149d8903fb4'),
    'nest_microphysics': (
        # Re-pinned for the mp=9 (Milbrandt-Yau) mixed-edge ratification
        # (audit R-003): the generic microphysics_edge_field kernel gained
        # the parent's base and perturbation potential temperature, its
        # pressure and the scheme's ck constant vector, plus the reference
        # pressure / R-over-cp pair and the flag saying whether the base
        # theta is a column or a field -- all read by the mp=9 arm alone,
        # placeholders for every other target -- plus my2_edge_field and
        # the two MY2 helper functions.  The absolute temperature that arm
        # reads is formed IN the kernel: the host has no array to form it
        # into on the tile-streamed nest route, where the launcher is
        # handed transition_parent_window's bounded namespace rather than a
        # DomainState, and a windowed mp=9 edge died there with an
        # AttributeError.  The frozen mp8_to_mp18_mass_diagnosed_field
        # entry point in the same file is byte-unchanged (its own bitwise
        # test still binds it), the P3 and NSSL arms are untouched, and no
        # mp=8 trajectory reaches the changed kernel: the edge matrix
        # launches only on MIXED nest edges, which an all-mp8 run never
        # resolves.  MEASURED on the 5070 Ti by
        # tests/test_milbrandt_nest_edge_gpu.py, whose windowed arm is
        # equal cell for cell to the resident parent's.  Previously pinned
        # at 541288c6/1598bc17 for the host-built temperature plane, and at
        # add31c69/fdf0a5b4 for the mp=50 ratification.
        #
        # RE-PINNED AGAIN for the mp=16 / mp=28 mixed-edge ratification:
        # the same generic kernel gained one wdm6_ccn scalar and the two
        # entry arms, and its field-code comment grew codes 24, 25 and 26.
        # Both reasons above hold unchanged -- the frozen
        # mp8_to_mp18_mass_diagnosed_field entry point is byte-identical,
        # and an all-mp8 run never launches the edge matrix at all.
        # Previously pinned at 9031874d/ae17b2a2.
        # Re-pinned for A146 (a98f2482e): every float division by a compile-time
        # constant is spelled __fdiv_rn, because NVRTC compiles x / C as a
        # multiply by the rounded reciprocal on Blackwell targets.  Reading:
        # sm_89 unchanged (the compute_89 PTX with __fdiv_rn spelled '/' is
        # identical to the base's); Blackwell cards now round these quotients
        # IEEE-correctly.  Previously 6584d2be/ab4befec.
        'a00a3f6cd14622bb53d27152652df31889267ae83c160a4f97128051b99c539c',
        '02bb1a0d5607306d502eba8d7054272bbd6e8a0b3c64961e2e2922eca1efa057'),
    # Re-pinned for the WOOF 1.0.0 text scrub: comments only (punctuation, host
    # labels), proven equal with comments stripped; the preprocessor drops
    # comments, so the compiled binary does not move.
    'noah': (
        # RE-PINNED by the WRF-parity FRZX repair (NOAH-01).  WRF renames
        # this quantity twice on its way down and woof followed the NAME
        # instead of the ARGUMENT: REDPRM (module_sf_noahlsm.F:2477-2478)
        # builds FRZX = FRZK*FRZFACT; SFLX passes FRZX at :769 and :784;
        # the receiving dummy is spelled FRZFACT in NOPAC (:1909), SNOPAC
        # (:3015) and SMFLX (:2670); SMFLX passes it on at :2785/:2794/:2803
        # and SRT (:3655) names it back to FRZX, spending it at :3795 as
        # ACRT = CVFRZ*FRZX/DICE.  noah_column was handing noah_smflx the
        # bare FRZFACT, so the frozen-ground infiltration limit ran on a
        # dimensionless ratio instead of FRZK*FRZFACT and SRT's ACRT
        # exponent collapsed.  This MOVES ANSWERS, hard, on frozen ground:
        # measured against the WRF oracle the sfcrunoff distance falls from
        # 60,641,303 ULP to 2,812, and sh2o, smcrel and smois go from 6,508
        # / 4,729 / 1,627 ULP to exactly bitwise.  noah is not an mp=8
        # translation unit and thompson.cu is byte-unchanged.  Measured by
        # tests/test_noah_wrf461_parity.py (the re-pinned BASELINE_MAX_ULP
        # table), ::test_the_mirror_reproduces_wrfs_frozen_ground_infiltration
        # and ::test_the_kernel_hands_smflx_the_same_word_the_mirror_does.
        #
        # RE-PINNED by the urban hand-over (739569b89, 27ca0e896;
        # sf_urban_physics 1-3): noah_column gained ten trailing arguments
        # and WRF's pre-SFLX urban remap (module_sf_noahdrv.F:964-990) under
        # `urban_col`, which is false on every column when urban_opt == 0,
        # plus the rural hand-over writes on urban columns only.  The
        # default path is unchanged: measured on test host A (RTX 4090), all 216
        # output arrays of the default-off call are byte-identical to the
        # pre-change kernel's (tests/test_urban_default_off_identity.py::
        # test_a_handover_with_no_urban_column_is_the_default_kernel), and
        # tests/test_noah_wrf461_parity.py holds its table unchanged.  The
        # urban arm is graded against WRF v4.7.1 by tests/
        # test_urban_noah_hook_wrf471_parity.py.  Previously
        # d7ae4d2c/b8adda8a.
        #
        # RE-PINNED because that claim did not hold for a whole forecast:
        # with the hand-over compiled into the one kernel behind a runtime
        # flag, NVRTC contracted a different set of products into FMAs, and
        # a 1 h default forecast (configs/hrrr_native_quick_demo.toml, RTX
        # 5090) differed from integrate/2.8's from its 15-minute history
        # on, while the same tree with the pre-hand-over kernel body was
        # byte-identical in all five.  The body is now `template <bool
        # URBAN>` with every hand-over statement under `if constexpr`:
        # noah_column (URBAN = false, the default launch) compiles from
        # exactly the pre-hand-over statements (545 of 545) and takes its
        # pre-hand-over arguments, and noah_column_urban carries the
        # hand-over.  Previously a21b1917/21faf648.
        #
        # Re-pinned for A146 (a98f2482e): every float division by a compile-time
        # constant is spelled __fdiv_rn, because NVRTC compiles x / C as a
        # multiply by the rounded reciprocal on Blackwell targets.  Reading:
        # sm_89 unchanged (the compute_89 PTX with __fdiv_rn spelled '/' is
        # identical to the base's); Blackwell cards now round these quotients
        # IEEE-correctly.  Previously d7ae4d2c/b8adda8a.
        #
        # MERGED with integrate/2.8 (urban) on lane/281-nvrtc-literal-div:
        # both changes above are in the file, so the digests are the
        # merged file's.
        # Re-pinned 2026-10-01 for the ordinary-land register bound.
        # Direct before/after output words match on RTX 4090 and RTX 5090.
        # Previously 2dcb1b59/d4dca1d0.
        #
        # RE-PINNED by golden/ports-pbl (2026-10-12), and this MOVES ANSWERS
        # on land-ice columns only: SFLX_GLACIAL (glacial PENMAN, SNOPAC,
        # HRTICE, SNOWZ0, the driver's ICE == -1 post-update) is ported where
        # the column used to return untouched.  Reading (G1 RTX PRO 6000,
        # sm_120, NVRTC 12.9.86 and 13.2 alike): the 41 ordinary land columns
        # of all four switch fixtures are byte-identical to cd336e62d's
        # kernel; case 28 and the 24-column land-ice fixtures sit at the
        # GLACIAL_BASELINE_MAX_ULP of tests/test_noah_wrf461_parity.py (TSK,
        # HFX and TSLB 0 ULP, worst flux SNOPCX 5).  Previously
        # aaff81f7/de546677.
        #
        # RE-PINNED by lane/290-A5-fixes (2026-10-14), MOVES ANSWERS on
        # sea-ice columns only: noah_seaice_column (WRF's seaice_noah,
        # verify-noah-lsm 217a273cc) is appended, and the unit takes
        # glibc_flt32.cuh for its gfk_exp/gfk_pow.  Reading (test host A RTX 4090,
        # sm_89, NVRTC 12.9.86): tests/test_noah_seaice_wrf461_parity.py 0 ULP
        # on its 24 x 3 oracle columns; the release short case (ICON-EU, two
        # domains, Noah, no sea ice) is byte-identical to integrate/2.9.0
        # 0bf505764 in all 21,885,246 history words.
        '1a8b9c227ce0d68590c77a63f2f78bd8343fca633c9446ab9e1a0e47a0db6395',
        '42d601c5a6af5b03ee7e3bb9127c42be635d0c5caa89405b273ce7101dcddb21'),
    'noahmp_bareflux': (
        '54fb5065e95b24d4cf676e2deda29bae44b3e9305d3d98cbc1abf5ed55f444ce',
        'fbb19fc8b5668ea2edbcc1270f8ffe367475124ffa0ce99d3bc34639f3f31e9f'),
    'noahmp_driver': (
        'bd555be10ccade5a5bdddcaf4c56b7f4353dae1208fc48a5586fb7ce7d32d643',
        'f1913fe0054adb74188effa6499b799e989cfb1e96ecd676edcd43df0083030e'),
    'noahmp_energy': (
        # RE-PINNED at 2.7.0 by the licence cut.  `nmpe_tanhf` loses glibc's
        # redundant `if (ix == 0) return x;` -- the one edit in the FDLIBM
        # group attributable to reading glibc's text and to nothing else, so
        # it is expression this Apache-2.0 distribution should not carry.
        # FDLIBM never had it and the tree already shipped tanh without it
        # (mynn_pbl.cu, mynn_dmp_sibling.cu).  It is redundant because the
        # |x| < 2**-55 branch returns x*(1+x), which is x for both signed
        # zeros.  NOTHING MOVES: the two forms were compiled side by side
        # with the shipped bodies and a host shim and compared on all
        # 4,294,967,296 float32 bit patterns -- 0 differ, tanh(+0) = +0 and
        # tanh(-0) = -0 in both.  Four lines of comment came in with it.
        '4fd4b5a86ec97371e01aac13d2c5e40e40546b64a77c39e6fb674c96be69d209',
        'bd91acf9bf987c5116e894b67e0ecbe47ae5bad5f1237095fe822da7fcd6b0a2'),
    'noahmp_fluxprep': (
        'eef473608e9d1c0176574c6bd72183659249b42a022f8d63d77c611de506f4ed',
        '4ae139ef5d31234b5e9fdbc04c4cfd3c5156cd91d0326e5544b5b78f7c17fb31'),
    'noahmp_leaves': (
        # RE-PINNED at 2.7.0 by the licence cut, and for the same reason as
        # noahmp_energy above.  `r_log10`'s zero path was glibc's
        # `-two25 / fabsf(x)`, a division whose only work is to raise
        # divide-by-zero on the way to -inf; FDLIBM divides by a `zero`
        # variable instead, so the spelling is glibc's own.  ArWen returns the
        # -inf directly.  NOTHING MOVES: compared on all 4,294,967,296 float32
        # bit patterns -- 0 differ, log10(+-0) = 0xFF800000 in both.  Five
        # lines of comment came in with it.
        '4a3b6ce94993ab1c7055de39880e1209b54527751dcda07f26aa4b4a612f3497',
        'e27a5b92bd5f9ee64e0e9a9e496228e4b4b7b4a43ca3f7708a272a8234030ebd'),
    'noahmp_libm_slab': (
        '0144fa7d142a8d24f5f0f52bd0dade987312cd97c2a2efad9a6f33edb1a35fda',
        'c7cdc57aa7d3d507d2b57935df13e16dba87dd783384e28a7240b95a536f39a7'),
    # RE-FROZEN by the WRF 4.7 Noah-MP night snow-aging port (golden/wrf-fixes,
    # merge 170b65f7b of upstream-noahmp-night-snow-age47 f77be93ec): the
    # unit gains the night-aging entry (noahmp_radiation_batch_night_age);
    # the daylight entry's statements are unchanged.  Reading: both
    # generations 0 of 19,614 words against WRF 4.7.0 / 4.6.1 full columns
    # (tests/test_noahmp_snow_age_gpu.py) and 0 of 9,120 leaf words against
    # each release's Fortran (tools/noahmp_snow_age_oracle/compare.py).
    'noahmp_radiation': (
        'f7317dcf918a73ac014aaf2b81f19757afb152e5cbe50c538253bedaedd5aa0e',
        '520df954dbfeb613997fd59cd4fde99ec7c3120fd361eefda553b1900a50dfca'),
    'noahmp_sflx': (
        '47f51fed351b3f720203c07ea2d5e5a8902f4162afe17b634f5196447728365f',
        'f17caed98ec58c4d9836778f3a9edea9e88cf13808dbe64d5e099fd81ffebc77'),
    'noahmp_snow': (
        'f46f850fb54acf3dadcaf7a7df9bb4c09d107be8de0c577b704536a20271b42a',
        'd375fe9595c514690fb7fe6db756b76f4c2341862dd5c3b70bae2131ccfcde22'),
    'noahmp_soilwater': (
        'a53adc6aaa1a46974b13497676e682c56587176442bf1a26003b4d377d87688e',
        'd9e8788e7bb479104faabbb3ca5d6ae698768a394a60f54b0e2840862bfbfa5a'),
    'noahmp_thermal': (
        '2598ca76b7f9c0d6de35631d5113901d1e25f24a014c8ea4e0df490158d16c86',
        'f7eef131507a29def54e2387686f34a4807851354f446684deaf5322dc226753'),
    # 300876fefd5: explicit libm-2.43 atan profile; default libm-2.39
    # retained. Native 8-column witness is words-exact; see
    # tools/noahowp_replay/ATAN-PROFILE.md and qualified_atan tests.
    # RE-PINNED by de768fc2770 (2.9.3, the merge of lane/speed2's
    # b3dd58fc9f3 with 300876fefd5 above): dev_vege_flux takes opt_stc
    # and fsno (the WRF-lineage STC=3 ground-temperature blend, Noah-MP
    # 848f54ad :4125-4136; STC=1, the Registry default, is the same reset
    # as before), k_vegeflux_stc3 and k_vegeflux_masked are appended
    # (fixed-width masked dispatch: an inactive column writes +0 and never
    # enters the leaf), and the GLIBC243 atan branch joins the profile
    # selector.  The default-route arithmetic is unchanged; the
    # composed-unit frame row is re-read for it in
    # kernel_frame_recordings.  Previously 295e9541/c3271c5a.
    'noahmp_vegeflux': (
        'd55c402a21caf3f52e5aed05a0dd727df76b747906a3e0d082446e6a7bcab014',
        '0d23b367d88497bb45a2e895e6d85abc826caf2f4467b8b47c7ba7be32d3a9b4'),
    'noahmp_vegprecip': (
        '9ce5667599d111be0efc7cb0871e5719d70259bb95b30fbe923e01964b75c26c',
        '599e85f0fe59a28ae5fd5437e47db7f41c31abd21783979f66b987b212f1f41a'),
    # 485b468bdf4: parent runoff units and explicit drainage export,
    # qualified by test_noahowp_coupling_cuda.py / NEXTGEN-NOAHOWP.md.
    # This freezes that landed source, not a whole-forecast skill claim.
    'noahmp_water': (
        '7946fb6612767fac098ec92a0dbb1d0e4cc768657b1cb81b297344cd39aa54f3',
        '46808ddb98445be32c5c8ab0a04527b919446081c33f7ef51c8b7a0ebed5b42a'),
    # Wave 3 folds the default twin using math aliases; original default
    # and strict arithmetic bytes remain independently pinned in
    # test_exact_chain_default_sources.py. Assembled text changes only.
    'nssl2': (
        # RE-PINNED by the WRF-parity qxmin(lh) repair (NSSLA-01).  The Bigg
        # rain-freezing gate used the graupel minimum mixing ratio 1.e-7 set
        # at module_mp_nssl_2mom.F:2095, missing the overwrite eight lines
        # later at :2103: `IF ( lh .gt. 1 .and. lnh .gt. 1 ) qxmin(lh) =
        # 1.0e-12`.  Under the option-18 default (ipconc=5 by
        # module_physics_init.F:4633-4641) the index block at :1650-1667
        # gives lnh = 14 and lh = 7 (:658), so the override is
        # unconditionally live at both use sites -- the minimum-transfer
        # gate (:17653) and the volume/SETVT gate (:14205/:14213).  The
        # sibling nssl2.cu:4274 already used 1.0e-12f.  This MOVES ANSWERS
        # in the five-decade window between the two constants, where the
        # transfer was being suppressed to zero.  nssl2 is not an mp=8
        # translation unit and thompson.cu is byte-unchanged.  Measured on
        # the device by tests/test_nssl2_gpu.py::
        # test_bigg_rain_freezing_uses_the_two_moment_graupel_qxmin.
        # 9a8b0da23 rounds Bigg's rain-mass multiply before subtracting,
        # preserving the WRF FP32 state used by the later number bound.
        # This changes mp=18, not Thompson's mp=8 translation unit. The
        # exact parent source reproduces both old pins through the real
        # loader; retained NSSL GPU oracle comparisons grade arithmetic
        # without widening the existing tolerances.
        # Re-pinned for A146 (a98f2482e): every float division by a compile-time
        # constant is spelled __fdiv_rn, because NVRTC compiles x / C as a
        # multiply by the rounded reciprocal on Blackwell targets.  Reading:
        # sm_89 unchanged (the compute_89 PTX with __fdiv_rn spelled '/' is
        # identical to the base's); Blackwell cards now round these quotients
        # IEEE-correctly.  Previously 0541eb4f/24a7e4af.
        # Re-pinned again by the A146 review repair: the snow-melt
        # constant c1sw, a C math function divisor the first pass
        # missed.
        '1952dfec0483c0002d5addb45e1ff3f10a54c8c42450efe8acfb124e7b2dbe9d',
        'f3f7f268018488f5cd076793c264ba36321743391ad6edcc87c494bc5fcd07fa'),
    'nssl2_diagnostics': (
        'aea3a5b1157ca95b69c8df077680e27058b752d5e707f2134f408d0aa7f42bd2',
        '331b4a9734959260ab515216e24ee7100eac18d8e6d646b1e0e8bf21c0c23374'),
    # The three NSSL translation units below are re-pinned for the WRF
    # v4.6.1 variant family (nssl_hail_on / nssl_ccn_on).  Each gained one
    # trailing `int` kernel parameter and a branch on it: the CCN load and
    # store in nssl2_driver_support, the graupel-to-hail conversion call in
    # nssl2_fused_gs, and the nucleation pool in nssl2_nucond.  None of them
    # is an mp=8 translation unit, so the mp=8 numerics guarantee this file
    # exists to protect is untouched.
    #
    # The pins move; the mp18 default path's OUTPUT does not.  That is the
    # justification, and it is measured rather than argued: with these exact
    # bytes, tools/nssl2_mp18_digest_probe.py reproduces all 30 committed
    # SHA-256 field digests in evidence/nssl2-variants/mp18-digest-baseline
    # .json byte for byte (re-run on this tree: 30 reproduced, 0
    # mismatched).  That is what rules out the FMA-contraction risk of
    # adding a parameter and a branch to a fused kernel.  Each digest below
    # was re-derived from the tree by this test's own fixture, not edited
    # to match.
    #
    # RE-PINNED by the speed lane's NSSL sedimentation (no reading moves):
    # for nz <= 64 one launch runs the six categories in parallel rows, one
    # thread per level, every flux computed before any level updates as
    # sediment1d does; graupel and hail read their gamma values from a
    # table the device fills once with the same tgamma calls; nz > 64 and
    # CUDA-graph captures keep the per-category column kernels.  The mp18
    # suite's 1 h forecasts of two convective cases (220 x 176 x 49) write
    # every wrfout frame byte-identical to d6929cb8d on an RTX 5090 and an
    # RTX 4090.
    'nssl2_driver_support': (
        # Re-pinned for A146 on top of that: every float division by a
        # compile-time constant is spelled __fdiv_rn, because NVRTC compiles x
        # / C as a multiply by the rounded reciprocal on Blackwell targets.
        # Reading: sm_89 unchanged; Blackwell cards now round these quotients
        # IEEE-correctly. Previously d56ab580/edda4361.
        'a5da944e7bcb5d63362265acef241a752ef835a080fa3a1a05b5eb5bd186ca35',
        'cfe99438d0ff9b7e7bc31c61daea9b33f0a77b82908807ee9b9c300719031320'),
    'nssl2_fused_gs': (
        # RE-PINNED by the WRF-parity vertical-velocity centering repair
        # (G-01).  The kernel averaged interface w to mass level TWICE,
        # delivering 0.25*w[k] + 0.5*w[k+1] + 0.25*w[k+2] where
        # module_mp_nssl_2mom.F:14174-14176 delivers 0.5*(w[k] + w[k+1]) --
        # a half-level upward shift plus 1-2-1 smoothing.  The premise of
        # the comment that put it there ("WRF's microphysics driver supplies
        # a mass-level W field") is false: solve_em.F hands the driver the
        # staggered grid%w_2 and :2827 copies it into the GS slab with no
        # de-staggering, so wvel = 0.5*(w(kp1)+w(kgs)) IS the single
        # interface-to-mass average and its Min(nz, kgs+1) clamp is on the
        # upper INTERFACE.  This MOVES ANSWERS for every mp_physics=18 run
        # with vertical shear in w: wvel is the linear factor in WRF's
        # icenucopt=1 primary-ice source (:20733-20738) and also its > 0
        # gate.  nssl2_fused_gs is not an mp=8 translation unit and
        # thompson.cu is byte-unchanged, so the mp=8 numerics guarantee is
        # untouched.  Measured against WRF's own instrumented oracle by
        # tests/test_nssl2_fused_gs.py::
        # test_official_wrf_oracle_pins_the_single_average_and_its_top_clamp
        # (all 240 rows satisfy w_center == 0.5*(w_lower+w_upper)) and ::
        # test_official_wrf_oracle_rows_reach_qiint_with_a_shifted_w_center
        # (12 rows reach the qiint computation, where the old rule overstated
        # wvel by 1.18x to 2.06x); the kernel's spelling is held by ::
        # test_cuda_centres_interface_w_onto_mass_levels_exactly_once, which
        # replaces a source pin that asserted the two-stage average.
        # Re-pinned for A146 (a98f2482e): every float division by a compile-time
        # constant is spelled __fdiv_rn, because NVRTC compiles x / C as a
        # multiply by the rounded reciprocal on Blackwell targets.  Reading:
        # sm_89 unchanged (the compute_89 PTX with __fdiv_rn spelled '/' is
        # identical to the base's); Blackwell cards now round these quotients
        # IEEE-correctly.  Previously 8b4ad70f/4606e906.
        # Re-pinned again by the A146 review repair: the graupel and
        # hail fall speeds' / tgammaf(4.0f) and / tgammaf(5.0f), which
        # compute_120 folds to a reciprocal multiply and the first
        # pass's compute_89 census could not see, and c1sw.
        '76a0685f4860042d558feafca633a38cc41b498058ec9c6230d83669336bc6a4',
        '15ec0480c24de0c6694d66cd183cb05412aa518d19c33a3ed4e9936027266a7f'),
    'nssl2_nucond': (
        # RE-PINNED by the WRF-parity raw-w repair (N-02).  The low-T cnuc
        # hack at module_mp_nssl_2mom.F:10122 tests the RAW staggered
        # element `w(igs(mgs),jgs,kgs(mgs))` -- the cell's own bottom face --
        # and the kernel averaged the two interfaces onto the mass level
        # first.  That is not WRF being sloppy: NUCOND spans :9611-12215 and
        # the only wvel assignment inside it is at :10381, 259 lines
        # downstream, so the routine has no averaged w to read at :10122.
        # The mass-level average survives everywhere WRF does compute one.
        # This MOVES ANSWERS on sheared columns below 265 K where the raw
        # face and the average straddle the 2.0 m/s gate.  nssl2_nucond is
        # not an mp=8 translation unit and thompson.cu is byte-unchanged.
        # Measured by tests/test_nssl2_contract.py::
        # test_the_low_temperature_cnuc_hack_reads_the_raw_staggered_w,
        # a two-sided gate: it also requires the mass-level average to
        # survive at its two legitimate sites, so it cannot go green by
        # deleting the averaging everywhere.
        # Re-pinned for A146 (a98f2482e): every float division by a compile-time
        # constant is spelled __fdiv_rn, because NVRTC compiles x / C as a
        # multiply by the rounded reciprocal on Blackwell targets.  Reading:
        # sm_89 unchanged (the compute_89 PTX with __fdiv_rn spelled '/' is
        # identical to the base's); Blackwell cards now round these quotients
        # IEEE-correctly.  Previously 224e22e7/311068aa.
        'a57ed276737bafee3c31709423bdc9fb23ef7c74906840838a71a3a7b9523355',
        '40864b3a5080c27998d0346d719281a535a1b8aea92c969bbc2a156de342e181'),
    'nssl2_qvexcess': (
        '6906dcd9f8822d73d87ff3cb6e545a1b1ddef567c16c658435d4f669f1f369dc',
        '89d27036499b7f57d780c594308766b83ef711fbc9d9c5d0ece2e79c270f6626'),
    'openbc': (
        # 6d9c9b999 folds CFL over owned columns of one domain sweep.
        # Measured 0d2a79ed4: 41 CUDA controls; public resident/streamed
        # 448 fields and resumed/continuous 150 fields exactly agree.
        # 399d95a86 appends the adaptive timestep's w_cfl_stat probe;
        # d7c5a9eca corrects its strict-threshold comment. The entire old
        # file remains an exact prefix: existing boundary kernels do not
        # change. Restoring the parent source through the real loader
        # reproduces both old pins; test_wrf_cfl_histogram grades the new
        # measurement against its independent CPU reference.
        # lane/281-w-crit-cfl (A165) moves it: w_damp takes WRF's onset
        # (w_crit_cfl under zadvect_implicit, else w_beta 1) and
        # w_crit_cfl as arguments where W_DAMP_BETA and W_CRIT_CFL were
        # literals, and w_cfl_stat counts cells above the same onset.  At
        # 1.0 and 1.0 the arithmetic is the old kernel's;
        # tests/test_w_crit_cfl.py grades it against WRF 4.7.1's compiled
        # w_damp at w_crit_cfl 1.0 and 2.0 with and without IEVA.
        # The compiled WRF big-step oracle exposed FMA mass rounding crossing
        # the strict CFL onset at 1/2. Explicit operator rounding fixes the
        # limiter and its diagnostic: nine real/edge cases now match every
        # WRF output word, plus four earlier compiled WRF fixtures.
        # tests/test_bigstep_coupling_wrf471_parity.py and test_w_crit_cfl.py
        # gate this default-on correction. Previously dead1a45/38f71150.
        # Re-pinned for the compiled WRF v4.7.1 oracles: w_damp rounds at WRF's
        # float32 operator boundaries (0673d8c51, merged 244794e42), which makes
        # all nine real/edge cases in tests/test_w_crit_cfl.py match every WRF
        # word, and the mapped radiation kernels include the map-factor divisor
        # (dc226e820, merged dde6ea687).  Previously d1d9c3e4/571bb4fa.
        'fd3ba041744821bd4e2b01db3ae53a1189cbc088e17b710e60848988d2203bf2',
        '5bf7babe4a953b1a12f8ddca403f04d86fa0aed0f7684992b5eabb5d82cbca57'),
    'pd_advection': (
        # Re-pinned for A146 (a98f2482e): every float division by a compile-time
        # constant is spelled __fdiv_rn, because NVRTC compiles x / C as a
        # multiply by the rounded reciprocal on Blackwell targets.  Reading:
        # sm_89 unchanged (the compute_89 PTX with __fdiv_rn spelled '/' is
        # identical to the base's); Blackwell cards now round these quotients
        # IEEE-correctly.  Previously 606e3968/d9e86499.
        # Re-pinned for the opt-in strict WRF arithmetic (lane/282-wrf-exact,
        # merged 1efb5a415): 2029c5726 (feat(verify): reproduce active WRF
        # advection and limiter order) adds `#if GPUWM_WRF_EXACT_C_ADVECTION`
        # branches, which the preprocessor keeps only when a GPUWM_WRF_EXACT
        # selector defines the macro.  Reading: with no selector,
        # module_source('pd_advection') at f3e4ca716 and at c2946a54e compiles
        # to byte-identical PTX, the whole text and 2 of 2 .entry kernels, for
        # compute_89, compute_90 and compute_120 under the loader's options and
        # under the RawModule options CuPy compiles with, with NVRTC 13.4 and
        # with 12.9 (tools/kernel_ptx_identity/receipts/,
        # wrf-exact-default-2.8.2-nvrtc13.4.json and -nvrtc12.9.json), so the
        # default build does not move.  Previously 5347b9a3/9cdc25d0.
        # Re-pinned for the bandwidth tuning (lane/282-bw-advection 8c69d26ab,
        # 48cfb1008; merged 7486d241c): in the default build pd_fluxes wraps
        # periodic indices with an in-range test and one signed remainder,
        # and pd_renorm_apply takes a neighbouring x donor's scale from the
        # adjacent lane through a warp shuffle (the same PD_SCALE value the
        # lane computed for its own cell) instead of evaluating it again;
        # warp edges and periodic seams keep the original evaluation, and
        # the WRF-exact build keeps both originals.  No floating expression,
        # reduction order or compile option moved, but the PTX of both
        # entries does on compute_89/90/120 (tools/kernel_ptx_identity/
        # receipts/, bw-advection-2.8.2-nvrtc13.4.json and -nvrtc12.9.json).
        # Reading: the product default suite (8 steps, 64x64x32) is
        # byte-identical, canonical state digest and history file, on
        # test host D's RTX 5070 Ti at 9eee49ce0 against 7486d241c; the lane's
        # report (DYCORE-SPEED-282/advection-RESULT.md) records forecast
        # identity on B200, H100 and RTX PRO 6000 (3 km CONUS) and on the
        # RTX 4090 and RTX 5090 (default suite); and the compiled-WRF advect
        # oracle reproduces every recorded word on the RTX 4090 and on
        # sm_120 (tools/advect_wrf471_oracle/README.md).
        # Previously 8284e197/7de1823f.
        # Re-pinned for lane/286-vadv5 (WRF v_sca_adv_order): pd_fluxes takes
        # a trailing int vorder; pd_zface_half runs WRF's vert_order == 5
        # ladder (pd_flux5v, advect_scalar_pd's flux5 with -vel) under 5 and
        # the unchanged vert_order 3 ladder under 3, the launcher default.
        # Reading: the compiled WRF 4.7.1 advect oracle reproduces every
        # recorded production word on sm_120 at vorder 3, and at vorder 5
        # the limiter's total tendency holds bitwise against WRF 4.7.1 under
        # the exact-C build on the HRRR-fork fixture's cases
        # (tests/test_advect_wrf_legacy_parity.py).
        # Previously 20a2dd79/b4832282.
        'de5a7aabcaafa9975bdf305834d771a68172388f73317788ff389db7ac64344c',
        '9011930b5b510770f81e11c56a7fd6764233e1592127edc05c1d9c20f7e5142b'),
    'refl': (
        # Re-pinned for A146 (a98f2482e): every float division by a compile-time
        # constant is spelled __fdiv_rn, because NVRTC compiles x / C as a
        # multiply by the rounded reciprocal on Blackwell targets.  Reading:
        # sm_89 unchanged (the compute_89 PTX with __fdiv_rn spelled '/' is
        # identical to the base's); Blackwell cards now round these quotients
        # IEEE-correctly.  Previously ff4e3c6d/8e3843a3.
        '1e2c45699a721b4b042769f9f868d85d8a6eb4960fc547ce9affb59b69f43f66',
        'c95b01ece7024d7ba3ad97cc1e6cc121c001e774217b2e2e45915630f55774c1'),
    'rrtmg_lw': (
        # Re-pinned for the buffer-march positivity fix (7ce2f5de7,
        # dp == DELTAP bitwise for in-contract tops per its message).
        # Re-pinned for TAUGB_GPOINT_PROLOGUE (lane/speed-rrtmg-legacy):
        # an added index macro for the batched per-g-point band helpers,
        # every existing kernel and macro byte-unchanged.  Reading:
        # tests/test_rrtmg_lw_batched_layout.py (generated band bodies
        # held to the standalone kernels statement for statement) and the
        # LW engine uint32-identical to the d6929cb8d engine on fifteen
        # decks at chunks 1, 256, 1,536 and 4,096, and six captured
        # product-suite calls uint32-identical to the d6929cb8d forecast.
        '241382acf34bc312e6361822acc057ec3fedf2f31514b477ff939ef723632592',
        'e6b66fe0cd2f2fbceb2c37aa7ab21636c24944df42b997b8908196fe1444069a'),
    'rrtmg_lw_chain': (
        'a71a779ea3a733e422414ee5d5d885cb904972fe32ac2bfe6438d6c5214c1252',
        'ee8f4c95e52248d0b641a48a8532d44d6aff0b75344dda59a416a3905ae972c1'),
    'rrtmg_lw_taugb02_10_11_12': (
        'fe5d57d1eb2649005d2748ae40f849c340b03a5f3d3a61c7fb506d0c69792485',
        'f7d4c98b0ed523e2c5912d4810925c60c3bcbb52d3e1a40d06a2708686cb632c'),
    'rrtmg_lw_taugb03_05': (
        '5de3f0c7c700cff993d5d5a3caa868bbd68d810ea30a902e6e92d652e5612cf7',
        '8a86d6cfcd46ea76bbc3027cb1f68f8842d4e37de5b011d681a5c502c1794b2e'),
    'rrtmg_lw_taugb06_09': (
        'ab677eb0de0faeb2b7a3f4a5923c6777e11128ab7950d9b63abc9cdb61125157',
        '4899dd33838614a7cf94bf3e74291039ed7bb3b35806495a22d3e1c559d01c01'),
    'rrtmg_lw_taugb13_16': (
        '104c3157451dc878e3f9d751db1cf01a4368b1a2892825678370c5159a787ff6',
        '56f315a0258d902c69f99abacbe5e53005f50ce478725c146647d3df26c71522'),
    'rrtmg_mcica_wrf': (
        # Re-frozen for 6fe6e9a6d7 (lane/radiation-fusion via sweep/all-294:
        # fused McICA column kernels rmcw_subcol_states/rmcw_walk_fill_column,
        # the same RMCW_* operations in the same order).  Reading, 2.9.4 fold
        # d61095cd3e on an RTX 5090: tests/test_rrtmg_mcica.py,
        # test_rrtmg_mcica_overlap_boundary.py and test_rrtmg_lw_batched_layout.py
        # 102 passed (bitwise against the certified NumPy port), and the R6
        # identity arms judge the whole forecast.
        # Re-pinned for rmcw_fill_outputs_column (lane/speed-rrtmg-legacy,
        # the longwave McICA slabs written straight into the batched
        # engine's (column, layer, g-point) layout): an appended kernel,
        # every existing kernel byte-unchanged.  Reading:
        # tests/test_rrtmg_lw_batched_layout.py (column layout == the
        # g-point layout transposed, uint32, on the WRF fixtures and real
        # prep decks) and tests/test_rrtmg_mcica.py green, and six
        # captured product-suite radiation calls uint32-identical to the
        # d6929cb8d forecast.  Not an mp=8 translation unit.
        '321f3ff02dcef9108017c4b2510f85479135e9fa601d36b8601253e974e6b11e',
        '5d0e7356e7bef98a6d7200ce7ac8d3204e3756761211343dacb9791eb459c488'),
    'rrtmg_sw': (
        # f7c2aadea removes an unused macro; host launch coverage now
        # includes every layer. Measured d8ef9d086: 156 CUDA controls
        # including independent WRF 80/129-layer reference fixtures.
        # Re-pinned for the one-instruction subnormal armor (65944605a,
        # exact in binary64 per its message; witness at 25ad40769).
        # Re-pinned for the interleaved spcvmc workspace (2c1bbb398,
        # lane/speed-rrtmg-legacy): addresses only, no arithmetic statement
        # moved.  Reading: tests/test_rrtmg_sw_cuda.py (Fortran oracle
        # max_ulp 0, batched == per-column) green, and six captured
        # product-suite radiation calls uint32-identical to the d6929cb8d
        # forecast on an RTX 4090 and an RTX 5090.  rrtmg_sw is not an mp=8
        # translation unit and thompson.cu is byte-unchanged.
        # Re-pinned for the coalesced shortwave slabs, the device laysolfr
        # scan and the fused spcvmc layer pass (lane/speed-rrtmg-legacy):
        # layouts, scheduling and storage only, every arithmetic statement
        # kept (tools/rrtmg_sw_witness/fusion_statement_manifest.json,
        # checked by tests/test_rrtmg_sw_layout_identity.py).  Reading:
        # tests/test_rrtmg_sw_cuda.py (Fortran oracle max_ulp 0, batched ==
        # per-column) green, real prep decks uint32-identical to the
        # previous engine at chunks 1, 256, 1,536 and 4,096, and six
        # captured product-suite radiation calls uint32-identical to the
        # d6929cb8d forecast.
        # e203fa9ea re-mapped rsw_taumol_b and the accumulation tile; a
        # clean RTX PRO 4500 profile measured both slower, so the next
        # commit restored this file byte for byte to its bfc177208 content,
        # whose pin this was.
        # Re-pinned for the solar source of columns with no layer above the
        # troposphere switch (fix/rrtmg-sw-sfluxzen): rsw_sfluxzen_body now
        # stores on every path, zero for bands 16, 17, 27, 28 and 29 when
        # laytrop == nlayers, as WRF's taumol_sw leaves them.  Bands 16 and
        # 27 used to return unstored, so the batched path read recycled
        # scratch there, and 17, 28 and 29 computed a flux WRF never sets.
        # Only rsw_sfluxzen and rsw_sfluxzen_b move: the other 15 entries
        # compile to identical PTX for compute_89, compute_90 and
        # compute_120 with NVRTC 13.4 and 12.9 (tools/kernel_ptx_identity/
        # receipts/, rrtmg-sw-sfluxzen-2.8.4-nvrtc13.4.json and
        # -nvrtc12.9.json).  Reading, on test host D's RTX 5070 Ti:
        # tests/test_rrtmg_sw_no_upper_flux_gpu.py (the unmodified WRF
        # Fortran's fixtures_shallow.npz at max_ulp 0, per column and
        # batched through NaN, 1e30 and stale-flux scratch) fails 7 of 9 on
        # the previous kernel and passes 9 of 9, tests/test_rrtmg_sw_cuda.py
        # passes 162 of 162, and a 360 s legacy-RRTMG forecast with a 50 hPa
        # top (133x125x50, 18 UTC) writes byte-identical wrfout files with
        # either kernel.  Previously 44a83f0e/c9e02eea.
        '43134d6d6d2be4b3e668c7ab92d5850f1f992a09812dd09c9536fce5d9ab791e',
        '663d12d7556f379bdd0724bb4f018113450ce45666e13f3ec3220615409acd08'),
    'rrtmgp_cloud': (
        # Re-pinned for fused profile preparation, broadcasts, temperature casts and flux copies.
        # Exact-array gates and all five seeded digests match the base.
        '512157c75edf4637d99a1cad490e8db71e51cd2c3643ff129adebefa81557560',
        '4ee90af7209167d012e195c011752f84e08d6ea741350977b1a735ac8ee444d4'),
    'rrtmgp_gas': (
        # Re-pinned for shared flavor weights, four cells and the frozen oracle.
        # Exact-array gates and all five seeded digests match the base.
        # Re-pinned 2026-10-01 for aligned flavor weights and prefix padding.
        # Direct before/after output words match on RTX 4090 and RTX 5090.
        'c3b14d451b0a3816c5bf7ecf6ea806034aeab3b7de5c4c12f15acff4421cf265',
        '3c85ba59159dadd97d6e5215cc0b552c4a017539d9c23b99bb77e1a788dee964'),
    'rrtmgp_mcica': (
        # Re-pinned for shared seeds, exact clear paths, SASS-matched FP64
        # intrinsics, bit-major GF(2) jumps and MWC-only alias walks.
        # Base oracle gates bits.
        '1de9a7a5408ffdbd043d547bfafbf2b1eb13c590caa8a03cd956149e1b1c2ad8',
        '4747b7e9cb34990a0650f7a654163f0c8ca2979afe38840e0613164b789520f0'),
    'rrtmgp_rte': (
        # Re-pinned for buffered folds, packed columns and 16-lane masks.
        # The partials/reduce oracle covers both geometries, cloud modes,
        # Planck paths and tile widths. Not an mp=8 unit.
        # Re-pinned for A146 (a98f2482e): every float division by a compile-time
        # constant is spelled __fdiv_rn, because NVRTC compiles x / C as a
        # multiply by the rounded reciprocal on Blackwell targets.  Reading:
        # sm_89 unchanged (the compute_89 PTX with __fdiv_rn spelled '/' is
        # identical to the base's); Blackwell cards now round these quotients
        # IEEE-correctly.  Previously b33d69b0/ceedc6be.
        # Re-pinned 2026-10-01 for vector coefficient/adding storage.
        # The serial recurrence and fold order are unchanged; direct output
        # words match on RTX 4090 and RTX 5090.
        # Re-pinned 2026-10-19 (compiled string only; the .cu is unchanged):
        # rad-history 29e917cbeb9 adds rrtmgp_planck_staged_default.cuh to the
        # default rrtmgp_rte loader closure. Reading: Planck sources bitwise to
        # the standalone kernel on 61,440,000 lay_source words; the A3 01Z
        # history frame differs in 0 of 139 variables. Previously 5d3c8dde.
        'fc10756eba4cc437572b3f141cd5746c6149b78b312b79d3386217144e8200d9',
        '5c251aa2b5dfc420f4fd8f0dbacba3459aff0654c6f685246e5fbfcbf7c68b86'),
    'rrtmgp_validation': (
        'c3a2554827e7c39db269d0fa1d5ad594d1623d6974be859a1ae3a044d438951f',
        '36f7a4dccc8c66be225e16fdd6088f0fafa7a357ac1aba9b4a99f98af297c370'),
    # Re-pinned for the WOOF 1.0.0 text scrub: comments only (punctuation, host
    # labels), proven equal with comments stripped; the preprocessor drops
    # comments, so the compiled binary does not move.
    'ruc': (
        # 2.8.4: WRF mosaic SOILVEGIN parameters and post-SFCTMP irrigation.
        # Twelve surface and 48 full-column WRF oracles match every bit;
        # the mixed-grid fused checks cover all selector pairs over three
        # steps. This pin prevents changed category order or missing root
        # irrigation from silently changing the surface water and fluxes.
        # Re-pinned for the RUC_NZS tier ladder (lane/ruc-column-nzs, step 3
        # of 3, final): the ladder, the 161-token macro substitution that
        # uses it, and the `#elif RUC_NZS == 6` arm of the depth table.  The
        # forecast column now compiles at both geometries WRF defines.
        #
        # The nine-level translation unit is unchanged through all three
        # steps.  Every macro expands to a bare decimal literal, so
        # `real zshalf[RUC_NZS]` is the token stream `real zshalf[9]` was,
        # and the `#elif` arm is deleted by the preprocessor at nine.
        #
        # The pin moves; the compiled binary does not.  Proved three ways in
        # tests/test_ruc_nzs_tier.py, all device-free: the shipped source is
        # run BACKWARDS to the pre-lift file and hashes to
        # d446b7462e4952416d3e21482b051823766a6f675163236686c7d9fab7fbbdb7
        # (the digest this row carried before), the token streams out of a
        # real host preprocessor are equal, and `nvcc -ptx` emits identical
        # PTX.  Each comparison has a negative control that must differ.
        #
        # MOVED A SECOND TIME, and this time the binary DOES change: the
        # `RUC_NZS DZSTOP` block.  ruc_soil_finalize computed
        # `dzstop = 1 / (0.01f - 0.0f)` -- WRF's NINE-level
        # zsmain(2) - zsmain(1), written as a literal rather than read from
        # ruc_soil_layer_depth, and therefore invisible to the macro sweep
        # above, which was looking for extents.  Harmless while nine was the
        # only geometry; at six it divides by 0.01 where the grid says 0.05,
        # and the kernel returned a ground heat flux five times too large --
        # MEASURED as grdflx -337.1 W m-2 against the host path's -67.4
        # on the same column, before the fix.
        #
        # WHY THIS ROW MAY MOVE FOR IT.  At nine levels
        # ruc_soil_layer_depth[1] IS 0.01f and [0] IS 0.00f, so the
        # subtraction, the divide and every number downstream are unchanged,
        # and that is measured on the hardware rather than argued: the full
        # 43-field host/device driver comparison in
        # tests/test_ruc_nzs_device.py is max_ulp 0 at nine, over snow-free,
        # water+ice and snow columns, and the eleven-file RUC suite is
        # unchanged against its lane baseline.  The generated code is NOT
        # unchanged -- a __constant__ load where an immediate was -- and
        # test_the_named_fix_is_a_real_change_to_the_generated_code asserts
        # that difference, so the fix cannot be waved through as inert.
        # The reconstruction above still reaches
        # d446b7462e4952416d3e21482b051823766a6f675163236686c7d9fab7fbbdb7,
        # through one extra named inversion step for this block, which is
        # what keeps "the lift changed nothing else" checkable.
        #
        # 2.8.5, SOILPROP by WRF lineage (ruc_soilprop).  ruc_soil_properties
        # compiles WRF v4.0-4.5's soil-water diffusivity and conductivity,
        # normalised by the moisture above the residual, and mineral
        # conductivity 2.0 at every quartz fraction (v4.5.2
        # module_sf_ruclsm.F:6154, 6213-6216, 6245; the same lines in the
        # operational RAP/HRRR branch); GPUWM_SOILPROP_WRF461 compiles the
        # v4.6.1 lines this unit carried before (:6198-6202, 6261-6267,
        # 6289), and under that define every WRF v4.6.1 column oracle still
        # matches as before (tests/test_ruc.py, test_ruc_gpu.py,
        # test_ruc_mosaic_gpu.py).  WHAT THE PIN PREVENTS: the v4.6.1 form
        # returning as the default.  Measured on a 3 km October afternoon
        # cut of an operational-HRRR start (106,671 land cells): it refills a
        # dry top soil level from below, 0.161 to 0.187 m3/m3 in the first
        # hour against 0.173 under this default and 0.157 in the operational
        # model, with latent heat 207 against 186 W/m2 (operational 142).
        # tests/test_ruc_nzs_tier.py inverts the five edits by name before
        # its historical digest, so the lift proof above still closes.
        # Previously 57e8c503/39d88927.
        #
        # 2.8.6, the snow scheme by WRF lineage (ruc_snow).  The default
        # compile is WRF v4.0-4.5's SFCTMP snow preparation and SNOWTEMP,
        # which the operational RAP/HRRR branch carries (constant snow
        # conductivity, cover from depth over the critical depth taken after
        # compaction, fresh-snow albedo from the depth on the ground, no 0.7
        # albedo floor, no freezing clamps on the second pass, melt entered
        # while the pack outlasts the step's evaporation with a cap
        # independent of the step, bottom melt capped unconditionally and
        # kept out of smelt); GPUWM_SNOW_WRF461 compiles the v4.6.1 lines
        # this unit carried before, and under that define every WRF v4.6.1
        # column oracle still matches as before (tests/test_ruc.py,
        # test_ruc_gpu.py); the default matches the branch build bit for
        # bit (tests/test_ruc_fork_oracle.py).  WHAT THE PIN PREVENTS: the
        # v4.6.1 snow scheme returning as the default, or either lineage
        # changing unseen.  tests/test_ruc_nzs_tier.py inverts the edits by
        # name before its historical digest.  Previously 844546e3/0af961d3.
        #
        # lane/verify-ruc-lsm, RUC's float32 libm words.  Every EXP, LOG,
        # LOG10, TANH, COS and REAL**REAL the RUC column evaluates moved from
        # a float64 evaluation rounded once (and CUDA's logf/cosf) to WOOF's
        # float32 routines: gfk_pow/gfk_exp/gfk_log, glibc_cosf (the module
        # now also takes glibc_trig_flt32.cuh) and the log10f/expm1f/tanhf
        # reductions in ruc.cu.  The binary changes on purpose.  WHAT IT
        # SHIPPED, measured by tools/ruc_lsm_gpu_oracle against unmodified
        # WRF v4.6.1 on box W1 (102 columns, six configurations, 8-12 steps,
        # strict and default arithmetic): 0 ULP on every compared word; before
        # it, 1,309 of 78,744 words differed on the nine-level case.  The
        # lsmruc.csv and sfctmp.csv residue maps (26 and 11 cells) are empty.
        # tests/test_ruc_nzs_tier.py inverts the hunks by name before its
        # historical digests.  Previously f2de42c4/48bb40b7.
        'f435298be63975bee606873f5aba969e4faae0ed1a506a8da9a973cca9dc10e2',
        '0b370cd9e57475cfb73718ead60784dd345d80e26f31c903e3ebacd7a0788af5'),
    'saxpy': (
        # 2.8.2: four values per thread with aligned vector loads/stores.
        # Scalar a*x+y and the default FMA policy are unchanged. The word
        # comparisons in test_bandwidth_word_kernels.py cover NaNs,
        # subnormals, misaligned slices, tails and exact in-place output.
        # This infrastructure kernel has no dedicated WRF Fortran oracle.
        'a057d932a4547a865da5458b3c6d01ff1d7551a58554cbbe0e2198a53f46387c',
        'e3bf69cd521ed814b1d25350dc81a07c43b1bb8934d0a77197c261cd981b9d86'),
    # Re-pinned for the WOOF 1.0.0 text scrub: comments only (punctuation, host
    # labels), proven equal with comments stripped; the preprocessor drops
    # comments, so the compiled binary does not move.
    'sfclay': (
        # Re-pinned on the 1.6 release line for the sm_120 DAZ hardening.
        # A roughness small enough to overflow the FP32 quotient sent Inf
        # into every downstream similarity quantity, so the logarithm is
        # rescued in float64 and a NaN roughness now propagates instead of
        # being floored into a plausible finite contrast.  sfclay is not an
        # mp=8 translation unit and thompson.cu is byte-unchanged across
        # this line, so the mp=8 numerics guarantee is untouched; the
        # healthy path is proven bit-exact under pinned fixtures and an
        # adversarial sweep.
        #
        # RE-PINNED by the WRF-parity USTM repair (SFC-01).  The kernel
        # produced no USTM at all: module_sf_sfclay.F:800-804 (and
        # physics_mmm/sf_sfclayrev.F90:759-763) relaxes a SECOND friction
        # velocity on WSPDI = sqrt(ux*ux+vx*vx) -- the speed WITHOUT the
        # Beljaars/Mahrt-Sun vconv/vsgd correction, without WSPD's 0.1 floor
        # and without UST's land floor -- and USTM is unconditional Registry
        # state (Registry.EM_COMMON:1954) that WRF hands to tke_rhs and
        # vertical_diffusion_2 (module_first_rk_step_part2.F:914,:1066).
        # ArWen recomputed it in a CuPy post-pass gated on
        # km_opt in (2,3,4) and bl_pbl_physics == 0; that post-pass is now
        # deleted and the line lives where WRF has it.  This MOVES ANSWERS:
        # <=1 ULP on the LES path (the kernel contracts uu*uu+vv*vv into an
        # FMA where the three CuPy kernels did not) and first-order for
        # km_opt=2 with a PBL scheme on, where the TKE surface shear source
        # was identically zero.  sfclay is not an mp=8 translation unit and
        # thompson.cu is byte-unchanged, so the mp=8 numerics guarantee is
        # untouched.  Measured on the CPU authority by tests/test_sfclay.py::
        # test_ustm_relaxes_on_the_uncorrected_wind_speed and ::
        # test_ustm_takes_neither_the_wind_floor_nor_the_land_floor, with the
        # kernel held to the float64 mirror by ::
        # test_sfclay_kernel_writes_ustm_and_not_a_copy_of_ust and by every
        # standing SFCLAY_OUTPUTS sweep, which now grades ustm.
        # Re-pinned for A146 (a98f2482e): every float division by a compile-time
        # constant is spelled __fdiv_rn, because NVRTC compiles x / C as a
        # multiply by the rounded reciprocal on Blackwell targets.  Reading:
        # sm_89 unchanged (the compute_89 PTX with __fdiv_rn spelled '/' is
        # identical to the base's); Blackwell cards now round these quotients
        # IEEE-correctly.  Previously 1e068781/c26f5f05.
        # Revised MM5 flux-off carry and land scalar EXP overflow fixes:
        # tests/test_sfclayrev_flux_disable_replay.py and
        # tests/test_sfclayrev_nonfinite.py grade this surface-layer slice;
        # tools/sfclayrev_wrf461_oracle/stress/check.py hashes every
        # WRF-finite field word. PHYSICS.md declares the overflow divergence.
        # Thompson's own translation unit is unchanged.
        # 2.8.8 (gfix/288-gpu-shards): the composed unit's sfclay_classic.cuh
        # keeps the incoming CHS/CHS2/CQS2 words when ISFFLX=0, as WRF and
        # the revised scheme do (tests/test_sfclayrev_flux_disable_replay.py
        # option 91); sfclay.cu is byte-unchanged.  Previously 650d6937.
        # Re-pinned 2.9.2 (fix/292-box-sfclay-compile): sf_log/sf_exp/sf_pow
        # are __noinline__ instead of __forceinline__, because inlining the
        # glibc libm words at every call site made this one module take
        # 114.6 s to compile on a cold cache (14.7 s now).  Reading: options
        # 1 and 91 over the stored WRF 4.6.1 surface fixture (6 arms x 3
        # steps x 110 columns x 34 outputs, 67,320 words each) are
        # byte-identical to the previous source under GPUWM_NUMERICS=fast and
        # =strict.  Previously 044c3246/45a053bb.
        # Re-pinned 2.9.10 (fix/sfclayrev-strict-o2-squares): the four x**2.
        # powers of psim/psih_unstable_full are products, as the pinned strict
        # wrf.exe 4.6.1 (gfortran 13.3 -O2) folds them; as powf calls they put
        # 1-ULP surface fluxes in 3 of 12,000 UAE 9 km columns after one
        # physics-on step.  Graded against the strict-recipe oracle
        # (tools/sfclayrev_wrf461_oracle/build_strict.sh).  sfclay is not an
        # mp=8 translation unit.  Previously 09ff5be4/010efce3.
        '4b12d778f6407625c625ee59601ee4816ace432b25c2491b0045837b7d0fc95b',
        'a204e41d2014bcb1f14dac9e950f823cecf08acd6e066ec8dba77cced9142fad'),
    # Re-pinned for the WOOF 1.0.0 text scrub: comments only (punctuation, host
    # labels), proven equal with comments stripped; the preprocessor drops
    # comments, so the compiled binary does not move.
    'smag2d': (
        # Re-pinned on the 1.5 integration line: feature/les-integration's
        # verified km_opt=2/3 work edits smag2d.cu after this table was
        # frozen on the mp28 lane.  smag2d is not an mp=8 translation unit;
        # the mp=8 numerics guarantee is untouched.
        #
        # Re-pinned again on the P1 moist lane, which adds the spec 3.3
        # MUTATION CONTROL to wrf_calc_n2: an `#ifdef
        # GPUWM_MUTATE_MOIST_N2_FORCE_DRY` block that assigns
        # `saturated = false` for instrument qualification.  Both digests
        # move because both are digests of SOURCE TEXT.  The generated code
        # of the production build does not: the guard's macro is never
        # defined in the assembly `load_module` builds -- asserted by
        # tests/test_les_moist_n2_mutation.py
        # ::test_the_mutation_define_is_absent_from_the_production_assembly
        # -- so the preprocessor deletes the block before nvrtc sees it, and
        # the production predicate line above it is byte-identical to what
        # it was.  The mutant is a SEPARATE translation unit compiled
        # through load_module_int_defines under its own cache key and its
        # own kernel-manifest entry; it is never this one.
        # Re-pinned for A146 (a98f2482e): every float division by a compile-time
        # constant is spelled __fdiv_rn, because NVRTC compiles x / C as a
        # multiply by the rounded reciprocal on Blackwell targets.  Reading:
        # sm_89 unchanged (the compute_89 PTX with __fdiv_rn spelled '/' is
        # identical to the base's); Blackwell cards now round these quotients
        # IEEE-correctly.  Previously c57ebd81/2e5a33dc.
        # Re-pinned for the compiled WRF v4.7.1 diffusion oracle (83fde6032,
        # merged dd4908a0f): WRF's boundary metric extension and tensor donors,
        # excluded outer mass rows in stability and the TKE source, masked
        # vertical TKE self-diffusion, and the prescribed-heat HFX refresh
        # (9,955,411 ULP to 1 ULP; tests/test_diffusion_drivers_wrf471_parity.py).
        # Previously eedf1fb3/975ec1e4.
        # Re-pinned for the opt-in strict WRF arithmetic (lane/282-wrf-exact,
        # merged 1efb5a415): 4d3d642bd (feat(verify): match active WRF diffusion
        # arithmetic and boundaries) adds `#ifdef GPUWM_WRF_EXACT_C_DIFFUSION`
        # branches, which the preprocessor keeps only when a GPUWM_WRF_EXACT
        # selector defines the macro.  Reading: with no selector,
        # module_source('smag2d') at f3e4ca716 and at c2946a54e compiles to
        # byte-identical PTX, the whole text and 27 of 27 .entry kernels, for
        # compute_89, compute_90 and compute_120 under the loader's options and
        # under the RawModule options CuPy compiles with, with NVRTC 13.4 and
        # with 12.9 (tools/kernel_ptx_identity/receipts/,
        # wrf-exact-default-2.8.2-nvrtc13.4.json and -nvrtc12.9.json), so the
        # default build does not move.  Previously 6b584aba/381bbd6f.
        # Ordinary scalar interpolation and sm_120 W stress reuse preserve
        # output words in the focused scalar and W identity tests. Strict
        # WRF source and cubins remain unchanged. Previously 0aab5788/c9165bb9.
        # Explicit stress, interpolation and scalar rounding preserves the
        # native sm_120 words in 248 randomized kernel fixtures and the
        # full short forecast. The direct W evaluator uses the same rounded
        # face arithmetic without reading the production face buffers.
        # Re-pinned for 1f625334f (perf: bounded 32-bit addressing for the
        # scalar Smagorinsky kernels, merged 8a9c9e63d): a GPUWM_SMAG_INDEX32
        # tier the host selects only after proving the largest staggered
        # allocation fits uint32; floating expressions and their explicit
        # rounding are unchanged.  Reading: byte-identical forecasts on Ada
        # and Blackwell (merge 8a9c9e63d), tests/test_smag2d_index32_*.py,
        # and all 178,128 diff_opt=1 coordinate words reproduced on an RTX
        # 5090 at this source (tests/data/wrf471_diff_opt1/
        # measured-source-transition.json).  Previously 1c8c29c1/565ad5fa.
        # Re-pinned for lane/verify-base4 (combo sweep round 3): strict-only
        # operation order inside `#if GPUWM_WRF_EXACT` -- vertical_diffusion_u_2/_v_2's rdz = -g/dnw then tend - rdz*diff.
        # Reading: with no selector the default smag2d PTX is byte-identical to
        # 932359b9/400d89d0 for compute_89/90/120 under both option sets
        # (tools/kernel_ptx_identity/receipts/wrf-exact-round3-default.json); the
        # strict words are graded in tests/test_wrf_exact_round3_gpu.py and against
        # WRF's own dumped words (tools/wrf_exact_localize).  Previously
        # 932359b9/400d89d0.
        '731d7f64e86af2d17344be7328fda2a3ecfd4f2e22371f6093128ca2f3192500',
        # smag2d's default composition keeps 2.9.1's glibc_flt32 text
        # (glibc_flt32_default.cuh); the exact chain's header is strict-only.
        '2cdfe1e9b38cfd901a837460526bed5887285bb367593992fd95b95df15e3d24'),
    'spec_bdy': (
        'bcc7090fbbb8ea307bd6dd6c65ab9b8a3f56948c4752ae3d744127b450d20161',
        'bc03ed595bacc546d8e041fbb1d11b5bb3b3b90760ef06ea1dd1f0f18b4de931'),
    'thompson': (
        # RE-FROZEN 2026-09-17 at 4ae7913df (fix(mp8): preserve rain
        # concentration and condensation history, shipped in 2.7.4): the
        # rain mass and number concentrations formed before cloud
        # adjustment persist until evaporation refreshes them at its own
        # incoming density, and a positive condensation decision
        # suppresses same-call rain evaporation, as WRF's
        # module_mp_thompson.F:3236, :3502 and :3568 do.  Its reading is
        # tests/fixtures/thompson-active-collision.json (three complete
        # classic columns through the pinned v4.6.1 driver),
        # tests/test_thompson_active_collision.py (the adapter holds
        # them at rtol 1e-5 on the mass and number fields and 2 ULP on
        # theta, and pins both mechanisms directly) and docs/thompson-
        # active-collision-accounting.md; the module docstring quotes
        # the numbers.  It did not re-freeze this entry, so the seven
        # thompson assertions in this file were red on the 2.7.4 and
        # 2.7.5 tips.  Previously 3ca6b7e9/8cb23f0a from the 1.4.1
        # re-anchor.
        # RE-FROZEN 2026-09-24 at 7727fda3c by the thirteen WRF v4.6.1
        # real-column repairs the module docstring's second reading lists
        # (the process rates, final state and echo of classic Thompson held
        # to WRF's own Fortran on 137,200 saved real-data columns, echo from
        # up to 43.9 dB off to within 0.045 dB).  Previously
        # 938bf573/f9b8547f from 4ae7913df.
        # RE-FROZEN 2026-09-30 by Thompson's exact shortcuts and
        # level-parallel fallout, byte-identical in bit tests and three 1 h
        # forecasts (see THOMPSON_CU_SHA256).  Previously d77977dc/e2ea3185.
        # RE-FROZEN 2026-09-30 by the fused classic adapter launches,
        # byte-identical in bit tests and three 1 h forecasts (see
        # THOMPSON_CU_SHA256).  Previously d0997611/fd8a053a.
        # Re-pinned for A146 on top of that: every float division by a
        # compile-time constant is spelled __fdiv_rn, because NVRTC compiles x
        # / C as a multiply by the rounded reciprocal on Blackwell targets.
        # Reading: sm_89 unchanged; Blackwell cards now round these quotients
        # IEEE-correctly. Previously a7955418/0508e4cc.
        # Re-frozen again by the A146 review repair (see THOMPSON_CU_SHA256).
        # Re-pinned for the network occupancy change and raw-word checks
        # in tests/test_thompson_speed_blocks.py; see THOMPSON_CU_SHA256.
        # RE-FROZEN 2026-10-08 for the one-slab rain-graupel reads
        # (thompson_racg_index); see THOMPSON_CU_SHA256 for the reading.
        # Previously 1ab8d031/b3bbf88b.
        '43f76b17febd32acdfe96e9cca4d9c932ce2d72c5a5378ed73b5f3eac33998d8',
        '35ac29fc801c66fc1eb64506267ded1efa2374f0998b0f80eea72f2afd5aa34a'),
    'uh_diag': (
        'cbfc98e8d025a4511fd7f8a41ca4bd163c261da4a48dec22bb979ec5a496b14e',
        '9dc88c6e14b2aaaa4249a9f844dc231f105431623375c988a2894e322de2f3ea'),
    # Re-pinned for the WOOF 1.0.0 text scrub: comments only (punctuation, host
    # labels), proven equal with comments stripped; the preprocessor drops
    # comments, so the compiled binary does not move.
    'vert_interp': (
        # Re-pinned for the source-column tier ladder: the unconditional
        # `#define WRF_VI_MAX_LEVELS 64` became an `#ifndef` guard around
        # the same literal (plus a comment), so woof/ingest/vert.py can
        # compile the WRF-real vertical kernel at 160 and 256 levels for
        # deep sources.  vert_interp is not an mp=8 translation unit and
        # the mp=8 numerics guarantee is untouched.  The guard is a
        # preprocessor no-op at the default tier: NVRTC emits
        # byte-identical PTX for the old and new source at compute_86,
        # compute_89 and compute_120, and tests/test_wrf_vert_interp.py
        # holds every tier to identical output bytes.  Previously
        # ab608d65/65b0fe8a.
        # Re-pinned for A146 (a98f2482e): every float division by a compile-time
        # constant is spelled __fdiv_rn, because NVRTC compiles x / C as a
        # multiply by the rounded reciprocal on Blackwell targets.  Reading:
        # sm_89 unchanged (the compute_89 PTX with __fdiv_rn spelled '/' is
        # identical to the base's); Blackwell cards now round these quotients
        # IEEE-correctly.  Previously d03c5656/f23a2817.
        # Re-pinned for gp-vert: Rust-order RN arithmetic, glibc pressure
        # log/pow, and bit-preserving tiny field operands, merged over A146's
        # __fdiv_rn spelling of the other entry point (vertical_interpolate_logp).
        # Previously 339e7266/b532daea on the lane and 3efb1a82/1c882556 on
        # integrate/2.8.
        # Native endpoint roundoff (69a0baec1): targets within four FP32
        # epsilons of the top pressure use that endpoint, matching the Rust
        # operator. tests/test_wrf_vert_interp.py grades both routes. The
        # file pin moved in b6f34356b; its assembled-source pin did not.
        # Re-pinned for 508691422 (fix(vert): co-locate a native moist top
        # within 2^-16 of its dry target, merged 2cc4baa2f): the top
        # co-location bound widens from four FP32 epsilons to 2^-16
        # relative in the kernel, the Rust operator, the Rust plan and the
        # launcher check alike, so an HRRR-native start whose top moist level
        # sits 4.9e-7 above the dry target no longer refuses.  Runs that
        # passed before are unchanged; tests/test_native_pressure_top.py and
        # tests/test_wrf_vert_interp.py grade both routes.  Previously
        # 1787bbbf/7ea882cb.
        'dadb7dff57ab3f7440bcec9700d28a7c1621dbddf35922ce27476c34986a6bf9',
        # Re-pinned for c0ffc53b5 and 2bd3bc505 shared-header additions.
        # The kernel file is unchanged. The added unused helpers disappear
        # from compiled code: sm_100 and sm_120 each retain identical ordered
        # PTX and SASS operations, operands and predicates against the header
        # before c0ffc53b5. Reading: vert-interp-preservation.json.
        # Moved again with the file by 508691422 (above).
        '7e9a3763bb02f7665250aa9687d2ee0398b2a9cd6cf0ffde4f5fc961dba47be3'),
    # Re-pinned for the WOOF 1.0.0 text scrub: comments only (punctuation, host
    # labels), proven equal with comments stripped; the preprocessor drops
    # comments, so the compiled binary does not move.
    'wsm6': (
        # Re-pinned for the column-workspace move (memory census item 12,
        # lane/292-mem-cuts 052440f32b4): the fifteen column arrays, the
        # sedimentation arrays and the reflectivity kernel's arrays left the
        # per-thread local frame for a launcher-owned global workspace,
        # stride-32 lane indexing, tile offset col0.  Placement only, unit
        # still --fmad=false: tests/test_wsm6_wrf461_column_exact.py 0
        # mismatches at both tiers, and 0 of 128,567,124 words differ from
        # integrate/2.9.2 on a 210,077-column state at the default tile and
        # at a 1,024-column tile.  Previously 3ca83093/59764c7a.
        # Compile string re-read at the 2.9.2 merge of lane/292-mem-cuts onto
        # the 2.9.1 forward merge: the lane wsm6.cu under the 2.9.1 recipe
        # (2.9.1 alone pinned ce5963f7); the file sha256 is the lane one.
        # Before that, re-pinned for correct mode, finding WB5 (correct/mpas-water-budget-290):
        # the WSM6_PLM_SURFACE arm books the below-ground part of the
        # straddling sedimentation cell with the remap's own sloped profile.
        # With the macro undefined (exact mode) the preprocessed unit is the
        # c8e0db79 bytes and NVRTC returns the same image
        # (tests/test_correct_mode_water_budget.py).  Previously c8e0db79:
        # RE-PINNED by lane/verify-wsm6-mp6 (2026-10-07), and this MOVES
        # ANSWERS on the default path: the kernel is rewritten as WRF
        # v4.6.1's own WSM6 call chain (module_mp_wsm6.F wrapper, mp_wsm6_run,
        # effectRad) plus refl10cm_wsm6 as a second kernel, with WRF's
        # association, mp_wsm6_init's float32 words, WOOF's own float32 and
        # binary64 libm words (glibc_flt32.cuh, glibc_flt64.cuh) and the unit
        # compiled with --fmad=false (_NO_FMAD_MODULES).  Reading (W1 RTX PRO
        # 6000, sm_120, NVRTC 12.9.86 and 13.4.92):
        # tests/test_wsm6_wrf461_column_exact.py, 0 of 1,351,090 output
        # words differ from gfortran WRF over 65 columns and six runs under
        # GPUWM_WRF_EXACT=1; the default build differs only in the two
        # columns whose WRF words are subnormal.  Previously 0028b0c2/d6798c07
        # (A146, a98f2482e), before that 0526192b/1a6d20da.
        'a4aa666e9d2011bd6fca29a9577978cbc66b3dd89a943c49af282cfd79e75dfc',
        # Compile string re-pinned with the file (correct/mpas-water-budget-290
        # repair, 2026-10-13): the WB5 commit above left the old file digest
        # c8e0db79 in place as a third element, so the entry no longer
        # unpacked.  The loader's string now carries the #ifdef
        # WSM6_PLM_SURFACE block (218,504 characters); with the macro
        # undefined the preprocessed unit and NVRTC's image are the
        # c8e0db79/b31ff1ff ones (test_wb5_exact_mode_compiles_the_same_ptx_as_2_8_7).
        # Previously b31ff1ff.
        '93d94836eedf855f14a96029b5357c0facc498732c613d0123a2a31f409fc65f'),
    'ysu': (
        # Re-pinned for the column-workspace move (17cf943ef): per-thread
        # local arrays became a caller-sized global workspace, stride-32
        # lane indexing, tile offset col0.  Placement-only -- proven by
        # --fmad=false digest equality (0 differing words of 903,168) and
        # the WRF v4.6.1 ULP-equality parity pins, with the instrument
        # validated in both directions
        # (docs/kernel_local_memory_bounds.md).  ysu is not an mp=8
        # translation unit; the mp=8 numerics guarantee is untouched.
        #
        # RE-PINNED AGAIN by the WRF-parity pblflg repair (par-pbl-ysu-01
        # and -04).  Two branch structures, both re-read at
        # phys/physics_mmm/bl_ysu.F90 before the edit: :703-728 guards the
        # thermal-enhanced Richardson sweep with if(pblflg(i)) and the
        # kernel did not, so a column WRF holds in the local-K regime was
        # being switched to full non-local YSU at convective onset; and
        # :765/:766 are two INDEPENDENT statements, so nesting the second
        # inside the first let a theta-li revival reach :832 with kpbl == 1
        # and index one level below the column.  This MOVES ANSWERS on both
        # paths -- see the release note -- and both are measured on the CPU
        # authority by tests/test_ysu.py::
        # test_the_thermal_enhanced_sweep_cannot_raise_pblflg and ::
        # test_a_theta_li_revival_that_leaves_kpbl_at_one_is_extinguished,
        # with the kernel's own spelling held to the mirror's by ::
        # test_the_kernel_spells_wrfs_two_pblflg_rules_like_the_mirror.
        # The same bytes also carry par-pbl-ysu-03 (the enhanced sweep's
        # result reaches the theta-li scan unclamped, because :718-728 has
        # no counterpart to the clamp at :646 and :823), landed in the same
        # window by the lane that owns it; the kernel and
        # woof.verify.npref.np_ysu_column carry it identically.
        #
        # RE-PINNED by WRF's flag_bep arm (dc7532c7f; sf_urban_physics 2/3):
        # the column body became `template <bool BEP>` and every BEP
        # statement sits under `if constexpr`, so ysu_column (BEP = false)
        # compiles from the statements it always had; a new entry point,
        # ysu_column_bep, carries the arm.  The non-BEP numerics are held
        # unchanged by tests/test_ysu_wrf461_parity.py (its ULP table did not
        # move, under NVRTC 13.4.92 and 12.9.86 alike) and the BEP arm is
        # graded by tests/test_ysu_bep_wrf471_parity.py.  Previously
        # 251dc846/4725831d.
        #
        # RE-PINNED by the declared rural-drag divergence (sf_urban_physics
        # 2/3 under YSU only): the flag_bep arm removes the whole of YSU's
        # own first-level drag instead of WRF's urban fraction of it, because
        # the BEP couple already carries the rural drag in a_u_bep and WRF
        # counts it twice.  One expression under `if constexpr (BEP)`;
        # ysu_column (BEP = false) compiles from unchanged statements.
        # Graded by tests/test_ysu_bep_wrf471_parity.py against a WRF build
        # with exactly that one-line change and by
        # tests/test_ysu_bep_rural_drag.py (a non-urban column under BEP
        # against the same column with urban off).  Previously
        # a42a4137/28be0b25.
        #
        # Re-pinned for A146 (a98f2482e): every float division by a compile-time
        # constant is spelled __fdiv_rn, because NVRTC compiles x / C as a
        # multiply by the rounded reciprocal on Blackwell targets.  Reading:
        # sm_89 unchanged (the compute_89 PTX with __fdiv_rn spelled '/' is
        # identical to the base's); Blackwell cards now round these quotients
        # IEEE-correctly.  Previously 251dc846/4725831d.
        #
        # MERGED with integrate/2.8 (urban) on lane/281-nvrtc-literal-div:
        # both changes above are in the file, so the digests are the
        # merged file's.  Previously 43dba307/7412be7d.
        #
        # RE-PINNED by WRF's topo_wind arm (lane/282-terrain-drag; topo_wind
        # = 1 or 2 under YSU): bl_ysu.F90's ctopo-present surface drag
        # (:1254-1314, the paj TKE profile, get_pblh and the Beljaars
        # convective velocity) and the hill-top 10 m blend (:1402-1408) under
        # `if constexpr (TOPO)`, entered only by the new ysu_column_topo; the
        # loader now prepends glibc_flt32.cuh for that arm's powf and
        # ysu_topo.cuh for its own pieces (get_pblh and the 10 m blend, kept
        # out of ysu.cu so every line before the momentum assembly keeps its
        # number: the registry and the FTZ claim census cite them).  Reading:
        # ysu_column and ysu_column_bep compile to byte-identical PTX before
        # and after (compute_120 and compute_89, -std=c++17 -ftz=true, the
        # loader's effective options), and tests/test_ysu_wrf461_parity.py's
        # ULP table did not move on either card.  The arm is graded against
        # WRF v4.7.1 by tests/test_terrain_drag_wrf471_parity.py.
        # Previously 41b9a899/57842bb8.
        #
        # RE-PINNED by lane/parity-286 (WRF parity sweep rows 2 and 9), and
        # this one MOVES ANSWERS on the default path.  (a) bl_ysu.F90:613
        # `br > 0` is compared in double through the bit-decoding ysu_f2d
        # (ysu_topo.cuh), because sm_120 DAZes a positive subnormal br in a
        # float32 compare and the column took the convective arm where WRF
        # takes the stable one.  (b) ysu_column and ysu_column_bep take WRF's
        # ctopo surface-drag arm (bl_ysu.F90:1308) with ctopo = ctopo2 = 1,
        # which module_bl_ysu.F:404 passes on every WRF run, instead of the
        # ctopo-absent :1315 no WRF run reaches.  Reading (box F RTX 5090,
        # sm_120, NVRTC 13.4): tests/test_ysu_wrf461_parity.py case 7 joins
        # the arithmetic table (wstar/delta 0 as WRF, momentum 4.5e7 -> 1457
        # ULP, theta 1.8e9 -> 0), du/dv graded against WRF's ctopo call with
        # maxima unchanged at 1457/23302, case 9 dv 22 -> 0; the BEP table
        # and the topo_wind arm's table did not move.
        # Previously d95a63ea/2d0e714d.
        #
        # RE-PINNED by lane/parity-286 (sweep row 10), and this MOVES ANSWERS
        # on the default path: glibc's powf/expf (gfk_pow/gfk_exp) in place
        # of CUDA's, every real-exponent `**` spelled as gfortran's powf call,
        # WRF's association for chi/temps/prnumfac, WRF's thx = (th*pi)/pi and
        # (ttend*pi)/pi round trips (ysu_thx), and the unit compiled with
        # --fmad=false (kernels/__init__.py _NO_FMAD_MODULES).  Reading (box K
        # RTX PRO 6000, sm_120, NVRTC 13.4.92 and 12.9.86 identical):
        # tests/test_ysu_wrf461_parity.py goes from du/dv 1457/23302, exch 7,
        # hpbl/wstar/delta/dtheta 1 ULP to bitwise WRF on every lane but the
        # 11 FTZ-flushed subnormal tendencies; the BEP table and the topo_wind
        # table go to 0 the same way.
        # Previously 46f4642c/bce0da68.
        #
        # RE-PINNED by golden/ports-pbl (2026-10-12), and this MOVES ANSWERS
        # wherever ust = hfx = qfx = 0 (or ust flushes to zero on the
        # card): the zero-flux short circuit is removed.  It returned zero
        # tendencies, PBLH = dz(1) and KPBL = 1 where bl_ysu_run has no such
        # branch and runs the whole scheme.  Reading (G1 RTX PRO 6000, sm_120,
        # NVRTC 12.9): tests/test_ysu_wrf461_parity.py case 12 joins the
        # arithmetic table at 0 ULP on every field (KPBL 9, PBLH 703.36 m, as
        # WRF), case 13 is WRF's word on every lane WRF wrote a number with
        # KPBL/PBLH exact, and the 23-column maxima did not move.  No other
        # statement changed; every column with any nonzero flux compiles from
        # the same statements as before.
        # Slimming: the recorder follows the retained ysu_default.cu file.
        # slimming-loader-source-replay.json proves identical compiler text.
        'f56e3ec7646a111d07c5447383ae136b2959b5f4545441a598b51be16ccdc768',
        # ysu_default.cu names a test host instead of a lab host since 2.9.1 (comment only).
        '6261061b62ef15ca30cf3746baf9177cdc1a0eafc2544937f41c674e0fd5a932'),
}

# -- R2 --------------------------------------------------------------------

CLASSIC_TABLE_ASSETS_PIN = (
    ("qr_acr_qg_V4.dat", 74_966_480,
     "89b779855847b2acdca1b40e24c5f1bd89b0c6ed105ca91a5a076d80c2437c3f"),
    ("qr_acr_qsV2.dat", 43_764_288,
     "47350be20bd59c9f31378dd5805ce7d35fd14bebcfafb4ade56626f6eed818d7"),
    ("freezeH2O.dat", 254_944_848,
     "c235d1ce6f8750a671b2273d0e216ed3acf9a869bfd52a14676826f87aab5c02"),
    ("thompson_aux_tables.dat", 6_164_536,
     "a1bda803cdb53aedce8a2970c04c355fad19e3744398e1c9b13a876f09730547"),
)
TABLE_SET_ID_PIN = "wrf-v4.6.1-classic-thompson-mp8-gfortran13-v1"
WRF_REFERENCE_VERSION_PIN = "v4.6.1"
WRF_REFERENCE_COMMIT_PIN = "d66e442fccc04111067e29274c9f9eaccc3cef28"

# -- R3 --------------------------------------------------------------------

EXTRA_MOIST_SPECIES_MP8 = ("qi", "qs", "qg", "nr", "ni")
EXTRA_MOIST_SPECIES_MP10 = ("qi", "qs", "qg", "nr", "ni", "ns", "ng")
TRANSPORTED_NUMBER_SPECIES_PIN = ("nr", "ni", "ns", "ng")

# -- R4.  Captured for the fixed probe config nx=8 ny=6 nz=4,
# moist=True, mp_physics=8 (see freeze.PREFLIGHT_PROBE_CONFIG).
# ------------------------------------------------------------------
STATE_ARRAY_SHAPES_MP8 = {
    'al': (4, 6, 8),
    'al_pp': (4, 6, 8),
    'alb': (4,),
    'alt': (4, 6, 8),
    'c1f': (5,),
    'c1h': (4,),
    'c2f': (5,),
    'c2h': (4,),
    'c3f': (5,),
    'c3h': (4,),
    'c4f': (5,),
    'c4h': (4,),
    'cosa': (6, 8),
    # The EOS's float64-derived base-thickness correction and the
    # float64-differenced full-level coefficient drops that replaced the
    # two cancelling subtractions in calc_p_alpha.  Derived setup, priced
    # like every other allocation; see woof/core/kernels/diagnostics.cu.
    'dc3f': (4,),
    'dc4f': (4,),
    'dn': (4,),
    'dnw': (4,),
    'dphb_resid': (4,),
    'e': (6, 8),
    'effc': (4, 6, 8),
    'effi': (4, 6, 8),
    'effs': (4, 6, 8),
    'f': (6, 8),
    'fnm': (4,),
    'fnp': (4,),
    'h_diabatic': (4, 6, 8),
    'ht': (6, 8),
    'msft': (6, 8),
    'msfu': (6, 9),
    'msfv': (7, 8),
    # CONUS strict MF_VX_INV (paper/conus-suite-exact, 2.9.11): the stored v map-factor inverse WRF keeps,
    # ones unless a WRF input supplies it; the default route reads it nowhere.
    'msfv_inv': (7, 8),
    'mu_pp': (6, 8),
    'mub2d': (6, 8),
    'mup': (6, 8),
    'mup0': (6, 8),
    'ni': (4, 6, 8),
    'ni0': (4, 6, 8),
    'nr': (4, 6, 8),
    'nr0': (4, 6, 8),
    'p': (4, 6, 8),
    'p_pp': (4, 6, 8),
    'p_pp_old': (4, 6, 8),
    'pb': (4,),
    'ph_pp': (5, 6, 8),
    'phb': (5,),
    'php': (5, 6, 8),
    'php0': (5, 6, 8),
    'qc': (4, 6, 8),
    'qc0': (4, 6, 8),
    'qg': (4, 6, 8),
    'qg0': (4, 6, 8),
    'qi': (4, 6, 8),
    'qi0': (4, 6, 8),
    'qr': (4, 6, 8),
    'qr0': (4, 6, 8),
    'qs': (4, 6, 8),
    'qs0': (4, 6, 8),
    'qv': (4, 6, 8),
    'qv0': (4, 6, 8),
    'rdn': (4,),
    'rdnw': (4,),
    'rmu_t': (6, 8),
    'rph_t': (5, 6, 8),
    'rth_t': (4, 6, 8),
    'ru_t': (4, 6, 9),
    'rv_t': (4, 7, 8),
    'rw_t': (5, 6, 8),
    'sina': (6, 8),
    'th_pp': (4, 6, 8),
    'thb': (4,),
    'thp': (4, 6, 8),
    'thp0': (4, 6, 8),
    'u': (4, 6, 9),
    'u0': (4, 6, 9),
    'u_pp': (4, 6, 9),
    'v': (4, 7, 8),
    'v0': (4, 7, 8),
    'v_pp': (4, 7, 8),
    'w': (5, 6, 8),
    'w0': (5, 6, 8),
    'w_pp': (5, 6, 8),
    'ww_pp': (5, 6, 8),
    'znu': (4,),
    'znw': (5,),
}
SCRATCH_SLOT_REGISTRY_MP8 = {
    'acoustic_a': (5, 6, 8),
    'acoustic_alpha': (5, 6, 8),
    'acoustic_c2a': (4, 6, 8),
    # The 59f7e280f default-CQ correction allocates all three acoustic faces.
    # The unchanged R4 probe records nx=8, ny=6, nz=4; existing slots retain
    # their shapes and the state-array shape digest remains unchanged.
    'acoustic_cqu': (4, 6, 9),
    'acoustic_cqv': (4, 7, 8),
    'acoustic_cqw': (5, 6, 8),
    'acoustic_gamma': (5, 6, 8),
    'acoustic_mu_pp_old': (6, 8),
    'acoustic_th_pp_old': (4, 6, 8),
    'adv_ru': (4, 6, 9),
    'adv_rv': (4, 7, 8),
    'adv_rw': (5, 6, 8),
    'integration_health_aux_ptr': (2048,),
    'integration_health_bounds': (1024, 2),
    'integration_health_field_ptr': (2048,),
    'integration_health_field_size': (2048,),
    'integration_health_flags': (1024,),
    'integration_health_partial': (1, 9),
    'integration_health_planes': (1024,),
    'integration_health_result': (8,),
    # Read-only gate geometry, 12 KiB per domain; physics slots retain
    # their shapes and lifetime aliases.
    'integration_health_rim': (1024, 3),
    'integration_health_status_bits': (2048,),
    'integration_health_validation': (4,),
    'moist_pd_q0': (4, 6, 8),
    'moist_rq_t': (4, 6, 8),
    'mp_dz8w': (4, 6, 8),
    'mp_graupelnc': (6, 8),
    'mp_graupelncv': (6, 8),
    'mp_pii': (4, 6, 8),
    'mp_rainnc': (6, 8),
    'mp_rainncv': (6, 8),
    'mp_snownc': (6, 8),
    'mp_snowncv': (6, 8),
    'mp_sr': (6, 8),
    'mp_th': (4, 6, 8),
    'mp_thompson_frozen_reference_density': (4, 6, 8),
    'mp_thompson_frozen_reference_temperature': (4, 6, 8),
    'mp_thompson_graupel_melt_marker': (4, 6, 8),
    'mp_thompson_graupel_number_shadow': (4, 6, 8),
    'mp_thompson_micro_columns': (6, 8),
    'mp_thompson_rain_reference_density': (4, 6, 8),
    'mp_thompson_snow_melt_marker': (4, 6, 8),
    'mp_thompson_snow_velocity_boost': (4, 6, 8),
    'mp_thompson_temperature': (4, 6, 8),
    'mp_z8w': (5, 6, 8),
    'pd_fxc': (4, 6, 9),
    'pd_fxl': (4, 6, 9),
    'pd_fyc': (4, 7, 8),
    'pd_fyl': (4, 7, 8),
    'pd_fzc': (5, 6, 8),
    'pd_fzl': (5, 6, 8),
    'hail_density': (4, 6, 8),
    'hail_height': (4, 6, 8),
    'hail_max2d': (6, 8),
    'hail_max2d_run': (6, 8),
    'hail_maxk1': (6, 8),
    'hail_maxk1_run': (6, 8),
    'hail_new_duration': (6, 8),
    'hail_new_mask': (6, 8),
    'hail_temperature': (4, 6, 8),
    'hail_zero': (4, 6, 8),
    'hailcast_dhail1': (6, 8),
    'hailcast_dhail2': (6, 8),
    'hailcast_dhail3': (6, 8),
    'hailcast_dhail4': (6, 8),
    'hailcast_dhail5': (6, 8),
    'hailcast_diam_max': (6, 8),
    'hailcast_diam_mean': (6, 8),
    'hailcast_diam_std': (6, 8),
    'hailcast_runmax': (6, 8),
    'hailcast_wdur': (6, 8),
    'hailcast_wup_mask': (6, 8),
    'physics_validation_status': (1,),
    'refl_10cm': (4, 6, 8),
    'refl_t': (4, 6, 8),
    'rk_ru': (4, 6, 9),
    'rk_ru_m': (4, 6, 9),
    'rk_rv': (4, 7, 8),
    'rk_rv_m': (4, 7, 8),
    'rk_ww': (5, 6, 8),
    'rk_ww_m': (5, 6, 8),
}
NEST_FIELD_KINDS_MP8 = (
    'u', 'v', 'w', 't', 'ph', 'mu',
    'qv', 'qc', 'qr', 'qi', 'qs', 'qg', 'nr', 'ni',
)
# 2.9.11: + msfv_inv (paper/conus-suite-exact, the stored MF_VX_INV). Previously 9bf52777.
STATE_ARRAY_SHAPES_DIGEST = (
    '242835fbbd30e4943471ca51c572e4e5ea53fe4e2673d0f776f14c1807a3b1e8')
#: RE-PINNED 2026-09-24: one slot added by 217e84e18,
#: ``mp_thompson_micro_columns`` (ny, nx), WRF's per-column no_micro flag
#: (module_mp_thompson.F:1646, :2020).  _apply_thompson takes it from the
#: entry state and the phase cleanup reads it for the terminal vapour floor
#: (:3974); its lifetime audit row is beeb8394a's.  Nothing else in the
#: arena moved, and the slot aliases no other buffer.  Was
#: cfa4fe7ed787889825d504ebb122e0a7042cc8de177ae33367b6cfc8f3ec6d2c.
# Default-CQ selection changed in e13fa45c0 (unmatched moist suites) and
# 59f7e280f (all moist selections); d5460e615 already refreshed this receipt.
# RE-PINNED 2026-10-02 after the upstream default-CQ correction: R4 adds
# only acoustic_cqu/cqv/cqw to the existing 56 slots.  Both the integration
# baseline and this lane reproduce the new 59-slot layout.  The numerical
# kernel pins and every other allocation contract are preserved.  Was
# f00b1b1748fba27988bdedfde16ed05f3559a5e0bbd33cad474d7bef35d41949.
# Re-pinned for the scanner's 12 KiB geometry slot. Every physics slot
# and alias retains its previous shape; the scanner never writes state.
# RE-PINNED for lane/cap-hail (97e9a3ca061): the default-on hail and HAILCAST
# diagnostics (hailcast_opt = -1, automatic) add 21 slots of their own
# (woof/core/hail.py scratch_shapes), named hail_* / hailcast_*.  None aliases
# a Thompson buffer; every existing slot keeps its shape.  Was 9cd991b5.
SCRATCH_SLOT_REGISTRY_DIGEST = (
    'e1d103fdd69bed32117386bbf6b56910834f17798fa76b95fa0ffe59c566c0b2')
ORACLE_FIXTURE_COUNT = 92
#: RE-PINNED with the corrected oracle, not with an edit.  The Thompson
#: oracle lane found that five committed fixtures were the output of a
#: libmvec-linked build and regenerated the whole set devectorised
#: (fix/thompson-oracle-devectorized, 576b755b/8290189d), which this branch
#: merges rather than reproduces.  See the inverted witness below.
ORACLE_FIXTURE_AGGREGATE_SHA256 = (
    '6c7f555df2a44206b1cb12013b68c50bd57ac94f4dd0967f1bf0595b7a9eac53')

# -- R5 --------------------------------------------------------------------

PORTED_MP_PHYSICS_PIN = (1, 6, 8, 10, 18)
#: The 20 pre-existing nest-edge field codes.  ``28`` must be APPENDED to
#: ``PORTED_MP_PHYSICS``; inserting it renumbers this table and silently
#: re-points the ratified mp8 -> mp18 nest edge at different fields.
EDGE_FIELD_CODES_PIN = {
    "qv": 0, "qc": 1, "qr": 2, "qi": 3, "qs": 4, "qg": 5,
    "nr": 6, "ni": 7, "ns": 8, "ng": 9,
    "qh": 10, "qndrop": 11, "qnr": 12, "qni": 13, "qns": 14,
    "qng": 15, "qnh": 16, "qnn": 17, "qvolg": 18, "qvolh": 19,
}

# -- R6.  The recorded _apply_thompson call graph.  Arguments are
# identity labels, not values: 'state.qc' or 'scratch[mp_dz8w]'.
# ------------------------------------------------------------------
ADAPTER_CALLS_NO_REFL = (
    # RE-PINNED 2026-09-30 by the fused classic adapter launches
    # (launch_adapter_prepare / _entry / _masks / _finish): after CuPy's
    # Exner power, the thermodynamics, entry markers, GRAUPELNCV reset,
    # save_pre_mp_theta and WRF's entry rewrite are one launch; WRF's per-column no_micro flag (217e84e18,
    # :1646, :2020) and the private graupel number, still taken on the
    # rewritten entry state before any source kernel, are the next; the two
    # post-source column masks are one; theta, moist_physics_finish and SR
    # are one.  Every output bit is unchanged (tests/test_thompson_speed_
    # glue.py and the lane's byte-identical 1 h forecasts).
    ('launch_adapter_prepare', (
        'state.thb',
        'state.thp',
        'state.phb',
        'state.php',
        'scratch[mp_th]',
        'scratch[mp_pii]',
        'scratch[mp_thompson_temperature]',
        'scratch[mp_dz8w]',
        'state.h_diabatic',
        'state.qc',
        'state.qi',
        'state.ni',
        'state.qr',
        'state.nr',
        'state.qs',
        'state.qg',
        'scratch[mp_thompson_frozen_reference_temperature]',
        'scratch[mp_thompson_graupel_melt_marker]',
        'scratch[mp_graupelncv]',
        'scratch[mp_thompson_micro_columns]',
     ), {}),
    ('launch_adapter_entry', (
        'state.qc',
        'state.qi',
        'state.qr',
        'state.qs',
        'state.qg',
        'scratch[mp_thompson_temperature]',
        'state.p',
        'state.qv',
        'scratch[mp_thompson_graupel_number_shadow]',
        'scratch[mp_thompson_micro_columns]',
     ), {}),
    ('launch_frozen_vapor_network_from_owner', (
        'state.qi',
        'state.ni',
        'state.qs',
        'state.qg',
        'state.qr',
        'state.nr',
        'scratch[mp_thompson_temperature]',
        'state.p',
        'state.qv',
        '<classic-table-owner>',
        '10.0',
     ), {
        'graupel_number_shadow':
            'scratch[mp_thompson_graupel_number_shadow]',
        'qc':
            'state.qc',
        'snow_velocity_boost':
            'scratch[mp_thompson_snow_velocity_boost]',
    }),
    ('launch_warm_frozen_source_network_from_owner', (
        'state.qc',
        'state.qr',
        'state.nr',
        'state.qs',
        'state.qg',
        'scratch[mp_thompson_graupel_number_shadow]',
        'scratch[mp_thompson_graupel_melt_marker]',
        'scratch[mp_thompson_snow_melt_marker]',
        'scratch[mp_thompson_temperature]',
        'state.p',
        'state.qv',
        '<classic-table-owner>',
        '10.0',
     ), {}),
    ('launch_adapter_masks', (
        'state.qr',
        'state.qg',
        'scratch[mp_thompson_frozen_reference_temperature]',
        'scratch[mp_rainncv]',
        'scratch[mp_sr]',
     ), {}),
    ('launch_cloud_saturation_adjust', (
        'scratch[mp_thompson_temperature]',
        'state.p',
        'state.qv',
        'state.qc',
     ), {
        'reference_density':
            'scratch[mp_thompson_frozen_reference_density]',
        'reference_temperature':
            'scratch[mp_thompson_frozen_reference_temperature]',
        # 4ae7913df: the positive-condensation decision is held in the
        # full-theta scratch (saved already) and read by rain evaporation.
        'condensation_marker':
            'scratch[mp_th]',
        # 6bd61312c: WRF's L_qc as the adjustment leaves it (:3485), held
        # in the rain reference density until rain evaporation rewrites it.
        'cloud_presence':
            'scratch[mp_thompson_rain_reference_density]',
    }),
    # 6bd61312c: the cloud fallout's ANY(L_qc) column gate (:3645) is taken
    # from the adjustment's L_qc, not from the post-source cloud.
    ('launch_hydrometeor_column_mask', (
        'scratch[mp_thompson_rain_reference_density]',
        'scratch[mp_snowncv]',
     ), {}),
    ('launch_rain_evaporation', (
        'state.qr',
        'state.nr',
        'scratch[mp_thompson_temperature]',
        'state.p',
        'state.qv',
        '10.0',
     ), {
        'graupel_melt_marker':
            'scratch[mp_thompson_graupel_melt_marker]',
        'reference_density':
            'scratch[mp_thompson_rain_reference_density]',
        # 4ae7913df: the rain concentrations formed before cloud
        # adjustment persist until evaporation actually refreshes them
        # (source_density), and a positive condensation decision
        # suppresses same-call rain evaporation (condensation_marker).
        'condensation_marker':
            'scratch[mp_th]',
        'source_density':
            'scratch[mp_thompson_frozen_reference_density]',
        # 7727fda3c: the evaporation writes WRF's L_qr (:3236) and the
        # :3568 rewrite into the rain reference density.
        'density_carries_rain_presence':
            'True',
    }),
    ('launch_cloud_sedimentation', (
        'state.qc',
        'scratch[mp_thompson_temperature]',
        'state.p',
        'state.qv',
        'state.w[view:(3, 1, 1)]',
        'scratch[mp_dz8w]',
        '10.0',
     ), {
        'cloud_active_columns':
            'scratch[mp_snowncv]',
        'rain_active_columns':
            'scratch[mp_rainncv]',
        'reference_density':
            'scratch[mp_thompson_frozen_reference_density]',
    }),
    ('launch_ice_sedimentation', (
        'state.qi',
        'state.ni',
        'scratch[mp_thompson_temperature]',
        'state.p',
        'state.qv',
        'scratch[mp_dz8w]',
        'scratch[mp_rainnc]',
        'scratch[mp_rainncv]',
        'scratch[mp_snownc]',
        'scratch[mp_snowncv]',
        '10.0',
     ), {
        'reference_density':
            'scratch[mp_thompson_frozen_reference_density]',
    }),
    ('launch_snow_sedimentation', (
        'state.qs',
        'scratch[mp_thompson_temperature]',
        'state.p',
        'state.qv',
        'scratch[mp_dz8w]',
        'scratch[mp_rainnc]',
        'scratch[mp_rainncv]',
        'scratch[mp_snownc]',
        'scratch[mp_snowncv]',
        '10.0',
     ), {
        'accumulate_surface':
            'True',
        'melt_rain_nr':
            'state.nr',
        'melt_rain_qr':
            'state.qr',
        # 4d6e551ef and 7727fda3c: melting snow blends with the rain pass's
        # own fall speed (:3612-3634, :3722-3724), read from its density.
        'melt_rain_density':
            'scratch[mp_thompson_rain_reference_density]',
        'melt_rain_density_carries_presence':
            'True',
        'reference_density':
            'scratch[mp_thompson_frozen_reference_density]',
        'reference_temperature':
            'scratch[mp_thompson_frozen_reference_temperature]',
        'snow_melt_marker':
            'scratch[mp_thompson_snow_melt_marker]',
        'velocity_boost':
            'scratch[mp_thompson_snow_velocity_boost]',
    }),
    ('launch_graupel_sedimentation', (
        'state.qg',
        'scratch[mp_thompson_temperature]',
        'state.p',
        'state.qv',
        'scratch[mp_dz8w]',
        'scratch[mp_rainnc]',
        'scratch[mp_rainncv]',
        'scratch[mp_graupelnc]',
        'scratch[mp_graupelncv]',
        '10.0',
     ), {
        'accumulate_surface':
            'True',
        'active_columns':
            'scratch[mp_sr]',
        'graupel_number_shadow':
            'scratch[mp_thompson_graupel_number_shadow]',
        'reference_density':
            'scratch[mp_thompson_frozen_reference_density]',
    }),
    ('launch_rain_sedimentation', (
        'state.qr',
        'state.nr',
        'scratch[mp_thompson_temperature]',
        'state.p',
        'state.qv',
        'scratch[mp_dz8w]',
        'scratch[mp_rainnc]',
        'scratch[mp_rainncv]',
        '10.0',
     ), {
        'accumulate_surface':
            'True',
        'reference_density':
            'scratch[mp_thompson_rain_reference_density]',
        # 7727fda3c: the rain fallout reads L_qr from that density.
        'density_carries_rain_presence':
            'True',
    }),
    ('launch_final_phase_cleanup', (
        'state.qc',
        'state.qi',
        'state.ni',
        'scratch[mp_thompson_temperature]',
        'state.p',
        'state.qv',
     ), {
        # 217e84e18: the terminal vapour floor (:3974) skips the columns
        # WRF leaves at its no-microphysics exit (:2020).
        'micro_columns':
            'scratch[mp_thompson_micro_columns]',
    }),
    ('launch_classic_graupel_number_finalize', (
        'state.qg',
        'scratch[mp_thompson_temperature]',
        'state.p',
        'state.qv',
        'scratch[mp_thompson_graupel_number_shadow]',
     ), {}),
    ('launch_effective_radius', (
        'scratch[mp_thompson_temperature]',
        'state.p',
        'state.qv',
        'state.qc',
        'state.qi',
        'state.ni',
        'state.qs',
        'state.effc',
        'state.effi',
        'state.effs',
     ), {}),
    ('launch_adapter_finish', (
        'scratch[mp_thompson_temperature]',
        'scratch[mp_pii]',
        'scratch[mp_th]',
        'state.thp',
        'state.h_diabatic',
        'scratch[mp_rainncv]',
        'scratch[mp_snowncv]',
        'scratch[mp_graupelncv]',
        'scratch[mp_sr]',
        '<SimpleNamespace>',
        '10.0',
     ), {}),
)

ADAPTER_CALLS_WITH_REFL = (
    # RE-PINNED 2026-09-30 by the fused classic adapter launches
    # (launch_adapter_prepare / _entry / _masks / _finish): after CuPy's
    # Exner power, the thermodynamics, entry markers, GRAUPELNCV reset,
    # save_pre_mp_theta and WRF's entry rewrite are one launch; WRF's per-column no_micro flag (217e84e18,
    # :1646, :2020) and the private graupel number, still taken on the
    # rewritten entry state before any source kernel, are the next; the two
    # post-source column masks are one; theta, moist_physics_finish and SR
    # are one.  Every output bit is unchanged (tests/test_thompson_speed_
    # glue.py and the lane's byte-identical 1 h forecasts).
    ('launch_adapter_prepare', (
        'state.thb',
        'state.thp',
        'state.phb',
        'state.php',
        'scratch[mp_th]',
        'scratch[mp_pii]',
        'scratch[mp_thompson_temperature]',
        'scratch[mp_dz8w]',
        'state.h_diabatic',
        'state.qc',
        'state.qi',
        'state.ni',
        'state.qr',
        'state.nr',
        'state.qs',
        'state.qg',
        'scratch[mp_thompson_frozen_reference_temperature]',
        'scratch[mp_thompson_graupel_melt_marker]',
        'scratch[mp_graupelncv]',
        'scratch[mp_thompson_micro_columns]',
     ), {}),
    ('launch_adapter_entry', (
        'state.qc',
        'state.qi',
        'state.qr',
        'state.qs',
        'state.qg',
        'scratch[mp_thompson_temperature]',
        'state.p',
        'state.qv',
        'scratch[mp_thompson_graupel_number_shadow]',
        'scratch[mp_thompson_micro_columns]',
     ), {}),
    ('launch_frozen_vapor_network_from_owner', (
        'state.qi',
        'state.ni',
        'state.qs',
        'state.qg',
        'state.qr',
        'state.nr',
        'scratch[mp_thompson_temperature]',
        'state.p',
        'state.qv',
        '<classic-table-owner>',
        '10.0',
     ), {
        'graupel_number_shadow':
            'scratch[mp_thompson_graupel_number_shadow]',
        'qc':
            'state.qc',
        'snow_velocity_boost':
            'scratch[mp_thompson_snow_velocity_boost]',
    }),
    ('launch_warm_frozen_source_network_from_owner', (
        'state.qc',
        'state.qr',
        'state.nr',
        'state.qs',
        'state.qg',
        'scratch[mp_thompson_graupel_number_shadow]',
        'scratch[mp_thompson_graupel_melt_marker]',
        'scratch[mp_thompson_snow_melt_marker]',
        'scratch[mp_thompson_temperature]',
        'state.p',
        'state.qv',
        '<classic-table-owner>',
        '10.0',
     ), {}),
    ('launch_adapter_masks', (
        'state.qr',
        'state.qg',
        'scratch[mp_thompson_frozen_reference_temperature]',
        'scratch[mp_rainncv]',
        'scratch[mp_sr]',
     ), {}),
    ('launch_cloud_saturation_adjust', (
        'scratch[mp_thompson_temperature]',
        'state.p',
        'state.qv',
        'state.qc',
     ), {
        'reference_density':
            'scratch[mp_thompson_frozen_reference_density]',
        'reference_temperature':
            'scratch[mp_thompson_frozen_reference_temperature]',
        # 4ae7913df: the positive-condensation decision is held in the
        # full-theta scratch (saved already) and read by rain evaporation.
        'condensation_marker':
            'scratch[mp_th]',
        # 6bd61312c: WRF's L_qc as the adjustment leaves it (:3485), held
        # in the rain reference density until rain evaporation rewrites it.
        'cloud_presence':
            'scratch[mp_thompson_rain_reference_density]',
    }),
    # 6bd61312c: the cloud fallout's ANY(L_qc) column gate (:3645) is taken
    # from the adjustment's L_qc, not from the post-source cloud.
    ('launch_hydrometeor_column_mask', (
        'scratch[mp_thompson_rain_reference_density]',
        'scratch[mp_snowncv]',
     ), {}),
    ('launch_rain_evaporation', (
        'state.qr',
        'state.nr',
        'scratch[mp_thompson_temperature]',
        'state.p',
        'state.qv',
        '10.0',
     ), {
        'graupel_melt_marker':
            'scratch[mp_thompson_graupel_melt_marker]',
        'reference_density':
            'scratch[mp_thompson_rain_reference_density]',
        # 4ae7913df: the rain concentrations formed before cloud
        # adjustment persist until evaporation actually refreshes them
        # (source_density), and a positive condensation decision
        # suppresses same-call rain evaporation (condensation_marker).
        'condensation_marker':
            'scratch[mp_th]',
        'source_density':
            'scratch[mp_thompson_frozen_reference_density]',
        # 7727fda3c: the evaporation writes WRF's L_qr (:3236) and the
        # :3568 rewrite into the rain reference density.
        'density_carries_rain_presence':
            'True',
    }),
    ('launch_cloud_sedimentation', (
        'state.qc',
        'scratch[mp_thompson_temperature]',
        'state.p',
        'state.qv',
        'state.w[view:(3, 1, 1)]',
        'scratch[mp_dz8w]',
        '10.0',
     ), {
        'cloud_active_columns':
            'scratch[mp_snowncv]',
        'rain_active_columns':
            'scratch[mp_rainncv]',
        'reference_density':
            'scratch[mp_thompson_frozen_reference_density]',
    }),
    ('launch_ice_sedimentation', (
        'state.qi',
        'state.ni',
        'scratch[mp_thompson_temperature]',
        'state.p',
        'state.qv',
        'scratch[mp_dz8w]',
        'scratch[mp_rainnc]',
        'scratch[mp_rainncv]',
        'scratch[mp_snownc]',
        'scratch[mp_snowncv]',
        '10.0',
     ), {
        'reference_density':
            'scratch[mp_thompson_frozen_reference_density]',
    }),
    ('launch_snow_sedimentation', (
        'state.qs',
        'scratch[mp_thompson_temperature]',
        'state.p',
        'state.qv',
        'scratch[mp_dz8w]',
        'scratch[mp_rainnc]',
        'scratch[mp_rainncv]',
        'scratch[mp_snownc]',
        'scratch[mp_snowncv]',
        '10.0',
     ), {
        'accumulate_surface':
            'True',
        'melt_rain_nr':
            'state.nr',
        'melt_rain_qr':
            'state.qr',
        # 4d6e551ef and 7727fda3c: melting snow blends with the rain pass's
        # own fall speed (:3612-3634, :3722-3724), read from its density.
        'melt_rain_density':
            'scratch[mp_thompson_rain_reference_density]',
        'melt_rain_density_carries_presence':
            'True',
        'reference_density':
            'scratch[mp_thompson_frozen_reference_density]',
        'reference_temperature':
            'scratch[mp_thompson_frozen_reference_temperature]',
        'snow_melt_marker':
            'scratch[mp_thompson_snow_melt_marker]',
        'velocity_boost':
            'scratch[mp_thompson_snow_velocity_boost]',
    }),
    ('launch_graupel_sedimentation', (
        'state.qg',
        'scratch[mp_thompson_temperature]',
        'state.p',
        'state.qv',
        'scratch[mp_dz8w]',
        'scratch[mp_rainnc]',
        'scratch[mp_rainncv]',
        'scratch[mp_graupelnc]',
        'scratch[mp_graupelncv]',
        '10.0',
     ), {
        'accumulate_surface':
            'True',
        'active_columns':
            'scratch[mp_sr]',
        'graupel_number_shadow':
            'scratch[mp_thompson_graupel_number_shadow]',
        'reference_density':
            'scratch[mp_thompson_frozen_reference_density]',
    }),
    ('launch_rain_sedimentation', (
        'state.qr',
        'state.nr',
        'scratch[mp_thompson_temperature]',
        'state.p',
        'state.qv',
        'scratch[mp_dz8w]',
        'scratch[mp_rainnc]',
        'scratch[mp_rainncv]',
        '10.0',
     ), {
        'accumulate_surface':
            'True',
        'reference_density':
            'scratch[mp_thompson_rain_reference_density]',
        # 7727fda3c: the rain fallout reads L_qr from that density.
        'density_carries_rain_presence':
            'True',
    }),
    ('launch_final_phase_cleanup', (
        'state.qc',
        'state.qi',
        'state.ni',
        'scratch[mp_thompson_temperature]',
        'state.p',
        'state.qv',
     ), {
        # 217e84e18: the terminal vapour floor (:3974) skips the columns
        # WRF leaves at its no-microphysics exit (:2020).
        'micro_columns':
            'scratch[mp_thompson_micro_columns]',
    }),
    ('launch_classic_graupel_number_finalize', (
        'state.qg',
        'scratch[mp_thompson_temperature]',
        'state.p',
        'state.qv',
        'scratch[mp_thompson_graupel_number_shadow]',
     ), {}),
    ('reflectivity', (
        '<_HostAdapterState>',
        '<SimpleNamespace>',
        'scratch[mp_thompson_temperature]',
        'state.p',
     ), {
        'thompson_graupel_number':
            'scratch[mp_thompson_graupel_number_shadow]',
    }),
    ('launch_effective_radius', (
        'scratch[mp_thompson_temperature]',
        'state.p',
        'state.qv',
        'state.qc',
        'state.qi',
        'state.ni',
        'state.qs',
        'state.effc',
        'state.effi',
        'state.effs',
     ), {}),
    ('launch_adapter_finish', (
        'scratch[mp_thompson_temperature]',
        'scratch[mp_pii]',
        'scratch[mp_th]',
        'state.thp',
        'state.h_diabatic',
        'scratch[mp_rainncv]',
        'scratch[mp_snowncv]',
        'scratch[mp_graupelncv]',
        'scratch[mp_sr]',
        '<SimpleNamespace>',
        '10.0',
     ), {}),
)

ACOUSTIC_N_MASS = {
    "mp1": 3, "mp6": 6, "mp8": 6, "mp10": 6, "mp18": 7,
}

# -- F2 --------------------------------------------------------------------

#: The retired four-file clean-rebuild exception set (warm/ice/mixed column
#: and mixed-surface).  Empty: a clean rebuild reproduces all four.
ORACLE_REBUILD_EXCEPTION_FILES: frozenset = frozenset()
#: The one fixture a rebuild on a libc other than the receipt's may differ in.
ORACLE_LIBC_SENSITIVE_FILES = frozenset({"cold-cloud-overlap-column.csv"})


# ==========================================================================
# R1 -- source identity
# ==========================================================================

@pytest.fixture(scope="module")
def r1():
    import importlib
    try:
        # This fixture records compiler inputs with CUDA operations
        # intercepted below. Keep the metadata dependency access from
        # classifying the file's raw-source CPU checks as GPU work.
        cp = importlib.import_module("cupy")
    except ImportError:
        # Preserve the receipt's existing no-CuPy fallback for raw-source
        # checks; its compiler-capture assertions retain their own marker.
        return freeze.receipt_r1_sources()
    compiler = importlib.import_module("cupy.cuda.compiler")
    from woof import nvrtc_ptx_cache
    from woof.core import kernels

    class SourceDevice:
        id = 0

        def __init__(self, *args):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class SourceImage:
        # A sentinel, never PTX/cubin bytes or a reported compiled image.
        def startswith(self, *args):
            return False

    class SourceModule:
        def load(self, image):
            assert isinstance(image, SourceImage)

    direct_inputs = []

    def capture_nvrtc(source, options=(), *args, **kwargs):
        # The receipt replaces RawModule per source. Feed that recorder
        # the exact inputs passed at the new direct compiler boundary.
        cp.RawModule(code=source, options=options)
        direct_inputs.append((freeze._sha256_bytes(source.encode("utf-8")),
                              tuple(options)))
        return SourceImage(), {}

    loaders = (kernels.load_module, kernels.load_module_int_defines,
               kernels._load_diffusion_module, kernels.compile_diffusion_source)
    for loader in loaders:
        loader.cache_clear()
    try:
        with pytest.MonkeyPatch.context() as patch:
            # Only CUDA ownership/loading and compiler calls are replaced;
            # source assembly, options and production dispatch stay live.
            patch.setattr(cp.cuda, "Device", SourceDevice)
            patch.setattr(cp.cuda.function, "Module", SourceModule)
            patch.setattr(compiler, "compile_using_nvrtc", capture_nvrtc)
            patch.setattr(nvrtc_ptx_cache, "compile_using_nvrtc", capture_nvrtc)
            receipt = freeze.receipt_r1_sources()
        receipt["source_capture_direct_inputs"] = direct_inputs
        return receipt
    finally:
        # A later numeric GPU check must construct its real modules.
        for loader in loaders:
            loader.cache_clear()


def test_thompson_cu_is_byte_frozen(r1):
    """The single most important assertion in the port."""
    module = r1["modules"]["thompson"]
    assert module["file_sha256"] == THOMPSON_CU_SHA256, (
        "woof/core/kernels/thompson.cu was edited by a commit that has not "
        "re-frozen this file, so an mp=8 result change is shipping without "
        "its reading.  If the commit shipped one (a fixture, a test and a "
        "page, as 4ae7913df did), re-freeze THOMPSON_CU_SHA256, the "
        "compile-string digest and the FROZEN_MODULE_DIGESTS entry here "
        "citing it; if it shipped none, record one first.  mp=28 kernels "
        "still belong in their own .cu files.")


# NEEDS CUPY INSTALLED, and opens no device: the compile string this
# assertion reads is captured by driving the real loader with a recording
# RawModule; without cupy the receipt falls back to `reconstructed`
# (preamble plus file), which is not the string nvrtc compiles (the loader
# assembles rrtmgp_rte with a header the reconstruction lacks), so the
# digest cannot be read here.  Green on the release node's card
# (proof/node-reds-276).
@requires_cupy
def test_thompson_compiled_source_string_is_frozen(r1):
    """Identical source string => identical PTX => identical FP results.

    Stronger than the file hash: this is what cupy actually compiles, so a
    change to ``_preamble()``, ``CUDA_DEFINES``, ``common.cuh`` or the
    loader itself fails here too.
    """
    module = r1["modules"]["thompson"]
    assert module["capture_method"] == "loader-capture", (
        "the compile string was reconstructed instead of captured from the "
        "real loader; the inertness claim is then unproven")
    moved = ("the string nvrtc compiles for thompson.cu moved and this file "
             "was not re-frozen: an mp=8 result change is shipping without "
             "its reading.  Re-freeze THOMPSON_COMPILED_SOURCE_SHA256 and "
             "_LEN here citing the commit's reading, or record one first.")
    assert module["compiled_source_len"] == THOMPSON_COMPILED_SOURCE_LEN, moved
    assert (module["compiled_source_sha256"]
            == THOMPSON_COMPILED_SOURCE_SHA256), moved


def test_preamble_and_common_header_are_frozen(r1):
    assert r1["preamble_sha256"] == PREAMBLE_SHA256
    assert r1["preamble_len"] == PREAMBLE_LEN
    assert r1["common_cuh_sha256"] == COMMON_CUH_SHA256
    assert r1["cuda_defines"] == CUDA_DEFINES_PIN


@requires_cupy
def test_every_frozen_kernel_module_is_unchanged(r1):
    """All 65 pre-existing translation units, file AND compile string.

    New mp=28 ``.cu`` files are allowed and ignored; a MISSING pinned name
    is a failure.
    """
    modules = r1["modules"]
    missing = sorted(set(FROZEN_MODULE_DIGESTS) - set(modules))
    assert not missing, f"frozen kernel modules disappeared: {missing}"
    drift = {}
    for name, (file_sha, compiled_sha) in sorted(
            FROZEN_MODULE_DIGESTS.items()):
        got = modules[name]
        if (got["file_sha256"], got["compiled_source_sha256"]) != (
                file_sha, compiled_sha):
            drift[name] = {
                "expected": (file_sha, compiled_sha),
                "actual": (got["file_sha256"],
                           got["compiled_source_sha256"]),
            }
    assert not drift, (
        f"kernel source drift: {drift}.  A frozen unit was edited by a "
        "commit that has not re-frozen it here, so its result change is "
        "shipping without its reading.  Re-freeze the entry with a comment "
        "naming the commit and the reading it shipped, as the annotated "
        "entries in FROZEN_MODULE_DIGESTS do, or record the reading first.")


def test_loader_hook_is_inert_for_every_frozen_module(r1):
    """``load_module`` still assembles ``_preamble() + <name>.cu`` exactly.

    WP-02 adds an ``_EXTRA_HEADERS`` allow-list to the shared loader.  This
    is the assertion that holds it to "every module not named in the dict
    assembles a byte-identical source string".
    """
    import importlib.util as _ilu

    from woof.core import kernels as _kernels

    _spec = _ilu.spec_from_file_location(
        "kernel_loader_inert_probe",
        Path(__file__).with_name("test_kernel_loader_inert.py"))
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    # Selected Group A diffusion and surface sources extend the historical
    # grant. Their raw headers and assembled source pairs are pinned above.
    _EXPECTED_HEADERS = {
        **_mod._EXPECTED_HEADERS,
        "acoustic": ("glibc_trig_flt32.cuh",),
        "myjsfc": ("glibc_flt32.cuh", "flt32_expf_fma.cuh"),
        "mynn_surface": ("mynn_libm.cuh", "surface_subnormal.cuh"),
        "sfclay": ("glibc_flt32.cuh", "sfclay_classic.cuh"),
        "smag2d": ("glibc_flt32.cuh",),
        "satmedmf": ("glibc_flt32.cuh", "glibc_flt64.cuh",
                     "gfs_physcons.cuh", "satmedmf_common.cuh",
                     "satmedmf_column.cuh"),
        # WRF's ocean mixed layer (2.9.1): the glibc expf/logf the column
        # module calls, pinned with oml.cu.
        "oml": ("glibc_flt32.cuh",),
    }

    # New port headers have measured composed-source pins.
    _EXPECTED_HEADERS.update({'fer_hires': ('glibc_flt32.cuh',)})
    # lane/cap-hail (97e9a3ca061): WRF's HAILCAST support and column headers,
    # prepended to the hail unit only, pinned in kernel_source_pins.json.
    # fix/hailcast: the operational HRRR v4.1.21 HAILCAST column header
    # (hailcast_version = "wrf_39_noaa"), pinned in kernel_source_pins.json.
    _EXPECTED_HEADERS.update({'hail': ('hailcast_support.cuh', 'hailcast_wrf.cuh', 'hailcast_wrf39.cuh')})
    # GSD output diagnostics (f80ac46aeb, 328ce7233b): raw sources and the
    # helper-only helicity header are pinned in kernel_source_pins.json.
    # uh_diag.cu remains its own frozen translation unit, never a header.
    _EXPECTED_HEADERS.update({
        'gsd_diagnostics': ('glibc_flt32.cuh', 'glibc_flt32_extra.cuh',
                            'glibc_flt64.cuh', 'gsd_hail_bins.cuh'),
        'gsd_helicity': ('gsd_helicity_helpers.cuh',),
    })

    # The header grant stays a CLOSED literal: the loader's own allow-list
    # must equal the historical mapping plus the selected literal grant,
    # so a module cannot gain a header without its source gate moving too.
    granted = {name: tuple(headers) for name, headers
               in _kernels._EXTRA_HEADERS.items()}
    assert granted == {name: tuple(headers) for name, headers
                       in _EXPECTED_HEADERS.items()}, (
        "the kernel loader's _EXTRA_HEADERS drifted from the closed "
        "historical mapping plus the selected Group A header pins")
    not_inert = sorted(
        name for name in FROZEN_MODULE_DIGESTS
        if name not in granted
        and not r1["modules"][name]["loader_matches_preamble_plus_file"])
    assert not not_inert, (
        "the kernel loader no longer assembles _preamble() + <name>.cu for "
        f"these pre-existing modules: {not_inert}")


def test_thompson_py_has_no_new_launcher(r1):
    assert r1["thompson_py_sha256"] == THOMPSON_PY_SHA256, (
        "woof/core/thompson.py was edited by a commit that has not "
        "re-frozen this file; if it shipped a reading of what it did to "
        "mp=8, re-freeze THOMPSON_PY_SHA256 here citing it, else record one "
        "first.  The two assertions below say whether the launch inventory "
        "itself moved.")
    assert r1["thompson_py_all"] == THOMPSON_PY_ALL
    assert (r1["thompson_py_launch_symbols"]
            == tuple(sorted(THOMPSON_PY_ALL))), (
        "woof/core/thompson.py grew or lost a launch_* symbol; mp=28 "
        "launchers belong in woof/core/thompson_aerosol_*.py")


def test_constant_droplet_number_inventory_is_unchanged(r1):
    """The 13 + 6 + 2 + 3 literal sites mp=28 must replace, not edit.

    If this moves, someone has been editing the frozen kernel in place --
    which the source digests would also catch, but this failure names the
    physics.
    """
    assert r1["thompson_cu_literal_sites"] == THOMPSON_CU_LITERAL_SITES, (
        "the constant-droplet-number sites in thompson.cu moved.  Counts "
        "that changed mean a site was edited in place; counts that held "
        "with shifted lines mean the file gained or lost lines above them, "
        "and this pin moves with the digests: re-freeze it here beside "
        "them, citing the commit's reading.")


# ==========================================================================
# R2 -- classic table contract
# ==========================================================================

@pytest.fixture(scope="module")
def r2():
    return freeze.receipt_r2_tables()


def test_classic_table_assets_are_unchanged(r2):
    got = tuple((a["filename"], a["bytes"], a["sha256"])
                for a in r2["classic_table_assets"])
    assert got == CLASSIC_TABLE_ASSETS_PIN
    assert r2["table_set_id"] == TABLE_SET_ID_PIN
    assert r2["mp_physics"] == 8
    assert tuple(r2["number_species"]) == ("ni", "nr")
    assert tuple(r2["mass_species"]) == ("qv", "qc", "qr", "qi", "qs", "qg")
    assert r2["wrf_reference_version"] == WRF_REFERENCE_VERSION_PIN
    assert r2["wrf_reference_commit"] == WRF_REFERENCE_COMMIT_PIN


def test_aerosol_blob_never_enters_the_classic_contract(r2):
    """CCN_ACTIVATE.BIN is third-party parcel-model output only mp=28 reads.

    It ships with woof as of 2026-08-01 and is listed in
    ``tables/MANIFEST.sha256``, which is exactly why this assertion still
    matters: that manifest is a ``sha256sum -c`` file nothing reads at run
    time, while ``CLASSIC_TABLE_ASSETS`` IS the four-asset inventory every
    mp=8 launch validates and every mp=8 restart identity binds.  Adding the
    blob there would change what a validated mp=8 table set means and make
    an mp=8 launch fail closed on a file it never reads.  WP-01 gives it its
    OWN contract and its own set id instead.
    """
    assert r2["aerosol_blob_in_classic_assets"] is False


# ==========================================================================
# R3 -- transported species
# ==========================================================================

def test_morrison_droplet_number_exclusion_survives():
    """``nc`` must not start being advected as a side effect of mp=28.

    mp_physics=10 already allocates ``state.nc`` and deliberately does not
    transport it.  A presence-based ``nc`` in
    ``TRANSPORTED_NUMBER_SPECIES`` would silently start advecting Morrison's
    diagnostic droplet number through all 8 generic dycore call sites.  The
    mp=10 probe below CARRIES ``nc``, so its absence from the result is
    proved rather than merely unexercised.
    """
    r3 = freeze.receipt_r3_species()
    assert tuple(r3["extra_moist_species_mp8"]) == EXTRA_MOIST_SPECIES_MP8
    assert tuple(r3["extra_moist_species_mp10"]) == EXTRA_MOIST_SPECIES_MP10
    assert "nc" not in r3["extra_moist_species_mp10"]
    assert "nwfa" not in r3["extra_moist_species_mp10"]
    assert "nifa" not in r3["extra_moist_species_mp10"]
    assert (tuple(r3["transported_number_species"])
            == TRANSPORTED_NUMBER_SPECIES_PIN)


# ==========================================================================
# R4 -- preflight allocation surface
# ==========================================================================

@pytest.fixture(scope="module")
def r4():
    return freeze.receipt_r4_preflight()


def test_mp8_state_allocation_list_is_unchanged(r4):
    """Every array DomainState allocates for mp=8, with its exact shape."""
    assert r4["probe_config"]["mp_physics"] == 8
    got = r4["state_array_shapes"]
    assert set(got) == set(STATE_ARRAY_SHAPES_MP8), {
        "added": sorted(set(got) - set(STATE_ARRAY_SHAPES_MP8)),
        "removed": sorted(set(STATE_ARRAY_SHAPES_MP8) - set(got)),
    }
    assert got == STATE_ARRAY_SHAPES_MP8
    assert r4["state_array_shapes_digest"] == STATE_ARRAY_SHAPES_DIGEST


def test_mp8_scratch_arena_layout_is_unchanged(r4):
    """The scratch registry IS the aliasing contract.

    ``_apply_thompson`` deliberately lifetime-aliases buffers (the graupel
    entry marker with the held-temperature buffer, RAINNCV/SNOWNCV/SR with
    column masks).  A new or resized slot moves the arena and can turn one
    of those aliases into a live conflict without any test failing on
    values.
    """
    got = r4["scratch_slot_registry"]
    assert set(got) == set(SCRATCH_SLOT_REGISTRY_MP8), {
        "added": sorted(set(got) - set(SCRATCH_SLOT_REGISTRY_MP8)),
        "removed": sorted(set(SCRATCH_SLOT_REGISTRY_MP8) - set(got)),
    }
    assert got == SCRATCH_SLOT_REGISTRY_MP8
    assert r4["scratch_slot_registry_digest"] == SCRATCH_SLOT_REGISTRY_DIGEST


def test_mp8_nest_field_kinds_are_unchanged(r4):
    assert tuple(r4["nest_field_kinds"]) == NEST_FIELD_KINDS_MP8
    assert "nc" not in r4["nest_field_kinds"]


# ==========================================================================
# R5 -- nest transition edge codes
# ==========================================================================

def test_edge_field_codes_for_the_twenty_pre_existing_names():
    """28 must be APPENDED to PORTED_MP_PHYSICS, never inserted.

    ``_EDGE_FIELD_CODES`` is ``enumerate`` over the de-duplicated union of
    every ported scheme's fields in PORTED_MP_PHYSICS order.  Inserting 28
    anywhere but the end renumbers the table, and the ratified mp8 -> mp18
    nest edge then selects different fields with no error anywhere.
    """
    r5 = freeze.receipt_r5_edge_codes()
    codes = r5["edge_field_codes"]
    for name, code in EDGE_FIELD_CODES_PIN.items():
        assert codes.get(name) == code, (
            f"nest edge field {name!r} moved from code {code} to "
            f"{codes.get(name)}")
    assert tuple(r5["ported_mp_physics"])[:5] == PORTED_MP_PHYSICS_PIN
    assert tuple(r5["all_edge_fields"])[:20] == tuple(EDGE_FIELD_CODES_PIN)


# ==========================================================================
# R6 -- acoustic selection and the adapter call graph
# ==========================================================================

def test_acoustic_moist_cq_selects_six_masses_for_mp8():
    """``calc_cq`` mass loading must not learn about aerosol numbers.

    WRF's ``calc_cq`` sums the ``moist`` Registry package; number moments
    live in the separate ``scalar`` package and are not mass loading.  mp=28
    adds nc/nwfa/nifa as scalars, so n_mass for mp=8 (and for mp=28) stays
    6.
    """
    n_mass = freeze.receipt_r6_call_graph()["acoustic_n_mass"]
    assert n_mass["mp8"] == 6
    assert n_mass == ACOUSTIC_N_MASS


def _as_tuple(calls):
    return tuple(
        (c["launcher"], tuple(c["args"]), dict(c["kwargs"])) for c in calls)


def test_apply_thompson_issues_the_identical_launcher_sequence():
    """Call-recording double over the real adapter.

    Every launcher, ``save_pre_mp_theta``, ``moist_physics_finish`` and the
    reflectivity entry point are replaced by spies, and the classic table
    owner by a sentinel, so no CUDA runs and no 380 MB table is read.  What
    is pinned is the ORDER of the calls and the IDENTITY of every argument
    -- which state field or which named scratch slot -- because that
    ordering and that aliasing are the mp=8 trajectory.

    The mp=28 adapter is a SEPARATE module
    (``woof/core/thompson_aerosol.py``); nothing in this sequence may
    change to accommodate it.
    """
    recorded = _as_tuple(freeze.record_adapter_calls(refl_10cm_due=False))
    assert recorded == ADAPTER_CALLS_NO_REFL, (
        "_apply_thompson's launcher sequence moved and this pin was not "
        "re-frozen: an mp=8 trajectory change is shipping without its "
        "reading.  Re-pin ADAPTER_CALLS_NO_REFL and _WITH_REFL here with a "
        "comment on the changed call naming the commit and its reading, as "
        "the 4ae7913df entries do, or record the reading first.")


def test_apply_thompson_reflectivity_call_graph_is_unchanged():
    recorded = _as_tuple(freeze.record_adapter_calls(refl_10cm_due=True))
    assert recorded == ADAPTER_CALLS_WITH_REFL, (
        "_apply_thompson's output-due launcher sequence moved and this pin "
        "was not re-frozen; re-pin it beside ADAPTER_CALLS_NO_REFL with the "
        "commit's reading, or record one first.")


def test_adapter_still_feeds_cloud_sedimentation_the_lower_w_slice():
    """WRF copies ``w(i,k,j)`` into ``w1d(k)`` with no averaging.

    mp=28's ``activ_ncloud`` needs the same field and is far more sensitive
    to it, so this is the argument the aerosol adapter must match.  Pinned
    here as its own named claim because a silent switch to a mass-level
    average would still pass the whole-sequence comparison's shape checks.
    """
    calls = {c["launcher"]: c
             for c in freeze.record_adapter_calls(refl_10cm_due=False)}
    assert calls["launch_cloud_sedimentation"]["args"][4] == (
        "state.w[view:(3, 1, 1)]")


# ==========================================================================
# F1 -- the committed mp=8 oracle fixtures
# ==========================================================================

def test_committed_mp8_oracle_fixtures_are_frozen():
    """92 CSVs, byte-for-byte.

    WP-03 adds ``woof/data/thompson/oracle-aero/``; it may not touch this
    directory.  Regenerating any file here moves a model-validated
    baseline.
    """
    f1 = freeze.receipt_f1_oracle_fixtures()
    assert f1["count"] == ORACLE_FIXTURE_COUNT
    assert f1["aggregate_sha256"] == ORACLE_FIXTURE_AGGREGATE_SHA256


# ==========================================================================
# F2 -- the documented four-file clean-rebuild exception
# ==========================================================================

def test_rebuild_exception_list_is_retired_and_the_libc_allowance_is_one_file():
    """No file may be added to an allowance to make a gate pass.

    The four-file set (warm, ice and mixed column, mixed-surface) recorded
    an older harness build; the fixtures were regenerated and a clean
    rebuild on 2026-10-05 reproduced all four byte for byte, so the set is
    empty.  What remains is the README's one measured libc sensitivity,
    allowed only when the rebuild's libc is not the receipt's.
    """
    assert set(freeze.ORACLE_REBUILD_EXCEPTIONS) == (
        ORACLE_REBUILD_EXCEPTION_FILES) == set()
    assert set(freeze.ORACLE_LIBC_SENSITIVE_FIXTURES) == (
        ORACLE_LIBC_SENSITIVE_FILES)
    assert freeze.ORACLE_LIBC_SENSITIVE_TABLES == {"freezeH2O.dat"}


def test_the_committed_fixtures_now_share_one_pressure_profile():
    """THE WITNESS, INVERTED, BECAUSE THE DEFECT IT WITNESSED WAS FIXED.

    This test used to be called
    ``test_the_three_original_fixtures_carry_a_foreign_pressure_profile``
    and it asserted the DEFECT: that the 46 committed column fixtures split
    into two ``p_pa`` profiles, 43 against {warm, mixed, ice}.  That was a
    true and carefully measured statement about a stale oracle, and the
    receipt beside it explained the split as a build-provenance difference
    -- "a different libm expf, i.e. a different machine or glibc" -- and
    recorded it rather than repairing it.

    The Thompson oracle lane found the actual mechanism and repaired it.
    It is not a different machine: from GCC 12 on, ``-O2`` implies
    ``-ftree-vectorize``, a vectorised ``exp``/``pow`` loop links glibc's
    libmvec SIMD entry points instead of the scalar routines, libmvec is
    not bit-identical to scalar libm, and whether any given loop vectorises
    is a cost-model decision that depends on how much UNRELATED source
    surrounds it.  ``run_column.F90`` was 227 lines when warm/mixed/ice
    were generated and roughly a thousand when the other 43 were, so the
    same ``-O2`` produced two different oracles.  ``build.sh`` now pins
    ``-fno-tree-vectorize``, which removes libmvec from the link entirely
    and makes the set invariant across gfortran 12/13/14/15 at -O1/-O2/-O3,
    and all five affected fixtures were regenerated
    (fix/thompson-oracle-devectorized, 576b755b and 8290189d).

    So the assertion INVERTS rather than being deleted: ONE profile, 46
    files, no minority group.  That is strictly stronger than what it
    asserted before, and a regression that reintroduces a second profile
    fails here exactly as the old form would have.

    The old docstring's remaining content is preserved because it is still
    the correction of record for the port spec's blocking unknown #2:

    HERMETIC witness for the CAUSE of the (former) exception.

    ``p_pa = p0 * exp(-z(k)/8000.0)`` (run_column.F90:219) is a pure
    function of the harness's own z grid and does not depend on the
    scenario, so all 46 committed column fixtures must print the same 24
    values.  They do not: there are exactly two profiles, and the minority
    one belongs to exactly {warm, mixed, ice} -- which
    ``woof/data/thompson/PROVENANCE.md`` records as the first three columns
    ever generated.  Those three came from a different build of the harness
    (a different libm ``expf``: they differ by one float32 ulp at 13 of 24
    levels), and every other difference in those files -- including the
    ``after`` rows and mixed-surface's rainnc -- follows from it through the
    nonlinear scheme.

    This CORRECTS the port spec's blocking unknown #2, which described the
    drift as a float32-vs-float64 vapour seed with "p_pa, pii and theta
    byte-identical".  p_pa and pii are NOT byte-identical, and the k=1
    vapour coincidence the spec generalised from does not hold at k=2..24.

    Reading only committed repository bytes, this test needs no gfortran,
    no WRF tree and no rebuild -- so the exception stays verified on every
    run instead of resting on a note.
    """
    witness = freeze.fixture_provenance_witness()
    assert witness["distinct_p_pa_profiles"] == 1, (
        "the committed column fixtures split into "
        f"{witness['distinct_p_pa_profiles']} pressure profiles again; "
        "p_pa = p0*exp(-z/8000) does not depend on the scenario, so more "
        "than one profile means the set is a stratigraphy of builds rather "
        "than one oracle run.  The minority group is "
        f"{sorted(witness['minority_group'])}")
    assert witness["group_sizes"] == [46]
    assert witness["minority_group"] == []
    assert witness["p_pa_detail"] == {}, (
        "the witness is reporting a p_pa divergence again; see "
        "tests/test_thompson_oracle_provenance.py for the same property "
        "asserted from the fixture bytes directly")
    # The port spec's blocking unknown #2 claimed the drift was a
    # float32-vs-float64 vapour seed, generalising from a k=1 coincidence.
    # With the oracle corrected, even that coincidence is gone: the
    # committed k=1 qv is the float32-throughout evaluation, not the
    # float64-then-REAL(4) one.  Both halves of the spec's claim are now
    # measured false, and both are asserted so rather than described.
    assert witness["seed_k1_note"]["committed_matches_float64_path"] is False
    assert witness["seed_k1_note"]["generalises_to_other_levels"] is False


@pytest.mark.skipif(
    not os.environ.get("WOOF_MP8_ORACLE_REBUILD_DIR"),
    reason="set WOOF_MP8_ORACLE_REBUILD_DIR to a completed "
           "tools/thompson_wrf461_oracle/build.sh output directory")
def test_clean_oracle_rebuild_reproduces_the_committed_oracle():
    """Opt-in empirical gate: the committed oracle is a re-derivable build.

    Opt-in because it needs gfortran, the pristine WRF tree and ~380 MB of
    regenerated coefficient tables.  On the receipt's own libc every table
    cache matches the receipt's [table_cache] and all 92 CSVs are
    byte-exact.  On another libc, freezeH2O.dat and one qg cell of
    cold-cloud-overlap-column.csv may move, by no more than measured
    (gfortran 13.3.0 / glibc 2.39, 2026-10-05: 91 of 92 byte-exact).

    The packaged CLASSIC_TABLE_ASSETS hashes are a different build from the
    receipt's tables and are reported in ``dat_assets``, not asserted.
    """
    build_dir = Path(os.environ["WOOF_MP8_ORACLE_REBUILD_DIR"])
    result = freeze.compare_rebuilt_oracle(build_dir)
    assert result["csv_missing"] == []
    assert result["csv_total"] == ORACLE_FIXTURE_COUNT
    assert result["rebuilt_libc"] is not None, (
        "the rebuild directory carries no build.sh PROVENANCE.txt")
    if result["same_libc"]:
        allowed_csv, allowed_tables = set(), set()
    else:
        allowed_csv = set(ORACLE_LIBC_SENSITIVE_FILES)
        allowed_tables = set(freeze.ORACLE_LIBC_SENSITIVE_TABLES)
    tables = result["receipt_tables"]
    assert len(tables) == 4, tables
    moved = {name for name, row in tables.items() if not row["matches"]}
    assert moved <= allowed_tables, tables
    assert set(result["csv_differing"]) <= allowed_csv, (
        result["csv_differing"])
    for name, diff in result["csv_differences"].items():
        pinned = freeze.ORACLE_LIBC_SENSITIVE_FIXTURES[name]
        assert set(diff["fields"]) <= set(pinned["fields"]), diff
        assert diff["max_relative_overall"] <= (
            pinned["max_relative_overall"] * 1.001), (
            f"{name} drifted beyond the recorded deviation")
