"""Every place ``mp_physics=28`` used to fall off the production runtime.

WHY THIS FILE EXISTS
--------------------
An auditor traced a real mp=28 forecast and found four places where the
scheme silently stopped being a first-class selector: the restart writer
refused it outright, three REFL_10CM admission gates and a fourth in the
prepared single-domain runner omitted it, the legacy-RRTMG radii table had
no row for it, and the offline downscale child could not read an mp=28
parent.  None of those is a numerics defect and none of them NaNs -- which
is precisely why none of the nineteen column fixtures, the 600-step
forecast smoke, or the three adversarial audits saw them.  They are
REACHABILITY defects, and the only thing that finds a reachability defect
is running the lane.

WHAT IS GATED HERE, AND WHY EACH GATE IS THE SHAPE IT IS
--------------------------------------------------------
1. THE CENSUS (``test_no_stale_pre_28_scheme_admission_tuple_survives``).
   The original pinned-gap test named three paths in ``woof/runtime.py``
   and therefore could not see the FOURTH copy of the same tuple in
   ``woof/prepared_single_domain_forecast.py``.  A test that enumerates
   known sites can only ever be as complete as the last audit.  This one
   walks the AST of every ``.py`` file under ``gpuwm/`` instead and fails
   on the stale literal wherever it appears, so a fifth copy added next
   month cannot hide.

2. THE RESTART ROUND TRIP.  ``tests/test_preflight.py::
   test_mp28_survives_a_restart_round_trip`` gates the manifest layer on
   host arrays.  This gates the FILE: a real device state, ``write_restart``
   to a real ``.npz``, ``restore_restart`` into a second state whose aerosol
   fields were zeroed first, and a BITWISE comparison of all five aerosol
   arrays.  Adding the identity string is what made an mp=28 checkpoint
   possible at all, and an identity string that lets a restart proceed while
   dropping the aerosol state would be worse than the old refusal -- so the
   two new inventory guards are gated in both directions too.

3. THE PACKAGING MEASUREMENT.  ``tests/test_package_data_coverage.py``
   already asserts the ``exclude-package-data`` DECLARATION out of
   ``tomllib`` and MEASURES the wheel contents with setuptools -- but the
   project virtualenv has no setuptools, so the measurement half skips
   exactly where it matters.  The gate here finds an interpreter that does
   have setuptools (this repo's CI image and the system python both do) and
   runs the measurement in a subprocess, so the strongest control on a file
   woof has no right to redistribute is not inert in the environment the
   suite actually runs in.

TWO THINGS THE MEASUREMENT CHANGED ABOUT THE PUBLISHED STORY
-------------------------------------------------------------
The evidence document described the REFL gap as "wrfout frames whose
REFL_10CM was never produced".  Reproducing it showed something stronger:
the tree model gates ``refl_10cm_due`` on the history handler and the clock
only (woof/core/model.py:669-675), so the mp=28 adapter DID stage the field
and the unconsumed stash then raised ``RuntimeError: REFL_10CM stash was not
consumed before reuse`` at the SECOND output frame.  See
:func:`test_the_missing_28_refl_admission_was_a_crash_not_a_missing_field`.

And the restart identity was only half the restart gap.  ``mp_physics=8``
has always bound its four table digests into the physics-setup fingerprint;
mp=28 bound none, so an mp=28 checkpoint could resume against a different
``freezeH2O.dat`` or a different ``CCN_ACTIVATE.BIN`` with every identity
check passing.  For this scheme that is a trajectory substitution: the
activation fraction that sets droplet number is READ out of ``tnccn_act``
(module_mp_thompson.F:5229-5230), not computed.  See
:func:`test_the_mp28_restart_identity_binds_every_table_it_reads`.

ONE DEFECT THIS FILE MEASURES AND DOES NOT FIX
-----------------------------------------------
``woof/core/rrtmgp.py``'s scheme map resolves mp=28 to ``"kessler"``.  The
file belongs to another owner, so it is reported rather than edited; the
measurement, the three consequences and the reason the census tolerates it
are in :data:`KNOWN_SCHEME_KEYED_DICTS_WITHOUT_28`.

TWO SKIP SITES, BOTH NAMED
---------------------------
``test_the_use_mp_re_citation_is_checkable_against_stock_wrf`` skips when the
stock WRF v4.6.1 reference tree is absent, and
``test_the_ccn_activation_blob_is_absent_from_the_built_package_data`` skips
when no interpreter on the machine has setuptools.  Neither swallows a
failure: each names the one missing thing.  If this module is adopted into
``tests/test_thompson_aerosol_gpu.py::_SUITE_MODULES`` they must be added to
that file's ``_SKIP_SITES`` census and to the evidence document's skip table
in the same change.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pytest

from conftest import requires_gpu

# Consumer expectations come from the independent generated declarations.
from woof.physics_registry import implemented_selector_values as _intake_selectors
from woof.physics_registry import consumer_rows_by_selector as _intake_rows
_INTAKE_MP = tuple(sorted(_intake_selectors("microphysics")))
_INTAKE_REFL = tuple(mp for mp in _INTAKE_MP if mp)
_INTAKE_EDGES = tuple(sorted(mp for mp, row in
    _intake_rows("microphysics", "nest_transition").items()
    if (row or {}).get("mixed_edge_ported")))
_INTAKE_PRODUCERS = tuple(sorted(mp for mp, row in
    _intake_rows("microphysics", "reflectivity_input_species").items()
    if row is not None))
_INTAKE_NATIVE = tuple(sorted(set(_INTAKE_REFL) - set(_INTAKE_PRODUCERS)))
_INTAKE_SNOW = tuple(sorted(mp for mp, row in
    _intake_rows("microphysics", "ring_guard").items()
    if "qs" in (row or {}).get("state_fields", ())))
_INTAKE_SIX = tuple(sorted(mp for mp, row in
    _intake_rows("microphysics", "ring_guard").items()
    if "qg" in (row or {}).get("state_fields", ())))

#: The three driven-door tests below type `woof run` and `woof run-plan`
#: and read the mp=28 dataset precondition off the door's own sentence.
#: On an install with no cupy those doors refuse the install first, at plan
#: acceptance, before any configuration is read, so the sentence they pin
#: is never reached there: declared skipped with the reason, never a green
#: that compared the wrong refusal.
needs_cupy_install = pytest.mark.skipif(
    importlib.util.find_spec("cupy") is None,
    reason=("cupy is not installed; woof run / run-plan refuse the install "
            "at plan acceptance before the mp=28 precondition is asked"))

REPO = pathlib.Path(__file__).resolve().parent.parent
PACKAGE = REPO / "woof"

#: The reflectivity-admission tuple as it was spelled BEFORE mp=28 existed.
#: Four production gates carried it verbatim.
STALE_REFL_ADMISSION = frozenset({1, 6, 8, 10, 18})

#: What every REFL_10CM admission gate must be now.  28 belongs on WRF's own
#: structure, not on a woof convenience: ``mp_gt_driver`` reaches
#: ``calc_refl10cm`` from ONE call site (module_mp_thompson.F:1458) gated on
#: ``diagflag .and. do_radar_ref == 1`` (:1449) and never on
#: ``is_aerosol_aware``, and the routine takes no droplet-number argument
#: (:5710-5711), so the aerosol-aware package publishes the same field on the
#: same cadence as the classic one.
#:
#: 16 (WDM6) joined 2026-08-09 with the mp=16 port, and it is RE-DERIVED on
#: the same WRF structure rather than added to make a red test green:
#: ``wdm6`` reaches ``refl10cm_wdm6`` from ONE call site
#: (module_mp_wdm6.F:291) inside ``IF (diagflag .and. do_radar_ref == 1)``
#: (:279-280), the identical gate and the identical
#: ``refl_10cm(i,k,j) = max(-35., dBZ(k))`` store (:294).  So mp=16 publishes
#: REFL_10CM on exactly the cadence the other admitted schemes do, and every
#: gate that admits 6/8/10 must admit it.  The lane already moved all four
#: PRODUCTION constants; this pin was the test-side copy and was stale, not
#: wrong-headed.  Unlike 28, WDM6's routine DOES take a number argument
#: (nr1d, :2957) -- that is a producer-side difference (the port gives it its
#: own kernel, woof/core/kernels/wdm6_refl.cu) and changes nothing about
#: admission.
#:
#: The named schemes (woof.microphysics_schemes, TEMPO = 900 with the RRFS v2
#: intake) join every admission gate through NAMED_MP_IDS: each computes its
#: own 10 cm reflectivity (``native_reflectivity``), so the carrier is due on
#: the same cadence.
from woof.microphysics_schemes import NAMED_MP_IDS  # noqa: E402
# CAM MG (11) has a cited absent reflectivity rail; NTU (56) has
# a native-distribution diagnostic. TEMPO (900) retains its rail.
REFL_ADMISSION = frozenset({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 16, 18, 24, 26, 28, 38, 40, 50, 51, 52, 53, 55, 56, 97, 106, 900})

#: The deliberate exceptions -- NONE remain.  This ledger is the census of
#: sites that are stale ON PURPOSE, so it is the last place that may itself
#: go stale; it had, and every clause below is what replaced what it said.
#:
#: It said mp=28 had no ported MIXED nest edge and that every one of its
#: mixed pairs was refused through ``UNVALIDATED_MIXED_EDGE_SELECTORS``.
#: Both edges are ratified: that tuple is now EMPTY
#: (``woof/core/microphysics_transition.py``) and 28 is a member of
#: ``PORTED_MP_PHYSICS`` beside it.  It cited
#: ``tests/test_preflight.py`` pinning ``28 not in mt.PORTED_MP_PHYSICS``;
#: that assertion is now ``28 in mt.PORTED_MP_PHYSICS``.  It derived the
#: HRRR route's admission as ``frozenset(PORTED_MP_PHYSICS) -
#: AEROSOL_LATERAL_BC_BLOCKED_MP_PHYSICS``; that subtraction constant is
#: deleted and ``SUPPORTED_MICROPHYSICS`` is ``frozenset(
#: PORTED_MP_PHYSICS)`` with nothing taken out.  And it restated the
#: retired route premise (no aerosol lateral boundary condition, revisit
#: when a QNWFA/QNIFA ingest lane exists) as though it were live: that
#: ingest lane exists (``woof/ingest/wif_climatology.py``), nwfa/nifa are
#: coupled scalars, and what survives is a DATASET precondition asked of
#: every source -- measured by
#: ``woof.config.mp28_aerosol_lateral_forcing_precondition``, raised
#: BEFORE THE FETCH by ``validate_experiment_preparation`` at the doors
#: that commit to building a forecast, kept as a floor at real
#: initialization, and reported at plan review from the registry's own
#: row.
#:
#: Nothing takes their place: the set is empty and every site the census
#: walks is expected to be current.
DELIBERATE_STALE_SITES: set[tuple[str, str]] = set()

#: Every ``mp_physics`` value ``woof/config.py`` accepts
#: (``MP_PHYSICS_ACCEPTED``).  Used to tell a microphysics scheme table
#: apart from an arbitrary integer-keyed map -- RRTMG's band and g-point
#: tables are keyed 1..16 and would otherwise look exactly like one.
#: Re-derived 2026-08-09 against the validator after mp=16 (WDM6) joined
#: it, and again 2026-08-30 for mp=9 (Milbrandt-Yau) and mp=50 (P3), which
#: had been accepted selectors since 1.9 while this copy stayed at the
#: 2026-08-09 census; the census below is a shape-detector, so a selector
#: missing here would make the detector blind to a dict keyed on it rather
#: than fail -- any dict keying 9 or 50 flunked the subset test and was
#: silently outside the census's domain.
#:
#: JUDGED at the 2026-08-30 widening -- six dicts entered the census's
#: domain, none a 28-omission hit, and every selector gap among them was
#: either fixed or is recorded here so the next reader finds a decision:
#:
#: * ``woof/verify/npref.py`` ``_CQ_MASS_SPECIES_BY_MP`` omitted 9 and
#:   16 (accepted selectors whose device cq path runs,
#:   woof/core/acoustic.py prepare_moist_cq) -- FIXED in the same lane,
#:   rows from Registry.EM_COMMON:3025/:3031, gated by
#:   tests/test_acoustic_npref.py.
#: * ``woof/io/restart.py`` ``MICROPHYSICS_ALGORITHM_IDENTITIES`` omits
#:   9: an mp=9 checkpoint write refuses by name (RestartManifestError,
#:   "cannot identify unsupported microphysics scheme 9").  Fail-loud,
#:   not silent, and composing Milbrandt-Yau's trajectory-identity
#:   string needs the mp=16/28/50 rows' level of scheme study --
#:   NAMED FOLLOW-UP for the restart lane, not a census suppression.
#: * ``woof/wrf_physics_inventory.py`` ``_INVENTORIES`` omits 9 and 16
#:   deliberately: it records EVIDENCED stock-WRF export contracts, which
#:   exist for exactly (6, 8, 10, 18, 24, 28, 50); the rw-wps consuming half
#:   documents the same decision (tools/rw_wps/crates/rw-wps/src/
#:   namelist.rs STOCK_WRF_INVENTORIED_MP_PHYSICS).
#: * ``woof/core/preflight.py`` and ``woof/core/rrtmg_legacy.py`` key
#:   all ten accepted selectors -- complete, nothing to judge.
ACCEPTED_MP_PHYSICS = frozenset({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 900, 13, 14, 15, 16, 18, 24, 26, 28, 30, 38, 40, 50, 51, 52, 53, 55, 56, 97, 106})

#: Scheme-KEYED DICTS that omit 28, and the judgement for each.  A dict is a
#: third shape the same defect comes in and the AST membership scan cannot
#: see it, so it is scanned separately.
#:
#: * ``microphysics_transition._MASS_FIELDS`` / ``_MOMENT_FIELDS`` are the
#:   MIXED nest-edge inventories.  28 is absent on purpose and the refusal
#:   is named (``UNVALIDATED_MIXED_EDGE_MOMENTS`` records exactly what an
#:   mp=28 mixed edge would have to move).
#:
#: * ``woof/core/rrtmgp.py``'s scheme map is an OPEN DEFECT, measured and
#:   reported by this package rather than fixed here (the file belongs to
#:   another owner).  ``{6: "wsm6", 8: "thompson", 10: "morrison",
#:   18: "nssl"}.get(mp_physics, "kessler")`` resolves mp=28 to "kessler",
#:   with three consequences on the DEFAULT radiation path for an mp=28 run
#:   (rte-rrtmgp): ``ice_active`` becomes False so ``cal_cldfra1`` is called
#:   with ``f_qi = f_qs = False``; ``effective_fields`` stays empty so the
#:   scheme's effc/effi/effs never reach cloud optics; and
#:   ``hydrometeor_paths`` takes its kessler branch, which returns CONSTANT
#:   10 um liquid and 50 um ice radii.  MEASURED on this tree: an
#:   ice-and-snow-bearing column at 258 K/60 kPa and 240 K/40 kPa gets
#:   CLDFRA 1.0 with ice active and 0.0 without -- an overcast ice cloud
#:   radiating as clear sky.  ``woof/core/preflight.py:2713`` already
#:   PRICES mp=28's effective-radius columns on WRF's use_mp_re table, so
#:   the tree currently budgets for radii the radiation never consumes.
#:
#: The assertion below is "no NEW scheme-keyed dict omits 28".  It
#: deliberately TOLERATES the rrtmgp row disappearing (a sibling package may
#: close it in this same wave) while refusing a fresh one.
KNOWN_SCHEME_KEYED_DICTS_WITHOUT_28 = {
    "woof/core/microphysics_transition.py",
    "woof/core/rrtmgp.py",
}


NON_MP_INTEGER_MAPS = {
    # _ABSB_SHAPES, native RRTM absorption-band table dimensions.
    ("woof/core/rrtm_tables.py", (1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 14)),
    # _LCZ_CLASS_NAMES, URBPARM land-class labels.
    ("woof/experiment.py", tuple(range(1, 12))),
    # WRF_UNPORTED_SCHEME_REASONS['sf_surface_physics'], land selectors.
    ("woof/config.py", (5, 6, 7, 8)),
    # WMO code table 4.7, DERIVED_FORECAST_NAMES, ensemble statistics.
    ("woof/member_prep.py", tuple(range(10))),
    # REGIMES in the independent YSU oracle, column fixture identifiers.
    ("woof/verify/ysu_wide_oracle.py", tuple(range(1, 10))),
}

# ---------------------------------------------------------------------------
# 1. The census.
# ---------------------------------------------------------------------------

def _literal_int_container(node: ast.AST) -> frozenset[int] | None:
    """``(1, 6, 8)`` / ``{1, 6, 8}`` / ``frozenset((...))`` -> the int set."""
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        values = []
        for element in node.elts:
            if (isinstance(element, ast.Constant)
                    and isinstance(element.value, int)
                    and not isinstance(element.value, bool)):
                values.append(element.value)
            else:
                return None
        return frozenset(values)
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in ("frozenset", "set") and len(node.args) == 1):
        return _literal_int_container(node.args[0])
    return None


def _mentions_mp_physics(node: ast.AST) -> bool:
    for sub in ast.walk(node):
        if isinstance(sub, ast.Attribute) and "mp_physics" in sub.attr:
            return True
        if isinstance(sub, ast.Name) and "mp_physics" in sub.id:
            return True
    return False


def scheme_selector_census() -> list[tuple[str, int, str, frozenset[int]]]:
    """Every mp_physics scheme-set literal under ``gpuwm/``.

    Two shapes are collected, because the four known defects came in both:
    a MEMBERSHIP test whose left side names ``mp_physics``, and a module
    CONSTANT holding a scheme set (which is what the fix turns the
    memberships into, and which therefore has to be scanned too or the fix
    would create a new blind spot).
    """
    rows: list[tuple[str, int, str, frozenset[int]]] = []
    for path in sorted(PACKAGE.rglob("*.py")):
        relative = path.relative_to(REPO).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare):
                for op, comparator in zip(node.ops, node.comparators):
                    if not isinstance(op, (ast.In, ast.NotIn)):
                        continue
                    values = _literal_int_container(comparator)
                    if values is None or not _mentions_mp_physics(node.left):
                        continue
                    rows.append((relative, node.lineno, "membership", values))
            elif (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)):
                values = _literal_int_container(node.value)
                if values is None:
                    continue
                name = node.targets[0].id
                if not any(token in name.upper() for token in
                           ("MP_PHYSICS", "MICROPHYSICS", "REFL", "SCHEME")):
                    continue
                rows.append((relative, node.lineno, name, values))
    return rows


def scheme_keyed_dict_census() -> list[tuple[str, int, tuple[int, ...]]]:
    """Dict literals keyed by mp_physics values that omit 28.

    The third shape of the same defect.  ``woof/core/rrtmgp.py``'s
    ``{6: "wsm6", 8: "thompson", ...}.get(mp_physics, "kessler")`` is
    invisible to a membership scan and to a constant scan, and it is the
    most consequential surviving one -- see
    :data:`KNOWN_SCHEME_KEYED_DICTS_WITHOUT_28`.
    """
    rows: list[tuple[str, int, tuple[int, ...]]] = []
    for path in sorted(PACKAGE.rglob("*.py")):
        relative = path.relative_to(REPO).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict) or len(node.keys) < 2:
                continue
            keys = []
            for key in node.keys:
                if (isinstance(key, ast.Constant)
                        and isinstance(key.value, int)
                        and not isinstance(key.value, bool)):
                    keys.append(key.value)
                else:
                    keys = None
                    break
            if not keys:
                continue
            values = set(keys)
            # A microphysics scheme table, not an arbitrary integer map:
            # every key must be an mp_physics selector woof accepts
            # (woof/config.py:1148's set), and it must key 8 plus at least
            # one other ported selector.  Without the subset test this
            # catches RRTMG's band/g-point tables, which are keyed 1..16.
            if (values <= ACCEPTED_MP_PHYSICS and 8 in values
                    and (values & {6, 10, 18}) and 28 not in values):
                rows.append((relative, node.lineno, tuple(sorted(values))))
    return rows


def test_no_new_scheme_keyed_dict_omits_28():
    """The dict shape, with both known rows judged in the constant above."""
    census = scheme_keyed_dict_census()
    unexpected = sorted(
        (path, line, keys) for path, line, keys in census
        if path not in KNOWN_SCHEME_KEYED_DICTS_WITHOUT_28
        and (path, keys) not in NON_MP_INTEGER_MAPS)
    assert unexpected == [], (
        "a NEW mp_physics-keyed dict omits 28:\n  "
        + "\n  ".join(f"{p}:{n} keys={k}" for p, n, k in unexpected)
        + "\nJudge it: either add 28, or record the reason in "
        "KNOWN_SCHEME_KEYED_DICTS_WITHOUT_28 with the measurement.")

    # The named-refusal machinery must stay CONSISTENT, whatever it holds.
    from woof.core import microphysics_transition as mt

    # One row per refused selector, and EVERY refused selector has one.
    # 50, then 16 and 28, left these tables as each entry closure was
    # ratified into PORTED_MP_PHYSICS -- the retired refusals died with their
    # defect, per the guard-retirement law.  What remains is FAST_SBM (30),
    # whose mixed edge would have to build 142 spectral distributions from
    # another scheme's bulk moments with no validated closure, and the named
    # schemes (TEMPO, 900): same-scheme nests run, and a mixed edge is
    # refused by name because no cross-scheme closure for their moments
    # (graupel number and volume) has been written or measured.
    assert (set(mt.UNVALIDATED_MIXED_EDGE_MOMENTS)
            == set(mt.UNVALIDATED_MIXED_EDGE_SELECTORS)
            == set(mt._UNVALIDATED_MIXED_EDGE_REASONS)
            == {3, 5, 14, 15, 26, 30, 38, 40, 51, 52, 53, 55, 56, 97} | set(NAMED_MP_IDS))
    # WDM6 is double-moment in cloud AND rain and carries a CCN reservoir:
    # ncr(:,:,1)=nn, ncr(:,:,2)=nc, ncr(:,:,3)=nr (module_mp_wdm6.F:238-240),
    # all three prognostic scalars in the WRF Registry.  It has no ni/ns/ng.
    # They are REAL moment rows now, with real field codes.
    assert set(mt._MOMENT_FIELDS[16]) == {"nr", "nc", "nn"}
    assert set(mt._MOMENT_FIELDS[28]) == {"nr", "ni", "nc", "nwfa", "nifa"}
    # Real host field codes: nc is mp=9's 22, shared; nn, nwfa and nifa
    # were appended after mp=9's nh at 23, moving nothing below them.
    assert {mt._EDGE_FIELD_CODES[name] for name in ("nc", "nn")} == {22, 24}
    assert {mt._EDGE_FIELD_CODES[name] for name in ("nwfa", "nifa")} == {25, 26}


def test_no_stale_pre_28_scheme_admission_tuple_survives():
    """SCAN, not a list of known paths.

    The pinned-gap test this replaces asserted
    ``runtime_source.count("mp_physics in (1, 6, 8, 10, 18)") == 3`` and was
    therefore blind to the identical fourth gate in
    ``woof/prepared_single_domain_forecast.py``.  Enumerating sites cannot
    find the site nobody enumerated; scanning can.
    """
    census = scheme_selector_census()
    assert census, "the census walk found nothing -- it is broken"

    stale = [
        (path, line, label) for path, line, label, values in census
        if values == STALE_REFL_ADMISSION
        and (path, label) not in DELIBERATE_STALE_SITES
    ]
    assert stale == [], (
        "these sites still carry the pre-mp=28 admission tuple "
        f"{sorted(STALE_REFL_ADMISSION)}:\n  "
        + "\n  ".join(f"{p}:{n} ({label})" for p, n, label in stale)
        + "\nEvery one of them is a gate an mp=28 forecast reaches.  If a "
        "site genuinely must exclude 28, add it to DELIBERATE_STALE_SITES "
        "with the reason -- do not delete this assertion.")


def test_the_hrrr_route_admission_is_derived_not_respelled():
    """The door DELIBERATE_STALE_SITES used to carry, in executable form.

    The route's admission is frozenset(PORTED_MP_PHYSICS) and nothing
    else, so a future port widens it without a table edit.  The
    aerosol-lateral-BC subtraction that used to sit beside it is GONE
    (audit R-044): it named a breakage the WIF climatology ingest and the
    coupled nwfa/nifa boundary had already fixed, it refused correct hrrr
    runs, and it left every other specified-BC route unchecked.  What is
    actually conditional -- the 225 MB dataset -- is measured for every
    route by woof.config.mp28_aerosol_lateral_forcing_precondition and
    raised before the fetch by validate_experiment_preparation, asserted
    below and walked door by door in the test that follows.
    The census walk is blind to derived sets (by design, it reads
    literals), so this pin is the census row's replacement.
    """
    from woof.core import microphysics_transition as mt
    from woof.hrrr_route_inputs import SUPPORTED_MICROPHYSICS

    assert SUPPORTED_MICROPHYSICS == frozenset(mt.PORTED_MP_PHYSICS)
    assert {16, 28, 50} <= SUPPORTED_MICROPHYSICS

    # The route table admits 28 with no subtraction; what decides an mp=28
    # run is the dataset, and it is source-independent by construction --
    # the precondition never sees a route or a source id.
    from woof.config import RUN_PREPARATION_PRECONDITIONS
    from woof.config import mp28_aerosol_lateral_forcing_precondition

    assert mp28_aerosol_lateral_forcing_precondition in (
        RUN_PREPARATION_PRECONDITIONS)

    # PORTED_MP_PHYSICS no longer equals the stale tuple: mp=50 was
    # APPENDED when its rime-pair closure was ratified, and 16 and 28 with
    # theirs (audit R-004), so the census above cannot match it and its
    # DELIBERATE_STALE_SITES row retired with the refusal.
    from woof.core import microphysics_transition as mt

    # mp=9 was APPENDED with mp=50, when its own entry closure (the
    # scheme's consistency block) was read into the edge kernel -- audit
    # R-003 -- so the ported set is the stale admission plus 50 and 9.
    # 16 and 28 were APPENDED next, with their own entry closures, so the
    # ported set is the stale admission plus 50, 9, 16 and 28.
    assert set(mt.PORTED_MP_PHYSICS) == STALE_REFL_ADMISSION | {50, 9, 16, 28, 2, 24, 13}
    # WSM3 (3, phase split), Ferrier-Aligo (5, rime factor) and FAST_SBM (30) are refused for a
    # genuine missing closure (no
    # bulk-to-spectrum entry closure is validated for a mixed edge), and the
    # named schemes are refused by name; both are therefore outside
    # PORTED_MP_PHYSICS and this route's admission.  Any other refused
    # selector is a regression.
    assert set(mt.UNVALIDATED_MIXED_EDGE_SELECTORS) == {3, 5, 14, 15, 26, 30, 38, 40, 51, 52, 53, 55, 56, 97} | set(NAMED_MP_IDS), (
        "a selector other than WSM3, Ferrier-Aligo, FAST_SBM, GCE or a named scheme is refused for a "
        "missing closure again; every other mixed edge in this release runs "
        "with its mapping named")


def _unpreparable_mp28_config(tmp_path, monkeypatch):
    """A loadable mp=28 experiment on a machine with no WIF dataset.

    The fixture pair every front-door test in this tree loads through
    (``tests/test_case_data.py``), with one switch changed: the root
    domain runs mp=28.  A root domain is ``specified`` by construction
    (``woof/experiment.py`` defaults it to ``is_root``), which is the
    condition the precondition is about, and the WIF search is pointed at
    a directory that does not exist -- including the working-directory
    rung, which is WRF's own ``constants_name`` rule.
    """

    from test_case_data import _EXPERIMENT_TOML, make_case_toml

    from woof.ingest import wif_climatology

    monkeypatch.delenv(wif_climatology.WIF_CLIMATOLOGY_PATH_ENV, raising=False)
    monkeypatch.delenv(wif_climatology.WIF_CLIMATOLOGY_ROOT_ENV, raising=False)
    monkeypatch.setenv("WOOF_WIF_DATA_ROOT", str(tmp_path / "no-staged-wif"))
    monkeypatch.setenv("HOME", str(tmp_path / "no-staged-wif"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "no-staged-wif"))

    case = tmp_path / "case"
    case.mkdir()
    monkeypatch.chdir(case)
    return make_case_toml(
        case,
        experiment=_EXPERIMENT_TOML + "moist = true\nmp_physics = 28\n")


def _says_the_precondition(said: str) -> None:
    """The one sentence, with both of its ways out."""

    assert "QNWFA_QNIFA_SIGMA_MONTHLY.dat" in said, said
    assert "$WOOF_WIF_CLIMATOLOGY" in said, said
    assert "mp28_aerosol_source=" in said, said
    assert "synthetic" in said, said


@needs_cupy_install
def test_the_run_dispatch_refuses_before_it_spawns_a_supervisor(
        tmp_path, monkeypatch, capsys):
    """``woof run``, driven rather than read.

    THE PREVIOUS VERSION OF THIS ASSERTED SOURCE TEXT -- that
    ``cli._dispatch`` contains the call.  That is satisfied by a call
    guarded out of every reachable branch, and it cannot see a door it
    was not told to grep.  So the door is DRIVEN: the refusal is the
    process's own exit code and printed sentence, and the supervisor is
    replaced by a seam that fails the test if the dispatch ever reaches
    it -- the supervisor being exactly what used to be spawned before the
    floor inside the time loop refused.
    """

    from woof import supervisor
    from woof.cli import main as cli_main

    config = _unpreparable_mp28_config(tmp_path, monkeypatch)

    def _never(args):                      # pragma: no cover - the point
        raise AssertionError(
            "the run dispatch spawned a supervisor for a config this "
            "machine cannot prepare")

    monkeypatch.setattr(supervisor, "supervise_from_cli", _never)

    outdir = tmp_path / "out"
    assert cli_main(["run", str(config), "--outdir", str(outdir)]) == 2
    said = capsys.readouterr().err
    assert "d01: " in said, said
    _says_the_precondition(said)
    assert not outdir.exists(), "the run root was claimed anyway"


@needs_cupy_install
def test_the_run_plan_door_refuses_before_it_fetches(tmp_path, monkeypatch,
                                                     capsys):
    """``woof run-plan``, which is what the desktop launches through.

    THE DEFECT THIS PREVENTS, measured.  ``woof run-plan`` loads the
    experiment and then runs its own fetch stage; the floor that raises
    this sentence is downstream of it, in preparation.  So the shape the
    other two doors were fixed for -- review says PASS, 10-15 GB is
    downloaded, then the run refuses -- survived on the front door a
    desktop forecast actually takes.  Both halves are driven here: plan
    review (``--resolve``) answers with the sentence, and a full
    execution never reaches the fetch stage, with ``_run_fetch`` replaced
    by a seam that fails the test if it is called.
    """

    import json

    from woof import runplan
    from woof.cli import main as cli_main

    config = _unpreparable_mp28_config(tmp_path, monkeypatch)
    run_root = tmp_path / "runs"
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps({
        "schema": runplan.PLAN_SCHEMA,
        "name": "mp28-unpreparable",
        "route": "experiment",
        "config": {"path": str(config)},
        "output_root": str(run_root),
        "fetch": {"args": ["--source", "gfs", "--cycle", "2026-07-29T18",
                           "--hours", "6"]},
    }), encoding="utf-8")

    # 1. PLAN REVIEW.  This is the call the desktop makes before it
    #    offers a Launch button: run-plan --resolve, then --estimate,
    #    then the run itself.
    assert cli_main(["run-plan", str(plan_path), "--resolve"]) == 2
    _says_the_precondition(capsys.readouterr().err)
    assert not run_root.exists(), "plan review claimed a run root"

    # 2. THE RUN, with the fetch stage armed to fail the test.
    def _never(*args, **kwargs):           # pragma: no cover - the point
        raise AssertionError(
            "run-plan reached its fetch stage for a config this machine "
            "cannot prepare")

    monkeypatch.setattr(runplan, "_run_fetch", _never)
    assert cli_main(["run-plan", str(plan_path)]) == 1
    stream = next(run_root.glob("**/" + runplan.EVENTS_FILENAME))
    events = [json.loads(line)
              for line in stream.read_text(encoding="utf-8").splitlines()
              if line.strip()]
    assert not [event for event in events
                if event.get("stage") == "fetch"], events
    # Nothing under the run root but the front door's own records: the
    # event stream, the manifest that names it, and the time ledger the
    # run observer keeps beside the events.  No fetch directory, no
    # prepared root, no forecast output.
    assert sorted(path.name for path in run_root.rglob("*")
                  if path.is_file()) == sorted(
                      [runplan.EVENTS_FILENAME, runplan.MANIFEST_FILENAME,
                       "timeline.jsonl"]), (
                          sorted(str(path) for path in run_root.rglob("*")))
    failed = [event for event in events if event.get("event") == "failed"]
    assert len(failed) == 1, events
    _says_the_precondition(failed[0]["message"])


def _named_stub_mp28_config(tmp_path, monkeypatch, *, via):
    """An mp=28 experiment whose dataset is NAMED and is not a dataset.

    Four bytes is what a download that died in its first packet leaves on
    disk, and the 225 MB file it was meant to be is the single most
    likely thing to arrive truncated.  ``via`` names which of the two
    human-chosen rungs points at it -- ``$WOOF_WIF_CLIMATOLOGY`` or
    ``[shared] wif_climatology_path`` -- because the resolver refuses on
    both and, before this lane's fix, neither door converted the refusal.
    """

    from test_case_data import _EXPERIMENT_TOML, make_case_toml

    from woof.ingest import wif_climatology

    stub = tmp_path / "QNWFA_QNIFA_SIGMA_MONTHLY.dat"
    stub.write_bytes(bytes(4))

    monkeypatch.delenv(wif_climatology.WIF_CLIMATOLOGY_PATH_ENV, raising=False)
    monkeypatch.delenv(wif_climatology.WIF_CLIMATOLOGY_ROOT_ENV, raising=False)
    monkeypatch.setenv("WOOF_WIF_DATA_ROOT", str(tmp_path / "no-staged-wif"))
    monkeypatch.setenv("HOME", str(tmp_path / "no-staged-wif"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "no-staged-wif"))

    experiment = _EXPERIMENT_TOML
    if via == "env":
        monkeypatch.setenv(wif_climatology.WIF_CLIMATOLOGY_PATH_ENV, str(stub))
    else:
        # In ``[shared]``: ``wif_climatology_path`` is not a per-domain
        # key, and ``woof/experiment.py`` refuses it on a ``[[domain]]``
        # table rather than dropping it.
        named = 'wif_climatology_path = "' + stub.as_posix() + '"'
        experiment = experiment.replace("[shared]\n", "[shared]\n" + named + "\n")

    case = tmp_path / "case"
    case.mkdir()
    monkeypatch.chdir(case)
    return stub, make_case_toml(
        case, experiment=experiment + "moist = true\nmp_physics = 28\n")


def _says_the_named_file_is_not_the_dataset(said: str, stub, *, origin) -> None:
    """One sentence naming the file, the breakage and the way out."""

    assert origin in said, said
    assert str(stub) in said, said
    assert "WPS intermediate" in said, said
    assert "woof fetch-tables --wif" in said, said
    # A config-shaped refusal is a sentence, not a stack.
    assert "Traceback" not in said, said
    assert "MissingWifClimatologyDataset" not in said, said


@needs_cupy_install
@pytest.mark.parametrize("via,origin", [
    ("env", "$WOOF_WIF_CLIMATOLOGY"),
    ("shared", "[shared] wif_climatology_path"),
])
def test_a_named_dataset_that_is_not_one_refuses_in_a_sentence(
        tmp_path, monkeypatch, capsys, via, origin):
    """THE DEFECT: a NAMED but invalid dataset tracebacked at every door.

    The resolver raises rather than degrading when a human chose the path
    -- an override that is silently ignored is how a run takes an initial
    condition nobody chose -- and the three doors convert only
    ``ValueError``.  So ``$WOOF_WIF_CLIMATOLOGY`` pointed at a truncated
    copy exited 1 from ``woof run``, ``woof run-plan --resolve`` and
    ``woof go`` with a raw
    ``woof.ingest.wif_dataset.MissingWifClimatologyDataset`` stack, while
    the absent-dataset case one line away answered in one sentence with
    exit 2.  Both committing doors in this file are driven here; ``woof
    go`` is driven by ``tests/test_mp28_runnable.py::
    test_the_go_chain_refuses_a_named_dataset_that_is_not_one``.
    """

    import json

    from woof import runplan, supervisor
    from woof.cli import main as cli_main

    stub, config = _named_stub_mp28_config(tmp_path, monkeypatch, via=via)

    def _never(*args, **kwargs):           # pragma: no cover - the point
        raise AssertionError(
            "a door committed to a forecast whose named dataset is a "
            "4-byte stub")

    monkeypatch.setattr(supervisor, "supervise_from_cli", _never)

    outdir = tmp_path / "out"
    assert cli_main(["run", str(config), "--outdir", str(outdir)]) == 2
    said = capsys.readouterr().err
    assert "d01: " in said, said
    _says_the_named_file_is_not_the_dataset(said, stub, origin=origin)
    assert not outdir.exists(), "the run root was claimed anyway"

    run_root = tmp_path / "runs"
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps({
        "schema": runplan.PLAN_SCHEMA,
        "name": "mp28-stub-dataset",
        "route": "experiment",
        "config": {"path": str(config)},
        "output_root": str(run_root),
        "fetch": {"args": ["--source", "gfs", "--cycle", "2026-07-29T18",
                           "--hours", "6"]},
    }), encoding="utf-8")
    monkeypatch.setattr(runplan, "_run_fetch", _never)
    assert cli_main(["run-plan", str(plan_path), "--resolve"]) == 2
    _says_the_named_file_is_not_the_dataset(
        capsys.readouterr().err, stub, origin=origin)
    assert not run_root.exists(), "plan review claimed a run root"


def test_every_door_that_commits_to_a_forecast_asks_the_precondition():
    """The census the previous tests' sentence rests on.

    A precondition raised only by ``initialize_real`` is raised AFTER the
    fetch on every route a user types: on ``woof go`` that call is in the
    prepare stage, downstream of the authority stage and of the whole
    downloaded cycle; on ``woof run`` it is inside the time loop; on
    ``woof run-plan`` the route's own fetch stage runs first.  So the
    doors that COMMIT to building a forecast ask the inventory themselves,
    before they spend anything.

    THE THREE DOORS ARE DRIVEN, NOT GREPPED: ``woof go`` by
    ``tests/test_mp28_runnable.py::
    test_the_go_chain_refuses_an_unpreparable_mp28_run_before_it_fetches``,
    ``woof run`` and ``woof run-plan`` by the two tests above.  What is
    left here is the part a behavioural test cannot state: that the three
    read ONE inventory rather than three copies of a sentence, and that
    the two reporting surfaces report instead of deciding.
    """

    import inspect

    from woof import cli, go_cli, runplan
    from woof.config import (RUN_PREPARATION_PRECONDITIONS,
                              experiment_preparation_refusals,
                              run_preparation_preconditions,
                              validate_experiment_preparation,
                              validate_run_preparation)
    from woof.core import preflight
    from woof.ingest import real

    # One inventory, four readers, no second list.
    assert RUN_PREPARATION_PRECONDITIONS
    for reader in (run_preparation_preconditions,
                   validate_run_preparation,
                   experiment_preparation_refusals,
                   validate_experiment_preparation):
        assert callable(reader)

    # Every door reaches the inventory through one of those readers
    # rather than spelling the question again.  A door that grew its own
    # copy of the sentence is what this catches; that each one FIRES is
    # measured behaviourally, above and in test_mp28_runnable.py.
    # _dispatch owns the output scope; its forecast preparation lives in
    # _dispatch_with_outputs. The behavioural tests above drive that wrapper.
    for door in (go_cli.plan_from_config, cli._dispatch_with_outputs,
                 runplan.resolve_plan):
        source = inspect.getsource(door)
        assert ("experiment_preparation_refusals" in source
                or "validate_experiment_preparation" in source), door
        assert "MP28_AEROSOL_LATERAL_FORCING_PRECONDITION" not in (
            source), door

    # ... and on `woof go` it is asked BEFORE the fetch stage is
    # composed.  The other two orders are measured behaviourally above.
    chain = inspect.getsource(go_cli._go_prepared_main)
    assert chain.index("plan_from_config(") < chain.index(
        '_run_stage("fetch"')

    # `woof check` reports it without changing a verdict: it can be asked
    # about a machine that is not this one, where "the dataset is not
    # here" says nothing about whether the run fits there.
    assert "_warn_unmet_run_preparation" in inspect.getsource(
        preflight.check_main)

    # And the floor stays where it was, so a RunConfig that reached
    # initialization without passing any door is still refused.
    assert "validate_run_preparation" in inspect.getsource(
        real.initialize_real)


def test_every_refl_10cm_admission_constant_admits_28():
    """The four production gates, by value rather than by source text.

    ``tests/test_refl.py::test_refl_stash_has_no_trajectory_or_restart_reader``
    pins the exact set of files that consume the one-frame REFL handoff.
    This asserts that each of them admits 28, so the pinned consumer census
    and the admission census cannot drift apart.
    """
    from woof import prepared_single_domain_forecast as psdf
    from woof import runtime
    from woof.verify.cases import nest_ideal_common, real74_n5s

    for module, attribute in (
            (runtime, "REFL_10CM_MICROPHYSICS"),
            (psdf, "REFL_10CM_MICROPHYSICS"),
            (nest_ideal_common, "REFL_10CM_MICROPHYSICS"),
            (real74_n5s, "REFLECTIVITY_MICROPHYSICS")):
        values = frozenset(getattr(module, attribute))
        assert values == REFL_ADMISSION, (
            f"{module.__name__}.{attribute} is {sorted(values)}, expected "
            f"{sorted(REFL_ADMISSION)}")

    # woof/core/refl.py is the producer side and is deliberately a
    # DIFFERENT set: 18 routes through NSSL's own native reflectivity and
    # never through compute_refl_10cm.  Assert the asymmetry on purpose so
    # a future "consistency" edit has to argue with it.
    refl_source = (PACKAGE / "core" / "refl.py").read_text(encoding="utf-8")
    # Re-derived 2026-08-09 for the WDM6 port: the producer gate gained 16
    # because compute_refl_10cm now HAS an mp=16 branch (:615, launching
    # woof/core/kernels/wdm6_refl.cu).  18 is still absent for the original
    # reason.  Re-derived again 2026-08-11 (1.9.1): mp=9 and mp=50 joined
    # the CONSUMER set because both were shipped staging a field nothing
    # consumed (the exact mp=28 crash, at the second output frame), and
    # both belong on 18's side of the asymmetry -- Milbrandt-Yau stashes
    # the scheme's own Zet (woof/core/milbrandt2.py, the INOUT dummy WRF
    # binds straight to refl_10cm) and P3 stashes its own diagnostic
    # reflectivity (woof/core/p3.py), so neither routes through
    # compute_refl_10cm.  The asymmetry the assertion protects -- consumer
    # set minus producer set == the native-reflectivity schemes -- is
    # asserted as arithmetic below rather than left implicit in two
    # hand-spelled tuples.
    # The producer gate reads the PUBLISHED input-species table (the same
    # row the physics registry carries per scheme) instead of a literal
    # tuple, so the pin is on the table's keys and on the source reading
    # the table; the asymmetry arithmetic below is unchanged.
    from woof.core.refl import REFL_10CM_INPUT_SPECIES
    producer = frozenset(REFL_10CM_INPUT_SPECIES)
    assert producer == frozenset({1, 2, 3, 4, 6, 8, 10, 13, 14, 16, 26, 28, 38, 40, 106})
    assert "cfg.mp_physics not in REFL_10CM_INPUT_SPECIES" in refl_source
    # The named schemes stash their own native reflectivity (TEMPO's,
    # ``native_reflectivity``), so they sit on 18's side as well.
    from woof.microphysics_schemes import SCHEMES_BY_ID
    assert REFL_ADMISSION - producer == {5, 7, 9, 15, 18, 24, 50, 51, 52, 53, 55, 56, 97} | {
        mp for mp, scheme in SCHEMES_BY_ID.items() if scheme.native_reflectivity}
    assert "elif cfg.mp_physics in (8, 28, 38):" in refl_source, (
        "mp=28 no longer shares mp=8's calc_refl10cm branch; WRF has ONE "
        "such routine with no aerosol-aware arm (module_mp_thompson.F:"
        "5710-6028), so a separate branch could only diverge from it")
    # WDM6 does NOT share it: refl10cm_wdm6 is its own Fortran routine
    # (module_mp_wdm6.F:2957-3131) taking a prognostic rain number, so it
    # gets its own branch and its own kernel.
    assert "elif cfg.mp_physics in (16, 26):" in refl_source


def _code_without_comments_or_strings(path: pathlib.Path) -> str:
    """Executable text only.

    A substring search over raw source cannot distinguish a live gate from
    a comment that QUOTES the retired gate, and these files document what
    they replaced -- deliberately, because that is what makes the fix
    legible.  Tokenizing drops comments and string literals so the search
    means what it says.
    """
    import io
    import tokenize

    pieces = []
    with path.open("rb") as handle:
        for token in tokenize.tokenize(io.BytesIO(handle.read()).readline):
            if token.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            pieces.append(token.string)
    return " ".join(pieces)


def test_the_runtime_refl_gates_are_reached_through_the_named_constant():
    """No inlined copy may survive in any of the four owning files.

    The defect was never "the tuple is wrong"; it was "there are four
    copies of the tuple".  Value equality above cannot see a fifth inlined
    copy inside the same file, so the code text is checked too.
    """
    for relative in ("woof/runtime.py",
                     "woof/prepared_single_domain_forecast.py",
                     "woof/verify/cases/nest_ideal_common.py",
                     "woof/verify/cases/real74_n5s.py"):
        code = _code_without_comments_or_strings(REPO / relative)
        normalized = code.replace(" ", "")
        assert "mp_physicsin(1,6,8,10,18)" not in normalized, relative
        assert "mp_physicsnotin(1,6,8,10,18)" not in normalized, relative

    tree = ast.parse((REPO / "woof/runtime.py").read_text(encoding="utf-8"))
    imports = [node for node in tree.body if isinstance(node, ast.ImportFrom)
               and node.module == "woof.core.physics_inventory"
               and any(alias.name == "REFL_10CM_MICROPHYSICS"
                       for alias in node.names)]
    assert len(imports) == 1
    # Case output now has separate ensemble-capture, streamed, and ordinary
    # resident consumers. Each uses the same admission inventory, along with
    # the substep schedule and the tree history handoff.
    gates = {}
    for function in tree.body:
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        references = [node for node in ast.walk(function)
                      if isinstance(node, ast.Name)
                      and node.id == "REFL_10CM_MICROPHYSICS"]
        if references:
            gates[function.name] = len(references)
            comparisons = [node for node in ast.walk(function)
                           if isinstance(node, ast.Compare)
                           and any(isinstance(value, ast.Name)
                                   and value.id == "REFL_10CM_MICROPHYSICS"
                                   for value in node.comparators)]
            assert len(comparisons) == len(references), function.name
            assert all(isinstance(node.left, ast.Attribute)
                       and node.left.attr == "mp_physics"
                       and len(node.ops) == 1
                       and isinstance(node.ops[0], ast.In)
                       for node in comparisons), function.name
    # write_case_output's fourth gate (01402fcb1): the analysis frame
    # writes WRF's initial all-zero REFL_10CM for a scheme that stashes one.
    assert gates == {"write_case_output": 4, "integrate_prepared_case": 1,
                     "_submit_tree_history_frame": 1}


def test_the_prepared_runner_consumes_the_mp28_refl_handoff(monkeypatch):
    """Behavioural, on the FOURTH site -- the one the old gap test missed.

    ``_consume_due_native_refl_10cm`` is a pure function of (ticks, state,
    consumer), so it is exercised directly rather than by standing up a
    prepared forecast.  A recording consumer proves the handoff is taken
    for 28 and, equally, that tick 0 still takes nothing.
    """
    from types import SimpleNamespace

    from woof.prepared_single_domain_forecast import (
        _consume_due_native_refl_10cm)

    from woof.core import refl
    # This test owns handoff routing; the GPU diagnostic has its own check.
    monkeypatch.setattr(refl, "analysis_refl_10cm",
                        lambda state, *, shape: np.full(shape, 15, dtype=np.float32))
    sentinel = object()
    calls = []

    def consumer(state):
        calls.append(state)
        return sentinel

    for mp_physics in sorted(REFL_ADMISSION):
        state = SimpleNamespace(
            qv=np.zeros(1),
            physics=SimpleNamespace(mp_physics=mp_physics))
        assert _consume_due_native_refl_10cm(state, 1, consumer) is sentinel
        # Tick 0 takes no stash. It diagnoses the initial hydrometeors.
        analysis = _consume_due_native_refl_10cm(state, 0, consumer,
                                                 shape=(1, 1, 1))
        assert analysis is not sentinel
        assert np.asarray(analysis).shape == (1, 1, 1)
        assert (np.asarray(analysis) == 15).all()
    assert len(calls) == len(REFL_ADMISSION)

    # A scheme with no native field must still take nothing.
    other = SimpleNamespace(qv=np.zeros(1),
                            physics=SimpleNamespace(mp_physics=0))
    assert _consume_due_native_refl_10cm(other, 1, consumer) is None


def test_the_runtime_tree_history_frame_consumes_the_mp28_refl_handoff():
    """Behavioural, on the THIRD runtime site (the nested-tree lane).

    ``_submit_tree_history_frame`` is the production handoff for every
    multi-domain run, and mp=28 is reachable exactly there -- the registry
    makes it a per-domain component override, never a base template
    (woof/physics_compat.py's readiness note).  So this is not a corner:
    it is the ONE lane an mp=28 user's history frames go through.
    """
    from types import SimpleNamespace

    from woof import runtime
    from woof.core import refl

    field = np.zeros((2, 3, 4), dtype=np.float32)
    submitted = []

    class _Writers:
        @staticmethod
        def submit(node, ticks, *, refl_field):
            submitted.append((ticks, refl_field))

    class _Refl:
        # The stub MIRRORS woof.core.refl: only the consumer is faked
        # (that is what this pin measures).  The due predicate and the
        # domain-start reader are the real ones, so a stub cannot make
        # the site look reachable by answering a question differently
        # from the module it stands in for.
        consume_refl_10cm = staticmethod(lambda state: field)
        refl_10cm_stash_is_due = staticmethod(refl.refl_10cm_stash_is_due)
        domain_start_ticks_of = staticmethod(refl.domain_start_ticks_of)
        # The GPU diagnostic is checked separately; pin the routing here.
        analysis_refl_10cm = staticmethod(
            lambda state, *args, shape, **kw: np.full(shape, 15, dtype=np.float32))

    class _UhDiag:
        @staticmethod
        def reset_up_heli_max(state):
            return None

    node = SimpleNamespace(
        state=SimpleNamespace(qv=np.zeros(1),
                              physics=SimpleNamespace(mp_physics=28)),
        # The analysis frame sizes WRF's initial array off the domain's
        # own run (01402fcb1).
        cfg=SimpleNamespace(run=SimpleNamespace(nz=2, ny=3, nx=4)))

    monkey = pytest.MonkeyPatch()
    try:
        monkey.setitem(sys.modules, "woof.core.refl", _Refl)
        monkey.setitem(sys.modules, "woof.core.uh_diag", _UhDiag)
        runtime._submit_tree_history_frame(_Writers, node, ticks=6)
        # Tick 0 diagnoses the initial hydrometeors without consuming a stash.
        runtime._submit_tree_history_frame(_Writers, node, ticks=0)
    finally:
        monkey.undo()

    assert [ticks for ticks, _ in submitted] == [6, 0], submitted
    assert submitted[0][1] is field
    assert submitted[1][1] is not field and (np.asarray(submitted[1][1]) == 15).all()


def test_the_missing_28_refl_admission_was_a_crash_not_a_missing_field():
    """MEASURED consequence, reproduced against the real refl contract.

    The published description of this gap said mp=28 wrfout frames would be
    written "with REFL_10CM never produced".  That understates it.  The tree
    model gates ``refl_10cm_due`` on the history handler and the clock ONLY
    (woof/core/model.py:669-675), never on ``mp_physics``, so the mp=28
    adapter DID stage the field on every history-due step
    (woof/core/microphysics_aerosol.py's ``compute_and_stash_refl_10cm``
    call).  With 28 missing from the consumer's admission set the stash was
    never taken, and ``refl.stash_refl_10cm``'s consume-once contract then
    raises on the NEXT history-due step.

    So the pre-fix behaviour on the ONLY lane mp=28 is reachable through
    (``tools.prepared_domain_tree_forecast``, the sole runner route with
    ``microphysics`` in ``allowed_component_overrides``) was a hard
    ``RuntimeError`` at the second output frame, not a quietly absent
    diagnostic.  This reproduces both sides.
    """
    from types import SimpleNamespace

    from woof.core.refl import consume_refl_10cm, stash_refl_10cm

    def three_history_steps(admission, mp_physics):
        state = SimpleNamespace(
            qv=np.zeros(1),
            physics=SimpleNamespace(mp_physics=mp_physics, refl_10cm=None))
        for _ in range(3):
            stash_refl_10cm(state,
                            np.zeros((2, 3, 4), dtype=np.float32))
            if state.physics.mp_physics in admission:
                consume_refl_10cm(state)

    # The retired admission set: mp=8 survives, mp=28 raises on step 2.
    three_history_steps(STALE_REFL_ADMISSION, 8)
    with pytest.raises(RuntimeError, match="not consumed before reuse"):
        three_history_steps(STALE_REFL_ADMISSION, 28)

    # The current one: both survive.
    three_history_steps(REFL_ADMISSION, 8)
    three_history_steps(REFL_ADMISSION, 28)

    # And the set the production consumer actually uses is the current one.
    from woof import runtime

    three_history_steps(runtime.REFL_10CM_MICROPHYSICS, 28)


def test_the_nest_history_refl_handoff_admits_28():
    """``woof/verify/cases/nest_ideal_common.py`` -- the fifth site.

    It is not one of the four the auditor traced; it was found by the
    census.  It matters because the ideal-nest cases exist to run
    production's seams, and a case that silently skipped the handoff for
    mp=28 would leave the stash unconsumed until ``refl.py`` raised at the
    NEXT history frame -- far from the omission.
    """
    from types import SimpleNamespace

    from woof.core import refl
    from woof.verify.cases import nest_ideal_common

    consumed = []

    class _Refl:
        # Mirrors woof.core.refl; only the consumer is faked.  See the
        # tree-history pin above for why the predicate stays real.
        consume_refl_10cm = staticmethod(consumed.append)
        refl_10cm_stash_is_due = staticmethod(refl.refl_10cm_stash_is_due)
        domain_start_ticks_of = staticmethod(refl.domain_start_ticks_of)

    node = SimpleNamespace(
        cfg=SimpleNamespace(run=SimpleNamespace(mp_physics=28), grid_id=1),
        state=SimpleNamespace(physics=object()))

    monkey = pytest.MonkeyPatch()
    try:
        monkey.setitem(sys.modules, "woof.core.refl", _Refl)
        nest_ideal_common.consume_history_reflectivity(node, ticks=3)
        assert consumed == [node.state]
        # tick 0 is not a period boundary; nothing is staged, nothing taken.
        nest_ideal_common.consume_history_reflectivity(node, ticks=0)
        assert consumed == [node.state]
    finally:
        monkey.undo()


# ---------------------------------------------------------------------------
# 2. Restart.
# ---------------------------------------------------------------------------

MP28_AEROSOL_FIELDS = ("nc", "nwfa", "nifa", "nwfa2d", "nifa2d")


def test_the_restart_identity_names_the_scheme_and_its_provenance():
    """A scheme with no identity row cannot be checkpointed at all.

    The string is not decoration: ``physics_setup_identity`` folds it into a
    SHA-256 that a resume must reproduce exactly, so it is the mechanism by
    which a trajectory-changing implementation change refuses an
    incompatible restart BEFORE any state is restored.  It therefore has to
    name the things that would change the trajectory, in the style of the
    mp=8 row, not just the scheme.
    """
    from woof.io import restart

    identity = restart.MICROPHYSICS_ALGORITHM_IDENTITIES[28]
    assert identity.startswith("thompson-aerosol-aware-wrf-v4.6.1-")
    for token in ("prognostic-nc", "nwfa", "nifa", "ccn-activate-table",
                  "demott", "koop", "scavenging", "surface-emission",
                  "synthetic-aerosol-init"):
        assert token in identity, token
    # It must be a NEW identity, not a re-use of the classic Thompson one.
    assert identity != restart.MICROPHYSICS_ALGORITHM_IDENTITIES[8]
    # And nothing else moved.  Re-derived 2026-08-09: the WDM6 port added
    # key 16 and NOTHING ELSE, which is the claim this pin exists to hold.
    # 16 is the only new key, its identity is its own string rather than a
    # re-use, and every pre-existing row is byte-unchanged (mp=8's exact
    # string is re-asserted below, and mp=28's above).
    #
    # RE-DERIVED AGAIN at the 1.9 gate: 50 is here because the P3
    # one-category port (mp_physics=50, woof/core/p3.py) registered its own
    # restart identity, which is exactly what a new scheme is required to
    # do -- woof/io/restart.py refuses to checkpoint a scheme with no
    # identity row at all.  It is a DELIBERATE pin update, not a widened
    # assertion: 50 is the only new key, and the two properties this case
    # exists for are re-asserted for it below on the same terms as 16's.
    #
    # RE-DERIVED 2026-09-10: 9 (Milbrandt-Yau) is here now, and the note
    # this replaces was wrong in the way a pin's prose can be wrong -- it
    # recorded the absence as deliberate ("a Milbrandt-Yau run cannot be
    # checkpointed") when the absence was the defect: mp=9 was accepted by
    # the loader and integrated by woof/core/milbrandt2.py, so a run of it
    # reached its first restart interval and died there with
    # "cannot identify unsupported microphysics scheme 9", losing the
    # forecast.  The row (and a plan-review gate that asks the question
    # before the run starts) is the fix.  9 is the only new key, its
    # identity is its own string rather than a re-use, and every
    # pre-existing row is byte-unchanged -- the same two properties 16 and
    # 50 are held to below.
    #
    # RE-DERIVED with the RRFS v2 intake: the named schemes (TEMPO, 900)
    # carry their own identity rows.  RE-DERIVED for 2.8.9: 30 (FAST_SBM)
    # registered its own identity with the port, held to the same two
    # properties below.  The WRF-numbered rows are otherwise unchanged.
    assert set(restart.MICROPHYSICS_ALGORITHM_IDENTITIES) == {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 16, 18, 24, 26, 28, 30, 38, 40, 50, 51, 52, 53, 55, 56, 97, 106} | set(NAMED_MP_IDS)
    fsbm_identity = restart.MICROPHYSICS_ALGORITHM_IDENTITIES[30]
    assert fsbm_identity not in {
        value for key, value in restart.MICROPHYSICS_ALGORITHM_IDENTITIES.items()
        if key != 30}
    milbrandt_identity = restart.MICROPHYSICS_ALGORITHM_IDENTITIES[9]
    assert milbrandt_identity.startswith("milbrandt-yau-wrf-v4.6.1-")
    assert milbrandt_identity not in {
        value for key, value in restart.MICROPHYSICS_ALGORITHM_IDENTITIES.items()
        if key != 9}
    p3_identity = restart.MICROPHYSICS_ALGORITHM_IDENTITIES[50]
    assert p3_identity.startswith("p3-one-category-wrf-v4.6.1-")
    assert p3_identity not in {
        value for key, value in restart.MICROPHYSICS_ALGORITHM_IDENTITIES.items()
        if key != 50}
    wdm6_identity = restart.MICROPHYSICS_ALGORITHM_IDENTITIES[16]
    assert wdm6_identity.startswith("wdm6-double-moment-warm-rain-wrf-v4.6.1-")
    assert wdm6_identity not in {
        value for key, value in restart.MICROPHYSICS_ALGORITHM_IDENTITIES.items()
        if key != 16}
    assert restart.MICROPHYSICS_ALGORITHM_IDENTITIES[8] == (
        "classic-thompson-wrf-v4.6.1-v5-cloud-fallout-"
        "refl10cm-ng-shadow-snow-rime-mass-number-velocity-rain-density-condensation-history-"
        "one-slab-rain-graupel-real-clamp-product")


@requires_gpu
def test_the_mp28_restart_identity_binds_every_table_it_reads():
    """mp=8 pinned its table bytes; mp=28 pinned none of them.

    ``physics_setup_identity`` gave mp=28 only ``{"scheme_id": 28}``, so a
    checkpoint could resume against a DIFFERENT ``freezeH2O.dat`` or a
    different ``CCN_ACTIVATE.BIN`` with every identity check passing.  That
    matters more here than for any other scheme: the droplet number that
    defines mp=28 is READ out of ``tnccn_act`` at a fixed radius/kappa index
    (module_mp_thompson.F:5229-5230), not computed, so a table substitution
    IS a trajectory substitution and nothing else in the header would show
    it.

    Both inventories are required.  mp=28 really does load the four classic
    tables (its adapter reuses the frozen mp=8 sedimentation and
    classic-graupel launchers), and it alone loads the activation blob.
    """
    from woof.core.thompson_aerosol_contract import (
        AEROSOL_TABLE_ASSETS, AEROSOL_TABLE_SET_ID)
    from woof.core.thompson_contract import CLASSIC_TABLE_ASSETS, TABLE_SET_ID
    from woof.io import restart

    import cupy as cp

    state, _forcing, cfg = _mp28_forecast_state(cp)
    identity = restart.physics_setup_identity(state, cfg)
    record = identity["microphysics"]["thompson_aerosol"]

    classic = record["classic_tables"]
    assert classic["table_set"] == TABLE_SET_ID
    assert [item["sha256"] for item in classic["assets"]] == [
        asset.sha256 for asset in CLASSIC_TABLE_ASSETS]

    aerosol = record["aerosol_tables"]
    assert aerosol["table_set"] == AEROSOL_TABLE_SET_ID
    assert aerosol["table_set"] != TABLE_SET_ID, (
        "the two coefficient inventories must stay tellable apart")
    assert [item["sha256"] for item in aerosol["assets"]] == [
        asset.sha256 for asset in AEROSOL_TABLE_ASSETS]

    # A PACKAGING fact must never enter the trajectory identity.  This dict
    # is hashed into physics_setup_fingerprint, so a "redistributed" key
    # would make flipping AEROSOL_ASSET_REDISTRIBUTED refuse every mp=28
    # checkpoint written before the flip -- while the bytes bound above,
    # which are the only thing that can move a float, stayed identical.
    # It was written here for part of 2026-08-01 and removed the same day.
    # Asserted as an absence so re-adding it is a test failure, not a
    # silently shipped incompatibility.
    assert "redistributed" not in aerosol, (
        "how the activation blob was DELIVERED is not part of the "
        "trajectory identity; its sha256 already binds what is")
    from woof.core.thompson_aerosol_contract import (
        AEROSOL_ASSET_REDISTRIBUTED)

    assert AEROSOL_ASSET_REDISTRIBUTED is True, (
        "the constant itself stays -- it is published on the registry row "
        "and drives the packaging gates; only the fingerprint is free of it")
    assert "redistributed" not in json.dumps(record), (
        "no nested record may smuggle the packaging flag back into the "
        "hashed physics setup")

    # The aerosol INITIAL CONDITION is part of the identity too: this build
    # runs thompson_init's synthetic profile, and a future WIF metgrid
    # ingest is a different initial condition, not a compatible resume.
    assert "thompson-init-synthetic" in record["aerosol_source"]
    assert "aer-init-opt-0" in record["aerosol_source"]

    # An mp=8 checkpoint must be unaffected -- it carries "thompson", never
    # "thompson_aerosol".
    from dataclasses import replace as _replace

    mp8_cfg = _replace(cfg, mp_physics=8)
    mp8_identity = restart.physics_setup_identity(state, mp8_cfg)
    assert "thompson" in mp8_identity["microphysics"]
    assert "thompson_aerosol" not in mp8_identity["microphysics"]


def _mp28_forecast_state(cp):
    """The G4 forecast domain, with the production aerosol profile."""
    sys.path.insert(0, str(REPO / "tests"))
    from test_mp28_forecast_smoke import (  # noqa: E402
        FORECAST_U, _attach_specified_boundaries, _build_states,
        _forecast_config, _tables_or_skip)

    _tables_or_skip()
    cfg = _forecast_config()
    state, forcing = _build_states(cp, cfg, bubble=False, wind=FORECAST_U)
    _attach_specified_boundaries(state, forcing, cfg)
    from woof.core import microphysics

    receipt = microphysics.microphysics_init(state, cfg)
    assert receipt, "microphysics_init returned an empty mp=28 receipt"
    cp.cuda.Stream.null.synchronize()
    return state, forcing, cfg


@requires_gpu
def test_an_mp28_restart_round_trips_every_aerosol_field_bitwise():
    """WRITE a real file, READ it back, compare BITS.

    The manifest-level gate in tests/test_preflight.py proves the fields are
    classified and copied.  This proves the whole file cycle -- header,
    physics-setup fingerprint, array manifest, npz, restore -- carries them,
    which is the claim a user depends on and the one that was refused
    outright before the identity row existed.

    ``nifa2d`` is deliberately included even though thompson_init leaves it
    at exactly zero (it is not even a thompson_init dummy argument,
    module_mp_thompson.F:424-444): a field that is legitimately zero is the
    easiest one for a restart to "round-trip" by accident, so ``nc`` is
    perturbed to a random field first and every array is compared as raw
    bytes rather than with a tolerance.
    """
    import cupy as cp

    from woof.io import restart

    state, _forcing, cfg = _mp28_forecast_state(cp)
    rng = np.random.default_rng(2028)
    state.nc[...] = cp.asarray(
        rng.random(state.nc.shape, dtype=np.float32) * np.float32(1.5e8))
    # nifa2d is exactly zero out of thompson_init (it is not even one of its
    # dummy arguments, module_mp_thompson.F:424-444) and an all-zero field is
    # the easiest thing in the world to "round-trip" by accident -- a build
    # that dropped it entirely would still pass.  A nonzero nifa2d is legal
    # state (WRF's Registry declares QNIFA2D in the restart stream precisely
    # because a WIF-ingest run has one), so seed one and require it back.
    state.nifa2d[...] = cp.asarray(
        rng.random(state.nifa2d.shape, dtype=np.float32) * np.float32(9.0))
    cp.cuda.Stream.null.synchronize()

    # The profile really is loaded, so this is not a zeros-round-trip.
    assert float(cp.max(state.nwfa)) > 1.0e7
    assert float(cp.max(state.nifa)) > 1.0e5
    assert float(cp.max(state.nwfa2d)) > 0.0

    saved = {name: cp.asnumpy(getattr(state, name)).copy()
             for name in MP28_AEROSOL_FIELDS}

    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "mp28-restart.npz"
        restart.write_restart(path, state, cfg)

        header = restart.read_restart_header(path)
        assert header["physics_setup"]["algorithms"]["microphysics"] == \
            restart.MICROPHYSICS_ALGORITHM_IDENTITIES[28]

        with np.load(path, allow_pickle=False) as stored:
            present = set(stored.files)
        for name in MP28_AEROSOL_FIELDS:
            assert f"state/{name}" in present, name

        restored, _forcing2, _cfg2 = _mp28_forecast_state(cp)
        for name in MP28_AEROSOL_FIELDS:
            getattr(restored, name)[...] = 0
        cp.cuda.Stream.null.synchronize()
        for name in MP28_AEROSOL_FIELDS:
            assert not np.array_equal(
                cp.asnumpy(getattr(restored, name)), saved[name]), name

        restart.restore_restart(path, restored, cfg)
        cp.cuda.Stream.null.synchronize()

    for name in MP28_AEROSOL_FIELDS:
        got = cp.asnumpy(getattr(restored, name))
        assert got.dtype == saved[name].dtype == np.float32, name
        assert got.tobytes() == saved[name].tobytes(), (
            f"state/{name} did not round-trip bitwise through the restart "
            f"file (max |delta| = {np.abs(got - saved[name]).max()!r})")


@requires_gpu
def test_an_mp28_restart_refuses_to_write_without_the_aerosol_state():
    """The guard that makes the identity row safe to add.

    An identity string alone would let a checkpoint proceed while dropping
    the aerosol state, and the resumed run would stay finite and bounded --
    WRF's terminal clamps (module_mp_thompson.F:3976-3982) hold nwfa/nifa at
    their floors and nc at 2/rho -- so nothing downstream would notice.  That
    is strictly worse than the refusal it replaced, which is why the write
    side fails closed on each of the five fields.
    """
    import cupy as cp

    from woof.io import restart

    state, _forcing, cfg = _mp28_forecast_state(cp)
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "ok.npz"
        restart.write_restart(path, state, cfg)   # baseline: it works
        for name in MP28_AEROSOL_FIELDS:
            held = getattr(state, name)
            setattr(state, name, None)
            try:
                with pytest.raises(restart.RestartManifestError) as excinfo:
                    restart.write_restart(
                        pathlib.Path(tmp) / f"missing-{name}.npz", state, cfg)
                assert f"state/{name}" in str(excinfo.value), name
            finally:
                setattr(state, name, held)


@requires_gpu
def test_an_mp28_restart_refuses_a_file_that_omits_the_aerosol_state():
    """The read side of the same guard.

    The generic inventory check compares the FILE against the RESUMING
    state, so two equally aerosol-less endpoints agree with each other.  The
    mp=28 check compares the file against the canonical inventory instead,
    which is what makes a truncated checkpoint a refusal rather than a
    silently aerosol-inert resume.
    """
    import cupy as cp

    from woof.io import restart

    state, _forcing, cfg = _mp28_forecast_state(cp)
    header_key = restart._HEADER_KEY
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "full.npz"
        restart.write_restart(path, state, cfg)
        with np.load(path, allow_pickle=False) as stored:
            payload = {key: stored[key] for key in stored.files}
        header = json.loads(
            bytes(bytearray(payload[header_key])).decode("utf-8"))

        for dropped in MP28_AEROSOL_FIELDS:
            # Drop the ARRAY and its manifest entry, i.e. exactly what a
            # build whose DomainState never allocated the field would
            # write.  Dropping only the array would trip the generic
            # array-manifest consistency check first and prove nothing
            # about the mp=28 inventory guard.
            wounded = json.loads(json.dumps(header))
            wounded["array_manifest"].pop(f"state/{dropped}")
            truncated = pathlib.Path(tmp) / f"drop-{dropped}.npz"
            reduced = {k: v for k, v in payload.items()
                       if k != f"state/{dropped}"}
            reduced[header_key] = np.frombuffer(
                json.dumps(wounded).encode("utf-8"), dtype=np.uint8)
            np.savez(truncated, **reduced)

            with pytest.raises(restart.RestartMismatchError) as excinfo:
                restart.restore_restart(truncated, state, cfg)
            message = str(excinfo.value)
            assert f"state/{dropped}" in message, dropped
            assert "aerosol" in message, (dropped, message)


@requires_gpu
def test_the_mp28_restart_identity_fails_closed_on_the_ccn_table(tmp_path):
    """It must refuse to WRITE an identity it cannot substantiate.

    ``CCN_ACTIVATE.BIN`` ships now, but ``WOOF_THOMPSON_CCN_ACTIVATE``
    still lets a run bind to another copy, so "the operator pointed it
    somewhere else" is a normal state of the world, not a corrupted install.
    A header that quietly omitted the aerosol table binding in that case
    would be the worst outcome: the checkpoint would look complete and would
    resume against anything.  Both failure modes -- absent and
    byte-different -- are therefore RestartManifestError at write time.
    """
    import cupy as cp

    from woof.io import restart

    state, _forcing, cfg = _mp28_forecast_state(cp)
    from woof.core.thompson_aerosol_contract import (
        AEROSOL_TABLE_PATH_ENV, resolve_aerosol_table_root,
        resolve_ccn_activation_path)

    good = resolve_ccn_activation_path(None, resolve_aerosol_table_root(None))

    monkey = pytest.MonkeyPatch()
    try:
        monkey.setenv(AEROSOL_TABLE_PATH_ENV, str(tmp_path / "absent.bin"))
        with pytest.raises(restart.RestartManifestError, match="CCN"):
            restart.physics_setup_identity(state, cfg)
    finally:
        monkey.undo()

    tampered = tmp_path / "CCN_ACTIVATE.BIN"
    payload = bytearray(good.read_bytes())
    payload[100] ^= 0xFF          # one flipped bit, correct byte count
    tampered.write_bytes(bytes(payload))
    assert tampered.stat().st_size == good.stat().st_size

    monkey = pytest.MonkeyPatch()
    try:
        monkey.setenv(AEROSOL_TABLE_PATH_ENV, str(tampered))
        with pytest.raises(restart.RestartManifestError, match="SHA-256"):
            restart.physics_setup_identity(state, cfg)
    finally:
        monkey.undo()

    # ...and the unmodified environment still succeeds, so the two refusals
    # above are about the asset and not about the code path.
    restart.physics_setup_identity(state, cfg)


@requires_gpu
def test_an_mp28_restart_refuses_a_different_ccn_activation_table():
    """The table identity is ESSENTIAL, not decorative.

    Writing the digests into the header only helps if the reader compares
    them.  This mutates the stored CCN activation SHA-256, recomputes the
    header's own self-consistency fingerprint so that check still passes,
    and requires the restore to refuse on the LIVE-versus-STORED comparison.

    That is the failure this record exists for: the droplet number mp=28
    predicts is READ out of ``tnccn_act`` at a fixed radius/kappa index
    (module_mp_thompson.F:5229-5230), so resuming a checkpoint against a
    different activation table is resuming a different model.
    """
    import cupy as cp

    from woof.io import restart

    state, _forcing, cfg = _mp28_forecast_state(cp)
    header_key = restart._HEADER_KEY
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "full.npz"
        restart.write_restart(path, state, cfg)
        with np.load(path, allow_pickle=False) as stored:
            payload = {key: stored[key] for key in stored.files}
        header = json.loads(
            bytes(bytearray(payload[header_key])).decode("utf-8"))

        setup = header["physics_setup"]
        record = setup["microphysics"]["thompson_aerosol"]
        original = record["aerosol_tables"]["assets"][0]["sha256"]
        record["aerosol_tables"]["assets"][0]["sha256"] = "0" * 64
        assert original != "0" * 64
        header["physics_setup_fingerprint"] = restart._json_sha256(setup)

        forged = pathlib.Path(tmp) / "forged.npz"
        payload[header_key] = np.frombuffer(
            json.dumps(header).encode("utf-8"), dtype=np.uint8)
        np.savez(forged, **payload)

        with pytest.raises(restart.RestartMismatchError) as excinfo:
            restart.restore_restart(forged, state, cfg)
        assert "different physics setup" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 3. Legacy RRTMG.
# ---------------------------------------------------------------------------

def test_legacy_rrtmg_declares_mp28_radii_on_wrfs_own_scheme_table():
    """``legacy_scheme_declares_radii(28, *)`` used to raise.

    That made ``mp_physics=28 + ra_rrtmg_variant='rrtmg_legacy'`` fail
    closed with a NotImplementedError telling the operator to extend a
    table -- a refusal WRF itself does not make.  WRF v4.6.1 lists
    THOMPSONAERO explicitly and separately from THOMPSON inside the same
    ``use_mp_re`` disjunction (phys/module_physics_init.F:1005 and :1006),
    and the P3/Jensen-Ishmael ``has_reqs = 0`` override at :1027-1033 does
    not touch it, so all three of has_reqc/has_reqi/has_reqs are 1.
    """
    from woof.core.rrtmg_legacy import (_LEGACY_ICE_ACTIVE_MICROPHYSICS,
                                         _MP_DECLARES_RADII,
                                         legacy_ice_active,
                                         legacy_scheme_declares_radii)

    assert _MP_DECLARES_RADII[28] is True
    assert legacy_scheme_declares_radii(28, 1) is True
    # use_mp_re = 0 still switches the whole table off, exactly as for mp=8.
    assert legacy_scheme_declares_radii(28, 0) is False
    assert legacy_scheme_declares_radii(8, 1) is True

    # cal_cldfra1's F_QI/F_QS: Registry.EM_COMMON:3036 gives thompsonaero
    # the identical ``moist:qv,qc,qr,qi,qs,qg`` inventory :3024 gives
    # thompson, so ice is active for 28 exactly as it is for 8.
    assert legacy_ice_active(28) is True
    assert 28 in _LEGACY_ICE_ACTIVE_MICROPHYSICS
    # Morrison stays out of BOTH tables for its own recorded reasons.
    assert legacy_scheme_declares_radii(10, 1) is False
    assert legacy_ice_active(10) is True


@pytest.mark.parametrize("stock_tree", [
    # Machine-local staging paths are deliberately not part of the
    # contract (PROVENANCE.md): name the tree, let the machine say where.
    pathlib.Path(os.environ.get("WOOF_WRF461_STOCK_TREE",
                                "wrf-stock-v461-gate-20260721")),
])
def test_the_use_mp_re_citation_is_checkable_against_stock_wrf(stock_tree):
    """The citation, verified rather than asserted -- when the tree is here.

    A line-numbered claim about a reference tree that nobody re-reads is a
    claim about the past.  This one is cheap to re-check, so it is checked.
    """
    source = stock_tree / "phys" / "module_physics_init.F"
    if not source.is_file():                                # pragma: no cover
        pytest.skip(f"stock WRF v4.6.1 tree not present at {stock_tree}")
    lines = source.read_text(encoding="utf-8",
                             errors="replace").splitlines()
    assert "THOMPSON" in lines[1004] and "THOMPSONAERO" not in lines[1004]
    assert "THOMPSONAERO" in lines[1005]
    # Both are inside the use_mp_re block that sets has_reqc/i/s.
    block = "\n".join(lines[986:1024])
    assert "use_mp_re" in "\n".join(lines[986:989])
    assert "has_reqc = 1" in block and "has_reqs = 1" in block


# ---------------------------------------------------------------------------
# 4. The N5S wrfinput inventory.
# ---------------------------------------------------------------------------

def _n5s_cfg(mp_physics):
    from types import SimpleNamespace

    return SimpleNamespace(moist=True, mp_physics=mp_physics)


def test_the_n5s_wrfinput_inventory_has_an_mp28_arm():
    """A 28 arm requiring the Thompson moments plus QNCLOUD/QNWFA/QNIFA.

    All three are declared scalars in WRF's Registry with the wrfinput
    stream in their IO strings (Registry.EM_COMMON:542 for QNCLOUD,
    registry.new3d_wif:87-90 for QNWFA/QNIFA), so real.exe writes them.
    QNCLOUD is REQUIRED here rather than optional: it is optional only for
    Morrison, which diagnoses cloud number, whereas for mp=28 it is the
    prognostic droplet number the entire scheme turns on.
    """
    from woof.verify.cases import real74_n5s as n5s

    required, allowed = n5s._active_moisture_inventory(_n5s_cfg(28))
    assert required == frozenset({
        "QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP",
        "QNRAIN", "QNICE", "QNCLOUD", "QNWFA", "QNIFA"})
    assert allowed == required

    mapping = n5s._active_moisture_map(_n5s_cfg(28))
    assert mapping["QNCLOUD"] == "nc"
    assert mapping["QNWFA"] == "nwfa"
    assert mapping["QNIFA"] == "nifa"

    # mp=8 must be untouched: no aerosol names may leak into its inventory.
    mp8_required, mp8_allowed = n5s._active_moisture_inventory(_n5s_cfg(8))
    assert mp8_required == frozenset({
        "QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP",
        "QNRAIN", "QNICE"})
    assert not ({"QNWFA", "QNIFA", "QNCLOUD"} & mp8_allowed)
    assert "QNWFA" not in n5s._active_moisture_map(_n5s_cfg(8))

    # Morrison keeps its documented OPTIONAL QNCLOUD.
    mp10_required, mp10_allowed = n5s._active_moisture_inventory(
        _n5s_cfg(10))
    assert "QNCLOUD" not in mp10_required
    assert "QNCLOUD" in mp10_allowed


def test_an_mp28_wrfinput_without_qncloud_is_refused():
    """The Morrison optional-QNCLOUD exemption must not extend to mp=28.

    ``_restore_active_moisture`` tolerates a missing wrfinput field only if
    it is in ``MORRISON_OPTIONAL_MOISTURE_WRFINPUT``.  QNCLOUD is in that
    tuple, so before the mp=28 arm existed an aerosol-aware wrfinput with no
    QNCLOUD would have restored quietly and started every column at nc = 0 --
    which WRF's terminal clamp (module_mp_thompson.F:3976) then holds at
    2/rho for the whole run.
    """
    from types import SimpleNamespace

    from woof.verify.cases import real74_n5s as n5s

    shape = (2, 3, 4)
    raw = {name: np.zeros(shape, dtype=np.float32) for name in
           ("QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP",
            "QNRAIN", "QNICE", "QNWFA", "QNIFA")}
    state = SimpleNamespace(**{
        name: np.zeros(shape, dtype=np.float32) for name in
        ("qv", "qc", "qr", "qi", "qs", "qg", "nr", "ni", "nc",
         "nwfa", "nifa")})

    with pytest.raises(ValueError, match="QNCLOUD"):
        n5s._restore_active_moisture(state, raw, _n5s_cfg(28), np)

    # With it present the same restore succeeds and lands on ``nc``.
    raw["QNCLOUD"] = np.full(shape, 1.25e8, dtype=np.float32)
    n5s._restore_active_moisture(state, raw, _n5s_cfg(28), np)
    np.testing.assert_array_equal(state.nc, raw["QNCLOUD"])
    np.testing.assert_array_equal(state.nwfa, raw["QNWFA"])


# ---------------------------------------------------------------------------
# 5. Packaging.
# ---------------------------------------------------------------------------

# argv: <project root> <distribution name> <top-level package directory>.
# Parameterised because the Thompson tables this file measures moved into
# the `recast-woof-data` companion distribution in 2.5.0 -- same repository, a
# second pyproject.toml -- and a probe hard-wired to the root project
# would have reported "not shipped" for files that ship perfectly well,
# one distribution over.
_WHEEL_PROBE = r"""
import json, os, sys, tomllib
from pathlib import Path
repo = Path(sys.argv[1])
name = sys.argv[2]
top = sys.argv[3]
os.chdir(repo)
import setuptools
from setuptools.command.build_py import build_py
with (repo / "pyproject.toml").open("rb") as stream:
    tools = tomllib.load(stream)["tool"]["setuptools"]
skip = {"__pycache__", ".pytest_cache"}
packages = []
for dirpath, dirnames, filenames in os.walk(repo / top):
    dirnames[:] = sorted(d for d in dirnames if d not in skip)
    if "__init__.py" in filenames:
        packages.append(".".join(Path(dirpath).relative_to(repo).parts))
distribution = setuptools.dist.Distribution({
    "name": name, "packages": packages,
    "package_data": tools["package-data"],
    "exclude_package_data": tools.get("exclude-package-data", {}),
})
command = build_py(distribution)
command.finalize_options()
shipped = set()
for _package, src_dir, _build_dir, names in (
        command.get_data_files_without_manifest()):
    for name in names:
        shipped.add(Path(src_dir, name).as_posix())
print(json.dumps(sorted(shipped)))
"""


def _interpreter_with_setuptools() -> str | None:
    candidates = [sys.executable]
    for name in ("python3", "python3.12", "python3.11", "python3.13"):
        found = shutil.which(name)
        if found:
            candidates.append(found)
    candidates.extend(("/usr/bin/python3", "/usr/bin/python3.12"))
    for candidate in dict.fromkeys(candidates):
        if not candidate or not os.path.exists(candidate):
            continue
        probe = subprocess.run(
            [candidate, "-c", "import setuptools, tomllib"],
            capture_output=True)
        if probe.returncode == 0:
            return candidate
    return None


def test_the_ccn_activation_blob_is_present_in_the_built_package_data():
    """DRY-RUN BUILD, not a declaration read.

    ``CCN_ACTIVATE.BIN`` is third-party parcel-model output (WRF's own
    comment at phys/module_mp_thompson.F:5102-5108) that this repository now
    redistributes -- ``AEROSOL_ASSET_REDISTRIBUTED = True``, reversed on
    2026-08-01 once WRF's public-domain dedication was checked against the
    committed blob.  Until then an ``[tool.setuptools.exclude-package-data]``
    entry kept it out of every wheel, so the measurement here is inverted
    rather than deleted: a leftover exclusion, or a reinstated .gitignore
    line, would publish an mp=28 that fails closed on a missing activation
    table for every user.

    The sibling declaration gate reads pyproject with ``tomllib`` and cannot
    tell a typo from an entry; the sibling measurement gate uses setuptools
    and SKIPS in this virtualenv, which has none.  This one finds an
    interpreter that does and measures in a subprocess, so the control is
    live in the environment the suite actually runs in.
    """
    # The COMPANION distribution's wheel since 2.5.0: woof/data/thompson/
    # tables moved to recast-woof-data/woof_data/data/thompson/tables when the
    # woof wheel measured 103.62 MiB against PyPI's 100 MiB cap.  The
    # claim under test is unchanged -- woof redistributes WRF's
    # CCN_ACTIVATE.BIN and a user's install must contain it -- so the probe
    # follows the file to the distribution that now carries it.
    project = REPO / "recast-woof-data"
    relative = "woof_data/data/thompson/tables/CCN_ACTIVATE.BIN"
    interpreter = _interpreter_with_setuptools()
    if interpreter is None:                                 # pragma: no cover
        pytest.skip("no interpreter with setuptools + tomllib is available")

    # encoding named: text=True alone decodes the child's stdout with the
    # host locale (cp1252 on Windows), and this stdout is JSON, which is
    # UTF-8 by specification whatever the console codepage is.
    result = subprocess.run(
        [interpreter, "-c", _WHEEL_PROBE, str(project),
         "recast-woof-data", "woof_data"],
        capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 0, result.stderr
    shipped = set(json.loads(result.stdout))

    assert relative in shipped, (
        "a wheel built from this tree would NOT contain "
        f"{relative}, which woof declares it redistributes; a stale "
        "[tool.setuptools.exclude-package-data] entry is the usual cause")
    # ...and nothing else moved.  The other Thompson tables that DO ship
    # must still ship, or this change silently broke mp=8.
    for still_shipped in (
            "woof_data/data/thompson/tables/qr_acr_qsV2.dat",
            "woof_data/data/thompson/tables/thompson_aux_tables.dat",
            "woof_data/data/thompson/tables/MANIFEST.sha256"):
        assert still_shipped in shipped, still_shipped
    # The two size-externalized tables stay out, as before.
    for externalized in ("woof_data/data/thompson/tables/freezeH2O.dat",
                         "woof_data/data/thompson/tables/qr_acr_qg_V4.dat"):
        assert externalized not in shipped, externalized

    # The wheel probe globs the FILESYSTEM, so a shipped-but-absent file
    # would pass vacuously.  Say so rather than assume.
    assert (project / relative).is_file(), (
        f"{relative} is missing from this tree, so this run measured an "
        "inclusion against an absent file")


def test_the_offline_child_lane_converts_every_ported_edge():
    """mp=28 is IN the lane for same-scheme AND for every mixed edge.

    Recorded here as one assertion rather than left implicit, because the
    failure mode this whole package exists to remove is a scheme that is
    half-supported by accident: the offline conversion is the online
    nest edge's own contract and kernel, so the admitted cross-scheme set
    is derived from PORTED_MP_PHYSICS.
    """
    from woof.offline_child import (OFFLINE_CHILD_MP_PHYSICS,
                                     PARENT_SCHEME_CONTRACT,
                                     offline_cross_scheme_refusal)
    from woof.offline_child_run import _CAPABILITIES
    from woof.core import microphysics_transition as mt

    assert 28 in OFFLINE_CHILD_MP_PHYSICS
    assert _CAPABILITIES["same_scheme_mp_physics"] == [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 13, 14, 15, 16, 18, 24, 26, 28, 40, 50, 51, 53, 55, 56, 97, 106]
    assert _CAPABILITIES["cross_scheme_transitions"]["mp_physics"] == sorted(
        PARENT_SCHEME_CONTRACT)
    assert PARENT_SCHEME_CONTRACT == (
        OFFLINE_CHILD_MP_PHYSICS & frozenset(mt.PORTED_MP_PHYSICS))
    assert {16, 28, 50} <= PARENT_SCHEME_CONTRACT
    for other in sorted(PARENT_SCHEME_CONTRACT - {28}):
        assert offline_cross_scheme_refusal(28, other) is None
        assert offline_cross_scheme_refusal(other, 28) is None


# ---------------------------------------------------------------------------
# 6. The offline-child lane, end to end.
# ---------------------------------------------------------------------------

@requires_gpu
def test_an_offline_mp28_child_keeps_its_inherited_surface_emission(tmp_path):
    """The interaction the two halves of this change had to agree on.

    ``thompson_init`` fills ``nwfa2d`` at module_mp_thompson.F:510 ONLY
    inside the "no initial CCN" branch (:493).  An offline child that
    inherits a parent's nonzero ``nwfa`` takes the ``has_CCN = .TRUE.``
    branch at :516-522 instead, which fills NOTHING -- so if the child's
    surface aerosol emission did not arrive by interpolation from the
    parent, the child would run its whole forecast at zero emission and
    ``microphysics_init`` would report success.

    This is the check that the two halves line up: the offline lane SINTs
    QNWFA2D from the parent, and the profile fill then declines to touch it.
    """
    import cupy as cp

    sys.path.insert(0, str(REPO / "tests"))
    from test_offline_child import (  # noqa: E402
        _history, _mp28_child_config, _mp28_placement, _physics_binding)

    from datetime import datetime

    from woof.core import microphysics
    from woof.offline_child import (build_offline_child_domain_state,
                                     interpolate_parent_initial_state)

    parent = tmp_path / "parent-mp28.nc"
    _history(parent, datetime(1974, 4, 3, 12), mp=28, ny=18, nx=20,
             signal=0.5)
    initial = interpolate_parent_initial_state(
        parent, _mp28_placement(),
        physics_binding=_physics_binding(tmp_path, mp=28), backend="cpu")
    child = build_offline_child_domain_state(
        initial, _mp28_child_config(), array_module=cp)
    cp.cuda.Stream.null.synchronize()

    assert float(cp.max(child.nwfa2d)) == pytest.approx(4321.0)
    inherited_nwfa2d = cp.asnumpy(child.nwfa2d).copy()
    inherited_nwfa = cp.asnumpy(child.nwfa).copy()

    receipt = microphysics.microphysics_init(child, _mp28_child_config())
    cp.cuda.Stream.null.synchronize()

    profile = receipt["thompson_aerosol_profile"]
    assert profile["ccn"] is False, (
        "the child inherited a nonzero CCN field, so WRF's has_CCN branch "
        "applies and thompson_init must NOT refill the profile")
    assert profile["in"] is False
    np.testing.assert_array_equal(cp.asnumpy(child.nwfa2d),
                                  inherited_nwfa2d)
    np.testing.assert_array_equal(cp.asnumpy(child.nwfa), inherited_nwfa)


@requires_gpu
def test_an_offline_mp28_child_can_write_its_final_checkpoint(tmp_path):
    """``woof downscale`` ends by writing ``gpuwmrst_dNN_<instant>.npz``.

    Before the identity row existed that raised ``RestartManifestError``
    AFTER the whole child forecast had run, i.e. the run's own evidence was
    destroyed at the last step.  It also ties the two halves of this
    package together: the write now REQUIRES nwfa2d/nifa2d to be present,
    which is exactly what the offline lane had to start interpolating.
    """
    import cupy as cp

    sys.path.insert(0, str(REPO / "tests"))
    from test_offline_child import (  # noqa: E402
        _history, _mp28_child_config, _mp28_placement, _physics_binding)

    from datetime import datetime

    from woof.io import restart
    from woof.offline_child import (build_offline_child_domain_state,
                                     interpolate_parent_initial_state)

    parent = tmp_path / "parent-mp28.nc"
    _history(parent, datetime(1974, 4, 3, 12), mp=28, ny=18, nx=20)
    initial = interpolate_parent_initial_state(
        parent, _mp28_placement(),
        physics_binding=_physics_binding(tmp_path, mp=28), backend="cpu")
    cfg = _mp28_child_config()
    child = build_offline_child_domain_state(initial, cfg, array_module=cp)
    cp.cuda.Stream.null.synchronize()

    path = tmp_path / "gpuwmrst_d01_1974-04-03_12_00_00.npz"
    restart.write_restart(path, child, cfg)
    assert path.is_file()
    header = restart.read_restart_header(path)
    assert header["physics_setup"]["algorithms"]["microphysics"] == \
        restart.MICROPHYSICS_ALGORITHM_IDENTITIES[28]
    with np.load(path, allow_pickle=False) as stored:
        for name in MP28_AEROSOL_FIELDS:
            assert f"state/{name}" in stored.files, name
