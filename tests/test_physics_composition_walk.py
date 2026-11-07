"""Is woof's physics arbitrary?  Measured, not argued.

The claim under test is that a user composes microphysics, PBL, surface
layer, land surface, radiation, cumulus and turbulence freely -- that there
is no engine lock quietly pinning a run to one of the shipped named suites.
It has been asserted in review threads more than once and believed by
nobody, including the people who wrote it, because the evidence for it was
always "read this file".

So it is measured.  ``tools/report_physics_composition_walk.py`` writes
the declared physics combinations into real experiment TOMLs and pushes every one
through :func:`woof.experiment.build_experiment` -- the single front door
``woof run``, ``woof go``, ``woof check``, both prepared runners and the
DA drivers reach a per-domain ``RunConfig`` through -- and records what
that call did.  This file pins the properties of the result:

* every admitted value of every axis appears in an ACCEPTED run, so no axis
  is decorative;
* the shipped presets are a tiny corner of what is admitted, so the presets
  are not the space;
* every ACCEPTED run kept every switch the file set, so an admission is
  never a silent rewrite;
* every refusal names why, and DOING WHAT THE REFUSAL SAYS produces an
  accepted run -- the property that caught the one defect this walk found
  (the ``km_opt=2`` message used to recommend ``km_opt=3``, which is
  refused for the same reason three lines above it);
* the axis tables are what the loader actually admits, measured by
  scanning integers at it rather than transcribed from reading
  ``woof/config.py``;
* and MYNN -- the suite family that started this -- composes with every
  land surface and every radiation pairing the loader admits at all.

Two controls keep the whole thing from being vacuous: the walk's window is
checked to contain no local night (or the 1.7.1 nocturnal guard would be
answering a different question), and a mutation control shows the harness
can still produce a refusal when one is warranted.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import report_physics_composition_walk as walk

MODEL = Path(__file__).resolve().parents[1]
RECEIPT = (
    MODEL / "docs" / "public" / "receipts" / "physics-composition-walk.json"
)


@pytest.fixture(scope="module")
def receipt() -> dict:
    return json.loads(RECEIPT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def regenerated() -> dict:
    """One complete loader walk, shared by every test in the file."""

    return walk.evaluate()


# ---------------------------------------------------------------------------
# The receipt is the measurement, and it is current.
# ---------------------------------------------------------------------------

def test_the_receipt_regenerates_byte_for_byte(regenerated) -> None:
    assert walk.render(regenerated) == RECEIPT.read_bytes(), (
        "docs/public/receipts/physics-composition-walk.json no longer "
        "matches what the loader does; regenerate it with "
        "tools/report_physics_composition_walk.py and land both together")


# ---------------------------------------------------------------------------
# Controls first: an instrument that cannot fail measures nothing.
# ---------------------------------------------------------------------------

def test_the_walk_window_contains_no_local_night(receipt) -> None:
    """The controlled variable, checked with the shipped scanner.

    ``build_experiment`` refuses an undeclared longwave-off/shortwave-on
    suite for any window containing local night.  That guard is correct and
    is pinned elsewhere; here it must stay OUT of the measurement, or a
    time-of-day refusal would be counted as a composition refusal.
    """

    assert receipt["harness"]["window_contains_local_night"] is False


def test_the_walk_grid_is_inside_every_shipped_vertical_bound() -> None:
    """The other harness control, stated as the bounds and not the number.

    An earlier draft ran nz=8 and reported cu_physics=3 as never admitted.
    That was the harness: Grell-Freitas declares a 12-level floor.  Pinning
    the INTERSECTION rather than the literal 40 means a bound that moves
    fails here instead of quietly re-introducing the artefact.
    """

    from woof import physics_vertical_contract as bounds
    from woof.config import SASE_MAX_NZ

    nz = walk.GRID["nz"]
    named = {
        name: getattr(bounds, name) for name in dir(bounds)
        if name.endswith("_VERTICAL_LEVEL_BOUNDS")
    }
    assert named, "no vertical bounds found to check the walk grid against"
    for name, (low, high) in named.items():
        assert low is None or nz >= low, f"{name} floor {low} > nz {nz}"
        assert high is None or nz <= high, f"{name} ceiling {high} < nz {nz}"
    assert nz <= SASE_MAX_NZ


def test_a_mutation_control_still_produces_a_refusal() -> None:
    """The harness can still refuse; it is not accepting everything.

    A value no schema admits must come back REFUSED with the axis named.
    Without this, "746 accepted" could mean the walk stopped checking.
    """

    combination = dict(walk.ANCHORS["ysu-mm5-noah-rrtmgp"], **walk.TIER_A_HELD)
    combination["mp_physics"] = 37
    outcome = walk.attempt(combination)
    assert outcome.verdict == "REFUSED"
    assert "mp_physics must be" in outcome.message


def test_every_anchor_suite_is_itself_accepted(receipt) -> None:
    """Tiers B, C and D are only coverage if their backgrounds load."""

    for name, row in receipt["anchors"].items():
        assert row["verdict"] == "ACCEPTED", (
            f"anchor {name} is refused, so every tier held against it "
            f"measured nothing: {row['message']}")


# ---------------------------------------------------------------------------
# The headline: the space is large, and it is not the preset list.
# ---------------------------------------------------------------------------

def test_the_walk_covers_what_it_says_it_covers(receipt) -> None:
    totals = receipt["totals"]
    assert totals["tried"] == totals["accepted"] + totals["refused"]
    assert totals["tried"] >= 6000, (
        "the walk shrank; a composition claim this size needs the space "
        "actually walked")
    tiers = receipt["tiers"]
    # Tier A is the full cartesian over the six coupled axes.  Spelled out
    # from the axis tables so a widened axis cannot silently shrink it.
    expected = 1
    for name in ("bl_pbl_physics", "sf_sfclay_physics", "sf_surface_physics",
                 "ra_lw_physics", "ra_sw_physics", "km_opt"):
        expected *= len(walk.AXIS_VALUES[name])
    assert tiers["A-nexus-cartesian"]["tried"] == expected


#: Admitted values held at the forecast door by a named, measured guard.
#: RCON-SPEED-01 (woof/config.py): mp_physics=29 is ported and schema-legal,
#: but its whole-forecast microphysics stage costs 1.885x MP28, outside the
#: 1.2x bar; the entry goes when that guard is retired.
HELD_AT_THE_DOOR = {"mp_physics=29": "RCON-SPEED-01"}


def test_every_admitted_value_of_every_axis_reaches_an_accepted_run(
        receipt) -> None:
    """No axis value is decorative; the only exceptions are named holds.

    ``ra_lw_physics=1`` used to be the one schema-legal, deliberately
    unexecutable value in the whole space: the WRF RRTM longwave kernels
    were not ported, so the schema advertised a choice the engine did not
    offer.  The 1.9 port makes it executable, so the exception set is
    EMPTY and every admitted value of every axis reaches an accepted run.

    The assertion is kept as an equality against the empty set rather than
    deleted.  An empty exception set is the strong claim, and a future
    scheme that lands schema-legal but unrunnable has to come back here
    and say so out loud.
    """

    unreachable = {
        f"{axis}={value}"
        for axis, values in receipt["per_axis"].items()
        for value, counts in values.items()
        if counts["accepted"] == 0
    }
    assert unreachable == set(HELD_AT_THE_DOOR), (
        "the set of admitted-but-unreachable selector values changed: "
        f"{sorted(unreachable)}")
    # Each held value says so out loud, citing the guard that holds it.
    for value, guard in HELD_AT_THE_DOOR.items():
        axis, number = value.split("=")
        held = walk.attempt({**walk.ANCHORS["ysu-mm5-noah-rrtmgp"],
                             **walk.TIER_A_HELD, axis: int(number)})
        assert held.verdict == "REFUSED", value
        assert "speed proof is open" in held.message, (guard, held.message)
    # Non-vacuous: the value that used to be the exception now runs, and
    # the pairing the old refusal named as unreachable is accepted.
    combination = dict(walk.ANCHORS["ysu-mm5-noah-rrtmgp"], **walk.TIER_A_HELD)
    combination.update(ra_lw_physics=1, ra_sw_physics=1)
    outcome = walk.attempt(combination)
    assert outcome.verdict == "ACCEPTED", outcome.message


def test_the_shipped_presets_are_a_corner_of_the_admitted_space(
        receipt) -> None:
    """The point of the whole exercise, as a number.

    If the accepted space were the preset list, "compose your own physics"
    would be marketing.  It is not: the walk accepts hundreds of distinct
    suites, and it walked a deliberately narrow slice of the full cartesian.
    """

    from woof.physics_compat import SINGLE_DOMAIN_PHYSICS_PROFILES

    accepted = receipt["totals"]["accepted"]
    presets = len(SINGLE_DOMAIN_PHYSICS_PROFILES)
    assert accepted > 10 * presets, (
        f"only {accepted} accepted suites against {presets} shipped "
        "presets; the composition claim no longer has room to be true")
    assert receipt["totals"]["full_eight_axis_cartesian"] > accepted


def test_an_accepted_run_never_rewrites_a_switch(receipt) -> None:
    """ACCEPTED has to mean the file got what it asked for.

    A loader that admits a configuration and then substitutes a switch
    would pass every count above while still being an engine lock.  The
    walk compares each accepted run's resolved per-domain RunConfig against
    the eight selectors the file set.
    """

    assert receipt["switch_rewrites"] == []
    assert receipt["totals"]["accepted_with_a_rewritten_switch"] == 0


# ---------------------------------------------------------------------------
# Refusals: every one names why, and its own advice works.
# ---------------------------------------------------------------------------

def test_every_refusal_names_the_selector_it_is_about(receipt) -> None:
    """A refusal that does not name a switch cannot be acted on."""

    vocabulary = set(walk.AXIS_NAMES) | {
        "num_soil_layers", "km_opt_zero_acknowledgement", "moist",
        "cudt_minutes", "nz"}
    for row in receipt["refusal_rules"]:
        for message in row["messages"]:
            named = {word for word in vocabulary if word in message["message"]}
            assert named, (
                "this refusal names no selector, so a user cannot tell what "
                f"to change: {message['message']!r}")


def test_every_refusal_rule_has_a_remedy_and_the_remedy_works(
        receipt) -> None:
    """The property that found the defect, kept as a standing gate.

    A refusal is only as good as the next thing it tells you to do.  Every
    distinct rule the walk produces carries an explicit before/after pair
    in the tool, and the after is built by following the message.  A new
    refusal cannot land without someone demonstrating that its advice
    reaches an accepted run.
    """

    remedies = {row["id"]: row for row in receipt["remedy_follow_through"]}
    for row in remedies.values():
        assert row["before"]["verdict"] == "REFUSED", row["id"]
        assert row["after"]["verdict"] == "ACCEPTED", (
            f"{row['id']}: doing what the refusal says still fails -- "
            f"{row['after']['message']}")
        assert row["remedy_works"], row["id"]

    covered = {row["before"]["rule"] for row in remedies.values()}
    observed = {row["rule"] for row in receipt["refusal_rules"]}
    assert observed - covered == set(), (
        "these refusal rules have no remedy pair, so nobody has shown that "
        f"their advice works: {sorted(observed - covered)}")
    assert covered - observed == set(), (
        "these remedy pairs no longer correspond to a rule the walk "
        f"produces and are dead weight: {sorted(covered - observed)}")


# WRF Fu-Liou-Gu dispatch has no one-spectrum-off CASE for these pairs.
# Keep this independent, literal contract: other mixed pairs still accept.
RADIATION_PAIR_REFUSALS = {
    (0, 7): "WRF radiation_driver has no ra_lw_physics=0 CASE for ra_sw_physics=7; select an implemented ra_lw_physics.",
    (7, 0): "WRF radiation_driver rejects ra_lw_physics=7 with ra_sw_physics=0: select an active ra_sw_physics (native CASE 0).",
}


def test_independent_radiation_spectra_preserve_every_selected_pair() -> None:
    """The retired pairing refusals must not return or rewrite a spectrum.

    Exercise the real loader across every pair, including mixed 4/0,
    4/90 and 1/4. The two native Fu-Liou-Gu dispatch exclusions must name
    their selectors; every other pair must accept without a rewrite.
    """

    base = dict(walk.ANCHORS["ysu-mm5-noah-rrtmgp"], **walk.TIER_A_HELD)
    for lw in walk.AXIS_VALUES["ra_lw_physics"]:
        for sw in walk.AXIS_VALUES["ra_sw_physics"]:
            outcome = walk.attempt(dict(base, ra_lw_physics=lw,
                                        ra_sw_physics=sw))
            if (lw, sw) in RADIATION_PAIR_REFUSALS:
                assert outcome.verdict == "REFUSED", (lw, sw, outcome)
                assert outcome.message == RADIATION_PAIR_REFUSALS[lw, sw]
            else:
                assert outcome.verdict == "ACCEPTED", (lw, sw, outcome.message)
                assert outcome.rewritten == (), (lw, sw, outcome.rewritten)


def test_the_prognostic_tke_refusal_no_longer_recommends_a_refused_value(
) -> None:
    """The defect this walk found, pinned so it cannot come back.

    ``km_opt=2`` used to be refused with "select the LES topology or
    km_opt 3/4".  With a PBL scheme on, ``km_opt=3`` is refused by the
    branch immediately above for exactly the same PBL-off gate, so the
    advice sent a user from one refusal to another.  Only ``km_opt=4``
    works, and now that is what the message says.
    """

    base = dict(walk.ANCHORS["ysu-mm5-noah-rrtmgp"], **walk.TIER_A_HELD)
    refused = walk.attempt(dict(base, km_opt=2))
    assert refused.verdict == "REFUSED"
    assert "km_opt=4" in refused.message
    # The measurement behind the wording: 3 does not work here, 4 does.
    assert walk.attempt(dict(base, km_opt=3)).verdict == "REFUSED"
    assert walk.attempt(dict(base, km_opt=4)).verdict == "ACCEPTED"


# ---------------------------------------------------------------------------
# The axis tables are measured, not transcribed.
# ---------------------------------------------------------------------------

def test_each_axis_admits_exactly_the_values_the_walk_declares(
        receipt) -> None:
    """Validate the instrument.

    The axis tables in the tool are a transcription of ``woof/config.py``,
    and a transcription can be wrong or stale.  Tier E offers the loader
    every integer in [-2, 106] plus 900 and 901 on each axis and classifies
    each by whether the refusal is that axis's own schema message.  What
    the loader admits and what the tool claims must be the same set.
    """

    for axis in receipt["axes"]:
        assert axis["measured_admitted_values"] == axis["admitted_values"], (
            f"{axis['name']}: the loader admits "
            f"{axis['measured_admitted_values']} but the walk declares "
            f"{axis['admitted_values']} ({axis['authority']})")


def test_the_declared_axes_match_the_exported_scheme_tables() -> None:
    """The half of the transcription that config.py does export."""

    from woof.config import (CU_SCHEMES, LAND_SURFACE_SCHEMES,
                              MP_PHYSICS_ACCEPTED, PBL_SCHEMES,
                              RA_LW_PHYSICS_ACCEPTED, RA_SW_PHYSICS_ACCEPTED,
                              SURFACE_LAYER_SCHEMES)

    assert walk.AXIS_VALUES["mp_physics"] == tuple(MP_PHYSICS_ACCEPTED)
    assert walk.AXIS_VALUES["ra_lw_physics"] == tuple(RA_LW_PHYSICS_ACCEPTED)
    assert walk.AXIS_VALUES["ra_sw_physics"] == tuple(RA_SW_PHYSICS_ACCEPTED)
    assert walk.AXIS_VALUES["bl_pbl_physics"] == tuple(PBL_SCHEMES)
    assert walk.AXIS_VALUES["sf_sfclay_physics"] == tuple(
        SURFACE_LAYER_SCHEMES)
    assert walk.AXIS_VALUES["sf_surface_physics"] == tuple(
        LAND_SURFACE_SCHEMES)
    assert walk.AXIS_VALUES["cu_physics"] == tuple(CU_SCHEMES)


def test_nothing_is_dropped_without_being_written_down(receipt) -> None:
    """Skips are logged, with a reason, or they are not skips."""

    assert receipt["skipped"], "a tiered walk that logs no skip is lying"
    for entry in receipt["skipped"]:
        assert entry["what"] and entry["why"]
    assert receipt["coverage_rule"]


# ---------------------------------------------------------------------------
# MYNN, the family this whole wave is about.
# ---------------------------------------------------------------------------

def test_mynn_composes_with_every_land_surface(receipt) -> None:
    from woof.config import LAND_SURFACE_SCHEMES

    assert (receipt["mynn_slice"]["land_surface_options_accepted"]
            == sorted(set(LAND_SURFACE_SCHEMES) - {8}))
    assert all(value.startswith("REFUSED:") for key, value in receipt["mynn_matrix"].items() if key.startswith("lsm8."))


def test_mynn_composes_with_every_radiation_pairing_the_loader_admits(
        receipt) -> None:
    """"Every admitted radiation option" stated exactly.

    MYNN reaches every declared pair except the two native Fu-Liou-Gu
    dispatch exclusions. Keep the expected set independent from the
    measured receipt, including all other longwave-enabled combinations.
    """

    accepted_anywhere = {
        f"{lw}/{sw}"
        for lw in walk.AXIS_VALUES["ra_lw_physics"]
        for sw in walk.AXIS_VALUES["ra_sw_physics"]
        if any(
            key.endswith(f".lw{lw}.sw{sw}") and verdict == "ACCEPTED"
            for key, verdict in receipt["mynn_matrix"].items())
    }
    assert set(receipt["mynn_slice"]["radiation_options_accepted"]) == \
        accepted_anywhere
    expected = {
        f"{lw}/{sw}"
        for lw in walk.AXIS_VALUES["ra_lw_physics"]
        for sw in walk.AXIS_VALUES["ra_sw_physics"]
        if (lw, sw) not in RADIATION_PAIR_REFUSALS
    }
    assert accepted_anywhere == expected
    longwave_on = {pair for pair in accepted_anywhere
                   if pair.split("/")[0] not in ("0",)}
    assert longwave_on == {
        f"{lw}/{sw}"
        for lw in walk.AXIS_VALUES["ra_lw_physics"] if lw != 0
        for sw in walk.AXIS_VALUES["ra_sw_physics"]
        if (lw, sw) not in RADIATION_PAIR_REFUSALS
    }


def test_the_mynn_matrix_is_the_same_in_every_land_surface_column(
        receipt) -> None:
    """Independence, which is what "arbitrary composition" actually means.

    The radiation verdict must not depend on which land-surface model is
    selected.  If the two axes were coupled anywhere, these four columns
    would differ.
    """

    from woof.config import LAND_SURFACE_SCHEMES

    columns = {}
    for lsm in set(LAND_SURFACE_SCHEMES) - {8}:
        columns[lsm] = tuple(
            receipt["mynn_matrix"][f"lsm{lsm}.lw{lw}.sw{sw}"]
            for lw in walk.AXIS_VALUES["ra_lw_physics"]
            for sw in walk.AXIS_VALUES["ra_sw_physics"])
    assert len(set(columns.values())) == 1, (
        "the radiation verdict depends on the land-surface model: "
        f"{columns}")
    assert "ACCEPTED" in columns[LAND_SURFACE_SCHEMES[0]]


#: The microphysics MYNN reaches in the walk's MYNN slice, measured.
#:
#: The slice's anchor runs RTE+RRTMGP on both radiation streams, and ONE
#: admitted microphysics value is refused against it, for the missing
#: thing this table is about -- a cloud-optics row in
#: woof.core.rrtmgp._MP_CLOUD_OPTICS_SCHEME:
#:
#: * mp_physics=9 (Milbrandt-Yau) publishes NO effective radii at all.  WRF
#:   leaves has_reqc/has_reqi/has_reqs at 0 for MILBRANDT2MOM
#:   (phys/module_physics_init.F:1004-1023) and the scheme's own
#:   effective-radius block is commented out.
#:
#: It is not a MYNN limitation, and it is not a place to invent physics:
#: handing RRTMGP radii nobody computed is what put mp=28 on Kessler's row
#: until 2026-08-01.  The refusal names two remedies and both work:
#: ra_rrtmg_variant = 'rrtmg_legacy', or the Dudhia pair.
#:
#: mp_physics=50 (P3) SAT BESIDE IT AND NO LONGER DOES, which is the record
#: of a fix rather than a widening.  P3 publishes cloud and ice radii but
#: no snow radius, and it was excluded only because no RTE+RRTMGP row
#: existed that did not assume one.  The row now does -- ``50: "p3"``,
#: WRF's own coupling, has_reqc=1, has_reqi=1, has_reqs=0
#: (module_physics_init.F:1022-1024 then :1033), with the wrappers'
#: ice-into-snow remap (module_ra_rrtmg_lw.F:12250-12261,
#: module_ra_rrtmg_sw.F:10851-10863) transcribed into hydrometeor_paths --
#: so the pairing resolves instead of refusing, and
#: woof.config.validate_p3_radiation is retired with it.
#: mp_physics=9 (Milbrandt-Yau) followed P3 out of the excluded set for
#: the same shape of reason.  It was excluded because the RTE+RRTMGP
#: adapter had no cloud-optics row for it and refused the pairing; the row
#: now exists -- ``9: "milbrandt2"``, the scheme's own radii from the
#: block WRF ships commented out (module_mp_milbrandt2mom.F:3351-3378),
#: evaluated over the transported number moments -- so the pairing
#: resolves, and woof.config.validate_milbrandt2_options no longer
#: carries the refusal.
# MP13 is measured on its native legacy RRTMG pairing in tier B. This
# union records accepted MYNN configurations across radiation variants.
MYNN_MICROPHYSICS_ACCEPTED = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 18, 24, 26, 28, 38, 40, 50, 51, 52, 53, 55, 56, 97, 106, 900]

#: The admitted microphysics values the MYNN slice does not reach.  50 was
#: here from the 1.9 gate until the P3 cloud-optics row landed, and 9 until
#: the Milbrandt-Yau one did; see MYNN_MICROPHYSICS_ACCEPTED.  FAST_SBM (30)
#: is here by a JUDGED exclusion, not a gap: its bin effective-radius
#: integrals are not qualified for RTE+RRTMGP cloud optics
#: (woof.core.rrtmgp._NO_CLOUD_OPTICS_COUPLING), so the slice's RTE+RRTMGP
#: anchor refuses it at the run door; its legacy-RRTMG pairing accepts
#: (measured below).
#: The WRF 4.7.1 port added three more judged RTE+RRTMGP cloud-optics
#: exclusions of the same kind, each accepting on legacy RRTMG: UDM (27,
#: module_mp_udm.F's uninitialised den1d radius input), Eta Ferrier (95,
#: total ice with no qi or radii) and MAD-WRF (96).  RCON (29) is held at the
#: door by RCON-SPEED-01 (see HELD_AT_THE_DOOR) under every suite.
MYNN_MICROPHYSICS_EXCLUDED = (27, 29, 30, 95, 96)


def test_mynn_composes_with_every_microphysics_and_cumulus_scheme(
        receipt) -> None:
    """The claim, stated as the measurement instead of as an aspiration.

    Pin the measured union across the MYNN radiation anchors. Native
    legacy cloud-optics pairings include MP2 and MP13; FAST_SBM remains
    excluded from these anchors for its missing RTE effective radii.
    Loader checks below retain that concrete refusal and its remedy.
    """

    from woof.config import CU_SCHEMES

    accepted = receipt["mynn_slice"]["microphysics_options_accepted"]
    assert accepted == MYNN_MICROPHYSICS_ACCEPTED
    # WRF CAM ZM needs MYJ or CAM UW's temperature excess; MYNN has
    # neither provider (module_cu_camzm_driver.F:978-1018).
    assert (receipt["mynn_slice"]["cumulus_options_accepted"]
            == sorted(set(CU_SCHEMES) - {7}))
    mynn_cam = dict(walk.ANCHORS["mynn-rrtmgp-noah"],
                    **walk.TIER_A_HELD)
    mynn_cam["cu_physics"] = 7
    refused_cam = walk.attempt(mynn_cam)
    assert refused_cam.verdict == "REFUSED"
    assert "CAMZMSCHEME requires MYJPBLSCHEME or CAMUWPBLSCHEME" in refused_cam.message
    native_cam = dict(mynn_cam, bl_pbl_physics=2, sf_sfclay_physics=2)
    assert walk.attempt(native_cam).verdict == "ACCEPTED"

    # The exclusions are exactly these values, and every one is admitted by
    # the schema -- so these are composition refusals, not a shrunken axis.
    excluded = sorted(set(walk.AXIS_VALUES["mp_physics"]) - set(accepted))
    assert excluded == sorted(MYNN_MICROPHYSICS_EXCLUDED)

    # MP50 and MP9 left the exclusion set when their cloud-optics rows
    # landed. Their default RTE+RRTMGP and former Dudhia remedies accept.
    anchor = dict(walk.ANCHORS["mynn-rrtmgp-noah"], **walk.TIER_A_HELD)
    for mp_physics in (9, 50):
        assert mp_physics in accepted
        on_rrtmgp = walk.attempt(dict(anchor, mp_physics=mp_physics))
        assert on_rrtmgp.verdict == "ACCEPTED", (
            mp_physics, on_rrtmgp.message)
        on_dudhia = walk.attempt(dict(
            anchor, mp_physics=mp_physics,
            ra_lw_physics=0, ra_sw_physics=1))
        assert on_dudhia.verdict == "ACCEPTED", (mp_physics, on_dudhia.message)
    # Both native cloud-optics exclusions name the legacy remedy, and
    # the legacy RRTMG pair that remedy names must accept each scheme.
    for mp_physics in (13, 30):
        on_rrtmgp = walk.attempt(dict(anchor, mp_physics=mp_physics))
        assert on_rrtmgp.verdict == "REFUSED", on_rrtmgp.message
        assert "rrtmg_legacy" in on_rrtmgp.message
        on_legacy = walk.attempt(dict(anchor, mp_physics=mp_physics,
                                      ra_rrtmg_variant="rrtmg_legacy"))
        assert on_legacy.verdict == "ACCEPTED", on_legacy.message


def _radiation_bearing(switches) -> bool:
    """Both streams on: the property the 1.8.8 MYNN family exists for."""

    return (int(switches["ra_lw_physics"]) != 0
            and int(switches["ra_sw_physics"]) != 0)


def test_the_shipped_mynn_presets_are_inside_the_measured_space() -> None:
    """The bridge from the walk to the product.

    Every shipped MYNN profile's own switch tuple must be one of the
    combinations the walk accepts -- otherwise the walk is measuring a
    space the presets do not live in.

    AND the radiation-bearing family is asserted SEPARATELY, per land
    surface.  Filtering on ``bl_pbl_physics == 5`` alone is satisfied by
    the three no-radiation MYNN rows that shipped long before this work,
    so a version of this test that stopped at the loop above would pass
    unchanged with the whole radiation-bearing family deleted: it would
    measure nothing the family added.  The property that discriminates
    is the one the family was built to establish -- every land surface
    that offers a no-radiation MYNN preset must also offer a
    radiation-bearing one, so a user who wants MYNN never has to leave
    the nocturnally valid block to get it.  Dropping any single row
    turns this red, and the pairing is derived from the shipped
    switches rather than a transcribed list of three names, so renaming
    a profile cannot quietly satisfy it either.
    """

    from woof.physics_compat import (SINGLE_DOMAIN_PHYSICS_PROFILES,
                                      single_domain_runtime_switches)

    mynn = [
        profile for profile in SINGLE_DOMAIN_PHYSICS_PROFILES
        if int(single_domain_runtime_switches(profile)["bl_pbl_physics"]) == 5
    ]
    assert mynn, "no MYNN profile ships at all"
    for profile in mynn:
        switches = single_domain_runtime_switches(profile)
        combination = {
            name: int(switches[name]) for name in walk.AXIS_NAMES
            if name in switches}
        combination.setdefault("km_opt", 4)
        outcome = walk.attempt(combination)
        assert outcome.verdict == "ACCEPTED", (
            f"{profile} resolves to a combination the walk refuses: "
            f"{outcome.message}")

    # Keyed by (microphysics, land surface): the MYNN + RUC pair ships with
    # WSM6 and with Thompson, and each microphysics owes its own
    # radiation-bearing partner, so one land surface can carry two pairs.
    by_land_surface: dict[tuple[int, int], dict[bool, list[str]]] = {}
    for profile in mynn:
        switches = single_domain_runtime_switches(profile)
        land_surface = (int(switches["mp_physics"]),
                        int(switches["sf_surface_physics"]))
        bucket = by_land_surface.setdefault(land_surface, {True: [], False: []})
        bucket[_radiation_bearing(switches)].append(profile)

    # The control.  Without a surviving no-radiation row the pairing
    # requirement below would be vacuously satisfied by an empty set.
    dark = {land_surface for land_surface, rows in by_land_surface.items()
            if rows[False]}
    assert dark, (
        "no shortwave-only MYNN preset ships any more; this test's "
        "pairing requirement has gone vacuous and needs rewriting")

    unpaired = sorted(
        land_surface for land_surface in dark
        if not by_land_surface[land_surface][True])
    assert unpaired == [], (
        "every land surface with a shortwave-only MYNN preset must also "
        "ship a radiation-bearing MYNN preset, or a user choosing MYNN "
        "from a menu has no nocturnally valid row to move to; "
        f"(mp_physics, sf_surface_physics)={unpaired} has none. Shipped "
        "MYNN rows: "
        + "; ".join(
            f"mp{land_surface[0]} lsm{land_surface[1]}: "
            f"lw+sw={rows[True]} sw-only={rows[False]}"
            for land_surface, rows in sorted(by_land_surface.items())))

    # And the radiation-bearing rows are in the measured space WITH
    # radiation on, which is the claim the walk is cited for.
    radiating = [profile for rows in by_land_surface.values()
                 for profile in rows[True]]
    # How many radiation-bearing MYNN rows each (mp_physics,
    # sf_surface_physics) pair ships, keyed by switches, not names.  It was
    # one per shortwave-only pair until 2026-10-03, when three HRRR-shaped
    # legacy-RRTMG suites joined the menu on purpose: Thompson mp8 over RUC
    # with prescribed monthly surface seeds (322308291, cc881f530) and its
    # solar-angle albedo sibling (67d633a15, 32f258956), and the
    # aerosol-aware Thompson mp28 with the GSD MYNN form (4193eb0da,
    # be69b1235).  A dropped or an unannounced row still turns this red.
    shipped = {land_surface: len(rows[True])
               for land_surface, rows in by_land_surface.items()
               if rows[True]}
    assert shipped == {(6, 2): 1, (6, 3): 1, (6, 4): 1, (8, 3): 3,
                       (28, 3): 1}, (
        f"radiation-bearing MYNN rows per (mp_physics, sf_surface_physics) "
        f"moved: got {shipped} from {radiating} against shortwave-only "
        f"pairs {sorted(dark)}")
    for profile in radiating:
        switches = single_domain_runtime_switches(profile)
        combination = {
            name: int(switches[name]) for name in walk.AXIS_NAMES
            if name in switches}
        combination.setdefault("km_opt", 4)
        assert combination["ra_lw_physics"] and combination["ra_sw_physics"]
        outcome = walk.attempt(combination)
        assert outcome.verdict == "ACCEPTED", (
            f"{profile} runs both radiation streams and the walk refuses "
            f"it: {outcome.message}")


def test_parallel_loader_walk_preserves_order_and_negative_controls():
    background = dict(walk.ANCHORS['ysu-mm5-noah-rrtmgp'], **walk.TIER_A_HELD)
    rows = [('probe', str(i), dict(background, **change)) for i, change in enumerate((
        {}, {'mp_physics': 40, 'cu_physics': 11, 'aercu_opt': 1},
        {'bl_pbl_physics': 2, 'sf_sfclay_physics': 1},
        {'ra_lw_physics': 999}, {'cu_physics': 0},
    ))]
    serial, parallel = walk.Walk(jobs=1), walk.Walk(jobs=2)
    serial.run_many(iter(rows))
    parallel.run_many(iter(rows))
    assert parallel.rows == serial.rows
    assert [row[1] for row in parallel.rows] == [str(i) for i in range(len(rows))]
    assert {row[3].verdict for row in parallel.rows} == {'ACCEPTED', 'REFUSED'}
