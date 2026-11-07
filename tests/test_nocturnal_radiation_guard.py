"""The nocturnal-radiation guard and the wizard's nocturnally sane default.

Provenance: a wizard-emitted 48 h real case bound
``thompson-mp8-ysu-mm5-noah-dudhia-daytime-v1`` (ra_lw_physics 0,
ra_sw_physics 1).  Shortwave heated the surface by day; at night the
surface radiated with no downward longwave, skin temperature cratered
and 2 m dewpoints read in the 50s F inside a 70s airmass.  Two fixes,
both bound here:

* the wizard's real-case default is a certified full lw+sw profile
  (registry maturity read off the registry, never asserted);
* any real experiment whose window includes local night refuses to LOAD
  an undeclared shortwave-on/longwave-off pairing, at the one loader
  every front door shares, with the physics, the profile and both
  remedies named.

The instrument is tested in both directions (a night window trips, a
daylight window does not) because a guard that cannot pass is as wrong
as one that cannot fire.
"""
from __future__ import annotations

from datetime import datetime

import tomllib

import pytest

from woof.domain_wizard import (DEFAULT_PHYSICS_PROFILE,
                                 HRRR_DEFAULT_PROFILE,
                                 experiment_from_text, render_config,
                                 resolved_physics_profile)
from woof.experiment import build_experiment
from woof.hrrr_route_inputs import ROUTE_DEFAULT_PHYSICS_PROFILE
from woof.physics_compat import (ASYMMETRIC_RADIATION_NOCTURNAL_ACK,
                                  CONSTANT_DOWNWARD_LONGWAVE_ACK,
                                  MORRISON_PROFILE_ID, THOMPSON_PROFILE_ID,
                                  WSM6_PROFILE_ID, first_local_night_time,
                                  single_domain_runtime_switches,
                                  solar_elevation_deg)

#: The reporting user's geometry: an Alabama reference point whose April
#: evening includes local night about 12 h into a 12Z-start window.
_PROJECTION = {
    "map_proj": "lambert", "ref_lat": 33.8, "ref_lon": -87.29,
    "truelat1": 23.8, "truelat2": 43.8, "stand_lon": -87.29,
}


def _emitted(profile, *, start=datetime(2011, 4, 26, 12), hours=48,
             acknowledgements=()):
    """The wizard's emitted bytes.

    ``acknowledgements`` defaults to NOTHING, which is the 1.8.8 change:
    :func:`render_config` no longer invents the nocturnal declaration, so
    a caller that wants one passes it, exactly as ``woof domain --ack``
    does.
    """
    return render_config(
        name="guardcase", start_time=start, hours=hours,
        projection=dict(_PROJECTION), dims=[(120, 100)], ratios=(),
        fetch_hints={"source": "era5"}, case_data=None, profile=profile,
        acknowledgements=tuple(acknowledgements))


def _raw(text):
    raw = tomllib.loads(text)
    raw.pop("fetch", None)
    raw.pop("case_data", None)
    return raw


# ---------------------------------------------------------------------------
# The default: certified, full lw+sw, taken from the registry.
# ---------------------------------------------------------------------------

def test_real_case_default_is_the_certified_full_radiation_profile():
    assert DEFAULT_PHYSICS_PROFILE == MORRISON_PROFILE_ID
    assert resolved_physics_profile("era5", None) == MORRISON_PROFILE_ID
    assert resolved_physics_profile("gfs", None) == MORRISON_PROFILE_ID
    # 1.8: HRRR is no longer the exception.  Its default is a different
    # profile from gfs/era5 -- its route refuses Kain-Fritsch at 3 km --
    # but it is full-radiation like theirs, and it comes from the route's
    # own authority rather than a door-local copy.
    assert resolved_physics_profile("hrrr", None) == HRRR_DEFAULT_PROFILE
    assert HRRR_DEFAULT_PROFILE == ROUTE_DEFAULT_PHYSICS_PROFILE
    assert (resolved_physics_profile("era5", THOMPSON_PROFILE_ID)
            == THOMPSON_PROFILE_ID)


def test_default_profile_is_registry_certified_and_nocturnally_valid():
    """The choice is re-derived from the authorities, not from this test."""
    from woof.physics_registry import physics_registry

    template = physics_registry()["templates"][DEFAULT_PHYSICS_PROFILE]
    assert template["maturity"] == "wrf-matched-run"
    switches = single_domain_runtime_switches(DEFAULT_PHYSICS_PROFILE)
    assert switches["ra_lw_physics"] == 4
    assert switches["ra_sw_physics"] == 4


def test_every_door_default_is_full_radiation_and_route_admissible():
    """Both facts, for every source, from the shipped tables.

    The second one is the fact the interactive door's own test could not
    see: it re-derived "offered by that source's route" from the registry
    and "strongest maturity" from the template, and both were TRUE of the
    Morrison suite on hrrr while ``validate_route_physics`` refused it on
    ``cu_physics``.  Registry reachability is not route admissibility, so
    this asserts the emission actually survives the gate that rejects it.
    """
    from woof.domain_interactive import DEFAULT_PHYSICS_PROFILE_BY_SOURCE
    from woof.hrrr_route_inputs import validate_route_physics

    for source in ("gfs", "era5", "hrrr"):
        for profile in (resolved_physics_profile(source, None),
                        DEFAULT_PHYSICS_PROFILE_BY_SOURCE[source]):
            switches = single_domain_runtime_switches(profile)
            assert switches["ra_lw_physics"] == 4, (
                f"{source} default {profile} runs longwave "
                f"{switches['ra_lw_physics']}")
            assert switches["ra_sw_physics"] == 4, (
                f"{source} default {profile} runs shortwave "
                f"{switches['ra_sw_physics']}")
    # And the hrrr default loads through the route's own physics gate.
    validate_route_physics(
        experiment_from_text(
            _emitted(resolved_physics_profile("hrrr", None)),
            source="<hrrr-default-emission>"))


def test_the_route_default_is_the_strongest_admissible_full_radiation_suite():
    """Re-derived from the switch tables and the route's gates.

    The historical strongest-maturity claim is superseded by the
    source-selected GSD4.1 route authority. Admissibility and full
    radiation do not establish numerical or nowcast certification.

    Not a restatement of the constant: this recomputes the admissible
    set the way :func:`validate_route_physics` does and checks the
    default is in it, so a profile that gains or loses admissibility
    moves this assertion rather than aging quietly beside it.
    """
    from woof.hrrr_route_inputs import route_physics_problems
    from woof.physics_compat import SINGLE_DOMAIN_PHYSICS_PROFILES
    # 4193eb0da, development branch: select the HRRR GSD4.1 composition.
    assert ROUTE_DEFAULT_PHYSICS_PROFILE == (
        "thompson-mp28-mynn-gsd41-mynn-ruc-rrtmg-legacy-v1")
    admissible = [profile for profile in SINGLE_DOMAIN_PHYSICS_PROFILES
                  if not route_physics_problems(single_domain_runtime_switches(profile))
                  and single_domain_runtime_switches(profile)["ra_lw_physics"] > 0
                  and single_domain_runtime_switches(profile)["ra_sw_physics"] > 0]
    assert ROUTE_DEFAULT_PHYSICS_PROFILE in admissible, (
        f"the route default is not admissible; admissible 4/4 suites are "
        f"{admissible}")
    # The route default's composition: Thompson microphysics, both
    # radiation streams on, no cumulus at its 3 km grid.  Which 4/4
    # implementation runs is a setting this assertion does not read.
    switches = single_domain_runtime_switches(ROUTE_DEFAULT_PHYSICS_PROFILE)
    # 4193eb0da, development branch: the named HRRR route now selects
    # MP28/GSD4.1/RUC. Source authority is not numerical certification.
    assert int(switches["mp_physics"]) == 28
    assert int(switches["bl_pbl_physics"]) == 5
    assert int(switches["sf_surface_physics"]) == 3
    assert switches["bl_mynn_version"] == "gsd_41"
    assert int(switches["cu_physics"]) == 0


def test_default_emission_is_nocturnally_valid_and_loads():
    text = _emitted(DEFAULT_PHYSICS_PROFILE)
    assert MORRISON_PROFILE_ID in text
    assert "# NOCTURNALLY VALID" in text
    assert ASYMMETRIC_RADIATION_NOCTURNAL_ACK not in text
    exp = experiment_from_text(text, source="<default-emission>")
    assert exp.root.run.ra_lw_physics == 4
    assert exp.root.run.ra_sw_physics == 4


def test_no_door_defaults_to_an_asymmetric_pairing():
    """The CHANGELOG claim line, over EVERY source.

    It looped ``("gfs", "era5")`` while 1.7.1's own CHANGELOG had to be
    corrected to say "neither the gfs nor the era5 door", because hrrr
    defaulted to a shortwave-on/longwave-off suite and this test was
    written not to look.  A claim test that skips the failing case is
    the claim, restated.
    """
    from woof.domain_interactive import DEFAULT_PHYSICS_PROFILE_BY_SOURCE

    for source in ("gfs", "era5", "hrrr"):
        for profile in (resolved_physics_profile(source, None),
                        DEFAULT_PHYSICS_PROFILE_BY_SOURCE[source]):
            switches = single_domain_runtime_switches(profile)
            assert not (switches["ra_sw_physics"] > 0
                        and switches["ra_lw_physics"] == 0), (
                f"{source} default {profile} is asymmetric")


# ---------------------------------------------------------------------------
# The guard, both directions.
# ---------------------------------------------------------------------------

def test_legacy_asymmetric_night_config_refuses_at_load_naming_everything():
    """The reporting user's shape: no declaration anywhere in the file."""
    raw = _raw(_emitted(THOMPSON_PROFILE_ID))
    raw["experiment"].pop("acknowledgements", None)  # legacy file
    with pytest.raises(ValueError) as caught:
        build_experiment(raw, source="<legacy-night>")
    message = str(caught.value)
    assert "local night" in message
    assert "ra_lw_physics 0" in message
    assert THOMPSON_PROFILE_ID in message           # the profile that did it
    assert "skin temperature" in message            # the physics
    # Remedy 1: a sane profile -- and one that is ROUTE-SAFE.
    #
    # This used to assert MORRISON_PROFILE_ID, and that was the defect
    # (2026-08-20): a loaded config carries no forcing source, so this
    # refusal cannot know which route will run it, and the example it
    # named is refused by the native HRRR route for cu_physics=1.  A
    # user who took it met a second refusal.  The example is computed
    # now, and the property it must have is the one asserted here:
    # every registered source's route admits it.
    from woof.domain_wizard import profile_route_blocker
    from woof.physics_menu import universally_admissible_profile
    from woof.source_adapters import source_adapters

    example = universally_admissible_profile()
    assert example is not None
    assert example in message
    for adapter in source_adapters():
        assert profile_route_blocker(example, adapter.source_id) is None, (
            adapter.source_id)
    assert ASYMMETRIC_RADIATION_NOCTURNAL_ACK in message  # remedy 2: declare


def test_asymmetric_night_with_declared_acknowledgement_loads():
    # BOTH tokens.  The nocturnal declaration now answers only the
    # nocturnal question; the suite also runs Noah with ra_lw_physics 0,
    # which is a constant downward longwave and its own declaration
    # (woof.physics_compat.constant_longwave_refusal).  Declaring one and
    # not the other is exactly the elision that let ten shipped configs
    # run on a frozen 300 W m-2.
    raw = _raw(_emitted(THOMPSON_PROFILE_ID))
    raw["experiment"]["acknowledgements"] = [
        ASYMMETRIC_RADIATION_NOCTURNAL_ACK,
        CONSTANT_DOWNWARD_LONGWAVE_ACK]
    exp = build_experiment(raw, source="<declared-night>")
    assert ASYMMETRIC_RADIATION_NOCTURNAL_ACK in exp.acknowledgements


def test_the_nocturnal_token_alone_no_longer_admits_a_frozen_longwave():
    """The bypass this lane closed, asserted as a refusal.

    The 1.7.1 token is checked before any physics is inspected, so a
    config carrying it was never asked where its downward longwave came
    from.  It must no longer be enough on its own.
    """
    raw = _raw(_emitted(THOMPSON_PROFILE_ID))
    raw["experiment"]["acknowledgements"] = [
        ASYMMETRIC_RADIATION_NOCTURNAL_ACK]
    with pytest.raises(ValueError) as caught:
        build_experiment(raw, source="<nocturnal-token-only>")
    message = str(caught.value)
    assert CONSTANT_DOWNWARD_LONGWAVE_ACK in message
    assert "300 W m-2" in message


def test_a_publisher_only_suite_refuses_at_load_not_mid_run():
    """lw=0/sw=1 with NO land surface: the DOOR refuses, not the engine.

    The scope finding this closes: the load refusal used to fire only
    for sf_surface_physics in {2, 3, 4} while initialize_physics also
    refused the publisher case (radiation active, so the fabricated GLW
    row reaches every wrfout frame), so this exact configuration loaded
    clean through build_experiment and died after ingest and prepare.
    All-daylight window on purpose: this is the constant guard firing,
    not the nocturnal one.
    """
    raw = _raw(_emitted(THOMPSON_PROFILE_ID,
                        start=datetime(2011, 4, 26, 15), hours=3))
    raw["shared"].update({"sf_surface_physics": 0, "sf_sfclay_physics": 0,
                          "bl_pbl_physics": 0, "km_opt": 1})
    raw["experiment"].pop("acknowledgements", None)
    with pytest.raises(ValueError) as caught:
        build_experiment(raw, source="<publisher-only>")
    message = str(caught.value)
    assert CONSTANT_DOWNWARD_LONGWAVE_ACK in message
    assert "wrfout frame" in message
    # And the same declaration that admits the consumed case admits this
    # one -- fail at load, always; run once declared.
    raw["experiment"]["acknowledgements"] = [CONSTANT_DOWNWARD_LONGWAVE_ACK]
    exp = build_experiment(raw, source="<publisher-declared>")
    assert CONSTANT_DOWNWARD_LONGWAVE_ACK in exp.acknowledgements


def test_asymmetric_daylight_window_loads_without_declaration():
    """Negative control: the NOCTURNAL guard must be passable by daylight.

    Daylight lifts the nocturnal guard and nothing else: the suite still
    runs Noah with no longwave scheme, so it still declares the constant.
    A frozen 300 W m-2 is wrong at noon too -- clear-sky downward
    longwave over a warm surface runs 350-400 W m-2 -- it is merely less
    catastrophic than it is at 3 a.m.
    """
    raw = _raw(_emitted(THOMPSON_PROFILE_ID,
                        start=datetime(2011, 4, 26, 15), hours=3))
    raw["experiment"]["acknowledgements"] = [CONSTANT_DOWNWARD_LONGWAVE_ACK]
    exp = build_experiment(raw, source="<daylight>")
    assert ASYMMETRIC_RADIATION_NOCTURNAL_ACK not in exp.acknowledgements


def test_full_radiation_night_window_is_not_guarded():
    raw = _raw(_emitted(MORRISON_PROFILE_ID))
    build_experiment(raw, source="<full-radiation-night>")


def test_idealized_experiment_without_projection_is_untouched():
    """No [projection] means no place and no clock: nothing to guard."""
    from woof.experiment import ExperimentConfig

    assert ExperimentConfig.__dataclass_fields__[
        "projection"].default is None
    # The guard body is scoped on projection presence; a config with no
    # projection table is an idealized one and never reaches it.  Bound
    # structurally because idealized fixtures build their own vertical
    # coordinates and belong to their own suites.


# ---------------------------------------------------------------------------
# The wizard's own emissions of explicitly selected asymmetric profiles.
# ---------------------------------------------------------------------------

def test_wizard_does_not_declare_a_nocturnal_experiment_by_itself():
    """1.8.8: the emitted bytes carry no declaration nobody made.

    Through 1.8.7 this exact call wrote
    ``acknowledgements = [ASYMMETRIC_RADIATION_NOCTURNAL_ACK]`` into the
    emitted ``[experiment]`` on its own, and that line disarmed the load
    guard at every other front door for the life of the file.  Now the
    file says what is wrong with it, names the flag, and does not load
    until its owner declares it.
    """
    text = _emitted(THOMPSON_PROFILE_ID)
    assert "# NOT NOCTURNALLY VALID, AND THIS FILE WILL NOT LOAD" in text
    assert ASYMMETRIC_RADIATION_NOCTURNAL_ACK in text   # named as the remedy
    assert (f'acknowledgements = ["{ASYMMETRIC_RADIATION_NOCTURNAL_ACK}"]'
            not in text)                                # but NOT declared
    with pytest.raises(ValueError, match="local night"):
        experiment_from_text(text, source="<undeclared-night>")


def test_wizard_writes_the_declaration_it_is_handed():
    """...and when its owner does declare it, the file loads and says so."""
    text = _emitted(THOMPSON_PROFILE_ID,
                    acknowledgements=(ASYMMETRIC_RADIATION_NOCTURNAL_ACK,))
    assert "# NOT NOCTURNALLY VALID" in text
    assert "because YOU declared it" in text
    # Membership, not the one-element spelling: this suite ALSO gets the
    # suite-derived constant-longwave token (see
    # test_wizard_daylight_asymmetric_emission_warns_without_declaring),
    # so the array has two members.  What this pins is that the token the
    # caller handed in is the one written.
    assert ASYMMETRIC_RADIATION_NOCTURNAL_ACK in _raw(text)[
        "experiment"]["acknowledgements"]
    # And the file it emits still loads through every front door's guard.
    exp = experiment_from_text(text, source="<explicit-night>")
    assert ASYMMETRIC_RADIATION_NOCTURNAL_ACK in exp.acknowledgements


def test_wizard_daylight_asymmetric_emission_warns_without_declaring():
    text = _emitted(THOMPSON_PROFILE_ID,
                    start=datetime(2011, 4, 26, 15), hours=3)
    assert "NOT NOCTURNALLY VALID" in text
    assert "all-daylight" in text
    assert ASYMMETRIC_RADIATION_NOCTURNAL_ACK not in text
    # It DOES declare the constant longwave, with a justification, because
    # daylight does not make a fabricated flux a computed one.
    assert CONSTANT_DOWNWARD_LONGWAVE_ACK in text
    assert f"# JUSTIFY {CONSTANT_DOWNWARD_LONGWAVE_ACK}:" in text
    experiment_from_text(text, source="<explicit-day>")


# ---------------------------------------------------------------------------
# The declaration the wizard writes must also be SPOKEN.
#
# Measured 2026-08-09 on 1.8.7: `woof domain --physics-profile <asymmetric>`
# over a night window emitted the acknowledgement -- which disarms the load
# guard at every other front door for that file -- and printed nothing about
# it on stdout or stderr, not even under --explain.  The only statement was a
# comment inside the emitted TOML.  The advisory is asserted through the
# WARNING OBSERVER rather than captured text because that is the channel a
# machine consumer reads: `woof run-plan --resolve` collects exactly these
# records into its `warnings` array, so one assertion covers the person at
# the terminal and the front end driving the intent route.
# ---------------------------------------------------------------------------

def _wizard_warnings(tmp_path, *, profile, cycle, hours, name, ack=(),
                     expect_code=0):
    """Run the real `woof domain` door and collect its warning records."""
    from woof import explain
    from woof.cli import main

    records: list[dict] = []
    explain.add_warning_observer(records.append)
    try:
        out = tmp_path / f"{name}.toml"
        argv = ["domain", "--point", "33.8,-87.29", "--source", "era5",
                "--cycle", cycle, "--hours", str(hours),
                "--physics-profile", profile, "--out", str(out)]
        for token in ack:
            argv += ["--ack", token]
        code = main(argv)
    finally:
        explain.remove_warning_observer(records.append)
    assert code == expect_code
    return out, [record for record in records
                 if "NOCTURNALLY VALID" in record["action"]]


def test_wizard_speaks_the_nocturnal_declaration_it_writes(tmp_path):
    out, spoken = _wizard_warnings(
        tmp_path, profile=THOMPSON_PROFILE_ID, cycle="2011-04-26T12",
        hours=48, name="night",
        ack=(ASYMMETRIC_RADIATION_NOCTURNAL_ACK,))
    # Membership, not the one-element spelling: this suite also gets the
    # suite-derived constant-longwave token, so the array has two members.
    # What this pins is that the token --ack handed in is the one written.
    assert ASYMMETRIC_RADIATION_NOCTURNAL_ACK in _raw(out.read_text())[
        "experiment"]["acknowledgements"]
    assert len(spoken) == 1, [record["action"] for record in spoken]
    action = spoken[0]["action"]
    assert THOMPSON_PROFILE_ID in action           # what did it
    assert ASYMMETRIC_RADIATION_NOCTURNAL_ACK in action  # what was declared
    assert MORRISON_PROFILE_ID in action           # the remedy
    assert "local night" in action
    assert "skin temperature" in spoken[0]["why"]  # the physics, layered


def test_wizard_speaks_only_where_there_is_something_to_declare(tmp_path):
    """Both negative controls, bound to the positive case so they can fail.

    A warning that cannot stay quiet is as wrong as one that cannot
    fire, and this one guards a deliberate, supported validation
    configuration -- crying on every emission would train the reader to
    skip the line that matters.  So the controls are real.

    But asserting ``spoken == []`` on the two controls ALONE is not a
    test: it holds on any tree where the wizard says nothing at all,
    including the tree from before this advisory existed, which is the
    exact-zero-delta failure mode -- a measurement that agrees with the
    hypothesis because the experiment never ran.  The controls are
    therefore measured as a CONTRAST against the one emission that must
    speak: 1 warning where a declaration is written into the file, 0
    where none is.  The positive arm is what makes this pair
    discriminate; drop it and the test goes green on a reverted tree.

    Only the counts live here.  What the spoken warning has to SAY is
    asserted in
    :func:`test_wizard_speaks_the_nocturnal_declaration_it_writes`.
    """
    _, declared = _wizard_warnings(
        tmp_path, profile=THOMPSON_PROFILE_ID, cycle="2011-04-26T12",
        hours=48, name="contrast-night",
        ack=(ASYMMETRIC_RADIATION_NOCTURNAL_ACK,))
    _, valid_suite = _wizard_warnings(
        tmp_path, profile=MORRISON_PROFILE_ID, cycle="2011-04-26T12",
        hours=48, name="valid")
    daylight_out, daylight = _wizard_warnings(
        tmp_path, profile=THOMPSON_PROFILE_ID, cycle="2011-04-26T15",
        hours=3, name="daylight")
    assert (len(declared), len(valid_suite), len(daylight)) == (1, 0, 0), (
        [record["action"]
         for record in declared + valid_suite + daylight])
    assert ASYMMETRIC_RADIATION_NOCTURNAL_ACK not in daylight_out.read_text()


# ---------------------------------------------------------------------------
# The instrument itself, against known answers in both directions.
# ---------------------------------------------------------------------------

def test_solar_elevation_known_answers():
    # Equinox, subsolar point: sun near the zenith at local noon...
    assert solar_elevation_deg(datetime(2026, 3, 20, 12, 0), 0.0, 0.0) > 80.0
    # ...and far below the horizon at local midnight.
    assert solar_elevation_deg(datetime(2026, 3, 20, 0, 0), 0.0, 0.0) < -80.0
    # June polar day: Svalbard's sun stays up at local midnight.
    assert solar_elevation_deg(datetime(2026, 6, 21, 23, 0),
                               78.0, 15.0) > 0.0


def test_first_local_night_time_both_directions():
    # The reporting user's window: night arrives the first evening.
    night = first_local_night_time(
        datetime(2011, 4, 26, 12), 48 * 3600.0,
        ref_lat=33.8, ref_lon=-87.29)
    assert night is not None
    assert datetime(2011, 4, 26, 22) < night < datetime(2011, 4, 27, 3)
    # A midday 3 h window at the same point has no night in it.
    assert first_local_night_time(
        datetime(2011, 4, 26, 15), 3 * 3600.0,
        ref_lat=33.8, ref_lon=-87.29) is None


# ---------------------------------------------------------------------------
# The wizard REFUSES rather than declare a nocturnal experiment for you.
#
# Measured 2026-08-09 on 1.8.7: `woof domain --physics-profile <asymmetric>`
# over a night window wrote acknowledgements = [<ack>] into the emitted file
# by itself.  the advisory change made that emission speak; speaking is necessary
# and not sufficient, because the FILE still carried a statement its owner
# had never made, and every later reader of the file -- `check`, `run`, `go`,
# run-plan, both prepared runners -- reads the file and not the terminal.
# ---------------------------------------------------------------------------

def _domain(argv):
    """Run the real `woof domain` door in process; return its exit code."""
    from woof.cli import main

    try:
        return main(argv)
    except SystemExit as exit_code:          # argparse-style exits
        return int(exit_code.code or 0)


def _loaded(path):
    """The emitted file through the shared loader, [case_data] and all."""
    from woof.experiment import load_experiment
    return load_experiment(path)


def test_wizard_door_refuses_an_undeclared_nocturnal_selection(tmp_path):
    """The door, run for real, with no --ack: refuses and writes nothing."""
    out = tmp_path / "undeclared.toml"
    code = _domain(["domain", "--point", "33.8,-87.29", "--source", "era5",
                    "--cycle", "2011-04-26T12", "--hours", "48",
                    "--physics-profile", THOMPSON_PROFILE_ID,
                    "--out", str(out)])
    assert code != 0
    # No file at all: a config that cannot load is worse than no config,
    # because it looks like progress.
    assert not out.exists()


def test_wizard_door_refusal_names_the_profile_the_window_and_both_ways_out(
        tmp_path, capsys):
    out = tmp_path / "undeclared2.toml"
    assert _domain(["domain", "--point", "33.8,-87.29", "--source", "era5",
                    "--cycle", "2011-04-26T12", "--hours", "48",
                    "--physics-profile", THOMPSON_PROFILE_ID,
                    "--out", str(out)]) != 0
    printed = capsys.readouterr()
    message = printed.out + printed.err
    assert THOMPSON_PROFILE_ID in message                 # what did it
    assert "local night" in message                       # why
    assert MORRISON_PROFILE_ID in message                 # remedy 1
    assert f"--ack {ASYMMETRIC_RADIATION_NOCTURNAL_ACK}" in message  # 2


def test_wizard_door_with_the_declaration_emits_and_the_file_loads(tmp_path):
    """The positive arm, through the same door: --ack is what writes it."""
    out = tmp_path / "declared.toml"
    assert _domain([
        "domain", "--point", "33.8,-87.29", "--source", "era5",
        "--cycle", "2011-04-26T12", "--hours", "48",
        "--physics-profile", THOMPSON_PROFILE_ID,
        "--ack", ASYMMETRIC_RADIATION_NOCTURNAL_ACK,
        "--out", str(out)]) == 0
    # The nocturnal token comes from --ack; the constant-longwave one is a
    # consequence of the named suite and the wizard writes it in ink.  Both
    # are expected here, and they stay DISTINCT tokens.
    assert _loaded(out).acknowledgements == (
        ASYMMETRIC_RADIATION_NOCTURNAL_ACK, CONSTANT_DOWNWARD_LONGWAVE_ACK)


def test_wizard_door_negative_controls_still_emit_without_any_ack(tmp_path):
    """NEGATIVE: the refusal must not fire on the cases it is not about.

    Bound as a CONTRAST against the refusing case above rather than as
    three bare successes, because three bare successes also pass on a
    tree where the refusal was never wired in.
    """
    # 1. The default (full lw+sw) over the SAME night window.
    full = tmp_path / "full.toml"
    assert _domain(["domain", "--point", "33.8,-87.29", "--source", "era5",
                    "--cycle", "2011-04-26T12", "--hours", "48",
                    "--out", str(full)]) == 0
    assert _loaded(full).acknowledgements == ()

    # 2. The asymmetric suite over an ALL-DAYLIGHT window.
    day = tmp_path / "day.toml"
    assert _domain(["domain", "--point", "33.8,-87.29", "--source", "era5",
                    "--cycle", "2011-04-26T15", "--hours", "3",
                    "--physics-profile", THOMPSON_PROFILE_ID,
                    "--out", str(day)]) == 0
    # No NOCTURNAL declaration -- that is what this control is about, and
    # the wizard makes none because the window is all daylight.  The
    # constant-longwave token IS written, because daylight does not make a
    # fabricated flux a computed one: the suite runs Noah with
    # ra_lw_physics = 0 at noon as much as at midnight.  Distinct tokens,
    # distinct claims.
    assert _loaded(day).acknowledgements == (CONSTANT_DOWNWARD_LONGWAVE_ACK,)
    assert ASYMMETRIC_RADIATION_NOCTURNAL_ACK not in day.read_text()

    # 3. And the hrrr door's own default, which is also full lw+sw.
    hrrr = tmp_path / "hrrr.toml"
    assert _domain(["domain", "--point", "33.8,-87.29", "--source", "hrrr",
                    "--cycle", "2026-08-05T00", "--hours", "6",
                    "--out", str(hrrr)]) == 0
    assert _loaded(hrrr).acknowledgements == ()
    # This point root is 12 km on a cone centred on the point, not the HRRR
    # row's cone, so since f0acb58ada7 (development branch:
    # no [static] source emitted for a HRRR grid off the row cone) the door
    # writes no [static] table: the metadata default stays a DEFAULT and
    # takes the WPS_GEOG fallback instead of becoming a declared source that
    # refuses the grid.  Both load seams build it, and a [static] table the
    # file does carry is still validated key by key.
    from woof.static.source_defaults import (defaulted_source_fallback,
                                              source_static_defaults,
                                              with_source_static_defaults)
    raw = tomllib.loads(hrrr.read_text())
    assert "static" not in raw
    assert defaulted_source_fallback(source_static_defaults("hrrr")["source"],
                                     raw["projection"])["used"] == "WPS_GEOG"
    assert with_source_static_defaults(raw, "hrrr") is raw
    assert experiment_from_text(hrrr.read_text(), source=str(hrrr)).root.run.mp_physics == 28
    assert build_experiment(_raw(hrrr.read_text()),
                            source="bare-wizard-case").root.run.mp_physics == 28
    invalid = hrrr.read_text() + "\n[static]\nnot_a_static_key = true\n"
    with pytest.raises(ValueError, match="not_a_static_key"):
        experiment_from_text(invalid, source=str(hrrr))


# ---------------------------------------------------------------------------
# The six templates whose NAME says "no-radiation" and whose radiation
# component is Dudhia shortwave, and the doors that used to default to one.
# ---------------------------------------------------------------------------

def test_every_no_radiation_named_template_says_what_it_actually_runs():
    """The name lies; the warning has to stop it from lying silently.

    ``wsm6-ysu-mm5-noah-no-radiation-v1`` was maturity ``supported`` with
    NO ``warnings`` key at all, and ``supported`` is one of
    ``warning_policy.nonwarning_maturities`` -- so selecting it produced
    nothing, at any door, while docs/public/STREAMING.md handed it to
    users in a copy-paste plan.  Recomputed from the registry, not from a
    list here: a seventh template named this way must fail this.
    """
    from woof.physics_registry import physics_registry

    registry = physics_registry()
    named = sorted(template_id for template_id in registry["templates"]
                   if "no-radiation" in template_id)
    assert len(named) == 6, named
    for template_id in named:
        template = registry["templates"][template_id]
        # The premise: the name is wrong because the component is Dudhia.
        assert template["components"]["radiation"] == "dudhia-shortwave"
        warnings = template.get("warnings", [])
        naming = [text for text in warnings
                  if "THE NAME IS WRONG" in text]
        assert len(naming) == 1, (template_id, warnings)
        text = naming[0]
        assert "dudhia-shortwave" in text
        assert "ra_lw_physics 0 with ra_sw_physics 1" in text
        assert ASYMMETRIC_RADIATION_NOCTURNAL_ACK in text
        # The label never lied, and the warning says to read it.
        assert "Dudhia SW" in template["label"]


def test_the_naming_warning_reaches_a_resolution_of_a_supported_template():
    """Emitted, not merely stored -- on the one whose maturity is silent."""
    from woof.physics_registry import physics_registry, validate_physics_plan

    template_id = WSM6_PROFILE_ID
    registry = physics_registry()
    assert registry["templates"][template_id]["maturity"] == "supported"
    assert "supported" in registry["warning_policy"]["nonwarning_maturities"]

    result = validate_physics_plan({
        "schema": registry["plan_schema"],
        "runner": "tools.prepared_single_domain_forecast",
        "source": "gfs",
        "domains": [{"grid_id": 1, "template_id": template_id}],
    })
    spoken = [warning["message"] for warning in result["warnings"]
              if "THE NAME IS WRONG" in warning["message"]]
    assert len(spoken) == 1, result["warnings"]


def test_no_shipped_door_defaults_to_an_asymmetric_pairing():
    """PHYSICS.md says this in as many words; here it is, measured.

    The storm-nowcast product made that sentence false through 1.8.7:
    ``tools/da_nowcast.py`` and its two sibling doors defaulted to
    ``wsm6-ysu-mm5-noah-no-radiation-v1`` on cases that are mostly
    nocturnal.
    """
    from woof.domain_interactive import DEFAULT_PHYSICS_PROFILE_BY_SOURCE
    from tools.da_nowcast import NOWCAST_DEFAULT_PHYSICS_PROFILE

    doors = {
        "woof domain (era5)": resolved_physics_profile("era5", None),
        "woof domain (gfs)": resolved_physics_profile("gfs", None),
        "woof domain (hrrr)": resolved_physics_profile("hrrr", None),
        "woof domain --interactive (era5)":
            DEFAULT_PHYSICS_PROFILE_BY_SOURCE["era5"],
        "woof domain --interactive (gfs)":
            DEFAULT_PHYSICS_PROFILE_BY_SOURCE["gfs"],
        "woof domain --interactive (hrrr)":
            DEFAULT_PHYSICS_PROFILE_BY_SOURCE["hrrr"],
        "tools.da_nowcast": NOWCAST_DEFAULT_PHYSICS_PROFILE,
    }
    asymmetric = {}
    for door, profile in doors.items():
        switches = single_domain_runtime_switches(profile)
        if int(switches["ra_sw_physics"]) > 0 \
                and int(switches["ra_lw_physics"]) == 0:
            asymmetric[door] = profile
    assert asymmetric == {}, asymmetric
    # The instrument, proved able to fire: the retired default IS the
    # shape this test looks for, so a door that went back to it fails.
    retired = single_domain_runtime_switches(WSM6_PROFILE_ID)
    assert int(retired["ra_sw_physics"]) > 0
    assert int(retired["ra_lw_physics"]) == 0
    # And the profile that replaced it resolves through the gate the
    # nowcast's own runner applies, on every source it can run on --
    # a default the runner would refuse is not a fix.
    from woof.prepared_single_domain_forecast import (
        SUPPORTED_SOURCES, _profile_runtime_switches)

    for source in sorted(SUPPORTED_SOURCES):
        switches = _profile_runtime_switches(
            source, NOWCAST_DEFAULT_PHYSICS_PROFILE)
        assert int(switches["ra_lw_physics"]) == 4, source
        assert int(switches["ra_sw_physics"]) == 4, source
        # No cumulus: the nowcast runs convection-permitting grids.
        assert int(switches["cu_physics"]) == 0, source


def test_the_nowcast_doors_all_bind_the_same_default():
    """One owner for the number, because 1.7.1 proved the alternative."""
    from tools.da_nowcast import NOWCAST_DEFAULT_PHYSICS_PROFILE
    from tools.da_nowcast_auto import (
        NOWCAST_DEFAULT_PHYSICS_PROFILE as auto_default)

    assert auto_default == NOWCAST_DEFAULT_PHYSICS_PROFILE
    assert NOWCAST_DEFAULT_PHYSICS_PROFILE == ROUTE_DEFAULT_PHYSICS_PROFILE
    import argparse
    from tools import da_nowcast, da_nowcast_auto, da_nowcast_launcher

    # Exercise each actual CLI registration, including the launcher sibling.
    root_parser = da_nowcast.build_parser()
    run_parser = next(action for action in root_parser._actions
                      if isinstance(action, argparse._SubParsersAction)).choices["run"]
    auto_parser = argparse.ArgumentParser()
    da_nowcast_auto.add_run_arguments(auto_parser)
    launcher_parser = argparse.ArgumentParser()
    da_nowcast_launcher.add_common(launcher_parser)
    for parser in (run_parser, auto_parser, launcher_parser):
        assert parser.get_default("physics_profile") == NOWCAST_DEFAULT_PHYSICS_PROFILE
    archived = "thompson-mp8-ysu-mm5-noah-rte-rrtmgp-v1"
    args = root_parser.parse_args([
        "run", "--site", "qqqq", "--window-end", "latest", "--out", "case",
        "--physics-profile", archived])
    assert args.physics_profile == archived


def test_nowcast_default_change_does_not_requalify_archived_measurements():
    """Existing measured records retain their actual physics composition."""
    import argparse
    import hashlib
    import json
    from pathlib import Path
    from tools import da_nowcast, da_nowcast_auto, da_nowcast_launcher

    archived = "wsm6-ysu-mm5-noah-no-radiation-v1"
    fixture = Path(__file__).resolve().parent / "fixtures/nowcast_archived_physics.json"
    payload = fixture.read_bytes()
    # 7db661c65, development branch: preserve the historical receipt
    # composition in public tests; release snapshots exclude evidence/**.
    assert hashlib.sha256(payload).hexdigest() == (
        "dfff21eccbb9309af3c3421e3952e740e4bc302eb400137566341457021255e6")
    provenance = json.loads(payload)
    assert provenance["schema"] == "gpuwm-nowcast.archived-physics-provenance.v1"
    assert len(provenance["receipts"]) == 2
    assert {receipt["physics_profile"] for receipt in provenance["receipts"]} == {archived}
    assert da_nowcast_launcher.REFERENCE["physics_profile"] == archived
    assert "does not qualify current" in da_nowcast_launcher.REFERENCE["caveat"]

    root_parser = da_nowcast.build_parser()
    run_parser = next(action for action in root_parser._actions
                      if isinstance(action, argparse._SubParsersAction)).choices["run"]
    auto_parser = argparse.ArgumentParser()
    da_nowcast_auto.add_run_arguments(auto_parser)
    for parser in (run_parser, auto_parser):
        help_text = " ".join(parser.format_help().split())
        assert "WSM6/Dudhia" in help_text
        assert "longwave off" in help_text
        assert "remeasurement" in help_text
