"""What each front door OFFERS, and whether its own route can run it.

Four defects sat behind one question nothing asked: does a suite a door
lists survive the route that door drives?  The nowcast door offered every
suite the single-domain runner resolves, the configuration door it calls
first offered a shorter list, and the namelist that door writes could not
be read back as the configuration beside it for three separate reasons --
an unimplemented cumulus selector, an ArWen-only boundary-layer selector,
and a turbulence row the renderer never authored.  A fourth suite class
ran on the route that reads the TOML and was refused on the route that
reads the namelist, because the declaration that admits it has no
namelist spelling.

Every test here is about the offer, not about a scheme: they enumerate
the offered set and hold the whole set to one rule, so a suite registered
tomorrow is covered by arithmetic rather than by an edit.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import tomllib

from woof.cli import main as cli_main
from woof.experiment import load_experiment
from woof.hrrr_route_inputs import route_input_paths, verify_round_trip
from woof.physics_compat import (CONSTANT_DOWNWARD_LONGWAVE_ACK,
                                  profile_declared_acknowledgements,
                                  single_domain_runtime_switches)

from woof.physics_menu import shipped_profiles

#: A daylight window, so a shortwave-only suite is emitted on its own
#: terms rather than on a nocturnal declaration the operator would have
#: to make.  The nocturnal guard is a claim about the WINDOW and is a
#: different question from the one these tests ask.
_CYCLE = "2026-07-29T18"
_HOURS = "3"
_POINT = "--point=41.6,-90.6"


def _choices_of(parser, option: str) -> tuple[str, ...]:
    """The choices a built parser offers for ``option``, subcommands too."""

    for action in parser._actions:
        if option in action.option_strings:
            return tuple(action.choices or ())
    for action in parser._actions:
        choices = getattr(action, "choices", None)
        if not hasattr(choices, "values"):
            # A plain value list, not a subcommand table.
            continue
        for sub in choices.values():
            if hasattr(sub, "_actions"):
                found = _choices_of(sub, option)
                if found:
                    return found
    return ()


def _wizard_offers() -> tuple[str, ...]:
    from woof.cli import build_parser

    return _choices_of(build_parser(), "--physics-profile")


def test_the_configuration_door_offers_every_shipped_suite():
    """What the configuration door offers is the one derived table.

    A storm-nowcast door read the same table until gpuwm 3.0.3 removed it
    with the local data assimilation; a name one door offered and another
    refused was a run that died in argument parsing with no physics reached.
    """

    wizard = _wizard_offers()
    assert wizard, "the configuration door offers no suite at all"
    assert set(wizard) == set(shipped_profiles())


def test_every_offered_suite_resolves_a_runtime_product():
    """What is offered is what the runner that runs it accepts."""

    for profile in shipped_profiles():
        switches = single_domain_runtime_switches(profile)
        assert switches.get("mp_physics") is not None, profile


def _emit(profile: str, out: Path) -> int:
    return cli_main(["domain", _POINT, "--card", "24gb",
                     "--root-dx", "3", "--source", "hrrr",
                     "--cycle", _CYCLE, "--hours", _HOURS,
                     "--physics-profile", profile, "--out", str(out)])


@pytest.mark.parametrize("profile", list(shipped_profiles()))
def test_an_offered_suite_emits_a_set_that_reads_back_as_itself(
        profile, tmp_path, capsys, monkeypatch):
    """Emit, import, compare -- over every suite the doors offer.

    This is the check the route already runs at emission
    (``verify_round_trip``); what was missing was running it over the
    whole offered set, which is where the renderer's unauthored
    turbulence row, the importer's two unmapped selectors and the
    unstated RRTMG lineage were all hiding.
    """
    from woof import capabilities

    # Configuration emission launches no kernels.  The capability-refusal
    # tests own missing module presence; this test owns the round trip.
    installed = capabilities.is_installed
    monkeypatch.setattr(capabilities, "is_installed",
                        lambda module: module == "cupy" or installed(module))

    out = tmp_path / "offer.toml"
    rc = _emit(profile, out)
    printed = capsys.readouterr()
    assert rc == 0, printed.out + printed.err
    exp = load_experiment(out)
    for key, value in single_domain_runtime_switches(profile).items():
        assert getattr(exp.root.run, key) == value, (profile, key)
    paths = route_input_paths(out)
    verify_round_trip(exp, paths["wps_namelist"], paths["namelist_input"])


@pytest.mark.parametrize("profile", list(shipped_profiles()))
def test_the_namelist_route_reads_the_named_suites_own_declaration(
        profile, tmp_path, capsys, monkeypatch):
    """The route that reads the namelist resolves what the TOML route does.

    A shortwave-only suite states, of itself, that its land surface
    integrates a declared constant downward longwave.  The configuration
    door writes that into the case TOML and WRF's namelist has no field
    for it, so the namelist-routed preparation refused exactly the suites
    the TOML-routed one ran.  It now carries the declaration through the
    profile it names, and the same load guard reads it from the same
    array on both routes.
    """

    from woof.hrrr_configuration import resolve_root_experiment
    from woof.ingest.hrrr_target import load_hrrr_target_domain
    from woof.vertical_contract import explicit_vertical_from_wrf_namelist
    from woof import capabilities

    installed = capabilities.is_installed
    monkeypatch.setattr(capabilities, "is_installed",
                        lambda module: module == "cupy" or installed(module))

    out = tmp_path / "offer.toml"
    rc = _emit(profile, out)
    printed = capsys.readouterr()
    assert rc == 0, printed.out + printed.err
    paths = route_input_paths(out)
    target = load_hrrr_target_domain(paths["target_domain"])
    configured, _tables = resolve_root_experiment(
        target=target,
        vertical=explicit_vertical_from_wrf_namelist(
            paths["namelist_input"], expected_nz=target.nz,
            context="offered-suite route read"),
        namelist_input=paths["namelist_input"],
        start_time=None, run_seconds=None,
        wps_namelist=paths["wps_namelist"], physics_profile=profile)

    # The switches are bound by the route's own validator, which
    # resolve_root_experiment runs when a profile is named and which
    # raises rather than returning: reaching this line at all is that
    # check passing.  Restating it here would restate it WRONGLY, because
    # a cumulus-off suite's cudt cadence is reconciled where identities
    # are compared rather than carried through the namelist.
    declared = profile_declared_acknowledgements(profile)
    assert set(declared) <= set(configured.acknowledgements), profile

    # The declaration is the suite's, not the window's: a suite that
    # computes its own longwave declares nothing and inherits nothing.
    if not declared:
        assert CONSTANT_DOWNWARD_LONGWAVE_ACK not in \
            tuple(configured.acknowledgements), profile


def test_the_case_toml_and_the_namelist_route_agree_on_the_declaration(
        tmp_path, capsys):
    """One token, written by the configuration door, read on both routes."""

    profile = next(p for p in shipped_profiles()
                   if profile_declared_acknowledgements(p))
    out = tmp_path / "declared.toml"
    assert _emit(profile, out) == 0
    capsys.readouterr()
    raw = tomllib.loads(Path(out).read_text(encoding="utf-8"))
    assert CONSTANT_DOWNWARD_LONGWAVE_ACK in raw["experiment"][
        "acknowledgements"]
