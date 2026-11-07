"""A memory refusal names only ways out that are not refused again.

When the smallest layout of a ladder does not fit the card, the wizard
names the levers that would move the refusal: a lighter physics suite, a
shallower ladder, a larger card.  Each named lever has to be one the same
door then accepts.  A suite that is cheaper than the refused one but
still over the same budget is not a way out: following it reproduces the
refusal that named it.
"""

from __future__ import annotations

import re

import pytest

from woof.cli import main as cli_main
from woof.core.preflight import EXTERNAL_MARGIN_BYTES, GIB
from woof.domain_wizard import (DEFAULT_PHYSICS_PROFILE,
                                 WIZARD_PHYSICS_PROFILES, LighterProfiles,
                                 _ladder_request_flags,
                                 _lighter_profiles_than,
                                 _minimum_layout_memory_remedy,
                                 profile_route_blocker,
                                 single_domain_runtime_switches)


def _advised(message: str) -> list[str]:
    match = re.search(r"a lighter --physics-profile \(([^)]+)\)", message)
    if match is None:
        return []
    return [name.strip() for name in match.group(1).split(",")]


# One 3 km domain on a small declared card: the default suite does not fit
# even at the minimum layout, and several lighter suites price below it,
# some of them still over the budget.
_SMALL_CARD_REQUEST = ["domain", "--point", "35.5,-97.5", "--root-dx", "3",
                       "--card", "5gb", "--hours", "1", "--source", "hrrr",
                       "--cycle", "2026-09-27T19"]


@pytest.mark.parametrize("choices", [
    None,
    # Scheme choices stay on the request that follows the advice, over the
    # suite it names, so a suite is priced with them.  The microphysics
    # row is aerosol-aware Thompson: since 4193eb0da the HRRR default is
    # the gsd41 template, whose scalar_pblmix=1 read Thompson-aerosol
    # fields only until U-E4, so any other microphysics over it was refused
    # by name before any pricing (2f65da582); the row's question, a scheme
    # pinned over every lighter suite, is asked with the HRRR scheme.
    '{"microphysics": "thompson-aerosol-mp28"}',
    '{"cumulus": "off"}',
])
def test_every_advised_suite_is_admitted_on_the_same_card(
        tmp_path, capsys, choices):
    """Follow each suite the refusal names; none may be refused again."""

    request = [*_SMALL_CARD_REQUEST,
               *([] if choices is None else ["--physics-choices", choices])]
    refused = tmp_path / "refused.toml"
    rc = cli_main([*request, "--out", str(refused)])
    captured = capsys.readouterr()
    message = captured.err + captured.out
    # Watched firing: the minimum-layout memory refusal, not another gate.
    assert rc == 2, message
    assert "does not fit" in message and "minimum layout" in message, message
    assert not refused.exists()
    advised = _advised(message)
    if not advised:
        assert "no lighter shipped --physics-profile fits" in message, message
    for rank, suite in enumerate(advised):
        out = tmp_path / f"advised-{rank}.toml"
        rc = cli_main([*request, "--physics-profile", suite,
                       "--out", str(out)])
        captured = capsys.readouterr()
        follow = captured.err + captured.out
        assert rc == 0 and out.exists(), (
            f"the refusal advised {suite} at #{rank + 1} and the same "
            f"request with it was refused: {follow}")
    # A single-domain ladder at its minimum layout has no shallower
    # ladder to fall back to, so that lever is not named.
    assert "shallower ladder" not in message
    assert "a larger card (this suite needs about" in message


def _named_shallower(message: str) -> str | None:
    match = re.search(r"a shallower ladder \(--ladder (\S+) fits", message)
    return None if match is None else match.group(1)


def test_a_named_shallower_ladder_is_the_deepest_one_that_fits(
        tmp_path, capsys):
    """The ladder lever names the deepest shallower ladder the card fits.

    Across cards that refuse the deepest preset at its minimum layout,
    the shallower ladder named is exactly the deepest shallower preset
    the same request then admits, and none is named when every one of
    them is refused again.
    """

    # The suite is named, with the grid deciding its cumulus as the
    # unnamed default does, so every ladder of the walk prices one suite:
    # this is the lever for a suite the user chose.  The unnamed default
    # is keyed on grid spacing, and its own lever is the test below.
    from woof.physics_menu import default_profile_for

    suite = ("--physics-profile", default_profile_for("era5"),
             "--cumulus", "grid")

    def _run(ladder: str, vram: str, out) -> tuple[int, str]:
        rc = cli_main(["domain", "--point", "39.1,-94.6", "--ladder", ladder,
                       "--vram-gib", vram, "--hours", "6", *suite,
                       "--cycle", "2026-08-12T00", "--out", str(out)])
        captured = capsys.readouterr()
        return rc, captured.err + captured.out

    shallower_presets = ("12-3-1", "12-3", "12")
    named_any = False
    # 4.6, 5 and 5.5 GiB since the 3.0.0 kernel set: its grid-independent
    # envelope (2.88 GiB + the 0.50 GiB margin) leaves a 4 or 4.5 GiB card
    # no budget for any layout, which is a different refusal; these three
    # still reach the minimum layout and name 12, 12 and 12-3-1.
    for vram in ("4.6", "5", "5.5"):
        rc, message = _run("12-3-1-0.5", vram, tmp_path / f"deep-{vram}.toml")
        # Watched firing: the minimum-layout memory refusal on every card.
        assert rc == 2, message
        assert "minimum layout" in message, message
        named = _named_shallower(message)
        deepest_admitted = None
        for ladder in shallower_presets:
            follow_rc, _follow = _run(ladder, vram,
                                      tmp_path / f"{ladder}-{vram}.toml")
            if follow_rc == 0:
                deepest_admitted = ladder
                break
        assert named == deepest_admitted, (
            f"on a {vram} GiB card the refusal named {named!r} as the "
            f"shallower ladder, and the deepest the same request admits "
            f"is {deepest_admitted!r}: {message}")
        named_any = named_any or named is not None
    assert named_any, "no card in the walk named a shallower ladder"


def _admissible_lighter(source: str) -> list[str]:
    names = []
    for name in WIZARD_PHYSICS_PROFILES:
        if name == DEFAULT_PHYSICS_PROFILE:
            continue
        try:
            single_domain_runtime_switches(name)
        except ValueError:
            continue
        if profile_route_blocker(name, source) is None:
            names.append(name)
    return names


def test_lighter_suites_are_filtered_by_the_budget():
    """Cheaper is necessary, not sufficient: the suite must fit the budget."""

    source = "gfs"
    candidates = _admissible_lighter(source)
    assert len(candidates) >= 4
    prices = {name: (index + 1) * GIB // 4
              for index, name in enumerate(candidates)}
    prices[DEFAULT_PHYSICS_PROFILE] = (len(candidates) + 1) * GIB // 4
    price = prices.get

    budget = 3 * GIB // 4
    result = _lighter_profiles_than(DEFAULT_PHYSICS_PROFILE, source, price,
                                    budget_bytes=budget)
    assert result.fitting, "suites priced within the budget must be named"
    assert all(prices[name] <= budget for name in result.fitting)
    # Smallest step down first, among the suites that fit.
    assert list(result.fitting) == sorted(
        result.fitting, key=prices.__getitem__, reverse=True)
    assert result.lightest == (candidates[0], prices[candidates[0]])
    assert result.compared

    nothing = _lighter_profiles_than(DEFAULT_PHYSICS_PROFILE, source, price,
                                     budget_bytes=GIB // 8)
    assert nothing.fitting == ()
    assert nothing.lightest == (candidates[0], prices[candidates[0]])


def test_remedy_says_when_no_lighter_suite_fits_and_names_the_card():
    envelope = int(4.28 * GIB)
    budget = int(3.75 * GIB)
    free = budget + EXTERNAL_MARGIN_BYTES
    text = _minimum_layout_memory_remedy(
        lighter=LighterProfiles(lightest=("suite-a", int(4.0 * GIB))),
        envelope_bytes=envelope, free_bytes=free, budget_bytes=budget,
        source="gfs", shallower=None)
    assert _advised(text) == []
    assert "no lighter shipped --physics-profile fits this budget" in text
    assert "suite-a, needs 4.00 GiB here, 0.25 GiB over" in text
    need = (envelope + EXTERNAL_MARGIN_BYTES) / GIB
    assert f"needs about {need:.2f} GiB free" in text
    assert f"presents about {free / GIB:.2f} GiB" in text
    assert "shallower ladder" not in text


def test_remedy_says_when_no_suite_prices_below_the_refused_one():
    budget = int(3.75 * GIB)
    text = _minimum_layout_memory_remedy(
        lighter=LighterProfiles(compared=True),
        envelope_bytes=int(4.28 * GIB),
        free_bytes=budget + EXTERNAL_MARGIN_BYTES, budget_bytes=budget,
        source="gfs", shallower=None)
    assert _advised(text) == []
    assert ("no lighter shipped --physics-profile fits this budget either: "
            "none that gfs can run prices below this one here") in text
    assert "a larger card" in text
    # Nothing was compared (a template's own suite): nothing is said of it.
    silent = _minimum_layout_memory_remedy(
        lighter=LighterProfiles(),
        envelope_bytes=int(4.28 * GIB),
        free_bytes=budget + EXTERNAL_MARGIN_BYTES, budget_bytes=budget,
        source="gfs", shallower=None)
    assert "physics-profile" not in silent


@pytest.mark.parametrize("shallower", [None, "--ladder 12"])
def test_remedy_names_fitting_suites_and_only_a_ladder_that_fits(shallower):
    budget = int(3.75 * GIB)
    text = _minimum_layout_memory_remedy(
        lighter=LighterProfiles(fitting=("suite-b", "suite-c"),
                                lightest=("suite-c", int(3.0 * GIB))),
        envelope_bytes=int(4.28 * GIB),
        free_bytes=budget + EXTERNAL_MARGIN_BYTES, budget_bytes=budget,
        source="gfs", shallower=shallower)
    assert _advised(text) == ["suite-b", "suite-c"]
    assert "no lighter" not in text
    if shallower is None:
        assert "shallower ladder" not in text
    else:
        assert f"a shallower ladder ({shallower} fits" in text
    assert "a larger card" in text


@pytest.mark.parametrize(("ratios", "root_dx_m", "flags"), [
    ((), 12000.0, "--ladder 12"),
    ((4, 3), 12000.0, "--ladder 12-3-1"),
    ((), 3000.0, "--root-dx 3"),
    ((3,), 3000.0, "--root-dx 3 --chain 3"),
    ((3, 2), 4500.0, "--root-dx 4.5 --chain 3,2"),
])
def test_a_named_ladder_is_given_as_the_flags_that_request_it(
        ratios, root_dx_m, flags):
    assert _ladder_request_flags(ratios, root_dx_m) == flags


def test_the_unnamed_default_names_the_shallower_ladder_its_own_default_fits(
        tmp_path, capsys):
    """Below 1 km the unnamed default is the spacing table's suite.

    It does not fit a 4.6 GiB card even at the 500 m ladder's minimum
    layout.  (Until dad24232a priced the Thompson kernels at the shallow
    exports nz <= 64 launches, its kernel set alone overflowed this card
    and the refusal said the ladder had no budget at all; it now prices
    the minimum layout and refuses that.)  A shallower ladder finishes at 1 km or coarser
    and binds the lighter source default, so it can be a way out: the
    refusal prices each shallower ladder with the suite it would run and
    names the deepest one whose minimum layout fits, and that is the
    deepest the same request then admits, instead of the refusal saying
    that no ladder can help.
    """

    def _run(*extra, out) -> tuple[int, str]:
        rc = cli_main(["domain", "--point", "39.1,-94.6", "--vram-gib", "4.6",
                       "--hours", "6", "--cycle", "2026-08-12T00", *extra,
                       "--out", str(out)])
        captured = capsys.readouterr()
        return rc, captured.err + captured.out

    rc, message = _run("--ladder", "12-3-1-0.5", out=tmp_path / "deep.toml")
    assert rc == 2, message
    assert ("ladder 12-3-1-0.5 does not fit" in message
            and "even at the minimum layout" in message), message
    assert "no smaller layout on any ladder can help" not in message
    named = _named_shallower(message)
    deepest_admitted = None
    for ladder in ("12-3-1", "12-3", "12"):
        follow_rc, _follow = _run("--ladder", ladder,
                                  out=tmp_path / f"{ladder}.toml")
        if follow_rc == 0:
            deepest_admitted = ladder
            break
    assert deepest_admitted is not None
    assert named == deepest_admitted, message


def test_the_source_default_is_named_first_when_it_fits():
    """The refused suite is the door's spacing default; the source's own fits.

    It rides in the fitting list ahead of the rest and is returned alone,
    and the sentence names it as the source's own default, apart from the
    lighter suites, so a reader is not left choosing among no-radiation
    and validation suites when the suite every coarser grid runs fits.
    """

    source = "gfs"
    candidates = _admissible_lighter(source)
    refused = candidates[-1]
    preferred = candidates[0]
    prices = {name: (index + 1) * GIB // 4
              for index, name in enumerate(candidates)}
    budget = (len(candidates) - 1) * GIB // 4
    result = _lighter_profiles_than(refused, source, prices.get,
                                    budget_bytes=budget, preferred=preferred)
    assert result.preferred == preferred
    assert result.fitting[0] == preferred and len(result.fitting) == 3
    # The heaviest that fit follow it, as without a preferred suite.
    plain = _lighter_profiles_than(refused, source, prices.get,
                                   budget_bytes=budget)
    assert plain.preferred is None
    assert result.fitting[1:] == plain.fitting[:2]

    text = _minimum_layout_memory_remedy(
        lighter=result, envelope_bytes=int(prices[refused]),
        free_bytes=budget + EXTERNAL_MARGIN_BYTES, budget_bytes=budget,
        source=source, shallower=None)
    assert (f"--source {source}'s own default suite (--physics-profile "
            f"{preferred}, which fits)") in text
    assert _advised(text) == list(result.fitting[1:])

    # One that does not fit is not named, as any other suite would not be.
    over = _lighter_profiles_than(refused, source, prices.get,
                                  budget_bytes=prices[preferred] - 1,
                                  preferred=preferred)
    assert over.preferred is None and preferred not in over.fitting


def test_a_small_card_names_the_source_default_and_prints_no_candidate_warnings(
        tmp_path, capsys):
    """5.75 GiB, the 500 m ladder, no suite named.

    (5.5 GiB until the 3.0.0 kernel set's grid-independent envelope grew:
    the source default no longer fits 5.5 GiB either, so 5.75 GiB is now
    the card where the sub-km default is refused and the source's own
    default fits, measured on fix/seal4-uh.)

    (6 GiB until dad24232a priced the Thompson kernels at the shallow
    exports nz <= 64 launches: the sub-km default's 4.68 GiB envelope now
    fits a 6 GiB card's 4.75 GiB budget, and 5.5 GiB is the card this
    refusal is still about.)

    The sub-km default does not fit the card at the ladder's minimum
    layout.  The refusal names the source's own default, which fits, and
    the same
    request naming it is admitted; the configurations the refusal priced
    and did not write say nothing on the screen.
    """

    from woof.physics_menu import default_profile_for

    request = ["domain", "--point", "39.1,-94.6", "--vram-gib", "5.75",
               "--ladder", "12-3-1-0.5", "--source", "gfs", "--hours", "6",
               "--cycle", "2026-08-12T00"]
    rc = cli_main([*request, "--out", str(tmp_path / "refused.toml")])
    captured = capsys.readouterr()
    message = captured.err + captured.out
    assert rc == 2, message
    assert "12-3-1-0.5" in message and "a larger card" in message, message
    own = default_profile_for("gfs")
    assert (f"--source gfs's own default suite (--physics-profile {own}, "
            "which fits)") in message, message
    assert "warning:" not in captured.err, captured.err
    out = tmp_path / "own.toml"
    rc = cli_main([*request, "--physics-profile", own, "--out", str(out)])
    follow = capsys.readouterr()
    assert rc == 0 and out.exists(), follow.err + follow.out
