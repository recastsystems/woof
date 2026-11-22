"""The one output-layering convention, and the ``woof setup`` wrapper.

Two complaints produced this module.  The CLI printed everything it
knew at once, so the next command was never findable; and a wheel
install still needed a list of staging chores nobody could infer the
order of.  The answers are one flag with one meaning everywhere, and
one command that runs the chores.

What these tests defend is the *contract*, not the wording: that both
halves of a layered message survive inside the exception, that the
default layer never hides a remedy, that every subcommand really does
take the flag the pointers name, and that setup never swallows a
refusal.
"""

from __future__ import annotations

import argparse

import pytest

from woof import doctor, explain, setup_cli
from woof.cli import main as cli_main


# ---------------------------------------------------------------------------
# The helper
# ---------------------------------------------------------------------------

def test_layered_keeps_both_halves_in_one_string():
    """Content contract: nothing a layered message says is thrown away.

    The refusals travel as ValueError through call chains this package
    does not own, and tests across the tree read ``str(error)``.
    Keeping both halves inside the one string is what lets the layer be
    chosen at the print boundary without changing what any of those
    readers see.
    """

    message = explain.layered("what happened\n  remedy: do this",
                              "because of the mechanism")
    assert "what happened" in message
    assert "remedy: do this" in message
    assert "because of the mechanism" in message

    action, why = explain.split(message)
    assert action == "what happened\n  remedy: do this"
    assert why == "because of the mechanism"


def test_an_unlayered_message_passes_through_untouched():
    """Most refusals are one sentence; layering must be a no-op on them."""

    assert explain.layered("just this", "") == "just this"
    assert explain.split("just this") == ("just this", "")
    for flag in (False, True):
        assert explain.render("just this", explain=flag,
                              command="woof x") == "just this"


def test_the_default_layer_keeps_the_remedy_and_names_the_flag():
    """The rule the whole convention rests on: a remedy never moves."""

    message = explain.layered("refused\n  remedy: pass --force",
                              "the mechanism paragraph")
    terse = explain.render(message, explain=False, command="woof fetch")
    assert "remedy: pass --force" in terse
    assert "the mechanism paragraph" not in terse
    assert "woof fetch --explain" in terse

    full = explain.render(message, explain=True, command="woof fetch")
    assert "remedy: pass --force" in full
    assert "the mechanism paragraph" in full


def test_the_sentinel_never_reaches_a_terminal_in_either_layer():
    """A marker a reader can see is a bug, whichever layer was asked for."""

    message = explain.layered("action", "why")
    for flag in (False, True):
        rendered = explain.render(message, explain=flag, command="woof x")
        assert explain.EXPLAIN_MARK.strip() not in rendered


def test_the_pointer_is_omitted_rather_than_guessed():
    """No command name, no pointer: a wrong one is worse than none."""

    message = explain.layered("action", "why")
    assert explain.render(message, explain=False, command=None) == "action"


# ---------------------------------------------------------------------------
# The flag really is everywhere the pointers claim
# ---------------------------------------------------------------------------

def _registered_subcommands() -> list[str]:
    """Every subcommand the real parser carries, read off the parser.

    Read rather than transcribed on purpose.  A hand-kept list is a
    sweep that shrinks silently: the day someone registers a
    subcommand and does not think about this file, the transcribed
    version keeps passing while the flag it is supposed to guarantee is
    missing from the new command -- which is exactly when the pointer
    starts lying.
    """

    from woof.cli import build_parser

    for action in build_parser()._actions:  # noqa: SLF001 - argparse only API
        if isinstance(action, argparse._SubParsersAction):
            return sorted(action.choices)
    raise AssertionError("woof's parser registers no subcommands")


def test_the_subcommand_sweep_covers_the_whole_surface():
    """Guard the guard: the enumeration must not come back empty."""

    names = _registered_subcommands()
    assert len(names) >= 15
    # A few anchors, so a parser that silently lost its registrars
    # cannot pass by returning some other non-empty list.
    for anchor in ("doctor", "setup", "domain", "fetch", "run"):
        assert anchor in names


@pytest.mark.parametrize("name", _registered_subcommands())
def test_every_subcommand_accepts_explain(capsys, name):
    """The pointer names ``woof <command> --explain`` for ANY command.

    That sentence is only true if the flag is on all of them, and this
    is what keeps a newly registered subcommand from quietly making it
    false.
    """

    with pytest.raises(SystemExit) as exit_info:
        cli_main([name, "--help"])
    assert exit_info.value.code == 0
    assert "--explain" in capsys.readouterr().out


def test_add_explain_flag_is_idempotent():
    """Two registrars share a parser; a second add must not be a crash."""

    parser = argparse.ArgumentParser()
    explain.add_explain_flag(parser)
    explain.add_explain_flag(parser)
    assert parser.parse_args([]).explain is False
    assert parser.parse_args(["--explain"]).explain is True


# ---------------------------------------------------------------------------
# Doctor: the same estate at two widths
# ---------------------------------------------------------------------------

def _fake(name, status, **kwargs):
    return doctor.Check(name, status, "detail text", **kwargs)


def test_the_terse_report_folds_a_shared_remedy_into_one_line():
    """Six identical remedies read as six problems; they are one."""

    checks = [
        _fake("bridge a", "missing", remedy="r",
              action="woof fetch-bridges", group="bridges"),
        _fake("bridge b", "missing", remedy="r",
              action="woof fetch-bridges", group="bridges"),
        _fake("bridge c", "missing", remedy="r",
              action="woof fetch-bridges", group="bridges"),
    ]
    brief = doctor.format_brief(checks)
    assert brief.count("woof fetch-bridges") == 1
    assert "bridges (3)" in brief
    # The full report still names every one of them.
    full = doctor.format_report(checks)
    for name in ("bridge a", "bridge b", "bridge c"):
        assert name in full


def test_folding_never_merges_across_status_or_remedy():
    """A fold is a presentation of sameness, never an assertion of it."""

    checks = [
        _fake("bridge a", "missing", action="woof fetch-bridges",
              group="bridges"),
        _fake("bridge b", "verified", group="bridges"),
        _fake("bridge c", "missing", action="cargo build", group="bridges"),
    ]
    brief = doctor.format_brief(checks)
    assert "woof fetch-bridges" in brief and "cargo build" in brief
    assert "(3)" not in brief


def test_the_terse_summary_points_at_setup_only_when_it_is_shorter():
    """One gap already printed its one command; a wrapper adds a step."""

    one = doctor.format_brief(
        [_fake("thompson tables", "missing", action="woof fetch-tables")])
    assert "woof setup" not in one

    both = doctor.format_brief([
        _fake("bridge a", "missing", action="woof fetch-bridges"),
        _fake("thompson tables", "missing", action="woof fetch-tables"),
    ])
    assert "woof setup" in both
    assert "woof fetch-bridges" in both and "woof fetch-tables" in both


def test_a_published_bundle_makes_the_fresh_install_summary_name_setup(
        monkeypatch):
    """The moment a PyPI user actually meets, end to end.

    On a release that published a bundle for the reader's platform,
    every Rust gap resolves to one download and the table gap to
    another -- which is exactly the pair ``woof setup`` runs.  This
    pins that the terse report reaches that conclusion rather than
    leaving the reader to notice it, and that the per-line commands are
    still there for anyone who would rather run them separately.
    """

    from woof import bridges

    monkeypatch.setattr(bridges, "prebuilt_bundle_offer",
                        lambda *a, **k: ("  woof fetch-bridges",))
    monkeypatch.setattr(bridges, "sources_present", lambda *a, **k: False)

    action = doctor._build_action()
    assert action == "woof fetch-bridges"

    brief = doctor.format_brief([
        _fake("bridge grib1_bridge", "missing", action=action,
              group=doctor._GROUP_BRIDGES),
        _fake("bridge gfs_grib2_bridge", "missing", action=action,
              group=doctor._GROUP_BRIDGES),
        _fake("thompson tables", "missing", action="woof fetch-tables"),
        _fake("WPS_GEOG", "missing", action="woof fetch-geog"),
    ])
    assert "woof setup runs woof fetch-bridges then woof fetch-tables" \
        in brief
    # Geog is NOT claimed by setup: it is opt-in, and the summary must
    # not imply the wrapper closes a gap it deliberately leaves open.
    assert "-> woof fetch-geog" in brief
    assert "fetch-geog" not in brief.splitlines()[-2]


def test_every_actionable_gap_carries_a_next_command():
    """A MISSING line with no action is a dead end on the default layer.

    The full report can afford a remedy that is six lines of comments;
    the one-line form cannot, so every gap has to have named the single
    thing to do -- even when that thing is a sentence rather than a
    command, as it is for a path only the reader knows.
    """

    for check in doctor.collect_checks():
        if check.status == "missing":
            assert check.action, check.name
            assert check.action.strip() == check.action


#: One check of every status, with a multi-line remedy, rendered by
#: :func:`woof.doctor.format_report`.  Verified byte-for-byte equal to
#: what v1.2.0's format_report produced for the same input (compared
#: against `git show 39984b0e:woof/doctor.py`), which is what "the
#: --explain layer is preserved verbatim" has to mean if it means
#: anything.
#:
#: RENEGOTIATED ONCE, deliberately, in 2.3.3+: the summary sentence
#: gained a severity census (`-- 1 broken`).  That is this test working
#: as designed rather than an exception to it -- the reason for the
#: change is that "N gap(s), M of them blocking" could not distinguish
#: a ~16 GB download nobody opted into from a box that cannot run a
#: forecast, and doctor printed the identical sentence over both.
#: Everything else about the layer -- the label column, the ten-space
#: remedy gutter, the order, the closing clause -- is unchanged and
#: still pinned here.
_GOLDEN_FULL_REPORT = """\
woof doctor: runtime estate (WOOF-ArWen)
  ok      python: 3.13 on this machine
  present manifest: schema only, so presence-only
  info    root: not set
  MISSING bridge x: not staged
          remedy: woof fetch-bridges
                  # one download, verified against the packaged pins
                  # before anything is staged.
woof doctor: 1 gap(s), 1 of them blocking (the exit code is 1) -- 1 broken.  \
Every remedy line above is either a command to run as printed, in the order \
printed, or a '#' comment."""


def test_the_explain_layer_still_renders_exactly_what_it_always_did():
    """The verbatim promise, pinned as output rather than as intent.

    ``--explain`` exists to hand back the long form unchanged.  A
    docstring saying so is not a guarantee; this is.  If a future edit
    reflows the label column, the ten-space remedy gutter, or the
    closing sentence, this fails and the promise gets renegotiated
    deliberately instead of quietly.
    """

    checks = [
        doctor.Check("python", "verified", "3.13 on this machine"),
        doctor.Check("manifest", "present", "schema only, so presence-only"),
        doctor.Check("root", "info", "not set"),
        doctor.Check(
            "bridge x", "missing", "not staged",
            "woof fetch-bridges\n"
            "  # one download, verified against the packaged pins\n"
            "  # before anything is staged.",
            action="woof fetch-bridges", brief="not staged",
            group="bridges"),
    ]
    assert doctor.format_report(checks) == _GOLDEN_FULL_REPORT


def test_the_terse_layer_reports_the_same_findings_as_the_full_one():
    """Two widths, one estate: neither layer may invent or lose a gap."""

    checks = doctor.collect_checks()
    brief = doctor.format_brief(checks)
    full = doctor.format_report(checks)

    gaps = sum(1 for check in checks if check.status == "missing")
    if gaps:
        assert f"{gaps} gap(s)" in brief
        assert f"{gaps} gap(s)" in full
    else:
        assert "no gaps" in brief and "no gaps" in full


def test_the_json_layer_carries_both_widths():
    """A front end reading --json gets the terse fields and the full ones."""

    payload = doctor.collect_checks()[0].__dict__
    for field in ("name", "status", "detail", "remedy", "action", "brief",
                  "group"):
        assert field in payload


# ---------------------------------------------------------------------------
# woof setup
# ---------------------------------------------------------------------------

def test_setup_takes_each_steps_own_defaults_not_a_transcription():
    """The wrapper must not become a second declaration of the defaults."""

    from woof import geog_assets

    geog = setup_cli.step_namespace("woof.geog_assets")
    # fetch-geog's own default, read off its parser: "default" (the WRF
    # and mesh set) since 86a30395d made the dust datasets opt-in.
    assert geog.datasets == "default"
    assert geog.bundle is False
    assert geog.root is None
    # The handler travels with the namespace, so there is no name to drift.
    assert geog.func is geog_assets.fetch_geog_main

    bridges_ns = setup_cli.step_namespace("woof.bridge_assets")
    assert bridges_ns.from_dir is None and bridges_ns.list is False


def test_setup_runs_the_steps_in_order_and_prints_one_line_each(
        monkeypatch, capsys):
    calls = []

    def fake(module_name, overrides):
        calls.append(module_name)
        return 0, f"{module_name}: chatty\nsecond line\n"

    monkeypatch.setattr(setup_cli, "_run_step", fake)
    monkeypatch.setattr("woof.doctor.collect_checks", lambda: [])

    rc = setup_cli.setup_main(argparse.Namespace(
        with_geog=False, from_dir=None, explain=False))
    printed = capsys.readouterr().out
    assert rc == 0
    assert calls == ["woof.bridge_assets", "woof.table_assets"]
    assert "  ok      bridges" in printed
    assert "  ok      tables" in printed
    # A succeeding step's chatter stays in its own command's output.
    assert "chatty" not in printed


def test_a_bare_setup_says_what_it_did_not_stage(monkeypatch, capsys):
    """N19: WPS_GEOG is excluded from ``setup`` by design, and a bare
    run never said so -- the tree surfaced as a refusal at prep time,
    several commands after the one that would have staged it.

    The size is the surprise, so the size is in the line, next to the
    command that closes the gap -- and it is the pin table's own figure,
    because a literal here understated the download by most of its size
    the day a dataset was added.
    """

    from woof.geog_assets import size_phrase

    monkeypatch.setattr(
        setup_cli, "_run_step", lambda module_name, overrides: (0, ""))
    monkeypatch.setattr("woof.doctor.collect_checks", lambda: [])

    setup_cli.setup_main(argparse.Namespace(
        with_geog=False, from_dir=None, explain=False))
    printed = capsys.readouterr().out
    assert "WPS_GEOG" in printed
    assert size_phrase() in printed
    assert "woof fetch-geog" in printed
    assert "--with-geog" in printed


def test_setup_with_geog_does_not_also_say_it_skipped_it(
        monkeypatch, capsys):
    """The note is about what was NOT staged, so asking for it removes
    the note rather than adding a contradiction."""

    monkeypatch.setattr(
        setup_cli, "_run_step", lambda module_name, overrides: (0, ""))
    monkeypatch.setattr("woof.doctor.collect_checks", lambda: [])

    setup_cli.setup_main(argparse.Namespace(
        with_geog=True, from_dir=None, explain=False))
    printed = capsys.readouterr().out
    assert setup_cli.GEOG_SIZE_NOTICE in printed
    assert "not staged" not in printed


def test_setup_explain_replays_every_steps_own_output(monkeypatch, capsys):
    monkeypatch.setattr(
        setup_cli, "_run_step",
        lambda module_name, overrides: (0, "the full receipt\n"))
    monkeypatch.setattr("woof.doctor.collect_checks", lambda: [])

    setup_cli.setup_main(argparse.Namespace(
        with_geog=False, from_dir=None, explain=True))
    assert "the full receipt" in capsys.readouterr().out


def test_setup_never_swallows_a_refusal(monkeypatch, capsys):
    """A wrapper that hides why a step refused is worse than no wrapper."""

    def fake(module_name, overrides):
        if module_name.endswith("bridge_assets"):
            return 2, "woof fetch-bridges: REFUSED: no bundle for this\n"
        return 0, ""

    monkeypatch.setattr(setup_cli, "_run_step", fake)
    monkeypatch.setattr("woof.doctor.collect_checks", lambda: [])

    rc = setup_cli.setup_main(argparse.Namespace(
        with_geog=False, from_dir=None, explain=False))
    printed = capsys.readouterr().out
    # Warn-not-block: the estate has no blocking gaps, so the refused
    # step is a note, not a failure -- but its output still prints
    # verbatim (a wrapper that hides why a step refused is worse than
    # no wrapper).
    assert rc == 0
    assert "FAILED  bridges" in printed
    assert "REFUSED: no bundle for this" in printed
    assert "already complete" in printed
    # And the independent step still ran.
    assert "  ok      tables" in printed

    # KEEP-HARD negative: the same refused step against an estate with
    # a real blocking gap keeps the step's exit code.
    from woof.doctor import Check
    monkeypatch.setattr(
        "woof.doctor.collect_checks",
        lambda: [Check("bridge gfs_grib2_bridge", "missing",
                       "not built yet")])
    rc = setup_cli.setup_main(argparse.Namespace(
        with_geog=False, from_dir=None, explain=False))
    assert rc == 2


def test_a_step_that_raises_keeps_what_it_had_already_printed(
        monkeypatch, capsys):
    """A download that died most of the way through IS the diagnosis.

    Letting the exception escape _run_step would have discarded the
    step's captured output along with it, leaving the reader a bare
    traceback where the useful part was the four lines above it.
    """

    def exploding(namespace):
        print("woof fetch-tables: downloading freezeH2O.dat (243 MiB)")
        raise OSError("connection reset by peer")

    monkeypatch.setattr(
        setup_cli, "step_namespace",
        lambda module_name: argparse.Namespace(func=exploding))
    from woof.doctor import Check
    monkeypatch.setattr(
        "woof.doctor.collect_checks",
        lambda: [Check("thompson tables", "missing", "not staged")])

    rc = setup_cli.setup_main(argparse.Namespace(
        with_geog=False, from_dir=None, explain=False))
    printed = capsys.readouterr().out
    # The step died AND the estate has a blocking gap: the failure
    # stands, with everything the step printed kept above it.
    assert rc == 2
    assert "downloading freezeH2O.dat" in printed
    assert "OSError: connection reset by peer" in printed
    # The estate report still prints: after a partial setup, "where do
    # I stand" is exactly the question.
    assert "woof doctor" in printed


def test_setup_does_not_fetch_geog_unless_asked_and_prints_the_size(
        monkeypatch, capsys):
    """Tens of GB is a decision, not a side effect of typing ``setup``."""

    from woof.geog_assets import size_phrase

    seen = []

    def fake(module_name, overrides):
        seen.append(module_name)
        return 0, ""

    monkeypatch.setattr(setup_cli, "_run_step", fake)
    monkeypatch.setattr("woof.doctor.collect_checks", lambda: [])

    setup_cli.setup_main(argparse.Namespace(
        with_geog=False, from_dir=None, explain=False))
    printed = capsys.readouterr().out
    assert "woof.geog_assets" not in seen
    # The bare run must not announce a download it is not going to make.
    # It DOES report the tree as not staged, with the same size, which is
    # a different sentence -- so this asserts on the notice rather than
    # on the number appearing anywhere (N19).
    assert setup_cli.GEOG_SIZE_NOTICE not in printed
    assert "will download" not in printed

    seen.clear()
    setup_cli.setup_main(argparse.Namespace(
        with_geog=True, from_dir=None, explain=False))
    printed = capsys.readouterr().out
    assert "woof.geog_assets" in seen
    assert size_phrase() in printed
    # The size is announced BEFORE the first byte moves.
    assert printed.index(size_phrase()) < printed.index("  ok      bridges")


def test_setup_is_registered_and_dispatches_through_the_real_cli(capsys):
    """cli._dispatch carries a hardcoded name list; this is its gate."""

    with pytest.raises(SystemExit) as exit_info:
        cli_main(["setup", "--help"])
    assert exit_info.value.code == 0
    printed = capsys.readouterr().out
    assert "--with-geog" in printed and "--explain" in printed


# ---------------------------------------------------------------------------
# The physics refusal, at both widths
# ---------------------------------------------------------------------------

def test_a_physics_refusal_keeps_its_rule_and_defers_its_mechanism():
    from woof.physics_compat import require_ready_wrf_physics

    with pytest.raises(ValueError) as error_info:
        require_ready_wrf_physics(
            mp_physics=8, sf_sfclay_physics=5, bl_pbl_physics=1,
            sf_surface_physics=2, num_soil_layers=4)
    message = str(error_info.value)

    terse = explain.render(message, explain=False, command="woof run")
    assert "WRF v4.6.1 PBL/surface-layer compatibility" in terse
    assert "WRF v4.6.1 refuses this pairing" in terse
    assert "no substitutions were applied" in terse
    # The mechanism waits to be asked for.
    assert "phys/module_physics_init.F:" not in terse
    assert "woof run --explain" in terse

    full = explain.render(message, explain=True, command="woof run")
    assert "phys/module_physics_init.F:3213-3219,3699-3701" in full
    # ArWen's OWN breakage travels with WRF's citation, in the mechanism
    # half: a refusal that says only "WRF refuses this" tells a user which
    # authority to argue with rather than what would go wrong here.
    assert "fm/fh" in full


def test_the_noahmp_budget_is_a_warning_not_a_blocker(capsys):
    """Warn-not-block: an unmeasured grid WIDTH is a performance
    projection, not a correctness gap.  One warning line carries the
    projection; the env variable that silences it stays named under
    --explain.  No blocker is raised."""

    from woof.physics_compat import (
        NOAHMP_EXPERT_COLUMN_BUDGET_ENV, pending_wrf_physics_components)

    blockers = pending_wrf_physics_components(
        mp_physics=8, sf_sfclay_physics=1, bl_pbl_physics=1,
        sf_surface_physics=4, num_soil_layers=4, columns=10_000_000)
    assert not [b for b in blockers
                if b.component == "Noah-MP column budget"]
    err = capsys.readouterr().err
    assert "warning:" in err
    assert "Noah-MP" in err and "slow, not wrong" in err

    explain.set_explain(True)
    try:
        pending_wrf_physics_components(
            mp_physics=8, sf_sfclay_physics=1, bl_pbl_physics=1,
            sf_surface_physics=4, num_soil_layers=4, columns=10_000_000)
        err = capsys.readouterr().err
        # The remedy is never withheld: --explain names the variable.
        assert NOAHMP_EXPERT_COLUMN_BUDGET_ENV in err
    finally:
        explain.set_explain(False)


# ---------------------------------------------------------------------------
# --explain belongs to ONE invocation
# ---------------------------------------------------------------------------

def test_a_front_door_gives_the_explain_flag_back_when_it_returns():
    """`--explain` must not outlive the command that asked for it.

    `_EXPLAIN_ACTIVE` is module state, because library code that calls
    `explain.warn` has no `args` in reach.  `woof.cli.main` is also an
    ordinary function that plenty of callers run in-process rather than
    by spawning: the console script, an embedder driving the CLI, and
    the test suite.  A door that stamped the flag and never gave it back
    therefore handed the flag to the NEXT invocation, whose warnings
    then printed a mechanism continuation nobody asked for.

    Measured before the fix: `pytest tests/test_doctor.py
    tests/test_fetch_engine_degrade_guard.py` failed with 2 stderr lines
    where the guard contracts 1, and the same two files in the opposite
    order passed -- so a battery could report this tree red or green on
    how it happened to pack its shards.
    """

    import woof.cli as cli_module

    assert explain._EXPLAIN_ACTIVE is False  # noqa: SLF001 - that IS the state
    inside = []
    with pytest.MonkeyPatch.context() as patch:
        # Dispatch stubbed so this reads the flag at the deepest point
        # the CLI sets it for, without running a subcommand.
        patch.setattr(
            cli_module, "_dispatch",
            lambda args: inside.append(explain._EXPLAIN_ACTIVE) or 0)  # noqa: SLF001
        assert cli_main(["version", "--explain"]) == 0
    assert inside == [True], (
        "--explain did not reach library code during the invocation that "
        "asked for it")
    assert explain._EXPLAIN_ACTIVE is False, (  # noqa: SLF001
        "woof.cli.main left --explain set after it returned; the next "
        "invocation in this interpreter inherits an explanation layer it "
        "never asked for")


def test_the_scope_restores_what_it_found_rather_than_clearing_it():
    """Nested doors: a scope hands its caller's layer back untouched."""

    with explain.explain_scope(True):
        assert explain._EXPLAIN_ACTIVE is True  # noqa: SLF001
        with explain.explain_scope(False):
            assert explain._EXPLAIN_ACTIVE is False  # noqa: SLF001
            explain.set_explain(True)
        assert explain._EXPLAIN_ACTIVE is True, (  # noqa: SLF001
            "the inner scope cleared its caller's layer instead of "
            "restoring it")
    assert explain._EXPLAIN_ACTIVE is False  # noqa: SLF001


def test_the_scope_restores_through_a_raised_exception():
    """A door that refuses still gives the flag back."""

    with pytest.raises(ValueError):
        with explain.explain_scope(True):
            raise ValueError("a refusal on the way out")
    assert explain._EXPLAIN_ACTIVE is False  # noqa: SLF001


def test_a_warning_after_an_explain_run_is_one_line_again(capsys):
    """The measured symptom itself, in one process.

    An `--explain` run followed by a bare `warn` printed the action line
    AND its `why` continuation, which is how
    tests/test_fetch_engine_degrade_guard.py counted 2 where it
    contracts 1. The contract is one line per warning unless THIS
    invocation asked otherwise.
    """

    import woof.cli as cli_module

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(cli_module, "_dispatch", lambda args: 0)
        assert cli_main(["version", "--explain"]) == 0
    capsys.readouterr()
    explain.warn("the transport degraded", "the mechanism prose")
    lines = [line for line in capsys.readouterr().err.splitlines() if line]
    assert lines == ["warning: the transport degraded"]
