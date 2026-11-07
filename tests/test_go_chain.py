"""``woof go``: the documented GFS chain, run without a human courier.

Five commands in a fixed order, three of them taking digests the
previous one printed.  ``go`` carries those values between stages by
reading the ARTIFACTS -- the manifest file, ``proof.json`` -- rather
than the printed prose, and runs the same commands a person would so
the provenance in the artifacts is the same either way.

The essential tests here are the two equivalence gates.  ``go``'s
composed ``rw-wps`` line must equal the one ``woof fetch
--author-front-door-manifest`` prints, and its composed forecast line
must equal the one the front door prints, because those printed
commands ARE the documented manual chain.  If either drifts, ``go``
stops being an automation of the documented route and becomes a second
route that happens to look like it.
"""

from __future__ import annotations

import io
import json
import shlex
import itertools
import shutil
import signal
import subprocess
from pathlib import Path

import pytest

from conftest import requires_cupy

from woof import go_cli, provenance, run_stamp
from woof.cli import main as cli_main


#: Flags whose value is a path INSIDE this run's tree, in the order the
#: stages are composed.  A stage fake reads the tree off its own command
#: rather than assuming one: every run claims its own timestamped folder
#: under ``--outdir`` (``woof.run_stamp``), so a hard-coded
#: ``<outdir>/prepared`` is a directory no stage writes to.
_STAGE_TREE_FLAGS = ("--output-directory", "--output-root", "--outdir",
                     "--prepared-root", "--render-dir")


def _stage_root(command) -> Path:
    """The run root a composed stage command points into."""

    for flag in _STAGE_TREE_FLAGS:
        if flag in command:
            return Path(command[command.index(flag) + 1]).parent
    raise AssertionError(
        f"no run-tree flag in {command!r}; the fake cannot tell which "
        "run folder this stage was composed for")


def _startup_notices(command: str) -> tuple:
    """Matchers for the stderr lines a healthy front door is documented
    to write, one predicate per line.

    Two, both intended and both matched by their exact text so anything
    else on stderr still fails the test that calls this:

    * the provenance banner, ``woof <door>: <banner>``, which
      :func:`woof.provenance_gate.announce` prints once per process so
      a log names the tree that ran.  It is prefixed by whichever door
      this process opened FIRST (a test that authors its config through
      ``woof domain`` sees ``woof domain:``), and whichever test opened
      it has consumed it, so a test run alone sees the line and a test
      run after its neighbours does not;
    * the Ctrl-C notice ``woof.cli`` prints for a long-running command
      when the process inherited SIGINT ignored, which is every test
      process a shell started in the background (``nohup``, ``&``).
    """

    banner = provenance.resolve().banner()

    def is_banner(line: str) -> bool:
        door, sep, text = line.partition(": ")
        return (bool(sep) and text == banner and door.startswith("woof ")
                and " " not in door[len("woof "):])

    notices = [is_banner]
    if signal.getsignal(signal.SIGINT) is signal.SIG_IGN:
        interrupt_notice = (
            "warning: SIGINT is set to ignore in this process, so Ctrl-C "
            f"cannot stop `woof {command}`; send SIGTERM to stop it")
        notices.append(lambda line: line == interrupt_notice)

    # A THIRD, conditional on the machine rather than on the door: two of
    # classic Thompson's four tables are excluded from the wheel, so a
    # FRESH INSTALL has not staged them and the preflight says so once,
    # as a warning, for any mp=8 configuration.  That notice is the
    # product working -- the fix is one command and nothing has been
    # downloaded yet -- but the tests below read stderr as an exact set,
    # so on a machine in that state they were failing for a correct
    # sentence.  Taken FROM the producer rather than transcribed, so it
    # matches only while the tables really are absent and any other line
    # still fails the test that calls this.
    try:
        from woof.table_assets import require_thompson_tables

        require_thompson_tables()
    except Exception as unstaged:  # noqa: BLE001 - the state, not a failure
        table_notice = "warning: " + " ".join(str(unstaged).split())
        notices.append(lambda line: line == table_notice)

    # A FOURTH, from the go chain's first stage: `woof check` states the
    # planning-time model-top verdict (woof.vertical_contract, 8f7e4c27247)
    # on stderr, so `--json` stdout stays one document.  Only the PASS form
    # is a healthy notice; a REFUSED top stops the door and still fails the
    # test that calls this.
    top_pass = "woof check: model-top check PASS: source "
    notices.append(lambda line: line.startswith(top_pass))
    return tuple(notices)


def _without_startup_notices(err: str, command: str) -> str:
    """``err`` minus the documented startup notices, each at most once."""

    lines = err.splitlines()
    for is_notice in _startup_notices(command):
        for index, line in enumerate(lines):
            if is_notice(line):
                del lines[index]
                break
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Fixtures: a wizard-authored single-domain GFS config, made once
# ---------------------------------------------------------------------------

PROFILE = "morrison-mp10-ysu-mm5-noah-kf-rte-rrtmgp-v1"


def _emit(tmp_path, name, *extra, source="gfs", ladder="12", profile=PROFILE,
          point="35.3,-97.5"):
    out = tmp_path / f"{name}.toml"
    argv = ["domain", f"--point={point}", "--card", "24gb",
            "--ladder", ladder, "--source", source,
            "--cycle", "2026-07-29T18", "--hours", "6",
            "--out", str(out), *extra]
    if profile is not None:
        argv += ["--physics-profile", profile]
    assert cli_main(argv) == 0
    return out


def _emit_unnamed_suite(tmp_path, name, *extra, **kwargs):
    """A config whose physics matches NO shipped profile.

    The chain still has an unnamed-suite branch -- ``plan["profile"]``
    is ``None``, ``--physics-profile`` is omitted from every composed
    command, and the verification status is stated instead -- and the
    two tests below are what hold it.  Until 2026-08-06 the wizard's own
    ``--physics-profile``-less emission produced such a config, so they
    built their fixture that way; 1.7.1 bound the gfs/era5 default to
    the certified Morrison profile (the nocturnal-radiation directive),
    which is a shipped profile, so that emission no longer reaches the
    branch and no door emits the unnamed suite any more.

    The branch itself is untouched and still reachable -- a hand-written
    config, an imported namelist, or the unshipped
    ``DEFAULT_SUITE_PHYSICS`` suite, which
    :data:`woof.domain_wizard.DEFAULT_PHYSICS_PROFILE`'s own docstring
    records as "reachable programmatically" -- so the fixture asks the
    wizard for that suite directly rather than hand-writing a stand-in
    that could agree with neither the emitter nor the loader.  Its
    ``[shared]`` block is the suite the pre-1.7.1 default emitted, key
    for key; only the emitted HEADER differs, because 1.7.1 states
    nocturnal validity on every file it writes.

    The assertion below is the fixture's own proof: a helper that
    quietly stopped producing an unmatched suite would leave these two
    tests passing against the branch they were written to leave.
    """

    from woof import domain_wizard

    bound = domain_wizard.DEFAULT_PHYSICS_PROFILE
    domain_wizard.DEFAULT_PHYSICS_PROFILE = None
    try:
        config = _emit(tmp_path, name, *extra, profile=None, **kwargs)
    finally:
        domain_wizard.DEFAULT_PHYSICS_PROFILE = bound
    # The fixture is only a fixture if it really matches nothing.
    from woof.experiment import load_experiment
    from woof.physics_compat import identify_single_domain_profile
    assert identify_single_domain_profile(
        load_experiment(config).root.run) is None
    return config


@pytest.fixture(scope="module")
def gfs_config(tmp_path_factory):
    return _emit(tmp_path_factory.mktemp("gfs"), "myarea")


#: What the pinned card below reports free.  Comfortably above this
#: file's fixture config (a 24 GiB-card ladder, ~19.94 GiB forecast peak
#: envelope) so the gate's verdict here is "fits", deterministically.
_PINNED_FREE_BYTES = 30 * 1024 ** 3


@pytest.fixture(autouse=True)
def _a_card_whose_free_vram_this_file_decides(monkeypatch):
    """Pin the ONE number the outside world moves in the memory gate.

    ``memory_gate`` refuses when the binding phase exceeds the free VRAM
    it measures *right now*, which is correct and is the whole point of
    gating before the fetch.  It also means that on a shared card every
    test below that drives the real chain -- the stage-failure replay,
    the one-line-per-stage report, ``--explain`` -- turns red whenever
    another run happens to hold the card, because the chain stops at a
    genuine refusal instead of reaching the stage behaviour under test.
    Proven: with 8 GiB reported free, five tests in this file fail; with
    the card idle they pass.  A test suite must not have that reading.

    So the card's free VRAM is pinned here and everything else in the
    gate stays real -- the phase estimates, the reserve policy, the
    verdict sentence, the refuse/warn arithmetic all run as shipped.
    The gate's own tests below substitute ``memory_gate`` wholesale from
    inside the test body, after this fixture, so they still choose their
    own numbers and are unaffected.

    The pin sits on the gate's subprocess probe seam: the gate asks the
    card nothing in-process (that stood up a CUDA context the go process
    then held for the whole chain as its progress printer), so the ONE
    place the outside world enters is ``device_memory_probe_subprocess``.
    Pinning it also makes this file identical on every box -- with or
    without a card, busy or idle -- where the old memGetInfo pin still
    left the no-device path machine-dependent.  ``profile: None`` prices
    the non-pool terms against the reference profile, deterministically.
    """

    from woof.core import preflight
    from types import SimpleNamespace
    from woof import doctor
    monkeypatch.setattr(doctor, "_cuda_headers_check", lambda **kwargs: SimpleNamespace(status="verified"))

    monkeypatch.setattr(
        preflight, "device_memory_probe_subprocess",
        lambda **_kwargs: {"free_bytes": _PINNED_FREE_BYTES,
                           "total_bytes": 32 * 1024 ** 3,
                           "profile": None})


def test_the_pinned_card_is_what_the_gate_reads(gfs_config, tmp_path):
    """The fixture above is essential; prove it reaches the gate."""

    plan = go_cli.plan_from_config(gfs_config, outdir=tmp_path / "out")
    gate = go_cli.memory_gate(plan)
    assert gate["free_bytes"] == _PINNED_FREE_BYTES
    assert not gate["refuse"]


# ---------------------------------------------------------------------------
# Refusals: never half-orchestrate
# ---------------------------------------------------------------------------

def test_wizard_era5_config_uses_the_shared_declared_input_launch(tmp_path, capsys, monkeypatch):
    # Repeated previews reuse published archive bounds rather than probing again.
    from datetime import datetime
    from time import monotonic
    from woof import source_availability as availability
    from woof.source_adapters import get_source_adapter
    from dataclasses import replace
    original = get_source_adapter("era5").archive_windows[0]
    window = replace(original, bounds_url="https://fixture.invalid/era5-bounds",
                     bounds_stop_keys=("end",))
    stop = datetime(2026, 8, 1, 23)
    monkeypatch.setattr(availability, "_BOUNDS", {window.bounds_url: (monotonic(), stop)})
    assert availability.published_stop(window) == stop
    config = _emit(tmp_path, "era5", source="era5")
    before = config.read_bytes()
    capsys.readouterr()  # Separate wizard output from the launch preview.
    output = tmp_path / "era5 launch"
    assert cli_main(["go", str(config), "--outdir", str(output), "--dry-run"]) == 0
    printed = capsys.readouterr().out
    assert "go: era5, 1 domain(s); fetch -> prepare -> forecast -> render" in printed
    assert "Run: woof go " in printed
    assert config.read_bytes() == before
    assert not output.exists()


def test_a_domain_tree_dry_runs_and_step_five_is_the_tree_runner(
        tmp_path, capsys, monkeypatch):
    """`woof go` runs a nest ladder; it used to refuse one.

    The refusal it replaces told a reader with a two-domain config to
    "re-emit the config without --ladder" -- throw the nests away --
    on the premise that "the single-domain runner this chain drives
    takes one".  The chain already drove the other runner: the plan's
    own ``runner`` key is the tree module here, and ``_run_forecast``
    already composed ``tree_forecast_command`` on that arm.
    """

    # A dry run prints commands and spends nothing, but `go` still
    # resolves the decoder bridge before it prints -- pinned here so
    # this test reads the dispatch and not the build state of the box.
    bridge = tmp_path / "gfs_grib2_bridge"
    bridge.write_bytes(b"stub")
    monkeypatch.setattr(go_cli, "resolve_bridge", lambda: bridge)

    config = _emit(tmp_path, "tree", ladder="12-3")
    assert cli_main(["go", str(config), "--outdir", str(tmp_path / "go"),
                     "--dry-run"]) == 0
    printed = capsys.readouterr().out
    # Step 5 is the TREE runner, module form, with the one preparation
    # receipt it binds -- not the single-domain runner's three digests.
    step5 = printed.split("5. forecast")[1].split("6. render")[0]
    assert go_cli.TREE_RUNNER_MODULE in step5
    assert "--preparation-receipt-sha256" in step5
    assert "--prepared-content-sha256" not in step5
    assert go_cli.RUNNER_MODULE not in step5
    # And the plan itself resolves to that runner with no keyword.
    plan = go_cli.plan_from_config(config, outdir=tmp_path / "plan")
    assert plan["domains"] == 2
    assert plan["runner"] == go_cli.TREE_RUNNER_MODULE


def test_the_legacy_stage_composer_refuses_other_input_formats(tmp_path):
    """The public door dispatches first; the GFS decoder stays format-bound."""

    tree = _emit(tmp_path, "tree2", ladder="12-3")
    era5 = tmp_path / "era5-tree.toml"
    era5.write_text(
        tree.read_text(encoding="utf-8")
        + '\n[case_data]\nschema = "gpuwm.case.v1"\n', encoding="utf-8")
    with pytest.raises(go_cli.GoRefusal) as refusal:
        go_cli.plan_from_config(era5, outdir=tmp_path / "o1")
    assert "[case_data]" in str(refusal.value)

    hrrr = tmp_path / "hrrr-tree.toml"
    hrrr.write_text(
        tree.read_text(encoding="utf-8").replace(
            'source = "gfs"', 'source = "hrrr"'), encoding="utf-8")
    with pytest.raises(go_cli.GoRefusal) as refusal:
        go_cli.plan_from_config(hrrr, outdir=tmp_path / "o2")
    assert "--source hrrr" in str(refusal.value)


#: The case_data refusal's remedy as each shell must receive it, written
#: out rather than derived so a generator that loses its shell rule cannot
#: also rewrite what it is judged against.  Windows PowerShell 5.1 rejects
#: `&&` with a parser error, and a bare `;` would start the run after the
#: check refused it; `$?` is false after a native command that failed.
CASE_DATA_REMEDY_FOR_SHELL = {
    False: "remedy: woof check era5.toml && woof run era5.toml",
    True: "remedy: woof check era5.toml; if ($?) { woof run era5.toml }",
}


@pytest.mark.parametrize("windows", (False, True))
def test_the_case_data_refusal_spells_check_then_run_for_the_shell(
        tmp_path, monkeypatch, windows):
    """The remedy line pastes into the reader's shell and keeps the gate.

    It printed `woof check X && woof run X` on every OS, which Windows
    PowerShell 5.1 cannot parse.
    """

    from woof import bridges

    monkeypatch.setattr(bridges, "WINDOWS_SHELL", windows)
    monkeypatch.chdir(tmp_path)
    Path("era5.toml").write_text(
        '[case_data]\nschema = "gpuwm.case.v1"\n', encoding="utf-8")
    with pytest.raises(go_cli.GoRefusal) as refusal:
        go_cli.plan_from_config(Path("era5.toml"), outdir=tmp_path / "o")
    message = str(refusal.value)
    assert CASE_DATA_REMEDY_FOR_SHELL[windows] in message, message
    if windows:
        assert "&&" not in message, (
            f"Windows PowerShell 5.1 cannot parse '&&': {message}")


@pytest.mark.parametrize("windows", (False, True))
def test_the_case_data_remedy_quotes_a_path_with_a_space_for_the_shell(
        tmp_path, monkeypatch, windows):
    """THE BREAKAGE: the remedy interpolated the config path bare, so a
    config in a folder with a space reached `woof check` as two
    arguments in either shell.  Single quotes hold the path as one word
    in both, and `$?` still gates the run in PowerShell."""

    from woof import bridges

    monkeypatch.setattr(bridges, "WINDOWS_SHELL", windows)
    monkeypatch.chdir(tmp_path)
    config = Path("my runs") / "era5.toml"
    config.parent.mkdir()
    config.write_text('[case_data]\nschema = "gpuwm.case.v1"\n',
                      encoding="utf-8")
    with pytest.raises(go_cli.GoRefusal) as refusal:
        go_cli.plan_from_config(config, outdir=tmp_path / "o")
    remedy = next(line.strip() for line in str(refusal.value).splitlines()
                  if line.strip().startswith("remedy:"))
    quoted = f"'{config}'"
    if windows:
        assert remedy == (f"remedy: woof check {quoted}; "
                          f"if ($?) {{ woof run {quoted} }}"), remedy
    else:
        assert remedy == (f"remedy: woof check {quoted} && "
                          f"woof run {quoted}"), remedy
        assert shlex.split(remedy.removeprefix("remedy: ")) == [
            "woof", "check", str(config), "&&", "woof", "run", str(config)]


def test_the_default_emission_is_what_the_default_runner_accepts(
        tmp_path, capsys, monkeypatch):
    """Default wizard output piped to the default runner composes.

    The 4090 user-zero stress run (2026-08-03) followed the obvious
    path: `woof domain --point ... --card 24gb --source gfs` with no
    --ladder, then `woof go` on the file it wrote -- and go refused
    it, because the flags door's --ladder default was `auto`, the
    deepest tree that fits.  The interactive door had already ruled on
    this exact seam (domain_interactive.DEFAULT_LADDER = "12": "two
    features that do not compose is not a feature"); this test pins
    the same ruling onto the flags door.

    Real emission, real plan reader, no profile flag: the default
    suite runs as written (owner ruling 2026-07-31), so nothing here
    needs one.
    """
    monkeypatch.setattr(go_cli, "resolve_bridge", lambda: tmp_path / "bridge")

    out = tmp_path / "default.toml"
    assert cli_main(["domain", "--point=35.3,-97.5", "--card", "24gb",
                     "--source", "gfs", "--cycle", "2026-07-29T18",
                     "--hours", "6", "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    # The wizard's own closing block names the runner its default
    # emission is for -- the one-command chain, not the tree route.
    assert "woof go " in printed

    plan = go_cli.plan_from_config(out)
    assert plan["source"] == "gfs"
    assert cli_main(["go", str(out), "--dry-run"]) == 0


# NEEDS CUPY INSTALLED, and opens no device: `woof go --dry-run`
# still returns 0 without it, but the door writes a stage 5 warning to
# stderr naming the absent module, and the sentence this test holds is
# that stderr carries NOTHING beyond the two documented startup
# notices. Measured: `_without_startup_notices(captured.err, "go")`
# reads "go: WARNING -- this install cannot run stage 5 (forecast):
# cupy (cupy-cuda12x / cupy-cuda13x) is not installed.", so the
# assertion fails on that warning rather than on anything about a
# shipped physics profile, which is this test's actual subject.
@requires_cupy
def test_a_config_with_no_shipped_profile_runs_with_status_stated(
        tmp_path, capsys, monkeypatch):
    """Converted (owner ruling 2026-07-31): the chain's last stage runs
    the config's own suite as written, so the first stage plans it
    instead of refusing it -- with the verification status stated in one
    sentence and no --physics-profile invented anywhere."""
    monkeypatch.setattr(go_cli, "resolve_bridge", lambda: tmp_path / "bridge")

    config = _emit_unnamed_suite(tmp_path, "default_suite")
    assert cli_main(["go", str(config), "--dry-run"]) == 0
    captured = capsys.readouterr()
    # Nothing on stderr beyond the two startup notices the door is
    # documented to write: no refusal, no warning about the profile.
    assert _without_startup_notices(captured.err, "go") == "", captured.err
    assert "supported, not yet WRF-verified" in captured.out
    assert "--physics-profile" not in captured.out

    plan = go_cli.plan_from_config(config, outdir=tmp_path / "go")
    assert plan["profile"] is None
    for command in (
            go_cli.authority_command(plan),
            go_cli.forecast_command(plan, {
                "proof": "a" * 64, "source_manifest": "b" * 64,
                "prepared_content": "c" * 64}),
    ):
        assert "--physics-profile" not in command


def test_a_missing_config_is_a_refusal_not_a_traceback(tmp_path, capsys):
    assert cli_main(["go", str(tmp_path / "nope.toml"), "--dry-run"]) == 2
    error = capsys.readouterr().err
    assert "does not exist" in error
    assert "Traceback" not in error


def test_a_config_without_a_fetch_table_is_refused(tmp_path, capsys,
                                                   gfs_config):
    stripped = tmp_path / "nofetch.toml"
    text = gfs_config.read_text(encoding="utf-8")
    stripped.write_text(text.split("[fetch]")[0], encoding="utf-8")
    assert cli_main(["go", str(stripped), "--dry-run"]) == 2
    assert "no [fetch] table" in capsys.readouterr().err


@pytest.mark.parametrize("source,chain", [("gfs", "prepared:go"),
                                           ("hrrr", "prepared:hrrr"),
                                           ("icon-eu", "prepared:staged")])
def test_native_launch_uses_the_registered_chain(source, chain):
    from woof.runplan import prepared_chain_for_source
    assert prepared_chain_for_source(source) == chain


# ---------------------------------------------------------------------------
# THE equivalence gates
# ---------------------------------------------------------------------------

def _flags(command: list[str]) -> dict:
    """``--flag -> value`` for one composed command."""

    out, index = {}, 0
    while index < len(command):
        token = command[index]
        if token.startswith("--"):
            following = (command[index + 1]
                         if index + 1 < len(command) else None)
            if following is not None and not following.startswith("--"):
                out[token] = following
                index += 2
                continue
            out[token] = True
        index += 1
    return out


def _printed_flags(lines) -> dict:
    """``--flag -> value`` out of a printed, pasteable command.

    Parenthetical notes are dropped before parsing.  The composer ends
    with "(--run-seconds and --history-interval-seconds default to the
    hash-bound experiment...)", which is prose ABOUT flags rather than
    flags -- and reading it as a flag is exactly the prose-scraping
    mistake ``go`` itself refuses to make.
    """

    command_lines = []
    for line in lines:
        text = str(line).rstrip()
        if not text.strip().startswith(("python", "rw-wps", "--", "woof")):
            continue
        # Strip only a TRAILING continuation backslash; a backslash
        # inside a value is part of that value.
        command_lines.append(text[:-1] if text.endswith("\\") else text)
    return _flags(shlex.split(" ".join(command_lines)))


def _stage_a_fetched_directory(out: Path, gfs_config: Path, authority: Path):
    """The minimum on-disk state ``author_gfs_front_door_manifest`` needs.

    A real fetch manifest, a real series file, and real role files, so
    the composer under test runs its true path rather than a mocked one.
    """

    out.mkdir(parents=True, exist_ok=True)
    authority.mkdir(parents=True, exist_ok=True)
    (authority / "namelist.wps").write_bytes(
        gfs_config.with_suffix(".namelist.wps").read_bytes())
    (authority / "experiment.toml").write_bytes(gfs_config.read_bytes())
    # Three forcing times, the 6 h window the config asks for at the 3 h
    # cadence: a manifest with one frame is refused because lateral
    # boundaries are interpolated between frames.
    hours = (0, 3, 6)
    (out / "gfs-series.tsv").write_text(
        "".join(f"{hour}\tgfs.f{hour:03d}.grib2\t81\n" for hour in hours),
        encoding="utf-8")
    for hour in hours:
        (out / f"gfs.f{hour:03d}.grib2").write_bytes(b"GRIB-stub")
    (out / "fetch-manifest.json").write_text(json.dumps({
        "schema": "gpuwm-fetch-manifest-v1", "source": "gfs",
        "cycle": "2026-07-29T18:00:00Z", "forecast_hours": list(hours),
        "files": [{"name": f"gfs.f{hour:03d}.grib2", "role": "gfs-subset",
                   "forecast_hour": hour, "sha256": "0" * 64}
                  for hour in hours],
    }), encoding="utf-8")


def test_the_prepare_stage_matches_what_fetch_prints(tmp_path, gfs_config):
    """go's rw-wps line IS the line the manual chain tells you to paste.

    Compared against the real ``author_gfs_front_door_manifest``, driven
    over real files, because a hand-written expectation here is a second
    opinion that can agree with neither.  An earlier draft of this test
    DID hand-write it, matched, and hid the fact that both were missing
    ``--physics-profile`` -- which the end-to-end run then found.

    go composes its command from the same inputs rather than parsing
    that printed line, so ``--explain`` moving the text cannot break the
    chain; this is what keeps the two from drifting apart anyway.
    """

    from woof.fetch import author_gfs_front_door_manifest

    plan = go_cli.plan_from_config(gfs_config, outdir=tmp_path / "go")
    bridge = tmp_path / "gfs_grib2_bridge"
    bridge.write_bytes(b"stub")
    _stage_a_fetched_directory(plan["data"], gfs_config, plan["authority"])

    printed: list[str] = []
    manifest, digest = author_gfs_front_door_manifest(
        out=plan["data"], bridge=bridge,
        wps_namelist=plan["authority"] / "namelist.wps",
        experiment_config=plan["authority"] / "experiment.toml",
        progress=printed.append)

    theirs = _printed_flags(
        [line for block in printed for line in str(block).splitlines()])
    mine = _flags(go_cli.prepare_command(
        plan, bridge, manifest=manifest, manifest_sha256=digest,
        cycle_stamp="2026-07-29_18:00:00",
        geog_root=Path(theirs["--geog-root"])))

    assert set(mine) == set(theirs), (
        f"go-only: {set(mine) - set(theirs)}, "
        f"printed-only: {set(theirs) - set(mine)}")
    # --output-root is the one deliberate difference: the printed line
    # suggests a child of the download directory, and go keeps its
    # authority/prepared/run trees together under its own root.  Every
    # other flag -- including the manifest digest and the profile -- has
    # to be identical, because those are the relay.
    for flag, value in theirs.items():
        if flag == "--output-root":
            continue
        assert str(mine[flag]).replace("\\", "/") == str(value), flag
    assert Path(mine["--output-root"]).name == "prepared"
    # The flag whose absence broke the documented chain.
    assert theirs["--physics-profile"] == PROFILE
    assert mine["--physics-profile"] == PROFILE


def test_the_printed_rw_wps_command_carries_the_physics_profile(tmp_path,
                                                                gfs_config):
    """The bug a real end-to-end found, pinned so it cannot come back.

    ``rw-wps``'s ``--physics-profile`` is spelled optional and is not:
    absent, `woof/source_cli.py` substitutes ``WSM6_PROFILE_ID`` and
    compares the experiment's physics against THAT, so the pasted
    command refused every config except a wsm6-no-radiation one --
    including the Morrison profile FIRST-LIGHT.md's own worked example
    uses.  Observed live: "selected physics differs from profile
    'wsm6-ysu-mm5-noah-no-radiation-v1'".
    """

    from woof.fetch import author_gfs_front_door_manifest

    data = tmp_path / "data"
    authority = tmp_path / "authority"
    _stage_a_fetched_directory(data, gfs_config, authority)
    bridge = tmp_path / "gfs_grib2_bridge"
    bridge.write_bytes(b"stub")

    printed: list[str] = []
    author_gfs_front_door_manifest(
        out=data, bridge=bridge,
        wps_namelist=authority / "namelist.wps",
        experiment_config=authority / "experiment.toml",
        progress=printed.append)
    text = "\n".join(str(block) for block in printed)
    assert f"--physics-profile {PROFILE}" in text


def test_a_config_matching_no_profile_prints_no_profile_flag(tmp_path):
    """Silence beats inventing a flag the runner would reject.

    A config matching no shipped profile cannot be rescued by naming
    one here; the front door explains that case itself after
    preparation, and a guessed flag would only move the refusal earlier
    without making it truer.
    """

    from woof.fetch import author_gfs_front_door_manifest

    config = _emit_unnamed_suite(tmp_path, "default_suite")
    data = tmp_path / "data"
    authority = tmp_path / "authority"
    _stage_a_fetched_directory(data, config, authority)
    bridge = tmp_path / "gfs_grib2_bridge"
    bridge.write_bytes(b"stub")

    printed: list[str] = []
    author_gfs_front_door_manifest(
        out=data, bridge=bridge,
        wps_namelist=authority / "namelist.wps",
        experiment_config=authority / "experiment.toml",
        progress=printed.append)
    text = "\n".join(str(block) for block in printed)
    assert "--physics-profile" not in text
    # ...and the rest of the command is still printed in full.
    assert "--source-manifest-sha256" in text and "--output-root" in text


def test_go_forwards_a_profile_only_when_the_whole_config_is_it(tmp_path):
    """The derivation behind ``--physics-profile``, fixed 2026-08-09.

    ``plan_from_config`` used to derive the forwarded profile from the
    ROOT domain alone, while stage 1's drift refusal reads every
    ``[[domain]]`` table -- so on the wizard's own ``--ladder`` trees
    (root = the profile, nests deliberately departing: ``cu_physics =
    0`` below the gray zone, tighter ``radt``, the ``diff_6th_factor``
    ladder) the chain composed a stage-1 command guaranteed to refuse
    its own config.  `woof run-plan`'s prepared route dispatches
    exactly this shape (``go_main``, the same call `woof go` makes).  Before the
    stage-1 refusal existed the same derivation was WORSE, not fine: the
    materializer silently flattened those nests onto the profile, which
    is the ledger #90 defect itself.

    The derivation now asks the materializer's own conflict predicate:
    a config the profile contradicts nowhere carries the assertion end
    to end, and one that deliberately says more runs as its own suite,
    unnamed (owner ruling 2026-07-31), with the verification status
    stated in the receipts.
    """

    tree = _emit(tmp_path, "tree", ladder="12-3")
    plan = go_cli.plan_from_config(tree, outdir=tmp_path / "go")
    assert plan["profile"] is None
    assert "--physics-profile" not in go_cli.authority_command(plan)
    # And stage 1 ACCEPTS what go now composes, publishing the config's
    # nest physics unchanged -- the whole point of omitting the flag.
    from woof.prepared_single_domain_forecast import (
        _render_materialized_experiment)
    _rendered, exp, _receipt = _render_materialized_experiment(
        tree.read_text(encoding="utf-8"), source="gfs", profile=None)
    assert int(exp.domains[1].run.cu_physics) == 0

    # Agreement-driven, not tree-driven: the same tree with its nest
    # brought onto the profile's values forwards the assertion again.
    agreeing = tmp_path / "agreeing.toml"
    agreeing.write_text(
        tree.read_text(encoding="utf-8")
        .replace("radt = 3.0", "radt = 12.0")
        .replace("cu_physics = 0", "cu_physics = 1")
        .replace("diff_6th_factor = 0.1\n", "diff_6th_factor = 0.12\n"),
        encoding="utf-8")
    shutil.copy(tree.with_suffix(".namelist.wps"),
                agreeing.with_suffix(".namelist.wps"))
    agreeing_plan = go_cli.plan_from_config(
        agreeing, outdir=tmp_path / "go-agree")
    assert agreeing_plan["profile"] == PROFILE

    # The single-domain emission was never affected and still binds.
    single = _emit(tmp_path, "single")
    assert go_cli.plan_from_config(
        single, outdir=tmp_path / "go-single")["profile"] == PROFILE


_BUBBLE = ("\n[[perturbation.bubbles]]\ncenter_lat = 35.3\n"
           "center_lon = -97.5\ncenter_height_m = 1500.0\nradius_km = 10.0\n"
           "depth_m = 1500.0\namplitude_k = 3.0\n")


def test_a_tree_carries_a_warm_bubble_through_the_authority_stage(tmp_path):
    """Stage 1 of `woof go` publishes a tree's bubble for the tree runner.

    GFS preparation defers [perturbation] to the prepared runner, which
    applies it to its restored states.  Stage 1 refused the block on every
    config, so `woof go` ran no bubble on any tree.  Since
    fix/single-domain-perturbation (9b04a0e9d5) the single-domain runner
    applies it too, so a single domain carries it through as well.
    """
    from woof.prepared_single_domain_forecast import (
        _render_materialized_experiment)
    tree = _emit(tmp_path, "tree", ladder="12-3")
    rendered, exp, _receipt = _render_materialized_experiment(
        tree.read_text(encoding="utf-8") + _BUBBLE, source="gfs",
        profile=None)
    assert len(exp.domains) == 2
    assert exp.perturbation.bubbles[0].amplitude_k == 3.0
    assert "[[perturbation.bubbles]]" in rendered
    single = _emit(tmp_path, "single")
    rendered, exp, _receipt = _render_materialized_experiment(
        single.read_text(encoding="utf-8") + _BUBBLE, source="gfs",
        profile=None)
    assert len(exp.domains) == 1
    assert exp.perturbation.bubbles[0].amplitude_k == 3.0
    assert "[[perturbation.bubbles]]" in rendered


def test_the_forecast_stage_matches_what_the_front_door_prints(tmp_path,
                                                               gfs_config):
    """go's forecast line IS the line rw-wps tells you to copy.

    Compared against ``gfs_direct.prepared_forecast_next_command`` -- the
    function that composes the printed command -- driven by a proof
    document of the shape the front door writes.  Same proof, same
    command, or ``go`` has drifted off the documented route.
    """

    from woof.gfs_direct import prepared_forecast_next_command

    plan = go_cli.plan_from_config(gfs_config, outdir=tmp_path / "go")
    plan["prepared"].mkdir(parents=True)
    proof = {
        "input_manifest_sha256": "a" * 64,
        "prepared_cache": {"content_sha256": "b" * 64},
        "physics": {"profile": PROFILE},
    }
    proof_path = plan["prepared"] / "proof.json"
    proof_path.write_text(json.dumps(proof), encoding="utf-8")

    digests = go_cli.proof_digests(plan["prepared"])
    assert digests["source_manifest"] == "a" * 64
    assert digests["prepared_content"] == "b" * 64

    mine = _flags(go_cli.forecast_command(plan, digests))
    theirs = _printed_flags(prepared_forecast_next_command(
        proof, output_root=plan["prepared"],
        experiment_config=plan["authority"] / "experiment.toml",
        wps_namelist=plan["authority"] / "namelist.wps", source="gfs"))

    # The composer prints --outdir as a sibling of the preparation; go
    # owns its own tree.  Every OTHER flag must match exactly, including
    # all three digests, which is the whole point.
    for flag in ("--source", "--prepared-root", "--proof-sha256",
                 "--source-manifest-sha256", "--prepared-content-sha256",
                 "--experiment-config", "--wps-namelist",
                 "--physics-profile", "--io-mode"):
        assert flag in mine and flag in theirs, flag
        assert str(mine[flag]).replace("\\", "/") == \
            str(theirs[flag]).replace("\\", "/"), flag

    # `--progress-format` is the second allowed difference, and the last.
    # The printed line is for a PERSON at a terminal, so it leaves the
    # runner's default in place and the WRF-shaped `Timing for main:`
    # lines land on their screen -- that is the reason to run the stage
    # by hand.  `go` owns the runner's stdout instead (its subprocess arm
    # would buffer tens of megabytes of discarded per-step lines) so it
    # asks for jsonl.  Subtracted rather than excused: the set equality
    # below still has to hold exactly, so no THIRD flag can drift in
    # behind this one.
    assert "--progress-format" in mine, (
        "go no longer sets the progress transport; drop this subtraction "
        "rather than leaving it to hide a real difference")
    assert "--progress-format" not in theirs, (
        "the printed line now sets a progress transport too, so these two "
        "should simply be compared directly")
    assert set(mine) - {"--progress-format"} == set(theirs)


def test_the_proof_digest_is_read_not_recomputed(tmp_path, gfs_config):
    """go transports the digests; it must never invent one.

    ``--proof-sha256`` is a hash OF the proof file, and the other two
    are values carried INSIDE it.  If go computed the inner two itself,
    the runner's comparison would be go checking its own arithmetic
    instead of checking the front door's claim.
    """

    prepared = tmp_path / "prepared"
    prepared.mkdir()
    proof = {"input_manifest_sha256": "a" * 64,
             "prepared_cache": {"content_sha256": "b" * 64}}
    path = prepared / "proof.json"
    path.write_text(json.dumps(proof), encoding="utf-8")

    from woof.fetch import sha256_file

    digests = go_cli.proof_digests(prepared)
    assert digests["proof"] == sha256_file(path)
    assert digests["source_manifest"] == "a" * 64
    assert digests["prepared_content"] == "b" * 64


def test_a_hierarchy_proof_is_refused_toward_the_tree_runner(tmp_path):
    """A proof with no single prepared-cache identity is a tree product."""

    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "proof.json").write_text(
        json.dumps({"input_manifest_sha256": "a" * 64}), encoding="utf-8")
    with pytest.raises(go_cli.GoRefusal, match="prepared_domain_tree"):
        go_cli.proof_digests(prepared)


def test_a_missing_proof_is_refused_rather_than_guessed(tmp_path):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    with pytest.raises(go_cli.GoRefusal, match="nothing to bind against"):
        go_cli.proof_digests(prepared)


# ---------------------------------------------------------------------------
# Stage failure: replay, and stop
# ---------------------------------------------------------------------------

class _FakePopen:
    """A ``subprocess.Popen`` double over this file's ``fake_run`` doubles.

    ``_run_stage`` spells ``subprocess.run`` out as ``Popen`` because
    the interrupt path has to be able to NAME the pid of the stage woof
    was waiting on without signalling it, and it reads the stage's two
    pipes as the stage writes them, so a preparer's steps reach the run
    while it runs.  The double's pipes hold the ``fake_run`` answer's
    text; ``communicate`` stays for a ``subprocess.run`` made while it is
    installed.
    """

    _pids = itertools.count(424242)

    def __init__(self, completed, command=None):
        self._completed = completed
        self.args = command
        self.pid = next(self._pids)
        self.returncode = None
        self.stdout = io.StringIO(completed.stdout or "")
        self.stderr = io.StringIO(completed.stderr or "")

    def communicate(self, input=None, timeout=None):
        self.returncode = self._completed.returncode
        return self._completed.stdout, self._completed.stderr

    # The rest of the protocol ``subprocess.run`` drives: it opens a Popen
    # as a context manager, so a ``run`` made while this double is
    # installed died with "'_FakePopen' object does not support the
    # context manager protocol".

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = self._completed.returncode
        return self.returncode

    def kill(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.wait()
        return False


#: The renderer as this file's Popen double answers it: its usage line, the
#: render contract, and a listing of two products a wrfout carries (the
#: products ``test_go_chain_events`` stands in for).
_STAND_IN_LISTING = (
    "group keywords: all, direct, derived, windowed\n"
    "  composite_reflectivity\n"
    "  2m_temperature\n"
    "selectable_slugs=2\n"
    "WRFOUT\tcomposite_reflectivity\tdirect\tdrawable\t0\tstored\n"
    "WRFOUT\t2m_temperature\tdirect\tdrawable\t0\tstored\n")


def _stand_in_renderer(command):
    """The double's answer to a renderer probe, or ``None`` for any other command.

    Disk admission and the product-spec check ask the renderer's own
    catalog before the download (``runplan.render_catalog``): ``--help``
    and ``--abi`` through ``subprocess.run``, then ``--list-products``.
    On a box where a renderer resolves, those launches reached this
    file's double and the test's own ``fake_run``, which answered them as
    a chain stage or refused them as one.  The double answers them as the
    renderer, so a chain stage's fake sees only chain stages.
    """

    from woof import rustwx

    if (len(command) != 2
            or Path(str(command[0])).stem != rustwx.RENDERER_NAME):
        return None
    if command[1] == "--help":
        return _FakeCompleted(
            0, stdout="usage: rw_wrfbatch --store-root DIR --out-dir DIR "
                      "wrfout...\n")
    if command[1] == "--abi":
        return _FakeCompleted(0, stdout=rustwx.RENDERER_ABI_MARKER + "\n")
    if command[1] == "--list-products":
        return _FakeCompleted(0, stdout=_STAND_IN_LISTING)
    return None


@pytest.fixture(autouse=True)
def _a_render_catalog_cache_of_its_own(monkeypatch):
    """Each test reads the renderer catalog itself, never a cached answer.

    The catalog is cached per process against the renderer binary, so a
    chain test's outcome depended on whether an earlier test in the same
    process had already asked: run alone, or on an xdist worker that had
    not, the chains here reached the renderer through the Popen double.
    The stand-in answer is kept out of the cache every later test reads.
    """

    from woof import runplan

    monkeypatch.setattr(runplan, "_RENDER_CATALOG_CACHE", {})


def _popen_double(fake_run):
    """Adapt a ``(command, **kwargs) -> CompletedProcess`` double."""

    def factory(command, **kwargs):
        answer = _stand_in_renderer(command)
        if answer is None:
            answer = fake_run(command, **kwargs)
        return _FakePopen(answer, command)

    return factory


class _FakeCompleted:
    def __init__(self, returncode, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _publish_chain_artifacts(command, *, frames=False):
    """Stage artifacts shared by the output-policy tests."""
    if "--author-front-door-manifest" in command:
        manifest = Path(command[command.index("--manifest-out") + 1])
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text("{}", encoding="utf-8")
    if "--output-root" in command:
        prepared = Path(command[command.index("--output-root") + 1])
        prepared.mkdir(parents=True, exist_ok=True)
        (prepared / "proof.json").write_text(json.dumps({
            "input_manifest_sha256": "a" * 64,
            "prepared_cache": {"content_sha256": "b" * 64}}), encoding="utf-8")
    if frames and "--io-mode" in command:
        directory = Path(command[command.index("--outdir") + 1]) / "wrfout"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "wrfout_d01_2026-07-29_18_00_00").write_text("", encoding="utf-8")


# NEEDS CUPY INSTALLED, and opens no device.  The chain tests marked below
# run the real `woof go` door; without the array library it refuses
# before the fetch stage (`this command needs cupy (cupy-cuda12x /
# cupy-cuda13x), which this install does not have`) and returns 2, so the
# stage outputs and refusal sentences these tests hold are never reached.
# Measured on the Linux release node: red without cupy, green with it
# (proof/node-reds-276).
@requires_cupy
@pytest.mark.parametrize("failing_index, failing_label", [
    (0, "authority"),
    (1, "fetch"),
    (2, "manifest"),
])
def test_a_failing_stage_replays_its_output_and_stops(
        tmp_path, capsys, monkeypatch, gfs_config, staged_geog, failing_index,
        failing_label):
    """Tested at more than one stage, because "stops" is the contract.

    Every later stage consumes the previous one's output, so unlike
    ``woof setup`` -- whose steps are independent and all run -- a
    failure here must end the chain.  The failing stage's whole output
    is replayed with or without ``--explain``: the reason a stage
    refused is the one thing an orchestrator must never summarize.
    """

    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if len(calls) - 1 == failing_index:
            return _FakeCompleted(
                3, stdout="line one of the real diagnosis\n",
                stderr="line two, naming the actual problem\n")
        return _FakeCompleted(0, stdout="chatty success\n")

    # This double counts chain stages. The concurrent kernel warm-up uses
    # Popen too, but is independent of stage failure and output replay.
    monkeypatch.setattr(go_cli.SpeedHooks, "warm_kernels_spawn",
                        lambda args, plan: None)
    # The hydro door's tool probe is stubbed outside the hydro test modules
    # (conftest _hydro_door_tools_unprobed_outside_hydro_tests), so only chain
    # stages reach this double.
    monkeypatch.setattr(subprocess, "Popen", _popen_double(fake_run))
    monkeypatch.setattr(go_cli, "resolve_bridge",
                        lambda: tmp_path / "gfs_grib2_bridge")

    rc = cli_main(["go", str(gfs_config), "--outdir", str(tmp_path / "go"),
                   "--geog-root", str(staged_geog)])
    printed = capsys.readouterr().out
    assert rc == 3
    # The hydro door's verdict sits in the memory gate's line block (test host: the
    # headline run had hydro off and only the receipt said so).
    lines = printed.splitlines()
    memory = next(i for i, line in enumerate(lines) if line.startswith("go: memory -- "))
    # Hydro is opt-in (2.9.4 hotfix): a run that did not ask says so in the same block.
    assert lines[memory + 1].startswith("go: hydro -- off (default_off)"), lines[memory + 1]
    assert f"FAILED  {failing_label}" in printed
    # Verbatim, without --explain, because this is the reason.
    assert "line one of the real diagnosis" in printed
    assert "line two, naming the actual problem" in printed
    assert "nothing after it ran" in printed
    # And nothing after it ran.
    assert len(calls) == failing_index + 1
    # A succeeding stage stays quiet.
    assert "chatty success" not in printed


@requires_cupy
def test_a_failed_chain_says_what_it_left_on_disk(
        tmp_path, capsys, monkeypatch, gfs_config, staged_geog):
    """D-05.  A failed prepare leaves a scratch tree of a few hundred MB
    and said nothing about it anywhere.

    The interrupted arm of this same try/except has always told the
    reader what is on disk; the failure arm returned the code in
    silence.  Nothing is deleted -- the tree is the evidence of what
    failed -- but it is now named, measured, and declared safe to remove.
    """

    root = tmp_path / "go"

    def fake_run(command, **kwargs):
        # Into THIS RUN's tree, read off the stage's own command rather
        # than assumed: every run claims its own timestamped folder under
        # --outdir, and a fake that wrote to the case root would be
        # measuring a directory no stage uses.
        run_root = _stage_root(command)
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "scratch.bin").write_bytes(b"x" * 3_000_000)
        return _FakeCompleted(3, stdout="the real diagnosis\n")

    monkeypatch.setattr(subprocess, "Popen", _popen_double(fake_run))
    monkeypatch.setattr(go_cli, "resolve_bridge",
                        lambda: tmp_path / "gfs_grib2_bridge")

    rc = cli_main(["go", str(gfs_config), "--outdir", str(root),
                   "--geog-root", str(staged_geog)])
    captured = capsys.readouterr()
    assert rc == 3
    assert str(root) in captured.err
    assert "partial tree with no certification capsule" in captured.err
    assert "MiB" in captured.err
    assert "safe to remove" in captured.err
    # and it really is still there: the note must not have tidied away
    # the evidence it is describing
    run_root = run_stamp.latest(root)
    assert run_root is not None
    assert (run_root / "scratch.bin").exists()


@requires_cupy
def test_outdir_and_data_dir_naming_one_directory_is_refused(
        tmp_path, capsys, gfs_config):
    """E-07.  The default puts the download inside the run root, so only
    an explicit --data-dir can make the two equal -- and when it does,
    the create-only run directory and the meant-to-be-reused download
    cache become the same directory, so the next run refuses against the
    reader's own cache."""

    shared = tmp_path / "both"
    shared.mkdir()
    rc = cli_main(["go", str(gfs_config), "--outdir", str(shared),
                   "--data-dir", str(shared)])
    err = capsys.readouterr().err
    assert rc == 2
    assert "cannot be both" in err
    assert "Traceback" not in err

    # the negative control: different directories still plan fine
    plan = go_cli.plan_from_config(
        gfs_config, outdir=tmp_path / "run", data_dir=tmp_path / "dl")
    assert plan["root"] != plan["data"]


@requires_cupy
def test_a_succeeding_chain_reports_one_line_per_stage(tmp_path, capsys,
                                                       monkeypatch,
                                                       staged_geog,
                                                       gfs_config):
    plan_root = tmp_path / "go"

    def fake_run(command, **kwargs):
        _publish_chain_artifacts(command, frames=True)
        return _FakeCompleted(0, stdout="detail nobody asked for\n")

    monkeypatch.setattr(subprocess, "Popen", _popen_double(fake_run))
    monkeypatch.setattr(go_cli, "resolve_bridge",
                        lambda: tmp_path / "gfs_grib2_bridge")

    monkeypatch.setattr(go_cli, "render_extra_missing", lambda: None)
    # The finalize compare stage (d63a454a40) probes rw_compare --abi through
    # subprocess.  The double above answers EVERY child with the stage
    # stdout, so on a host whose environment names an rw_compare it
    # impersonated a stale engine and the skip line quoted its answer.  Pin
    # the engine absent: this test reads the stage lines, and the compare
    # stage's own lines are pinned in tests/test_render_compare.py.
    from woof import rustwx_lanes

    def no_compare_engine():
        raise RuntimeError("rw_compare is not built")

    monkeypatch.setattr(rustwx_lanes, "require_compare_bin", no_compare_engine)

    rc = cli_main(["go", str(gfs_config), "--outdir", str(plan_root),
                   "--geog-root", str(staged_geog)])
    printed = capsys.readouterr().out
    assert rc == 0
    for label in ("authority", "fetch", "manifest", "prepare", "forecast",
                  "render"):
        assert f"  ok      {label}" in printed
    assert "detail nobody asked for" not in printed
    # No "next:" block, and that absence is the point.
    #
    # This chain used to end by printing `woof render ...` for the
    # reader to paste, which is the same baton pass `go` exists to
    # remove: a command printed instead of run reads as "it stopped".
    # Rendering is the sixth stage now, so a finished `go` leaves
    # pictures, not homework.
    assert "next:" not in printed
    assert "go: rendered " in printed


# NEEDS CUPY INSTALLED, and opens no device: this test runs `woof go` to
# completion; without cupy the door refuses ahead of the fetch stage and
# returns 2.
@requires_cupy
def test_a_passing_stages_note_survives_the_output_capture(
        tmp_path, capsys, monkeypatch, staged_geog, gfs_config):
    """A skipped product must not become a silent success one level up.

    `woof render` says, in one sentence, when a frame's declared inputs
    are absent and a product was therefore not drawn -- it exists so
    that skip is never silent.  `go` captures every stage's output and
    prints `ok render`, so without this the sentence is produced and
    swallowed, and the chain re-creates the silence the render change
    removed.

    `warning:` already survived; `note:` is this tree's word for "true,
    worth knowing, not a fault", and it survives on the same terms.  The
    third assertion is the boundary: ordinary chatter still does not.
    """
    plan_root = tmp_path / "go"

    def fake_run(command, **kwargs):
        _publish_chain_artifacts(command)
        return _FakeCompleted(0, stdout=(
            "note: render skipped 1 product render(s) (refl)\n"
            "warning: something to know\n"
            "render: /png/t2.png\n"))

    # Popen, not run: `_run_stage` spells subprocess.run out as Popen +
    # communicate so the interrupt path can NAME the child's pid.  This
    # test was written against the older seam and merged forward
    # unchanged -- a patch on `run` is simply inert now, so the stages
    # ran for real, fetched over the network, and died on the bridge
    # stub.  Same double as every sibling in this file.
    monkeypatch.setattr(subprocess, "Popen", _popen_double(fake_run))
    monkeypatch.setattr(go_cli, "resolve_bridge",
                        lambda: tmp_path / "gfs_grib2_bridge")
    assert cli_main(["go", str(gfs_config), "--outdir", str(plan_root),
                     "--geog-root", str(staged_geog)]) == 0
    printed = capsys.readouterr().out
    assert "note: render skipped 1 product render(s) (refl)" in printed
    assert "warning: something to know" in printed
    assert "/png/t2.png" not in printed, \
        "a passing stage's ordinary output still stays behind --explain"


# NEEDS CUPY INSTALLED, and opens no device: this test replays every stage
# of `woof go`; without cupy the door refuses before the first one and
# returns 2.
@requires_cupy
def test_explain_replays_every_stage(tmp_path, capsys, monkeypatch,
                                     staged_geog,
                                     gfs_config):
    plan_root = tmp_path / "go"

    def fake_run(command, **kwargs):
        _publish_chain_artifacts(command)
        return _FakeCompleted(0, stdout="the full receipt\n")

    monkeypatch.setattr(subprocess, "Popen", _popen_double(fake_run))
    monkeypatch.setattr(go_cli, "resolve_bridge",
                        lambda: tmp_path / "gfs_grib2_bridge")
    assert cli_main(["go", str(gfs_config), "--outdir", str(plan_root),
                     "--geog-root", str(staged_geog),
                     "--explain"]) == 0
    assert "the full receipt" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# --dry-run
# ---------------------------------------------------------------------------

def test_dry_run_prints_five_filled_in_commands_and_runs_nothing(
        tmp_path, capsys, monkeypatch, gfs_config):
    def explode(*args, **kwargs):
        raise AssertionError("--dry-run must not run anything")

    monkeypatch.setattr(subprocess, "Popen", explode)
    monkeypatch.setattr(go_cli, "resolve_bridge",
                        lambda: tmp_path / "gfs_grib2_bridge")

    assert cli_main(["go", str(gfs_config), "--dry-run",
                     "--outdir", str(tmp_path / "go")]) == 0
    printed = capsys.readouterr().out
    for step in ("1. authority", "2. fetch", "3. manifest", "4. prepare",
                 "5. forecast"):
        assert step in printed
    # Filled in from the config, not left as placeholders.
    assert "--physics-profile " + PROFILE in printed
    assert "--cycle 2026-07-29T18" in printed
    assert "--hours 6" in printed
    # The two values that cannot exist yet name the file they come from
    # rather than showing a plausible-looking hash.
    assert "after step 3" in printed
    assert "proof.json" in printed


def test_the_fetch_area_keeps_its_equals_form_for_a_negative_box(
        tmp_path, capsys, monkeypatch, gfs_config):
    """A leading-minus area is an option token unless it is joined.

    The same rule the wizard's printed command follows; getting it wrong
    here would make the fetch stage fail with "expected one argument"
    on every western-hemisphere domain.
    """

    monkeypatch.setattr(go_cli, "resolve_bridge",
                        lambda: tmp_path / "gfs_grib2_bridge")
    # A point low enough that the forcing box's first corner is a
    # southern latitude whatever the sizing model of the day emits: this
    # test is about the "=" joining rule, not about how many cells fit.
    southern = _emit(tmp_path, "southern", point="12.0,-97.5")
    assert cli_main(["go", str(southern), "--dry-run",
                     "--outdir", str(tmp_path / "go")]) == 0
    printed = capsys.readouterr().out
    assert "--area=-" in printed
    assert "--area -" not in printed


def test_plan_review_reads_products_with_the_engine_tokenizer(
        tmp_path, capsys, monkeypatch, gfs_config):
    """A section's level list and the term that closes it are one product.

    ``xsec:QCLOUD=0.01,0.1/wa`` is a spelling the renderer draws: its
    level list is comma-separated too, and ``0.1/wa`` is the list's last
    level plus the overlay that closes it.  Plan review read it with a
    private splitter that joined only purely numeric tokens, so
    ``0.1/wa`` became a product of its own and the request was refused
    as unknown.  A misspelled product is still refused by name, before
    anything is fetched or created: it would otherwise reach the
    renderer only after the whole forecast.
    """

    import woof.runplan as runplan

    def explode(*args, **kwargs):
        raise AssertionError("plan review must not run anything")

    monkeypatch.setattr(subprocess, "Popen", explode)
    monkeypatch.setattr(go_cli, "resolve_bridge",
                        lambda: tmp_path / "gfs_grib2_bridge")
    monkeypatch.setattr(runplan, "render_catalog", lambda: {
        "engine": "rust",
        "products": [{"name": "composite_reflectivity"},
                     {"name": "2m_temperature"}],
        "group_keywords": ["direct", "derived", "windowed"]})
    out = tmp_path / "go"
    for spec in ("composite_reflectivity,xsec:QCLOUD=0.01,0.1/wa",
                 "xsec:wa=1,2,5@5"):
        capsys.readouterr()
        assert cli_main(["go", str(gfs_config), "--dry-run", "--outdir",
                         str(out), "--products", spec,
                         "--section=38.3,-99.0,38.3,-98.4"]) == 0, spec
        printed = capsys.readouterr()
        assert "6. render" in printed.out, spec
        assert "catalog does not carry" not in printed.err, spec
    unmade = tmp_path / "refused"
    assert cli_main(["go", str(gfs_config), "--dry-run", "--outdir", str(unmade),
                     "--products", "composite_reflectivity,compsite_reflectivity,"
                     "xsec:QCLOUD=0.01,0.1/wa",
                     "--section=38.3,-99.0,38.3,-98.4"]) == 2
    refused = capsys.readouterr()
    assert "'compsite_reflectivity'" in refused.err
    assert "catalog does not carry" in refused.err
    assert "'0.1/wa'" not in refused.err
    assert "6. render" not in refused.out
    assert not unmade.exists()


_SECTION_LINE = "38.3,-99.0,38.3,-98.4"


def test_go_carries_its_section_line_to_every_render_it_runs(
        tmp_path, capsys, monkeypatch, gfs_config):
    """``woof go --section`` reaches the runner's renders and the batch.

    ``woof go`` had no ``--section``: an ``xsec:`` term in ``--products``
    passed review and every render dropped it with advice to add a flag
    this command did not have.  The line is recorded in the plan, handed
    to the runner (which draws each frame as it lands and the first
    products) joined with ``=`` so a line starting with a minus sign is
    not read as an option, and put on the end-of-run batch render.
    """

    import woof.runplan as runplan
    from woof import prepared_single_domain_forecast as single

    def explode(*args, **kwargs):
        raise AssertionError("a dry run must not run anything")

    monkeypatch.setattr(subprocess, "Popen", explode)
    monkeypatch.setattr(go_cli, "resolve_bridge",
                        lambda: tmp_path / "gfs_grib2_bridge")
    monkeypatch.setattr(go_cli, "render_extra_missing", lambda: None)
    monkeypatch.setattr(runplan, "render_catalog", lambda: {
        "engine": "rust",
        "products": [{"name": "composite_reflectivity"}],
        "group_keywords": ["direct", "derived", "windowed"]})
    spec = "composite_reflectivity,xsec:QCLOUD=0.01,0.1/wa"
    assert cli_main(["go", str(gfs_config), "--dry-run", "--outdir",
                     str(tmp_path / "go"), "--products", spec,
                     f"--section={_SECTION_LINE}"]) == 0
    printed = capsys.readouterr().out
    render = printed.split("6. render", 1)[1]
    assert f"--section={_SECTION_LINE}" in render

    plan = go_cli.plan_from_config(gfs_config, outdir=tmp_path / "go2",
                                   render_products=spec,
                                   render_section=_SECTION_LINE)
    assert plan["render_section"] == _SECTION_LINE
    frames = [tmp_path / "wrfout_d01_2026-07-28_05_00_00"]
    assert f"--section={_SECTION_LINE}" in go_cli.render_command(plan, frames)
    command = go_cli.forecast_command(
        plan, {"proof": "a" * 64, "source_manifest": "b" * 64,
               "prepared_content": "c" * 64},
        early_render=spec)
    # The runner reads back exactly the line go composed.
    parsed = single.build_parser().parse_args(command[3:])
    assert parsed.render_section == _SECTION_LINE
    assert parsed.render_products == spec
    # And the plan the runner draws with carries it.
    armed = single._route_owned_first_products(
        parsed, outdir=tmp_path / "run", observer=None, started=0.0)
    assert armed._plan["render_section"] == _SECTION_LINE
    # A southern line keeps its sign on every command.
    south = "-33.9,151.2,-34.1,151.3"
    plan["render_section"] = south
    assert f"--section={south}" in go_cli.render_command(plan, frames)
    assert f"--render-section={south}" in go_cli.forecast_command(
        plan, {"proof": "a" * 64, "source_manifest": "b" * 64,
               "prepared_content": "c" * 64}, early_render=spec)


def test_review_refuses_a_section_with_no_line_before_anything_runs(
        tmp_path, capsys, monkeypatch, gfs_config):
    """An ``xsec:`` term with no line is refused by name at review.

    Before, a request of only section terms ran the whole forecast and
    then the render stage refused with "nothing left to draw"; beside
    other products the term was dropped after the forecast.  A line the
    renderer cannot read is refused at review too, because the renderer
    refuses it for the whole invocation.
    """

    import woof.runplan as runplan

    def explode(*args, **kwargs):
        raise AssertionError("plan review must not run anything")

    monkeypatch.setattr(subprocess, "Popen", explode)
    monkeypatch.setattr(go_cli, "resolve_bridge",
                        lambda: tmp_path / "gfs_grib2_bridge")
    monkeypatch.setattr(runplan, "render_catalog", lambda: {
        "engine": "rust",
        "products": [{"name": "composite_reflectivity"}],
        "group_keywords": ["direct", "derived", "windowed"]})
    for number, (spec, section, expected) in enumerate((
            ("xsec:QCLOUD=0.01,0.1/wa", None, "'xsec:QCLOUD=0.01,0.1/wa'"),
            ("composite_reflectivity,xsec:wa=1,2,5@5", None,
             "'xsec:wa=1,2,5@5'"),
            ("composite_reflectivity,xsec:wa", "38.3,-99.0,38.3,-99.0",
             "less than 1 km apart"),
            ("composite_reflectivity,xsec:wa", "95,-99.0,38.3,-98.4",
             "invalid geographic coordinate"),
            ("composite_reflectivity,xsec:wa", str(tmp_path / "nope.json"),
             "neither 'lat,lon,lat,lon' nor a readable JSON file"))):
        out = tmp_path / f"refused-{number}"
        capsys.readouterr()
        argv = ["go", str(gfs_config), "--outdir", str(out),
                "--products", spec]
        if section is not None:
            argv.append(f"--section={section}")
        assert cli_main([*argv, "--dry-run"]) == 2, spec
        refused = capsys.readouterr()
        assert expected in refused.err, (spec, refused.err)
        assert "--section" in refused.err
        assert "6. render" not in refused.out
        assert not out.exists()


def test_the_cwd_relative_fetch_out_key_is_not_trusted(tmp_path, gfs_config):
    """`[fetch].out` is written relative to the wizard's cwd, not the file.

    A config emitted from one directory records a download path that
    means something else read from another -- six `..` hops walked past
    the drive root and clamped at `C:/AppData/...` in the run that found
    this.  The table calls itself advisory; go honours the values the
    domain was SIZED against and owns where the bytes land.
    """

    import tomllib

    recorded = tomllib.loads(
        gfs_config.read_text(encoding="utf-8"))["fetch"]["out"]
    assert recorded and recorded != str(tmp_path / "go" / "data")

    plan = go_cli.plan_from_config(gfs_config, outdir=tmp_path / "go")
    assert plan["data"].parent == tmp_path / "go" / "downloads"

    override = tmp_path / "already-fetched"
    plan = go_cli.plan_from_config(gfs_config, outdir=tmp_path / "go",
                                   data_dir=override)
    assert plan["data"] == override


def _record_cached_request(folder, request):
    """Use the fetch writer's schema so its real identity guard reads this cache."""
    from woof import fetch

    folder.mkdir(parents=True, exist_ok=True)
    payload = fetch._manifest_payload(
        source=request["source"],
        cycle=fetch.parse_cycle(request["cycle"], request["source"]),
        hours=(0, 3), area=fetch.parse_area(request["area"]), files=[])
    (folder / fetch.FETCH_MANIFEST_NAME).write_text(json.dumps(payload), encoding="utf-8")
    (folder / "existing-input").write_bytes(b"retain this earlier download")


@pytest.mark.parametrize("changed", [
    {"cycle": "2026-07-30T00"}, {"area": "30,-110,40,-90"},
    {"source": "hrrr"}, {"hours": 12}, {"cadence": 1},
    {"forecast_start_hour": 3}, {"product": "prs"},
])
def test_repeat_forecasts_select_their_own_inputs_automatically(tmp_path, changed):
    from woof import fetch

    request = {"source": "gfs", "cycle": "2026-07-29T18", "hours": 6,
               "area": "25,-105,45,-85", "cadence": 3}
    root = tmp_path / "runs"
    first = go_cli.managed_download_dir(root, request)
    _record_cached_request(first, request)
    before = {p.name: p.read_bytes() for p in first.iterdir()}
    second_request = request | changed
    second = go_cli.managed_download_dir(root, second_request)
    assert second != first and not second.exists()
    fetch.check_prior_request(
        second, source=second_request["source"],
        cycle=fetch.parse_cycle(second_request["cycle"], second_request["source"]),
        area=fetch.parse_area(second_request["area"]))
    _record_cached_request(second, second_request)
    assert go_cli.managed_download_dir(root, request) == first
    assert go_cli.managed_download_dir(root, second_request) == second
    assert {p.name: p.read_bytes() for p in first.iterdir()} == before


def test_identical_fetches_share_cache_despite_config_output_spelling(tmp_path):
    request = {"source": "gfs", "cycle": "2026-07-29T18", "hours": 6,
               "area": "25,-105,45,-85", "out": "old-folder"}
    first = go_cli.managed_download_dir(tmp_path, request)
    second = go_cli.managed_download_dir(
        tmp_path, request | {"area": "45.0,-85.0,25.0,-105.0", "out": "new-folder"})
    assert first == second
    assert not list(tmp_path.iterdir()), "Planning must not allocate downloads"


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), object()])
def test_unserializable_download_settings_are_a_named_refusal(tmp_path, invalid):
    request = {"source": "gfs", "cycle": "2026-07-29T18", "hours": invalid,
               "area": "25,-105,45,-85"}
    with pytest.raises(go_cli.GoRefusal, match="forecast download settings are invalid"):
        go_cli.managed_download_dir(tmp_path / "runs", request)
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("source", ["gfs", "hrrr", "icon-eu"])
@pytest.mark.parametrize("explicit_data", [False, True])
def test_saved_latest_is_refused_before_a_dry_run_probes_or_selects_cache(
        tmp_path, monkeypatch, capsys, gfs_config, source, explicit_data):
    from woof import fetch

    def unexpected(*args, **kwargs):
        raise AssertionError("A saved latest refusal must not probe, fetch or choose a cache")

    monkeypatch.setattr(fetch, "resolve_latest_cycle", unexpected)
    monkeypatch.setattr(go_cli, "managed_download_dir", unexpected)
    monkeypatch.setattr(subprocess, "Popen", unexpected)
    config = tmp_path / "saved-latest.toml"
    config.write_text(gfs_config.read_text(encoding="utf-8").replace(
        'cycle = "2026-07-29T18"', 'cycle = "latest"').replace(
        'source = "gfs"', f'source = "{source}"'), encoding="utf-8")
    root = tmp_path / "runs"
    argv = ["go", str(config), "--dry-run", "--outdir", str(root)]
    if explicit_data:
        argv += ["--data-dir", str(tmp_path / "existing-inputs")]

    assert cli_main(argv) == 2
    message = capsys.readouterr().err
    assert "saved [fetch].cycle must be a concrete UTC cycle" in message
    assert "[experiment].start_time" in message
    assert "woof domain --cycle latest" in message
    assert not root.exists()


@pytest.mark.parametrize("start_hour", [0, 3])
def test_a_saved_fetch_cannot_shift_the_experiment_start(
        tmp_path, gfs_config, start_hour):
    config = tmp_path / "wrong-cycle.toml"
    text = gfs_config.read_text(encoding="utf-8").replace(
        'cycle = "2026-07-29T18"', 'cycle = "2026-07-30T00"')
    if start_hour:
        text = text.replace('[fetch]', f'[fetch]\nforecast_start_hour = {start_hour}')
    config.write_text(text, encoding="utf-8")
    with pytest.raises(go_cli.GoRefusal, match="download and experiment agree"):
        go_cli.plan_from_config(config, outdir=tmp_path / "runs")
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("cycle", ["latest", "2026-07-30T00"])
def test_runplan_refuses_an_unbound_saved_fetch_clock_without_probing(
        tmp_path, gfs_config, monkeypatch, cycle):
    from woof import fetch, runplan

    def unexpected(*args, **kwargs):
        raise AssertionError("Resolving a saved config must not query latest")

    monkeypatch.setattr(fetch, "resolve_latest_cycle", unexpected)
    config = tmp_path / "unbound-clock.toml"
    config.write_text(gfs_config.read_text(encoding="utf-8").replace(
        'cycle = "2026-07-29T18"', f'cycle = "{cycle}"'), encoding="utf-8")
    plan = runplan.build_plan(
        {"schema": runplan.PLAN_SCHEMA, "name": "saved-clock", "route": "prepared",
         "config": {"path": str(config)}, "output_root": str(tmp_path / "runs")},
        source="saved-clock.json", base_dir=tmp_path, sha256="0" * 64)
    with pytest.raises(runplan.PlanError, match=r"\[experiment\].start_time"):
        runplan.resolve_plan(plan, require_inputs=False)
    assert not (tmp_path / "runs").exists()


def test_explicit_runplan_latest_resolves_once_before_managed_cache_selection(
        tmp_path, monkeypatch):
    from datetime import datetime
    from woof import fetch, runplan

    probes = []

    def resolve(source, last_hour):
        probes.append((source, last_hour))
        return datetime(2026, 7, 29, 18)

    monkeypatch.setattr(fetch, "resolve_latest_cycle", resolve)
    arguments, resolutions, _ = runplan.resolve_fetch_cycle(
        ["--source", "gfs", "--cycle", "latest", "--hours", "6",
         "--forecast-start-hour", "3", "--whole-cycle"])
    cycle = arguments[arguments.index("--cycle") + 1]
    request = {"source": "gfs", "cycle": cycle, "hours": 6,
               "forecast_start_hour": 3, "area": "25,-105,45,-85"}
    cache = go_cli.managed_download_dir(tmp_path, request)

    assert cache == go_cli.managed_download_dir(tmp_path, request)
    assert runplan.resolve_fetch_cycle(arguments) == (arguments, [], [])
    assert resolutions[0]["value"] == cycle == "2026-07-29T18"
    assert probes == [("gfs", 9)]
    assert not list(tmp_path.iterdir()), "Planning must not allocate downloads"


@pytest.mark.parametrize("state", ["missing", "invalid", "different"])
def test_an_interrupted_managed_cache_is_preserved_and_recovers_automatically(tmp_path, state):
    from woof import fetch

    request = {"source": "gfs", "cycle": "2026-07-29T18", "hours": 6,
               "area": "25,-105,45,-85"}
    first = go_cli.managed_download_dir(tmp_path, request)
    first.mkdir(parents=True)
    (first / "partial-input").write_bytes(b"interrupted transfer evidence")
    if state == "invalid":
        (first / fetch.FETCH_MANIFEST_NAME).write_text("{broken", encoding="utf-8")
    elif state == "different":
        _record_cached_request(first, request | {"cycle": "2026-07-30T00"})
    before = {p.name: p.read_bytes() for p in first.iterdir()}
    repaired = go_cli.managed_download_dir(tmp_path, request)
    assert repaired != first and not repaired.exists()
    _record_cached_request(repaired, request)
    assert go_cli.managed_download_dir(tmp_path, request) == repaired
    assert {p.name: p.read_bytes() for p in first.iterdir()} == before


def test_old_flat_downloads_do_not_break_a_new_quick_forecast(gfs_config, tmp_path):
    import tomllib

    root = tmp_path / "runs"
    request = tomllib.loads(gfs_config.read_text(encoding="utf-8"))["fetch"]
    legacy = root / "data"
    _record_cached_request(legacy, request | {"cycle": "2026-07-30T00"})
    before = {p.name: p.read_bytes() for p in legacy.iterdir()}
    plan = go_cli.plan_from_config(gfs_config, outdir=root)
    assert plan["data"].parent == root / "downloads"
    assert not plan["data"].exists()
    assert {p.name: p.read_bytes() for p in legacy.iterdir()} == before


def test_an_active_partial_download_is_shared_instead_of_duplicated(tmp_path, monkeypatch):
    from woof import fetch_guard

    monkeypatch.setenv(fetch_guard.LOCK_ROOT_ENV, str(tmp_path / "locks"))
    request = {"source": "gfs", "cycle": "2026-07-29T18", "hours": 6,
               "area": "25,-105,45,-85"}
    cache = go_cli.managed_download_dir(tmp_path, request)
    cache.mkdir(parents=True)
    (cache / "unfinished.part").write_bytes(b"downloading")
    with fetch_guard.hold("fetch-out", cache):
        assert go_cli.managed_download_dir(tmp_path, request) == cache
    assert go_cli.managed_download_dir(tmp_path, request) != cache


def test_table_route_cache_reuses_its_own_real_manifest_schema(tmp_path):
    from woof import fetch, fetch_routes

    request = {"source": "icon-eu", "cycle": "2026-07-29T18", "hours": 6}
    cache = go_cli.managed_download_dir(tmp_path, request)
    cache.mkdir(parents=True)
    plan = fetch_routes.resolve_request(
        "icon-eu", cycle=fetch.parse_cycle(request["cycle"], "icon-eu"), hours=6)
    payload = {"schema": fetch_routes.ROUTE_MANIFEST_SCHEMA,
               "request": fetch_routes._request_identity(plan), "files": []}
    (cache / fetch_routes.MANIFEST_NAME).write_text(json.dumps(payload), encoding="utf-8")
    assert go_cli.managed_download_dir(tmp_path, request) == cache
    payload["request"]["cycle"] = "2026-07-30T00Z"
    (cache / fetch_routes.MANIFEST_NAME).write_text(json.dumps(payload), encoding="utf-8")
    assert go_cli.managed_download_dir(tmp_path, request) != cache


@requires_cupy
def test_a_second_go_into_the_same_tree_is_refused_in_its_own_words(
        tmp_path, capsys, gfs_config, monkeypatch):
    """Re-running the command is the second thing anyone does.

    Every stage is create-only, so a chain refuses a tree an earlier run
    owns -- correctly, since merging two runs would publish receipts
    describing neither.  The runner's own message names
    ``--output-directory``, a flag nobody typed to get here, and it
    arrives after a stage has already been spent reaching it, so `go`
    answers first in the vocabulary of the command that was run.

    What CHANGED in 2.5.0 is which command reaches it.  A bare re-run no
    longer does: each run claims its own timestamped folder under
    ``--outdir``, which is what the collaborator running this in a loop
    asked for.  The refusal now guards the two ways a caller can still
    put two runs in one tree -- naming an existing run folder, and
    ``--run-stamp off`` -- and this pins the second, with its wording.
    """

    plan_root = tmp_path / "go"
    (plan_root / "authority").mkdir(parents=True)

    def fail_if_called(*args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("a stage ran after an up-front refusal")

    monkeypatch.setattr(subprocess, "Popen", fail_if_called)
    monkeypatch.setattr(go_cli, "resolve_bridge",
                        lambda: tmp_path / "gfs_grib2_bridge")

    assert cli_main(["go", str(gfs_config), "--outdir", str(plan_root),
                     "--run-stamp", "off"]) == 2
    printed = capsys.readouterr()
    message = printed.out + printed.err
    assert "already exists" in message
    assert "--outdir" in message
    assert "Traceback" not in message


def test_a_bare_second_go_claims_its_own_folder_rather_than_refusing(
        tmp_path, gfs_config):
    """The complaint, answered: two runs of one config, two trees.

    The stages stay create-only; what changed is that a run no longer
    walks into the last one's directory to find out.
    """

    plan_root = tmp_path / "go"
    (plan_root / "authority").mkdir(parents=True)   # a previous flat run
    first = go_cli.claim_run_root(
        go_cli.plan_from_config(gfs_config, outdir=plan_root))
    second = go_cli.claim_run_root(
        go_cli.plan_from_config(gfs_config, outdir=plan_root))
    assert first["root"] != second["root"]
    for plan in (first, second):
        assert plan["root"].parent == plan_root
        assert not plan["authority"].exists(), (
            "a freshly claimed run folder already holds an authority "
            "tree, so the create-only refusal would fire on it")


# ---------------------------------------------------------------------------
# The memory gate: before the download, never after it
# ---------------------------------------------------------------------------

def test_the_memory_gate_prices_both_phases_and_names_the_binding_one(
        gfs_config, tmp_path):
    """`go` must know what this run costs BEFORE `woof fetch` runs.

    The bug this closes: the only estimate anyone computed described the
    forecast, so a domain sized to a 12 GB card downloaded 81 GFS files
    and then died in preprocessing at 15.82 GB.  Both phases are priced
    here, from the config alone, with no device and no download.
    """
    plan = go_cli.plan_from_config(gfs_config, outdir=tmp_path / "out")
    gate = go_cli.memory_gate(plan)
    phases = gate["phases"]
    assert phases.ingest_priced
    assert phases.ingest.n_forcing_times >= 2
    # ONE resident forcing time: a single-domain adapter builds the start
    # time first, writes it into the prepared head and releases it before
    # the next time (woof/ingest/boundary_stream.py); a domain tree builds
    # it last (woof/ingest/lateral_bc.py:start_last_forcing_order).  Either
    # way nothing is held across the loop, and the gate prices that.
    assert phases.ingest.resident_times == 1
    assert phases.binding_phase in ("forecast", "ingest")
    assert phases.binding_phase in gate["verdict"]
    assert "forecast" in gate["verdict"] and "ingest" in gate["verdict"]


def test_the_memory_gate_refuses_ahead_of_the_fetch_stage(gfs_config,
                                                          tmp_path,
                                                          monkeypatch,
                                                          capsys):
    """A refusal must land with the download still un-started.

    Every stage command is replaced by a recorder, so if `fetch` appears
    in the record at all the gate ran too late.
    """
    ran: list[str] = []

    def _record(label, command, **kwargs):
        ran.append(label)

    monkeypatch.setattr(go_cli, "_run_stage", _record)
    monkeypatch.setattr(go_cli, "resolve_bridge", lambda: Path("bridge"))
    monkeypatch.setattr(
        go_cli, "memory_gate",
        lambda plan, **kw: {
            "verdict": "preprocessing (ingest) is the memory-binding phase "
                       "at 40.00 GiB peak envelope",
            "refuse": True, "warn": True, "free_bytes": 8 * 1024 ** 3,
        })
    args = _args(gfs_config, tmp_path / "out")
    with pytest.raises(go_cli.GoRefusal) as refusal:
        go_cli.go_main(args)
    assert ran == []
    message = str(refusal.value)
    assert "BEFORE the fetch stage" in message
    assert "memory-binding phase" in message
    # The measured free-VRAM accuracy stays; the remedy must be
    # REACHABLE: the 3080 walk followed `woof domain --vram-gib <free>`
    # verbatim and was refused at every grid size, because the flag
    # names a card and the number fed to it was a free-VRAM figure.
    assert "8.00 GiB free right now" in message
    assert "woof domain" in message
    assert "--vram-gib" not in message
    assert "--no-memory-gate" in message


def test_unstaged_geography_is_refused_ahead_of_the_fetch_stage(
        gfs_config, tmp_path, monkeypatch):
    """The first-run wall, measured on the 1.4.0 wheel and closed here.

    `woof doctor` prints `MISSING WPS_GEOG ... -> woof fetch-geog` and
    exits 0, which is right: a ~16 GB download nobody opted into is not
    a broken install.  `woof go` then ran three stages, downloaded the
    forcing, and died in the fourth with the whole of:

        FAILED  prepare (exit 2)
          rw-wps --source gfs: /.../WPS_GEOG/topo_gmted2010_30s/index.

    No verb, no remedy, and the answer had been on screen a minute
    earlier from a different command.  `go` asks the same check now, on
    the same side of the download as the memory gate.
    """
    ran: list[str] = []
    monkeypatch.setattr(go_cli, "_run_stage",
                        lambda label, command, **kw: ran.append(label))
    monkeypatch.setattr(go_cli, "resolve_bridge", lambda: Path("bridge"))

    absent = tmp_path / "never-fetched" / "WPS_GEOG"
    args = _args(gfs_config, tmp_path / "out", geog_root=absent)
    with pytest.raises(go_cli.GoRefusal) as refusal:
        go_cli.go_main(args)
    assert ran == [], "a stage ran before the geography check"

    message = str(refusal.value)
    assert "woof fetch-geog" in message
    assert str(absent) in message
    assert "--geog-root" in message
    # The layered half carries the why, including the accurate account of
    # doctor's exit 0 on the same gap.
    assert "before the fetch stage" in message
    assert "exits 0" in message


def test_a_partial_geography_tree_names_what_is_wrong_with_it(tmp_path):
    """A dataset present but unindexed is a partial download, not a
    missing opt-in, and the refusal has to distinguish them by name."""

    geog = _staged_geog_tree(tmp_path)
    victim = sorted(p for p in geog.iterdir() if p.is_dir())[0]
    (victim / "index").unlink()

    message = go_cli.geography_refusal(geog)
    assert message is not None
    assert victim.name in message
    assert "woof fetch-geog" in message


def test_a_staged_geography_tree_passes_the_check(staged_geog):
    """Non-vacuity: the check says yes to a tree shaped like a real one."""

    assert go_cli.geography_refusal(staged_geog) is None


def test_the_memory_gate_warns_without_blocking_and_can_be_skipped(
        gfs_config, tmp_path, monkeypatch, capsys):
    """Over budget but inside free VRAM is an advisory, not a refusal."""
    ran: list[str] = []

    def _stage(label, command, **kwargs):
        ran.append(label)
        if label == "manifest":  # stop the chain where the real work starts
            raise go_cli.GoStageFailed(9)

    monkeypatch.setattr(go_cli, "_run_stage", _stage)
    monkeypatch.setattr(go_cli, "resolve_bridge", lambda: Path("bridge"))
    monkeypatch.setattr(
        go_cli, "memory_gate",
        lambda plan, **kw: {"verdict": "the forecast is the memory-binding "
                                       "phase at 9.00 GiB peak envelope",
                            "refuse": False, "warn": True,
                            "free_bytes": 12 * 1024 ** 3})
    assert go_cli.go_main(_args(gfs_config, tmp_path / "warn")) == 9
    assert ran == ["authority", "fetch", "manifest"]
    printed = capsys.readouterr().out
    assert "WARNING" in printed
    assert "memory-binding phase" in printed

    called: list[str] = []
    ran.clear()
    monkeypatch.setattr(
        go_cli, "memory_gate",
        lambda plan, **kw: called.append("gate") or {})
    args = _args(gfs_config, tmp_path / "skipped")
    args.no_memory_gate = True
    assert go_cli.go_main(args) == 9
    assert called == []
    assert ran == ["authority", "fetch", "manifest"]


# ---------------------------------------------------------------------------
# The memory gate must not stand up a CUDA context in the go process
# ---------------------------------------------------------------------------

#: A card as the subprocess probe reports one: the two device questions
#: the gate prices against, answered together.
_PROBE_PROFILE = {
    "name": "pinned probe card",
    "multiprocessor_count": 170,
    "max_threads_per_multiprocessor": 1536,
    "default_stack_limit_bytes": 1024,
}


@pytest.fixture
def _in_process_cupy_poisoned(monkeypatch):
    """Any in-process touch of cupy's CUDA half is the defect, said loudly.

    ``memGetInfo`` and ``deviceGetLimit`` cannot be asked without
    standing up a CUDA primary context, and the ``woof go`` process
    outlives its own gate as nothing but the stage orchestrator and
    progress printer -- so a context stood up there sits on the card for
    the entire chain (measured 0.486 GiB on the RTX 5090) as a consumer
    no term of the budget the gate just computed names.  The poison
    replaces cupy in ``sys.modules``: importing it stays legal (imports
    allocate nothing), touching any attribute raises.
    """

    import sys as _sys
    import types

    poison = types.ModuleType("cupy")

    def _refuse(name):
        raise AssertionError(
            f"the go process touched in-process cupy.{name}: that stands "
            "up a CUDA primary context that then sits on the card for the "
            "whole chain while this process does nothing but print "
            "progress")

    poison.__getattr__ = _refuse
    monkeypatch.setitem(_sys.modules, "cupy", poison)


def test_the_gate_asks_the_card_in_a_subprocess_and_prices_its_answer(
        gfs_config, tmp_path, monkeypatch, _in_process_cupy_poisoned):
    """The gate's device questions run in a short-lived subprocess.

    The card's answers must be the ones the gate prices -- free VRAM
    from the probe, the reserve from the probe's device profile -- and
    in-process cupy (poisoned above) must never be touched.  On code
    that still asks in-process, the poison sends the gate down its
    no-device path and the first assertion states the defect: the probe
    seam was ignored.
    """

    from woof.core import preflight

    payload = {"free_bytes": 30 * 1024 ** 3, "total_bytes": 32 * 1024 ** 3,
               "profile": dict(_PROBE_PROFILE)}
    # raising=False so unfixed code FAILS the assertion below (the
    # defect, stated) instead of erroring on a not-yet-existing seam.
    monkeypatch.setattr(preflight, "device_memory_probe_subprocess",
                        lambda **_kwargs: dict(payload), raising=False)
    plan = go_cli.plan_from_config(gfs_config, outdir=tmp_path / "probe")
    gate = go_cli.memory_gate(plan)
    assert gate["free_bytes"] == payload["free_bytes"]

    from woof.core.preflight import (EXTERNAL_MARGIN_BYTES, ReservePolicy,
                                      _load_experiment_any,
                                      profile_from_device_probe)

    profile = profile_from_device_probe(payload)
    assert profile is not None
    assert profile.name == _PROBE_PROFILE["name"]
    exp = _load_experiment_any(plan["config"])
    # The budget the ENVELOPE is warned against is free VRAM less the
    # other-process margin -- the only thing the envelope does not
    # already model.  It used to be free less the whole ALLOCATION
    # reserve, and that reserve carries the CUDA context and the
    # local-memory backing store the envelope carries too, so the gate
    # warned about a card the configuration fits (task 206).
    assert gate["budget_bytes"] == (payload["free_bytes"]
                                    - EXTERNAL_MARGIN_BYTES)
    # The allocation reserve is still what the allocation gate spends,
    # and it is still priced against the card the PROBE described.
    reserve = ReservePolicy.n0_alloc(
        exp, profile=profile,
        estimate_bytes=gate["phases"].forecast.alloc_estimate_bytes)
    assert reserve.device_overhead_bytes > 0
    peak = gate["phases"].peak_envelope_bytes
    assert gate["refuse"] == (peak > payload["free_bytes"])
    assert gate["warn"] == (peak > gate["budget_bytes"])


def test_a_card_the_probe_cannot_see_never_refuses(
        gfs_config, tmp_path, monkeypatch, _in_process_cupy_poisoned):
    """No readable device (a planning box, a CI runner): the phases are
    still priced and the verdict still prints, but nothing refuses on a
    card nobody measured -- through the probe seam, and still without
    touching in-process cupy."""

    from woof.core import preflight

    monkeypatch.setattr(preflight, "device_memory_probe_subprocess",
                        lambda **_kwargs: None, raising=False)
    plan = go_cli.plan_from_config(gfs_config, outdir=tmp_path / "nodev")
    gate = go_cli.memory_gate(plan)
    assert gate["free_bytes"] is None
    assert gate["refuse"] is False
    assert gate["warn"] is False
    assert gate["verdict"]
    assert "phases" in gate


def _staged_geog_tree(root: Path) -> Path:
    """A WPS_GEOG tree shaped the way `woof fetch-geog` leaves one.

    Directory names and the `index` file are the whole of what the
    prepare stage's precondition reads, so the nine empty-but-indexed
    directories are a faithful stand-in for 16 GB of terrain.  Names
    come from `geog_assets`, the module that stages the real one -- the
    same source doctor reads -- so this fixture cannot drift from the
    check under test.
    """

    # The one stand-in (tests/_staged_geog.py) that stages each container
    # archive's child indexes (soilgrids, the permissive default soil set
    # since c3e7e0d647), which is where the go door's geography check looks.
    from _staged_geog import staged_geog_tree

    return staged_geog_tree(root / "WPS_GEOG")


@pytest.fixture(scope="module")
def staged_geog(tmp_path_factory):
    return _staged_geog_tree(tmp_path_factory.mktemp("geog"))


_DEFAULT_GEOG: list[Path] = []


def _args(config, outdir, geog_root=None):
    import argparse
    import tempfile

    if geog_root is None:
        # Every chain test past the memory gate now needs a usable
        # geography tree, because `go` checks for one there (it used to
        # find out in the prepare stage, after the download).  One
        # stand-in for the whole file rather than a fixture threaded
        # through thirty call sites.
        if not _DEFAULT_GEOG:
            _DEFAULT_GEOG.append(
                _staged_geog_tree(Path(tempfile.mkdtemp(prefix="gowps-"))))
        geog_root = _DEFAULT_GEOG[0]
    return argparse.Namespace(
        config=config, outdir=outdir, data_dir=None, geog_root=geog_root,
        dry_run=False, no_memory_gate=False, explain=False)


# ---------------------------------------------------------------------------
# The forecast lead: a start of cycle + K, carried through the chain
# ---------------------------------------------------------------------------

def test_go_carries_the_configs_forecast_lead_into_its_fetch(tmp_path):
    """A lead in [fetch] is essential, exactly like cycle/hours/area.

    ``woof go`` is step 3 of what the wizard itself prints, so a config
    whose start_time is cycle + K has to reach a fetch that downloads
    f{K}..  Ignoring the lead here would download f000.. and then hand
    the front door a series that does not carry the lead the experiment
    starts from -- a refusal produced by the orchestrator, from a config
    that is entirely correct.
    """

    config = _emit(tmp_path, "lead", "--forecast-start-hour", "174")
    plan = go_cli.plan_from_config(config, outdir=tmp_path / "go")
    assert plan["forecast_start_hour"] == 174
    # The cycle stays the CYCLE; the lead is carried beside it.
    assert plan["cycle"] == "2026-07-29T18"
    fetch = go_cli.fetch_command(plan)
    assert "--forecast-start-hour" in fetch
    assert fetch[fetch.index("--forecast-start-hour") + 1] == "174"

    # And an analysis-start config still prints the command it always did.
    plain = go_cli.plan_from_config(
        _emit(tmp_path, "plain"), outdir=tmp_path / "go-plain")
    assert plain["forecast_start_hour"] == 0
    assert "--forecast-start-hour" not in go_cli.fetch_command(plain)


def test_go_derives_the_statics_corridor_from_a_follow_config(tmp_path,
                                                              gfs_config):
    """A config that declares a [relocation] follow source gets
    --statics-corridor on the prepare stage; every other config's
    prepare line is byte-for-byte what it always was."""

    two_domain = tmp_path / "follow.toml"
    two_domain.write_text(_write_follow_tree_config(gfs_config),
                          encoding="utf-8")
    plan = go_cli.plan_from_config(two_domain, outdir=tmp_path / "go")
    assert plan["statics_corridor"] is True
    command = go_cli.prepare_command(
        plan, tmp_path / "bridge", manifest=tmp_path / "m.json",
        manifest_sha256="a" * 64, cycle_stamp="2026-07-29_18:00:00",
        geog_root=tmp_path / "geog")
    assert "--statics-corridor" in command

    plain = go_cli.plan_from_config(gfs_config, outdir=tmp_path / "go2")
    assert plain["statics_corridor"] is False
    unchanged = go_cli.prepare_command(
        plain, tmp_path / "bridge", manifest=tmp_path / "m.json",
        manifest_sha256="a" * 64, cycle_stamp="2026-07-29_18:00:00",
        geog_root=tmp_path / "geog")
    assert "--statics-corridor" not in unchanged


def _write_follow_tree_config(gfs_config) -> str:
    """The wizard's own single-domain emission, grown into a two-domain
    follow tree: a child plus a [relocation] itinerary on it."""

    from woof.experiment import load_experiment

    base = load_experiment(gfs_config)
    dt = float(base.root.run.dt)
    text = gfs_config.read_text(encoding="utf-8")
    return text + f"""
[[domain]]
grid_id = 2
parent_id = 1
i_parent_start = 30
j_parent_start = 30
parent_grid_ratio = 3
parent_time_step_ratio = 3
nx = 45
ny = 45
history_interval_s = 3600.0

[relocation]
enabled = true
grid_id = 2

[[relocation.move]]
at_seconds = {dt * 2:.1f}
di_parent_cells = 1
dj_parent_cells = 0
"""


def test_the_printed_rw_wps_line_and_go_agree_on_the_corridor(tmp_path,
                                                              gfs_config):
    """The pasted manual line and go's driven line derive the corridor
    flag from the same config predicate, so neither can drift: both
    carry --statics-corridor for a follow config."""

    from woof.fetch import author_gfs_front_door_manifest

    config = tmp_path / "follow.toml"
    config.write_text(_write_follow_tree_config(gfs_config),
                      encoding="utf-8")
    config.with_suffix(".namelist.wps").write_bytes(
        gfs_config.with_suffix(".namelist.wps").read_bytes())
    plan = go_cli.plan_from_config(config, outdir=tmp_path / "go")
    bridge = tmp_path / "gfs_grib2_bridge"
    bridge.write_bytes(b"stub")
    _stage_a_fetched_directory(plan["data"], config, plan["authority"])

    printed: list[str] = []
    manifest, digest = author_gfs_front_door_manifest(
        out=plan["data"], bridge=bridge,
        wps_namelist=plan["authority"] / "namelist.wps",
        experiment_config=plan["authority"] / "experiment.toml",
        progress=printed.append)
    theirs = _printed_flags(
        [line for block in printed for line in str(block).splitlines()])
    mine = _flags(go_cli.prepare_command(
        plan, bridge, manifest=manifest, manifest_sha256=digest,
        cycle_stamp="2026-07-29_18:00:00",
        geog_root=Path(theirs["--geog-root"])))
    assert theirs.get("--statics-corridor") is True
    assert mine.get("--statics-corridor") is True
    assert set(mine) == set(theirs)


# ---------------------------------------------------------------------------
# UX finding N18 (2026-08-18 upgrader walk): the run-folder line prints on
# the real path BEFORE the gates, so a refused `woof go` still teaches the
# 2.5.0 layout.  Both of the walk's real attempts refused (memory gate,
# then WPS_GEOG) and the announcement -- which --dry-run prints second --
# never appeared.
# ---------------------------------------------------------------------------

def test_a_refused_real_go_still_announces_the_run_folder(
        gfs_config, tmp_path, monkeypatch, capsys):
    ran: list[str] = []
    monkeypatch.setattr(go_cli, "_run_stage",
                        lambda label, command, **kw: ran.append(label))
    monkeypatch.setattr(go_cli, "resolve_bridge", lambda: Path("bridge"))
    monkeypatch.setattr(
        go_cli, "memory_gate",
        lambda plan, **kw: {
            "verdict": "the forecast is the memory-binding phase at "
                       "40.00 GiB peak envelope",
            "refuse": True, "warn": True, "free_bytes": 8 * 1024 ** 3,
        })
    args = _args(gfs_config, tmp_path / "out")
    with pytest.raises(go_cli.GoRefusal):
        go_cli.go_main(args)
    assert ran == [], "the gate still fires before any stage"
    printed = capsys.readouterr().out
    assert "go: run folder" in printed, (
        "a refused real run must still say where a run WOULD land")
    # The same line the dry run prints second: folder name, case root,
    # and the subtree inventory.
    assert "authority/, prepared/, run/ and png/" in printed


def test_the_real_run_folder_line_agrees_with_the_dry_run(
        gfs_config, tmp_path, monkeypatch, capsys):
    """One line, one function, both paths -- the path a reader plans
    against is the path they get."""

    monkeypatch.setattr(go_cli, "resolve_bridge", lambda: Path("bridge"))
    args = _args(gfs_config, tmp_path / "out")
    args.dry_run = True
    assert go_cli.go_main(args) == 0
    dry = [line for line in capsys.readouterr().out.splitlines()
           if line.startswith("go: run folder")]
    assert len(dry) == 1

    monkeypatch.setattr(
        go_cli, "memory_gate",
        lambda plan, **kw: {
            "verdict": "the forecast is the memory-binding phase",
            "refuse": True, "warn": True, "free_bytes": 8 * 1024 ** 3,
        })
    args = _args(gfs_config, tmp_path / "out")
    with pytest.raises(go_cli.GoRefusal):
        go_cli.go_main(args)
    real = [line for line in capsys.readouterr().out.splitlines()
            if line.startswith("go: run folder")]
    assert len(real) == 1
    # Same shape up to the stamp (the two invocations claim different
    # stamped names on a shared case root).
    assert dry[0].split("run-", 1)[0] == real[0].split("run-", 1)[0]


def test_a_go_whose_gates_take_two_seconds_announces_one_folder_that_exists(
        gfs_config, tmp_path, monkeypatch, capsys):
    """F10: every run announced two run folders, and the first never existed.

    The plan named ``run-<launch>Z`` before the gates and the claim read
    the clock again after them.  The memory and geography gates take more
    than a second on a real card, so the "correction" line fired on every
    run and the first path printed was a folder nobody made.  Here the
    gate really takes two seconds; the chain is stopped at its first stage,
    just after the claim.
    """

    import time

    def slow_gate(plan, **kw):
        time.sleep(2.1)
        return {"verdict": "fits", "refuse": False, "warn": False,
                "free_bytes": 30 * 1024 ** 3}

    def stop_at_first_stage(label, command, **kw):
        raise go_cli.GoRefusal(f"test stops the chain before {label}")

    monkeypatch.setattr(go_cli, "memory_gate", slow_gate)
    monkeypatch.setattr(go_cli, "_require_forecast_device", lambda: None)
    monkeypatch.setattr(go_cli, "resolve_bridge", lambda: Path("bridge"))
    monkeypatch.setattr(go_cli, "_run_stage", stop_at_first_stage)
    case_root = tmp_path / "out"
    with pytest.raises(go_cli.GoRefusal, match="before authority"):
        go_cli.go_main(_args(gfs_config, case_root))
    lines = [line for line in capsys.readouterr().out.splitlines()
             if line.startswith("go: run folder")]
    assert len(lines) == 1, lines
    name = lines[0].split("go: run folder ", 1)[1].split(" ", 1)[0]
    assert (case_root / name).is_dir(), (name, sorted(
        path.name for path in case_root.iterdir()))
    assert [path.name for path in case_root.iterdir()
            if path.name.startswith("run-")] == [name]


# ---------------------------------------------------------------------------
# The run-folder line's "cached at" clause named `<case_root>/data`
# unconditionally, so `--data-dir` announced a directory no stage ever
# writes to while the fetch, manifest and prepare stages all used the
# directory the reader named.  The wizard's own "next" block ends with a
# `woof go ... --data-dir <path>` line, so this was the first thing a
# reader following the printed route saw, and it sent them looking for
# their download in an empty path.
# ---------------------------------------------------------------------------

def _announced_cache(printed: str) -> str:
    """The download directory the run-folder line ANNOUNCES."""

    lines = [line for line in printed.splitlines()
             if line.startswith("go: run folder")]
    assert len(lines) == 1, f"expected one run-folder line, got {lines!r}"
    marker = "the download is cached at "
    assert marker in lines[0], lines[0]
    return lines[0].split(marker, 1)[1].strip()


def _fetch_stage_out(printed: str) -> str:
    """The download directory the fetch stage WRITES, off the same run.

    Read from the composed command the dry run prints rather than from a
    second plan: the two must agree within ONE invocation, which is the
    whole claim.
    """

    for line in printed.splitlines():
        stripped = line.strip()
        if "woof.cli fetch" in stripped and "--cycle" in stripped:
            tokens = shlex.split(stripped, posix=False)
            return tokens[tokens.index("--out") + 1].strip("'\"")
    raise AssertionError(f"no composed fetch stage in:\n{printed}")


@pytest.mark.parametrize("named_cache", [True, False])
def test_the_announced_download_cache_is_the_one_the_stages_use(
        gfs_config, tmp_path, capsys, monkeypatch, named_cache):
    """One directory, said once and used once.

    Parameterized over both cases on purpose: the default -- the
    download under the case root, cached across runs of one config --
    is what the line was written for and must not move, and
    ``--data-dir`` is the case it got wrong.
    """
    from woof.render_layout import fs_path

    def canonical(path):
        # Windows' long-path spelling and the ordinary spelling identify
        # the same directory.  Compare the filesystem paths the stages use.
        return Path(fs_path(path, descend=True)).resolve()

    monkeypatch.setattr(go_cli, "resolve_bridge", lambda: tmp_path / "bridge")
    argv = ["go", str(gfs_config), "--dry-run",
            "--outdir", str(tmp_path / "out")]
    if named_cache:
        argv += ["--data-dir", str(tmp_path / "mycache")]
    assert cli_main(argv) == 0
    printed = capsys.readouterr().out

    announced = canonical(_announced_cache(printed))
    used = canonical(_fetch_stage_out(printed))
    assert announced == used, (
        f"go announces the download cache at {announced} but the fetch "
        f"stage writes to {used}")
    # And it is the directory the reader NAMED, not one derived from the
    # run root: equality above would also hold if both had drifted.
    if named_cache:
        assert announced == canonical(tmp_path / "mycache")
    else:
        assert announced.parent == canonical(tmp_path / "out" / "downloads")
