"""A long forecast's pictures start however many frames it has.

The render stage named every frame on its command line.  A 48 h nested
forecast under an ordinary Documents folder is 242 frames, which came to a
36,519 character command, and Windows starts no program whose command line
passes 32,767: the forecast finished and ``[WinError 206] The filename or
extension is too long`` escaped as an exception, so the stage observer heard
the stage begin and never end.  These tests hold the three halves of the
repair: a stage whose command would pass the limit hands its frames over in
a file (one that fits runs as it always did), the renderer is started on
names short enough to fit, and a stage whose program cannot start is a
failed stage like any other.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from woof import go_cli, rustwx
from woof import render as render_module

#: The longest command line Windows starts a program with.
WINDOWS_COMMAND_LIMIT = 32_767


def _long_folder(root: Path) -> Path:
    """Where an ordinary desktop install keeps one long forecast's frames."""

    return (root / "Zoe Weather" / "Documents" / "WOOF-ArWen forecasts"
            / "Oklahoma long forecast"
            / "run-20260926-190000Z_i202609260000Z" / "run" / "wrfout")


def _series(folder: Path, *, create: bool = False) -> list[Path]:
    """48 hours of hourly parent frames and a 15 minute nest: 242 frames."""

    start = datetime(2026, 9, 26)
    frames = []
    for domain, step_minutes, count in ((1, 60, 49), (2, 15, 193)):
        for k in range(count):
            valid = start + timedelta(minutes=step_minutes * k)
            frames.append(folder / (f"wrfout_d{domain:02d}_"
                                    f"{valid:%Y-%m-%d_%H_%M_%S}"))
    if create:
        folder.mkdir(parents=True, exist_ok=True)
        for frame in frames:
            frame.write_bytes(b"one durable history frame")
    assert len(frames) == 242
    return frames


def _plan(folder: Path) -> dict:
    run = folder.parent
    return {"run": run, "wrfout_dir": folder, "render": run.parent / "render",
            "render_products": "all"}


def _length(command) -> int:
    return len(subprocess.list2cmdline([str(part) for part in command]))


def spelled_out(command) -> list[str]:
    """A render stage's command with its frame file read back in.

    For a stub standing in for the stage's process: a long series' frames
    arrive in a file that is gone once the stage ends, so the stub reads it
    while the stage is still running and gets the command spelled out,
    every frame on it.  A command with no frame file comes back as it is.
    """

    command = [str(part) for part in command]
    if "--inputs-from" not in command:
        return command
    at = command.index("--inputs-from")
    record = json.loads(Path(command[at + 1]).read_text(encoding="utf-8"))
    spelled = command[:at] + record["wrfout"] + command[at + 2:]
    for frame in record["context_wrfout"]:
        spelled += ["--context-wrfout", frame]
    return spelled


# ---------------------------------------------------------------------------
# The stage's command
# ---------------------------------------------------------------------------


def test_a_48_hour_nested_series_starts_under_the_windows_limit(tmp_path):
    folder = _long_folder(tmp_path)
    frames = _series(folder)
    context = frames[:2]
    plan = _plan(folder)

    # What the stage used to run: every frame spelled out, past the limit.
    spelled = go_cli.render_command(plan, frames, context_frames=context)
    assert _length(spelled) > WINDOWS_COMMAND_LIMIT

    listing = tmp_path / "inputs.json"
    command = go_cli.render_command(plan, frames, context_frames=context,
                                    inputs_file=listing)
    assert _length(command) < 2_000
    assert command[command.index("--inputs-from") + 1] == str(listing)
    assert not any("wrfout_d0" in part for part in command)
    assert "--context-wrfout" not in command
    record = json.loads(listing.read_text(encoding="utf-8"))
    assert record["schema"] == render_module.RENDER_INPUTS_SCHEMA
    assert record["wrfout"] == [str(frame) for frame in frames]
    assert record["context_wrfout"] == [str(frame) for frame in context]


def test_a_printed_command_still_names_its_frames(tmp_path):
    """The line a reader pastes has no frame file to point at."""

    frames = _series(_long_folder(tmp_path))[:3]
    command = go_cli.render_command(_plan(_long_folder(tmp_path)), frames)
    assert "--inputs-from" not in command
    assert [str(frame) for frame in frames] == [
        part for part in command if "wrfout_d0" in part]


def _frames_named(command) -> list[str]:
    """The frames a render command draws, read from its frame file if it has one."""

    command = [str(part) for part in command]
    if "--inputs-from" in command:
        listing = Path(command[command.index("--inputs-from") + 1])
        return json.loads(listing.read_text(encoding="utf-8"))["wrfout"]
    return [part for at, part in enumerate(command)
            if "wrfout_d0" in part and command[at - 1] != "--context-wrfout"]


def test_the_render_stage_hands_a_long_series_over_in_a_file_it_removes(
        tmp_path, monkeypatch):
    monkeypatch.setattr(render_module, "drawable_engine",
                        lambda: ("rust", "declared by the test"))
    folder = _long_folder(tmp_path)
    frames = _series(folder, create=True)
    seen = []

    def stage(label, command, **kw):
        command = [str(part) for part in command]
        listing = (Path(command[command.index("--inputs-from") + 1])
                   if "--inputs-from" in command else None)
        seen.append((label, _length(command), listing, _frames_named(command)))

    monkeypatch.setattr(go_cli, "_run_stage", stage)
    assert go_cli._render_stage(_plan(folder), explain=False, observer=None)

    # Every pass starts; the one over the whole series goes through a file.
    assert seen and all(label == "render" and length < WINDOWS_COMMAND_LIMIT
                        for label, length, _, _ in seen)
    whole = [entry for entry in seen
             if sorted(entry[3]) == sorted(str(f) for f in frames)]
    assert len(whole) == 1
    listing = whole[0][2]
    assert listing is not None
    # The file is the stage's own and does not outlive it.
    assert not listing.exists()


def test_a_series_that_fits_runs_the_command_it_always_did(
        tmp_path, monkeypatch):
    """Only a command past the budget changes: a short one names its frames."""

    folder = tmp_path / "run" / "wrfout"
    frames = _series(folder, create=True)[:6]
    calls = []
    monkeypatch.setattr(go_cli, "_run_stage",
                        lambda label, command, **kw: calls.append((command, kw)))

    go_cli.run_render_pass(_plan(folder), frames, explain=False)

    assert len(calls) == 1
    command, kw = calls[0]
    assert "--inputs-from" not in command and "shown" not in kw
    assert command == go_cli.render_command(_plan(folder), frames)


def test_the_stage_event_names_the_frames_not_the_frame_file(
        tmp_path, monkeypatch):
    """events.jsonl records a command a reader can still use after the stage.

    The process gets ``--inputs-from`` and a file the stage removes when it
    ends; the ``stage_started`` record carries the frames spelled out, as it
    did before the file existed, in the same list-of-strings field.
    """

    from woof.chain_events import GoChainEvents, read_chain_events

    monkeypatch.setattr(render_module, "drawable_engine",
                        lambda: ("rust", "declared by the test"))
    folder = _long_folder(tmp_path)
    frames = _series(folder, create=True)
    ran = []

    class _Renderer:
        """The render process: reads its frame file while it runs."""

        def __init__(self, command, **_kwargs):
            ran.append((list(command), spelled_out(command)))
            self.pid = 4242
            self.returncode = 0
            # Empty pipes, read while the renderer runs.
            self.stdout = io.StringIO()
            self.stderr = io.StringIO()

        def wait(self, timeout=None):
            return self.returncode

        def communicate(self):
            return "", ""

    monkeypatch.setattr(go_cli.subprocess, "Popen", _Renderer)
    stream = tmp_path / "events.jsonl"
    observer = GoChainEvents()
    observer.open(stream)
    try:
        go_cli._render_stage(_plan(folder), explain=False, observer=observer)
    finally:
        observer.close()

    started = [record["command"] for record in read_chain_events(stream)
               if record["event"] == "stage_started"
               and record["stage"] == "render"]
    assert started and len(started) == len(ran)
    for recorded, (command, spelled) in zip(started, ran):
        assert all(isinstance(part, str) for part in recorded)
        assert "--inputs-from" not in recorded
        # Each record is exactly the command its process ran, with any
        # frame file read back in.
        assert recorded == spelled
    whole = [at for at, recorded in enumerate(started)
             if sorted(_frames_named(recorded)) == sorted(str(f) for f in frames)]
    assert len(whole) == 1
    command = ran[whole[0]][0]
    # The process itself was handed the short command.
    assert "--inputs-from" in command and _length(command) < 2_000
    listing = Path(command[command.index("--inputs-from") + 1])
    assert not listing.exists()


def test_a_stage_records_the_command_it_was_shown_and_runs_its_own(capsys):
    observer = _Observer()
    go_cli._run_stage("render", [sys.executable, "-c", "pass"],
                      explain=False, observer=observer, heartbeat_seconds=60.0,
                      shown=["woof", "render", "wrfout_d01_a", "wrfout_d01_b"])
    begin = observer.events[0]
    assert begin == ("stage_begin", {
        "label": "render",
        "command": ["woof", "render", "wrfout_d01_a", "wrfout_d01_b"]})
    assert observer.events[-1][0] == "stage_end"
    assert observer.events[-1][1]["ok"] is True
    capsys.readouterr()


def test_the_frame_file_reaches_render_as_the_frames_themselves(tmp_path):
    folder = _long_folder(tmp_path)
    frames = _series(folder)[:4]
    context = [folder / "wrfout_d01_2026-09-25_23_00_00"]
    listing = render_module.write_render_inputs(
        tmp_path / "inputs.json", frames, context)

    parser = argparse.ArgumentParser()
    render_module.register_cli(parser.add_subparsers(dest="command"))
    out = ["--series", "--out", str(tmp_path / "png")]
    by_file = parser.parse_args(["render", "--inputs-from", str(listing), *out])
    spelled = parser.parse_args(
        ["render", *map(str, frames), *out,
         *[token for frame in context for token in ("--context-wrfout", str(frame))]])

    assert render_module._fold_render_inputs(by_file) is None
    assert render_module._fold_render_inputs(spelled) is None
    assert by_file.wrfout == spelled.wrfout == frames
    assert by_file.context_wrfout == spelled.context_wrfout == context


def test_the_explain_pointer_names_the_frames_not_the_frame_file(tmp_path):
    """The stage removes its frame file, so a pointer at it is a dead end."""

    from woof import explain

    folder = _long_folder(tmp_path)
    frames = _series(folder)[:2]
    context = [folder / "wrfout_d01_2026-09-25_23_00_00"]
    listing = render_module.write_render_inputs(
        tmp_path / "inputs.json", frames, context)
    parser = argparse.ArgumentParser()
    render_module.register_cli(parser.add_subparsers(dest="command"))
    typed = ["render", "--inputs-from", str(listing), "--series",
             "--out", "png"]
    args = parser.parse_args(typed)
    with explain.explain_scope(False):
        explain.set_invocation(typed)
        assert render_module._fold_render_inputs(args) is None
        line = explain.reinvocation()
    assert line.startswith("woof render ")
    assert "--inputs-from" not in line and str(listing) not in line
    assert all(str(frame) in line for frame in frames)
    assert f"--context-wrfout \"{context[0]}\"" in line
    assert line.endswith("--series --out png --context-wrfout "
                         f"\"{context[0]}\"")


def test_a_refusal_before_render_starts_names_the_frames_too(
        tmp_path, capsys, monkeypatch):
    """The provenance gate and the capability preflight refuse before render_main.

    Their pointer is the recorded invocation, so the frame file is read in
    as the command is parsed, not only when render_main runs: otherwise
    the line names a file the stage has already removed.
    """

    import woof.cli as cli
    from woof import explain, provenance_gate

    frames = _series(_long_folder(tmp_path))[:2]
    listing = render_module.write_render_inputs(tmp_path / "inputs.json", frames)

    def refuse(command, *_, **__):
        raise ValueError(explain.layered(
            "this install cannot say which tree is running",
            "the mechanism, shown under --explain"))

    monkeypatch.setattr(provenance_gate, "announce", refuse)
    code = cli.main(["render", "--inputs-from", str(listing), "--series",
                     "--out", str(tmp_path / "png")])
    err = capsys.readouterr().err
    assert code == 2
    assert "this install cannot say which tree is running" in err
    assert str(listing) not in err and "--inputs-from" not in err
    assert all(str(frame) in err for frame in frames)


@pytest.mark.parametrize("content,words", [
    (None, "could not be read"),
    ("{not json", "could not be read"),
    ('{"wrfout": []}', "not a frame list this version reads"),
    ('{"schema": "gpuwm.render-inputs/v1", "wrfout": [3]}',
     '"wrfout" must be a list of file paths'),
])
def test_a_bad_frame_file_is_refused_in_words(tmp_path, capsys, content, words):
    import woof.cli as cli

    listing = tmp_path / "inputs.json"
    if content is not None:
        listing.write_text(content, encoding="utf-8")
    code = cli.main(["render", "--inputs-from", str(listing),
                     "--out", str(tmp_path / "png")])
    captured = capsys.readouterr()
    assert code == 2
    assert words in captured.err
    assert "Traceback" not in captured.err + captured.out


def test_a_series_too_long_to_spell_out_points_at_its_frame_file(
        tmp_path, capsys, monkeypatch):
    """242 frames spelled out were a 48,000 character pointer no shell runs.

    Past the command-line budget the pointer keeps ``--inputs-from`` and
    the file, which a failed stage leaves in place, so the line runs as
    printed and stays short; the frames still reach render.
    """

    import woof.cli as cli
    from woof import explain, provenance_gate

    frames = _series(_long_folder(tmp_path))
    listing = render_module.write_render_inputs(tmp_path / "inputs.json", frames)
    spelled = ["woof", "render", *map(str, frames), "--series", "--out", "png"]
    assert _length(spelled) > rustwx.COMMAND_LINE_BUDGET

    def refuse(command, *_, **__):
        raise ValueError(explain.layered(
            "this install cannot say which tree is running",
            "the mechanism, shown under --explain"))

    monkeypatch.setattr(provenance_gate, "announce", refuse)
    code = cli.main(["render", "--inputs-from", str(listing), "--series",
                     "--out", str(tmp_path / "png")])
    err = capsys.readouterr().err
    assert code == 2
    assert "this install cannot say which tree is running" in err
    pointer = [line for line in err.splitlines() if "--explain" in line]
    assert len(pointer) == 1
    assert "--inputs-from" in pointer[0] and str(listing) in pointer[0]
    assert len(pointer[0]) < 2_000
    assert not any(str(frame) in err for frame in frames)

    # The fold itself: every frame still reaches render.
    parser = argparse.ArgumentParser()
    render_module.register_cli(parser.add_subparsers(dest="command"))
    args = parser.parse_args(["render", "--inputs-from", str(listing), "--series"])
    assert render_module._fold_render_inputs(args) is None
    assert args.wrfout == frames


@pytest.mark.parametrize("outcome", ["passed", "failed", "did not start"])
def test_the_frame_file_outlives_only_a_stage_that_ran_and_failed(
        tmp_path, monkeypatch, outcome):
    """A failed stage's pointer names the file; nothing else needs it."""

    folder = _long_folder(tmp_path)
    frames = _series(folder, create=True)
    handed = []

    def stage(label, command, **kw):
        command = [str(part) for part in command]
        handed.append(Path(command[command.index("--inputs-from") + 1]))
        if outcome == "failed":
            raise go_cli.GoStageFailed(2, "render: refused")
        if outcome == "did not start":
            raise go_cli.GoStageFailed(go_cli._STAGE_START_FAILED,
                                       "the render stage could not start")

    monkeypatch.setattr(go_cli, "_run_stage", stage)
    if outcome == "passed":
        go_cli.run_render_pass(_plan(folder), frames, explain=False)
    else:
        with pytest.raises(go_cli.GoStageFailed):
            go_cli.run_render_pass(_plan(folder), frames, explain=False)
    assert len(handed) == 1
    listing = handed[0]
    assert listing.exists() == (outcome == "failed")
    if listing.exists():
        record = json.loads(listing.read_text(encoding="utf-8"))
        assert record["wrfout"] == [str(frame) for frame in frames]
        listing.unlink()


def test_a_kept_frame_file_goes_a_week_later_with_the_next_long_stage(
        tmp_path, monkeypatch):
    """A failed long stage's file used to stay in the temporary folder for good.

    One built up per failed long render, from go, the desktop's queue or a
    DA member's pictures.  The next long stage removes those kept past a
    week, and nothing younger, nothing it did not name, and nothing of its
    own that a failure keeps.
    """

    temporary = tmp_path / "temp"
    temporary.mkdir()
    monkeypatch.setattr(go_cli.tempfile, "tempdir", str(temporary))
    week = go_cli.RENDER_INPUTS_KEEP_SECONDS
    now = time.time()

    def planted(name, age):
        path = temporary / name
        path.write_text("{}", encoding="utf-8")
        os.utime(path, (now - age, now - age))
        return path

    stale = planted("gpuwm-render-inputs-old1.json", week + 3600)
    older = planted("gpuwm-render-inputs-old2.json", 30 * 24 * 3600)
    recent = planted("gpuwm-render-inputs-new1.json", week - 3600)
    other = planted("someone-elses-file.json", 30 * 24 * 3600)
    not_ours = planted("gpuwm-render-inputs-old3.txt", 30 * 24 * 3600)
    folder = temporary / "gpuwm-render-inputs-folder.json"
    folder.mkdir()
    os.utime(folder, (now - 30 * 24 * 3600, now - 30 * 24 * 3600))

    frames = _series(_long_folder(tmp_path), create=True)
    handed = []

    def stage(label, command, **kw):
        command = [str(part) for part in command]
        handed.append(Path(command[command.index("--inputs-from") + 1]))
        raise go_cli.GoStageFailed(2, "render: refused")

    monkeypatch.setattr(go_cli, "_run_stage", stage)
    with pytest.raises(go_cli.GoStageFailed):
        go_cli.run_render_pass(_plan(_long_folder(tmp_path)), frames,
                               explain=False)

    assert not stale.exists() and not older.exists()
    assert recent.exists() and other.exists() and not_ours.exists()
    assert folder.is_dir()
    # this stage's own file is in the same folder, kept for its refusal line
    assert handed and handed[0].parent == temporary and handed[0].exists()
    # and a week on, the stage after it removes it
    later = now + week + 60
    assert handed[0] in go_cli.sweep_kept_render_inputs(temporary, now=later)
    assert not handed[0].exists()


def test_a_failed_stage_diagnostic_keeps_its_refusal_beside_a_long_line(capsys):
    """One line of frame paths used to fill the whole diagnostic.

    The desktop and hosted readers show the ``stage_failed`` diagnostic;
    it was the last 8,192 characters of the last eight lines, which one
    long pointer filled, so it began mid-path and never said the refusal.
    """

    refusal = "render: REFUSING: the renderer bridge is from another tree"
    pointer = "  (run woof render " + " ".join(
        f"C:\\forecasts\\run\\wrfout\\wrfout_d02_{k:04d}" for k in range(1_500)
    ) + " --explain for the reason)"
    script = f"import sys; print({refusal!r}, file=sys.stderr); " \
             f"print({pointer!r}, file=sys.stderr); sys.exit(2)"
    observer = _Observer()
    with pytest.raises(go_cli.GoStageFailed) as caught:
        go_cli._run_stage("render", [sys.executable, "-c", script],
                          explain=False, observer=observer,
                          heartbeat_seconds=60.0)
    capsys.readouterr()
    failed = dict(observer.events)["stage_failed"]
    assert failed["diagnostic"] == caught.value.diagnostic
    diagnostic = caught.value.diagnostic
    assert diagnostic.splitlines()[0] == refusal
    assert len(diagnostic) <= 8 * 1_000 + 7
    # the long line keeps its start and its end
    last = diagnostic.splitlines()[-1]
    assert last.startswith("  (run woof render ")
    assert last.endswith(" --explain for the reason)") and " ... " in last


# ---------------------------------------------------------------------------
# A stage that cannot start
# ---------------------------------------------------------------------------


class _Observer:
    def __init__(self):
        self.events = []

    def stage_begin(self, **fields):
        self.events.append(("stage_begin", fields))

    def stage_failed(self, **fields):
        self.events.append(("stage_failed", fields))

    def stage_end(self, **fields):
        self.events.append(("stage_end", fields))


def _start_failure(command, capsys):
    observer = _Observer()
    with pytest.raises(go_cli.GoStageFailed) as caught:
        go_cli._run_stage("render", command, explain=False,
                          observer=observer, heartbeat_seconds=60.0)
    printed = capsys.readouterr()
    assert [name for name, _ in observer.events] == [
        "stage_begin", "stage_failed", "stage_end"]
    failed = observer.events[1][1]
    ended = observer.events[2][1]
    assert failed["exit_code"] == ended["exit_code"] == 127
    assert ended["ok"] is False
    assert caught.value.code == 127
    assert caught.value.diagnostic == failed["diagnostic"]
    assert "FAILED  render (did not start)" in printed.out
    assert "go: stopped at render" in printed.out
    assert "Traceback" not in printed.out + printed.err
    return failed["diagnostic"]


def test_a_command_line_the_system_will_not_start_is_a_failed_stage(capsys):
    # A harmless child with one argument longer than any of the three
    # systems starts a program with: the operating system refuses it
    # before anything runs (WinError 206 on Windows, E2BIG elsewhere).
    command = [sys.executable, "-c", "pass", "x" * 3_000_000]
    reason = _start_failure(command, capsys)
    assert reason.startswith("the render stage could not start: its "
                             "command line is 3,0")
    assert "more than this system will start a program with" in reason


def test_a_program_that_is_not_there_is_a_failed_stage(tmp_path, capsys):
    reason = _start_failure([str(tmp_path / "no-such-program")], capsys)
    assert reason.startswith("the render stage could not start: ")


# ---------------------------------------------------------------------------
# The renderer's own launch
# ---------------------------------------------------------------------------


def test_a_short_renderer_command_is_launched_exactly_as_built(tmp_path):
    frames = [str(frame) for frame in _series(_long_folder(tmp_path))[:3]]
    command = ["rw_wrfbatch", "--store-root", "store", *frames]
    env = {"RUSTWX_BASEMAP_DIR": "maps"}
    assert rustwx.fit_series_command(command, len(frames), env) == (
        command, None, env)


def test_a_long_renderer_command_starts_in_the_frames_folder(tmp_path):
    folder = _long_folder(tmp_path)
    frames = _series(folder, create=True)
    basemaps = tmp_path / "basemaps"
    basemaps.mkdir()
    head = [str(tmp_path / "rw_wrfbatch"), "--store-root",
            os.path.relpath(tmp_path / "store"), "--out-dir",
            os.path.relpath(tmp_path / "png"), "--products", "t2",
            "--frames", "all"]
    command = [*head, *map(str, frames)]
    assert _length(command) > rustwx.COMMAND_LINE_BUDGET
    env = {"RUSTWX_BASEMAP_DIR": os.path.relpath(basemaps), "OTHER": "x"}

    fitted, cwd, moved = rustwx.fit_series_command(command, len(frames), env)

    assert _length(fitted) < WINDOWS_COMMAND_LIMIT
    assert Path(cwd) == folder
    # Every frame is its bare name, and names the same file from there.
    names = fitted[len(head):]
    assert names == [frame.name for frame in frames]
    assert all((Path(cwd) / name) == frame for name, frame in zip(names, frames))
    # Every other path the renderer opens names the same place it did.
    assert Path(fitted[fitted.index("--store-root") + 1]) == tmp_path / "store"
    assert Path(fitted[fitted.index("--out-dir") + 1]) == tmp_path / "png"
    assert Path(moved["RUSTWX_BASEMAP_DIR"]) == basemaps
    assert moved["OTHER"] == "x"
    assert fitted[fitted.index("--products") + 1] == "t2"
