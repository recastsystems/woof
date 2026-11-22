"""The one launcher every desktop package ships, on both of its routes.

``tools/release/launch_arwen.py`` is the file each package carries, so it is
tested here by path and, where a whole start is exercised, by copying it into
a package built in ``tmp_path`` exactly as a cut ships it: it is not a package
module.  A package that carries its own runtime starts on that runtime and
takes no ``--python``; a package that carries none takes ``--python`` or
``ARWEN_PYTHON`` and verifies the engine installed in it.  Both routes share
the folder preferences, the terminal restore, the notices and the
``--verify-launcher`` record.

THE BREAKAGE THIS PREVENTS: the packaged runtime is verified by hash and
must stay byte-identical after use, and one ``doctor`` run from it wrote
554 ``__pycache__`` files (134 MB) into ``runtime/Lib/site-packages``
because the launcher's own ``-B`` covered the launcher process only.  The
flag has to travel in the environment to every engine, terminal and GUI
subprocess, and the same environment carries the CDS credentials path and
the GUI's chosen cache folder.  Through 2.7.3 this file served the bundled
runtime alone, so the Windows desktop and the Linux tarball each shipped a
launcher maintained outside the repository and a repair committed here
reached them a release late, or not at all.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "tools" / "release" / "launch_arwen.py"
# The package this computer can actually start, and the names that package uses.
PLATFORM = "windows-x86_64" if os.name == "nt" else "linux-x86_64"
CONTROLLER = "arwen-tui.exe" if os.name == "nt" else "arwen-tui"
COMPANION = "arwen-companion.exe" if os.name == "nt" else "arwen-companion"
RUNTIME_PYTHON = "runtime/python.exe" if os.name == "nt" else "runtime/bin/python3"
START_SCRIPT = "Start WOOF-ArWen.cmd" if os.name == "nt" else "Start WOOF-ArWen.sh"
# What the two programs load once they are running, as the shipped packages record it.
PAYLOAD = (["arwen-weather.exe", "maplibre-native-c.dll", "vcruntime140.dll"] if os.name == "nt"
           else ["arwen-weather", "libmaplibre-native-c.so"])


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def launcher():
    if not LAUNCHER.is_file():
        pytest.skip("the launcher template needs the source tree")
    return _load(LAUNCHER, "_arwen_launcher")


def _package(tmp_path: Path, name: str, *, bundled: bool, platform: str = PLATFORM,
             engine_version: str = "2.7.4", revision: str = "0" * 40):
    """A package holding this launcher, the way a cut ships it: the file itself,
    the two programs it starts and the identity document of its runtime."""

    if not LAUNCHER.is_file():
        pytest.skip("the launcher template needs the source tree")
    root = tmp_path / name
    root.mkdir(parents=True)
    shutil.copy2(LAUNCHER, root / "launch_arwen.py")
    if platform == "windows-x86_64":
        contents = ["arwen-tui.exe", "arwen-companion.exe", "arwen-weather.exe",
                    "maplibre-native-c.dll", "vcruntime140.dll"]
    else:
        contents = ["arwen-tui", "arwen-companion", "arwen-weather", "libmaplibre-native-c.so"]
    if bundled:
        contents += ["runtime/python.exe" if platform == "windows-x86_64" else "runtime/bin/python3",
                     "runtime/ARWEN-RUNTIME.json"]
    else:
        contents.append("ENGINE-PYTHON.json")
    rows = []
    for relative in contents:
        member = root / relative
        member.parent.mkdir(parents=True, exist_ok=True)
        member.write_bytes(relative.encode("utf-8"))
        rows.append({"path": relative, "sha256": hashlib.sha256(relative.encode("utf-8")).hexdigest()})
    (root / "ARWEN-DESKTOP.json").write_text(json.dumps({
        "schema": "arwen.desktop-package.v1", "status": "READY_FOR_ACCEPTANCE", "platform": platform,
        "desktop_version": "1.0.4", "engine_version": engine_version, "engine_source_revision": revision,
        "launch_files": rows}), encoding="utf-8")
    return root, _load(root / "launch_arwen.py", "_arwen_launcher_" + name)


def _interpreter(tmp_path: Path) -> Path:
    python = tmp_path / "venv" / ("Scripts" if os.name == "nt" else "bin") / ("python.exe" if os.name == "nt" else "python3")
    python.parent.mkdir(parents=True)
    python.write_text("#", encoding="utf-8")
    return python


def _parent_env() -> dict[str, str]:
    return {"PATH": "C:\\Windows\\System32", "PYTHONPATH": "C:\\elsewhere",
            "WOOF_DOCTOR_STATE": "x", "ARWEN_CACHE_DIR": "y",
        "NO_COLOR": "1", "HOME": "C:\\WOOF-ArWen\\profile"}


def test_children_never_write_bytecode_into_the_package(launcher, tmp_path):
    env = launcher.child_environment(tmp_path / "state", tmp_path / "rt" / "python.exe",
                                     parent=_parent_env())
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"
    # The rest of the sealed-runtime posture is unchanged.
    assert env["PYTHONNOUSERSITE"] == "1"
    assert env["PYTHONSAFEPATH"] == "1"
    assert env["PYTHONUTF8"] == "1"
    # Host Python/woof/ArWen settings never leak into the package's children.
    assert "PYTHONPATH" not in env and "WOOF_DOCTOR_STATE" not in env
    # Everything else the caller set is the caller's, the NO_COLOR instruction included.
    assert env["NO_COLOR"] == "1"
    assert env["HOME"] == "C:\\WOOF-ArWen\\profile"
    assert env["PATH"].startswith(str(tmp_path / "rt") + os.pathsep)


def test_a_no_color_instruction_reaches_the_controller(launcher, tmp_path):
    """NO_COLOR is the user's instruction, and the controller implements it.

    The controller drops to a monochrome theme when NO_COLOR names a value, so a
    launcher that removed the variable made a package ignore an instruction its
    own terminal obeys. It travels untouched, empty value included, on both
    package routes.
    """
    env = launcher.child_environment(tmp_path / "state", tmp_path / "python",
                                     parent={"NO_COLOR": "1", "PATH": "/usr/bin"})
    assert env["NO_COLOR"] == "1"
    # An empty value is the terminal library's way of saying colour is wanted after
    # all, so it is neither dropped nor promoted to a value.
    env = launcher.child_environment(tmp_path / "state", tmp_path / "python",
                                     parent={"NO_COLOR": "", "PATH": "/usr/bin"})
    assert env["NO_COLOR"] == ""


def test_the_terminal_the_caller_stands_in_describes_itself(launcher, tmp_path):
    """A caller's own terminal description reaches the controller; a window that gives none gets a default.

    The controller reads TERM for keys and colours and COLORTERM for whether
    24-bit colour is real, so naming both outright told a 256-colour terminal
    it was truecolor.
    """
    env = launcher.child_environment(tmp_path / "state", tmp_path / "python",
                                     parent={"TERM": "screen-256color", "COLORTERM": "256color"})
    assert env["TERM"] == "screen-256color"
    assert env["COLORTERM"] == "256color"
    # A window that describes nothing, which is what the desktop application and the
    # file browser hand their children, still gets a terminal the controller can draw in.
    for parent in ({}, {"TERM": "", "COLORTERM": ""}):
        env = launcher.child_environment(tmp_path / "state", tmp_path / "python", parent=parent)
        assert env["TERM"] == "xterm-256color"
        assert env["COLORTERM"] == "truecolor"


def test_the_cache_folder_follows_the_gui_preference(launcher, tmp_path):
    state = tmp_path / "state"
    assert launcher.cache_directory(state) == state / "cache"
    pref = state / "appdata" / "ArWenCompanion" / "preferences.json"
    pref.parent.mkdir(parents=True)
    chosen = tmp_path / "chosen"
    pref.write_text(json.dumps({"data_folder": str(chosen)}), encoding="utf-8")
    assert launcher.cache_directory(state) == chosen / "cache"
    env = launcher.child_environment(state, tmp_path / "python.exe", parent={})
    assert env["ARWEN_CACHE_DIR"] == str(chosen / "cache")
    # A relative or malformed preference falls back rather than escaping.
    pref.write_text(json.dumps({"data_folder": "relative/folder"}), encoding="utf-8")
    assert launcher.cache_directory(state) == state / "cache"
    pref.write_text("not json", encoding="utf-8")
    assert launcher.cache_directory(state) == state / "cache"


def test_the_forecast_output_folder_follows_the_gui_preference(launcher, tmp_path):
    state = tmp_path / "state"
    assert launcher.output_arguments(state) == ["--output", str(state / "runs")]
    pref = state / "appdata" / "ArWenCompanion" / "preferences.json"
    pref.parent.mkdir(parents=True)
    chosen = tmp_path / "chosen-forecasts"
    pref.write_text(json.dumps({"schema": "arwen.desktop-preferences.v1", "data_folder": str(tmp_path / "data"),
                                "forecast_output_folder": str(chosen)}), encoding="utf-8")
    # The controller opens on the chosen folder and keeps the profile's own runs listed.
    assert launcher.output_arguments(state) == ["--output", str(chosen), "--saved-runs", str(state / "runs")]
    assert launcher.cache_directory(state) == tmp_path / "data" / "cache"
    # The controller log the signal line names moves with the chosen folder.
    command = ["arwen-tui.exe", "--python", "python.exe"] + launcher.output_arguments(state)
    assert launcher.controller_log(command, state) == chosen / ".arwen-tui" / "controller.log"
    # A choice equal to the default names no extra folder.
    pref.write_text(json.dumps({"forecast_output_folder": str(state / "runs")}), encoding="utf-8")
    assert launcher.output_arguments(state) == ["--output", str(state / "runs")]
    # A relative, blank, non-string or absent preference is the default.
    for document in ({"forecast_output_folder": "relative/runs"}, {"forecast_output_folder": "  "},
                     {"forecast_output_folder": 7}, {"data_folder": str(tmp_path / "data")}, [1]):
        pref.write_text(json.dumps(document), encoding="utf-8")
        assert launcher.output_arguments(state) == ["--output", str(state / "runs")]
    pref.write_text("not json", encoding="utf-8")
    assert launcher.output_arguments(state) == ["--output", str(state / "runs")]


def test_every_forecast_folder_chosen_before_stays_named_once_and_never_the_output_folder(launcher, tmp_path):
    state = tmp_path / "state"
    pref = state / "appdata" / "ArWenCompanion" / "preferences.json"
    pref.parent.mkdir(parents=True)
    first, second, third = (tmp_path / "first", tmp_path / "second", tmp_path / "third")
    # After two moves (first, then second, now third) both earlier folders are named, after the profile's own.
    pref.write_text(json.dumps({"schema": "arwen.desktop-preferences.v1", "forecast_output_folder": str(third),
                                "saved_run_folders": [str(state / "runs"), str(first), str(second)]}), encoding="utf-8")
    assert launcher.output_arguments(state) == ["--output", str(third), "--saved-runs", str(state / "runs"),
                                                "--saved-runs", str(first), "--saved-runs", str(second)]
    assert launcher.preference_folders(state, "saved_run_folders") == [state / "runs", first, second]
    # A listed folder equal to the output folder, a duplicate, a relative entry and a non-string are skipped;
    # a trailing separator neither hides a duplicate nor survives into the argument.
    pref.write_text(json.dumps({"forecast_output_folder": str(third),
                                "saved_run_folders": [str(third), str(first), "relative/runs", 7, "  ", str(first) + os.sep, str(second) + os.sep]}),
                    encoding="utf-8")
    assert launcher.output_arguments(state) == ["--output", str(third), "--saved-runs", str(state / "runs"),
                                                "--saved-runs", str(first), "--saved-runs", str(second)]
    # Back on the profile's own folder, the earlier chosen folders are still named and the default only once.
    pref.write_text(json.dumps({"saved_run_folders": [str(first), str(state / "runs")]}), encoding="utf-8")
    assert launcher.output_arguments(state) == ["--output", str(state / "runs"), "--saved-runs", str(first)]
    # A list that is not a list, or is absent, names only the profile's own folder.
    for document in ({"forecast_output_folder": str(third), "saved_run_folders": "not a list"},
                     {"forecast_output_folder": str(third), "saved_run_folders": {"kept": True}}):
        pref.write_text(json.dumps(document), encoding="utf-8")
        assert launcher.output_arguments(state) == ["--output", str(third), "--saved-runs", str(state / "runs")]
        assert launcher.preference_folders(state, "saved_run_folders") == []


def test_a_saved_forecast_folder_that_cannot_be_created_falls_back_and_the_notice_names_it_and_the_way_out(launcher, tmp_path):
    state = tmp_path / "state"
    pref = state / "appdata" / "ArWenCompanion" / "preferences.json"
    pref.parent.mkdir(parents=True)
    blocker = tmp_path / "a-file-not-a-folder"
    blocker.write_text("", encoding="utf-8")
    unavailable = blocker / "forecasts"
    pref.write_text(json.dumps({"forecast_output_folder": str(unavailable), "saved_run_folders": [str(tmp_path / "first")]}),
                    encoding="utf-8")
    # THE BREAKAGE THIS PREVENTS: the controller given --output on a folder it cannot create exits before
    # the GUI opens, and Settings, the only door that changes the folder, is unreachable.
    folder, notice = launcher.output_folder(state)
    assert folder == state / "runs"
    assert notice.startswith("The saved forecast output folder " + str(unavailable) + " is unavailable: ")
    assert notice.endswith("New forecasts go to " + str(state / "runs")
                           + " until a folder that can be created is chosen in Settings, Forecast output folder.")
    assert launcher.output_arguments(state) == ["--output", str(state / "runs"), "--saved-runs", str(tmp_path / "first")]
    assert not unavailable.exists()
    # A chosen folder that can be created is created before the controller starts, with no notice.
    chosen = tmp_path / "chosen" / "forecasts"
    pref.write_text(json.dumps({"forecast_output_folder": str(chosen)}), encoding="utf-8")
    assert launcher.output_folder(state) == (chosen, None)
    assert chosen.is_dir()
    assert launcher.output_folder(tmp_path / "no-preferences") == (tmp_path / "no-preferences" / "runs", None)


def test_a_saved_folder_holding_a_wildcard_or_a_null_is_no_preference_at_all(launcher, tmp_path):
    """THE BREAKAGE THIS PREVENTS: ``Path.is_absolute`` accepts both, and the launcher then
    hands the controller a folder Windows cannot create, or raises ``ValueError`` out of
    ``mkdir`` before the GUI, and its Settings, can open to change the folder."""

    state = tmp_path / "state"
    pref = state / "appdata" / "ArWenCompanion" / "preferences.json"
    pref.parent.mkdir(parents=True)
    for name in (str(tmp_path / "wild?card"), str(tmp_path / "star*"), str(tmp_path / "null") + "\x00y",
                 str(tmp_path / "pipe|d"), str(tmp_path / "bell") + "\x07"):
        pref.write_text(json.dumps({"forecast_output_folder": name, "data_folder": name}), encoding="utf-8")
        assert launcher.preference_folder(state, "forecast_output_folder") is None, name
        assert launcher.output_folder(state) == (state / "runs", None), name
        assert launcher.output_arguments(state) == ["--output", str(state / "runs")], name
        assert launcher.cache_directory(state) == state / "cache", name
    # One refused entry never discards the entries after it.
    first, second = tmp_path / "first", tmp_path / "second"
    pref.write_text(json.dumps({"saved_run_folders": [str(first), str(tmp_path / "bad|entry"), str(second)]}), encoding="utf-8")
    assert launcher.preference_folders(state, "saved_run_folders") == [first, second]
    assert launcher.absolute_folder(str(first)) == first
    assert launcher.absolute_folder("relative/runs") is None
    assert launcher.absolute_folder(7) is None


def test_cds_credentials_reach_the_children_or_refuse_by_name(launcher, tmp_path):
    rc = tmp_path / "cdsapirc"
    rc.write_text("url: https://example.invalid\nkey: 0\n", encoding="utf-8")
    env = launcher.child_environment(tmp_path / "state", tmp_path / "python.exe", rc, parent={})
    assert env["CDSAPI_RC"] == str(rc)
    env = launcher.child_environment(tmp_path / "state", tmp_path / "python.exe", parent={})
    assert "CDSAPI_RC" not in env
    with pytest.raises(ValueError, match="CDS credentials file is unavailable"):
        launcher.child_environment(tmp_path / "state", tmp_path / "python.exe",
                                   tmp_path / "missing", parent={})


def test_the_launcher_parses_the_packaging_flags(launcher):
    """The flags the shipped package added by hand are in the template."""

    source = LAUNCHER.read_text(encoding="utf-8")
    for flag in ("--verify-launcher", "--tui-only", "--state-dir", "--cds-credentials"):
        assert f"'{flag}'" in source, flag
    assert "'cds_credentials_configured'" in source
    assert "'cache_directory'" in source


def test_the_restore_bytes_are_the_controllers_own_sequence(launcher):
    """SGR/urxvt/any-event/button/normal mouse off, paste off, main screen, cursor."""

    assert launcher.TERMINAL_RESTORE == (
        "\x1b[?1006l\x1b[?1015l\x1b[?1003l\x1b[?1002l\x1b[?1000l"
        "\x1b[?2004l\x1b[?1049l\x1b[?25h")
    # Once the controller lane carrying TERMINAL_RESTORE is in, the two
    # sequences have to stay the same one: a launcher that disabled a
    # different set would leave exactly the modes it missed turned on.
    controller = REPO_ROOT / "tools" / "arwen-tui" / "src" / "main.rs"
    source = controller.read_text(encoding="utf-8") if controller.is_file() else ""
    match = re.search(r"const TERMINAL_RESTORE: &\[u8\] =\s*b\"([^\"]*)\";", source)
    if match is None:
        pytest.skip("this tree's controller does not define TERMINAL_RESTORE yet")
    assert match.group(1).replace("\\x1b", "\x1b") == launcher.TERMINAL_RESTORE


def test_a_redirected_launcher_leaves_the_stream_alone(launcher, tmp_path, monkeypatch):
    """THE BREAKAGE THIS PREVENTS: escape bytes in a log file or a pipeline."""

    sink = tmp_path / "captured.txt"
    with sink.open("w", encoding="utf-8") as stream:
        monkeypatch.setattr(sys, "stdout", stream)
        assert launcher.terminal_mode() is None
        launcher.restore_terminal(None)
    assert sink.read_text(encoding="utf-8") == ""


@pytest.mark.skipif(os.name == "nt", reason="the restore is POSIX-only by design")
def test_a_killed_controller_hands_back_a_usable_terminal(launcher, monkeypatch):
    import termios
    import tty

    controller, terminal = os.openpty()
    try:
        with os.fdopen(terminal, "w", closefd=False) as stream:
            monkeypatch.setattr(sys, "stdout", stream)
            before = termios.tcgetattr(terminal)
            saved = launcher.terminal_mode()
            assert saved is not None
            # What the controller does, and what a SIGKILL leaves behind.
            tty.setraw(terminal)
            stream.write("\x1b[?1049h\x1b[?1003h\x1b[?1006h\x1b[?25l")
            stream.flush()
            os.read(controller, 65536)
            launcher.restore_terminal(saved)
        assert os.read(controller, 65536).decode() == launcher.TERMINAL_RESTORE
        assert termios.tcgetattr(terminal) == before
    finally:
        os.close(controller)
        os.close(terminal)


def test_the_signal_line_names_the_signal_and_the_controller_log(launcher, tmp_path):
    state = tmp_path / "state"
    runs = state / "runs"
    command = ["/opt/arwen/arwen-tui", "--python", "/venv/bin/python", "--output", str(runs)]
    line = launcher.signal_report(-9, command, state)
    assert "\n" not in line
    assert "arwen-tui" in line
    # The line is only ever printed on the platform that has the signal.
    if os.name != "nt":
        assert "SIGKILL" in line
    assert str(runs / ".arwen-tui" / "controller.log") in line
    # A forwarded --output moves the log with it; an unknown signal still reports.
    chosen = tmp_path / "elsewhere"
    command[-1] = str(chosen)
    assert str(chosen / ".arwen-tui" / "controller.log") in launcher.signal_report(-9, command, state)
    assert "signal 99" in launcher.signal_report(-99, command, state)
    assert launcher.controller_log(["/opt/arwen/arwen-tui"], state) == runs / ".arwen-tui" / "controller.log"


def test_a_package_without_a_bundled_runtime_starts_on_the_python_it_is_given(tmp_path, monkeypatch, capsys):
    """THE BREAKAGE THIS PREVENTS: through 2.7.3 this file refused a forwarded ``--python``
    outright, so the Windows desktop and the Linux tarball each shipped a launcher built
    outside the repository and a repair committed here never reached them."""

    root, launcher = _package(tmp_path, "installed", bundled=False)
    python = _interpreter(tmp_path)
    state = tmp_path / "profile"
    proof = {"engine_version": "2.7.4", "engine_source_revision": "0" * 40, "python": str(python),
             "python_version": "3.11.9", "python_files_verified": 812, "installed_tui_sha256": "ab" * 32}
    seen = {}

    def probe(selected, identity, env):
        seen.update(python=selected, identity=identity, cache=env["ARWEN_CACHE_DIR"])
        return proof

    monkeypatch.setattr(launcher, "installed_engine", probe)
    assert launcher.launch(["--python", str(python), "--state-dir", str(state), "--verify-launcher"]) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["status"] == "PASS"
    assert record["runtime_source"] == "installed"
    assert record["runtime"] == proof
    assert record["python"] == str(python)
    assert record["command"][:5] == [str(root / CONTROLLER), "--python", str(python),
                                     "--companion", str(root / COMPANION)]
    assert record["command"][5] == "--open-companion"
    assert record["output_directory"] == str(state / "runs")
    assert record["checked_components"] == 3 + len(PAYLOAD)
    # The engine is verified in the Python that was chosen, against this package's document.
    assert seen["python"] == python and seen["identity"] == root / "ENGINE-PYTHON.json"
    assert seen["cache"] == str(state / "cache")


def test_the_chosen_folders_and_the_terminal_restore_serve_the_unbundled_package_too(tmp_path, monkeypatch, capsys):
    """One code path: the folder preferences that only the bundled variant honored are read
    on the route the desktop packages actually take."""

    root, launcher = _package(tmp_path, "folders", bundled=False)
    python = _interpreter(tmp_path)
    state = tmp_path / "profile"
    chosen, earlier = tmp_path / "forecasts", tmp_path / "earlier"
    preference = state / "appdata" / "ArWenCompanion" / "preferences.json"
    preference.parent.mkdir(parents=True)
    preference.write_text(json.dumps({"forecast_output_folder": str(chosen),
                                      "saved_run_folders": [str(earlier)],
                                      "data_folder": str(tmp_path / "data")}), encoding="utf-8")
    monkeypatch.setattr(launcher, "installed_engine",
                        lambda *arguments: {"engine_source_revision": "0" * 40, "python": str(python)})
    assert launcher.launch(["--python", str(python), "--state-dir", str(state), "--verify-launcher"]) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["command"][-6:] == ["--output", str(chosen), "--saved-runs", str(state / "runs"),
                                      "--saved-runs", str(earlier)]
    assert record["cache_directory"] == str(tmp_path / "data" / "cache")
    assert record["output_directory"] == str(chosen)
    # The terminal restore and the signal line are in this one file, for both routes.
    assert launcher.TERMINAL_RESTORE and callable(launcher.restore_terminal) and callable(launcher.signal_report)


def test_a_bundled_package_refuses_a_forwarded_python_by_naming_its_own_runtime(tmp_path):
    """A refusal names what it prevents and the way out: this package's own Python,
    and the script that starts it."""

    root, launcher = _package(tmp_path, "bundled_refusal", bundled=True)
    with pytest.raises(ValueError) as refusal:
        launcher.launch(["--python", str(tmp_path / "elsewhere" / "python.exe"), "--verify-launcher"])
    assert str(root / RUNTIME_PYTHON) in str(refusal.value)
    # The way out is the same command without the flag. The start scripts forward every
    # argument they are given, so naming one of them here would name the command the
    # caller just ran.
    assert "again without --python" in str(refusal.value)
    assert START_SCRIPT not in str(refusal.value)


def test_a_bundled_package_selects_the_runtime_it_carries(tmp_path, monkeypatch, capsys):
    root, launcher = _package(tmp_path, "bundled", bundled=True)
    state = tmp_path / "profile"
    monkeypatch.setattr(launcher, "bundled_engine",
                        lambda *arguments: {"engine_version": "2.7.4", "engine_source_revision": "0" * 40},
                        raising=False)
    assert launcher.launch(["--state-dir", str(state), "--tui-only", "--verify-launcher"]) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["status"] == "PASS"
    assert record["runtime_source"] == "bundled"
    assert record["python"] == str(root / RUNTIME_PYTHON)
    assert record["command"][:5] == [str(root / CONTROLLER), "--python", str(root / RUNTIME_PYTHON),
                                     "--companion", str(root / COMPANION)]
    assert "--open-companion" not in record["command"]
    assert record["checked_components"] == 4 + len(PAYLOAD)


def test_only_the_bundled_runtime_starts_a_bundled_package(tmp_path):
    root, launcher = _package(tmp_path, "bundled_interpreter", bundled=True)
    row = launcher.PACKAGE_PLATFORMS[PLATFORM]
    with pytest.raises(ValueError, match=re.escape(START_SCRIPT)):
        launcher.bundled_engine(root, root / "runtime" / "another-python", {}, row)


def test_a_package_built_for_another_computer_is_refused_by_name(tmp_path):
    other = "linux-x86_64" if os.name == "nt" else "windows-x86_64"
    root, launcher = _package(tmp_path, "other_platform", bundled=False, platform=other)
    with pytest.raises(ValueError, match="this computer cannot run"):
        launcher.launch(["--verify-launcher"])
    root, launcher = _package(tmp_path, "unknown_platform", bundled=False, platform="solaris-sparc")
    with pytest.raises(ValueError, match="solaris-sparc"):
        launcher.launch(["--verify-launcher"])


def test_a_manifest_that_does_not_record_what_the_launcher_starts_is_refused_by_name(tmp_path):
    root, launcher = _package(tmp_path, "incomplete", bundled=False)
    manifest = json.loads((root / "ARWEN-DESKTOP.json").read_text(encoding="utf-8"))
    manifest["launch_files"] = [row for row in manifest["launch_files"] if row["path"] != COMPANION]
    (root / "ARWEN-DESKTOP.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match=re.escape(COMPANION)):
        launcher.launch(["--verify-launcher"])


def _started(launcher, monkeypatch):
    """Start the controller with the process calls replaced, and report what they were given."""

    seen = {}
    restored = []

    class _Ended:
        returncode = 0

    def run(command, **keywords):
        seen.update(command=command, keywords=keywords)
        return _Ended()

    monkeypatch.setattr(launcher.subprocess, "run", run)
    monkeypatch.setattr(launcher, "terminal_mode", lambda: "the mode before the controller")
    monkeypatch.setattr(launcher, "restore_terminal", restored.append)
    return seen, restored


def test_the_controller_keeps_the_directory_the_caller_stands_in(tmp_path, monkeypatch):
    """THE BREAKAGE THIS PREVENTS: the controller reads its own working directory once and
    then starts every engine job, every node operation and the file browser there, and
    resolves a relative configuration path a caller forwards against it. Naming a directory
    here moves all of those, and it would put the two doors of one package into
    disagreement: the desktop bootstrap starts the same controller in the package folder."""

    for name, bundled in (("start_installed", False), ("start_bundled", True)):
        root, launcher = _package(tmp_path, name, bundled=bundled)
        python = _interpreter(tmp_path / (name + "-environment"))
        state = tmp_path / (name + "-profile")
        monkeypatch.setattr(launcher, "installed_engine",
                            lambda *arguments: {"engine_source_revision": "0" * 40}, raising=False)
        monkeypatch.setattr(launcher, "bundled_engine",
                            lambda *arguments: {"engine_source_revision": "0" * 40}, raising=False)
        seen, restored = _started(launcher, monkeypatch)
        arguments = ["--state-dir", str(state)] + ([] if bundled else ["--python", str(python)])
        assert launcher.launch(arguments) == 0
        assert "cwd" not in seen["keywords"]
        assert seen["command"][0] == str(root / CONTROLLER)
        assert seen["keywords"]["env"]["ARWEN_COMPANION"] == str(root / COMPANION)
        # The terminal is put back after the controller ends, on both routes.
        assert restored == ["the mode before the controller"]


def test_a_manifest_that_does_not_record_the_payload_the_programs_load_is_refused_by_name(tmp_path):
    """THE BREAKAGE THIS PREVENTS: a row a package does not record is a file nothing
    verifies, so a package assembled without the weather program or the map library
    reaches the interface and fails there, on the first map or the first forecast."""

    missing = PAYLOAD[-1]
    root, launcher = _package(tmp_path, "no_payload", bundled=False)
    manifest = json.loads((root / "ARWEN-DESKTOP.json").read_text(encoding="utf-8"))
    manifest["launch_files"] = [row for row in manifest["launch_files"] if row["path"] != missing]
    (root / "ARWEN-DESKTOP.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match=re.escape(missing)):
        launcher.launch(["--verify-launcher"])
    # What the launcher requires is a row of the platform table, not a branch.
    assert set(PAYLOAD) <= launcher.required_components(launcher.PACKAGE_PLATFORMS[PLATFORM], False)
    assert set(PAYLOAD) <= launcher.required_components(launcher.PACKAGE_PLATFORMS[PLATFORM], True)


def test_a_package_missing_its_identity_document_and_one_not_marked_complete_each_name_a_way_out(tmp_path):
    """A refusal names the breakage and the way out. An unreadable identity document and a
    package that never finished assembly are two different situations with two answers."""

    root, launcher = _package(tmp_path, "no_identity", bundled=False)
    (root / "ARWEN-DESKTOP.json").unlink()
    with pytest.raises(ValueError) as refusal:
        launcher.launch(["--verify-launcher"])
    assert "ARWEN-DESKTOP.json" in str(refusal.value)
    assert "Extract the complete WOOF-ArWen desktop package" in str(refusal.value)
    root, launcher = _package(tmp_path, "half_built", bundled=False)
    manifest = json.loads((root / "ARWEN-DESKTOP.json").read_text(encoding="utf-8"))
    manifest["status"] = "IN_PROGRESS"
    (root / "ARWEN-DESKTOP.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Use a released package"):
        launcher.launch(["--verify-launcher"])


def test_a_forwarded_companion_flag_is_refused_with_the_start_that_works(tmp_path):
    """A refusal names the breakage and the way out, and this one named only the breakage.

    The launcher starts the terminal and the application a package carries, so an
    interface flag naming either one is not passed through; the sentence now says
    what to start instead.
    """

    root, launcher = _package(tmp_path, "companion_flag", bundled=False)
    with pytest.raises(ValueError) as refusal:
        launcher.launch(["--verify-launcher", "--open-companion"])
    assert "included in this package" in str(refusal.value)
    assert "without them" in str(refusal.value) and "--tui-only" in str(refusal.value)


def test_a_package_is_refused_on_a_computer_that_cannot_run_its_programs(tmp_path, monkeypatch):
    """THE BREAKAGE THIS PREVENTS: the programs of one package run on one family of
    computer, and a package accepted on another one fails later, on exec, with the
    operating system's own message instead of a sentence."""

    root, launcher = _package(tmp_path, "elsewhere_platform", bundled=False, platform="linux-x86_64")
    monkeypatch.setattr(launcher.sys, "platform", "darwin")
    with pytest.raises(ValueError, match="this computer cannot run"):
        launcher.launch(["--verify-launcher"])
    # The family a package runs on is a field of its row, not a branch in the code.
    assert launcher.PACKAGE_PLATFORMS["linux-x86_64"]["system"] == "linux"
    assert launcher.PACKAGE_PLATFORMS["windows-x86_64"]["system"] == "win32"


def test_every_start_script_says_what_to_do_when_no_python_is_installed(tmp_path):
    """THE BREAKAGE THIS PREVENTS: a package with no runtime of its own, started on a
    computer with no Python, failed with the shell's own message about a missing
    command and named neither WOOF-ArWen nor the way out."""

    for name in ("Start WOOF-ArWen.cmd", "Start WOOF-ArWen Terminal.cmd", "Start WOOF-ArWen.sh", "Start WOOF-ArWen Terminal.sh"):
        script = REPO_ROOT / "tools" / "release" / name
        text = script.read_text(encoding="utf-8")
        assert "Cannot start WOOF-ArWen" in text
        assert "ARWEN_PYTHON" in text
        assert "Install Python 3.11 or newer" in text


def test_the_help_text_names_the_version_this_package_carries(tmp_path, capsys):
    """A version written into this file would be a release behind on the day it ships."""

    root, launcher = _package(tmp_path, "help_text", bundled=False, engine_version="9.9.9")
    with pytest.raises(SystemExit) as exit_code:
        launcher.launch(["--help"])
    assert exit_code.value.code == 0
    assert "woof 9.9.9" in capsys.readouterr().out
    # No release version is written into a string the launcher prints or compares.
    import ast

    tree = ast.parse(LAUNCHER.read_text(encoding="utf-8"))
    holders = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    documented = {id(ast.get_docstring(node, clean=False)) for node in ast.walk(tree)
                  if isinstance(node, holders)}
    spoken = [node.value for node in ast.walk(tree)
              if isinstance(node, ast.Constant) and isinstance(node.value, str)
              and id(node.value) not in documented]
    # The module docstring is prose, except its first line, which argparse prints as the
    # description: a version written there would reach --help unread by the rest of this check.
    spoken.append((ast.get_docstring(tree, clean=False) or "").splitlines()[0])
    assert [text for text in spoken if re.search(r"\b\d+\.\d+\.\d+\b", text)] == []
