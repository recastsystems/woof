"""A build error must never wear a data error's face.

Measured 2026-08-20 in a fresh worktree whose Rust bridge had never been
built.  ``woof check --alloc CONFIG`` reaches the ERA5 GRIB1 route,
which builds ``tools/grib1_bridge`` with cargo on first use.  With the
crate's ``gpuwm_preprocess_cpu.dll`` held open by another process --
another lane's run, an editor, a previous woof still exiting -- cargo
cannot relink and fails::

    error: failed to remove file `...\\target\\release\\gpuwm_preprocess_cpu.dll`
    Caused by:
      The process cannot access the file because it is being used by
      another process. (os error 32)

Three things were wrong with what the reader then saw.

1. The sentence was ``could not decode/merge forcing inputs:`` followed
   by ten lines of unrelated Rust compiler WARNINGS, with the one line
   that names the cause last.  Nothing said "this is a build, not your
   data".
2. Five more failures followed, all of them false -- ``forcing
   inventory is missing [22 fields]``, ``forcing has no pressure
   levels`` -- measured against an EMPTY catalog because the decoder
   never ran.  A reader hunting a missing variable is hunting nothing.
3. In ``--json`` mode the whole report goes to stderr and stdout carries
   the memory document, which is written only if the input preflight
   returned zero.  So a caller running ``woof check --alloc --json``
   got an EMPTY stdout and a ``json.loads`` exception: a build error
   reaching a program as a corrupt reply.

Each of the three is guarded below.
"""

from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

#: Cargo's REAL output for the locked-DLL failure, captured verbatim
#: from the reproduction (Windows, RTX 3080 box, 2026-08-20).  Kept
#: whole -- warning wall included -- because the point of the classifier
#: is that it finds the one line that matters inside exactly this.
LOCKED_DLL_OUTPUT = r"""warning: value assigned to `raw_data` is never read
   --> vendor\grib-core\src\grib2\parser.rs:571:33
    |
571 |     let mut raw_data: Vec<u8> = Vec::new();
    |                                 ^^^^^^^^^^
    |
    = help: maybe it is overwritten before being read?
    = note: `#[warn(unused_assignments)]` (part of `#[warn(unused)]`) on by default

warning: `grib-core` (lib) generated 1 warning
   Compiling grib1_bridge v0.1.0 (C:\woof\tools\grib1_bridge)
error: failed to remove file `C:\woof\tools\grib1_bridge\target\release\gpuwm_preprocess_cpu.dll`

Caused by:
  The process cannot access the file because it is being used by another process. (os error 32)
"""


def test_the_locked_artifact_failure_names_itself_and_not_the_warnings():
    from woof import bridges

    message = bridges.cargo_build_refusal(
        "grib1_bridge", "tools/grib1_bridge",
        returncode=101, output=LOCKED_DLL_OUTPUT)
    lowered = message.lower()
    # The CLASS, in words a reader can act on.
    assert "another process" in lowered
    assert "build" in lowered
    # The artifact that could not be replaced.
    assert "gpuwm_preprocess_cpu.dll" in message
    # A remedy, and one that exists.
    assert "remedy:" in lowered
    # And NOT the wall: the unrelated compiler warning must not be what
    # the reader has to read past to reach the cause.
    assert "raw_data" not in message
    assert "unused_assignments" not in message


def test_a_held_build_lock_is_a_different_class_than_a_held_artifact():
    from woof import bridges

    message = bridges.cargo_build_refusal(
        "grib1_bridge", "tools/grib1_bridge", returncode=101,
        output="Blocking waiting for file lock on build directory\n"
               "error: build failed")
    assert "lock" in message.lower()
    assert "remedy:" in message.lower()


def test_an_unclassified_cargo_failure_still_says_it_was_a_build():
    """No needle matched is not permission to relay a raw wall."""

    from woof import bridges

    message = bridges.cargo_build_refusal(
        "grib1_bridge", "tools/grib1_bridge", returncode=101,
        output="warning: something\nerror: could not compile `grib-core`")
    lowered = message.lower()
    assert "build" in lowered and "cargo" in lowered
    assert "could not compile `grib-core`" in message
    assert "remedy:" in lowered


def test_the_grib1_bridge_route_raises_the_named_refusal(monkeypatch):
    """The real call site, with cargo's real failure under it."""

    from woof import bridges
    from woof.ingest import grib

    def fake_run(command, **kwargs):
        # The RESOLVED toolchain, so this holds on a machine whose cargo
        # is only in rustup's own home: the route runs an absolute
        # ~/.cargo/bin/cargo there, and asserting the bare word would
        # have refused the fix that made ten fixture errors go away.
        assert Path(command[0]).stem == "cargo"
        return types.SimpleNamespace(
            returncode=101, stdout="", stderr=LOCKED_DLL_OUTPUT)

    monkeypatch.delenv(bridges.BRIDGE_ENV["grib1_bridge"], raising=False)
    monkeypatch.setattr(grib.subprocess, "run", fake_run)
    with pytest.raises(bridges.BridgeBuildError) as excinfo:
        grib.build_rust_bridge()
    message = str(excinfo.value)
    assert "another process" in message.lower()
    assert "raw_data" not in message


def test_a_rustup_install_off_PATH_is_still_found(monkeypatch, tmp_path):
    """THE MEASURED DEFECT: rustup works, PATH does not know it.

    A non-login shell -- ``ssh host 'pytest ...'``, cron, systemd, a
    desktop-launched process -- never runs rustup's profile edit, so
    ``cargo`` is absent from PATH on a machine whose ``~/.cargo/bin/cargo``
    answers ``1.93.1``.  Every bridge build then refused with "no Rust
    toolchain is on PATH" and told its owner to install what was already
    installed; ten tests in tests/test_domain_wizard_forcing.py errored in
    their module fixture on exactly that, measured on test host 1 2026-09-17.

    The ladder is asserted in order, because the order is the contract: an
    explicit choice, then the shell's, then rustup's own home.
    """

    from woof import bridges

    name = "cargo.exe" if bridges.os.name == "nt" else "cargo"
    home = tmp_path / "home"
    (home / ".cargo" / "bin").mkdir(parents=True)
    shim = home / ".cargo" / "bin" / name
    shim.write_text("#!/bin/sh\n", encoding="utf-8")

    monkeypatch.delenv("CARGO", raising=False)
    monkeypatch.delenv("CARGO_HOME", raising=False)
    monkeypatch.setattr(bridges.shutil, "which", lambda *_a, **_k: None)
    monkeypatch.setattr(bridges.Path, "home", classmethod(lambda _cls: home))
    assert bridges.cargo_executable() == str(shim)
    assert bridges.cargo_is_installed(), (
        "a reachable rustup toolchain is reported as no toolchain, which is "
        "the refusal that told a working machine to install Rust")

    # CARGO_HOME moves rustup's home, and the resolver follows it.
    moved = tmp_path / "elsewhere"
    (moved / "bin").mkdir(parents=True)
    (moved / "bin" / name).write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setenv("CARGO_HOME", str(moved))
    assert bridges.cargo_executable() == str(moved / "bin" / name)

    # An explicit CARGO outranks both, so a lane can still choose.
    chosen = tmp_path / "chosen-cargo"
    chosen.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setenv("CARGO", str(chosen))
    assert bridges.cargo_executable() == str(chosen)

    # And a machine with none of the three still answers "none", so the
    # refusal this file is about keeps firing where it should.
    monkeypatch.delenv("CARGO", raising=False)
    monkeypatch.delenv("CARGO_HOME", raising=False)
    monkeypatch.setattr(bridges.Path, "home",
                        classmethod(lambda _cls: tmp_path / "empty"))
    assert bridges.cargo_executable() is None
    assert not bridges.cargo_is_installed()


def test_no_cargo_on_path_is_a_refusal_and_not_an_oserror(monkeypatch):
    """``cargo`` absent used to escape as a bare WinError 2 traceback."""

    from woof import bridges
    from woof.ingest import grib

    def fake_run(command, **kwargs):
        raise FileNotFoundError(2, "The system cannot find the file specified")

    monkeypatch.delenv(bridges.BRIDGE_ENV["grib1_bridge"], raising=False)
    monkeypatch.setattr(grib.subprocess, "run", fake_run)
    with pytest.raises(bridges.BridgeBuildError) as excinfo:
        grib.build_rust_bridge()
    lowered = str(excinfo.value).lower()
    assert "cargo" in lowered
    assert "remedy:" in lowered


def test_the_grib2_tool_route_raises_the_same_named_refusal(monkeypatch):
    """The second cargo call site shares the classifier, not a copy."""

    from woof import bridges, mapped_source

    monkeypatch.setattr(bridges, "find_bridge", lambda name: None)
    monkeypatch.setattr(
        mapped_source.subprocess, "run",
        lambda command, **kwargs: types.SimpleNamespace(
            returncode=101, stdout="", stderr=LOCKED_DLL_OUTPUT))
    if not (mapped_source._grib2_tools_crate() /  # noqa: SLF001
            "Cargo.toml").is_file():
        pytest.skip("no checkout crate here, so cargo is never reached")
    with pytest.raises(bridges.BridgeBuildError) as excinfo:
        mapped_source._build_grib2_tools()  # noqa: SLF001
    assert "another process" in str(excinfo.value).lower()


# --------------------------------------------------------------------------
# The cascade, and the malformed JSON
# --------------------------------------------------------------------------

def test_a_decoder_that_could_not_be_built_is_not_reported_as_missing_data():
    """The report says which failure is the cause and which are echoes."""

    from woof.ingest.preflight import (PreflightIssue, PreflightReport,
                                        _empty_catalog)

    report = PreflightReport(
        _empty_catalog("ERA5"),
        (PreflightIssue("decoder-build",
                        "the GRIB1 decoder could not be built here: "
                        "another process is holding it"),
         PreflightIssue("inventory", "forcing inventory is missing ['T']"),
         PreflightIssue("levels", "forcing has no pressure levels")),
        ("resolved input SHA-256 catalog",))
    text = report.format()
    assert "decoder-build" in text
    # The naming, so nobody hunts a variable that was never looked for.
    assert "measured no data" in text or "never ran" in text


#: The smallest experiment TOML that reaches the input preflight.
#:
#: WHY THIS IS BUILT HERE AND NOT READ OUT OF configs/.
#: These two gates are about ONE thing: that ``woof check --json`` always
#: puts a parseable document on stdout when the preflight cannot produce a
#: report.  They said so, and then pointed at a repository config that
#: declares staged CDS ERA5 GRIB files and a reference bundle under
#: ``${GPUWM_CASE_DATA_ROOT}``.  On a machine without that data the loader
#: refuses BEFORE ``preflight_report`` is reached, so the monkeypatched
#: cause never entered the document and both gates failed while asserting
#: about a branch they had not run.  Measured on two machines at
#: 674133103 and at this branch's tip: the Linux node answered "does not
#: exist" for the declared inputs, and the Windows cut box answered
#: "version identity is ambiguous" from the same loader.  A gate about a
#: channel must not be gated on somebody's staged dataset.
#:
#: ``_check_command`` needs exactly two things from the file: that the
#: config authority can read it, and that it declares ``[case_data]`` so
#: neither of the two early returns (legacy RunConfig shape, prepared
#: route) is taken.  Nothing below that line is read by these gates.
_CHANNEL_CONFIG = """
[experiment]
name = "json_channel_gate"
start_time = 2020-01-01T00:00:00
run_seconds = 3600.0

[case_data]
forcing = ["forcing.grib"]
forcing_interval_s = 21600.0
"""


@pytest.fixture
def channel_config(tmp_path, monkeypatch):
    """A config that reaches the preflight on every machine.

    The loader is stubbed, and that is the point rather than a shortcut.
    ``load_experiment_case`` answers "what experiment is this, and are its
    declared inputs on this disk"; it has its own coverage, and its answer
    is upstream of everything these two gates measure.  What they measure
    is what ``_check_command`` writes to stdout once the preflight is
    reached -- so the fixture's job is to reach it, deterministically,
    with no dataset and no installed distribution involved.

    ``preflight_report`` is what each gate then replaces with its own
    failure, which is the cause whose journey into the document is the
    thing under test.
    """
    path = tmp_path / "json_channel_gate.toml"
    path.write_text(_CHANNEL_CONFIG, encoding="utf-8", newline="\n")

    from woof import case_data

    monkeypatch.setattr(case_data, "load_experiment_case",
                        lambda *_a, **_k: (object(), object()))
    return path


def test_check_json_emits_a_document_when_the_decoder_cannot_be_built(
        capsys, monkeypatch, channel_config):
    """stdout must ALWAYS parse: a build error is not a corrupt reply.

    THE #241 defect.  ``woof check --alloc --json`` printed the report
    to stderr, returned 1 before the memory estimator wrote anything,
    and left stdout EMPTY -- so a caller doing ``json.loads(stdout)``
    saw a JSON parse error where a build failure had happened.
    """

    from woof.ingest import preflight

    def explode(*_args, **_kwargs):
        raise RuntimeError(
            "the Rust bridge `grib1_bridge` is not built here and building "
            "it FAILED: another process has "
            "target/release/gpuwm_preprocess_cpu.dll open")

    monkeypatch.setattr(preflight, "preflight_report", explode)
    args = types.SimpleNamespace(config=channel_config, json=True)
    code = preflight._check_command(args)  # noqa: SLF001
    captured = capsys.readouterr()
    assert code != 0
    document = json.loads(captured.out)
    assert document["ok"] is False
    assert "gpuwm_preprocess_cpu.dll" in json.dumps(document)
    # The cause reached the document as a refusal and not as invented
    # data failures, which is finding 2 of the three in this module's
    # docstring: an empty catalog used to produce five false ones.
    assert document["refusal_type"] == "RuntimeError"
    assert document["failures"] == []


def test_check_json_emits_a_document_for_an_ordinary_preflight_failure(
        capsys, monkeypatch, channel_config):
    """The guarantee is about the CHANNEL, so it cannot be class-bound."""

    from woof.ingest.preflight import (PreflightIssue, PreflightReport,
                                        _empty_catalog)
    from woof.ingest import preflight

    monkeypatch.setattr(
        preflight, "preflight_report",
        lambda *_a, **_k: PreflightReport(
            _empty_catalog("ERA5"),
            (PreflightIssue("levels", "forcing has no pressure levels"),),
            ("resolved input SHA-256 catalog",)))
    args = types.SimpleNamespace(config=channel_config, json=True)
    code = preflight._check_command(args)  # noqa: SLF001
    document = json.loads(capsys.readouterr().out)
    assert code != 0
    assert document["ok"] is False
    assert any(issue["code"] == "levels"
               for issue in document["failures"])


def test_check_json_emits_a_document_when_the_config_itself_is_refused(
        capsys, monkeypatch, channel_config):
    """The branch the old fixture was reaching by accident, stated.

    A config the loader will not accept -- inputs that are not on this
    disk, a version identity it cannot resolve -- stops the preflight
    just as surely as a build failure does, and the same caller is still
    running ``json.loads(stdout)``.  Both machines this branch measured
    took this branch instead of the one the two gates above name, and
    nothing held it, so it is held here.
    """

    from woof import case_data
    from woof.ingest import preflight

    monkeypatch.setattr(case_data, "load_experiment_case", _refuse)
    args = types.SimpleNamespace(config=channel_config, json=True)
    code = preflight._check_command(args)  # noqa: SLF001
    document = json.loads(capsys.readouterr().out)
    assert code != 0
    assert document["ok"] is False
    assert document["refusal_type"] == "ValueError"
    assert "declared inputs are not on this disk" in document["refusal"]


def _refuse(*_a, **_k):
    raise ValueError("experiment config: declared inputs are not on this "
                     "disk")


def test_text_mode_still_raises_rather_than_printing_a_document(
        monkeypatch, channel_config):
    """--json is what owes stdout a document; plain text owes a refusal.

    Without this the fix above could drift into swallowing every loader
    refusal on the text route too, where the front door's own refusal
    boundary is what prints it.
    """

    from woof import case_data
    from woof.ingest import preflight

    monkeypatch.setattr(case_data, "load_experiment_case", _refuse)
    args = types.SimpleNamespace(config=channel_config, json=False)
    with pytest.raises(ValueError):
        preflight._check_command(args)  # noqa: SLF001
