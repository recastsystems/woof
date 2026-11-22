# SPDX-License-Identifier: Apache-2.0
"""The recorder finds the wheel's own runner.

THE BREAKAGE (hex 0.6.0 candidate, a fresh ``[ad]`` install on an RTX 4090, 2026-11-16): ``python -m woof.hex.ad.record
record --lam ...`` stopped on "no replay runner: pass --runner, set HEXAD_RUNNER, or install the hexad runner" although
the wheel carries ``woof/hex/libexec/ad-binder``. ``which_runner`` looked for ``woof.hex.ad.runner``, a module no wheel
has, and never asked ``woof.hex.hexad``, which the doors use.
"""
import os

from woof.hex import hexad
from woof.hex.ad.record import env as record_env


def _runner(tmp_path, name="ad-binder"):
    path = tmp_path / name
    path.write_text("#!/bin/sh" + chr(10) + "exit 0" + chr(10))
    path.chmod(0o755)
    return path


def test_the_packaged_runner_is_found_without_a_flag(monkeypatch, tmp_path):
    packaged = _runner(tmp_path)
    monkeypatch.delenv("HEXAD_RUNNER", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    monkeypatch.setattr(hexad, "binary", lambda: packaged)
    assert record_env.which_runner(None) == packaged


def test_a_named_runner_and_the_variable_come_first(monkeypatch, tmp_path):
    packaged = _runner(tmp_path)
    (tmp_path / "other").mkdir()
    named = _runner(tmp_path / "other", "mine")
    monkeypatch.setattr(hexad, "binary", lambda: packaged)
    monkeypatch.delenv("HEXAD_RUNNER", raising=False)
    assert record_env.which_runner(str(named)) == named
    monkeypatch.setenv("HEXAD_RUNNER", str(named))
    assert record_env.which_runner(None) == named


def test_an_install_without_the_runner_falls_to_path_then_none(monkeypatch, tmp_path):
    def missing():
        raise hexad.HexadNotInstalled("ad-binder is not installed")

    monkeypatch.setattr(hexad, "binary", missing)
    monkeypatch.delenv("HEXAD_RUNNER", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    assert record_env.which_runner(None) is None
    on_path = _runner(tmp_path)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + str(tmp_path / "empty"))
    assert record_env.which_runner(None) == on_path
