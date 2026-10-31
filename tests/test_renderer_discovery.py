"""The imported package owns renderer selection, even with two installs."""

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from woof import bridges, rustwx


@pytest.fixture
def estate(tmp_path, monkeypatch):
    package = tmp_path / "installed" / "woof"
    package.mkdir(parents=True)
    # Relocate only the package anchors. Discovery, acceptance and the native
    # executable remain real; no candidate or subprocess function is stubbed.
    monkeypatch.setattr(rustwx, "__file__", str(package / "rustwx.py"))
    monkeypatch.setattr(bridges, "__file__", str(package / "bridges.py"))
    monkeypatch.setattr(rustwx, "default_bridge_dir", lambda: tmp_path / "staged")
    monkeypatch.setattr(rustwx, "legacy_bridge_candidates", lambda _: ())
    monkeypatch.delenv(rustwx.RENDERER_ENV, raising=False)
    name = bridges.executable_name("rw_wrfbatch")
    paths = {
        "bundled": package / "libexec" / "bridges" / name,
        "checkout": package.parent / "tools" / "rustwx" / "target" / "release" / name,
        "sealed": package.parent / "libexec" / "bridges" / name,
        "staged": tmp_path / "staged" / name,
        "foreign": tmp_path / "another-checkout" / "bin" / name,
    }
    return paths


def _binary(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    # A native executable, usable on every supported test host. This tests
    # discovery rather than the renderer ABI, which test_render_rust covers.
    shutil.copy2(sys.executable, path)
    return path.resolve()


@pytest.mark.parametrize("competitor", ["staged", "foreign"])
def test_bundle_wins_with_wrong_binary_first_on_path(
        estate, monkeypatch, capsys, competitor):
    own = _binary(estate["bundled"])
    wrong = _binary(estate[competitor])
    monkeypatch.setenv("PATH", os.pathsep.join((str(wrong.parent), str(own.parent))))
    monkeypatch.chdir(wrong.parent)
    assert Path(shutil.which(own.name)).resolve() == wrong
    assert rustwx.find_renderer() == own
    assert str(own) in capsys.readouterr().err
    assert subprocess.run([str(own), "--version"], capture_output=True,
                          timeout=10).returncode == 0


def test_checkout_wins_over_foreign_path_without_a_bundle(estate, monkeypatch, capsys):
    own = _binary(estate["checkout"])
    wrong = _binary(estate["foreign"])
    monkeypatch.setenv("PATH", os.pathsep.join((str(wrong.parent), str(own.parent))))
    monkeypatch.chdir(wrong.parent)
    assert rustwx.find_renderer() == own
    assert str(own) in capsys.readouterr().err


def test_explicit_override_still_wins(estate, monkeypatch, capsys):
    _binary(estate["bundled"])
    declared = _binary(estate["foreign"])
    monkeypatch.setenv(rustwx.RENDERER_ENV, str(declared))
    assert rustwx.find_renderer() == declared
    assert str(declared) in capsys.readouterr().err


def test_missing_override_does_not_silently_use_bundle(estate, monkeypatch):
    _binary(estate["bundled"])
    monkeypatch.setenv(rustwx.RENDERER_ENV, str(estate["foreign"]))
    with pytest.raises(FileNotFoundError, match="names a missing file"):
        rustwx.find_renderer()


def test_missing_package_renderer_does_not_adopt_path_binary(estate, monkeypatch):
    wrong = _binary(estate["foreign"])
    monkeypatch.setenv("PATH", str(wrong.parent))
    assert rustwx.find_renderer() is None


@pytest.mark.parametrize("rung", ["checkout", "sealed"])
def test_developer_build_wins_over_older_staged_bundle(estate, capsys, rung):
    """Wheel staging must not shadow a later checkout or sealed build."""
    old = _binary(estate["bundled"])
    os.utime(old, (1, 1))
    fresh = _binary(estate[rung])
    assert fresh.stat().st_mtime > old.stat().st_mtime
    assert rustwx.find_renderer() == fresh
    assert str(fresh) in capsys.readouterr().err
