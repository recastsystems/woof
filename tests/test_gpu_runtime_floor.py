"""The front door refuses a CuPy older than the engine's floor, by name.

Breakage this prevents: the dycore's bookkeeping launches pass by-value
tables as ``numpy.void`` (since 2.8.1), which CuPy 13 rejects.  A box that
still held CuPy 13 (a rented 8x RTX 5090 image, 2026-10-02) passed the
front door, because the door only asked whether CuPy resolved, and stopped
at its first model step with "TypeError: Unsupported type <class
'numpy.void'>".  The floor is the one the GPU extras declare.
"""
from __future__ import annotations

import importlib
from pathlib import Path
import re
import sys

import pytest

from woof import capabilities

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _fake_cupy(root: Path, version: str) -> None:
    package = root / "cupy"
    package.mkdir()
    (package / "__init__.py").write_text("raise ImportError('not for import')\n",
                                         encoding="utf-8")
    (package / "_version.py").write_text(f"__version__ = '{version}'\n",
                                         encoding="utf-8")


@pytest.mark.parametrize("version, expected", [("13.6.0", (13, 6)),
                                               ("14.2.0", (14, 2))])
def test_the_version_is_read_beside_the_module_without_importing_it(
        tmp_path, monkeypatch, version, expected):
    _fake_cupy(tmp_path, version)
    monkeypatch.delitem(sys.modules, "cupy", raising=False)
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    assert capabilities.installed_cupy_version() == expected
    assert "cupy" not in sys.modules


def test_an_unreadable_version_is_not_refused(tmp_path, monkeypatch):
    package = tmp_path / "cupy"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    monkeypatch.delitem(sys.modules, "cupy", raising=False)
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    assert capabilities.installed_cupy_version() is None
    assert capabilities.outdated_gpu_runtime() is None


@pytest.mark.parametrize("command", ["sim", "go"])
def test_cupy_13_is_refused_at_the_door_with_the_floor_and_the_remedy(
        monkeypatch, command):
    monkeypatch.setattr(capabilities, "is_installed", lambda module: True)
    monkeypatch.setattr(capabilities, "installed_cupy_version", lambda: (13, 6))
    with pytest.raises(capabilities.CapabilityMissing) as refused:
        capabilities.require_for_command(command)
    text = str(refused.value)
    assert "CuPy 14.0 or newer" in text and "CuPy 13.6" in text
    assert "recast-woof[gpu-cu12]" in text and "recast-woof[gpu-cu13]" in text
    assert refused.value.requirement is capabilities.GPU_RUNTIME


@pytest.mark.parametrize("found", [(14, 0), (14, 2), (15, 0), None])
def test_a_cupy_at_or_above_the_floor_or_unreadable_passes(monkeypatch, found):
    monkeypatch.setattr(capabilities, "is_installed", lambda module: True)
    monkeypatch.setattr(capabilities, "installed_cupy_version", lambda: found)
    capabilities.require_for_command("sim")
    assert capabilities.unmet_run_requirements() == ()


def test_readiness_reports_an_outdated_cupy_as_unmet(monkeypatch):
    monkeypatch.setattr(capabilities, "is_installed", lambda module: True)
    monkeypatch.setattr(capabilities, "installed_cupy_version", lambda: (13, 6))
    assert capabilities.unmet_run_requirements() == (capabilities.GPU_RUNTIME,)


def test_the_floor_is_the_one_the_gpu_extras_declare():
    text = (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    floors = set()
    for extra in ("gpu-cu12", "gpu-cu13"):
        # The CuPy requirement comes first; a constraint may follow it (gpu-cu13 keeps NVRTC below 13.5).
        line = re.search(rf'^{extra}\s*=\s*\["cupy-cuda1[23]x\[ctk\]>=(\d+)\.(\d+)"(?:,\s*"[^"]*")*\]',
                         text, re.MULTILINE)
        assert line, f"{extra} no longer declares a CuPy floor this test can read"
        floors.add((int(line.group(1)), int(line.group(2))))
    assert floors == {capabilities.GPU_RUNTIME_MINIMUM}
