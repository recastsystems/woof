"""Two CuPy packages in one environment are a doctor gap, not OK.

THE BREAKAGE (0.6.0, a fresh install from the published files, one RTX 5090, a fresh Python 3.14 venv):
`pip install "recast-woof[gpu]"` and then `pip install "recast-woof[ad]"` left cupy-cuda12x and cupy-cuda13x both
installed.  Every later command printed CuPy's "may not function correctly because multiple CuPy packages are
installed" and `woof hex doctor` still printed `OK cupy (the CUDA lane)`.  These tests fail if doctor goes back to
reading only whether `import cupy` works, and if the two extras ever name two CuPy packages a user is told to
install together.
"""

from __future__ import annotations

import importlib
from pathlib import Path
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _doctor(monkeypatch, driver_major=13):
    doctor = importlib.import_module("woof.hex.doctor")
    monkeypatch.setenv("WOOF_BACKEND", "cuda")
    import woof.gpu_smi as smi
    monkeypatch.setattr(smi, "is_amd", lambda: False)
    monkeypatch.setattr(doctor, "_import_probe", lambda module: (True, "imported, 14.2.0"))
    monkeypatch.setattr(doctor, "_cupy_runtime_probe", lambda: {"cupy_version": "14.2.0", "runtime_version": 13020})
    monkeypatch.setattr(doctor, "_driver_cuda_major", lambda: driver_major)
    return doctor


def test_two_cupy_packages_are_a_required_gap_with_one_command(monkeypatch):
    doctor = _doctor(monkeypatch)
    monkeypatch.setattr(doctor, "_installed_cupy_wheels", lambda: ["cupy-cuda12x", "cupy-cuda13x"])
    findings = doctor.check_gpu_runtime()
    assert [f.status for f in findings] == [doctor.MISSING]
    gap = findings[0]
    assert "cupy-cuda12x" in gap.detail and "cupy-cuda13x" in gap.detail
    assert gap.required, "two CuPy packages must make doctor exit 1"
    assert doctor.blocking_gaps(findings) == findings
    # The compact view prints the first command line only: it must be the whole remedy, removal and install.
    compact = doctor.render(findings, explain=False)
    assert "MISSING" in compact and "OK " not in compact
    extra = "ad" if sys.version_info >= (3, 14) else "gpu-cu13"
    assert f'pip uninstall -y cupy-cuda12x cupy-cuda13x && pip install "recast-woof[{extra}]"' in compact, compact


def test_two_cupy_packages_on_a_cuda_12_driver_name_the_forecast_extra(monkeypatch):
    doctor = _doctor(monkeypatch, driver_major=12)
    monkeypatch.setattr(doctor, "_installed_cupy_wheels", lambda: ["cupy-cuda12x", "cupy-cuda13x"])
    gap = doctor.check_gpu_runtime()[0]
    assert gap.status == doctor.MISSING and gap.required
    assert 'pip uninstall -y cupy-cuda12x cupy-cuda13x && pip install "recast-woof[gpu-cu12]"' in gap.remedy
    commands = [line for line in gap.remedy.splitlines() if not line.strip().startswith("#")]
    assert len(commands) == 1 and '[ad]"' not in commands[0]


def test_one_cupy_package_is_still_verified(monkeypatch):
    doctor = _doctor(monkeypatch)
    monkeypatch.setattr(doctor, "_installed_cupy_wheels", lambda: ["cupy-cuda13x"])
    findings = doctor.check_gpu_runtime()
    assert [f.status for f in findings] == [doctor.VERIFIED]


def test_the_lister_names_only_cupy_distributions(monkeypatch):
    doctor = importlib.import_module("woof.hex.doctor")
    from importlib import metadata

    class D:
        def __init__(self, name):
            self.metadata = {"Name": name}

    monkeypatch.setattr(metadata, "distributions",
                        lambda: [D("cupy-cuda12x"), D("CuPy-CUDA13x"), D("cupyx-extras-fake"), D("numpy")])
    assert doctor._installed_cupy_wheels() == ["cupy-cuda12x", "cupy-cuda13x"]


def _project():
    """The distribution's own pyproject table: the nearest pyproject.toml up from the tests that declares `[ad]`.

    In woof hex's tree that is the tree's own; where this package is folded into another distribution it is
    that distribution's, whose extras carry the same CuPy packages under its own name."""
    for folder in (ROOT, *ROOT.parents):
        path = folder / "pyproject.toml"
        if path.is_file():
            project = tomllib.loads(path.read_text(encoding="utf-8")).get("project", {})
            if "ad" in project.get("optional-dependencies", {}):
                return project
    import pytest
    pytest.skip("no pyproject.toml declaring the [ad] extra above the tests")


def _cupy_of(extra, extras, seen=(), name=None):
    """The CuPy distributions an extra of this package installs, following its own `<name>[...]` references."""
    names = set()
    prefix = f"{name or _project()['name']}["
    for requirement in extras[extra]:
        head = requirement.split(";")[0].strip()
        if head.startswith(prefix):
            for inner in head[len(prefix):].split("]")[0].split(","):
                if inner.strip() not in seen:
                    names |= _cupy_of(inner.strip(), extras, (*seen, extra), name)
        elif head.lower().startswith("cupy-"):
            names.add(head.split("[")[0].split("=")[0].split(">")[0].split("<")[0].strip().lower())
    return names


def test_each_extra_brings_one_cupy_and_the_pages_say_ad_is_installed_alone():
    extras = _project()["optional-dependencies"]
    for extra in ("gpu", "gpu-cu12", "gpu-cu13", "ad"):
        assert len(_cupy_of(extra, extras)) == 1, (extra, _cupy_of(extra, extras))
    # [gpu] and [ad] bring different CuPy packages, so no page may tell a user to install one on top of the other.
    assert _cupy_of("gpu", extras) != _cupy_of("ad", extras)
    for page in ("README.md", "docs/differentiable/FIRST-HOUR.md", "docs/differentiable/README.md"):
        text = " ".join((ROOT / page).read_text(encoding="utf-8").split())
        assert "not on top of" in text and "[gpu]" in text, (
            f"{page} must say the [ad] extra goes in its own environment, not on top of [gpu]")
