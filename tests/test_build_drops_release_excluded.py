"""A wheel or sdist built from the development tree carries nothing the export drops.

THE BREAKAGE THIS PREVENTS: a 3.0.3 candidate wheel built from the
development tree carried 45 files RELEASE-EXCLUDE.txt keeps out of the public
tree (44 configuration records whose paths name the build machine's home
directory, and one development script), because ``configs = ["**/*"]`` and
the ``tools`` package swept them in.  The export strips them, so a wheel built
from the public tree never held them.  ``setup.py``'s build_py and sdist now
drop every RELEASE-EXCLUDE'd file; this measures that with setup.py's own
command classes against the real tree, and that the plain setuptools
selection would still have taken them (so the measurement is not vacuous).

The public tree carries no RELEASE-EXCLUDE.txt, so there is nothing to
measure there and the module skips.
"""

from __future__ import annotations

import os
from pathlib import Path
import runpy
import tomllib

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
EXCLUDE = REPO_ROOT / "RELEASE-EXCLUDE.txt"

pytestmark = [
    pytest.mark.skipif(
        not (EXCLUDE.is_file() and (REPO_ROOT / "setup.py").is_file()),
        reason="the export list is not in this tree (published snapshot)"),
    # (WOOF 1.1.7) WOOF's tree is built from the engine's export (components/SOURCES.json records the fold), so
    # every RELEASE-EXCLUDE'd file was dropped before it arrived and a build here has nothing to drop; the
    # engine's development tree, where the 45 files of the 3.0.3 breakage live, is where this measures the filter.
    pytest.mark.skipif(
        (REPO_ROOT / "components" / "SOURCES.json").is_file(),
        reason="WOOF's tree is the engine's export: no RELEASE-EXCLUDE'd file is present to be dropped; "
               "the engine's development tree runs this measurement"),
]


def _rules() -> list[str]:
    from tools.release_exclusions import read_exclusions
    return read_exclusions(REPO_ROOT)


def _setup(monkeypatch: pytest.MonkeyPatch) -> tuple[dict, dict]:
    import setuptools

    captured: dict = {}
    monkeypatch.setattr(setuptools, "setup", lambda **kwargs: captured.update(kwargs))
    namespace = runpy.run_path(str(REPO_ROOT / "setup.py"), run_name="setup_under_test")
    return namespace, captured


def _selection(monkeypatch: pytest.MonkeyPatch, build_py_class) -> set[str]:
    """Every module and data file a build_py class would put in the wheel."""

    import setuptools
    from setuptools.config import expand

    config = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["tool"]["setuptools"]
    find = config["packages"]["find"]
    packages = expand.find_packages(include=find.get("include", ["*"]),
                                    exclude=find.get("exclude", []), root_dir=str(REPO_ROOT))
    distribution = setuptools.dist.Distribution({
        "name": "woof", "packages": packages, "package_data": config["package-data"],
        "exclude_package_data": config.get("exclude-package-data", {})})
    distribution.script_name = "setup.py"
    command = build_py_class(distribution)
    command.finalize_options()
    command.manifest_files = {}  # the declaration alone; a manifest member would pass the same filter
    monkeypatch.chdir(REPO_ROOT)
    selected = {Path(filename).as_posix() for _p, _m, filename in command.find_all_modules()}
    for package in packages:
        selected |= {Path(name).as_posix()
                     for name in command.find_data_files(package, command.get_package_dir(package))}
    return selected


def test_the_wheel_build_drops_every_release_excluded_file(monkeypatch: pytest.MonkeyPatch) -> None:
    from setuptools.command.build_py import build_py as plain
    from tools.release_exclusions import matches

    rules = _rules()
    namespace, captured = _setup(monkeypatch)
    operational = captured["cmdclass"]["build_py"]

    everything = _selection(monkeypatch, plain)
    swept = sorted(name for name in everything if matches(name, rules))
    assert swept, ("plain setuptools selects no RELEASE-EXCLUDE'd file, so this "
                   "measurement no longer shows the filter doing anything")
    assert any(name.endswith(".py") for name in swept) and any(not name.endswith(".py") for name in swept), (
        f"the measurement should cover a module and a data file: {swept[:5]}")

    shipped = _selection(monkeypatch, operational)
    leaked = sorted(name for name in shipped if matches(name, rules))
    assert not leaked, (f"{len(leaked)} file(s) the export drops would ship in a wheel built "
                        f"here: {leaked[:10]}")

    # Nothing else left: what the operational build drops beyond them is its
    # older classes (the verification oracles, the development probes).
    def older_class(name: str) -> bool:
        path = Path(name)
        return namespace["is_verify_oracle_file"](name) or (
            path.suffix == ".py" and namespace["is_development_module"](".".join(path.parent.parts), path.stem))

    lost = sorted(name for name in everything - shipped if name not in swept and not older_class(name))
    assert not lost, f"the filter dropped files the export keeps: {lost[:10]}"


def test_the_sdist_drops_them_too(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from setuptools.command.build_py import build_py as plain
    from tools.release_exclusions import matches

    rules = _rules()
    namespace, captured = _setup(monkeypatch)
    is_excluded = namespace["is_release_excluded"]
    private = next(name for name in sorted(_selection(monkeypatch, plain))
                   if matches(name, rules) and not name.endswith(".py"))
    public = "pyproject.toml"
    assert is_excluded(private) and is_excluded(REPO_ROOT / private)
    assert is_excluded(private.replace("/", os.sep))
    assert not is_excluded(public) and not is_excluded(REPO_ROOT / public)

    import setuptools

    sdist_class = captured["cmdclass"]["sdist"]
    distribution = setuptools.dist.Distribution({"name": "woof", "version": "0.0.0"})
    distribution.script_name = "setup.py"
    command = sdist_class(distribution)
    command.ensure_finalized()
    base = tmp_path / "gpuwm-0.0.0"
    command.make_release_tree(str(base), [public, private])
    assert (base / public).is_file()
    assert not (base / private).exists(), f"the sdist tree still carries {private}"
