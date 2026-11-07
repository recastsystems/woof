"""``configs/`` ships, because two headline pages send readers to it.

The state this file closes: 97 files in ``configs/``, 0 of them in any
wheel.  ``[tool.setuptools.packages.find]`` named ``woof*``, ``tools`` and
``tilestream*`` and nothing else, so the directory was in no distribution
at all -- and two of the project's most-read pages point at it anyway:

* ``docs/public/VERIFICATION.md`` section 7 reproduces the headline
  Thompson/RRTMG comparison with
  ``woof check configs/real74_thompson_1218z_rrtmg_legacy_4dom.toml``.
* ``docs/public/LES.md`` sends a reader to ``configs/`` for "a real,
  config-driven run" and names the frozen archives under
  ``configs/frozen/`` that the LES receipts are pinned to.

Both instructions worked from a git checkout and from nowhere else.  A
reader who followed the documented install line got a refusal, and the
refusal was the accurate one -- ``woof/verify/cases/_repo_config.py``
already said "``configs/`` is not a Python package and ships in no wheel"
and offered ``--config``/``WOOF_CONFIGS_ROOT``.  An accurate refusal is
still a headline claim nobody can reach.

Where the files land is not incidental.  ``_repo_config`` resolves a
repository config as ``Path(woof/verify/cases/_repo_config.py).parents[3]
/ "configs"``, which inside an install is ``<site-packages>/configs``.
Shipping ``configs`` as a top-level directory of the wheel therefore puts
it exactly where the resolver has always looked, and
:func:`test_the_shipped_configs_land_where_the_resolver_looks` pins that
agreement rather than leaving it to a coincidence of two files.

The measurement runs setuptools' own ``build_py.find_data_files`` against
the real ``pyproject.toml`` -- the same discipline as
``tests/test_package_data_coverage.py`` -- so what it reports is what the
wheel would really contain, and it expands the declared
``packages.find`` include list rather than restating it.
"""

from __future__ import annotations

import os
from pathlib import Path
import tomllib

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
CONFIGS_ROOT = REPO_ROOT / "configs"

_MODULE_SUFFIXES = (".py", ".pyc", ".pyo")
_SKIP_DIRS = frozenset({"__pycache__", ".pytest_cache", ".ruff_cache"})

#: The exact files the two headline pages name.  A glob that silently
#: stopped matching one of these would leave the page's own command
#: unrunnable, so they are pinned by name as well as by coverage.
_DOCUMENTED_CONFIGS = (
    # docs/public/VERIFICATION.md section 7, "Reproduce this".
    "configs/real74_thompson_1218z_rrtmg_legacy_4dom.toml",
    # docs/public/LES.md: the frozen archives the LES receipts pin.
    "configs/frozen/les_tornado_100m_mayfield_20211210.toml",
    "configs/frozen/les_nest_250m_grayzone.toml",
    "configs/frozen/les_nest_250m_km3.toml",
)


def _require_source_tree() -> None:
    if not PYPROJECT.is_file() or not CONFIGS_ROOT.is_dir():
        pytest.skip("the configs packaging gate needs the source tree")


def _pyproject() -> dict:
    with PYPROJECT.open("rb") as stream:
        return tomllib.load(stream)


def _declared_packages() -> list[str]:
    """Expand the real ``packages.find`` directive, rather than restating it."""

    from setuptools.config import expand

    find = _pyproject()["tool"]["setuptools"]["packages"]["find"]
    return expand.find_packages(
        include=find.get("include", ["*"]),
        exclude=find.get("exclude", []),
        root_dir=str(REPO_ROOT),
    )


def _files_setuptools_would_ship(monkeypatch: pytest.MonkeyPatch) -> set[str]:
    """Ask setuptools which files the declaration selects, in wheel paths."""

    try:
        import setuptools
    except ModuleNotFoundError as exc:      # pragma: no cover - env defect
        raise AssertionError(
            "setuptools is not installed, so this file cannot measure what "
            "the wheel would contain -- and a silent skip of a packaging gate "
            "reports green. pyproject.toml's [build-system] requires it, so "
            "any environment that can build this project has it. Remedy: "
            "pip install setuptools") from exc

    from setuptools.command.build_py import build_py

    config = _pyproject()["tool"]["setuptools"]
    distribution = setuptools.dist.Distribution({
        "name": "woof",
        "packages": _declared_packages(),
        "package_data": config["package-data"],
        "exclude_package_data": config.get("exclude-package-data", {}),
    })
    command = build_py(distribution)
    command.finalize_options()

    monkeypatch.chdir(REPO_ROOT)        # setuptools globs relative to cwd
    shipped: set[str] = set()
    for _package, src_dir, _build_dir, filenames in (
            command.get_data_files_without_manifest()):
        for name in filenames:
            shipped.add(Path(src_dir, name).as_posix())
    return shipped


def _configs_on_disk() -> set[str]:
    present: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(CONFIGS_ROOT):
        dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS)
        for name in filenames:
            if name.endswith(_MODULE_SUFFIXES):
                continue
            present.add(Path(dirpath, name).relative_to(REPO_ROOT).as_posix())
    return present


def test_the_configs_directory_is_a_declared_package() -> None:
    """The declaration half, readable without setuptools.

    ``packages.find`` is what decides whether the directory exists in the
    distribution at all; ``package-data`` only decides which of its files
    come along. Both have to name it, and a tree without setuptools can
    still check that much.
    """

    _require_source_tree()
    config = _pyproject()["tool"]["setuptools"]
    include = config["packages"]["find"].get("include", [])
    assert any(pattern.rstrip("*") == "configs" for pattern in include), (
        f"[tool.setuptools.packages.find] include is {include} and names no "
        "configs entry, so configs/ is in no wheel -- while "
        "docs/public/VERIFICATION.md and docs/public/LES.md both send "
        "readers to it")
    assert "configs" in config["package-data"], (
        "configs is a declared package but [tool.setuptools.package-data] "
        "names no pattern for it, so the wheel would carry the directory "
        "with none of its files")


def test_every_config_in_the_tree_reaches_the_wheel(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """The measurement half: 97 files in the tree, 97 in the wheel."""

    _require_source_tree()
    present = _configs_on_disk()
    assert len(present) >= 50, (
        f"found only {len(present)} file(s) under configs/ -- the walk is "
        "broken, and a broken walk passes this file trivially")

    shipped = _files_setuptools_would_ship(monkeypatch)
    missing = sorted(present - shipped)
    assert not missing, (
        f"{len(missing)} of {len(present)} file(s) under configs/ would be "
        "omitted from a wheel. Widen the glob in "
        "[tool.setuptools.package-data] under the 'configs' key (do not "
        "append filenames):\n  " + "\n  ".join(missing[:20]))


def test_the_configs_the_headline_docs_name_are_in_the_wheel(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Named files, because a page whose command cannot run is worse than none.

    ``VERIFICATION.md``'s reproduce recipe and ``LES.md``'s frozen archives
    are the project's two loudest claims. Pinning them by name means a
    renamed or moved config fails here rather than in somebody's terminal.
    """

    _require_source_tree()
    shipped = _files_setuptools_would_ship(monkeypatch)
    from _woof_snapshot import left_out
    for relative in _DOCUMENTED_CONFIGS:
        if left_out(relative):
            continue
        assert (REPO_ROOT / relative).is_file(), (
            f"a headline document names {relative}, which is not in the tree")
        assert relative in shipped, (
            f"{relative} is named by a headline document but would not ship "
            f"in a wheel, so following that page from a pip install fails")


def test_the_shipped_configs_land_where_the_resolver_looks() -> None:
    """Packaging and resolution must be about the same directory.

    ``_repo_config`` looks for ``<woof package parent>/configs``. Shipping
    ``configs`` as a top-level package puts it exactly there inside an
    install. If either side moves -- the package renamed, or the resolver
    re-rooted -- the files would ship to a path nothing reads, which is
    indistinguishable from not shipping them.
    """

    _require_source_tree()
    from woof.verify.cases import _repo_config

    roots = _repo_config.config_roots()
    assert roots, "the repository-config resolver searches nowhere at all"
    package_parent = Path(_repo_config.__file__).resolve().parents[3]
    assert package_parent / "configs" in [Path(r) for r in roots], (
        f"the resolver searches {[str(r) for r in roots]}, none of which is "
        f"the {package_parent / 'configs'} a top-level `configs` package "
        f"installs to")
    # And the packaged name really is top-level, not nested under gpuwm/:
    # a `gpuwm.configs` package would install one level too deep.
    assert "configs" in _pyproject()["tool"]["setuptools"]["package-data"]


def _sdist_default_files(monkeypatch: pytest.MonkeyPatch) -> set[str]:
    """The Python file set setuptools puts in an sdist before MANIFEST.in.

    ``sdist._add_defaults_python`` is the step that folds ``package_data``
    into the source distribution, so this measures whether the configs
    reach the sdist by the same declaration that puts them in the wheel --
    rather than assuming they do.
    """

    import setuptools
    from setuptools.command.sdist import sdist

    config = _pyproject()["tool"]["setuptools"]
    distribution = setuptools.dist.Distribution({
        "name": "woof",
        "version": _pyproject()["project"]["version"],
        "packages": _declared_packages(),
        "package_data": config["package-data"],
        "exclude_package_data": config.get("exclude-package-data", {}),
    })
    distribution.script_name = "setup.py"
    command = sdist(distribution)
    command.finalize_options()

    from distutils.filelist import FileList

    command.filelist = FileList()
    monkeypatch.chdir(REPO_ROOT)
    command._add_defaults_python()
    return {name.replace(os.sep, "/") for name in command.filelist.files}


def test_the_sdist_carries_the_configs_too(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """The sdist is the other route to the reproduce recipe.

    ``exclude-package-data`` governs the wheel only, and ``MANIFEST.in``
    governs the sdist, so the two can drift. This measures the sdist's own
    default file set rather than trusting that ``package-data`` reaches it,
    then applies ``MANIFEST.in`` and checks the public configs survive while
    private snapshot exclusions stay out of a direct worktree sdist build.
    """

    _require_source_tree()
    present = _configs_on_disk()
    in_sdist = _sdist_default_files(monkeypatch)
    missing = sorted(present - in_sdist)
    assert not missing, (
        f"{len(missing)} of {len(present)} configs would be absent from the "
        "sdist:\n  " + "\n  ".join(missing[:20]))

    from setuptools._distutils.filelist import FileList
    from tools.release_exclusions import matches, read_exclusions

    rules = (read_exclusions(REPO_ROOT)
             if (REPO_ROOT / "RELEASE-EXCLUDE.txt").is_file() else [])
    public = {name for name in present if not matches(name, rules)}
    files = FileList()
    files.allfiles = [os.fspath(Path(name)) for name in sorted(in_sdist)]
    files.files = list(files.allfiles)
    manifest = REPO_ROOT / "MANIFEST.in"
    if manifest.is_file():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.lstrip().startswith("#"):
                files.process_template_line(line)
    included = {name.replace(os.sep, "/") for name in files.files}
    assert public <= included, (
        f"MANIFEST.in drops public configs from the sdist: {sorted(public - included)}")
    assert not (present - public) & included, (
        "the sdist includes private RELEASE-EXCLUDE'd config receipts")
    excluded = _pyproject()["tool"]["setuptools"].get(
        "exclude-package-data", {})
    assert "configs" not in excluded, (
        "[tool.setuptools.exclude-package-data] excludes configs files from "
        f"the wheel: {excluded.get('configs')}")
