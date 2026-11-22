"""``docs/mpas-seam.md`` ships, because an external engine pins it.

The state this file closes, measured against the published 2.5.x wheels:
woof hex verifies the physics seam it builds against by hashing sixteen
engine files by SHA-256, keyed on their REPOSITORY paths, and it runs
that one key set against two roots -- a woof source checkout, and
``site-packages`` (the directory ``importlib.util.find_spec("woof")``'s
origin sits two levels under).  Fifteen of the sixteen resolve under
both, because they live inside the ``woof`` package and a wheel carries
that package whole.  ``docs/mpas-seam.md`` -- the document that STATES
the seam contract those fifteen implement -- resolved under the checkout
only, because no distribution placed it anywhere at all.  Its verifier
reported it ``absent`` on every installed engine, and the remedy it
printed told the reader to clone the repository beside the wheel they
had just installed.

Where the file lands is the whole point and is not incidental.  A
consumer running one manifest against two roots needs the SAME key to
resolve under both, so the installed copy has to sit at
``<site-packages>/docs/mpas-seam.md``.  ``woof/docs/mpas-seam.md``
would work only for a consumer willing to carry a second key for one
document, which is how a pin quietly grows a special case.  That is the
same reasoning ``configs`` shipped under
(``tests/test_configs_are_packaged.py``), and this file measures the
agreement rather than leaving it to a coincidence of two declarations.

The measurement asks setuptools' own ``build_py`` what the declaration
selects, exactly as the configs gate does, so what it reports is what a
wheel would really contain.  The wheel-level test opens a built wheel.
"""

from __future__ import annotations

from pathlib import Path
import tomllib
import zipfile

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"

#: The repository path, which is also the archive path in the wheel and
#: the install path under site-packages.  One string, because the whole
#: claim of this file is that those three are the same string.
SEAM_DOC = "docs/mpas-seam.md"


def _require_source_tree() -> None:
    if not PYPROJECT.is_file() or not (REPO_ROOT / SEAM_DOC).is_file():
        pytest.skip("the seam-document packaging gate needs the source tree")


def _pyproject() -> dict:
    with PYPROJECT.open("rb") as stream:
        return tomllib.load(stream)


def _declared_packages() -> list[str]:
    from setuptools.config import expand

    find = _pyproject()["tool"]["setuptools"]["packages"]["find"]
    return expand.find_packages(
        include=find.get("include", ["*"]),
        exclude=find.get("exclude", []),
        root_dir=str(REPO_ROOT),
    )


def _files_setuptools_would_ship(monkeypatch: pytest.MonkeyPatch) -> set[str]:
    import setuptools
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


def test_docs_is_a_declared_package_and_is_anchored() -> None:
    """``docs``, never ``docs*``, and the difference is 200+ files.

    The repository's ``docs/`` holds receipt trees that carry their own
    ``.py`` (``docs/internal/receipts/...``).  An unanchored pattern
    would make every one of those subdirectories a package of this
    wheel and ship the scripts inside them -- development record, in
    site-packages, under import names nobody declared.
    """

    _require_source_tree()

    find = _pyproject()["tool"]["setuptools"]["packages"]["find"]
    assert "docs" in find["include"], (
        "pyproject's packages.find no longer includes `docs`, so "
        f"{SEAM_DOC} reaches no wheel and woof hex's seam pin is back to "
        "demanding a source checkout beside the install")
    assert not any(pattern.startswith("docs") and pattern != "docs"
                   for pattern in find["include"]), (
        "packages.find includes a docs pattern wider than the anchored "
        "`docs`; that sweeps docs/internal/receipts/**.py into the wheel")

    packages = _declared_packages()
    assert "docs" in packages
    strays = sorted(name for name in packages if name.startswith("docs."))
    assert not strays, (
        "the docs include pattern now matches subpackages, which ships "
        f"development record in the wheel: {strays}")


def test_the_sdist_manifest_cannot_widen_the_docs_payload(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """The pages MANIFEST.in ships in the source archive stay out of the wheel.

    A pyproject build turns ``include-package-data`` on, so every sdist
    member under a declared package also reaches the wheel.  ``docs`` is a
    declared package, and when the 2.9.2 source archive started carrying
    the public pages the manylinux wheel grew by 126 ``docs/`` members and
    1.85 MB, over the cut's per-file limit.  setup.py's build_py drops the
    manifest contribution for ``docs``; this runs that class with a manifest
    contribution present and reads back the selection.
    """

    _require_source_tree()

    import runpy
    import setuptools

    captured: dict = {}
    monkeypatch.setattr(setuptools, "setup",
                        lambda **kwargs: captured.update(kwargs))
    namespace = runpy.run_path(str(REPO_ROOT / "setup.py"),
                               run_name="setup_under_test")
    assert "docs" in namespace["DECLARATION_ONLY_DATA_PACKAGES"]
    build_py_class = captured["cmdclass"]["build_py"]

    config = _pyproject()["tool"]["setuptools"]
    distribution = setuptools.dist.Distribution({
        "name": "woof",
        "packages": _declared_packages(),
        "package_data": config["package-data"],
        "exclude_package_data": config.get("exclude-package-data", {}),
    })
    command = build_py_class(distribution)
    command.finalize_options()
    monkeypatch.chdir(REPO_ROOT)
    pages = sorted(p.relative_to(REPO_ROOT).as_posix()
                   for p in (REPO_ROOT / "docs" / "public").glob("*.md"))
    assert pages, "docs/public holds no pages to measure with"
    command.manifest_files = {"docs": pages + ["docs/install.md"]}

    from_docs = sorted(Path(name).as_posix()
                       for name in command.find_data_files("docs", "docs"))
    assert from_docs == [SEAM_DOC], (
        "the sdist manifest widened the wheel's docs/ payload beyond the "
        f"seam contract document: {from_docs}")


def test_the_seam_document_is_the_only_file_the_docs_key_ships(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """One file, and the enumeration is what keeps it one file.

    A glob under the ``docs`` key would ship the whole documentation
    tree.  This measures setuptools' own selection rather than reading
    the pattern back.
    """

    _require_source_tree()

    shipped = _files_setuptools_would_ship(monkeypatch)
    from_docs = sorted(path for path in shipped
                       if path.split("/")[0] == "docs")
    assert from_docs == [SEAM_DOC], (
        "the wheel's docs/ payload is not exactly the seam contract "
        f"document: {from_docs}")


def test_the_seam_document_lands_where_a_manifest_keyed_on_it_looks(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """The install path equals the repository path, character for character.

    woof hex resolves ``<root>/docs/mpas-seam.md`` for both of its
    roots.  Under an install ``<root>`` is site-packages, and a
    top-level ``docs`` package puts the file at exactly
    ``<site-packages>/docs/mpas-seam.md``.  Moving it under ``gpuwm/``
    would still ship the bytes and would still break that consumer,
    which is why this is pinned by PATH and not by presence.
    """

    _require_source_tree()

    shipped = _files_setuptools_would_ship(monkeypatch)
    assert SEAM_DOC in shipped, (
        f"{SEAM_DOC} is in no wheel path, so an installed engine's seam "
        "check reports it absent and sends the reader to `git clone`")


def test_a_built_wheel_carries_the_seam_document(gpuwm_wheel: Path) -> None:
    """The artifact, opened.  Not the declaration that predicts it."""

    with zipfile.ZipFile(gpuwm_wheel) as wheel:
        names = set(wheel.namelist())
    assert SEAM_DOC in names, (
        f"{gpuwm_wheel.name} does not carry {SEAM_DOC}; it holds "
        + ", ".join(sorted(n for n in names if n.startswith("docs"))
                    or ["no docs/ member at all"]))
    with zipfile.ZipFile(gpuwm_wheel) as wheel:
        packed = wheel.read(SEAM_DOC)
    assert packed == (REPO_ROOT / SEAM_DOC).read_bytes(), (
        "the wheel's copy of the seam contract document is not the "
        "repository's bytes, so its sha256 is not the one a consumer pins")


@pytest.fixture(scope="module")
def gpuwm_wheel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the real wheel once, or skip saying why.

    Marked so a suite that cannot afford a build still runs the three
    declaration tests above: those measure setuptools' own selection and
    are the fast gate; this one is the artifact.
    """

    _require_source_tree()
    build = pytest.importorskip(
        "build", reason="`python -m build` is what makes the artifact this "
                        "test opens; without it there is nothing to open")
    del build

    import subprocess
    import sys

    out = tmp_path_factory.mktemp("wheel")
    result = subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(out),
         str(REPO_ROOT)],
        capture_output=True, text=True)
    if result.returncode != 0:
        pytest.skip("`python -m build --wheel` failed in this environment, "
                    "so there is no artifact to open:\n"
                    + (result.stderr or result.stdout)[-2000:])
    wheels = sorted(out.glob("gpuwm-*.whl"))
    assert len(wheels) == 1, wheels
    return wheels[0]
