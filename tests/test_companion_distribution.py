"""The ``recast-woof-data`` companion: same bytes, same version, named refusals.

One repository, two distributions.  ``woof`` measured 103.62 MiB against
PyPI's 100 MiB per-file cap once the Rust bridges were staged into it, and
the two pure-data directories below were 64.21 MiB of that, so they ship in
``recast-woof-data`` and ``woof`` pins it ``==``.

What this file gates, and the breakage each one prevents:

* **Byte identity.**  A packaging move that quietly re-encoded, truncated or
  substituted a table would change every radiation and microphysics answer
  while every other test still passed -- physics reads what it is handed.
  So a member of each moved tree is hashed THROUGH the resolver against a
  recorded digest, which also proves the resolver reaches the real files
  rather than an empty directory.
* **Version lockstep.**  The version string is restated in three places
  that pip and setuptools each read separately.  If they drift, `pip
  install woof` resolves to a companion that was never built with it, and
  the failure is numeric rather than loud.
* **The refusals.**  A missing or skewed companion must say so by name and
  end in the pip line that fixes it.  A bare ``FileNotFoundError`` deep in a
  NetCDF open names nothing.
* **The front doors.**  The refusal has to reach a READER, which means
  ``woof check`` and ``woof domain`` print it as a sentence at a nonzero
  exit with no traceback -- not a twenty-frame relay ending in the same
  words.  Measured in a real subprocess against an install-shaped tree,
  because that is the only shape where the checkout fallback is absent.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
import tomllib

import pytest

from woof import data_assets

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
COMPANION_ROOT = REPO_ROOT / "recast-woof-data"
COMPANION_PYPROJECT = COMPANION_ROOT / "pyproject.toml"
COMPANION_VERSION_FILE = COMPANION_ROOT / "woof_data" / "VERSION"

#: One moved member per moved tree, with the SHA-256 it had in the ``woof``
#: wheel before the split.  Recorded, not recomputed: a digest derived from
#: the same file it is checking proves nothing.
MOVED_MEMBER_SHA256 = {
    "rrtmgp/rrtmgp-gas-lw-g256.nc":
        "4048360199d1917ed8f2ccaae2ec097d0f990da3bbad9830337b739b4fa01be7",
    "rrtmgp/rrtmgp-clouds-sw-bnd.nc":
        "7671835992a45afe66244b591a02c0b3df73d7d59ecb746bbffd9763497651cd",
    "thompson/tables/thompson_aux_tables.dat":
        "a1bda803cdb53aedce8a2970c04c355fad19e3744398e1c9b13a876f09730547",
    "thompson/tables/CCN_ACTIVATE.BIN":
        "f2b8d3916560f9046f89f8ac5f32c5292a1800498fd75301e422f147c82a3dbd",
    # Moved in 2.8.7 as single files (their directories keep other members
    # in the woof wheel); the digests are the ones they had at e9d54cc17.
    "case-catalog/historical.zip":
        "39a2318da1608ac7e77aa71a946b950370bdb7c8888a7efbc1ca1f1629a20195",
    # Re-provisioned after the move by development branch: the same
    # captures rebuilt on the oracle platform (gfortran 13.3.0, glibc 2.39);
    # the moved bytes were 9bc14a88c46b5454... (gfortran 15.2.0, glibc 2.43).
    # 2.9.1: its GF captures re-run on the corrected GF harness (receipt.json
    # gf_rebuild); fb969d191daa1ede... was the archive before that.
    "spp/native-wrf461.zip":
        "201025994da48275c72fb06d37e38074c924780139a1aed8ed4409682b3a688e",
    # Moved in 2.9.0 as a single file, re-encoded on the way (xz of byte
    # planes of longitude differences; woof/core/rrtmg_sw.py
    # encode_aer1_climatology).  These are the re-encoded bytes; that the
    # DECODED words are the ones the 2.9.0 aer_opt=1 port was proven on is
    # rrtmg_aer1_manifest.json's arrays.climatology sha256, which the loader
    # checks on every load.
    "wrf_radiation/rrtmg_aer1_climatology.npz":
        "c48005c8f802c60f42824b6daa358dd13aec7c3d41200a68876ef7eaa3c04509",
    # Moved later in 2.8.7 as a directory, beside the Linux bridges' symbol
    # strip; the digests are the ones they had at e81291059.
    "chem/optics/table-sha256sums.txt":
        "d39e1c78ddd3465f0bf3a5e4b329e626be508ac21847ca94fa3a5a11f50486af",
    "chem/optics/extplw.bin":
        "239b29e9ffb68549a4cdab88db258ef288d8bd5a13f95397c94dc49a26d25e28",
}


def _require_source_tree() -> None:
    if not PYPROJECT.is_file() or not COMPANION_PYPROJECT.is_file():
        pytest.skip("companion gate needs the source tree, not an install")


def _declared_version() -> str:
    with PYPROJECT.open("rb") as stream:
        return tomllib.load(stream)["project"]["version"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@pytest.mark.parametrize("relative", sorted(MOVED_MEMBER_SHA256))
def test_a_moved_member_is_byte_identical_through_the_resolver(relative: str
                                                               ) -> None:
    """The bytes physics reads are the bytes that were in the old wheel.

    Read through ``data_assets.data_path`` -- the path every caller now
    takes -- rather than by joining onto the companion directory, so a
    resolver that pointed at the wrong root, or at a stale in-package copy
    left behind by the move, fails here instead of producing quietly
    different radiation.
    """

    path = data_assets.data_path(relative)
    assert path.is_file(), (
        f"{relative} does not resolve to a file ({path}); the companion is "
        "missing or woof.data_assets.COMPANION_TREES disagrees with where "
        "the files actually landed")
    assert _sha256(path) == MOVED_MEMBER_SHA256[relative], (
        f"{relative} resolved to {path} but its bytes are not the ones the "
        "woof wheel shipped before the companion split")


def test_every_moved_tree_is_covered_by_a_recorded_hash() -> None:
    """No tree may move out without a byte-identity witness.

    The parametrised test above is only as strong as its table.  Moving a
    third directory into ``COMPANION_TREES`` and forgetting to record a
    member would leave that directory's bytes unchecked, which is exactly
    the state the split was supposed to make impossible.
    """

    # A moved entry is covered when a recorded member is the entry itself
    # (a single moved file) or lies inside it (a moved directory).
    missing = sorted(
        entry for entry in data_assets.COMPANION_TREES
        if not any(relative == entry or relative.startswith(entry + "/")
                   for relative in MOVED_MEMBER_SHA256))
    assert not missing, (
        f"woof.data_assets.COMPANION_TREES moved {missing} into the "
        "companion with no recorded member hash; add one file from each to "
        "MOVED_MEMBER_SHA256")


def test_the_two_distributions_state_one_version() -> None:
    """pyproject version, companion VERSION file, and the `==` pin agree.

    Three restatements, one truth: ``[project].version`` in the root
    pyproject.toml.  It stays a static literal there because
    ``woof.provenance`` reads it straight out of the file for the
    code-version receipt, so this gate -- not a dynamic lookup -- is what
    keeps the other two from drifting.

    Drift is not cosmetic.  ``pip install recast-woof==2.5.0`` resolving a
    companion built for another release pairs versioned physics with
    versioned tables, and the result is numbers rather than an error.
    """

    _require_source_tree()
    declared = _declared_version()

    version_file = COMPANION_VERSION_FILE.read_text(encoding="utf-8").strip()
    assert version_file == declared, (
        f"recast-woof-data/woof_data/VERSION is {version_file!r} but "
        f"pyproject.toml's [project].version is {declared!r}")

    with PYPROJECT.open("rb") as stream:
        dependencies = tomllib.load(stream)["project"]["dependencies"]
    pins = [entry for entry in dependencies
            if entry.replace("_", "-").startswith(
                data_assets.COMPANION_DISTRIBUTION)]
    assert pins == [f"{data_assets.COMPANION_DISTRIBUTION}=={declared}"], (
        f"woof must depend on exactly "
        f"{data_assets.COMPANION_DISTRIBUTION}=={declared}; found {pins}")


def test_the_companion_declares_no_dependency_of_its_own() -> None:
    """It must not depend on woof, and it must not need anything else.

    A cycle would make either distribution unresolvable alone, and any
    other dependency would let a table become unreadable for a reason that
    has nothing to do with tables.
    """

    _require_source_tree()
    with COMPANION_PYPROJECT.open("rb") as stream:
        project = tomllib.load(stream)["project"]
    assert project["dependencies"] == []
    assert "dependencies" not in project.get("dynamic", [])


def test_a_missing_companion_refuses_by_name_with_the_pip_line(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The refusal names the breakage and ends in the command that fixes it.

    Both rungs are removed for this: the importable package and the source
    checkout beside ``gpuwm/``.  What must NOT happen is a bare
    ``FileNotFoundError`` from a NetCDF open twenty frames deeper, which is
    what a resolver without a refusal would produce and which names nothing
    a reader can act on.
    """

    def _no_package(_name):
        raise ModuleNotFoundError("No module named 'woof_data'")

    import importlib.resources as resources
    monkeypatch.setattr(resources, "files", _no_package)
    monkeypatch.setattr(data_assets, "_checkout_root", lambda: None)

    with pytest.raises(ModuleNotFoundError) as caught:
        data_assets.companion_root()
    message = str(caught.value)
    assert data_assets.COMPANION_DISTRIBUTION in message
    assert "REFUSES" in message
    assert "radiation" in message and "mp_physics=8/28" in message
    assert f"pip install {data_assets.COMPANION_DISTRIBUTION}==" in message


def test_a_version_mismatch_refuses_by_name_with_the_pip_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Skew is refused even though both halves are present and readable.

    This is the quiet one.  A companion from another release has every
    file, opens cleanly and returns arrays -- it is a different numerical
    setup wearing the right filenames, and no certification capsule can
    see it.  So the check is on metadata, not on presence.
    """

    # The check memoises a successful match, so an environment that has
    # already resolved a table would otherwise skip it and this test would
    # pass by not running.
    monkeypatch.setattr(data_assets, "_VERSION_CHECKED", False)
    monkeypatch.setattr(data_assets, "_required_companion_version",
                        lambda: "2.5.0")
    monkeypatch.setattr("importlib.metadata.version", lambda _name: "2.4.1")

    with pytest.raises(ImportError) as caught:
        data_assets._check_version()
    message = str(caught.value)
    assert "2.4.1" in message and "2.5.0" in message
    assert "REFUSES" in message
    assert f"pip install {data_assets.COMPANION_DISTRIBUTION}==2.5.0" in message


def test_an_installed_gpuwm_never_falls_back_to_a_stray_sibling(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The checkout rung is guarded on woof running from a real CHECKOUT.

    Without a working guard, an INSTALLED woof two directories below any
    folder named ``recast-woof-data`` reads unrelated bytes and says nothing --
    the one failure mode this module exists to make impossible.

    The site-packages built here is the real shape, ``woof/__init__.py``
    included, because that file is what an install and a checkout have in
    COMMON: a guard written on it is true in both and separates nothing.
    An earlier version of this test omitted it and passed vacuously while
    the fallback was live on every installed wheel.
    """

    site_packages = tmp_path / "site-packages"
    installed = site_packages / "woof"
    installed.mkdir(parents=True)
    (installed / "__init__.py").write_text("", encoding="utf-8", newline="\n")
    (site_packages / "recast-woof-data" / "woof_data" / "data").mkdir(parents=True)
    monkeypatch.setattr(data_assets, "__file__",
                        str(installed / "data_assets.py"))

    assert data_assets._checkout_root() is None


def test_the_checkout_rung_still_answers_inside_this_repository() -> None:
    """The other direction, or the guard is just a switch that is off.

    A checkout must keep reading its tables with no ``pip install -e
    recast-woof-data`` first -- CONTRIBUTING.md promises exactly that -- so the
    marker the guard is written on has to be present HERE, in a real
    working tree, and has to point at the real sibling directory.
    """

    _require_source_tree()
    found = data_assets._checkout_root()
    assert found is not None, (
        "the checkout fallback no longer answers inside the repository, so "
        "a contributor reading a table out of a working tree is refused")
    assert found == COMPANION_ROOT / data_assets.COMPANION_PACKAGE / "data"


# ---------------------------------------------------------------------------
# The front doors, in a real process, against an install-shaped tree
# ---------------------------------------------------------------------------
#
# Everything below runs the REAL CLI in a REAL subprocess.  Nothing here
# stubs a woof function: the only thing arranged is the environment, and
# it is arranged into the exact shape a `pip install recast-woof` followed by a
# `pip uninstall recast-woof-data` leaves behind.  That shape cannot be reached
# by running out of this repository -- a checkout always has the sibling
# `recast-woof-data/` directory and the fallback always answers -- which is why
# the fixture below exists and why the defect it catches survived every
# in-process test in this file.


#: Hides the companion from the child interpreter whatever the ambient
#: environment happens to have installed.  CONTRIBUTING.md tells
#: contributors to run `pip install -e recast-woof-data`, so on their box the
#: import rung WOULD answer and these gates would pass by not running --
#: the vacuous-green shape this project has already paid for once, when
#: five wheel-content assertions sat skipped on the assembly venvs.
_HIDE_COMPANION = '''\
import sys
from importlib.abc import MetaPathFinder


class _HideCompanion(MetaPathFinder):
    """Make woof_data unimportable, as an uninstall does."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "woof_data" or fullname.startswith("woof_data."):
            raise ModuleNotFoundError(
                "No module named %r" % (fullname,), name=fullname)
        return None


sys.meta_path.insert(0, _HideCompanion())
'''


def _link_directory(source: Path, link: Path) -> None:
    """Point ``link`` at ``source`` without copying 107 MiB of package.

    A symlink where the platform grants one, a directory JUNCTION on
    Windows where it does not -- junctions need no privilege, so this
    never has to skip and never has to copy the tables.
    """

    try:
        os.symlink(source, link, target_is_directory=True)
        return
    except OSError as error:
        detail = str(error)
    if os.name == "nt":
        made = subprocess.run(["cmd", "/c", "mklink", "/J",
                               str(link), str(source)],
                              capture_output=True, text=True)
        if made.returncode == 0:
            return
        detail = (made.stderr or made.stdout).strip() or detail
    raise AssertionError(
        f"cannot link {source} into the install-shaped tree ({detail}); "
        "the front-door gates in this file need one, and copying the "
        "package instead is 107 MiB per run")


def _link_file(source: Path, link: Path) -> None:
    """One member without the copy: hardlink where the volume allows it.

    Hardlinks need no privilege on NTFS and the tables are the whole cost
    -- ``rrtmgp-gas-sw-g224.nc`` alone is 10.6 MiB.  A cross-volume tmpdir
    falls back to a real copy rather than skipping.
    """

    try:
        os.link(source, link)
    except OSError:
        shutil.copy2(source, link)


def _installed_shaped_tree(destination: Path) -> Path:
    """``<site-packages>/gpuwm`` serving THIS working tree's code.

    An installed wheel and a source checkout differ in exactly one way
    that :mod:`woof.data_assets` can see: the directory ABOVE ``gpuwm/``
    is ``site-packages`` in one and the repository in the other.
    Everything else is identical -- ``woof/__init__.py``, ``woof/data``,
    every subpackage -- which is precisely why a guard written on
    ``woof/__init__.py`` alone cannot tell them apart.

    Top-level modules are COPIED, because ``Path.resolve()`` follows a
    link straight back to the repository and would hand the resolver the
    checkout it is supposed to be blind to.  Subdirectories are linked,
    so the subpackages and the packaged data are the real ones and this
    costs about five megabytes rather than a hundred.
    """

    package = destination / "woof"
    package.mkdir(parents=True)
    for entry in sorted((REPO_ROOT / "woof").iterdir()):
        if entry.name == "__pycache__":
            continue
        if entry.is_dir():
            _link_directory(entry, package / entry.name)
        else:
            shutil.copy2(entry, package / entry.name)
    # The installed engine also carries this sibling package.  Omitting
    # it makes host-memory sizing fail before the companion-data door is
    # reached, so the fixture would test an incomplete engine instead.
    for sibling in ("tilestream", "tools"):
        _link_directory(REPO_ROOT / sibling, destination / sibling)
    # Bind the copied package to its own wheel metadata, including the
    # exact companion requirement.  Borrowed checkout metadata does not
    # own these files and cannot supply the installed package's remedy.
    project = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]
    version = project["version"]
    info = destination / f"gpuwm-{version}.dist-info"
    info.mkdir()
    lines = ["Metadata-Version: 2.1", "Name: woof", f"Version: {version}"]
    lines += [f"Requires-Dist: {dep}" for dep in project["dependencies"]]
    (info / "METADATA").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (info / "RECORD").write_text("woof/__init__.py,,\n", encoding="utf-8")
    (destination / "sitecustomize.py").write_text(
        _HIDE_COMPANION, encoding="utf-8", newline="\n")
    return destination


def _run_front_door(root: Path, cwd: Path, *argv: str
                    ) -> subprocess.CompletedProcess:
    """One ``woof`` subcommand, in its own process, out of ``root``.

    ``cwd`` must not be the repository: ``python -m`` puts the working
    directory first on ``sys.path``, which would import this checkout's
    ``woof`` and restore the very fallback the fixture removed.
    """

    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root)
    # 3.13 colourises tracebacks, which puts SGR codes between the
    # indent and the word `File` and would hide every frame from the
    # count below.  Belt and braces: the count strips them as well.
    environment["PYTHON_COLORS"] = "0"
    return subprocess.run([sys.executable, "-m", "woof.cli", *argv],
                          cwd=str(cwd), env=environment,
                          capture_output=True, text=True, timeout=900)


#: SGR escape sequences, so a coloured frame still counts as a frame.
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _traceback_frames(text: str) -> list[str]:
    return [line for line in _ANSI.sub("", text).splitlines()
            if line.lstrip().startswith('File "')]


#: ``woof domain``'s argv, minus the ``--out`` a caller must place.
_DOMAIN_ARGV = ("domain", "--point=35.3,-97.5", "--card", "24gb",
                "--ladder", "12", "--source", "gfs",
                "--cycle", "2024-10-09T00", "--hours", "6")


@pytest.fixture(scope="module")
def install_shaped_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _installed_shaped_tree(tmp_path_factory.mktemp("site-packages"))


@pytest.fixture(scope="module")
def experiment_config(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A real experiment TOML, written by the real ``woof domain``.

    Generated rather than committed, and generated by the command whose
    output ``woof check`` names in its own refusal ("pass the experiment
    .toml that `woof domain` wrote"), so this fixture cannot drift away
    from the file the front door actually consumes.
    """

    out = tmp_path_factory.mktemp("case") / "case.toml"
    written = subprocess.run(
        [sys.executable, "-m", "woof.cli", *_DOMAIN_ARGV, "--out", str(out)],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=900)
    assert written.returncode == 0, (
        "the wizard could not write a config for the front-door gates:\n"
        + written.stdout + written.stderr)
    return out


def test_the_install_shaped_tree_really_hides_the_companion(
    tmp_path: Path,
) -> None:
    """Validate the instrument before trusting what it measures.

    Two properties, and a gate that assumed either one would report green
    from an environment that had neither: the companion must be
    unimportable in the child, and the tree above ``gpuwm/`` must not be
    a checkout.
    """

    root = _installed_shaped_tree(tmp_path / "site-packages")
    assert (root / "woof" / "__init__.py").is_file()
    assert not (root / "pyproject.toml").exists()
    assert not (root / "recast-woof-data").exists()

    probe = subprocess.run([sys.executable, "-c", "import woof_data"],
                           cwd=str(tmp_path), timeout=300,
                           env={**os.environ, "PYTHONPATH": str(root)},
                           capture_output=True, text=True)
    assert probe.returncode != 0, (
        "woof_data imported in the child, so the front-door gates below "
        "would be measuring an install that is not missing anything")
    assert "woof_data" in probe.stderr


@pytest.mark.parametrize("door,flags", [
    ("check", ()), ("check", ("--alloc",)),
    ("check", ("--budget-gib", "24")), ("check", ("--free-gib", "24")),
    ("domain", ()),
])
def test_a_front_door_without_the_companion_refuses_without_a_traceback(
    door: str, flags: tuple[str, ...], install_shaped_root: Path,
    experiment_config: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The breakage: `woof check` relayed a 20-frame traceback at exit 1.

    ``woof.core.rrtmgp`` resolves its data directory at MODULE level, so
    an install whose companion was uninstalled hits the refusal during an
    import, deep inside the preflight.  The refusal itself was already
    written and already named its remedy -- and none of it reached the
    reader as a refusal, because :func:`woof.cli.main`'s
    ``ModuleNotFoundError`` branch derives the remedy from
    ``ModuleNotFoundError.name`` and this one carried none.

    So both halves are asserted: a reader sees the sentence and the pip
    line, and a reader does NOT see a stack.
    """

    # Both gaps are real here: the companion is absent and local GPU use
    # is disabled. The cheap CPU dependency refusal must win in every mode.
    monkeypatch.setenv("GPUWM_NO_LOCAL_GPU", "1")
    argv = ((door, str(experiment_config), *flags) if door == "check"
            else (*_DOMAIN_ARGV, "--out", str(tmp_path / "case.toml")))
    result = _run_front_door(install_shaped_root, tmp_path, *argv)
    output = result.stdout + result.stderr

    assert result.returncode != 0, (
        f"woof {door} exited 0 without the companion:\n{output}")
    assert "woof GPU readiness:" not in output, output
    frames = _traceback_frames(output)
    assert "Traceback (most recent call last)" not in output and not frames, (
        f"woof {door} relayed a traceback ({len(frames)} frames) instead "
        f"of refusing:\n{output}")
    assert data_assets.COMPANION_DISTRIBUTION in result.stderr, (
        f"woof {door} refused without naming the companion:\n{output}")
    spoken = [line for line in result.stderr.splitlines() if line.strip()]
    if spoken[-1].strip().startswith("(run woof "):
        # Every layered refusal in this product ends with the `--explain`
        # pointer; it is the reader's way INTO the mechanism half and is
        # not part of the action half being asserted here.
        spoken.pop()
    assert spoken[-1].strip().startswith(
        f"remedy: {data_assets.companion_install_command()}"), (
        "the refusal must END in the command that fixes it; its last line "
        f"is {spoken[-1]!r}\n{output}")


def test_installed_check_with_companion_preserves_disabled_gpu_refusal(
    tmp_path: Path, experiment_config: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Restoring the dependency exposes the genuine GPU refusal unchanged."""
    root = _installed_shaped_tree(tmp_path / "site-packages")
    (root / "sitecustomize.py").unlink()
    _link_directory(COMPANION_ROOT / data_assets.COMPANION_PACKAGE,
                    root / data_assets.COMPANION_PACKAGE)
    monkeypatch.setenv("GPUWM_NO_LOCAL_GPU", "1")
    result = _run_front_door(root, tmp_path, "check", str(experiment_config))
    output = result.stdout + result.stderr
    assert result.returncode == 2, output
    assert "woof GPU readiness: UNVERIFIED. device not touched." in output
    assert "woof doctor --explain" in output
    assert "ERA5 native-GRIB1 route only" not in output
    assert "Traceback" not in output and not _traceback_frames(output)
    assert "pip install recast-woof-data" not in output


def test_a_front_door_with_an_incomplete_companion_refuses_without_a_traceback(
    tmp_path: Path,
) -> None:
    """Present but incomplete is the THIRD companion state, and it is real.

    The first public-tree CI run produced it: a repository-wide ``*.nc``
    gitignore rule swallowed every ``data/rrtmgp/*.nc`` when the release
    snapshot was ``git add``-ed, so the companion wheel built from that
    checkout was importable, versioned correctly, carried a ``data/``
    directory -- and had no k-distributions in it.  ``woof domain`` died
    with a bare ``FileNotFoundError`` out of a NetCDF open deep in the
    preflight's radiation sizing: a traceback that names no distribution
    and no remedy, which is exactly the shape the module docstring above
    promises cannot happen.

    The missing-companion and decoy tests cannot catch it: both remove or
    replace the whole package, and ``companion_root()``'s directory-level
    check passes an incomplete tree.  So the state is arranged exactly --
    a real companion beside the install with ONE member absent -- and the
    door must refuse with the member's name and the pip line, tracebackless.
    """

    root = _installed_shaped_tree(tmp_path / "site-packages")
    # The companion must IMPORT here -- incomplete, not hidden.
    (root / "sitecustomize.py").unlink()
    source = COMPANION_ROOT / data_assets.COMPANION_PACKAGE
    package = root / data_assets.COMPANION_PACKAGE
    rrtmgp = package / "data" / "rrtmgp"
    rrtmgp.mkdir(parents=True)
    shutil.copy2(source / "__init__.py", package / "__init__.py")
    shutil.copy2(source / "VERSION", package / "VERSION")
    absent = "rrtmgp-gas-lw-g256.nc"
    for entry in sorted((source / "data" / "rrtmgp").iterdir()):
        if entry.name != absent:
            _link_file(entry, rrtmgp / entry.name)
    _link_directory(source / "data" / "thompson",
                    package / "data" / "thompson")

    result = _run_front_door(root, tmp_path, *_DOMAIN_ARGV,
                             "--out", str(tmp_path / "case.toml"))
    output = result.stdout + result.stderr

    assert result.returncode != 0, (
        "woof domain exited 0 against a companion missing "
        f"{absent}:\n{output}")
    frames = _traceback_frames(output)
    assert "Traceback (most recent call last)" not in output and not frames, (
        f"woof domain relayed a traceback ({len(frames)} frames) instead "
        f"of refusing the incomplete companion:\n{output}")
    assert data_assets.COMPANION_DISTRIBUTION in result.stderr, (
        f"the refusal does not name the companion:\n{output}")
    assert absent in result.stderr, (
        f"the refusal does not name the absent member:\n{output}")
    assert "REFUS" in result.stderr.upper(), output
    assert "pip install" in result.stderr, (
        f"the refusal does not end in the command that fixes it:\n{output}")


def test_an_installed_wheel_refuses_a_decoy_companion_beside_it(
    tmp_path: Path,
) -> None:
    """The decoy, driven end to end through the real front door.

    This is the whole failure the guard exists to prevent, arranged
    exactly as it occurs: a wheel install, the companion uninstalled, and
    a directory named ``recast-woof-data`` sitting beside ``woof`` in
    site-packages -- left behind by a partial uninstall, unpacked there by
    hand, or (until the packaging fix one commit back) written there by
    ``pip install recast-woof`` itself.

    Reading it is worse than failing to: the version check is skipped
    entirely on that rung, so bytes of unknown provenance become the
    k-distribution for every radiation call in the run, and the receipt
    records a clean pass.  The front door must REFUSE and name the
    companion instead.
    """

    root = _installed_shaped_tree(tmp_path / "site-packages")
    decoy = root / "recast-woof-data" / data_assets.COMPANION_PACKAGE / "data"
    (decoy / "rrtmgp").mkdir(parents=True)
    (decoy / "rrtmgp" / "rrtmgp-gas-lw-g256.nc").write_bytes(
        b"not a k-distribution")

    result = _run_front_door(root, tmp_path, *_DOMAIN_ARGV,
                             "--out", str(tmp_path / "case.toml"))
    output = result.stdout + result.stderr

    assert result.returncode != 0, (
        "woof domain succeeded with a decoy recast-woof-data beside the "
        f"install, so it read the decoy's bytes as reference data:\n{output}")
    assert not _traceback_frames(output), (
        f"the decoy produced a traceback rather than a refusal:\n{output}")
    assert data_assets.COMPANION_DISTRIBUTION in result.stderr, (
        "the failure did not name the companion, so the decoy was read "
        f"and something downstream of it broke instead:\n{output}")
    assert "REFUS" in result.stderr.upper(), output


# ---------------------------------------------------------------------------
# The lock compares PUBLIC versions, not full version strings
# ---------------------------------------------------------------------------
#
# The breakage, reproduced against the published 2.6.1 pair: a wheel built
# from this tree with a PEP 440 local label -- `2.6.1+<label>`, which is what
# every private rebuild of a published release carries -- refused the public
# `recast-woof-data 2.6.1` it was cut against, because the comparison was on the
# whole string.  The refusal then printed `pip install recast-woof-data==2.6.1+
# <label>`, a line that resolves to nothing, so the remedy could not even be
# run.  The label says WHO built the wheel; it does not change which release's
# tables the physics needs.  A different public version is still refused --
# that is the skew the lock exists for and it must survive this.


def test_a_local_label_accepts_the_companion_of_its_public_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`2.6.1+<label>` is a private build of 2.6.1 and takes 2.6.1's tables."""

    monkeypatch.setattr(data_assets, "_VERSION_CHECKED", False)
    monkeypatch.setattr(data_assets, "_required_companion_version",
                        lambda: "2.6.1")
    monkeypatch.setattr("importlib.metadata.version", lambda _name: "2.6.1")

    data_assets._check_version()  # must not raise
    assert data_assets._VERSION_CHECKED


def test_a_locally_labelled_companion_is_accepted_too(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The strip is symmetric: a rebuilt COMPANION is the same data cut.

    Same reasoning as the wheel side.  A companion rebuilt from the release
    commit carries the release's bytes whatever label the builder stamped on
    it, so refusing it would name no breakage.
    """

    monkeypatch.setattr(data_assets, "_VERSION_CHECKED", False)
    monkeypatch.setattr(data_assets, "_required_companion_version",
                        lambda: "2.6.1")
    monkeypatch.setattr("importlib.metadata.version",
                        lambda _name: "2.6.1+rebuild.4")

    data_assets._check_version()
    assert data_assets._VERSION_CHECKED


def test_a_different_public_version_is_still_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The refusal the lock exists for survives the local-label strip."""

    monkeypatch.setattr(data_assets, "_VERSION_CHECKED", False)
    monkeypatch.setattr(data_assets, "_required_companion_version",
                        lambda: "2.6.1")
    monkeypatch.setattr("importlib.metadata.version",
                        lambda _name: "2.6.0+rebuild.4")

    with pytest.raises(ImportError) as caught:
        data_assets._check_version()
    message = str(caught.value)
    assert "REFUSES" in message
    assert "2.6.0" in message and "2.6.1" in message
    assert f"pip install {data_assets.COMPANION_DISTRIBUTION}==2.6.1" in message


def test_the_remedy_names_a_version_that_was_published(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`pip install recast-woof-data==2.6.1+<label>` resolves to nothing.

    The companion is only ever published under a public version, so a remedy
    carrying the wheel's local label is a command that cannot succeed -- the
    same reasoning the module already applies to an uninstalled source tree,
    where it drops the `==` pin rather than print `==0+unknown`.
    """

    import woof

    monkeypatch.setattr(woof, "__version__", "2.6.1+label.1")
    assert data_assets.companion_install_command() == (
        f"pip install {data_assets.COMPANION_DISTRIBUTION}==2.6.1")


def test_the_sentinel_for_an_uninstalled_tree_is_not_stripped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`0+unknown` is a sentinel, not a labelled version.

    It is compared by identity at three call sites; stripping it to `0` would
    make every one of them miss and turn "woof is an uninstalled source tree"
    into "gpuwm 0 requires recast-woof-data 0".
    """

    import woof

    monkeypatch.setattr(woof, "__version__", data_assets._UNKNOWN_VERSION)
    assert (data_assets._required_companion_version()
            == data_assets._UNKNOWN_VERSION)
    assert data_assets.companion_install_command() == (
        f"pip install {data_assets.COMPANION_DISTRIBUTION}")


def test_public_version_keeps_the_prerelease_segment() -> None:
    """`.public`, not `.base_version`, and this is the difference.

    `2.7.0rc1` and `2.7.0` are two different cuts that ship two different
    companions.  `.base_version` collapses them, which would accept exactly
    the pairing this lock exists to refuse.
    """

    assert data_assets._public_version("2.6.1+label.1") == "2.6.1"
    assert data_assets._public_version("2.6.1") == "2.6.1"
    assert data_assets._public_version("2.7.0rc1+label.1") == "2.7.0rc1"
    assert data_assets._public_version("2.7.0rc1") != "2.7.0"
