"""One resolver for every packaged reference-data path under ``woof/data``.

Two directories of that tree do not ship inside the ``woof`` wheel any
more.  They ship in the ``recast-woof-data`` distribution, which ``woof``
declares as a hard dependency at its own exact version, and this module is
the only place that knows which is which.

Why
---
PyPI rejects any file over 100 MiB.  The 2.5.0 ``woof`` wheel measured
103.62 MiB (108,649,669 bytes) with the Rust bridge artifacts staged.  The
RRTMGP directory and the Thompson table directory were 64.21 MiB of that
compressed -- ``qr_acr_qsV2.dat`` at 29.79 MiB and
``rrtmgp-gas-lw-g256.nc`` at 19.69 MiB alone are half the wheel -- and
they are pure data: no code, no platform coupling, nothing a wheel tag
describes.  Moving them puts ``woof`` at 39.41 MiB and the companion at
64.31 MiB, both single ``py3-none-any``-eligible files well under the cap,
with room for the native artifacts still landing on the 2.5.0 line.

Nothing else about them changed.  The companion mirrors the old layout
exactly (``woof_data/data/rrtmgp/...`` for ``woof/data/rrtmgp/...``), so
the same bytes reach the same call sites at the same relative path.
``tests/test_companion_distribution.py`` hashes a moved member through
this resolver against a recorded SHA-256 to keep that literal.

The 2.8.7 line crossed the cap again: three new native executables put
the manylinux wheel at about 105 MB.  Two single archives moved out with
them -- the bundled historical case catalog (2.73 MB) and the WRF v4.6.1
SPP oracle captures (2.51 MB), both incompressible ZIPs -- beside the
kernel-set trim in ``rw-post``.  The air-quality line then put it at
106.4 MB, and the WRF-Chem Mie tables (``chem/optics``, 1.00 MB
compressed) moved out beside the Linux bridges' symbol strip.

The shape of the rule
---------------------
:data:`COMPANION_TREES` names DIRECTORIES wherever a whole directory
moves.  An enumeration of files would drift behind the tree the way
``package-data`` once drifted 52 files behind it; a directory rule cannot,
because a table added beside its siblings is already covered.  Moving the
next directory out is one entry in that tuple plus the ``git mv`` -- no
new call site, no new code path.  A single FILE is named only when its
directory keeps other members in this wheel (the case-catalog schema and
examples, the SPP ULP receipts), and the same prefix rule resolves it.

The refusals
------------
Both name a concrete breakage and both end in the command that fixes it:

* companion missing -- every radiation scheme and every Thompson run
  cannot read its tables, which on a bare ``pip install recast-woof`` should be
  impossible (it is a hard dependency), so the case this catches is an
  install someone edited: ``pip uninstall recast-woof-data``, a partially
  restored venv, a vendored tree copied without it.
* version skew -- the two distributions are cut from one commit and
  pinned ``==``, so a mismatch means a hand-installed sibling.  The tables
  are versioned data: reading 2.4.1's RRTMGP k-distribution under 2.5.0's
  radiation code is a silently different numerical setup, not a missing
  file, and that is exactly the class of error a certification capsule
  cannot see.
"""

from __future__ import annotations

from pathlib import Path

#: PyPI name of the companion distribution, spelled once.
COMPANION_DISTRIBUTION = "recast-woof-data"

#: Import name of the package inside it.
COMPANION_PACKAGE = "woof_data"

#: ``woof/data``-relative directories the companion owns, as POSIX-style
#: relative paths.  Everything else under ``woof/data`` still ships
#: inside the ``woof`` wheel and is resolved by :func:`package_data_root`.
#:
#: The two directories are bulk reference tables with a single loader each --
#: ``woof.core.rrtmgp.DATA_DIR`` and
#: ``woof.physics_compat.packaged_thompson_table_root`` -- which is why
#: they were chosen over the same number of megabytes spread across the
#: oracle directories: the split had to be measurable in the wheel and
#: invisible everywhere else.  The two files each have one reader too:
#: ``woof.case_catalog.builtin_catalog_path`` and the SPP consumer gate
#: (tests/test_spp_consumers.py).  ``chem/optics`` has one loader,
#: ``woof.core.chem_optics.data_dir``, and its parity gate
#: (tests/test_chem_optics_wrf471_parity.py).
COMPANION_TREES: tuple[str, ...] = (
    "rrtmgp",
    "thompson/tables",
    "case-catalog/historical.zip",
    "spp/native-wrf461.zip",
    "chem/optics",
    # 2.9.0: the aer_opt=1 aerosol climatology (6.0 MB) put the platform
    # wheel at 101,425,913 bytes, over PyPI's per-file limit.  Its
    # directory keeps the other WRF radiation tables in this wheel.
    "wrf_radiation/rrtmg_aer1_climatology.npz",
)

#: In-package data root: ``<site-packages>/woof/data``.
_PACKAGE_DATA_ROOT = Path(__file__).resolve().parent / "data"

#: What ``woof.__version__`` reports when no distribution provides the
#: code that is running -- a source tree nobody installed.
_UNKNOWN_VERSION = "0+unknown"


def package_data_root() -> Path:
    """The ``woof/data`` directory that still ships inside this wheel."""

    return _PACKAGE_DATA_ROOT


def _public_version(version: str) -> str:
    """``version`` with any PEP 440 local label (``+...``) removed.

    ``.public`` and deliberately NOT ``.base_version``.  The local label
    is the only part a private rebuild of a published release adds, and
    dropping it is the whole job here; ``.base_version`` would also drop
    the pre-release segment, collapsing ``2.7.0rc1`` into ``2.7.0``.
    Those are two different cuts that ship two different companions, so
    accepting one for the other is exactly the skew this lock exists to
    refuse -- the strip has to be narrower than that.

    Guarded the way :mod:`woof.version_cli` guards its comparisons:
    ``packaging`` is not a declared dependency, and without it the
    partition on ``+`` gives the same answer for every version string
    PEP 440 admits, because a local label is the only thing ``+`` can
    introduce.
    """

    try:
        from packaging.version import InvalidVersion, Version
    except ImportError:
        return version.partition("+")[0]
    try:
        return Version(version).public
    except InvalidVersion:
        return version.partition("+")[0]


def _required_companion_version() -> str:
    """The PUBLIC version the installed ``woof`` was built against.

    Read from distribution metadata rather than restated, because the two
    are cut from one commit at one version and ``woof`` pins
    ``recast-woof-data==`` that exact string.

    PUBLIC, because the companion is cut per public release and only
    ever published under one.  A PEP 440 local label (``2.6.1+<label>``)
    marks a private rebuild of that SAME release: it reads the same
    tables, and no companion wearing that label exists to install.
    Comparing the full string refused the correct companion from every
    locally-built wheel and then printed a remedy -- ``pip install
    recast-woof-data==2.6.1+<label>`` -- that resolves to nothing.

    ``_UNKNOWN_VERSION`` is returned untouched: it is a sentinel the
    callers below compare against by identity, and its own ``+unknown``
    is not a version label to strip.
    """

    from woof import __version__

    if __version__ == _UNKNOWN_VERSION:
        return __version__
    return _public_version(__version__)


def companion_install_command() -> str:
    """The one pip line that fixes a missing or skewed companion.

    Spelled ONCE, and consumed by both refusals in this module and by
    :data:`woof.capabilities.COMPANION_DATA` -- so the sentence a
    reader meets at a front door and the sentence a table load raises
    cannot drift into naming different commands for the same gap.  Same
    discipline as :mod:`woof.static.geog_stack`, which owns the
    geography stack's remedy for the same reason.

    Version-less when ``woof`` itself is an uninstalled source tree:
    ``pip install recast-woof-data==0+unknown`` is a line that cannot be
    typed, and a remedy nobody can run is not a remedy.
    """

    required = _required_companion_version()
    if required == _UNKNOWN_VERSION:
        return f"pip install {COMPANION_DISTRIBUTION}"
    return f"pip install {COMPANION_DISTRIBUTION}=={required}"


def _refuse_missing(detail: str) -> "ModuleNotFoundError":
    """The refusal, carrying the MODULE that is missing in ``.name``.

    ``name=`` is not decoration.  :func:`woof.cli.main` answers a
    ``ModuleNotFoundError`` by looking ``ModuleNotFoundError.name`` up in
    :data:`woof.capabilities.REQUIREMENTS` and printing a refusal for
    what it finds; a raise without ``name`` resolves to nothing, falls
    through the branch and re-raises.  Measured, on an install-shaped
    tree with the companion uninstalled: `woof check` ended in 15
    traceback frames at exit 1 and `woof domain` in 17, both with this
    carefully-worded message as the last line of the stack.  The words
    were already right; nothing was reading them as a refusal.
    """

    if _required_companion_version() == _UNKNOWN_VERSION:
        # woof is a source tree nobody installed, so the companion's
        # absence is a missing sibling directory rather than an edited
        # install, and the remedy says so.
        return ModuleNotFoundError(
            f"woof REFUSES to resolve packaged reference data: the "
            f"{COMPANION_DISTRIBUTION} distribution is not importable "
            f"({detail}), and this woof is an uninstalled source tree "
            f"with no sibling recast-woof-data/ directory beside it.  A "
            f"checkout carries both; this one is incomplete.  Either "
            f"restore recast-woof-data/ from the repository, or install the "
            f"companion:\n    {companion_install_command()}",
            name=COMPANION_PACKAGE)
    return ModuleNotFoundError(
        f"woof REFUSES to resolve packaged reference data: the "
        f"{COMPANION_DISTRIBUTION} distribution is not importable "
        f"({detail}).  It carries the RRTMGP k-distribution and "
        f"cloud-optics tables, the Thompson microphysics lookup "
        f"tables, the WRF-Chem aerosol optics tables and the bundled "
        f"historical case catalog -- so without "
        f"it every radiation scheme and every "
        f"mp_physics=8/28 run fails at table load, and `woof check` "
        f"cannot complete a preflight.  It is a HARD dependency of "
        f"woof and a plain `pip install recast-woof` installs it; this state "
        f"means the install was edited afterwards.  Fix it:\n"
        f"    {companion_install_command()}",
        name=COMPANION_PACKAGE)


#: Where the companion's sources sit inside a source checkout, relative to
#: the repository root.  One repository, two distributions.
_CHECKOUT_RELATIVE = ("recast-woof-data", COMPANION_PACKAGE, "data")


#: The marker that says "the directory above ``gpuwm/`` is THIS project's
#: repository root".  It is the root ``pyproject.toml`` with
#: ``[project] name = "woof"`` in it, and the property that makes it work
#: is that no wheel of either distribution can place such a file there:
#: setuptools ships package data, the file sits at the project root, and
#: the project root is not a package of either distribution.  Measured on
#: both built wheels -- neither has a top-level ``pyproject.toml`` member --
#: and gated in tests/test_package_data_coverage.py, which pins that no
#: declared package is rooted at the repository root.
_CHECKOUT_MARKER = "pyproject.toml"


def _is_checkout_root(candidate: Path) -> bool:
    """Whether ``candidate`` is this project's repository root.

    Deliberately not "does a pyproject.toml exist": ANY project's root
    has one of those, and an install unpacked inside somebody else's
    source tree would answer yes.  The name is what is checked.
    """

    import tomllib

    marker = candidate / _CHECKOUT_MARKER
    try:
        with marker.open("rb") as stream:
            document = tomllib.load(stream)
    except (OSError, ValueError):
        # Absent, unreadable, or not TOML.  Not a checkout of this
        # project, and a malformed file is not a reason to start
        # trusting a directory.
        return False
    project = document.get("project")
    return isinstance(project, dict) and project.get("name") == "recast-woof"


def _checkout_root() -> Path | None:
    """The companion's sources beside a checkout's own ``gpuwm/``, or None.

    The second rung, and the same shape as
    :func:`woof.bridges.artifact_candidates`: a checkout's own copy
    outranks nothing and answers only when the installed package cannot,
    so a working tree needs no ``pip install -e`` of the sibling before
    ``python -m woof.cli`` can read a table.

    Guarded on the directory above ``gpuwm/`` being THIS PROJECT'S
    REPOSITORY, which is a different question from the one this guard
    used to ask.  It tested ``(repo_root / "woof" / "__init__.py")``,
    and that file is exactly what an install and a checkout have in
    common: inside site-packages the test is TRUE, so the fallback was
    live on every installed wheel.  Reproduced end to end -- an
    install-shaped tree with a `recast-woof-data/woof_data/data/rrtmgp`
    directory beside it, `woof domain` opened the decoy's bytes as the
    k-distribution -- and the version check cannot catch it, because
    this rung returns before :func:`_check_version` is ever reached.  So
    unlabelled bytes become the numerical setup of the run and the
    receipt records a clean pass, which is the exact failure this module
    exists to make impossible.

    The marker is the root ``pyproject.toml`` naming ``woof``.  It is in
    every checkout by construction -- it is what builds this project --
    and it is in no wheel of either distribution, because the project
    root is not a package and only package data ships.
    """

    repo_root = Path(__file__).resolve().parent.parent
    if not _is_checkout_root(repo_root):
        return None                    # installed: no checkout to fall to
    candidate = repo_root.joinpath(*_CHECKOUT_RELATIVE)
    return candidate if candidate.is_dir() else None


def companion_root() -> Path:
    """Directory the companion lays its ``woof/data`` mirror out under.

    Resolved with :mod:`importlib.resources` against the installed
    package -- not by walking up from ``gpuwm/``, which would find
    nothing in a wheel install, and not by an environment variable, which
    would make "which tables did this run read" unanswerable from the
    capsule.  A source checkout falls to its own sibling directory; see
    :func:`_checkout_root` for why that cannot fire on an install.
    """

    import importlib.resources as resources

    detail: str
    try:
        anchor = resources.files(COMPANION_PACKAGE)
    except ModuleNotFoundError as error:
        detail = str(error)
    else:
        root = Path(str(anchor)) / "data"
        if root.is_dir():
            _check_version()
            return root
        detail = (f"{COMPANION_PACKAGE} imported from {anchor} but "
                  f"carries no data/ directory")
    checkout = _checkout_root()
    if checkout is not None:
        return checkout
    raise _refuse_missing(detail)


#: Set once the installed pair has been checked.  Every table load asks
#: for a directory, and an ``importlib.metadata`` lookup walks
#: site-packages; the versions of two installed distributions cannot
#: change inside one process, so asking twice buys nothing.
_VERSION_CHECKED = False


def _check_version() -> None:
    """Refuse a companion whose version is not this ``woof``'s."""

    from importlib.metadata import PackageNotFoundError, version

    global _VERSION_CHECKED
    if _VERSION_CHECKED:
        return
    required = _required_companion_version()
    if required == _UNKNOWN_VERSION:
        # woof itself is being read out of a source tree that was never
        # installed.  There is no version to match against, so there is
        # no skew to detect and a refusal here could name no breakage.
        return
    try:
        found = version(COMPANION_DISTRIBUTION)
    except PackageNotFoundError:
        # Importable but not installed: a source checkout on sys.path.
        # Same reasoning as above -- nothing to compare.
        return
    # Both sides stripped to their public version: the comparison asks
    # "were these two cut from the same release", and a local label on
    # either half does not change that answer.
    if _public_version(found) == required:
        _VERSION_CHECKED = True
        return
    from woof import __version__ as installed
    raise ImportError(
        f"woof REFUSES to read packaged reference data from a "
        f"mismatched companion: woof {installed} requires "
        f"{COMPANION_DISTRIBUTION} {required}, found {found}.  The two "
        f"are cut from one commit and pinned `==`, so this install was "
        f"edited.  The tables are versioned data, not interchangeable "
        f"files: running {required}'s radiation and microphysics against "
        f"{found}'s k-distribution and lookup tables is a different "
        f"numerical setup that produces numbers instead of an error, and "
        f"no certification capsule can see it.  Fix it:\n"
        f"    {companion_install_command()}")


class CompanionDataMissing(FileNotFoundError):
    """A companion that imports, version-checks, and lost a member.

    The third companion state, beside "missing" and "skewed", and it is
    the one the other two refusals cannot see: :func:`companion_root`'s
    check is directory-level, so an importable package carrying a
    ``data/`` directory answers even when a member a run needs is not in
    it.  Real, not hypothetical: the first public-tree CI build shipped
    a companion wheel with no ``rrtmgp/*.nc`` in it (a repository-wide
    ``*.nc`` gitignore rule swallowed them when the release snapshot was
    staged), and every front door died in a bare ``FileNotFoundError``
    out of a NetCDF open mid-preflight.

    ``FileNotFoundError`` so that ``errno``-minded callers keep working;
    its own class so :func:`woof.cli.main` can print it as the refusal
    it is instead of relaying fifteen frames.
    """


def require_companion_member(relative: str | Path) -> Path:
    """Resolve a companion-owned path and refuse BY NAME if it is absent.

    The named refusal for the state :class:`CompanionDataMissing`
    documents.  Loaders that open companion members directly call this
    instead of joining onto a directory, so an incomplete companion is a
    sentence with the member's name and the pip line -- at the front
    door, before any traceback -- rather than a NetCDF open error.
    """

    posix = str(relative).replace("\\", "/").strip("/")
    path = data_path(posix)
    if path.is_file():
        return path
    if _required_companion_version() == _UNKNOWN_VERSION:
        # An uninstalled source tree: the sibling checkout directory is
        # answering, and the file is gone from it.
        raise CompanionDataMissing(
            f"woof REFUSES to read packaged reference data: {posix} is "
            f"absent from the {COMPANION_DISTRIBUTION} data tree at "
            f"{path.parent}.  This woof is an uninstalled source tree "
            f"answering out of its sibling recast-woof-data/ directory, and a "
            f"checkout always carries this member -- this one lost it.  "
            f"Restore the file from the repository, or install the "
            f"companion:\n    {companion_install_command()}")
    remedy = companion_reinstall_command()
    raise CompanionDataMissing(
        f"woof REFUSES to read packaged reference data: {posix} is "
        f"absent from the {COMPANION_DISTRIBUTION} data tree at "
        f"{path.parent}.  The companion is importable and its version "
        f"matches, but this member is not in it, so the scheme that "
        f"reads it fails at table load -- an intact "
        f"{COMPANION_DISTRIBUTION} always carries it, which means this "
        f"install was edited or its wheel was built from an incomplete "
        f"tree.  Fix it:\n    {remedy}")


def _is_companion(relative: str) -> bool:
    posix = relative.replace("\\", "/").strip("/")
    return any(posix == tree or posix.startswith(tree + "/")
               for tree in COMPANION_TREES)


def data_path(relative: str | Path) -> Path:
    """Resolve a ``woof/data``-relative path to where its bytes live.

    ``data_path("rrtmgp/rrtmgp-gas-lw-g256.nc")`` answers out of the
    companion; ``data_path("noah_tables/VEGPARM.TBL")`` answers out of
    this wheel.  Callers state the path they always stated and never
    which distribution carries it -- that is the whole point of routing
    both roots through one function.
    """

    posix = str(relative).replace("\\", "/").strip("/")
    if _is_verify_oracle(posix):
        return verify_oracle_path(posix)
    root = companion_root() if _is_companion(posix) else _PACKAGE_DATA_ROOT
    return root.joinpath(*posix.split("/")) if posix else root


#: The two directories, resolved.  Named because they are what the
#: loaders ask for, and a loader asking for a directory should not have
#: to know the spelling of the tree that holds it.
def rrtmgp_data_dir() -> Path:
    """``woof/data/rrtmgp`` as it now resolves (companion)."""

    return data_path("rrtmgp")


def thompson_table_dir() -> Path:
    """``woof/data/thompson/tables`` as it now resolves (companion)."""

    return data_path("thompson/tables")


#: The companion directory holding the renderer's map assets, relative to
#: :func:`companion_root`.  Not a :data:`COMPANION_TREES` entry: those are
#: trees that MOVED out of ``woof/data``, and these shapefiles never lived
#: there.  Their copy of record is ``tools/rustwx/assets/basemap`` in the
#: source repository; the companion carries its three layer directories so
#: that a wheel install draws coastlines, borders and state lines with no
#: step after ``pip install``.
COMPANION_BASEMAP = "basemap"


def companion_basemap_dir() -> Path | None:
    """The map assets the companion carries, or ``None`` when it has none.

    Never raises.  A missing or mismatched companion is refused by name
    wherever a physics table is read; a map is not a table load, and the
    picture must still be drawn, so this answers ``None`` and the caller
    says what the picture lacks (``woof.render.missing_basemap_notice``).
    """

    try:
        root = companion_root()
    except ImportError:          # absent (ModuleNotFoundError) or skewed
        return None
    candidate = root / COMPANION_BASEMAP
    return candidate if candidate.is_dir() else None


def companion_reinstall_command() -> str:
    """The pip line that restores a companion's files in place.

    ``--force-reinstall`` because the state it answers is a companion that
    imports and lost files (or an older one without the map assets); a
    plain install of a version already present does nothing.
    """

    return companion_install_command().replace(
        "pip install", "pip install --force-reinstall", 1)



# ---------------------------------------------------------------------------
# The verification oracles (3.0.0)
# ---------------------------------------------------------------------------
#
# The native-WRF captures the GPU ports are graded against (the CSV, binary
# and archive fixtures under ``woof/data/<scheme>/oracle`` and its
# siblings) are verification data: ``woof.verify`` reads them, the
# forecast path never does.  The 3.0.0 manylinux wheel measured
# 109,246,370 bytes against the release's 100,000,000-byte per-file gate,
# and these trees were 12.0 MB of it compressed.  The ``recast-woof-data``
# companion had 1.37 MB of room left (98,629,793 bytes), so they ship in
# the ``gpuwm-kernels`` companion instead: a hard ``==`` dependency like
# ``recast-woof-data``, pip-installable and usable offline, with 43.7 MB of room.
#
# The source tree keeps its copy where it always was; the woof wheel
# leaves these trees out (``[tool.setuptools.exclude-package-data]``) and
# the gpuwm-kernels build copies them in from the same commit
# (``gpuwm-kernels/setup.py``), with a SHA-256 manifest.  A checkout reads
# its own copy; an install reads the companion's.
#
# Two oracle directories stay in this wheel because the FORECAST reads a
# member of each at run time: ``ruc/oracle`` (``tbq.csv``, the RUC
# saturation table, ``woof.core.ruc._load_ruc_saturation_table``) and
# ``urban/oracle`` (``ucm480/schema.json``, ``woof.core.urban_ucm``).

#: PyPI name and import name of the companion that carries them.
VERIFY_ORACLE_DISTRIBUTION = "gpuwm-kernels"
VERIFY_ORACLE_PACKAGE = "gpuwm_kernels"

#: Directory inside that package mirroring ``woof/data`` for these trees.
VERIFY_ORACLE_SUBDIR = "verify_oracles"

#: ``woof/data``-relative trees that ship in the companion, not this wheel.
#: Kept in step with the root pyproject.toml's exclude-package-data entry and
#: gpuwm-kernels/setup.py by tests/test_package_data_coverage.py.
VERIFY_ORACLE_TREES: tuple[str, ...] = (
    "fogdes/oracle",
    "gbmpbl/oracle",
    "gecros/oracle",
    "gfs_pbl/oracle",
    "keps/oracle",
    "lake/oracle",
    "mrf/oracle",
    "mynn/oracle",
    "noah/oracle",
    "noah/seaice_oracle",
    "noah/ua_oracle",
    "nssl2/driver-oracle",
    "nssl2/fused-gs-oracle",
    "nssl2/oracle",
    "pxlsm/oracle",
    "pxsfclay/oracle",
    "ruc/oracle_fork",
    "ruc/spp_oracle",
    "sbu_ylin/oracle",
    "shinhong/oracle",
    "terrain_drag/oracle",
    "thompson/oracle",
    "thompson/oracle-aero",
    "thompson/oracle-hail",
    "ysu/oracle",
)


def _is_verify_oracle(relative: str) -> bool:
    posix = relative.replace("\\", "/").strip("/")
    return any(posix == tree or posix.startswith(tree + "/")
               for tree in VERIFY_ORACLE_TREES)


def _running_from_checkout() -> bool:
    return _is_checkout_root(Path(__file__).resolve().parent.parent)


def verify_oracle_path(relative: str | Path) -> Path:
    """Where a verification-oracle path's bytes live.  Never raises.

    A checkout answers out of its own ``woof/data``; an install answers out
    of the ``gpuwm-kernels`` companion.  With the companion absent the
    in-package spelling comes back, which does not exist on an install;
    :func:`require_verify_oracle` turns that into a plain refusal.  Module
    constants use this form so importing a verifier never fails.
    """

    posix = str(relative).replace("\\", "/").strip("/")
    local = _PACKAGE_DATA_ROOT.joinpath(*posix.split("/"))
    if _running_from_checkout():
        return local
    import importlib.resources as resources

    try:
        anchor = resources.files(VERIFY_ORACLE_PACKAGE)
    except ModuleNotFoundError:
        return local
    return Path(str(anchor)).joinpath(VERIFY_ORACLE_SUBDIR, *posix.split("/"))


def verify_oracle_install_command() -> str:
    """The pip line that restores the companion carrying the oracles."""

    required = _required_companion_version()
    if required == _UNKNOWN_VERSION:
        return f"pip install --force-reinstall {VERIFY_ORACLE_DISTRIBUTION}"
    return (f"pip install --force-reinstall "
            f"{VERIFY_ORACLE_DISTRIBUTION}=={required}")


def require_verify_oracle(path: str | Path) -> Path:
    """``path`` if it exists, else the plain refusal naming the companion."""

    path = Path(path)
    if path.exists():
        return path
    raise CompanionDataMissing(
        f"woof cannot find its verification data: {path} is absent.  The "
        f"native-WRF reference captures that `woof verify` and the parity "
        f"checks grade against ship in the {VERIFY_ORACLE_DISTRIBUTION} "
        f"companion (a hard dependency of woof, installed by a plain "
        f"`pip install recast-woof`), and this install does not have them.  "
        f"Forecasts are not affected.  Fix it:\n"
        f"    {verify_oracle_install_command()}")

__all__ = ["COMPANION_BASEMAP", "COMPANION_DISTRIBUTION", "COMPANION_PACKAGE",
           "COMPANION_TREES", "CompanionDataMissing",
           "companion_basemap_dir", "companion_install_command",
           "companion_reinstall_command", "companion_root", "data_path",
           "package_data_root", "require_companion_member",
           "rrtmgp_data_dir", "require_verify_oracle", "thompson_table_dir",
           "VERIFY_ORACLE_DISTRIBUTION", "VERIFY_ORACLE_PACKAGE",
           "VERIFY_ORACLE_SUBDIR", "VERIFY_ORACLE_TREES",
           "verify_oracle_install_command", "verify_oracle_path"]
