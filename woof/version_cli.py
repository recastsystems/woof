"""``woof version``: which code is actually running, and will pip move it.

A field user upgraded repeatedly and nothing changed.  He had run
``pip install -e ~/gpuwm`` months earlier, at 1.6.2; the editable
install's path entry shadows every wheel pip puts in site-packages, so
each ``pip install --upgrade recast-woof`` succeeded, reported a new version
in its own output, and left the 1.6.2 source tree serving every import.
Nothing in the product told him which code he was running, and the one
number he could see -- ``woof.__version__`` -- is read from
distribution METADATA and therefore reported the number pip had just
written, not the number of the code doing the reporting.

So this command answers four questions the version number alone cannot:

  * WHICH version -- taken from the distribution that actually provides
    the imported package (``installed_distribution`` locates the package
    file that was imported), never from whichever ``.dist-info`` happens
    to carry the name;
  * WHERE the running code lives -- a site-packages wheel, or a source
    tree, with the path;
  * WHETHER pip can move it -- an editable install cannot be upgraded by
    ``pip install --upgrade``, and saying so is the whole point;
  * WHAT the index has -- one PyPI lookup, ONLY when asked for with
    ``--check-pypi``, so "am I behind" does not require a second tool.

The network is opt-in, and when opted into it is strictly last.  A
version command is run inside self-contained installs (the desktop
package's sealed runtime, air-gapped nodes) where an outbound request on
every invocation is a privacy and reproducibility defect, so the default
asks nothing.  With ``--check-pypi`` every identity line is still printed
BEFORE the lookup is attempted, the lookup has a short timeout, and any
failure at all -- offline, proxy, DNS, a 404, a slow index -- prints
nothing rather than an error.  The one line that waits for the lookup is
the wheel upgrade advice, because on an install AHEAD of the index that
advice points backwards and is dropped (UX finding N16); a silent index
keeps it, one timeout later.  ``--offline`` is kept as a no-op so scripts
that passed it keep working.

Like :mod:`woof.update_cli`, this prints and never executes: replacing
a package's files under the process importing them is how a
half-upgraded install happens.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

#: The index lookup is a courtesy, not a dependency.  Two seconds is
#: long enough for a warm connection and short enough that a reader on a
#: dead network sees the local answer without noticing the wait.
PYPI_TIMEOUT_S = 2.0

# git is no longer run from this module at all; :mod:`woof.provenance`
# owns every git question this product asks about its own tree, and its
# GIT_TIMEOUT_S is the one that applies.


def _direct_url(distribution) -> dict:
    """``direct_url.json`` for a distribution, or ``{}``.

    Delegates to :mod:`woof.provenance`, which is the one resolver for
    "which tree is executing".  Kept under this name because it is what
    this module's readers and tests already call.
    """

    from woof.provenance import direct_url

    return direct_url(distribution)


def _git_identity(root: Path) -> dict:
    """``{"commit": ..., "branch": ...}`` for a checkout, or ``{}``.

    A thin adapter over :func:`woof.provenance.git_identity`, which
    asks git ONE question (``status --porcelain=v2 --branch``) instead
    of the two ``rev-parse`` calls this used to make, and which is the
    single implementation of every guard that matters: no git on PATH, a
    git that hangs, a directory that is not a repository, a repository
    with no commits, a detached HEAD, and -- the one that would
    otherwise bind a stranger's history -- a venv created inside
    somebody ELSE's repository.

    The shape stays ``{}``-for-absent and omits ``branch`` on a detached
    HEAD, because :func:`headline` distinguishes those and this module's
    output is pinned by tests.
    """

    from woof.provenance import git_identity

    found = git_identity(root)
    if not found:
        return {}
    identity = {"commit": found["commit"]}
    if found.get("branch"):
        identity["branch"] = found["branch"]
    return identity


def describe_install(package_root: Path, distribution) -> dict:
    """What an install IS, given where its code lives and what claims it.

    The pure half, separated from the machine, so a test can build BOTH
    arguments out of real files -- a real package directory and a real
    ``importlib.metadata`` distribution reading a real ``.dist-info`` --
    instead of asserting against whatever happens to be installed on the
    box running the suite.  That mattered here: the reference box has a
    stale woof whose ``__file__`` is ``None``, which is a shape no
    fixture would have thought to invent.

    ``editable`` is decided by two independent signals because one is
    not enough: PEP 610's ``dir_info.editable`` covers a modern
    ``pip install -e``, and "the package is not inside the directory its
    own metadata lives in" covers legacy ``setup.py develop``, a
    hand-written ``.pth``, and anything else that makes an import
    resolve outside site-packages while a distribution still claims it.
    Either one means pip cannot replace this code by upgrading.
    """

    package_root = Path(package_root).resolve()
    source_root = package_root.parent
    shape: dict = {
        "package_root": package_root,
        "source_root": source_root,
        "distribution": None,
        "version": None,
        "editable": False,
        "site_dir": None,
        "git": _git_identity(source_root),
    }
    if distribution is None:
        # Nothing pip knows about provides the code that is running:
        # a plain source tree on PYTHONPATH, or a wheel that a checkout
        # is shadowing.  Reporting woof.__version__ here is exactly the
        # trap this command exists to close -- that number would come
        # from some OTHER distribution's metadata.
        return shape
    shape["distribution"] = str(distribution.metadata["Name"])
    shape["version"] = str(distribution.metadata["Version"])
    try:
        site_dir = Path(distribution.locate_file("")).resolve()
        shape["site_dir"] = site_dir
        outside = not package_root.is_relative_to(site_dir)
    except (OSError, ValueError):
        outside = False
    marked = bool(_direct_url(distribution).get("dir_info", {})
                  .get("editable", False))
    from woof.provenance import editable_root, source_inventory

    shape["editable"] = bool(marked or outside or editable_root(distribution)
                             or source_inventory(distribution, source_root) is not None)
    return shape


def install_shape() -> dict:
    """:func:`describe_install` for the woof that is actually imported.

    The distribution is located by
    :func:`woof.provenance.providing_distribution` rather than by
    :func:`woof.runtime_manifest.installed_distribution`, which cannot
    answer for the very case this command exists to report.  A PEP 660
    editable install puts no package in ``site-packages`` -- a ``.pth``
    redirects the import -- so ``locate_file`` names a file that is not
    there and the match fails.  Measured on the reference box: an
    editable ``gpuwm 1.8.7`` pointing at its own source tree was
    reported as "no installed distribution provides it", which dropped
    the version from the headline of the one command whose job is to
    print it.
    """

    import woof
    from woof.provenance import providing_distribution

    location = getattr(woof, "__file__", None)
    if location is None:
        # A namespace package, or a half-removed install: there is no
        # single file to point at, so there is no location to report.
        # Seen in the wild on the reference box, from a stale wheel.
        location = Path(woof.__path__[0]) / "__init__.py"
    package_root = Path(location).resolve().parent
    try:
        distribution = providing_distribution(package_root)
    except Exception:                                   # noqa: BLE001
        distribution = None
    return describe_install(package_root, distribution)


def headline(shape: dict | None = None) -> str:
    """One line: the version, where it lives, and its git identity.

    Also used as ``woof doctor``'s header, so that the report which
    diagnoses an estate says up front WHICH copy of woof produced it.
    """

    shape = install_shape() if shape is None else shape
    if shape["distribution"] is None:
        lead = "woof"
        where = [f"source tree at {shape['source_root']}"]
    else:
        lead = f"woof {shape['version']}"
        where = [f"editable source at {shape['source_root']}"
                 if shape["editable"]
                 else f"installed wheel at {shape['package_root']}"]
    git = shape["git"]
    if git.get("commit"):
        branch = git.get("branch")
        where[0] += (f", git {git['commit']}"
                     + (f" on {branch}" if branch and branch != "HEAD"
                        else " on a detached HEAD"))
    if shape["distribution"] is None:
        where.append("no installed distribution provides it")
    return f"{lead} ({'; '.join(where)})"


def pypi_latest(distribution: str, *,
                timeout: float = PYPI_TIMEOUT_S) -> str | None:
    """The index's latest version, or ``None`` for ANY reason.

    Offline, proxied, rate-limited, renamed, slow, or serving something
    that is not the JSON we expect: all of them are ``None``.  This is
    a courtesy line on an informational command, and there is no failure
    here worth showing a reader instead of their own version.
    """

    url = f"https://pypi.org/pypi/{distribution}/json"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8-sig"))
        latest = payload["info"]["version"]
    except Exception:                                   # noqa: BLE001
        return None
    return str(latest) if latest else None


def _is_behind(installed: str | None, latest: str | None) -> bool | None:
    """``True``/``False``, or ``None`` when the two cannot be compared."""

    if not installed or not latest:
        return None
    try:
        from packaging.version import InvalidVersion, Version

        try:
            return Version(installed) < Version(latest)
        except InvalidVersion:
            return None
    except ImportError:
        # packaging is not a declared dependency.  Without it the only
        # accurate comparison is equality: "different" is not "behind".
        return None if installed == latest else True


def _is_ahead(installed: str | None, latest: str | None) -> bool:
    """Is this install NEWER than the index's latest?

    The case the walks measured missing (UX finding N16): a 2.5.0
    install against a 2.4.1 index was reported "current", because
    "not behind" had only two sentences.  Ahead is a distinct fact --
    a source or pre-release install -- and it is the one comparison
    where pip upgrade advice points backwards.  ``False`` whenever the
    comparison cannot be made, because withholding the upgrade line
    needs evidence.
    """

    if not installed or not latest:
        return False
    try:
        from packaging.version import InvalidVersion, Version

        try:
            return Version(installed) > Version(latest)
        except InvalidVersion:
            return False
    except ImportError:
        return False


def local_report(shape: dict | None = None) -> list[str]:
    """Every line that needs no network, built so a test can read them."""

    shape = install_shape() if shape is None else shape
    lines = [headline(shape)]
    if shape["distribution"] is None:
        lines.append(
            "  Nothing pip knows about provides this code, so "
            "`pip install --upgrade` would install a SECOND copy beside "
            "it rather than replace it.")
        lines.append(f"  Update the tree at {shape['source_root']} the way "
                     "you obtained it.")
        return lines
    if shape["editable"]:
        lines.append(
            "  This is an EDITABLE install: the source tree above is what "
            "every `import woof` resolves to, and it shadows any wheel "
            "pip puts in site-packages.")
        lines.append(
            "  `pip install --upgrade "
            f"{shape['distribution']}` will report success and change "
            "NOTHING about the code that runs.")
        if shape["git"].get("commit"):
            lines.append(
                f"  Update it with: git -C {shape['source_root']} pull")
        else:
            lines.append(f"  Update the tree at {shape['source_root']} the "
                         "way you obtained it.")
    else:
        advice = _wheel_upgrade_advice(shape)
        if advice is not None:
            lines.append(advice)
    return lines


def _wheel_upgrade_advice(shape: dict) -> str | None:
    """The wheel-install upgrade line, or ``None`` off the wheel case.

    A separate function because :func:`version_main` holds this ONE
    line back until the index has answered: on an install that is
    AHEAD of PyPI, `pip install --upgrade` points backwards, and the
    walks measured it being advised anyway (UX finding N16).  Every
    other outcome -- behind, current, no answer, ``--offline`` --
    still prints it, so a wheel user always keeps the one command that
    upgrades a wheel unless it is demonstrably wrong advice.
    """

    if shape["distribution"] is None or shape["editable"]:
        return None
    from woof.update_cli import upgrade_command

    return (f"  Installed as a wheel; upgrade with: "
            f"{upgrade_command(shape['distribution'])}")


def _is_prerelease(version: str | None) -> bool:
    """PEP 440 pre-release (``2.7.0rc1``, ``2.7.0a1``, ``2.7.0.dev3``)?

    ``False`` when the answer cannot be given: an unparsable version, or
    no ``packaging`` on the box, must not turn a release wheel into a
    "pre-release" in the sentence below.
    """

    if not version:
        return False
    try:
        from packaging.version import InvalidVersion, Version

        try:
            return Version(version).is_prerelease
        except InvalidVersion:
            return False
    except ImportError:
        return False


def _ahead_label(shape: dict) -> str:
    """What an install that is newer than the index actually IS.

    Read off the shape, never inferred from the comparison alone: a wheel
    pip installed (the sealed desktop runtime, a wheel from a release
    asset, a PyPI upload the index has not surfaced yet) whose metadata
    version is a plain release number is a RELEASE install that is newer
    than what the index lists.  A pre-release number is a pre-release
    install whatever its shape.  An editable checkout or a bare source
    tree is a source install.
    """

    if _is_prerelease(shape.get("version")):
        return "a pre-release install"
    if shape.get("distribution") is None or shape.get("editable"):
        return "a source install"
    return "a release install newer than the index lists"


#: "No latest was passed in" -- distinct from None, which is a real
#: answer meaning the index did not answer.
_ASK_THE_INDEX = object()


def pypi_report(shape: dict, *, timeout: float = PYPI_TIMEOUT_S,
                latest=_ASK_THE_INDEX) -> list[str]:
    """The staleness line, or no lines at all.  Never raises.

    ``latest`` lets :func:`version_main` ask the index ONCE and share
    the answer between this line and the upgrade-advice decision;
    omitted, the index is asked here, which is what every other caller
    wants.
    """

    if shape["distribution"] is None:
        return []
    if latest is _ASK_THE_INDEX:
        latest = pypi_latest(shape["distribution"], timeout=timeout)
    if latest is None:
        return []
    behind = _is_behind(shape["version"], latest)
    line = f"  PyPI latest is {latest}"
    if _is_ahead(shape["version"], latest):
        # Ahead is not "current": saying "current" against an older index
        # number reads as a broken comparison (measured, UX finding N16).
        # Each version number appears once in the sentence: the first
        # spelling of it said the index's number twice in eleven words,
        # which reads as a template that forgot to substitute.  WHAT kind
        # of ahead is read off the install, not guessed: the 2.7.0 desktop
        # package -- a release wheel whose metadata version is the release
        # -- was being told it was "a source or pre-release install".
        return [f"{line} -- this install is {shape['version']}, ahead of "
                f"it: {_ahead_label(shape)}."]
    if behind is False:
        return [f"{line} -- this install is current."]
    if behind is None:
        return [f"{line} (installed: {shape['version']})."]
    if shape["editable"]:
        return [f"{line}, and this install will NOT get there via pip; "
                "see the update line above."]
    return [f"{line} -- this install is behind it."]


def version_main(args=None) -> int:
    """Print the local lines; ask the index only with ``--check-pypi``.

    Never fails.  The default is the whole local answer and no network:
    the command runs inside sealed and air-gapped installs, and an
    outbound request nobody asked for is a defect there (measured on the
    2.7.0 desktop package, whose every `woof version` phoned PyPI).

    With ``--check-pypi`` there is one exception to local-first, and it
    is one line: the wheel upgrade-advice waits for the index's answer,
    because on an install that is AHEAD of PyPI that advice points
    backwards and is dropped (UX finding N16).  The identity lines still
    print before the lookup, and the advice still prints -- after at most
    the lookup's two-second timeout -- whenever the index is silent,
    refused, or reports the install behind or current.  ``--offline`` is
    accepted and changes nothing: it names the default.
    """

    shape = install_shape()
    offline = not getattr(args, "check_pypi", False)
    advice = None if offline else _wheel_upgrade_advice(shape)
    for line in local_report(shape):
        if advice is not None and line == advice:
            continue                     # held for the index's answer
        print(line, flush=True)
    if offline:
        return 0
    latest = (pypi_latest(shape["distribution"],
                          timeout=getattr(args, "pypi_timeout",
                                          PYPI_TIMEOUT_S))
              if shape["distribution"] is not None else None)
    if advice is not None and not _is_ahead(shape["version"], latest):
        print(advice, flush=True)
    for line in pypi_report(shape, latest=latest):
        print(line)
    return 0


def register_cli(subparsers):
    # A lookup that fails prints nothing by design, so a timeout of zero,
    # below zero or not a number was a lookup that could never answer and
    # said so to nobody; the parser refuses those instead.
    from woof.cli_numbers import positive_float

    parser = subparsers.add_parser(
        "version",
        help="which woof is actually running: version, import location, "
             "whether it is an editable install pip cannot upgrade, its "
             "git identity, and how that compares to PyPI")
    parser.add_argument(
        "--check-pypi", action="store_true", dest="check_pypi",
        help="also ask pypi.org for the latest published version and say "
             "whether this install is behind, current or ahead of it.  Off "
             "by default: `woof version` makes no network request unless "
             "this flag is given, and a lookup that does not answer prints "
             "nothing rather than an error")
    parser.add_argument(
        "--offline", action="store_true",
        help="accepted for older scripts; names the default (no PyPI "
             "lookup) and changes nothing")
    parser.add_argument(
        "--pypi-timeout", type=positive_float, default=PYPI_TIMEOUT_S,
        metavar="SECONDS", dest="pypi_timeout",
        help=f"seconds to wait for the index under --check-pypi (default "
             f"{PYPI_TIMEOUT_S})")
    parser.set_defaults(func=version_main)
    return parser
