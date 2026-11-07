"""The woof package.

``__version__`` is read from the installed distribution's metadata, not
declared here.  A hand-maintained constant is a promise to update two
files at every cut, and that promise was broken: the wheel shipped
``0.1.1`` for four releases, and 1.1.1's prepared-cache provenance
refusal used it to tell a 1.1.1 user "this is gpuwm 0.1.1" -- a sentence
whose whole job is to name the release that is speaking.

The metadata read is the one belonging to the distribution that installed
THIS package, found by its own files (:func:`owning_distribution`), not
the first distribution of a matching name on the path.  Asked by name, a
staged ``rw-wps`` package, or any woof run beside a second woof
install (a venv and a ``--user`` install), read the other install's
number, and the rw-wps runtime check then refused it ("rw-wps version
mismatch").

The fallbacks cover what ownership cannot answer.  A tree no installed
distribution provides reads the first matching name on the path, as
before, which is what lets :mod:`woof.provenance` call that number
BORROWED and :mod:`woof.provenance_gate` refuse it when it disagrees
with the tree's own declaration.  With no distribution to ask at all it
says so rather than inventing a number.
"""

from __future__ import annotations

import time as _time

#: The earliest instant any woof code runs in this process, on both
#: clocks: :data:`LAUNCH_MONOTONIC` for durations and
#: :data:`LAUNCH_UNIX_MS` for the wall a receipt is stamped with.
#:
#: Captured HERE, on the first line of the package that any entry point
#: must import, and before this module's own metadata lookup, because
#: everything a chain wants to measure is measured FROM launch.  The
#: 2026-08-16 pre-sim audit had to answer "how long from typing the
#: command to step 1" with an external wrapper's timestamps, and the
#: reason nothing inside the product could answer it was that nothing
#: inside the product knew when it started.
#:
#: This is not the process's start: the interpreter has already booted
#: and imported this package's parents by now.  It is the earliest
#: instant woof can observe, it is stated as such wherever it is
#: reported, and the difference is milliseconds.
LAUNCH_MONOTONIC = _time.monotonic()
LAUNCH_UNIX_MS = int(_time.time() * 1000)

# WOOF_* is the spelling of every setting; the ArWen-era GPUWM_* names
# still work.  See woof/_env.py.
from woof import _env as _woof_env  # noqa: E402

_woof_env.apply()

def _no_numpy_huge_page_advice() -> None:
    """Stop numpy advising the kernel to back its arrays with huge pages.

    numpy calls ``madvise(MADV_HUGEPAGE)`` on every large array it
    allocates.  On a host whose transparent-huge-page defrag is
    ``madvise`` (a rented box: ``always defer defer+madvise [madvise] never``),
    each such allocation then compacts memory synchronously; with most of
    a 723 GB host in page cache the compaction mostly fails and retries.
    Measured there on the 9 km CONUS DA controller: 14,770 s of kernel
    time against 448 s of user time in one analysis, every card idle,
    compact_stall 28 M; a member worker stalled the same way decoding its
    radar file.  Without the advice the kernel still uses huge pages
    where it can do so cheaply; array contents are unchanged.

    ``NUMPY_MADVISE_HUGEPAGE`` is set for this process and every child it
    starts unless the caller already set it, which is honoured either
    way; numpy reads it at import, so a numpy imported before woof is
    switched through its own setter too.
    """
    import os
    import sys

    value = os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")
    numpy = sys.modules.get("numpy")
    if numpy is None or value.strip() not in ("0", ""):
        return
    for owner in ("_core", "core"):
        setter = getattr(getattr(getattr(numpy, owner, None), "multiarray",
                                 None), "_set_madvise_hugepage", None)
        if setter is not None:
            try:
                setter(False)
            except Exception:
                pass
            return


_no_numpy_huge_page_advice()


def _die_with_launcher() -> None:
    """Arm the parent-pid watchdog a launching woof process asked for.

    Here because every stage is ``python -m woof.<module>`` and imports
    this package first, so one line covers the forecast supervisor and
    every other stage a chain starts.  It does nothing unless a woof
    launcher set :data:`woof.parent_death.PARENT_ENV` for this process.
    THE BREAKAGE: a ``woof go`` killed by pid left its forecast
    supervisor alive, reparented to init, holding 63.9 GB on a card for
    43 minutes (a rented box, 2026-10-07).  See :mod:`woof.parent_death`.
    """

    import os

    if "WOOF_PARENT_DEATH_PID" not in os.environ:
        return
    try:
        from woof.parent_death import arm_from_environment
    except ImportError:
        return
    arm_from_environment()


_die_with_launcher()

from importlib import metadata as _metadata  # noqa: E402
from pathlib import Path as _Path  # noqa: E402

#: The distribution this package is published as.
DISTRIBUTION_NAME = "recast-woof"

#: Every distribution that ships this package directory, in the order the
#: version is read: woof, then the standalone RW-WPS preparation package
#: built from the same source.  Reading woof alone, a clean rw-wps
#: install reported 0+unknown and its own runtime check refused it
#: ("rw-wps version mismatch: metadata=2.8.1, module=0+unknown").
DISTRIBUTION_NAMES = (DISTRIBUTION_NAME, "rw-wps")

#: The RECORD row a distribution that installed this package carries for
#: the package's own ``__init__.py``.  Both names ship the same package
#: directory, so this one row is what tells their installs apart.
PACKAGE_INIT = "woof/__init__.py"


def candidate_distributions() -> list:
    """Every distribution on the path that may publish this package.

    In the order the version is read: each of :data:`DISTRIBUTION_NAMES`
    in turn and, under one name, every distribution the path carries,
    first on the path first.  ``importlib.metadata.version(name)`` stops
    at the first of a name, so a second install of it further down the
    path -- a venv's beside a ``--user`` one -- was never seen, even when
    it was the one serving the import.
    """

    found: list = []
    for name in DISTRIBUTION_NAMES:
        try:
            found.extend(_metadata.distributions(name=name))
        except Exception:                               # noqa: BLE001
            # An unreadable path entry must not fail ``import woof``.
            continue
    return found


def _resolved(path) -> _Path | None:
    try:
        return _Path(path).resolve()
    except (OSError, RuntimeError, TypeError, ValueError):
        return None


def installs_package(distribution, init_file, *,
                     entry: str = PACKAGE_INIT) -> bool:
    """True when ``distribution``'s own files put ``init_file`` where it is.

    ``init_file`` is a resolved path.  A wheel, and ``pip install
    --target``, lists every file it wrote in ``RECORD``: the row for
    ``entry``, located the way the distribution locates its files, must
    BE ``init_file``.  A ``.dist-info`` whose RECORD has no such row
    installed no such file however close by it sits -- an editable
    install's redirect, or metadata left beside another distribution's
    package -- and does not own it.  Metadata with no RECORD at all (an
    ``.egg-info``) sits beside the package it describes, so the file it
    locates for ``entry`` is its whole claim.
    """

    try:
        located = _resolved(distribution.locate_file(entry))
    except Exception:                                   # noqa: BLE001
        return False
    if located is None or located != init_file:
        return False
    try:
        record = distribution.read_text("RECORD")
    except Exception:                                   # noqa: BLE001
        record = None
    if record is None:
        return True
    if entry not in record.replace("\\", "/"):
        return False
    import csv
    import io

    return any(row and row[0].replace("\\", "/") == entry
               for row in csv.reader(io.StringIO(record)))


def editable_source_root(distribution) -> _Path | None:
    """The source directory a PEP 610 editable install points at, or None.

    ``direct_url.json``'s ``dir_info.editable`` marks the install, and its
    ``file:`` url names the tree.  An editable install's RECORD describes
    only its redirect, so this is the one claim it makes on the code it
    serves.
    """

    try:
        text = distribution.read_text("direct_url.json")
    except Exception:                                   # noqa: BLE001
        return None
    if not text:
        return None
    import json

    try:
        info = json.loads(text)
    except ValueError:
        return None
    if not isinstance(info, dict):
        return None
    dir_info = info.get("dir_info")
    if not isinstance(dir_info, dict) or not dir_info.get("editable", False):
        return None
    url = info.get("url")
    if not isinstance(url, str) or not url.startswith("file:"):
        return None
    try:
        from urllib.parse import unquote, urlparse

        path = unquote(urlparse(url).path)
        # ``file:///C:/x`` parses to ``/C:/x`` on Windows.
        if len(path) > 2 and path[0] == "/" and path[2] == ":":
            path = path[1:]
        return _Path(path).resolve()
    except Exception:                                   # noqa: BLE001
        return None


def owning_distribution(package_dir=None, *, candidates=None):
    """The installed distribution that provides the package, or None.

    ``package_dir`` is a ``woof`` package directory, this running one by
    default.  Every candidate is asked for exact evidence before any is
    asked for loose evidence:

    1. its own files install ``package_dir/__init__.py``
       (:func:`installs_package`), which is a wheel, a ``--target``
       install or an ``.egg-info`` beside its source;
    2. it is an editable install whose source directory holds
       ``package_dir`` (:func:`editable_source_root`).

    In that order because a venv created inside an editable install's
    checkout puts a wheel's package inside the editable root: the wheel's
    RECORD is the proof, the editable root only a neighbourhood.
    """

    init_file = _resolved(__file__ if package_dir is None
                          else _Path(package_dir) / "__init__.py")
    package = _resolved(_Path(__file__).parent if package_dir is None
                        else package_dir)
    if init_file is None or package is None:
        return None
    if candidates is None:
        candidates = candidate_distributions()
    for distribution in candidates:
        if installs_package(distribution, init_file):
            return distribution
    for distribution in candidates:
        root = editable_source_root(distribution)
        if root is not None and (package == root
                                 or package.is_relative_to(root)):
            return distribution
    return None


def version_distribution():
    """The distribution ``__version__`` is read from, or None.

    The one that provides this package (:func:`owning_distribution`).
    Only when none does, the first of :data:`DISTRIBUTION_NAMES` on the
    path, the order the version was always read in: the number then
    describes another install, and :mod:`woof.provenance` reports it as
    borrowed rather than this function hiding it.
    """

    candidates = candidate_distributions()
    owner = owning_distribution(candidates=candidates)
    if owner is not None:
        return owner
    return candidates[0] if candidates else None


def _installed_version() -> str:
    distribution = version_distribution()
    if distribution is not None:
        try:
            found = distribution.version
        except Exception:                               # noqa: BLE001
            found = None
        if found:
            return str(found)
    return "0+unknown"  # an uninstalled source tree


__version__ = _installed_version()

# Before any woof module imports CuPy: on Windows with the CUDA 12 toolkit
# wheels, CuPy 14.2 warns on import that it cannot find CUDA when it can,
# and that line headed every failed command's text in the desktop.  See
# woof/cupy_windows_warning.py.
from woof import cupy_windows_warning as _cupy_windows_warning  # noqa: E402

_cupy_windows_warning.quiet_false_cuda_path_warning()

# Verification is selected before importing any kernel module, by the
# GPUWM_WRF_EXACT* environment: woof/wrf_exact.py reads the values, and its
# install() returns unless they select it, so the default branch imports no
# compiler and preserves every existing compile route.  A process carrying
# none of those variables, which is every default run, does not import the
# module here at all, so a bare ``import woof`` still runs only this file
# and cupy_windows_warning.py: the closure tests/test_provenance_gate.py and
# tests/test_provenance.py copy into a stranger checkout.
import os as _os  # noqa: E402

# Without CuPy the process compiles no CUDA, so there is no compiler to hook.
# Breakage (2.9.1 public-CI rehearsal, route B, 2026-10-16): the Thompson
# host-parity scripts import woof with GPUWM_WRF_EXACT=1 on a CPU-only install,
# and install() died with "No module named cupy" before the host backend could
# install its CPU shim.  The guard sits here, not in wrf_exact.py, because that
# file is sealed by tests/data/assembled_legacy_proofs/source-extension.json.
# A cupy already in sys.modules counts as installed: find_spec raises
# ValueError on a module with no __spec__, such as the stand-in compiler
# tests/test_correct_mode_default.py installs on hosts without CuPy (2.9.1
# public-CI rehearsal, routes B and C, 2026-10-16).
import importlib.util as _importlib_util  # noqa: E402
import sys as _sys  # noqa: E402

if (any(_name.startswith("GPUWM_WRF_EXACT") for _name in _os.environ)
        and ("cupy" in _sys.modules or _importlib_util.find_spec("cupy") is not None)):
    from woof import wrf_exact as _wrf_exact

    _wrf_exact.install()

# Kernel numerics (owner ruling 2026-10-16: fast is the default, strict the opt-in).  woof/numerics.py
# imports nothing heavy: it arms a hook that applies the fast compile boundary when CuPy's compiler loads,
# and arms nothing under GPUWM_WRF_EXACT=1 or GPUWM_NUMERICS=strict changes nothing it compiles.
from woof import numerics as _numerics  # noqa: E402

_numerics.arm()

__all__ = ["DISTRIBUTION_NAME", "DISTRIBUTION_NAMES", "LAUNCH_MONOTONIC",
           "LAUNCH_UNIX_MS", "PACKAGE_INIT", "__version__",
           "candidate_distributions", "editable_source_root",
           "installs_package", "owning_distribution",
           "version_distribution"]
