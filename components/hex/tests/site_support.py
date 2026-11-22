"""What the tests need from outside a checkout, where they look for it, and how to fetch it.

A public checkout carries no MPAS source tree, no native forecast publications
and no proof-lane receipts.  A test that compares against one of them asks
here, and when the thing is absent it SKIPS BY NAME with the recipe that
provides it, instead of failing on a path that only ever existed on the
machine the test was written on.

THE BREAKAGE THIS PREVENTS: nine CPU-tier tests opened a source tree, a queue
program or a receipt folder by an absolute home path, so on every other host
(public CI included) they failed with FileNotFoundError instead of saying what
they needed (release hygiene, 2026-10-14).

MPAS-A v8.4.2 source
    ``MPAS_REF`` (the older spellings ``MPAS_V842_SOURCE``, ``MPAS_V842_REF``,
    ``MPAS_ORACLE_SOURCE`` and ``MPAS_ORACLE_NATIVE_SOURCE`` are read first
    where a test used them).  Recipe: ``MPAS_SOURCE_RECIPE``.
Proof receipts a test writes
    the named environment variable, else a temporary folder.
The pinned RRFS fork and its published decks
    the test's own variable, else ``ref/rrfs/`` under ``HEX_REFERENCE_ROOT`` on a
    maintainers' pool host, else ``vendor/`` in this checkout (the order
    tools/check_mpas_citations.py resolves the fork in).  Recipe: ``RRFS_SOURCE_RECIPE``.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest

MPAS_V842_COMMIT = "f34984b2c89353cf31c235b08d3b4acd96b37bb3"
MMM_PHYSICS_COMMIT = "a4baf7f3243d1db0dbc5f63473f895bdbdc05c30"
UGWP_COMMIT = "c1c893edcf171af5639af60e3a3a528816f6cc2b"
MPAS_DATA_COMMIT = "c57dbc7be629802c6e848770a9e44b9bc602be41"

MPAS_SOURCE_RECIPE = (
    "fetch the pinned MPAS-A v8.4.2 source and set MPAS_REF to it: "
    "git clone https://github.com/MPAS-Dev/MPAS-Model.git && "
    f"git -C MPAS-Model checkout {MPAS_V842_COMMIT}; then place the externals MPAS pins "
    f"(MMM-physics {MMM_PHYSICS_COMMIT} in src/core_atmosphere/physics/physics_mmm, "
    f"UGWP {UGWP_COMMIT} in src/core_atmosphere/physics/physics_noaa/UGWP) and the "
    f"MPAS-Data {MPAS_DATA_COMMIT} tables in src/core_atmosphere/physics/physics_wrf/files"
)

SOURCE_VARIABLES = ("MPAS_REF",)


def mpas_source_root(*preferred: str) -> Path | None:
    """The MPAS source root named by the environment, or None (no skip)."""
    for key in (*preferred, *SOURCE_VARIABLES):
        value = os.environ.get(key)
        if value:
            return Path(value)
    return None


def mpas_source(*relative: str, preferred: tuple[str, ...] = ()) -> Path:
    """The pinned MPAS source root; skips by name, with the fetch recipe, when it is absent."""
    root = mpas_source_root(*preferred)
    if root is None:
        names = ", ".join((*preferred, *SOURCE_VARIABLES))
        pytest.skip(f"needs the pinned MPAS-A v8.4.2 source ({names} unset): {MPAS_SOURCE_RECIPE}",
                    allow_module_level=True)
    missing = [part for part in relative if not (root / part).exists()]
    if not root.is_dir() or missing:
        pytest.skip(f"needs the pinned MPAS-A v8.4.2 source; {root} lacks "
                    f"{', '.join(missing) or 'the tree'}: {MPAS_SOURCE_RECIPE}", allow_module_level=True)
    return root


RRFS_COMMIT = "1d6e370e7b43a332074809f170532a216c2212fb"

RRFS_SOURCE_RECIPE = (
    "fetch the pinned RRFS fork of MPAS and name it with the variable the test reads, or place it at "
    "vendor/MPAS-Model-rrfs: git clone https://github.com/RRFSx/MPAS-Model.git MPAS-Model-rrfs && "
    f"git -C MPAS-Model-rrfs checkout {RRFS_COMMIT} && "
    "git -C MPAS-Model-rrfs submodule update --init --recursive"
)


def rrfs_reference(relative: str, *preferred: str, vendor: str | None = None) -> Path:
    """A pinned RRFS reference a maintainers' pool keeps under ``ref/rrfs/<relative>``.

    The first of ``preferred`` that is set names it; else a host that exports ``HEX_REFERENCE_ROOT``
    keeps it under ``$HEX_REFERENCE_ROOT/ref/rrfs/``; else it is ``vendor/<vendor or relative>`` in this
    checkout.  No skip: the caller keeps its own behaviour for an absent reference.
    """
    for key in preferred:
        value = os.environ.get(key)
        if value:
            return Path(value)
    lanes = os.environ.get("HEX_REFERENCE_ROOT")
    if lanes:
        return Path(lanes) / "ref" / "rrfs" / relative
    return Path(__file__).resolve().parents[1] / "vendor" / (vendor or relative)


def rrfs_source(*preferred: str) -> Path:
    """The pinned RRFS fork checkout; skips by name, with the fetch recipe, when it is absent."""
    root = rrfs_reference("MPAS-Model-rrfs", *preferred)
    if not root.is_dir():
        names = ", ".join((*preferred, "HEX_REFERENCE_ROOT"))
        pytest.skip(f"needs the pinned RRFS fork source ({names}; {root} is absent): {RRFS_SOURCE_RECIPE}")
    return root


def receipt_dir(variable: str) -> Path:
    """Where a proof test writes its receipts: ``$variable``, else a temporary folder."""
    value = os.environ.get(variable)
    path = Path(value) if value else Path(tempfile.gettempdir()) / "hex-test-receipts" / variable.lower()
    path.mkdir(parents=True, exist_ok=True)
    return path


def tools_or_skip(*programs: str, what: str) -> None:
    """Skip by name when a test's build toolchain is not on PATH.

    THE BREAKAGE THIS PREVENTS (release 0.4.0 lane tail, B8, the public CPU tier run off the lab
    host in a clean python:3.13-slim-bookworm container, 2026-10-14): the native-oracle tests
    compile their Fortran fixtures with make and gfortran, and on a host without them 32 tests
    errored with "No such file or directory: 'make'" instead of saying what they need.  GitHub's
    ubuntu runners carry make, gfortran and rustc, so the public tier still runs every one of them.
    """
    missing = [name for name in programs if shutil.which(name) is None]
    if missing:
        pytest.skip(f"{what} needs {', '.join(missing)} on PATH; install the build toolchain "
                    f"({', '.join(programs)}) to run it")


def setuptools_or_skip(what: str):
    """``setuptools``, or a skip that names the test dependency.

    THE BREAKAGE THIS PREVENTS (0.6.0 candidate, every whole-directory run since dist47): since
    Python 3.12 a fresh venv carries no setuptools, and 22 packaging tests that load ``setup.py`` or
    drive the sdist template engine ended in ModuleNotFoundError (20 errors at fixture setup, 2
    failures) instead of saying what they need.  ``setuptools`` is declared in the ``dev`` extra;
    where it is installed these tests run exactly as before.
    """
    try:
        import setuptools
    except ModuleNotFoundError:
        pytest.skip(f"{what} needs setuptools, which this interpreter does not have (a fresh venv "
                    'since Python 3.12 carries none); install the test dependency: pip install '
                    '"woof hex[dev]" (or setuptools>=77)')
    return setuptools


def contract_hostprep_or_skip() -> Path:
    """The ``hex_contract_hostprep`` executable, or a skip that says how to get it.

    THE BREAKAGE THIS PREVENTS (same runs): the wheel carries the helper in ``woof/hex/libexec``; a
    source checkout does not until it is built, and 14 bitwise tests failed there with the door's
    own refusal.  The refusal is right for a user; a test tier says what is missing and moves on.
    """
    from woof.hex.drivers import contract_hostprep
    from woof.hex.errors import MpasPortError

    try:
        return Path(contract_hostprep.engine())
    except MpasPortError:
        pytest.skip("hex_contract_hostprep is not built in this checkout (the wheel carries it in "
                    "woof/hex/libexec): cargo build --release --locked --manifest-path "
                    "rust/host-prep/Cargo.toml, or name one with $WOOF_HEX_CONTRACT_HOSTPREP")


def current_bridge_or_skip(spec, mark: bytes, speaks: str, what: str) -> Path:
    """The ``spec`` bridge this machine resolves, when it is new enough for this release.

    A bridge found on a machine's own ladder (``~/.woof/bridges``, an older engine wheel, PATH) can
    predate what this release needs; the release's wheel carries current ones in ``woof/hex/libexec``.
    THE BREAKAGE THIS PREVENTS (same runs): 11 tests failed on a host whose staged ``rw_mpas_stream``
    and ``rw_mpas_geometry`` were a month older than the pinned ones.  An old bridge on the machine
    is a skip that names it.  An old bridge in THIS distribution's ``woof/hex/libexec``, or one the
    environment names on purpose, is a failure: that is the thing a user would run.
    """
    from woof.hex import engines

    path = engine_bridge_or_skip(spec, what)
    if mark in Path(path).read_bytes():
        return path
    named = [name for name in (*spec.env_names, spec.gpuwm_env) if name and os.environ.get(name)]
    if named or Path(path).resolve() == engines.bundled_candidate(spec).resolve():
        pytest.fail(f"{path} does not speak {speaks}, and it is the {spec.name} "
                    f"{'$' + named[0] + ' names' if named else 'this distribution carries'}")
    names = " or ".join("$" + name for name in (*spec.env_names, spec.gpuwm_env) if name)
    pytest.skip(f"{what} needs a {spec.name} that speaks {speaks}; the one this machine resolves "
                f"({path}) is older.  The woof hex wheel carries a current one in woof/hex/libexec; "
                f"in a checkout run tools/build_engine_bridges.sh, or name one with {names}")


def engine_bridge_or_skip(spec, what: str) -> Path:
    """The installed engine's ``spec`` bridge; skips by name, with the remedy, when it is absent.

    THE BREAKAGE THIS PREVENTS (same run as above): with an engine wheel that does not carry
    the bridge, 62 swath tests errored with the door's own EngineRefusal at fixture setup.  The
    refusal is right for a user; a test tier says which thing is missing and moves on.  Where
    the bridge is present (any host whose engine carries it, or one named in the environment)
    these tests run exactly as before.
    """
    from woof.hex import engines

    path, where = engines.locate(spec)
    if path is None:
        names = " or ".join("$" + name for name in (*spec.env_names, spec.gpuwm_env) if name)
        pytest.skip(f"{what} needs the engine's {spec.name} ({where}); without it "
                    f"{spec.what_breaks}.  Install a woof whose wheel carries {spec.name}, "
                    f"or name one with {names}")
    return path
