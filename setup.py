"""Platform tagging and source provenance for the built distribution.

Everything about this distribution is declared in ``pyproject.toml``.
This file handles build details setuptools cannot express declaratively:
stamping the source commit, and tagging a wheel for the platform its
prebuilt Rust artifacts run on.

The rule is mechanical, and it reads the tree rather than a flag:

* ``woof/libexec/bridges`` is absent -> the wheel is pure Python and is
  tagged ``py3-none-any``, exactly as every release through 2.3.3 was.
  This is still published, and it is what pip resolves on a platform no
  bundle exists for (macOS, aarch64).  On it, every Rust door refuses BY
  NAME with a build-from-source remedy; nothing degrades into Python.
* ``woof/libexec/bridges`` is present -> the wheel carries native code
  for exactly one platform and is tagged ``py3-none-<platform>``
  (``py3-none-win_amd64``, ``py3-none-manylinux_2_28_x86_64``).  The
  platform comes from the staged ``BUNDLE.json``, never from the machine
  running the build, so a cross-staged bundle cannot be mis-tagged as
  the builder's own platform.

The ABI tag stays ``none`` and the Python tag stays ``py3`` on purpose.
The artifacts are executables and cdylibs launched or ``ctypes``-loaded
by path -- they are not CPython extension modules, they link no libpython
and they see no ``Py_`` symbol.  Letting bdist_wheel derive a
``cp313-cp313`` tag from ``has_ext_modules`` would multiply the release
matrix by every supported Python minor to describe a coupling that does
not exist.

``tools/stage_wheel_bridges.py`` is what creates that directory; see its
docstring for what is staged and what deliberately is not.
"""

from __future__ import annotations

import functools
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py as _BuildPy
from setuptools.command.sdist import sdist as _Sdist
from setuptools.dist import Distribution

try:  # setuptools >= 70.1 vendors bdist_wheel; older trees use the wheel package
    from setuptools.command.bdist_wheel import bdist_wheel as _BdistWheel
except ImportError:  # pragma: no cover - older setuptools
    from wheel.bdist_wheel import bdist_wheel as _BdistWheel

_HERE = Path(__file__).resolve().parent
_STAGED = _HERE / "woof" / "libexec" / "bridges"
_MANIFEST = _STAGED / "BUNDLE.json"


def _build_provenance(version: str) -> dict:
    """Stamp at build time; retain the stamp when a wheel comes from an sdist."""
    identity = dict(schema="gpuwm-build-provenance-v1", version=version,
                    commit="unknown")
    if (_HERE / ".git").exists():
        try:
            result = subprocess.run(
                ["git", "-C", str(_HERE), "rev-parse", "HEAD"],
                check=True, capture_output=True, text=True, timeout=10)
            commit = result.stdout.strip()
            if len(commit) == 40 and all(c in "0123456789abcdef" for c in commit):
                identity["commit"] = commit
        except (OSError, subprocess.SubprocessError):
            pass
    else:
        try:
            stamp = json.loads((_HERE / "gpuwm-build-provenance.json").read_text())
            if (isinstance(stamp, dict)
                    and stamp.get("schema") == identity["schema"]
                    and stamp.get("version") == version):
                commit = stamp.get("commit")
                if (isinstance(commit, str) and len(commit) == 40
                        and all(c in "0123456789abcdef" for c in commit)):
                    identity["commit"] = commit
        except (OSError, ValueError, KeyError, TypeError):
            pass
    return identity


#: Platform key (woof's spelling) -> the wheel platform tag it installs on.
#:
#: manylinux_2_28 rather than a plain ``linux_x86_64``: PyPI rejects the
#: bare tag outright. The Linux release build and all native probes run
#: inside the pinned manylinux_2_28 image, with a checked symbol/dependency
#: policy; the Ubuntu host's newer libc is not the payload's baseline.
PLATFORM_TAGS = {
    "win-x86_64": "win_amd64",
    "linux-x86_64": "manylinux_2_28_x86_64",
}

#: The glibc the Linux tag promises, and the reading that checks the staged
#: bytes keep that promise.  The linker writes each version requirement's
#: name (``GLIBC_2.34`` and the like) into the file, so the newest one any
#: staged ELF names is the oldest glibc that can load every artifact.
_LINUX_POLICY_GLIBC = (2, 28)
_GLIBC_VERSION = re.compile(rb"GLIBC_(\d+)\.(\d+)")
_ELF_MAGIC = b"\x7fELF"


@functools.lru_cache(maxsize=None)
def _staged_glibc_floor(staged: Path = _STAGED) -> tuple[tuple[int, int], str] | None:
    """``((major, minor), artifact)``: the newest glibc a staged ELF needs, or None."""

    newest = None
    for path in sorted(staged.rglob("*")):
        if not path.is_file():
            continue
        with path.open("rb") as handle:
            if handle.read(4) != _ELF_MAGIC:
                continue
        found = [(int(a), int(b)) for a, b in _GLIBC_VERSION.findall(path.read_bytes())]
        if found and (newest is None or max(found) > newest[0]):
            newest = (max(found), path.relative_to(staged).as_posix())
    return newest


def _platform_tag(platform: str) -> str:
    """The wheel platform tag for the staged bytes, read from the bytes.

    The breakage this prevents (2.9.4 cut, 2026-10-22): the battery's
    measurement wheel was built from bridges compiled on an Ubuntu glibc
    2.43 host and still tagged ``manylinux_2_28_x86_64``, because the tag
    came from ``PLATFORM_TAGS`` alone.  pip installed it on the woof hex
    CI runner (Ubuntu 24.04, glibc 2.39) and 398 tests failed with
    ``version `GLIBC_2.43' not found`` (rw_mpas_hostprep, rw_mpas_static,
    rw_wrfbatch).  A Linux wheel whose artifacts need a newer glibc than
    the policy is now tagged with the glibc they need, so pip refuses it
    where it cannot load; :func:`_refuse_off_policy_wheel` refuses to build
    such a wheel at all outside a development build.
    """

    tag = PLATFORM_TAGS[platform]
    if platform != "linux-x86_64":
        return tag
    floor = _staged_glibc_floor()
    if floor is None or floor[0] <= _LINUX_POLICY_GLIBC:
        return tag
    (major, minor), _artifact = floor
    return f"manylinux_{major}_{minor}_x86_64"


def _refuse_off_policy_wheel(platform: str) -> None:
    """No release wheel carries Linux artifacts newer than the manylinux_2_28 policy.

    The release builds the Linux bridges inside the pinned manylinux_2_28
    image (``tools/build_linux_release_bridges.py``, the cut's native stage),
    where nothing can bind a newer symbol.  Bridges built on the host bind
    its glibc (2.43 on the Ubuntu nodes: ``atan2f``, ``acosf``, ``libm``), and
    a wheel of them installs on RHEL 8/9 and Ubuntu 22.04/24.04 as helpers
    that cannot start.  A development or measurement wheel
    (``WOOF_ALLOW_UNPINNED_WHEEL=1``, never published) is built, tagged with
    the glibc its bytes need.
    """

    tag = _platform_tag(platform)
    if tag == PLATFORM_TAGS[platform]:
        return
    (major, minor), artifact = _staged_glibc_floor()
    if os.environ.get(_UNPINNED_OK_ENV) == "1":
        print(f"setup.py: WARNING: woof/libexec/bridges/{artifact} needs glibc "
              f"{major}.{minor}, newer than the {PLATFORM_TAGS[platform]} policy; this "
              f"development wheel is tagged {tag}, so pip installs it only where it loads.",
              file=sys.stderr)
        return
    raise SystemExit(
        f"setup.py: REFUSED: woof/libexec/bridges/{artifact} needs glibc "
        f"{major}.{minor}, newer than the {PLATFORM_TAGS[platform]} policy the "
        "release promises: these bridges were built on a host, not in the pinned "
        "manylinux_2_28 image, and on any host older than glibc "
        f"{major}.{minor} every door that starts one fails with \"version "
        f"`GLIBC_{major}.{minor}' not found\".  Build the Linux bridges with "
        "tools/build_linux_release_bridges.py (the cut's native stage), stage "
        "them, then build.")


#: The packaged bridge pins document of the tree being built.  Read
#: directly rather than through :mod:`woof.bridge_assets`, because
#: setup.py must not import the package it is packaging.
_PINS_PATH = _HERE / "woof" / "data" / "bridges" / "bridge-pins.json"

#: Explicit opt-in for building a wheel from an UNPINNED tree.  Dev and
#: test fixtures set it (their wheels never leave a temp directory); a
#: release must not, and a wheel built under it is unpublishable by
#: construction.
_UNPINNED_OK_ENV = "WOOF_ALLOW_UNPINNED_WHEEL"


def _refuse_unpinned_wheel() -> None:
    """No wheel is built while the packaged bridge pins pin nothing.

    The breakage this prevents happened, on the 2.5.0 candidate:
    ``tools/build_bridge_bundle.py pin`` was skipped before the wheel
    build, the wheel shipped ``release: null, platforms: {}``, and on a
    clean home ``pip install recast-woof && woof setup`` reported FAILED
    bridges -- no GRIB decoder, no NetCDF decoder, renderer
    ``rw_wrfbatch`` not built -- while every check on the builder's own
    box passed, because its bridges were staged long ago.  The pin step
    existed; nothing made skipping it fail.  This does.

    Only the two facts the skip erases are checked -- a named release
    and at least one pinned platform.  Deep validation belongs to
    ``woof.bridge_assets.parse_pins``, which the pin tool already runs
    on every document it writes.
    """

    if os.environ.get(_UNPINNED_OK_ENV) == "1":
        print(f"setup.py: WARNING: {_UNPINNED_OK_ENV}=1 -- building a "
              "wheel from an unpinned tree.  It must never be published: "
              "on a clean home `woof setup` cannot stage the bridges "
              "from it.", file=sys.stderr)
        return
    try:
        payload = json.loads(_PINS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise SystemExit(
            f"setup.py: REFUSED: the packaged bridge pins document "
            f"{_PINS_PATH.relative_to(_HERE)} is unreadable ({error}); a "
            "wheel built without it installs on a clean home as FAILED "
            "bridges.  Restore woof/data/bridges/bridge-pins.json, run "
            "tools/build_bridge_bundle.py pin, then build.")
    release = payload.get("release") if isinstance(payload, dict) else None
    platforms = payload.get("platforms") if isinstance(payload, dict) else None
    if release and isinstance(platforms, dict) and platforms:
        return
    raise SystemExit(
        "setup.py: REFUSED: woof/data/bridges/bridge-pins.json declares "
        "no release and no platforms, and a wheel built from an unpinned "
        "tree is the 2.5.0 release blocker verbatim: on a clean home, "
        "`pip install recast-woof && woof setup` reports FAILED bridges -- no "
        "GRIB decoder, no NetCDF decoder, renderer rw_wrfbatch not built "
        "-- while every check on the builder's own box passes because its "
        "bridges were staged long ago.  Run the pin step first:\n"
        "  python tools/build_bridge_bundle.py pin --release <tag> "
        "--source-rev <commit> --bundle <each bundle from `pack`>\n"
        "(RELEASE_CHECKLIST.md, 'Per-cut: the assets a wheel's pins point "
        f"at').  For a local dev/test wheel that will never be published, "
        f"set {_UNPINNED_OK_ENV}=1.")


def _refuse_staged_sdist() -> None:
    """No source distribution is built while a platform's bytes are staged.

    The breakage this prevents was measured on the 2.5.0 candidate, on
    Linux: ``tools/stage_wheel_bridges.py --platform linux-x86_64``
    followed by a bare ``python -m build`` swept every staged native ELF
    artifact into the SOURCE distribution -- the 18 the declaration
    carried at that commit, 20 extra members under
    ``gpuwm-2.5.0/woof/libexec/bridges/``, the tarball up from
    95,236,252 B to 114,906,421 B, and Linux executables shipping as
    source.  The published shape escaped only because
    ``.github/workflows/publish.yml`` runs ``--clean`` immediately
    before ``python -m build``; nothing made the wrong build fail.

    ``MANIFEST.in`` prunes the directory, so the tarball's shape is
    right however it is built.  That alone would be a quieter defect,
    not a fix: ``python -m build`` builds its wheel FROM the sdist it
    just wrote, so a pruned sdist yields a tree with nothing staged and
    a ``py3-none-any`` wheel -- while the operator was staging for a
    platform one, and the wheel says nothing about it.  So the sdist
    step refuses instead, and names both ways out, because which is
    right depends on which distribution is wanted.

    There is deliberately no override.  Neither published shape needs
    one: the pure pair cleans first, and the platform wheel is built
    with ``--wheel``.
    """

    platform = _staged_platform()
    if platform is None:
        return
    raise SystemExit(
        "setup.py: REFUSED: woof/libexec/bridges is staged with the "
        f"{platform} artifacts, and an sdist built now carries them: a "
        "SOURCE distribution shipping one platform's native binaries "
        "(measured on the 2.5.0 candidate at +19,670,169 B, 20 extra "
        "members, ELF executables inside gpuwm-<version>.tar.gz).\n"
        "Pick the distribution you actually want:\n"
        "  the pure pair, which is what publish.yml ships:\n"
        "    python tools/stage_wheel_bridges.py --clean\n"
        "    python -m build\n"
        "  the platform wheel, which needs the staged bytes and no "
        "sdist:\n"
        "    python -m build --wheel\n"
        "(`python -m build` builds its wheel from the sdist, so it can "
        "never produce a platform wheel; that is why this is a refusal "
        "and not a silent prune.)")


def _staged_platform() -> str | None:
    """The platform the staged artifacts are for, or None if none staged."""

    if not _MANIFEST.is_file():
        return None
    try:
        manifest = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise SystemExit(
            f"{_MANIFEST} is unreadable ({error}); a staged bridge directory "
            "with no manifest cannot be tagged, because nothing says which "
            "platform its bytes are for.  Re-run tools/stage_wheel_bridges.py "
            "or pass --clean to build the pure wheel.") from None
    platform = manifest.get("platform")
    if platform not in PLATFORM_TAGS:
        raise SystemExit(
            f"{_MANIFEST} names platform {platform!r}, which has no wheel tag "
            f"in setup.py PLATFORM_TAGS; known: {sorted(PLATFORM_TAGS)}")
    return platform


class _BinaryDistribution(Distribution):
    """Declares the distribution impure when it carries staged artifacts."""

    def has_ext_modules(self) -> bool:  # noqa: D102 - setuptools protocol
        return _staged_platform() is not None

    def is_pure(self) -> bool:  # noqa: D102 - setuptools protocol
        return _staged_platform() is None


#: Unix mode the staged executables and shared libraries must carry
#: inside the wheel.
#:
#: The regular-file type bit (``stat.S_IFREG``, 0o100000) is part of the
#: constant, not decoration.  A zip's ``external_attr`` high half is a
#: WHOLE ``st_mode``, type bits included, and pip's installer tests it as
#: one -- ``pip._internal.utils.unpacking.zip_item_is_executable`` is
#: ``mode and stat.S_ISREG(mode) and mode & 0o111`` -- so a bare 0o755
#: fails ``S_ISREG`` and pip drops the execute bits it was asked to keep.
#: Measured on the published 2.7.4 manylinux wheel: ``pip install`` wrote
#: the bridge binaries 0644 and the first door to reach one refused by
#: name with rc 2, while ``uv``, which does not apply that predicate,
#: installed the same wheel 0775 and never showed it.
#: ``tools/build_bridge_bundle.py`` already writes ``0o100755 << 16``,
#: which is why the standalone bundles were unaffected and only the wheel
#: carried the defect.  tests/test_wheel_bridge_staging.py holds both
#: constants against pip's own predicate.
_EXECUTABLE_MODE = 0o100755
_DEFAULT_MODE = 0o100644

#: Archive prefix of the staged artifacts, as it appears in the wheel.
_STAGED_PREFIX = "woof/libexec/bridges/"

#: Deflate level of the platform wheel's rewrite.  9 rather than zlib's
#: default 6 is 0.77 MB on the Linux wheel, measured at 2.8.9, against
#: the 100,000,000-byte per-file limit both platform wheels were over.
_WHEEL_DEFLATE_LEVEL = 9


def _force_executable_bits(wheel_path: Path) -> int:
    """Stamp 0755 on every staged artifact inside a built wheel.

    A wheel records each member's unix mode in the zip's
    ``external_attr``, and ``bdist_wheel`` copies it from the file on
    disk.  On Windows there is no executable bit to copy, so a
    manylinux wheel cross-built there ships its binaries 0644 -- present,
    correct, and unrunnable.  Measured, not theorised: installing that
    wheel into a clean venv on a real Linux userland put all eleven
    artifacts in place and then died with ``PermissionError: [Errno 13]
    Permission denied`` on the first one executed.

    Forcing it here rather than trusting the build host is deliberate.
    The mode is a property of what the file IS -- a program -- not of
    the machine that packed it, and a release must not depend on having
    been assembled on a platform that happens to preserve it.

    Rewrites the archive, which RECORD tolerates: RECORD hashes member
    CONTENT, and no content changes.  The rewrite is also where the
    platform wheel's deflate level is set (:data:`_WHEEL_DEFLATE_LEVEL`).
    The level is given to ``writestr`` because a member written from a
    ``ZipInfo`` ignores the level the archive was opened with.
    """

    import shutil
    import tempfile
    import zipfile

    stamped = 0
    with tempfile.TemporaryDirectory() as work:
        rebuilt = Path(work) / wheel_path.name
        with zipfile.ZipFile(wheel_path) as source:
            # Level 9 keeps the platform wheel under the 100,000,000-byte per-file limit.
            with zipfile.ZipFile(rebuilt, "w", zipfile.ZIP_DEFLATED,
                                 compresslevel=_WHEEL_DEFLATE_LEVEL) as target:
                for info in source.infolist():
                    payload = source.read(info.filename)
                    updated = zipfile.ZipInfo(info.filename, info.date_time)
                    updated.compress_type = info.compress_type
                    updated.external_attr = info.external_attr
                    if (info.filename.startswith(_STAGED_PREFIX)
                            and not info.filename.endswith("/")
                            and not info.filename.endswith("BUNDLE.json")):
                        updated.external_attr = (_EXECUTABLE_MODE << 16)
                        stamped += 1
                    elif not info.external_attr >> 16:
                        updated.external_attr = (_DEFAULT_MODE << 16)
                    target.writestr(updated, payload,
                                    compresslevel=_WHEEL_DEFLATE_LEVEL)
        shutil.move(str(rebuilt), str(wheel_path))
    return stamped


class _PlatformWheel(_BdistWheel):
    """Tags the wheel ``py3-none-<platform>`` when artifacts are staged."""

    def finalize_options(self) -> None:
        platform = _staged_platform()
        if platform is not None:
            self.plat_name = _platform_tag(platform)
            self.plat_name_supplied = True
        super().finalize_options()
        self.root_is_pure = platform is None

    def get_tag(self) -> tuple[str, str, str]:
        python_tag, abi_tag, plat_tag = super().get_tag()
        if _staged_platform() is None:
            return python_tag, abi_tag, plat_tag
        # Keep py3/none: these are subprocess binaries and ctypes cdylibs,
        # not CPython extension modules.  See the module docstring.
        return "py3", "none", plat_tag

    def run(self) -> None:
        # Metadata and PEP 660 editable installs finalize bdist_wheel to
        # obtain tags; they do not build a distributable package.  Gate
        # the actual wheel build, before it copies any package bytes.
        _refuse_unpinned_wheel()
        if _staged_platform() is not None:
            _refuse_off_policy_wheel(_staged_platform())
        super().run()
        if _staged_platform() is None:
            return
        for _command, _version, path in self.distribution.dist_files:
            wheel_path = Path(path)
            if wheel_path.suffix != ".whl":
                continue
            stamped = _force_executable_bits(wheel_path)
            self.announce(
                f"stamped 0{_EXECUTABLE_MODE & 0o7777:o} on {stamped} staged "
                f"artifact(s) in {wheel_path.name}", level=2)

    def write_wheelfile(self, wheelfile_base, *args, **kwargs):
        super().write_wheelfile(wheelfile_base, *args, **kwargs)
        stamp = _build_provenance(self.distribution.get_version())
        (Path(wheelfile_base) / "build-provenance.json").write_text(
            json.dumps(stamp, sort_keys=True) + "\n", encoding="utf-8")


class _SourceDistribution(_Sdist):
    """Refuses to pack a source distribution out of a staged tree."""

    def finalize_options(self) -> None:
        # Before a file list is built: every route that makes an sdist
        # -- python -m build, pip install <tree> (via build_sdist),
        # python setup.py sdist -- passes through this command's
        # finalize_options; the wheel's pin gate guards its run method.
        _refuse_staged_sdist()
        super().finalize_options()

    def make_release_tree(self, base_dir, files):
        super().make_release_tree(base_dir, files)
        (Path(base_dir) / "gpuwm-build-provenance.json").write_text(
            json.dumps(_build_provenance(self.distribution.get_version()),
                       sort_keys=True) + "\n", encoding="utf-8")


#: Module-name globs, per package, that are development material and never
#: reach a wheel: the pytest suites and the fault-injection probes that live
#: BESIDE tilestream's operational modules rather than under tests/.
#:
#: The breakage this prevents was measured on the 2.7.0 candidate: all
#: three woof wheels carried 42 ``tilestream/test_*.py`` suites (one of
#: them 136,872 B) and the six ``tilestream/skeptic_*.py`` probes at import
#: top level, because ``[tool.setuptools.packages.find] exclude`` matches
#: PACKAGES (``tilestream.skeptic*`` named a sub-package that does not
#: exist) and ``exclude-package-data`` matches data files, and neither can
#: name a MODULE.  ``build_py.find_package_modules`` is the one place a
#: module is selected for the wheel, so the filter lives there.  MANIFEST.in
#: carries the matching ``recursive-exclude`` for the sdist; both halves
#: are measured by tests/test_operational_package_excludes_probes.py.
DEVELOPMENT_MODULE_GLOBS: dict[str, tuple[str, ...]] = {
    "tilestream": ("test_*", "skeptic_*"),
}


def is_development_module(package: str, module: str) -> bool:
    """Is ``package.module`` development material that never ships?"""

    import fnmatch

    return any(fnmatch.fnmatchcase(module, pattern)
               for pattern in DEVELOPMENT_MODULE_GLOBS.get(package, ()))


#: Packages whose wheel payload is exactly their ``package-data``
#: declaration.  setuptools' ``include-package-data`` (on by default for a
#: pyproject build) also hands every sdist member that lives under a
#: declared package to the wheel, and ``docs`` is a declared package for one
#: file (``docs/mpas-seam.md``, see pyproject.toml).  MEASURED on the 2.9.2
#: candidate: when MANIFEST.in started shipping the public pages in the
#: source archive, the manylinux wheel grew from 3,161 to 3,287 members
#: (+1,850,932 compressed bytes, all ``docs/``) and tripped the cut's
#: 100,000,000-byte per-file limit.  The sdist keeps its pages; the wheel
#: keeps its declaration.  tests/test_mpas_seam_doc_is_packaged.py measures
#: it with a manifest contribution present.
DECLARATION_ONLY_DATA_PACKAGES: tuple[str, ...] = ("docs",)


@functools.lru_cache(maxsize=None)
def verify_oracle_trees() -> tuple[str, ...]:
    """``woof.data_assets.VERIFY_ORACLE_TREES``, read without importing woof."""

    import ast

    source = _HERE / "woof" / "data_assets.py"
    for node in ast.parse(source.read_text(encoding="utf-8")).body:
        if (isinstance(node, ast.AnnAssign)
                and getattr(node.target, "id", None) == "VERIFY_ORACLE_TREES"):
            return tuple(ast.literal_eval(node.value))
    raise SystemExit("woof/data_assets.py declares no VERIFY_ORACLE_TREES")


def is_verify_oracle_file(path: str) -> bool:
    """Is ``path`` inside a verification-oracle tree the gpuwm-kernels companion ships?

    pyproject's exclude-package-data drops these trees from the wheel for
    files the ``woof`` package-data glob finds, but MANIFEST.in keeps them
    in the sdist, and a wheel built FROM that sdist sees them as manifest
    files of the PEP 420 namespace packages under woof/data (``namespaces =
    true``), where a ``woof`` key cannot reach.  Measured before this
    filter: the wheel built from the sdist carried all 713 oracle files
    again.  Filtering on the path covers every package they could land in.
    """

    posix = "/" + Path(path).as_posix().lstrip("/")
    return any(f"/woof/data/{tree}/" in posix for tree in verify_oracle_trees())


@functools.lru_cache(maxsize=None)
def wheel_xz_data() -> tuple[str, ...]:
    """``woof.packed_data.WHEEL_XZ_DATA``, read without importing woof."""

    import ast

    source = _HERE / "woof" / "packed_data.py"
    for node in ast.parse(source.read_text(encoding="utf-8")).body:
        if (isinstance(node, ast.AnnAssign)
                and getattr(node.target, "id", None) == "WHEEL_XZ_DATA"):
            return tuple(ast.literal_eval(node.value))
    raise SystemExit("woof/packed_data.py declares no WHEEL_XZ_DATA")


def pack_wheel_xz_data(build_lib: Path) -> list[tuple[str, int, int]]:
    """Re-encode the :func:`wheel_xz_data` members of ``build_lib`` as ``<name>.xz``.

    The breakage this prevents (WOOF 1.1.6): the platform wheel built at
    105,480,112 bytes against PyPI's 100,000,000-byte per-file limit.  These
    four members deflate to 13.9 MB in a wheel and to 3.6 MB as xz; their
    readers (woof.packed_data) take either form, so only the wheel changes:
    the tree, the sdist and an editable install keep the plain files.
    Returns (path, plain bytes, xz bytes) per member re-encoded.
    """

    import lzma
    from concurrent.futures import ThreadPoolExecutor

    def one(relative: str):
        plain = build_lib / relative
        if not plain.is_file():
            return None
        raw = plain.read_bytes()
        packed = lzma.compress(raw, preset=9)
        if lzma.decompress(packed) != raw:
            raise SystemExit(f"{relative}: xz round trip differs; prevents shipping altered data")
        plain.with_name(plain.name + ".xz").write_bytes(packed)
        plain.unlink()
        return relative, len(raw), len(packed)

    with ThreadPoolExecutor(max_workers=len(wheel_xz_data()) or 1) as pool:
        return [row for row in pool.map(one, wheel_xz_data()) if row is not None]


class _OperationalBuildPy(_BuildPy):
    """Copies every module of a package into the wheel EXCEPT the probes."""

    def build_package_data(self) -> None:  # noqa: D102
        super().build_package_data()
        if getattr(self, "editable_mode", False):
            return  # an editable install reads the checkout's plain files
        for relative, plain, packed in pack_wheel_xz_data(Path(self.build_lib)):
            self.announce(f"packed {relative} as xz: {plain:,} -> {packed:,} B", level=2)

    def find_package_modules(self, package, package_dir):  # noqa: D102
        return [row for row in super().find_package_modules(package, package_dir)
                if not is_development_module(row[0], row[1])]

    def find_data_files(self, package, src_dir):  # noqa: D102
        return [path for path in self._declared_data_files(package, src_dir)
                if not is_verify_oracle_file(path)]

    def _declared_data_files(self, package, src_dir):
        if package not in DECLARATION_ONLY_DATA_PACKAGES:
            return super().find_data_files(package, src_dir)
        manifest = self.manifest_files
        self.manifest_files = {}
        try:
            return super().find_data_files(package, src_dir)
        finally:
            self.manifest_files = manifest


setup(distclass=_BinaryDistribution,
      cmdclass={"bdist_wheel": _PlatformWheel, "sdist": _SourceDistribution,
                "build_py": _OperationalBuildPy})
