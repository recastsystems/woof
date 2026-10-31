"""Acquire the exact WRF 3.9 fork coefficient set without changing its pins.

The set is published as four release assets of every release from 2.8.7 on
(``thompson-wrf39-noaa-<table>``; tools/release/cut/identities.py names them
and the cut uploads them).  A machine with no named source downloads them
from its own release, then from the first release that carried them, each
file verified against :data:`woof.core.thompson_contract.FORK_TABLE_ASSETS`
before it is installed.  The set can also come from a local directory
(``--from``, ``WOOF_THOMPSON_FORK_TABLE_SOURCE_ROOT``), from
recast-woof-data/data/thompson/wrf39-noaa, or from an explicitly selected mirror.
When no route answers, the runtime builds unchanged, hash-pinned public
Fortran inputs with a pinned portable libc on a local CPU. Every route
enforces the same canonical manifest before installation. Different
official HRRR floating-point bytes are refused.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import secrets
import shutil

from woof import fetch_guard
from woof.core.thompson_contract import (
    FORK_REFERENCE_SOURCE, FORK_TABLE_ASSETS, FORK_TABLE_SET_ID,
    validate_table_assets,
)
from woof.table_assets import (
    TableAssetError, classify_assets, fetch_asset_from_dir, fetch_asset_from_url,
)

FORK_TABLE_SOURCE_ROOT_ENV = "WOOF_THOMPSON_FORK_TABLE_SOURCE_ROOT"
FORK_TABLE_ASSET_URL_BASE_ENV = "WOOF_THOMPSON_FORK_TABLE_ASSET_URL_BASE"
#: The published route for a machine with no Fortran compiler: the four
#: pinned files as release assets of the engine's own release, each under
#: the prefix below because the classic set publishes two of the same
#: filenames.  Every downloaded byte is held to the same size and SHA-256
#: pins as every other route.  ``WOOF_THOMPSON_FORK_RELEASE_BASE``
#: overrides the bases with one, and an empty value turns the published
#: route off (the source build remains).
FORK_RELEASE_BASE_ENV = "WOOF_THOMPSON_FORK_RELEASE_BASE"
FORK_PUBLISHED_PREFIX = "thompson-wrf39-noaa-"
#: The first release whose assets carry the set.  Every later cut carries
#: the same pinned bytes again, so a release hosts the set its own wheel
#: pins; this tag answers for a version whose own release is not published
#: yet (a source checkout ahead of the last release).
FORK_FIRST_RELEASE_TAG = "v1.1.0"
_FIRST_RELEASE = (1, 1, 0)
_VERIFIED_ROOTS: dict[str, tuple] = {}


def _signature(root: Path) -> tuple | None:
    """Stat-only identity of the installed set: one ``stat`` per file.

    Size, mtime_ns, ctime_ns and inode of every asset.  A replaced,
    rewritten or truncated file changes it, so the cache below falls
    through to the full SHA-256 verification under the lock.
    """
    try:
        signature = []
        for asset in FORK_TABLE_ASSETS:
            info = (root / asset.filename).stat()
            signature.append((asset.filename, info.st_size, info.st_mtime_ns,
                              info.st_ctime_ns, info.st_ino))
        return tuple(signature)
    except OSError:
        return None


def _verified_in_process(root: Path, key: str) -> bool:
    """True when this process already verified ``root`` and nothing moved."""
    signature = _signature(root)
    return signature is not None and _VERIFIED_ROOTS.get(key) == signature


def _packaged_source() -> Path | None:
    try:
        from woof_data import data_root
    except ImportError:
        return None
    candidate = data_root() / "thompson" / "wrf39-noaa"
    return candidate if candidate.is_dir() else None


def _release_download_root() -> str:
    """``https://github.com/<owner>/<repo>/releases/download``."""
    from woof.table_assets import RELEASE_ASSET_BASE_URL
    return RELEASE_ASSET_BASE_URL.rsplit("/", 1)[0]


def _own_release_tag() -> str | None:
    """``v<version>`` of this engine when that release can carry the set."""
    try:
        from woof import __version__ as version
    except ImportError:  # pragma: no cover - the package always has one
        return None
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", str(version or ""))
    if match is None or tuple(int(part) for part in match.groups()) < _FIRST_RELEASE:
        return None
    return "v" + match.group(0)


def _published_bases() -> tuple[str, ...]:
    """The release bases the fork set is published under, in order; () is off.

    This engine's own release first, then the first release that carried
    the set.  The bytes are the same pinned bytes under either, so the
    order only decides which host answers.
    """
    override = os.environ.get(FORK_RELEASE_BASE_ENV)
    if override is not None:
        base = override.strip().rstrip("/")
        return (base,) if base else ()
    root = _release_download_root()
    bases = []
    tag = _own_release_tag()
    if tag is not None:
        bases.append(f"{root}/{tag}")
    first = f"{root}/{FORK_FIRST_RELEASE_TAG}"
    if first not in bases:
        bases.append(first)
    return tuple(bases)


def published_asset_name(asset) -> str:
    """The release-asset filename of one fork table."""
    return FORK_PUBLISHED_PREFIX + asset.filename


def _fetch_published(root: Path, absent, base: str) -> None:
    """Download every absent pinned file from the published release."""
    for asset in absent:
        fetch_asset_from_url(root, asset, base + "/" + published_asset_name(asset))


def _build_source(work: Path, log: Path) -> Path:
    from woof.thompson_fork_build import build_canonical_fork_tables
    return build_canonical_fork_tables(work, log_path=log)


def _cleanup_build(work: Path, root: Path) -> None:
    # This unique work directory was created by this call, never supplied by
    # the operator. Record every spent file and size before removing it.
    if work.parent.resolve() != root.resolve() or not work.name.startswith(".fork-build-"):
        raise RuntimeError("Fork build cleanup escaped its owned cache directory")
    if not work.exists():
        return
    files = []
    for directory, subdirs, names in os.walk(work, followlinks=False):
        for name in list(subdirs):
            candidate = Path(directory) / name
            if candidate.is_symlink():
                files.append({"path": str(candidate.relative_to(work)), "bytes": candidate.lstat().st_size})
                subdirs.remove(name)
        for name in names:
            candidate = Path(directory) / name
            files.append({"path": str(candidate.relative_to(work)), "bytes": candidate.lstat().st_size})
    fetch_guard.atomic_write_text(root / "fork-table-build-cleanup.json", json.dumps(
        {"owned_work": work.name, "files": files, "logical_bytes": sum(item["bytes"] for item in files)}, indent=2) + "\n")
    shutil.rmtree(work)


def ensure_thompson_fork_tables(root=None, *, source_dir=None) -> Path:
    """Verify or acquire the selected fork's complete set under one root lock.

    Existing wrong bytes are refused without overwrite. A named source wins
    over packaged data and a mirror and must contain the complete canonical
    set. Transfers use unique temporary files and per-file atomic publication;
    a failed transfer can leave verified files to resume, never wrong bytes.
    All consumers enter this transaction before reading the selected set.

    Every fork Thompson microphysics call comes through here, so a root this
    process already verified returns on its stat-only signature BEFORE the
    lock: the lock and the SHA-256 verification run once per process, and
    again only when a file's size, times or inode change.  The defect this
    order fixes: the lock used to be taken first on every model step, and
    member processes sharing one TMPDIR and table root (the box-A DA cycles,
    2026-10-06; about 70 processes) queued on one lock file at its 0.25 s
    poll with their GPUs near 0%.

    What the lock still prevents: two processes acquiring, downloading or
    building the set into the same root at once, which would interleave
    partial files and could publish a receipt describing the other one's
    bytes.  What the verification still prevents: fork kernels reading a root
    whose bytes differ from the pins in FORK_TABLE_ASSETS (a damaged file, a
    foreign build's floating-point bytes), whose tables index the fork's
    records wrongly and change the physics silently.
    """
    if root is None:
        from woof.physics_compat import thompson_fork_table_root
        root = thompson_fork_table_root()
    root = Path(root)
    key = str(root.resolve())
    if _verified_in_process(root, key):
        return root
    with fetch_guard.hold("fetch-tables", root):
        # Another thread of this process may have verified it while this
        # one waited; the signature is re-read under the lock.
        signature = _signature(root)
        if signature is not None and _VERIFIED_ROOTS.get(key) == signature:
            return root
        _valid, invalid, absent = classify_assets(root, FORK_TABLE_ASSETS)
        if invalid:
            raise TableAssetError(
                "fork Thompson cache contains different bytes; refusing "
                "without overwrite: " + "; ".join(invalid))
        if not absent:
            _VERIFIED_ROOTS[key] = signature
            return root
        source = source_dir
        if source is None:
            source = os.environ.get(FORK_TABLE_SOURCE_ROOT_ENV) or _packaged_source()
        mirror = os.environ.get(FORK_TABLE_ASSET_URL_BASE_ENV, "").strip().rstrip("/")
        owned_work = None
        published_errors: list[str] = []
        if source is None and not mirror:
            for published in _published_bases():
                root.mkdir(parents=True, exist_ok=True)
                try:
                    _fetch_published(root, absent, published)
                    validate_table_assets(root, FORK_TABLE_ASSETS)
                except (OSError, ValueError, TableAssetError) as error:
                    # Verified files stay to resume; wrong bytes never land
                    # (fetch_asset_from_url verifies before installing).
                    published_errors.append(f"{published}: {error}")
                    _valid, invalid, absent = classify_assets(root, FORK_TABLE_ASSETS)
                    if invalid:
                        raise TableAssetError(
                            "fork Thompson cache contains different bytes; "
                            "refusing without overwrite: " + "; ".join(invalid))
                    continue
                else:
                    fetch_guard.atomic_write_text(
                        root / "fork-table-acquisition.json", json.dumps({
                            "schema": 1, "table_set": FORK_TABLE_SET_ID,
                            "reference_source": FORK_REFERENCE_SOURCE,
                            "acquired_from": published,
                            "assets": [{"filename": asset.filename,
                                        "published_as": published_asset_name(asset),
                                        "bytes": asset.bytes,
                                        "sha256": asset.sha256}
                                       for asset in FORK_TABLE_ASSETS],
                        }, indent=2) + "\n")
                    _VERIFIED_ROOTS[key] = _signature(root)
                    return root
        if source is None and not mirror:
            root.mkdir(parents=True, exist_ok=True)
            owned_work = root / (".fork-build-" + secrets.token_hex(8))
            try:
                source = _build_source(owned_work, root / "fork-table-source-build.log")
            except (OSError, ValueError, TableAssetError) as error:
                _cleanup_build(owned_work, root)
                published_note = ("" if not published_errors else
                                  " The published release also failed: "
                                  + "; ".join(published_errors) + ".")
                raise FileNotFoundError(
                    "thompson_version='wrf_39_noaa' canonical table acquisition "
                    f"failed at {root}: {error}.{published_note} Offline: woof fetch-tables "
                    "--thompson-fork --thompson-fork-only --from DIR; alternatively "
                    f"set {FORK_TABLE_SOURCE_ROOT_ENV} or {FORK_TABLE_ASSET_URL_BASE_ENV}. "
                    "The exact oracle route is tools/thompson_fork_oracle/build.sh; "
                    "different generated bytes are refused, never re-pinned.") from error
        if source is not None:
            source = Path(source)
            try:
                validate_table_assets(source, FORK_TABLE_ASSETS)
            except (OSError, ValueError) as error:
                if owned_work is not None:
                    _cleanup_build(owned_work, root)
                raise TableAssetError(
                    f"fork Thompson source {source} is not the canonical "
                    f"complete set: {error}") from error
        try:
            root.mkdir(parents=True, exist_ok=True)
            for asset in absent:
                if source is not None:
                    fetch_asset_from_dir(root, asset, source)
                else:
                    fetch_asset_from_url(root, asset, mirror + "/" + asset.filename)
            validate_table_assets(root, FORK_TABLE_ASSETS)
            receipt = {
                "schema": 1, "table_set": FORK_TABLE_SET_ID,
                "reference_source": FORK_REFERENCE_SOURCE,
                "acquired_from": "pinned-public-source-build" if owned_work is not None else (str(source) if source is not None else mirror),
                "assets": [{"filename": asset.filename, "bytes": asset.bytes,
                            "sha256": asset.sha256} for asset in FORK_TABLE_ASSETS],
            }
            if owned_work is not None:
                receipt["source_build"] = json.loads((owned_work / "source-build-receipt.json").read_text())
            fetch_guard.atomic_write_text(
                root / "fork-table-acquisition.json", json.dumps(receipt, indent=2) + "\n")
            _VERIFIED_ROOTS[key] = _signature(root)
            return root
        finally:
            if owned_work is not None:
                _cleanup_build(owned_work, root)


def stage_thompson_fork_tables(source_dir=None, root=None, *, raise_errors=False) -> int:
    """CLI acquisition with the runtime transaction; preserve errors for API callers."""
    try:
        staged = ensure_thompson_fork_tables(root, source_dir=source_dir)
    except (OSError, ValueError, TableAssetError, fetch_guard.FetchLockBusy) as error:
        if raise_errors:
            raise
        print(f"woof fetch-tables --thompson-fork: REFUSED: {error}")
        return 2
    print(f"woof fetch-tables --thompson-fork: verified {staged} "
          f"({FORK_TABLE_SET_ID})")
    return 0
