# SPDX-License-Identifier: Apache-2.0
"""The differentiable engine's operator-row assets, fetched once on first use.

An operator row (``data/ad-doors.v1.json``) needs its derivative kernel set, the L2 scheme bundle, the parameter
carrier bundle, the two TEMPO adjoint images and the mesh files, plus an example plan package to differentiate.
They are far too large for the wheel (about 52 MB and 889 MB compressed), so, like the kernel packs, they are pinned
by size and sha256 in ``data/ad-assets.json`` and fetched from the release on first use, then unpacked under the
hex cache (``<cache>/ad/<pins digest>/``) and reused. Nothing here runs unless a door asks for its asset root.

    WOOF_HEX_AD_ASSETS        an asset root to use as is (no fetch)
    WOOF_HEX_AD_ASSETS_FROM   a directory holding the pinned files (offline install; no network)
    WOOF_HEX_AD_ASSETS_URL    a mirror base URL for the pinned files

A file that differs from its pin is refused by name and never unpacked (no fallback).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

from .. import assets as _assets

PINS = Path(__file__).resolve().parents[1] / "data" / "ad-assets.json"
ROOT_ENV = "WOOF_HEX_AD_ASSETS"
FROM_ENV = "WOOF_HEX_AD_ASSETS_FROM"
URL_ENV = "WOOF_HEX_AD_ASSETS_URL"
MARKER = ".complete.json"


class AdAssetError(RuntimeError):
    """An operator-row asset is missing, unreachable or differs from its pin."""


def load_pins() -> dict:
    pins = json.loads(PINS.read_text())
    if pins.get("schema") != _assets.SCHEMA or not isinstance(pins.get("assets"), dict) or not pins["assets"]:
        raise AdAssetError(f"{PINS}: expected {_assets.SCHEMA} with at least one asset")
    for name, row in pins["assets"].items():
        if (Path(name).name != name or not name.endswith(".tar.gz") or not isinstance(row.get("bytes"), int)
                or len(row.get("sha256", "")) != 64):
            raise AdAssetError(f"{PINS}: invalid pin for {name!r}")
    return pins


def pinned_holds(pins: dict | None = None) -> list[str] | None:
    """The asset-root paths the pinned files hold (every pin's ``holds`` list of path prefixes), or None when a pin
    does not say. A door reads it before it fetches anything, to refuse a row whose files no pinned file holds."""
    pins = pins or load_pins()
    out: list[str] = []
    for row in pins["assets"].values():
        holds = row.get("holds")
        if not isinstance(holds, list) or not holds or not all(isinstance(h, str) and h for h in holds):
            return None
        out += holds
    return sorted(out)


def asset_url(name: str, pins: dict) -> str:
    release = pins["release_assets"]
    base = os.environ.get(URL_ENV) or f"https://github.com/{release['repository']}/releases/download/{release['tag']}"
    return base.rstrip("/") + "/" + name


def _fetch(name: str, row: dict, pins: dict, cache) -> Path:
    target = _assets.asset_path(name, row, cache)
    try:
        _assets.verify(target, row, memoise=True)
        return target
    except _assets.AssetError:
        pass
    target.parent.mkdir(parents=True, exist_ok=True)
    source = os.environ.get(FROM_ENV)
    origin = {"source_dir": str(Path(source).resolve())} if source else {"url": asset_url(name, pins)}
    with _assets._lock(target.parent / ".fetch.lock"):
        try:
            _assets.verify(target, row, memoise=True)
            return target
        except _assets.AssetError:
            pass
        where = origin.get("source_dir") or origin["url"]
        print(f"woof hex: fetching {name} ({row['bytes'] / 1e6:.0f} MB, once) from {where}", file=sys.stderr, flush=True)
        fd, temporary = tempfile.mkstemp(prefix=".partial-", dir=target.parent)
        os.close(fd)
        try:
            _assets._download(name, row, source, origin, temporary)
            os.replace(temporary, target)
        except (OSError, ValueError) as error:
            raise AdAssetError(f"{name}: {error} (offline: put the pinned files in a directory and name it with "
                               f"{FROM_ENV}; or name an unpacked asset root with {ROOT_ENV})") from error
        finally:
            Path(temporary).unlink(missing_ok=True)
    return target


def _partial_root(pins: dict, names: list[str], root: Path, digest: str, cache) -> Path:
    """Fetch and unpack NAMES into ROOT, each archive once (marker ``.part-<name>.json``); the others stay unfetched."""
    todo = [n for n in names if not (root / f".part-{n}.json").is_file()]
    if not todo:
        return root
    archives = {name: _fetch(name, pins["assets"][name], pins, cache) for name in todo}
    root.mkdir(parents=True, exist_ok=True)
    with _assets._lock(root.parent / f".{digest}.lock"):
        for name, path in archives.items():
            mark = root / f".part-{name}.json"
            if mark.is_file():
                continue
            _unpack(path, root)
            mark.write_text(json.dumps({"asset": name, "sha256": pins["assets"][name]["sha256"]}) + "\n")
    return root


def _unpack(archive: Path, into: Path) -> None:
    with tarfile.open(archive, "r:gz") as tar:
        try:
            tar.extractall(into, filter="data")
        except TypeError:   # Python without extraction filters: refuse members that leave the folder
            for member in tar.getmembers():
                path = (into / member.name).resolve()
                if not str(path).startswith(str(into.resolve()) + os.sep) or member.issym() or member.islnk():
                    raise AdAssetError(f"{archive.name}: member {member.name!r} leaves the asset folder")
            tar.extractall(into)


def wanted(pins: dict, want, rows: bool = True) -> list[str]:
    """The pinned files a door needs: every operator-row file (kind ``ad-row``), and an example's mesh, plan or case
    file only when a path the door reads lies under what it holds. ``want`` None: every pinned file. ``rows`` False:
    only the files that hold ``want`` (the door's mesh check before the plan fetches the mesh file alone)."""
    if want is None:
        return sorted(pins["assets"])
    paths = [str(p) for p in want if p and not os.path.isabs(str(p))]
    return sorted(name for name, row in pins["assets"].items()
                  if (rows and (row.get("kind") == "ad-row" or not row.get("holds")))
                  or any(p.startswith(h) or h.startswith(p.rstrip("/") + "/") for h in row.get("holds") or ()
                         for p in paths))


def asset_root(cache=None, want=None, rows: bool = True) -> Path:
    """The unpacked asset root (fetching and unpacking the pinned files once), or a named refusal.

    ``want`` (asset-root paths the door will read) limits the fetch to the files that hold them, unpacked into the same
    root one file at a time (each marked when whole). THE BREAKAGE THIS PREVENTS: the release pins two examples (the
    120 km globe, about 2.4 GB, and the 3 km storm box, about 2.9 GB); fetching every pin on first use made a 120 km
    user download the storm box too, and a user of either download both. ``rows`` False fetches only the files that
    hold ``want`` (see ``wanted``); a later call with the rows fetches the rest into the same root."""
    pins = load_pins()
    digest = hashlib.sha256("".join(r["sha256"] for _, r in sorted(pins["assets"].items())).encode()).hexdigest()[:16]
    root = _assets.cache_root(cache) / "ad" / digest
    if (root / MARKER).is_file():
        return root
    if want is not None:
        return _partial_root(pins, wanted(pins, want, rows=rows), root, digest, cache)
    archives = {name: _fetch(name, row, pins, cache) for name, row in pins["assets"].items()}
    root.parent.mkdir(parents=True, exist_ok=True)
    with _assets._lock(root.parent / f".{digest}.lock"):
        if (root / MARKER).is_file():
            return root
        staging = Path(tempfile.mkdtemp(prefix=f".{digest}-", dir=root.parent))
        try:
            for name, path in archives.items():
                _unpack(path, staging)
            (staging / MARKER).write_text(json.dumps({"assets": {n: pins["assets"][n]["sha256"] for n in archives}},
                                                     indent=1, sort_keys=True) + "\n")
            if root.exists():
                shutil.rmtree(root)
            os.replace(staging, root)
        finally:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
    return root
