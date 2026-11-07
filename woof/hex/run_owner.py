"""Claim transient run trees; reclaim only proven dead, same-host owners."""
from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import uuid

import psutil

MARKER = ".hex-run-owner.json"
SCHEMA = "gpuwm-hex.run-owner/v1"


def _process(pid):
    try:
        value = psutil.Process(int(pid))
        if value.status() == psutil.STATUS_ZOMBIE:
            return None
        return value.create_time()
    except psutil.NoSuchProcess:
        return None


def _alive(record):
    try:
        return _process(record["pid"]) == record["started"]
    except (psutil.AccessDenied, OSError, ValueError, TypeError, OverflowError, KeyError):
        return True  # inability to prove death is never permission to delete


def _record():
    return {"pid": os.getpid(), "started": psutil.Process().create_time()}


def _write(path, record):
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex)
    temporary.write_text(json.dumps(record), encoding="utf-8")
    os.replace(temporary, path)


def _directory(path, *, create, reuse_current=False):
    """Create a fresh transient tree, or replace a dead tree recorded by this code.

    Unknown, foreign-host, linked and live trees are preserved. The process
    start time protects a new live process whose PID was reused after a crash.
    Callers must register children before they can outlive the owner.
    Explicit resident reuse retains a tree only when this exact process is
    its sole recorded owner; ordinary claims keep the fresh-tree contract.
    """
    from .static_cache import _lock
    import shutil

    path = Path(path).absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock(path.with_name("." + path.name + ".owner.lock")):
        if path.exists() or path.is_symlink():
            try:
                marker = path / MARKER
                owner = {} if marker.is_symlink() else json.loads(marker.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                owner = {}
            known = (not path.is_symlink() and isinstance(owner, dict) and owner.get("schema") == SCHEMA
                     and owner.get("host") == platform.node()
                     and isinstance(owner.get("processes"), list)
                     and bool(owner["processes"]))
            if known and reuse_current and owner["processes"] == [_record()]:
                return path
            if not known or any(_alive(record) for record in owner["processes"]):
                raise FileExistsError(f"{path}: existing tree has a live, foreign or unknown owner; preserved")
            shutil.rmtree(path)
        if create:
            # Publish the tree only after its owner is durable. A crash between
            # mkdir and writing the marker must not leave an unclaimed run tree.
            staging = path.with_name("." + path.name + ".claim." + uuid.uuid4().hex)
            staging.mkdir()
            try:
                _write(staging / MARKER, {"schema": SCHEMA, "host": platform.node(), "processes": [_record()]})
                os.rename(staging, path)
            finally:
                if staging.exists():
                    shutil.rmtree(staging)
    return path


def claim_directory(path, *, reuse_current=False):
    return _directory(path, create=True, reuse_current=reuse_current)


def reclaim_dead_directory(path, *, reuse_current=False):
    """Remove a proven-dead tree, or retain an explicitly reused current tree."""
    return _directory(path, create=False, reuse_current=reuse_current)


def register_child(path, pid):
    """Keep a surviving rank's files when its parent has died."""
    from .static_cache import _lock
    path = Path(path).absolute()
    with _lock(path.with_name("." + path.name + ".owner.lock")):
        marker = path / MARKER
        owner = json.loads(marker.read_text(encoding="utf-8"))
        if owner.get("host") != platform.node() or _record() not in owner["processes"]:
            raise RuntimeError(f"{path}: child registration by a non-owner")
        started = psutil.Process(pid).create_time()
        owner["processes"].append({"pid": pid, "started": started})
        _write(marker, owner)
