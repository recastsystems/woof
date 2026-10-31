"""Drives no test may read, and the staging root that keeps off them.

Named breakage (2026-10-29): on the development workstation the default
model-gauntlet staging root, ``~/gpuwm-model-gauntlet-staging``, is a
directory junction onto ``F:``, a drive holding an archive that no test
or agent may read.  A desktop ``pytest`` of the mapped parity battery
followed that junction and read ``F:`` for about ten minutes.  ``F:`` and
``H:`` are reserved there: nothing in the suite may open, list or stat a
path on them, whether the path spells the drive or reaches it through a
junction or symlink.

Deciding that a path is reserved never touches the reserved drive: the
path is walked from its root, each link's target is read from the link
itself (``os.readlink``) and checked before anything beyond it is opened.
"""

from __future__ import annotations

import os
from pathlib import Path
import stat
import tempfile

#: Drive letters reserved on the development workstation.
RESERVED_DRIVES = frozenset({"F", "H"})

#: The environment variable naming the model-gauntlet staging root.
STAGING_ENV = "GPUWM_MODEL_GAUNTLET_STAGING"

#: The staging root when the environment names none.
DEFAULT_STAGING = Path.home() / "gpuwm-model-gauntlet-staging"

#: Where the default goes when it would reach a reserved drive: a scratch
#: spelling with nothing staged in it, so every real-bytes row takes its
#: "staged bytes absent" path exactly as on a box with no staging tree.
SCRATCH_STAGING = Path(tempfile.gettempdir()) / "gpuwm-model-gauntlet-staging"

_MAX_LINKS = 40
_EXTENDED_PREFIXES = ("\\\\?\\", "\\??\\")


class ReservedDriveError(RuntimeError):
    """A test or tool was pointed at a path on a reserved drive."""


def _strip_extended(path: str) -> str:
    for prefix in _EXTENDED_PREFIXES:
        if (path.startswith(prefix)
                and not path[len(prefix):].upper().startswith("UNC\\")):
            return path[len(prefix):]
    return path


def _drive_letter(path: str) -> str | None:
    drive = os.path.splitdrive(path)[0]
    if len(drive) == 2 and drive[1] == ":":
        return drive[0].upper()
    parts = path.replace("\\", "/").split("/")
    if len(parts) >= 3 and parts[:2] == ["", "mnt"] and len(parts[2]) == 1:
        return parts[2].upper()          # the WSL spelling of a drive
    return None


def _is_link(path: str) -> bool:
    try:
        info = os.lstat(path)
    except (OSError, ValueError):
        return False
    if stat.S_ISLNK(info.st_mode):
        return True
    return getattr(info, "st_reparse_tag", 0) == getattr(
        stat, "IO_REPARSE_TAG_MOUNT_POINT", None)


def reserved_drive(path) -> str | None:
    """The reserved drive ``path`` is on or links onto (``"F:"``), else None."""

    pending = _strip_extended(os.path.abspath(os.fspath(path)))
    for _ in range(_MAX_LINKS):
        drive = _drive_letter(pending)
        if drive in RESERVED_DRIVES:
            return f"{drive}:"
        parts = Path(pending).parts
        current = parts[0] if parts else pending
        for index in range(1, len(parts)):
            current = os.path.join(current, parts[index])
            if _is_link(current):
                target = _strip_extended(os.readlink(current))
                if not os.path.isabs(target):
                    target = os.path.join(os.path.dirname(current), target)
                pending = os.path.abspath(
                    os.path.join(target, *parts[index + 1:]))
                break
        else:
            return None
    raise ReservedDriveError(
        f"{path}: more than {_MAX_LINKS} links deep; refused rather than "
        "followed to a drive that could not be checked")


def refuse_reserved(path, what: str) -> Path:
    """``path`` as a Path, or :class:`ReservedDriveError` naming the drive."""

    drive = reserved_drive(path)
    if drive is not None:
        raise ReservedDriveError(
            f"{what} = {path} is on, or links onto, {drive}, a drive "
            "reserved on the development workstation that no test or tool "
            "may read.  Point it at a copy on another drive.")
    return Path(path)


def gauntlet_staging_root() -> Path:
    """The model-gauntlet staging root every real-bytes suite reads.

    ``GPUWM_MODEL_GAUNTLET_STAGING`` when set, refused with
    :class:`ReservedDriveError` if it names or links onto a reserved
    drive.  Otherwise ``~/gpuwm-model-gauntlet-staging``, unless that
    default reaches a reserved drive through a junction or symlink: then
    the empty :data:`SCRATCH_STAGING`, so the rows run as on a box with
    nothing staged instead of reading the reserved drive.
    """

    named = os.environ.get(STAGING_ENV)
    if named is not None:
        return refuse_reserved(named, STAGING_ENV)
    if reserved_drive(DEFAULT_STAGING) is not None:
        return SCRATCH_STAGING
    return DEFAULT_STAGING
