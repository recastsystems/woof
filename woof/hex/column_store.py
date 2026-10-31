"""The any-point column store of a hex run: every output hour, from the converted lat-lon frames.

Every run (WOOF and hex) writes the tiled compact column store for every output hour, so a
sounding at any point and any hour costs nothing extra. The store is rw-store version 2 (``RWSTORE2``, written by
the engine's ``rw_soundings --columns`` through :func:`woof.soundings.write_columns`; an engine before 2.9.7 writes
the same tiles as ``GPWMCOL1``, which every reader of the store also reads). It needs the 3-D fields, so it is
written from the converted frames of either field set: the render door's (``--field-set full``, its default) and
the export door's. The surface set carries the column too (T, P, PB, QVAPOR, PH, PHB, U, V and W, rw-mpas
``field_set_allows``, in every woof this package runs on, 2.9.7 and newer), so a surface export writes its store.

Files: ``<out>/<domain>/<YYYYMMDDTHHMMSSZ>.wcol``, the frames on whole hours plus the last frame when the run ends
off the hour. ``WOOF_COLUMNS=off`` writes none. A store that cannot be written is said and never fails the door.
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path

_STAMP = re.compile(r"(\d{4}-\d{2}-\d{2})_(\d{2})[_:](\d{2})[_:](\d{2})")


def enabled(environ=None) -> bool:
    value = (os.environ if environ is None else environ).get("WOOF_COLUMNS", "").strip().lower()
    return value not in ("0", "off", "false", "no")


def chosen(frames: list[Path], every_minutes: int = 60) -> list[Path]:
    """The frames on the cadence, plus the last one."""
    frames = sorted(frames, key=lambda p: p.name)
    keep = []
    for f in frames:
        m = _STAMP.search(f.name)
        if m is None or every_minutes <= 0 or ((int(m[2]) * 60 + int(m[3])) % every_minutes == 0 and int(m[4]) == 0):
            keep.append(f)
    if frames and frames[-1] not in keep:
        keep.append(frames[-1])
    return keep


_FRAME = re.compile(r"^wrfout_(d\d{2})_(\d{4})-(\d{2})-(\d{2})_(\d{2})[_:](\d{2})[_:](\d{2})$")


def stored(frame: Path, out: Path) -> Path | None:
    """The store file a frame's columns already sit in under ``out`` (``<out>/<domain>/<YYYYMMDDTHHMMSSZ>.wcol``,
    rw-store's own name for the frame's domain and valid time), else None. rw-store writes each file beside and renames
    it in, so a file there is whole."""
    m = _FRAME.match(Path(frame).name)
    if m is None:
        return None
    path = Path(out) / m[1] / f"{m[2]}{m[3]}{m[4]}T{m[5]}{m[6]}{m[7]}Z.wcol"
    return path if path.is_file() and path.stat().st_size > 0 else None


def write(frames: list[Path], out: Path, *, field_set: str = "full", say=print) -> dict:
    """Write the store for these converted frames under ``out``; returns a receipt (files, bytes, seconds) or the
    reason none was written.

    A frame whose store file is already in ``out`` keeps it (``kept``). THE BREAKAGE THIS PREVENTS (LAM-vs-zoom
    measurement 2026-10-27, defect 3): a zoom globe's products pass draws
    its whole-run maps over every frame, so it converts every frame again and wrote every frame's store again over the
    ones its live loop had written: 162 s after a 3 h forecast at 3 km (the global and refined-context stores), about
    15 to 20 min after a 24 h one, on one core while the card waited."""
    started = time.monotonic()
    if not enabled():
        return {"files": 0, "skipped": "WOOF_COLUMNS=off"}
    # No refusal of the surface set: it carries every field a sounding reads (the module notes). The refusal that
    # stood here named a breakage that was not true of any engine this door runs on, and kept a surface export (the
    # WOOF site's, every ensemble member's) from writing the store its own frames make.
    picked = chosen([Path(f) for f in frames])
    if not picked:
        return {"files": 0, "skipped": "no frames"}
    kept = [f for f in picked if stored(f, out) is not None]
    todo = [f for f in picked if f not in kept]
    if not todo:
        say(f"COLUMNS files=0 kept={len(kept)} out={out} seconds={round(time.monotonic() - started, 3)}", flush=True)
        return {"files": 0, "kept": len(kept), "bytes": 0, "out": str(out),
                "seconds": round(time.monotonic() - started, 3), "failed": []}
    try:
        from woof.soundings import write_columns
    except ImportError as error:
        return {"files": 0, "kept": len(kept), "skipped": f"the engine has no column store writer ({error})"}
    rows, failed = [], []
    for frame in todo:
        try:
            rows += write_columns(frame, out)
        except Exception as error:  # a store never fails the door
            failed.append({"frame": str(frame), "error": f"{type(error).__name__}: {error}"[:300]})
    receipt = {"files": len(rows), "kept": len(kept), "bytes": int(sum(r.get("bytes", 0) for r in rows)),
               "out": str(out), "field_set": field_set, "seconds": round(time.monotonic() - started, 3),
               "failed": failed}
    say(f"COLUMNS files={receipt['files']} kept={len(kept)} bytes={receipt['bytes']} out={out} "
        f"seconds={receipt['seconds']}"
        + (f" failed={len(failed)}: {failed[0]['error']}" if failed else ""), flush=True)
    return receipt
