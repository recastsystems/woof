"""Stage a WOOF-ArWen frame dump as a wrfout-convention file for the t=0 digest.

``tools/matched_wrfout_t0_state_digest.py`` scores a candidate/reference
pair of wrfout-convention netCDF frames.  The matched-trajectory lane's
frame writer (``run_arwen.py``) dumps ``.npz`` archives whose keys are
already WRF variable names on WRF's own staggering, so the only thing
standing between those dumps and the registered full-state digest is the
container format.  This tool re-containers one frame, exactly:

* every staged array is written float32, byte-for-byte the ``.npz`` value --
  no cast, no transpose, no fill handling, nothing derived;
* the SHA-256 of the source archive is embedded in the output's global
  attributes, so the staged file names the raw artifact it came from and a
  reader can tie it back to a committed hash manifest;
* any field excluded from staging is named in the global attributes with
  the caller's reason.  An exclusion the file itself does not disclose
  would be curation; this one is part of the record.

The one expected exclusion is ``RAINNC``.  A WOOF-ArWen run restarted from a
mature WRF history frame begins with a fresh accumulator (zero) while the
frame carries the storm's accumulated rain to that hour; both gate
documents pre-declare that comparison as two different quantities
(mp28-shortwindow-gate.md section 3, "published, never gated").  Staging it
into a state-identity digest would score that design decision, not the
state read-back.  Re-run this tool without ``--exclude`` to stage it
anyway; the digest will then report exactly that accumulator difference.

Usage:
    python stage_frame_for_t0_digest.py --frame FRAME.npz --out OUTFILE \\
        [--exclude FIELD=reason ...]

The output filename should follow the wrfout convention
(``wrfout_d<NN>_<time>``) so the digest's frame discovery pairs it; on
filesystems that forbid ``:`` in names the digest accepts ``_`` in the
timestamp.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def spatial_dimensions(shape: tuple[int, ...],
                       nz: int, ny: int, nx: int) -> tuple[str, ...]:
    """WRF dimension names for an array on WRF's own staggering."""
    if shape == (nz, ny, nx):
        return ("bottom_top", "south_north", "west_east")
    if shape == (nz, ny, nx + 1):
        return ("bottom_top", "south_north", "west_east_stag")
    if shape == (nz, ny + 1, nx):
        return ("bottom_top", "south_north_stag", "west_east")
    if shape == (nz + 1, ny, nx):
        return ("bottom_top_stag", "south_north", "west_east")
    if shape == (ny, nx):
        return ("south_north", "west_east")
    raise ValueError(f"array shape {shape} is not on WRF's staggering for "
                     f"an (nz={nz}, ny={ny}, nx={nx}) mass grid")


def stage(frame: Path, out: Path, excludes: dict[str, str]) -> list[str]:
    import netCDF4

    archive = np.load(frame)
    unstaggered = [archive[name].shape for name in archive.files
                   if archive[name].ndim == 3]
    nz, ny, nx = (min(s[0] for s in unstaggered),
                  min(s[1] for s in unstaggered),
                  min(s[2] for s in unstaggered))

    staged: list[str] = []
    out.parent.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(out, "w", format="NETCDF4") as target:
        target.createDimension("Time", None)
        for name, size in (("bottom_top", nz), ("bottom_top_stag", nz + 1),
                           ("south_north", ny), ("south_north_stag", ny + 1),
                           ("west_east", nx), ("west_east_stag", nx + 1)):
            target.createDimension(name, size)
        for name in sorted(archive.files):
            if name in excludes:
                continue
            array = np.asarray(archive[name], dtype=np.float32)
            dims = ("Time",) + spatial_dimensions(array.shape, nz, ny, nx)
            variable = target.createVariable(name, "f4", dims)
            variable[0] = array
            staged.append(name)
        target.source_archive = frame.name
        target.source_archive_sha256 = sha256_file(frame)
        target.staged_fields = " ".join(staged)
        target.excluded_fields = " ".join(
            f"{name}: {reason}" for name, reason in sorted(excludes.items())
        ) or "none"
        target.staging_tool = "tools/mp28_matched/stage_frame_for_t0_digest.py"
        target.staging_note = (
            "arrays are byte-for-byte the source archive's float32 values; "
            "this file changes the container, not the state")
    return staged


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", required=True, type=Path,
                        help="frame_t*.npz written by run_arwen.py")
    parser.add_argument("--out", required=True, type=Path,
                        help="wrfout-convention netCDF file to write")
    parser.add_argument("--exclude", action="append", default=[],
                        metavar="FIELD=reason",
                        help="leave FIELD unstaged, recording the reason in "
                             "the output's global attributes")
    args = parser.parse_args()

    excludes: dict[str, str] = {}
    for item in args.exclude:
        name, _, reason = item.partition("=")
        if not reason:
            parser.error(f"--exclude {item!r}: an exclusion without a "
                         "recorded reason is curation; use FIELD=reason")
        excludes[name] = reason

    staged = stage(args.frame, args.out, excludes)
    print(f"[stage-t0] {args.frame} -> {args.out}")
    print(f"[stage-t0] staged {len(staged)}: {', '.join(staged)}")
    for name, reason in sorted(excludes.items()):
        print(f"[stage-t0] excluded {name}: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
