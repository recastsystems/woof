#!/usr/bin/env python3
"""Cut a smaller wrfout out of a big one, so the renderer can zoom.

``woof render --engine rust`` draws a whole file; it has no crop switch, and
a 1200x900 sheet at 3 km shows a supercell as four pixels.  The accurate way to
zoom is therefore not to resample the picture but to write a SMALLER FILE that
covers the sub-rectangle -- same forecast, same valid time, same 3 km cells,
fewer of them -- and render that.  Nothing is interpolated and nothing is
re-derived: every value in the crop is the identical float32 the full frame
carried.

WHAT IS ADJUSTED, AND WHY EACH ONE
----------------------------------
* Every field is sliced on its own stagger.  A mass field keeps
  ``[j0:j1, i0:i1]``; a ``west_east_stag`` field keeps ``[j0:j1, i0:i1+1]``
  because its closing face belongs to the last mass cell; likewise
  ``south_north_stag`` in y.  Slicing all of them as mass would drop the
  closing face and give the writer a shape its dimension table refuses --
  which is the good outcome; the bad one is a file that opens and is one
  column short.
* ``CEN_LAT``/``CEN_LON`` are moved to the crop's own centre and ``GRID_ID``
  becomes 2.  Both are read by the renderer -- the first for the map, the
  second for the ``d0N-3km`` filename token that keeps two renders of one run
  from overwriting each other.  ``MOAD_CEN_LAT``, ``TRUELAT*`` and
  ``STAND_LON`` are NOT touched: the projection is unchanged, only the window
  onto it.
* ``XLAT``/``XLONG`` come along as data, so the crop needs no reprojection at
  all and cannot disagree with the parent about where a cell is.

THIS IS A CROP, NOT A NEST.  ``GRID_ID = 2`` names a second grid because the
extents really are different, but there is no second integration behind it:
the cells are d01's own, at d01's 3 km spacing, and any figure made from one
must say "crop" rather than imply a nested downscale.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


#: Variables the writer owns and a caller must not hand it.
_WRITER_OWNED = frozenset({"Times", "XTIME", "ITIMESTEP"})


def window_from_lonlat(lat, lon, west, east, south, north):
    """Index window ``(i0, i1, j0, j1)`` covering a lon/lat box.

    The grid is curvilinear on a Lambert cone, so the box's corners are not
    at constant i or j; the window is the bounding box of every mass cell
    inside the requested rectangle, which is the smallest index window that
    contains it.
    """
    inside = ((lon >= west) & (lon <= east)
              & (lat >= south) & (lat <= north))
    if not inside.any():
        raise SystemExit(
            f"no cell of this file lies in lon [{west}, {east}] "
            f"lat [{south}, {north}]; the file covers lon "
            f"[{lon.min():.2f}, {lon.max():.2f}] lat "
            f"[{lat.min():.2f}, {lat.max():.2f}]")
    jj, ii = np.nonzero(inside)
    return int(ii.min()), int(ii.max()) + 1, int(jj.min()), int(jj.max()) + 1


def crop(src_path, dst_path, i0, i1, j0, j1, *, grid_id: int = 2) -> Path:
    """Write the ``[j0:j1, i0:i1]`` window of ``src_path`` as its own wrfout."""
    import netCDF4

    from woof.io.wrfout import WrfoutWriter

    src_path, dst_path = Path(src_path), Path(dst_path)
    with netCDF4.Dataset(src_path) as ds:
        nx, ny = int(i1 - i0), int(j1 - j0)
        nz = len(ds.dimensions["bottom_top"])
        soil = (len(ds.dimensions["soil_layers_stag"])
                if "soil_layers_stag" in ds.dimensions else None)
        time_str = str(netCDF4.chartostring(ds.variables["Times"][:])[0])
        fields: dict[str, np.ndarray] = {}
        for name, var in ds.variables.items():
            if name in _WRITER_OWNED:
                continue
            dims = var.dimensions
            arr = np.asarray(var[:])
            if dims and dims[0] == "Time":
                arr = arr[0]
                dims = dims[1:]
            # A vertical-only row (ZNU/ZNW) and a scalar (P_TOP) have no
            # horizontal axes at all and are carried through untouched;
            # only a field with BOTH horizontal dimensions is windowed.
            horizontal = (len(dims) >= 2
                          and dims[-1] in ("west_east", "west_east_stag")
                          and dims[-2] in ("south_north",
                                           "south_north_stag"))
            if horizontal:
                si = slice(i0, i1 + 1) if dims[-1] == "west_east_stag" \
                    else slice(i0, i1)
                sj = slice(j0, j1 + 1) if dims[-2] == "south_north_stag" \
                    else slice(j0, j1)
                arr = arr[..., sj, si]
            # ``np.ascontiguousarray`` PROMOTES a 0-d array to shape (1,),
            # which routes P_TOP -- WRF's one scalar row -- onto no axis at
            # all and raises in the writer's dimension table.
            fields[name] = (np.asarray(arr) if np.ndim(arr) == 0
                            else np.ascontiguousarray(arr))
        attrs = {k: ds.getncattr(k) for k in ds.ncattrs()}
        lat = np.asarray(ds.variables["XLAT"][0])[j0:j1, i0:i1]
        lon = np.asarray(ds.variables["XLONG"][0])[j0:j1, i0:i1]

    for drop in ("DX", "DY", "WEST-EAST_GRID_DIMENSION",
                 "SOUTH-NORTH_GRID_DIMENSION", "BOTTOM-TOP_GRID_DIMENSION",
                 "TITLE", "GPUWM_VERSION"):
        attrs.pop(drop, None)
    attrs["CEN_LAT"] = np.float32(lat[ny // 2, nx // 2])
    attrs["CEN_LON"] = np.float32(lon[ny // 2, nx // 2])
    attrs["GRID_ID"] = np.int32(grid_id)
    attrs["GPUWM_CROP_OF"] = src_path.name
    attrs["GPUWM_CROP_WINDOW"] = f"i={i0}..{i1} j={j0}..{j1} of the d01 grid"

    with netCDF4.Dataset(src_path) as ds:
        dx, dy = float(ds.DX), float(ds.DY)
        title = str(getattr(ds, "TITLE", "WOOF-ArWen"))
    writer = WrfoutWriter(dst_path, nx=nx, ny=ny, nz=nz, dx=dx, dy=dy,
                          title=title, global_attrs=attrs,
                          field_schema=fields, soil_layers=soil)
    try:
        writer.write_frame(time_str, fields)
    except BaseException:
        writer.abort()
        raise
    writer.close()
    return dst_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sources", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--box", type=float, nargs=4, default=None,
                    metavar=("WEST", "EAST", "SOUTH", "NORTH"))
    ap.add_argument("--window", type=int, nargs=4, default=None,
                    metavar=("I0", "I1", "J0", "J1"))
    ap.add_argument("--grid-id", type=int, default=2)
    args = ap.parse_args(argv)
    if (args.box is None) == (args.window is None):
        raise SystemExit("give exactly one of --box or --window")

    import netCDF4

    args.out.mkdir(parents=True, exist_ok=True)
    window = tuple(args.window) if args.window else None
    for src in args.sources:
        if window is None:
            with netCDF4.Dataset(src) as ds:
                lat = np.asarray(ds.variables["XLAT"][0])
                lon = np.asarray(ds.variables["XLONG"][0])
            window = window_from_lonlat(lat, lon, *args.box)
            print(f"window i={window[0]}..{window[1]} "
                  f"j={window[2]}..{window[3]} "
                  f"({window[1] - window[0]}x{window[3] - window[2]} cells)")
        out = crop(src, args.out / src.name, *window, grid_id=args.grid_id)
        print(f"wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
