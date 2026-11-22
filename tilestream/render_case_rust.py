#!/usr/bin/env python3
"""Drive ``woof render --engine rust`` over a streamed case's wrfout frames.

This is a THIN driver, on purpose.  ``RENDERING.md`` is explicit that the
production renderer is the marketing tier and that hand-rolled matplotlib
styling has no place in forecast imagery, so nothing here draws anything:
it selects files, calls ``woof render``, and reports what came back --
including, loudly, the products the engine refused and why.

    python -m tilestream.render_case_rust --run WORK/run_ks \
        --products composite_reflectivity_uh,composite_reflectivity \
        --out WORK/png --size 2400x1800

``--every N`` renders every Nth frame (the whole sequence is the default),
``--at HH:MM`` renders one valid time, and ``--crop W E S N`` first cuts the
named lon/lat box out of each selected frame with
:mod:`tilestream.case_crop` and renders the crop instead -- same cells, same
values, a smaller file.

``--boundary-zone N`` and ``--reports FILE.json`` are the two overlays this
lane's matplotlib renderer (``tilestream/render_case.py``) drew by hand and
this one could not: the dashed frame marking where the forecast stops being
a forecast, and the storm reports it is being read against.  Both go to the
production renderer through ``woof render --overlays``, which projects
them into the same frame as the fill, so no panel here is drawn twice by
two engines.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def select(run_dir: Path, every: int, at: tuple[str, ...]) -> list[Path]:
    files = sorted((run_dir / "wrfout").glob("wrfout_d01_*"))
    if not files:
        raise SystemExit(f"no wrfout frames under {run_dir / 'wrfout'}")
    if at:
        keep = [f for f in files
                if any(stamp in f.name for stamp in at)]
        if not keep:
            raise SystemExit(
                f"no frame matches {at}; have "
                + ", ".join(f.name[-19:] for f in files))
        return keep
    return files[::max(1, every)]



#: Marker colours by report kind, the SPC convention: red tornado, green
#: hail, blue wind.  Named here rather than in the renderer because it is
#: a reporting convention, not a rendering one.
REPORT_COLORS = {"torn": "#d0202c", "hail": "#1a8a3c", "wind": "#1f5fd0"}


def build_overlays(args, sample_frame: Path):
    """The overlay JSON this run needs, or ``None``.

    Merged rather than chosen between: a caller who has a hand-written
    ``--overlays`` file and also wants the boundary box should get both,
    and having to pick would send them back to hand-editing JSON.
    """

    import json

    features = {"lines": [], "points": [], "labels": [], "rings": []}
    if args.overlays is not None:
        given = json.loads(Path(args.overlays).read_text(encoding="utf-8"))
        for key in features:
            features[key].extend(given.get(key, []))

    if args.boundary_zone > 0:
        box = boundary_zone_box(sample_frame, args.boundary_zone)
        if box is not None:
            features["lines"].append(
                {"points": box, "closed": True, "color": "#20242c",
                 "width": 2})

    if args.reports is not None:
        reports = json.loads(Path(args.reports).read_text(encoding="utf-8"))
        for report in reports:
            kind = str(report.get("kind", "torn")).lower()
            features["points"].append({
                "lat": float(report["lat"]), "lon": float(report["lon"]),
                "color": REPORT_COLORS.get(kind, "#20242c"),
                "radius_px": 7, "width_px": 2,
                "shape": "circle" if kind == "torn" else "cross"})

    if not any(features.values()):
        return None
    out = args.out / "overlays.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(features, indent=1), encoding="utf-8")
    return out


def boundary_zone_box(frame: Path, cells: int):
    """``[(lat, lon), ...]`` of the box ``cells`` in from the domain edge.

    Read off the frame's OWN ``XLAT``/``XLONG`` rather than computed from
    a namelist: the box has to sit on the grid the forecast was actually
    written on, and a frame that has been cropped is a different grid than
    the one the run was configured with.
    """

    import netCDF4
    import numpy as np

    with netCDF4.Dataset(frame) as dataset:
        lat = np.asarray(dataset.variables["XLAT"][0])
        lon = np.asarray(dataset.variables["XLONG"][0])
    ny, nx = lat.shape
    if cells * 2 >= min(ny, nx):
        print(f"render: --boundary-zone {cells} is at least half of the "
              f"{ny}x{nx} grid; no box drawn")
        return None
    j0, j1 = cells, ny - 1 - cells
    i0, i1 = cells, nx - 1 - cells
    corners = [(j0, i0), (j0, i1), (j1, i1), (j1, i0)]
    return [[float(lat[j, i]), float(lon[j, i])] for j, i in corners]

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--products", default="composite_reflectivity")
    ap.add_argument("--size", default="1600x1200")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--at", nargs="*", default=(),
                    help="valid-time stamps to render, e.g. 2019-05-28_23_00")
    ap.add_argument("--crop", type=float, nargs=4, default=None,
                    metavar=("WEST", "EAST", "SOUTH", "NORTH"))
    ap.add_argument("--crop-dir", type=Path, default=None)
    ap.add_argument("--source-label", default="WOOF-ArWen")
    ap.add_argument("--heavy", action="store_true")
    ap.add_argument("--boundary-zone", type=int, default=0, metavar="CELLS",
                    help="draw a box CELLS in from the domain edge: "
                         "nothing outside it is a forecast, and a reader "
                         "who cannot see where the relaxation zone ends "
                         "will read boundary artefacts as weather")
    ap.add_argument("--reports", type=Path, default=None, metavar="FILE.json",
                    help="storm reports to mark, as a JSON list of "
                         '{"lat":..,"lon":..,"kind":"torn|hail|wind"}')
    ap.add_argument("--overlays", type=Path, default=None,
                    metavar="FILE.json",
                    help="an overlay file to pass through verbatim; "
                         "--boundary-zone and --reports are merged into it")
    args = ap.parse_args(argv)

    files = select(args.run, args.every, tuple(args.at))
    print(f"render: {len(files)} frame(s) selected from {args.run}")

    if args.crop is not None:
        from tilestream import case_crop
        crop_dir = args.crop_dir or (args.run / "wrfout_crop")
        crop_dir.mkdir(parents=True, exist_ok=True)
        cropped = []
        window = None
        import netCDF4
        import numpy as np
        for src in files:
            if window is None:
                with netCDF4.Dataset(src) as ds:
                    lat = np.asarray(ds.variables["XLAT"][0])
                    lon = np.asarray(ds.variables["XLONG"][0])
                window = case_crop.window_from_lonlat(lat, lon, *args.crop)
                print(f"render: crop window i={window[0]}..{window[1]} "
                      f"j={window[2]}..{window[3]} "
                      f"({window[1] - window[0]}x{window[3] - window[2]} "
                      "cells of the 1200x900 domain)")
            dst = crop_dir / src.name
            case_crop.crop(src, dst, *window, grid_id=2)
            cropped.append(dst)
        files = cropped

    args.out.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "woof.cli", "render",
               "--engine", "rust", "--products", args.products,
               "--out", str(args.out), "--size", args.size,
               "--source-label", args.source_label]
    if args.heavy:
        command.append("--heavy")
    overlay_file = build_overlays(args, files[0])
    if overlay_file is not None:
        command.extend(("--overlays", str(overlay_file)))
        print(f"render: overlays {overlay_file}")
    command.extend(str(f) for f in files)
    result = subprocess.run(command, cwd=str(REPO))
    written = sorted(args.out.glob("*.png"))
    print(f"render: {len(written)} PNG(s) in {args.out}")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
