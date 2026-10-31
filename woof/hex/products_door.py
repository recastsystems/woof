"""The product door: ``woof hex products``.

One command after a forecast -- and, by default, the last stage of the
forecast door itself -- turns a run's history on ANY mesh (uniform global,
variable-resolution, refined patch) into the standard product set:

* maps through the Rust renderer, every view filed
  ``<out>/<case>/<domain>/<product>/<valid-day>/``:
  the whole sphere at the mesh's own spacing, the mesh's refined core at its
  own spacing (and a context view around it) when the mesh has one, a
  default region when it does not, and any ``--region`` asked for;
* soundings (skew-T log-p) and meteograms at points, filed
  ``<out>/<case>/points/<product>/<day>/``.

Python here is orchestration only (the project's boundary for 2.5.0): every
field the pictures show is read, remapped and drawn by the Rust binaries
``rw_mpas_convert`` (``--remap product``: linear state, area-conservative
precipitation), ``rw_wrfbatch`` and ``rw_hex_points``.  This module chooses
views, names folders and writes one manifest.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime as _dt
import html as _html
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time

from .errors import MpasPortError
from . import engines
from . import render_door


class ProductDoorError(MpasPortError):
    """A product-door refusal; the message names the breakage."""


MANIFEST_SCHEMA = "gpuwm-hex.products-manifest/v1"

#: The standard map set.  The renderer's own slugs: a member a history
#: cannot draw is named in the manifest with the catalog's reason, never
#: dropped silently (``render_door`` standard-set mode).
GLOBAL_PRODUCTS = (
    "2m_temperature_10m_winds",
    "10m_wind_speed_and_direction",
    "mslp_10m_winds",
    "500mb_absolute_vorticity_height_winds",
    "250mb_height_winds",
    "total_qpf",
    "precipitable_water",
    # The model-column vapour transport this door's history feeds.  THE BREAKAGE THIS PREVENTS (rolling 2.9.4
    # run, hex global-60km, 2026-10-21): the renderer's plain "ivt" became the 1000-300 hPa product of the
    # isobaric lanes when the capability landing joined both, and a hex history stores no isobaric IVT
    # ("not stored: ivt_1000_300_entire_atmosphere"), so the default IVT map was never drawn.
    "ivt_column",
    "ivt_vectors",
    "isotherm_height_0c",
    "300mb_height_winds",
    "simulated_ir_satellite",
    "sbcape",
    "composite_reflectivity",
    "precipitation_type",
)
#: Interval products: each is the difference (or the extreme) of several
#: stored frames, so each needs the frames its window spans.  The value is
#: (the spacing in hours every stored frame inside the window must keep, the
#: run length in hours the window needs): ``qpf_1h`` and the run/daily extremes
#: need every hour stored, ``qpf_6h`` every 6 h, ``qpf_24h`` a day of run.
#: THE BREAKAGE THIS PREVENTS (rolling 2.9.4 run, hex global-60km 24 h,
#: 2026-10-21): the default set asked for hourly windows from the row's
#: 3-hourly history, so every global run reported qpf_1h, 10m_wind_run_max
#: and the 0-24 h temperature extremes unavailable.  The door now requests a
#: window only when the stored frames can close it (windowed_for_history) and
#: names every window it did not request, with the cadence that rules it out,
#: in the manifest; the global rows store hourly where that is cheap.
WINDOWED_REQUIREMENTS = {
    "qpf_1h": (1, 1),
    "qpf_6h": (6, 6),
    "qpf_24h": (24, 24),
    "10m_wind_run_max": (1, 1),
    "2m_temp_0_24h_max": (1, 24),
    "2m_temp_0_24h_min": (1, 24),
}
WINDOWED_PRODUCTS = tuple(WINDOWED_REQUIREMENTS)
BASE_GLOBAL_PRODUCTS = GLOBAL_PRODUCTS
REGIONAL_EXTRA_PRODUCTS = (
    "2m_dewpoint_10m_winds",
    "500mb_height_winds",
    "850mb_temperature_height_winds",
    "850mb_height_winds",
    "925mb_height_winds",
    "mucape",
    "cloud_cover_levels",
)
GLOBAL_PRODUCTS = BASE_GLOBAL_PRODUCTS + WINDOWED_PRODUCTS
REGIONAL_PRODUCTS = GLOBAL_PRODUCTS + REGIONAL_EXTRA_PRODUCTS


def windowed_for_history(histories: list[Path], start: str) -> tuple[tuple[str, ...], dict[str, str]]:
    """The window products the stored frames can close, and why each other one is not requested.

    A frame's lead hour is its valid time minus the run start.  A window with spacing S and length L is
    requested when the run reaches L hours and every multiple of S up to the last stored lead is stored,
    so the renderer's every-hour / every-6-hour differencing never meets a gap.  Frames without a
    valid-time stamp leave the full set requested (the renderer names its own reasons).
    """

    parse = _parse_stamp
    stamps = [frame_stamp(path) for path in histories]
    if not histories or any(stamp is None for stamp in stamps):
        return WINDOWED_PRODUCTS, {}
    origin = parse(start)
    leads = {round((parse(stamp) - origin).total_seconds() / 3600.0, 6) for stamp in stamps}
    whole = {int(lead) for lead in leads if float(lead).is_integer() and lead >= 0}
    end = max(whole) if whole else 0
    requested, withheld = [], {}
    for slug, (spacing, length) in WINDOWED_REQUIREMENTS.items():
        missing = [h for h in range(0, end + 1, spacing) if h not in whole]
        if end < length:
            withheld[slug] = f"the run's stored frames reach F{end:03d}; this window needs {length} h"
        elif missing:
            withheld[slug] = (f"the history does not store every {spacing} h (first missing F{missing[0]:03d}); "
                              "store hourly frames (--history-every-minutes 60) to draw it")
        else:
            requested.append(slug)
    return tuple(requested), withheld

#: Where a mesh WITHOUT a refined region is also drawn regionally.  A view,
#: not a case: ground extents in km, dx from the mesh cells inside it.
DEFAULT_REGIONS = {
    "north-america": "region:40,-97,7000,4600",
    "europe": "region:50,12,5200,3800",
}

#: Points a sounding and meteogram are drawn at when none are named.  A
#: refined mesh takes those inside its refined core; a uniform mesh takes the
#: first ``DEFAULT_GLOBAL_POINT_COUNT``.
DEFAULT_POINTS = (
    ("Oklahoma City", 35.47, -97.52),
    ("Wichita", 37.69, -97.34),
    ("Dodge City", 37.76, -99.97),
    ("Kansas City", 39.10, -94.58),
    ("Amarillo", 35.22, -101.83),
    ("Tulsa", 36.15, -95.99),
    ("Denver", 39.74, -104.99),
    ("Chicago", 41.88, -87.63),
    ("Miami", 25.76, -80.19),
    ("London", 51.51, -0.13),
    ("Tokyo", 35.68, 139.69),
    ("Sydney", -33.87, 151.21),
    ("Sao Paulo", -23.55, -46.63),
    ("Nairobi", -1.29, 36.82),
    ("New Delhi", 28.61, 77.21),
)
DEFAULT_GLOBAL_POINT_COUNT = 6
EARTH_RADIUS_KM = 6371.229

#: Cities a LIMITED-AREA run takes its soundings at when none are named: the
#: ones on its mesh, nearest the domain centre first, at most
#: ``DEFAULT_REGION_POINT_COUNT``; a domain with none of them gets its centre.
#: THE BREAKAGE THIS PREVENTS (graded-plateau 1 km region, box J,
#: 2026-10-24): a regional run fell back to the world set (London, Tokyo,
#: Sydney), rw_hex_points refused London as 7,025 km off the mesh, and that
#: one refusal took every sounding with it.
REGION_CITIES = DEFAULT_POINTS + (
    ("Dallas", 32.78, -96.80), ("Houston", 29.76, -95.37), ("San Antonio", 29.42, -98.49),
    ("Austin", 30.27, -97.74), ("Lubbock", 33.58, -101.85), ("Midland", 32.00, -102.08),
    ("El Paso", 31.76, -106.49), ("Albuquerque", 35.08, -106.65), ("Phoenix", 33.45, -112.07),
    ("Las Vegas", 36.17, -115.14), ("Salt Lake City", 40.76, -111.89), ("Omaha", 41.26, -95.93),
    ("Lincoln", 40.81, -96.68), ("Topeka", 39.05, -95.68), ("Des Moines", 41.59, -93.62),
    ("Sioux Falls", 43.55, -96.73), ("Rapid City", 44.08, -103.23), ("Bismarck", 46.81, -100.78),
    ("Minneapolis", 44.98, -93.27), ("Milwaukee", 43.04, -87.91), ("St. Louis", 38.63, -90.20),
    ("Springfield", 37.21, -93.29), ("Little Rock", 34.75, -92.29), ("Fort Smith", 35.39, -94.40),
    ("Shreveport", 32.53, -93.75), ("Memphis", 35.15, -90.05), ("Nashville", 36.16, -86.78),
    ("Jackson", 32.30, -90.18), ("Birmingham", 33.52, -86.80), ("Atlanta", 33.75, -84.39),
    ("New Orleans", 29.95, -90.07), ("Tampa", 27.95, -82.46), ("Jacksonville", 30.33, -81.66),
    ("Charlotte", 35.23, -80.84), ("Raleigh", 35.78, -78.64), ("Washington", 38.91, -77.04),
    ("Philadelphia", 39.95, -75.17), ("New York", 40.71, -74.01), ("Boston", 42.36, -71.06),
    ("Pittsburgh", 40.44, -79.99), ("Cleveland", 41.50, -81.69), ("Detroit", 42.33, -83.05),
    ("Indianapolis", 39.77, -86.16), ("Columbus", 39.96, -83.00), ("Cincinnati", 39.10, -84.51),
    ("Louisville", 38.25, -85.76), ("Seattle", 47.61, -122.33), ("Portland", 45.52, -122.68),
    ("San Francisco", 37.77, -122.42), ("Los Angeles", 34.05, -118.24), ("San Diego", 32.72, -117.16),
    ("Boise", 43.62, -116.20), ("Billings", 45.78, -108.50), ("Cheyenne", 41.14, -104.82),
    ("Anchorage", 61.22, -149.90), ("Honolulu", 21.31, -157.86), ("San Juan", 18.47, -66.11),
    ("Toronto", 43.65, -79.38), ("Montreal", 45.50, -73.57), ("Winnipeg", 49.90, -97.14),
    ("Calgary", 51.05, -114.07), ("Vancouver", 49.28, -123.12), ("Monterrey", 25.69, -100.32),
    ("Mexico City", 19.43, -99.13), ("Havana", 23.11, -82.37), ("Bogota", 4.71, -74.07),
    ("Lima", -12.05, -77.04), ("Santiago", -33.45, -70.67), ("Buenos Aires", -34.60, -58.38),
    ("Rio de Janeiro", -22.91, -43.17), ("Paris", 48.86, 2.35), ("Madrid", 40.42, -3.70),
    ("Lisbon", 38.72, -9.14), ("Dublin", 53.35, -6.26), ("Amsterdam", 52.37, 4.90),
    ("Brussels", 50.85, 4.35), ("Berlin", 52.52, 13.40), ("Munich", 48.14, 11.58),
    ("Zurich", 47.38, 8.54), ("Vienna", 48.21, 16.37), ("Rome", 41.90, 12.50), ("Milan", 45.46, 9.19),
    ("Warsaw", 52.23, 21.01), ("Prague", 50.08, 14.44), ("Budapest", 47.50, 19.04),
    ("Copenhagen", 55.68, 12.57), ("Stockholm", 59.33, 18.07), ("Oslo", 59.91, 10.75),
    ("Helsinki", 60.17, 24.94), ("Athens", 37.98, 23.73), ("Istanbul", 41.01, 28.98),
    ("Kyiv", 50.45, 30.52), ("Moscow", 55.76, 37.62), ("Cairo", 30.04, 31.24),
    ("Lagos", 6.52, 3.38), ("Johannesburg", -26.20, 28.05), ("Cape Town", -33.92, 18.42),
    ("Riyadh", 24.71, 46.68), ("Dubai", 25.20, 55.27), ("Tehran", 35.69, 51.39),
    ("Karachi", 24.86, 67.01), ("Mumbai", 19.08, 72.88), ("Kolkata", 22.57, 88.36),
    ("Dhaka", 23.81, 90.41), ("Bangkok", 13.76, 100.50), ("Singapore", 1.35, 103.82),
    ("Jakarta", -6.21, 106.85), ("Manila", 14.60, 120.98), ("Hong Kong", 22.32, 114.17),
    ("Shanghai", 31.23, 121.47), ("Beijing", 39.90, 116.41), ("Seoul", 37.57, 126.98),
    ("Osaka", 34.69, 135.50), ("Taipei", 25.03, 121.57), ("Perth", -31.95, 115.86),
    ("Brisbane", -27.47, 153.03), ("Melbourne", -37.81, 144.96), ("Darwin", -12.46, 130.84),
    ("Auckland", -36.85, 174.76),
)
DEFAULT_REGION_POINT_COUNT = 8
#: The sphere rw_hex_points measures distances on (rw_mpas MPAS_EARTH_RADIUS_M).
MPAS_EARTH_RADIUS_M = 6371229.0


def mesh_point_check(mesh, points) -> tuple[list, list, tuple[float, float]] | None:
    """``(on_mesh, off_mesh, (centre_lat, centre_lon))`` for ``points`` on a mesh, or ``None``.

    The rule is rw_hex_points' own (rw_mpas ``PointSampler``): a point is off
    the mesh when its nearest cell centre is further away than that cell's
    spacing, ``sqrt(2 * areaCell / sqrt(3))`` on the unit sphere times the
    MPAS radius.  ``off_mesh`` items are ``(name, lat, lon, nearest_km,
    spacing_km)``.  ``None`` when the mesh cannot be read; the caller then
    passes the points through and rw_hex_points decides.
    """
    if mesh is None:
        return None
    try:
        import numpy as np
        from netCDF4 import Dataset

        from .netcdf_lock import NETCDF
        with NETCDF, Dataset(mesh) as grid:
            lat = np.asarray(grid["latCell"][:], dtype=np.float64)
            lon = np.asarray(grid["lonCell"][:], dtype=np.float64)
            area = np.asarray(grid["areaCell"][:], dtype=np.float64)
            radius = float(grid.getncattr("sphere_radius"))
    except (OSError, ValueError, KeyError, AttributeError, ImportError):
        return None
    if lat.size == 0 or not (math.isfinite(radius) and radius > 0.0):
        return None
    cells = np.stack((np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)), axis=1)
    mean = cells.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if norm > 0.0:
        centre = mean / norm
        centre_deg = (math.degrees(math.asin(max(-1.0, min(1.0, float(centre[2]))))),
                      math.degrees(math.atan2(float(centre[1]), float(centre[0]))))
        reach = float(np.arccos(np.clip(cells @ centre, -1.0, 1.0)).max())
    else:
        centre, centre_deg, reach = None, (0.0, 0.0), math.pi
    spacing_km = np.sqrt(2.0 * (area / (radius * radius)) / math.sqrt(3.0)) * MPAS_EARTH_RADIUS_M / 1000.0
    reach += float(spacing_km.max()) / (MPAS_EARTH_RADIUS_M / 1000.0)
    on_mesh, off_mesh = [], []
    for name, plat, plon in points:
        p, q = math.radians(plat), math.radians(plon)
        target = np.array((math.cos(p) * math.cos(q), math.cos(p) * math.sin(q), math.sin(p)))
        if centre is not None and math.acos(max(-1.0, min(1.0, float(centre @ target)))) > reach:
            # Beyond every cell plus the widest spacing: off the mesh, and the
            # distance to the domain edge is all a note needs.
            beyond = (math.acos(max(-1.0, min(1.0, float(centre @ target)))) - reach) * MPAS_EARTH_RADIUS_M / 1000.0
            off_mesh.append((name, plat, plon, None, beyond))
            continue
        dots = cells @ target
        index = int(np.argmax(dots))
        chord = float(np.linalg.norm(cells[index] - target))
        km = 2.0 * math.asin(max(0.0, min(1.0, chord / 2.0))) * MPAS_EARTH_RADIUS_M / 1000.0
        if km > float(spacing_km[index]):
            off_mesh.append((name, plat, plon, km, float(spacing_km[index])))
        else:
            on_mesh.append((name, plat, plon))
    return on_mesh, off_mesh, centre_deg


def region_default_points(check: tuple[list, list, tuple[float, float]]) -> list:
    """The cities of :data:`REGION_CITIES` on a limited-area mesh, nearest its
    centre first, or the domain centre when none is on it."""

    on_mesh, _off, (clat, clon) = check
    on_mesh = sorted(on_mesh, key=lambda p: great_circle_km(clat, clon, p[1], p[2]))
    return on_mesh[:DEFAULT_REGION_POINT_COUNT] or [("Domain centre", round(clat, 3), round(clon, 3))]

POINTS_ENV = engines.POINTS.env_names[0]


def add_label_arguments(parser) -> None:
    """``--model-label`` and ``--brand`` on a door that ends in the product
    stage, forwarded to ``woof hex products`` by :func:`label_argv`.

    THE BREAKAGE THIS PREVENTS (WOOF sounding labels, 2026-10-23): the
    products door took ``--model-label`` but ``woof hex global`` (and the
    region, variable-resolution, storm-following and ensemble doors) had no
    way to pass it, so every point sounding a host launched through a door
    said "hex 121 km" top left with nothing top right, whatever the host
    called the model.  Unset, nothing is forwarded and the products door
    keeps its own defaults."""

    parser.add_argument(
        "--model-label", default=None, metavar="NAME",
        help="the model the maps and soundings name, top left (e.g. 'WOOF Hex'; "
             "the mesh spacing is added; a map's row takes 32 characters, a longer "
             "name ends in its whole words and '...'); forwarded to the product "
             "stage. Default: hex")
    parser.add_argument(
        "--brand", default=None, metavar="TEXT",
        help="bold text at the top right of every sounding and meteogram (e.g. "
             "'WOOF Hex · Global 120 km · Init GFS 22 Oct 2026 00Z'); forwarded "
             "to the product stage. Default: composed from the run's own records "
             "(shape, finest spacing, init source and time)")
    add_view_label_argument(parser, forwarded=True)


def add_view_label_argument(parser, *, forwarded: bool = False) -> None:
    """``--view-label VIEW=TEXT``: a view's map headline in a caller's words."""

    parser.add_argument(
        "--view-label", action="append", default=None, metavar="VIEW=TEXT",
        help="the headline of one view's maps (repeatable), e.g. 'refined-core=Core zoom 8-38 km'; "
             "'{spacing}' in TEXT becomes the spacing of the mesh cells the picture samples, "
             "'·' and '→' are drawn as ' ' and '-', at most 32 characters"
             + ("; forwarded to the product stage" if forwarded else "")
             + ". Default: the view's name (Core zoom, Around the core, Whole globe, Region, "
               "Moving core, or the view's own name) and the spacing the picture samples")


def label_argv(args) -> list[str]:
    """The ``woof hex products`` flags a door forwards for its labels."""

    argv: list[str] = []
    if getattr(args, "model_label", None):
        argv += ["--model-label", str(args.model_label)]
    if getattr(args, "brand", None):
        argv += ["--brand", str(args.brand)]
    for item in getattr(args, "view_label", None) or ():
        argv += ["--view-label", str(item)]
    return argv


# ---------------------------------------------------------------------------
# each view's headline name
# ---------------------------------------------------------------------------
#: What each view is called in its pictures' headline: the words the WOOF
#: site's grid menu uses for it (the site grid menu names, gridName and
#: messages productBrowser.grids), so the picture and the menu say one thing.
#: Any other view (north-america, europe, a --region NAME) is called by its
#: own name in title case, as the menu calls it (viewTitle).
VIEW_TITLES = {
    "refined-core": "Core zoom",
    "refined-context": "Around the core",
    "event": "Around the point",
    "region": "Region",
    "global": "Whole globe",
}
#: A storm-following run's core, which moves with what it follows.
MOVING_CORE_TITLE = "Moving core"
#: The widest text the converter puts in place of ``{spacing}`` ("12.5-128 km"):
#: a view's name is cut to its whole words that leave this much room.
SPACING_TEXT_ROOM = len("12.5-128 km")


def view_title(view: str, *, moving: bool = False) -> str:
    """The headline a view's pictures carry, for the converter to finish:
    ``Core zoom {spacing}`` -> ``Core zoom 8-38 km``, where the converter
    reads the spacing off the mesh cells that picture samples, finest first
    (rw_mpas_convert --grid-label).

    THE BREAKAGE THIS PREVENTS (2026-10-30): every view of a zoom globe
    was headlined by the WHOLE mesh's range and the picture grid's spacing,
    so the site's ``Core zoom · 8 → 38 km`` opened a picture titled ``(hex
    133-6.9 km) ... Δx 8.063 km`` (run_713c9e1913164e02) and ``Around the core
    · 6 → 96 km`` one saying ``Δx 6.08 km`` (run_c108e049fecb4bd0)."""

    if moving and view == "refined-core":
        name = MOVING_CORE_TITLE
    else:
        name = VIEW_TITLES.get(view) or " ".join(
            word[:1].upper() + word[1:] for word in re.split(r"[-_\s]+", view) if word)
    words = name.split()
    token = render_door.GRID_LABEL_SPACING_TOKEN
    while words and len(" ".join(words)) + 1 + SPACING_TEXT_ROOM > render_door.GRID_LABEL_MAX_CHARS:
        words.pop()
    return f"{' '.join(words)} {token}" if words else token


def caller_view_label(text: str) -> str:
    """A caller's headline for a view (``--view-label VIEW=TEXT``) as the
    picture draws it: the menu's separators made plain (``Core zoom · 8 → 38
    km`` -> ``Core zoom 8-38 km``: the map font draws ASCII), whitespace made
    one space, and held to the headline's limits.  Refused by name rather
    than dropped: rw_wrfbatch draws no label longer than 32 characters or
    holding '|', and the converter refuses one."""

    label = " ".join(str(text).replace("\u00b7", " ").split())
    label = re.sub(r"\s*\u2192\s*", "-", label)
    if not label:
        raise ProductDoorError("--view-label needs text after VIEW=")
    if (len(label.replace(render_door.GRID_LABEL_SPACING_TOKEN, "0-000 km"))
            > render_door.GRID_LABEL_MAX_CHARS):
        raise ProductDoorError(
            f"--view-label {label!r} is longer than {render_door.GRID_LABEL_MAX_CHARS} characters, "
            f"the map headline's limit")
    if any(ord(ch) < 32 or ch == "|" for ch in label):
        raise ProductDoorError(
            f"--view-label {label!r} holds a control character or '|', the map metadata row's separator")
    return label


def view_labels(items) -> dict[str, str]:
    """``--view-label VIEW=TEXT`` items as ``{view: headline}``; a view named
    twice keeps its last text, as a repeated flag does."""

    labels: dict[str, str] = {}
    for item in items or ():
        view, sep, text = str(item).partition("=")
        if not sep:
            raise ProductDoorError(f"--view-label {item!r} is not VIEW=TEXT (refined-core=Core zoom 8-38 km)")
        labels[render_door.checked_view(view.strip())] = caller_view_label(text)
    return labels


def points_take_brand(points_exe: Path) -> bool:
    """Whether this ``rw_hex_points`` draws ``--brand``.

    THE BREAKAGE THIS PREVENTS: a ``rw_hex_points`` built before the flag
    refuses any flag it does not know ("unknown flag --brand"), so passing
    it unconditionally would cost every point picture on an older engine.
    Such a build draws without the brand and the products log says so."""

    try:
        probe = subprocess.run([str(points_exe), "--help"], capture_output=True,
                               text=True, errors="replace", timeout=60)
    except (OSError, subprocess.SubprocessError):
        return False
    return "--brand" in (probe.stdout or "") + (probe.stderr or "")


# ---------------------------------------------------------------------------
# the sounding brand, from the run's own records
# ---------------------------------------------------------------------------
#: The model a hex sounding names top right.  A sounding never shows a default
#: tool name or nothing there; the header comes from the run's own metadata on
#: every path.
BRAND_MODEL = "WOOF Hex"
#: The brand with nothing known about the run, not even the model.
BRAND_FALLBACK = "WOOF"
BRAND_SHAPES = {
    "global": "Global",
    "variable-resolution": "Variable-resolution globe to",
    "region": "Region",
    "storm-following": "Storm-following",
}
_BRAND_SHAPES_WITHOUT_SPACING = {"variable-resolution": "Variable-resolution globe"}
#: Words a source's display name may lead with that name its agency, not the model.
_SOURCE_AGENCIES = {"NCEP", "ECMWF", "DWD", "NOAA"}
_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def spacing_text(km, *, measured: bool = False) -> str | None:
    """``3``, ``120``, ``3.75``: a spacing in km as the brand writes it.

    A row's declared spacing is written as declared.  A spacing measured
    from the grid's edges is not a round number (a 120 km globe's median
    edge is 120.4 km), so it is written to the kilometre at 10 km and up,
    and to the nearest quarter kilometre below when it is that close."""

    try:
        km = float(km)
    except (TypeError, ValueError):
        return None
    if not (km > 0.0) or km != km or km == float("inf"):
        return None
    if measured:
        if km >= 10.0:
            km = float(round(km))
        else:
            quarter = round(km * 4.0) / 4.0
            km = quarter if quarter > 0 and abs(km - quarter) <= 0.05 * km else round(km, 2)
    return f"{round(km, 4):g}"


def brand_time(start: str | None) -> str | None:
    """``2026-10-23_18:00:00`` -> ``23 Oct 2026 18Z`` (``18:30Z`` off the hour)."""

    if not start:
        return None
    text = str(start).strip().replace("T", "_").rstrip("Z")
    for layout in ("%Y-%m-%d_%H:%M:%S", "%Y-%m-%d_%H:%M", "%Y-%m-%d_%H", "%Y%m%d%H"):
        try:
            when = _dt.datetime.strptime(text, layout)
            break
        except ValueError:
            continue
    else:
        return None
    hour = f"{when.hour:02d}" if when.minute == 0 else f"{when.hour:02d}:{when.minute:02d}"
    return f"{when.day} {_MONTHS[when.month - 1]} {when.year} {hour}Z"


def default_brand(*, shape: str | None = None, spacing_km=None, source: str | None = None,
                  start: str | None = None, model: str | None = BRAND_MODEL,
                  spacing_measured: bool = False) -> str:
    """The top-right text of a sounding when no ``--brand`` is given:
    ``WOOF Hex · Storm-following 3 km · Init GFS 23 Oct 2026 18Z``.

    ``shape`` is one of :data:`BRAND_SHAPES` (global row, variable-resolution
    globe, limited-area region, storm-following).  A part the run does not
    record is dropped, never guessed: no source gives ``Init 23 Oct 2026
    18Z``, nothing at all gives ``WOOF Hex``, and no model ``WOOF``.  Every
    door that ends in the product stage gets its brand here, so none can
    fall back to an empty top right."""

    parts = [str(model).strip() if model and str(model).strip() else BRAND_FALLBACK]
    km = spacing_text(spacing_km, measured=spacing_measured)
    if shape in BRAND_SHAPES:
        parts.append(f"{BRAND_SHAPES[shape]} {km} km" if km
                     else _BRAND_SHAPES_WITHOUT_SPACING.get(shape, BRAND_SHAPES[shape]))
    elif km:
        parts.append(f"{km} km")
    init = " ".join(p for p in (source, brand_time(start)) if p)
    if init:
        parts.append(f"Init {init}")
    return " · ".join(parts)


def _source_short_name(display_name: str) -> str | None:
    """``NCEP GFS 0.25 degree, cycle step 0`` -> ``GFS``; ``ECMWF IFS ENS open
    data ...`` -> ``IFS ENS``: the model words of a source row's display name."""

    words = display_name.split()
    while words and words[0] in _SOURCE_AGENCIES:
        words = words[1:]
    kept: list[str] = []
    for word in words:
        bare = word.rstrip(",;:")
        if not bare or not bare[0].isupper():
            break
        kept.append(bare)
        if bare != word:
            break
    return " ".join(kept) or None


def brand_source(text: str | None, table: dict | None = None) -> str | None:
    """The init source a receipt's ``init.source`` names, as the brand
    writes it (``GFS``, ``ERA5``, ``HRRR``), or ``None``.

    The doors write a source row's id or display name first (``gfs
    2026-10-23T18; global row ...``, ``gfs analysis; moving run member
    m0``), so the name is looked up in the init-source table; text that
    names no row (``staged analysis``) gives no source rather than a guess."""

    if not text:
        return None
    if table is None:
        try:
            from . import init_sources
            table = init_sources.load_table()
        except Exception:  # noqa: BLE001 - no table, no source named
            return None
    rows = table.get("sources") or ()
    lowered = str(text).strip().lower()
    by_display = [row for row in rows if row.get("display_name")
                  and lowered.startswith(str(row["display_name"]).lower())]
    if by_display:
        row = max(by_display, key=lambda r: len(r["display_name"]))
        return _source_short_name(row["display_name"])
    token = re.split(r"[\s;,]+", lowered, maxsplit=1)[0].replace("_", "-")
    for row in rows:
        if token in {str(n).strip().lower().replace("_", "-")
                     for n in (row.get("id"), *row.get("aliases", ())) if n}:
            return _source_short_name(str(row.get("display_name") or "")) or str(row["id"]).upper()
    return None


def moving_controller(forecast_dir: Path | None) -> dict | None:
    """The moving run's ``controller.json`` when ``forecast_dir`` is one of
    its segments (``<run>/segment-NNN/fc``), else ``None``."""

    if forecast_dir is None:
        return None
    segment = Path(forecast_dir).resolve().parent
    if not segment.name.startswith("segment-"):
        return None
    try:
        state = json.loads((segment.parent / "controller.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return state if isinstance(state, dict) and "segments" in state else None


def _mesh_row_spacing(mesh_name: str | None, mesh: Path | None,
                      extra_rows=()) -> tuple[float | None, float | None]:
    """(finest, coarsest) km a registered mesh row declares for this run's mesh."""

    try:
        from . import mesh_rows
        files = [*mesh_rows.row_files(), *[Path(p) for p in extra_rows if p]]
        rows = [row for location in files if Path(location).is_file()
                for row in mesh_rows.read_rows(location)]
    except Exception:  # noqa: BLE001 - an unreadable row file names no spacing
        rows = []
    grid = str(mesh) if mesh is not None else None
    for row in rows:
        if (mesh_name and row.name == mesh_name) or (grid and row.grid == grid):
            fine = row.fine_dx_m or row.nominal_dx_m
            return (float(fine) / 1000.0 if fine else None,
                    float(row.background_km) if row.background_km else None)
    if mesh_name:
        try:
            from .forecast_door import load_registry
            row = load_registry().get(mesh_name)
        except Exception:  # noqa: BLE001
            row = None
        if row is not None and row.nominal_dx_m:
            coarse = row.coarsest_dx_m or row.nominal_dx_m
            return float(row.nominal_dx_m) / 1000.0, float(coarse) / 1000.0
    return None, None


def _mesh_file_spacing(mesh: Path | None) -> tuple[float | None, float | None]:
    """(finest, coarsest) km measured from the grid's own ``dcEdge``: the
    median edge on a uniform mesh (its nominal spacing), the shortest on a
    variable one."""

    if mesh is None:
        return None, None
    try:
        import numpy as np
        from netCDF4 import Dataset

        from .netcdf_lock import NETCDF
        with NETCDF, Dataset(mesh) as grid:
            if "dcEdge" not in grid.variables:
                return None, None
            edges = np.asarray(grid["dcEdge"][:], dtype=np.float64)
    except Exception:  # noqa: BLE001 - not a readable grid
        return None, None
    edges = edges[edges > 0.0]
    if edges.size == 0:
        return None, None
    low, high = float(edges.min()), float(edges.max())
    fine = float(np.median(edges)) if high < 2.0 * low else low
    return fine / 1000.0, high / 1000.0


def init_source_receipt(init: Path | None) -> dict:
    """The ``init-source-receipt.json`` the init-source door wrote beside
    the run's init (``source`` row id, ``cycle``), or ``{}``."""

    if init is None:
        return {}
    try:
        doc = json.loads((Path(init).parent / "init-source-receipt.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return doc if isinstance(doc, dict) else {}


def run_brand_metadata(*, forecast_dir: Path | None, receipt: dict, mesh: Path | None,
                       limited_area: bool, uniform: bool | None,
                       start: str | None, init: Path | None = None) -> dict:
    """What a run records about itself for its brand: shape, finest spacing,
    init source and start, each ``None`` when the run does not record it.

    * shape: a segment of a moving run (its ``controller.json``) is
      storm-following; a limited-area mesh a region; else the refined-core
      view's verdict (``uniform``), else the declared or measured spacing
      range, decides global row against variable-resolution globe.
    * spacing: the mesh row the run was bound to (``$WOOF_HEX_MESH_ROWS``,
      a moving run's member rows, the built-in registry), else the grid.
    * source: the receipt's ``init.source``, through the init-source table,
      else the init-source receipt beside the init (a first look draws
      before the forecast receipt exists).
    * start: ``--simulation-start`` or the receipt's ``schedule.start_time``,
      else that init-source receipt's cycle."""

    controller = moving_controller(forecast_dir)
    member_rows = [m.get("mesh_rows") for m in ((controller or {}).get("members") or {}).values()
                   if isinstance(m, dict)]
    mesh_name = receipt_text(receipt, "mesh", "name")
    fine, coarse = _mesh_row_spacing(mesh_name, mesh, member_rows)
    measured = False
    if fine is None:
        fine, coarse = _mesh_file_spacing(mesh)
        measured = fine is not None
    if controller is not None:
        shape = "storm-following"
    elif limited_area:
        shape = "region"
    else:
        if uniform is None and fine and coarse:
            uniform = coarse < 2.0 * fine
        shape = None if uniform is None else ("global" if uniform else "variable-resolution")
    source = brand_source(receipt_text(receipt, "init", "source"))
    if source is None or not start:
        made = init_source_receipt(init)
        source = source or brand_source(made.get("source") if isinstance(made.get("source"), str) else None)
        start = start or (made.get("cycle") if isinstance(made.get("cycle"), str) else None)
    return {"shape": shape, "spacing_km": fine, "spacing_measured": measured,
            "source": source, "start": start}


def great_circle_km(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    c = math.sin(p1) * math.sin(p2) + math.cos(p1) * math.cos(p2) * math.cos(dl)
    return EARTH_RADIUS_KM * math.acos(max(-1.0, min(1.0, c)))


#: The request the MPAS suite route (woof.hex.mpasphys.forecast_v842) writes beside its stream files.
NATIVE_REQUEST = "native-physics-request.json"


def native_request(forecast_dir: Path | None) -> dict:
    """The MPAS suite route's own request for a run folder, or ``{}``."""

    if forecast_dir is None:
        return {}
    try:
        doc = json.loads((Path(forecast_dir) / NATIVE_REQUEST).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return doc if isinstance(doc, dict) else {}


def native_output_frames(forecast_dir: Path) -> list[Path]:
    """The frames of the MPAS suite route's ``output`` stream, from the template its request names.

    THE BREAKAGE THIS PREVENTS: ``woof hex products RUN_DIR`` refused every MPAS suite run, because that
    route writes MPAS's own stream files (``output.$Y-$M-$D_$h.$m.$s.nc`` in a native deck,
    ``history.$Y-...`` from the Registry default) and the door only looked for ``cuda-history.*``/``history.*``
    (hexmx proof, 2026-10-16).  The stream's filename template is read from the run's own request, so any
    deck's naming is found without a guess.
    """

    for stream in native_request(forecast_dir).get("streams") or ():
        if not isinstance(stream, dict) or stream.get("name") != "output":
            continue
        template = str(stream.get("filename") or "")
        if not template or "output" not in (stream.get("directions") or ()):
            continue
        pattern = re.sub(r"\$[YMDhms]", "*", template)
        return sorted(p for p in Path(forecast_dir).glob(pattern) if p.is_file())
    return []


def histories_without_surface_pressure(histories: list[Path]) -> list[Path]:
    """The source's frames when its first carries neither ``surface_pressure`` nor ``PSFC``, else ``[]``.

    One header read by the Rust reader: every frame of a run comes from the same stream, so the first
    answers for all of them.
    """

    from woof.netcdf_bridge import open_dataset

    if not histories:
        return []
    try:
        names = set(open_dataset(Path(histories[0])).variables)
    except Exception:
        return []  # unreadable here: rw_hex_points names the defect itself
    return [] if names & {"surface_pressure", "PSFC"} else [Path(p) for p in histories]


def discover_histories(source: Path) -> list[Path]:
    """History frames in a forecast output folder, valid-time order."""

    source = Path(source)
    found = (sorted(source.glob("cuda-history.*.nc")) or sorted(source.glob("history.*.nc"))
             or native_output_frames(source))
    if not found:
        raise ProductDoorError(
            f"{source} holds no cuda-history.*.nc or history.*.nc frame, nor the "
            f"output stream a {NATIVE_REQUEST} names; the "
            f"product door draws history frames and there are none to draw. "
            f"Point --forecast at the forecast door's --out folder or pass "
            f"--history.")
    return found


_STAMP = re.compile(r"(\d{4})-(\d{2})-(\d{2})_(\d{2})[.:](\d{2})[.:](\d{2})")


def frame_stamp(path: Path) -> str | None:
    m = _STAMP.search(Path(path).name)
    if not m:
        return None
    y, mo, d, h, mi, s = m.groups()
    return f"{y}-{mo}-{d}_{h}:{mi}:{s}"


#: The maps a viewer opens on, in this order: fields that have content in every frame, the start included.
#: THE BREAKAGE THIS PREVENTS (hex 1 km LAM fast start, GFS 2025-05-19 18Z over Oklahoma City, 2026-10-23): a
#: cold start carries no hydrometeors, so the f000 composite reflectivity -- drawn 12.7 s into the forecast and
#: the map a viewer showed first -- was an empty map.  Reflectivity joins the opening maps from the first frame
#: that has echoes (``first_maps``), and a viewer that opens it starts on that frame.
FIRST_MAPS = ("2m_temperature_10m_winds", "10m_wind_speed_and_direction", "sbcape")
ECHO_MAP = "composite_reflectivity"
#: The weakest reflectivity that counts as an echo (dBZ): the first step of the renderer's reflectivity scale.
ECHO_DBZ = 10.0
#: Reflectivity as a history stores it: the 2-D column maximum where the run writes one, else the 3-D field.
ECHO_FIELDS = ("refl10cm_max", "refl10cm")
#: Values per read of the reflectivity scan, so a fine global mesh is read in cell blocks, not in one array.
ECHO_BLOCK_VALUES = 16_000_000


def _parse_stamp(stamp: str) -> _dt.datetime:
    return _dt.datetime.strptime(stamp.replace("T", "_").replace(".", ":")[:19], "%Y-%m-%d_%H:%M:%S")


def diag_companion(path: Path) -> Path | None:
    """The ``diag`` stream file written at the same time as a history frame, when it is on disk.

    The converter's rule (rw-mpas ``diag_companion``): the last ``history`` token of the name becomes ``diag``,
    in the same folder (``history.2025-05-19_13.00.00.nc`` -> ``diag.2025-05-19_13.00.00.nc``).
    """

    name = Path(path).name
    at = name.rfind("history")
    if at < 0:
        return None
    companion = Path(path).with_name(name[:at] + "diag" + name[at + len("history"):])
    return companion if companion.is_file() else None


def frame_echo_dbz(path: Path) -> float | None:
    """The strongest reflectivity in one history frame (dBZ), read by the Rust reader; None when none is stored.

    THE BREAKAGE THIS PREVENTS (case study 2025-05-19, H2, 2026-10-24): the MPAS-physics route writes MPAS-A's
    ``output`` stream as its history, which carries no reflectivity; MPAS writes ``refl10cm_max`` to the
    ``diag`` stream of the same time.  Reading the history alone found no echo in any frame, so the viewer never
    offered the reflectivity map.  A field the history lacks is read from its diag companion.
    """

    peak = _echo_dbz_in(Path(path))
    if peak is None:
        companion = diag_companion(Path(path))
        if companion is not None:
            peak = _echo_dbz_in(companion)
    return peak


def _echo_dbz_in(path: Path) -> float | None:
    import numpy as np
    from woof.netcdf_bridge import open_dataset

    dataset = open_dataset(Path(path))
    for name in ECHO_FIELDS:
        variable = dataset.variables.get(name)
        if variable is None or variable.size == 0:
            continue
        if variable.ndim < 2 or variable.size <= ECHO_BLOCK_VALUES:
            blocks = [np.asarray(variable[...], dtype=float)]
        else:
            # (Time, nCells, nVertLevels): blocks of cells, every level, so the peak is the column maximum.
            rows, columns = variable.shape[-2], variable.shape[-1]
            outer = max(1, variable.size // (rows * columns))
            step = max(1, ECHO_BLOCK_VALUES // (columns * outer))
            blocks = (np.asarray(variable.read_window(i0=0, ni=columns, j0=j0, nj=min(step, rows - j0)), dtype=float)
                      for j0 in range(0, rows, step))
        peak = None
        for block in blocks:
            finite = block[np.isfinite(block)]
            if finite.size:
                peak = float(finite.max()) if peak is None else max(peak, float(finite.max()))
        return peak
    return None


def first_maps(histories: list[Path], start: str, drawn) -> dict:
    """The maps a viewer opens on: ``FIRST_MAPS`` that were drawn, then reflectivity from its first echo.

    Frames are read in valid-time order and the scan stops at the first frame with an echo of ``ECHO_DBZ`` or
    more, so a run with storms at the start reads one frame.  A frame that cannot be read ends the scan with a
    note; nothing here can fail the products.
    """

    from .render_door import lead_token

    drawn = set(drawn or ())
    record: dict = {"maps": [slug for slug in FIRST_MAPS if slug in drawn], "echo_map": ECHO_MAP,
                    "echo_dbz": ECHO_DBZ, "reflectivity_from": None, "reflectivity_frame": None,
                    "reflectivity_peak_dbz": None}
    if ECHO_MAP not in drawn:
        return record
    for path in histories:
        try:
            peak = frame_echo_dbz(path)
        except Exception as error:  # the scan only orders the opening maps; the maps themselves are drawn
            record["note"] = f"reflectivity scan stopped at {Path(path).name}: {type(error).__name__}: {error}"
            break
        if peak is None or peak < ECHO_DBZ:
            continue
        stamp = frame_stamp(path)
        if stamp is None:
            record["note"] = f"{Path(path).name} has echoes but no valid-time stamp in its name"
            break
        seconds = (_parse_stamp(stamp) - _parse_stamp(start)).total_seconds()
        record.update(reflectivity_from=lead_token(int(round(seconds)), Path(path).name),
                      reflectivity_frame=Path(path).name, reflectivity_peak_dbz=round(peak, 1))
        record["maps"].append(ECHO_MAP)
        break
    return record


#: The receipt the forecast door writes beside its frames.
FORECAST_RECEIPT = "forecast-receipt.json"


def forecast_receipt(forecast_dir: Path | None) -> dict:
    """The forecast door's own receipt for a run folder, or ``{}``.

    It names the grid the run was bound to, the init it started from and its
    start time, which is everything this door needs besides the frames -- so
    after a forecast the command is ``woof hex products RUN/fc`` and nothing
    the forecast already knew has to be typed again."""

    if forecast_dir is None:
        return {}
    try:
        doc = json.loads((Path(forecast_dir) / FORECAST_RECEIPT).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        doc = None
    if isinstance(doc, dict):
        return doc
    # An MPAS suite run writes no forecast-receipt.json; its request names the same three things: the grid
    # (else the init, which carries the whole mesh), the init its input stream read, and its start.
    request = native_request(forecast_dir)
    if not request:
        return {}
    init = next((s.get("input_file") for s in request.get("streams") or ()
                 if isinstance(s, dict) and s.get("name") == "input" and s.get("input_file")), None)
    init = request.get("init") or init
    grid = request.get("grid") or init
    start = request.get("start")
    synthesized: dict = {}
    if grid:
        synthesized["mesh"] = {"grid": str(grid)}
    if init:
        synthesized["init"] = {"path": str(init)}
    if start:
        synthesized["schedule"] = {"start_time": str(start)}
    return synthesized


def mesh_is_limited_area(mesh) -> bool:
    """A mesh with boundary rings (bdyMaskCell > 0) is a limited-area mesh.

    THE BREAKAGE THIS PREVENTS (hex fast start, 2026-10-23): the first map of a 937.5 m LAM, drawn
    from its f000 frame with --history while the forecast ran (no receipt yet), also rendered a 2,030 km
    context view of a 405 km domain: 152 s (a 10.9 GB conversion) beside the 19 s mesh view.
    """
    if mesh is None:
        return False
    try:
        from netCDF4 import Dataset

        from .netcdf_lock import NETCDF
        with NETCDF, Dataset(mesh) as grid:
            return "bdyMaskCell" in grid.variables and int(grid["bdyMaskCell"][:].max()) > 0
    except (OSError, ValueError, KeyError):
        return False


def receipt_text(receipt: dict, *keys: str) -> str | None:
    """A nested string of the receipt (``mesh`` -> ``grid``), or ``None``."""

    value = receipt
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value if isinstance(value, str) and value else None


def simulation_start(explicit: str | None, forecast_dir: Path | None,
                     histories: list[Path]) -> str:
    """Run start: as given, else the forecast receipt's, else the earliest
    frame (a forecast writes its initial state as its first frame)."""

    if explicit:
        return explicit
    if forecast_dir is not None:
        value = receipt_text(forecast_receipt(forecast_dir), "schedule", "start_time")
        if value and _STAMP.search(value.replace("T", "_")):
            return value
        for name in (FORECAST_RECEIPT, "cuda-v841-forecast-receipt.json"):
            path = Path(forecast_dir) / name
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            for key in ("start_time", "simulation_start"):
                value = doc.get(key) or (doc.get("request") or {}).get(key)
                if isinstance(value, str) and _STAMP.search(value.replace("T", "_")):
                    return value
    stamp = frame_stamp(histories[0])
    if stamp is None:
        raise ProductDoorError(
            f"no run start: {histories[0].name} carries no valid-time stamp and "
            f"no --simulation-start was given, so lead hours cannot be numbered")
    return stamp


def case_token(explicit: str | None, mesh: Path, start: str,
               mesh_name: str | None = None) -> str:
    """``<mesh-name>-<YYYYMMDDHH>`` unless named: read from the data, never a
    hardcoded case.  The mesh name is the one the forecast ran under when its
    receipt gives one (``mesh_name``), else the grid file's own name, else
    the folder that holds it -- a grid stored as ``ordered.grid.nc`` has no
    name of its own, and every such run was filed as ``mesh-<date>``
    (measured 2026-10-12 on the 30 km global case)."""

    if explicit:
        token = explicit
    else:
        stem = Path(mesh).name
        for suffix in (".nc", ".grid", ".static", ".init", ".ordered"):
            stem = stem.replace(suffix, "")
        stem = stem.replace("ordered", "").strip("._-")
        stem = (mesh_name or "").strip() or stem or Path(mesh).parent.name or "mesh"
        day = re.sub(r"[^0-9]", "", start)[:10]
        token = f"{stem}-{day}"
    token = re.sub(r"[^A-Za-z0-9._-]+", "-", token).strip("-")
    if not token:
        raise ProductDoorError("--case reduces to an empty folder name")
    return token


def resolve_points_exe(explicit: str | None, convert_exe: Path) -> Path:
    """``rw_hex_points``: as named, else the build that sits beside the
    converter in use (the three are built together, and a sounding should
    come from the same build as the map under it), else the shared engine
    ladder every door of this distribution reads."""

    named = explicit or any(os.environ.get(name) for name in
                            (*engines.POINTS.env_names, engines.POINTS.gpuwm_env))
    if not named:
        beside = Path(convert_exe).with_name(engines.POINTS.name)
        if beside.is_file():
            return beside
    try:
        return engines.resolve(engines.POINTS, explicit)
    except engines.EngineRefusal as error:
        raise ProductDoorError(
            f"{error}\n  # or draw the maps without the point pictures:\n"
            f"  woof hex products ... --no-points") from None


def parse_point_text(text: str) -> list[tuple[str, float, float]]:
    out = []
    for item in [s.strip() for s in text.split(";") if s.strip()]:
        name, _, coords = item.partition("@")
        lat, _, lon = coords.partition(",")
        try:
            out.append((name.strip(), float(lat), float(lon)))
        except ValueError:
            raise ProductDoorError(f"--points item {item!r} is not NAME@LAT,LON") from None
    return out


#: A mesh whose 95th-percentile cell spacing is more than this many times its 5th is variable-resolution (a zoom
#: globe or any refined mesh); a quasi-uniform mesh is within about 1.5.
VARIABLE_RESOLUTION_SPACING_RATIO = 2.0


def mesh_cell_spacing_km(mesh) -> "list[float] | None":
    """Each cell's spacing (km) from its area as a regular hexagon, or None when the grid cannot be read."""

    if mesh is None:
        return None
    try:
        from netCDF4 import Dataset

        from .netcdf_lock import NETCDF
        with NETCDF, Dataset(mesh) as grid:
            area = grid["areaCell"][:]
            radius = float(getattr(grid, "sphere_radius", 1.0) or 1.0)
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return None
    scale = 6371.229 / radius if radius < 1000.0 else 0.001  # a unit sphere's areas, else metres
    return [math.sqrt(2.0 * float(a) / math.sqrt(3.0)) * scale for a in area]


def global_window(mesh, size: str) -> tuple[str, str | None]:
    """The global view's window and, when it is not the mesh's own, why.

    ``global`` (the whole sphere at the mesh's median spacing) is right for a quasi-uniform mesh: the globe is its
    only view and the median is its spacing. A variable-resolution mesh's median is set by its refined cells, so its
    whole sphere would be converted at the refined spacing to fill one picture. THE BREAKAGE THIS PREVENTS (an
    Arizona 96 -> 3 km zoom globe, 800 km core, 36 h, 8 x RTX PRO 6000, woof hex 0.4.10, 2026-10-30): the median
    is the refined cells' few km, so the converter laid the sphere down at its 9-million-point ceiling, a 4240 x 2121
    grid at 9.4 km (11 GB of converted frame, 117 s of conversion a frame), drawn into an 1800 x 1000 picture of mostly
    48-96 km cells, and the globe was still converting 38 min after the forecast ended. Such a mesh's globe is drawn on the lat-lon grid of its picture's pixels (0.2 degrees
    across 1800 pixels); its refined region's own detail is the refined-core and refined-context views. Never finer
    than the mesh: a mesh whose median is coarser than a pixel keeps ``global``."""

    try:
        width = int(str(size).lower().split("x", 1)[0])
    except ValueError:
        return "global", None
    if width < 2 or width % 2:
        return "global", None
    spacing = mesh_cell_spacing_km(mesh)
    if not spacing:
        return "global", None
    ordered = sorted(spacing)
    low, high = ordered[len(ordered) // 20], ordered[(19 * len(ordered)) // 20]
    if low <= 0.0 or high / low < VARIABLE_RESOLUTION_SPACING_RATIO:
        return "global", None
    pixel_degrees = 360.0 / width
    median_degrees = ordered[len(ordered) // 2] / (math.pi * 6370.0 / 180.0)
    if median_degrees >= pixel_degrees:
        return "global", None
    east = 180.0 - pixel_degrees
    window = f"latlon:-90,90,-180,{east:.6g},{pixel_degrees:.6g}"
    why = (f"variable-resolution mesh (cell spacing {low:.1f}-{high:.1f} km, median {ordered[len(ordered) // 2]:.1f} km):"
           f" the globe drawn at its {width}-pixel picture's {pixel_degrees:.3g} degrees")
    return window, why


#: Below this |latitude| the converter frames a region on lat-lon, not Lambert (rw-mpas
#: MESH_FOCUS_MIN_LAMBERT_LATITUDE_DEG); the context view there keeps the converter's own region window.
REGION_MIN_LAMBERT_LATITUDE = 10.0


def context_window(lat: float, lon: float, extent_km: float, size: str, core_dx_m: float | None) -> tuple[str, str | None]:
    """The refined-context view's window and, when it is not the converter's own region window, why.

    ``region:`` takes its dx from the median of the mesh cells inside it, which around a refined core is the core's
    own spacing, so the converter lays the whole context box down at the core's spacing (or at its 2000-points-a-side
    ceiling). THE BREAKAGE THIS PREVENTS (an Arizona 96 -> 3 km zoom globe, 800 km core, 36 h, 8 x RTX PRO 6000,
    woof hex 0.4.10, 2026-10-30): the 6,850 x 4,795 km context box was a 2000 x 1401 grid at 3.4 km (3.3 GB of
    converted frame, 40 s of conversion a frame) drawn into a 1400 x 1000 picture, 4.9 km a pixel, of a box that is
    the 6-48 km rings outside its middle. The same box is converted on the Lambert grid the converter would lay down
    with the picture's pixel as its dx (same centre, true latitudes and stand_lon), which is all one picture shows;
    the core's own detail is the refined-core view. Never finer than the core's window, and the converter's own
    window near the equator, where it frames regions on lat-lon."""

    region = f"region:{lat:.3f},{lon:.3f},{extent_km:.0f},{0.7 * extent_km:.0f}"
    try:
        width = int(str(size).lower().split("x", 1)[0])
    except ValueError:
        return region, None
    clat, clon = round(lat, 3), round(lon, 3)
    if width < 2 or abs(clat) < REGION_MIN_LAMBERT_LATITUDE:
        return region, None
    width_m, height_m = round(extent_km) * 1000.0, round(0.7 * extent_km) * 1000.0
    dx = width_m / (width - 1)
    if core_dx_m is not None and dx <= core_dx_m:
        return region, None
    nx = math.ceil(width_m / dx - 1e-9) + 1
    ny = math.ceil(height_m / dx - 1e-9) + 1
    hemi = 1.0 if clat >= 0.0 else -1.0
    half = min(5.0, 0.5 * abs(clat))
    truelat1, truelat2 = hemi * (abs(clat) - half), hemi * min(abs(clat) + half, 89.0)
    window = f"lambert:{clat!r},{clon!r},{dx!r},{nx},{ny},{truelat1!r},{truelat2!r},{clon!r}"
    why = (f"the {extent_km:.0f} x {0.7 * extent_km:.0f} km context box on {nx} x {ny} points at {dx / 1000.0:.2f} km, "
           f"the pixel of its {size} picture")
    return window, why


def live_view_manifest(out_case: Path, view: str) -> dict | None:
    """The newest render receipt a view of this tree holds whose converter window is known, or None."""

    newest = None
    for path in Path(out_case, view).glob("render-manifest*.json"):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
            stamp = path.stat().st_mtime
        except (OSError, ValueError):
            continue
        if isinstance(doc, dict) and ((doc.get("convert") or {}).get("window_spec")):
            if newest is None or stamp > newest[0]:
                newest = (stamp, doc)
    return newest[1] if newest else None


def window_centre(manifest: dict) -> tuple[float, float, float] | None:
    """(lat, lon, half-extent km) of a derived window, from the converter's
    own window spec as the render manifest recorded it (Lambert:
    centre/dx/nx/ny; the near-equatorial lat-lon fallback: its bounds)."""

    spec = (manifest.get("convert") or {}).get("window_spec")
    if isinstance(spec, str):
        try:
            spec = json.loads(spec)
        except ValueError:
            return None
    if not isinstance(spec, dict):
        return None
    try:
        if spec.get("kind") == "lambert":
            half = 0.5 * float(spec["dx_metres"]) * min(int(spec["nx"]), int(spec["ny"])) / 1000.0
            return float(spec["centre_lat"]), float(spec["centre_lon"]), half
        if spec.get("kind") == "latlon":
            south, north = float(spec["south"]), float(spec["north"])
            west, east = float(spec["west"]), float(spec["east"])
            lat = 0.5 * (south + north)
            half = 0.5 * min(north - south, (east - west) * math.cos(math.radians(lat))) * 111.195
            return lat, 0.5 * (west + east), half
    except (KeyError, TypeError, ValueError):
        return None
    return None


#: Memory one frame in flight is budgeted, as a multiple of its history
#: file.  MEASURED (2026-10-12, eight frames in flight): 78.2 GB resident
#: for 30 km global frames whose history files are 1.24 GB (7.9 files per
#: frame, G1) and 31.2 GB for 48-3 km patch frames of 0.57 GB (6.8);
#: the renderer's import of the converted frame is the peak.
FRAME_MEMORY_FACTOR = 8.0
#: The share of the host's available memory frames in flight may plan on.
FRAME_MEMORY_SHARE = 0.5
#: The converter's own ceiling (``rw_mpas_convert --jobs`` takes 1-64). Memory and cores bound it before that.
#: THE BREAKAGE THIS PREVENTS (an Arizona zoom globe, 8 x RTX PRO 6000, 192 vCPU, 2 TB, woof hex 0.4.10,
#: 2026-10-30): a fixed ceiling of 8 held each view to 8 frames at a time on a host whose memory and cores carry
#: 28, so the whole-run maps' 37 conversions a view ran in five rounds after the forecast while most cores waited.
MAX_FRAME_JOBS = 64


def available_memory_bytes() -> int | None:
    """``MemAvailable`` from ``/proc/meminfo``; ``None`` where there is no
    such file (the door then runs one frame at a time)."""

    try:
        with open("/proc/meminfo", encoding="ascii", errors="replace") as stream:
            for line in stream:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def auto_frame_jobs(histories: list[Path], view_jobs: int, *,
                    available: int | None = None, cpus: int | None = None) -> int:
    """Frames each view converts and imports at a time, from the host.

    Bounded by memory (half of what is available, shared by the views that
    draw side by side, at :data:`FRAME_MEMORY_FACTOR` history files per
    frame in flight), by cores (two per frame per view), by the number of
    frames and by :data:`MAX_FRAME_JOBS`.  One when the host's memory cannot
    be read: a guess that is too high is an out-of-memory kill at the end of
    a forecast."""

    if available is None:
        available = available_memory_bytes()
    if available is None or not histories:
        return 1
    cpus = cpus or os.cpu_count() or 1
    view_jobs = max(1, view_jobs)
    per_frame = FRAME_MEMORY_FACTOR * max(Path(h).stat().st_size for h in histories)
    by_memory = int(FRAME_MEMORY_SHARE * available // max(1.0, per_frame * view_jobs))
    by_cores = cpus // (2 * view_jobs)
    return max(1, min(by_memory, by_cores, len(histories), MAX_FRAME_JOBS))


#: The first look's receipt (woof/hex/first_look.py), read by ``--live-frames``.
LIVE_FRAMES_SCHEMA = "gpuwm-hex.live-frames/v1"


def live_drawn(receipt: Path | str | None, out_case: Path, histories: list[Path], start: str) -> dict | None:
    """The frames the first look already rendered into THIS tree, or None (the full pass draws everything).

    THE BREAKAGE THIS PREVENTS (hex global 60 km 24 h, 2026-10-24): every hour was drawn while the forecast
    ran, then the full pass rendered all 25 frames again after it (81 s), so the run was done no sooner. A frame counts as
    drawn only when the receipt names this tree and case, the frame's size and mtime still match what was drawn (a
    forecast rerun in the same folder rewrites them), and pictures for its lead are on disk in the tree."""

    if not receipt:
        return None
    try:
        doc = json.loads(Path(receipt).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(doc, dict) or doc.get("schema") != LIVE_FRAMES_SCHEMA or not doc.get("out") or not doc.get("case"):
        return None
    try:
        if (Path(doc["out"]) / str(doc["case"])).resolve() != Path(out_case).resolve() or not Path(out_case).is_dir():
            return None
    except OSError:
        return None
    stat = doc.get("stat") or {}
    named = set(doc.get("drawn") or ())
    leads = {png.stem.rsplit("_", 1)[-1] for png in Path(out_case).rglob("*.png")
             if png.relative_to(out_case).parts[0] != "points"}
    origin = _parse_stamp(start)
    frames = []
    for path in histories:
        if path.name not in named:
            continue
        stamp = frame_stamp(path)
        try:
            st = path.stat()
        except OSError:
            continue
        if stamp is None or list(stat.get(path.name) or ()) != [st.st_size, st.st_mtime_ns]:
            continue
        seconds = int(round((_parse_stamp(stamp) - origin).total_seconds()))
        if render_door.lead_token(seconds, path.name) in leads:
            frames.append(path)
    if not frames:
        return None
    return {"receipt": str(receipt), "frames": frames, "products": {str(p) for p in doc.get("products") or ()}}


def _merge_views(parts: list[tuple[int, dict | None, str | None]]) -> tuple[int, dict | None, str | None]:
    """One view drawn in parts (the whole-run maps over every frame, the rest over the frames not drawn live), told
    as one: the first non-zero exit and refusal, the first manifest with the parts' counts summed and their drawn and
    unavailable products joined."""

    rc = next((r for r, _, _ in parts if r), 0)
    refusal = next((x for _, _, x in parts if x is not None), None)
    manifests = [m for _, m, _ in parts if m is not None]
    if not manifests:
        return rc, None, refusal
    merged = dict(manifests[0])
    counts: dict = {}
    drawn: list[str] = []
    unavailable: dict = {}
    for m in manifests:
        for key, value in (m.get("counts") or {}).items():
            counts[key] = counts.get(key, 0) + value if isinstance(value, (int, float)) else value
        listed = m.get("products_drawn") or ()
        for slug in (listed.split(",") if isinstance(listed, str) else listed):
            if slug and slug.strip() and slug.strip() not in drawn:
                drawn.append(slug.strip())
        unavailable.update(m.get("products_unavailable") or {})
    merged.update(counts=counts, products_drawn=",".join(drawn), products_unavailable=unavailable)
    return rc, merged, refusal


def _render_view(*, histories, mesh, init, start, out_case, scratch_root, view,
                 window, products, size, model_label, remap, exes, basemap_dir,
                 keep_scratch, frame_jobs=1, source_label=None, part="",
                 tiles="none", grid_label=None, frame_cache=None) -> tuple[int, dict | None, str | None]:
    """One view through the render door in standard-set mode."""

    args = argparse.Namespace(
        history=[str(p) for p in histories], mesh=str(mesh),
        init=str(init) if init else None, compose=None, compose_base_only=False,
        out=str(out_case), products=",".join(products), frames="all",
        window=window, view=view, model_label=model_label, source_label=source_label,
        grid_label=grid_label, field_set="full",
        label=None, simulation_start=start, size=size, heavy=False,
        nc_format="cdf5", scratch=str(Path(scratch_root) / f"render-{view}{part}"),
        keep_scratch=keep_scratch, basemap_dir=basemap_dir,
        convert_exe=exes[0], renderer_exe=exes[1], remap=remap,
        standard_set=True, series=True, frame_jobs=frame_jobs, tiles=tiles, frame_cache=frame_cache)
    print(f"PRODUCTS view={view} window={window}", flush=True)
    try:
        rc = render_door.run_render(args)
    except MpasPortError as error:
        return 2, None, str(error)
    except (OSError, ValueError) as error:
        # Views draw side by side: one view's file error is that view's
        # failure, named, and the others still deliver.
        return 2, None, f"{type(error).__name__}: {error}"
    manifests = sorted((Path(out_case) / view).glob("render-manifest*.json"),
                       key=lambda p: p.stat().st_mtime)
    manifest = None
    if manifests:
        try:
            manifest = json.loads(manifests[-1].read_text(encoding="utf-8"))
        except (OSError, ValueError):
            manifest = None
    return rc, manifest, None


def _place_points(raw: Path, out_case: Path, start: str) -> list[str]:
    """``<case>/points/<sounding-slug|meteogram-slug>/<day>/file.png``."""

    placed = []
    init_day = start[:10]
    for png in sorted(Path(raw).rglob("*.png")):
        name = png.name
        m = re.match(r"(sounding|meteogram)_(.+?)_(\d{8})_(\d{2,4})", name)
        if not m:
            continue
        kind, slug, ymd = m.group(1), m.group(2), m.group(3)
        day = f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:]}" if kind == "sounding" else init_day
        dest = Path(out_case) / "points" / f"{kind}-{slug}" / day / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(png), str(dest))
        placed.append(dest.relative_to(out_case).as_posix())
    return placed


_LEAD_IN_NAME = re.compile(r"_(f\d{3}(?:h\d{2}m)?)\.png$")


_INDEX_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#f4f5f7;--panel:#ffffff;--ink:#1c2026;--dim:#5d6772;--line:#d5d9df;--pick:#1f5fbf;--pickink:#ffffff}
@media (prefers-color-scheme: dark){:root{--bg:#14171a;--panel:#1d2126;--ink:#e8eaed;--dim:#9aa3ab;--line:#353b42;--pick:#5b9bf0;--pickink:#0d1014}}
*{box-sizing:border-box}
body{margin:0;font:15px/1.4 system-ui,sans-serif;background:var(--bg);color:var(--ink)}
header{padding:12px 16px;border-bottom:1px solid var(--line)}
h1{font-size:18px;margin:0 0 2px}
header p{margin:0;color:var(--dim);font-size:13px}
main{display:flex;flex-wrap:wrap;gap:12px;padding:12px 16px}
nav{flex:0 0 250px;max-width:100%}
nav select{width:100%;padding:6px;font:inherit;margin-bottom:8px;background:var(--panel);color:var(--ink);border:1px solid var(--line)}
#products{max-height:70vh;overflow:auto;border:1px solid var(--line);background:var(--panel)}
#products button{display:block;width:100%;text-align:left;padding:6px 8px;font:inherit;font-size:13px;border:0;border-bottom:1px solid var(--line);background:none;color:var(--ink);cursor:pointer;overflow-wrap:anywhere}
#products button[aria-pressed=true]{background:var(--pick);color:var(--pickink)}
#stage{flex:1 1 520px;min-width:0}
#bar{display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin-bottom:8px}
#bar button{padding:5px 12px;font:inherit;border:1px solid var(--line);background:var(--panel);color:var(--ink);cursor:pointer}
#slider{flex:1 1 160px;min-width:120px}
#lead{font-variant-numeric:tabular-nums;min-width:8ch}
#frame{display:block;max-width:100%;height:auto;background:#fff;border:1px solid var(--line)}
#open{font-size:13px;color:var(--dim)}
</style></head><body>
<header><h1>__TITLE__</h1><p>woof hex products. Run start __START__. __TOTAL__ pictures. Choose a view and a product; the slider and the arrow keys step through its frames.</p></header>
<main>
<nav><label for="view">View</label><select id="view"></select><div id="products"></div></nav>
<section id="stage">
<div id="bar"><button id="prev" type="button">Previous</button><button id="play" type="button">Play</button><button id="next" type="button">Next</button>
<input id="slider" type="range" min="0" max="0" value="0" aria-label="Frame"><span id="lead"></span><a id="open" href="#">open picture</a></div>
<img id="frame" alt="">
</section>
</main>
<script>
const FIRST = __FIRST__;
const DATA = __DATA__;
const el = id => document.getElementById(id);
let view = Object.keys(DATA)[0], product = null, index = 0, timer = null;
function frames(){ return (DATA[view] || {})[product] || []; }
function show(){
  const f = frames(); if (!f.length) return;
  index = Math.max(0, Math.min(index, f.length - 1));
  el('frame').src = f[index][1]; el('frame').alt = product + ' ' + f[index][0];
  el('open').href = f[index][1]; el('lead').textContent = f[index][0] + '  ' + (index + 1) + '/' + f.length;
  el('slider').max = f.length - 1; el('slider').value = index;
}
function pick(name){
  const before = frames()[index];
  const keepLead = before ? before[0] : null;
  product = name;
  const at = frames().findIndex(x => x[0] === keepLead);
  index = at >= 0 ? at : 0;
  // Reflectivity opens on its first frame with echoes: an earlier one (a cold start's f000) is an empty map.
  if (name === FIRST.echo_map && FIRST.reflectivity_from) {
    const echo = frames().findIndex(x => x[0] === FIRST.reflectivity_from);
    if (echo > index) index = echo;
  }
  for (const b of el('products').children) b.setAttribute('aria-pressed', b.dataset.name === name);
  show();
}
function list(){
  const box = el('products'); box.textContent = '';
  for (const name of Object.keys(DATA[view])) {
    const b = document.createElement('button'); b.type = 'button'; b.dataset.name = name;
    b.textContent = name + ' (' + DATA[view][name].length + ')'; b.onclick = () => pick(name); box.appendChild(b);
  }
  pick(Object.keys(DATA[view]).includes(product) ? product : Object.keys(DATA[view])[0]);
}
function step(d){ const n = frames().length; if (n) { index = (index + d + n) % n; show(); } }
function stop(){ if (timer) { clearInterval(timer); timer = null; el('play').textContent = 'Play'; } }
el('prev').onclick = () => { stop(); step(-1); };
el('next').onclick = () => { stop(); step(1); };
el('play').onclick = () => { if (timer) stop(); else { timer = setInterval(() => step(1), 600); el('play').textContent = 'Pause'; } };
el('slider').oninput = e => { stop(); index = +e.target.value; show(); };
document.addEventListener('keydown', e => { if (e.target.tagName === 'SELECT') return;
  if (e.key === 'ArrowLeft') { stop(); step(-1); } if (e.key === 'ArrowRight') { stop(); step(1); } });
const sel = el('view');
for (const name of Object.keys(DATA)) { const o = document.createElement('option'); o.value = o.textContent = name; sel.appendChild(o); }
sel.onchange = () => { stop(); view = sel.value; list(); };
list();
</script></body></html>
"""


def index_tree(out_case: Path) -> dict[str, dict[str, list[list[str]]]]:
    """``{view: {product: [[caption, relative path], ...]}}`` for every
    picture under a case folder, frames in valid-day then lead order."""

    tree: dict[str, dict[str, list[list[str]]]] = {}
    out_case = Path(out_case)
    for view_dir in sorted(p for p in out_case.iterdir() if p.is_dir()):
        products: dict[str, list[list[str]]] = {}
        for product_dir in sorted(p for p in view_dir.iterdir() if p.is_dir()):
            # A tiled view (render --tiles) files each product's pictures under its tile:
            # <product>/full/<day>/ and <product>/<cut>/<day>/.  Each tile is its own entry
            # (the whole window keeps the product's own name), so the frames of one tile are
            # never interleaved with another's.
            groups = [(product_dir.name, product_dir)]
            tiled = [child for child in product_dir.iterdir()
                     if child.is_dir() and child.name in render_door.TILE_NAMES]
            if tiled:
                groups = [(product_dir.name if child.name == render_door.TILE_FULL
                           else f"{product_dir.name}/{child.name}", child)
                          for child in sorted(tiled, key=lambda c: render_door.TILE_NAMES.index(c.name))]
            for key, folder in groups:
                pictures = sorted(folder.rglob("*.png"),
                                  key=lambda p: (p.parent.name, p.name))
                rows = []
                for png in pictures:
                    lead = _LEAD_IN_NAME.search(png.name)
                    rows.append([lead.group(1) if lead else png.parent.name,
                                 png.relative_to(out_case).as_posix()])
                if rows:
                    products[key] = rows
        if products:
            tree[view_dir.name] = products
    return tree


COLUMNS_INDEX_SCHEMA = "gpuwm-hex.columns-index/v1"
COLUMNS_INDEX_NAME = "columns-index.json"


def _window_spec(record: dict, out_case: Path) -> dict | None:
    """A view's converter window, as the render manifest recorded it (the view's own record, else the
    render-manifest.json beside its maps, which a view drawn live has)."""
    spec = record.get("window_spec")
    if spec is None:
        try:
            doc = json.loads((Path(out_case) / record["view"] / "render-manifest.json").read_text(encoding="utf-8"))
            spec = (doc.get("convert") or {}).get("window_spec")
        except (OSError, ValueError, AttributeError):
            spec = None
    if isinstance(spec, str):
        try:
            spec = json.loads(spec)
        except ValueError:
            return None
    return spec if isinstance(spec, dict) else None


def write_columns_index(out_case: Path, views: list[dict]) -> Path | None:
    """``<case>/columns-index.json``: every view's any-point column store (column_store.py), finest grid first.

    Each view writes its own store beside its maps (``<case>/<view>/columns/<dNN>/<stamp>.wcol``, every output hour),
    and every store's own frame file names its grid ``d01``, so a reader cannot tell from the store's path which grid
    it holds. Here each is listed with its view, its spacing (km) and its window, and ranked finest first, so a
    sounding at a point is read from the finest grid that holds it and the whole globe outside a refined core. THE
    BREAKAGE THIS SERVES (WOOF production, 2026-10-28): a reader that filed every store by the frame's ``d01`` took a
    96 -> 3 km zoom globe's ~25 km globe for its 3 km core. Nothing is written when no view has a store."""

    rows = []
    for record in views:
        name = record.get("view")
        folder = Path(out_case) / str(name) / "columns"
        if not name or name == "points" or not folder.is_dir():
            continue
        files = sorted(f for f in folder.glob("d[0-9][0-9]/*.wcol") if f.stat().st_size > 0)
        if not files:
            continue
        spec = _window_spec(record, out_case) or {}
        spacing = None
        try:
            if spec.get("dx_metres"):
                spacing = round(float(spec["dx_metres"]) / 1000.0, 3)
            elif spec.get("spacing_degrees"):
                spacing = round(float(spec["spacing_degrees"]) * 111.195, 3)
        except (TypeError, ValueError):
            spacing = None
        bounds = None
        if spec.get("kind") == "latlon":
            try:
                bounds = [float(spec["west"]), float(spec["south"]), float(spec["east"]), float(spec["north"])]
            except (KeyError, TypeError, ValueError):
                bounds = None
        centre = window_centre({"convert": {"window_spec": spec}}) if spec else None
        rows.append({
            "view": name,
            "window": record.get("window"),
            "folder": "/".join((name, "columns", files[0].parent.name)),
            "spacing_km": spacing,
            "bounds": bounds,
            "centre": ({"lat": round(centre[0], 4), "lon": round(centre[1], 4), "half_extent_km": round(centre[2], 1)}
                       if centre else None),
            "valid_times": [f"{f.stem[0:4]}-{f.stem[4:6]}-{f.stem[6:8]}T{f.stem[9:11]}:{f.stem[11:13]}:{f.stem[13:15]}Z"
                            for f in files],
            "bytes": sum(f.stat().st_size for f in files),
        })
    if not rows:
        return None
    rows.sort(key=lambda r: (r["spacing_km"] is None, r["spacing_km"] or 0.0, r["view"]))
    for rank, row in enumerate(rows):
        row["rank"] = rank
    path = Path(out_case) / COLUMNS_INDEX_NAME
    render_door.atomic_write_text(path, json.dumps({"schema": COLUMNS_INDEX_SCHEMA, "order": "finest first",
                                                    "views": rows}, indent=2, sort_keys=True) + "\n")
    return path


def write_index(out_case: Path, case: str, start: str, first: dict | None = None) -> Path | None:
    """``<case>/index.html``: a one-file viewer over every delivered
    picture -- view, product, a frame slider and play.  Built from the tree
    as it is on disk, so it shows what was delivered and nothing else; the
    links are relative and the page loads nothing from the network, so the
    case folder can be copied anywhere and still open.  It lists and opens
    on the ``first`` maps (``first_maps``) and opens reflectivity on its
    first frame with echoes."""

    tree = index_tree(out_case)
    if not tree:
        return None
    first = first or {"maps": list(FIRST_MAPS), "echo_map": ECHO_MAP, "reflectivity_from": None}
    leading = list(first.get("maps") or [])
    if first.get("echo_map") and first["echo_map"] not in leading:
        leading.append(first["echo_map"])  # no echoes yet: still listed right after the opening maps
    rank = {slug: position for position, slug in enumerate(leading)}
    # "global" first, then the mesh's own views, points last.
    order = sorted(tree, key=lambda name: (name == "points", name != "global", name))
    data = {name: {product: tree[name][product]
                   for product in sorted(tree[name], key=lambda p: (rank.get(p, len(rank)), p))}
            for name in order}
    opening = {"echo_map": first.get("echo_map"), "reflectivity_from": first.get("reflectivity_from")}
    total = sum(len(rows) for products in tree.values() for rows in products.values())
    page = (_INDEX_PAGE
            .replace("__TITLE__", _html.escape(case))
            .replace("__START__", _html.escape(start))
            .replace("__TOTAL__", str(total))
            .replace("__FIRST__", json.dumps(opening).replace("</", "<\\/"))
            .replace("__DATA__", json.dumps(data).replace("</", "<\\/")))
    path = Path(out_case) / "index.html"
    render_door.atomic_write_text(path, page)
    return path


def run_products(args) -> int:
    started = time.monotonic()
    named = getattr(args, "forecast", None)
    positional = getattr(args, "forecast_dir", None)
    if named and positional and Path(named) != Path(positional):
        raise ProductDoorError(
            f"two forecast folders were given ({positional} and --forecast "
            f"{named}); the door draws one run")
    forecast_dir = Path(named or positional) if (named or positional) else None
    receipt = forecast_receipt(forecast_dir)
    run_receipt = receipt
    from .chem_transport import product_slugs
    chemistry_products = product_slugs(forecast_dir)
    if args.history:
        histories = [Path(p) for p in args.history]
    elif forecast_dir is not None:
        histories = discover_histories(forecast_dir)
    else:
        raise ProductDoorError("give --forecast DIR (the forecast door's --out) or --history FILE...")
    for path in histories:
        if not path.is_file():
            raise ProductDoorError(f"--history names a missing file: {path}")
    # What the forecast already recorded is read from its receipt; a flag
    # still wins, and a path the receipt names that is no longer there (the
    # run folder was copied without its mesh) is refused as the receipt's.
    mesh_text = args.mesh or receipt_text(receipt, "mesh", "grid")
    if not mesh_text:
        raise ProductDoorError(
            "no mesh: --mesh was not given and the forecast folder carries no "
            f"{FORECAST_RECEIPT} naming the grid the run was bound to. The "
            "converter places the mesh's cells on the render grid from it; "
            "pass --mesh GRID.nc.")
    mesh = Path(mesh_text)
    mesh_source = "--mesh" if args.mesh else f"{FORECAST_RECEIPT} (mesh.grid)"
    if not mesh.is_file():
        raise ProductDoorError(
            f"{mesh_source} names a missing file: {mesh}. The converter places the "
            f"mesh's cells on the render grid from it; give the run's grid file "
            f"with --mesh.")
    init_text = args.init or receipt_text(receipt, "init", "path")
    init = Path(init_text) if init_text else None
    if init is not None and not init.is_file():
        init_source = "--init" if args.init else f"{FORECAST_RECEIPT} (init.path)"
        raise ProductDoorError(
            f"{init_source} names a missing file: {init}. zgrid, the column "
            f"heights behind the pressure-level maps and the soundings, lives "
            f"there; pass --init INIT.nc.")
    start = simulation_start(args.simulation_start, forecast_dir, histories)
    windowed, windows_not_requested = windowed_for_history(histories, start)
    global_products = BASE_GLOBAL_PRODUCTS + windowed
    regional_products = global_products + REGIONAL_EXTRA_PRODUCTS
    for slug, reason in windows_not_requested.items():
        print(f"PRODUCTS window not requested: {slug}: {reason}", flush=True)
    case = case_token(args.case, mesh, start,
                      receipt_text(receipt, "mesh_binding", "mesh")
                      or receipt_text(receipt, "admission", "mesh"))
    if args.out:
        out_root = Path(args.out)
    elif forecast_dir is not None:
        out_root = forecast_dir / "products"
    else:
        raise ProductDoorError(
            "no --out: with --history there is no forecast folder to file "
            "the products beside, so name the delivered tree's root")
    args.out = str(out_root)
    out_case = Path(args.out) / case
    scratch_root = Path(args.scratch) if args.scratch else Path(args.out).parent / (Path(args.out).name + ".products-scratch")
    # Frames converted once a view and reused by later passes of this run (render_door.convert_frames_cached): the
    # live draws fill it, the full pass reads it and removes it.
    # ("scratch" in its name: hosts that file a run's frames, such as the WOOF worker, skip such folders.)
    frame_cache = Path(args.out).parent / (Path(args.out).name + ".frame-cache-scratch") / case
    render_door.refuse_scratch_inside_out(scratch_root, Path(args.out))
    out_case.mkdir(parents=True, exist_ok=True)
    convert_exe = render_door.resolve_convert_exe(args.convert_exe)
    renderer_exe = render_door.resolve_renderer_exe(args.renderer_exe)
    exes = (str(convert_exe), str(renderer_exe))
    print(f"PRODUCTS case={case} frames={len(histories)} start={start} out={out_case}", flush=True)
    live = live_drawn(getattr(args, "live_frames", None), out_case, histories, start)
    live_names = {p.name for p in (live or {}).get("frames", ())}
    redraw = [p for p in histories if p.name not in live_names]
    if live is not None:
        print(f"PRODUCTS live: {len(live_names)} of {len(histories)} frames already drawn while the forecast ran; "
              f"this pass draws the whole-run maps over every frame and the maps of {len(redraw)} frame(s)", flush=True)
    # The opening maps (first_maps): the reflectivity scan reads frames beside the views, off their path.
    opening: dict = {}
    opening_scan = threading.Thread(
        target=lambda: opening.update(first_maps(histories, start, (*FIRST_MAPS, ECHO_MAP))),
        name="first-maps", daemon=True)
    opening_scan.start()

    views: list[dict] = []
    # Each view's map headline: the caller's words where given, else the view's
    # own name and the spacing its pictures sample (view_title).
    captions = view_labels(getattr(args, "view_label", None))
    moving = moving_controller(forecast_dir) is not None

    def view(name, window, products, size, tiles=None):
        view_started = time.monotonic()
        if tiles is None:
            tiles = getattr(args, "tiles", "auto") if window != "global" else "none"
        common = dict(mesh=mesh, init=init, start=start, out_case=out_case, scratch_root=scratch_root, view=name,
                      grid_label=captions.get(name) or view_title(name, moving=moving),
                      window=window, size=size, model_label=args.model_label, remap=args.remap, exes=exes,
                      basemap_dir=args.basemap_dir, keep_scratch=args.keep_scratch, frame_jobs=frame_jobs,
                      source_label=getattr(args, "source_label", None), tiles=tiles,
                      frame_cache=frame_cache)
        if live is None:
            rc, manifest, refusal = _render_view(
                histories=histories, products=tuple(products) + chemistry_products, **common)
        else:
            # The frames drawn live keep their pictures (the same door rendered them, byte for byte); this pass draws
            # the windowed maps, which need the whole series, and every map of the frames no live draw took, side
            # by side. A view with nothing else to draw redraws the last frame, so it still names its window.
            whole_run = tuple(p for p in products if p in WINDOWED_PRODUCTS)
            per_frame = tuple(p for p in products if p not in WINDOWED_PRODUCTS) + chemistry_products
            rest = redraw or ([] if whole_run else histories[-1:])
            jobs_here = []
            if whole_run:
                jobs_here.append(dict(histories=histories, products=whole_run, part="-series"))
            if rest:
                jobs_here.append(dict(histories=rest, products=per_frame, part="-rest"))
            results: list = [None] * len(jobs_here)

            def part(index):
                result = _render_view(**jobs_here[index], **common)
                if (jobs_here[index]["part"] == "-series" and result[0] and result[2]
                        and result[2].startswith("none of the standard products")):
                    result = (0, None, None)  # no whole-run map this history can draw: not a failed view
                results[index] = result

            helpers = [threading.Thread(target=part, args=(i,), name=f"view-{name}-part{i}")
                       for i in range(1, len(jobs_here))]
            for helper in helpers:
                helper.start()
            part(0)
            for helper in helpers:
                helper.join()
            if not any(r for r, _, _ in results) and all(m is None for _, m, _ in results):
                results.append(_render_view(histories=histories[-1:], products=per_frame, part="-rest", **common))
            rc, manifest, refusal = _merge_views(results)
        record = {"view": name, "window": window, "rc": rc, "refusal": refusal,
                  "elapsed_seconds": round(time.monotonic() - view_started, 3)}
        if refusal is not None and "no refined region" not in refusal:
            print(f"PRODUCTS view={name} refused: {refusal}", flush=True)
        if manifest is not None:
            record.update({
                "counts": manifest.get("counts"),
                "products_drawn": manifest.get("products_drawn"),
                "products_unavailable": manifest.get("products_unavailable"),
                "window_description": (manifest.get("convert") or {}).get("window_description"),
                "window_spec": (manifest.get("convert") or {}).get("window_spec"),
                "remap": (manifest.get("convert") or {}).get("remap"),
                "product_remap": (manifest.get("convert") or {}).get("product_remap"),
                "precipitation_budget": (manifest.get("convert") or {}).get("precipitation_budget"),
                "tiles": ({"why": manifest["tiles"].get("why"),
                           "cuts": [{key: cut.get(key) for key in ("name", "window", "rc", "counts",
                                                                   "elapsed_seconds")}
                                    for cut in manifest["tiles"].get("cuts") or []]}
                          if manifest.get("tiles") else None),
            })
        views.append(record)
        return rc, manifest, refusal

    # Views are independent passes over the same frames (their own window,
    # their own scratch), so they draw side by side.  MEASURED (G1,
    # 2026-10-12, nine 30 km frames): one after another the three views took
    # 25 min 54 s, most of it each view's own import.
    # What the point pictures need is asked for BEFORE any view is drawn: a
    # missing rw_hex_points or --init used to be discovered after every map
    # was on disk, minutes into the command.
    points_exe = None
    if not args.no_points:
        points_exe = resolve_points_exe(args.points_exe, convert_exe)
        if init is None:
            raise ProductDoorError(
                "soundings need the run's --init (zgrid, the column heights, "
                "lives there and nowhere else); pass --init or --no-points")
    extra_regions: dict[str, str] = {}
    for item in args.region or []:
        name, _, spec = item.partition("=")
        if not spec:
            raise ProductDoorError(f"--region {item!r} is not NAME=region:CLAT,CLON,W_KM,H_KM")
        extra_regions[render_door.checked_view(name)] = spec

    jobs = max(1, int(getattr(args, "jobs", 3) or 1))
    requested = getattr(args, "frame_jobs", None)
    if requested in (None, "auto"):
        frame_jobs = auto_frame_jobs(histories, jobs)
    else:
        frame_jobs = render_door.checked_frame_jobs(requested)
    print(f"PRODUCTS views side by side={jobs} frames at a time per view={frame_jobs}",
          flush=True)
    pool = ThreadPoolExecutor(max_workers=jobs, thread_name_prefix="view")
    pending = []
    # A LIMITED-AREA run holds data inside its own domain only, and the mesh
    # view frames all of it.  THE BREAKAGE THIS PREVENTS (woof hex regional,
    # 2026-10-20, a 43,360-cell cull, 3 h, 4 frames): the
    # global view converted every frame onto a whole-globe window at the
    # cull's own spacing (35 GB per frame, 1,444 s for the conversion) and
    # the 1,500 km context view another 550 s, together 131 GB of scratch,
    # to draw maps that are empty outside a 270 km domain; the mesh view of
    # the same frames took 61 s.  Named --region views are still drawn.
    limited_area = (receipt_text(receipt, "admission", "configuration") == "limited-area"
                    or mesh_is_limited_area(mesh))
    if limited_area:
        print("PRODUCTS limited-area run: the mesh view frames the whole domain; "
              "no global or context view", flush=True)
    if not args.no_global and not limited_area:
        globe, globe_why = global_window(mesh, args.global_size)
        if globe_why:
            print(f"PRODUCTS global window {globe}: {globe_why}", flush=True)
        pending.append(pool.submit(view, "global", globe, global_products, args.global_size, "none"))
    refined_centre = None
    mesh_uniform = None
    if not args.no_mesh_view:
        core_future = pool.submit(view, "refined-core", "mesh", regional_products, args.regional_size)
        # The context view is placed from the core's window. When the live draws of this run already rendered the core
        # into this tree, its window is on disk, so the context starts beside the core's whole-run maps instead of
        # after them. THE BREAKAGE THIS PREVENTS (an Arizona zoom globe, 8 x RTX PRO 6000, woof hex 0.4.10,
        # 2026-10-30): after the forecast the context view, the slowest, waited for the core view's 37 conversions
        # and its renders before its own first conversion began.
        drawn_core = live_view_manifest(out_case, "refined-core") if live is not None else None
        if drawn_core is not None:
            pending.append(core_future)
            rc, manifest, refusal = 0, drawn_core, None
        else:
            rc, manifest, refusal = core_future.result()
        if refusal is not None and "no refined region" in refusal:
            mesh_uniform = True
            # A uniform mesh has no refined region: the converter refuses
            # --window mesh by that name.  Not a failure of this run.
            core = next(v for v in views if v["view"] == "refined-core")
            core["rc"] = 0
            core["note"] = "uniform mesh: no refined region to frame"
            print("PRODUCTS uniform mesh: no refined-core view; default regions instead", flush=True)
            regions = dict(DEFAULT_REGIONS)
        else:
            regions = {}
            if manifest is not None:
                refined_centre = window_centre(manifest)
                mesh_uniform = False
            if refined_centre is not None and not limited_area:
                lat, lon, half = refined_centre
                extent = max(1500.0, 8.0 * half)
                core_spec = (manifest.get("convert") or {}).get("window_spec") if manifest else None
                core_dx = core_spec.get("dx_metres") if isinstance(core_spec, dict) else None
                regions["refined-context"], context_why = context_window(
                    lat, lon, extent, args.regional_size, float(core_dx) if core_dx else None)
                if context_why:
                    print(f"PRODUCTS context window {regions['refined-context']}: {context_why}", flush=True)
    else:
        regions = {}
    regions.update(extra_regions)
    for name, spec in regions.items():
        # The context view around a refined core is one picture, never tiles. THE BREAKAGE THIS PREVENTS (an
        # Arizona 96 -> 3 km zoom globe, 800 km core, 36 h, 8 x RTX PRO 6000, woof hex 0.4.10, 2026-10-30): the
        # 6,850 x 4,795 km context window takes its dx from the median of the cells inside it, which is the core's
        # 3 km, so its 2000 x 1401 grid was "wider than its 1400 x 1000 picture" and --tiles auto cut it into six
        # more passes: 2,996 maps where 0.4.9 rendered 477, still drawing 31 min after the core view was done. Outside
        # the core that window holds the 6-48 km rings, which one 1400-pixel picture (4.9 km a pixel) already
        # shows; the core's own 3 km detail is the refined-core view. The WOOF worker files <view>/<product>/<day>/
        # and reads a tile segment as the day. Tiles stay on a view whose mesh is that fine throughout: the mesh
        # view of a limited-area or uniform mesh (RRFS conus3km) and a named --region; --tiles regional still
        # cuts every regional view, this one included.
        tiles = "none" if name == "refined-context" and getattr(args, "tiles", "auto") == "auto" else None
        pending.append(pool.submit(view, name, spec, regional_products, args.regional_size, tiles))

    def draw_points() -> dict:
        points_started = time.monotonic()
        lacking = histories_without_surface_pressure(histories)
        if lacking:
            # THE BREAKAGE THIS PREVENTS (hexmx proof, 2026-10-16): an MPAS suite run's history carries no
            # surface_pressure (the suite route does not write that diag field yet; its receipt lists it under
            # missing_fields), so rw_hex_points refused and the whole products command exited 1 with every map
            # drawn.  A sounding and a meteogram need PSFC, so the point pictures are skipped for such a source,
            # said once, and the maps stand.
            reason = (f"{lacking[0].name} carries no surface_pressure (PSFC), which a sounding and a meteogram "
                      f"need; point pictures skipped for this source ({len(lacking)} frames)")
            print(f"POINTS skipped: {reason}", flush=True)
            return {"rc": 0, "skipped": reason, "points": [],
                    "elapsed_seconds": round(time.monotonic() - points_started, 3)}
        dropped: list[dict] = []
        on_mesh_already = False
        if args.points:
            points = parse_point_text(args.points)
        elif limited_area and (check := mesh_point_check(mesh, REGION_CITIES)) is not None:
            points = region_default_points(check)
            on_mesh_already = True
            print(f"POINTS default: {len(points)} on this limited-area mesh: "
                  + ", ".join(name for name, _, _ in points), flush=True)
        elif refined_centre is not None:
            lat, lon, half = refined_centre
            radius = max(150.0, 0.9 * half)
            points = [p for p in DEFAULT_POINTS if great_circle_km(lat, lon, p[1], p[2]) <= radius]
            if not points:
                points = [("Refined core centre", lat, lon)]
        else:
            points = list(DEFAULT_POINTS[:3]) + [p for p in DEFAULT_POINTS if p[0] in ("London", "Tokyo", "Sydney")]
            points = points[:DEFAULT_GLOBAL_POINT_COUNT]
        if limited_area and not on_mesh_already and (check := mesh_point_check(mesh, points)) is not None:
            # A point off a limited-area mesh is dropped with a note and the
            # rest are drawn: rw_hex_points refuses the whole list over one
            # off-mesh point (graded-plateau 1 km region, 2026-10-24).
            points = check[0]
            for name, plat, plon, km, spacing in check[1]:
                where = (f"{km:.1f} km from the nearest cell centre, beyond that cell's {spacing:.1f} km spacing"
                         if km is not None else f"{spacing:.0f} km beyond the domain edge")
                note = f"{name} ({plat:.2f}, {plon:.2f}) is off this limited-area mesh: {where}"
                dropped.append({"name": name, "lat": plat, "lon": plon, "note": note})
                print(f"POINTS dropped: {note}", flush=True)
            if not points:
                reason = "every point is off this limited-area mesh; no sounding or meteogram drawn"
                print(f"POINTS skipped: {reason}", flush=True)
                return {"rc": 0, "skipped": reason, "points": [], "dropped": dropped,
                        "elapsed_seconds": round(time.monotonic() - points_started, 3)}
        raw = scratch_root / "points"
        raw.mkdir(parents=True, exist_ok=True)
        receipt = raw / "points.json"
        command = [str(points_exe), "--mesh", str(mesh), "--init", str(init),
                   "--out-dir", str(raw), "--simulation-start", start,
                   "--model-label", args.model_label,
                   "--sounding-every-hours", str(args.sounding_every_hours),
                   "--points", ";".join(f"{n}@{la},{lo}" for n, la, lo in points),
                   "--json", str(receipt)]
        # The top right is never a default tool name or nothing.  THE BREAKAGE
        # THIS PREVENTS: a storm-following run (GFS 2026-10-23 18Z) rendered its
        # soundings with an empty top right because no door step passed
        # --brand.  An explicit --brand still wins; else the run's own records.
        brand = getattr(args, "brand", None)
        brand_from = "--brand"
        if not brand:
            metadata = run_brand_metadata(
                forecast_dir=forecast_dir, receipt=run_receipt, mesh=mesh,
                limited_area=limited_area, uniform=mesh_uniform, init=init,
                start=args.simulation_start or receipt_text(run_receipt, "schedule", "start_time"))
            brand = default_brand(**metadata)
            brand_from = "run metadata"
        if points_take_brand(points_exe):
            command += ["--brand", brand]
            print(f"POINTS brand ({brand_from}): {brand}", flush=True)
        else:
            print(f"POINTS brand not drawn: {points_exe} predates --brand; "
                  "install the woof release that carries it", flush=True)
        command += ["--history", *[str(p) for p in histories]]
        print("POINTS $ " + " ".join(command), flush=True)
        result = subprocess.run(command, capture_output=True, text=True, errors="replace")
        for line in (result.stdout or "").splitlines():
            if line.startswith(("POINT\t", "METEOGRAM\t", "FINISHED\t", "POINTS MISSING\t")):
                print(f"POINTS {line}", flush=True)
        points_record = {"rc": result.returncode, "points": points, "dropped": dropped, "invocation": command,
                         "brand": brand, "brand_from": brand_from,
                         "elapsed_seconds": round(time.monotonic() - points_started, 3)}
        if result.returncode != 0:
            tail = [l for l in (result.stderr or "").splitlines() if l.strip()]
            points_record["refusal"] = tail[-1] if tail else f"exit {result.returncode}"
            print(f"POINTS refused: {points_record['refusal']}", flush=True)
        else:
            try:
                points_record["receipt"] = json.loads(receipt.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                points_record["receipt"] = None
            points_record["placed"] = _place_points(raw, out_case, start)
            (out_case / "points").mkdir(parents=True, exist_ok=True)
            shutil.copyfile(receipt, out_case / "points" / "points.json")
            points_record["points_exe_sha256"] = render_door.sha256_file(points_exe)
        return points_record

    # The point pictures read the same frames as the views and wait on none
    # of them, so they draw beside the regional views.
    points_future = None if args.no_points else pool.submit(draw_points)
    for future in pending:
        future.result()
    points_record = points_future.result() if points_future is not None else None
    pool.shutdown()
    order = ["global", "refined-core", *regions]
    views.sort(key=lambda v: order.index(v["view"]) if v["view"] in order else len(order))
    if not args.keep_scratch:
        shutil.rmtree(scratch_root, ignore_errors=True)
        if not getattr(args, "keep_frame_cache", False):
            shutil.rmtree(frame_cache, ignore_errors=True)
            try:
                frame_cache.parent.rmdir()  # the run's cache folder, once no case is left in it
            except OSError:
                pass

    rendered = sum((v.get("counts") or {}).get("rendered", 0) for v in views)
    opening_scan.join()
    drawn = set((live or {}).get("products", ()))
    for v in views:
        listed = v.get("products_drawn") or ()
        drawn.update(p.strip() for p in (listed.split(",") if isinstance(listed, str) else listed) if p)
    if opening:
        opening["maps"] = [slug for slug in opening["maps"] if slug in drawn]
        if opening["reflectivity_from"]:
            print(f"PRODUCTS reflectivity has echoes from {opening['reflectivity_from']} "
                  f"({opening['reflectivity_peak_dbz']} dBZ)", flush=True)
        print(f"PRODUCTS opening maps: {', '.join(opening['maps']) or 'none'}", flush=True)
    failed_views = [v["view"] for v in views if v.get("rc") not in (0, None)]
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "case": case,
        "simulation_start": start,
        "mesh": str(mesh),
        "init": str(init) if init else None,
        "histories": [str(p) for p in histories],
        "layout": "<out>/<case>/<domain>/<product>/<valid-day>/ ; points: <out>/<case>/points/<sounding|meteogram>-<point>/<day>/",
        "remap": args.remap,
        "view_jobs": jobs,
        "frame_jobs": frame_jobs,
        "views": views,
        "points": points_record,
        "chemistry_products": list(chemistry_products),
        "windows_not_requested": windows_not_requested,
        "maps_rendered": rendered,
        "first_maps": opening or None,
        "live_frames": ({"receipt": live["receipt"], "drawn_live": sorted(live_names),
                         "drawn_here": [p.name for p in redraw]} if live is not None else None),
        "failed_views": failed_views,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    index = write_index(out_case, case, start, opening or None)
    manifest["index"] = index.name if index is not None else None
    columns_index = write_columns_index(out_case, views)
    manifest["columns_index"] = columns_index.name if columns_index is not None else None
    if columns_index is not None:
        print(f"PRODUCTS columns index {columns_index}", flush=True)
    path = out_case / "products-manifest.json"
    render_door.atomic_write_text(path, json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    if index is not None:
        print(f"PRODUCTS index {index}", flush=True)
    soundings = len((points_record or {}).get("placed") or [])
    print(f"PRODUCTS done maps={rendered} point_pictures={soundings} "
          f"failed_views={','.join(failed_views) or 'none'} manifest={path}", flush=True)
    if failed_views or (points_record is not None and points_record.get("rc")):
        return 1
    return 0 if rendered > 0 else 1


def add_products_arguments(parser) -> None:
    parser.add_argument("forecast_dir", nargs="?", default=None, metavar="RUN_DIR",
                        help="the forecast door's --out folder: its history frames, and its "
                             "receipt, which names the mesh, the init and the start time, so "
                             "after a forecast this one argument is the whole command")
    parser.add_argument("--forecast", default=None, metavar="DIR",
                        help="the same folder, by flag")
    parser.add_argument("--history", nargs="+", default=None, metavar="FILE",
                        help="history frames, instead of --forecast")
    parser.add_argument("--mesh", default=None, metavar="FILE",
                        help="the run's grid file (cell centres, connectivity, areas). "
                             "Default: the grid the forecast receipt names")
    parser.add_argument("--init", default=None, metavar="FILE",
                        help="the run's init file: zgrid for pressure-level maps and soundings. "
                             "Default: the init the forecast receipt names")
    parser.add_argument("--out", default=None, metavar="DIR",
                        help="delivered tree root: <out>/<case>/<domain>/<product>/<valid-day>/. "
                             "Default: <RUN_DIR>/products")
    parser.add_argument("--case", default=None,
                        help="case folder name (default: <mesh>-<YYYYMMDDHH> from the data)")
    parser.add_argument("--simulation-start", default=None, metavar="YYYY-MM-DD_HH:MM:SS")
    parser.add_argument("--region", action="append", metavar="NAME=region:CLAT,CLON,W_KM,H_KM",
                        help="an extra regional view (repeatable); dx from the mesh cells inside it")
    parser.add_argument("--points", default=None, metavar="'Name@LAT,LON;...'",
                        help="sounding/meteogram points (default: on a limited-area mesh the cities "
                             "on it, else cities inside a refined core, else a world set). On a "
                             "limited-area mesh a point off the mesh is dropped with a note")
    parser.add_argument("--sounding-every-hours", type=float, default=6.0)
    parser.add_argument("--remap", default=render_door.REMAP_AUTO, choices=render_door.REMAP_CHOICES,
                        help="see `woof hex render --help`; default: product (linear state, "
                             "conservative precipitation)")
    parser.add_argument("--model-label", default=render_door.DEFAULT_MODEL_LABEL,
                        help="the model the maps and soundings name, top left (the mesh "
                             "spacing is added, e.g. 'hex 121 km'; a map's row takes 32 "
                             "characters, a longer name ends in its whole words and '...'); "
                             "default: hex")
    parser.add_argument("--brand", default=None, metavar="TEXT",
                        help="bold text at the top right of every sounding and meteogram, "
                             "e.g. 'WOOF Hex · Global 120 km · Init GFS 22 Oct 2026 00Z' "
                             "(at most 120 characters); default: composed from the run's own "
                             "records (products_door.default_brand)")
    parser.add_argument("--source-label", default=None, metavar="TEXT",
                        help="source credit printed on each map (1-160 printable ASCII "
                             "characters), e.g. which ensemble member it came from; default: "
                             "Hex forecast where the renderer supports it")
    add_view_label_argument(parser)
    parser.add_argument("--global-size", default="1800x1000")
    parser.add_argument("--regional-size", default="1400x1000")
    parser.add_argument("--jobs", type=int, default=3, metavar="N",
                        help="views drawn side by side (default 3; each is its own "
                             "converter and renderer pass over the frames)")
    parser.add_argument("--frame-jobs", default="auto", metavar="N|auto",
                        help="frames each view converts and imports at a time. auto "
                             "(default): from the host's available memory and cores "
                             "(8 history files of memory per frame in flight, half of "
                             "what is available, at most 8); 1 where memory cannot be read")
    parser.add_argument("--tiles", default="auto", choices=render_door.TILE_MODES,
                        help="regional tiles on the regional views (see `woof hex render --tiles`): "
                             "auto (default) adds the six cuts where a view has more grid points "
                             "across than its picture has pixels (a 3 km CONUS domain); regional "
                             "always; none never. The global view is never tiled")
    parser.add_argument("--no-global", action="store_true")
    parser.add_argument("--no-mesh-view", action="store_true")
    parser.add_argument("--no-points", action="store_true")
    parser.add_argument("--live-frames", default=None, metavar="FILE",
                        help="the first look's live-frames.json: frames it rendered into this tree keep their maps; "
                             "this pass draws the whole-run maps, the other frames and the points (the doors pass it)")
    parser.add_argument("--scratch", default=None, metavar="DIR")
    parser.add_argument("--keep-scratch", action="store_true")
    parser.add_argument("--keep-frame-cache", action="store_true",
                        help="leave this run's converted frames for a later pass to reuse (the live draws pass it; "
                             "the full pass reuses them and removes them)")
    parser.add_argument("--basemap-dir", default=None, metavar="DIR")
    parser.add_argument("--convert-exe", default=None, metavar="FILE")
    parser.add_argument("--renderer-exe", default=None, metavar="FILE")
    parser.add_argument("--points-exe", default=None, metavar="FILE")


def add_products_parser(commands) -> None:
    parser = commands.add_parser(
        "products",
        help="forecast history -> the standard product set (maps, soundings, "
             "meteograms) through the Rust renderer, on any mesh")
    add_products_arguments(parser)
    parser.set_defaults(handler=run_products)
