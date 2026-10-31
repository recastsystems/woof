"""The MPAS render door: history files in, product PNGs out, Rust path only.

``mpas-port render`` is the user-reachable command over two Rust binaries:

* ``rw_mpas_convert`` (woof ``tools/rustwx/crates/rw-mpas``) resamples one or
  more MPAS history frames onto a named render window and writes
  wrfout-shaped netCDF frames, with a JSON receipt carrying the mesh,
  weights and per-frame output digests.
* ``rw_wrfbatch`` (woof ``tools/rustwx/crates/rw-wrfbatch``) imports each
  frame and draws the product PNGs.  Product availability is proven by the
  renderer's own ``--list-products`` catalog against the stored fields,
  never guessed from filenames.

This module is deliberately subprocess-and-stdlib only: every byte of field
data is decoded, resampled and drawn in Rust.  Python here is orchestration --
resolving executables, sequencing the two engines, filing the finished PNGs
into the ruled ``<domain>/<product>/<valid-day>/`` layout and writing the
manifest.  There is no fallback plotter behind it.

Native resolution and names (2026-10-10).  Every picture is drawn at the
spacing of the MESH, never a fixed grid: ``--window global`` is the whole
sphere at the mesh's own median spacing and ``--window region:...`` takes
its dx from the mesh cells inside it, both derived by the converter.  Each
PNG is filed as ``<view>/<product>/<valid-day>/<mesh>_<view>_<product>_<lead>.png``
-- ``hex15km_global_500mb_height_winds_f006.png`` -- where ``<mesh>`` is the
grid label the converter read off the mesh.  THE BREAKAGE THIS PREVENTS,
MEASURED: the live global hex gallery drew a 15 km mesh on 20 km and 28 km
grids, headlined every map ``(d01 20 km)`` and named every file
``rustwx_wrf_20261010_0z_f000_d01-20km_<product>.png`` -- the renderer's
generic WRF template, which says neither which model nor which mesh.

Scratch discipline: conversion frames, per-frame render stores and the raw
engine output live in a scratch directory that is a SIBLING of ``--out``,
never inside it.  A ``.rwstore-*`` scratch tree inside a delivered png tree
is a shipped defect this door refuses to repeat; an explicit ``--scratch``
inside ``--out`` is refused by name.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
import datetime as _dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from . import engines
from .errors import MpasPortError


class RenderDoorError(MpasPortError):
    """A named render-door refusal: what breaks, and the remedy."""


#: Environment variables naming the two engine binaries.  The renderer also
#: honors woof's own spelling so a box with a woof install needs no second
#: variable pointed at the same file.
# The preferred spellings carry the distribution's own name.  The
# MPAS_PORT_* pair came from the import package's ORIGINAL name -- the
# package was `mpas_port` through 0.1.1 and is `woof.hex` from 0.2.0 -- and it
# predates the decision that the brand and the spellings a user types never
# carry the MPAS token.  They are kept AHEAD of nothing and BEHIND the new
# names, never removed: an install line that already works must not be
# invalidated by a rename, and a variable that silently stops being read is
# the worst possible way to learn about one.  That rule is why the 0.2.0
# package rename did not touch this pair either: these two strings are the
# one place the retired name is still load-bearing for a user.
#
# The spellings are stated here and CONSUMED by :mod:`woof.hex.engines`,
# which owns the one resolution ladder every door in this distribution
# uses.  Both halves matter: the names live beside the door that
# documents them, and the search itself is not reimplemented per door --
# which is how this door came to read three variables and PATH while the
# init door read one variable and PATH, and neither read the directory
# `woof fetch-bridges` actually stages into.
CONVERT_ENV = "WOOF_HEX_RW_MPAS_CONVERT"
CONVERT_ENV_LEGACY = "MPAS_PORT_RW_MPAS_CONVERT"
RENDERER_ENV = "WOOF_HEX_RW_WRFBATCH"
RENDERER_ENV_LEGACY = "MPAS_PORT_RW_WRFBATCH"
RENDERER_ENV_GPUWM = "WOOF_RW_WRFBATCH"

# The ladder reads them in this order, preferred first, legacy behind,
# woof's own spelling last of the three.  A test binds these tuples to
# the constants above so a rename cannot drop a variable from the search.
assert engines.CONVERT.env_names == (CONVERT_ENV, CONVERT_ENV_LEGACY)
assert engines.RENDERER.env_names == (RENDERER_ENV, RENDERER_ENV_LEGACY)
assert engines.RENDERER.gpuwm_env == RENDERER_ENV_GPUWM

#: The renderer reads its basemap root from this variable first, ahead of
#: everything it can find beside itself.  The door always sets it.
BASEMAP_ENV = "RUSTWX_BASEMAP_DIR"
BASEMAP_ASSETS_ENV = "RUSTWX_ASSETS_DIR"

#: What makes a directory a basemap root: one of the Natural Earth sets the
#: renderer draws coastlines, borders and state lines from.
BASEMAP_MARKERS = (
    ("natural_earth_10m", "ne_10m_coastline.shp"),
    ("natural_earth_110m", "ne_110m_coastline.shp"),
)

#: The manifest schema written beside the delivered tree.
MANIFEST_SCHEMA = "mpas-port.render-door-manifest/v1"

#: Layout tokens for a frame whose facts could not be read (same spellings
#: as woof.render_layout, so a reader of either tree learns one vocabulary).
NATIVE_GRID = "native_grid"
UNDATED = "undated"
UNCLASSIFIED = "unclassified"

#: ``--products`` tokens the renderer expands itself; they name groups, not
#: single catalog rows, so per-slug coverage refusal does not apply to them.
_GROUP_KEYWORDS = frozenset({"all", "direct", "derived", "heavy", "windowed",
                             "variables"})

#: The door's window names and the converter window each one asks for.
#: ``global`` is the whole sphere AT THE MESH'S OWN SPACING
#: (``global-mesh``): the frozen 0.25 degree overview drew a 15 km mesh on a
#: 27.8 km grid, so it is no longer what a bare ``--window global`` means.
#: It stays reachable, unchanged, as ``global-0.25`` -- its weights digests
#: are recorded in evidence and its geometry does not move.
WINDOW_TO_CONVERTER = {
    "mesh": "mesh",
    "focus": "focus",
    "global": "global-mesh",
    "global-0.25": "global",
    "composite": "composite",
}

#: Window specifications passed to the converter as written: ``region:``
#: (ground extent, dx from the mesh), ``lambert:`` (an explicit grid) and
#: ``latlon:`` (an explicit lat-lon grid: the products door's globe of a
#: variable-resolution mesh, at its picture's pixels).
WINDOW_SPEC_PREFIXES = ("region:", "region-latlon:", "lambert:", "latlon:")

#: ``--remap`` vocabulary (see :func:`resolve_remap`).
REMAP_AUTO = "auto"
REMAP_CHOICES = (REMAP_AUTO, "product", "nearest")


def resolve_remap(requested: str, composite: bool) -> str:
    """The converter remap for this render.

    ``auto`` is ``product`` -- linear state, conservative precipitation --
    for a single run, the door default because a nearest-cell map steps at
    every cell edge and carries no accumulation budget; and ``nearest`` for
    a composite, which switches source point by point and has no single mesh
    to interpolate across.  An explicit ``product`` with ``--compose`` is
    refused by that name rather than quietly downgraded."""

    requested = str(requested or REMAP_AUTO)
    if requested not in REMAP_CHOICES:
        raise RenderDoorError(
            f"--remap {requested!r} is not one of {', '.join(REMAP_CHOICES)}")
    if requested == REMAP_AUTO:
        return "nearest" if composite else "product"
    if requested == "product" and composite:
        raise RenderDoorError(
            "--remap product interpolates across one mesh, and a --compose "
            "frame takes each point from whichever run covers it, so there "
            "is no single mesh to interpolate across; use --remap nearest "
            "(or auto) for a composite")
    return requested


#: Regional tiles (``--tiles``).  A window wider than its picture puts several grid points into each
#: pixel, so a 3 km domain the size of the CONUS (RRFS v2's 1800 x 1060 grid) shows none of its own
#: detail on one map.  NOAA's HRRR and RRFS graphics answer that with the whole domain plus regional
#: cuts; this door draws the same set: ``full`` and six cuts, three columns by two rows of the window's
#: own grid points, overlapping so a feature on a seam is whole in at least one of them.  They are
#: product subfolders (render folder law): ``<view>/<product>/<tile>/<valid-day>/``.  Every cut is an
#: exact sub-grid of the full window (same projection, spacing and grid points), converted from the
#: history by the same Rust remap.
TILE_FULL = "full"
#: name -> ((west, east) column fractions, (south, north) row fractions) of the full window.
TILE_CUTS = {
    "nw": ((0.0, 0.4), (0.45, 1.0)),
    "nc": ((0.3, 0.7), (0.45, 1.0)),
    "ne": ((0.6, 1.0), (0.45, 1.0)),
    "sw": ((0.0, 0.4), (0.0, 0.55)),
    "sc": ((0.3, 0.7), (0.0, 0.55)),
    "se": ((0.6, 1.0), (0.0, 0.55)),
}
TILE_NAMES = (TILE_FULL, *TILE_CUTS)
#: ``none`` draws the window whole (the layout with no tile segment); ``regional`` always adds the
#: six cuts; ``auto`` adds them when a Lambert window has more grid points across than its picture
#: has pixels.
TILE_MODES = ("none", "auto", "regional")

_WRF_EARTH_RADIUS_M = 6_370_000.0


def _lambert_cone(truelat1: float, truelat2: float) -> float:
    rad = math.pi / 180.0
    if abs(truelat1 - truelat2) > 0.1:
        return ((math.log10(math.cos(truelat1 * rad)) - math.log10(math.cos(truelat2 * rad)))
                / (math.log10(math.tan((45.0 - abs(truelat1) / 2.0) * rad))
                   - math.log10(math.tan((45.0 - abs(truelat2) / 2.0) * rad))))
    return math.sin(abs(truelat1) * rad)


def lambert_ijll(spec: dict, i: float, j: float) -> tuple[float, float]:
    """Latitude and longitude of grid point ``(i, j)`` (1-based) of a converter Lambert window.

    WRF ``set_lc`` / ``ijll_lc`` as ``rw_mpas::window::LambertProjection`` transcribes them, with
    the window's known point at its centre ``((nx+1)/2, (ny+1)/2)``.  Used only to name the centre
    of a tile cut; the converter lays down every grid point itself."""

    rad, deg = math.pi / 180.0, 180.0 / math.pi
    t1, t2 = float(spec["truelat1"]), float(spec["truelat2"])
    stand_lon = float(spec["stand_lon"])
    radius_m = float(spec.get("earth_radius_m", _WRF_EARTH_RADIUS_M))
    cone = _lambert_cone(t1, t2)
    hemi = 1.0 if t1 >= 0.0 else -1.0
    rebydx = radius_m / float(spec["dx_metres"])
    known_lat, known_lon = float(spec["centre_lat"]), float(spec["centre_lon"])
    known_i, known_j = 0.5 * (int(spec["nx"]) + 1.0), 0.5 * (int(spec["ny"]) + 1.0)
    delta_lon = known_lon - stand_lon
    if delta_lon > 180.0:
        delta_lon -= 360.0
    if delta_lon < -180.0:
        delta_lon += 360.0
    ctl1r = math.cos(t1 * rad)
    rsw = rebydx * ctl1r / cone * (math.tan((90.0 * hemi - known_lat) * rad / 2.0)
                                   / math.tan((90.0 * hemi - t1) * rad / 2.0)) ** cone
    arg = cone * (delta_lon * rad)
    polei = hemi * known_i - hemi * rsw * math.sin(arg)
    polej = hemi * known_j + rsw * math.cos(arg)
    chi1 = (90.0 - hemi * t1) * rad
    chi2 = (90.0 - hemi * t2) * rad
    xx = hemi * i - polei
    yy = polej - hemi * j
    radius_squared = xx * xx + yy * yy
    radius = math.sqrt(radius_squared) / rebydx
    longitude = (stand_lon + deg * math.atan2(hemi * xx, yy) / cone + 360.0) % 360.0
    if abs(chi1 - chi2) < 1.0e-12:
        chi = 2.0 * math.atan((radius / math.tan(chi1)) ** (1.0 / cone) * math.tan(chi1 * 0.5))
    else:
        chi = 2.0 * math.atan((radius * cone / math.sin(chi1)) ** (1.0 / cone) * math.tan(chi1 * 0.5))
    latitude = (90.0 - chi * deg) * hemi
    if radius_squared == 0.0:
        latitude, longitude = hemi * 90.0, stand_lon
    if longitude > 180.0:
        longitude -= 360.0
    elif longitude < -180.0:
        longitude += 360.0
    return latitude, longitude


def _cut(points: int, fractions: tuple[float, float]) -> tuple[int, int]:
    """1-based first and last grid point of a fraction range of ``points`` points."""

    first = 1 + int(math.floor(fractions[0] * (points - 1) + 1e-9))
    last = 1 + int(math.ceil(fractions[1] * (points - 1) - 1e-9))
    return first, max(last, first + 1)


def plan_tiles(window_spec, mode: str, width: int, height: int) -> tuple[list[dict], str]:
    """The regional cuts of a converted window: ``([{name, window, i, j}, ...], why)``.

    ``window`` is the converter window spec of the cut (``lambert:`` or ``latlon:``), an exact
    sub-grid of the full window; ``i``/``j`` are its 1-based point ranges in the full grid.  An
    empty list means the window is drawn whole, and ``why`` says why."""

    if mode not in TILE_MODES:
        raise RenderDoorError(f"--tiles {mode!r} is not one of {', '.join(TILE_MODES)}")
    if mode == "none":
        return [], "tiles off"
    spec = window_spec
    if isinstance(spec, str):
        try:
            spec = json.loads(spec)
        except ValueError:
            spec = None
    if not isinstance(spec, dict) or spec.get("kind") not in ("lambert", "latlon"):
        if mode == "regional":
            raise RenderDoorError(
                "--tiles regional cuts the tiles from the converter's window grid (Lambert or "
                "lat-lon), and this converter's receipt names none, so no tile could be placed on "
                "the full window's grid points; use --tiles none")
        return [], "the converter named no window grid"
    kind = spec["kind"]
    if mode == "auto" and kind != "lambert":
        return [], "a lat-lon (global or region-latlon) window is tiled only with --tiles regional"
    try:
        if kind == "lambert":
            nx, ny = int(spec["nx"]), int(spec["ny"])
            spacing = None
            lambert_ijll(spec, 1.0, 1.0)
        else:
            spacing = float(spec["spacing_degrees"])
            nx = int(round((float(spec["east"]) - float(spec["west"])) / spacing)) + 1
            ny = int(round((float(spec["north"]) - float(spec["south"])) / spacing)) + 1
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        if mode == "regional":
            raise RenderDoorError(
                f"--tiles regional cuts the tiles from the converter's window grid, and its receipt's "
                f"{kind} window spec is incomplete ({type(error).__name__}: {error}), so no tile could "
                f"be placed on the full window's grid points; use --tiles none") from None
        return [], f"the converter's {kind} window spec is incomplete"
    if mode == "auto":
        if nx <= width and ny <= height:
            return [], f"the {nx} x {ny} window fits its {width} x {height} picture"
    plan = []
    for name, (columns, rows) in TILE_CUTS.items():
        i0, i1 = _cut(nx, columns)
        j0, j1 = _cut(ny, rows)
        if kind == "lambert":
            lat, lon = lambert_ijll(spec, 0.5 * (i0 + i1), 0.5 * (j0 + j1))
            window = (f"lambert:{lat!r},{lon!r},{float(spec['dx_metres'])!r},{i1 - i0 + 1},{j1 - j0 + 1},"
                      f"{float(spec['truelat1'])!r},{float(spec['truelat2'])!r},{float(spec['stand_lon'])!r}")
        else:
            south, west = float(spec["south"]), float(spec["west"])
            window = (f"latlon:{south + (j0 - 1) * spacing!r},{south + (j1 - 1) * spacing!r},"
                      f"{west + (i0 - 1) * spacing!r},{west + (i1 - 1) * spacing!r},{spacing!r}")
        plan.append({"name": name, "window": window, "i": [i0, i1], "j": [j0, j1]})
    why = f"{nx} x {ny} window" + ("" if mode == "regional" else f" wider than its {width} x {height} picture")
    return plan, why


def tile_workers(count: int, histories: list[Path] | None, frame_jobs: int) -> tuple[int, int]:
    """``(tile passes at a time, frames each pass converts at a time)``.

    The cuts are independent passes, so they run side by side on the host's cores.  Every pass
    reads whole histories, so how many run at once is also bounded by memory (half of what is
    available, three history files per pass in flight), and the frames-at-a-time the caller sized
    for one pass are shared among the passes, never multiplied."""

    cpus = os.cpu_count() or 1
    workers = max(1, min(count, cpus // 8 or 1))
    available = None
    try:
        with open("/proc/meminfo", encoding="ascii", errors="replace") as stream:
            for line in stream:
                if line.startswith("MemAvailable:"):
                    available = int(line.split()[1]) * 1024
                    break
    except (OSError, ValueError, IndexError):
        available = None
    size = 0
    for item in histories or []:
        try:
            size = max(size, Path(item).stat().st_size)
        except OSError:
            pass
    if available is not None and size > 0:
        workers = max(1, min(workers, int(0.5 * available // (3 * size))))
    return workers, max(1, int(frame_jobs) // workers)


#: carry_georef reads, merges and rewrites one file per view; tile passes run side by side.
_GEOREF_LOCK = threading.Lock()


#: The model this door renders.  Every history it reads was written by
#: woof hex, so its plots say ``hex`` where the wrfout-shaped frames would
#: otherwise import as ``WRF``.
DEFAULT_MODEL_LABEL = "hex"

#: The longest model name the converter writes into a frame, in characters
#: (rw_mpas_convert --model-label; rustwx-products MODEL_LABEL_MAX_CHARS):
#: the metadata row also carries the init time, lead, valid time and grid
#: spacing.  A longer name is refused by the converter, so the door fits it.
MODEL_LABEL_MAX_CHARS = 32

#: The longest headline grid name the converter writes (rw_mpas_convert
#: --grid-label; rw_wrfbatch ignores a longer GPUWM_GRID_LABEL).
GRID_LABEL_MAX_CHARS = 32

#: The token the converter replaces with the spacing of the mesh cells a
#: picture samples (rw_mpas_convert --grid-label 'Core zoom {spacing}').
GRID_LABEL_SPACING_TOKEN = "{spacing}"

#: Converters already said to predate --grid-label, so the door says it once.
_GRID_LABEL_UNSUPPORTED: set[str] = set()
_GRID_LABEL_LOCK = threading.Lock()


def drawn_grid_label(convert_exe: Path, label: str | None) -> str | None:
    """The view label this converter can draw (``--grid-label``), or ``None``.

    THE BREAKAGE THIS PREVENTS (2026-10-30): every hex picture was
    headlined by the WHOLE mesh's range and the picture grid's spacing, so
    the core picture of run_713c9e1913164e02 read ``(hex 133-6.9 km) ...
    Δx 8.063 km`` under the site's ``Core zoom · 8 → 38 km``, and the view
    around the core of run_c108e049fecb4bd0 ``Δx 6.08 km`` under ``Around the
    core · 6 → 96 km``.  A converter built before ``--grid-label`` refuses
    the flag, so it is asked first; such a build keeps the mesh's headline,
    said once per converter rather than failing the view."""

    if not label:
        return None
    if engine_accepts(convert_exe, "--grid-label"):
        return label
    with _GRID_LABEL_LOCK:
        if str(convert_exe) not in _GRID_LABEL_UNSUPPORTED:
            _GRID_LABEL_UNSUPPORTED.add(str(convert_exe))
            print(f"GRID-LABEL {convert_exe} predates --grid-label; its maps keep the mesh's "
                  f"own headline instead of {label!r}", flush=True)
    return None


def fitted_model_label(label: str | None) -> str | None:
    """The model name as the metadata row draws it: runs of whitespace made
    one space, and a name longer than :data:`MODEL_LABEL_MAX_CHARS` shortened
    to its longest run of whole words that fits with ``...`` after it.

    THE BREAKAGE THIS PREVENTS (hex 0.4.10 gate maps, 2026-10-29): the name
    was sliced at 32 characters mid-word with no word said, so every map of a
    run named 'hex 0.4.10 RC HB18 bec7ce68: RRFS ...' read '... bec7ce68: RRF'
    and the MPAS-physics route's read 'MPAS physics (d'."""

    if label is None:
        return None
    text = " ".join(str(label).split())
    if len(text) <= MODEL_LABEL_MAX_CHARS:
        return text
    room = MODEL_LABEL_MAX_CHARS - 3
    head = ""
    for word in text.split(" "):
        candidate = f"{head} {word}" if head else word
        if len(candidate) > room:
            break
        head = candidate
    # A first word longer than the row is cut inside itself; a kept run ends
    # on its last whole word, without a dangling separator before the dots.
    head = (head or text[:room]).rstrip(" :;,-/(")
    return f"{head or text[:room]}..."


#: A view name: the layout's domain folder and the filename's second token.
_VIEW = re.compile(r"^[a-z0-9][a-z0-9-]{0,39}$")

#: The renderer's output filename, as rustwx-products formats it::
#:
#:     rustwx_<model>_<YYYYMMDD>_<H>z_f<NNN>_<domain-slug>_<product-slug>.png
_LEAD_SEGMENT = re.compile(r"^f\d{3}$")

#: ``YYYY-MM-DD`` at the head of an ISO instant or WRF Times stamp.
_DAY = re.compile(r"^(\d{4}-\d{2}-\d{2})(?:[T_ ]|$)")


@dataclass
class FrameResult:
    """What one history frame became."""

    history: str
    wrfout: str
    valid_time: str
    valid_day: str
    output_sha256: str
    absent_wrf_fields: list[str]
    rendered: list[str] = field(default_factory=list)
    skipped: list[dict[str, str]] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    placed: list[str] = field(default_factory=list)
    # Composed frames only: how far the runs disagree where the composite
    # changes source, per surface field.  Empty for a single-run render,
    # which has no seam to measure.
    seam: list[dict] = field(default_factory=list)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(spec: engines.EngineSpec, explicit: str | None) -> Path:
    """Drive the shared ladder, re-raising as this door's own refusal type.

    The refusal TEXT is the ladder's, unedited: it already names every
    rung it searched and both commands that supply the binary, and a
    second paraphrase here would be a second thing to keep true.  Only
    the exception CLASS changes, so a caller catching
    :class:`RenderDoorError` still catches it.
    """

    try:
        return engines.resolve(spec, explicit)
    except engines.EngineRefusal as error:
        raise RenderDoorError(str(error)) from None


def resolve_convert_exe(explicit: str | None = None) -> Path:
    """The ``rw_mpas_convert`` binary, or a named refusal.

    Reads ``--convert-exe``, ``$CONVERT_ENV``, ``$CONVERT_ENV_LEGACY``,
    woof's ``$WOOF_RW_MPAS_CONVERT`` and its bridge directories, then
    ``PATH``.
    """

    return _resolve(engines.CONVERT, explicit)


def resolve_renderer_exe(explicit: str | None = None) -> Path:
    """The ``rw_wrfbatch`` binary, or a named refusal.

    Reads ``--renderer-exe``, ``$RENDERER_ENV``, ``$RENDERER_ENV_LEGACY``,
    ``$RENDERER_ENV_GPUWM`` and woof's bridge directories, then ``PATH``.
    """

    return _resolve(engines.RENDERER, explicit)


def is_basemap_root(path: Path) -> bool:
    """Whether ``path`` holds a Natural Earth set the renderer can draw."""

    return any((Path(path) / folder / shapefile).is_file()
               for folder, shapefile in BASEMAP_MARKERS)


def basemap_candidates(renderer_exe: Path,
                       explicit: str | None = None) -> list[Path]:
    """Where the door looks for the renderer's map geometry, best first.

    1. ``--basemap-dir``;
    2. ``$RUSTWX_BASEMAP_DIR`` and ``$RUSTWX_ASSETS_DIR/basemap`` -- the
       renderer's own variables, which stay the user's to set;
    3. ``assets/basemap`` above the renderer binary, which is where a
       renderer built in a woof checkout already finds it;
    4. woof's own ladder (``woof.rustwx.wrapper_basemap_candidates``:
       the checkout's assets, the ``recast-woof-data`` companion every install
       pulls, the ones ``woof fetch-bridges`` staged) when woof imports;
    5. ``~/.woof/bridges/assets/basemap``, which needs no woof import.
    """

    found: list[Path] = []

    def push(path) -> None:
        path = Path(path)
        if path not in found:
            found.append(path)

    if explicit:
        push(explicit)
    if os.environ.get(BASEMAP_ENV):
        push(os.environ[BASEMAP_ENV])
    if os.environ.get(BASEMAP_ASSETS_ENV):
        push(Path(os.environ[BASEMAP_ASSETS_ENV]) / "basemap")
    try:
        parent = Path(renderer_exe).resolve().parent
    except OSError:
        parent = Path(renderer_exe).parent
    for ancestor in (parent, *parent.parents)[:8]:
        push(ancestor / "assets" / "basemap")
    try:
        from woof import rustwx  # noqa: PLC0415 - lazy on purpose
        for path in rustwx.wrapper_basemap_candidates():
            push(path)
    except Exception:  # noqa: BLE001 - woof absent, or older than the ladder
        pass
    push(engines.USER_BRIDGE_DIR / "assets" / "basemap")
    return found


def resolve_basemap_dir(renderer_exe: Path,
                        explicit: str | None = None) -> Path:
    """The basemap root every renderer call of this door is given.

    THE BREAKAGE THIS PREVENTS, MEASURED (2026-10-11): the renderer finds
    its coastlines, borders and state lines beside its own build tree, and
    the door launched it bare.  A renderer binary copied to another folder
    therefore found none, drew every field with no geography under it,
    exited 0 and reported every product RENDERED: 59 delivered maps had no
    basemap.  The regional path never had this because it launches the
    same binary through ``woof.rustwx``, which pins the root.  So the door
    resolves the root itself, hands it to the renderer by name, and refuses
    before converting anything when there is none.
    """

    candidates = basemap_candidates(renderer_exe, explicit)
    if explicit and not is_basemap_root(Path(explicit)):
        raise RenderDoorError(
            f"--basemap-dir {explicit} holds no Natural Earth set "
            f"(natural_earth_10m/ne_10m_coastline.shp or "
            f"natural_earth_110m/ne_110m_coastline.shp), so every map would "
            f"be drawn without coastlines, borders or state lines. Point it "
            f"at a woof checkout's tools/rustwx/assets/basemap."
        )
    for candidate in candidates:
        if is_basemap_root(candidate):
            return candidate
    searched = ", ".join(str(path) for path in candidates)
    raise RenderDoorError(
        f"no basemap geometry was found for the renderer {renderer_exe}, so "
        f"every map would be drawn without coastlines, borders or state "
        f"lines; nothing is rendered. Searched: {searched}. Pass "
        f"--basemap-dir (or set ${BASEMAP_ENV}) to a directory holding "
        f"natural_earth_10m/ or natural_earth_110m/ -- a woof checkout's "
        f"tools/rustwx/assets/basemap -- or install recast-woof-data, which "
        f"carries them."
    )


#: The credit the start data's licence asks for on every map, named by the host: WOOF sets it for ECMWF's CC BY 4.0
#: open data ("© 2026 ECMWF, CC BY 4.0, modified") and ECCC's Datamart. Each map's source line ends with it, composed
#: here per map so the map keeps its own line (an ensemble member's number) and fitted to the canvas so nothing is
#: cut. Unset (every run whose data asks for no credit), no theme is written and every map is drawn as before.
#: THE BREAKAGE THIS PREVENTS (WOOF, 2026-10-29): a host could print the credit only through one render theme for the
#: whole run, whose source line replaced every map's own, so each hex ensemble member's maps lost their member number;
#: and a line longer than the renderer's source slot (at most half the subtitle row) was cut with "...".
SOURCE_CREDIT_ENV = "WOOF_HEX_SOURCE_CREDIT"

#: The longest source line, in characters, that leaves both halves of the subtitle row whole on a fixed canvas of this
#: size (rw_wrfbatch 2.9.9, measured 2026-10-29 with the ECMWF and ECCC credit lines; at 1200 x 900 a 42-character
#: line already cuts the left half, the run's own provenance). Any other size takes width / 30.
SOURCE_LINE_FITS = {(1800, 1000): 69, (1400, 1000): 56, (1200, 900): 40}


def source_line_budget(width: int, height: int) -> int:
    """How many characters of source line a ``width`` x ``height`` map draws whole (SOURCE_LINE_FITS)."""

    return SOURCE_LINE_FITS.get((int(width), int(height)), max(16, int(width) // 30))


def credited_source_line(label: str | None, credit: str, width: int, height: int) -> str:
    """The map's own source line with the credit after it, in the longest form the canvas draws whole: the whole line
    (``source: Hex forecast, member mem003 of 36 · <credit>``), then shorter forms of the map's own part (``source: Hex
    mem003 of 36``, ``Hex mem003 of 36``, ``mem003``), then the credit alone. The credit, a licence's words, is never
    shortened."""

    own = " ".join(str(label or "Hex forecast").split())
    member = re.fullmatch(r"(?P<model>.*?)(?: forecast)?,? member (?P<id>\S+)(?P<of> of \d+)?", own)
    if member:
        short = " ".join(x for x in (member["model"], member["id"]) if x) + (member["of"] or "")
        heads = (f"source: {own}", f"source: {short}", short, member["id"])
    else:
        short = re.sub(r" forecast\b", "", own).strip() or own
        heads = (f"source: {own}", f"source: {short}", short)
    budget = source_line_budget(width, height)
    for head in dict.fromkeys(heads):
        line = f"{head} · {credit}"
        if len(line) <= budget:
            return line
    return credit


def source_credit_theme(scratch: Path, label: str | None, width: int, height: int) -> str | None:
    """The render theme carrying this map's credited source line (credited_source_line), written in ``scratch``; None
    when the host names no credit (``$WOOF_HEX_SOURCE_CREDIT``) or names a theme of its own (``$RUSTWX_THEME``). The
    theme names no ``extends``: the renderer keeps its plain look and only the source line changes."""

    credit = " ".join(os.environ.get(SOURCE_CREDIT_ENV, "").split())
    if not credit or os.environ.get("RUSTWX_THEME"):
        return None
    path = Path(scratch) / "source-credit-theme.json"
    path.write_text(json.dumps({"text": {"source_label": credited_source_line(label, credit, width, height)}})
                    + "\n", encoding="utf-8")
    return str(path)


def renderer_environment(basemap_dir: Path) -> dict[str, str]:
    """The renderer's environment: this process's, with the basemap root
    named.  The explicit root outranks ``$RUSTWX_ASSETS_DIR`` inside the
    renderer, so one resolved answer is the one that is drawn."""

    env = dict(os.environ)
    env[BASEMAP_ENV] = str(basemap_dir)
    return env


def valid_day(stamp: str | None) -> str:
    """``YYYY-MM-DD`` from an ISO instant, or :data:`UNDATED` -- a wrong
    date is worse than an honest ``undated`` because a reader believes it."""

    if not stamp:
        return UNDATED
    match = _DAY.match(str(stamp).strip())
    if match is None:
        return UNDATED
    try:
        _dt.date.fromisoformat(match.group(1))
    except ValueError:
        return UNDATED
    return match.group(1)


def split_engine_name(filename: str, slug: str) -> tuple[str, str]:
    """``(domain_token, product_dir)`` for one engine output filename.

    The RENDERED event names the product ``slug``, but the FILENAME spells
    it the engine's filesystem-safe way: a generic ``var:wrf_psfc`` event
    becomes ``..._var_wrf_psfc_<hash>.png`` on disk (colon to underscore,
    plus a content-hash suffix).  The product directory uses that safe
    spelling too -- a ``var:`` directory is unrepresentable on Windows and
    would strand the delivered tree on the render node.  The domain token
    is what sits between the ``fNNN`` lead segment and the slug's first
    appearance.  A name that does not parse files under
    (:data:`NATIVE_GRID`, safe slug) -- somewhere nameable, never dropped.
    """

    safe_slug = slug.replace(":", "_")
    stem = Path(filename).stem
    parts = stem.split("_")
    for index, part in enumerate(parts):
        if _LEAD_SEGMENT.match(part):
            tail = "_".join(parts[index + 1 :])
            marker = tail.find(f"_{safe_slug}")
            if marker > 0:
                return tail[:marker], safe_slug
            return NATIVE_GRID, safe_slug
    return NATIVE_GRID, safe_slug


#: The engine's own filename head: ``rustwx_<model>_<YYYYMMDD>_<H>z_f<NNN>_``.
_ENGINE_HEAD = re.compile(r"^rustwx_[^_]+_(?P<day>\d{8})_(?P<hour>\d{1,2})z_f(?P<fhr>\d{3})_")
#: The engine's lead suffix: ``..._lead_<HHH>h<MM>m<SS>s``.
_LEAD_SUFFIX = re.compile(r"_lead_(?P<h>\d+)h(?P<m>\d{2})m(?P<s>\d{2})s$")


def default_label(mesh: Path | None) -> str:
    """``hex_<mesh>`` from the mesh file name: ``x1.2621442.static.nc`` ->
    ``hex_x1.2621442``.  The role tokens (``ordered``, ``static``, ``grid``,
    ``nc``) are dropped; a name that is nothing but role tokens falls back to
    its directory's name, so a label is never empty."""

    if mesh is None:
        return "hex"
    roles = {"ordered", "static", "grid", "nc", "init"}
    tokens = [token for token in Path(mesh).name.split(".") if token and token not in roles]
    stem = ".".join(tokens) or Path(mesh).parent.name
    return f"hex_{stem}" if stem else "hex"


def legacy_product_file_name(label: str, engine_name: str, product: str) -> str:
    """``<label>_<YYYYMMDD>_<HH>z_f<lead>_<product>.png`` for one engine PNG.

    THE BREAKAGE THIS PREVENTS (the owner's naming ruling, restated 2026-10-10):
    the engine names every frame it imports from a wrfout-shaped file
    ``rustwx_wrf_<date>_<H>z_f<NNN>_d01-<dx>km_...``, so a hex global
    forecast's maps were filed under another model's name and a resolution
    token.  The delivered name says the model and mesh (``label``), the
    cycle, the lead and the product.  A sub-hour lead keeps its minutes
    (``f000h30m``) so two frames in one hour never share a name.  An engine
    name that does not parse is kept as it is rather than guessed at.
    """

    stem = Path(engine_name).stem
    head = _ENGINE_HEAD.match(stem)
    if head is None:
        return engine_name
    lead = f"f{head['fhr']}"
    tail = _LEAD_SUFFIX.search(stem)
    if tail is not None:
        hours, minutes, seconds = int(tail["h"]), int(tail["m"]), int(tail["s"])
        lead = f"f{hours:03d}" + (f"h{minutes:02d}m" if minutes or seconds else "")
        if seconds:
            lead += f"{seconds:02d}s"
    return f"{label}_{head['day']}_{int(head['hour']):02d}z_{lead}_{product}.png"
def window_argument(text: str) -> str:
    """argparse ``type`` for ``--window``: a door window name or a spec."""

    import argparse

    value = str(text).strip()
    if value in WINDOW_TO_CONVERTER or value.startswith(WINDOW_SPEC_PREFIXES):
        return value
    raise argparse.ArgumentTypeError(
        f"unknown window {value!r}: choose {', '.join(WINDOW_TO_CONVERTER)}, "
        f"or region:CLAT,CLON,WIDTH_KM,HEIGHT_KM[,TRUELAT1,TRUELAT2] (dx from "
        f"the mesh), region-latlon:CLAT,CLON,WIDTH_KM,HEIGHT_KM (the same box on a "
        f"regular lat-lon grid: what a WOOF domain reads as its parent), or "
        f"lambert:CLAT,CLON,DX_M,NX,NY,TRUELAT1,TRUELAT2,STAND_LON")


def converter_window(window: str) -> str:
    """The converter window a door window asks for (specs pass through)."""

    if window.startswith(WINDOW_SPEC_PREFIXES):
        return window
    try:
        return WINDOW_TO_CONVERTER[window]
    except KeyError:
        raise RenderDoorError(
            f"--window {window!r} is not a render window this door knows; "
            f"choose {', '.join(WINDOW_TO_CONVERTER)} or a region:/lambert: "
            f"spec.") from None


#: The most grid columns the ``global`` window's picture grid carries: the converter's own ceiling for an explicit
#: lat-lon window (rw_mpas_convert refuses more: "explicit latlon exceeds 2000 points per side").
PICTURE_GRID_MAX_COLUMNS = 2000
#: MPAS's sphere: rw_mpas_convert's per-cell spacing is sqrt(2 * areaCell / sqrt(3)) on the unit sphere times this.
_MPAS_EARTH_RADIUS_M = 6_371_229.0
#: The lat-lon window's sphere (rw-mpas WRF_EARTH_RADIUS_M): global-mesh's columns are pi * this / spacing, times 2.
#: Its own name, not the tile planner's _WRF_EARTH_RADIUS_M, so this block stands in a render door without tiles
#: (woof hex 0.4.9 carries it as a hotfix row; its render door has no such name: NameError on canary
#: run_35e9999842049ddb).
_GLOBAL_WINDOW_RADIUS_M = 6_370_000.0


def mesh_median_spacing_m(mesh: Path) -> float | None:
    """The median per-cell spacing of a mesh in metres, the figure ``global-mesh`` is sized from (rw-mpas
    ``read_mesh_cell_spacing``: ``sqrt(2 * areaCell / sqrt(3))`` on the unit sphere, MPAS's radius), read through woof's
    Rust NetCDF decoder. None when the mesh states no ``areaCell`` or ``sphere_radius``: the converter then decides."""

    try:
        import numpy as np
        from woof.netcdf_bridge import open_dataset

        grid = open_dataset(Path(mesh))
        radius = float(grid.getncattr("sphere_radius"))
        area = np.asarray(grid.variables["areaCell"][:], dtype=np.float64).ravel()
    except Exception as error:  # noqa: BLE001 - an unreadable mesh is the converter's to name
        print(f"GLOBAL mesh spacing not read ({type(error).__name__}: {error}); the converter's global-mesh window",
              flush=True)
        return None
    if not (radius > 0.0) or area.size == 0 or not np.all(np.isfinite(area)) or np.any(area <= 0.0):
        return None
    spacing = np.sqrt(2.0 * (area / (radius * radius)) / math.sqrt(3.0)) * _MPAS_EARTH_RADIUS_M
    return float(np.median(spacing))


def global_picture_window(mesh: Path | None, width: int) -> tuple[str, str] | None:
    """The converter window of the ``global`` door window when the mesh is finer than its picture can show, with the
    reason, or None (the mesh's own spacing, ``global-mesh``, as before).

    The picture's grid is the whole sphere on a lat-lon grid with one column per pixel of the picture's width, at
    most :data:`PICTURE_GRID_MAX_COLUMNS`: an 1800-pixel picture is 0.2 degrees (22.2 km), 1800 x 901 = 1.62 M points,
    whatever the mesh and however long the run. A mesh coarser than that (60 and 120 km globes) keeps its own spacing,
    and its picture is unchanged; its subtitle names the render grid either way.

    THE BREAKAGE THIS PREVENTS, MEASURED (owner canary run_b8d705f1f8db9c92, 48 h 96 -> 3 km Arizona zoom globe on
    one RTX PRO 6000 box, 2026-10-30): ``global-mesh`` takes the mesh's MEDIAN cell spacing, three cells in four of that
    mesh are its 3 km core, so the global view was the whole sphere at 9.4 km, 4240 x 2121 = 9.0 M points, for a
    picture 1800 pixels across. The renderer's import holds about 10 KB per grid point per frame (RTX 5090 host, real hex
    frames: 15-17 GB for one 1.62 M-point frame, rising and falling frame by frame, not with the run's length), so
    the global view's renderer needed 85-96 GB, was killed for host memory 21 times, and the products stage exited 1
    after a 3 h forecast (its live passes listed failed_views=global too). The same rule asked 3.6 M points of a
    uniform 15 km globe for the same picture."""

    if mesh is None or width <= 0:
        return None
    columns = min(int(width), PICTURE_GRID_MAX_COLUMNS) // 2 * 2
    median = mesh_median_spacing_m(mesh) if columns >= 4 else None
    if median is None or median <= 0.0:
        return None
    mesh_columns = 2 * math.ceil(math.pi * _GLOBAL_WINDOW_RADIUS_M / median)
    if mesh_columns <= columns:
        return None
    step = 180.0 / (columns // 2)
    why = (f"the mesh's median spacing {median / 1000.0:.3f} km asks {mesh_columns} columns for a picture "
           f"{width} pixels across; drawn at {step:g} degrees ({columns} x {columns // 2 + 1})")
    return f"latlon:-90,90,-180,{180.0 - step!r},{step!r}", why


def default_view(window: str) -> str:
    """The view name a window files under when ``--view`` is not given."""

    if window.startswith(WINDOW_SPEC_PREFIXES):
        return "region"
    return "global" if window.startswith("global") else window


def checked_view(view: str) -> str:
    if not _VIEW.match(view or ""):
        raise RenderDoorError(
            f"--view {view!r} must be lower-case letters, digits and '-', "
            f"starting with a letter or digit (at most 40): it becomes a "
            f"folder and part of every filename, so a space or a slash would "
            f"split one picture's name into a path.")
    return view


def grid_token(grid_label: str | None, engine_domain: str) -> str:
    """``hex 15 km`` -> ``hex15km``; the engine's own domain token when the
    converter named no grid (an older engine), so a name is never empty."""

    if grid_label:
        token = "".join(ch for ch in str(grid_label).lower()
                        if ch.isascii() and (ch.isalnum() or ch in "-."))
        if token:
            return token
    return engine_domain or NATIVE_GRID


def lead_token(lead_seconds, filename: str) -> str:
    """``f006`` for a whole-hour lead, ``f000h30m`` for a sub-hourly one.

    Read from the converter's own ``lead_seconds`` when it reported one;
    otherwise the engine filename's ``fNNN`` segment (whole hours only)."""

    try:
        seconds = int(lead_seconds)
    except (TypeError, ValueError):
        seconds = None
    if seconds is not None and seconds >= 0:
        hours, rest = divmod(seconds, 3600)
        minutes, seconds = divmod(rest, 60)
        lead = f"f{hours:03d}"
        if minutes or seconds:
            lead += f"h{minutes:02d}m"
        if seconds:
            lead += f"{seconds:02d}s"
        return lead
    for part in Path(filename).stem.split("_"):
        if _LEAD_SEGMENT.match(part):
            return part
    return "f-unknown"


def product_file_name(token: str, view: str, product: str, lead: str | None = None) -> str:
    """``hex15km_global_500mb_height_winds_f006.png``: model and mesh,
    view, product, lead."""

    if lead is None:
        return legacy_product_file_name(token, view, product)
    return f"{token}_{view}_{product}_{lead}.png"


def atomic_write_text(path: Path, text: str) -> None:
    """Write ``path`` whole or not at all: a temporary file beside it, then a rename over it.

    Live draws run as several door processes at once into one tree (first_look.py), and a host (the WOOF
    worker) reads these files while they draw; a reader never sees half a JSON file or half a page."""

    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


@contextlib.contextmanager
def folder_lock(folder: Path, name: str):
    """An exclusive lock between door processes on one file of a view folder (POSIX ``flock``; on a host
    without it, the in-process lock only). Held only around a read-merge-write."""

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    with _GEOREF_LOCK:
        try:
            import fcntl
        except ImportError:  # Windows: live draws run on the Linux run boxes; tests run in one process
            yield
            return
        with open(folder / f".{name}.lock", "a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def write_new_json(folder: Path, stem: str, doc: dict) -> Path:
    """``<folder>/<stem>.json``, or ``<stem>-<UTC stamp>.json`` when that name is taken, created exclusively.

    A second render into the same tree keeps the first one's receipt; two door processes drawing one view
    at once (live draws) can no longer both find the plain name free and write over each other."""

    folder = Path(folder)
    text = json.dumps(doc, indent=2, sort_keys=True) + "\n"
    temporary = folder / f".{stem}.{os.getpid()}.{threading.get_ident()}.tmp"
    temporary.write_text(text, encoding="utf-8")
    try:
        candidates = [folder / f"{stem}.json"]
        while True:
            for target in candidates:
                try:
                    os.link(temporary, target)  # atomic and exclusive: fails when the name exists
                    return target
                except FileExistsError:
                    continue
                except OSError:  # a filesystem without hard links: an exclusive create, then the write
                    try:
                        with open(target, "x", encoding="utf-8") as stream:
                            stream.write(text)
                        return target
                    except FileExistsError:
                        continue
            stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            candidates = [folder / f"{stem}-{stamp}.json"]
    finally:
        temporary.unlink(missing_ok=True)


def carry_georef(raw_out: Path, moved: dict[str, str], folder: Path) -> int:
    """Re-key the renderer's ``render-georef.json`` to the filed names.

    The renderer writes one georeference per picture, keyed by the name it
    gave the picture inside its own output folder.  Those names are gone
    once the door files the pictures, so the records are carried to
    ``<folder>/render-georef.json`` (the view's folder) keyed by paths
    relative to that folder, the renderer's own convention; without this
    every delivered picture lost the transform that puts a latitude and
    longitude on its pixels.  Returns the records carried."""

    source = Path(raw_out) / "render-georef.json"
    if not moved or not source.is_file():
        return 0
    try:
        held = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 0
    target = Path(folder) / "render-georef.json"
    # Read, merge and write under one lock: tile passes run side by side in this process, and live draws
    # (first_look.py) run as several door processes into the same view folder.
    with folder_lock(folder, "render-georef"):
        try:
            merged = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            merged = {key: value for key, value in held.items()
                      if key not in ("panels", "without_georeference")}
            merged["panels"] = {}
            merged["without_georeference"] = []
        merged.setdefault("panels", {})
        merged.setdefault("without_georeference", [])
        carried = 0
        for key, record in (held.get("panels") or {}).items():
            new = moved.get(Path(key).as_posix())
            if new is not None:
                merged["panels"][new] = record
                carried += 1
        for absence in held.get("without_georeference") or []:
            new = moved.get(Path(str(absence.get("path", ""))).as_posix())
            if new is not None:
                merged["without_georeference"].append(dict(absence, path=new))
                carried += 1
        for key in ("generated_utc", "schema"):
            if key in held:
                merged[key] = held[key]
        atomic_write_text(target, json.dumps(merged, indent=2, sort_keys=True) + "\n")
    return carried


def place_path(out_root: Path, domain: str, product: str, day: str,
               filename: str, tile: str | None = None) -> Path:
    """``<out>/<domain>/<product>/<valid-day>/<filename>`` -- the ruled
    layout, and the only one this door writes.  Nothing may invert it back
    to flat.  A tiled render (``--tiles``) files each picture one segment
    deeper, under its tile: ``<out>/<domain>/<product>/<tile>/<valid-day>/``."""

    product_dir = Path(out_root) / (domain or NATIVE_GRID) / (product or UNCLASSIFIED)
    if tile:
        product_dir = product_dir / tile
    return product_dir / (day or UNDATED) / filename


def default_scratch(out_root: Path) -> Path:
    """A scratch sibling of ``--out``: same parent, never inside the tree."""

    out_root = Path(out_root)
    return out_root.parent / (out_root.name + ".render-scratch")


def refuse_scratch_inside_out(scratch: Path, out_root: Path) -> None:
    scratch_abs = Path(os.path.abspath(scratch))
    out_abs = Path(os.path.abspath(out_root))
    if scratch_abs == out_abs or out_abs in scratch_abs.parents:
        raise RenderDoorError(
            f"--scratch {scratch} is inside the delivered png tree "
            f"{out_root}. Renderer scratch inside a delivered tree is a "
            f"shipped defect (readers publish half-finished temporaries as "
            f"products); give --scratch a directory outside --out, or omit "
            f"it for the sibling default."
        )


def _require_file(path: Path, flag: str, remedy: str) -> Path:
    path = Path(path)
    if not path.is_file():
        raise RenderDoorError(
            f"{flag} names a missing file: {path}. {remedy}"
        )
    return path


def _run(command: list[str], *, log_prefix: str,
         env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    printable = " ".join(str(part) for part in command)
    print(f"{log_prefix} $ {printable}", flush=True)
    return subprocess.run(
        [str(part) for part in command],
        capture_output=True, text=True, errors="replace", env=env,
    )


def _run_renderer(command: list[str], *, log_prefix: str,
                  env: dict[str, str] | None) -> subprocess.CompletedProcess:
    """A renderer call.  ``env`` carries the basemap root; ``None`` (a
    caller that resolved none, as the unit tests of one stage do) leaves
    the renderer to its own search."""

    if env is None:
        return _run(command, log_prefix=log_prefix)
    return _run(command, log_prefix=log_prefix, env=env)


def convert_frames(convert_exe: Path, *, histories: list[Path] | None,
                   mesh: Path | None, scratch: Path, window: str,
                   field_set: str, simulation_start: str | None,
                   nc_format: str, compose: Path | None = None,
                   compose_base_only: bool = False,
                   init: Path | None = None,
                   diagnostics: Path | None = None,
                   model_label: str | None = None,
                   remap: str = "nearest",
                   jobs: int = 1,
                   explicit_jobs: bool = False,
                   grid_label: str | None = None) -> tuple[dict, list[str]]:
    """Run ``rw_mpas_convert`` once over the whole series.

    Returns ``(report, invocation)`` where ``report`` is the converter's own
    JSON receipt (mesh/weights digests, per-frame output digests and absent
    WRF fields).  A converter refusal is relayed verbatim -- it already
    names the file and the field.

    With ``compose``, the source list in that file owns the meshes and the
    histories and ``histories``/``mesh`` are not passed: two source lists
    that can disagree about which run is the base is a worse failure than
    one flag being unavailable, so the converter refuses the pair and this
    door does not construct it."""

    pairing = None
    legacy_diagnostics = False
    if diagnostics is not None:
        if compose is not None or len(histories or []) != 1:
            raise RenderDoorError("--diagnostics requires one history and no --compose: it names the "
                                  "diag stream file of that one history's valid time")
        # A converter that names --diagnostics-sha256 in its usage reads the file through that flag
        # (the 2026-10-14 split-history reader); the converter woof ships reads a history's diag
        # stream file of the same time from beside the history (history.T.nc -> diag.T.nc) and has no
        # such flag.  THE BREAKAGE THIS PREVENTS: the door passed --diagnostics to that converter and
        # every `woof hex render --diagnostics` exited 2 ("unknown flag --diagnostics", RRFS bigmesh
        # maps, 2026-10-28).  The named file is now paired with the history under the names the
        # converter pairs on, and the converter's receipt must name it back (checked below).
        legacy_diagnostics = engine_accepts(convert_exe, "--diagnostics-sha256")
        if legacy_diagnostics:
            if init is None:
                raise RenderDoorError(
                    "--diagnostics requires one history, --init, and no --compose: this converter "
                    "reads a named diagnostics file only beside a pinned --init")
        else:
            pairing = pair_diagnostics(Path(histories[0]), Path(diagnostics), Path(scratch))
            histories = [pairing["history"]]
    frames_dir = Path(scratch) / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(scratch) / "convert-report.json"
    command: list[str] = [str(convert_exe)]
    if compose is not None:
        command.extend(("--compose", str(compose)))
        if compose_base_only:
            command.append("--compose-base-only")
    else:
        command.extend(("--history", *[str(item) for item in histories or []]))
        command.extend(("--mesh", str(mesh)))
        if init is not None:
            # The converter sources PHB (g * zgrid) from the run's init file
            # and nothing else: a history frame carries no zgrid, so without
            # this every pressure-level product (500 hPa height and winds,
            # 850 hPa) is refused as missing-fields on every run.
            command.extend(("--init", str(init)))
        if diagnostics is not None and legacy_diagnostics:
            command.extend(("--diagnostics", str(diagnostics),
                            "--diagnostics-sha256", sha256_file(diagnostics),
                            "--init-sha256", sha256_file(init)))
    command.extend((
        "--out-dir", str(frames_dir),
        "--window", window,
        "--field-set", field_set,
        "--format", nc_format,
        "--clobber",
        "--json", str(report_path),
    ))
    if simulation_start:
        command.extend(("--simulation-start", simulation_start))
    drawn_label = fitted_model_label(model_label)
    if drawn_label:
        # The plot's metadata row names this model instead of the generic
        # WRF identity every wrfout-shaped frame imports under, fitted to
        # the converter's limit (fitted_model_label) and said so when cut.
        if drawn_label != " ".join(str(model_label).split()):
            print(f"LABEL model label shortened to the metadata row's {MODEL_LABEL_MAX_CHARS} "
                  f"characters: {model_label!r} -> {drawn_label!r}", flush=True)
        command.extend(("--model-label", drawn_label))
    if grid_label:
        # The picture's own headline name (drawn_grid_label): passed only to a
        # converter that takes it, which also keeps the mesh's label in its
        # receipt as mesh_grid_label for the delivered file names.
        command.extend(("--grid-label", grid_label))
    if remap != "nearest":
        # Linear state, conservative accumulations, nearest categories
        # (rw_mpas_convert --help).  Passed only when asked so an older
        # converter that predates the flag still runs a nearest render.
        command.extend(("--remap", remap))
    if (jobs > 1 or explicit_jobs) and compose is None:
        # Frames are independent, so the converter takes several at a time;
        # passed only when asked, so an older converter still runs.  A tile
        # pass passes it even at 1: left to itself the converter sizes its
        # frames in flight against the host's whole free memory, which the
        # tile passes beside it are using too.
        command.extend(("--jobs", str(jobs)))
    result = _run(command, log_prefix="CONVERT")
    for line in (result.stdout or "").splitlines():
        if line.startswith(("WINDOW\t", "WEIGHTS\t", "CONVERTED\t",
                            "COMPOSITE\t", "SEAM\t", "FINISHED\t",
                            "REMAP\t", "BUDGET\t")):
            print(f"CONVERT {line}", flush=True)
    # A composite drops any hour a source does not carry, and says so on
    # stderr.  Relayed rather than swallowed: a frame silently missing from
    # a delivered series reads as a gap in the weather.
    for line in (result.stderr or "").splitlines():
        if "dropping" in line:
            print(f"CONVERT {line.strip()}", flush=True)
    if result.returncode != 0:
        tail = [line for line in (result.stderr or "").splitlines()
                if line.strip()]
        raise RenderDoorError(
            f"rw_mpas_convert exited {result.returncode}: "
            f"{tail[-1] if tail else 'no stderr'}"
        )
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RenderDoorError(
            f"rw_mpas_convert exited 0 but its JSON receipt at "
            f"{report_path} is unreadable ({error}); the receipt is the "
            f"digest evidence, so the render does not proceed without it."
        ) from error
    if pairing is not None:
        report["diagnostics"] = checked_diagnostics_pairing(report, pairing)
    return report, [str(part) for part in command]


#: Free space a frame cache leaves on its disk: at least this much and this share of the disk, so the forecast still
#: writing its frames beside it never runs short (a frame the cache does not keep is converted again, never lost).
FRAME_CACHE_FREE_MARGIN = 20 * 1024 ** 3
FRAME_CACHE_FREE_SHARE = 0.10


def _frame_cache_key(convert_exe: Path, **inputs) -> str:
    """The cache folder of one converter setting: the converter's bytes and every input that shapes its frames."""

    parts = [sha256_file(convert_exe)]
    for name in sorted(inputs):
        value = inputs[name]
        if isinstance(value, Path):
            try:
                st = value.stat()
                value = f"{value.resolve()}:{st.st_size}:{st.st_mtime_ns}"
            except OSError:
                value = str(value)
        parts.append(f"{name}={value}")
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:24]


def _history_token(history: Path) -> str | None:
    try:
        st = Path(history).stat()
    except OSError:
        return None
    return f"{Path(history).name}.{st.st_size}.{st.st_mtime_ns}"


def convert_frames_cached(cache: Path | None, convert_exe: Path, *, histories: list[Path], **kwargs) -> tuple[dict, list[str]]:
    """:func:`convert_frames`, reusing the frames this run's live draws already converted for the same view.

    THE BREAKAGE THIS PREVENTS (an Arizona zoom globe, 8 x RTX PRO 6000, woof hex 0.4.10, 2026-10-30, and the
    RTX 5090 replay of the same door): the whole-run maps (qpf_1h, the run's maximum wind, ...) need every frame in one
    renderer store, so after the forecast each view converted every frame again, the ones its live draws had just
    converted included; on an RTX 5090 that second conversion was 70 of the pass's 110 s. A live draw links each frame it
    converts into ``cache`` (``<products>.frame-cache-scratch/<case>/<view>/<key>/``, the key being the converter's bytes and
    every input that shapes a frame), and a later pass of the same view converts only the frames not there. The bytes
    are the converter's own from the same inputs, so the pictures are the same. The full pass removes the cache."""

    window = kwargs.get("window")
    if cache is None or not histories or kwargs.get("compose") is not None or kwargs.get("diagnostics") is not None:
        return convert_frames(convert_exe, histories=histories, **kwargs)
    key = _frame_cache_key(convert_exe, mesh=kwargs.get("mesh"), init=kwargs.get("init"), window=window,
                           field_set=kwargs.get("field_set"), simulation_start=kwargs.get("simulation_start"),
                           nc_format=kwargs.get("nc_format"), model_label=kwargs.get("model_label"),
                           remap=kwargs.get("remap"), grid_label=kwargs.get("grid_label"))
    folder = Path(cache) / key
    found: list[dict | None] = []
    for history in histories:
        token = _history_token(history)
        record = None
        if token is not None:
            try:
                record = json.loads((folder / token / "frame.json").read_text(encoding="utf-8"))
                if not Path(record["output"]).is_file():
                    record = None
            except (OSError, ValueError, KeyError, TypeError):
                record = None
        found.append(record)
    head = None
    try:
        head = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        head = None
    missing = [h for h, r in zip(histories, found) if r is None]
    invocation: list[str] = ["frame-cache", str(folder)]
    if missing or head is None:
        if head is None:
            missing, found = list(histories), [None] * len(histories)
        report, invocation = convert_frames(convert_exe, histories=missing, **kwargs)
        new = list(report.get("frames") or [])
        if len(new) != len(missing):
            return report if len(missing) == len(histories) else convert_frames(
                convert_exe, histories=histories, **kwargs)
        head = {k: v for k, v in report.items() if k != "frames"}
        try:
            folder.mkdir(parents=True, exist_ok=True)
            atomic_write_text(folder / "report.json", json.dumps(head, sort_keys=True) + "\n")
        except OSError:
            pass
        made = iter(new)
        for index, (history, record) in enumerate(zip(histories, found)):
            if record is not None:
                continue
            meta = dict(next(made))
            found[index] = meta
            token = _history_token(history)
            output = Path(meta.get("output", ""))
            try:
                size = output.stat().st_size
                disk = shutil.disk_usage(folder)
                if token is None or disk.free - size < max(FRAME_CACHE_FREE_MARGIN, FRAME_CACHE_FREE_SHARE * disk.total):
                    continue
                target = folder / token / output.name
                target.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(output, target)
                except FileExistsError:
                    pass
                atomic_write_text(folder / token / "frame.json", json.dumps({**meta, "output": str(target)}) + "\n")
            except OSError:
                continue  # a disk that takes no hard link (or is short of room): this frame is converted again
    reused = len(histories) - len(missing)
    if reused:
        print(f"FRAME-CACHE {reused} of {len(histories)} frames reused from this run's earlier draws "
              f"({folder.name}); {len(missing)} converted now", flush=True)
    return {**head, "frames": found}, invocation


#: The valid-time stamp MPAS stream file names carry (``history.2023-03-31_18.00.00.nc``).
_STREAM_STAMP = re.compile(r"\d{4}-\d{2}-\d{2}_\d{2}[.:]\d{2}[.:]\d{2}")


def pair_diagnostics(history: Path, diagnostics: Path, scratch: Path) -> dict:
    """Put ``diagnostics`` where the converter reads a history's diag stream file.

    ``rw_mpas_convert`` completes a history from the file whose name is the history's with its last
    ``history`` token replaced by ``diag``, in the same directory (``history.T.nc`` ->
    ``diag.T.nc``, the template MPAS-A and the RRFS fork both write).  When the named file already
    is that file the history is converted where it is; otherwise both are linked (hard link, else
    symbolic link, else copied) into ``<scratch>/diagnostics-pair/`` under those names.  Returns
    ``{history, diag, given, sha256, how}``."""

    name = history.name
    at = name.rfind("history")
    if at < 0:
        raise RenderDoorError(
            f"--diagnostics pairs the diag file with its history by name (history.T.nc beside "
            f"diag.T.nc), and {name} carries no 'history' token to pair on, so the converter "
            f"could not find it; rename the history file to the MPAS template or drop --diagnostics")
    companion = name[:at] + "diag" + name[at + len("history"):]
    # The converter checks a diag file's cells and levels, not its time: the pairing by name is
    # what makes it the same valid time.  A named file whose own name carries another time is
    # refused, so the maps never mix two times.
    stamps = [_STREAM_STAMP.search(text) for text in (name, Path(diagnostics).name)]
    if all(stamps) and stamps[0].group(0) != stamps[1].group(0):
        raise RenderDoorError(
            f"--diagnostics {Path(diagnostics).name} is the diag stream of {stamps[1].group(0)}, and "
            f"--history {name} is {stamps[0].group(0)}; a frame completed from it would draw two "
            f"valid times as one")
    record = {"given": str(diagnostics), "sha256": sha256_file(diagnostics)}
    beside = history.with_name(companion)
    try:
        if beside.exists() and os.path.samefile(beside, diagnostics):
            return {**record, "history": history, "diag": beside, "how": "beside"}
    except OSError:
        pass
    pair = Path(scratch) / "diagnostics-pair"
    pair.mkdir(parents=True, exist_ok=True)
    hows = []
    for source, target in ((history, pair / name), (diagnostics, pair / companion)):
        if target.is_symlink() or target.exists():
            target.unlink()
        try:
            os.link(source, target)
            hows.append("hard link")
        except OSError:
            try:
                os.symlink(Path(source).resolve(), target)
                hows.append("symbolic link")
            except OSError:
                shutil.copyfile(source, target)
                hows.append("copy")
    print(f"DIAGNOSTICS paired {diagnostics} with {history.name} as {companion} "
          f"({', '.join(sorted(set(hows)))})", flush=True)
    return {**record, "history": pair / name, "diag": pair / companion, "how": ", ".join(hows)}


def checked_diagnostics_pairing(report: dict, pairing: dict) -> dict:
    """The converter's receipt must name the paired diag file (by path, and by digest where it
    records one); a frame it completed from another file is refused, never drawn."""

    frames = report.get("frames") or []
    meta = frames[0] if frames else {}
    named = None
    named_sha = None
    fields: list = []
    for key in ("split_streams", "diag_stream"):
        block = meta.get(key) if isinstance(meta, dict) else None
        if isinstance(block, dict) and block.get("diag"):
            named = block.get("diag")
            named_sha = block.get("diag_sha256")
            fields = list(block.get("diag_fields") or [])
            break
    record = {"file": pairing["given"], "sha256": pairing["sha256"], "paired_as": str(pairing["diag"]),
              "how": pairing["how"], "history": str(pairing["history"]), "read": named is not None,
              "fields": fields}
    if named is None:
        print("DIAGNOSTICS not read: the history carries every field the converter maps", flush=True)
        return record
    try:
        same = os.path.samefile(named, pairing["diag"])
    except OSError:
        same = Path(named) == Path(pairing["diag"])
    if not same or (named_sha and named_sha != pairing["sha256"]):
        raise RenderDoorError(
            f"rw_mpas_convert completed {pairing['history']} from {named} (sha256 {named_sha}), not "
            f"the --diagnostics file {pairing['given']} (sha256 {pairing['sha256']}); the frame is "
            f"not drawn from fields of another file")
    print(f"DIAGNOSTICS read {len(fields)} field(s) from {pairing['given']}: {', '.join(fields)}", flush=True)
    return record


def list_catalog(renderer_exe: Path, wrfout: Path | list[Path], *, scratch: Path,
                 heavy: bool, store: Path | None = None,
                 products: str | None = None, env: dict[str, str] | None = None,
                 import_jobs: int = 1
                 ) -> tuple[dict[str, tuple[str, str, str]], str, list[str]]:
    """The renderer's own product catalog for one converted frame, or for
    a whole series imported into one store (a list of frames): only a
    store that holds several frames can close an accumulation window, so
    only the series listing can prove ``qpf_6h`` and its kin.

    Returns ``(rows, summary, invocation)``; ``rows`` maps slug ->
    (kind, status, detail).  Statuses are proven against the real stored
    import, never guessed.

    With ``store`` and ``products`` the listing imports the frame into the
    store its render will use, exactly as that render imports it, so the
    render finds the import there and does not repeat it.  MEASURED (8-card RTX PRO 6000 host,
    2026-10-10, one 15 km global frame at native spacing): the import is
    about two minutes of the render and drawing ten products 1.3 s, and the
    door imported every frame twice -- once in full into a catalog store,
    once for the render."""

    store = Path(store) if store is not None else Path(scratch) / "catalog-store"
    store.mkdir(parents=True, exist_ok=True)
    command = [
        str(renderer_exe),
        "--store-root", str(store),
        "--out-dir", str(store),
        "--list-products",
    ]
    if products is not None:
        command.extend(("--products", products))
    if heavy:
        command.append("--heavy")
    if import_jobs > 1:
        command.extend(("--import-jobs", str(import_jobs)))
    if isinstance(wrfout, (list, tuple)):
        command.extend(str(item) for item in wrfout)
        wrfout = wrfout[0]
    else:
        command.append(str(wrfout))
    result = _run_renderer(command, log_prefix="CATALOG", env=env)
    if result.returncode != 0:
        tail = [line for line in (result.stderr or "").splitlines()
                if line.strip()]
        raise RenderDoorError(
            f"rw_wrfbatch --list-products exited {result.returncode} on "
            f"{wrfout}: {tail[-1] if tail else 'no stderr'}"
        )
    rows: dict[str, tuple[str, str, str]] = {}
    summary = ""
    for line in (result.stdout or "").splitlines():
        if line.startswith("PRODUCT\t"):
            # Five fields or more, never exactly five. The renderer's row
            # grew a sixth field (a machine reason code) in woof 2.8.0; a
            # parser that demanded exactly five dropped every row of that
            # build and refused every render with "no catalog rows".
            parts = line.split("\t")
            if len(parts) >= 5:
                rows[parts[1]] = (parts[2], parts[3], parts[4])
        elif line.startswith("CATALOG "):
            summary = line[len("CATALOG "):]
    if not rows:
        raise RenderDoorError(
            f"rw_wrfbatch produced no catalog rows for {wrfout}; the import "
            f"failed silently, so product availability cannot be proven and "
            f"the render does not proceed."
        )
    return rows, summary, [str(part) for part in command]


def refuse_uncovered_products(products: str, rows: dict, history: Path,
                              absent: list[str]) -> None:
    """Refuse, by product name, when a requested product cannot be drawn.

    Group keywords (``all``, ``derived``, ...) expand to whatever IS
    renderable, so they never refuse; an explicitly named product that the
    catalog marks non-renderable stops the run before anything is drawn,
    so a partial gallery cannot pass as the requested one."""

    requested = [token.strip() for token in products.split(",")
                 if token.strip()]
    for token in requested:
        if token in _GROUP_KEYWORDS:
            continue
        row = rows.get(token)
        if row is None:
            known = ", ".join(sorted(slug for slug, (_k, status, _d)
                                     in rows.items()
                                     if status == "renderable"))
            raise RenderDoorError(
                f"product '{token}' is not in the renderer catalog for "
                f"{history}; nothing by that name can be drawn. Renderable "
                f"products from this history: {known}."
            )
        kind, status, detail = row
        if status != "renderable":
            hint = (f" (fields absent from the source history: "
                    f"{', '.join(absent)})" if absent else "")
            if kind == "windowed":
                # The absent-field list alone read as the cause of a window
                # refusal (RRFS hrrrv5 qpf_1h, 2026-10-28: "RAINC" named,
                # while one frame per store was the cause); say what a
                # window needs first.
                hint = (" (a window is the difference of stored frames of "
                        "one run: give --history two or more frames with "
                        "distinct leads, every hour for an hourly window"
                        + (f"; fields absent from the source history: "
                           f"{', '.join(absent)}" if absent else "") + ")")
            raise RenderDoorError(
                f"product '{token}' is not renderable from {history}: "
                f"{status} -- {detail}{hint}. The render stops before "
                f"drawing anything so a partial gallery cannot pass as the "
                f"requested one; drop the product from --products or rerun "
                f"the model with the fields it needs."
            )


#: The one store and raw folder a series render uses (:func:`run_render`).
SERIES_TAG = "series"


#: Graticule spacings, degrees, smallest first.
GRATICULE_STEPS = (0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 30.0)
#: Most lines across a regional window's longer side.
GRATICULE_MAX_LINES = 6
GRATICULE_LINE_COLOUR = "#4a525c80"


def _wrap180(value):
    return ((value + 180.0) % 360.0) - 180.0


def _degree_text(value: float, step: float, positive: str, negative: str) -> str:
    number = f"{abs(value):.0f}" if step >= 1.0 else f"{abs(value):g}"
    if abs(value) < 1e-9 or abs(abs(value) - 180.0) < 1e-9:
        return f"{number}°"
    return f"{number}°{positive if value > 0 else negative}"


def graticule_overlays(lat, lon) -> dict | None:
    """Latitude/longitude lines and their labels for one converted frame,
    in the renderer's ``--overlays`` schema (degrees).

    THE BREAKAGE THIS PREVENTS (hex storm-following AL10, GFS 2026-10-22
    18Z, f012): a 3 km moving region over open ocean was drawn as
    reflectivity on ocean fill with no line of geography at all, so a
    reader could not tell where the storm was.  The renderer draws
    coastlines and borders, but a window with no land in it had nothing
    else; every hex map now carries a labelled graticule.  ``lat``/``lon``
    are the frame's XLAT/XLONG (2-D, south row first)."""

    import numpy as np

    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    if lat.ndim != 2 or lat.shape != lon.shape or min(lat.shape) < 4:
        return None
    ny, nx = lat.shape
    centre = float(lon[ny // 2, nx // 2])
    offset = _wrap180(lon - centre)
    if not (np.isfinite(lat).all() and np.isfinite(offset).all()):
        return None
    south, north = float(lat.min()), float(lat.max())
    west, east = float(offset.min()), float(offset.max())
    whole_globe = east - west > 300.0
    if whole_globe:
        step = 30.0
    else:
        mid = math.radians(0.5 * (south + north))
        span = max(north - south, (east - west) * max(0.2, math.cos(mid)))
        step = next((s for s in GRATICULE_STEPS if span / s <= GRATICULE_MAX_LINES),
                    GRATICULE_STEPS[-1])
    edge = 1e-3
    west, east = west + edge, east - edge
    parallels = [k * step for k in range(math.ceil(south / step), math.floor(north / step) + 1)
                 if south < k * step < north and abs(k * step) < 90.0]
    meridians = [k * step for k in range(math.ceil((centre + west) / step),
                                         math.floor((centre + east) / step) + 1)
                 if west < k * step - centre < east]
    densify = 2.0 if whole_globe else step / 10.0
    lines, labels = [], []
    for value in parallels:
        count = max(2, int(math.ceil((east - west) / densify)) + 1)
        points = [(value, _wrap180(centre + west + (east - west) * i / (count - 1)))
                  for i in range(count)]
        lines.append({"points": points, "color": GRATICULE_LINE_COLOUR, "width": 1})
    for value in meridians:
        count = max(2, int(math.ceil((north - south) / densify)) + 1)
        points = [(south + (north - south) * i / (count - 1), _wrap180(value))
                  for i in range(count)]
        lines.append({"points": points, "color": GRATICULE_LINE_COLOUR, "width": 1})
    # Labels sit on the lines just inside the west and south edges (on a
    # whole globe: between the first two meridians and two parallels).
    if whole_globe:
        label_lon = _wrap180(centre - 165.0)
        label_lat = -45.0
        for value in parallels:
            labels.append({"lat": value, "lon": label_lon,
                           "text": _degree_text(value, step, "N", "S")})
        for value in meridians:
            labels.append({"lat": label_lat, "lon": _wrap180(value),
                           "text": _degree_text(_wrap180(value), step, "E", "W")})
    else:
        column = min(nx - 1, max(1, nx // 20))
        row = min(ny - 1, max(1, ny // 20))
        for value in parallels:
            j = int(np.argmin(np.abs(lat[:, column] - value)))
            if abs(lat[j, column] - value) > 0.5 * step or j in (0, ny - 1):
                continue
            labels.append({"lat": value, "lon": float(lon[j, column]),
                           "text": _degree_text(value, step, "N", "S")})
        for value in meridians:
            i = int(np.argmin(np.abs(offset[row, :] - (value - centre))))
            if abs(offset[row, i] - (value - centre)) > 0.5 * step or i in (0, nx - 1):
                continue
            labels.append({"lat": float(lat[row, i]), "lon": _wrap180(value),
                           "text": _degree_text(_wrap180(value), step, "E", "W")})
    if not lines:
        return None
    return {"lines": lines, "labels": labels, "step_deg": step}


def write_graticule_overlays(frame: Path, scratch: Path) -> tuple[Path | None, dict | None]:
    """Write :func:`graticule_overlays` for ``frame`` (a converted
    wrfout-shaped file) into ``scratch``; ``(None, None)`` when the frame
    carries no usable XLAT/XLONG."""

    import netCDF4

    from .netcdf_lock import NETCDF

    try:
        with NETCDF, netCDF4.Dataset(str(frame)) as dataset:
            if "XLAT" not in dataset.variables or "XLONG" not in dataset.variables:
                return None, None
            lat = dataset.variables["XLAT"][:]
            lon = dataset.variables["XLONG"][:]
    except OSError as error:
        # The renderer reads the same file and names its own failure.
        print(f"GRATICULE unreadable frame {frame}: {error}", flush=True)
        return None, None
    while getattr(lat, "ndim", 0) > 2:
        lat, lon = lat[0], lon[0]
    overlays = graticule_overlays(lat, lon)
    if overlays is None:
        return None, None
    step = overlays.pop("step_deg")
    path = Path(scratch) / "graticule-overlays.json"
    path.write_text(json.dumps(overlays) + "\n", encoding="utf-8")
    return path, {"step_deg": step, "lines": len(overlays["lines"]),
                  "labels": len(overlays["labels"])}


def render_frame(renderer_exe: Path, wrfout: Path | list[Path], *, scratch: Path,
                 index: int, products: str, frames: str, width: int,
                 height: int, heavy: bool,
                 env: dict[str, str] | None = None,
                 tag: str | None = None,
                 import_jobs: int = 1,
                 source_label: str | None = None,
                 overlays: Path | None = None
                 ) -> tuple[list[tuple[str, Path]], list[dict[str, str]],
                            list[str], list[str]]:
    """One renderer invocation over one store: a per-frame store (campaign
    convention), or with ``tag`` and a list of frames the series store.

    Returns ``(rendered, skipped, failed, invocation)`` where ``rendered``
    is ``[(slug, png_path), ...]``.  Only FAILED is a failure; an honest
    SKIPPED is relayed, never hidden."""

    tag = tag or f"{index:03d}"
    store = Path(scratch) / f"store-{tag}"
    raw_out = Path(scratch) / f"raw-{tag}"
    store.mkdir(parents=True, exist_ok=True)
    raw_out.mkdir(parents=True, exist_ok=True)
    command = [
        str(renderer_exe),
        "--store-root", str(store),
        "--out-dir", str(raw_out),
        "--products", products,
        "--frames", frames,
        "--width", str(width),
        "--height", str(height),
    ]
    if heavy:
        command.append("--heavy")
    if import_jobs > 1:
        command.extend(("--import-jobs", str(import_jobs)))
    if source_label is not None:
        command.extend(("--source-label", source_label))
    if overlays is not None:
        command.extend(("--overlays", str(overlays)))
    if isinstance(wrfout, (list, tuple)):
        command.extend(str(item) for item in wrfout)
    else:
        command.append(str(wrfout))
    result = _run_renderer(command, log_prefix="RENDER", env=env)
    rendered: list[tuple[str, Path]] = []
    skipped: list[dict[str, str]] = []
    failed: list[str] = []
    for line in (result.stdout or "").splitlines():
        if line.startswith("RENDERED "):
            slug, _, path = line[len("RENDERED "):].partition(" ")
            if path:
                rendered.append((slug, Path(path)))
        elif line.startswith("SKIPPED "):
            slug, _, reason = line[len("SKIPPED "):].partition(" ")
            skipped.append({"product": slug,
                            "reason": reason or "no reason given"})
    for line in (result.stderr or "").splitlines():
        if line.startswith("FAILED "):
            failed.append(line[len("FAILED "):])
    if result.returncode != 0 and not failed:
        tail = [line for line in (result.stderr or "").splitlines()
                if line.strip()]
        failed.append(tail[-1] if tail else f"exit {result.returncode}")
    return rendered, skipped, failed, [str(part) for part in command]


def engine_lead_seconds(filename: str) -> int | None:
    """The lead one engine PNG is valid at, from its own name: the
    ``_lead_HHHhMMmSSs`` suffix a sub-hourly frame carries, else the
    ``fNNN`` segment.  ``None`` when the name is not the engine's."""

    stem = Path(filename).stem
    tail = _LEAD_SUFFIX.search(stem)
    if tail is not None:
        return int(tail["h"]) * 3600 + int(tail["m"]) * 60 + int(tail["s"])
    head = _ENGINE_HEAD.match(stem)
    if head is not None:
        return int(head["fhr"]) * 3600
    return None


def series_leads(frames_meta: list[dict]) -> list[int] | None:
    """Each frame's lead in seconds when a series can share one store:
    every frame reports a lead and no two frames share one.  ``None``
    otherwise, and the render falls back to a store per frame -- a picture
    is attributed to its frame by its lead, so two frames at one lead (or a
    converter that reported none) cannot be told apart in a shared store."""

    leads: list[int] = []
    for meta in frames_meta:
        try:
            lead = int(meta.get("lead_seconds"))
        except (TypeError, ValueError):
            return None
        if lead < 0 or lead in leads:
            return None
        leads.append(lead)
    return leads if len(leads) > 1 else None


def engine_accepts(exe: Path, flag: str) -> bool:
    """Whether an engine binary's own usage text names ``flag``.

    The frame-parallel flags (``rw_mpas_convert --jobs``, ``rw_wrfbatch
    --import-jobs``) arrived with the product door; an engine built before
    them refuses an unknown flag, so the door asks the binary instead of
    assuming, and runs one frame at a time on a build that does not say
    yes."""

    try:
        result = subprocess.run([str(exe), "--help"], capture_output=True,
                                text=True, errors="replace", timeout=60)
    except (OSError, subprocess.SubprocessError):
        return False
    return flag in (result.stdout or "") + (result.stderr or "")


def checked_frame_jobs(value) -> int:
    try:
        jobs = int(value)
    except (TypeError, ValueError):
        jobs = 0
    if not 1 <= jobs <= 64:
        raise RenderDoorError(
            f"--frame-jobs {value!r} must be a whole number from 1 to 64: it "
            f"is how many frames are converted and imported at a time")
    return jobs


def run_render(args) -> int:
    """The ``mpas-port render`` handler.  Returns the process exit code."""

    return _run_render(args)[0]


def _run_tiles(args, plan: list[dict], *, view: str, scratch: Path, histories: list[Path] | None,
               frame_jobs: int, why: str) -> dict:
    """Draw every cut of ``plan`` (see :func:`plan_tiles`), side by side, each its own pass."""

    import argparse

    workers, per_tile = tile_workers(len(plan), histories, frame_jobs)
    print(f"TILES {len(plan)} cuts ({why}); {workers} at a time, {per_tile} frame(s) each", flush=True)
    started = time.monotonic()

    def one(cut: dict) -> dict:
        tile_args = argparse.Namespace(**vars(args))
        tile_args.frame_jobs = per_tile
        tile_args.scratch = str(Path(scratch) / f"tile-{cut['name']}")
        tile_args.keep_scratch = bool(getattr(args, "keep_scratch", False))
        if histories is not None:
            tile_args.history = [str(item) for item in histories]
        # The full pass already paired and checked the diagnostics file; the cut converts the paired
        # history, whose diag file the converter finds beside it.
        tile_args.diagnostics = None
        cut_started = time.monotonic()
        record = {"name": cut["name"], "window": cut["window"], "i": cut["i"], "j": cut["j"]}
        try:
            rc, manifest = _run_render(tile_args, tile={**cut, "view": view})
        except (MpasPortError, OSError, ValueError) as error:
            print(f"TILE {cut['name']} refused: {error}", flush=True)
            record.update(rc=2, refusal=str(error), counts={"rendered": 0, "skipped": 0, "failed": 1})
        else:
            record.update(rc=rc, counts=manifest.get("counts"),
                          window_description=(manifest.get("convert") or {}).get("window_description"),
                          products_drawn=manifest.get("products_drawn"),
                          products_unavailable=manifest.get("products_unavailable"),
                          frames=manifest.get("frames"), invocations=manifest.get("invocations"))
        record["elapsed_seconds"] = round(time.monotonic() - cut_started, 3)
        return record

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="tile") as pool:
        cuts = list(pool.map(one, plan))
    return {"why": why, "workers": workers, "frame_jobs_each": per_tile, "cuts": cuts,
            "elapsed_seconds": round(time.monotonic() - started, 3)}


def _run_render(args, tile: dict | None = None) -> tuple[int, dict]:
    """One render pass: ``(exit code, manifest)``.

    ``tile`` (internal) is one cut of a tiled render, ``{name, window, view, ...}``: the pass
    converts the cut's own window, files its pictures under the tile folder, and leaves the
    manifest file, the column store and the further cuts to the full pass that called it."""

    started = time.monotonic()
    out_root = Path(args.out)
    scratch = Path(args.scratch) if args.scratch else default_scratch(out_root)
    refuse_scratch_inside_out(scratch, out_root)

    # Argument coherence FIRST, before any binary is looked for: an
    # incoherent command is incoherent whether or not an engine is
    # installed, and making the user install one to be told so wastes
    # their time and hides the real complaint behind a second one.
    compose = getattr(args, "compose", None)
    compose_base_only = bool(getattr(args, "compose_base_only", False))
    if compose is not None:
        if args.history or args.mesh:
            raise RenderDoorError(
                "--compose owns the mesh and history list; passing --mesh "
                "or --history as well would leave two source lists that "
                "can disagree about which run is the base. Put every run "
                "in the compose file."
            )
    else:
        if compose_base_only:
            raise RenderDoorError(
                "--compose-base-only describes what to leave OUT of a "
                "composite, and there is no composite without --compose. "
                "A single-source render is already the base alone."
            )
        if args.window == "composite":
            raise RenderDoorError(
                "--window composite is sized to the fine regions named in "
                "--compose, and no source list was given, so there is no "
                "fine region to frame. Use --window mesh for one run's own "
                "refined core, or add --compose."
            )
        if not args.mesh or not args.history:
            raise RenderDoorError(
                "--history and --mesh are required unless --compose names "
                "them instead."
            )

    remap = resolve_remap(getattr(args, "remap", REMAP_AUTO), compose is not None)
    frame_jobs = checked_frame_jobs(getattr(args, "frame_jobs", 1) or 1)
    tiles_mode = str(getattr(args, "tiles", None) or "none")
    if tiles_mode not in TILE_MODES:
        raise RenderDoorError(f"--tiles {tiles_mode!r} is not one of {', '.join(TILE_MODES)}")
    if compose is not None and tiles_mode == "regional":
        raise RenderDoorError(
            "--tiles regional cuts tiles out of one run's converted window, and a --compose frame "
            "takes its window from the source list, which the converter frames itself, so there is "
            "no window grid the cuts could be placed on; draw the composite with --tiles none")

    convert_exe = resolve_convert_exe(args.convert_exe)
    renderer_exe = resolve_renderer_exe(args.renderer_exe)
    source_label = getattr(args, "source_label", None)
    # These are hex forecast fields. The renderer's generic local-file
    # default names the WOOF engine version, not this forecast's source.
    # Default the credit only where the renderer takes --source-label: a
    # renderer that predates it keeps its own credit instead of refusing
    # every default render. A label the caller names still refuses there.
    if source_label is None and engine_accepts(renderer_exe, "--source-label"):
        source_label = "Hex forecast"
    if source_label is not None:
        if not source_label or len(source_label) > 160 or any(
                ord(c) < 32 or ord(c) > 126 for c in source_label):
            raise RenderDoorError("--source-label needs 1-160 printable ASCII characters")
        if not engine_accepts(renderer_exe, "--source-label"):
            raise RenderDoorError(f"{renderer_exe} does not support --source-label")
    grid_label_asked = getattr(args, "grid_label", None)
    if grid_label_asked and compose is not None:
        raise RenderDoorError(
            "--grid-label names one run's picture, and a --compose frame switches between "
            "several meshes point by point, so the converter writes it no grid label; draw "
            "the composite without --grid-label")
    convert_grid_label = drawn_grid_label(convert_exe, grid_label_asked)
    convert_jobs = import_jobs = 1
    if frame_jobs > 1:
        if engine_accepts(convert_exe, "--jobs"):
            convert_jobs = frame_jobs
        else:
            print(f"FRAME-JOBS {convert_exe} predates --jobs; it converts one "
                  f"frame at a time", flush=True)
        if engine_accepts(renderer_exe, "--import-jobs"):
            import_jobs = frame_jobs
        else:
            # rw_wrfbatch takes no --import-jobs: since woof 18506efc96 (2.9.4; this package needs 2.9.10) it
            # imports a series' frames on lanes it sizes itself from the host's threads and memory. THE BREAKAGE
            # THIS PREVENTS (an Arizona zoom globe, 8 x RTX PRO 6000, woof hex 0.4.10, 2026-10-30): the door
            # printed "predates --import-jobs; it imports one frame at a time" on every view, which named a stale
            # renderer as the cause of a slow products tail when the renderer was current and parallel.
            print(f"FRAME-JOBS {renderer_exe} takes no --import-jobs; it sets its own import lanes from the "
                  f"host's threads and memory", flush=True)
    # Before the converter runs: a render with no map geometry is refused
    # here, by name, not discovered after minutes of resampling.
    basemap_dir = resolve_basemap_dir(
        renderer_exe, getattr(args, "basemap_dir", None))
    renderer_env = renderer_environment(basemap_dir)
    print(f"BASEMAP {basemap_dir}", flush=True)
    if compose is not None:
        compose = _require_file(
            Path(compose), "--compose",
            "Give a JSON file with a 'base' entry and an 'overlays' list, "
            "each carrying 'label', 'mesh' and 'history'.")
        mesh, histories = None, None
    else:
        mesh = _require_file(
            Path(args.mesh), "--mesh",
            "The converter cannot place Voronoi cells on the render grid "
            "without the mesh's cell coordinates; point --mesh at the run's "
            "grid/static netCDF (the file the run was initialized on).")
        histories = [
            _require_file(
                Path(item), "--history",
                "Give the run's history.*.nc output files; diag files do "
                "not carry the 3-D state the converter reads.")
            for item in args.history
        ]
    init = None
    if getattr(args, "init", None):
        if compose is not None:
            raise RenderDoorError(
                "--init belongs to a single-source render; a compose file "
                "names each run's own files.")
        init = _require_file(
            Path(args.init), "--init",
            "Point --init at the init netCDF the run started from; the "
            "converter reads zgrid from it and from nothing else.")

    diagnostics = None
    if getattr(args, "diagnostics", None):
        diagnostics = _require_file(Path(args.diagnostics), "--diagnostics",
                                    "Give the native diagnostics file at this history valid time.")

    label = getattr(args, "label", None) or default_label(mesh)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", label):
        raise RenderDoorError(
            f"--label {label!r} must be letters, digits, '.', '_' or '-' so it "
            "can lead a filename on every platform the tree is copied to")
    window = str(getattr(args, "window", "mesh") or "mesh")
    convert_window = tile["window"] if tile else converter_window(window)
    view = checked_view(tile["view"] if tile else (getattr(args, "view", None) or default_view(window)))
    # The plot names the model from --model-label alone: --label is the
    # filenames' leading token (hex_x1.2621442), not a model name.
    model_label = getattr(args, "model_label", None) or DEFAULT_MODEL_LABEL

    try:
        width_text, height_text = str(args.size).lower().split("x", 1)
        width, height = int(width_text), int(height_text)
    except ValueError:
        raise RenderDoorError(
            f"--size must be WIDTHxHEIGHT (e.g. 1200x900), got {args.size!r}"
        ) from None
    if tile is None and window == "global" and compose is None:
        # The whole sphere at no more grid columns than its picture has pixels (global_picture_window).
        picture = global_picture_window(mesh, width)
        if picture is not None:
            convert_window = picture[0]
            print(f"GLOBAL picture grid {convert_window}: {picture[1]}", flush=True)

    scratch.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)
    invocations: list[list[str]] = []
    # The start data's credit after this map's own source line, fitted to this canvas (SOURCE_CREDIT_ENV).
    credit_theme = source_credit_theme(scratch, source_label, width, height)
    if credit_theme is not None:
        renderer_env = {**renderer_env, "RUSTWX_THEME": credit_theme}

    frame_cache = getattr(args, "frame_cache", None) if tile is None else None
    report, convert_invocation = convert_frames_cached(
        Path(frame_cache) / view if frame_cache else None,
        convert_exe, histories=histories, mesh=mesh, scratch=scratch,
        window=convert_window, field_set=args.field_set,
        simulation_start=args.simulation_start, nc_format=args.nc_format,
        compose=compose, compose_base_only=compose_base_only, init=init, diagnostics=diagnostics,
        model_label=model_label, remap=remap, jobs=convert_jobs,
        explicit_jobs=tile is not None and engine_accepts(convert_exe, "--jobs"),
        grid_label=convert_grid_label)
    grid_label = report.get("grid_label")
    # The delivered file names keep the MESH's token (hex128-8km) whatever the
    # headline says: the WOOF worker and every tree already delivered file
    # pictures by it.  A converter given --grid-label reports the mesh's own
    # label as mesh_grid_label; without it the two are the same label.
    file_grid_label = report["mesh_grid_label"] if "mesh_grid_label" in report else grid_label
    invocations.append(convert_invocation)
    # Regional tiles: decided on the window the converter actually laid down (a derived window's
    # size is known only now), so a render files whole or tiled from its first picture on.
    tile_plan: list[dict] = []
    tile_why = ""
    if tile is None:
        tile_plan, tile_why = plan_tiles(report.get("window_spec"),
                                         "none" if compose is not None else tiles_mode, width, height)
        if tiles_mode != "none":
            print(f"TILES {','.join(cut['name'] for cut in tile_plan) or 'none'} ({tile_why})", flush=True)
    tile_folder = tile["name"] if tile else (TILE_FULL if tile_plan else None)
    file_view = view if tile is None else f"{view}-{tile['name']}"

    frames_meta = report.get("frames", [])
    if compose is not None:
        # A composite converts only the hours EVERY source carries, so its
        # frame count is an intersection and cannot be checked against one
        # history list.  What still has to hold is that it produced
        # something: an empty series would otherwise reach the renderer as
        # a clean run that delivered no images.
        if not frames_meta:
            raise RenderDoorError(
                f"rw_mpas_convert composed no frames from {compose}; no "
                f"valid time is carried by every source in the list, so "
                f"there is nothing to render."
            )
        histories = [Path(meta["output"]) for meta in frames_meta]
    elif len(frames_meta) != len(histories):
        raise RenderDoorError(
            f"rw_mpas_convert reported {len(frames_meta)} frames for "
            f"{len(histories)} history files; the receipt does not cover "
            f"the request, so the render does not proceed."
        )

    # Coverage gate: one real catalog per distinct absent-field signature.
    # Frames from one converter run differ only when the histories
    # themselves carry different fields.
    catalogs: dict[tuple[str, ...], dict] = {}
    catalog_summary = ""
    # One store for the whole series (--series): the frames are imported
    # once, side by side, which is the only arrangement in which the
    # renderer can close an accumulation window (qpf_6h is the stored total
    # at f012 minus the one at f006) and which costs one renderer start per
    # view, not two per frame.  THE BREAKAGE THIS PREVENTS, MEASURED
    # (2026-10-12): with a store per frame every windowed product listed as
    # "windowed accumulations need more than one stored whole-hour frame",
    # so a 48 h run could draw precipitation since init and nothing else.
    leads = None
    if getattr(args, "series", False) and compose is None:
        leads = series_leads(frames_meta)
        if leads is None:
            print("SERIES off: the frames do not carry distinct leads (or "
                  "there is one frame); one store per frame, so no windowed "
                  "product", flush=True)
    elif compose is None and not getattr(args, "standard_set", False):
        # A windowed product NAMED in --products turns the series store on
        # by itself when the frames can close a window (two or more frames,
        # distinct leads); the renderer catalog's own kind column says which
        # products are windowed (49 today, qpf_1h among them).
        # THE BREAKAGE THIS PREVENTS (RRFS hrrrv5 conus3km, 2026-10-28):
        # `woof hex render --history <three hours> --products qpf_1h`
        # refused every run with "excluded -- windowed accumulations need
        # more than one stored whole-hour frame (fields absent from the
        # source history: ... RAINC ...)", because without --series each
        # frame had its own store; the absent-field hint read as the cause,
        # the conus3km gallery's 1 h QPF was drawn by hand without a
        # basemap, and the box kit's QPF pass exited 2 on every run.
        named = [token.strip() for token in (args.products or "").split(",")
                 if token.strip() and token.strip() not in _GROUP_KEYWORDS]
        possible = series_leads(frames_meta) if named else None
        if possible is not None:
            first = frames_meta[0]
            signature = tuple(first.get("absent_wrf_fields", []))
            rows, catalog_summary, catalog_invocation = list_catalog(
                renderer_exe, Path(first["output"]), scratch=scratch,
                heavy=args.heavy, store=Path(scratch) / "store-000",
                products=args.products, env=renderer_env)
            catalogs[signature] = rows
            invocations.append(catalog_invocation)
            windowed = [token for token in named
                        if rows.get(token, ("",))[0] == "windowed"]
            if windowed:
                leads = possible
                print(f"SERIES on: {', '.join(windowed)} named; a window is "
                      f"the difference of stored frames, so the "
                      f"{len(frames_meta)} frames share one store", flush=True)
    series_outputs = [Path(meta["output"]) for meta in frames_meta]
    # Every map carries a labelled latitude/longitude graticule (see
    # graticule_overlays).  One window per render, so the first frame's
    # coordinates hold for the series.
    overlays_path, graticule = None, None
    if frames_meta and engine_accepts(renderer_exe, "--overlays"):
        overlays_path, graticule = write_graticule_overlays(
            Path(frames_meta[0]["output"]), scratch)
    print(f"GRATICULE {graticule if graticule else 'none'}", flush=True)
    if leads is not None:
        rows, catalog_summary, catalog_invocation = list_catalog(
            renderer_exe, series_outputs, scratch=scratch, heavy=args.heavy,
            store=Path(scratch) / f"store-{SERIES_TAG}",
            products=args.products, env=renderer_env, import_jobs=import_jobs)
        catalogs[(SERIES_TAG,)] = rows
        invocations.append(catalog_invocation)
        if not getattr(args, "standard_set", False):
            absent = sorted({name for meta in frames_meta
                             for name in meta.get("absent_wrf_fields", [])})
            refuse_uncovered_products(args.products, rows, histories[0], absent)
    for index, (meta, history) in enumerate(zip(frames_meta, histories)):
        if leads is not None:
            break
        signature = tuple(meta.get("absent_wrf_fields", []))
        if signature not in catalogs:
            rows, catalog_summary, catalog_invocation = list_catalog(
                renderer_exe, Path(meta["output"]), scratch=scratch,
                heavy=args.heavy, store=Path(scratch) / f"store-{index:03d}",
                products=args.products, env=renderer_env)
            catalogs[signature] = rows
            invocations.append(catalog_invocation)
        if not getattr(args, "standard_set", False):
            refuse_uncovered_products(
                args.products, catalogs[signature], history,
                list(meta.get("absent_wrf_fields", [])))

    unavailable: dict[str, str] = {}
    products_effective = args.products
    if getattr(args, "standard_set", False):
        # A standard set is a GROUP the door owns, like the renderer's own
        # 'all': it expands to the members this history can draw, and every
        # member it cannot is named with the catalog's own reason in stdout
        # and the manifest -- never silently dropped.
        keep: list[str] = []
        for token in [t.strip() for t in args.products.split(",") if t.strip()]:
            statuses = [rows.get(token) for rows in catalogs.values()]
            if all(row is not None and row[1] == "renderable" for row in statuses):
                keep.append(token)
            else:
                row = next((r for r in statuses if r is not None and r[1] != "renderable"), None)
                unavailable[token] = (f"{row[1]}: {row[2]}" if row is not None
                                      else "not in the renderer catalog")
                print(f"UNAVAILABLE {token} {unavailable[token]}", flush=True)
        if not keep:
            raise RenderDoorError(
                f"none of the standard products can be drawn from this "
                f"history: {'; '.join(f'{k} ({v})' for k, v in unavailable.items())}")
        products_effective = ",".join(keep)

    results: list[FrameResult] = []
    total_rendered = total_skipped = total_failed = 0
    series_parts: dict[int, tuple[list, list, list]] = {}
    if leads is not None:
        # One renderer start draws every frame; each picture, skip and
        # failure is then handed to the frame it belongs to: a picture by
        # the lead in its own name, a skip or failure by the frame file the
        # renderer names in it (the last frame when it names none).
        rendered, skipped, failed, invocation = render_frame(
            renderer_exe, series_outputs, scratch=scratch, index=0,
            products=products_effective, frames=args.frames, width=width,
            height=height, heavy=args.heavy, env=renderer_env, tag=SERIES_TAG,
            import_jobs=import_jobs, source_label=source_label,
            overlays=overlays_path)
        invocations.append(invocation)
        by_lead = {lead: index for index, lead in enumerate(leads)}
        names = sorted(((str(path), index) for index, path in enumerate(series_outputs)),
                       key=lambda item: -len(item[0]))
        last = len(leads) - 1
        series_parts = {index: ([], [], []) for index in range(len(leads))}

        def owner(text: str) -> int:
            for name, index in names:
                if name in text:
                    return index
            return last

        for slug, png in rendered:
            index = by_lead.get(engine_lead_seconds(png.name))
            if index is None:
                # A picture no frame owns is a failure, said by name: filed
                # under a guessed frame it would carry a wrong valid day.
                series_parts[last][2].append(
                    f"{slug} {png.name}: its lead matches no frame of the series")
                continue
            series_parts[index][0].append((slug, png))
        for item in skipped:
            series_parts[owner(item["reason"])][1].append(item)
        for item in failed:
            series_parts[owner(item)][2].append(item)
    for index, (meta, history) in enumerate(zip(frames_meta, histories)):
        frame = FrameResult(
            history=str(history),
            wrfout=str(meta["output"]),
            valid_time=str(meta.get("valid_time", "")),
            valid_day=valid_day(meta.get("valid_time")),
            output_sha256=str(meta.get("output_sha256", "")),
            absent_wrf_fields=list(meta.get("absent_wrf_fields", [])),
            seam=list(meta.get("seam", [])),
        )
        if leads is not None:
            rendered, skipped, failed = series_parts[index]
            raw_out = Path(scratch) / f"raw-{SERIES_TAG}"
        else:
            rendered, skipped, failed, invocation = render_frame(
                renderer_exe, Path(meta["output"]), scratch=scratch, index=index,
                products=products_effective, frames=args.frames, width=width,
                height=height, heavy=args.heavy, env=renderer_env,
                source_label=source_label, overlays=overlays_path)
            invocations.append(invocation)
            raw_out = Path(scratch) / f"raw-{index:03d}"
        frame.skipped = skipped
        frame.failed = failed
        moved: dict[str, str] = {}
        lead = None
        for slug, png in rendered:
            domain, product = split_engine_name(png.name, slug)
            lead = lead_token(meta.get("lead_seconds"), png.name)
            name = product_file_name((getattr(args, "label", None) or grid_token(file_grid_label, domain)), file_view,
                                     product, lead)
            destination = place_path(out_root, view, product,
                                     frame.valid_day, name, tile=tile_folder)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(png), str(destination))
            placed = str(destination.relative_to(out_root)).replace(os.sep, "/")
            in_view = destination.relative_to(out_root / view).as_posix()
            try:
                moved[png.relative_to(raw_out).as_posix()] = in_view
            except ValueError:
                moved[png.name] = in_view
            frame.rendered.append(slug)
            frame.placed.append(placed)
        carry_georef(raw_out, moved, out_root / view)
        results.append(frame)
        total_rendered += len(frame.rendered)
        total_skipped += len(skipped)
        total_failed += len(failed)
        print(f"FRAME {frame.valid_time} rendered={len(frame.rendered)} "
              f"skipped={len(skipped)} failed={len(failed)}", flush=True)
        for item in skipped:
            print(f"  SKIPPED {item['product']} {item['reason']}",
                  flush=True)
        for item in failed:
            print(f"  FAILED {item}", flush=True)

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "engine": {
            "convert_exe": str(convert_exe),
            "convert_exe_sha256": sha256_file(convert_exe),
            "renderer_exe": str(renderer_exe),
            "renderer_exe_sha256": sha256_file(renderer_exe),
            "basemap_dir": str(basemap_dir),
        },
        "view": view,
        "window": window,
        "model_label": model_label,
        "model_label_drawn": fitted_model_label(model_label),
        "source_label": source_label,
        "grid_label": grid_label,
        "mesh_grid_label": file_grid_label,
        "graticule": graticule,
        "convert": {
            "schema": report.get("schema"),
            "engine": report.get("engine"),
            "window": report.get("window"),
            "window_spec": report.get("window_spec"),
            "window_description": report.get("window_description"),
            "mesh_spacing_km": report.get("mesh_spacing_km"),
            "mesh": str(mesh) if mesh is not None else None,
            "mesh_sha256": report.get("mesh_sha256"),
            "weights_sha256": report.get("weights_sha256"),
            "resample_method": report.get("resample_method"),
            "remap": remap,
            "product_remap": report.get("product_remap"),
            "precipitation_budget": [frame.get("precipitation_budget")
                                     for frame in frames_meta],
            "field_set": report.get("field_set"),
            # Present only on a composed render: which runs went in, how
            # many grid points each won, and how wide the seam is.  A
            # composite whose manifest did not say which runs it came from
            # would be indistinguishable from a single-run render.
            "compose_spec": str(compose) if compose is not None else None,
            "compose_base_only": compose_base_only,
            "composite": report.get("composite"),
            "diagnostics": report.get("diagnostics"),
        },
        "tile": tile_folder,
        "catalog_summary": catalog_summary,
        "label": label,
        "products_requested": args.products,
        "products_drawn": products_effective,
        "series_store": leads is not None,
        "frame_jobs": {"requested": frame_jobs, "convert": convert_jobs,
                       "import": import_jobs if leads is not None else 1},
        "products_unavailable": unavailable,
        "frames_requested": args.frames,
        "size": f"{width}x{height}",
        "counts": {"rendered": total_rendered, "skipped": total_skipped,
                   "failed": total_failed},
        "frames": [vars(frame) for frame in results],
        "invocations": invocations,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    # The any-point column store, every output hour (column_store.py), from the converted frames before the
    # scratch they sit in is cleared. A composite blends several runs, so it is no run's own column: none.
    if tile is not None:
        # A cut of a tiled render: its pictures are filed; the full pass keeps the receipt, the
        # column store (the full window's) and the scratch it sits in.
        ok = total_failed == 0 and total_rendered > 0
        if ok and not args.keep_scratch:
            shutil.rmtree(scratch, ignore_errors=True)
        print(f"TILE {tile['name']} rendered={total_rendered} skipped={total_skipped} "
              f"failed={total_failed}", flush=True)
        return (0 if ok else 1), manifest
    if tile_plan:
        tiles = _run_tiles(args, tile_plan, view=view, scratch=scratch, histories=(
            [Path((report.get("diagnostics") or {}).get("history"))] if report.get("diagnostics")
            else histories if compose is None else None), frame_jobs=frame_jobs, why=tile_why)
        manifest["tiles"] = tiles
        for cut in tiles["cuts"]:
            for key in ("rendered", "skipped", "failed"):
                manifest["counts"][key] += int((cut.get("counts") or {}).get(key) or 0)
        total_rendered = manifest["counts"]["rendered"]
        total_skipped = manifest["counts"]["skipped"]
        total_failed = manifest["counts"]["failed"]
        manifest["elapsed_seconds"] = round(time.monotonic() - started, 3)
    if compose is None:
        from .column_store import write as write_columns
        # The converted frames this pass rendered, wherever they are (its scratch, or the run's frame cache).
        manifest["columns"] = write_columns(sorted({Path(meta["output"]) for meta in frames_meta},
                                                   key=lambda path: path.name),
                                            out_root / view / "columns", field_set=args.field_set)
    # A second run into the same delivered tree is legitimate (one full
    # catalog pass plus a restricted pass is the proof shape); its receipt
    # must not overwrite the first run's.  Each view keeps its receipts in
    # its own folder, so views of one run rendered side by side never race
    # for one file.
    receipt_dir = out_root / view
    receipt_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = write_new_json(receipt_dir, "render-manifest", manifest)
    print(f"MANIFEST {manifest_path}", flush=True)

    ok = total_failed == 0 and total_rendered > 0
    if ok and not args.keep_scratch:
        shutil.rmtree(scratch, ignore_errors=True)
        print(f"SCRATCH cleared {scratch}", flush=True)
    else:
        print(f"SCRATCH kept {scratch}", flush=True)
    print(f"DOOR rendered={total_rendered} skipped={total_skipped} "
          f"failed={total_failed} out={out_root}", flush=True)
    if total_rendered == 0:
        print("DOOR nothing rendered: every requested product was skipped "
              "or failed; see the manifest's frames[] for each reason",
              flush=True)
        return 1, manifest
    return (0 if total_failed == 0 else 1), manifest


def add_render_arguments(parser) -> None:
    """Wire the ``render`` subcommand's arguments onto ``parser``."""

    parser.add_argument(
        "--history", nargs="+", default=None, metavar="FILE",
        help="MPAS history netCDF file(s), one frame each. Required unless "
             "--compose names them instead")
    parser.add_argument(
        "--mesh", default=None, metavar="FILE",
        help="the run's grid/static netCDF with cell coordinates. Required "
             "unless --compose names it instead")
    parser.add_argument(
        "--init", default=None, metavar="FILE",
        help="the run's init netCDF (the file the forecast started from). "
             "Optional: it is the converter's only source of zgrid, so the "
             "pressure-level products (500mb_height_winds, "
             "850mb_temperature_height_winds) render only when it is given")
    parser.add_argument(
        "--diagnostics", default=None, metavar="FILE",
        help="the diag stream file (diag.T.nc) of exactly one --history's valid time, when it is "
             "not already beside the history under that name: the door pairs the two where the "
             "converter reads a history's diag stream, and refuses a frame the converter completed "
             "from any other file (its SHA-256 is in the manifest)")
    parser.add_argument(
        "--tiles", default="none", choices=TILE_MODES,
        help="regional tiles: the whole window plus six cuts of it (nw, nc, ne, sw, sc, se: three "
             "columns by two rows, overlapping), each an exact sub-grid of the window, filed as "
             "<view>/<product>/<tile>/<valid-day>/ with 'full' for the whole window. 'regional' "
             "always draws them; 'auto' draws them when a Lambert window has more grid points "
             "across than the picture has pixels (a 3 km CONUS domain), and the window whole "
             "otherwise; 'none' (default) draws the window whole, unfiled by tile")
    parser.add_argument(
        "--compose", default=None, metavar="FILE",
        help="compose SEVERAL runs of the same hours into one frame: the "
             "coarse run everywhere, each fine run inside the ground that "
             "fine run actually resolves. The file lists a base and any "
             "number of overlays, each with its own mesh and history "
             "files. Pair it with --window composite, which frames all the "
             "fine regions at once. THE BREAKAGE THIS PREVENTS, MEASURED "
             "(2026-08-26): a cascade produces a coarse global forecast and "
             "one fine forecast per placed grid, and the only way to look "
             "at both was two separate images at two different scales. "
             "Compositing them as PICTURES would need every pixel's "
             "geographic transform and would still leave two projections "
             "and two colour scales to reconcile; compositing the DATA is "
             "one resample onto one grid and one render, so no seam can "
             "come from the drawing")
    parser.add_argument(
        "--compose-base-only", action="store_true",
        help="with --compose, frame the same window from the same sources "
             "but convert the BASE alone. The coarse counterpart to a "
             "composite, over identical ground, so a comparison between "
             "them cannot be confounded by a different map")
    parser.add_argument(
        "--out", required=True, metavar="DIR",
        help="delivered PNG tree root (<domain>/<product>/<valid-day>/)")
    parser.add_argument(
        "--products", default="all",
        help="renderer product selection: all|direct|derived|heavy|windowed "
             "or comma-separated slugs (default: all)")
    parser.add_argument(
        "--frames", default="all",
        help="renderer frame selection within each store (default: all)")
    parser.add_argument(
        "--window", default="mesh", type=window_argument,
        help="render window. 'mesh' is DERIVED from the mesh you "
             "pass: the cells within 2x the minimum spacing are its refined "
             "region, and the window is centred on them at their median "
             "spacing. 'global' is the whole sphere at the MESH'S OWN "
             "spacing (a 15 km mesh draws at 15 km, a 60 km mesh at 60 km). "
             "'region:CLAT,CLON,WIDTH_KM,HEIGHT_KM[,TRUELAT1,TRUELAT2]' is a "
             "Lambert zoom of that ground whose dx is the spacing of the mesh "
             "cells inside it; 'lambert:...' is an explicit WRF-style grid. "
             "'focus' is a fixed 22 km Lambert box over the CONUS and "
             "'global-0.25' the old fixed 0.25 degree overview; both are "
             "frozen geometry whose weights digests are recorded in evidence. "
             "THE BREAKAGE 'mesh' PREVENTS, MEASURED (2026-08-26): a mesh "
             "this project PLACED had its refined 4.5 km core in the "
             "tropical Atlantic, and the only windows that existed were a "
             "CONUS box 5,000 km away -- which samples nothing but 75 km "
             "background cells -- and a 0.25 degree (about 28 km) global "
             "grid, which cannot resolve a 4.5 km core at all. There was no "
             "way to render the run's own fine region, and every future "
             "placement would have needed another hardcoded box. "
             "'composite' needs --compose and frames every fine region in "
             "the source list at once (default: mesh)")
    parser.add_argument(
        "--view", default=None, metavar="NAME",
        help="what the window shows, as the folder and filename token "
             "(north-america, polar, ...); lower case, digits and '-'. "
             "Default: 'global' for the global windows, 'region' for a "
             "region:/lambert: spec, else the window name")
    parser.add_argument(
        "--model-label", default=DEFAULT_MODEL_LABEL, metavar="NAME",
        help="the model the plots name (metadata row and headline grid "
             "label, e.g. 'hex 15 km'), at most 32 characters, the metadata "
             "row's limit; a longer name is shortened to its whole words "
             "that fit, with '...', and the door says so; default: hex")
    parser.add_argument(
        "--source-label", default=None, metavar="TEXT",
        help="source credit printed on each picture (1-160 printable ASCII "
             "characters); default: Hex forecast where the renderer supports it")
    parser.add_argument(
        "--grid-label", default=None, metavar="TEXT",
        help="the picture's headline grid name in place of the mesh's (e.g. "
             "'Core zoom {spacing}'; '{spacing}' becomes the spacing of the mesh "
             "cells the picture samples, finest first: 'Core zoom 8-38 km'), at "
             "most 32 characters. Needs a converter that takes rw_mpas_convert "
             "--grid-label; an older one keeps the mesh's headline and the door "
             "says so. File names keep the mesh's token. Not with --compose")
    parser.add_argument(
        "--field-set", default="full", choices=("full", "surface"),
        help="converter field set (default: full)")
    parser.add_argument(
        "--remap", default=REMAP_AUTO, choices=REMAP_CHOICES,
        help="how mesh fields reach the render grid. 'product' (the "
             "default for a single run): linear barycentric on the mesh's "
             "own Delaunay triangles for state, first-order area-conservative "
             "for accumulated precipitation (the frame records the window "
             "integral against the mesh's), nearest cell for categories and "
             "soil. 'nearest': every field the value of the nearest cell. "
             "'auto' is product for a single run and nearest for --compose, "
             "which switches source point by point and has no single mesh to "
             "interpolate across")
    parser.add_argument(
        "--series", action="store_true",
        help="import every frame into ONE store and draw the series with one "
             "renderer start. Needed for the windowed products (qpf_1h, "
             "qpf_6h, qpf_24h, 2m_temp_0_24h_max, ...): an accumulation "
             "window is the difference of two stored frames, and a store "
             "per frame holds one. Off by default so an existing render "
             "keeps its per-frame stores; the product door turns it on. "
             "Frames without distinct leads fall back to a store per frame")
    parser.add_argument(
        "--frame-jobs", type=int, default=1, metavar="N",
        help="frames converted at a time and, with --series, imported at a "
             "time (rw_mpas_convert --jobs, rw_wrfbatch --import-jobs). The "
             "pictures are the same for every N; each frame in flight holds "
             "its fields in memory, so the default is 1 and the product "
             "door sizes N from the host's free memory. An engine built "
             "before these flags runs one frame at a time and the door says "
             "so")
    parser.add_argument(
        "--label", default=None, metavar="MODEL_MESH",
        help="leading token of every delivered filename, which reads "
             "<label>_<YYYYMMDD>_<HH>z_f<lead>_<product>.png (default: hex_ "
             "plus the mesh file's name without its role tokens, e.g. "
             "hex_x1.2621442); the plots name the model from --model-label")
    parser.add_argument(
        "--simulation-start", default=None, metavar="YYYY-MM-DD_HH:MM:SS",
        help="run start, for true lead hours in filenames")
    parser.add_argument(
        "--size", default="1200x900",
        help="output pixels as WIDTHxHEIGHT (default: 1200x900)")
    parser.add_argument(
        "--heavy", action="store_true",
        help="include the heavy product family (off by default)")
    parser.add_argument(
        "--nc-format", default="cdf2", choices=("cdf2", "cdf5"),
        help="converted frame netCDF variant (default: cdf2)")
    parser.add_argument(
        "--scratch", default=None, metavar="DIR",
        help="conversion/render scratch dir OUTSIDE --out "
             "(default: sibling <out>.render-scratch)")
    parser.add_argument(
        "--keep-scratch", action="store_true",
        help="keep the scratch dir after a clean run")
    parser.add_argument(
        "--basemap-dir", default=None, metavar="DIR",
        help=f"the renderer's map geometry (natural_earth_10m/, "
             f"natural_earth_110m/, us_counties_5m/). Default: "
             f"${BASEMAP_ENV}, then assets/basemap above the renderer, then "
             f"woof's own assets. A render with none is refused: it would "
             f"draw every map without coastlines, borders or state lines")
    parser.add_argument(
        "--convert-exe", default=None, metavar="FILE",
        help=f"rw_mpas_convert binary (else ${CONVERT_ENV}, then PATH)")
    parser.add_argument(
        "--renderer-exe", default=None, metavar="FILE",
        help=f"rw_wrfbatch binary (else ${RENDERER_ENV}, "
             f"${RENDERER_ENV_GPUWM}, then PATH)")
