"""A mesh as a TABLE ROW: what can be asked of the graded generator, written down once.

WHY THIS EXISTS.  ``rw_mpas_mesh`` reads a resolution spec: a background spacing and a list of
regions, each a shape, a spacing and a ramp width, with the finest region winning where they overlap.
That spec can already say nested zones, several separate zones, ellipses and drawn outlines.  The
doors in front of it could not: ``mesh-plan --point`` wrote one ladder of caps about one point, refused
any fine spacing that was not the background halved a whole number of times, and the zoom-globe door
refused a drawn core unless the caller authored the generator's spec by hand.  This module is the one
place a caller's ROW becomes that spec:

* ``fine_dx_m`` is ANY spacing.  The generator refines by midpoint insertion, so every plateau on one
  mesh is the finest spacing times a power of two.  The finest zone is what was bought and is kept
  exactly; the background (and any coarser zone) is moved onto its ladder, to the NEAREST rung by
  default (``"fit": "finer"`` or ``"coarser"`` choose a side, ``"exact"`` refuses instead), and the
  plan says what was asked and what is built, with the columns of each choice;
* ``zones`` is a list: each a shape (cap, ellipse, rounded_box, lat_lon_box, polygon, GeoJSON) and a
  fine spacing.  A zone inside a coarser zone is a NESTED core: its rings climb only to the spacing of
  the zone that holds it;
* ``grading`` says how the rings are laid: the ramp of each ring in cells of its own spacing
  (``ramp_cells``, or ``growth_per_cell``, the steepest growth the ramp asks for), and the flat
  width of each ring (``ring_cells``; none is a ramp with no plateau);
* ``domain`` is the whole globe, or a region: the rim distance, the spacing held out to the rim and
  the boundary rings a limited-area run needs.

Nothing here generates.  :func:`expand` returns the generator's own spec, the cut for a region, and
the notes a plan prints; :func:`plan` prices it (the generator's dry run for the count, the engine's
own step rule and card estimate).  A ROW that names only one cap, the legacy ladder factors and a
background its fine spacing halves into expands to the spec ``mesh_point.ladder_spec`` writes, byte
for byte (tested), so the doors that call this change no existing mesh.

GEOMETRY 2 (``"generator": {"geometry": 2}`` in the row, or a door's ``--geometry 2``).  The
generator's second way of grading has no halving: the fine spacing holds to the zone's own line and
then grows by ONE rate per cell out to the background.  Under it a row is built AS WRITTEN: every
zone at the spacing asked, the background at the spacing asked, one region a zone and no rings.
``fit`` then has nothing to move, and the plan says what geometry 1 would have built instead.  The
row's words do not change; only the grading words that describe flat rings are refused, by name.
Geometry 2 is the default since 0.6.2: a row or door that names no geometry is built under it.  A
row that names ``{"geometry": 1}``, or a door's ``--geometry 1``, expands to the bytes it always did.
THE BREAKAGE THIS PREVENTS: under geometry 1 a 1.96 M-cell 3 km CONUS zoom globe took 10 to 12 minutes
a try on a 384-thread host and was refused twice for one four-edge cell left after 40 surgery
rounds (20 Nov 2026); geometry 2 built the same ask in 96 s with no four-edge cell and no obtuse
triangle (receipt ``conus3km_globe_g2.receipt.json``).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .errors import MpasPortError

ROW_KIND = "gpuwm-hex.mesh-ask/v1"
EARTH_RADIUS_KM = 6371.229
HEXAGON_AREA_FACTOR = math.sqrt(3.0) / 2.0

#: The legacy ladder's two numbers (``mesh_point``): a ramp scale of 18 cells and each ring's edge
#: three ramp scales of the next finer ring outside it.
LEGACY_TRANSITION_FACTOR = 18.0
LEGACY_RING_FACTOR = 3.0
#: How far outside the asked outline a zone's own region is drawn, in ramp scales, when the row asks
#: for the fine spacing to hold on the whole outline.  The generator's ramp is a tanh in 1/spacing
#: centred on a region's edge: 1.5 scales inside it the requested spacing is within 5 % of the fine
#: one (0.5 * (1 + tanh 1.5) = 0.953).  MEASURED on the legacy ladder, which draws the region AT the
#: asked radius: a 3 km core asked to 1,000 km held 3 km to about 925 km (OWN-MESH, 18 Nov 2026).
HOLD_RAMP_SCALES = 1.5
#: Finer than this and the dynamics' own admissions have never run; coarser than the background makes
#: no zone.  A limit with a name, so a refusal can say it.
FINEST_FINE_M = 50.0
#: Backgrounds the generator has built.  Finer than 12 km a globe is over five million columns before
#: any zone; the refusal names the count rather than a rule.
COARSEST_BACKGROUND_KM = 480.0
#: The limited-area boundary zone the regional runs read (MPAS-A's seven relaxation and specified rings).
BOUNDARY_RINGS = 7
#: Vertices a ring's outline is simplified to.  The generator measures the steepest growth of every
#: region on probe points laid along each edge of its outline, out to six ramp scales either side,
#: and one reading may spend four million of them (rw-mpas ``PROBE_BUDGET``).  MEASURED 18 Nov 2026,
#: an ellipse of 900 by 550 km at 3 km on a 96 km globe: 76 vertices on the fine ring spent the
#: budget and the build refused the spec as unmeasured; 64 measured it whole (3.63 million probes).
#: A ring is drawn at least 1.5 ramp scales outside the outline asked, so the kilometre or two this
#: costs in the outline's detail is far inside the ramp.
RING_VERTICES = 64
FITS = ("nearest", "finer", "coarser", "exact")
#: The generator's two geometries.  1 is the halving ladder every premade mesh was built on.  2 grows
#: the spacing by one rate per cell and hands out no mesh with an obtuse triangle.
GEOMETRIES = (1, 2)
#: The geometry a row or door that names none is built under (0.6.2: geometry 2; ``--geometry 1`` for the ladder).
DEFAULT_GEOMETRY = 2
#: Geometry 2's ramp (rw-mpas ``RampOptions::default``): outside a region the spacing is
#: ``h * (1 + s / (1.5 W))`` at distance ``s`` from its line, ``W`` the region's transition width, so
#: it doubles in 1.5 widths and the growth per cell is ``1 / (1.5 x the width in cells)``: 3.7 % for
#: the doors' 18 cells.
RAMP_WIDTHS_PER_DOUBLING = 1.5
#: Geometry 2 holds every pair of neighbouring cell centres at or above this share of the spacing
#: asked there (its finishing stage; rw-mpas ``AcuteOptions::edge_floor``).  MEASURED 18 Nov 2026 on
#: the geometry 2 meshes of OWN-MESH.md: 0.79 to 0.83.
SMALLEST_EDGE_OVER_FINE_RAMP = 0.79
#: Cells of rim spacing held beyond the last boundary ring of a region cut under geometry 2, which
#: holds a region's spacing to its own line (geometry 1 needs two ramp scales there instead).
RIM_MARGIN_CELLS = 3.0
#: A coarser zone that cannot be held beside the finest one is built FINER unless its row says
#: otherwise: the generator's own rule for a region ("a caller always gets at least the resolution
#: asked for", rw-mpas ladder_snap).  The background is moved to the NEAREST rung instead, because
#: its cells are most of a globe's cost and nobody forecasts on them.
ZONE_FIT = "finer"


class MeshAskRefusal(MpasPortError):
    """A row cannot be built, and the message names the limit in plain words."""


# ---------------------------------------------------------------------------
# the ladder
# ---------------------------------------------------------------------------
def fit_to_ladder(asked_km: float, fine_km: float, how: str = "nearest") -> dict[str, Any]:
    """``asked_km`` moved onto the halving ladder that stands on ``fine_km``.

    Returns the spacing built, how many halvings separate it from the fine spacing, and both
    neighbours, so a plan can price each.  ``exact`` refuses a spacing that is not on the ladder.
    """

    if how not in FITS:
        raise MeshAskRefusal(f"fit {how!r} is not one of {', '.join(FITS)}")
    if not asked_km > fine_km:
        raise MeshAskRefusal(
            f"a {asked_km:g} km spacing is not coarser than the {fine_km:g} km fine spacing it has to stand over")
    x = math.log2(asked_km / fine_km)
    if abs(x - round(x)) <= 1e-9:
        k = int(round(x))
        built = fine_km * 2.0 ** k
        return {"asked_km": asked_km, "built_km": built, "halvings": k, "moved": False,
                "finer_km": built, "coarser_km": built}
    below, above = int(math.floor(x)), int(math.ceil(x))
    finer, coarser = fine_km * 2.0 ** below, fine_km * 2.0 ** above
    if how == "exact":
        raise MeshAskRefusal(
            f"{fine_km * 1000.0:g} m does not divide {asked_km:g} km by halving, and the generator refines only "
            f"by halving, so one mesh cannot hold both exactly.  Keep the {fine_km * 1000.0:g} m core and take "
            f"{finer:g} km or {coarser:g} km (fit: finer / coarser / nearest), or keep {asked_km:g} km and take a "
            f"core of {asked_km / 2.0 ** above * 1000.0:g} m or {asked_km / 2.0 ** below * 1000.0:g} m")
    if how == "finer" or (how == "nearest" and below >= 1 and (x - below) <= (above - x)):
        k = below
    else:
        k = above
    if how == "finer" and below < 1:
        k = above
    return {"asked_km": asked_km, "built_km": fine_km * 2.0 ** k, "halvings": k, "moved": True,
            "finer_km": finer if below >= 1 else None, "coarser_km": coarser}


def rungs_between(fine_km: float, ceiling_km: float) -> list[float]:
    """The ring spacings from ``fine_km`` up to, not including, ``ceiling_km``, finest first."""

    out = []
    spacing = fine_km
    while spacing < ceiling_km * (1.0 - 1e-9):
        out.append(spacing)
        spacing *= 2.0
    return out


# ---------------------------------------------------------------------------
# shapes
# ---------------------------------------------------------------------------
def _pair(value: Any, what: str) -> tuple[float, float]:
    try:
        lat, lon = float(value[0]), float(value[1])
    except (TypeError, ValueError, IndexError) as error:
        raise MeshAskRefusal(f"{what} is not [latitude, longitude]: {value!r}") from error
    if not -90.0 <= lat <= 90.0 or not -180.0 <= lon <= 360.0:
        raise MeshAskRefusal(f"{what} {value!r} is not a place: latitude -90..90, longitude -180..360")
    return lat, (lon - 360.0 if lon > 180.0 else lon)


def rounded_box_row(lat: float, lon: float, width_km: float, height_km: float, corner_km: float,
                    azimuth_deg: float = 0.0, *, per_corner: int = 24) -> dict[str, Any]:
    """A box with rounded corners on the azimuthal-equidistant plane about its centre.

    ``height_km`` runs along ``azimuth_deg`` (clockwise from north), ``width_km`` across it.
    """

    import numpy as np

    from . import mesh_shape

    if not (width_km > 0.0 and height_km > 0.0):
        raise MeshAskRefusal(f"a rounded box of {width_km:g} x {height_km:g} km encloses nothing")
    limit = 0.5 * min(width_km, height_km)
    if not 0.0 <= corner_km <= limit:
        raise MeshAskRefusal(
            f"a corner radius of {corner_km:g} km does not fit a {width_km:g} x {height_km:g} km box; "
            f"the largest is half the short side, {limit:g} km (which makes the ends round)")
    hx, hy = 0.5 * width_km - corner_km, 0.5 * height_km - corner_km
    points = []
    for quadrant, (cx, cy) in enumerate(((hx, hy), (-hx, hy), (-hx, -hy), (hx, -hy))):
        start = quadrant * 0.5 * math.pi
        steps = per_corner if corner_km > 0.0 else 1
        for step in range(steps + (1 if corner_km > 0.0 else 0)):
            angle = start + 0.5 * math.pi * step / max(steps, 1)
            points.append((cx + corner_km * math.cos(angle), cy + corner_km * math.sin(angle)))
    # Straight sides are walked too, so a long side follows the plane's line and not a great circle
    # between two far corners.
    dense = []
    for (x0, y0), (x1, y1) in zip(points, points[1:] + points[:1]):
        pieces = max(1, int(math.ceil(math.hypot(x1 - x0, y1 - y0) / 25.0)))
        for piece in range(pieces):
            t = piece / pieces
            dense.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t))
    az = math.radians(azimuth_deg)
    xy = np.asarray(dense, dtype=float)
    east = xy[:, 0] * math.cos(az) + xy[:, 1] * math.sin(az)
    north = -xy[:, 0] * math.sin(az) + xy[:, 1] * math.cos(az)
    plat, plon = mesh_shape._Azimuthal(lat, lon).inverse(east, north)
    return {"kind": "polygon", "vertices_deg": [[float(a), float(b)] for a, b in zip(plat, plon)]}


def shape_row(document: Mapping[str, Any], what: str = "a zone's shape") -> dict[str, Any]:
    """A row's shape as a generator Shape row: a cap, or a polygon with [lat, lon] vertices."""

    from . import mesh_shape

    if not isinstance(document, Mapping):
        raise MeshAskRefusal(f"{what} is not an object with a kind")
    kind = document.get("kind")
    try:
        if kind == "cap":
            lat, lon = _pair(document.get("center_deg"), f"{what}'s center_deg")
            radius = float(document.get("radius_km", 0.0))
            if not radius > 0.0:
                raise MeshAskRefusal(f"{what} is a circle of radius {radius:g} km, which encloses nothing")
            if radius > 0.5 * math.pi * EARTH_RADIUS_KM:
                raise MeshAskRefusal(
                    f"{what} is a circle of radius {radius:g} km, more than a quarter of the way round the "
                    f"earth ({0.5 * math.pi * EARTH_RADIUS_KM:,.0f} km); a zone that size is the background")
            return {"kind": "cap", "center_deg": [lat, lon], "radius_km": radius}
        if kind == "ellipse":
            lat, lon = _pair(document.get("center_deg"), f"{what}'s center_deg")
            a = float(document["semi_major_km"])
            b = float(document["semi_minor_km"])
            azimuth = float(document.get("azimuth_deg", 0.0))
            row = mesh_shape.ellipse_row(lat, lon, max(a, b), min(a, b), azimuth if a >= b else azimuth + 90.0)
        elif kind == "rounded_box":
            lat, lon = _pair(document.get("center_deg"), f"{what}'s center_deg")
            row = rounded_box_row(lat, lon, float(document["width_km"]), float(document["height_km"]),
                                  float(document.get("corner_km", 0.0)), float(document.get("azimuth_deg", 0.0)))
        else:
            return mesh_shape.shape_from_document(document)
    except KeyError as error:
        raise MeshAskRefusal(f"{what} of kind {kind!r} is missing {error.args[0]!r}") from error
    except mesh_shape.ShapeRefusal as error:
        raise MeshAskRefusal(f"{what}: {error}") from error
    try:
        mesh_shape._check_size(row)
    except mesh_shape.ShapeRefusal as error:
        raise MeshAskRefusal(f"{what}: {error}") from error
    return row


def grown(shape: Mapping[str, Any], distance_km: float, tolerance_km: float) -> dict[str, Any]:
    """``shape`` with every point within ``distance_km`` of it added (a cap grows by its radius)."""

    from . import mesh_shape

    if distance_km <= 0.0:
        return dict(shape)
    if shape["kind"] == "cap":
        return {"kind": "cap", "center_deg": list(shape["center_deg"]),
                "radius_km": float(shape["radius_km"]) + float(distance_km)}
    try:
        return mesh_shape.grow(shape, distance_km, tolerance_km=tolerance_km, max_vertices=RING_VERTICES)
    except mesh_shape.ShapeRefusal as error:
        raise MeshAskRefusal(str(error)) from error


def shape_centre(shape: Mapping[str, Any]) -> tuple[float, float]:
    from . import mesh_shape

    if shape["kind"] == "cap":
        return float(shape["center_deg"][0]), float(shape["center_deg"][1])
    return mesh_shape.centroid(shape)


def shape_area_km2(shape: Mapping[str, Any]) -> float:
    from . import mesh_shape

    if shape["kind"] == "cap":
        return 2.0 * math.pi * EARTH_RADIUS_KM ** 2 * (1.0 - math.cos(float(shape["radius_km"]) / EARTH_RADIUS_KM))
    return mesh_shape.area_km2(shape)


def signed_distance_km(shape: Mapping[str, Any], lat: Any, lon: Any) -> Any:
    """Great-circle distance from points to the shape's edge, km, negative inside."""

    import numpy as np

    from . import mesh_shape

    xyz = mesh_shape._xyz(lat, lon)
    if shape["kind"] == "cap":
        centre = mesh_shape._xyz(*shape["center_deg"])
        return EARTH_RADIUS_KM * np.arccos(np.clip(xyz @ centre, -1.0, 1.0)) - float(shape["radius_km"])
    from scipy.spatial import cKDTree

    ring = mesh_shape._ring(shape)
    dense = mesh_shape._densify(mesh_shape._xyz(*ring.T), 2.0 / EARTH_RADIUS_KM)
    chord, _ = cKDTree(dense).query(np.atleast_2d(xyz), workers=-1)
    distance = 2.0 * EARTH_RADIUS_KM * np.arcsin(np.clip(0.5 * chord, 0.0, 1.0))
    lat0, lon0 = mesh_shape.centroid(shape)
    proj = mesh_shape._Azimuthal(lat0, lon0)
    dlat, dlon = mesh_shape._latlon(dense)
    px, py = proj.forward(dlat, dlon)
    qx, qy = proj.forward(np.atleast_1d(lat), np.atleast_1d(lon))
    inside = np.zeros(qx.shape, dtype=bool)
    # Even-odd crossing on the plane about the outline's own centre; points beyond a quarter turn
    # from it are outside any outline this module admits.
    near = (mesh_shape._xyz(np.atleast_1d(lat), np.atleast_1d(lon)) @ mesh_shape._xyz(lat0, lon0)) > 0.0
    if near.any():
        x, y = qx[near], qy[near]
        hit = np.zeros(x.shape, dtype=bool)
        x0, y0 = px, py
        x1, y1 = np.roll(px, -1), np.roll(py, -1)
        for a0, b0, a1, b1 in zip(x0, y0, x1, y1):
            if b0 == b1:
                continue
            crosses = (b0 > y) != (b1 > y)
            at = a0 + (y - b0) * (a1 - a0) / (b1 - b0)
            hit ^= crosses & (x < at)
        inside[near] = hit
    return np.where(inside, -distance, distance).reshape(np.shape(lat))


# ---------------------------------------------------------------------------
# grading
# ---------------------------------------------------------------------------
def grading_numbers(grading: Mapping[str, Any] | None) -> dict[str, Any]:
    """The ramp scale (cells) and the distance between ring edges, from a row's ``grading``.

    THE FIELD THESE NUMBERS DRIVE.  A region's spacing blends to the background as a tanh in
    1/spacing centred on its edge, with scale ``ramp scale x its own spacing``.  At the edge the
    region asks for twice its spacing, which is the next ring's, and the growth it asks for there is
    ``2 / ramp scale`` per cell: that is the steepest of the ramp, so ``growth_per_cell`` g is a ramp
    scale of ``2 / g`` cells (18 cells, the legacy number, is 11 % per cell).  The fine spacing is
    reached about two scales inside the edge, so ring k's edge sits ``ring_cells + 2 x scale`` of
    its own cells outside ring k-1's for a flat ring ``ring_cells`` wide; the legacy ladder's
    ``ring_factor`` 3 is 1.5 scales, a ramp with no flat part.
    """

    grading = dict(grading or {})
    kind = grading.pop("kind", "ladder")
    if kind not in ("ladder", "rings", "ramp"):
        raise MeshAskRefusal(f"grading kind {kind!r} is not ladder, rings or ramp")
    growth = grading.pop("growth_per_cell", None)
    ramp_cells = grading.pop("ramp_cells", None)
    factor = grading.pop("transition_factor", None)
    ring_cells = grading.pop("ring_cells", None)
    ring_factor = grading.pop("ring_factor", None)
    if grading:
        raise MeshAskRefusal(f"grading has words this door does not know: {', '.join(sorted(grading))}")
    given = [name for name, value in (("growth_per_cell", growth), ("ramp_cells", ramp_cells),
                                      ("transition_factor", factor)) if value is not None]
    if len(given) > 1:
        raise MeshAskRefusal(f"grading names the ramp twice ({' and '.join(given)}); give one")
    if growth is not None:
        growth = float(growth)
        if not 0.0 < growth <= 0.25:
            raise MeshAskRefusal(
                f"growth_per_cell {growth:g} is not between 0 and 0.25 (25 % a cell).  The generator refuses a "
                f"ramp steeper than its repairs can follow, and a growth of nothing is no ramp")
        scale = 2.0 / growth
    elif ramp_cells is not None:
        # The ramp from one ring's spacing to the next occupies about two scales.
        scale = 0.5 * float(ramp_cells)
    elif factor is not None:
        scale = float(factor)
    else:
        scale = LEGACY_TRANSITION_FACTOR
    if not scale > 0.0:
        raise MeshAskRefusal("the ramp must have a width; a ramp of no cells is a jump between two cells")
    if ring_cells is not None and ring_factor is not None:
        raise MeshAskRefusal("grading names the ring width twice (ring_cells and ring_factor); give one")
    if kind == "rings" and ring_cells is None and ring_factor is None:
        raise MeshAskRefusal("grading kind rings needs ring_cells, the flat width of each ring in its own cells")
    if kind == "ramp" and (ring_cells is not None or ring_factor is not None):
        raise MeshAskRefusal("grading kind ramp has no flat rings; use kind rings to give ring_cells")
    if ring_cells is not None:
        ring_cells = float(ring_cells)
        if ring_cells < 0.0:
            raise MeshAskRefusal(f"ring_cells {ring_cells:g} is negative; a ring's flat width is zero or more cells")
        step_cells = ring_cells + 2.0 * scale
    else:
        ring_factor = LEGACY_RING_FACTOR if ring_factor is None else float(ring_factor)
        if not ring_factor > 0.0:
            raise MeshAskRefusal("ring_factor must be positive")
        # Legacy: an edge sits ring_factor ramp scales OF THE NEXT FINER ring outside it, which is
        # half that many of its own.
        step_cells = 0.5 * ring_factor * scale
        ring_cells = step_cells - 2.0 * scale
    return {"kind": kind, "ramp_scale_cells": scale, "edge_step_cells": step_cells,
            "ring_cells": ring_cells, "steepest_growth_per_cell": 2.0 / scale,
            "legacy_ring_factor": 2.0 * step_cells / scale}


def generator_request(row: Mapping[str, Any], geometry: int | None = None,
                      ramp_onset: float | None = None) -> dict[str, Any]:
    """The generator geometry a row is built under and the words ``rw_mpas_mesh`` is given for it.

    A door's own word (``geometry``, ``ramp_onset``) wins over the row's ``generator`` object.
    With no geometry named anywhere it is :data:`DEFAULT_GEOMETRY` (2).  Geometry 1 is no words at all: the
    generator is called as it always was, so ``--geometry 1`` gives the meshes of 0.6.1 and before.
    """

    document = row.get("generator")
    if document is None:
        document = {}
    if not isinstance(document, Mapping):
        raise MeshAskRefusal('the row\'s generator is an object: {"geometry": 2}')
    unknown = set(document) - {"geometry", "ramp_onset"}
    if unknown:
        raise MeshAskRefusal(f"generator has words this door does not know: {', '.join(sorted(unknown))}")
    chosen: Any = geometry if geometry is not None else document.get("geometry", DEFAULT_GEOMETRY)
    try:
        number = int(chosen)
        if isinstance(chosen, bool) or float(chosen) != number:
            raise ValueError(chosen)
    except (TypeError, ValueError):
        number = None
    if number not in GEOMETRIES:
        raise MeshAskRefusal(
            f"generator geometry {chosen!r} is not 1 or 2.  1 is the halving ladder every premade mesh was "
            f"built on; 2 grows the spacing by one rate per cell, builds every spacing as asked and hands out "
            f"no mesh with an obtuse triangle")
    onset: Any = ramp_onset if ramp_onset is not None else document.get("ramp_onset")
    if onset is not None:
        try:
            onset = float(onset)
        except (TypeError, ValueError) as error:
            raise MeshAskRefusal("ramp_onset is a number of transition widths") from error
        if number != 2:
            raise MeshAskRefusal(
                "ramp_onset belongs to geometry 2: it turns the corner where the ramp leaves the fine zone into "
                "a bend.  Geometry 1 has rings and no ramp to bend; name geometry 2 with it")
        if not 0.0 <= onset <= 6.0:
            raise MeshAskRefusal(
                f"ramp_onset {onset:g} is not between 0 and 6: it is the number of transition widths over which "
                f"the ramp's growth rises to its full rate")
    words: list[str] = []
    if number == 2:
        words = ["--geometry", "2"]
        if onset:
            words += ["--ramp-onset", f"{onset:g}"]
    return {"geometry": number, "ramp_onset": onset if onset else None, "words": words}


def add_generator_arguments(group: Any) -> None:
    """``--geometry`` and ``--ramp-onset``, the same two words on every door that builds a mesh."""

    group.add_argument(
        "--geometry", type=int, default=None, metavar="{1,2}",
        help="the generator's geometry.  2 (the default) builds every spacing as asked (no halving ladder), holds "
             "the fine spacing to the outline, grows it by one rate per cell (3.7 %% for the default 18-cell "
             "width) and hands out no mesh with an obtuse triangle; the same ask is a few per cent more columns "
             "than under 1, and the plan counts them.  1 is the halving ladder every premade mesh was built on "
             "(the meshes of 0.6.1 and before)")
    group.add_argument(
        "--ramp-onset", type=float, default=None, metavar="WIDTHS",
        help="geometry 2 only: transition widths (0 to 6) over which the ramp's growth rises to its full rate, "
             "so the fine zone's line is a bend and not a corner; fewer cells that are not hexagons inside the "
             "core, a few per cent more columns.  Default 0")


def door_generator(arguments: Any, row: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The generator a door's own words ask for (``--geometry``, ``--ramp-onset``), over a row's if one is given."""

    return generator_request(row or {}, getattr(arguments, "geometry", None), getattr(arguments, "ramp_onset", None))


def door_argv(arguments: Any) -> list[str]:
    """``--geometry`` and ``--ramp-onset`` as a door passes them on to the door it calls (nothing when not given)."""

    out: list[str] = []
    if getattr(arguments, "geometry", None) is not None:
        out += ["--geometry", str(int(arguments.geometry))]
    if getattr(arguments, "ramp_onset", None) is not None:
        out += ["--ramp-onset", repr(float(arguments.ramp_onset))]
    return out


def ramp_numbers(grading: Mapping[str, Any] | None) -> dict[str, Any]:
    """A row's ``grading`` under geometry 2: one growth rate per cell, from the zone's line outward.

    The generator is given a transition width ``W`` for each region and grows the spacing by
    ``1 / (1.5 x W in cells)`` a cell.  ``growth_per_cell`` g is therefore a width of ``1 / (1.5 g)``
    cells; ``ramp_cells`` N is the number of cells over which the spacing doubles (g = 2^(1/N) - 1);
    ``transition_factor`` is the width itself, the older doors' word (18 cells: 3.7 % a cell).
    There are no flat rings to describe, so the ring words are refused with the reason.
    """

    grading = dict(grading or {})
    kind = grading.pop("kind", "ladder")
    if kind not in ("ladder", "rings", "ramp"):
        raise MeshAskRefusal(f"grading kind {kind!r} is not ladder, rings or ramp")
    growth = grading.pop("growth_per_cell", None)
    ramp_cells = grading.pop("ramp_cells", None)
    factor = grading.pop("transition_factor", None)
    ring_cells = grading.pop("ring_cells", None)
    ring_factor = grading.pop("ring_factor", None)
    if grading:
        raise MeshAskRefusal(f"grading has words this door does not know: {', '.join(sorted(grading))}")
    if kind == "rings" or ring_cells is not None or (ring_factor is not None and kind != "ladder"):
        raise MeshAskRefusal(
            "geometry 2 has no flat rings to lay: the spacing grows by one rate per cell from the zone's line "
            "to the background.  Give growth_per_cell or ramp_cells (grading kind ramp), or build this row "
            "with geometry 1, which lays rings")
    given = [name for name, value in (("growth_per_cell", growth), ("ramp_cells", ramp_cells),
                                      ("transition_factor", factor)) if value is not None]
    if len(given) > 1:
        raise MeshAskRefusal(f"grading names the ramp twice ({' and '.join(given)}); give one")
    if growth is not None:
        growth = float(growth)
        if not 0.0 < growth <= 0.25:
            raise MeshAskRefusal(
                f"growth_per_cell {growth:g} is not between 0 and 0.25 (25 % a cell).  The generator refuses a "
                f"ramp steeper than its repairs can follow, and a growth of nothing is no ramp")
        scale = 1.0 / (RAMP_WIDTHS_PER_DOUBLING * growth)
    elif ramp_cells is not None:
        ramp_cells = float(ramp_cells)
        if not ramp_cells > 0.0:
            raise MeshAskRefusal("the ramp must have a width; a ramp of no cells is a jump between two cells")
        growth = 2.0 ** (1.0 / ramp_cells) - 1.0
        if growth > 0.25:
            raise MeshAskRefusal(
                f"ramp_cells {ramp_cells:g} doubles the spacing in so few cells that it grows "
                f"{100.0 * growth:.0f} % a cell; the steepest ramp is 25 % a cell, which is 3.2 cells a doubling")
        scale = 1.0 / (RAMP_WIDTHS_PER_DOUBLING * growth)
    else:
        scale = LEGACY_TRANSITION_FACTOR if factor is None else float(factor)
        if not scale > 0.0:
            raise MeshAskRefusal("the ramp must have a width; a ramp of no cells is a jump between two cells")
        growth = 1.0 / (RAMP_WIDTHS_PER_DOUBLING * scale)
        if growth > 0.25:
            raise MeshAskRefusal(
                f"transition_factor {scale:g} is a ramp of {100.0 * growth:.0f} % a cell; the steepest is 25 %")
    return {"kind": "ramp", "ramp_scale_cells": scale, "edge_step_cells": None, "ring_cells": 0.0,
            "steepest_growth_per_cell": growth, "growth_per_cell": growth,
            "cells_per_doubling": math.log(2.0) / math.log1p(growth), "legacy_ring_factor": None}


# ---------------------------------------------------------------------------
# the row
# ---------------------------------------------------------------------------
def load_row(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise MeshAskRefusal(f"{path}: cannot read a JSON row ({error})") from error
    if not isinstance(document, dict):
        raise MeshAskRefusal(f"{path}: a row is a JSON object")
    return document


def is_row(document: Mapping[str, Any]) -> bool:
    """A mesh-ask row, as against the generator's own resolution spec (which has ``regions``)."""

    return document.get("kind") == ROW_KIND or ("zones" in document and "regions" not in document)


_ROW_WORDS = {"kind", "name", "background_km", "fit", "grading", "zones", "domain", "hold_fine_to_outline",
              "generator", "comment"}
_ZONE_WORDS = {"name", "shape", "fine_dx_m", "inside", "fit", "grading", "comment"}


def _zones(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    zones = row.get("zones")
    if not isinstance(zones, Sequence) or isinstance(zones, (str, bytes)) or not zones:
        raise MeshAskRefusal("a row needs zones: a list of at least one {shape, fine_dx_m}")
    out = []
    names = set()
    for index, zone in enumerate(zones):
        if not isinstance(zone, Mapping):
            raise MeshAskRefusal(f"zone {index} is not an object")
        unknown = set(zone) - _ZONE_WORDS
        if unknown:
            raise MeshAskRefusal(f"zone {index} has words this door does not know: {', '.join(sorted(unknown))}")
        name = str(zone.get("name") or f"zone{index}")
        if name in names:
            raise MeshAskRefusal(f"two zones are named {name!r}; a nested zone names the zone that holds it")
        names.add(name)
        try:
            fine_m = float(zone["fine_dx_m"])
        except (KeyError, TypeError, ValueError) as error:
            raise MeshAskRefusal(f"zone {name!r} needs fine_dx_m, its spacing in metres") from error
        if not fine_m >= FINEST_FINE_M:
            raise MeshAskRefusal(
                f"zone {name!r} asks for {fine_m:g} m; the finest spacing this door writes is {FINEST_FINE_M:g} m")
        out.append({"name": name, "index": index, "asked_fine_km": fine_m / 1000.0,
                    "asked_shape": shape_row(zone.get("shape"), f"zone {name!r}'s shape"),
                    "inside": zone.get("inside"), "fit": zone.get("fit"), "grading": zone.get("grading")})
    return out


def _holder(zone: dict[str, Any], zones: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    """The zone that holds ``zone``: the one it names, else the finest coarser zone whose outline
    contains its centre."""

    if zone["inside"] is not None:
        for other in zones:
            if other["name"] == zone["inside"] and other is not zone:
                return other
        raise MeshAskRefusal(f"zone {zone['name']!r} says it is inside {zone['inside']!r}, and no zone has that name")
    lat, lon = shape_centre(zone["asked_shape"])
    best = None
    for other in zones:
        if other is zone or not other["asked_fine_km"] > zone["asked_fine_km"]:
            continue
        if float(signed_distance_km(other["asked_shape"], lat, lon)) < 0.0:
            if best is None or other["asked_fine_km"] < best["asked_fine_km"]:
                best = other
    return best


def _outline_points(shape: Mapping[str, Any], step_km: float = 2.0):
    """Points along a shape's line (latitudes, longitudes), about ``step_km`` apart."""

    import numpy as np

    from . import mesh_shape

    if shape["kind"] == "cap":
        lat0, lon0 = (float(v) for v in shape["center_deg"])
        radius = float(shape["radius_km"])
        count = max(64, int(math.ceil(2.0 * math.pi * radius / step_km)))
        az = np.linspace(0.0, 2.0 * math.pi, count, endpoint=False)
        return mesh_shape._Azimuthal(lat0, lon0).inverse(radius * np.sin(az), radius * np.cos(az))
    ring = mesh_shape._ring(shape)
    return mesh_shape._latlon(mesh_shape._densify(mesh_shape._xyz(*ring.T), step_km / EARTH_RADIUS_KM))


def _check_holders(zones: Sequence[dict[str, Any]]) -> None:
    """A zone that names its holder lies inside it, at least in part.

    THE BREAKAGE IT PREVENTS: ``inside`` was taken on the zone's word.  A zone wholly outside the zone
    it names was built with its grading stopped at the holder's spacing, out where the holder is not,
    so the spacing jumped from the holder's to the background's at the zone's outer edge (geometry 1),
    or the ramp climbed to a spacing no zone there asked for (geometry 2).
    """

    import numpy as np

    for zone in zones:
        holder = zone.get("holder")
        if holder is None or zone["inside"] is None:
            continue
        lat, lon = _outline_points(zone["asked_shape"])
        clat, clon = shape_centre(zone["asked_shape"])
        d = np.append(np.asarray(signed_distance_km(holder["asked_shape"], lat, lon)).ravel(),
                      float(signed_distance_km(holder["asked_shape"], clat, clon)))
        hlat, hlon = shape_centre(holder["asked_shape"])
        if float(d.min()) > 0.0 and not float(signed_distance_km(zone["asked_shape"], hlat, hlon)) < 0.0:
            raise MeshAskRefusal(
                f"zone {zone['name']!r} says it is inside {holder['name']!r}, but lies wholly outside it: its "
                f"nearest point is {float(d.min()):,.0f} km beyond {holder['name']!r}'s line.  A nested zone sits "
                f"inside the zone that holds it.  Move it inside, name the zone it is in, or leave out inside "
                f"to make it a zone of its own")


def _check_ramp_room(zones: Sequence[dict[str, Any]]) -> list[str]:
    """Under geometry 2 a nested zone's ramp reaches its holder's spacing somewhere inside the holder.

    THE BREAKAGE IT PREVENTS: a zone finer than its holder by a ratio whose ramp is longer than the
    holder is wide leaves no cell of the holder at the holder's spacing: the zone the row names is
    built nowhere, and the ramp runs on out past the holder's line at spacings no zone asked for.
    Where the ramp reaches the holder's spacing on one side and not on another (the stepped row's 1 km core
    inside a 3 km zone, which stepped the model cleanly), the plan says so in a note and builds.
    """

    import numpy as np

    notes: list[str] = []
    for zone in zones:
        holder = zone.get("holder")
        if holder is None:
            continue
        reach = float(zone["reach_km"])
        growth = 100.0 * zone["grading_numbers"]["growth_per_cell"]
        hlat, hlon = _outline_points(holder["asked_shape"])
        farthest = float(np.asarray(signed_distance_km(zone["asked_shape"], hlat, hlon)).max())
        if reach >= farthest:
            raise MeshAskRefusal(
                f"zone {zone['name']!r} ({zone['fine_km'] * 1000.0:g} m) is {holder['fine_km'] / zone['fine_km']:.2f} "
                f"times finer than {holder['name']!r} ({holder['fine_km'] * 1000.0:g} m), and geometry 2's ramp "
                f"needs {reach:,.0f} km beyond {zone['name']!r}'s line to grow from {zone['fine_km']:g} km to "
                f"{holder['fine_km']:g} km at {growth:.1f} % a cell; no point of {holder['name']!r} is farther than "
                f"{max(farthest, 0.0):,.0f} km from that line, so {holder['name']!r}'s {holder['fine_km']:g} km would "
                f"be built nowhere.  Make {holder['name']!r} reach at least {math.ceil(reach):,} km beyond "
                f"{zone['name']!r}, ask a coarser {zone['name']!r} or a finer {holder['name']!r}, or give a steeper "
                f"grading")
        zlat, zlon = _outline_points(zone["asked_shape"])
        room = -float(np.asarray(signed_distance_km(holder["asked_shape"], zlat, zlon)).max())
        if reach > room:
            notes.append(
                f"zone {zone['name']!r}: its ramp reaches {holder['fine_km']:g} km {reach:,.0f} km beyond its line, "
                f"and on its nearest side {holder['name']!r}'s line is {room:,.0f} km away, so there the cells "
                f"cross {holder['name']!r}'s line still finer than {holder['fine_km']:g} km (the columns above count them)")
    return notes


def expand(row: Mapping[str, Any], *, geometry: int | None = None,
           ramp_onset: float | None = None) -> dict[str, Any]:
    """A row as the generator's spec, with the cut for a region and what a plan has to say.

    ``geometry`` and ``ramp_onset`` are a door's own words; they win over the row's ``generator``.
    """

    unknown = set(row) - _ROW_WORDS
    if unknown:
        raise MeshAskRefusal(f"the row has words this door does not know: {', '.join(sorted(unknown))}")
    if row.get("kind") not in (None, ROW_KIND):
        raise MeshAskRefusal(f"the row's kind is {row.get('kind')!r}; this door reads {ROW_KIND}")
    fit = str(row.get("fit", "nearest"))
    if fit not in FITS:
        raise MeshAskRefusal(f"fit {fit!r} is not one of {', '.join(FITS)}")
    zones = _zones(row)
    try:
        asked_background = float(row.get("background_km", 120.0))
    except (TypeError, ValueError) as error:
        raise MeshAskRefusal("background_km is not a number") from error
    if not 0.0 < asked_background <= COARSEST_BACKGROUND_KM:
        raise MeshAskRefusal(
            f"background_km {asked_background:g} is not between 0 and {COARSEST_BACKGROUND_KM:g} km")
    hold = bool(row.get("hold_fine_to_outline", True))
    notes: list[str] = []
    generator = generator_request(row, geometry, ramp_onset)
    if generator["geometry"] == 2:
        return _expand_ramp(row, zones, asked_background, fit, generator)

    # One ladder, standing on the finest zone.
    anchor = min(zone["asked_fine_km"] for zone in zones)
    for zone in zones:
        if zone["asked_fine_km"] <= anchor * (1.0 + 1e-12):
            zone["fine_km"] = anchor
            zone["fit_record"] = None
            continue
        record = fit_to_ladder(zone["asked_fine_km"], anchor, str(zone["fit"] or ZONE_FIT))
        zone["fine_km"] = record["built_km"]
        zone["fit_record"] = record if record["moved"] else None
        if record["moved"]:
            notes.append(
                f"zone {zone['name']!r}: asked {zone['asked_fine_km'] * 1000.0:g} m, built "
                f"{record['built_km'] * 1000.0:g} m.  One mesh holds only spacings that halve into each other, "
                f"and the finest zone ({anchor * 1000.0:g} m) is kept exact; the other side is "
                f"{(record['coarser_km'] if record['built_km'] == record['finer_km'] else (record['finer_km'] or anchor)) * 1000.0:g} m")
    if not asked_background > max(zone["fine_km"] for zone in zones):
        raise MeshAskRefusal(
            f"the {asked_background:g} km background is not coarser than the coarsest zone "
            f"({max(zone['fine_km'] for zone in zones):g} km), so there is nothing to refine")
    background = fit_to_ladder(asked_background, anchor, fit)
    if background["built_km"] <= max(zone["fine_km"] for zone in zones) * (1.0 + 1e-12):
        background = fit_to_ladder(asked_background, anchor, "coarser")
    if background["moved"]:
        other = background["coarser_km"] if background["built_km"] == background["finer_km"] else background["finer_km"]
        notes.append(
            f"background: asked {asked_background:g} km, built {background['built_km']:g} km, the nearest "
            f"spacing the {anchor * 1000.0:g} m core halves up to"
            + (f" (the other side is {other:g} km; fit: finer / coarser chooses)" if other else ""))

    # Who holds whom, and each zone's rings.
    for zone in zones:
        zone["holder"] = _holder(zone, zones)
    _check_holders(zones)
    for zone in zones:
        holder = zone["holder"]
        if holder is not None and not holder["fine_km"] > zone["fine_km"]:
            raise MeshAskRefusal(
                f"zone {zone['name']!r} ({zone['fine_km'] * 1000.0:g} m) is inside {holder['name']!r} "
                f"({holder['fine_km'] * 1000.0:g} m), which is no coarser; a nested zone is finer than its holder")
    regions: list[dict[str, Any]] = []
    region_owner: list[dict[str, Any]] = []
    for zone in zones:
        numbers = grading_numbers(zone["grading"] if zone["grading"] is not None else row.get("grading"))
        zone["grading_numbers"] = numbers
        ceiling = zone["holder"]["fine_km"] if zone["holder"] is not None else background["built_km"]
        rungs = rungs_between(zone["fine_km"], ceiling)
        scale = numbers["ramp_scale_cells"]
        offset = HOLD_RAMP_SCALES * scale * zone["fine_km"] if hold else 0.0
        zone["rings"] = []
        for position, spacing in enumerate(rungs):
            if position:
                offset += numbers["edge_step_cells"] * spacing
            tolerance = min(0.5 * spacing, 2.0) if position == 0 else 0.5 * spacing
            shape = grown(zone["asked_shape"], offset, tolerance)
            region = {"shape": shape, "spacing_km": spacing, "transition_km": scale * spacing}
            regions.append(region)
            region_owner.append({"zone": zone["name"], "spacing_km": spacing, "edge_beyond_outline_km": offset})
            zone["rings"].append({"spacing_km": spacing, "edge_beyond_outline_km": round(offset, 3),
                                  "ramp_scale_km": scale * spacing})
        zone["reach_km"] = offset

    domain = dict(row.get("domain") or {"kind": "globe"})
    cut = None
    if domain.get("kind", "globe") == "globe":
        if set(domain) - {"kind"}:
            raise MeshAskRefusal("a globe has no rim; rim words belong to domain kind region")
    elif domain.get("kind") == "region":
        cut, extra, rim_notes = _region_domain(domain, zones, background, row)
        regions.extend(extra)
        notes.extend(rim_notes)
    else:
        raise MeshAskRefusal(f"domain kind {domain.get('kind')!r} is not globe or region")

    # Coarsest first, the order the legacy ladder writes (the field is a max; order moves no cell).
    order = sorted(range(len(regions)), key=lambda i: (-regions[i]["spacing_km"], i))
    regions = [regions[i] for i in order]
    name = row.get("name") or _default_name(zones, background)
    spec = {"name": str(name), "background_km": float(background["built_km"]), "regions": regions}
    return {
        "kind": ROW_KIND,
        "spec": spec,
        "cut": cut,
        "notes": notes,
        "anchor_fine_km": anchor,
        "background": background,
        "hold_fine_to_outline": hold,
        "domain": domain,
        "zones": [{
            "name": zone["name"], "asked_fine_m": zone["asked_fine_km"] * 1000.0, "fine_m": zone["fine_km"] * 1000.0,
            "inside": zone["holder"]["name"] if zone["holder"] is not None else None,
            "shape": zone["asked_shape"], "centre_deg": list(shape_centre(zone["asked_shape"])),
            "area_km2": shape_area_km2(zone["asked_shape"]), "rings": zone["rings"],
            "grading": zone["grading_numbers"], "reach_km": zone["reach_km"],
        } for zone in zones],
    }


def _outline(shape: Mapping[str, Any], fine_km: float) -> dict[str, Any]:
    """A zone's own outline as the generator's region under geometry 2.

    A circle, or a drawn outline of up to ``RING_VERTICES`` corners, goes in as asked.  A denser one
    is redrawn half a cell outside itself with that many corners (see ``RING_VERTICES``), so the
    zone still covers everything asked.
    """

    if shape["kind"] == "cap" or len(shape.get("vertices_deg") or ()) <= RING_VERTICES:
        return dict(shape)
    return grown(shape, 0.5 * fine_km, min(0.5 * fine_km, 2.0))


def ramp_reach_km(fine_km: float, ceiling_km: float, width_km: float, onset_widths: float = 0.0) -> float:
    """How far outside a zone's line geometry 2's ramp reaches ``ceiling_km``."""

    return (RAMP_WIDTHS_PER_DOUBLING * width_km * (ceiling_km / fine_km - 1.0)
            + 0.5 * onset_widths * width_km)


def _expand_ramp(row: Mapping[str, Any], zones: list[dict[str, Any]], asked_background: float, fit: str,
                 generator: Mapping[str, Any]) -> dict[str, Any]:
    """A row under geometry 2: every spacing as asked, one region a zone, no rings."""

    notes: list[str] = []
    anchor = min(zone["asked_fine_km"] for zone in zones)
    for zone in zones:
        zone["fine_km"] = zone["asked_fine_km"]
        zone["fit_record"] = None
    coarsest = max(zone["fine_km"] for zone in zones)
    if not asked_background > coarsest:
        raise MeshAskRefusal(
            f"the {asked_background:g} km background is not coarser than the coarsest zone "
            f"({coarsest:g} km), so there is nothing to refine")
    background = {"asked_km": asked_background, "built_km": asked_background, "halvings": None, "moved": False,
                  "finer_km": asked_background, "coarser_km": asked_background}
    # What the row's fit words would have done on the halving ladder, said once, so a caller who
    # priced the row under geometry 1 knows why the columns moved.
    for zone in zones:
        if zone["fine_km"] > anchor * (1.0 + 1e-12):
            how = str(zone["fit"] or ZONE_FIT)
            ladder = fit_to_ladder(zone["fine_km"], anchor, ZONE_FIT if how == "exact" else how)
            if ladder["moved"]:
                notes.append(
                    f"zone {zone['name']!r}: built at the {zone['fine_km'] * 1000.0:g} m asked (geometry 1 would "
                    f"build {ladder['built_km'] * 1000.0:g} m, a halving of the {anchor * 1000.0:g} m core)")
    ladder = fit_to_ladder(asked_background, anchor, "nearest" if fit == "exact" else fit)
    if ladder["moved"]:
        notes.append(
            f"background: built at the {asked_background:g} km asked (geometry 1 would build "
            f"{ladder['built_km']:g} km, the spacing the {anchor * 1000.0:g} m core halves up to).  Geometry 2 "
            f"grows the spacing by one rate per cell and has no halving ladder, so fit has nothing to move")
    for zone in zones:
        zone["holder"] = _holder(zone, zones)
    _check_holders(zones)
    for zone in zones:
        holder = zone["holder"]
        if holder is not None and not holder["fine_km"] > zone["fine_km"]:
            raise MeshAskRefusal(
                f"zone {zone['name']!r} ({zone['fine_km'] * 1000.0:g} m) is inside {holder['name']!r} "
                f"({holder['fine_km'] * 1000.0:g} m), which is no coarser; a nested zone is finer than its holder")
    onset = float(generator.get("ramp_onset") or 0.0)
    regions: list[dict[str, Any]] = []
    for zone in zones:
        numbers = ramp_numbers(zone["grading"] if zone["grading"] is not None else row.get("grading"))
        zone["grading_numbers"] = numbers
        ceiling = zone["holder"]["fine_km"] if zone["holder"] is not None else background["built_km"]
        width_km = numbers["ramp_scale_cells"] * zone["fine_km"]
        regions.append({"shape": _outline(zone["asked_shape"], zone["fine_km"]), "spacing_km": zone["fine_km"],
                        "transition_km": width_km})
        reach = ramp_reach_km(zone["fine_km"], ceiling, width_km, onset)
        zone["rings"] = [{"spacing_km": zone["fine_km"], "edge_beyond_outline_km": 0.0, "ramp_scale_km": width_km}]
        zone["ramp"] = {"growth_per_cell": numbers["growth_per_cell"], "to_km": ceiling,
                        "reaches_it_km_beyond_outline": round(reach, 3)}
        zone["reach_km"] = reach
    notes.extend(_check_ramp_room(zones))

    domain = dict(row.get("domain") or {"kind": "globe"})
    cut = None
    if domain.get("kind", "globe") == "globe":
        if set(domain) - {"kind"}:
            raise MeshAskRefusal("a globe has no rim; rim words belong to domain kind region")
    elif domain.get("kind") == "region":
        cut, extra, rim_notes = _region_domain_ramp(domain, zones, background, onset)
        regions.extend(extra)
        notes.extend(rim_notes)
    else:
        raise MeshAskRefusal(f"domain kind {domain.get('kind')!r} is not globe or region")

    order = sorted(range(len(regions)), key=lambda i: (-regions[i]["spacing_km"], i))
    regions = [regions[i] for i in order]
    name = row.get("name") or _default_name(zones, background)
    spec = {"name": str(name), "background_km": float(background["built_km"]), "regions": regions}
    # Every region of a geometry 2 row is a zone someone asked for, so the generator's search for
    # the older doors' halving rings (which it replaces by the ramp) is told to keep them all.  The
    # search judges a region against the LADDER's line, which is gentler than the ramp built: a
    # 12 km rim held 420 km out from a 3 km zone lies under that line (10.8 km there), would be
    # taken for a ring and dropped, and the cut would then fall on 14 to 19 km cells.
    words = [*generator["words"], "--rings", "keep"]
    return {
        "kind": ROW_KIND,
        "spec": spec,
        "cut": cut,
        "notes": notes,
        "anchor_fine_km": anchor,
        "background": background,
        "hold_fine_to_outline": True,
        "domain": domain,
        "generator": {"geometry": 2, "ramp_onset": generator.get("ramp_onset"), "words": words},
        "zones": [{
            "name": zone["name"], "asked_fine_m": zone["asked_fine_km"] * 1000.0, "fine_m": zone["fine_km"] * 1000.0,
            "inside": zone["holder"]["name"] if zone["holder"] is not None else None,
            "shape": zone["asked_shape"], "centre_deg": list(shape_centre(zone["asked_shape"])),
            "area_km2": shape_area_km2(zone["asked_shape"]), "rings": zone["rings"], "ramp": zone["ramp"],
            "grading": zone["grading_numbers"], "reach_km": zone["reach_km"],
        } for zone in zones],
    }


def _region_base(domain: Mapping[str, Any], zones: Sequence[dict[str, Any]]) -> tuple[dict[str, Any], float]:
    """The zone a region is cut around and the rim distance, with the refusals both geometries share."""

    unknown = set(domain) - {"kind", "rim_km", "rim_spacing_km", "boundary_rings", "around"}
    if unknown:
        raise MeshAskRefusal(f"domain has words this door does not know: {', '.join(sorted(unknown))}")
    outer = [zone for zone in zones if zone["holder"] is None]
    around = domain.get("around")
    if around is not None:
        chosen = [zone for zone in zones if zone["name"] == around]
        if not chosen:
            raise MeshAskRefusal(f"domain.around names {around!r}, and no zone has that name")
        base = chosen[0]
    elif len(outer) == 1:
        base = outer[0]
    else:
        raise MeshAskRefusal(
            f"a region is cut around one zone, and this row has {len(outer)} zones that sit in no other "
            f"({', '.join(zone['name'] for zone in outer)}); name one with domain.around")
    try:
        rim_km = float(domain["rim_km"])
    except (KeyError, TypeError, ValueError) as error:
        raise MeshAskRefusal("domain kind region needs rim_km: the distance from the zone's outline to the cut") from error
    if not rim_km > 0.0:
        raise MeshAskRefusal(
            f"rim_km {rim_km:g} is not positive: a cut on the fine zone's own outline lays the boundary rings on "
            f"fine cells with no coarser cells to carry the driving model in")
    rings = int(domain.get("boundary_rings", BOUNDARY_RINGS))
    if rings != BOUNDARY_RINGS:
        raise MeshAskRefusal(
            f"boundary_rings {rings} is not {BOUNDARY_RINGS}: the regional runs relax and specify over exactly "
            f"{BOUNDARY_RINGS} rings of cells, and the cut tool marks that many")
    return base, rim_km


def _region_domain_ramp(domain: Mapping[str, Any], zones: Sequence[dict[str, Any]],
                        background: Mapping[str, Any], onset: float):
    """The cut of a limited-area mesh under geometry 2, and the region that holds the rim spacing.

    The words are ``_region_domain``'s.  What differs is the arithmetic: there are no rings to reach
    the rim spacing, the ramp does, at its one rate; and a region's spacing holds to its own line,
    so the rim region ends ``RIM_MARGIN_CELLS`` cells beyond the last boundary ring.
    """

    base, rim_km = _region_base(domain, zones)
    notes: list[str] = []
    extra: list[dict[str, Any]] = []
    rim_spacing = domain.get("rim_spacing_km")
    if rim_spacing is not None:
        rim_spacing = float(rim_spacing)
        if rim_spacing < base["fine_km"] * (1.0 - 1e-12):
            raise MeshAskRefusal(
                f"rim_spacing_km {rim_spacing:g} is finer than the zone it rims ({base['fine_km']:g} km)")
        if not rim_spacing < background["built_km"] * (1.0 + 1e-12):
            raise MeshAskRefusal(
                f"the rim spacing {rim_spacing:g} km is coarser than the {background['built_km']:g} km background")
        numbers = base["grading_numbers"]
        ends = ramp_reach_km(base["fine_km"], rim_spacing, numbers["ramp_scale_cells"] * base["fine_km"], onset)
        if rim_km < ends:
            raise MeshAskRefusal(
                f"the ramp needs {ends:,.0f} km beyond the outline to grow from {base['fine_km']:g} km to the "
                f"{rim_spacing:g} km rim at {100.0 * numbers['growth_per_cell']:.1f} % a cell, and rim_km is "
                f"{rim_km:g}: the cut would fall on the ramp, where the boundary rings would change size across "
                f"the boundary zone.  Give rim_km of at least {math.ceil(ends):,} km, a finer rim_spacing_km, or "
                f"a steeper grading")
        if rim_spacing < background["built_km"] * (1.0 - 1e-9):
            hold_to = rim_km + (BOUNDARY_RINGS + RIM_MARGIN_CELLS) * rim_spacing
            extra.append({"shape": grown(base["asked_shape"], hold_to, 0.5 * rim_spacing), "spacing_km": rim_spacing,
                          "transition_km": numbers["ramp_scale_cells"] * rim_spacing})
            notes.append(f"rim: {rim_spacing:g} km cells are held from where the ramp reaches them ({ends:,.0f} km "
                         f"outside the outline) to the cut at {rim_km:g} km and across the {BOUNDARY_RINGS} "
                         f"boundary rings")
    cut = grown(base["asked_shape"], rim_km, 1.0)
    return ({"region": cut, "around": base["name"], "rim_km": rim_km, "boundary_rings": BOUNDARY_RINGS,
             "rim_spacing_km": rim_spacing}, extra, notes)


def _default_name(zones: Sequence[dict[str, Any]], background: Mapping[str, Any]) -> str:
    first = zones[0]
    lat, lon = shape_centre(first["asked_shape"])
    more = f" and {len(zones) - 1} more" if len(zones) > 1 else ""
    return (f"ask {first['fine_km'] * 1000.0:g}m {first['asked_shape']['kind']} at {lat:.4f}N {lon:.4f}E{more} "
            f"on {background['built_km']:g}km")


def _region_domain(domain: Mapping[str, Any], zones: Sequence[dict[str, Any]], background: Mapping[str, Any],
                   row: Mapping[str, Any]):
    """The cut of a limited-area mesh, and the region that holds the rim spacing out to it.

    ``rim_km`` is the distance from the outline of the zone the region is built around (the first
    zone with no holder) to the cut.  ``rim_spacing_km`` is the spacing held from the end of the
    rings out to the cut and across the seven boundary rings: the cells the lateral boundary data is
    laid on.  Without it the rim takes whatever the rings have reached there.
    """

    base, rim_km = _region_base(domain, zones)
    notes = []
    extra: list[dict[str, Any]] = []
    anchor = min(zone["fine_km"] for zone in zones)
    rim_spacing = domain.get("rim_spacing_km")
    rim_record = None
    if rim_spacing is not None:
        rim_spacing = float(rim_spacing)
        if rim_spacing < base["fine_km"] * (1.0 - 1e-12):
            raise MeshAskRefusal(
                f"rim_spacing_km {rim_spacing:g} is finer than the zone it rims ({base['fine_km']:g} km)")
        if rim_spacing <= base["fine_km"] * (1.0 + 1e-12):
            rim_built = base["fine_km"]
        else:
            rim_record = fit_to_ladder(rim_spacing, anchor, str(row.get("fit", "nearest")))
            rim_built = rim_record["built_km"]
            if rim_record["moved"]:
                notes.append(f"rim: asked {rim_spacing:g} km, built {rim_built:g} km (the halving ladder)")
        if not rim_built < background["built_km"] * (1.0 + 1e-12):
            raise MeshAskRefusal(
                f"the rim spacing {rim_built:g} km is coarser than the {background['built_km']:g} km background")
        numbers = base["grading_numbers"]
        reach = None
        for ring in base["rings"]:
            if abs(ring["spacing_km"] - rim_built) <= 1e-9 * rim_built:
                reach = ring["edge_beyond_outline_km"]
        ends = max((ring["edge_beyond_outline_km"] for ring in base["rings"]
                    if ring["spacing_km"] < rim_built * (1.0 - 1e-9)), default=0.0)
        if rim_km < ends:
            raise MeshAskRefusal(
                f"the rings need {ends:,.0f} km beyond the outline to climb from {base['fine_km']:g} km to the "
                f"{rim_built:g} km rim, and rim_km is {rim_km:g}: the cut would fall on the ramp, where the "
                f"boundary rings would change size across the boundary zone.  Give rim_km of at least "
                f"{math.ceil(ends):,} km, a finer rim_spacing_km, or a steeper grading")
        # The rim plateau runs past the cut by the boundary rings and two ramp scales, so the cut and
        # its rings sit on flat rim spacing.
        hold_to = rim_km + (BOUNDARY_RINGS + 2.0 * numbers["ramp_scale_cells"]) * rim_built
        if rim_built < background["built_km"] * (1.0 - 1e-9) and (reach is None or reach < hold_to):
            tolerance = 0.5 * rim_built
            extra.append({"shape": grown(base["asked_shape"], hold_to, tolerance), "spacing_km": rim_built,
                          "transition_km": numbers["ramp_scale_cells"] * rim_built})
            notes.append(f"rim: {rim_built:g} km cells are held from the rings out to the cut at {rim_km:g} km "
                         f"and across the {BOUNDARY_RINGS} boundary rings")
    cut = grown(base["asked_shape"], rim_km, 1.0)
    return ({"region": cut, "around": base["name"], "rim_km": rim_km, "boundary_rings": BOUNDARY_RINGS,
             "rim_spacing_km": rim_spacing if rim_spacing is None else (rim_record or {}).get("built_km", rim_spacing)},
            extra, notes)


# ---------------------------------------------------------------------------
# the legacy doors, as rows
# ---------------------------------------------------------------------------
def legacy_point_row(point: tuple[float, float], *, fine_dx_m: float, radius_km: float, background_km: float,
                     transition_factor: float = LEGACY_TRANSITION_FACTOR, ring_factor: float = LEGACY_RING_FACTOR,
                     name: str | None = None) -> dict[str, Any]:
    """The row that expands to ``mesh_point.ladder_spec`` of the same arguments, byte for byte."""

    row: dict[str, Any] = {
        "kind": ROW_KIND, "background_km": float(background_km), "fit": "exact", "hold_fine_to_outline": False,
        "grading": {"kind": "ladder", "transition_factor": float(transition_factor), "ring_factor": float(ring_factor)},
        "zones": [{"shape": {"kind": "cap", "center_deg": [point[0], point[1]], "radius_km": float(radius_km)},
                   "fine_dx_m": float(fine_dx_m)}],
    }
    if name is not None:
        row["name"] = name
    return row


# ---------------------------------------------------------------------------
# the plan: what a row costs, before anything is generated
# ---------------------------------------------------------------------------
#: The smallest cell-centre distance of a generated mesh over its fine spacing.  MEASURED on five
#: meshes the line's generator made on 18 Nov 2026 (OWN-MESH.md table): 0.727, 0.746, 0.757, 0.745,
#: 0.795.  A plan takes the smallest, so the step it prints is one the built mesh has admitted.
SMALLEST_EDGE_OVER_FINE = 0.72
#: Points of the census lattice.  At four million a sample stands for 128 km2, so a 150 km core is
#: about 550 samples and its count is good to a few per cent; the whole-mesh count a plan prints is
#: the generator's own dry run, not this.
CENSUS_SAMPLES = 4_000_000
PLAN_CARD_COUNTS = (1, 2, 4, 8)
PLAN_CARDS = ("96gb", "32gb")
#: (suite, levels) a plan prices when the caller names none: the NOAA suite's own 59 levels and the
#: default suite's 55.
PLAN_SUITES = (("hrrrv5", 59), ("convection_permitting", 55))


def _lattice(samples: int):
    import numpy as np

    k = np.arange(samples, dtype=float) + 0.5
    lat = np.degrees(np.arcsin(1.0 - 2.0 * k / samples))
    lon = (np.degrees(k * math.pi * (3.0 - math.sqrt(5.0))) + 180.0) % 360.0 - 180.0
    return lat, lon


def _reach(shape: Mapping[str, Any]) -> tuple[tuple[float, float], float]:
    """A circle that holds the shape: its centre and radius, km."""

    import numpy as np

    from . import mesh_shape

    if shape["kind"] == "cap":
        return (float(shape["center_deg"][0]), float(shape["center_deg"][1])), float(shape["radius_km"])
    lat0, lon0 = mesh_shape.centroid(shape)
    ring = mesh_shape._ring(shape)
    cos = np.clip(mesh_shape._xyz(*ring.T) @ mesh_shape._xyz(lat0, lon0), -1.0, 1.0)
    return (lat0, lon0), float(EARTH_RADIUS_KM * np.arccos(cos).max())


def field_spacing_km(asked: Mapping[str, Any], lat: Any, lon: Any) -> Any:
    """The spacing (km) the generator relaxes to at the given points: :func:`census`'s field, point by point.

    Geometry 2 grows each region's spacing as ``h * (1 + s / (1.5 W))`` at distance s outside its line (with an
    onset, a parabola first); geometry 1 blends 1/spacing to the background by tanh.  The finest wins, capped at
    the background, exactly as :func:`census` evaluates it on its lattice.
    """

    import numpy as np

    spec = asked["spec"]
    lat = np.atleast_1d(np.asarray(lat, dtype=float))
    lon = np.atleast_1d(np.asarray(lon, dtype=float))
    back_inv = 1.0 / float(spec["background_km"])
    inv = np.full(lat.shape, back_inv)
    generator = asked.get("generator") or {}
    ramp = int(generator.get("geometry", 1)) == 2
    onset = float(generator.get("ramp_onset") or 0.0)
    for region in spec["regions"]:
        width = float(region["transition_km"])
        fine = float(region["spacing_km"])
        if ramp:
            doubling = RAMP_WIDTHS_PER_DOUBLING * width
            s_out = np.maximum(signed_distance_km(region["shape"], lat, lon), 0.0)
            bend = onset * width
            if bend > 0.0:
                allowed = np.where(s_out < bend, fine * (1.0 + s_out * s_out / (2.0 * bend * doubling)),
                                   fine * (1.0 + (s_out - 0.5 * bend) / doubling))
            else:
                allowed = fine * (1.0 + s_out / doubling)
            inv = np.maximum(inv, 1.0 / allowed)
            continue
        distance = signed_distance_km(region["shape"], lat, lon)
        inv = np.maximum(inv, back_inv + (1.0 / fine - back_inv) * 0.5 * (1.0 - np.tanh(distance / width)))
    return 1.0 / inv


#: The walked rings' width is scaled by this.  A relaxed mesh's rings are not straight rows of the field's spacing:
#: measured on two geometry-2 regions, the outermost kept cell lay past the cut by 1.045 times the walked width (10 km,
#: 800 km at 38.5,-93.0, cut 1,080 km: 173.8 against 166.3 km) and by 1.17 times (the 3 km storm box at 40.6,-91.3,
#: cut 405 km: 66.3 against 56.6 km).
RING_REACH_SCALE = 1.25


def boundary_ring_reach_km(asked: Mapping[str, Any], *, bearings: int = 72) -> float:
    """How far past the cut its boundary rings reach, km: each ring walked outward at its own spacing.

    THE BREAKAGE THIS PREVENTS (0.6.2, 20 Nov 2026): the plan gave the rings' width as their count times the
    spacing AT the cut, but past the cut geometry 2's spacing keeps growing, so every ring is wider than the one
    inside it.  The 10 km, 800 km region at 38.5,-93.0 from HRRR (cut 1,080 km, 20.4 km cells at the cut) was
    given 142.7 km; its rings reached 173.8 km, and with the window's 30 km margin one of its 35,043 cells lay
    0.0006 degrees outside a ``--from-plan`` HRRR window, so the init refused after a 28-minute intermediate.
    Here each ring advances by the field's spacing at its own outer side (solved by a few fixed-point steps), along
    every bearing from the cut's centre, scaled by :data:`RING_REACH_SCALE`; the farthest bearing is returned.
    """

    import numpy as np

    cut = asked["cut"]
    (lat0, lon0), radius = _reach(cut["region"])
    rings = int(cut["boundary_rings"])
    theta = np.radians(np.arange(bearings) * 360.0 / bearings)
    phi0, lam0 = math.radians(lat0), math.radians(lon0)

    def spacing_at(distance_km):
        delta = distance_km / EARTH_RADIUS_KM
        phi = np.arcsin(np.clip(math.sin(phi0) * np.cos(delta) + math.cos(phi0) * np.sin(delta) * np.cos(theta),
                                -1.0, 1.0))
        lam = lam0 + np.arctan2(np.sin(theta) * np.sin(delta) * math.cos(phi0),
                                np.cos(delta) - math.sin(phi0) * np.sin(phi))
        return field_spacing_km(asked, np.degrees(phi), (np.degrees(lam) + 540.0) % 360.0 - 180.0)

    reach = np.full(theta.shape, float(radius))
    for _ in range(rings):
        step = spacing_at(reach)
        for _ in range(4):
            step = spacing_at(reach + step)
        reach = reach + step
    return RING_REACH_SCALE * float((reach - radius).max())


def census(asked: Mapping[str, Any], *, samples: int = CENSUS_SAMPLES) -> dict[str, Any]:
    """Columns by zone and the spacing percentiles, from the generator's own field on a lattice.

    The field is the one ``rw_mpas_mesh`` relaxes to (rw-mpas ``PreparedSpec::spacing_m``): each
    region blends 1/spacing to the background as ``0.5 * (1 - tanh(distance / ramp scale))`` and the
    finest wins.  Columns are the area over one hexagon of the local spacing.
    """

    import numpy as np

    from . import mesh_shape

    spec = asked["spec"]
    lat, lon = _lattice(samples)
    xyz = mesh_shape._xyz(lat, lon)
    back_inv = 1.0 / float(spec["background_km"])
    inv = np.full(samples, back_inv)
    generator = asked.get("generator") or {}
    ramp = int(generator.get("geometry", 1)) == 2
    onset = float(generator.get("ramp_onset") or 0.0)
    for region in spec["regions"]:
        width = float(region["transition_km"])
        fine_inv = 1.0 / float(region["spacing_km"])
        (lat0, lon0), radius = _reach(region["shape"])
        if ramp:
            # Geometry 2's field (rw-mpas ``RampField``): the region's spacing inside its line, and
            # ``h * (1 + s / (1.5 W))`` at distance s outside it, the smallest of any region winning,
            # capped at the background.  With an onset the first ``onset x W`` is a parabola.
            fine = float(region["spacing_km"])
            doubling = RAMP_WIDTHS_PER_DOUBLING * width
            far = ramp_reach_km(fine, float(spec["background_km"]), width, onset)
            near = np.flatnonzero(xyz @ mesh_shape._xyz(lat0, lon0)
                                  >= math.cos(min(math.pi, (radius + far) / EARTH_RADIUS_KM)))
            if near.size == 0:
                continue
            s_out = np.maximum(signed_distance_km(region["shape"], lat[near], lon[near]), 0.0)
            bend = onset * width
            if bend > 0.0:
                allowed = np.where(s_out < bend, fine * (1.0 + s_out * s_out / (2.0 * bend * doubling)),
                                   fine * (1.0 + (s_out - 0.5 * bend) / doubling))
            else:
                allowed = fine * (1.0 + s_out / doubling)
            inv[near] = np.maximum(inv[near], 1.0 / allowed)
            continue
        # Past 8 ramp scales the blend is under 1e-7 of the step and moves no count.
        near = np.flatnonzero(xyz @ mesh_shape._xyz(lat0, lon0)
                              >= math.cos(min(math.pi, (radius + 8.0 * width) / EARTH_RADIUS_KM)))
        if near.size == 0:
            continue
        distance = signed_distance_km(region["shape"], lat[near], lon[near])
        here = back_inv + (fine_inv - back_inv) * 0.5 * (1.0 - np.tanh(distance / width))
        inv[near] = np.maximum(inv[near], here)
    spacing = 1.0 / inv
    area = 4.0 * math.pi * EARTH_RADIUS_KM ** 2 / samples
    columns = area / (HEXAGON_AREA_FACTOR * spacing ** 2)
    owner = np.full(samples, -1)
    zones = asked["zones"]
    # Finest last, so a nested zone takes its own cells from the zone that holds it.
    for index in sorted(range(len(zones)), key=lambda i: -zones[i]["fine_m"]):
        (lat0, lon0), radius = _reach(zones[index]["shape"])
        near = np.flatnonzero(xyz @ mesh_shape._xyz(lat0, lon0) >= math.cos(min(math.pi, radius / EARTH_RADIUS_KM)))
        if near.size:
            inside = signed_distance_km(zones[index]["shape"], lat[near], lon[near]) <= 0.0
            owner[near[inside]] = index
    kept = np.ones(samples, dtype=bool)
    cut = asked.get("cut")
    out: dict[str, Any] = {"basis": "area integral of the generator's field", "samples": samples}
    if cut is not None:
        (lat0, lon0), radius = _reach(cut["region"])
        rim_spacing = None
        near = np.flatnonzero(xyz @ mesh_shape._xyz(lat0, lon0)
                              >= math.cos(min(math.pi, (radius + 600.0) / EARTH_RADIUS_KM)))
        distance = np.full(samples, np.inf)
        distance[near] = signed_distance_km(cut["region"], lat[near], lon[near])
        edge = np.abs(distance) < 25.0
        rim_spacing = float(np.median(spacing[edge])) if edge.any() else float(spec["background_km"])
        kept = distance <= cut["boundary_rings"] * rim_spacing
        out["spacing_at_cut_km"] = round(rim_spacing, 3)
        out["columns_in_boundary_rings"] = int(round(float(columns[kept & (distance > 0.0)].sum())))
    total = float(columns[kept].sum())
    out["columns"] = int(round(total))
    out["zones"] = []
    for index, zone in enumerate(zones):
        mine = kept & (owner == index)
        out["zones"].append({"name": zone["name"], "fine_m": zone["fine_m"],
                             "columns": int(round(float(columns[mine].sum()))),
                             "median_spacing_km": round(float(np.median(spacing[mine])), 3) if mine.any() else None})
    out["columns_in_rings_and_background"] = int(round(float(columns[kept & (owner < 0)].sum())))
    order = np.argsort(spacing[kept])
    weight = np.cumsum(columns[kept][order])
    weight /= weight[-1]
    sorted_spacing = spacing[kept][order]
    out["spacing_km_by_column"] = {
        name: round(float(sorted_spacing[min(len(sorted_spacing) - 1, int(np.searchsorted(weight, q)))]), 3)
        for name, q in (("min", 0.0), ("p01", 0.01), ("p25", 0.25), ("p50", 0.5), ("p75", 0.75), ("p99", 0.99),
                        ("max", 1.0))}
    return out


def plan(row: Mapping[str, Any], *, engine: Any = None, cards: Sequence[str] | None = None,
         suites: Sequence[tuple[str, int]] | None = None, samples: int = CENSUS_SAMPLES,
         geometry: int | None = None, ramp_onset: float | None = None) -> dict[str, Any]:
    """Everything a caller needs to size and price a row, before a cell is generated."""

    from . import card_fit, mesh_point
    from .swath.sizing import dry_run

    asked = expand(row, geometry=geometry, ramp_onset=ramp_onset)
    spec = asked["spec"]
    generator = asked.get("generator") or {"geometry": 1, "words": []}
    ramp = int(generator["geometry"]) == 2
    # Geometry 2 is counted by the generator under the ramp it will build (its own --dry-run with
    # the word), never under geometry 1's steps: that count was 8 to 30 % low (18 Nov 2026).
    sizing = dry_run(spec, engine=engine, generator_words=generator["words"]) if ramp else dry_run(spec, engine=engine)
    counted = census(asked, samples=samples)
    globe = asked["cut"] is None
    whole = float(sizing["predicted_cells"])
    # A globe is priced on the generator's own count; a region on the cut, scaled by how the census
    # and the generator agree on the whole sphere.
    if globe:
        columns = whole
    else:
        sphere = census({**asked, "cut": None}, samples=samples)["columns"]
        columns = counted["columns"] * whole / max(sphere, 1)
    fine_m = asked["anchor_fine_km"] * 1000.0
    step, runs, memory = price(columns, fine_m, ramp=ramp, cards=cards, suites=suites)
    planned = _planned(asked, spec, globe, columns, whole, counted, step, memory, sizing)
    planned["forecast_step"] = runs
    if ramp:
        planned["generator"] = dict(generator)
        for target, zone in zip(planned["zones"], asked["zones"]):
            target["ramp"] = zone["ramp"]
    return planned


def price(columns: float, fine_m: float, *, ramp: bool, cards: Sequence[str] | None = None,
          suites: Sequence[tuple[str, int]] | None = None) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """What a mesh of ``columns`` at ``fine_m`` costs to run: the step the mesh admits, the step each
    suite's forecast will run on it, and the card memory for 1, 2, 4 and 8 cards.

    The mesh's own step is the doors' rule on the smallest edge a build of this geometry holds.  The
    step a forecast RUNS can differ: hrrrv5 takes NOAA's deck step (20 s near 3 km) where the mesh's
    limit admits it (``physics_default.forecast_step``), and a plan that prices 18 s steps for it
    counts 11 % too many.
    """

    from . import card_fit, mesh_point, physics_default

    edge_share = SMALLEST_EDGE_OVER_FINE_RAMP if ramp else SMALLEST_EDGE_OVER_FINE
    try:
        step = mesh_point.choose_timestep(edge_share * fine_m, fine_dx_m=fine_m)
        step = {"dt_seconds": step["dt_seconds"], "rule": step["rule"],
                "courant_limit_seconds": round(float(step["courant_limit_seconds"]), 2),
                "smallest_edge_assumed_m": round(edge_share * fine_m, 1),
                "basis": (f"smallest edge taken as {edge_share:g} x the fine spacing, the floor geometry 2 holds"
                          if ramp else
                          f"smallest edge taken as {edge_share:g} x the fine spacing; the build measures it")}
    except MpasPortError as error:
        raise MeshAskRefusal(f"no time step is admitted at {fine_m:g} m: {error}") from error
    runs: dict[str, Any] = {}
    memory: dict[str, Any] = {}
    for suite, levels in (suites or PLAN_SUITES):
        try:
            run = physics_default.forecast_step({"suite": suite}, float(step["dt_seconds"]), fine_m,
                                                float(step["courant_limit_seconds"]))
        except MpasPortError as error:
            run = {"dt_seconds": None, "source": str(error)}
        runs[suite] = run
    for card_key in (cards or PLAN_CARDS):
        card = mesh_point.resolve_card(card_key)
        for suite, levels in (suites or PLAN_SUITES):
            run_dt = runs[suite].get("dt_seconds") or step["dt_seconds"]
            try:
                fit = card_fit.by_card_count(columns, card=card, levels=levels, dt_seconds=float(run_dt),
                                             hours=1.0, suite=suite)
            except MpasPortError as error:
                memory[f"{card_key}/{suite}"] = {"refused": str(error)}
                continue
            rows = {}
            for entry in fit.get("by_cards") or ():
                count = int(entry["cards"])
                if count in PLAN_CARD_COUNTS:
                    rows[str(count)] = {"peak_mib_per_card": round(float(entry["predicted_peak_mib"])),
                                        "required_free_mib": round(float(entry["required_free_mib"])),
                                        "fits": bool(entry.get("fits"))}
            memory[f"{card_key}/{suite}"] = {"levels": levels, "fewest_cards_that_fit": fit.get("fewest_cards_that_fit"),
                                             "cards": rows}
    return step, runs, memory


def price_lines(step: Mapping[str, Any], runs: Mapping[str, Any], memory: Mapping[str, Any]) -> list[str]:
    """The step and memory lines of a plan."""

    lines = [f"  time step        {step['dt_seconds']:g} s admitted (limit {step['courant_limit_seconds']:g} s; "
             f"{step['basis']})"]
    for suite, run in (runs or {}).items():
        if run.get("dt_seconds") is None:
            lines.append(f"  step {suite:<26} {run.get('source')}")
            continue
        margin = f", {run['margin']:.2f}x inside the limit" if "margin" in run else ""
        lines.append(f"  step {suite:<26} the forecast runs {float(run['dt_seconds']):g} s: {run['source']}{margin}")
    for key, entry in memory.items():
        if "refused" in entry:
            lines.append(f"  memory {key:<28} not priced: {entry['refused']}")
            continue
        cells = ", ".join(f"{count} card{'s' if count != '1' else ''} {row['peak_mib_per_card']:,} MiB"
                          f"{'' if row['fits'] else ' (no)'}" for count, row in entry["cards"].items())
        fewest = entry["fewest_cards_that_fit"]
        lines.append(f"  memory {key:<28} {cells}; fewest that fit: {fewest if fewest else 'more than 8'}")
    return lines


def _planned(asked, spec, globe, columns, whole, counted, step, memory, sizing) -> dict[str, Any]:
    return {
        "schema": "gpuwm-hex.mesh-ask-plan/v1",
        "name": spec["name"],
        "domain": "globe" if globe else "region",
        "columns": int(round(columns)),
        "columns_whole_sphere": int(round(whole)),
        "columns_basis": ("the generator's own dry run" if globe else
                          "the generator's dry run of the whole sphere, times the share of it inside the cut"),
        "census": counted,
        "background": asked["background"],
        "zones": [{k: zone[k] for k in ("name", "asked_fine_m", "fine_m", "inside", "area_km2", "rings", "grading")}
                  for zone in asked["zones"]],
        "cut": None if globe else {k: asked["cut"][k] for k in ("around", "rim_km", "rim_spacing_km", "boundary_rings")},
        "notes": list(asked["notes"]),
        "step": step,
        "memory": memory,
        "steepest_gradient_percent_per_cell": sizing.get("steepest_requested_gradient_percent_per_cell"),
        "gates_applied_by_hexcore": sizing.get("gates_applied_by_hexcore"),
        "ladder_snap": sizing.get("ladder_snap"),
        "spec": spec,
        "cull_region": None if globe else asked["cut"]["region"],
    }


def report(planned: Mapping[str, Any]) -> list[str]:
    """The plan in plain lines."""

    counted = planned["census"]
    spacing = counted["spacing_km_by_column"]
    lines = [f"mesh-plan --spec (a mesh row): {planned['name']}"]
    ramp = int((planned.get("generator") or {}).get("geometry", 1)) == 2
    if ramp:
        lines.append("  generator        geometry 2: every spacing built as asked, one growth rate per cell, "
                     "no obtuse triangle (" + " ".join(planned["generator"]["words"]) + ")")
    for note in planned["notes"]:
        lines.append(f"  NOTE             {note}")
    lines.append(f"  columns          {planned['columns']:,} in all ({planned['columns_basis']})")
    for zone in counted["zones"]:
        asked = next(z for z in planned["zones"] if z["name"] == zone["name"])
        inside = f", inside {asked['inside']}" if asked["inside"] else ""
        lines.append(f"    zone {zone['name']:<12} {zone['columns']:>12,} at {zone['fine_m']:g} m"
                     f" over {asked['area_km2']:,.0f} km2{inside}")
    lines.append(f"    rings, background {counted['columns_in_rings_and_background']:>9,}"
                 + (f" (of them {counted['columns_in_boundary_rings']:,} in the boundary rings)"
                    if "columns_in_boundary_rings" in counted else ""))
    if planned["cut"] is not None:
        cut = planned["cut"]
        lines.append(f"  region           cut {cut['rim_km']:g} km outside zone {cut['around']}, "
                     f"{cut['boundary_rings']} boundary rings on {counted['spacing_at_cut_km']:g} km cells")
    else:
        lines.append(f"  globe            whole sphere, no rim; background {planned['background']['built_km']:g} km")
    lines.append("  spacing by column  " + ", ".join(f"{name} {spacing[name]:g} km"
                                                    for name in ("min", "p01", "p25", "p50", "p75", "p99", "max")))
    for zone in planned["zones"]:
        if ramp:
            grown_to = zone["ramp"]
            lines.append(f"  ramp of {zone['name']:<9} {zone['fine_m'] / 1000.0:g} km held to the outline, then "
                         f"+{100.0 * grown_to['growth_per_cell']:.1f} % a cell, reaching {grown_to['to_km']:g} km "
                         f"{grown_to['reaches_it_km_beyond_outline']:,.0f} km outside it")
            continue
        rings = " > ".join(f"{ring['spacing_km']:g} km to +{ring['edge_beyond_outline_km']:,.0f} km" for ring in zone["rings"])
        grading = zone["grading"]
        lines.append(f"  rings of {zone['name']:<8} {rings}; ramp scale {grading['ramp_scale_cells']:g} cells "
                     f"(steepest growth asked {100.0 * grading['steepest_growth_per_cell']:.1f} % a cell), "
                     f"flat ring {max(grading['ring_cells'], 0.0):g} cells")
    lines.extend(price_lines(planned["step"], planned.get("forecast_step") or {}, planned["memory"]))
    gradient = planned.get("steepest_gradient_percent_per_cell")
    if gradient is not None:
        lines.append(f"  generator        steepest growth in its own field {float(gradient):.1f} % a cell")
    return lines
