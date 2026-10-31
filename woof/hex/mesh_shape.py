"""Drawn regions for the regional door: a polygon or an ellipse instead of a circle.

WHY THIS EXISTS.  ``rw_mpas_mesh`` refines and cuts any Shape row (cap,
lat_lon_box, polygon), but ``mesh-plan --point`` and ``woof hex regional``
only ever wrote caps, so a user who wanted Oklahoma (or a tilted ellipse along
a front) got the circle that encloses it.  This module turns a drawn shape into
the same three things the circle door builds from a point and a radius:

* the halving ladder (:func:`shape_ladder_spec`): the drawn outline GROWN by
  one ramp of the fine rung is the fine region, and each coarser rung is that
  outline grown by the same ring offsets the cap ladder adds to its radius;
* the limited-area cut (:func:`shape_cull_region`): the fine region grown by
  the shipped pad (``(pad_scale - 1) x`` the equal-area radius, the circle's
  ``1.35 r`` rule restated for a shape);
* the halo the geography is built on (:func:`shape_halo_region`).

WHY THE FINE REGION IS THE DRAWN OUTLINE GROWN BY ONE RAMP.  A region's
spacing is a ``tanh`` ramp centred on its boundary, so the requested spacing is
only reached about one ramp width INSIDE the line (shapes study, 2026-10-23:
3.75 km is reached 67.5 km inside, and the 55 km wide Oklahoma panhandle got 5
to 6 km cells).  Growing the outline by one ramp puts the fine spacing (within
6 %) on everything the user outlined.

HOW AN OUTLINE IS GROWN.  Exactly, on the sphere, with no geometry library:
the drawn boundary is densified along its great circles, the geodesic
distance to it is read on an azimuthal-equidistant grid through a k-d tree on
every core, the region is ``distance <= d`` with its holes filled (the drawn
interior), and its outer edge is traced by marching squares and simplified to
the tolerance the rung needs.  The grown outline is therefore the true offset
set to within a grid cell, for convex and non-convex outlines alike.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .errors import MpasPortError

EARTH_RADIUS_KM = 6371.229
HEXAGON_AREA_FACTOR = math.sqrt(3.0) / 2.0
#: Vertices a region row may carry.  The generator evaluates every edge for
#: every probe and every relaxation sweep, so a row is simplified to the
#: tolerance its rung needs and then capped (the shapes study's rows).
MAX_REGION_VERTICES = 180
#: Grid cells per axis the offset raster may use, at most.
MAX_GRID = 1200
#: Vertices of an ellipse before any growing; dense, because it is only read
#: as a distance field.
ELLIPSE_VERTICES = 360


class ShapeRefusal(MpasPortError):
    """A drawn region is refused, and the message names what would break."""


# ---------------------------------------------------------------------------
# sphere and projections
# ---------------------------------------------------------------------------
def _xyz(lat_deg: Any, lon_deg: Any) -> np.ndarray:
    lat = np.radians(np.asarray(lat_deg, dtype=float))
    lon = np.radians(np.asarray(lon_deg, dtype=float))
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)


def _latlon(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xyz = np.asarray(xyz, dtype=float)
    norm = np.linalg.norm(xyz, axis=-1)
    lat = np.degrees(np.arcsin(np.clip(xyz[..., 2] / norm, -1.0, 1.0)))
    lon = np.degrees(np.arctan2(xyz[..., 1], xyz[..., 0]))
    return lat, lon


class _Azimuthal:
    """Azimuthal equidistant (``equal_area=False``) or Lambert equal-area
    projection of the sphere about a centre, in km."""

    def __init__(self, lat0: float, lon0: float, *, equal_area: bool = False) -> None:
        self.phi0 = math.radians(lat0)
        self.lam0 = math.radians(lon0)
        self.equal_area = equal_area

    def forward(self, lat_deg: Any, lon_deg: Any) -> tuple[np.ndarray, np.ndarray]:
        phi = np.radians(np.asarray(lat_deg, dtype=float))
        dlam = np.radians(np.asarray(lon_deg, dtype=float)) - self.lam0
        cos_c = np.clip(math.sin(self.phi0) * np.sin(phi)
                        + math.cos(self.phi0) * np.cos(phi) * np.cos(dlam), -1.0, 1.0)
        c = np.arccos(cos_c)
        if self.equal_area:
            k = np.sqrt(2.0 / np.maximum(1.0 + cos_c, 1e-300))
        else:
            k = np.where(c > 1e-12, c / np.where(c > 1e-12, np.sin(c), 1.0), 1.0)
        x = EARTH_RADIUS_KM * k * np.cos(phi) * np.sin(dlam)
        y = EARTH_RADIUS_KM * k * (math.cos(self.phi0) * np.sin(phi)
                                   - math.sin(self.phi0) * np.cos(phi) * np.cos(dlam))
        return x, y

    def inverse(self, x: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        rho = np.hypot(x, y)
        if self.equal_area:
            c = 2.0 * np.arcsin(np.clip(rho / (2.0 * EARTH_RADIUS_KM), -1.0, 1.0))
        else:
            c = rho / EARTH_RADIUS_KM
        safe = np.where(rho > 0.0, rho, 1.0)
        phi = np.arcsin(np.clip(np.cos(c) * math.sin(self.phi0)
                                + y * np.sin(c) * math.cos(self.phi0) / safe, -1.0, 1.0))
        lam = self.lam0 + np.arctan2(
            x * np.sin(c), rho * math.cos(self.phi0) * np.cos(c) - y * math.sin(self.phi0) * np.sin(c))
        phi = np.where(rho > 0.0, phi, self.phi0)
        lam = np.where(rho > 0.0, lam, self.lam0)
        lon = (np.degrees(lam) + 180.0) % 360.0 - 180.0
        return np.degrees(phi), lon


# ---------------------------------------------------------------------------
# rings
# ---------------------------------------------------------------------------
def _ring(row: Mapping[str, Any]) -> np.ndarray:
    """A polygon row's vertices as ``(n, 2)`` [lat, lon] degrees, no repeat."""

    vertices = np.asarray(row["vertices_deg"], dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 2:
        raise ShapeRefusal("a polygon's vertices_deg is a list of [lat, lon] pairs")
    if len(vertices) > 1 and np.allclose(vertices[0], vertices[-1]):
        vertices = vertices[:-1]
    if len(vertices) < 3:
        raise ShapeRefusal(f"a polygon of {len(vertices)} vertices encloses no area to refine")
    return vertices


def centroid(row: Mapping[str, Any]) -> tuple[float, float]:
    """The normalised vertex centroid, the point the generator itself uses
    as a polygon's representative interior point."""

    acc = _xyz(*_ring(row).T).sum(axis=0)
    norm = float(np.linalg.norm(acc))
    if norm <= 0.0:
        raise ShapeRefusal("the drawn outline's vertices cancel to the centre of the sphere")
    lat, lon = _latlon(acc / norm)
    return float(lat), float(lon)


def _densify(ring_xyz: np.ndarray, step_rad: float) -> np.ndarray:
    """The closed ring walked along its great circles at ``step_rad``."""

    out = []
    n = len(ring_xyz)
    for k in range(n):
        a, b = ring_xyz[k], ring_xyz[(k + 1) % n]
        angle = math.acos(max(-1.0, min(1.0, float(a @ b))))
        pieces = max(1, int(math.ceil(angle / step_rad)))
        if angle < 1e-15:
            out.append(a[None, :])
            continue
        t = np.arange(pieces) / pieces
        sin_angle = math.sin(angle)
        wa = np.sin((1.0 - t) * angle) / sin_angle
        wb = np.sin(t * angle) / sin_angle
        out.append(wa[:, None] * a[None, :] + wb[:, None] * b[None, :])
    return np.concatenate(out, axis=0)


def _check_size(row: Mapping[str, Any]) -> tuple[float, float]:
    lat0, lon0 = centroid(row)
    c = _xyz(lat0, lon0)
    reach = float(np.degrees(np.arccos(np.clip(_xyz(*_ring(row).T) @ c, -1.0, 1.0))).max())
    if reach >= 60.0:
        raise ShapeRefusal(
            f"the drawn outline reaches {reach:.1f} deg from its centre; the regional door "
            f"builds a limited-area domain, and a region that size is a global variable-"
            f"resolution mesh (woof hex mesh-plan --global-mesh --spec)")
    return lat0, lon0


def area_km2(row: Mapping[str, Any]) -> float:
    """Area enclosed by the outline (Lambert equal-area at its centroid)."""

    lat0, lon0 = centroid(row)
    proj = _Azimuthal(lat0, lon0, equal_area=True)
    dense = _densify(_xyz(*_ring(row).T), 10.0 / EARTH_RADIUS_KM)
    lat, lon = _latlon(dense)
    x, y = proj.forward(lat, lon)
    return float(abs(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))


def equal_area_radius_km(row: Mapping[str, Any]) -> float:
    """The radius of the spherical cap of the same area."""

    area = area_km2(row)
    cos_r = 1.0 - area / (2.0 * math.pi * EARTH_RADIUS_KM ** 2)
    return EARTH_RADIUS_KM * math.acos(max(-1.0, min(1.0, cos_r)))


# ---------------------------------------------------------------------------
# input shapes
# ---------------------------------------------------------------------------
def ellipse_row(lat: float, lon: float, semi_major_km: float, semi_minor_km: float,
                azimuth_deg: float, *, vertices: int = ELLIPSE_VERTICES) -> dict[str, Any]:
    """An ellipse on the azimuthal-equidistant plane about its centre; the
    major axis points ``azimuth_deg`` clockwise from north."""

    if not (semi_major_km > 0.0 and semi_minor_km > 0.0):
        raise ShapeRefusal(f"ellipse semi-axes {semi_major_km} x {semi_minor_km} km enclose nothing")
    az = math.radians(azimuth_deg)
    major = np.array([math.sin(az), math.cos(az)])
    minor = np.array([math.cos(az), -math.sin(az)])
    t = 2.0 * math.pi * np.arange(vertices) / vertices
    pts = (semi_major_km * np.cos(t))[:, None] * major + (semi_minor_km * np.sin(t))[:, None] * minor
    plat, plon = _Azimuthal(lat, lon).inverse(pts[:, 0], pts[:, 1])
    return {"kind": "polygon", "vertices_deg": [[float(a), float(b)] for a, b in zip(plat, plon)]}


def parse_ellipse(text: str) -> dict[str, Any]:
    parts = [piece.strip() for piece in str(text).split(",")]
    if len(parts) not in (4, 5):
        raise ShapeRefusal(
            f"--ellipse {text!r} is not LAT,LON,SEMI_MAJOR_KM,SEMI_MINOR_KM[,AZIMUTH_DEG]")
    try:
        values = [float(piece) for piece in parts]
    except ValueError as error:
        raise ShapeRefusal(f"--ellipse {text!r} is not numbers: {error}") from error
    lat, lon, a, b = values[:4]
    azimuth = values[4] if len(values) == 5 else 0.0
    if not -90.0 <= lat <= 90.0:
        raise ShapeRefusal(f"--ellipse latitude {lat} is outside -90..90")
    return ellipse_row(lat, lon, max(a, b), min(a, b), azimuth if a >= b else azimuth + 90.0)


def _box_polygon(lat_deg: Sequence[float], lon_deg: Sequence[float], per_side: int = 64) -> dict[str, Any]:
    lat0, lat1 = sorted(float(v) for v in lat_deg)
    lon0 = float(lon_deg[0])
    span = (float(lon_deg[1]) - lon0) % 360.0
    if span == 0.0:
        raise ShapeRefusal("lat_lon_box longitude span is zero")
    s = np.arange(per_side) / per_side
    lats = np.concatenate([np.full(per_side, lat0), lat0 + (lat1 - lat0) * s,
                           np.full(per_side, lat1), lat1 - (lat1 - lat0) * s])
    lons = np.concatenate([lon0 + span * s, np.full(per_side, lon0 + span),
                           lon0 + span * (1.0 - s), np.full(per_side, lon0)])
    lons = (lons + 180.0) % 360.0 - 180.0
    return {"kind": "polygon", "vertices_deg": [[float(a), float(b)] for a, b in zip(lats, lons)]}


def _geojson_polygon(document: Mapping[str, Any]) -> dict[str, Any]:
    kind = document.get("type")
    if kind == "FeatureCollection":
        rings = [_geojson_polygon(feature) for feature in document.get("features", ())
                 if (feature.get("geometry") or {}).get("type") in ("Polygon", "MultiPolygon")]
        if not rings:
            raise ShapeRefusal("the GeoJSON FeatureCollection holds no Polygon or MultiPolygon")
        return max(rings, key=area_km2)
    if kind == "Feature":
        return _geojson_polygon(document.get("geometry") or {})
    if kind == "Polygon":
        candidates = [document["coordinates"][0]]
    elif kind == "MultiPolygon":
        candidates = [polygon[0] for polygon in document["coordinates"]]
    else:
        raise ShapeRefusal(f"GeoJSON type {kind!r} is not a Polygon, MultiPolygon, Feature or "
                           f"FeatureCollection")
    rows = []
    for ring in candidates:
        raw = {"vertices_deg": [[float(p[1]), float(p[0])] for p in ring]}
        rows.append({"kind": "polygon", "vertices_deg": [[float(a), float(b)] for a, b in _ring(raw)]})
    return max(rows, key=area_km2)


def load_shape(path: Path) -> dict[str, Any]:
    """A drawn region from a file: a Shape row (the rows ``rw_mpas_mesh``
    reads: cap, lat_lon_box, polygon with [lat, lon] vertices) or GeoJSON
    (Polygon, MultiPolygon, Feature, FeatureCollection; [lon, lat]
    coordinates, the largest outer ring is the region)."""

    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ShapeRefusal(f"--shape {path}: cannot read a JSON shape ({error})") from error
    return shape_from_document(document)


def shape_from_document(document: Mapping[str, Any]) -> dict[str, Any]:
    kind = document.get("kind")
    if kind == "cap":
        return {"kind": "cap", "center_deg": [float(v) for v in document["center_deg"]],
                "radius_km": float(document["radius_km"])}
    if kind == "polygon":
        row = {"kind": "polygon",
               "vertices_deg": [[float(v[0]), float(v[1])] for v in _ring(document)]}
    elif kind == "lat_lon_box":
        row = _box_polygon(document["lat_deg"], document["lon_deg"])
    elif "type" in document:
        row = _geojson_polygon(document)
    else:
        raise ShapeRefusal(
            "the shape is neither a Shape row (kind cap, lat_lon_box or polygon) nor GeoJSON")
    _check_size(row)
    return row


# ---------------------------------------------------------------------------
# growing an outline
# ---------------------------------------------------------------------------
def _distance_grid(row: Mapping[str, Any], reach_km: float, cell_km: float, *, stretched: bool = False):
    """Geodesic distance (km) from an azimuthal-equidistant grid to the
    drawn boundary, with the grid's projection and axes.

    ``stretched`` sizes the margin for the projection's own stretch.  The
    azimuthal-equidistant scale is 1 along a radius but ``c / sin(c)`` along a
    circle about the centre (``c`` the angular distance), so a point ``reach``
    km from the boundary can project more than ``reach`` km from the
    boundary's projected box.  Every point of a geodesic of length ``reach``
    from a boundary point lies within ``c_b + reach`` of the centre, so it
    projects at most ``reach * c_max / sin(c_max)`` away, and no point within
    ``reach`` of the boundary projects farther than ``r_b + reach`` from the
    centre (``r_b`` the boundary's largest projected radius).
    """

    from scipy.spatial import cKDTree

    lat0, lon0 = _check_size(row)
    proj = _Azimuthal(lat0, lon0)
    ring = _xyz(*_ring(row).T)
    dense = _densify(ring, 0.5 * cell_km / EARTH_RADIUS_KM)
    blat, blon = _latlon(dense)
    bx, by = proj.forward(blat, blon)
    margin = reach_km + 4.0 * cell_km
    x0, x1 = float(bx.min()) - margin, float(bx.max()) + margin
    y0, y1 = float(by.min()) - margin, float(by.max()) + margin
    if stretched:
        rb = float(np.hypot(bx, by).max())
        c_max = (rb + reach_km) / EARTH_RADIUS_KM
        stretch = c_max / math.sin(c_max) if 1e-12 < c_max < math.pi - 1e-6 else (1.0 if c_max <= 1e-12 else math.inf)
        margin = reach_km * stretch + 4.0 * cell_km
        limit = rb + reach_km + 4.0 * cell_km
        x0, x1 = max(float(bx.min()) - margin, -limit), min(float(bx.max()) + margin, limit)
        y0, y1 = max(float(by.min()) - margin, -limit), min(float(by.max()) + margin, limit)
    cell = max(cell_km, (x1 - x0) / MAX_GRID, (y1 - y0) / MAX_GRID)
    xs = np.arange(x0, x1 + cell, cell)
    ys = np.arange(y0, y1 + cell, cell)
    gx, gy = np.meshgrid(xs, ys)
    glat, glon = proj.inverse(gx.ravel(), gy.ravel())
    chord, _ = cKDTree(dense).query(_xyz(glat, glon), workers=-1)
    dist = 2.0 * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0)) * EARTH_RADIUS_KM
    return proj, xs, ys, dist.reshape(gy.shape), cell


def _march(field: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> list[np.ndarray]:
    """Closed loops of ``field == 0`` (inside is negative), by marching squares."""

    inside = field < 0.0
    corners = (inside[:-1, :-1].astype(int) + 2 * inside[:-1, 1:] + 4 * inside[1:, 1:]
               + 8 * inside[1:, :-1])
    cells = np.argwhere((corners != 0) & (corners != 15))

    def point(edge):
        kind, i, j = edge
        if kind == "h":   # (i, j) -> (i, j + 1)
            a, b = field[i, j], field[i, j + 1]
            t = a / (a - b)
            return (xs[j] + t * (xs[j + 1] - xs[j]), ys[i])
        a, b = field[i, j], field[i + 1, j]  # (i, j) -> (i + 1, j)
        t = a / (a - b)
        return (xs[j], ys[i] + t * (ys[i + 1] - ys[i]))

    links: dict[tuple, list[tuple]] = {}
    for i, j in cells:
        case = int(corners[i, j])
        bottom, right, top, left = ("h", i, j), ("v", i, j + 1), ("h", i + 1, j), ("v", i, j)
        if case in (5, 10):
            centre = 0.25 * (field[i, j] + field[i, j + 1] + field[i + 1, j + 1] + field[i + 1, j])
            a_and_c = case == 5
            if (centre < 0.0) == a_and_c:
                pairs = [(bottom, right), (top, left)]
            else:
                pairs = [(left, bottom), (right, top)]
        else:
            crossing = []
            if (case & 1) != ((case >> 1) & 1):
                crossing.append(bottom)
            if ((case >> 1) & 1) != ((case >> 2) & 1):
                crossing.append(right)
            if ((case >> 2) & 1) != ((case >> 3) & 1):
                crossing.append(top)
            if ((case >> 3) & 1) != (case & 1):
                crossing.append(left)
            pairs = [(crossing[0], crossing[1])]
        for a, b in pairs:
            links.setdefault(a, []).append(b)
            links.setdefault(b, []).append(a)

    loops = []
    seen: set[tuple] = set()
    for start in links:
        if start in seen:
            continue
        loop, previous, current = [], None, start
        while current not in seen:
            seen.add(current)
            loop.append(point(current))
            ahead = [e for e in links[current] if e != previous and e not in seen]
            if not ahead:
                break
            previous, current = current, ahead[0]
        if len(loop) >= 3:
            loops.append(np.asarray(loop))
    return loops


def _simplify(loop: np.ndarray, tolerance: float) -> np.ndarray:
    """Douglas-Peucker on a closed planar loop."""

    n = len(loop)
    if n <= 4:
        return loop
    far = int(np.argmax(np.hypot(*(loop - loop[0]).T)))
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[far] = True
    stack = [(0, far), (far, n)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        pa, pb = loop[a], loop[b % n]
        inner = loop[a + 1:b]
        d = pb - pa
        length = math.hypot(*d)
        if length == 0.0:
            dist = np.hypot(*(inner - pa).T)
        else:
            dist = np.abs(d[0] * (inner[:, 1] - pa[1]) - d[1] * (inner[:, 0] - pa[0])) / length
        k = int(np.argmax(dist))
        if dist[k] > tolerance:
            mid = a + 1 + k
            keep[mid] = True
            stack.extend([(a, mid), (mid, b)])
    return loop[keep]


def _touches_border(mask: np.ndarray) -> bool:
    return bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())


def grow(row: Mapping[str, Any], distance_km: float, *, tolerance_km: float,
         max_vertices: int = MAX_REGION_VERTICES) -> dict[str, Any]:
    """The outline of every point within ``distance_km`` of the drawn
    region, as a polygon row simplified to ``tolerance_km``."""

    if not distance_km > 0.0:
        raise ShapeRefusal(f"an outline is grown by a positive distance, not {distance_km} km")
    from scipy import ndimage

    asked = max(0.5 * tolerance_km, 0.25)
    proj, xs, ys, dist, cell = _distance_grid(row, distance_km, asked)
    near = dist <= distance_km
    # THE BREAKAGE A CLIPPED RASTER HAD: the grown band is connected, so a band that reaches the raster's border
    # has lost ground past it, and marching squares then traces open chains; the largest was closed by a chord
    # across the region. MEASURED 2026-10-29: `woof hex country --polygon` CONUS at 96 -> 3 km grew its 24 km
    # rung 1,188 km (stretched past the old reach + 4 cells box by the projection), and the rung lost its whole
    # western side (29 vertices, west edge -132.2 deg); the 12 km rung's ramp then fell straight to 48 km over the
    # Pacific, and rw_mpas_mesh refused the spec at 18.4 % per cell. A band inside the old box is whole, so it is
    # kept bit for bit; only a clipped one is re-rastered over the stretched box, and a band still clipped there
    # is refused rather than traced.
    if _touches_border(near):
        proj, xs, ys, dist, cell = _distance_grid(row, distance_km, asked, stretched=True)
        near = dist <= distance_km
        if _touches_border(near):
            raise ShapeRefusal(
                f"growing the outline by {distance_km:g} km reaches past its own raster even with the "
                f"projection's stretch allowed for; tracing it would close the clipped band with a chord "
                f"across the region")
    region = ndimage.binary_fill_holes(near)
    labels, count = ndimage.label(region)
    if count > 1:
        sizes = ndimage.sum(region, labels, range(1, count + 1))
        region = labels == (1 + int(np.argmax(sizes)))
    field = np.where(region, np.minimum(dist - distance_km, -1e-9), np.maximum(dist - distance_km, 1e-9))
    loops = _march(field, xs, ys)
    if not loops:
        raise ShapeRefusal("growing the drawn outline traced no boundary")
    loop = max(loops, key=lambda p: abs(0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1)
                                                       - np.roll(p[:, 0], -1) * p[:, 1])))
    tol = max(tolerance_km, 0.5 * cell)
    while True:
        simple = _simplify(loop, tol)
        if len(simple) <= max_vertices:
            break
        tol *= 1.5
    lat, lon = proj.inverse(simple[:, 0], simple[:, 1])
    return {"kind": "polygon",
            "vertices_deg": [[round(float(a), 5), round(float(b), 5)] for a, b in zip(lat, lon)]}


# ---------------------------------------------------------------------------
# the door's three regions
# ---------------------------------------------------------------------------
def shape_ladder_spec(drawn: Mapping[str, Any], *, rungs_km: Sequence[float],
                      background_km: float, transition_factor: float, ring_factor: float,
                      name: str) -> dict[str, Any]:
    """The halving ladder around a drawn region.

    The fine region is the drawn outline grown by one ramp of the fine rung
    (the module docstring says why); each coarser rung is that outline grown
    by the ring offsets :func:`woof.hex.mesh_point.ladder_spec` adds to a cap's
    radius, rung for rung.  Rows are simplified to half their own spacing
    (2 km at most for the fine one), as the shapes study's rows were.
    """

    regions: list[dict[str, Any]] = []
    fine = float(rungs_km[-1])
    offset = transition_factor * fine
    for spacing in reversed([float(r) for r in rungs_km]):
        transition = transition_factor * spacing
        tolerance = min(0.5 * spacing, 2.0) if spacing == fine else 0.5 * spacing
        regions.append({"shape": grow(drawn, offset, tolerance_km=tolerance),
                        "spacing_km": spacing, "transition_km": transition})
        offset += ring_factor * transition
    regions.reverse()
    return {"name": name, "background_km": float(background_km), "regions": regions}


def fine_region_growth_km(fine_km: float, transition_factor: float) -> float:
    return transition_factor * fine_km


def shape_cull_region(drawn: Mapping[str, Any], *, grown_km: float, pad_scale: float,
                      tolerance_km: float = 1.0) -> dict[str, Any]:
    """The cut: the fine region (the drawn outline grown by ``grown_km``)
    grown by ``(pad_scale - 1)`` equal-area radii of the drawn outline."""

    if pad_scale < 1.0:
        raise ShapeRefusal(f"--cull-pad-scale {pad_scale} is below 1.0; a cut inside the fine "
                           f"core hands the boundary rings fine cells the parent's ramp never reaches")
    pad = (float(pad_scale) - 1.0) * equal_area_radius_km(drawn)
    return grow(drawn, grown_km + pad, tolerance_km=tolerance_km)


def shape_halo_region(drawn: Mapping[str, Any], *, grown_km: float, pad_scale: float,
                      halo_km: float) -> dict[str, Any]:
    """The working halo: the cut grown by ``halo_km``."""

    if not halo_km > 0.0:
        raise ShapeRefusal(f"--halo-km {halo_km} is not positive")
    pad = (float(pad_scale) - 1.0) * equal_area_radius_km(drawn)
    return grow(drawn, grown_km + pad + halo_km, tolerance_km=5.0)


# ---------------------------------------------------------------------------
# pricing the cut
# ---------------------------------------------------------------------------
def _inside(row: Mapping[str, Any], proj: _Azimuthal, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Grid cells inside a polygon row, by scanline on its densified ring."""

    dense = _densify(_xyz(*_ring(row).T), 5.0 / EARTH_RADIUS_KM)
    lat, lon = _latlon(dense)
    px, py = proj.forward(lat, lon)
    qx, qy = np.roll(px, -1), np.roll(py, -1)
    mask = np.zeros((len(ys), len(xs)), dtype=bool)
    for i, y in enumerate(ys):
        crosses = (py <= y) != (qy <= y)
        if not crosses.any():
            continue
        t = (y - py[crosses]) / (qy[crosses] - py[crosses])
        xc = np.sort(px[crosses] + t * (qx[crosses] - px[crosses]))
        idx = np.searchsorted(xc, xs)
        mask[i] = (idx % 2) == 1
    return mask


def _boundary_distance(row: Mapping[str, Any], proj: _Azimuthal, gx: np.ndarray, gy: np.ndarray,
                       step_km: float) -> np.ndarray:
    from scipy.spatial import cKDTree

    dense = _densify(_xyz(*_ring(row).T), step_km / EARTH_RADIUS_KM)
    glat, glon = proj.inverse(gx, gy)
    chord, _ = cKDTree(dense).query(_xyz(glat, glon), workers=-1)
    return 2.0 * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0)) * EARTH_RADIUS_KM


def predicted_shape_cull_cells(*, spec: Mapping[str, Any], drawn: Mapping[str, Any],
                               cull: Mapping[str, Any], attained_fine_km: float | None = None,
                               boundary_rings: int = 7) -> dict[str, Any]:
    """Cells the cut of a drawn region holds: the spec's own spacing field
    (the generator's tanh blend of every region, in 1/h) integrated over the
    cut and its boundary rings on an equal-area grid.  ``basis:
    area_integral``, the circle door's basis, so the two are priced alike;
    the cull receipt replaces it with the count."""

    lat0, lon0 = centroid(cull)
    proj = _Azimuthal(lat0, lon0, equal_area=True)
    background = float(spec["background_km"])
    regions = list(spec["regions"])
    fine = min(float(r["spacing_km"]) for r in regions)
    cull_lat, cull_lon = _latlon(_xyz(*_ring(cull).T))
    cx, cy = proj.forward(cull_lat, cull_lon)
    # The rings are never wider than seven background cells.
    reach = boundary_rings * background
    cell = max(0.5 * fine, (max(cx.max() - cx.min(), cy.max() - cy.min()) + 2 * reach) / 900.0)
    reach += 2.0 * cell
    xs = np.arange(cx.min() - reach, cx.max() + reach + cell, cell)
    ys = np.arange(cy.min() - reach, cy.max() + reach + cell, cell)
    gx, gy = np.meshgrid(xs, ys)
    inv = np.full(gx.shape, 1.0 / background)
    back_inv = 1.0 / background
    for region in regions:
        spacing = float(region["spacing_km"])
        if attained_fine_km is not None and spacing == fine:
            spacing = float(attained_fine_km)
        width = float(region.get("transition_km") or float(region["transition_cells"]) * spacing)
        shape = region["shape"]
        dist = _boundary_distance(shape, proj, gx.ravel(), gy.ravel(), 0.5 * cell).reshape(gx.shape)
        signed = np.where(_inside(shape, proj, xs, ys), -dist, dist)
        blend = 0.5 * (1.0 - np.tanh(signed / width))
        inv = np.maximum(inv, back_inv + (1.0 / spacing - back_inv) * blend)
    h = 1.0 / inv
    in_cut = _inside(cull, proj, xs, ys)
    cut_dist = _boundary_distance(cull, proj, gx.ravel(), gy.ravel(), 0.5 * cell).reshape(gx.shape)
    on_rim = (~in_cut) & (cut_dist <= 1.5 * cell)
    spacing_at_cut = float(np.median(h[on_rim])) if on_rim.any() else background
    halo = boundary_rings * spacing_at_cut
    counted = in_cut | (cut_dist <= halo)
    per_cell = cell * cell / (HEXAGON_AREA_FACTOR * h * h)
    core = _inside(drawn, proj, xs, ys)
    return {
        "basis": "area_integral",
        "predicted_cells": round(float(per_cell[counted].sum()), 1),
        "core_cells": round(float(per_cell[core].sum()), 1),
        "core_area_km2": round(area_km2(drawn), 1),
        "core_equal_area_radius_km": round(equal_area_radius_km(drawn), 2),
        "spacing_at_cut_km": spacing_at_cut,
        "boundary_rings": boundary_rings,
        "halo_km": halo,
        "fine_spacing_on_drawn_area_km": {
            "max": round(float(h[core].max()), 3) if core.any() else None,
            "median": round(float(np.median(h[core])), 3) if core.any() else None,
        },
    }


__all__ = [
    "ShapeRefusal", "area_km2", "centroid", "ellipse_row", "equal_area_radius_km",
    "fine_region_growth_km", "grow", "load_shape", "parse_ellipse", "predicted_shape_cull_cells",
    "shape_cull_region", "shape_from_document", "shape_halo_region", "shape_ladder_spec",
]
