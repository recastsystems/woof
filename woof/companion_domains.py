"""Candidate-only domain edits using WOOF's parser and native grid geometry.

This endpoint authors ordinary configuration metadata. It does not run a
forecast, locate a storm, decode forcing, or implement a projection/tracker.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import ctypes
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tomllib

REQUEST_SCHEMA = "arwen.companion-domain-edit.v1"
RESULT_SCHEMA = "arwen.companion-domain-result.v1"


class DomainScopeError(ValueError):
    """A per-domain edit of a setting this configuration's route holds
    tree-wide, carrying the edit that publishes instead.

    ``all_domains_settings`` is the whole edit this door MEASURED as
    publishing -- the same settings at every domain, plus whatever the
    route's own namelists state for that selection -- or ``None`` where
    nothing it tried published.  It is measured through the renderer the
    save publishes through, never derived from the setting names, so a
    panel, a front end and a test read one answer instead of three, and
    the sentence cannot offer a way out that the save then refuses.
    """

    def __init__(self, message, all_domains_settings=None):
        super().__init__(message)
        self.all_domains_settings = (None if all_domains_settings is None
                                     else dict(all_domains_settings))

# Explicit example settings already shipped in this configuration. This is
# an editable preset, not a new detection algorithm or a model-wide default.
#
# The proposal ceiling uses the single-axis overlap limit: 30% of a
# 40-parent-cell nest is 12 cells. The relocation runner then clamps the
# actual vector to retain 70% overlap (six cells per axis on a diagonal).
# A fixed diagonal ceiling of six also constrained zonal motion, halving
# available reach. The MSLP eye is consulted and may move on every 15 min cadence.
VORTEX_PRESET_SOURCE = "configs/cyclone_nest_slots_12km.toml: domain 2 follow"
VORTEX_PRESET = {
    "field": "pressure", "threshold": 1100.0, "level_hpa": 0.0,
    "radius_km": 60.0, "search_margin_cells": 20, "min_shift_cells": 2,
    "max_shift_cells": 12, "cooldown_seconds": 900.0,
    "cadence_seconds": 900.0, "max_move_parent_cells": 12,
    "min_overlap_fraction": 0.7,
    # MSLP minimum, with the 850 hPa circulation as fallback when no
    # qualifying minimum is found (the parked trailer-tracking study).
    "pressure_center": "minimum", "fallback_level_hpa": 850.0,
    "fallback_deficit_m": 25.0,
}


def _exact_keys(value, allowed, required=(), *, where):
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be a JSON object")
    if set(value) - set(allowed):
        raise ValueError(f"{where} has unknown fields: {sorted(set(value) - set(allowed))}")
    if set(required) - set(value):
        raise ValueError(f"{where} is missing fields: {sorted(set(required) - set(value))}")


def _integer(value, name, *, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _utc(value):
    if not isinstance(value, str):
        raise ValueError("Target/start time must be an ISO UTC date and time")
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is not None:
        result = result.astimezone(timezone.utc).replace(tzinfo=None)
    return result


def _json(value):
    from woof.companion_query import _json_value
    return json.dumps(value, allow_nan=False, indent=2, default=_json_value) + "\n"


def _build(raw, path):
    from woof.experiment import build_experiment_from_config_tables
    return build_experiment_from_config_tables(raw, source=str(path), base_dir=path.parent)


def _domain_table(raw, grid_id):
    grid_id = _integer(grid_id, "grid_id")
    matches = [row for row in raw["domain"] if row["grid_id"] == grid_id]
    if len(matches) != 1:
        raise ValueError(f"Configuration has no unique domain d{grid_id:02}")
    return matches[0]


def _native_grids(exp):
    from woof.static import rust_bridge
    from woof.static.projection import grids_from_projection_config
    bridge = rust_bridge.route("companion domain editing")
    if bridge is None:
        raise ValueError("Domain editing requires the native static-fields bridge")
    return bridge, dict(zip((d.grid_id for d in exp.domains), grids_from_projection_config(exp)))


def _transform(bridge, grid, direction, points):
    """Marshal only; the existing native grid performs all coordinate math."""
    first = (ctypes.c_double * len(points))(*(p[0] for p in points))
    second = (ctypes.c_double * len(points))(*(p[1] for p in points))
    library = bridge.load()
    code = library.gpuwm_static_grid_transform(
        grid._rust_handle(bridge), direction, first, second, len(points))
    if code:
        raise ValueError(bridge.last_error(library))
    result = list(zip(first, second))
    if any(not math.isfinite(x) or not math.isfinite(y) for x, y in result):
        raise ValueError("Native projection cannot represent the requested coordinates")
    return result


def native_domain_outlines(exp):
    """Native perimeter coordinates with O(edge samples), not O(grid cells), storage."""
    bridge, grids = _native_grids(exp)
    result = []
    for domain in exp.domains:
        grid = grids[domain.grid_id]
        # Corner coordinates are the same 0.5..e_we-0.5 grid as latlon_c().
        xs = sorted({0, domain.run.nx, *range(0, domain.run.nx, max(1, domain.run.nx // 96))})
        ys = sorted({0, domain.run.ny, *range(0, domain.run.ny, max(1, domain.run.ny // 96))})
        points = ([(x + .5, .5) for x in xs]
                  + [(domain.run.nx + .5, y + .5) for y in ys[1:]]
                  + [(x + .5, domain.run.ny + .5) for x in reversed(xs[:-1])]
                  + [(.5, y + .5) for y in reversed(ys[1:-1])])
        perimeter = _transform(bridge, grid, 0, points)
        center = _transform(bridge, grid, 0, [(grid.e_we / 2, grid.e_sn / 2)])[0]
        result.append({"grid_id": domain.grid_id, "parent_id": domain.parent_id,
            "nx": domain.run.nx, "ny": domain.run.ny, "nz": domain.run.nz,
            "history_interval_s": domain.history_interval_s,
            "dx_m": domain.run.dx, "dy_m": domain.run.dy,
            "width_km": domain.run.nx * domain.run.dx / 1000,
            "height_km": domain.run.ny * domain.run.dy / 1000,
            "center_latlon": center, "perimeter_latlon": perimeter})
    return result


# RFC 7946 3.1.9: an exported ring may not carry an edge that crosses the
# antimeridian, and a Cartesian consumer reads such an edge as the 340-degree
# complement of a 20-degree domain. These helpers cut the ring at the seam.
# They shift a longitude only by whole turns and only when a step wraps, so a
# domain that stays inside one -180..180 strip exports the ring it always did.


def _seam_lift(points):
    """Ordered [lon, lat] pairs carried onto one continuous longitude branch.

    Each step follows the native short edge, which is what the perimeter
    sampler produced; the result is the domain's real extent instead of its
    complement. Longitudes are only ever shifted by an exact multiple of 360,
    and only after a step wraps, so an ordinary domain's values are unchanged.
    """
    lifted = []
    turns = 0
    for index, (lon, lat) in enumerate(points):
        value = lon if turns == 0 else lon + 360. * turns
        if index == 0:
            while value >= 180.:
                value -= 360.
                turns -= 1
            while value < -180.:
                value += 360.
                turns += 1
        else:
            previous = lifted[-1][0]
            while value - previous > 180.:
                value -= 360.
                turns -= 1
            while value - previous < -180.:
                value += 360.
                turns += 1
            if abs(value - previous) == 180.:
                raise ValueError(
                    "A perimeter step of exactly 180 degrees of longitude has two equally "
                    "short arcs, so the exported domain would have no defined interior; "
                    "move the domain off that antipodal edge or resize it so its sampled "
                    "edges are shorter than half a turn")
        lifted.append([value, lat])
    return lifted


def _seam_undoubled(points):
    """Drop steps of zero length, which is how +180 and -180 for one point meet."""
    result = []
    for point in points:
        if not result or result[-1] != point:
            result.append(point)
    return result


def _seam_turn(points):
    """Whole turns of longitude the closed cycle of a lifted ring makes.

    Zero for a ring that stays on one side of the world, and exactly one turn
    for a ring that encircles a pole: every step is a short arc, so the total
    around a closed cycle can only be a multiple of 360.
    """
    if len(points) < 2:
        return 0.
    closing = (points[0][0] - points[-1][0] + 180.) % 360. - 180.
    return points[-1][0] + closing - points[0][0]


def _seam_strips(lifted):
    """Indices k of the -180+360k..180+360k strips the lifted coordinates reach."""
    lons = [lon for lon, _ in lifted]
    low = math.floor((min(lons) + 180.) / 360.)
    high = math.ceil((max(lons) - 180.) / 360.)
    return range(int(low), int(high) + 1)


def _seam_cross(first, second, boundary):
    """The GeoJSON-linear point where an existing sampled edge meets a strip edge."""
    share = (boundary - first[0]) / (second[0] - first[0])
    return [boundary, first[1] + share * (second[1] - first[1])]


def _seam_clip_ring(ring, low, high):
    """The part of a lifted ring inside one strip, as a closed ring or nothing."""
    for boundary, inside in ((low, lambda lon: lon >= low), (high, lambda lon: lon <= high)):
        crossings = sum(1 for a, b in zip(ring, ring[1:] + ring[:1])
                        if inside(a[0]) != inside(b[0]))
        if crossings > 2:
            raise ValueError(
                "A domain perimeter that re-enters the antimeridian more than once cannot "
                "be cut into canonical parts by this exporter; resize the domain so its "
                "perimeter crosses the antimeridian at most once")
        clipped = []
        for a, b in zip(ring, ring[1:] + ring[:1]):
            if inside(a[0]):
                clipped.append(a)
                if not inside(b[0]):
                    clipped.append(_seam_cross(a, b, boundary))
            elif inside(b[0]):
                clipped.append(_seam_cross(a, b, boundary))
        ring = _seam_undoubled(clipped)
        if len(ring) > 1 and ring[0] == ring[-1]:
            ring.pop()
        if len(ring) < 3:
            return None
    return ring


def _seam_polygons(perimeter):
    """One closed ring per antimeridian-cut part, west to east, [lon, lat] order."""
    points = _seam_undoubled(_seam_lift([[lon, lat] for lat, lon in perimeter]))
    if len(points) > 1 and points[0] == points[-1]:
        points.pop()
    if len(points) < 3:
        raise ValueError(
            "A domain perimeter needs at least three distinct corners before it can be "
            "exported as an area; pass the outlines native_domain_outlines returned for "
            "this configuration rather than a hand-built domain record")
    if abs(_seam_turn(points)) >= 180.:
        raise ValueError(
            "A domain perimeter that encircles a pole spans a full turn of longitude, so "
            "it has no antimeridian crossing to cut: every part of such a cut keeps a "
            "world-spanning edge unless the exporter invents an edge through the pole "
            "itself, which would change the domain. This exporter is not the wall that "
            "matters most for such a domain: lat-lon source interpolation and static-tile "
            "windowing are not pole-capable, so a domain whose footprint encloses a pole "
            "is not a domain this pipeline can run, on any projection. The way out is "
            "that wall's: move the domain centre away from the pole, or shrink the "
            "domain, until its footprint clears the pole")
    parts = []
    for turn in _seam_strips(points):
        ring = _seam_clip_ring(points, -180. + 360. * turn, 180. + 360. * turn)
        if ring is None:
            continue
        if turn:
            ring = [[lon - 360. * turn, lat] for lon, lat in ring]
        parts.append([ring + [ring[0]]])
    return parts


def _seam_lines(track):
    """The lifted track split at every strip edge, in travel order."""
    lifted = _seam_lift(track)
    points = _seam_undoubled(lifted)
    if len(points) < 2:
        # A target that does not move between two scheduled times collapses to a
        # single position, and a LineString of one position is malformed (RFC 7946
        # 3.1.4), so the repeated position is carried through as it arrived.
        return [[[lon, lat] for lon, lat in lifted]] if len(lifted) >= 2 else []
    lines, turn = [], math.floor((points[0][0] + 180.) / 360.)
    current = [points[0]]
    for first, second in zip(points, points[1:]):
        step = math.floor((second[0] + 180.) / 360.) - turn
        for _ in range(abs(step)):
            boundary = 180. + 360. * turn if step > 0 else -180. + 360. * turn
            crossing = _seam_cross(first, second, boundary)
            current.append(crossing)
            lines.append(current if not turn else
                         [[lon - 360. * turn, lat] for lon, lat in current])
            turn += 1 if step > 0 else -1
            current = [[crossing[0], crossing[1]]]
        current.append(second)
    lines.append(current if not turn else
                 [[lon - 360. * turn, lat] for lon, lat in current])
    return [line for line in lines if len(_seam_undoubled(line)) >= 2]


def domain_geojson(domains, target_points=()):
    """MapLibre-ready metadata; coordinates remain the native grid's values.

    A domain that crosses the antimeridian is emitted as the two seam-cut parts
    of one feature (RFC 7946 3.1.9), keeping its own identifier and properties.
    A domain that does not cross it is emitted exactly as before.
    """
    features = []

    def polygon(domain, *, target_time=None):
        parts = _seam_polygons(domain["perimeter_latlon"])
        properties = {"kind": "domain" if target_time is None else "target_domain",
            "grid_id": domain["grid_id"], "parent_id": domain["parent_id"],
            "label": f"d{domain['grid_id']:02}", "dx_m": domain["dx_m"],
            "nx": domain["nx"], "ny": domain["ny"],
            "width_km": domain["width_km"], "height_km": domain["height_km"]}
        if target_time is not None:
            properties["target_time_utc"] = target_time
        geometry = ({"type": "Polygon", "coordinates": parts[0]} if len(parts) == 1
                    else {"type": "MultiPolygon", "coordinates": parts})
        features.append({"type": "Feature", "id": f"{properties['kind']}-{domain['grid_id']}-{target_time or 'initial'}",
            "properties": properties, "geometry": geometry})

    for domain in domains:
        polygon(domain)
    track = []
    for point in target_points:
        time = point["time"].isoformat() + "Z" if isinstance(point["time"], datetime) else str(point["time"])
        domain = point["outline"]
        polygon(domain, target_time=time)
        lat, lon = point["requested_latlon"]
        track.append([lon, lat])
        features.append({"type": "Feature", "id": f"target-{domain['grid_id']}-{time}",
            "properties": {"kind": "target_point", "grid_id": domain["grid_id"], "target_time_utc": time,
                           "i_parent_start": point["i_parent_start"], "j_parent_start": point["j_parent_start"]},
            "geometry": {"type": "Point", "coordinates": [lon, lat]}})
    if len(track) >= 2:
        lines = _seam_lines(track)
        features.append({"type": "Feature", "id": "scheduled-target-track",
            "properties": {"kind": "target_track", "grid_id": target_points[0]["outline"]["grid_id"]},
            "geometry": ({"type": "LineString", "coordinates": lines[0]} if len(lines) == 1
                         else {"type": "MultiLineString", "coordinates": lines})})
    return {"type": "FeatureCollection", "features": features}


def _placement(exp, grid_id, value):
    from woof.core.storm_tracking import _round_cells
    from woof.experiment import validate_spawn_placement

    _exact_keys(value, {"kind", "i_parent_start", "j_parent_start", "latitude", "longitude"},
                {"kind"}, where="placement")
    child = exp.domain(grid_id)
    if child.parent_id == 0:
        raise ValueError("A root domain has no parent placement; use Fit area for the root")
    if value["kind"] == "parent_cells":
        _exact_keys(value, {"kind", "i_parent_start", "j_parent_start"},
                    {"kind", "i_parent_start", "j_parent_start"}, where="parent-cell placement")
        i = _integer(value["i_parent_start"], "i_parent_start")
        j = _integer(value["j_parent_start"], "j_parent_start")
    elif value["kind"] == "center":
        _exact_keys(value, {"kind", "latitude", "longitude"},
                    {"kind", "latitude", "longitude"}, where="center placement")
        latitude = _number(value["latitude"], "latitude")
        longitude = _number(value["longitude"], "longitude")
        if not -90 <= latitude <= 90:
            raise ValueError("Target latitude must lie between -90 and 90 degrees")
        bridge, grids = _native_grids(exp)
        child_grid, parent_grid = grids[grid_id], grids[child.parent_id]
        current_center = _transform(bridge, child_grid, 0,
            [(child_grid.e_we / 2, child_grid.e_sn / 2)])[0]
        current, target = _transform(bridge, parent_grid, 1,
                                    [current_center, (latitude, longitude)])
        # The native registration supplies both centers. The tracker's existing
        # symmetric whole-cell rounding supplies the discrete placement choice.
        i = child.i_parent_start + _round_cells(target[0] - current[0])
        j = child.j_parent_start + _round_cells(target[1] - current[1])
    else:
        raise ValueError("Placement kind must be parent_cells or center")
    validate_spawn_placement(exp, grid_id, i, j)
    return i, j


def _clamp_start(start, parent_size, child_size, ratio, clearance):
    """Nearest legal WPS start; dimensions already satisfy parent containment."""
    return max(clearance + 1, min(start, parent_size - child_size // ratio - clearance + 1))


def _refresh_root_fetch(raw, exp):
    """Keep the existing acquisition source's geographic hint with the root."""
    from woof import domain_wizard as dw
    fetch = raw.get("fetch", {})
    source = fetch.get("source")
    if source and dw.source_fetch_takes_a_crop_box(source):
        root = next(domain for domain in exp.domains if domain.parent_id == 0)
        fetch["area"] = dw.fetch_area_hint(raw["projection"], root.run.nx, root.run.ny,
                                          source=source, root_dx_m=root.run.dx)
        for key in ("point", "radius_km"):
            fetch.pop(key, None)


def _move_domain(raw, action, output):
    from woof.core.storm_tracking import _round_cells
    _exact_keys(action, {"kind", "grid_id", "latitude", "longitude"},
                {"kind", "grid_id", "latitude", "longitude"}, where="move_domain")
    grid_id = _integer(action["grid_id"], "grid_id")
    latitude = _number(action["latitude"], "latitude")
    longitude = _number(action["longitude"], "longitude")
    if not -90 < latitude < 90:
        raise ValueError("Domain center latitude must lie between -90 and 90 degrees")
    exp = _build(raw, output)
    domain = exp.domain(grid_id)
    row = _domain_table(raw, grid_id)
    bridge, grids = _native_grids(exp)
    if domain.parent_id == 0:
        # A root reference is its geometric center. Keeping the projection
        # parameters and every child registration moves the whole tree together.
        _transform(bridge, grids[grid_id], 1, [(latitude, longitude)])
        raw["projection"].update(ref_lat=latitude, ref_lon=longitude)
        _refresh_root_fetch(raw, _build(raw, output))
    else:
        grid, parent_grid = grids[grid_id], grids[domain.parent_id]
        center = _transform(bridge, grid, 0, [(grid.e_we / 2, grid.e_sn / 2)])[0]
        current, target = _transform(bridge, parent_grid, 1, [center, (latitude, longitude)])
        parent = exp.domain(domain.parent_id)
        clearance = exp.spec_bdy_width + exp.blend_width
        for key, axis, size, parent_size in (
                ("i_parent_start", 0, domain.run.nx, parent.run.nx),
                ("j_parent_start", 1, domain.run.ny, parent.run.ny)):
            proposed = row[key] + _round_cells(target[axis] - current[axis])
            row[key] = _clamp_start(proposed, parent_size, size, domain.parent_grid_ratio, clearance)


def _minimum_domain_axis(exp, run):
    """The same boundary/stencil floor for every authored domain size."""
    from woof.grid_requirements import FIFTH_ORDER_STENCIL_AXIS, boundary_axis
    return max(boundary_axis(exp.spec_bdy_width, interior_points=1),
               boundary_axis(max(run.spec_zone, run.relax_zone), interior_points=1),
               FIFTH_ORDER_STENCIL_AXIS if run.h_sca_adv_order == 5 else 1)


def _resize_row(raw, domain):
    """Read mass counts from the validated config, including WPS aliases."""
    row = _domain_table(raw, domain.grid_id)
    for key in ("nx", "ny"):
        row[key] = getattr(domain.run, key)
    return row


def _set_domain_axis(row, key, size):
    row[key] = size
    alias = "e_we" if key == "nx" else "e_sn"
    if alias in row:
        row[alias] = size + 1


def _resize_domain(raw, action, output):
    """Resize one registered native grid, retaining spacing and the domain tree."""
    from woof.core.storm_tracking import _round_cells
    _exact_keys(action, {"kind", "grid_id", "bounds"}, {"kind", "grid_id", "bounds"}, where="resize_domain")
    grid_id = _integer(action["grid_id"], "grid_id")
    bounds = action["bounds"]
    _exact_keys(bounds, {"south", "west", "north", "east"}, {"south", "west", "north", "east"}, where="bounds")
    south, west, north, east = (_number(bounds[key], key) for key in ("south", "west", "north", "east"))
    if not -90 < south < north < 90:
        raise ValueError("Resize bounds require -90 < south < north < 90")
    if east <= west:
        east += 360
    if not 0 < east - west < 180:
        raise ValueError("Resize longitude bounds must span less than 180 degrees")
    exp = _build(raw, output)
    domain = exp.domain(grid_id)
    row = _resize_row(raw, domain)
    bridge, grids = _native_grids(exp)
    grid = grids[grid_id]
    # Sample only rectangle edges. The native bridge owns the projection;
    # Python chooses integer dimensions and WPS registration on that lattice.
    points = []
    for step in range(65):
        fraction = step / 64
        longitude = west + (east - west) * fraction
        latitude = south + (north - south) * fraction
        points.extend(((south, longitude), (north, longitude), (latitude, west), (latitude, east)))
    projected = _transform(bridge, grid, 1, points)
    low = [min(point[axis] for point in projected) for axis in (0, 1)]
    high = [max(point[axis] for point in projected) for axis in (0, 1)]
    center = [(a + b) / 2 for a, b in zip(low, high)]
    clearance = exp.spec_bdy_width + exp.blend_width
    children = [child for child in exp.domains if child.parent_id == grid_id]
    quantum = domain.parent_grid_ratio if domain.parent_id else 1
    minimum = _minimum_domain_axis(exp, domain.run)
    parent = exp.domain(domain.parent_id) if domain.parent_id else None
    for axis, key in enumerate(("nx", "ny")):
        min_size = max([minimum] + [getattr(child.run, key) // child.parent_grid_ratio + 2 * clearance
                                   for child in children])
        min_size = math.ceil(min_size / quantum) * quantum
        size = max(min_size, _round_cells((high[axis] - low[axis]) / quantum) * quantum)
        if parent is not None:
            size = min(size, (getattr(parent.run, key) - 2 * clearance) * quantum)
            start_key = "i_parent_start" if axis == 0 else "j_parent_start"
            old_size = getattr(domain.run, key)
            shift = (center[axis] - (old_size + 1) / 2 + (old_size - size) / 2) / quantum
            proposed = row[start_key] + _round_cells(shift)
            row[start_key] = _clamp_start(proposed, getattr(parent.run, key), size, quantum, clearance)
        _set_domain_axis(row, key, size)
    if parent is None:
        latitude, longitude = _transform(bridge, grid, 0, [center])[0]
        raw["projection"].update(ref_lat=latitude, ref_lon=longitude)
    # Direct children retain their dimensions and registrations wherever legal.
    # A smaller parent moves only those that would cross its boundary zone;
    # descendants keep their registration within those unchanged child grids.
    for child in children:
        child_row = _domain_table(raw, child.grid_id)
        for key, axis in (("i_parent_start", "nx"), ("j_parent_start", "ny")):
            child_row[key] = _clamp_start(child_row[key], row[axis], getattr(child.run, axis),
                                          child.parent_grid_ratio, clearance)
    if parent is None:
        _refresh_root_fetch(raw, _build(raw, output))


def _resize_domain_edges(raw, action, output):
    """Apply pointer movement on the original native lattice, anchoring opposite edges."""
    from woof.core.storm_tracking import _round_cells
    required = {"kind", "grid_id", "handle", "start", "end"}
    _exact_keys(action, required, required, where="resize_domain_edges")
    handle = action["handle"]
    if not isinstance(handle, str) or handle not in ("nw", "n", "ne", "e", "se", "s", "sw", "w"):
        raise ValueError("Resize handle must be nw, n, ne, e, se, s, sw or w")
    points = []
    for key in ("start", "end"):
        point = action[key]
        _exact_keys(point, {"latitude", "longitude"}, {"latitude", "longitude"}, where=key)
        latitude = _number(point["latitude"], key + ".latitude")
        longitude = _number(point["longitude"], key + ".longitude")
        if not -90 < latitude < 90:
            raise ValueError("Resize pointer latitude must lie between -90 and 90 degrees")
        points.append((latitude, longitude))
    grid_id = _integer(action["grid_id"], "grid_id")
    exp = _build(raw, output)
    domain = exp.domain(grid_id)
    row = _resize_row(raw, domain)
    bridge, grids = _native_grids(exp)
    grid = grids[grid_id]
    start, end = _transform(bridge, grid, 1, points)
    quantum = domain.parent_grid_ratio if domain.parent_id else 1
    moves = [_round_cells((end[axis] - start[axis]) / quantum) * quantum
             if any(side in handle for side in sides) else 0
             for axis, sides in enumerate(("we", "sn"))]
    _apply_domain_edge_moves(raw, output, exp, domain, row, bridge, grid,
                             handle, moves)


def _resize_domain_cells(raw, action, output):
    """Apply the native dimensions requested by a rectangular map preview.

    Screen north/east are not native Lambert axes: in the southern example
    a northeast screen drag projected to +657,-585 cells and collapsed the
    saved height. The UI now sends its preview's dimensions explicitly. The
    original lattice still owns quantization, minimum size and edge anchors.
    """
    from woof.core.storm_tracking import _round_cells
    required = {"kind", "grid_id", "handle", "nx", "ny"}
    _exact_keys(action, required, required, where="resize_domain_cells")
    handle = action["handle"]
    if not isinstance(handle, str) or handle not in ("nw", "n", "ne", "e", "se", "s", "sw", "w"):
        raise ValueError("Resize handle must be nw, n, ne, e, se, s, sw or w")
    grid_id = _integer(action["grid_id"], "grid_id")
    requested = [_integer(action[key], key) for key in ("nx", "ny")]
    exp = _build(raw, output)
    domain = exp.domain(grid_id)
    row = _resize_row(raw, domain)
    quantum = domain.parent_grid_ratio if domain.parent_id else 1
    moves = []
    for axis, (key, sides, low_side) in enumerate((("nx", "we", "w"), ("ny", "sn", "s"))):
        old = getattr(domain.run, key)
        if not any(side in handle for side in sides):
            if requested[axis] != old:
                raise ValueError(f"Resize handle {handle!r} must preserve {key}={old}")
            moves.append(0)
            continue
        delta = requested[axis] - old
        moves.append(_round_cells(delta / quantum) * quantum
                     * (-1 if low_side in handle else 1))
    bridge, grids = _native_grids(exp)
    _apply_domain_edge_moves(raw, output, exp, domain, row, bridge,
                             grids[grid_id], handle, moves)


def _apply_domain_edge_moves(raw, output, exp, domain, row, bridge, grid,
                             handle, moves):
    """Shared native anchor/clamp rule for geographic and dimension actions."""
    grid_id = domain.grid_id
    quantum = domain.parent_grid_ratio if domain.parent_id else 1
    if not any(moves):
        return  # An unchanged or sub-cell gesture preserves exact saved geometry.
    clearance = exp.spec_bdy_width + exp.blend_width
    children = [child for child in exp.domains if child.parent_id == grid_id]
    minimum = _minimum_domain_axis(exp, domain.run)
    parent = exp.domain(domain.parent_id) if domain.parent_id else None
    low_shift = [0, 0]
    changed = False
    for axis, (key, start_key, low_handle) in enumerate((
            ("nx", "i_parent_start", "w"), ("ny", "j_parent_start", "s"))):
        if not moves[axis]:
            continue
        old_size = getattr(domain.run, key)
        moving_low = low_handle in handle
        min_size = max([minimum] + [getattr(child.run, key) // child.parent_grid_ratio + 2 * clearance
                                   for child in children])
        min_size = math.ceil(min_size / quantum) * quantum
        size = max(min_size, old_size + (-moves[axis] if moving_low else moves[axis]))
        if parent is not None:
            # Each maximum is measured from the unchanged opposite edge.
            # Clamping never slides that anchor to make room for the gesture.
            maximum = (old_size + (row[start_key] - 1 - clearance) * quantum if moving_low else
                       (getattr(parent.run, key) - clearance - (row[start_key] - 1)) * quantum)
            size = min(size, maximum)
        if moving_low:
            low_shift[axis] = old_size - size
            if parent is not None:
                row[start_key] += low_shift[axis] // quantum
        _set_domain_axis(row, key, size)
        changed |= size != old_size
    if not changed:
        return
    if parent is None:
        center = (.5 + low_shift[0] + row["nx"] / 2, .5 + low_shift[1] + row["ny"] / 2)
        latitude, longitude = _transform(bridge, grid, 0, [center])[0]
        raw["projection"].update(ref_lat=latitude, ref_lon=longitude)
    # Rebase child registrations when a low edge moves, retaining their
    # geographic position wherever the parent's new boundary zones permit it.
    for child in children:
        child_row = _domain_table(raw, child.grid_id)
        for axis, (key, size_key) in enumerate((("i_parent_start", "nx"), ("j_parent_start", "ny"))):
            child_row[key] = _clamp_start(child_row[key] - low_shift[axis], row[size_key],
                getattr(child.run, size_key), child.parent_grid_ratio, clearance)
    if parent is None:
        _refresh_root_fetch(raw, _build(raw, output))


def capabilities():
    from woof.core.storm_tracking import FOLLOW_KEYS, TRACKED_FIELDS
    from woof.core.nest_lifecycle import DOMAIN_FOLLOW_EXTRA_KEYS
    from woof.core.nest_spawn import SPAWN_KEYS, SPAWN_TRIGGERS
    return {"schema": "arwen.companion-domain-capabilities.v1",
        "actions": ["add_nest", "remove_nest", "move_domain", "resize_domain", "resize_domain_edges", "resize_domain_cells", "set_output", "set_tiles", "set_placement", "set_activation", "set_follow", "set_spawn", "set_targets", "set_physics"],
        "output_policy": {"history_interval_s": "Saved forecast output interval in seconds for the selected domain; positive.",
                          "restart_interval_s": "Restart checkpoint interval in seconds for the whole experiment; 0 disables checkpoints, omitted or null preserves the existing value.",
                          "validation": "The engine validates whole-second output timestamps, exact domain time-step multiples and existing tracking cadence requirements."},
        "physics_components": physics_components(),
        "follow_fields": list(TRACKED_FIELDS), "follow_keys": sorted(FOLLOW_KEYS | DOMAIN_FOLLOW_EXTRA_KEYS),
        "spawn_triggers": list(SPAWN_TRIGGERS), "spawn_keys": sorted(SPAWN_KEYS),
        "presets": [{"id": "cyclone-mslp-example", "label": "Follow the MSLP minimum",
                     "source": VORTEX_PRESET_SOURCE, "settings": dict(VORTEX_PRESET),
                     "threshold_units": "hPa (absolute MSLP ceiling)",
                     "description": "Existing cyclone example; review its explicit cadence and movement bounds for this grid."}],
        "target_mode": "scheduled discrete relocation in whole parent cells",
        "limits": ["A candidate carries every file its route reads, not the configuration alone: the TOML, its namelist.wps, and on the native regional route the two namelists and the target-domain document beside them, rendered from the edited configuration and listed in route_companions.",
                   "One manual target itinerary can drive one global mover; multiple weather followers use per-domain follow.",
                   "Target points center a fixed-size nest at explicit UTC times; there is no interpolation between targets.",
                   "Use set_placement for a target at experiment start; scheduled relocation targets occur later than the start.",
                   "Absolute target itineraries need an active or scheduled-start nest; trigger-spawned or retiring slots can use weather follow instead.",
                   "Map editing validates configuration and geometry; Run review remains responsible for input and memory admission."]}


def physics_components():
    """Expose the installed engine's own choices, settings and couplings.

    THE COUPLINGS ARE PART OF THE ANSWER.  The registry already carries
    every pairing law as table data -- ``requires_components`` says which
    sibling option an option needs (MYJ needs the Eta surface layer, and
    the Eta layer needs MYJ), ``refused_when`` says which combination is
    refused outright -- and two engine doors read it,
    woof.physics_registry.validate_physics_plan and
    woof.physics_compat.validate_physics_capabilities.  This payload
    dropped both, so a panel built on it could show a scheme with no way
    to say what it must be paired with; the reader met the pairing law
    only as a refusal after choosing.  Forwarded verbatim, from the same
    table those doors read, so a front end cannot invent a second
    vocabulary for one law -- and a pairing added to the registry
    reaches the panel with no code change here.

    ``refused_when`` is filtered through
    :func:`woof.physics_registry._conditional_refusals`, the same filter
    the validator applies, so a malformed rule is dropped in one place
    rather than reaching a caller that would render it.  Couplings key
    off ``registry_option_id``: the ``ra_rrtmg_variant`` fan-out gives one
    registry option several ids here, and every one of them carries the
    registry option's couplings.
    """
    from woof.case_catalog import _native_contract
    from woof.physics_registry import _conditional_refusals, _same_value
    registry, shared, domains = _native_contract()
    allowed = shared | domains
    result = []
    for component, spec in registry["components"].items():
        options = []
        for key, option in spec.get("options", {}).items():
            if not option.get("implemented", False):
                continue
            settings = dict(option.get("parameters", {}))
            settings.update(option.get("constraints", {}).get("required_settings", {}))
            # A conditional rule whose settings clause holds at the
            # registry's own defaults asks, of the option as offered, what
            # a required setting asks: development branch moved the
            # 1.5-order TKE closure's bl_pbl_physics = 0 out of
            # required_settings into a diff_opt = 2 rule, and without this
            # the option stopped carrying it, so choosing TKE on a PBL
            # domain greyed the cell instead of switching the PBL off with
            # it.  The rule's own remedy is the companion edit.  A rule
            # scoped to a source, or with no settings clause, is left to
            # the repair flow.
            for rule in _conditional_refusals(option.get("constraints", {})):
                clause, remedy = rule.get("settings"), rule.get("remedy_settings")
                if (isinstance(clause, dict) and clause and isinstance(remedy, dict)
                        and not isinstance(rule.get("sources"), list)
                        and all(isinstance(values, list) and any(
                            _same_value(registry["parameters"].get(name, {}).get("default"), value)
                            for value in values) for name, values in clause.items())):
                    settings.update(remedy)
            settings.update(option.get("selectors", {}))
            settings = {k: v for k, v in settings.items() if k in allowed and isinstance(v, (str, bool, int, float))}
            if not all(k in settings for k in option.get("selectors", {})):
                continue
            variants = registry["parameters"]["ra_rrtmg_variant"]["enum"] if option.get("selectors") == {"ra_lw_physics": 4, "ra_sw_physics": 4} else [None]
            for variant in variants:
                selected_settings = dict(settings)
                selectors = dict(option.get("selectors", {}))
                label = option.get("label", key)
                if variant is not None:
                    selected_settings.update(ra_rrtmg_variant=variant, wrf_rrtmg_compatibility="none")
                    selectors["ra_rrtmg_variant"] = variant
                    label = "WRF RRTMG longwave and shortwave" if variant == "rrtmg_legacy" else label
                constraints = option.get("constraints", {})
                # AUDIT R-023: the filter above is ``implemented`` only, so
                # this menu offered options the registry declared
                # UNREACHABLE and said nothing about it -- the desktop
                # listed land surface off, surface layer off and the
                # analytic 90/90 proxy while the registry called all three
                # unreachable.  Reachability is now carried to the caller
                # with the blocker that explains it, so a door can rank,
                # gray or warn rather than guess, and an option that
                # becomes unreachable stops being offered silently.
                reachability = option.get("reachability") or {}
                options.append({"id": key if variant is None else f"{key}:{variant}", "registry_option_id": key, "label": label,
                    "selectors": selectors, "settings": selected_settings,
                    "shared_settings": sorted(set(selected_settings) - domains),
                    "reachability": reachability.get("state", ""),
                    "blocker": reachability.get("blocker", ""),
                    "maturity": option.get("maturity", ""), "warnings": option.get("warnings", []),
                    "requires_components": constraints.get("requires_components", {}),
                    "refused_when": _conditional_refusals(constraints)})
        result.append({"id": component, "label": component.replace("_", " ").title(), "options": options})
    return result


def _remove_nest(raw, grid_id, include_children):
    """Remove a complete, explicitly selected subtree; stable IDs survive."""
    if not isinstance(include_children, bool):
        raise ValueError("include_children must be a boolean")
    rows = raw.get("domain")
    if not isinstance(rows, list) or not rows:
        raise ValueError("The configuration needs its root domain before removing a nest")
    parents = {}
    for row in rows:
        domain_id = _integer(row.get("grid_id"), "grid_id")
        parent_id = _integer(row.get("parent_id"), "parent_id", minimum=0)
        if domain_id in parents:
            raise ValueError(f"Duplicate domain d{domain_id:02}; correct its ID in Settings first")
        parents[domain_id] = parent_id
    if sum(parent == 0 for parent in parents.values()) != 1:
        raise ValueError("The configuration needs exactly one root domain before removing a nest")
    for domain_id in parents:
        current, visited = domain_id, set()
        while current != 0:
            if current in visited:
                raise ValueError("Domain parent links contain a cycle; correct them in Settings first")
            visited.add(current)
            if current not in parents:
                raise ValueError("A domain's parent is missing; correct parent_id in Settings first")
            current = parents[current]
    if grid_id not in parents:
        raise ValueError(f"Configuration has no domain d{grid_id:02}")
    if parents[grid_id] == 0:
        raise ValueError("Keep the root domain; use Fit area to change it, or select a child to remove")
    removed = {grid_id}
    while True:
        descendants = {domain_id for domain_id, parent in parents.items()
                       if parent in removed and domain_id not in removed}
        if not descendants:
            break
        removed.update(descendants)
    if len(removed) > 1 and not include_children:
        children = ", ".join(f"d{domain_id:02}" for domain_id in sorted(removed - {grid_id}))
        raise ValueError(f"d{grid_id:02} has child domains {children}; explicitly include its children or remove the leaves first")
    for row in rows:
        domain_id = row["grid_id"]
        if domain_id in removed:
            continue
        reference = row.get("follow", {}).get("refine_grid_id")
        if reference in removed:
            raise ValueError(f"d{domain_id:02} still uses d{reference:02} in follow.refine_grid_id; change that tracking source in Settings first")
    relocation = raw.get("relocation", {})
    owns_relocation = relocation.get("grid_id") in removed
    if not owns_relocation:
        for table, key in (("containment", "grid_id"), ("follow", "refine_grid_id")):
            reference = relocation.get(table, {}).get(key)
            if reference in removed:
                raise ValueError(f"relocation.{table}.{key} still uses d{reference:02}; change that reference in Settings first")
    output_domain = raw.get("case_data", {}).get("output_domain")
    if output_domain in removed:
        raise ValueError(f"case_data.output_domain still selects d{output_domain:02}; choose a surviving output domain in Settings first")
    raw["domain"] = [row for row in rows if row["grid_id"] not in removed]
    if owns_relocation:
        raw.pop("relocation", None)


def _apply(raw, action, output):
    _exact_keys(action, {"kind", "parent_id", "nx", "ny", "parent_grid_ratio", "parent_time_step_ratio",
        "history_interval_s", "placement", "grid_id", "mode", "time", "settings", "points",
        "max_move_parent_cells", "min_overlap_fraction", "cadence_seconds", "include_children",
        "bounds", "latitude", "longitude", "restart_interval_s", "handle", "start", "end"}, {"kind"}, where="action")
    kind = action["kind"]
    targeted = []
    if kind == "set_tiles":
        from woof.core.streaming import STREAMING_MODES
        _exact_keys(action, {"kind", "mode"}, {"kind", "mode"}, where=kind)
        if action["mode"] not in STREAMING_MODES:
            raise ValueError("Tile mode must be off, auto or on")
        raw.setdefault("tiles", {})["mode"] = action["mode"]
    elif kind == "set_output":
        _exact_keys(action, {"kind", "grid_id", "history_interval_s", "restart_interval_s"},
                    {"kind", "grid_id", "history_interval_s"}, where=kind)
        row = _domain_table(raw, _integer(action["grid_id"], "grid_id"))
        row["history_interval_s"] = _number(action["history_interval_s"], "history_interval_s")
        if action.get("restart_interval_s") is not None:
            raw["experiment"]["restart_interval_s"] = _number(action["restart_interval_s"], "restart_interval_s")
        # edit_configuration runs the complete engine parser before publishing:
        # positivity, whole-second writes, exact step divisibility and follower
        # dependencies all retain the same admission rules as a hand-edited TOML.
    elif kind == "move_domain":
        _move_domain(raw, action, output)
    elif kind == "resize_domain":
        _resize_domain(raw, action, output)
    elif kind == "resize_domain_edges":
        _resize_domain_edges(raw, action, output)
    elif kind == "resize_domain_cells":
        _resize_domain_cells(raw, action, output)
    elif kind == "set_physics":
        _exact_keys(action, {"kind", "grid_id", "settings"}, {"kind", "grid_id", "settings"}, where=kind)
        from woof.case_catalog import validate_native_overrides, _native_contract
        grid_id = _integer(action["grid_id"], "grid_id", minimum=0)
        rows = raw["domain"] if grid_id == 0 else [_domain_table(raw, grid_id)]
        if not isinstance(action["settings"], dict):
            raise ValueError("Physics settings must be an object")
        from woof.hrrr_route_inputs import route_shared_domain_keys
        _, shared_keys, domain_keys = _native_contract()
        shared_settings = {k:v for k,v in action["settings"].items() if k in shared_keys and k not in domain_keys}
        if grid_id and shared_settings:
            raise ValueError("These settings apply to all domains: " + ", ".join(sorted(shared_settings)) + ". Select All domains to change them.")
        # A key this configuration's own input route carries once for the
        # whole tree is a tree-wide setting HERE, whatever the schema
        # admits per domain: the route runs the namelists written beside
        # the candidate, and they have one column for the tree. Asked and
        # answered before the candidate is rendered, so the reader meets
        # the sentence above with its way out instead of the importer's
        # words about a namelist column they never asked to write.
        route_shared = {k:v for k,v in action["settings"].items()
                        if k in route_shared_domain_keys((raw.get("fetch") or {}).get("source"))}
        if grid_id and route_shared:
            from woof.hrrr_route_inputs import _settings_phrase
            # THE WAY OUT IS DRIVEN, NOT OFFERED.  "Select All domains"
            # is what this door says for a tree-wide setting, and for
            # two of these keys it is not enough on its own: the route
            # states a runtime switch for the suite that is selected,
            # and the round trip refuses the tree-wide edit until that
            # switch is stated with it.  Told only the first step, a
            # reader takes it and meets the importer's words about a
            # namelist column they never asked to write.
            remedy, refused = _all_domains_remedy(raw, action["settings"], output)
            sentence = ("These settings apply to all domains of a forecast on this input route: "
                + ", ".join(sorted(route_shared)) + ". Select All domains to change them.")
            extra = {k: v for k, v in (remedy or {}).items()
                     if k not in action["settings"]}
            if extra:
                sentence += (" This route states " + _settings_phrase(extra)
                    + " for that selection and runs the namelists it writes beside the"
                    " configuration, so change " + _settings_phrase(remedy) + " together.")
            elif remedy is None:
                sentence += (" Selecting All domains is refused here as well, so read its"
                    " own way out first: " + str(refused))
            raise DomainScopeError(sentence, remedy)
        domain_settings = {k:v for k,v in action["settings"].items() if k not in shared_settings}
        changes = {"shared": shared_settings, "domains": [{"grid_id": row["grid_id"], "settings": domain_settings} for row in rows]}
        validate_native_overrides(changes)
        raw.setdefault("shared", {}).update(copy.deepcopy(shared_settings))
        for row in rows:
            row.update(copy.deepcopy(domain_settings))
    elif kind == "remove_nest":
        _exact_keys(action, {"kind", "grid_id", "include_children"}, {"kind", "grid_id", "include_children"}, where=kind)
        _remove_nest(raw, _integer(action["grid_id"], "grid_id"), action["include_children"])
    elif kind == "add_nest":
        from woof.core.storm_tracking import _round_cells
        required = {"kind", "parent_id", "nx", "ny", "parent_grid_ratio", "parent_time_step_ratio",
                    "history_interval_s", "placement"}
        _exact_keys(action, required, required, where="add_nest")
        exp = _build(raw, output)
        parent = exp.domain(_integer(action["parent_id"], "parent_id"))
        child_id = max(d.grid_id for d in exp.domains) + 1
        row = {"grid_id": child_id, "parent_id": parent.grid_id,
               "specified": False, "nested": True,
               "i_parent_start": exp.spec_bdy_width + exp.blend_width + 1,
               "j_parent_start": exp.spec_bdy_width + exp.blend_width + 1}
        for key in ("nx", "ny", "parent_grid_ratio", "parent_time_step_ratio"):
            row[key] = _integer(action[key], key)
        # A drawn rectangle supplies mass-cell counts, while a WPS child must
        # span whole parent cells (e_we/e_sn = n * ratio + 1). Snap before the
        # first child build; the same boundary/stencil minimum as resizing
        # prevents a tiny gesture from authoring an unusable grid. The parser
        # still owns containment, time-step and physics admission.
        ratio = row["parent_grid_ratio"]
        minimum = math.ceil(_minimum_domain_axis(exp, parent.run) / ratio) * ratio
        for key in ("nx", "ny"):
            row[key] = max(minimum, _round_cells(row[key] / ratio) * ratio)
        row["history_interval_s"] = _number(action["history_interval_s"], "history_interval_s")
        # A nest INHERITS its parent's radiation cadence. The rule is the
        # emission door's (woof.domain_wizard.radt_ladder_minutes:
        # radiative transfer varies on cloud timescales, not grid scales),
        # and a row written without it fell to the schema default instead,
        # so a nest added here could join a tree on one cadence and run
        # another. Written into BOTH spellings at the parent's effective
        # value: radt is what the emission door puts on its own nest rows
        # and what the route's namelist carries, radt_minutes is what a
        # zero cadence needs (a zero radt means "not stated", not "every
        # step"), and agreeing values leave the pair with nothing to
        # resolve.
        from woof.config import effective_radt_minutes
        row["radt"] = row["radt_minutes"] = effective_radt_minutes(parent.run)
        raw["domain"].append(row)
        exp = _build(raw, output)
        row["i_parent_start"], row["j_parent_start"] = _placement(exp, child_id, action["placement"])
    else:
        grid_id = _integer(action.get("grid_id"), "grid_id")
        row = _domain_table(raw, grid_id)
        if row["parent_id"] == 0:
            raise ValueError("This action applies to a nested grid; use Fit area to change the root")
        if kind == "set_placement":
            _exact_keys(action, {"kind", "grid_id", "placement"}, {"kind", "grid_id", "placement"}, where=kind)
            row["i_parent_start"], row["j_parent_start"] = _placement(_build(raw, output), grid_id, action["placement"])
        elif kind == "set_activation":
            _exact_keys(action, {"kind", "grid_id", "mode", "time"}, {"kind", "grid_id", "mode"}, where=kind)
            if row.get("retire") or row.get("rearm"):
                raise ValueError("This nest has retirement/re-arm policy; change its trigger explicitly with set_spawn")
            if row.get("spawn", {}).get("trigger") not in (None, "time"):
                raise ValueError("This nest already has a weather trigger; edit or remove it explicitly with set_spawn")
            row.pop("start_time", None)
            row.pop("spawn", None)
            if action["mode"] == "scheduled":
                row["start_time"] = _utc(action.get("time"))
            elif action["mode"] == "spawn_time":
                exp = _build(raw, output)
                row["spawn"] = {"trigger": "time", "at_s": (_utc(action.get("time")) - exp.start_time).total_seconds()}
            elif action["mode"] != "immediate":
                raise ValueError("Activation mode must be immediate, scheduled or spawn_time")
        elif kind == "set_follow":
            _exact_keys(action, {"kind", "grid_id", "settings"}, {"kind", "grid_id", "settings"}, where=kind)
            from woof.core.nest_lifecycle import build_domain_follow_config, DOMAIN_FOLLOW_EXTRA_KEYS
            settings = action["settings"]
            if settings is not None:
                build_domain_follow_config(settings, str(output), grid_id=grid_id)
            relocation = raw.get("relocation", {})
            if relocation.get("grid_id") == grid_id:
                if relocation.get("move") and settings is not None:
                    raise ValueError("This nest has a manual target itinerary; remove it explicitly before enabling weather follow")
                if settings is None and (relocation.get("containment") or relocation.get("track")):
                    raise ValueError("This global follower has containment or track-output settings; remove those explicitly in Settings before disabling its tracker")
                relocation.pop("follow", None)
                if settings is not None:
                    relocation["follow"] = {k: v for k, v in settings.items() if k not in DOMAIN_FOLLOW_EXTRA_KEYS}
                    relocation.update({k: v for k, v in settings.items() if k in DOMAIN_FOLLOW_EXTRA_KEYS})
                    relocation["enabled"] = True
                elif not relocation.get("move"):
                    relocation.pop("cadence_seconds", None)
            elif settings is None:
                row.pop("follow", None)
            else:
                row["follow"] = copy.deepcopy(settings)
        elif kind == "set_spawn":
            _exact_keys(action, {"kind", "grid_id", "settings"}, {"kind", "grid_id", "settings"}, where=kind)
            from woof.core.nest_spawn import build_spawn_config
            settings = action["settings"]
            if settings is None:
                if row.get("retire") or row.get("rearm"):
                    raise ValueError("Remove retirement/re-arm policy before removing its required spawn trigger")
                row.pop("spawn", None)
            else:
                build_spawn_config(settings, str(output), grid_id=grid_id)
                row.pop("start_time", None)
                row["spawn"] = copy.deepcopy(settings)
        elif kind == "set_targets":
            allowed = {"kind", "grid_id", "points", "max_move_parent_cells", "min_overlap_fraction", "cadence_seconds"}
            _exact_keys(action, allowed, allowed - {"cadence_seconds"}, where=kind)
            exp = _build(raw, output)
            relocation = raw.get("relocation", {})
            points = action["points"]
            if not isinstance(points, list):
                raise ValueError("Target points must be an array")
            if relocation.get("enabled") and relocation.get("grid_id") not in (None, grid_id):
                raise ValueError("WOOF supports one global manual mover; another nest already owns relocation")
            if not points:
                relocation.pop("move", None)
                if not relocation.get("follow"):
                    relocation.pop("cadence_seconds", None)
                return targeted
            if row.get("follow") or relocation.get("follow") or relocation.get("containment") or relocation.get("track"):
                raise ValueError("Manual targets and automatic tracking/containment have separate placement authorities; disable automatic policy first")
            activation = exp.start_time
            member = exp.domain(grid_id)
            while True:
                if member.spawn is not None or member.retire is not None or member.rearm is not None:
                    raise ValueError(
                        f"d{member.grid_id:02} has a trigger-spawn/retirement lifecycle. Absolute target itineraries require a scheduled UTC start; triggered slots can use weather follow instead.")
                activation = max(activation, member.start_time or exp.start_time)
                if not member.parent_id:
                    break
                member = exp.domain(member.parent_id)
            ancestor = exp.domain(row["parent_id"])
            while ancestor.parent_id:
                if ancestor.follow is not None:
                    raise ValueError("Geographic target itineraries require a stationary parent frame; an ancestor follows weather")
                ancestor = exp.domain(ancestor.parent_id)
            moves, previous = [], (row["i_parent_start"], row["j_parent_start"])
            for point in points:
                _exact_keys(point, {"time", "latitude", "longitude"}, {"time", "latitude", "longitude"}, where="target point")
                moment = _utc(point["time"])
                if moment < activation:
                    raise ValueError(f"Target {moment.isoformat()} is before d{grid_id:02} and its parent chain are active at {activation.isoformat()}")
                i, j = _placement(exp, grid_id, {"kind": "center", "latitude": point["latitude"], "longitude": point["longitude"]})
                moves.append({"at_seconds": (moment-exp.start_time).total_seconds(),
                              "di_parent_cells": i-previous[0], "dj_parent_cells": j-previous[1]})
                targeted.append({"time": moment, "requested_latlon": [point["latitude"], point["longitude"]],
                                 "i_parent_start": i, "j_parent_start": j})
                previous = i, j
            relocation.update(enabled=True, grid_id=grid_id, move=moves,
                max_move_parent_cells=_integer(action["max_move_parent_cells"], "max_move_parent_cells"),
                min_overlap_fraction=_number(action["min_overlap_fraction"], "min_overlap_fraction"))
            if action.get("cadence_seconds") is not None:
                relocation["cadence_seconds"] = _number(action["cadence_seconds"], "cadence_seconds")
            else:
                relocation.pop("cadence_seconds", None)
            raw["relocation"] = relocation
            resolved = _build(raw, output)
            from woof.core.nest_relocation import Placement, placement_of, plan_relocation, check_admissible
            child = resolved.domain(grid_id)
            previous = placement_of(child)
            for index, point in enumerate(targeted):
                destination = Placement(grid_id, point["i_parent_start"], point["j_parent_start"], index+1)
                plan = plan_relocation(placement_from=previous, placement_to=destination,
                    parent_grid_ratio=child.parent_grid_ratio, child_nx=child.run.nx, child_ny=child.run.ny)
                point["admission"] = check_admissible(plan, resolved.relocation)
                target_raw = copy.deepcopy(raw)
                target_row = _domain_table(target_raw, grid_id)
                target_row.update(i_parent_start=destination.i_parent_start, j_parent_start=destination.j_parent_start)
                point["outline"] = next(d for d in native_domain_outlines(_build(target_raw, output)) if d["grid_id"] == grid_id)
                previous = destination
        else:
            raise ValueError(f"Unknown domain action {kind!r}")
    return targeted


def _wps_text(exp, original_path, output_path, raw, original_count, original_domain_ids=None):
    from woof.hrrr_prepared_bundle import render_wps_namelist
    from woof.namelist_import import parse_namelist_text
    from woof.starter_template import _tiles_wps
    from woof.stream import _render_namelist_values
    from woof.wps_domain_ids import domain_ids_from_wps_text, validated_domain_ids, with_domain_ids
    generated = parse_namelist_text(render_wps_namelist(exp))
    original_ids = validated_domain_ids(
        tuple(range(1, original_count + 1)) if original_domain_ids is None else original_domain_ids)
    if len(original_ids) != original_count:
        raise ValueError("The original WPS domain order is not uniquely bound to the configuration")
    if original_path is not None and original_path.is_file():
        original = original_path.read_text(encoding="utf-8-sig")
        tables = parse_namelist_text(_tiles_wps(original_path, output_path, text=original))
        declared_count = int(tables.get("share", {}).get("max_dom", [1])[0])
        if domain_ids_from_wps_text(original, declared_count) != original_ids:
            raise ValueError("The original WPS domain identity differs from the configuration; reconcile its domain IDs before editing")
    else:
        tables = copy.deepcopy(generated)
    tables.setdefault("share", {})["max_dom"] = [len(exp.domains)]
    interval = raw.get("case_data", {}).get("forcing_interval_s")
    if interval is None and raw.get("fetch", {}).get("cadence") is not None:
        interval = raw["fetch"]["cadence"] * 3600
    if interval is not None:
        if interval != int(interval):
            raise ValueError("WPS requires a whole-second forcing interval")
        tables["share"]["interval_seconds"] = [int(interval)]
    geogrid = tables.setdefault("geogrid", {})
    for key, value in generated["geogrid"].items():
        if key != "geog_data_res":
            geogrid[key] = value
    resolutions = geogrid.get("geog_data_res", ["default"])
    # Fortran initializes unspecified entries to default; a single custom
    # value belongs only to the first domain, not to every child.
    by_id = {domain_id: resolutions[index] if index < len(resolutions) else "default"
             for index, domain_id in enumerate(original_ids)}
    for domain in exp.domains:
        if domain.grid_id not in by_id:
            by_id[domain.grid_id] = by_id.get(domain.parent_id, "default")
    geogrid["geog_data_res"] = [by_id[domain.grid_id] for domain in exp.domains]
    for key, moment in (("start_date", exp.start_time), ("end_date", exp.start_time+timedelta(seconds=exp.run_seconds))):
        tables["share"][key] = [moment.strftime("%Y-%m-%d_%H:%M:%S")] * len(exp.domains)
    text = "\n".join("&"+section+"\n"+"\n".join(
        f" {key} = {_render_namelist_values(value)}" for key, value in table.items())+"\n/"
        for section, table in tables.items())+"\n"
    if parse_namelist_text(text) != tables:
        raise ValueError("Candidate WPS did not preserve its parsed settings")
    return with_domain_ids(text, [domain.grid_id for domain in exp.domains])


def candidate_route_files(raw, original_exp, exp, output, *, original_wps=None):
    """The WPS text and the route companions a candidate publishes.

    ONE renderer for the door that SAVES and for every measurement of
    what a save would meet.  Measured any other way, a panel offered
    "Select All domains to change them" for an edit whose save the
    route's own importer then refused, which is a remedy that does not
    work told to a reader as though it had been tried.

    ``original_wps`` is the WPS namelist a save PRESERVES.  A
    measurement passes ``None`` and gets the candidate's own rendered
    geography instead: every field :func:`~woof.hrrr_route_inputs.
    verify_round_trip` compares comes from the experiment, and the
    preserved file's remaining settings (its resolution choices, its
    other sections) reach none of them.
    """

    from woof.hrrr_route_inputs import candidate_companions

    wps = candidate_wps_text(raw, original_exp, exp, output,
                             original_wps=original_wps)
    return wps, candidate_companions(
        output, exp, wps_text=wps,
        source=(raw.get("fetch") or {}).get("source"))


def candidate_wps_text(raw, original_exp, exp, output, *, original_wps=None):
    """The WPS namelist this candidate publishes, on its own.

    Split out because it is the INPUT to the route's question rather
    than part of the answer: a layout that cannot be reconciled with
    the one being preserved is a refusal of this edit, and it is not
    the route saying anything about the physics it was asked about.
    """

    return _wps_text(exp, original_wps, output.with_suffix(".namelist.wps"),
                     raw, len(original_exp.domains),
                     original_domain_ids=[domain.grid_id
                                          for domain in original_exp.domains])


def configuration_wps_namelist(config_path, exp, *, raw, into=None):
    """The WPS namelist one run of ``config_path`` hands its preparation.

    ``<stem>.namelist.wps`` beside the configuration when it is there: a
    door wrote it, and its geography choices are the user's.  Otherwise
    the run renders it from the configuration by the doors' own renderer
    (:func:`candidate_wps_text`, the one a save and a re-time use), holds
    the bytes to the forecast stage's own geometry comparison
    (:func:`woof.native_wrf_contract.validate_wps_geometry`), and writes
    it into ``into`` -- never beside the user's file, which may sit in an
    installed package.  Every number in it already exists in the
    configuration's ``[projection]`` and ``[[domain]]`` tables and its
    ``[fetch]`` cadence, so a twin file kept by hand beside each
    configuration could only ever drift from them.

    The boundary interval is ``[case_data] forcing_interval_s``, else the
    ``[fetch]`` cadence, else the source registry's forcing interval -- the
    interval ``woof domain`` writes for a configuration it emits.

    ``into=None`` is the door's question (dry run, plan review): the
    namelist is rendered and checked in a scratch folder and ``None``
    comes back, so a configuration the run could not render for is
    refused before anything is fetched.

    Breakage it prevents: ``woof go`` on every shipped HRRR configuration
    recipe stopped before any stage ("the staged route reads
    hrrr_configuration_cut.namelist.wps beside ..."), because none of
    them carried a WPS twin (WOOF 1.0.3 GPU smoke, 2026-10-06).

    Raises ``ValueError`` naming what the renderer or the geometry check
    refused.
    """

    import tempfile

    from woof.native_wrf_contract import validate_wps_geometry

    config_path = Path(config_path)
    beside = config_path.with_name(f"{config_path.stem}.namelist.wps")
    if beside.is_file():
        return beside
    fetch = raw.get("fetch") if isinstance(raw.get("fetch"), dict) else {}
    if ((raw.get("case_data") or {}).get("forcing_interval_s") is None
            and fetch.get("cadence") is None and fetch.get("source")):
        from woof.source_adapters import source_forcing_interval_seconds

        try:
            interval = source_forcing_interval_seconds(str(fetch["source"]))
        except (KeyError, ValueError):
            interval = None
        if interval is not None and float(interval) % 3600 == 0:
            raw = {**raw, "fetch": {**fetch, "cadence": int(interval) // 3600}}
    text = candidate_wps_text(raw, exp, exp, config_path)
    data = text.encode("utf-8")
    source = str(fetch.get("source") or "configuration")

    def checked(path: Path) -> Path:
        path.write_bytes(data)
        try:
            validate_wps_geometry(exp, path, source_name=source)
        except ValueError as error:
            raise ValueError(
                f"the WPS namelist rendered from {config_path.name} does not "
                f"match its own grid ({error}); put the namelist `woof "
                f"domain` writes beside it as {beside.name}") from error
        return path

    if into is None:
        with tempfile.TemporaryDirectory(prefix="arwen-wps-") as staging:
            checked(Path(staging) / beside.name)
        return None
    into = Path(into)
    into.mkdir(parents=True, exist_ok=True)
    target = into / beside.name
    if target.is_file() and target.read_bytes() == data:
        return target
    staged = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    try:
        checked(staged)
    except BaseException:
        staged.unlink(missing_ok=True)
        raise
    os.replace(staged, target)
    return target


def candidate_route_blocker(raw, original_exp, exp, output):
    """What this candidate's ROUTE would refuse about it, or ``None``.

    The question every caller that says what a save would meet has to
    ask, asked of the renderer itself rather than of a table of setting
    names.

    ONE question: what the route's own namelist writer and importer say
    about this candidate.  Both halves of that answer count -- a
    selection those namelists have no spelling for, and a set they read
    back as a different tree -- because a save meets either one.

    A route that reads the configuration itself is not asked at all,
    and a candidate whose WPS namelist could not be rendered returns
    ``None``: that step is the input to the question rather than the
    answer, so a layout that cannot be reconciled with the one being
    preserved is not reported as the route refusing a physics
    selection.
    """

    from woof.source_drivability import candidate_route_chain

    if candidate_route_chain((raw.get("fetch") or {}).get("source")) != "prepared:hrrr":
        return None
    try:
        wps = candidate_wps_text(raw, original_exp, exp, output)
    except Exception:  # the question was never reached, so it is unanswered
        return None
    from woof.hrrr_route_inputs import candidate_companions
    try:
        candidate_companions(output, exp, wps_text=wps,
                             source=(raw.get("fetch") or {}).get("source"))
    except (ValueError, NotImplementedError) as error:
        return error
    except Exception:  # not the route's answer, so not reported as one
        return None
    return None


def _all_domains_remedy(raw, settings, output):
    """The tree-wide edit that publishes, driven, and what refused it.

    Returns ``(settings_that_published, refusal)``, at most one of them
    set.  Two attempts, no more: these settings at every domain, and --
    where the route's round trip refused them and said what its own
    namelist column carries -- the same settings with those fields
    stated beside them.  Nothing is claimed that was not driven through
    the renderer the save publishes through, so a door that names
    "Select All domains" has tried it.
    """

    def attempt(candidate_settings):
        """``(published, refusal)`` for these settings at every domain.

        The parser and the route are asked separately because only the
        route's own refusal has a second step to name.  Where the
        renderer falls over for a reason of its own the answer is the
        sentence this door already shipped, with nothing added: an
        unmeasured second step is not stated at all.
        """

        candidate = copy.deepcopy(raw)
        try:
            original_exp = _build(raw, output)
            _apply(candidate, {"kind": "set_physics", "grid_id": 0,
                               "settings": dict(candidate_settings)}, output)
            exp = _build(candidate, output)
        except (ValueError, NotImplementedError) as error:
            return False, error
        except Exception:  # a probe that could not read is not an answer
            return False, None
        blocker = candidate_route_blocker(candidate, original_exp, exp, output)
        if blocker is not None:
            return False, blocker
        return True, None

    published, refused = attempt(settings)
    if published:
        return dict(settings), None
    stated = getattr(refused, "namelist_values", None)
    if stated:
        wider = dict(settings, **stated)
        published, second = attempt(wider)
        if published:
            return wider, None
        refused = second or refused
    return None, refused


def edit_configuration(request):
    from woof.config_authority import read_config_authority
    from woof.case_data import resolved_case_data_paths
    from woof.experiment import experiment_config_document, domain_config_document
    from woof.starter_template import changes, _publish_new_files
    from woof.toml_document import emit_experiment_toml

    required = {"schema", "config_path", "expected_sha256", "output_path", "action"}
    _exact_keys(request, required, required, where="domain edit request")
    if request["schema"] != REQUEST_SCHEMA:
        raise ValueError("Unsupported domain edit request schema")
    authority = read_config_authority(request["config_path"])
    if request["expected_sha256"] != authority.sha256:
        raise ValueError("The selected configuration changed; refresh its domain layers before editing")
    output = Path(request["output_path"]).expanduser().resolve()
    wps_output, receipt_output = output.with_suffix(".namelist.wps"), output.with_suffix(".domains.json")
    # Every path this door may write, including the native route's three
    # extra companions. They were left out of the sweep while the door
    # did not write them, so an edit published next to an earlier edit's
    # namelists silently left those in place for the run to read.
    from woof.hrrr_route_inputs import route_input_paths
    reserved = (output, wps_output, receipt_output, *route_input_paths(output).values())
    if output == authority.source or any(os.path.lexists(p) for p in reserved):
        raise FileExistsError("Choose a new candidate output path; existing configurations and companions are preserved")
    original = tomllib.loads(authority.payload.decode("utf-8-sig"))
    original_exp = _build(original, authority.source)
    raw = copy.deepcopy(original)
    original_wps = None
    if "case_data" in raw:
        raw["case_data"] = resolved_case_data_paths(raw["case_data"], base_dir=authority.base_dir, source=str(authority.source))
        if raw["case_data"].get("wps_namelist"):
            original_wps = Path(raw["case_data"]["wps_namelist"])
            raw["case_data"]["wps_namelist"] = str(wps_output)
    if "static" in raw:
        from woof.static.highres_production import parse_static_table
        config = parse_static_table(raw["static"], source=str(authority.source), base_dir=authority.base_dir)
        if config is not None and "highres" in raw["static"]:
            raw["static"]["highres"]["cache_root"] = str(config.cache_root.resolve())
    targets = _apply(raw, request["action"], output)
    exp = _build(raw, output)
    text = "# Candidate domain edit; original configuration preserved. Review before Run.\n" + emit_experiment_toml(raw)
    config_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    outlines = native_domain_outlines(exp)
    configuration = {"schema": "arwen.companion-configuration.v1", "config_path": str(output),
        "config_sha256": config_sha, "geometry_backend": "rust-static-fields", "domains": outlines,
        "domains_geojson": domain_geojson(outlines, targets),
        "experiment": experiment_config_document(exp), "tiles": exp.tiles.to_mapping(), "fetch": raw.get("fetch", {}),
        "case_data": {k:v for k,v in raw.get("case_data", {}).items() if k in
                      ("forcing_interval_s", "start_time", "end_time", "wps_namelist")}}
    # Rendered from the EDITED experiment, so an edited cadence reaches
    # the namelist the route reads instead of the TOML it does not.
    # The candidate's own [fetch].source, which is exactly what the
    # dispatcher reads to choose the chain (woof/runplan.py _chain_key).
    # Not "None when [case_data] is present": a configuration carrying
    # both would then be given a short candidate and still routed to the
    # regional chain, which is the failure this door was fixed for.
    wps, companions = candidate_route_files(raw, original_exp, exp, output,
                                            original_wps=original_wps)
    result = {"schema": RESULT_SCHEMA, "created": True, "forecast_started": False,
        "config_path": str(output), "config_sha256": config_sha, "wps_path": str(wps_output),
        "route_companions": [str(path) for path, _text in companions],
        "receipt_path": str(receipt_output), "source_path": str(authority.source), "source_sha256": authority.sha256,
        "configuration": configuration, "domains": [domain_config_document(d) for d in exp.domains],
        "map_geojson": domain_geojson(outlines, targets),
        "target_points": targets, "changes": [{"field": k,"before": a,"after": b} for k,a,b in changes(original, raw)],
        "validation": {"configuration_parser": "passed", "native_geometry": "passed", "forecast_or_memory_admission": "not_run"},
        "notes": capabilities()["limits"]}
    if hashlib.sha256(authority.source.read_bytes()).hexdigest() != authority.sha256:
        raise ValueError("The original configuration changed while preparing the candidate; refresh before editing")
    output.parent.mkdir(parents=True, exist_ok=True)
    # The existing create-only publication helper owns rollback and preserves a
    # replaced file. The configuration is published after its companions and receipt.
    _publish_new_files([*companions, (receipt_output, _json(result)), (output, text)])
    return result


def main(args):
    try:
        with contextlib.redirect_stdout(sys.stderr):
            if args.capabilities:
                result = capabilities()
            else:
                if args.request is None:
                    raise ValueError("Provide --request or --capabilities")
                path = Path(args.request)
                if not path.is_file() or path.stat().st_size > 256 * 1024:
                    raise ValueError("Domain edit request must be a JSON file no larger than 256 KiB")
                request = json.loads(path.read_text(encoding="utf-8-sig"))
                if args.repairs:
                    from woof.companion_physics import repairs
                    result = repairs(request)
                elif args.availability:
                    from woof.companion_physics import availability
                    result = availability(request)
                else:
                    result = edit_configuration(request)
        print(_json(result), end="")
        return 0
    except Exception as error:
        print(_json({"schema": RESULT_SCHEMA, "created": False, "forecast_started": False, "error": str(error)}), end="")
        return 1


def register_cli(subparsers):
    parser = subparsers.add_parser("companion-domains", help="create a candidate domain edit using native geometry")
    parser.add_argument("--request", type=Path, help="a domain edit request document; the candidate it publishes carries every file its route reads, named in the result's route_companions")
    parser.add_argument("--capabilities", action="store_true", help="report the actions, presets and limits this door offers, and write nothing")
    parser.add_argument("--repairs", action="store_true", help="check compatible physics replacements without writing a candidate")
    parser.add_argument("--availability", action="store_true", help="report why each installed physics option is open or closed to a draft")
    parser.set_defaults(func=main)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--capabilities", action="store_true")
    parser.add_argument("--repairs", action="store_true")
    parser.add_argument("--availability", action="store_true")
    raise SystemExit(main(parser.parse_args()))
