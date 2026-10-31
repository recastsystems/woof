"""Nests that spawn later on conditions the user sets: the ``[hex_nests]`` table.

THE BREAKAGE THIS PREVENTS: a regional nest inside a hex parent could only be
placed at launch (shape 1) or by hand while the run went (shape 2).  A user who
wants "3 km on the first big storm in the Southern Plains" had to watch the run
and click, or pay for a nest over an area where nothing happened.  This module
looks at the running parent on the card, decides when and where the asked-for
conditions hold, and writes one firing row per spawn to
``<run>/nest-triggers.jsonl``.  The add door (``woof hex add-nest --watch``)
reads that file and starts each nest exactly as a hand-placed add, so the nest
itself is the same code in all three shapes.

WHAT IS DATA.  Every attribute a condition can test is one row of
``data/nest-trigger-fields.v1.json`` (its units, how it reduces over a column,
and which parent array it reads).  A new attribute is a table row whose
``source`` names an array a route already has on the card; nothing here
branches on a field name except through that table.  A field whose source the
running route does not compute is refused at arm time by name, because a test
that can never see its field would silently never fire.

WHERE IT RUNS.  In the forecast's step loop, every ``cadence_s`` of model time
and only while armed: the column reduction, the tests, the coverage sums and
the storm-object labelling all run on the card over the area's cells only
(:class:`DeviceEvaluator`).  They reduce to a short record that comes back by
one asynchronous copy into pinned memory; the loop reads it when its event has
completed (normally at the next step), so a look adds no device sync and no
host pass over fields.  The firing decision (:class:`TriggerState`) is host
arithmetic on that record.

THE TABLE (all keys; any other is refused by name)::

    [hex_nests]
    max_firings   = 2           # hard cap on spawns in the run
    max_live      = 2           # nests alive at once
    earliest      = "+6h"       # "+Nh", "+Nm", seconds, or an ISO time
    latest        = "+36h"
    cadence_s     = 900         # how often the card looks
    sustain_looks = 2           # looks in a row a block must hold
    lead_back_s   = 0           # start the nest this far before the firing look
    cooldown_s    = 1800        # after a firing, no new firing for this long
    merge_km      = 0           # 0: half the nest size; a centroid this close to a live nest is not new
    area          = { box = { south = 30, north = 40, west = -104, east = -92 } }
                    # or { circle = { lat, lon, radius_km } } or { polygon = [[lat, lon], ...] }
    nest          = { dx_km = 3, size_km = 400, hours = 6 }

    [[hex_nests.when]]          # any one block is enough; inside a block every test must hold
    all = [ { field = "refl", op = ">=", value = 45, cover_km2 = 500 } ]

A test is ``{field, op, value}`` plus at most one of:

* ``cover_km2``    the area of cells meeting the test must reach this;
* ``object_km2``   one connected object of such cells must reach this (the
                   storm-object test: connected cells on the mesh adjacency).

Without either, the test holds when any area cell meets it.

WHERE A NEST GOES.  On a storm OBJECT, for every test type: the cells meeting
the block's first test are labelled into connected objects on the mesh, and
the firing point is the area-weighted centroid of the biggest object that is
not already inside a live nest.  The record carries the biggest
``OBJECT_SLOTS`` objects, so a second firing goes to the second storm instead
of being held back by the first nest.  (Before 2026-10-23 a ``cover_km2`` or
point test fired at the centroid of EVERY meeting cell in the area, which with
two storm clusters in the box put the nest between them, and the second
firing's centroid barely moved, so it stayed inside ``merge_km`` of the first
nest and never fired: shapes 2-3 checker.)

ECHO INSIDE A LIVE NEST DOES NOT COUNT.  Every live nest on the parent (the
loop's own firings, and every nest add-nest lists in ``<run>/live-nests.jsonl``:
placed at launch or added by hand) masks its whole square footprint out of the
look before anything is summed or labelled, so a nest's own storms neither
reach a ``cover_km2`` / ``object_km2`` threshold nor take one of the record's
object slots.  (2026-10-24: the Oklahoma storms inside shape 1's nest
made up 1,441 of the 1,834 km2 that fired a paid nest, which then went onto a
197 km2 Kansas echo, below the table's own 500 km2; and with only the 4
biggest objects reported, all inside live nests, a fifth storm could never be
picked.  The footprint is the nest's square, not a circle of half its size, so
a storm in a nest's corner is covered.)

WHERE A NEST GOES, AND WHERE ITS STORM WILL BE.  A nest is centred on the
storm's track over the nest's hours, not on where the storm is at the look:
the object is matched to the same storm in the looks before (its centroid
moves less than ``MATCH_SPEED_MS`` between looks), its motion fitted over up
to ``TRACK_LOOKS`` looks and capped at ``MAX_STORM_SPEED_MS``, and the centre
moved half the nest's run along it, never so far that the storm as it is now
sits closer than ``PLACEMENT_EDGE_FRACTION`` of the nest's size to an edge.
(2026-10-24: trigger-0 was centred on the storm at the look, the storm
system grew 250-300 km to the south-east, and from +3 h a 60-70 dBZ area sat
against the nest's south and south-east edges.)
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from datetime import datetime, timedelta
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Sequence

import numpy as np

TABLE = "hex_nests"
SCHEMA = "gpuwm-hex.nest-triggers.v1"
FIRING_SCHEMA = "gpuwm-hex.nest-firing.v1"
FIRINGS_NAME = "nest-triggers.jsonl"
LOOKS_NAME = "nest-trigger-looks.jsonl"
STATE_NAME = "nest-trigger-state.json"
#: Every nest add-nest starts on this parent (hand-placed, added mid-run, or a
#: firing), one row when its forecast starts and one when it ends.  The loop reads
#: it so a firing never lands on a storm a nest the user already has covers.
LIVE_NESTS_NAME = "live-nests.jsonl"
EARTH_RADIUS_M = 6371229.0
FIELDS_PATH = Path(__file__).with_name("data") / "nest-trigger-fields.v1.json"

KEYS = frozenset({
    "max_firings", "max_live", "earliest", "latest", "cadence_s", "sustain_looks",
    "lead_back_s", "cooldown_s", "merge_km", "area", "nest", "when", "hold",
})
#: The site's prices for this run's trigger (apps/web nest-addon.ts triggerHold),
#: carried in the table so the loop's fire-time hold check reads the numbers the
#: user authorised (woof.hex.nest_budget).  Absent on a run nobody pays for.
HOLD_KEYS = frozenset({"hold_cents", "reserve_cents", "per_firing_cents", "per_firing_quote_cents",
                       "overrun_factor", "usd_per_hour", "output_seconds"})
NEST_KEYS = frozenset({"dx_km", "size_km", "hours"})
TEST_KEYS = frozenset({"field", "op", "value", "cover_km2", "object_km2"})
OPS = {">=": np.greater_equal, ">": np.greater, "<=": np.less_equal, "<": np.less}
DEFAULTS = {
    "max_firings": 2,
    "max_live": 2,
    "earliest": 0.0,
    "latest": None,
    "cadence_s": 900.0,
    "sustain_looks": 2,
    "lead_back_s": 0.0,
    "cooldown_s": 1800.0,
    "merge_km": 0.0,
}
NEST_DEFAULTS = {"dx_km": 3.0, "size_km": 400.0, "hours": 6.0}
#: Fixed label-propagation sweeps per look.  Each sweep is a min over the
#: neighbours followed by a pointer jump, so a component of graph diameter D
#: settles in about log2(D) + 2 sweeps; 40 settles any object a 3,000 km core can
#: hold at 3 km and costs no host round trip to decide when to stop.
LABEL_SWEEPS = 40


class NestTriggerRefusal(ValueError):
    """A ``[hex_nests]`` request this module will not arm."""


# --------------------------------------------------------------------------- table

def load_field_table(path: Path | None = None) -> dict[str, dict[str, Any]]:
    data = json.loads((path or FIELDS_PATH).read_text(encoding="utf-8"))
    return {row["name"]: row for row in data["fields"]}


def _seconds(value: Any, *, key: str, start: datetime | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise NestTriggerRefusal(f"[{TABLE}] {key} must be seconds, \"+Nh\", \"+Nm\" or an ISO time")
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    match = re.fullmatch(r"\+?\s*([0-9]+(?:\.[0-9]+)?)\s*([hms])", text)
    if match:
        return float(match.group(1)) * {"h": 3600.0, "m": 60.0, "s": 1.0}[match.group(2)]
    if start is None:
        raise NestTriggerRefusal(f"[{TABLE}] {key}={text!r} is a time; the run start is needed to read it")
    try:
        moment = datetime.fromisoformat(text.replace("Z", "").replace("_", "T"))
    except ValueError:
        raise NestTriggerRefusal(
            f"[{TABLE}] {key}={text!r} is not seconds, \"+Nh\", \"+Nm\" or an ISO time") from None
    return (moment - start).total_seconds()


def _positive(table: Mapping[str, Any], key: str, *, allow_zero: bool = False) -> float:
    value = table[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise NestTriggerRefusal(f"[{TABLE}] {key} must be a finite number, got {value!r}")
    if float(value) < 0 or (float(value) == 0 and not allow_zero):
        raise NestTriggerRefusal(f"[{TABLE}] {key} must be {'>= 0' if allow_zero else '> 0'}, got {value!r}")
    return float(value)


def resolve(raw: Mapping[str, Any], *, start: datetime | None = None,
            fields: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """The ``[hex_nests]`` table with defaults filled in and every key checked."""

    if not isinstance(raw, Mapping):
        raise NestTriggerRefusal(f"[{TABLE}] must be a table")
    unknown = sorted(set(raw) - KEYS)
    if unknown:
        raise NestTriggerRefusal(f"[{TABLE}] does not have key(s) {unknown}; it takes {sorted(KEYS)}")
    fields = load_field_table() if fields is None else fields
    out: dict[str, Any] = dict(DEFAULTS)
    out.update(raw)
    for key in ("max_firings", "max_live", "sustain_looks"):
        value = out[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise NestTriggerRefusal(f"[{TABLE}] {key} must be a whole number >= 1, got {value!r}")
    out["cadence_s"] = _positive(out, "cadence_s")
    for key in ("lead_back_s", "cooldown_s", "merge_km"):
        out[key] = _positive(out, key, allow_zero=True)
    out["earliest_s"] = _seconds(out.pop("earliest"), key="earliest", start=start) or 0.0
    latest = out.pop("latest")
    out["latest_s"] = _seconds(latest, key="latest", start=start)
    if out["latest_s"] is not None and out["latest_s"] <= out["earliest_s"]:
        raise NestTriggerRefusal(f"[{TABLE}] latest must come after earliest")
    out["area"] = _resolve_area(out.get("area"))
    nest = out.get("nest") or {}
    if not isinstance(nest, Mapping):
        raise NestTriggerRefusal(f"[{TABLE}] nest must be a table")
    bad = sorted(set(nest) - NEST_KEYS)
    if bad:
        raise NestTriggerRefusal(f"[{TABLE}] nest does not have key(s) {bad}; it takes {sorted(NEST_KEYS)}")
    resolved_nest = dict(NEST_DEFAULTS)
    resolved_nest.update(nest)
    for key in ("dx_km", "size_km", "hours"):
        resolved_nest[key] = _positive(resolved_nest, key)
    out["nest"] = resolved_nest
    if not out["merge_km"]:
        out["merge_km"] = 0.5 * resolved_nest["size_km"]
    blocks = out.get("when")
    if not isinstance(blocks, list) or not blocks:
        raise NestTriggerRefusal(f"[{TABLE}] needs at least one [[{TABLE}.when]] block")
    resolved_blocks = []
    for b, block in enumerate(blocks):
        if not isinstance(block, Mapping) or set(block) != {"all"} or not isinstance(block["all"], list) \
                or not block["all"]:
            raise NestTriggerRefusal(f"[{TABLE}] when[{b}] must be {{ all = [ test, ... ] }}")
        tests = []
        for t, test in enumerate(block["all"]):
            where = f"[{TABLE}] when[{b}].all[{t}]"
            if not isinstance(test, Mapping):
                raise NestTriggerRefusal(f"{where} must be a table")
            bad = sorted(set(test) - TEST_KEYS)
            if bad:
                raise NestTriggerRefusal(f"{where} does not have key(s) {bad}; it takes {sorted(TEST_KEYS)}")
            name = test.get("field")
            if name not in fields:
                raise NestTriggerRefusal(f"{where} field {name!r} is not a row of the field table; "
                                         f"rows are {sorted(fields)}")
            op = test.get("op", ">=")
            if op not in OPS:
                raise NestTriggerRefusal(f"{where} op must be one of {sorted(OPS)}, got {op!r}")
            if "value" not in test:
                raise NestTriggerRefusal(f"{where} needs value")
            value = test["value"]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise NestTriggerRefusal(f"{where} value must be a finite number, got {value!r}")
            value = float(value)
            if "cover_km2" in test and "object_km2" in test:
                raise NestTriggerRefusal(f"{where} takes cover_km2 or object_km2, not both")
            row = {"field": name, "op": op, "value": value}
            for key in ("cover_km2", "object_km2"):
                if key in test:
                    row[key] = _positive(test, key)
            tests.append(row)
        resolved_blocks.append(tests)
    out["when"] = resolved_blocks
    hold = out.get("hold")
    if hold is not None:
        if not isinstance(hold, Mapping) or set(hold) - HOLD_KEYS or (
                not {"hold_cents", "reserve_cents", "per_firing_cents", "overrun_factor"} <= set(hold)):
            raise NestTriggerRefusal(f"[{TABLE}] hold takes {sorted(HOLD_KEYS)} (hold_cents, reserve_cents, "
                                     f"per_firing_cents and overrun_factor required)")
        out["hold"] = {k: float(v) for k, v in hold.items()}
        if out["hold"]["hold_cents"] < out["hold"]["reserve_cents"] + out["hold"]["per_firing_cents"]:
            raise NestTriggerRefusal(f"[{TABLE}] hold_cents does not cover one firing plus the reserve")
    out["schema"] = SCHEMA
    return out


def _resolve_area(area: Any) -> dict[str, Any]:
    if area is None:
        return {"kind": "everywhere"}
    if not isinstance(area, Mapping) or len(area) != 1:
        raise NestTriggerRefusal(f"[{TABLE}] area must be one of {{box}}, {{circle}} or {{polygon}}")
    (kind, spec), = area.items()
    if kind == "box":
        need = ("south", "north", "west", "east")
        if not isinstance(spec, Mapping) or any(k not in spec for k in need):
            raise NestTriggerRefusal(f"[{TABLE}] area.box needs {list(need)}")
        box = {k: float(spec[k]) for k in need}
        if box["north"] <= box["south"]:
            raise NestTriggerRefusal(f"[{TABLE}] area.box north must exceed south")
        return {"kind": "box", **box}
    if kind == "circle":
        need = ("lat", "lon", "radius_km")
        if not isinstance(spec, Mapping) or any(k not in spec for k in need):
            raise NestTriggerRefusal(f"[{TABLE}] area.circle needs {list(need)}")
        return {"kind": "circle", **{k: float(spec[k]) for k in need}}
    if kind == "polygon":
        if not isinstance(spec, list) or len(spec) < 3:
            raise NestTriggerRefusal(f"[{TABLE}] area.polygon needs at least three [lat, lon] corners")
        return {"kind": "polygon", "corners": [[float(a), float(b)] for a, b in spec]}
    raise NestTriggerRefusal(f"[{TABLE}] area kind {kind!r} is not box, circle or polygon")


def _wrap(lon_deg: np.ndarray) -> np.ndarray:
    return (np.asarray(lon_deg) + 180.0) % 360.0 - 180.0


def _great_circle_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    h = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2.0 * EARTH_RADIUS_M / 1000.0 * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))


def area_mask(area: Mapping[str, Any], lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    """Which cells (degrees) lie in the table's area."""

    lat = np.asarray(lat_deg, dtype=np.float64)
    lon = _wrap(lon_deg)
    kind = area["kind"]
    if kind == "everywhere":
        return np.ones(lat.shape, dtype=bool)
    if kind == "box":
        west, east = float(_wrap(area["west"])), float(_wrap(area["east"]))
        in_lon = (lon >= west) & (lon <= east) if west <= east else (lon >= west) | (lon <= east)
        return (lat >= area["south"]) & (lat <= area["north"]) & in_lon
    if kind == "circle":
        return _great_circle_km(area["lat"], area["lon"], lat, lon) <= area["radius_km"]
    corners = np.asarray(area["corners"], dtype=np.float64)
    py, px = corners[:, 0], _wrap(corners[:, 1])
    inside = np.zeros(lat.shape, dtype=bool)
    j = len(corners) - 1
    for i in range(len(corners)):
        crosses = (py[i] > lat) != (py[j] > lat)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_at = (px[j] - px[i]) * (lat - py[i]) / (py[j] - py[i]) + px[i]
        inside ^= crosses & (lon < x_at)
        j = i
    return inside


# --------------------------------------------------------------------------- device look

@dataclass
class AreaGeometry:
    """The area's cells, their areas (km^2), unit vectors and local adjacency."""

    cells: np.ndarray          # global cell index of each area cell
    area_km2: np.ndarray       # float64
    xyz: np.ndarray            # (n, 3) float64 unit vectors
    neighbours: np.ndarray     # (n, max_edges) local indices, -1 where absent or outside the area

    @property
    def size(self) -> int:
        return int(self.cells.size)


def area_geometry(area: Mapping[str, Any], *, lat_rad: np.ndarray, lon_rad: np.ndarray,
                  area_cell: np.ndarray, cells_on_cell: np.ndarray,
                  n_edges_on_cell: np.ndarray, sphere_radius_m: float | None = None,
                  owned_cells: int | None = None) -> AreaGeometry:
    """Cut the area out of the mesh once, at arm time (host, vectorised)."""

    n = int(owned_cells if owned_cells is not None else np.asarray(lat_rad).size)
    lat = np.degrees(np.asarray(lat_rad, dtype=np.float64)[:n])
    lon = np.degrees(np.asarray(lon_rad, dtype=np.float64)[:n])
    inside = area_mask(area, lat, lon)
    cells = np.flatnonzero(inside)
    if cells.size == 0:
        raise NestTriggerRefusal(f"[{TABLE}] the area holds no cell of this mesh")
    areas = np.asarray(area_cell, dtype=np.float64)[:n]
    if sphere_radius_m is None:
        # MPAS grids are stored on the unit sphere or scaled to the Earth; the
        # total tells which, so a unit-sphere mesh is not read as m^2.
        total = float(np.sum(np.asarray(area_cell, dtype=np.float64)))
        sphere_radius_m = 1.0 if abs(total - 4 * math.pi) < 0.5 else None
    scale = (EARTH_RADIUS_M ** 2) if sphere_radius_m == 1.0 else 1.0
    area_km2 = areas[cells] * scale / 1.0e6
    local = np.full(n, -1, dtype=np.int64)
    local[cells] = np.arange(cells.size)
    coc = np.asarray(cells_on_cell, dtype=np.int64)[:n][cells]
    if coc.min(initial=0) >= 1 and coc.max(initial=0) >= n:
        coc = coc - 1  # one-based file convention
    nedges = np.asarray(n_edges_on_cell, dtype=np.int64)[:n][cells]
    slot = np.arange(coc.shape[1])[None, :]
    valid = (slot < nedges[:, None]) & (coc >= 0) & (coc < n)
    neighbours = np.where(valid, local[np.clip(coc, 0, n - 1)], -1)
    la, lo = np.radians(lat[cells]), np.radians(lon[cells])
    xyz = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=1)
    return AreaGeometry(cells=cells, area_km2=area_km2, xyz=xyz, neighbours=neighbours)


#: The biggest storm objects a look reports per block, outside every live nest
#: (a firing takes the biggest).
OBJECT_SLOTS = 8
#: Placement (see the module text): looks kept for a storm's track, the fastest
#: centroid move between two looks that is still the same storm, the speed cap
#: of the fitted motion, and how far inside an edge the storm must start.
TRACK_LOOKS = 4
MATCH_SPEED_MS = 60.0
MAX_STORM_SPEED_MS = 25.0
PLACEMENT_EDGE_FRACTION = 0.15
#: Record layout per block (float64): held, max_value_first_test, size_km2
#: (coverage outside the live nests, or the biggest object there for an object
#: test), n_cells_meeting outside the live nests, the area meeting the first
#: test everywhere in the area (live nests included: the look's whole echo),
#: then OBJECT_SLOTS x (centroid x, y, z, object_km2), biggest first,
#: zero-padded.
RECORD_HEAD = 5
RECORD_WIDTH = RECORD_HEAD + 4 * OBJECT_SLOTS


class DeviceEvaluator:
    """One look: reduce, test, label and sum on the card; one async copy home.

    ``xp`` is the array module of the parent's arrays (CuPy on the card; NumPy
    when a host test drives it).  The record of a look is ``(n_blocks,
    RECORD_WIDTH)`` float64.
    """

    def __init__(self, table: Mapping[str, Any], geometry: AreaGeometry,
                 fields: Mapping[str, Mapping[str, Any]], *, xp: Any = np) -> None:
        self.table = table
        self.geometry = geometry
        self.fields = fields
        self.xp = xp
        self._cells = xp.asarray(geometry.cells)
        self._area = xp.asarray(geometry.area_km2)
        self._xyz = xp.asarray(geometry.xyz)
        self._nbr = xp.asarray(geometry.neighbours)
        self._has = self._nbr >= 0
        self._nbr_safe = xp.where(self._has, self._nbr, 0)
        self._index = xp.arange(geometry.size, dtype=xp.int64)
        self._pending: tuple[Any, Any, dict[str, Any]] | None = None
        self._pinned = None
        self._stream = None
        self._covered_key: tuple = ()
        self._covered = None

    def covered(self, nests: Sequence[Mapping[str, Any]]) -> Any:
        """Area cells inside any live nest's square footprint (on the card; cached per nest set).

        Each nest is ``{lat, lon, size_km}``.  The footprint is the square the
        nest's grid covers, centred on its point and aligned with its local
        meridian: the cell's offset east and north of the centre on the
        tangent plane, at most half the size each way.
        """

        xp = self.xp
        key = tuple(sorted((round(float(n["lat"]), 4), round(float(n["lon"]), 4), round(float(n["size_km"]), 3))
                           for n in nests))
        if key == self._covered_key and self._covered is not None:
            return self._covered
        mask = xp.zeros(self.geometry.size, dtype=bool)
        radius_km = EARTH_RADIUS_M / 1000.0
        for lat, lon, size in key:
            la, lo = math.radians(lat), math.radians(lon)
            east = xp.asarray([-math.sin(lo), math.cos(lo), 0.0])
            north = xp.asarray([-math.sin(la) * math.cos(lo), -math.sin(la) * math.sin(lo), math.cos(la)])
            centre = xp.asarray([math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)])
            x = radius_km * (self._xyz @ east)
            y = radius_km * (self._xyz @ north)
            half = 0.5 * size
            mask |= ((self._xyz @ centre) > 0.0) & (xp.abs(x) <= half) & (xp.abs(y) <= half)
        self._covered_key, self._covered = key, mask
        return mask

    def needed_sources(self) -> list[str]:
        return sorted({self.fields[t["field"]]["source"] for block in self.table["when"] for t in block})

    def _column(self, name: str, arrays: Mapping[str, Any]) -> Any:
        row = self.fields[name]
        array = arrays[row["source"]]
        xp = self.xp
        if row.get("reduce") == "column_max":
            # (nlev, ncol): the column maximum over the area's cells only.
            return xp.max(array[:, self._cells], axis=0).astype(xp.float64)
        if row.get("reduce") == "column_min":
            return xp.min(array[:, self._cells], axis=0).astype(xp.float64)
        return array[self._cells].astype(xp.float64)

    def _label(self, mask: Any) -> Any:
        """Connected components of ``mask`` on the area adjacency (min-label sweeps)."""

        xp = self.xp
        big = self.geometry.size
        labels = xp.where(mask, self._index, big)
        nbr_mask = mask[self._nbr_safe] & self._has & mask[:, None]
        for _ in range(LABEL_SWEEPS):
            neighbour = xp.where(nbr_mask, labels[self._nbr_safe], big)
            labels = xp.minimum(labels, neighbour.min(axis=1))
            # pointer jump: follow the label to its own label
            labels = xp.where(mask, labels[xp.minimum(labels, big - 1)], big)
        return labels

    def _objects(self, meets: Any, minimum_km2: float) -> tuple[Any, Any, Any]:
        """Connected objects of ``meets``: (labels, per-label area, top-slot rows).

        The top rows are ``(OBJECT_SLOTS, 4)``: centroid x, y, z (area-weighted
        unit vectors, unnormalised) and the object's area, biggest first;
        objects under ``minimum_km2`` have area 0 there.
        """

        xp = self.xp
        n = self.geometry.size
        labels = self._label(meets)
        sizes = xp.bincount(labels, weights=self._area, minlength=n + 1)[:n]
        sums = [xp.bincount(labels, weights=self._area * self._xyz[:, k], minlength=n + 1)[:n]
                for k in range(3)]
        kept = xp.where(sizes >= minimum_km2, sizes, 0.0)
        slots = min(OBJECT_SLOTS, n)
        order = xp.argsort(-kept)[:slots]
        top = xp.stack([sums[0][order], sums[1][order], sums[2][order], kept[order]], axis=1)
        if slots < OBJECT_SLOTS:
            top = xp.concatenate([top, xp.zeros((OBJECT_SLOTS - slots, 4), dtype=top.dtype)])
        return labels, sizes, top

    def _record(self, arrays: Mapping[str, Any], nests: Sequence[Mapping[str, Any]] = ()) -> Any:
        xp = self.xp
        columns = {name: self._column(name, arrays) for name in
                   {t["field"] for block in self.table["when"] for t in block}}
        open_area = ~self.covered(nests) if nests else None
        rows = []
        for block in self.table["when"]:
            held = xp.asarray(True)
            head = None
            for test in block:
                values = columns[test["field"]]
                meets = OPS[test["op"]](values, test["value"])
                every = xp.sum(xp.where(meets, self._area, 0.0))
                if open_area is not None:
                    # Echo inside a live nest is that nest's (module text).
                    meets = meets & open_area
                top = None
                if "object_km2" in test:
                    _, sizes, top = self._objects(meets, float(test["object_km2"]))
                    size = sizes.max()
                    ok = size >= test["object_km2"]
                elif "cover_km2" in test:
                    size = xp.sum(xp.where(meets, self._area, 0.0))
                    ok = size >= test["cover_km2"]
                else:
                    size = xp.sum(xp.where(meets, self._area, 0.0))
                    ok = xp.any(meets)
                held = held & ok
                if head is None:
                    if top is None:
                        # Placement is on objects for every test type.
                        _, _, top = self._objects(meets, 0.0)
                    head = (values, meets, size, top, every)
            values, meets, size, top, every = head
            rows.append(xp.concatenate([
                xp.stack([held.astype(xp.float64), xp.max(values).astype(xp.float64),
                          xp.asarray(size, dtype=xp.float64), meets.sum().astype(xp.float64),
                          xp.asarray(every, dtype=xp.float64)]),
                top.astype(xp.float64).reshape(-1),
            ]))
        return xp.stack(rows)

    def launch(self, arrays: Mapping[str, Any], *, step: int, model_seconds: float,
               nests: Sequence[Mapping[str, Any]] = ()) -> None:
        """Queue one look; the record comes home without a device sync.

        ``nests`` are the live nests (``{lat, lon, size_km}``) whose footprints
        the look leaves out.
        """

        record = self._record(arrays, nests)
        meta = {"step": int(step), "model_seconds": float(model_seconds)}
        if self.xp is np:
            self._pending = (record, None, meta)
            return
        cp = self.xp
        if self._pinned is None:
            memory = cp.cuda.alloc_pinned_memory(record.nbytes)
            self._pinned = np.frombuffer(memory, dtype=np.float64, count=record.size).reshape(record.shape)
        stream = cp.cuda.get_current_stream()
        record.get(out=self._pinned, stream=stream, blocking=False)
        event = cp.cuda.Event(block=False, disable_timing=True)
        event.record(stream)
        self._pending = (record, event, meta)

    def ready(self) -> tuple[np.ndarray, dict[str, Any]] | None:
        """The last look's record once its copy has landed (a non-blocking query)."""

        if self._pending is None:
            return None
        record, event, meta = self._pending
        if event is not None and not event.done:
            return None
        self._pending = None
        host = np.array(record if event is None else self._pinned, dtype=np.float64, copy=True)
        return host, meta

    def drain(self) -> tuple[np.ndarray, dict[str, Any]] | None:
        """Wait for the last look (used once, at the end of the run)."""

        if self._pending is not None and self._pending[1] is not None:
            self._pending[1].synchronize()
        return self.ready()


# --------------------------------------------------------------------------- firing decision

def _latlon(vector: Sequence[float]) -> tuple[float, float]:
    x, y, z = (float(v) for v in vector)
    norm = math.sqrt(x * x + y * y + z * z)
    if norm == 0.0:
        return float("nan"), float("nan")
    return math.degrees(math.asin(z / norm)), math.degrees(math.atan2(y, x))


@dataclass
class TriggerState:
    """Host bookkeeping for the firings: rides the checkpoint as a dict."""

    table: dict[str, Any]
    run_start: str
    firings: list[dict[str, Any]] = dataclass_field(default_factory=list)
    sustain: list[int] = dataclass_field(default_factory=list)
    last_fire_s: float | None = None
    looks: int = 0
    #: The parent forecast's length (model seconds); a firing whose nest could
    #: not run its asked hours before the parent ends is not fired.
    parent_seconds: float | None = None
    #: The last TRACK_LOOKS looks' storm objects (placement reads a storm's track).
    track: list[dict[str, Any]] = dataclass_field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.sustain:
            self.sustain = [0] * len(self.table["when"])

    def to_dict(self) -> dict[str, Any]:
        return {"schema": SCHEMA, "table": self.table, "run_start": self.run_start,
                "firings": self.firings, "sustain": self.sustain,
                "last_fire_s": self.last_fire_s, "looks": self.looks, "parent_seconds": self.parent_seconds,
                "track": self.track}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TriggerState":
        return cls(table=dict(value["table"]), run_start=str(value["run_start"]),
                   firings=list(value["firings"]), sustain=list(value["sustain"]),
                   last_fire_s=value.get("last_fire_s"), looks=int(value.get("looks", 0)),
                   parent_seconds=value.get("parent_seconds"), track=list(value.get("track") or []))

    def live(self, model_seconds: float) -> list[dict[str, Any]]:
        return [f for f in self.firings
                if f["nest_start_s"] <= model_seconds < f["nest_start_s"] + f["nest"]["hours"] * 3600.0]

    def consider(self, record: np.ndarray, *, model_seconds: float,
                 fits: Callable[[Mapping[str, Any]], tuple[bool, str]] | None = None,
                 hold_left: Callable[[Mapping[str, Any]], tuple[bool, str]] | None = None,
                 others: Sequence[Mapping[str, Any]] = (),
                 ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        """Apply sustain, window, caps, cooldown, object placement, hold and card fit to one look."""

        table = self.table
        self.looks += 1
        look: dict[str, Any] = {"model_seconds": model_seconds, "blocks": []}
        candidate = None
        for b, row in enumerate(record):
            held = bool(row[0] > 0.5)
            self.sustain[b] = self.sustain[b] + 1 if held else 0
            objects = []
            for k in range(OBJECT_SLOTS):
                x, y, z, area = (float(v) for v in row[RECORD_HEAD + 4 * k: RECORD_HEAD + 4 * k + 4])
                if area > 0.0:
                    lat, lon = _latlon((x, y, z))
                    objects.append({"lat": round(lat, 4), "lon": round(lon, 4), "km2": round(area, 1)})
            look["blocks"].append({"held": held, "sustain": self.sustain[b], "max": float(row[1]),
                                   "size_km2": round(float(row[2]), 1), "cells": int(row[3]),
                                   "all_km2": round(float(row[4]), 1), "objects": objects})
            if candidate is None and self.sustain[b] >= table["sustain_looks"]:
                candidate = (b, objects, row)
        earlier = list(self.track)
        self.track = (earlier + [{"model_seconds": model_seconds,
                                  "objects": [blk["objects"] for blk in look["blocks"]]}])[-TRACK_LOOKS:]
        if candidate is None:
            look["decision"] = "not held"
            return None, look
        b, objects, row = candidate
        reasons = []
        if model_seconds < table["earliest_s"]:
            reasons.append("before earliest")
        if table["latest_s"] is not None and model_seconds > table["latest_s"]:
            reasons.append("after latest")
        if len(self.firings) >= table["max_firings"]:
            reasons.append(f"max_firings {table['max_firings']} used")
        live = self.live(model_seconds)
        if len(live) >= table["max_live"]:
            reasons.append(f"max_live {table['max_live']} alive")
        if self.last_fire_s is not None and model_seconds - self.last_fire_s < table["cooldown_s"]:
            reasons.append("cooldown")
        nest_hours = float(table["nest"]["hours"])
        if self.parent_seconds is not None and                 model_seconds - min(float(table["lead_back_s"]), model_seconds) + nest_hours * 3600.0                 > float(self.parent_seconds) + 1e-6:
            # 2026-10-23: a firing at +14 h of a 15 h parent ran a 1 h nest
            # that the site charged as the 6 h nest the user asked for.
            reasons.append(f"the parent ends before a {nest_hours:g} h nest from here would")
        # Nests this loop did not fire (placed at launch, added by hand), alive now.
        covering = [o for o in others if o["start_s"] <= model_seconds < o["end_s"]]
        if covering:
            look["other_live_nests"] = [o["name"] for o in covering]
        target = None
        for obj in objects:
            if all(float(_great_circle_km(obj["lat"], obj["lon"], nest["lat"], nest["lon"])) >= table["merge_km"]
                   for nest in live) and                all(float(_great_circle_km(obj["lat"], obj["lon"], o["lat"], o["lon"])) >= o["radius_km"]
                   for o in covering):
                target = obj
                break
        if target is None:
            reasons.append("no storm object outside the live nests" if not objects else
                           f"every storm object is inside a live nest ({len(objects)} seen)")
        if reasons:
            look["decision"] = "held but not fired: " + "; ".join(reasons)
            return None, look
        nest_spec = dict(table["nest"])
        lead_back = min(float(table["lead_back_s"]), model_seconds)
        placed = place_on_track(target, earlier, block=b, model_seconds=model_seconds,
                                size_km=float(nest_spec["size_km"]), hours=float(nest_spec["hours"]),
                                lead_back_s=lead_back)
        firing = {
            "schema": FIRING_SCHEMA, "index": len(self.firings), "block": b,
            "lat": placed["lat"], "lon": placed["lon"], "object_km2": target["km2"],
            "storm": {"lat": target["lat"], "lon": target["lon"]}, "placement": placed["placement"],
            "look_model_seconds": model_seconds,
            "nest_start_s": model_seconds - lead_back, "lead_back_s": lead_back,
            "max_value": float(row[1]), "size_km2": round(float(row[2]), 1),
            "sustain_looks": self.sustain[b], "nest": nest_spec,
        }
        # The hold first: a card reservation is a claim, made only for a firing
        # that will go.
        if hold_left is not None:
            ok, why = hold_left(firing)
            if not ok:
                reasons.append(f"hold: {why}")
        if not reasons and fits is not None:
            ok, why = fits(firing)
            if not ok:
                reasons.append(f"card fit: {why}")
        if reasons:
            look["decision"] = "held but not fired: " + "; ".join(reasons)
            return None, look
        self.firings.append(firing)
        self.last_fire_s = model_seconds
        self.sustain[b] = 0
        look["decision"] = f"fired nest {firing['index']} on a {target['km2']:.0f} km2 object"
        return firing, look


def _offset_km(lat0: float, lon0: float, lat: float, lon: float) -> tuple[float, float]:
    """East and north km of (lat, lon) from (lat0, lon0) on the local tangent plane."""

    r = EARTH_RADIUS_M / 1000.0
    la0, lo0, la, lo = (math.radians(v) for v in (lat0, lon0, lat, lon))
    x = math.cos(la) * math.sin(lo - lo0)
    y = math.cos(la0) * math.sin(la) - math.sin(la0) * math.cos(la) * math.cos(lo - lo0)
    return r * x, r * y


def _move_km(lat: float, lon: float, east_km: float, north_km: float) -> tuple[float, float]:
    r = EARTH_RADIUS_M / 1000.0
    lat2 = lat + math.degrees(north_km / r)
    lon2 = lon + math.degrees(east_km / (r * max(math.cos(math.radians(0.5 * (lat + lat2))), 1e-6)))
    return lat2, (lon2 + 180.0) % 360.0 - 180.0


def place_on_track(target: Mapping[str, Any], earlier: Sequence[Mapping[str, Any]], *, block: int,
                   model_seconds: float, size_km: float, hours: float,
                   lead_back_s: float = 0.0) -> dict[str, Any]:
    """The nest centre for ``target``: on its track over the nest's hours (see the module text).

    ``earlier`` are the looks before this one (``{model_seconds, objects: [per block]}``).
    Walking back look by look, each step takes the object nearest the storm's
    last position whose move is at most ``MATCH_SPEED_MS``; the motion is the
    least-squares fit of the matched centroids over time, capped at
    ``MAX_STORM_SPEED_MS``.  The centre is the storm's position half way
    through the nest's run, pulled back so the storm as it is now starts at
    least ``PLACEMENT_EDGE_FRACTION`` x size inside every edge (its own radius
    included).  With no earlier match, the nest is centred on the storm.
    """

    lat0, lon0 = float(target["lat"]), float(target["lon"])
    points = [(float(model_seconds), 0.0, 0.0)]
    last = (lat0, lon0, float(model_seconds))
    for look in reversed(list(earlier)):
        objects = look.get("objects") or []
        objects = objects[block] if block < len(objects) else []
        dt = last[2] - float(look["model_seconds"])
        if dt <= 0:
            continue
        best = None
        for obj in objects:
            east, north = _offset_km(last[0], last[1], float(obj["lat"]), float(obj["lon"]))
            distance = math.hypot(east, north)
            if distance * 1000.0 <= MATCH_SPEED_MS * dt and (best is None or distance < best[0]):
                best = (distance, obj)
        if best is None:
            break
        obj = best[1]
        east, north = _offset_km(lat0, lon0, float(obj["lat"]), float(obj["lon"]))
        points.append((float(look["model_seconds"]), east, north))
        last = (float(obj["lat"]), float(obj["lon"]), float(look["model_seconds"]))
    if len(points) < 2:
        return {"lat": lat0, "lon": lon0,
                "placement": {"rule": "on the storm (no earlier look of it)", "track_looks": 1}}
    t = np.asarray([p[0] for p in points]) - float(model_seconds)
    e = np.asarray([p[1] for p in points])
    n = np.asarray([p[2] for p in points])
    tc = t - t.mean()
    ue = float(np.sum(tc * (e - e.mean())) / np.sum(tc * tc))   # km per s
    un = float(np.sum(tc * (n - n.mean())) / np.sum(tc * tc))
    speed = math.hypot(ue, un) * 1000.0                         # m/s
    if speed > MAX_STORM_SPEED_MS:
        ue, un = ue * MAX_STORM_SPEED_MS / speed, un * MAX_STORM_SPEED_MS / speed
    # The storm half way through the nest's run (the nest starts lead_back before the look).
    seconds = max(0.0, 0.5 * hours * 3600.0 - float(lead_back_s))
    east, north = ue * seconds, un * seconds
    asked = math.hypot(east, north)
    radius = math.sqrt(max(float(target.get("km2", 0.0)), 0.0) / math.pi)
    room = max(0.0, 0.5 * size_km - PLACEMENT_EDGE_FRACTION * size_km - radius)
    largest = max(abs(east), abs(north))
    if largest > room and largest > 0:
        # The nest is a square: the storm stays inside along each axis.
        east, north = east * room / largest, north * room / largest
    lat, lon = _move_km(lat0, lon0, east, north)
    return {"lat": round(lat, 4), "lon": round(lon, 4),
            "placement": {"rule": "on the storm's track, half the nest's run ahead",
                          "track_looks": len(points),
                          "motion_ms": [round(ue * 1000.0, 2), round(un * 1000.0, 2)],
                          "shift_km": [round(east, 1), round(north, 1)], "asked_shift_km": round(asked, 1),
                          "room_km": round(room, 1)}}


def valid_time(run_start: str, model_seconds: float) -> str:
    start = datetime.strptime(run_start.replace("T", "_"), "%Y-%m-%d_%H:%M:%S")
    return (start + timedelta(seconds=model_seconds)).strftime("%Y-%m-%dT%H:%M:%SZ")


def append_row(path: Path, row: Mapping[str, Any]) -> None:
    """Append one whole JSON line and fsync it (readers skip a partial last line)."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


# --------------------------------------------------------------------------- loop binding

class LoopTrigger:
    """What the forecast loop holds: when to look, the evaluator, the state, the files.

    The loop calls :meth:`due` before a step (so the step forms the fields a
    look reads, e.g. ``refl10cm``), :meth:`look` after the committed step, and
    :meth:`poll` after every step.  ``arrays`` maps each field row's ``source``
    to the parent's resident array.
    """

    #: The dycore sources the route forms for this table's looks
    #: (:class:`woof.hex.nest_look_dycore.DycoreLookSources`), set by the route at arm time.
    dycore_sources: Any = None

    def __init__(self, table: Mapping[str, Any], *, run_dir: Path, run_start: str, dt_s: float,
                 geometry: AreaGeometry, xp: Any, fields: Mapping[str, Mapping[str, Any]] | None = None,
                 available_sources: Sequence[str] | None = None,
                 state: TriggerState | None = None,
                 fits: Callable[[Mapping[str, Any]], tuple[bool, str]] | None = None,
                 hold_left: Callable[[Mapping[str, Any]], tuple[bool, str]] | None = None,
                 parent_seconds: float | None = None) -> None:
        self.fields = load_field_table() if fields is None else dict(fields)
        self.table = dict(table)
        self.run_dir = Path(run_dir)
        self.run_start = run_start
        self.dt_s = float(dt_s)
        every = self.table["cadence_s"] / self.dt_s
        self.every_steps = max(1, int(round(every)))
        self.evaluator = DeviceEvaluator(self.table, geometry, self.fields, xp=xp)
        if available_sources is not None:
            missing = sorted(set(self.evaluator.needed_sources()) - set(available_sources))
            if missing:
                names = sorted(t["field"] for block in self.table["when"] for t in block
                               if self.fields[t["field"]]["source"] in missing)
                raise NestTriggerRefusal(
                    f"[{TABLE}] field(s) {names} read {missing}, which this physics route does not form on the "
                    f"card (it forms {sorted(available_sources)}); a test on them could never fire")
        self.state = state or TriggerState(table=self.table, run_start=run_start)
        if parent_seconds is not None:
            self.state.parent_seconds = float(parent_seconds)
        if self.table.get("hold"):
            from .nest_budget import HOLD_NAME, HOLD_SCHEMA

            hold_path = self.run_dir / HOLD_NAME
            hold_path.parent.mkdir(parents=True, exist_ok=True)
            hold_path.write_text(json.dumps({"schema": HOLD_SCHEMA, **self.table["hold"]}, sort_keys=True) + "\n",
                                 encoding="utf-8")
        self.fits = fits
        self.hold_left = hold_left
        self.look_seconds = 0.0
        self.looks_launched = 0
        append_row(self.run_dir / LOOKS_NAME, {"armed": True, "schema": SCHEMA, "table": self.table,
                                               "area_cells": geometry.size,
                                               "area_km2": round(float(geometry.area_km2.sum()), 1),
                                               "every_steps": self.every_steps})

    def due(self, step: int) -> bool:
        return step % self.every_steps == 0

    def needed_sources(self) -> list[str]:
        """The sources this table's tests read (a route forms only these on a look step)."""

        return self.evaluator.needed_sources()

    def live_footprints(self, model_seconds: float) -> list[dict[str, Any]]:
        """Every live nest on the parent at ``model_seconds``: this loop's firings and add-nest's list."""

        own = [{"lat": f["lat"], "lon": f["lon"], "size_km": f["nest"]["size_km"]}
               for f in self.state.live(model_seconds)]
        others = [{"lat": o["lat"], "lon": o["lon"], "size_km": 2.0 * o["radius_km"]}
                  for o in read_other_nests(self.run_dir, self.run_start)
                  if o["start_s"] <= model_seconds < o["end_s"]]
        return own + others

    def look(self, step: int, arrays: Mapping[str, Any]) -> None:
        import time as _time
        mark = _time.perf_counter()
        model_seconds = step * self.dt_s
        self.evaluator.launch(arrays, step=step, model_seconds=model_seconds,
                              nests=self.live_footprints(model_seconds))
        self.look_seconds += _time.perf_counter() - mark
        self.looks_launched += 1

    def poll(self, *, final: bool = False) -> dict[str, Any] | None:
        got = self.evaluator.drain() if final else self.evaluator.ready()
        if got is None:
            return None
        record, meta = got
        firing, look = self.state.consider(record, model_seconds=meta["model_seconds"],
                                           fits=self.fits, hold_left=self.hold_left,
                                           others=read_other_nests(self.run_dir, self.run_start))
        look["step"] = meta["step"]
        look["valid"] = valid_time(self.run_start, meta["model_seconds"])
        append_row(self.run_dir / LOOKS_NAME, look)
        # Firings used, sustain counters and cooldown ride beside the run's
        # restarts, so a run resumed after a reclaim neither re-fires a nest
        # nor forgets one (read back by the loop's arm on a resumed start).
        state = self.run_dir / STATE_NAME
        temporary = state.with_name(f".{state.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps({"step": meta["step"], "state": self.state.to_dict()}, sort_keys=True),
                             encoding="utf-8")
        os.replace(temporary, state)
        if firing is not None:
            firing = {**firing, "look_valid": look["valid"],
                      "nest_start_valid": valid_time(self.run_start, firing["nest_start_s"]),
                      "written_unix": round(__import__("time").time(), 3)}
            append_row(self.run_dir / FIRINGS_NAME, firing)
        return firing

    def receipt(self) -> dict[str, Any]:
        return {"schema": SCHEMA, "looks": self.looks_launched, "every_steps": self.every_steps,
                "launch_host_seconds": round(self.look_seconds, 4),
                "firings": self.state.firings, "state": self.state.to_dict()}


def read_other_nests(run_dir: Path, run_start: str) -> list[dict[str, Any]]:
    """The live nests on this parent that no firing of this loop started (see the module text).

    Each is ``{name, lat, lon, radius_km, start_s, end_s}`` in this run's model
    seconds; a nest whose end row is written is gone.  A small host file read once
    per look (every ``cadence_s`` of model time), off the card.
    """

    try:
        text = (Path(run_dir) / LIVE_NESTS_NAME).read_text(encoding="utf-8")
    except OSError:
        return []
    start = datetime.strptime(run_start.replace("T", "_"), "%Y-%m-%d_%H:%M:%S")
    nests: dict[str, dict[str, Any]] = {}
    for line in text.splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        name = str(row.get("name"))
        if row.get("event") == "ended":
            nests.pop(name, None)
            continue
        if row.get("event") != "started" or row.get("firing") is not None:
            continue  # this loop's own firings are in its state already
        try:
            begin = datetime.strptime(str(row["start_valid"]).replace("T", "_").rstrip("Z"), "%Y-%m-%d_%H:%M:%S")
            start_s = (begin - start).total_seconds()
            nests[name] = {"name": name, "lat": float(row["lat"]), "lon": float(row["lon"]),
                           "radius_km": float(row["size_km"]) / 2.0, "start_s": start_s,
                           "end_s": start_s + float(row["hours"]) * 3600.0}
        except (KeyError, TypeError, ValueError):
            continue
    return list(nests.values())


def note_live_nest(run_dir: Path, row: Mapping[str, Any]) -> None:
    """add-nest's side of :data:`LIVE_NESTS_NAME`: one whole line per start or end."""

    append_row(Path(run_dir) / LIVE_NESTS_NAME, row)


def read_firings(run_dir: Path) -> list[dict[str, Any]]:
    """Every whole firing row written so far (a partial last line is skipped)."""

    try:
        text = (Path(run_dir) / FIRINGS_NAME).read_text(encoding="utf-8")
    except OSError:
        return []
    rows = []
    for line in text.splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def load_table_file(path: Path, *, start: datetime | None = None) -> dict[str, Any]:
    """Read ``[hex_nests]`` from a TOML or JSON file."""

    text = Path(path).read_text(encoding="utf-8")
    if str(path).endswith(".json"):
        raw = json.loads(text)
    else:
        try:
            import tomllib
        except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
            import tomli as tomllib  # type: ignore[no-redef]
        raw = tomllib.loads(text)
    if TABLE in raw:
        raw = raw[TABLE]
    return resolve(raw, start=start)


# --------------------------------------------------------------------------- arming (both physics routes)

#: The forecast door carries ``--nest-triggers FILE`` to the step loop of either
#: physics route as this variable (it sets it for its own driver call only).
NEST_TRIGGERS_ENV = "WOOF_HEX_NEST_TRIGGERS"


def arm_loop(table_path: Path | str, *, run_dir: Path, start_text: str, start_step: int, dt_s: float,
             parent_seconds: float, mesh: Any, owned_cells: int, available_sources: Sequence[str],
             cp: Any) -> LoopTrigger:
    """The armed look for one forecast's step loop: the ArWen-physics driver and the MPAS-physics route both call this.

    THE BREAKAGE ONE ARMING PREVENTS: until 2026-10-24 the arming lived inside
    the WOOF driver only, so a ``--nest-triggers`` run on the MPAS-physics
    route finished rc 0 with the loop never armed, no look written and nothing
    refused (parity run group 3, capability 11).

    ``mesh`` is the run's :class:`woof.hex.mesh.Mesh`; ``owned_cells`` the cells
    the look's arrays cover (the cells this process owns).  ``available_sources``
    are the arrays the route forms on the card; a table reading another is
    refused here by :class:`LoopTrigger`, before the first step.  A resumed run
    (``start_step`` > 0) carries its firings, sustain counters and cooldown.
    """

    from .card_ledger import CardLedger
    from . import nest_budget

    start = datetime.strptime(start_text, "%Y-%m-%d_%H:%M:%S")
    table = load_table_file(Path(table_path), start=start)
    geometry = area_geometry(
        table["area"], lat_rad=np.asarray(mesh.latCell), lon_rad=np.asarray(mesh.lonCell),
        area_cell=np.asarray(mesh.areaCell), cells_on_cell=np.asarray(mesh.cellsOnCell),
        n_edges_on_cell=np.asarray(mesh.nEdgesOnCell), owned_cells=int(owned_cells))
    state = None
    state_path = Path(run_dir) / STATE_NAME
    if start_step > 0 and state_path.is_file():
        state = TriggerState.from_dict(json.loads(state_path.read_text(encoding="utf-8"))["state"])
    # The two fire-time checks (woof.hex.nest_budget): the remaining hold must
    # cover one more nest (the site's prices, <out>/nest-hold.json), and the
    # nest's memory is RESERVED on the card atomically, so two quick firings
    # cannot both pass.  Free memory is the card's own count (every process),
    # read only for a firing that passed every other check.
    device = int(cp.cuda.runtime.getDevice())
    ledger = CardLedger(device=device, free_bytes=lambda: int(cp.cuda.runtime.memGetInfo()[0]))
    holder: dict[str, Any] = {}
    trigger = LoopTrigger(
        table, run_dir=Path(run_dir), run_start=start_text, dt_s=float(dt_s),
        geometry=geometry, xp=cp, available_sources=list(available_sources), state=state,
        parent_seconds=float(parent_seconds),
        fits=nest_budget.card_fit_hook(Path(run_dir), ledger),
        hold_left=nest_budget.hold_left_hook(Path(run_dir), lambda: holder["trigger"].state.firings))
    holder["trigger"] = trigger
    print(json.dumps({"nest_triggers": {"armed": True, "table": str(table_path),
                                        "area_cells": geometry.size,
                                        "every_steps": trigger.every_steps}}), flush=True)
    return trigger
