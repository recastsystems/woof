"""Starting a fine grid in the middle of its parent's window.

THE COST THIS REMOVES.  A swath is placed for weather the coarse forecast says
will be there at hour 12, not at hour 0.  Until now the only initial condition
a limited-area cull could have was its parent's INIT -- the state at hour 0 --
so covering that threat meant integrating the fine grid from hour 0 and
throwing the first twelve hours away.  On the 1.35x cut of the 4.6 km cascade
that is 6 h of fine forecast spent on an atmosphere nobody placed a grid for,
per cycle, per swath, and it grows with every hour of lead.

THE SEAM THAT MADE IT IMPOSSIBLE, and it is the delayed-start seam.  The parent's own
history stream is a PRODUCT stream, not a restart stream.  Measured on a real
121,182-cell parent frame: it publishes ``theta``, ``rho``, ``qv``, the five
hydrometeors, ``normal_u``, ``w``, ``pressure``, the surface fields and the
soil columns -- but it carries **no** ``xtime``, **no** ``zgrid``/``zz``, **no**
base state, and none of the mesh connectivity.  It cannot be handed to
anything as an initial condition, and that is why a fine grid could only ever
start where its parent started.

THE ROUTE THIS SHIPS.  Everything the history stream is missing is
TIME-INVARIANT, and all of it is in the parent's own init: the vertical grid,
the base state, the terrain, the geography, the connectivity.  So a mid-window
initial condition is the culled init with its PROGNOSTIC state replaced by the
parent's state at the hour the swath wanted, and its clock moved to match.

* the cull is the supported route and stays the supported route -- grid,
  static and init out of the parent in about a second, against 775 s to
  build that parent's own init (v4.75.121182, 2026-08-26) rather than a
  native regional init;
* the parent's history frame is gathered onto the child's cells by EXACT
  COORDINATE MATCH.  A cull moves no cell centre, so a child cell's
  ``latCell``/``lonCell`` are the parent's float64 bits unchanged; the map is
  built on those bits as an integer pair and **a miss is a refusal**, never a
  nearest neighbour.  Folding the two words into one hash would admit a
  collision, and two different cells reading as one another is exactly the
  failure a state transplant cannot survive;
* no engine change and no second grammar: this is a gather, in Python, over
  files the shipped chain already writes.

WHAT A DELAYED START DOES NOT CARRY, measured against the two files rather
than asserted, and reported on every composition it makes:

* **the ice-phase hydrometeors.** The parent publishes ``qi``/``qs``/``qg``
  and the init has no slot for them, so the fine grid starts with the liquid
  condensate the parent had and no frozen condensate at all.  A run started
  inside an active ice cloud re-forms it, and the first hour is a spin-up.
* **the land-surface memory the history stream does not publish** -- snow
  cover, sea ice, canopy state.  Soil temperature, soil moisture, soil liquid
  water, skin temperature, 2 m temperature, snow water equivalent and snow
  depth ARE carried, because the parent publishes them.
* **turbulent kinetic energy**, for the same reason.

Every one of those is named in the receipt this module writes, per field, with
whether it was carried, and from which file.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .errors import DelayedStartRefusal
from .land_state import LAND_FIELDS, LAND_BY_NAME, collect_land, write_land, land_receipt

XTIME_FORMAT = "%Y-%m-%d_%H:%M:%S"


def _stamp_slot(variable: Any) -> Any:
    """The index that addresses a whole clock string in ``variable``.

    MPAS writes ``xtime`` as ``(Time, StrLen)``, so the stamp is row 0; the
    port's inits (hex >= 0.4.6) write ``initial_time`` as ``(StrLen,)``, so
    the stamp is the whole variable.  ``[0]`` on the 1-D shape addresses ONE
    CHARACTER: the transplant wrote only that character and left
    ``' 025-05-19_06:00:00'``, and every member started from it stopped with
    "Invalid isoformat string".
    """

    return 0 if len(variable.dimensions) > 1 else slice(None)


#: What a mid-window state is made of: ``(init variable, history variable)``.
#:
#: The history variable is the name the hex history stream publishes (the
#: ``output`` stream: ``u``, ``t2m``, ``skintemp``, ``sh2o``, ``snow``,
#: ``snowh``); :data:`HISTORY_ALIASES` lists older names read when that one is
#: absent.  Nothing here is converted: both streams carry the same physical
#: quantity in the same units, and the loader that reads the composed init
#: derives ``rho_zz``/``theta_m`` from the public ``rho``/``theta``/``qv``
#: exactly as it does for a parent init.
#:
#: Breakage this prevents: the table read ``normal_u``, ``t2`` and ``tsk`` and
#: had no ``sh2o``/``snow``/``snowh``, while the history publishes ``u``,
#: ``t2m``, ``skintemp``, ``sh2o``, ``snow`` and ``snowh``.  Every composed
#: init kept its template's (the cold start's) wind, 2 m and skin temperature,
#: soil liquid water and snow: a cycled 24-member ensemble's backgrounds
#: (2026-10-28, 07-08 18Z) had 0 K 2 m temperature spread while the history
#: differed from the init by up to 29 K (skin temperature 41 K).
PROGNOSTIC_TRANSPLANT: tuple[tuple[str, str], ...] = (
    ("u", "u"),
    ("w", "w"),
    ("theta", "theta"),
    ("rho", "rho"),
    ("qv", "qv"),
    ("qc", "qc"),
    ("qr", "qr"),
    ("surface_pressure", "surface_pressure"),
    ("u10", "u10"),
    ("v10", "v10"),
) + tuple((f.name, f.name) for f in LAND_FIELDS)

#: Older history names, read only when the history lacks the name above.
HISTORY_ALIASES: dict[str, tuple[str, ...]] = {
    "u": ("normal_u",),
    **{f.name: f.aliases for f in LAND_FIELDS if f.aliases},
}

#: What the parent HAS and the init has no slot for, so a delayed start cannot
#: carry it.  Named here rather than discovered later: each entry is a real
#: spin-up cost the receipt reports.
NOT_CARRIED_NO_INIT_SLOT: tuple[tuple[str, str], ...] = (
    ("qi", "cloud ice: the parent publishes it and the init stream has no "
           "slot, so a grid started inside an ice cloud re-forms it"),
    ("qs", "snow mixing ratio, same reason"),
    ("qg", "graupel mixing ratio, same reason"),
)

#: What the init HAS and the parent's history stream does not publish, so the
#: composed state keeps the value from the parent's own init hour.
NOT_CARRIED_NOT_PUBLISHED: tuple[tuple[str, str], ...] = (
    ("seaice", "sea-ice fraction, same reason"),
    ("xice", "sea-ice fraction on the land mask, same reason"),
    ("sst", "sea-surface temperature, same reason"),
    ("tke", "turbulent kinetic energy: not published, so the PBL re-spins it"),
    ("relhum", "a diagnostic the model recomputes on its first step"),
    ("rh2", "a diagnostic the model recomputes on its first step"),
    ("precipw", "a diagnostic the model recomputes on its first step"),
)


@dataclass
class TransplantReport:
    """What a composition actually did, field by field."""

    parent_history: str
    parent_history_sha256: str
    parent_grid: str
    child_grid: str
    init_path: str
    valid_time: str
    init_time_before: str
    cells_matched: int
    edges_matched: int
    carried: list[dict[str, Any]] = field(default_factory=list)
    not_carried: list[dict[str, Any]] = field(default_factory=list)
    land_state: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "gpuwm-hex.delayed-start/v1",
            "parent_history": self.parent_history,
            "parent_history_sha256": self.parent_history_sha256,
            "parent_grid": self.parent_grid,
            "child_grid": self.child_grid,
            "init_path": self.init_path,
            "valid_time": self.valid_time,
            "init_time_before": self.init_time_before,
            "cells_matched": self.cells_matched,
            "edges_matched": self.edges_matched,
            "carried": self.carried,
            "land_state": self.land_state,
            "not_carried": self.not_carried,
            "what_this_does_not_carry": (
                "A delayed start replaces the PROGNOSTIC state of a culled "
                "init with the parent's state at a later hour.  Everything "
                "outside the strict land contract that the history does not publish stays at the "
                "parent's own init hour, and everything the init stream has "
                "no slot for is not carried at all.  Both lists are above, "
                "per field, with the reason.  The largest of them is the "
                "ice-phase condensate."
            ),
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def coordinate_keys(lat: np.ndarray, lon: np.ndarray) -> list[tuple[int, int]]:
    """The exact bits of each element's centre, as an integer PAIR.

    Bit equality, not a tolerance, and a pair rather than a mixed hash --
    the same convention ``tools/measure_domain_size_agreement.py`` uses, and
    for the same reason: folding two words into one integer would be faster
    and would admit a collision, and two different cells reading as the same
    cell is precisely the failure a state transplant cannot survive.
    """

    lat64 = np.ascontiguousarray(np.asarray(lat, dtype=np.float64))
    lon64 = np.ascontiguousarray(np.asarray(lon, dtype=np.float64))
    return list(zip(lat64.view(np.int64).tolist(), lon64.view(np.int64).tolist()))


def parent_index_map(
    child_lat: np.ndarray,
    child_lon: np.ndarray,
    parent_lat: np.ndarray,
    parent_lon: np.ndarray,
    *,
    what: str,
) -> np.ndarray:
    """Which parent element each child element IS.  A miss refuses."""

    lookup: dict[tuple[int, int], int] = {}
    for index, key in enumerate(coordinate_keys(parent_lat, parent_lon)):
        lookup.setdefault(key, index)
    child_keys = coordinate_keys(child_lat, child_lon)
    mapped = np.empty(len(child_keys), dtype=np.int64)
    missing = 0
    first_missing = -1
    for index, key in enumerate(child_keys):
        found = lookup.get(key)
        if found is None:
            missing += 1
            if first_missing < 0:
                first_missing = index
            continue
        mapped[index] = found
    if missing:
        raise DelayedStartRefusal(
            f"a delayed start cannot be composed: {missing} of "
            f"{len(child_keys)} {what} in the cull do not appear in the "
            f"parent grid by exact coordinate bits (first at child index "
            f"{first_missing}).  A cull moves no cell centre, so every child "
            f"{what[:-1]} must BE a parent one; a miss means these two files "
            f"are not a parent and its cull, and gathering the parent's state "
            f"onto them would put one column's atmosphere on another column"
        )
    if np.unique(mapped).size != mapped.size:
        raise DelayedStartRefusal(
            f"a delayed start cannot be composed: two {what} in the cull map "
            f"to the same parent element, so the parent grid carries "
            f"duplicate coordinates and the gather is not a permutation"
        )
    return mapped


def frame_valid_time(path: Path) -> datetime | None:
    """The valid time a history frame declares, or ``None`` if it declares none.

    THE SEAM, MEASURED: the port's history stream carries no ``xtime``
    variable.  It is a product stream and its valid time lives in its file
    NAME and in the run receipt beside it.  This reads the variable when a
    stream grows one and returns ``None`` otherwise, so the caller states the
    time it is composing for rather than inferring one that is not there.
    """

    from netCDF4 import Dataset

    with Dataset(str(path)) as dataset:
        dataset.set_auto_maskandscale(False)
        if "xtime" not in dataset.variables:
            return None
        variable = dataset.variables["xtime"]
        raw = np.asarray(variable[_stamp_slot(variable)])
        text = (
            raw.tobytes().decode("utf-8", "replace")
            if raw.dtype.kind in ("S", "U")
            else bytes(raw).decode("utf-8", "replace")
        )
    return datetime.strptime(text.strip()[:19], XTIME_FORMAT)


def compose_mid_window_init(
    *,
    child_init: Path,
    child_grid: Path,
    parent_grid: Path,
    parent_history: Path,
    valid_time: datetime,
    receipt_path: Path | None = None,
) -> TransplantReport:
    """Move a culled init's clock and state to ``valid_time``, in place.

    ``child_init`` is modified.  It is the cull's own file and the cascade
    made it seconds earlier; the caller owns it.
    """

    from netCDF4 import Dataset

    for path, flag in (
        (child_init, "the culled init"),
        (child_grid, "the culled grid"),
        (parent_grid, "the parent grid"),
        (parent_history, "the parent history frame"),
    ):
        if not Path(path).is_file():
            raise DelayedStartRefusal(
                f"a delayed start needs {flag} and {path} is not a file"
            )

    with Dataset(str(child_grid)) as grid:
        child_lat_cell = np.asarray(grid.variables["latCell"][:], np.float64)
        child_lon_cell = np.asarray(grid.variables["lonCell"][:], np.float64)
        child_lat_edge = np.asarray(grid.variables["latEdge"][:], np.float64)
        child_lon_edge = np.asarray(grid.variables["lonEdge"][:], np.float64)
    with Dataset(str(parent_grid)) as grid:
        parent_lat_cell = np.asarray(grid.variables["latCell"][:], np.float64)
        parent_lon_cell = np.asarray(grid.variables["lonCell"][:], np.float64)
        parent_lat_edge = np.asarray(grid.variables["latEdge"][:], np.float64)
        parent_lon_edge = np.asarray(grid.variables["lonEdge"][:], np.float64)

    cell_map = parent_index_map(
        child_lat_cell, child_lon_cell,
        parent_lat_cell, parent_lon_cell,
        what="cells",
    )
    edge_map = parent_index_map(
        child_lat_edge, child_lon_edge,
        parent_lat_edge, parent_lon_edge,
        what="edges",
    )

    carried: list[dict[str, Any]] = []
    not_carried: list[dict[str, Any]] = []
    stamp = valid_time.strftime(XTIME_FORMAT)

    # Validate the entire land payload before touching the child. The shared
    # contract also creates missing canopy/snow-age slots in older templates.
    with Dataset(str(parent_history)) as h, Dataset(str(child_init)) as c:
        h.set_auto_maskandscale(False)
        c.set_auto_maskandscale(False)
        if (len(h.dimensions['nCells']) != parent_lat_cell.size
                or len(h.dimensions['nEdges']) != parent_lat_edge.size):
            raise DelayedStartRefusal('parent history is from a different mesh: grid dimensions differ')
        try:
            land, land_pairs = collect_land(h, c, cell_map)
        except ValueError as exc:
            raise DelayedStartRefusal(str(exc)) from exc
    write_land(child_init, land)
    carried.extend({"field": n, "from": src, "on": "cells", "shape": list(land[n].shape)}
                   for n, src in land_pairs)

    with Dataset(str(parent_history)) as history, Dataset(str(child_init), "a") as init:
        history.set_auto_maskandscale(False)
        init.set_auto_maskandscale(False)
        before = "unrecorded"
        if "xtime" in init.variables:
            variable = init.variables["xtime"]
            raw = np.asarray(variable[_stamp_slot(variable)])
            before = (
                raw.tobytes().decode("utf-8", "replace")
                if raw.dtype.kind in ("S", "U")
                else bytes(raw).decode("utf-8", "replace")
            ).strip()

        parent_cells = int(history.dimensions["nCells"].size)
        parent_edges = int(history.dimensions["nEdges"].size)
        if parent_cells != parent_lat_cell.size or parent_edges != parent_lat_edge.size:
            raise DelayedStartRefusal(
                f"a delayed start cannot be composed: the parent history "
                f"frame carries {parent_cells} cells and {parent_edges} "
                f"edges while the parent grid carries {parent_lat_cell.size} "
                f"and {parent_lat_edge.size}.  The gather is by parent index, "
                f"so a frame from a different mesh would read another mesh's "
                f"rows"
            )

        for init_name, history_name in PROGNOSTIC_TRANSPLANT:
            if init_name in LAND_BY_NAME:
                continue
            if init_name not in init.variables:
                not_carried.append({
                    "field": init_name,
                    "reason": "the culled init has no such variable",
                })
                continue
            if history_name not in history.variables:
                history_name = next(
                    (alias for alias in HISTORY_ALIASES.get(init_name, ())
                     if alias in history.variables),
                    history_name,
                )
            if history_name not in history.variables:
                not_carried.append({
                    "field": init_name,
                    "from": history_name,
                    "reason": (
                        "the parent's history stream does not publish it, so "
                        "the parent's own init-hour value stays"
                    ),
                })
                continue
            source = np.asarray(history.variables[history_name][0])
            target = init.variables[init_name]
            on_edges = "nEdges" in target.dimensions
            index = edge_map if on_edges else cell_map
            gathered = source[index, ...]
            wanted = target.shape
            payload = gathered[None, ...] if wanted[0] == 1 and len(wanted) == gathered.ndim + 1 else gathered
            if tuple(payload.shape) != tuple(wanted):
                raise DelayedStartRefusal(
                    f"a delayed start cannot be composed: {init_name} in the "
                    f"init has shape {tuple(wanted)} and the gathered "
                    f"{history_name} is {tuple(payload.shape)}.  Writing it "
                    f"would silently broadcast one column's profile across "
                    f"another's"
                )
            target[...] = payload.astype(target.dtype, copy=False)
            carried.append({
                "field": init_name,
                "from": history_name,
                "on": "edges" if on_edges else "cells",
                "shape": list(wanted),
            })

        for name, reason in NOT_CARRIED_NO_INIT_SLOT:
            not_carried.append({
                "field": name,
                "present_in_parent_history": name in history.variables,
                "reason": reason,
            })
        for name, reason in NOT_CARRIED_NOT_PUBLISHED:
            if name in init.variables:
                not_carried.append({
                    "field": name,
                    "present_in_parent_history": name in history.variables,
                    "reason": reason,
                })

        # THE CLOCK, LAST, so a composition that refused above leaves a file
        # whose stamp still says what its state is.
        #
        # AND AN INIT'S CLOCK LIVES IN THREE PLACES, which is worth stating
        # because moving two of the three is worse than moving none.  The
        # ``xtime`` variable is what a reader of the record sees; the
        # ``initial_time`` variable is what the history stream stamps its
        # frames against; and the ``config_start_time`` GLOBAL ATTRIBUTE is
        # what the forecast door asserts ``--start-time`` against.  MEASURED:
        # a transplant that moved the first two produced a file whose own
        # attribute contradicted its own variable, and the door refused it by
        # name -- "--start-time '..._12:00:00' disagrees with the init's
        # config_start_time '..._06:00:00'".  That refusal was right and its
        # cause was here.  All three move together now, and the one they
        # replaced is recorded so a reader can see what the file was.
        for name in ("xtime", "initial_time"):
            if name in init.variables:
                variable = init.variables[name]
                width = int(variable.shape[-1])
                variable[_stamp_slot(variable)] = np.array(
                    list(stamp.ljust(width)[:width]), dtype="S1"
                )
        attribute_before = None
        if "config_start_time" in init.ncattrs():
            attribute_before = str(init.getncattr("config_start_time"))
        init.setncattr("config_start_time", stamp)
        init.setncattr("gpuwm_hex_delayed_start_valid_time", stamp)
        init.setncattr(
            "gpuwm_hex_delayed_start_config_start_time_before",
            attribute_before if attribute_before is not None else "(absent)",
        )
        init.setncattr("gpuwm_hex_delayed_start_parent_history", str(parent_history))

    report = TransplantReport(
        parent_history=str(parent_history),
        parent_history_sha256=_sha256(Path(parent_history)),
        parent_grid=str(parent_grid),
        child_grid=str(child_grid),
        init_path=str(child_init),
        valid_time=stamp,
        init_time_before=before,
        cells_matched=int(cell_map.size),
        edges_matched=int(edge_map.size),
        carried=carried,
        not_carried=not_carried,
        land_state=land_receipt(land),
    )
    if receipt_path is not None:
        Path(receipt_path).parent.mkdir(parents=True, exist_ok=True)
        Path(receipt_path).write_text(
            json.dumps(report.as_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    return report


__all__ = [
    "HISTORY_ALIASES",
    "NOT_CARRIED_NOT_PUBLISHED",
    "NOT_CARRIED_NO_INIT_SLOT",
    "PROGNOSTIC_TRANSPLANT",
    "TransplantReport",
    "XTIME_FORMAT",
    "compose_mid_window_init",
    "coordinate_keys",
    "frame_valid_time",
    "parent_index_map",
]
