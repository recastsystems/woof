"""A moving fine region on the MPAS suite route: the move hooks of ``runtime_v842``.

The WOOF route's moving hooks live in its one-card class executor
(``woof.hex.drivers.fullstep_lts_forecast_v841``).  This module gives the MPAS-A v8.4.2 suite route
(``--physics-backend mpasphys``, and the RRFS fork's suite through the same route) the same five
hooks, driven by the same ``WOOF_HEX_MOVING_*`` switches the moving controller sets
(``woof.hex.moving_run``), so ``woof hex moving run`` and ``moving fixed`` work unchanged:

* **stop request** -- at every history frame (``--history-every-minutes`` of model time from the case
  start) the forecast writes a tracker frame, tells the controller (``woof.hex.moving_control``) and
  leaves its step loop when asked to stop there;
* **carrier export** -- the dycore state and the native pools' physics state, persistent scheme state
  included, are sealed in one carrier (``WOOF_HEX_MOVING_CHECKPOINT_OUT``, and recovery seals);
* **remapped resume** -- the next member continues from the carrier: the atmosphere through the same
  conservative, balanced remap as the WOOF route (``woof.hex.moving_state.remap_atmosphere``), every
  carried physics field through the same overlap operator (atmospheric fields in physical height,
  surface and soil state from one coherent donor column of the same land/water/ice role);
* **clock offset** -- a resumed segment is a native MPAS restart at the carrier's valid time
  (``config_start_time``, ``config_do_restart``), so its physics initialises as MPAS restarts it and a
  move does not reset convection, PBL or land memory;
* **post-move shock series** -- max |dpsfc/dt| and max |w| after every step (``WOOF_HEX_MOVING_SHOCK``).

Which fields are carried is read from the Registry, never listed by scheme: every cell field of a
physics pool that is in the restart stream (what MPAS itself needs to continue), plus the single-level
time-dependent diagnostics (so output across a move stays continuous), less the lower-boundary
geography each member brings from its own static and init.  With no ``WOOF_HEX_MOVING_*`` switch set
nothing here is imported.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import pickle
import time
from types import SimpleNamespace
from typing import Any

import numpy as np

SCHEMA = "gpuwm-hex.moving-boundary-mpas/v1"
TIME_FORMAT = "%Y-%m-%d_%H:%M:%S"
STATE_FIELDS = ("rho", "rho_theta", "rho_u", "rho_w", "scalars")
SAVED_FIELDS = ("theta_m", "exner", "density_perturbation", "rho_theta_perturbation",
                "pressure_perturbation", "normal_velocity", "vertical_velocity")

#: Pools the dycore owns (state, its diagnostics and tendencies) or that hold geometry: never carried as
#: physics.  The dycore's own state crosses a move through the atmospheric remap.
DYCORE_POOLS = frozenset(("mesh", "state", "diag", "tend", "lbc", "halo_scratch"))
#: Dycore-pool fields that are physics memory: the lagged microphysics heating (restart state,
#: mpas_atm_time_integration.F:4436-4437).
CARRIED_DYCORE_FIELDS = frozenset({("tend", "rt_diabatic_tend")})
#: Lower-boundary data each member brings from its own static and init (Registry.xml sfc_input): the
#: geography of the new mesh, never the old mesh's.
GEOGRAPHY = frozenset((
    "isltyp", "ivgtyp", "lu_index", "landmask", "soilcat_top", "soilcat_bot", "shdmin", "shdmax", "snoalb",
    "ter", "albedo12m", "greenfrac", "dzs", "sfc_albbck", "sst", "tmn", "vegfra", "seaice", "xice", "xland",
    "var2d", "con", "oa1", "oa2", "oa3", "oa4", "ol1", "ol2", "ol3", "ol4", "isice_lu", "iswater_lu",
    "mminlu", "lakemask"))
#: Vertical axes remapped in physical height; any other leading axis (soil, ozone, lags) is horizontal.
LAYER_AXIS, INTERFACE_AXIS = "nVertLevels", "nVertLevelsP1"
#: Zonal/meridional pairs carried as one earth-Cartesian vector (moving_physics.carry_horizontal_vector).
VECTOR_PAIRS = (("rucuten", "rvcuten"), ("rublten", "rvblten"), ("rubldiff", "rvbldiff"),
                ("dtaux3d", "dtauy3d"), ("tend_uzonal", "tend_umerid"))
#: Surface roles a coherent donor column keeps (moving_physics.LAND_CATEGORIES on MPAS's names).
CATEGORY_FIELDS = ("xland", "xice", "isltyp", "ivgtyp")


# --------------------------------------------------------------------------- the carried field set

def _has_time(layout, row) -> bool:
    element = layout.root.find(f"./var_struct[@name='{row['pool']}']/*[@name='{row['name']}']")
    return element is not None and "Time" in element.get("dimensions", "").split()


def stream_members(layout, name: str) -> set[str]:
    """Registry names a stream reads or writes, packages and nested streams expanded (Registry.xml:498-1351)."""
    definitions = {s.get("name"): s for s in layout.root.findall("./streams/stream")}
    members: set[str] = set()

    def include(stream, seen):
        for item in stream:
            if not layout.enabled(item):
                continue
            if item.tag in ("var", "var_array"):
                members.add(item.get("name"))
            elif item.tag == "var_struct":
                members.update(f["name"] for f in layout.fields if f["pool"] == item.get("name"))
            elif item.tag == "stream" and item.get("name") not in seen:
                include(definitions[item.get("name")], seen | {item.get("name")})

    if name in definitions:
        include(definitions[name], {name})
    return members


def carried_rows(layout) -> list[dict[str, Any]]:
    """The physics fields a move carries, from the Registry (see the module docstring)."""
    restart = stream_members(layout, "restart")
    rows = []
    for row in layout.fields:
        dims = row["dimensions"]
        if (not dims or dims[-1] != "nCells" or row["kind"] == "character" or row.get("components")
                or any(n == 0 for n in row["shape"])):
            continue
        if (row["pool"], row["name"]) in CARRIED_DYCORE_FIELDS:
            rows.append(row)
            continue
        if row["pool"] in DYCORE_POOLS or row["name"] in GEOGRAPHY:
            continue
        if row["name"] in restart or (len(dims) == 1 and _has_time(layout, row)):
            rows.append(row)
    return rows


def _key(row, level: int) -> str:
    return f"{row['pool']}.{row['name']}.{level}"


# --------------------------------------------------------------------------- carrier I/O

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_verified(path: Path, size: int, sha256: str, what: str = "moving carrier"):
    """The file's bytes in memory after their SHA256 matches ``sha256`` (one read; the hash on a second thread).

    Returns the uint8 buffer; the caller unpickles from it.  THE COST THIS REMOVES: a full hash pass and then a full
    pickle read of the ~3 GB carrier, one after the other, on one core at the start of every resumed segment
    (box-aws2x 2026-10-24).
    """
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor
    mismatch = f"{what} does not match its sealed receipt"
    if path.stat().st_size != size:
        raise ValueError(mismatch)
    data = np.empty(size, dtype=np.uint8)
    view = memoryview(data)
    digest = hashlib.sha256()
    block = 64 << 20
    with path.open("rb", buffering=0) as stream, ThreadPoolExecutor(max_workers=1) as hasher:
        pending = None
        for start in range(0, len(data), block):
            chunk = view[start:start + block]
            filled = 0
            while filled < len(chunk):
                got = stream.readinto(chunk[filled:])
                if not got:
                    raise ValueError(mismatch)
                filled += got
            if pending is not None:
                pending.result()
            pending = hasher.submit(digest.update, chunk)
        if pending is not None:
            pending.result()
        if stream.read(1):
            raise ValueError(mismatch)
    if digest.hexdigest() != sha256:
        raise ValueError(mismatch)
    return data


def load_carrier(path) -> dict[str, Any]:
    """Read this route's own carrier after its sealed receipt matches (never an untrusted download).

    A carrier sealed on same-host ranks is a manifest plus one shard per rank (checkpoint_shards_v842): the shards
    are read and verified in parallel and assembled into the payload one card seals, so a reader never sees the
    difference.
    """
    path = Path(path)
    from . import moving_resident_v842 as resident
    carried = resident.take_carrier(path)
    if carried is not None:
        # Handed over in memory by the previous segment of this resident process (its file was never written).
        return carried
    receipt = json.loads(path.with_suffix(path.suffix + ".json").read_text())
    data = _read_verified(path, int(receipt["bytes"]), receipt["sha256"])
    payload = pickle.loads(memoryview(data))
    del data
    from .checkpoint_shards_v842 import SCHEMA as SHARDED, assemble_sharded
    if payload.get("schema") == SHARDED:
        payload = assemble_sharded(path, payload)
    if payload.get("schema") != SCHEMA:
        raise ValueError("this moving carrier was sealed on the WOOF route; a moving run keeps one physics "
                         "route from its first segment to its last (the case's physics_args select it)")
    return payload


def _write_sealed(path: Path, payload: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    began = time.perf_counter()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    digest = hashlib.sha256()

    class HashingWriter:
        # The receipt hash comes from the bytes as pickle writes them, in order: no second full read of
        # the ~3 GB carrier on one core after writing it (box-aws2x 2026-10-24).
        def __init__(self, stream):
            self.stream = stream

        def write(self, data):
            digest.update(data)
            return self.stream.write(data)
    with temporary.open("wb") as stream:
        pickle.dump(payload, HashingWriter(stream), protocol=5)
    temporary.replace(path)
    receipt = dict(schema=SCHEMA, path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest(),
                   time_seconds=payload["time_seconds"], valid_time=payload["valid_time"], registers_empty=True,
                   **summary, wall_seconds=time.perf_counter() - began)
    # The receipt is written last: its presence means the carrier is complete (moving_control.wait_for_carrier).
    path.with_suffix(path.suffix + ".json").write_text(json.dumps(receipt, indent=2, default=str) + "\n")
    return receipt


# --------------------------------------------------------------------------- physics field remap

def largest_donors(weights) -> np.ndarray:
    """Each target cell's largest-overlap donor (first on ties): the vectorized, cached moving_physics one."""
    from ..moving_physics import largest_donors as donors
    return donors(weights)


def _roles(categories: dict[str, np.ndarray], threshold: float, ice_class: int) -> np.ndarray:
    """(water, ice, soil class, vegetation class, glacier) per cell."""
    water = np.asarray(categories["xland"], dtype=float) >= 1.5
    ice = np.asarray(categories["xice"], dtype=float) >= threshold
    soil = np.asarray(categories["isltyp"]).astype(np.int64)
    vegetation = np.asarray(categories["ivgtyp"]).astype(np.int64)
    return np.column_stack((water, ice, soil, vegetation, vegetation == ice_class)).astype(np.int64)


def coherent_donors(weights, source_roles, target_roles, source_xyz, target_xyz):
    """One donor column per target cell with the same surface role (moving_physics.land_state_donors).

    The largest-overlap source cell is kept when its role (land or water, sea ice, soil and vegetation
    class, glacier) matches; else the largest overlapping cell of a matching role, relaxing soil and then
    vegetation class; else the nearest cell anywhere with the same land/water, ice and glacier role.
    """
    from scipy.spatial import cKDTree
    matrix = weights.overlap.tocsr()
    donors = largest_donors(weights)
    mismatch = np.flatnonzero(np.any(source_roles[donors] != target_roles, axis=1))
    labels = None
    if (getattr(weights, 'backend', 'cpu') == 'cuda'
            and os.environ.get('WOOF_HEX_MOVING_SURFACE_GPU','1') != '0'
            and np.isfinite(matrix.data).all()):
        from ..moving_surface_cuda import overlap_donors
        donors, labels = overlap_donors(weights, source_roles, target_roles, donors)
    relaxed: dict[str, int] = {}
    nearest, nearest_km = 0, 0.0
    trees: dict[tuple, Any] = {}
    for cell in mismatch:
        if labels is not None and labels[cell] >= 0:
            tier = ((0,1,2,3,4),(0,1,2,4),(0,1,3,4),(0,1,4))[int(labels[cell])]
            if len(tier) < 5:
                name = "+".join(("water", "ice", "soil", "vegetation", "glacier")[i] for i in tier)
                relaxed[name] = relaxed.get(name, 0) + 1
            continue
        start, stop = matrix.indptr[cell], matrix.indptr[cell + 1]
        candidates, areas = matrix.indices[start:stop], matrix.data[start:stop]
        found = False
        for tier in (() if labels is not None else ((0, 1, 2, 3, 4), (0, 1, 2, 4), (0, 1, 3, 4), (0, 1, 4))):
            match = np.all(source_roles[candidates][:, tier] == target_roles[cell, tier], axis=1)
            if match.any():
                picks = np.flatnonzero(match)
                donors[cell] = candidates[picks[int(np.argmax(areas[picks]))]]
                if len(tier) < 5:
                    name = "+".join(("water", "ice", "soil", "vegetation", "glacier")[i] for i in tier)
                    relaxed[name] = relaxed.get(name, 0) + 1
                found = True
                break
        if found:
            continue
        role = tuple(target_roles[cell, (0, 1, 4)])
        if role not in trees:
            pool = np.flatnonzero(np.all(source_roles[:, (0, 1, 4)] == role, axis=1))
            if not len(pool):
                raise ValueError(f"no carried column has the new cell's land/ice/glacier role {role}")
            trees[role] = (pool, cKDTree(source_xyz[pool]))
        pool, tree = trees[role]
        chord, index = tree.query(target_xyz[cell])
        donors[cell] = pool[index]
        nearest += 1
        nearest_km = max(nearest_km, float(2 * np.arcsin(min(chord / 2, 1.0)) * 6371.229))
    return donors, dict(target_cells=int(len(target_roles)), role_mismatch_cells=int(len(mismatch)),
                        relaxed_overlap_donors=relaxed, nearest_role_donors=nearest,
                        nearest_role_donor_distance_max_km=nearest_km)


def _unit_xyz(mesh) -> np.ndarray:
    xyz = np.stack([np.asarray(mesh.arrays[k], dtype=np.float64) for k in ("xCell", "yCell", "zCell")], axis=-1)
    return xyz / np.linalg.norm(xyz, axis=-1, keepdims=True)


def build_weights(*, source_mesh, source_vertical, operator, source_grid, target_grid, mesh, vertical):
    """The move's remap map from the carrier's mesh (``source_mesh``, ``source_vertical``) onto this member's.

    Reads only geometry (the two members' grids, the overlap operator, both meshes' areas and heights), never the
    carried state, so a resident process builds it while the previous segment steps (moving_resident_v842).
    """
    from ..moving_remap import ConservativeMap, SphericalCells
    source, target = SphericalCells.read(source_grid), SphericalCells.read(target_grid)
    radius = float(mesh.attrs["sphere_radius"])
    if float(source_mesh.attrs["sphere_radius"]) != radius:
        raise ValueError("moving forecast earth radii differ")
    scale = (radius / source.radius_m) ** 2
    if (radius / target.radius_m) ** 2 != scale:
        raise ValueError("source and target operator spheres use different area units")
    if len(source.area) != len(source_mesh.arrays["areaCell"]) or len(target.area) != len(mesh.arrays["areaCell"]):
        raise ValueError("the move's grids do not have the carrier's and this member's cell counts")
    # The generator geometry binds the operator; the carrier's coordinates were read single precision
    # (runtime_v842.load_native_dynamics_inputs), so they identify the source grid to FP32.
    if not np.allclose(_unit_xyz(source_mesh), source.centers, rtol=0, atol=2e-6):
        raise ValueError("source grid does not identify the stored moving carrier")
    weights = ConservativeMap.load(operator, source=source, target=target)
    weights = ConservativeMap(weights.overlap * scale, weights.source_area * scale, weights.target_area * scale,
                              dict(weights.receipt, effective_area_radius_m=radius, generator_area_scale=scale))
    weights = weights.with_physical_areas(source_mesh.arrays["areaCell"], mesh.arrays["areaCell"])
    height = os.environ.get("WOOF_HEX_MOVING_HEIGHT_REMAP")
    if height in ("1", "absolute"):
        from ..moving_vertical import HeightConservativeMap
        weights = HeightConservativeMap(weights, source_vertical.zgrid, vertical.zgrid,
                                        source_vertical.dzw, vertical.dzw,
                                        endpoint_alignment=height != "absolute")
    return weights


def map_field(values, dims, weights, *, donors=None):
    """One carried field (leading axes, then cells) onto the target cells."""
    values = np.asarray(values)
    if donors is not None:
        return values[..., donors]
    if dims[0] == LAYER_AXIS and len(dims) == 2:
        layer = getattr(weights, "layer_intensive", None)
        return layer(values) if layer is not None else weights.intensive(values)
    if dims[0] == INTERFACE_AXIS and len(dims) == 2:
        interface = getattr(weights, "interface_intensive", None)
        return interface(values) if interface is not None else weights.intensive(values)
    return weights.intensive(values)


# --------------------------------------------------------------------------- the route object

@dataclass
class MovingRoute:
    """Everything the MPAS route needs to move; built from the WOOF_HEX_MOVING_* switches."""

    carrier: dict[str, Any] | None = None
    carrier_path: str | None = None
    operator: str | None = None
    source_grid: str | None = None
    export_path: str | None = None
    shock: bool = False
    control: Any = None
    frame_seconds: float = 3600.0
    origin_time: str | None = None
    offset_seconds: float = 0.0
    shock_rows: list = field(default_factory=list)
    remap_receipt: dict | None = None
    export_receipt: dict | None = None
    frames: list = field(default_factory=list)
    _pressure: Any = None

    # ----------------------------------------------------------------- construction
    @classmethod
    def from_environ(cls, arguments, environ=os.environ) -> "MovingRoute":
        if environ.get("WOOF_HEX_MOVING_ZONE"):
            # THE BREAKAGE THIS PREVENTS: the frozen-far-field zone transfer (woof.hex.moving_zone_state) maps the
            # atmosphere only; on this route every carried physics field would be left on the old member's
            # cell order.  The sentence starts with the door's moving refusal so the controller stops at once.
            from ..fullstep_lts_admission import MOVING_ROUTE_REFUSAL
            raise ValueError(f"{MOVING_ROUTE_REFUSAL}, but zone carry (a corridor case, or moving run --zone-carry) "
                             "is a named gap on the MPAS suite route: the frozen-far-field transfer carries no "
                             "physics state yet. Run the case without a corridor; its moves take the overlap remap")
        # Absolute paths: the native step loop runs inside the output folder (native_working_directory).
        def absolute(name):
            value = environ.get(name)
            return str(Path(value).expanduser().absolute()) if value else None

        route = cls(operator=absolute("WOOF_HEX_MOVING_OPERATOR"), source_grid=absolute("WOOF_HEX_MOVING_SOURCE_GRID"),
                    export_path=absolute("WOOF_HEX_MOVING_CHECKPOINT_OUT"),
                    shock=environ.get("WOOF_HEX_MOVING_SHOCK") == "1")
        minutes = getattr(arguments, "history_every_minutes", None)
        route.frame_seconds = float(minutes) * 60.0 if minutes else 3600.0
        carrier = absolute("WOOF_HEX_MOVING_CHECKPOINT_IN")
        if carrier:
            if environ.get("WOOF_HEX_MOVING_CHECKPOINT_WAIT"):
                from ..moving_control import wait_for_carrier
                wait_for_carrier(carrier, float(environ["WOOF_HEX_MOVING_CHECKPOINT_WAIT"]))
            route.carrier_path = carrier
            route.carrier = load_carrier(carrier)
            route.origin_time = route.carrier["origin_time"]
            route.offset_seconds = float(route.carrier["time_seconds"])
            if route.operator and not route.source_grid:
                raise ValueError("a remapped move names its operator but not the source member's grid "
                                 "(WOOF_HEX_MOVING_SOURCE_GRID)")
        if environ.get("WOOF_HEX_MOVING_CONTROL"):
            from ..moving_control import MovingControl, SEAL_ENV
            route.control = MovingControl(Path(absolute("WOOF_HEX_MOVING_CONTROL")),
                                          float(environ.get(SEAL_ENV, "0") or 0))
        return route

    def resume_namelist(self, text: str) -> str:
        """A resumed segment starts at the carrier's valid time (its restart flag: ``mark_restart``)."""
        from .forecast_v842 import namelist_option, set_namelist_options
        if self.carrier is None:
            return text
        named = namelist_option(text, "nhyd_model", "config_start_time")
        if named is not None and str(named).strip() != self.origin_time:
            raise ValueError(f"the moving carrier belongs to a run that started {self.origin_time}; this case "
                             f"starts {named}")
        return set_namelist_options(text, {"nhyd_model": {"config_start_time": self.carrier["valid_time"]}})

    def mark_restart(self, layout) -> None:
        """A resumed segment initialises its physics as a native restart (mpas_atm_core.F:109-150).

        Set after the pools are read from the member's init, never in the namelist the pool reader sees.
        THE BREAKAGE THAT ORDER PREVENTS: with config_do_restart in the request, the reader also takes the
        restart stream's fields from the init, and an init's soil-depth ``zs`` (nSoilLevels) is not the
        Registry's diag_physics ``zs`` ("physics pool: zs: dimension rank differs").  The carrier, not the
        init, supplies every restart-stream field (``restore_physics``).
        """
        if self.carrier is not None:
            layout.configs["config_do_restart"] = True

    def begin(self, start: str) -> None:
        if self.origin_time is None:
            self.origin_time = start
        if self.offset_seconds % self.frame_seconds:
            raise ValueError("a moving carrier must lie on a history frame")

    # ----------------------------------------------------------------- remapped resume (dycore)
    def _weights(self, mesh, vertical):
        carrier = self.carrier
        build = lambda: build_weights(source_mesh=carrier["mesh"], source_vertical=carrier["vertical"],
                                      operator=self.operator, source_grid=self.source_grid,
                                      target_grid=self.target_grid, mesh=mesh, vertical=vertical)
        # A resident moving process prepared this map beside the previous segment (moving_resident_v842).
        from . import moving_resident_v842 as resident
        return resident.weights(self, mesh, vertical, build)

    def move_weights(self, *, mesh, grid, target_grid):
        """This move's remap map before the atmosphere is remapped (None on the same mesh), so the physics
        restore can run beside the atmosphere remap (moving_mpi_v842.prepare_moving)."""
        if self.carrier is None or not self.operator:
            return None
        self.target_grid = target_grid
        return self._weights(mesh, grid.vertical_grid)

    def resume_dycore(self, *, mesh, grid, reference, state, saved, names, target_grid, weights=None):
        """The carrier's atmosphere on this member: (state, saved, weights or None for the same mesh).

        ``weights``: this move's map when the caller already built it (``move_weights``)."""
        from ..driver import DrySavedDiagnostics
        from ..state import PrognosticState
        carrier = self.carrier
        if tuple(carrier["scalar_names"]) != tuple(names):
            raise ValueError("moving carrier species differ from this run's MPAS suite: "
                             f"{carrier['scalar_names']} against {list(names)}")
        values = carrier["state"]
        if not self.operator:
            source = carrier["mesh"].arrays
            for name in ("areaCell", "cellsOnEdge", "edgesOnCell"):
                if not np.array_equal(source[name], mesh.arrays[name]):
                    raise ValueError(f"same-mesh moving resume on another mesh: {name} differs")
            state = PrognosticState(*(np.ascontiguousarray(values[n]) for n in STATE_FIELDS),
                                    time_seconds=float(carrier["time_seconds"]))
            saved = DrySavedDiagnostics(**{n: np.ascontiguousarray(values[n]) for n in SAVED_FIELDS})
            self.remap_receipt = dict(kind="same mesh", time_seconds=carrier["time_seconds"])
            return state, saved, None
        began = time.perf_counter()
        self.target_grid = target_grid
        if weights is None:
            weights = self._weights(mesh, grid.vertical_grid)
        from ..moving_state import remap_atmosphere
        prepared = SimpleNamespace(mesh=mesh, vertical=grid.vertical_grid, reference=reference,
                                   terrain_metrics=grid.terrain_metrics)
        payload = dict(state=values, vertical=carrier["vertical"], mesh=carrier["mesh"],
                       scalar_names=list(names), time_seconds=float(carrier["time_seconds"]))
        state, saved, receipt = remap_atmosphere(payload, weights, source_mesh=carrier["mesh"],
                                                 target_prepared=prepared, config=None)
        # The CUDA upload takes C-ordered words only (the remap's edge projection is an einsum result).
        state = PrognosticState(*(np.ascontiguousarray(getattr(state, n)) for n in STATE_FIELDS),
                                time_seconds=float(carrier["time_seconds"]))
        saved = DrySavedDiagnostics(**{n: np.ascontiguousarray(getattr(saved, n)) for n in SAVED_FIELDS})
        receipt.update(kind="remap", operator=weights.receipt if hasattr(weights, "receipt") else None,
                       time_seconds=carrier["time_seconds"], atmosphere_remap_seconds=time.perf_counter() - began)
        receipt["conservation"] = conservation(receipt)
        self.remap_receipt = receipt
        return state, saved, weights

    # ----------------------------------------------------------------- remapped resume (physics)
    def restore_physics(self, session, layout, weights, mesh) -> dict[str, Any]:
        """Carried physics state into this member's native pools, before the (restart) physics init."""
        began = time.perf_counter()
        carrier, native = self.carrier, session.native
        physics = carrier["physics"]
        rows = self.carried_rows(layout)
        absent = sorted(f"{r['pool']}.{r['name']}" for r in rows if f"{r['pool']}.{r['name']}" not in physics["fields"])
        if absent:
            raise ValueError("this member's physics needs state the moving carrier does not hold (the case's "
                             "physics selection changed between segments): " + ", ".join(absent[:12]))
        if carrier["physics"]["suite"] != getattr(session.config, "suite", None):
            raise ValueError(f"the moving carrier ran suite {carrier['physics']['suite']}, this member "
                             f"{getattr(session.config, 'suite', None)}")
        n_target = layout.dimensions["nCells"]
        receipt = dict(fields=0, methods={})
        stages = {}
        stage_began = time.perf_counter()
        donors = surface = None
        if weights is not None:
            threshold = 0.02 if layout.configs.get("config_frac_seaice") else 0.5
            target_categories = {name: self._pool_cells(native, layout, "sfc_input", name) for name in CATEGORY_FIELDS}
            ice_class = int(np.asarray(native.get_field("sfc_input", self._pool_name(layout, "sfc_input", "isice_lu")))
                            .reshape(-1)[0])
            source_roles = _roles(physics["categories"], threshold, physics["isice"])
            target_roles = _roles(target_categories, threshold, ice_class)
            donors, surface = coherent_donors(weights, source_roles, target_roles,
                                              _unit_xyz(carrier["mesh"]), _unit_xyz(mesh))
            receipt["surface_donors"] = surface
        stages['surface_donors'] = time.perf_counter()-stage_began
        mapped_vectors = {}
        computed = {}
        if weights is not None:
            from ..moving_physics import carry_horizontal_vector
            from ..host_parallel import run_tasks
            names = {r["name"]: r for r in rows}
            vector_tasks = []
            for u_name, v_name in VECTOR_PAIRS:
                if u_name in names and v_name in names and names[u_name]["dimensions"][0] == LAYER_AXIS:
                    for level in range(1, names[u_name]["nlevels"] + 1):
                        vector_tasks.append((_key(names[u_name], level), _key(names[v_name], level)))

            def carry_vector(u_key, v_key):
                u = physics["pools"][u_key][..., :len(weights.source_area)]
                v = physics["pools"][v_key][..., :len(weights.source_area)]
                return carry_horizontal_vector(u, v, weights, carrier["mesh"], mesh, atmospheric=True)

            def map_one(key, dims):
                # Exactly the per-field branch the sequential loop ran; only the pool write stays below.
                cells = physics["pools"][key][..., :len(weights.source_area)]
                surface_field = len(dims) == 1 or dims[0] not in (LAYER_AXIS, INTERFACE_AXIS)
                finite = bool(np.all(np.isfinite(cells))) if cells.dtype.kind == "f" else True
                if not finite:
                    # A word the scheme has not written yet (never a carried state): keep its pattern.
                    return finite, cells[..., largest_donors(weights)], "largest-overlap donor (unset words)"
                if key in mapped_vectors:
                    return finite, mapped_vectors[key], "earth-Cartesian vector, physical height"
                if np.issubdtype(cells.dtype, np.integer) and not surface_field:
                    return finite, cells[..., largest_donors(weights)], "largest-overlap donor"
                if surface_field and (len(dims) == 1 or dims[0] == "nSoilLevels"):
                    return finite, map_field(cells, dims, weights, donors=donors), "coherent surface donor"
                mapped = map_field(cells, dims, weights)
                method = ("physical-height layers" if dims[0] == LAYER_AXIS and hasattr(weights, "layer_intensive")
                          else "physical-height interfaces" if dims[0] == INTERFACE_AXIS
                          and hasattr(weights, "interface_intensive") else "conservative overlap")
                return finite, mapped, method

            # THE COST THIS REMOVES: every carried physics field (vectors first) was mapped one after another on
            # one thread, 7.9 s of a resumed MPAS-route storm-following segment (box-aws2x 2026-10-24, sampled).
            # The maps are independent calls; they run on all cores, and the native pool writes stay in order.
            largest_donors(weights)  # cached on the map before any task needs it
            jobs = [(_key(row, level), row['dimensions']) for row in rows for level in range(1, row['nlevels']+1)]
            if (getattr(weights, 'backend', 'cpu') == 'cuda'
                    and hasattr(weights, 'source_dzw')
                    and os.environ.get('WOOF_HEX_MOVING_PHYSICS_BATCH','1') != '0'):
                from .moving_rates_batch_v842 import remap
                stage_began = time.perf_counter()
                scalar_keys = [key for key, dims in jobs if len(dims) == 2 and dims[0] == LAYER_AXIS]
                batched, detail = remap(weights, physics['pools'], scalar_keys, vector_tasks,
                                        carrier['mesh'], mesh)
                computed.update(batched)
                vector_tasks = [pair for pair in vector_tasks if pair[0] not in computed]
                receipt['rate_batches'] = detail
                stages['rate_batches'] = time.perf_counter()-stage_began
            stage_began = time.perf_counter()
            if vector_tasks:
                first, rest = vector_tasks[0], vector_tasks[1:]
                results = [carry_vector(*first)]  # builds the map's device geometry once
                results += run_tasks([lambda pair=pair: carry_vector(*pair) for pair in rest])
                for (u_key, v_key), (mu, mv) in zip(vector_tasks, results):
                    mapped_vectors[u_key] = mu
                    mapped_vectors[v_key] = mv
            jobs = [(key, dims) for key, dims in jobs if key not in computed]
            for (key, _), result in zip(jobs, run_tasks([lambda key=key, dims=dims: map_one(key, dims)
                                                         for key, dims in jobs])):
                computed[key] = result
            stages['remaining_maps'] = time.perf_counter()-stage_began
        stage_began = time.perf_counter()
        install = getattr(native, 'restore_mapped_fields', None) if weights is not None else None
        if install is not None:
            jobs = []
            for row in rows:
                for level in range(1, row['nlevels']+1):
                    key = _key(row, level)
                    finite, mapped, method = computed.pop(key)
                    jobs.append((row, level, finite, mapped))
                    receipt['fields'] += 1
                    receipt['methods'].setdefault(method, []).append(key)
            install(jobs, n_target)
        for row in rows:
            if install is not None:
                break
            dims = row["dimensions"]
            for level in range(1, row["nlevels"] + 1):
                key = _key(row, level)
                source = physics["pools"][key]
                buffer = native.get_field(row["pool"], row["pool_name"], level=level)
                if weights is None:
                    if source.shape != buffer.shape or source.dtype != buffer.dtype:
                        raise ValueError(f"same-mesh moving resume: {key} layout differs")
                    mapped, method = source, "same mesh"
                else:
                    finite, mapped, method = computed.pop(key)
                    if finite and not np.all(np.isfinite(mapped)):
                        raise ValueError(f"moving physics remap produced a non-finite value in {key}")
                    buffer[..., :n_target] = np.asarray(mapped).astype(buffer.dtype)
                    mapped = buffer
                native.put_field(row["pool"], row["pool_name"], mapped, level=level, nlevels=row["nlevels"])
                receipt["fields"] += 1
                receipt["methods"].setdefault(method, []).append(key)
        stages['pool_writes'] = time.perf_counter()-stage_began
        stage_began = time.perf_counter()
        self._restore_windows(session, weights, donors)
        stages['windows'] = time.perf_counter()-stage_began
        receipt['stages_seconds'] = stages
        receipt["seconds"] = time.perf_counter() - began
        if self.remap_receipt is not None:
            self.remap_receipt["physics"] = receipt
        return receipt

    def restore_alarm_phase(self, session, config) -> dict[str, Any] | None:
        """Continue the carried physics alarms' phase on this segment's clock (correct mode).

        MPAS-A v8.4.2 starts every recurring physics alarm again at a restart's start time
        (mpas_atmphys_manager.F:385-766), so a checkpoint between two radiation calls (a 10-minute frame
        under a 30-minute radiation interval) calls radiation at the restart and the restart leaves the
        straight run.  THE BREAKAGE THIS FIXES: such a restart differed from the straight run in every
        cell's glw at its first frame (HEX-PHYSICS-PARITY run group 6 #21, a 12:10 restart).  Exact mode keeps
        MPAS's own reset (bit for bit native, restart included); correct mode, the door default, carries the
        phase, so a restart from any checkpoint frame is the straight run.
        """
        carried = self.carrier["physics"].get("manager")
        manager = getattr(getattr(session, "driver", None), "manager", None)
        if carried is None or manager is None or getattr(config, "mode", "exact") == "exact":
            return None
        elapsed = Fraction(self.carrier["physics"]["manager_elapsed"])
        # The alarms' phase is in model seconds, so a resume under another timestep keeps it: an
        # alarm rings at the first step at or past its next ring, as on any step grid.
        if set(carried["alarms"]) != set(manager.alarms):
            raise ValueError("the checkpoint's physics alarms differ from this run's suite")
        for name, item in carried["alarms"].items():
            alarm = manager.alarms[name]
            if Fraction(item["period"]) != alarm.period:
                raise ValueError(f"the checkpoint's {name} alarm interval differs from this run's")
            alarm.next_ring = Fraction(item["next_ring"]) - elapsed
        return {name: str(alarm.next_ring) for name, alarm in manager.alarms.items()}

    def _restore_windows(self, session, weights, donors):
        physics = self.carrier["physics"]
        window = getattr(session, "precipitation_window", None)
        if (window is None) != (physics.get("window") is None):
            raise ValueError("the moving carrier and this member differ in the surface precipitation window (RP6)")
        if window is not None:
            state = dict(physics["window"])
            if weights is not None:
                for key, value in state.items():
                    cells = np.asarray(value)[..., :len(weights.source_area)]
                    state[key] = (cells[donors] if np.issubdtype(cells.dtype, np.integer)
                                  else weights.intensive(cells).astype(cells.dtype))
            window.restore(state)
        service = getattr(session, "surface_service", None)
        if service is not None and physics.get("surface_service") is not None:
            service.restore_restart_state(physics["surface_service"])

    @staticmethod
    def _pool_name(layout, pool, name):
        return next(r["pool_name"] for r in layout.fields if r["pool"] == pool and r["name"] == name)

    def _pool_cells(self, native, layout, pool, name):
        return np.asarray(native.get_field(pool, self._pool_name(layout, pool, name)))[..., :layout.dimensions["nCells"]]

    # ----------------------------------------------------------------- export
    def carried_rows(self, layout):
        """Moving remaps keep target geography; same-mesh checkpoints may carry it."""
        return carried_rows(layout)

    def continuation_state(self, session) -> dict[str, Any]:
        """Carrier words a same-mesh checkpoint continuation adds (``CheckpointRoute``); a move adds none."""
        return {}

    def physics_clock_offset(self, dt):
        """(steps, seconds) the physics clock starts at: a move starts its own clock (``CheckpointRoute``)."""
        return 0, 0.0

    def export(self, path, *, session, dycore, mesh, vertical, names, layout, step, dt, handoff=None) -> dict[str, Any]:
        """Seal the dycore state and the carried native physics state at this step."""
        payload, summary = self.carrier_payload(session=session, dycore=dycore, mesh=mesh, vertical=vertical,
                                                names=names, layout=layout, step=step, dt=dt)
        if handoff is not None:
            # A move whose next segment runs in this resident process: the carrier stays in memory.
            from . import moving_resident_v842 as resident
            return resident.store_carrier(Path(path), payload, summary, handoff)
        return _write_sealed(Path(path), payload, summary)

    def carrier_payload(self, *, session, dycore, mesh, vertical, names, layout, step, dt):
        """The carrier ``export`` seals, and its receipt summary.

        Same-host ranks seal each rank's words in its own shard (checkpoint_shards_v842); ``load_carrier``
        assembles the shards and builds this same payload from them, so every reader gets the words one card's
        carrier holds.
        """
        native, cp = session.native, dycore.cp
        atmosphere = dycore.atmosphere
        model_seconds = self.offset_seconds + step * dt
        state = {name: cp.asnumpy(getattr(atmosphere.state, name)) for name in STATE_FIELDS}
        state.update({name: cp.asnumpy(getattr(atmosphere.saved, name)) for name in SAVED_FIELDS})
        if getattr(dycore, "regional_v841", None) is not None:
            # A limited-area dycore holds native's garbage element as the last cell/edge of every state array
            # (cuda_regional_forecast_v841._PaddedState/_PaddedSaved); the carrier holds the solve extent, the
            # shape the regional dycore pads again when it opens on a resume.  THE BREAKAGE THIS FIXES
            # (fix/hex-lam-da-v842): a limited-area checkpoint resumed with the garbage element padded twice.
            n_cells, n_edges = int(mesh.dimensions["nCells"]), int(mesh.dimensions["nEdges"])
            extent = {name: n_edges if name in ("rho_u", "normal_velocity") else n_cells
                      for name in (*STATE_FIELDS, *SAVED_FIELDS)}
            for name, value in state.items():
                if value.shape[-1] != extent[name] + 1:
                    raise ValueError(f"limited-area carrier: {name} is not the padded solve extent "
                                     f"({value.shape[-1]} against {extent[name]} + 1)")
                state[name] = np.ascontiguousarray(value[..., :extent[name]])
        rows = self.carried_rows(layout)
        pools = {_key(row, level): native.get_field(row["pool"], row["pool_name"], level=level)
                 for row in rows for level in range(1, row["nlevels"] + 1)}
        window = getattr(session, "precipitation_window", None)
        service = getattr(session, "surface_service", None)
        physics = dict(suite=getattr(session.config, "suite", None), pools=pools,
                       fields={f"{r['pool']}.{r['name']}": dict(dimensions=r["dimensions"], nlevels=r["nlevels"],
                                                                 kind=r["kind"]) for r in rows},
                       categories={name: self._pool_cells(native, layout, "sfc_input", name) for name in CATEGORY_FIELDS},
                       isice=int(np.asarray(native.get_field(
                           "sfc_input", self._pool_name(layout, "sfc_input", "isice_lu"))).reshape(-1)[0]),
                       window=None if window is None else window.snapshot(),
                       surface_service=None if service is None else service.restart_state(),
                       # The physics alarms' phase (radiation, convection, buckets) on this segment's elapsed
                       # clock, and that clock at the seal (restore_alarm_phase rebases it).  A checkpoint
                       # continuation's clock starts at its offset (CheckpointRoute.physics_clock_offset).
                       manager=_manager_state(session),
                       manager_elapsed=str(Fraction(str(self.physics_clock_offset(dt)[1]))
                                           + Fraction(str(step)) * Fraction(str(dt))))
        physics.update(self.continuation_state(session))
        valid = (datetime.strptime(self.origin_time, TIME_FORMAT) + timedelta(seconds=model_seconds)).strftime(TIME_FORMAT)
        payload = dict(schema=SCHEMA, route="mpas-suite", time_seconds=float(model_seconds),
                       origin_time=self.origin_time, valid_time=valid, state=state, mesh=mesh, vertical=vertical,
                       scalar_names=list(names), physics=physics,
                       stochastic=(None if getattr(dycore, 'stochastic', None) is None else
                                   dycore.stochastic.process.snapshot()),
                       chemistry=(None if getattr(dycore, 'chemistry', None) is None else
                                  dycore.chemistry.restart_state()))
        return payload, dict(
            physics_fields=len(rows), physics_words=int(sum(np.asarray(v).size for v in pools.values())),
            state={k: list(v.shape) for k, v in state.items()})

    # ----------------------------------------------------------------- per step: shock series, frames, stop
    def after_step(self, step, *, steps, dt, columns, session, dycore, mesh, vertical, names, layout, out) -> bool:
        """Shock series, tracker frame and the controller's verdict after a completed step; True stops."""
        model_seconds = self.offset_seconds + step * dt
        cp = dycore.cp
        n = layout.dimensions["nCells"]
        if self.shock:
            pressure = cp.asnumpy(columns.psfc_p).reshape(-1)[:n].astype(np.float64)
            row = dict(model_seconds=model_seconds, surface_pressure_tendency_abs_max_pa_s=None,
                       vertical_velocity_abs_max=float(cp.abs(dycore.atmosphere.saved.vertical_velocity).max()))
            if self._pressure is not None:
                difference = (pressure - self._pressure) / dt
                peak = int(np.argmax(abs(difference)))
                area = np.asarray(mesh.arrays["areaCell"], dtype=np.float64)[:n]
                row.update(surface_pressure_tendency_abs_max_pa_s=float(abs(difference[peak])),
                           surface_pressure_tendency_area_rms_pa_s=float(np.sqrt(np.dot(difference ** 2, area) / area.sum())),
                           surface_pressure_tendency_peak_latitude_deg=float(np.degrees(mesh.arrays["latCell"][peak])),
                           surface_pressure_tendency_peak_longitude_deg=float(np.degrees(mesh.arrays["lonCell"][peak])))
            self._pressure = pressure
            self.shock_rows.append(row)
        if self.control is None or model_seconds % self.frame_seconds or (step == 0 and self.carrier is not None):
            return False
        final = step >= steps
        path = Path(out) / f"tracker-frame.{int(model_seconds):08d}.nc"
        write_tracker_frame(path, mesh=mesh, native=session.native, layout=layout, columns=columns,
                            vertical=vertical, cp=cp)
        self.frames.append(str(path))
        self.control.frame_written(model_seconds=model_seconds, history_path=str(path), final=final)
        if step == 0:
            # As on the ArWen route, the start frame is only reported: a stop, a verdict or a seal is taken
            # after a completed step, never before the first (a move at the start has nothing to carry).
            return False
        seal = lambda target: self.export(target, session=session, dycore=dycore, mesh=mesh, vertical=vertical,
                                          names=names, layout=layout, step=step, dt=dt)
        return bool(self.control.at_frame(model_seconds=model_seconds, final=final, seal=seal))

    def finish(self, *, step, session, dycore, mesh, vertical, names, layout, dt) -> None:
        if self.export_path:
            # A move whose next segment runs in this resident process hands the carrier over in memory
            # (moving_resident_v842); every other stop seals it to its file as before.
            from . import moving_resident_v842 as resident
            self.export_receipt = self.export(self.export_path, session=session, dycore=dycore, mesh=mesh,
                                              vertical=vertical, names=names, layout=layout, step=step, dt=dt,
                                              handoff=resident.hands_off(self.control))

    def receipt(self) -> dict[str, Any]:
        return {"moving_boundary_export": self.export_receipt, "moving_remap": self.remap_receipt,
                "moving_control": None if self.control is None else self.control.receipt(),
                "clock_offset_seconds": self.offset_seconds, "remap_shock_series": self.shock_rows,
                "moving_origin_time": self.origin_time, "tracker_frames": self.frames}


def _manager_state(session):
    manager = getattr(getattr(session, "driver", None), "manager", None)
    return None if manager is None else manager.restart_state()


def conservation(receipt: dict[str, Any]) -> dict[str, float]:
    """Largest relative change of dry mass, theta mass and each tracer across the move."""
    def relative(item):
        before = float(item["before"])
        return abs(float(item["absolute_error"]) / before) if before else abs(float(item["absolute_error"]))
    tracers = {name: relative(item) for name, item in receipt["tracer_totals_kg"].items()}
    return dict(mass=relative(receipt["mass_total_kg"]), theta_mass=relative(receipt["theta_mass_total_kg_kelvin"]),
                tracer=max(tracers.values(), default=0.0), tracers=tracers)


def write_tracker_frame(path, *, mesh, native, layout, columns, vertical, cp) -> None:
    """The fields the moving controller's tracker reads (moving_run.history_fields), for one frame."""
    import netCDF4
    n = layout.dimensions["nCells"]
    rows = {(r["pool"], r["name"]): r["pool_name"] for r in layout.fields}

    def column(pool, name):
        key = (pool, rows.get((pool, name)))
        if key[1] is None or key not in native._fields:
            return None
        return np.asarray(native.get_field(*key))[..., :n]

    levels = int(layout.dimensions["nVertLevels"])
    zeros = np.zeros(n, np.float32)
    path = Path(path)
    temporary = path.with_suffix(".partial.nc")
    with netCDF4.Dataset(temporary, "w") as d:
        d.createDimension("Time", 1)
        d.createDimension("nCells", n)
        d.createDimension("nVertLevels", levels)
        for name in ("latCell", "lonCell"):
            d.createVariable(name, "f8", ("nCells",))[:] = np.asarray(mesh.arrays[name], dtype=np.float64)[:n]
        d.createVariable("ter", "f4", ("nCells",))[:] = np.asarray(vertical.zgrid)[0, :n]
        d.createVariable("surface_pressure", "f4", ("Time", "nCells"))[0] = cp.asnumpy(columns.psfc_p).reshape(-1)[:n]
        for target, (pool, name) in (("pressure", ("diag", "pressure")), ("theta", ("diag", "theta")),
                                     ("u_zonal", ("diag", "uReconstructZonal")),
                                     ("v_meridional", ("diag", "uReconstructMeridional"))):
            d.createVariable(target, "f4", ("Time", "nCells", "nVertLevels"))[0] = column(pool, name).T
        for target in ("u10", "v10"):
            value = column("diag_physics", target)
            d.createVariable(target, "f4", ("Time", "nCells"))[0] = zeros if value is None else value
        # The setup targets' fields (moving_run.history_fields): cape_shear reads qv columns and its storms
        # (refl10cm), boundary the 2 m fields. THE BREAKAGE THIS PREVENTS (owner run run_51882d2a2a758db6, CAPE x
        # shear 3 km on MPAS convection-permitting, 2026-10-24): the frame carried only the cyclone fields and the
        # controller stopped at the first frame with "IndexError: qv not found".
        qv = scalar_column(native, layout, "qv", n, levels)
        if qv is not None:
            d.createVariable("qv", "f4", ("Time", "nCells", "nVertLevels"))[0] = qv
        refl = column("diag_physics", "refl10cm_max")
        if refl is not None:
            d.createVariable("refl10cm", "f4", ("Time", "nCells"))[0] = np.asarray(refl).reshape(-1)[:n]
        for target, source in (("t2m", "t2m"), ("q2", "q2")):
            value = column("diag_physics", source)
            if value is not None:
                d.createVariable(target, "f4", ("Time", "nCells"))[0] = np.asarray(value).reshape(-1)[:n]
    temporary.replace(path)


def scalar_column(native, layout, name, n, levels):
    """One species of the native ``scalars`` var_array as (nCells, nVertLevels), or None when it is not active."""
    row = next((r for r in layout.fields if r["name"] == "scalars"), None)
    components = tuple((row or {}).get("components") or ())
    if row is None or name not in components or (row["pool"], row["pool_name"]) not in native._fields:
        return None
    value = np.asarray(native.get_field(row["pool"], row["pool_name"]))
    axis = next(a for a, size in enumerate(value.shape) if size == len(components))
    species = np.take(value, components.index(name), axis=axis)
    if species.ndim != 2:
        return None
    if species.shape[0] == levels and species.shape[1] != levels:
        species = species.T
    return np.ascontiguousarray(species[:n, :levels], dtype=np.float32)
