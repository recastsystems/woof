"""Native MPAS-A v8.4.2 IAU (``woof.hex.iau_v842``) on hex's forecast step.

``woof.hex.iau_v842`` holds the MPAS-A v8.4.2 IAU arithmetic as CUDA kernels on MPAS pool
fields (``atm_add_tend_anal_incr``: the increments of the ``iau`` stream turned into coupled
tendencies against the CURRENT state, every step of the window).  This module feeds those
kernels from hex's own step, at the same slot ``woof.hex.iau_forcing`` uses: after the column
physics is coupled and before the dynamics, onto the held physics tendencies (MPAS
``tend_ru_physics``, ``tend_rtheta_physics``, ``tend_rho_physics`` and the scalar tendencies).

The two IAU routes differ by construction, and both stay selectable (``--iau-route``):

* ``native`` (this module): the increment is held in MPAS's uncoupled variables (``u``, dry
  ``rho``, ``theta``, mixing ratios) and converted to coupled tendencies every step with the
  state of that step.  ``physics.mode`` picks the kernels: ``exact`` is MPAS-A v8.4.2 word for
  word (NINT step count, the physics vapour tendency counted again in the theta_m conversion,
  no ``u*d(rho_edge)`` term); ``correct`` applies IN1-IN3 (``woof.hex.iau_v842``).
* ``hex`` (``woof.hex.iau_forcing``): the increment is formed once in hex's coupled variables
  (analysis minus background, both recoupled) and added as a constant tendency.

Mapping from hex to the MPAS pool names the kernels read:

    mesh.zz                driver.atmosphere.vertical.zz        [level, cell]
    mesh.cellsOnEdge       physics geometry cells_on_edge       [edge, 2], zero-based; a
                           limited-area mesh's garbage column is written as -1 (exterior)
    state.rho_zz           state.rho
    state.theta_m          state.rho_theta / state.rho          (hex carries rho*theta_m)
    state.scalars          state.scalars
    state.u                the step's recovered edge velocity
    diag.rho_edge          the step's recovered edge density
    tend.scalars_tend      held.scalars
    tend_ru/rtheta/rho     held.rho_u / held.rho_theta / held.rho

Each rate class of a local-time-step run carries its own copy of the increments (its local
cells and edges, halos included, exactly as ``woof.hex.iau_forcing`` does), its own step count
(exact) and its own window position (correct), so every class receives the whole window at its
own step length.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .iau_v842 import IAUConfig, atm_iau_coef, atm_iau_coef_window

#: The MPAS moist group in the order hex rows declare it; the IAU stream increments qv..qg.
MOIST_SPECIES = ("qv", "qc", "qr", "qi", "qs", "qg", "qh")
INCREMENT_SPECIES = ("qv", "qc", "qr", "qi", "qs", "qg")


def moist_block(scalar_names: Sequence[str]) -> int:
    """moist_end (one-based, inclusive) of the leading moist block; qv must lead it."""

    names = tuple(scalar_names)
    if not names or names[0] != "qv":
        raise ValueError(f"native IAU needs qv as the first model scalar (index_qv = 1), got {names[:3]}")
    end = 0
    for name in names:
        if name not in MOIST_SPECIES:
            break
        end += 1
    stray = [name for name in names[end:] if name in MOIST_SPECIES]
    if stray:
        raise ValueError(f"moist species {stray} sit outside the leading moist block of {names}; MPAS's "
                         "moist_start..moist_end range would not reach them")
    return end


def physical_increment(increments: Mapping[str, np.ndarray], scalar_names: Sequence[str]) -> dict[str, np.ndarray]:
    """Increment-file fields (woof.hex.analysis_bundle.read_increment) as tend_iau arrays, global order.

    ``scalars`` is [moist_end, level, cell] in the model's scalar order; a moist species the
    file has no increment for (qh) is zero.
    """

    end = moist_block(scalar_names)
    rho = np.ascontiguousarray(increments["rho_amb"], dtype=np.float32)
    scalars = np.zeros((end,) + rho.shape, dtype=np.float32)
    carried = []
    for index, name in enumerate(tuple(scalar_names)[:end]):
        key = f"{name}_amb"
        if key in increments:
            scalars[index] = increments[key]
            carried.append(name)
    return {"u": np.ascontiguousarray(increments["u_amb"], dtype=np.float32), "rho": rho,
            "theta": np.ascontiguousarray(increments["theta_amb"], dtype=np.float32),
            "scalars": scalars, "species": tuple(carried), "moist_end": end}


def _take(values: np.ndarray, index: np.ndarray, width: int, axis: int) -> np.ndarray:
    """values[..., index] padded with zero columns up to ``width`` (a limited-area garbage column)."""

    taken = np.take(values, index, axis=axis)
    pad = width - taken.shape[axis]
    if pad < 0:
        raise ValueError(f"class extent {width} is narrower than its {taken.shape[axis]} mapped entities")
    if pad:
        shape = list(taken.shape)
        shape[axis] = pad
        taken = np.concatenate([taken, np.zeros(shape, dtype=taken.dtype)], axis=axis)
    return np.ascontiguousarray(taken)


@dataclass
class _ClassIau:
    u: Any
    rho: Any
    theta: Any
    scalars: Any
    cells: np.ndarray
    edges: np.ndarray
    itimestep: int = 0
    position: np.ndarray = field(default_factory=lambda: np.array(0.0, dtype=np.float32))
    weight_seconds: float = 0.0
    calls: int = 0
    active_calls: int = 0
    first_start: float | None = None
    max_position_lag_seconds: float = 0.0
    cells_on_edge: Any = None
    zz: Any = None
    released: bool = False


class NativeIauForcing:
    """Per-class device increments and the MPAS IAU clock, applied through woof.hex.iau_v842.

    ``layouts`` give each class its local-to-global cell and edge maps (one identity layout
    for a fixed-step run).  ``apply`` is ``woof.hex.iau_v842.atm_add_tend_anal_incr``.
    """

    def __init__(self, increment: Mapping[str, Any], layouts: Sequence[Any], *, window_seconds: float,
                 mode: str, scalar_names: Sequence[str], cp: Any,
                 apply: Callable[..., None] | None = None, window_start: float = 0.0):
        if not window_seconds > 0:
            raise ValueError("the IAU window must be positive; a zero window is direct insertion")
        self.config = IAUConfig("on", float(window_seconds), mode)
        self.window = float(window_seconds)
        # Model time (seconds since the run's original start) where the window begins: 0 on a cold start,
        # the carried start when a restart continues a window (correct mode, IN3).  A class's correct-mode
        # position starts at its first step's time minus this.
        self.window_start = float(window_start)
        self.mode = mode
        self.cp = cp
        self.scalar_names = tuple(scalar_names)
        self.moist_end = int(increment["moist_end"])
        if self.moist_end != moist_block(self.scalar_names):
            raise ValueError("the increment's moist block does not match the run's scalars")
        self.species = tuple(increment["species"])
        if apply is None:
            from .iau_v842 import atm_add_tend_anal_incr as apply
        self._apply = apply
        self._host = {name: np.asarray(increment[name], dtype=np.float32) for name in ("u", "rho", "theta", "scalars")}
        self.classes: list[_ClassIau] = []
        for layout in layouts:
            cells = np.asarray(layout.cell_l2g, dtype=np.int64)
            edges = np.asarray(layout.edge_l2g, dtype=np.int64)
            self.classes.append(_ClassIau(u=None, rho=None, theta=None, scalars=None, cells=cells, edges=edges))

    def _bind(self, klass: _ClassIau, n_cells: int, n_edges: int) -> None:
        cp = self.cp
        host = self._host
        n_global = host["rho"].shape[1]
        cells = klass.cells
        if cells.size and int(cells.max()) >= n_global:
            # A class that replicates the limited-area garbage element maps it past the last
            # real cell; it has no analysis, so its increment is zero.
            cells = np.where(cells < n_global, cells, -1)
        def cell_take(values, axis):
            safe = np.where(cells < 0, 0, cells)
            taken = _take(values, safe, n_cells, axis)
            if (cells < 0).any():
                index = [slice(None)] * taken.ndim
                index[axis] = np.flatnonzero(cells < 0)
                taken[tuple(index)] = 0
            return taken
        klass.rho = cp.asarray(cell_take(host["rho"], 1))
        klass.theta = cp.asarray(cell_take(host["theta"], 1))
        klass.scalars = cp.asarray(cell_take(host["scalars"], 2))
        klass.u = cp.asarray(_take(host["u"], klass.edges, n_edges, 1))

    def add(self, index: int, held: Any, start: float, dt: float, *, driver: Any, rho_edge: Any,
            normal_velocity: Any, geometry: Any, state: Any, **_unused: Any) -> None:
        """This class step's IAU tendencies onto the held physics tendencies (in place)."""

        klass = self.classes[index]
        if klass.first_start is None:
            klass.first_start = float(start)
            if self.config.correct:
                klass.position[...] = np.float32(float(start) - self.window_start)
        klass.calls += 1
        if klass.released:
            return
        nl, nc = (int(n) for n in held.rho.shape)
        ne = int(held.rho_u.shape[1])
        if self.config.correct:
            lag = abs(float(klass.position) - (float(start) - self.window_start))
            klass.max_position_lag_seconds = max(klass.max_position_lag_seconds, lag)
            weight = atm_iau_coef_window(self.config, float(klass.position), dt)
        else:
            klass.itimestep += 1
            weight = atm_iau_coef(self.config, klass.itimestep, dt)
        if not weight > np.float32(1.0e-12):
            # The window is over for this class (MPAS's own early return): release the device
            # copies.  The correct-mode clock still advances so the receipt shows where it ended.
            klass.u = klass.rho = klass.theta = klass.scalars = None
            klass.released = True
            if self.config.correct:
                klass.position[...] = np.float32(klass.position + np.float32(dt))
            return
        if klass.rho is None:
            self._bind(klass, nc, ne)
        cp = self.cp
        if klass.zz is None:
            zz = driver.atmosphere.vertical.zz
            if not isinstance(zz, cp.ndarray) or tuple(zz.shape) != (nl, nc):
                zz = np.asarray(zz if not isinstance(zz, cp.ndarray) else cp.asnumpy(zz), dtype=np.float32)
                if zz.shape[1] == nc - 1:
                    zz = np.concatenate([zz, np.ones((nl, 1), np.float32)], axis=1)
                zz = cp.asarray(np.ascontiguousarray(zz))
            klass.zz = zz
            cells_on_edge = geometry.cells_on_edge
            garbage = getattr(geometry, "regional_garbage_cell", None)
            if garbage is not None:
                host = np.asarray(cp.asnumpy(cells_on_edge) if isinstance(cells_on_edge, cp.ndarray)
                                  else cells_on_edge, dtype=np.int32).copy()
                host[host == int(garbage)] = -1
                cells_on_edge = cp.asarray(host)
            klass.cells_on_edge = cells_on_edge
        garbage = getattr(geometry, "regional_garbage_cell", None)
        solved = nc - 1 if garbage is not None and int(garbage) == nc - 1 else nc
        theta_m = cp.ascontiguousarray(state.rho_theta / state.rho)
        structs = {
            "mesh": {"nVertLevels": nl, "nCells": nc, "nEdges": ne, "nCellsSolve": solved,
                     "nEdgesSolve": ne, "zz": klass.zz, "cellsOnEdge": klass.cells_on_edge},
            "state": {"rho_zz": state.rho, "theta_m": theta_m, "scalars": state.scalars,
                      "u": normal_velocity, "moist_start": 1, "moist_end": self.moist_end, "index_qv": 1},
            "diag": {"rho_edge": rho_edge, "iau_window_elapsed": klass.position},
            "tend": {"scalars_tend": held.scalars},
            "tend_iau": {"u": klass.u, "rho": klass.rho, "theta": klass.theta, "scalars": klass.scalars},
        }
        self._apply(self.config, structs, klass.itimestep if not self.config.correct else 1, dt,
                    held.rho_u, held.rho_theta, held.rho, cache=driver.cache)
        klass.weight_seconds += float(weight) * float(dt)
        if weight > 0:
            klass.active_calls += 1

    def receipt(self) -> dict[str, Any]:
        rows = [{"weight_times_seconds": klass.weight_seconds, "calls": klass.calls,
                 "active_calls": klass.active_calls, "first_start_seconds": klass.first_start,
                 **({"window_position_seconds": float(klass.position),
                     "max_position_lag_seconds": klass.max_position_lag_seconds}
                    if self.config.correct else {"itimestep": klass.itimestep})}
                for klass in self.classes]
        delivered = [row["weight_times_seconds"] for row in rows]
        # The share of the window this run covers: all of it on a cold start, the rest of it when a
        # restart continues a carried window (correct mode).
        due = [1.0 if not self.config.correct or klass.first_start is None else
               max(0.0, min(1.0, (self.window_start + self.window - max(float(klass.first_start), self.window_start))
                                 / self.window))
               for klass in self.classes]
        return {"route": "native", "kernels": "woof.hex.iau_v842 (MPAS-A v8.4.2 mpas_atm_iau.F)",
                "mode": self.mode,
                "corrections": (["IN1", "IN2", "IN3"] if self.config.correct else []),
                "window_seconds": self.window, "window_start_seconds": self.window_start,
                "species_incremented": list(self.species),
                "moist_end": self.moist_end, "classes": rows,
                # sum of weight*dt per class: 1 means the whole increment was admitted
                "fraction_admitted": delivered, "fraction_due": due,
                "complete": all(abs(value - share) < 1e-5 for value, share in zip(delivered, due)),
                "theta_m": "state.rho_theta / state.rho on the device (hex carries rho*theta_m)",
                "applied_on": "every local cell and edge of the class (halos included); a limited-area "
                              "garbage column is outside nCellsSolve and an exterior endpoint of its edges"}


def identity_layout(n_cells: int, n_edges: int) -> Any:
    """The one class of a fixed-step run: every cell and edge in global order."""

    from types import SimpleNamespace

    return SimpleNamespace(cell_l2g=np.arange(int(n_cells), dtype=np.int64),
                           edge_l2g=np.arange(int(n_edges), dtype=np.int64), n_owned_cells=int(n_cells))


def build_iau_forcing(iau: Mapping[str, Any], *, prepared: Any, scalar_names: Sequence[str],
                      layouts: Sequence[Any], cp: Any, work_dir: Any, n_cells: int | None,
                      n_edges: int | None, n_levels: int | None, pad_cells: int = 0,
                      devices: Sequence[int] | None = None,
                      cell_widths: Sequence[int] | None = None) -> tuple[Any, dict[str, Any]]:
    """The IAU forcing an executor applies, and its receipt, for ``iau["route"]``.

    ``iau``: ``route`` ("native" or "hex"), ``mode`` (physics.mode), ``window_seconds``,
    ``background`` (the run's --init), and ``analysis`` and/or ``increment``.  The native
    route reads MPAS IAU increments (``increment``; formed here from ``analysis`` minus
    ``background`` when none is given); the hex route forms its coupled increment from the
    two initial-conditions files.  ``pad_cells`` appends zero columns (a limited-area
    garbage column) to the hex route's cell increments.
    """

    source = iau_increment_source(iau, prepared=prepared, scalar_names=scalar_names, work_dir=work_dir,
                                  n_cells=n_cells, n_edges=n_edges, n_levels=n_levels, pad_cells=pad_cells)
    try:
        return iau_forcing_from_source(source, layouts=layouts, cp=cp, devices=devices, cell_widths=cell_widths)
    finally:
        source["data"] = None


def iau_increment_source(iau: Mapping[str, Any], *, prepared: Any, scalar_names: Sequence[str], work_dir: Any,
                         n_cells: int | None, n_edges: int | None, n_levels: int | None,
                         pad_cells: int = 0) -> dict[str, Any]:
    """The host half of :func:`build_iau_forcing`: the increment in global order and its receipt, no device work.

    Formed once per run; :func:`iau_forcing_from_source` makes any set of ranks' forcing from it (all ranks in one
    process, or one rank per process on the same-host process executor), so every rank admits the same words.
    """

    import time
    from pathlib import Path

    began = time.perf_counter()
    route = str(iau.get("route") or "hex")
    window = float(iau["window_seconds"])
    window_start = float(iau.get("window_start_seconds") or 0.0)
    mode = None
    if route == "hex":
        from .iau_forcing import analysis_increment

        if iau.get("analysis") is None:
            raise ValueError("--iau-route hex forms its increment from two recoupled initial-conditions "
                             "files and needs --iau-analysis; an MPAS increment file alone runs --iau-route native")
        increment, receipt = analysis_increment(prepared=prepared, scalar_names=scalar_names,
                                                analysis=Path(iau["analysis"]), background=Path(iau["background"]))
        if pad_cells:
            for name, axis in (("rho", 1), ("rho_theta", 1), ("scalars", 2)):
                shape = list(increment[name].shape)
                shape[axis] = pad_cells
                increment[name] = np.concatenate([increment[name], np.zeros(shape, np.float32)], axis=axis)
        data = increment
        receipt.update(route="hex", mode=iau.get("mode"),
                       note=("coupled analysis minus background, recoupled once, added as a constant "
                             "tendency with overlap weights: the IN1-IN3 defects of MPAS's own IAU are not "
                             "in this route's arithmetic, and it carries no restart window position"))
    elif route == "native":
        from .analysis_bundle import read_increment, sha256_file, write_increment

        path = iau.get("increment")
        derived = None
        if path is None:
            if iau.get("analysis") is None:
                raise ValueError("--iau-route native needs --iau-increment, or --iau-analysis to form one")
            path = Path(work_dir) / "iau-increment.nc"
            derived = write_increment(iau["analysis"], iau["background"], path)
        increments = read_increment(path, n_cells=n_cells, n_edges=n_edges, n_levels=n_levels)
        tend_iau = physical_increment(increments, scalar_names)
        mode = str(iau.get("mode") or "exact")
        data = tend_iau

        def rms(values: np.ndarray) -> float:
            return float(np.sqrt(np.mean(np.asarray(values, np.float64) ** 2)))

        receipt = {"route": "native", "mode": mode, "increment": str(path), "increment_sha256": sha256_file(path),
                   "formed_from_analysis_and_background": derived,
                   "rms": {name: rms(values) for name, values in increments.items()},
                   "species": list(tend_iau["species"])}
        if mode == "correct":
            from . import correct_mode

            correct_mode._record(("IN1", "IN2", "IN3"), ())
    else:
        raise ValueError(f"IAU route {route!r} is not 'native' or 'hex'")
    receipt["window_seconds"] = window
    receipt["window_start_seconds"] = window_start
    if iau.get("window_rule"):
        receipt["window_rule"] = iau["window_rule"]
    return {"route": route, "window": window, "window_start": window_start, "mode": mode,
            "scalar_names": tuple(scalar_names), "data": data, "receipt": receipt,
            "host_seconds": time.perf_counter() - began}


def iau_forcing_from_source(source: Mapping[str, Any], *, layouts: Sequence[Any], cp: Any,
                            devices: Sequence[int] | None = None,
                            cell_widths: Sequence[int] | None = None) -> tuple[Any, dict[str, Any]]:
    """The device half of :func:`build_iau_forcing`: ``layouts``' forcing from an increment source."""

    import time

    began = time.perf_counter()
    window, window_start = float(source["window"]), float(source["window_start"])
    if source["route"] == "hex":
        from .iau_forcing import IauForcing

        forcing = IauForcing(source["data"], layouts, window_seconds=window, cp=cp, window_start=window_start,
                             devices=devices, cell_widths=cell_widths)
    else:
        forcing = NativeIauForcing(source["data"], layouts, window_seconds=window, mode=source["mode"],
                                   scalar_names=source["scalar_names"], cp=cp, window_start=window_start)
    receipt = dict(source["receipt"])
    receipt["prepare_seconds"] = float(source["host_seconds"]) + time.perf_counter() - began
    return forcing, receipt


#: The checkpoint key that carries an IAU window across a restart (every executor's checkpoint dict).
IAU_WINDOW_KEY = "iau_window"
_IDENTITIES: dict[tuple, str] = {}


def iau_identity(iau: Mapping[str, Any]) -> str:
    """The increment a window admits: SHA-256 over its route, window length and source files.

    The native route's increment file (or the analysis and background it is formed from); the hex route's
    analysis and background.  Two runs with the same identity admit the same increment.
    """
    from .analysis_bundle import sha256_file

    route = str(iau.get("route") or "hex")
    keys = ("increment",) if route == "native" and iau.get("increment") is not None else ("analysis", "background")
    files = tuple((key, str(iau[key])) for key in keys if iau.get(key) is not None)
    memo = (route, float(iau["window_seconds"]), files)
    if memo not in _IDENTITIES:
        import hashlib

        parts = [route, repr(float(iau["window_seconds"]))] + [f"{key}:{sha256_file(path)}" for key, path in files]
        _IDENTITIES[memo] = hashlib.sha256("|".join(parts).encode()).hexdigest()
    return _IDENTITIES[memo]


def window_record(iau: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """What a checkpoint carries of the run's IAU window (None for a run without one)."""

    if iau is None:
        return None
    return {"schema": "gpuwm-hex.iau-window/v1", "route": str(iau.get("route") or "hex"),
            "mode": str(iau.get("mode") or "exact"), "window_seconds": float(iau["window_seconds"]),
            "window_start_seconds": float(iau.get("window_start_seconds") or 0.0), "identity": iau_identity(iau)}


def resolve_restart_window(iau: Mapping[str, Any] | None, *, carried: Mapping[str, Any] | None,
                           restart_seconds: float) -> dict[str, Any] | None:
    """Where a restarted run's IAU window begins (IN3), from the window its checkpoint carries.

    * The checkpoint carries a window of the SAME increment (same identity, route and length): the run
      continues it, so the increment is admitted once across the restart (correct mode; the hex route).
    * Exact mode on the native route is MPAS-A v8.4.2 word for word: a new window at every run start.
    * Otherwise (a new increment, e.g. a DA-cycling restart, or a checkpoint with no window): a new window
      begins at the restart.  The hex route forms analysis minus --init, which is not a restarted run's
      background, so a new window there needs the MPAS increment file (--iau-increment, the native route).
    Returns the run's IAU request with ``window_start_seconds`` and ``window_rule`` set.
    """

    if iau is None:
        if carried is not None:
            end = float(carried["window_start_seconds"]) + float(carried["window_seconds"])
            if end > restart_seconds:
                print(json.dumps({"iau": {"note": "the checkpoint was written inside an IAU window and this run "
                                                  "names no IAU: the rest of that window is not admitted",
                                          "dropped_seconds": end - restart_seconds}}), flush=True)
        return None
    resolved = dict(iau)
    route, mode = str(iau.get("route") or "hex"), str(iau.get("mode") or "exact")
    if route == "native" and mode == "exact":
        resolved.update(window_start_seconds=float(restart_seconds),
                        window_rule="exact mode: MPAS-A v8.4.2 begins the window at every run start (IN3 left in)")
        return resolved
    identity = iau_identity(iau)
    if (carried is not None and carried.get("identity") == identity and carried.get("route") == route
            and float(carried.get("window_seconds", -1.0)) == float(iau["window_seconds"])):
        start = float(carried["window_start_seconds"])
        resolved.update(window_start_seconds=start,
                        window_rule=f"restart continues the checkpoint's window ({restart_seconds - start:g} s of "
                                    f"{float(iau['window_seconds']):g} s already admitted)")
        return resolved
    if route == "hex":
        # THE BREAKAGE THIS PREVENTS: the hex route's increment is analysis minus the run's --init; after a
        # restart the background is the checkpoint's state, so that increment would be the wrong one.
        raise ValueError("--iau-route hex forms its increment as analysis minus --init, which is not a restarted "
                         "run's background (its checkpoint state); a new IAU window on a restart takes the MPAS "
                         "increment file: --iau-increment FILE --iau-route native")
    dropped = 0.0
    if carried is not None:
        dropped = max(0.0, float(carried["window_start_seconds"]) + float(carried["window_seconds"]) - restart_seconds)
    resolved.update(window_start_seconds=float(restart_seconds),
                    window_rule="a new increment: the window begins at the restart"
                                + (f" (the checkpoint's own window had {dropped:g} s left, not admitted)" if dropped else ""))
    return resolved
