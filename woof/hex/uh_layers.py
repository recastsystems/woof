"""Running-maximum updraft helicity in 0-1, 0-2, 0-3 and 2-5 km AGL layers, on both physics routes.

What it writes.  Four history fields, each the maximum over the output interval (reset at every
history frame, as HRRR's hourly-max UH and WRF's ``UP_HELI_MAX`` are), in m2 s-2:

* ``updraft_helicity_max``     2-5 km  (on the MPAS-physics routes written here only when no output
  stream of the run carries MPAS's own field; MPAS's native 2-5 km words are never replaced)
* ``updraft_helicity_max_01``  0-1 km
* ``updraft_helicity_max_02``  0-2 km
* ``updraft_helicity_max_03``  0-3 km

The names are RRFS-MPAS's (MPAS-Model-rrfs Registry), so ``rw_mpas_convert`` already maps them to
``UP_HELI_MAX`` / ``UP_HELI_MAX01`` / ``UP_HELI_MAX02`` / ``UP_HELI_MAX03`` in the wrfout frames the
renderer reads.

The formula is MPAS's (mpas_convective_diagnostics.F:99-198, the 2-5 km ``updraft_helicity_max``):
the vertex relative vorticity kite-averaged to the cell (sum of kiteAreasOnVertex * vorticity over
the cell's vertices, over areaCell), times max(w, 0) at mid-layer, each factor floored at zero,
summed over the part of each layer inside the band above the ground.  The 0-x km bands use the
same column pass, so all four come from one kernel launch per step.

Why it exists: a WOOF severe-weather user asked for 0-3 km UH (2026-11-02).  Only the RRFS route
formed the low-level layers; the MPAS-A v8.4.2 route (the hex default) formed 2-5 km only and the
ArWen-physics route on hex formed none.  One implementation serves both routes, so they write the
same fields with the same names and the same running maximum.

Defaults, per route.  The MPAS-physics routes write MPAS's own 2-5 km ``updraft_helicity_max`` in every run, so
the ArWen-physics route writes the same field from here by default (``ARWEN_DEFAULT_NAMES``): every hex capability
on every physics route.  THE BREAKAGE THIS PREVENTS (both-routes matrix, woof hex 0.4.14 candidate, 2026-11-06):
the MPAS route rendered composite_reflectivity_uh, uh_2to5km and 1km_reflectivity_uh2to5km and the WOOF route rendered none
of them (global leg, 225 vs 207 maps).  ``WOOF_HEX_UH_LAYERS`` on (the door's ``--uh-layers on``) adds the 0-1/0-2/
0-3 km layers on both routes; off writes none of these on the WOOF route (the MPAS routes keep MPAS's own 2-5 km)
and then nothing here is constructed and no kernel is compiled.  When armed: one kernel per step (plus the dycore's
own vorticity solve where the step did not already form it), no host sync per step; one small device-to-host copy
per history frame.

Partitioned WOOF executors (the full-step LTS classes, one card or several, and same-host multi-card ranks) arm
one set of layers per class or rank over its own local mesh (``attach_rank``), fold it after each of that class's
or rank's steps (``rank_update``) and hand its words to the history capture (``rank_history_arrays``), whose owned
blocks the executors already scatter to global cells by name (samehost_identity_v841.assemble_owned_arrays).  Each
cell sums its kites in its own ``verticesOnCell`` order (MPAS's loop, and the same on every partition), so a
partition's owned words are the whole mesh's words whenever its atmosphere is.

Known gap: same-host multi-rank MPAS-physics runs (several cards in one process) do not get the
fields yet; the gathered writer would need each rank's owned words.  The receipt says so.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Mapping

import numpy as np

#: Environment knob both routes read (the forecast door's ``--uh-layers`` sets it for its driver).
ENV = "WOOF_HEX_UH_LAYERS"

#: (history name, bottom m AGL, top m AGL), in kernel band order.
BANDS = (
    ("updraft_helicity_max", 2000.0, 5000.0),
    ("updraft_helicity_max_01", 0.0, 1000.0),
    ("updraft_helicity_max_02", 0.0, 2000.0),
    ("updraft_helicity_max_03", 0.0, 3000.0),
)
NAMES = tuple(name for name, _, _ in BANDS)
UNITS = "m2 s-2"

_SOURCE = r"""
extern "C" __global__ void uh_layers_update(
    const int n, const int nl, const int degree,
    const int* __restrict__ offsets, const int* __restrict__ entries,
    const float* __restrict__ kite,
    const float* __restrict__ vort, const long long vstride,
    const float* __restrict__ w, const long long wstride,
    const float* __restrict__ z, const long long zstride,
    const float* __restrict__ area, float* __restrict__ acc)
{
    const int c = blockDim.x * blockIdx.x + threadIdx.x;
    if (c >= n) return;
    const float bot[4] = {2000.0f, 0.0f, 0.0f, 0.0f};
    const float top[4] = {5000.0f, 1000.0f, 2000.0f, 3000.0f};
    float r[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const float z0 = z[c];
    const float a = area[c];
    const int e0 = offsets[c], e1 = offsets[c + 1];
    for (int k = 0; k < nl; k++) {
        const float zl = z[(long long)k * zstride + c] - z0;
        if (zl >= 5000.0f) break;                      // above every band
        const float zu = z[(long long)(k + 1) * zstride + c] - z0;
        float s = 0.0f;
        for (int e = e0; e < e1; e++) {
            const int ix = entries[e];
            s += kite[ix] * vort[(long long)k * vstride + ix / degree];
        }
        const float wm = 0.5f * (w[(long long)k * wstride + c] + w[(long long)(k + 1) * wstride + c]);
        const float h = fmaxf(wm, 0.0f) * fmaxf(s / a, 0.0f);
        #pragma unroll
        for (int b = 0; b < 4; b++) {
            const float zb = fmaxf(zl, bot[b]), zt = fminf(zu, top[b]);
            r[b] += h * fmaxf(0.0f, zt - zb);
        }
    }
    #pragma unroll
    for (int b = 0; b < 4; b++) acc[(long long)b * n + c] = fmaxf(r[b], acc[(long long)b * n + c]);
}
"""


def enabled() -> bool:
    """True when the run asked for the layers (``WOOF_HEX_UH_LAYERS`` = 1/on/true/yes)."""

    return os.environ.get(ENV, "").strip().lower() in ("1", "on", "true", "yes")


#: What the ArWen-physics route writes when the run names no ``--uh-layers``: the 2-5 km maximum, the field every
#: MPAS-physics run carries natively.
ARWEN_DEFAULT_NAMES = ("updraft_helicity_max",)


def arwen_names() -> tuple[str, ...]:
    """The fields the ArWen-physics route writes: all four when asked for, none when turned off, else the default."""

    value = os.environ.get(ENV, "").strip().lower()
    if value in ("1", "on", "true", "yes"):
        return NAMES
    if value in ("0", "off", "false", "no"):
        return ()
    return ARWEN_DEFAULT_NAMES


def _plan(cells_on_vertex: np.ndarray, n: int, vertices_on_cell: np.ndarray | None = None,
          n_edges_on_cell: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Per cell, the flat (vertex*degree + slot) indices that name it, as CSR (offsets, entries).

    ``cells_on_vertex`` is (nVertices, degree), zero-based; slots naming no cell in [0, n) (MPAS's
    nCells+1 sentinel, a culled neighbour) are dropped.  Built once per run on the host.

    With ``vertices_on_cell`` (nCells, maxEdges, zero-based) and ``n_edges_on_cell``, each cell's entries
    follow its own vertex order (MPAS's ``do i=1,nEdgesOnCell`` loop), so the kite sum is the same sum on a
    partition-local mesh, whose vertices are renumbered, as on the whole mesh.  THE BREAKAGE THIS PREVENTS:
    ordered by flat vertex index, a cell's sum ran in global vertex order on one card and in local order on a
    rank, so the ranks' owned UH words could differ from the one-card run's in the last bits while every
    atmosphere word was equal.  A vertex whose row does not name the cell (a clamped halo reference) is dropped.
    """

    if vertices_on_cell is not None and n_edges_on_cell is not None:
        cov2 = np.asarray(cells_on_vertex, dtype=np.int64)
        degree = int(cov2.shape[1])
        voc = np.asarray(vertices_on_cell, dtype=np.int64)[:n]
        count = np.asarray(n_edges_on_cell, dtype=np.int64).reshape(-1)[:n]
        valid = (np.arange(voc.shape[1])[None, :] < count[:, None]) & (voc >= 0) & (voc < cov2.shape[0])
        vertex = np.where(valid, voc, 0)
        match = cov2[vertex] == np.arange(n, dtype=np.int64)[:, None, None]
        present = valid & match.any(axis=2)
        entries = (vertex * degree + match.argmax(axis=2))[present].astype(np.int32)
        offsets = np.zeros(n + 1, dtype=np.int32)
        offsets[1:] = np.cumsum(present.sum(axis=1)).astype(np.int32)
        return offsets, entries
    cov = np.asarray(cells_on_vertex, dtype=np.int64).ravel()
    flat = np.nonzero((cov >= 0) & (cov < n))[0]
    cells = cov[flat]
    order = np.argsort(cells, kind="stable")
    entries = flat[order].astype(np.int32)
    offsets = np.zeros(n + 1, dtype=np.int32)
    offsets[1:] = np.cumsum(np.bincount(cells, minlength=n)[:n]).astype(np.int32)
    return offsets, entries


class UpdraftHelicityLayers:
    """The four running maxima on the card, updated once a step, taken (and reset) per history frame."""

    def __init__(self, cp: Any, *, cells_on_vertex: np.ndarray, kite_areas_on_vertex: np.ndarray,
                 area_cell: np.ndarray, n_cells: int, n_out: int | None = None,
                 names: tuple[str, ...] = NAMES, vertices_on_cell: np.ndarray | None = None,
                 n_edges_on_cell: np.ndarray | None = None) -> None:
        cov = np.asarray(cells_on_vertex)
        kite = np.asarray(kite_areas_on_vertex, dtype=np.float32)
        if cov.ndim != 2 or kite.shape != cov.shape:
            raise ValueError(f"cellsOnVertex {cov.shape} and kiteAreasOnVertex {kite.shape} must be (nVertices, degree)")
        self.cp = cp
        self.n = int(n_cells)
        self.n_out = int(n_out if n_out is not None else n_cells)
        if self.n_out < self.n:
            raise ValueError("history cell count is smaller than the computed cells")
        self.names = tuple(names)
        self.degree = int(cov.shape[1])
        offsets, entries = _plan(cov, self.n, vertices_on_cell, n_edges_on_cell)
        self.offsets = cp.asarray(offsets)
        self.entries = cp.asarray(entries)
        self.kite = cp.asarray(np.ascontiguousarray(kite.ravel()))
        self.area = cp.asarray(np.ascontiguousarray(np.asarray(area_cell, dtype=np.float32)[:self.n]))
        self.acc = cp.zeros((len(BANDS), self.n_out), dtype=cp.float32)
        self._kernel = cp.RawKernel(_SOURCE, "uh_layers_update")
        self._zgrid = None
        self.updates = 0
        self.frames = 0
        # A fault here never stops the forecast: the layers switch off for the run, loudly (stderr + receipt),
        # and their frames then simply lack the fields (the renderer names those maps unavailable).
        self.error = None

    def fail(self, where: str, error: BaseException) -> None:
        if self.error is None:
            self.error = f"{where}: {type(error).__name__}: {error}"
            print(f"uh_layers: switched off for this run ({self.error}); the forecast goes on without the UH layers",
                  file=sys.stderr, flush=True)

    @staticmethod
    def _rows(array: Any, rows: int, name: str) -> Any:
        if array.ndim != 2 or array.shape[0] < rows:
            raise ValueError(f"{name} must be (levels, cells), got {array.shape}")
        return array

    def _float32(self, array: Any) -> Any:
        cp = self.cp
        if array.dtype != cp.float32:
            array = array.astype(cp.float32)
        return array if array.flags.c_contiguous else cp.ascontiguousarray(array)

    def update(self, *, vorticity: Any, w: Any, zgrid: Any) -> None:
        """Fold this step's state: vorticity (nl, vertices), w and zgrid (nl+1, cells), level-major."""

        if self.error is not None:
            return
        try:
            self._update(vorticity, w, zgrid)
        except Exception as error:  # noqa: BLE001 -- an optional diagnostic never stops the forecast
            self.fail("update", error)

    def _update(self, vorticity: Any, w: Any, zgrid: Any) -> None:
        nl = int(vorticity.shape[0])
        vort = self._float32(self._rows(vorticity, nl, "vorticity"))
        w = self._float32(self._rows(w, nl + 1, "w"))
        z = zgrid
        if z.dtype != self.cp.float32 or not z.flags.c_contiguous:
            # The mesh heights do not move: keep one float32 contiguous copy.
            if self._zgrid is None or self._zgrid[0] is not zgrid:
                self._zgrid = (zgrid, self._float32(zgrid))
            z = self._zgrid[1]
        self._rows(z, nl + 1, "zgrid")
        if min(int(w.shape[1]), int(z.shape[1])) < self.n:
            raise ValueError(f"w/zgrid hold fewer than {self.n} cells")
        self._kernel(((self.n + 127) // 128,), (128,), (
            np.int32(self.n), np.int32(nl), np.int32(self.degree),
            self.offsets, self.entries, self.kite,
            vort, np.int64(vort.shape[1]), w, np.int64(w.shape[1]), z, np.int64(z.shape[1]),
            self.area, self.acc))
        self.updates += 1

    def take(self) -> dict[str, np.ndarray]:
        """This interval's maxima on the host, by name; the maxima restart from zero."""

        if self.error is not None:
            return {}
        try:
            host = self.cp.asnumpy(self.acc)
            self.acc.fill(0)
        except Exception as error:  # noqa: BLE001
            self.fail("take", error)
            return {}
        self.frames += 1
        return {name: host[NAMES.index(name)] for name in self.names}

    def history_fields(self) -> dict[str, tuple[np.ndarray, tuple[str, ...], str]]:
        """``take()`` in the MPAS writer's extra-field form: name -> (array, ('nCells',), units)."""

        return {name: (array, ("nCells",), UNITS) for name, array in self.take().items()}

    def receipt(self) -> dict[str, Any]:
        return {"fields": list(self.names), "updates": self.updates, "frames": self.frames,
                "cells": self.n, "kernel": "uh_layers_update, one launch per step", "error": self.error}


def _zero_based(index: np.ndarray, limit: int) -> np.ndarray:
    index = np.asarray(index, dtype=np.int64)
    if index.size and index.min() >= 1 and index.max() >= limit:
        index = index - 1  # one-based file convention
    return index


def attach_mpas(diagnostics: Any, output: Any, *, halo: Any = None) -> dict[str, Any] | None:
    """Arm the layers on the MPAS-physics routes (v8.4.2 and RRFS); returns its receipt row or None.

    The maxima fold at every running-maximum update the route's output diagnostics already make
    (``_update_convective``: once a step, deferred to the next synchronize, and at every output
    event), on the same pool words MPAS's own 2-5 km maximum reads.  They are written as extra
    history fields (``PhysicsStreams.extra_history``) and reset at each history frame.  A name the
    run's history stream already carries (RRFS's Registry fields, MPAS's 2-5 km when the deck lists
    it) is left to the route's own computation.
    """

    if not enabled():
        return None
    try:
        return _attach_mpas(diagnostics, output, halo=halo)
    except Exception as error:  # noqa: BLE001 -- an optional diagnostic never stops the forecast
        print(f"uh_layers: not armed ({type(error).__name__}: {error}); the forecast goes on without the UH layers",
              file=sys.stderr, flush=True)
        return {"enabled": False, "error": f"{type(error).__name__}: {error}"}


def _attach_mpas(diagnostics: Any, output: Any, *, halo: Any = None) -> dict[str, Any]:
    if halo is not None:
        return {"enabled": False, "reason": "same-host multi-rank output not wired for uh layers yet"}
    if diagnostics is None or not hasattr(output, "streams"):
        return {"enabled": False, "reason": "this route has no resident output diagnostics"}
    # A name any output stream of the run already writes (MPAS's 2-5 km in diag.*.nc, RRFS's layers) stays
    # the route's own: rw_mpas_convert reads a field the history lacks from the diag file beside it.
    written: set[str] = set()
    for stream in output.streams.values():
        if "output" in stream.options["type"].split(";"):
            written.update(output.stream_fields(stream))
    names = tuple(name for name in NAMES if name not in written)
    if not names:
        return {"enabled": False, "reason": "the run's output streams already carry every layer"}
    cp = diagnostics.cp
    dims = diagnostics.session.layout.dimensions
    n_solve, n_cells = int(dims["nCellsSolve"]), int(dims["nCells"])
    mesh, state = diagnostics.pools["mesh"], diagnostics.pools["state"]
    # Native (degree, nVertices), one-based with the nCells+1 sentinel; _plan drops the sentinel.
    cov = cp.asnumpy(mesh["cellsOnVertex"]).T.astype(np.int64) - 1
    kite = cp.asnumpy(mesh["kiteAreasOnVertex"]).T
    area = cp.asnumpy(mesh["areaCell"]).reshape(-1)[:n_solve]
    layers = UpdraftHelicityLayers(cp, cells_on_vertex=cov, kite_areas_on_vertex=kite, area_cell=area,
                                   n_cells=n_solve, n_out=n_cells, names=names)
    original = diagnostics._update_convective

    def update_convective_and_layers():
        original()
        if layers.error is None:
            try:
                layers.update(vorticity=diagnostics.buffer("diag", "vorticity"), w=state.view("w"),
                              zgrid=mesh.view("zgrid"))
            except Exception as error:  # noqa: BLE001 -- an optional diagnostic never stops the forecast
                layers.fail("pool views", error)

    diagnostics._update_convective = update_convective_and_layers
    # The deferred per-step update (and its vorticity solve) runs only for an accumulating run.
    diagnostics.accumulates = True
    previous = getattr(output, "extra_history", None)

    def extra_history():
        fields = layers.history_fields()
        cells = int(diagnostics.session.layout.dimensions["nCells"])
        if fields and cells != layers.n_out:
            # A frame whose cell count moved (a re-cut region) never gets words of another mesh.
            layers.fail("history", ValueError(f"nCells {cells} differs from the {layers.n_out} the layers hold"))
            fields = {}
        return {**(previous() if previous else {}), **fields}

    output.extra_history = extra_history
    diagnostics.uh_layers = layers
    return {"enabled": True, **layers.receipt()}


def _mesh_member(mesh: Any, name: str) -> Any:
    arrays = getattr(mesh, "arrays", None)
    if isinstance(arrays, Mapping) and name in arrays:
        return arrays[name]
    return getattr(mesh, name, None)


def for_arwen(cp: Any, mesh: Any, *, n_cells: int, n_out: int | None = None) -> UpdraftHelicityLayers | None:
    """The layers on the ArWen-physics route (the hex dycore's mesh, whole or one partition's local mesh), or None
    when turned off (``arwen_names``).  ``n_out`` is the history's cell extent when it carries more columns than
    the mesh solves (a limited-area driver's garbage column)."""

    names = arwen_names()
    if not names:
        return None
    try:
        n_cells = int(n_cells)
        cov = _zero_based(np.asarray(_mesh_member(mesh, "cellsOnVertex")), n_cells)
        voc = _mesh_member(mesh, "verticesOnCell")
        count = _mesh_member(mesh, "nEdgesOnCell")
        if voc is not None and count is not None:
            voc = _zero_based(np.asarray(voc), int(cov.shape[0]))
        else:
            voc = count = None
        return UpdraftHelicityLayers(cp, cells_on_vertex=cov,
                                     kite_areas_on_vertex=np.asarray(_mesh_member(mesh, "kiteAreasOnVertex")),
                                     area_cell=np.asarray(_mesh_member(mesh, "areaCell")), n_cells=n_cells,
                                     n_out=n_out, names=names, vertices_on_cell=voc, n_edges_on_cell=count)
    except Exception as error:  # noqa: BLE001 -- an optional diagnostic never stops the forecast
        print(f"uh_layers: not armed ({type(error).__name__}: {error}); the forecast goes on without the UH layers",
              file=sys.stderr, flush=True)
        return None


def update_from_driver(layers: UpdraftHelicityLayers, driver: Any) -> None:
    """Fold the driver's committed atmosphere (ArWen-physics route; call after each step)."""

    if layers.error is not None:
        return
    try:
        _update_from_driver(layers, driver)
    except Exception as error:  # noqa: BLE001
        layers.fail("vorticity", error)


def _update_from_driver(layers: UpdraftHelicityLayers, driver: Any) -> None:
    atmosphere = driver.atmosphere
    state, saved = atmosphere.state, atmosphere.saved
    horizontal = driver.horizontal
    if hasattr(horizontal, "solve_vorticity"):
        vorticity = horizontal.solve_vorticity(state.rho, state.rho_u, normal_velocity=saved.normal_velocity)
    else:
        vorticity = horizontal.solve_diagnostics(
            state.rho, state.rho_u, dt=float(driver.config.config_dt),
            apvm_upwinding=float(getattr(driver.config, "config_apvm_upwinding", 0.0)),
            normal_velocity=saved.normal_velocity, rk_step=3).vorticity
    layers.update(vorticity=vorticity, w=saved.vertical_velocity, zgrid=atmosphere.vertical.zgrid)


def arwen_history_arrays(layers: UpdraftHelicityLayers) -> Mapping[str, np.ndarray]:
    """``take()`` for the ArWen-route writer, which writes every (nCells,) array by name."""

    return layers.take()


def attach_rank(stack: dict[str, Any], mesh: Any, cp: Any, *, initial: bool) -> dict[str, Any] | None:
    """Arm the layers on one partition (an LTS class or a same-host rank) of the ArWen-physics route.

    ``mesh`` is the partition's local mesh (its solved cells first); the history extent is the driver's cell
    count.  ``initial``: fold the start state now, as the one-card driver does before its F000 frame (not on a
    restart or a resumed moving segment, whose first frame is not re-captured).  Returns the receipt row, or None
    when the run turned the layers off.
    """

    driver = stack["driver"]
    solved = int(np.asarray(_mesh_member(mesh, "areaCell")).reshape(-1).size)
    layers = for_arwen(cp, mesh, n_cells=min(solved, int(driver.horizontal.ncells)),
                       n_out=int(driver.horizontal.ncells))
    stack["uh_layers"] = layers
    if layers is None:
        return None
    if initial:
        update_from_driver(layers, driver)
    return layers.receipt()


def rank_update(stack: Mapping[str, Any]) -> None:
    """Fold this partition's committed step (call after its step, inside its device scope)."""

    layers = stack.get("uh_layers")
    if layers is not None:
        update_from_driver(layers, stack["driver"])


def rank_history_arrays(stack: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """This partition's interval maxima for its history capture (every name on every partition).

    A partition whose layers switched off after a fault hands zeros under the same names: every rank's capture
    must carry the same fields (assemble_owned_arrays refuses diverging paths), and a fault must never stop the
    forecast.  The fault is on stderr and in the layers' receipt.
    """

    layers = stack.get("uh_layers")
    if layers is None:
        return {}
    taken = layers.take()
    if not taken:
        return {name: np.zeros(layers.n_out, dtype=np.float32) for name in layers.names}
    return taken
