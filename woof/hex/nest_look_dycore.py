"""``[hex_nests]`` sources both physics routes form from the hex dycore's own resident arrays.

THE BREAKAGE THIS PREVENTS (WOOF owner canary run_9948508bd7cfdee3, 2026-10-29): the trigger
table (``data/nest-trigger-fields.v1.json``) offers ``wmax`` (source ``w``), ``mucape``
(``cape``) and ``uh25`` (``updraft_helicity_max``), and the WOOF Hex form offers Updraft and
CAPE chips, but both routes passed ``refl10cm`` as their only source.  Every such table was
refused at arm time ("which this physics route does not form on the card"), after its box had
been rented.  ``w`` is the dycore's own resident array, so leaving it out was a binding
omission; CAPE and 2-5 km updraft helicity are formed here from the dycore's arrays.

Both routes run the same CUDA dycore (``driver.atmosphere``), so one implementation serves
both and a fix lands on both at once.  Everything runs on the card, on a look step only, over
the trigger area's cells only:

* ``w``: ``atmosphere.saved.vertical_velocity`` (nVertLevels+1, nCells), the end-of-step w the
  history writer writes.  No copy.
* ``cape``: most-unstable CAPE from MPAS-A v8.4.2's own convective-diagnostics kernel
  (``woof.hex.mpasdiag.convective_v842`` ``compute_diag``, mask cape|cin: getcape with
  source=2, the most-unstable parcel), fed the area's columns of the dycore's theta_m
  (rho_theta / rho), qv, exner, pressure_base and pressure_p.  Chunked so its work array stays
  bounded.  Written into a scratch array, never into an output pool: no output word changes.
* ``updraft_helicity_max``: 2-5 km updraft helicity at the look, MPAS's formula
  (mpas_convective_diagnostics.F:99-198): the vertex vorticity (the dycore's own
  ``solve_vorticity``) kite-averaged to the cell, times max(w, 0) at mid-layer, over the part
  of each layer between 2 and 5 km above the ground.  Each look is its own value (no running
  maximum across looks: a look tests the storm it sees).

A route adds these to the sources it forms (``SOURCES``); the table's rows say which field
reads which source, and nothing here branches on a field name.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

#: The trigger sources formed here from the dycore, on every physics route.
SOURCES = ("w", "cape", "updraft_helicity_max")

#: Area columns per CAPE kernel launch: the kernel's work array is (columns, 17*nlev+3) float32,
#: 3.7 kB per column at 55 levels, so a chunk holds about 120 MB however big the area is.
CAPE_CHUNK_COLUMNS = 32768


def _zero_based(index: np.ndarray, limit: int) -> np.ndarray:
    index = np.asarray(index, dtype=np.int64)
    if index.size and index.min() >= 1 and index.max() >= limit:
        index = index - 1  # one-based file convention
    return index


def helicity_weights(mesh: Any, cells: np.ndarray, *, owned_cells: int) -> tuple[np.ndarray, np.ndarray]:
    """Per area cell, its vertices and kite weights (kiteArea / areaCell): (n_area, maxEdges) each.

    A slot past the cell's edge count, or a vertex that does not list the cell, has weight 0.
    """

    n_vertices = int(np.asarray(mesh.cellsOnVertex).shape[0])
    voc = _zero_based(np.asarray(mesh.verticesOnCell)[:owned_cells], n_vertices)[cells]
    nedges = np.asarray(mesh.nEdgesOnCell, dtype=np.int64)[:owned_cells][cells]
    cov = _zero_based(np.asarray(mesh.cellsOnVertex), int(owned_cells))
    kite = np.asarray(mesh.kiteAreasOnVertex, dtype=np.float64)
    area = np.asarray(mesh.areaCell, dtype=np.float64)[:owned_cells][cells]
    slot = np.arange(voc.shape[1])[None, :]
    valid = (slot < nedges[:, None]) & (voc >= 0) & (voc < n_vertices)
    vertex = np.where(valid, voc, 0)
    # Which of the vertex's cells is this cell: (n_area, maxEdges, degree).
    match = cov[vertex] == cells[:, None, None]
    weight = np.where(valid & match.any(axis=2),
                      (kite[vertex] * match).sum(axis=2) / area[:, None], 0.0)
    return vertex.astype(np.int64), weight.astype(np.float32)


class DycoreLookSources:
    """The dycore sources a trigger needs, formed on its look steps (see the module text)."""

    def __init__(self, cp: Any, *, needed: Sequence[str], cells: np.ndarray, owned_cells: int,
                 qv_index: int | None, mesh: Any | None = None) -> None:
        self.cp = cp
        self.needed = tuple(name for name in SOURCES if name in set(needed))
        self.owned_cells = int(owned_cells)
        self.cells_host = np.asarray(cells, dtype=np.int64)
        self.cells = cp.asarray(self.cells_host)
        self.qv_index = qv_index
        if "cape" in self.needed and qv_index is None:
            raise ValueError("[hex_nests] a cape test needs the run's water vapour (qv): this run carries none")
        self._full: dict[str, Any] = {}
        self._helicity = None
        if "updraft_helicity_max" in self.needed:
            if mesh is None:
                raise ValueError("[hex_nests] updraft helicity needs the mesh's vertices and kite areas")
            vertex, weight = helicity_weights(mesh, self.cells_host, owned_cells=self.owned_cells)
            self._helicity = (cp.asarray(vertex), cp.asarray(weight))
        self._cape_module = None
        if "cape" in self.needed:
            # Compiled at arm time, so a missing compiler or header fails before step 1.
            self._cape_module = self._module()

    def __bool__(self) -> bool:
        return bool(self.needed)

    def _module(self) -> Any:
        import os

        from .mpasdiag import convective_v842

        cp = self.cp
        return convective_v842._module(cp.cuda.Device().id, os.environ.get("MPAS_CONVECTIVE_IMAGE")
                                       or os.environ.get("MPAS_CONVECTIVE_PTX"))

    def _target(self, name: str) -> Any:
        cp = self.cp
        if name not in self._full:
            self._full[name] = cp.zeros((self.owned_cells,), dtype=cp.float32)
        return self._full[name]

    def form(self, driver: Any) -> dict[str, Any]:
        """The needed sources from the driver's committed atmosphere (call after the step)."""

        atmosphere = driver.atmosphere
        out: dict[str, Any] = {}
        if "w" in self.needed:
            out["w"] = atmosphere.saved.vertical_velocity[:, :self.owned_cells]
        if "cape" in self.needed:
            out["cape"] = self._cape(atmosphere)
        if "updraft_helicity_max" in self.needed:
            out["updraft_helicity_max"] = self._updraft_helicity(driver, atmosphere)
        return out

    def _cape(self, atmosphere: Any) -> Any:
        cp = self.cp
        state, saved = atmosphere.state, atmosphere.saved
        zgrid = atmosphere.vertical.zgrid
        nl = int(state.rho.shape[0])
        target = self._target("cape")
        rvord = np.float32(np.float32(461.6) / np.float32(287.0))
        function = self._cape_module.get_function("compute_diag")
        for start in range(0, int(self.cells.size), CAPE_CHUNK_COLUMNS):
            cells = self.cells[start:start + CAPE_CHUNK_COLUMNS]
            n = int(cells.size)

            def columns(array: Any) -> Any:
                return cp.ascontiguousarray(array[:, cells].T.astype(cp.float32, copy=False))

            height = columns(zgrid)
            # theta_m from the prognostic pair, so a look after the step's microphysics sees it.
            theta = cp.ascontiguousarray((state.rho_theta[:, cells] / state.rho[:, cells]).T.astype(cp.float32,
                                                                                                    copy=False))
            qv = columns(state.scalars[self.qv_index])
            exner = columns(saved.exner)
            pb = columns(atmosphere.reference.pressure_base)
            pp = columns(saved.pressure_perturbation)
            # CAPE reads no wind; the kernel copies level 1 of u and v into its (scratch) output.
            wind = cp.zeros((n, nl), dtype=cp.float32)
            result = cp.zeros((17, n), dtype=cp.float32)
            work = cp.empty((n, 17 * nl + 3), dtype=cp.float32)
            function(((n + 127) // 128,), (128,),
                     (np.int32(n), np.int32(n), np.int32(nl), np.uint32(3), rvord,
                      height, wind, wind, theta, qv, exner, pb, pp, result, work))
            target[cells] = result[0]
        return target

    def _updraft_helicity(self, driver: Any, atmosphere: Any) -> Any:
        cp = self.cp
        state, saved = atmosphere.state, atmosphere.saved
        horizontal = driver.horizontal
        if hasattr(horizontal, "solve_vorticity"):
            vorticity = horizontal.solve_vorticity(state.rho, state.rho_u, normal_velocity=saved.normal_velocity)
        else:
            vorticity = horizontal.solve_diagnostics(
                state.rho, state.rho_u, dt=float(driver.config.config_dt),
                apvm_upwinding=float(getattr(driver.config, "config_apvm_upwinding", 0.0)),
                normal_velocity=saved.normal_velocity, rk_step=3).vorticity
        vertex, weight = self._helicity
        nl = int(vorticity.shape[0])
        zeta = cp.zeros((nl, int(self.cells.size)), dtype=cp.float32)
        for slot in range(int(vertex.shape[1])):
            zeta += weight[:, slot][None, :] * vorticity[:, vertex[:, slot]]
        w = saved.vertical_velocity[:, self.cells]
        z = atmosphere.vertical.zgrid[:, self.cells]
        above = z - z[0:1]
        bottom = cp.maximum(above[:-1], 2000.0)
        top = cp.minimum(above[1:], 5000.0)
        mid_w = cp.maximum(0.5 * (w[:-1] + w[1:]), 0.0)
        layer = mid_w * cp.maximum(zeta, 0.0) * cp.maximum(top - bottom, 0.0)
        target = self._target("updraft_helicity_max")
        target[self.cells] = layer.sum(axis=0).astype(cp.float32)
        return target


def for_trigger(trigger: Any, cp: Any, *, owned_cells: int, qv_index: int | None,
                mesh: Any | None) -> DycoreLookSources | None:
    """The dycore sources an armed trigger reads, or None when it reads none of them."""

    if trigger is None:
        return None
    needed = [name for name in trigger.needed_sources() if name in SOURCES]
    if not needed:
        return None
    return DycoreLookSources(cp, needed=needed, cells=np.asarray(trigger.evaluator.geometry.cells),
                             owned_cells=owned_cells, qv_index=qv_index, mesh=mesh)


def look_arrays(*, refl: Any | None, needed: Sequence[str], dycore: DycoreLookSources | None,
                driver: Any) -> Mapping[str, Any] | None:
    """Every source a look needs, or None when the step formed no reflectivity a test reads."""

    arrays: dict[str, Any] = {}
    if "refl10cm" in needed:
        if refl is None:
            return None
        arrays["refl10cm"] = refl
    if dycore:
        arrays.update(dycore.form(driver))
    return arrays
