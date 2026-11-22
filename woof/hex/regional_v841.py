"""MPAS-A regional (limited-area) runtime, audited against v8.4.2.

The historical module name is retained for the existing driver interface.
Direct strict-GNU replay: tests/test_regional_native_v842.py.

Transcribed against the native v8.4.1 sources of the pinned reference rig
(``src/MPAS-intel/src/core_atmosphere``, the exact tree that built the
CANDIDATE-REGIONAL / CANDIDATE-REGIONAL-DRY reference executables):

* ``mpas_atm_boundaries.F``: zone constants, ``specZoneMask*`` derivation,
  ``nearestRelaxationCell``, the two-level LBC value/tendency pool semantics
  (``mpas_atm_update_bdy_tend`` / ``mpas_atm_get_bdy_state`` /
  ``mpas_atm_get_bdy_tend``) and the limited-area admission checks
  (``mpas_atm_bdy_checks``).
* ``mpas_atm_time_integration.F``: the regional insertions in ``atm_srk3``
  and the boundary-adjust subroutines at F:7839-8505.
* ``mpas_atm_core.F:1123-1180``, ``atm_compute_mesh_scaling`` for
  ``meshScalingRegionalCell/Edge``.

Everything here is a transcription, not a design: where the native source
carries a quirk, the quirk is replicated and documented at the site, because
the dycore pin law makes native bytes the definition of correct.  All
arithmetic is performed in the state dtype (float32 for the pinned
authority) with native operation order.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from . import rkind_libm
from .native_powf import powf_native_gnu
from .errors import ConfigurationRefusal, MpasPortError
from .lbc import LbcAdmissionError, LbcFile, LbcInventory, LbcPool, rkind_interval_seconds

FloatArray = NDArray[np.floating[Any]]
IntArray = NDArray[np.integer[Any]]

#: mpas_atm_boundaries.F:36-38.  The nearestRelaxationCell derivation below
#: assumes nSpecZone == 2 exactly as the native comment at F:34-35 states.
N_SPEC_ZONE = 2
N_RELAX_ZONE = 5
N_BDY_ZONE = N_SPEC_ZONE + N_RELAX_ZONE

#: mpas_constants.F: ``rvord = rv/rgas`` is a compile-time REAL(RKIND)
#: parameter division of the float32 constants 461.6 and 287.0.
RVORD_F32 = np.float32(np.float32(461.6) / np.float32(287.0))


def scalar_index_for_field(name: str, scalar_names: Sequence[str]) -> int | None:
    """Resolve native constituent fallback (mpas_atm_boundaries.F:398-419,535-548)."""
    try:
        return scalar_names.index(f"lbc_{name}")
    except ValueError:
        return None


def _mesh_array(mesh: object, name: str) -> NDArray[Any]:
    try:
        return np.asarray(getattr(mesh, name))
    except AttributeError:
        arrays = getattr(mesh, "arrays", None)
        if arrays is None or name not in arrays:
            raise AttributeError(f"mesh has no MPAS field {name!r}") from None
        return np.asarray(arrays[name])


def _rows(array: NDArray[Any], count: int, name: str) -> NDArray[Any]:
    value = np.asarray(array)
    if value.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional")
    if value.shape[0] == count:
        return value
    if value.shape[1] == count:
        return value.T
    raise ValueError(f"{name} shape {value.shape} does not carry {count} rows")


class RegionalAdmissionError(MpasPortError):
    """A limited-area consistency check failed (mpas_atm_bdy_checks)."""


# ---------------------------------------------------------------------------
# masks and mesh-derived regional fields
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegionalMasks:
    """The 7-ring boundary masks and their derived specified-zone masks.

    ``bdy_mask_*`` are the int masks stored by the culling tool (0=interior,
    rings 1..7 outward).  ``spec_zone_mask_*`` are the REAL masks derived at
    model setup: 1.0 where ``bdyMask > nRelaxZone`` (rings 6-7), else 0.0
    (mpas_atm_boundaries.F:692-699, Registry default 0.0).

    ``nearest_relaxation_cell`` is derived per F:702-751; it is 0-based with
    the native "no candidate" value nCells+1 mapped to ``n_cells`` (the
    garbage index).  v8.4.1's ``atm_zero_gradient_w_bdy_work`` no longer
    consumes it (the 2024-08-06 WCS fix hard-zeroes w instead) but the field
    is derived and carried so the transcription surface stays complete.
    """

    bdy_mask_cell: IntArray
    bdy_mask_edge: IntArray
    bdy_mask_vertex: IntArray
    spec_zone_mask_cell: FloatArray
    spec_zone_mask_edge: FloatArray
    spec_zone_mask_vertex: FloatArray
    nearest_relaxation_cell: IntArray
    # Ascending-index element lists (0-based), precomputed so the stage
    # functions below touch only boundary elements.  Iterating these in
    # ascending order reproduces the native ascending whole-domain loops
    # bit for bit because non-member elements contribute nothing.
    spec_cells: IntArray          # bdyMaskCell > nRelaxZone
    spec_edges: IntArray          # bdyMaskEdge > nRelaxZone
    relax_cells: IntArray         # 1 < bdyMaskCell <= nRelaxZone
    relax_edges: IntArray         # 1 < bdyMaskEdge <= nRelaxZone
    nudged_cells: IntArray        # bdyMaskCell > 1  (scalar copy-back set)

    @property
    def n_cells(self) -> int:
        return int(self.bdy_mask_cell.size)

    @property
    def n_edges(self) -> int:
        return int(self.bdy_mask_edge.size)


def derive_regional_masks(mesh: object, dtype: np.dtype[Any]) -> RegionalMasks:
    """Transcribe ``mpas_atm_setup_bdy_masks`` (mpas_atm_boundaries.F:659-753)."""

    bdy_mask_cell = np.asarray(_mesh_array(mesh, "bdyMaskCell"), dtype=np.int64)
    bdy_mask_edge = np.asarray(_mesh_array(mesh, "bdyMaskEdge"), dtype=np.int64)
    bdy_mask_vertex = np.asarray(
        _mesh_array(mesh, "bdyMaskVertex"), dtype=np.int64
    )
    n_cells = bdy_mask_cell.size

    spec_cell = np.where(
        bdy_mask_cell > N_RELAX_ZONE, dtype.type(1.0), dtype.type(0.0)
    ).astype(dtype)
    spec_edge = np.where(
        bdy_mask_edge > N_RELAX_ZONE, dtype.type(1.0), dtype.type(0.0)
    ).astype(dtype)
    spec_vertex = np.where(
        bdy_mask_vertex > N_RELAX_ZONE, dtype.type(1.0), dtype.type(0.0)
    ).astype(dtype)

    counts = np.asarray(_mesh_array(mesh, "nEdgesOnCell"), dtype=np.int64)
    cells_on_cell = _rows(
        _mesh_array(mesh, "cellsOnCell"), n_cells, "cellsOnCell"
    ).astype(np.int64, copy=False)
    x_cell = np.asarray(_mesh_array(mesh, "xCell"), dtype=dtype)
    y_cell = np.asarray(_mesh_array(mesh, "yCell"), dtype=dtype)
    z_cell = np.asarray(_mesh_array(mesh, "zCell"), dtype=dtype)

    # F:702: nearestRelaxationCell(:) = nCells+1, i.e. the garbage index.
    nearest = np.full(n_cells, n_cells, dtype=np.int64)
    # F:708-722: inner specified zone (nRelaxZone+1) searches cellsOnCell for
    # bdyMaskCell == nRelaxZone.  F:728-751: outer specified zone
    # (nRelaxZone+2) searches cellsOnCell of cellsOnCell.  Distances use the
    # native REAL(RKIND) arithmetic on x/y/zCell (v8.4.2
    # mpas_atm_boundaries.F:667-678,713,737). File storage precision does
    # not determine pool precision. Double arithmetic changes nearest-cell
    # selection when squared distances tie after single-precision rounding.
    for cell in range(n_cells):
        mask = bdy_mask_cell[cell]
        if mask == N_RELAX_ZONE + 1:
            dmin = dtype.type(1.0e36)
            for slot in range(int(counts[cell])):
                neighbor = int(cells_on_cell[cell, slot])
                if neighbor < 0 or neighbor >= n_cells:
                    continue  # stored-0 slot on a ring-7 row: no cell there
                if bdy_mask_cell[neighbor] == N_RELAX_ZONE:
                    dx = x_cell[neighbor] - x_cell[cell]
                    dy = y_cell[neighbor] - y_cell[cell]
                    dz = z_cell[neighbor] - z_cell[cell]
                    d = (dx * dx + dy * dy) + dz * dz
                    if d < dmin:
                        dmin = d
                        nearest[cell] = neighbor
        elif mask == N_RELAX_ZONE + 2:
            dmin = dtype.type(1.0e36)
            for slot in range(int(counts[cell])):
                neighbor = int(cells_on_cell[cell, slot])
                if neighbor < 0 or neighbor >= n_cells:
                    continue
                if bdy_mask_cell[neighbor] == N_RELAX_ZONE + 1:
                    for slot2 in range(int(counts[neighbor])):
                        outer = int(cells_on_cell[neighbor, slot2])
                        if outer < 0 or outer >= n_cells:
                            continue
                        if bdy_mask_cell[outer] == N_RELAX_ZONE:
                            dx = x_cell[outer] - x_cell[cell]
                            dy = y_cell[outer] - y_cell[cell]
                            dz = z_cell[outer] - z_cell[cell]
                            d = (dx * dx + dy * dy) + dz * dz
                            if d < dmin:
                                dmin = d
                                nearest[cell] = outer

    return RegionalMasks(
        bdy_mask_cell=bdy_mask_cell,
        bdy_mask_edge=bdy_mask_edge,
        bdy_mask_vertex=bdy_mask_vertex,
        spec_zone_mask_cell=spec_cell,
        spec_zone_mask_edge=spec_edge,
        spec_zone_mask_vertex=spec_vertex,
        nearest_relaxation_cell=nearest,
        spec_cells=np.flatnonzero(bdy_mask_cell > N_RELAX_ZONE).astype(np.int64),
        spec_edges=np.flatnonzero(bdy_mask_edge > N_RELAX_ZONE).astype(np.int64),
        relax_cells=np.flatnonzero(
            (bdy_mask_cell > 1) & (bdy_mask_cell <= N_RELAX_ZONE)
        ).astype(np.int64),
        relax_edges=np.flatnonzero(
            (bdy_mask_edge > 1) & (bdy_mask_edge <= N_RELAX_ZONE)
        ).astype(np.int64),
        nudged_cells=np.flatnonzero(bdy_mask_cell > 1).astype(np.int64),
    )


def regional_bdy_checks(
    masks: RegionalMasks,
    *,
    config_apply_lbcs: bool,
    lbc_input_interval_valid: bool,
) -> None:
    """Transcribe ``mpas_atm_bdy_checks`` (mpas_atm_boundaries.F:775-874)."""

    max_mask = int(np.max(masks.bdy_mask_cell, initial=0))
    if not config_apply_lbcs and max_mask > 0:
        raise RegionalAdmissionError(
            "Boundary cells found in the bdyMaskCell field, but "
            "config_apply_lbcs = false.  Please ensure that "
            "config_apply_lbcs = true for limited-area simulations."
        )
    if config_apply_lbcs and max_mask == 0:
        raise RegionalAdmissionError(
            "config_apply_lbcs = true, but no boundary cells found in the "
            "bdyMaskCell field.  Please ensure that config_apply_lbcs = "
            "false for global simulations."
        )
    if config_apply_lbcs and not lbc_input_interval_valid:
        raise RegionalAdmissionError(
            "Input interval for the 'lbc_in' stream must be a valid "
            "interval when config_apply_lbcs = true."
        )


def compute_mesh_scaling_regional(
    mesh: object,
    dtype: np.dtype[Any],
    *,
    config_h_scale_with_mesh: bool,
) -> Tuple[FloatArray, FloatArray]:
    """Transcribe ``atm_compute_mesh_scaling`` (mpas_atm_core.F:1163-1180).

    ``meshScalingRegionalEdge(iEdge) = 1/((meshDensity(c1)+meshDensity(c2))/2)**0.25``
    and ``meshScalingRegionalCell(iCell) = 1/meshDensity(iCell)**0.25``, both
    only under ``config_h_ScaleWithMesh``; otherwise 1.0 everywhere.  At a
    one-cell (ring-7) boundary edge the native gathers meshDensity at the
    garbage cell, whose pool allocation is 0.0; the resulting value is inert
    (ring-7 edges never carry a relaxation coefficient) but is reproduced so
    the field is native byte for byte.
    """

    density = np.asarray(_mesh_array(mesh, "meshDensity"), dtype=dtype)
    n_cells = density.size
    cells_on_edge = _rows(
        _mesh_array(mesh, "cellsOnEdge"),
        _mesh_array(mesh, "dcEdge").size,
        "cellsOnEdge",
    ).astype(np.int64, copy=False)
    scaling_cell = np.ones(n_cells, dtype=dtype)
    scaling_edge = np.ones(cells_on_edge.shape[0], dtype=dtype)
    if not config_h_scale_with_mesh:
        return scaling_cell, scaling_edge
    padded_density = np.concatenate([density, np.asarray([0.0], dtype=dtype)])
    c0 = np.where(cells_on_edge[:, 0] < 0, n_cells, cells_on_edge[:, 0])
    c1 = np.where(cells_on_edge[:, 1] < 0, n_cells, cells_on_edge[:, 1])
    one = dtype.type(1.0)
    two = dtype.type(2.0)
    quarter = dtype.type(0.25)
    # gfortran evaluates RKIND ``**0.25`` with GNU libm powf, which is not correctly rounded: on the
    # 2024-05-21 storm mesh the correctly rounded hook differs from it on 64 edges and 33 cells, enough to
    # move the relaxation zone after one step. powf_native_gnu is GNU's own algorithm (0 of those differ).
    scaling_edge = (
        one
        / powf_native_gnu(
            (padded_density[c0] + padded_density[c1]) / two, quarter
        )
    ).astype(dtype)
    scaling_cell = (one / powf_native_gnu(density, quarter)).astype(dtype)
    return scaling_cell, scaling_edge


# ---------------------------------------------------------------------------
# the driving state: derived coupled fields over the LBC pool
# ---------------------------------------------------------------------------

#: Field name -> ("cell"|"edge"|"w", derived?) for the pool this runtime
#: maintains.  File fields come from woof.hex.lbc; the derived coupled
#: fields are computed here exactly as mpas_atm_update_bdy_tend does
#: (mpas_atm_boundaries.F:217-262) at every admission.
_DRIVING_FIELDS: Dict[str, str] = {
    "u": "edge",
    "ru": "edge",
    "rho_edge": "edge",
    "w": "w",
    "rho": "cell",
    "rho_zz": "cell",
    "theta": "cell",
    "rtheta_m": "cell",
}


class RegionalDrivingState:
    """The model-side two-level LBC pool with its derived coupled fields.

    Wraps the file-level :class:`~woof.hex.lbc.LbcPool` admission and
    timekeeping and adds what ``mpas_atm_update_bdy_tend`` computes after
    every read: ``lbc_rho_zz = lbc_rho / zz``, edge-averaged
    ``lbc_rho_edge`` (including the zero-valued native garbage cell at
    an outer edge), ``lbc_ru = lbc_u * lbc_rho_edge``, and
    ``lbc_rtheta_m = lbc_theta * lbc_rho_zz * (1 + rvord*qv)``, all in
    float32 with native operation order.

    Tendencies are ``(new - old) * (1/interval)`` with the interval formed
    in float32 and inverted once (F:265-309).  ``state_at`` implements
    ``mpas_atm_get_bdy_state``: ``state(end) - dt_remaining * tend`` with
    ``dt_remaining = float32(end - step_start) - delta_t`` (F:491-551).
    """

    def __init__(
        self,
        inventory: LbcInventory,
        mesh: object,
        zz: FloatArray,
        *,
        scalar_names: Sequence[str] = ("lbc_qv",),
    ) -> None:
        self._pool = LbcPool(inventory, scalar_names=scalar_names)
        self._scalar_names = tuple(scalar_names)
        if not self._scalar_names:
            raise ConfigurationRefusal(
                "scalar_names",
                (),
                "the lbc_scalars var_array always carries at least qv",
                "scalar_names=('lbc_qv', ...)",
            )
        zz_array = np.asarray(zz, dtype=np.float32)
        self._zz = zz_array
        cells_on_edge = _rows(
            _mesh_array(mesh, "cellsOnEdge"),
            _mesh_array(mesh, "dcEdge").size,
            "cellsOnEdge",
        ).astype(np.int64, copy=False)
        # MPAS-A v8.4.2 mpas_block_creator.F:1490-1496 remaps absent
        # file neighbors to the allocated, zero-valued garbage cell.
        cells_on_edge = np.where(cells_on_edge < 0, zz_array.shape[1], cells_on_edge)
        self._cells_on_edge = cells_on_edge
        self._both_present = (cells_on_edge[:, 0] >= 0) & (
            cells_on_edge[:, 1] >= 0
        )
        n_edges = cells_on_edge.shape[0]
        nlev = zz_array.shape[0]
        # Initial pool allocation before the first boundary admission.
        self._rho_edge_slot = np.zeros((nlev, n_edges), dtype=np.float32)
        self._state: Dict[str, np.ndarray] | None = None
        self._state_scalars: np.ndarray | None = None
        self._tend: Dict[str, np.ndarray] | None = None
        self._tend_scalars: np.ndarray | None = None

    # -- admission ---------------------------------------------------------

    def start(self, when: datetime) -> None:
        admitted = self._pool.start(when)
        self._state, self._state_scalars = self._derive(admitted)
        self._tend = None
        self._tend_scalars = None

    def advance(self, when: datetime | None = None) -> None:
        if self._state is None or self._state_scalars is None:
            raise LbcAdmissionError(
                "advance was called on a driving state that never started"
            )
        old_state = self._state
        old_scalars = self._state_scalars
        old_end = self._pool.interval_end
        admitted = self._pool.advance(when)
        new_state, new_scalars = self._derive(admitted)
        # F:265-272: dt from whole days + seconds in REAL(RKIND), then the
        # single reciprocal; every tendency is (new-old)*that reciprocal.
        delta = admitted.valid_time - old_end
        dt = rkind_interval_seconds(delta)
        inv_dt = np.float32(np.float32(1.0) / dt)
        self._tend = {
            name: np.asarray(
                (new_state[name] - old_state[name]) * inv_dt,
                dtype=np.float32,
            )
            for name in _DRIVING_FIELDS
        }
        self._tend_scalars = np.asarray(
            (new_scalars - old_scalars) * inv_dt, dtype=np.float32
        )
        self._state = new_state
        self._state_scalars = new_scalars

    def _derive(
        self, admitted: LbcFile
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        # File slabs are (entity, level); the model-side pool is
        # (level, entity).  The transpose changes no bit of any elementwise
        # operation below.
        u = np.ascontiguousarray(admitted.fields["lbc_u"].T)
        w = np.ascontiguousarray(admitted.fields["lbc_w"].T)
        rho = np.ascontiguousarray(admitted.fields["lbc_rho"].T)
        theta = np.ascontiguousarray(admitted.fields["lbc_theta"].T)
        scalars = np.stack(
            [
                np.ascontiguousarray(admitted.fields[name].T)
                for name in self._scalar_names
            ],
            axis=0,
        )
        qv = scalars[0]
        # F:217-231: the garbage column of zz is forced to 1.0 before the
        # division; this port's arrays carry no garbage column, so the real
        # columns divide by the file zz directly (identical bytes).
        rho_zz = np.asarray(rho / self._zz, dtype=np.float32)
        # MPAS-A v8.4.2 mpas_atm_boundaries.F:234-247 averages both
        # reindexed neighbors, including the zero-valued garbage cell.
        padded_rho = np.pad(rho_zz, ((0, 0), (0, 1)))
        rho_edge = self._rho_edge_slot.copy()
        both = self._both_present
        c0 = self._cells_on_edge[both, 0]
        c1 = self._cells_on_edge[both, 1]
        rho_edge[:, both] = np.float32(0.5) * (padded_rho[:, c0] + padded_rho[:, c1])
        ru = np.asarray(u * rho_edge, dtype=np.float32)
        # F:257-263: rtheta_m = theta * rho_zz * (1 + rvord*qv), left to
        # right in float32.
        rtheta_m = np.asarray(
            theta * rho_zz * (np.float32(1.0) + RVORD_F32 * qv),
            dtype=np.float32,
        )
        state = {
            "u": np.asarray(u, dtype=np.float32),
            "ru": ru,
            "rho_edge": rho_edge,
            "w": np.asarray(w, dtype=np.float32),
            "rho": np.asarray(rho, dtype=np.float32),
            "rho_zz": rho_zz,
            "theta": np.asarray(theta, dtype=np.float32),
            "rtheta_m": rtheta_m,
        }
        return state, np.asarray(scalars, dtype=np.float32)

    # -- consumption -------------------------------------------------------

    @property
    def interval_end(self) -> datetime:
        return self._pool.interval_end

    def _require_ready(self) -> None:
        if self._tend is None or self._tend_scalars is None:
            raise LbcAdmissionError(
                "the regional driving state holds no complete interval; "
                "start then advance must both run before boundary "
                "tendencies or interpolated state exist"
            )

    def tendency(self, name: str) -> np.ndarray:
        """``mpas_atm_get_bdy_tend``, constant over the interval."""

        self._require_ready()
        assert self._tend is not None
        if name == "scalars":
            assert self._tend_scalars is not None
            return self._tend_scalars
        scalar_index = scalar_index_for_field(name, self._scalar_names)
        if name not in _DRIVING_FIELDS and scalar_index is not None:
            assert self._tend_scalars is not None
            return self._tend_scalars[scalar_index]
        if name not in _DRIVING_FIELDS:
            raise ConfigurationRefusal(
                "field",
                name,
                "not a driving field of the regional lbc pool",
                f"one of {sorted(_DRIVING_FIELDS)} or 'scalars'",
            )
        return self._tend[name]

    def state_at(
        self, name: str, step_start: datetime, delta_t: np.float32
    ) -> np.ndarray:
        """``mpas_atm_get_bdy_state``: state(end) - dt_remaining*tend.

        ``dt_remaining = float32(seconds(interval_end - step_start)) -
        delta_t``, with ``delta_t`` the float32 in-step offset the caller
        formed exactly as ``atm_srk3`` forms ``time_dyn_step``.
        """

        self._require_ready()
        assert self._state is not None and self._tend is not None
        assert self._state_scalars is not None
        assert self._tend_scalars is not None
        end = self._pool.interval_end
        delta = end - step_start
        remaining = rkind_interval_seconds(delta)
        dt = np.float32(remaining - np.float32(delta_t))
        if name == "scalars":
            return np.asarray(
                self._state_scalars - dt * self._tend_scalars,
                dtype=np.float32,
            )
        scalar_index = scalar_index_for_field(name, self._scalar_names)
        if name not in _DRIVING_FIELDS and scalar_index is not None:
            return np.asarray(
                self._state_scalars[scalar_index] - dt * self._tend_scalars[scalar_index],
                dtype=np.float32,
            )
        if name not in _DRIVING_FIELDS:
            raise ConfigurationRefusal(
                "field",
                name,
                "not a driving field of the regional lbc pool",
                f"one of {sorted(_DRIVING_FIELDS)} or 'scalars'",
            )
        return np.asarray(
            self._state[name] - dt * self._tend[name], dtype=np.float32
        )


# ---------------------------------------------------------------------------
# time_dyn_step: the float32 in-step offsets of atm_srk3
# ---------------------------------------------------------------------------


def dynamics_time_offset(
    *,
    outer_dt: float,
    dynamics_split: int,
    dynamics_substep: int,
    rk_timestep: float,
) -> np.float32:
    """``time_dyn_step = dt_dynamics*real(substep-1) + rk_timestep(rk_step)``.

    atm_srk3 line 2329/2451, formed in float32: ``dt_dynamics`` is the
    float32 quotient ``dt/real(dynamics_split)``, and ``rk_timestep`` is the
    caller's float32 stage timestep.
    """

    dt_dynamics = np.float32(np.float32(outer_dt) / np.float32(dynamics_split))
    return np.float32(
        dt_dynamics * np.float32(dynamics_substep - 1) + np.float32(rk_timestep)
    )


def rk_timestep_f32(*, outer_dt: float, dynamics_split: int, rk_step: int, order: int = 3) -> np.float32:
    """The float32 ``rk_timestep(rk_step)`` of atm_srk3:2066-2070."""

    if order not in (2, 3):
        raise ValueError("order must be 2 or 3")
    # Order 2 selects the existing half-step branch at both first stages
    # (native v8.4.1 atm_srk3:2080-2092).
    if order == 2 and rk_step == 1:
        rk_step = 2
    dt_dynamics = np.float32(np.float32(outer_dt) / np.float32(dynamics_split))
    if rk_step == 1:
        return np.float32(dt_dynamics / np.float32(3.0))
    if rk_step == 2:
        return np.float32(dt_dynamics / np.float32(2.0))
    if rk_step == 3:
        return np.float32(dt_dynamics)
    raise ValueError("rk_step must be 1, 2, or 3")


def transport_rk_timestep_f32(*, outer_dt: float, rk_step: int, order: int = 3) -> np.float32:
    """The float32 split-transport ``rk_timestep`` of atm_srk3:2676-2678."""

    if order not in (2, 3):
        raise ValueError("order must be 2 or 3")
    if order == 2 and rk_step == 1:
        rk_step = 2
    dt = np.float32(outer_dt)
    if rk_step == 1:
        return np.float32(dt / np.float32(3.0))
    if rk_step == 2:
        return np.float32(dt / np.float32(2.0))
    if rk_step == 3:
        return dt
    raise ValueError("rk_step must be 1, 2, or 3")


# ---------------------------------------------------------------------------
# the boundary-adjust stages (atm_srk3 regional insertions)
# ---------------------------------------------------------------------------


def adjust_dynamics_speczone_tend(
    *,
    masks: RegionalMasks,
    tend_ru: FloatArray,
    tend_rho: FloatArray,
    tend_rt: FloatArray,
    tend_rw: FloatArray,
    ru_driving_tend: FloatArray,
    rt_driving_tend: FloatArray,
    rho_driving_tend: FloatArray,
    rt_diabatic_tend: FloatArray | None = None,
) -> None:
    """``atm_bdy_adjust_dynamics_speczone_tend`` (F:7906-7967), in place.

    Cells with ``bdyMaskCell > nRelaxZone``: ``tend_rho``/``tend_rt`` take
    the driving tendencies, ``tend_rw`` (the omega tendency, rows
    1..nVertLevels) and ``rt_diabatic_tend`` are zeroed.  The dry authority
    may omit the diabatic tendency pool (identically zero with
    config_physics_suite='none'); physics callers supply it to reproduce
    mpas_atm_time_integration.F:7949.  Edges with ``bdyMaskEdge > nRelaxZone``: ``tend_ru`` takes the
    driving tendency.
    """

    cells = masks.spec_cells
    edges = masks.spec_edges
    nlev = tend_rho.shape[0]
    tend_rho[:, cells] = rho_driving_tend[:, cells]
    tend_rt[:, cells] = rt_driving_tend[:, cells]
    # Native zeroes k=1..nVertLevels of tend_rw; the top interface row of
    # the omega tendency is untouched at F:7948.
    tend_rw[:nlev, cells] = tend_rw.dtype.type(0.0)
    if rt_diabatic_tend is not None:
        rt_diabatic_tend[:nlev, cells] = rt_diabatic_tend.dtype.type(0.0)
    tend_ru[:, edges] = ru_driving_tend[:, edges]


def adjust_dynamics_relaxzone_tend(
    mesh: object,
    *,
    masks: RegionalMasks,
    mesh_scaling_regional_cell: FloatArray,
    mesh_scaling_regional_edge: FloatArray,
    config_relax_zone_divdamp_coef: float,
    dt: float,
    tend_ru: FloatArray,
    tend_rho: FloatArray,
    tend_rt: FloatArray,
    ru: FloatArray,
    theta_m: FloatArray,
    rho_zz: FloatArray,
    ru_driving_values: FloatArray,
    rt_driving_values: FloatArray,
    rho_driving_values: FloatArray,
) -> None:
    """``atm_bdy_adjust_dynamics_relaxzone_tend`` (F:7971-8198), in place.

    The hardwired relaxation-zone shape: Rayleigh coefficients
    ``(mask-1)/nRelaxZone/(50*dt*meshScalingRegional)`` and Laplacian filter
    coefficients with the ``10*dt`` clock, both against the FULL outer
    timestep ``dt`` (atm_srk3 passes ``dt``, not ``dt_dynamics``, at line
    2337).  The u-filter combines a divergence part scaled by
    ``config_relax_zone_divdamp_coef`` (Registry default 6.0) and a
    vorticity part with ``r_dv = min(invDvEdge, 4*invDcEdge)``, on the
    deviation ``ru - ru_driving``.
    """

    dtype = tend_rho.dtype
    one = dtype.type(1.0)
    five = dtype.type(N_RELAX_ZONE)
    fifty_dt = dtype.type(50.0) * dtype.type(dt)
    ten_dt = dtype.type(10.0) * dtype.type(dt)
    divdamp = dtype.type(config_relax_zone_divdamp_coef)

    counts = np.asarray(_mesh_array(mesh, "nEdgesOnCell"), dtype=np.int64)
    n_cells = counts.size
    edges_on_cell = _rows(
        _mesh_array(mesh, "edgesOnCell"), n_cells, "edgesOnCell"
    ).astype(np.int64, copy=False)
    cells_on_edge = _rows(
        _mesh_array(mesh, "cellsOnEdge"),
        _mesh_array(mesh, "dcEdge").size,
        "cellsOnEdge",
    ).astype(np.int64, copy=False)
    n_edges = cells_on_edge.shape[0]
    vertices_on_edge = _rows(
        _mesh_array(mesh, "verticesOnEdge"), n_edges, "verticesOnEdge"
    ).astype(np.int64, copy=False)
    edges_on_vertex = _rows(
        _mesh_array(mesh, "edgesOnVertex"),
        _mesh_array(mesh, "areaTriangle").size,
        "edgesOnVertex",
    ).astype(np.int64, copy=False)
    vertex_degree = edges_on_vertex.shape[1]
    dc_edge = np.asarray(_mesh_array(mesh, "dcEdge"), dtype=dtype)
    dv_edge = np.asarray(_mesh_array(mesh, "dvEdge"), dtype=dtype)
    inv_dc = np.asarray(np.reciprocal(dc_edge), dtype=dtype)
    inv_dv = np.asarray(
        np.reciprocal(np.asarray(_mesh_array(mesh, "dvEdge"), dtype=dtype)),
        dtype=dtype,
    )
    inv_area_cell = np.asarray(
        np.reciprocal(np.asarray(_mesh_array(mesh, "areaCell"), dtype=dtype)),
        dtype=dtype,
    )
    inv_area_triangle = np.asarray(
        np.reciprocal(
            np.asarray(_mesh_array(mesh, "areaTriangle"), dtype=dtype)
        ),
        dtype=dtype,
    )
    from .acoustic import edge_signs_on_cells
    from .diagnostics import edge_signs_on_vertices

    cell_signs = edge_signs_on_cells(mesh).astype(dtype, copy=False)
    vertex_signs = edge_signs_on_vertices(mesh).astype(dtype, copy=False)

    bdy_cell = masks.bdy_mask_cell
    bdy_edge = masks.bdy_mask_edge

    # --- first, Rayleigh damping for rho/rtheta at relax cells (F:8054-8065)
    for cell in masks.relax_cells:
        coef = dtype.type(
            (dtype.type(np.float32(bdy_cell[cell])) - one)
            / five
            / (fifty_dt * mesh_scaling_regional_cell[cell])
        )
        tend_rho[:, cell] = tend_rho[:, cell] - coef * (
            rho_zz[:, cell] - rho_driving_values[:, cell]
        )
        tend_rt[:, cell] = tend_rt[:, cell] - coef * (
            rho_zz[:, cell] * theta_m[:, cell] - rt_driving_values[:, cell]
        )

    # --- Rayleigh damping for ru at relax edges (F:8067-8076)
    for edge in masks.relax_edges:
        coef = dtype.type(
            (dtype.type(np.float32(bdy_edge[edge])) - one)
            / five
            / (fifty_dt * mesh_scaling_regional_edge[edge])
        )
        tend_ru[:, edge] = tend_ru[:, edge] - coef * (
            ru[:, edge] - ru_driving_values[:, edge]
        )

    # --- second, the horizontal (dimensionless) Laplacian filter for
    #     rtheta_m and rho at relax cells (F:8080-8109)
    for cell in masks.relax_cells:
        filter_coef = dtype.type(
            (dtype.type(np.float32(bdy_cell[cell])) - one)
            / five
            / (ten_dt * mesh_scaling_regional_cell[cell])
        )
        for slot in range(int(counts[cell])):
            edge = int(edges_on_cell[cell, slot])
            edge_sign = (
                cell_signs[cell, slot]
                * dv_edge[edge]
                * inv_dc[edge]
                * filter_coef
            )
            cell1 = int(cells_on_edge[edge, 0])
            cell2 = int(cells_on_edge[edge, 1])
            tend_rt[:, cell] = tend_rt[:, cell] + edge_sign * (
                (
                    rho_zz[:, cell2] * theta_m[:, cell2]
                    - rt_driving_values[:, cell2]
                )
                - (
                    rho_zz[:, cell1] * theta_m[:, cell1]
                    - rt_driving_values[:, cell1]
                )
            )
            tend_rho[:, cell] = tend_rho[:, cell] + edge_sign * (
                (rho_zz[:, cell2] - rho_driving_values[:, cell2])
                - (rho_zz[:, cell1] - rho_driving_values[:, cell1])
            )

    # --- third, the u filter: divergence and vorticity of (ru - driving)
    #     (F:8111-8194)
    nlev = tend_rho.shape[0]
    for edge in masks.relax_edges:
        # Fortran ``dcEdge(iEdge)**2`` with an integer exponent compiles to
        # the exact product; ``**`` through libm pow would not be it.
        dc_squared = dtype.type(dc_edge[edge] * dc_edge[edge])
        filter_coef = dtype.type(
            dc_squared
            * (dtype.type(np.float32(bdy_edge[edge])) - one)
            / five
            / (ten_dt * mesh_scaling_regional_edge[edge])
        )
        cell1 = int(cells_on_edge[edge, 0])
        cell2 = int(cells_on_edge[edge, 1])
        vertex1 = int(vertices_on_edge[edge, 0])
        vertex2 = int(vertices_on_edge[edge, 1])
        r_dc = inv_dc[edge]
        r_dv = min(inv_dv[edge], dtype.type(4) * inv_dc[edge])

        divergence1 = np.zeros(nlev, dtype=dtype)
        divergence2 = np.zeros(nlev, dtype=dtype)
        vorticity1 = np.zeros(nlev, dtype=dtype)
        vorticity2 = np.zeros(nlev, dtype=dtype)

        inv_area = inv_area_cell[cell1]
        for slot in range(int(counts[cell1])):
            edge_div = int(edges_on_cell[cell1, slot])
            edge_sign = inv_area * dv_edge[edge_div] * cell_signs[cell1, slot]
            divergence1 = divergence1 + edge_sign * (
                ru[:, edge_div] - ru_driving_values[:, edge_div]
            )
        inv_area = inv_area_cell[cell2]
        for slot in range(int(counts[cell2])):
            edge_div = int(edges_on_cell[cell2, slot])
            edge_sign = inv_area * dv_edge[edge_div] * cell_signs[cell2, slot]
            divergence2 = divergence2 + edge_sign * (
                ru[:, edge_div] - ru_driving_values[:, edge_div]
            )
        for slot in range(vertex_degree):
            edge_vort = int(edges_on_vertex[vertex1, slot])
            edge_sign = (
                inv_area_triangle[vertex1]
                * dc_edge[edge_vort]
                * vertex_signs[vertex1, slot]
            )
            vorticity1 = vorticity1 + edge_sign * (
                ru[:, edge_vort] - ru_driving_values[:, edge_vort]
            )
        for slot in range(vertex_degree):
            edge_vort = int(edges_on_vertex[vertex2, slot])
            edge_sign = (
                inv_area_triangle[vertex2]
                * dc_edge[edge_vort]
                * vertex_signs[vertex2, slot]
            )
            vorticity2 = vorticity2 + edge_sign * (
                ru[:, edge_vort] - ru_driving_values[:, edge_vort]
            )

        tend_ru[:, edge] = tend_ru[:, edge] + filter_coef * (
            divdamp * (divergence2 - divergence1) * r_dc
            - (vorticity2 - vorticity1) * r_dv
        )


def overwrite_speczone_u_ru(
    *,
    masks: RegionalMasks,
    normal_velocity: FloatArray,
    rho_u: FloatArray,
    u_driving_values: FloatArray,
    ru_driving_values: FloatArray,
) -> None:
    """The post-recover specified-zone velocity reset (atm_srk3:2442-2485).

    ``u`` and ``ru`` at edges with ``bdyMaskEdge > nRelaxZone`` are replaced
    with the interpolated driving 'u' and 'ru' states; recover computed them
    through the garbage-cell density convention and the native comment says
    plainly it "will not have set outermost edge velocities correctly".
    """

    edges = masks.spec_edges
    normal_velocity[:, edges] = u_driving_values[:, edges]
    rho_u[:, edges] = ru_driving_values[:, edges]


def zero_speczone_w(*, masks: RegionalMasks, w: FloatArray) -> None:
    """``atm_zero_gradient_w_bdy_work`` (F:7868-7902) plus its context.

    Native zeroes rows k=2..nVertLevels of w at cells with
    ``bdyMaskCell > nRelaxZone`` (the 2024-08-06 WCS hard-zero replacing the
    commented-out nearest-relaxation copy).  Rows 1 and nVertLevels+1 are
    already zero at those cells: recover's unmasked first pass wrote
    ``w(1)=0``/``w(top)=0`` and its terrain-following second pass, the only
    writer of a nonzero ``w(1)``, is masked to ``bdyMaskCell <= nRelaxZone``
    (F:4492).  The port recovers w through the unmasked terrain path, so all
    rows including the endpoints are zeroed here to land on the identical
    all-zero specified-zone column.
    """

    w[:, masks.spec_cells] = w.dtype.type(0.0)


def bdy_adjust_scalars(
    mesh: object,
    *,
    masks: RegionalMasks,
    mesh_scaling_regional_cell: FloatArray,
    scalars_new: FloatArray,
    scalars_driving: FloatArray,
    dt: float,
    dt_rk: np.float32,
) -> None:
    """``atm_bdy_adjust_scalars_work`` (F:8305-8416), in place.

    Relaxation-zone cells (1 < mask <= nRelaxZone) receive a dimensionless
    Laplacian filter toward the driving scalars with coefficient
    ``dt_rk*(mask-1)/nRelaxZone/(10*dt*meshScalingRegionalCell)`` and a
    Rayleigh term with one fifth of that coefficient; specified-zone cells
    take the driving values outright.  Both land in temporary storage and
    are copied back for every cell with ``mask > 1``: ring 1 is never
    nudged (the native copy-back condition at F:8402).
    """

    dtype = scalars_new.dtype
    one = dtype.type(1.0)
    five = dtype.type(N_RELAX_ZONE)
    ten_dt = dtype.type(10.0) * dtype.type(dt)

    counts = np.asarray(_mesh_array(mesh, "nEdgesOnCell"), dtype=np.int64)
    n_cells = counts.size
    edges_on_cell = _rows(
        _mesh_array(mesh, "edgesOnCell"), n_cells, "edgesOnCell"
    ).astype(np.int64, copy=False)
    cells_on_edge = _rows(
        _mesh_array(mesh, "cellsOnEdge"),
        _mesh_array(mesh, "dcEdge").size,
        "cellsOnEdge",
    ).astype(np.int64, copy=False)
    dv_edge = np.asarray(_mesh_array(mesh, "dvEdge"), dtype=dtype)
    inv_dc = np.asarray(
        np.reciprocal(np.asarray(_mesh_array(mesh, "dcEdge"), dtype=dtype)),
        dtype=dtype,
    )
    from .acoustic import edge_signs_on_cells

    cell_signs = edge_signs_on_cells(mesh).astype(dtype, copy=False)
    bdy_cell = masks.bdy_mask_cell

    updates: Dict[int, FloatArray] = {}
    for cell in masks.relax_cells:
        filter_coef = dtype.type(
            dtype.type(dt_rk)
            * (dtype.type(np.float32(bdy_cell[cell])) - one)
            / five
            / (ten_dt * mesh_scaling_regional_cell[cell])
        )
        rayleigh_coef = dtype.type(filter_coef / dtype.type(5.0))
        column = scalars_new[:, :, cell].copy()
        for slot in range(int(counts[cell])):
            edge = int(edges_on_cell[cell, slot])
            edge_sign = (
                cell_signs[cell, slot]
                * dv_edge[edge]
                * inv_dc[edge]
                * filter_coef
            )
            cell1 = int(cells_on_edge[edge, 0])
            cell2 = int(cells_on_edge[edge, 1])
            column = column + edge_sign * (
                (scalars_new[:, :, cell2] - scalars_driving[:, :, cell2])
                - (scalars_new[:, :, cell1] - scalars_driving[:, :, cell1])
            )
        column = column - rayleigh_coef * (
            scalars_new[:, :, cell] - scalars_driving[:, :, cell]
        )
        updates[int(cell)] = column
    for cell in masks.spec_cells:
        updates[int(cell)] = scalars_driving[:, :, cell].copy()
    # F:8399-8412: copy back where bdyMaskCell > 1, exactly the union of
    # the relaxation set (2..5) and the specified set (6..7) computed above.
    for cell in masks.nudged_cells:
        scalars_new[:, :, cell] = updates[int(cell)]


def bdy_set_scalars(
    *,
    masks: RegionalMasks,
    scalars_new: FloatArray,
    scalars_driving: FloatArray,
) -> None:
    """``atm_bdy_set_scalars_work`` (F:8462-8505): spec zone := driving."""

    cells = masks.spec_cells
    scalars_new[:, :, cells] = scalars_driving[:, :, cells]


def reset_speczone_values(
    *,
    masks: RegionalMasks,
    theta_m: FloatArray,
    rho_theta: FloatArray,
    rt_driving_values: FloatArray,
    rho_driving_values: FloatArray,
) -> None:
    """``atm_bdy_reset_speczone_values`` (F:8201-8244), in place.

    At the end of the full timestep the specified-zone ``theta_m`` becomes
    ``rt_driving/rho_driving`` and ``rtheta_p`` becomes
    ``rt_driving - rtheta_base``.  The port carries
    ``rho_theta = rtheta_p + rtheta_base``, so the equivalent update is
    ``rho_theta := rt_driving``: bit-identical downstream because every
    native consumer re-adds ``rtheta_base`` to ``rtheta_p`` before use.
    The two assignments are deliberately inconsistent through ``rho_zz``
    (native leaves the integrated specified-zone density in place), which is
    why ``theta_m`` is carried explicitly rather than recomputed.
    """

    cells = masks.spec_cells
    theta_m[:, cells] = rt_driving_values[:, cells] / rho_driving_values[:, cells]
    rho_theta[:, cells] = rt_driving_values[:, cells]


def clamp_negative_scalars(scalars: FloatArray) -> None:
    """The unconditional end-of-step clamp of atm_srk3:2798-2800.

    ``where (scalars_2 < 0) scalars_2 = 0`` executes in every DO_PHYSICS
    build regardless of the physics suite: the reference dry arms ran the
    same physics-capable executable with config_physics_suite='none', so
    the clamp is part of the pinned trajectory.  It runs BEFORE the
    regional specified-zone resets, which can (and measurably do)
    reintroduce small negatives from the driving data.
    """

    # mpas_atm_time_integration.F:2799-2800 is a conditional write, not
    # MAX: negative zero and NaN payloads must remain untouched.
    scalars[scalars < scalars.dtype.type(0.0)] = scalars.dtype.type(0.0)


__all__ = [
    "PaddedRegionalMesh",
    "RegionalRuntime",
    "compute_moist_coefficients",
    "pad_cells_column",
    "regional_normal_velocity",
    "N_BDY_ZONE",
    "N_RELAX_ZONE",
    "N_SPEC_ZONE",
    "RVORD_F32",
    "RegionalAdmissionError",
    "RegionalDrivingState",
    "RegionalMasks",
    "adjust_dynamics_relaxzone_tend",
    "adjust_dynamics_speczone_tend",
    "bdy_adjust_scalars",
    "bdy_set_scalars",
    "clamp_negative_scalars",
    "compute_mesh_scaling_regional",
    "derive_regional_masks",
    "dynamics_time_offset",
    "overwrite_speczone_u_ru",
    "regional_bdy_checks",
    "reset_speczone_values",
    "rk_timestep_f32",
    "transport_rk_timestep_f32",
    "zero_speczone_w",
]


# ---------------------------------------------------------------------------
# padded regional geometry: the native garbage-element memory model
# ---------------------------------------------------------------------------


class PaddedRegionalMesh:
    """A mesh view carrying the native garbage elements explicitly.

    Native MPAS allocates every connectivity and field array with one extra
    "garbage" element per dimension (nCells+1, nEdges+1, nVertices+1) and
    maps absent neighbours to it at read time.  The port's arrays carry no
    garbage element, so regional gathers through ring-7 stored-0 slots need
    this view: connectivity is remapped (sentinel -> garbage index) and
    per-element geometry is padded with the pool-allocation value the native
    garbage element holds (0 for lengths/areas/weights, so dead-lane
    arithmetic reproduces native bytes including signed zeros).
    """

    def __init__(self, mesh: object) -> None:
        n_cells = int(_mesh_array(mesh, "areaCell").size)
        n_edges = int(_mesh_array(mesh, "dcEdge").size)
        n_vertices = int(_mesh_array(mesh, "areaTriangle").size)
        self.n_cells = n_cells
        self.n_edges = n_edges
        self.n_vertices = n_vertices
        self.garbage_cell = n_cells
        self.garbage_edge = n_edges
        self.garbage_vertex = n_vertices

        def remap(name: str, count: int, garbage_row: int, garbage: int):
            rows = _rows(_mesh_array(mesh, name), count, name).astype(
                np.int64, copy=True
            )
            rows[rows < 0] = garbage
            pad = np.full((1, rows.shape[1]), garbage_row, dtype=np.int64)
            return np.concatenate([rows, pad], axis=0)

        def pad_real(name: str, value: float = 0.0):
            data = np.asarray(_mesh_array(mesh, name))
            if data.ndim == 1:
                return np.concatenate(
                    [data, np.asarray([value], dtype=data.dtype)]
                )
            pad = np.full((1, data.shape[1]), value, dtype=data.dtype)
            return np.concatenate([data, pad], axis=0)

        arrays: Dict[str, NDArray[Any]] = {}
        arrays["cellsOnEdge"] = remap(
            "cellsOnEdge", n_edges, self.garbage_cell, self.garbage_cell
        )
        arrays["verticesOnEdge"] = remap(
            "verticesOnEdge", n_edges, self.garbage_vertex, self.garbage_vertex
        )
        arrays["edgesOnVertex"] = remap(
            "edgesOnVertex", n_vertices, self.garbage_edge, self.garbage_edge
        )
        arrays["edgesOnCell"] = remap(
            "edgesOnCell", n_cells, self.garbage_edge, self.garbage_edge
        )
        arrays["edgesOnEdge"] = remap(
            "edgesOnEdge", n_edges, self.garbage_edge, self.garbage_edge
        )
        arrays["verticesOnCell"] = remap(
            "verticesOnCell", n_cells, self.garbage_vertex, self.garbage_vertex
        )
        arrays["cellsOnVertex"] = remap(
            "cellsOnVertex", n_vertices, self.garbage_cell, self.garbage_cell
        )
        arrays["nEdgesOnCell"] = np.concatenate(
            [
                np.asarray(_mesh_array(mesh, "nEdgesOnCell"), dtype=np.int64),
                np.asarray([0], dtype=np.int64),
            ]
        )
        arrays["nEdgesOnEdge"] = np.concatenate(
            [
                np.asarray(_mesh_array(mesh, "nEdgesOnEdge"), dtype=np.int64),
                np.asarray([0], dtype=np.int64),
            ]
        )
        arrays["dcEdge"] = pad_real("dcEdge", 0.0)
        arrays["dvEdge"] = pad_real("dvEdge", 0.0)
        # areaCell/areaTriangle garbage pads are 1.0 so a non-inverse caller
        # cannot trap on a dead division; the v8.4.1 lane always consumes
        # the precomputed inverses, which pad with the native allocation 0.0.
        arrays["areaCell"] = pad_real("areaCell", 1.0)
        arrays["areaTriangle"] = pad_real("areaTriangle", 1.0)
        arrays["weightsOnEdge"] = pad_real("weightsOnEdge", 0.0)
        arrays["kiteAreasOnVertex"] = pad_real("kiteAreasOnVertex", 0.0)
        for optional in ("fVertex", "fEdge", "meshDensity"):
            try:
                arrays[optional] = pad_real(optional, 0.0)
            except AttributeError:
                pass
        self.arrays = arrays


def pad_cells_column(array: FloatArray, value: float = 0.0) -> FloatArray:
    """Append one garbage column along the last axis."""

    data = np.asarray(array)
    pad = np.full(data.shape[:-1] + (1,), value, dtype=data.dtype)
    return np.concatenate([data, pad], axis=-1)


def regional_normal_velocity(
    rho: FloatArray,
    rho_u: FloatArray,
    cells_on_edge_remapped: IntArray,
) -> FloatArray:
    """Recover u on a regional mesh exactly as native recover does.

    ``atm_recover_large_step_variables_work`` F:4385-4392 forces the garbage
    cell of rho_zz to 1.0 before computing ``u = 2*ru/(rho(c1)+rho(c2))`` at
    every edge; a one-cell ring-7 edge therefore divides by
    ``rho(present)+1.0``.  Those edges are overwritten with driving u
    immediately afterwards, but the arithmetic is reproduced so nothing in
    this lane depends on which values are dead.
    """

    padded_rho = pad_cells_column(rho, 1.0)
    denominator = (
        padded_rho[:, cells_on_edge_remapped[:, 0]]
        + padded_rho[:, cells_on_edge_remapped[:, 1]]
    )
    if np.any(denominator == 0.0):
        raise FloatingPointError("zero edge-density denominator")
    return rho.dtype.type(2.0) * rho_u / denominator


def compute_moist_coefficients(
    scalars: FloatArray,
    *,
    moist_indices: Sequence[int],
    cells_on_edge_remapped: IntArray,
    nlev: int,
) -> Tuple[FloatArray, FloatArray, FloatArray]:
    """``atm_compute_moist_coefficients`` (F:3188-3283): qtot, cqw, cqu.

    ``qtot`` sums the moist scalar group per cell; ``cqw(k>=2) =
    1/(1 + 0.5*(qtot(k)+qtot(k-1)))`` and ``cqu = 1/(1 + sum_iq
    0.5*(q(c1)+q(c2)))`` at every edge touching an owned cell (all edges in
    the serial lane); one-cell ring-7 edges read the zeroed garbage column
    exactly as native reads the scalars garbage element.  ``cqw`` row 0 is
    native-untouched (Registry allocation); it is set to one here because no
    consumer reads it and a poisoned row would be indistinguishable from a
    defect.
    """

    dtype = scalars.dtype
    half = dtype.type(0.5)
    one = dtype.type(1.0)
    qtot = np.zeros(scalars.shape[1:], dtype=dtype)
    for index in moist_indices:
        qtot = qtot + scalars[index]
    cqw = np.ones_like(qtot)
    for level in range(1, nlev):
        qtotal = half * (qtot[level] + qtot[level - 1])
        cqw[level] = one / (one + qtotal)
    padded = pad_cells_column(scalars, 0.0)
    n_edges = cells_on_edge_remapped.shape[0]
    qtotal_edge = np.zeros((nlev, n_edges), dtype=dtype)
    for index in moist_indices:
        qtotal_edge = qtotal_edge + half * (
            padded[index][:, cells_on_edge_remapped[:, 0]]
            + padded[index][:, cells_on_edge_remapped[:, 1]]
        )
    cqu = one / (one + qtotal_edge)
    return qtot, cqw, np.asarray(cqu, dtype=dtype)


class RegionalRuntime:
    """Everything the whole-step driver needs to run the regional branch."""

    def __init__(
        self,
        mesh: object,
        *,
        dtype: np.dtype[Any],
        lbc_paths: Sequence[str],
        start_time: datetime,
        config_h_scale_with_mesh: bool,
        zz: FloatArray,
        config_relax_zone_divdamp_coef: float = 6.0,
        scalar_names: Sequence[str] = ("lbc_qv",),
        moist_indices: Sequence[int] = (0,),
    ) -> None:
        self.masks = derive_regional_masks(mesh, dtype)
        scaling_cell, scaling_edge = compute_mesh_scaling_regional(
            mesh, dtype, config_h_scale_with_mesh=config_h_scale_with_mesh
        )
        self.mesh_scaling_cell = scaling_cell
        self.mesh_scaling_edge = scaling_edge
        self.config_relax_zone_divdamp_coef = float(
            config_relax_zone_divdamp_coef
        )
        self.padded_mesh = PaddedRegionalMesh(mesh)
        self.moist_indices = tuple(moist_indices)
        self.driving = RegionalDrivingState(
            LbcInventory(lbc_paths),
            mesh,
            zz,
            scalar_names=scalar_names,
        )
        self.clock = start_time
        self._started = False

    @property
    def cells_on_edge_remapped(self) -> IntArray:
        return self.padded_mesh.arrays["cellsOnEdge"][:-1]

    def ensure_interval(self) -> None:
        """The lbc_in read cadence of atm_core_run (mpas_atm_core.F:735-781).

        The initial LATEST_BEFORE read happens once; the alarm still rings
        at the first timestep, so the first in-loop call shifts and reads
        EARLIEST_STRICTLY_AFTER, and every later ring (the clock reaching an
        interval end) does the same.
        """

        if not self._started:
            self.driving.start(self.clock)
            self.driving.advance(self.clock)
            self._started = True
            return
        if self.clock >= self.driving.interval_end:
            self.driving.advance(self.clock)

    def advance_clock(self, seconds: float) -> None:
        from datetime import timedelta

        self.clock = self.clock + timedelta(seconds=seconds)
