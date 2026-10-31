"""The regional (limited-area) v8.4.1 forecast on the card.

L5 put 22 regional kernels on the device and pinned every one of them,
bitwise, against the v8.4.1 CPU authority.  What it did not do is run a
forecast: the device step had no regional branch and
``cuda_backend.regional_admission.ADMITTED_REGIONS`` was empty, so regional
CUDA execution refused by name.  This module is the missing half -- the
device-side residency, memory model and stage sequencing that let
:class:`woof.hex.cuda_driver.CudaDryDycoreDriver` run
``config_apply_lbcs=true`` on a culled mesh.

The memory model, and why no shared kernel source moves
-------------------------------------------------------
Native MPAS allocates every connectivity and field array with one extra
"garbage" element per dimension (``nCells+1``, ``nEdges+1``,
``nVertices+1``), remaps absent neighbours to it at read time, and never
enters a loop for it.  The CPU authority reproduces that by padding an
array, computing, and slicing the pad off
(:class:`woof.hex.regional_v841.PaddedRegionalMesh`,
:func:`woof.hex.regional_v841.pad_cells_column`, and the pad/strip pairs in
``driver.py``).

A device launch cannot do "compute but not for the last element": the same
integer that bounds the thread index is the array stride in ``lidx``.  So
the device carries the padded element resident, lets it be computed, and
restores the native pool value into every garbage column after each launch.
That is *exactly* pad-compute-strip, expressed as a residency discipline
instead of a copy: every kernel reads inputs whose garbage column holds the
native allocation value, and no garbage output survives to be read.

The consequence is the point: **not one shared CUDA kernel source changes.**
The twelve division-by-garbage-geometry sites and the
``divergence_damping_f32`` sentinel early-out recorded in
``evidence/regional-cuda-l5-20260826/probes/padded-element-blockers.json``
are all retired without touching ``cuda_horizontal.py``,
``cuda_transport.py`` or the CUDA source string in ``cuda_driver.py``.  See
:class:`RegionalGarbageDiscipline` for how each site is retired, one by one.

The one place the padded convention is *not* what a kernel wants is the
acoustic normal-momentum update: ``acoustic_ru_regional_v841`` mirrors the
CPU authority's ``acoustic_v841.py:385-405``, which tests ``cellsOnEdge < 0``
to find a one-cell ring-7 edge, because native multiplies that edge's
garbage-gathered pressure gradient by exactly zero (F:3909).  That kernel is
therefore handed a second, sentinel-preserving connectivity array; every
other consumer takes the remapped one.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping, Sequence

import numpy as np

from .cuda_backend.containers import DeviceAtmosphere, TransferStats
from .cuda_regional_v841 import (
    CudaRegionalKernels,
    DRIVING_FIELD_KIND,
    DeviceRegionalMasks,
    DeviceRegionalDrivingState,
)
from .diagnostics import edge_signs_on_vertices
from .lbc import LbcFile, LbcInventory
from .regional_v841 import (
    N_BDY_ZONE,
    RegionalAdmissionError,
    N_RELAX_ZONE,
    RVORD_F32,
    PaddedRegionalMesh,
    RegionalMasks,
    compute_mesh_scaling_regional,
    derive_regional_masks,
    dynamics_time_offset,
    regional_bdy_checks,
    rk_timestep_f32,
    transport_rk_timestep_f32,
)

__all__ = [
    "assemble_regional_driver_v841",
    "open_regional_forecast_v841",
    "REGIONAL_GARBAGE_POOL",
    "PaddedRegionalHostMesh",
    "RegionalGarbageDiscipline",
    "CudaRegionalRuntimeV841",
    "advance_acoustic_step_regional_cuda_v841",
    "advance_scalars_regional_cuda_v841",
    "build_regional_device_atmosphere",
    "regional_padded_context_v841",
    "regional_padded_advection_coefficients",
    "pad_regional_physics_column",
    "pad_regional_physics_host",
]


# ---------------------------------------------------------------------------
# the native pool values a garbage element holds
# ---------------------------------------------------------------------------

#: What native's garbage element holds, by role.  ``0.0`` is the pool
#: allocation; ``1.0`` appears exactly where the CPU authority pads with 1.0,
#: which is where native itself writes it (``atm_recover_large_step_variables``
#: forces ``rho_zz`` at the garbage cell to 1.0, F:4385-4392) or where a
#: divisor would otherwise trap on a lane native never enters.
REGIONAL_GARBAGE_POOL: Mapping[str, float] = {
    "default": 0.0,
    "rho_zz": 1.0,
    "zz": 1.0,
    "area": 1.0,
    "spec_zone_mask": 1.0,
}


def _padded(array: Any, axis: int, value: float) -> np.ndarray:
    """``np.concatenate([array, np.full(.., value)], axis)`` as a C-ordered array, copied on host threads.

    THE COST THIS REMOVES (mv3km 3 km limited area, V6000, 2026-10-28): padding every regional array with
    the garbage element was a strided copy plus a concatenate on the main thread, about 1.8 s of the
    start-up (the dycore build and the first boundary admissions) while the card waited.  The copy is written in blocks of
    the leading non-padded axis on host threads (woof.hex.host_parallel); the words are the same.
    """

    from .host_parallel import run_blocks

    data = np.asarray(array)
    if data.ndim == 0:
        raise ValueError("zero-dimensional arrays cannot be concatenated")
    axis = axis % data.ndim
    shape = list(data.shape)
    shape[axis] += 1
    out = np.empty(tuple(shape), dtype=data.dtype)
    tail = [slice(None)] * data.ndim
    tail[axis] = slice(data.shape[axis], None)
    np.copyto(out[tuple(tail)], value, casting="unsafe")
    head = [slice(None)] * data.ndim
    head[axis] = slice(0, data.shape[axis])
    # Blocks along the longest axis (the padded one included: blocks then cover its first n elements). A level
    # axis (55) or scalar axis (59) is shorter than one block (host_parallel.MIN_BLOCK_COLUMNS), so blocking
    # there copied every (level, cell) and (scalar, level, cell) array on one thread (mv3km, 2026-10-28).
    block_axis = int(np.argmax(data.shape))

    def copy(block: slice) -> None:
        index = list(head)
        index[block_axis] = block
        out[tuple(index)] = data[tuple(index)]

    run_blocks(copy, data.shape[block_axis], elements=data.size)
    return out


def _pad_last(array: Any, value: float) -> np.ndarray:
    """Append one garbage element along the LAST axis."""

    return _padded(array, -1, value)


def _pad_axis(array: Any, axis: int, value: float) -> np.ndarray:
    """Append one garbage element along ``axis``."""

    return _padded(array, axis, value)


# ---------------------------------------------------------------------------
# the padded host mesh handed to DeviceMesh.from_host
# ---------------------------------------------------------------------------


def regional_garbage_donor(attrs: Mapping[str, Any], count: int) -> int:
    """Keep a declared physical donor when entity numbering changes."""
    donor = attrs.get("regional_garbage_donor_cell", count - 1)
    if isinstance(donor, (bool, np.bool_)) or not isinstance(donor, (int, np.integer)) or not 0 <= donor < count:
        raise ValueError("regional garbage donor must identify a real cell; prevents lending an absent physics column")
    return int(donor)


class PaddedRegionalHostMesh:
    """A host mesh view carrying native's garbage element explicitly.

    :class:`woof.hex.regional_v841.PaddedRegionalMesh` already builds the
    connectivity remap and the geometry pads the CPU authority uses; this
    view adds everything else ``DeviceMesh.from_host`` reads, declares the
    padded dimensions so every downstream shape assertion is satisfied
    without changing one of them, and carries the regional
    ``spec_zone_mask_edge`` the shared ``divergence_damping_f32`` already
    consumes.

    The ``spec_zone_mask_*`` pads are ``1.0``: the garbage element then
    behaves as a specified-zone element everywhere the masks are read, which
    is what makes its computed values the native pool zeros rather than
    junk that the discipline has to erase.
    """

    def __init__(self, mesh: object, masks: RegionalMasks) -> None:
        padded = PaddedRegionalMesh(mesh)
        self.base = mesh
        self.padded = padded
        self.n_cells_solve = padded.n_cells
        self.n_edges_solve = padded.n_edges
        self.n_vertices_solve = padded.n_vertices
        self.garbage_cell = padded.garbage_cell
        self.garbage_edge = padded.garbage_edge
        self.garbage_vertex = padded.garbage_vertex
        layout = getattr(mesh, "partition_layout_v841", None)
        self.partition_source_mesh = getattr(mesh, "partition_source_mesh", mesh)
        if layout is None:
            self.regional_garbage_donor_cell = regional_garbage_donor(
                getattr(mesh, "attrs", {}), padded.garbage_cell)
        else:
            donor_global = regional_garbage_donor(
                getattr(self.partition_source_mesh, "attrs", {}), int(layout._global_cells))
            donor_local = int(layout.cell_g2l()[donor_global])
            if donor_local < 0:
                raise ValueError(
                    "partition lacks the canonical regional garbage donor; "
                    "build layouts with replicated_cells containing the declared global donor"
                )
            self.regional_garbage_donor_cell = donor_local
            self.partition_layout_v841 = layout
            # The driver validator checks padded row extents.  The garbage
            # rows are allocation, not topology; the real layout remains
            # unchanged for ownership and halo exchange.
            from types import SimpleNamespace
            self.partition_layout = SimpleNamespace(
                partition=layout.partition,
                cell_row_complete=np.concatenate((layout.cell_row_complete, [False])),
                edge_row_complete=np.concatenate((layout.edge_row_complete, [False])),
                vertex_row_complete=np.concatenate((layout.vertex_row_complete, [False])),
            )

        base_dimensions = dict(getattr(mesh, "dimensions", {}) or {})
        arrays: dict[str, np.ndarray] = dict(padded.arrays)

        # The sentinel-preserving connectivity: acoustic_ru_regional_v841 and
        # the CPU authority it is pinned to both detect a one-cell ring-7 edge
        # by a negative entry (acoustic_v841.py:385-405, native F:3909).  The
        # garbage ROW itself is remapped, not sentinel, so the garbage edge
        # takes the interior branch and lands on zeros instead of tripping the
        # kernel's "one-cell edge outside the specified zone" refusal flag.
        # PaddedRegionalMesh remaps the seven arrays the CPU dycore gathers
        # through; ``cellsOnCell`` is not one of them (mixing_v841 builds its
        # own local remap at mixing_v841.py:539), but DeviceMesh uploads it,
        # so it is remapped here by the same rule: sentinel -> garbage cell,
        # and a garbage row that points at the garbage cell.
        raw_coc = np.asarray(_mesh_array(mesh, "cellsOnCell"), dtype=np.int64)
        raw_coc = np.where(raw_coc < 0, padded.garbage_cell, raw_coc)
        arrays["cellsOnCell"] = np.concatenate(
            [
                raw_coc,
                np.full((1, raw_coc.shape[1]), padded.garbage_cell, np.int64),
            ],
            axis=0,
        )

        raw_coe = np.asarray(_mesh_array(mesh, "cellsOnEdge"), dtype=np.int64)
        sentinel_row = np.full((1, 2), padded.garbage_cell, dtype=np.int64)
        arrays["cellsOnEdgeSentinel"] = np.concatenate(
            [raw_coe, sentinel_row], axis=0
        )

        for name in (
            "latCell",
            "lonCell",
            "meshDensity",
        ):
            arrays[name] = _pad_last(_mesh_array(mesh, name), 0.0)
        for name in ("latEdge", "lonEdge", "angleEdge", "fEdge"):
            arrays[name] = _pad_last(_mesh_array(mesh, name), 0.0)
        arrays["fVertex"] = _pad_last(_mesh_array(mesh, "fVertex"), 0.0)
        # defc_a/defc_b are the v8.2.3 deformation coefficients.  The v8.4.1
        # mixing branch consumes the deformation WEIGHTS instead, so the
        # v8.4.1 lane attaches inactive zero placeholders when a mesh file
        # does not carry them (tools/run_cuda_v841_full_physics_x4.py:1059).
        # A regional cull is such a file; the placeholders are made here, at
        # the padded extent, and never claim an active mixing lane.
        counts = np.asarray(_mesh_array(mesh, "nEdgesOnCell"))
        placeholder = (int(counts.size) + 1, int(
            np.asarray(_mesh_array(mesh, "edgesOnCell")).shape[1]
        ))
        self.inactive_deformation_placeholders = []
        for name in ("defc_a", "defc_b"):
            existing = _mesh_value(mesh, name)
            if existing is None:
                arrays[name] = np.zeros(placeholder, dtype=np.float32, order="C")
                self.inactive_deformation_placeholders.append(name)
            else:
                arrays[name] = _pad_axis(np.asarray(existing), 0, 0.0)
        arrays["spec_zone_mask_edge"] = np.concatenate(
            [
                np.asarray(masks.spec_zone_mask_edge, dtype=np.float32),
                np.asarray([REGIONAL_GARBAGE_POOL["spec_zone_mask"]], np.float32),
            ]
        )
        arrays["spec_zone_mask_cell"] = np.concatenate(
            [
                np.asarray(masks.spec_zone_mask_cell, dtype=np.float32),
                np.asarray([REGIONAL_GARBAGE_POOL["spec_zone_mask"]], np.float32),
            ]
        )
        # ------------------------------------------------------------------
        # what the ArWen physics seam reads, and the dycore does not
        # ------------------------------------------------------------------
        # THE BREAKAGE THIS PREVENTS, MEASURED on r4.75.11020 (2026-08-26):
        # the three physics geometry constructors are cell-count agnostic, so
        # the SAME constructors the global stack calls can be called on this
        # padded view -- but only if the view honours the two conventions the
        # dycore never had to.  Without the block below,
        # ``CudaMpasToPhysGeometryV841.from_host`` refuses with "edgesOnCell
        # padding must be canonical -1" and then "coeffs_reconstruct" is
        # absent, and a limited-area run gets a dry dycore instead of ArWen.
        #
        # 1. ``edgesOnCell`` inactive slots.  ``PaddedRegionalMesh.remap``
        #    sends EVERY negative entry to the garbage edge, which is right
        #    for the sentinel slots the dycore gathers through and wrong for
        #    the slots past ``nEdgesOnCell``, where the loader's canonical
        #    padding is -1 and the physics prep requires exactly that.  Only
        #    the inactive slots are restored: every ACTIVE entry is left
        #    bit-identical, and no kernel on either side reads an inactive
        #    slot (both loop ``slot < nEdgesOnCell``; cuda_driver's own host
        #    validation checks ``edges_on_cell[cell, :count]`` only,
        #    cuda_driver.py:1848-1852).  The global lane already ships -1 in
        #    those slots, so this makes the two views agree rather than
        #    inventing a third convention.
        counts64 = np.asarray(arrays["nEdgesOnCell"], dtype=np.int64)
        eoc = np.array(arrays["edgesOnCell"], dtype=np.int64, copy=True)
        slot_index = np.arange(eoc.shape[1], dtype=np.int64)[None, :]
        inactive = slot_index >= counts64[:, None]
        eoc[inactive] = -1
        arrays["edgesOnCell"] = eoc
        # 2. The reconstruction carriers.  ``DeviceMesh`` uploads neither
        #    ``coeffs_reconstruct`` nor ``edgeNormalVectors`` (they are not
        #    fields of it), so carrying them here costs the dycore nothing
        #    and no device byte.  ``coeffs_reconstruct`` padding must be
        #    bitwise +0 on inactive slots for the same prep contract; the
        #    garbage cell's whole row is +0, which is what a cell with
        #    ``nEdgesOnCell == 0`` reconstructs from.
        # Both carriers are OPTIONAL here: the dry regional lane never reads
        # them, and a mesh built for it (or a test fixture) carries neither.
        # The physics geometry constructors refuse a mesh without them, by
        # their own names, which is the right place for that refusal.
        raw_coeffs_value = _mesh_value(mesh, "coeffs_reconstruct")
        if raw_coeffs_value is not None:
            raw_coeffs = np.asarray(raw_coeffs_value)
            coeffs = np.zeros(
                (raw_coeffs.shape[0] + 1,) + raw_coeffs.shape[1:],
                dtype=raw_coeffs.dtype,
            )
            coeffs[: raw_coeffs.shape[0]] = raw_coeffs
            real_inactive = inactive[: raw_coeffs.shape[0]]
            coeffs[: raw_coeffs.shape[0]][real_inactive] = raw_coeffs.dtype.type(0.0)
            arrays["coeffs_reconstruct"] = coeffs
        normals = _mesh_value(mesh, "edgeNormalVectors")
        if normals is not None:
            arrays["edgeNormalVectors"] = _pad_axis(np.asarray(normals), 0, 0.0)
        # 3. The boundary rings themselves, so a consumer handed this view can
        #    still tell it is regional and hash the triple.  The garbage
        #    element is ring 0 (interior): it is not a boundary element, it is
        #    not an element at all.
        for name, source in (
            ("bdyMaskCell", masks.bdy_mask_cell),
            ("bdyMaskEdge", masks.bdy_mask_edge),
            ("bdyMaskVertex", masks.bdy_mask_vertex),
        ):
            arrays[name] = np.concatenate(
                [np.asarray(source, dtype=np.int32), np.zeros(1, np.int32)]
            )
        for name, value in list(arrays.items()):
            arrays[name] = np.ascontiguousarray(value)

        self.arrays = arrays
        self.dimensions = {
            **base_dimensions,
            "nCells": padded.n_cells + 1,
            "nEdges": padded.n_edges + 1,
            "nVertices": padded.n_vertices + 1,
        }
        self.attrs = {"on_a_sphere": True}
        self.nominalMinDc = float(np.asarray(_mesh_value(mesh, "nominalMinDc")))
        self.registry_row = _mesh_value(mesh, "registry_row")
        self.is_regional = True

    @property
    def n_cells(self) -> int:
        return self.n_cells_solve + 1

    @property
    def n_edges(self) -> int:
        return self.n_edges_solve + 1

    @property
    def n_vertices(self) -> int:
        return self.n_vertices_solve + 1

    def __getattr__(self, name: str) -> Any:
        arrays = self.__dict__.get("arrays")
        if arrays is not None and name in arrays:
            return arrays[name]
        raise AttributeError(name)


def _mesh_value(mesh: object, name: str) -> Any:
    value = getattr(mesh, name, None)
    if value is not None:
        return value
    arrays = getattr(mesh, "arrays", None)
    if isinstance(arrays, dict) and name in arrays:
        return arrays[name]
    attrs = getattr(mesh, "attrs", None)
    if isinstance(attrs, dict) and name in attrs:
        return attrs[name]
    return None


def _mesh_array(mesh: object, name: str) -> np.ndarray:
    value = _mesh_value(mesh, name)
    if value is None:
        raise AttributeError(f"mesh is missing {name}")
    return np.asarray(value)


# ---------------------------------------------------------------------------
# padded host shims for the remaining device containers
# ---------------------------------------------------------------------------


class _PaddedState:
    __slots__ = ("rho", "rho_theta", "rho_u", "rho_w", "scalars", "time_seconds")

    def __init__(self, state: object) -> None:
        self.rho = _pad_last(getattr(state, "rho"), REGIONAL_GARBAGE_POOL["rho_zz"])
        self.rho_theta = _pad_last(getattr(state, "rho_theta"), 0.0)
        self.rho_u = _pad_last(getattr(state, "rho_u"), 0.0)
        self.rho_w = _pad_last(getattr(state, "rho_w"), 0.0)
        self.scalars = _pad_last(getattr(state, "scalars"), 0.0)
        self.time_seconds = float(getattr(state, "time_seconds", 0.0))


class _PaddedSaved:
    __slots__ = (
        "theta_m",
        "exner",
        "density_perturbation",
        "rho_theta_perturbation",
        "pressure_perturbation",
        "normal_velocity",
        "vertical_velocity",
    )

    def __init__(self, saved: object) -> None:
        for name in self.__slots__:
            setattr(self, name, _pad_last(getattr(saved, name), 0.0))


class _PaddedReference:
    """The dry reference state with a non-trapping garbage column.

    MEASURED on the reference cull: no kernel in the v8.4.1 step reads
    ``rho_base``/``rho_theta_base``/``exner_base``/``pressure_base`` at a
    GATHERED index -- every read is at the thread's own element (a grep of
    ``cuda_*.py`` for those names returns only ``[i]``/``[index]``/``[cell]``
    forms).  Their garbage column is therefore invisible to every real
    element, and its value is a device-only choice about what the garbage
    thread computes rather than a claim about native's pool.

    The choice is 1.0 for the three that appear in denominators.  At 0.0 the
    garbage thread forms ``0/0`` in ``acoustic_coefficients_v841``'s
    ``cofwt`` and in ``recover_cells_f32``'s ``rtheta/rho``, and the
    coefficient kernel's own singular-denominator check -- which fires
    DURING the launch, where the garbage discipline cannot reach -- refuses
    the step.  At 1.0 the garbage cell recovers ``rho_zz = 1.0``, which is
    exactly the value native writes there
    (``atm_recover_large_step_variables_work`` F:4385-4392) and the value the
    discipline holds for ``rho_zz`` anyway.
    """

    __slots__ = ("rho_base", "rho_theta_base", "pressure_base", "exner_base")

    def __init__(self, reference: object) -> None:
        for name in ("rho_base", "rho_theta_base", "exner_base"):
            setattr(self, name, _pad_last(getattr(reference, name), 1.0))
        self.pressure_base = _pad_last(getattr(reference, "pressure_base"), 0.0)


class _PaddedVertical:
    """The vertical grid on the padded cell/edge axes.

    ``zz`` pads with 1.0, the value the CPU authority uses when it pads for
    ``recover_velocities`` (``driver.py:2391-2394``) and the value that keeps
    the garbage cell's own acoustic normalization a division of zero by one
    instead of zero by zero.  No real element ever gathers ``zz`` at the
    garbage index: the only sentinel path into a cell array is
    ``cellsOnEdge`` on a ring-7 edge, and the acoustic kernel takes its
    one-cell branch there without gathering.
    """

    _COLUMN = ("zw", "dzw", "rdzw", "zu", "dzu", "rdzu", "rdzwp", "rdzwm",
               "fzp", "fzm", "ah")
    _CELL = ("hx", "zgrid", "dss")
    _EDGE = ("zxu",)

    def __init__(self, vertical: object) -> None:
        for name in self._COLUMN:
            setattr(self, name, np.ascontiguousarray(getattr(vertical, name)))
        for name in self._CELL:
            setattr(self, name, _pad_last(getattr(vertical, name), 0.0))
        for name in self._EDGE:
            setattr(self, name, _pad_last(getattr(vertical, name), 0.0))
        self.zz = _pad_last(getattr(vertical, "zz"), REGIONAL_GARBAGE_POOL["zz"])
        self.cf1 = float(getattr(vertical, "cf1"))
        self.cf2 = float(getattr(vertical, "cf2"))
        self.cf3 = float(getattr(vertical, "cf3"))
        self.first_height_level = int(getattr(vertical, "first_height_level"))


class _PaddedTerrain:
    __slots__ = ("zb_cell", "zb3_cell")

    def __init__(self, terrain: object) -> None:
        self.zb_cell = _pad_axis(getattr(terrain, "zb_cell"), 1, 0.0)
        self.zb3_cell = _pad_axis(getattr(terrain, "zb3_cell"), 1, 0.0)


class _PaddedAdvectionCoefficients:
    """The order-three stencil with a garbage edge row.

    The stencil's cell indices already reach the garbage cell -- the CPU
    authority builds them that way with ``allow_regional_sentinels=True``
    (``transport.py:314-350``, ``garbage = n_cells``) -- so only the edge
    axis needs the extra row, with zero stencil cells on it.
    """

    __slots__ = (
        "adv_coefs",
        "adv_coefs_3rd",
        "n_adv_cells_for_edge",
        "adv_cells_for_edge",
        "horizontal_order",
    )

    def __init__(self, coefficients: object, garbage_cell: int) -> None:
        self.adv_coefs = _pad_axis(getattr(coefficients, "adv_coefs"), 0, 0.0)
        self.adv_coefs_3rd = _pad_axis(
            getattr(coefficients, "adv_coefs_3rd"), 0, 0.0
        )
        counts = np.asarray(getattr(coefficients, "n_adv_cells_for_edge"))
        self.n_adv_cells_for_edge = np.ascontiguousarray(
            np.concatenate([counts, np.zeros(1, dtype=counts.dtype)])
        )
        cells = np.asarray(getattr(coefficients, "adv_cells_for_edge"))
        pad = np.full((1, cells.shape[1]), garbage_cell, dtype=cells.dtype)
        self.adv_cells_for_edge = np.ascontiguousarray(
            np.concatenate([cells, pad], axis=0)
        )
        self.horizontal_order = int(getattr(coefficients, "horizontal_order"))


def regional_padded_advection_coefficients(
    coefficients: object, garbage_cell: int
) -> _PaddedAdvectionCoefficients:
    return _PaddedAdvectionCoefficients(coefficients, garbage_cell)


# ---------------------------------------------------------------------------
# the ArWen physics statics on the padded extent
# ---------------------------------------------------------------------------


def pad_regional_physics_column(array: Any, *, donor_cell: int | None = None) -> np.ndarray:
    """Append one garbage column that DUPLICATES the last real one.

    Every other pad in this module carries native's pool value, because the
    dycore's garbage lane is arithmetic native also performs and the pool
    value is what native's garbage element holds.  The physics pad is the one
    place where that rule gives the wrong answer, and the reason is worth
    stating because it is the difference between a limited-area forecast and
    a refusal.

    Native never runs physics on its garbage element: ``mpas_atmphys_*`` loops
    ``1..nCellsSolve``.  A device launch cannot skip its last element, so the
    garbage column runs the column physics whether or not anyone wants its
    answer -- and a column of pool zeros is not a column.  Measured on
    ``r4.75.11020``: ``dx_column_m`` at 0.0 is refused outright by
    ``SealedArwenConstructorV841.from_mapping`` ("dx_column_m must be finite
    and positive", cuda_arwen_physics_v841.py:409); ``meshDensity`` at 0.0 is
    refused by ``native_cell_dx_m``; a 0 K skin temperature and a 0 K soil
    column put the surface layer, the land-surface model and WSM6 outside
    every table they interpolate, and the NaNs that come back are not local:
    the ring-7 one-cell edges gather the garbage cell through ``cellsOnEdge``
    in ``project_edges_v841_f32``, and the step health gate reduces over the
    whole array.

    Duplicating the last real column gives the garbage lane a well-posed
    throwaway answer instead.  Nothing real consumes it: the only path from
    the garbage column into a real element is that ring-7 edge projection,
    and every ring-7 edge is a SPECIFIED-zone edge whose momentum is
    overwritten with the driving value in the same step
    (``overwrite_speczone_u``, F:2442-2485).
    """

    data = np.ascontiguousarray(np.asarray(array))
    if data.ndim < 1 or data.shape[-1] < 1:
        raise ValueError("a physics column carrier needs at least one column")
    donor = regional_garbage_donor(
        {} if donor_cell is None else {"regional_garbage_donor_cell": donor_cell},
        int(data.shape[-1]))
    return np.ascontiguousarray(
        np.concatenate([data, data[..., donor:donor + 1]], axis=-1)
    )


def pad_regional_physics_host(
    constructor_values: Mapping[str, Any],
    gwdo_host: Mapping[str, Any],
    *,
    n_cells_solve: int,
    donor_cell: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Put the WOOF constructor and the GWDO statics on the padded extent.

    The global stack and the limited-area stack call the SAME three physics
    geometry constructors and the SAME sealed constructor; the only thing
    that differs is that a limited-area device atmosphere is one column
    wider, because that is native's own allocation.  This moves the host
    carriers onto that extent so no constructor needs a regional branch.

    Returns ``(constructor_values, gwdo_host, receipt)``.
    """

    solve = int(n_cells_solve)
    padded_values = dict(constructor_values)
    declared = int(padded_values.get("n_columns", solve))
    if declared != solve:
        raise ValueError(
            f"the regional physics pad expects the sealed constructor to "
            f"carry the solve column count {solve}; it carries {declared}"
        )
    padded_names: list[str] = []
    for name, value in list(padded_values.items()):
        if not isinstance(value, np.ndarray):
            continue
        if value.ndim == 0 or int(value.shape[-1]) != solve:
            # z_interface_nominal_m is a column PROFILE, not a per-column
            # carrier; its last axis is nVertLevels+1 and it is shared.
            continue
        padded_values[name] = pad_regional_physics_column(value, donor_cell=donor_cell)
        padded_names.append(name)
    padded_values["n_columns"] = solve + 1
    padded_gwdo = dict(gwdo_host)
    gwdo_names: list[str] = []
    for name, value in list(padded_gwdo.items()):
        array = np.asarray(value)
        if array.ndim != 1 or int(array.shape[-1]) != solve:
            continue
        padded_gwdo[name] = pad_regional_physics_column(array, donor_cell=donor_cell)
        gwdo_names.append(name)
    receipt = {
        "convention": "duplicate-last-real-column",
        "n_columns_solve": solve,
        "n_columns_padded": solve + 1,
        "constructor_arrays_padded": sorted(padded_names),
        "gwdo_arrays_padded": sorted(gwdo_names),
        "reason": (
            "native runs column physics over 1..nCellsSolve and a device "
            "launch cannot skip its last element; the garbage column is "
            "given a duplicate of a real column so it computes a well-posed "
            "answer nothing consumes, instead of the pool zeros the dycore "
            "lane holds, which every physics table refuses"
        ),
    }
    if donor_cell is not None:
        receipt["convention"] = "duplicate-declared-physical-column"
        receipt["donor_cell"] = int(donor_cell)
    return padded_values, padded_gwdo, receipt


def build_regional_device_atmosphere(
    padded_mesh: PaddedRegionalHostMesh,
    state: object,
    vertical: object,
    reference: object,
    saved: object,
    terrain: object,
) -> DeviceAtmosphere:
    """Upload one regional atmosphere in native's padded memory model."""

    return DeviceAtmosphere.from_host(
        padded_mesh,
        _PaddedState(state),
        _PaddedVertical(vertical),
        _PaddedReference(reference),
        _PaddedSaved(saved),
        _PaddedTerrain(terrain),
        dtype=np.float32,
        index_dtype=np.int32,
    )


def regional_padded_context_v841(
    mesh: object,
    offcentering: object,
    reference_wind: object,
    *,
    n_vert_levels: int,
) -> Any:
    """The v8.4.1 resident inverses with native's garbage allocation.

    ``PaddedRegionalMesh`` pads ``areaCell``/``areaTriangle`` with 1.0 so a
    non-inverse caller cannot trap on a dead division, and records that the
    v8.4.1 lane consumes the precomputed inverses, **which pad with the
    native allocation 0.0** (``regional_v841.py:1100-1104``).  Building the
    inverses from the padded geometry would put 1.0 there instead, so they
    are built on the unpadded mesh and padded explicitly.
    """

    import cupy as cp

    from .cuda_v841 import CudaV841Context
    from .dynamics_v841 import precomputed_mesh_inverse_v841

    selected = np.dtype(np.float32)
    host = {
        "inv_area_cell": _pad_last(
            precomputed_mesh_inverse_v841(mesh, "areaCell", selected), 0.0
        ),
        "inv_area_triangle": _pad_last(
            precomputed_mesh_inverse_v841(mesh, "areaTriangle", selected), 0.0
        ),
        "inv_dc_edge": _pad_last(
            precomputed_mesh_inverse_v841(mesh, "dcEdge", selected), 0.0
        ),
        "inv_dv_edge": _pad_last(
            precomputed_mesh_inverse_v841(mesh, "dvEdge", selected), 0.0
        ),
        "etp": np.ascontiguousarray(np.asarray(offcentering.etp, selected)),
        "etm": np.ascontiguousarray(np.asarray(offcentering.etm, selected)),
        "ewp": np.ascontiguousarray(np.asarray(offcentering.ewp, selected)),
        "ewm": np.ascontiguousarray(np.asarray(offcentering.ewm, selected)),
        "u_init": np.ascontiguousarray(
            np.asarray(reference_wind.u_init, selected)
        ),
        "v_init": np.ascontiguousarray(
            np.asarray(reference_wind.v_init, selected)
        ),
    }
    for name, value in host.items():
        if not np.all(np.isfinite(value)):
            raise ValueError(f"regional v8.4.1 context {name} must be finite")
    started = time.perf_counter()
    device = {name: cp.asarray(value) for name, value in host.items()}
    cp.cuda.get_current_stream().synchronize()
    elapsed = time.perf_counter() - started
    transferred = int(sum(int(value.nbytes) for value in device.values()))
    del n_vert_levels
    return CudaV841Context(**device, h2d=TransferStats(transferred, elapsed))


# ---------------------------------------------------------------------------
# the garbage-element discipline
# ---------------------------------------------------------------------------

#: Translation units whose kernels own their own garbage column, and which
#: :class:`RegionalGarbageDiscipline` therefore must not rewrite after.
#:
#: This is the ArWen physics seam, and it is a closed set of four modules:
#: the MPAS-to-physics preparation, the tendency coupling, the YSU gravity
#: wave drag statics, and the two-phase backend.  Every one of them runs
#: COLUMN physics, which native runs over 1..nCellsSolve and a device launch
#: cannot; the seam handles that itself by lending the garbage column a real
#: column for the duration of a preparation and giving it back
#: (``cuda_physics_prep_v841.prepare_mpas_to_phys_cuda_v841``).  Anything
#: NOT in this set is dycore, and the dycore wants the pool value restored.
#:
#: The fifth member is the solve-region check unit
#: (``woof.hex.cuda_solve_region_v841``): its kernels READ the state arrays
#: they are handed (the health envelope, the density and recovered-state
#: validations) or ARE the restoration (the fused garbage scrub), and they
#: write nothing but a flag, an envelope or the pool value itself.  A launch
#: that dirties no garbage column has nothing to restore, and rewriting a
#: state array's garbage column after the health gate reads it would write
#: the default pool over ``rho_zz``'s unit pool between the commit and the
#: next ``begin_step`` -- the same breakage ``release_unit_pool`` records.
SELF_MANAGED_GARBAGE_MODULES: frozenset[str] = frozenset(
    {
        "hexcore.cuda_physics_prep_v841",
        "hexcore.cuda_physics_v841",
        "hexcore.cuda_gwdo_v841",
        "hexcore.cuda_arwen_physics_v841",
        "hexcore.cuda_solve_region_v841",
    }
)


class RegionalGarbageDiscipline:
    """Restore native's pool value into every garbage column after a launch.

    This is pad-compute-strip, held resident.  The CPU authority pads an
    array, runs the shared kernel over the padded extent, and slices the pad
    off, so the computed garbage value never reaches a consumer.  A device
    launch cannot skip its last element -- ``lidx(k,i,n)`` makes the loop
    bound and the array stride the same integer -- so instead every garbage
    column is rewritten with the native allocation value immediately after
    the launch that may have dirtied it.  A consumer therefore only ever
    gathers the pool value, which is what native gathers.

    **What this retires, site by site**, from
    ``probes/padded-element-blockers.json``:

    * ``theta_finish_f32`` / ``w_finish_f32`` (``cuda_driver.py:1069,1158``)
      divide by ``areaCell`` at the launch index.  ``PaddedRegionalMesh``
      pads ``areaCell`` with 1.0 for exactly this reason, so the garbage
      cell's tendency is ``0/1``, not ``0/0``.
    * ``vertex_diagnostics_f32`` / ``cell_diagnostics_f32``
      (``cuda_horizontal.py:216,217,249``) are the v8.2.3 kernels; the
      v8.4.1 lane runs ``*_v841_f32``, which consume the precomputed
      ``inv_area_*`` and never divide in-kernel.
    * ``fct_horizontal_low_order`` / ``fct_finish``
      (``cuda_transport.py:357,463``) are the monotonic limiter, which the
      pinned regional record does not run (``config_monotonic=false``).
    * ``momentum_filter_lap2_f32`` / ``momentum_filter_lap4_f32`` /
      ``w_filter_lap2_f32`` (``cuda_horizontal.py:574,575,670,671,806``)
      DO run, and DO form ``1/dcEdge`` at the garbage edge, whose pad is the
      native allocation 0.0.  Their garbage-element output is therefore
      non-finite -- and is erased here before any gather or any finiteness
      validation can see it.  Keeping the pad at 0.0 rather than making the
      divisor safe is deliberate: 0.0 is what the CPU authority gathers
      through ``edgesOnVertex`` on a ring-7 row, and a nonzero pad would
      move a signed zero in the vorticity sum.
    * ``divergence_damping_f32`` needs no sentinel early-out at all once the
      connectivity is remapped: its test is
      ``cell0 >= n_cells_solve && cell1 >= n_cells_solve`` with
      ``n_cells_solve`` the SOLVE count, so a ring-7 one-cell edge (one real
      cell, one garbage cell) fails the AND, takes the normal path, and
      indexes ``rtheta_pp`` at a valid garbage column whose contribution the
      ``(1 - specZoneMaskEdge)`` factor multiplies by zero.
    * ``DeviceMesh.from_host``'s ``edge_sign_on_cell`` derivation is correct
      unchanged: ``nEdgesOnCell`` at the garbage cell is 0, so the loop body
      never runs for the garbage row and it keeps the pre-filled 0.0; on a
      real ring-7 row ``cellsOnEdge[edge,0] == cell`` is false for the
      remapped garbage entry exactly as it was false for ``-1``.

    The discipline also MEASURES itself: :attr:`dirty_values` counts how many
    garbage entries were not already at their pool value, which is the
    evidence that says whether the erasure is doing work or merely
    confirming an invariant.
    """

    def __init__(
        self,
        *,
        n_cells_solve: int,
        n_edges_solve: int,
        n_vertices_solve: int,
        measure: bool = False,
    ) -> None:
        import cupy as cp

        self.cp = cp
        self.n_cells_solve = int(n_cells_solve)
        self.n_edges_solve = int(n_edges_solve)
        self.n_vertices_solve = int(n_vertices_solve)
        self._extents = {
            self.n_cells_solve + 1: self.n_cells_solve,
            self.n_edges_solve + 1: self.n_edges_solve,
            self.n_vertices_solve + 1: self.n_vertices_solve,
        }
        if len(self._extents) != 3:
            raise ValueError(
                "a regional cull whose padded cell/edge/vertex extents "
                "collide cannot be disciplined by extent; this mesh has "
                f"nCells={n_cells_solve} nEdges={n_edges_solve} "
                f"nVertices={n_vertices_solve}"
            )
        self._immutable: set[int] = set()
        # Strong references, deliberately: an unregistered pointer defaults to
        # the 0.0 pool, so a freed rho whose address CuPy recycles for an
        # unrelated buffer must not still be listed as a 1.0 pool.  Holding
        # the array keeps the address reserved while the entry lives; the
        # dictionary is cleared at every step boundary.
        self._unit_pool: dict[int, Any] = {}
        self.measure = bool(measure)
        self.dirty_values = 0
        self.scrubs = 0
        self.enabled = True
        self.last_kernel = ""
        self.audit: Any | None = None
        self._permanent_unit_pool: dict[int, Any] = {}
        #: The fused restoration (``cuda_solve_region_v841``), attached by
        #: the runtime that arms this discipline on a KernelCache.  Absent,
        #: every column is restored by its own ``cupy`` fill.
        self.fused: Any | None = None
        self.fused_launches = 0

    # -- registration ------------------------------------------------------

    def hold_immutable(self, *arrays: Any) -> None:
        """Never rewrite these: they are read-only residency, already padded."""

        for array in arrays:
            if array is None:
                continue
            if getattr(array, "ndim", 0) == 0:
                continue
            self._immutable.add(int(array.data.ptr))

    def bind_unit_pool(self, array: Any, *, permanent: bool = False) -> Any:
        """Register ``rho_zz``: its garbage cell is 1.0, not 0.0.

        ``permanent`` survives :meth:`release_unit_pool`.  The dynamics binds
        and releases the rho_zz images it forms WITHIN a step; the resident
        atmosphere's own rho_zz is not one of those -- native's 1.0 at its
        garbage cell is true for the whole run, not for the interval between
        two dynamics calls.

        Native writes it (``atm_recover_large_step_variables_work``
        F:4385-4392) and the CPU authority pads with 1.0 at every gather
        (``driver.py:2246``, ``2391``, ``transport.py:2015-2019``).  The
        value is also a fixed point of the continuity update at the garbage
        cell -- its flux divergence is zero -- so re-binding it after each
        rebinding of the state is a restatement, not a correction.
        """

        if array is None:
            return array
        if permanent:
            self._permanent_unit_pool[int(array.data.ptr)] = array
        self._unit_pool[int(array.data.ptr)] = array
        array[..., self.n_cells_solve] = np.float32(
            REGIONAL_GARBAGE_POOL["rho_zz"]
        )
        return array

    def bound_to_unit_pool(self, array: Any) -> bool:
        """Whether a scrub of ``array`` would write the 1.0 pool right now.

        The one public reading of the registry.  A launch's scrub depends
        on the registrations live AT THAT LAUNCH, and they are released at
        every step boundary, so anything that replays a recorded launch
        after the run (``tools/kernel_element_ab.py``) has to ask this
        while the launch is live and re-bind before it replays; asked
        after the run it answers 0.0 for every step-bound image, which is
        what the replay of 2026-09-14 wrote into four kernels' input
        garbage columns and reported as a difference.
        """

        return int(array.data.ptr) in self._unit_pool

    def release_unit_pool(self) -> None:
        """Drop every per-step ``rho_zz`` registration at a step boundary.

        THE BREAKAGE THIS PREVENTS, MEASURED on r4.75.11020 (2026-08-26):
        clearing the resident atmosphere's registration too meant that
        between one step's release and the next step's dynamics, a scrub
        wrote the DEFAULT pool 0.0 into rho_zz's garbage cell.  With a dry
        dycore nothing looked -- but the physics phase runs in exactly that
        window (``recover_edge_fields`` is launched, and scrubbed, before
        the coupling), and ``couple_cells_v841_f32`` requires rho_zz
        strictly positive at every column.  A limited-area full-physics
        forecast refused its first step on "state.rho (must be > 0): 55
        value(s) ... column 11020" for this and nothing else.
        """

        self._unit_pool.clear()
        self._unit_pool.update(self._permanent_unit_pool)

    # -- the discipline ----------------------------------------------------

    def scrub(
        self, name: str, args: Sequence[Any], *, module_key: str = ""
    ) -> None:
        """Restore pool values in every padded argument of one launch.

        ``module_key`` names the translation unit that launched, and it is
        load-bearing.  One KernelCache serves the whole run -- the dycore AND
        the WOOF physics seam -- so before 2026-08-26 this discipline
        rewrote the garbage column of every float32 argument of every physics
        launch too, and that is the wrong answer for exactly one reason: the
        physics seam MANAGES ITS OWN garbage column.

        THE BREAKAGE IT CAUSED, MEASURED on r4.75.11020: with the physics
        stack attached, prep_mass_v841_f32 wrote a well-posed garbage column
        and this discipline zeroed it between that launch and the next, so
        prepare_mpas_to_phys_cuda_v841 saw rho_dry = th_p = pres_p = 0 there
        and refused the step -- and it also zeroed the SOURCE state arrays it
        was handed, because they are arguments too.  A limited-area
        full-physics forecast could not take one step.

        The dycore's kernels want this discipline: they compute a garbage
        value and native's pool value is what a real element must gather
        instead.  The physics seam does not: its garbage column is a
        deliberate duplicate of a real column, lent for the duration of the
        preparation and given back (cuda_physics_prep_v841), and a pool value
        written into the middle of that is not a restoration, it is a hole.
        """

        if not self.enabled:
            return
        if module_key in SELF_MANAGED_GARBAGE_MODULES:
            return
        self.last_kernel = name
        cp = self.cp
        # One launch restores every padded argument of this launch when the
        # fused kernel is attached and nothing wants to watch the columns.
        # The measuring and audited forms keep the per-column fills, because
        # they READ each column before rewriting it, and the value written
        # is the same float32 either way.  Measured before the fusion:
        # 3,328 ``cupy_fill`` launches a step on the 43,884-cell point mesh,
        # ~40 us of host time each, 51 ms of a 0.38 s step.
        fused = None if (self.measure or self.audit is not None) else self.fused
        batch: list[tuple[int, int, int, int, float]] = []
        for value in args:
            if not isinstance(value, cp.ndarray):
                continue
            if value.dtype != cp.float32 or value.ndim == 0:
                continue
            solve = self._extents.get(int(value.shape[-1]))
            if solve is None:
                continue
            pointer = int(value.data.ptr)
            if pointer in self._immutable:
                continue
            pool = np.float32(
                REGIONAL_GARBAGE_POOL["rho_zz"]
                if pointer in self._unit_pool
                else REGIONAL_GARBAGE_POOL["default"]
            )
            if fused is not None and value.flags.c_contiguous:
                stride = int(value.shape[-1])
                batch.append((pointer, int(value.size // stride), stride, solve, float(pool)))
                self.scrubs += 1
                continue
            column = value[..., solve]
            if self.measure:
                same = cp.equal(column, pool)
                self.dirty_values += int(column.size) - int(same.sum())
            column[...] = pool
            self.scrubs += 1
        if batch:
            fused.scrub_columns(batch)
            self.fused_launches += 1
        if self.audit is not None:
            self.audit(name, args)

    def receipt(self) -> dict[str, Any]:
        return {
            "n_cells_solve": self.n_cells_solve,
            "n_edges_solve": self.n_edges_solve,
            "n_vertices_solve": self.n_vertices_solve,
            "garbage_columns_restored": self.scrubs,
            "fused_restoration_launches": self.fused_launches,
            "measured": self.measure,
            "garbage_values_off_pool": self.dirty_values if self.measure else None,
        }


# ---------------------------------------------------------------------------
# the lateral-boundary pool, on the padded stride
# ---------------------------------------------------------------------------


class _PaddedMaskExtents:
    """The extent pair ``DeviceRegionalDrivingState`` sizes its pool from."""

    __slots__ = ("n_cells", "n_edges")

    def __init__(self, n_cells: int, n_edges: int) -> None:
        self.n_cells = int(n_cells)
        self.n_edges = int(n_edges)


def _ring_lbc_on() -> bool:
    """WOOF_HEX_REGIONAL_RING_LBC=0 restores the full-extent LBC admission derivation (A/B only)."""

    import os

    return os.environ.get("WOOF_HEX_REGIONAL_RING_LBC", "1").strip().lower() not in ("0", "off", "false", "no")


class PaddedRegionalDrivingState(DeviceRegionalDrivingState):
    """L5's device lateral-boundary pool on native's padded stride.

    The pool arrays have to carry the garbage column because every consumer
    kernel indexes them with the SAME ``ncells``/``nedges`` it uses for the
    tendency arrays it writes -- ``lidx`` makes the stride and the bound one
    integer.  Only the upload changes: each file slab gains a garbage column
    at the native allocation 0.0, and the derivation then runs unmodified,
    landing 0.0 in every derived field at the garbage element because
    ``lbc_rho`` is 0 there and ``zz`` is 1.
    """

    def __init__(self, *args: Any, garbage_cell: int, garbage_edge: int, **kwargs: Any) -> None:
        self._garbage_cell = int(garbage_cell)
        self._garbage_edge = int(garbage_edge)
        # Device index sets of the boundary support (compact_to_boundary); None keeps the full-extent pool.
        self._support: dict[str, Any] | None = None
        self._derive_support: dict[str, Any] | None = None
        super().__init__(*args, **kwargs)
        # _derive makes the zeros it needs; holding a full-extent edge array of zeros for the run bought nothing.
        self._rho_edge_zero = None

    # -- the boundary-support pool ------------------------------------------
    #
    # THE BREAKAGE THIS PREVENTS (vram census 2026-10-28, RESULT.md: limited-area boundary arrays at full-mesh
    # extent, 14.5 KB per column): the two-level driving pool (state and interval tendency, eight coupled fields
    # and every boundary scalar) was held over every cell and edge of the rank, although only the seven boundary
    # rings use it (about 2 % of NOAA conus3km's columns), and on 4 x 96 GB cards rank 1 ran out of memory in its
    # first step.  The pool now holds the boundary rings plus three neighbour rings (the relaxation filters and the
    # scalar Laplacian read the first interior neighbours) and the garbage element; consumers get full-extent
    # arrays scattered from it, zeros elsewhere, the same words at every element a boundary kernel reads.

    def compact_to_boundary(self, bdy_mask_cell: Any, *, neighbour_rings: int = 3) -> None:
        import cupy as cp

        mask = cp.asarray(bdy_mask_cell).ravel()
        if int(mask.size) != self.ncells:
            raise ValueError(f"boundary mask has {int(mask.size)} cells, the driving pool {self.ncells}")
        coe = self._cells_on_edge.reshape(-1, 2)
        if int(coe.shape[0]) != self.nedges:
            raise ValueError(f"cellsOnEdge has {int(coe.shape[0])} edges, the driving pool {self.nedges}")
        valid = (coe >= 0) & (coe < self.ncells)
        safe = cp.where(valid, coe, 0)
        cell_in = mask > 0
        cell_in[self._garbage_cell] = True
        for _ in range(int(neighbour_rings)):
            edge_in = (cell_in[safe] & valid).any(axis=1)
            grown = cell_in.copy()
            touched = safe[edge_in][valid[edge_in]]
            grown[touched] = True
            cell_in = grown
        edge_in = (cell_in[safe] & valid).any(axis=1)
        edge_in[self._garbage_edge] = True
        cells = cp.flatnonzero(cell_in)
        edges = cp.flatnonzero(edge_in)
        self._support = {"cell": cells, "interface": cells, "edge": edges}
        self.support_counts = (int(cells.size), int(edges.size))
        # The admission derivation runs on the support too (see _derive_on_support): its cells are the support cells
        # plus both cells of every support edge (rho_edge averages the two), in rank order; cellsOnEdge of the support
        # edges is renumbered to that set, an absent or out-of-rank neighbour to one past its end (the kernel's 0.0).
        coe_se = coe[edges]
        valid_se = (coe_se >= 0) & (coe_se < self.ncells)
        derive_in = cell_in.copy()
        derive_in[coe_se[valid_se]] = True
        derive_cells = cp.flatnonzero(derive_in)
        n_derive = int(derive_cells.size)
        local = cp.full(self.ncells, n_derive, dtype=cp.int32)
        local[derive_cells] = cp.arange(n_derive, dtype=cp.int32)
        coe_local = cp.where(valid_se, local[cp.where(valid_se, coe_se, 0)], cp.int32(n_derive))
        zz = self._zz.reshape(-1)[: self.nlev * self.ncells].reshape(self.nlev, self.ncells)
        self._derive_support = {
            "n": n_derive,
            "cells_host": cp.asnumpy(derive_cells),
            "edges_host": cp.asnumpy(edges),
            "cells_on_edge": cp.ascontiguousarray(coe_local.astype(cp.int32).ravel()),
            "zz": cp.ascontiguousarray(zz[:, derive_cells]),
            "support_in_derive": cp.ascontiguousarray(local[cells]),
        }

    def _support_index(self, name: str) -> Any:
        assert self._support is not None
        return self._support[DRIVING_FIELD_KIND.get(name, "cell")]

    def _expand(self, name: str, compact: Any, out: Any = None) -> Any:
        import cupy as cp

        if self._support is None:
            if out is None:
                return compact
            out[...] = compact
            return out
        extent = self.nedges if DRIVING_FIELD_KIND.get(name) == "edge" else self.ncells
        shape = tuple(compact.shape[:-1]) + (extent,)
        if out is None or tuple(out.shape) != shape:
            out = cp.zeros(shape, dtype=compact.dtype)
        out[..., self._support_index(name)] = compact
        return out

    def _missing_scalar(self, index: int) -> Any:
        if self._support is None or self._tend_scalars is None:
            return super()._missing_scalar(index)
        return self._expand("scalars", self._tend_scalars[index])

    def _derive(self, admitted: LbcFile) -> tuple[dict[str, Any], Any]:
        import cupy as cp

        if self._support is not None and getattr(self, "_derive_support", None) is not None and _ring_lbc_on():
            return self._derive_on_support(admitted)
        state, scalars = self._derive_full(admitted)
        if self._support is None:
            return state, scalars
        compact = {name: cp.ascontiguousarray(value[..., self._support_index(name)])
                   for name, value in state.items()}
        compact_scalars = cp.ascontiguousarray(scalars[..., self._support["cell"]])
        del state, scalars
        return compact, compact_scalars

    def tendency(self, name: str) -> Any:
        return self._expand(name, super().tendency(name))

    def state_at(self, name: str, step_start: Any, delta_t: Any, out: Any = None) -> Any:
        return self._expand(name, super().state_at(name, step_start, delta_t), out=out)

    def state_at_support(self, name: str, step_start: Any, delta_t: Any) -> tuple[Any, Any]:
        """The driving values on the boundary support only, and the index set they sit on.

        ``(values, None)`` when the pool is full extent.  Same words as :meth:`state_at` at every
        support element; nothing is scattered to the rank's full extent.
        """
        values = super().state_at(name, step_start, delta_t)
        if self._support is None:
            return values, None
        return values, self._support_index(name)

    # THE BREAKAGE THIS PREVENTS (rrfs review 2026-10-29: the VRAM census's largest "sized for all cells" item, the
    # limited-area LBC arrays at 14.5 KB per column, cuda_regional_v841 advance / _derive / upload): the held pool is
    # compact since compact_to_boundary, but every admission (the first step, then each LBC interval) still uploaded
    # every file slab at the rank's full extent, stacked the scalar block a second time, and derived rho_zz, rtheta_m,
    # rho_edge and ru over every cell and edge before keeping ~2-3 % of them: about 10 KB per local column at once,
    # ~5.5 GiB per card for NOAA conus3km on 4 ranks, on top of a card already at its peak.  The admission now takes
    # only the derivation support's rows from the host slab (the garbage element is the padded upload's 0.0), runs
    # the same two kernels on support-local numbering, and keeps the support.  Same inputs per element, same kernels,
    # same order of operations: the same words at every support element.  WOOF_HEX_REGIONAL_RING_LBC=0 restores the
    # full-extent derivation for A/B.

    def _missing_scalar_on_derive(self, index: int) -> Any:
        """``_missing_scalar`` on the derivation support: the held tendency on the boundary support, 0.0 elsewhere."""

        import cupy as cp

        ds = self._derive_support
        assert ds is not None
        out = cp.zeros((self.nlev, ds["n"]), dtype=cp.float32)
        if self._tend_scalars is not None:
            out[:, ds["support_in_derive"]] = self._tend_scalars[index]
        return out

    def _derive_on_support(self, admitted: LbcFile) -> tuple[dict[str, Any], Any]:
        import cupy as cp

        ds = self._derive_support
        assert ds is not None
        n_derive = int(ds["n"])
        n_edges = int(ds["edges_host"].size)

        def upload(name: str, rows: np.ndarray, extent: int) -> Any:
            host = np.asarray(admitted.fields[name])
            count = int(host.shape[0])
            if count + 1 != extent:
                raise RuntimeError(
                    f"LBC field {name} has {count} rows; the padded pool expects {extent - 1} plus the garbage element"
                )
            inside = rows < count
            picked = np.zeros((int(rows.size), int(host.shape[1])), dtype=np.float32)
            picked[inside] = np.asarray(host[rows[inside]], dtype=np.float32)
            return cp.ascontiguousarray(cp.asarray(picked).T)

        cells, edges = ds["cells_host"], ds["edges_host"]
        u = upload("lbc_u", edges, self.nedges)
        w = upload("lbc_w", cells, self.ncells)
        rho = upload("lbc_rho", cells, self.ncells)
        theta = upload("lbc_theta", cells, self.ncells)
        scalars = cp.ascontiguousarray(
            cp.stack([upload(name, cells, self.ncells) if name in admitted.fields
                      else self._missing_scalar_on_derive(j)
                      for j, name in enumerate(self._scalar_names)], axis=0)
        )
        qv = cp.ascontiguousarray(scalars[0])
        rho_zz = cp.empty((self.nlev, n_derive), dtype=cp.float32)
        rtheta_m = cp.empty((self.nlev, n_derive), dtype=cp.float32)
        self._kernels.launch(
            "regional_lbc_derive_v841",
            self.nlev * n_derive,
            (
                np.int32(self.nlev * n_derive), np.float32(RVORD_F32),
                ds["zz"], rho, theta, qv, rho_zz, rtheta_m,
            ),
        )
        rho_edge = cp.zeros((self.nlev, n_edges), dtype=cp.float32)
        ru = cp.empty((self.nlev, n_edges), dtype=cp.float32)
        self._kernels.launch(
            "regional_lbc_rho_edge_v841",
            n_edges,
            (
                np.int32(self.nlev), np.int32(n_derive),
                np.int32(n_edges), ds["cells_on_edge"], rho_zz, u,
                rho_edge, ru,
            ),
        )
        keep = ds["support_in_derive"]

        def on_support(values: Any) -> Any:
            return cp.ascontiguousarray(values[..., keep])

        state = {
            "u": u, "ru": ru, "rho_edge": rho_edge, "w": on_support(w), "rho": on_support(rho),
            "rho_zz": on_support(rho_zz), "theta": on_support(theta), "rtheta_m": on_support(rtheta_m),
        }
        return state, on_support(scalars)

    def _derive_full(self, admitted: LbcFile) -> tuple[dict[str, Any], Any]:
        import cupy as cp

        def upload(name: str) -> Any:
            # The file slab (element, level) goes to the card as stored; the (level, element) transpose and
            # the garbage column at 0.0 are written there.  THE COST THIS REMOVES (mv3km, V6000,
            # 2026-10-28): the strided host transpose plus padding of 17 slabs took about 1.3 s of one core
            # per boundary admission (2.6 s of the first step) while the card waited.  Same words.
            slab = cp.asarray(np.ascontiguousarray(admitted.fields[name], dtype=np.float32))
            count = slab.shape[0]
            padded = cp.empty((slab.shape[1], count + 1), dtype=cp.float32)
            padded[:, count] = 0.0
            padded[:, :count] = slab.T
            return padded

        u = upload("lbc_u")
        w = upload("lbc_w")
        rho = upload("lbc_rho")
        theta = upload("lbc_theta")
        scalars = cp.ascontiguousarray(
            cp.stack([upload(name) if name in admitted.fields else self._missing_scalar(j)
                      for j,name in enumerate(self._scalar_names)], axis=0)
        )
        qv = cp.ascontiguousarray(scalars[0])
        rho_zz = cp.empty(self._cell_shape, dtype=cp.float32)
        rtheta_m = cp.empty(self._cell_shape, dtype=cp.float32)
        self._kernels.launch(
            "regional_lbc_derive_v841",
            self.nlev * self.ncells,
            (
                np.int32(self.nlev * self.ncells), np.float32(RVORD_F32),
                self._zz, rho, theta, qv, rho_zz, rtheta_m,
            ),
        )
        rho_edge = cp.zeros(self._edge_shape, dtype=cp.float32)
        ru = cp.empty(self._edge_shape, dtype=cp.float32)
        self._kernels.launch(
            "regional_lbc_rho_edge_v841",
            self.nedges,
            (
                np.int32(self.nlev), np.int32(self.ncells),
                np.int32(self.nedges), self._cells_on_edge, rho_zz, u,
                rho_edge, ru,
            ),
        )
        return (
            {
                "u": u, "ru": ru, "rho_edge": rho_edge, "w": w, "rho": rho,
                "rho_zz": rho_zz, "theta": theta, "rtheta_m": rtheta_m,
            },
            scalars,
        )


# ---------------------------------------------------------------------------
# the regional acoustic substep and scalar transport
# ---------------------------------------------------------------------------


def advance_acoustic_step_regional_cuda_v841(
    runtime: "CudaRegionalRuntimeV841",
    mesh: Any,
    state: Any,
    forcing: Any,
    coefficients: Any,
    *,
    context: Any,
    dts: float,
    small_step: int,
    fzm: Any,
    fzp: Any,
    rdzw: Any,
    gravity: float = 9.80616,
    rgas: float = 287.0,
    cp_dry: float = 1004.5,
) -> Any:
    """One regional acoustic substep: L5's three kernels plus the shared prepare.

    Kernel for kernel this is ``advance_acoustic_step_cuda_v841`` with the
    three specified-zone-aware entrypoints substituted -- the pressure
    gradient masking (F:3909), the rs/ts skip and the implicit-solve skip
    (F:4093-4103).  ``acoustic_prepare_v841`` is the shared kernel unchanged:
    its work is the sub-step bookkeeping native does for every cell.
    """

    import cupy as cp

    kernels = runtime.kernels
    out = state
    nlev, nedges = map(int, out.ru_p.shape)
    ncells = int(out.rho_pp.shape[1])
    max_edges = int(mesh.max_edges)
    kernels.launch(
        "acoustic_ru_regional_v841",
        nlev * nedges,
        (
            np.int32(nlev), np.int32(nedges), np.int32(ncells),
            np.int32(small_step), np.float32(dts), np.float32(gravity),
            np.float32(rgas), np.float32(cp_dry),
            runtime.cells_on_edge_sentinel, context.inv_dc_edge,
            mesh.spec_zone_mask_edge, forcing.zz, forcing.exner, forcing.cqu,
            forcing.zxu, forcing.tend_ru, out.rho_pp, out.rtheta_pp,
            out.ru_p, out.ru_avg, runtime.acoustic_invalid,
        ),
    )
    from .cuda_acoustic_fusion_v841 import fusion_enabled, advance_cell
    if fusion_enabled():
        advance_cell(runtime, mesh, out, forcing, coefficients, context,
                     nlev=nlev, ncells=ncells, nedges=nedges, max_edges=max_edges,
                     dts=dts, small_step=small_step, fzm=fzm, fzp=fzp, rdzw=rdzw)
        return out
    runtime.shared_launch(
        "acoustic_prepare_v841",
        ncells,
        (
            np.int32(nlev), np.int32(ncells), np.int32(small_step),
            out.rw_p, out.rtheta_pp, out.rtheta_pp_old, out.rho_pp,
            out.ww_avg,
        ),
    )
    rs = cp.zeros((nlev, ncells), dtype=cp.float32)
    ts = cp.zeros_like(rs)
    kernels.launch(
        "acoustic_rs_ts_regional_v841",
        nlev * ncells,
        (
            np.int32(nlev), np.int32(ncells), np.int32(nedges),
            np.int32(max_edges), np.float32(dts), mesh.n_edges_on_cell,
            mesh.edges_on_cell, mesh.cells_on_edge, mesh.edge_sign_on_cell,
            mesh.dv_edge, context.inv_area_cell, runtime.spec_zone_mask_cell,
            forcing.theta_m, rdzw, coefficients.cofrz, coefficients.coftz,
            context.ewm, out.ru_p, out.rw_p, out.rho_pp, out.rtheta_pp,
            forcing.tend_rho, forcing.tend_rt, rs, ts,
        ),
    )
    kernels.launch(
        "acoustic_column_solve_regional_v841",
        ncells,
        (
            np.int32(nlev), np.int32(ncells), np.float32(dts), forcing.zz,
            forcing.rho_zz, fzm, fzp, rdzw, forcing.dss, forcing.w,
            forcing.rw, forcing.rw_save, forcing.tend_rw, forcing.tend_rho,
            forcing.tend_rt, rs, ts, runtime.spec_zone_mask_cell,
            coefficients.cofwr, coefficients.cofwz, coefficients.coftz,
            coefficients.cofwt, coefficients.cofrz, coefficients.a_tri,
            coefficients.alpha_tri, coefficients.gamma_tri, context.etp,
            context.etm, context.ewp, context.ewm, out.rw_p, out.rho_pp,
            out.rtheta_pp, out.ww_avg,
        ),
    )
    return out


def advance_scalars_regional_cuda_v841(
    runtime: "CudaRegionalRuntimeV841",
    mesh: Any,
    context: Any,
    scalar_old: Any,
    scalar_stage: Any,
    rho_zz_old: Any,
    rho_zz_new: Any,
    uh_avg: Any,
    ww_avg: Any,
    dt: float,
    *,
    coefficients: Any,
    fzm: Any,
    fzp: Any,
    rdzw: Any,
    rk_step: int,
    config_time_integration_order: int = 3,
    config_coef_3rd_order: float,
    validation_flag: Any,
    scalar_tendency: Any | None = None,
) -> Any:
    """The regional split scalar transport; mpas_atm_time_integration.F:4764-4945.

    Two of the four kernels are L5's regional variants: the edge-value
    three-way zone split (full stencil, mask-4/5 first-order upwind, ring-6/7
    skipped) and the cell finish with its specified-zone skip.  The vertical
    flux and the target-density interpolation are the shared kernels,
    unchanged -- including the inherited one-ulp
    ``mpas_div(..., 12.0f)`` divergence in ``transport_vertical_flux`` that
    L5 measured and reported (it is a property of the released global lane,
    not of this branch).
    """

    import cupy as cp

    ntracers, nlev, ncells = map(int, scalar_stage.shape)
    nedges = int(uh_avg.shape[1])
    max_edges = int(mesh.max_edges)
    width = int(coefficients.adv_coefs.shape[1])
    chunk = _regional_scalar_chunk(ntracers, nlev, nedges, scalar_stage, scalar_old)
    weight = np.float32(
        1.0 / 3.0 if rk_step == 1 and config_time_integration_order == 3
        else 0.5 if rk_step in (1, 2) else 1.0
    )
    target = cp.empty_like(rho_zz_old)
    runtime.shared_transport_v841_launch(
        "transport_interpolate_target_v841",
        nlev * ncells,
        (np.int32(nlev * ncells), weight, rho_zz_old, rho_zz_new, target),
    )
    runtime.validate_density(target, validation_flag)
    # MPAS-A v8.4.2 mpas_atm_time_integration.F:4893-4898 adds the
    # held physics source after horizontal divergence on every RK stage.
    if scalar_tendency is None:
        source = None
        zero_source = cp.zeros((chunk, nlev, ncells), dtype=cp.float32)
    else:
        source = cp.ascontiguousarray(scalar_tendency)
        if source.dtype != cp.float32 or source.shape != scalar_old.shape:
            raise ValueError("regional scalar tendency must match native scalar storage")
    output = cp.empty_like(scalar_old)
    # Tracer groups (see _regional_scalar_chunk): every kernel here computes each tracer on its own, so a group is the
    # same words as the whole block; the edge-value and vertical-flux scratch are group-sized.
    edge_values = cp.zeros((chunk, nlev, nedges), dtype=cp.float32)
    vertical_flux = cp.zeros((chunk, nlev + 1, ncells), dtype=cp.float32)
    for first in range(0, ntracers, chunk):
        last = min(ntracers, first + chunk)
        group = last - first
        group_edges = edge_values[:group]
        group_flux = vertical_flux[:group]
        if first:
            group_edges.fill(0.0)
            group_flux.fill(0.0)
        stage_group = scalar_stage[first:last]
        runtime.kernels.launch(
            "transport_edge_values_regional_v841",
            group * nlev * nedges,
            (
                np.int32(group), np.int32(nlev), np.int32(ncells),
                np.int32(nedges), np.int32(width),
                np.float32(config_coef_3rd_order), runtime.bdy_mask_edge,
                mesh.dv_edge, mesh.cells_on_edge, stage_group, uh_avg,
                coefficients.adv_coefs, coefficients.adv_coefs_3rd,
                coefficients.n_adv_cells_for_edge, coefficients.adv_cells_for_edge,
                group_edges,
            ),
        )
        runtime.shared_transport_launch(
            "transport_vertical_flux",
            ncells,
            (
                np.int32(group), np.int32(nlev), np.int32(ncells),
                np.float32(config_coef_3rd_order), stage_group, ww_avg,
                fzm, fzp, group_flux,
            ),
        )
        runtime.kernels.launch(
            "transport_standard_finish_regional_v841",
            group * nlev * ncells,
            (
                np.int32(group), np.int32(nlev), np.int32(ncells),
                np.int32(nedges), np.int32(max_edges), np.float32(dt),
                runtime.bdy_mask_cell, mesh.n_edges_on_cell, mesh.edges_on_cell,
                mesh.edge_sign_on_cell, context.inv_area_cell, uh_avg,
                group_edges, group_flux, rdzw, scalar_old[first:last], rho_zz_old, target,
                zero_source[:group] if source is None else source[first:last], stage_group, output[first:last],
            ),
        )
    return output, target


# THE BREAKAGE THIS PREVENTS (V8X6000 2026-10-29 06:46-07:54Z): NOAA conus3km hrrrv5 on 4 x RTX PRO 6000 (44c0849 row,
# hex 0.4.10 tree 236be507) died in step 1 with CUDA_ERROR_ILLEGAL_ADDRESS on every route (peer and process), named by
# CUDA_LAUNCH_BLOCKING=1 as transport_edge_values_regional_v841.  The regional scalar transport launched all ~20 TEMPO
# tracers at once: ntracers x nlev x nedges = ~2.1e9 elements per rank, past 2^30, so the kernels' int grid-stride
# step (element += gridDim.x * blockDim.x) wrapped negative and indexed outside the arrays; 8 ranks (~1.05e9) sat just
# under the bound.  The same single launch held a (ntracers, nlev, nedges) edge-value block and a (ntracers, nlev+1,
# ncells) vertical-flux block for the stage: ~11 GB per card at 4 ranks.  Tracer groups keep every launch at or under
# WOOF_HEX_REGIONAL_SCALAR_CHUNK_ELEMENTS (default 2^28) edge values, so both kernels' strides stay inside int and the
# scratch (edge values, vertical flux, the zero source when there is no physics source) is group-sized.  Each tracer is computed on its own in all three kernels, so the words are the same.
# WOOF_HEX_REGIONAL_SCALAR_CHUNK_ELEMENTS=0 restores the single launch (A/B only; it overflows past 2^30 elements).

_REGIONAL_SCALAR_CHUNK_ELEMENTS = 1 << 28
_INT_GRID_STRIDE_BOUND = 1 << 30


def _regional_scalar_chunk(ntracers: int, nlev: int, nedges: int, scalar_stage: Any, scalar_old: Any) -> int:
    import os

    raw = os.environ.get("WOOF_HEX_REGIONAL_SCALAR_CHUNK_ELEMENTS", "").strip()
    cap = _REGIONAL_SCALAR_CHUNK_ELEMENTS if not raw else int(raw)
    if cap <= 0:
        return max(int(ntracers), 1)
    per_tracer = max(int(nlev) * int(nedges), 1)
    if per_tracer > _INT_GRID_STRIDE_BOUND:
        raise ValueError(
            f"one tracer's {nlev} x {nedges} edge values exceed the int grid-stride bound (2^30) of the regional "
            "transport kernels; this rank's mesh is too large for one card's limited-area transport"
        )
    if not (scalar_stage.flags.c_contiguous and scalar_old.flags.c_contiguous):
        return max(int(ntracers), 1)  # a strided block cannot be cut into tracer groups by pointer; one launch, as before
    return max(1, min(int(ntracers), cap // per_tracer))


# ---------------------------------------------------------------------------
# the device-side regional runtime the driver hooks call
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _StageOffsets:
    dynamics_substep: int = 1


class CudaRegionalRuntimeV841:
    """Everything ``_step_device_v841`` needs for the regional branch.

    One object per run.  It owns the device masks, the lateral-boundary
    pool, the mesh scalings and the garbage discipline, and exposes one
    method per graft site in the CPU authority's ``driver.py`` so the CUDA
    driver's hooks stay one line each.
    """

    def __init__(
        self,
        mesh: object,
        padded: PaddedRegionalHostMesh,
        *,
        lbc_paths: Sequence[str],
        start_time: datetime,
        config_h_scale_with_mesh: bool,
        config_apply_lbcs: bool,
        outer_dt: float,
        dynamics_split: int,
        time_integration_order: int = 3,
        n_vert_levels: int,
        kernel_cache: Any,
        device_atmosphere: DeviceAtmosphere,
        v841_context: Any,
        config_relax_zone_divdamp_coef: float = 6.0,
        scalar_names: Sequence[str] = ("lbc_qv",),
        measure_garbage: bool = False,
    ) -> None:
        self.driven_scalars = int(len(tuple(scalar_names)))
        import cupy as cp

        self.cp = cp
        dtype = np.dtype(np.float32)
        self.masks_host = derive_regional_masks(mesh, dtype)
        source_mesh = getattr(mesh, "partition_source_mesh", mesh)
        regional_bdy_checks(
            self.masks_host if source_mesh is mesh else derive_regional_masks(source_mesh, dtype),
            config_apply_lbcs=bool(config_apply_lbcs),
            lbc_input_interval_valid=True,
        )
        self.masks = DeviceRegionalMasks.from_host(self.masks_host)
        self.masks.validate()
        self.padded = padded
        self.n_cells_solve = padded.n_cells_solve
        self.n_edges_solve = padded.n_edges_solve
        self.nlev = int(n_vert_levels)
        self.outer_dt = float(outer_dt)
        self.dynamics_split = int(dynamics_split)
        self.time_integration_order = int(time_integration_order)
        self.config_relax_zone_divdamp_coef = float(config_relax_zone_divdamp_coef)
        self.kernels = CudaRegionalKernels(kernel_cache)

        scaling_cell, scaling_edge = compute_mesh_scaling_regional(
            mesh, dtype, config_h_scale_with_mesh=config_h_scale_with_mesh
        )
        self.mesh_scaling_cell = cp.asarray(
            np.ascontiguousarray(scaling_cell, dtype=np.float32)
        )
        self.mesh_scaling_edge = cp.asarray(
            np.ascontiguousarray(scaling_edge, dtype=np.float32)
        )

        self.atmosphere = device_atmosphere
        self.context = v841_context
        self.kernel_cache = kernel_cache
        device_mesh = device_atmosphere.mesh
        self.cells_on_edge = device_mesh.cells_on_edge
        self.cells_on_edge_sentinel = cp.asarray(
            np.ascontiguousarray(
                padded.arrays["cellsOnEdgeSentinel"], dtype=np.int32
            )
        )
        self.spec_zone_mask_cell = cp.asarray(
            np.ascontiguousarray(
                padded.arrays["spec_zone_mask_cell"], dtype=np.float32
            )
        )
        self.spec_zone_mask_edge = device_mesh.spec_zone_mask_edge
        # The zone masks the transport kernels index by THREAD, so they carry
        # the garbage element.  Its ring is the outermost, ``N_BDY_ZONE``:
        # the garbage element is a specified-zone element in every branch
        # that reads a mask, which is what makes its computed values the
        # native pool zeros instead of junk.
        self.bdy_mask_cell = cp.asarray(
            np.ascontiguousarray(
                np.append(
                    np.asarray(self.masks_host.bdy_mask_cell, np.int32),
                    np.int32(N_BDY_ZONE),
                )
            )
        )
        self.bdy_mask_edge = cp.asarray(
            np.ascontiguousarray(
                np.append(
                    np.asarray(self.masks_host.bdy_mask_edge, np.int32),
                    np.int32(N_BDY_ZONE),
                )
            )
        )
        self.edge_sign_on_vertex = cp.asarray(
            _pad_axis(
                np.asarray(edge_signs_on_vertices(mesh), dtype=np.float32),
                0,
                0.0,
            )
        )
        self.zz_solve = cp.ascontiguousarray(
            device_atmosphere.vertical.zz[:, : self.n_cells_solve]
        )

        self.driving = PaddedRegionalDrivingState(
            LbcInventory(list(lbc_paths)),
            _PaddedMaskExtents(padded.n_cells, padded.n_edges),
            garbage_cell=padded.garbage_cell,
            garbage_edge=padded.garbage_edge,
            cells_on_edge=self.cells_on_edge_sentinel,
            zz=device_atmosphere.vertical.zz,
            n_vert_levels=self.nlev,
            kernels=self.kernels,
            scalar_names=scalar_names,
        )
        self.driving.compact_to_boundary(self.bdy_mask_cell)
        layout = getattr(mesh, "partition_layout_v841", None)
        if layout is not None:
            from .partition_regional_v841 import PartitionLbcPoolV841
            self.driving._pool = PartitionLbcPoolV841(self.driving._pool, layout)
        self.clock = start_time
        self._started = False
        self._offsets = _StageOffsets()
        self._u_scratch: Any = None

        self.discipline = RegionalGarbageDiscipline(
            n_cells_solve=padded.n_cells_solve,
            n_edges_solve=padded.n_edges_solve,
            n_vertices_solve=padded.n_vertices_solve,
            measure=measure_garbage,
        )
        self._hold_residency()
        # rho_zz's garbage cell is 1.0 for the WHOLE run, not only inside the
        # dynamics: native writes it in atm_recover_large_step_variables
        # (F:4385-4392) and every ring-7 one-cell edge divides by
        # rho(present) + rho(garbage) whenever velocity is recovered --
        # including the recovery the PHYSICS phase performs before any
        # dynamics of this step has run.
        self.discipline.bind_unit_pool(
            device_atmosphere.state.rho, permanent=True
        )
        self.acoustic_invalid = cp.zeros((1,), dtype=cp.int32)
        # The solve-region check unit: one launch per validation and per
        # restoration instead of one CuPy reduction or fill per array.
        from .cuda_solve_region_v841 import SolveRegionKernels

        self.solve_region = SolveRegionKernels.for_cache(kernel_cache)
        self.discipline.fused = self.solve_region
        # Arm the discipline on the cache every kernel of the step resolves
        # through, so no shared translation unit needs a line for it.
        kernel_cache.post_launch = self.discipline.scrub

    # -- residency ---------------------------------------------------------

    def _hold_residency(self) -> None:
        """Mesh geometry and the resident inverses are never rewritten."""

        mesh = self.atmosphere.mesh
        vertical = self.atmosphere.vertical
        reference = self.atmosphere.reference
        terrain = self.atmosphere.terrain
        self.discipline.hold_immutable(
            *(
                getattr(mesh, name)
                for name in (
                    "weights_on_edge", "dc_edge", "dv_edge", "area_cell",
                    "area_triangle", "kite_areas_on_vertex", "lat_cell",
                    "lon_cell", "lat_edge", "lon_edge", "angle_edge",
                    "mesh_density", "defc_a", "defc_b", "f_vertex", "f_edge",
                    "edge_sign_on_cell", "spec_zone_mask_edge",
                )
            ),
            *(
                getattr(vertical, name)
                for name in ("hx", "zgrid", "zz", "zxu", "dss")
            ),
            *(
                getattr(reference, name)
                for name in (
                    "rho_base", "rho_theta_base", "pressure_base", "exner_base"
                )
            ),
            *(
                getattr(self.context, name)
                for name in (
                    "inv_area_cell", "inv_area_triangle", "inv_dc_edge",
                    "inv_dv_edge"
                )
            ),
            self.mesh_scaling_cell,
            self.mesh_scaling_edge,
            self.spec_zone_mask_cell,
            *(() if terrain is None else (terrain.zb_cell, terrain.zb3_cell)),
        )

    def bind_state_rho(self, rho: Any) -> None:
        """Re-declare ``rho_zz``'s garbage cell after a state rebinding."""

        self.discipline.bind_unit_pool(rho)

    def begin_step(self, state: Any) -> None:
        """Everything the CPU authority does before the step's arithmetic."""

        self.discipline.release_unit_pool()
        self.discipline.bind_unit_pool(state.rho)
        self.ensure_interval()

    # -- clock and the lbc_in cadence -------------------------------------

    def ensure_interval(self) -> None:
        """``atm_core_run``'s ``lbc_in`` read cadence (F:735-781)."""

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

    def begin_dynamics_substep(self, substep: int) -> None:
        self._offsets.dynamics_substep = int(substep)

    def _stage_offset(self, rk_step: int) -> np.float32:
        return dynamics_time_offset(
            outer_dt=self.outer_dt,
            dynamics_split=self.dynamics_split,
            dynamics_substep=self._offsets.dynamics_substep,
            rk_timestep=rk_timestep_f32(
                outer_dt=self.outer_dt,
                dynamics_split=self.dynamics_split,
                rk_step=int(rk_step),
                order=self.time_integration_order,
            ),
        )

    def driving_values(self, name: str, rk_step: int) -> Any:
        return self.driving.state_at(name, self.clock, self._stage_offset(rk_step))

    # -- graft: the tendency adjust pair (atm_srk3:2300-2351) --------------

    def adjust_dynamics_tendencies(
        self,
        *,
        tend_ru: Any,
        tend_rho: Any,
        tend_rt: Any,
        tend_omega: Any,
        rho_u: Any,
        theta_m: Any,
        rho_zz: Any,
        rk_step: int,
        rt_diabatic_tend: Any | None = None,
    ) -> None:
        """Boundary stages from mpas_atm_time_integration.F:7906-8198.

        Physics callers supply the diabatic tendency pool cleared at F:7949.
        """
        nlev = self.nlev
        ncells = int(tend_rho.shape[-1])
        nedges = int(tend_ru.shape[-1])
        masks = self.masks
        self.kernels.launch(
            "regional_speczone_tend_cell_v841",
            max(nlev * masks.n_spec_cells, 1),
            (
                np.int32(nlev), np.int32(ncells),
                np.int32(masks.n_spec_cells), masks.spec_cells,
                self.driving.tendency("rho_zz"),
                self.driving.tendency("rtheta_m"),
                tend_rho, tend_rt, tend_omega,
            ),
        )
        self.kernels.launch(
            "regional_speczone_tend_edge_v841",
            max(nlev * masks.n_spec_edges, 1),
            (
                np.int32(nlev), np.int32(nedges),
                np.int32(masks.n_spec_edges), masks.spec_edges,
                self.driving.tendency("ru"), tend_ru,
            ),
        )
        if rt_diabatic_tend is not None:
            # mpas_atm_time_integration.F:7949: clear only the specified
            # zone, reusing the existing elementwise zero kernel. No CUDA
            # translation-unit change or extra host transfer is needed.
            self.kernels.launch(
                "regional_zero_speczone_w_v841",
                max(nlev * masks.n_spec_cells, 1),
                (
                    np.int32(nlev), np.int32(ncells),
                    np.int32(masks.n_spec_cells), masks.spec_cells,
                    rt_diabatic_tend,
                ),
            )
        values_ru = self.driving_values("ru", rk_step)
        values_rt = self.driving_values("rtheta_m", rk_step)
        values_rho = self.driving_values("rho_zz", rk_step)
        fifty_dt = np.float32(np.float32(50.0) * np.float32(self.outer_dt))
        ten_dt = np.float32(np.float32(10.0) * np.float32(self.outer_dt))
        relax_zone = np.float32(N_RELAX_ZONE)
        mesh = self.atmosphere.mesh
        self.kernels.launch(
            "regional_relaxzone_rayleigh_cell_v841",
            max(nlev * masks.n_relax_cells, 1),
            (
                np.int32(nlev), np.int32(ncells),
                np.int32(masks.n_relax_cells), fifty_dt, relax_zone,
                masks.relax_cells, masks.bdy_mask_cell,
                self.mesh_scaling_cell, rho_zz, theta_m,
                values_rho, values_rt, tend_rho, tend_rt,
            ),
        )
        self.kernels.launch(
            "regional_relaxzone_rayleigh_edge_v841",
            max(nlev * masks.n_relax_edges, 1),
            (
                np.int32(nlev), np.int32(nedges),
                np.int32(masks.n_relax_edges), fifty_dt, relax_zone,
                masks.relax_edges, masks.bdy_mask_edge,
                self.mesh_scaling_edge, rho_u, values_ru, tend_ru,
            ),
        )
        self.kernels.launch(
            "regional_relaxzone_filter_cell_v841",
            max(nlev * masks.n_relax_cells, 1),
            (
                np.int32(nlev), np.int32(ncells), np.int32(nedges),
                np.int32(mesh.max_edges), np.int32(masks.n_relax_cells),
                ten_dt, relax_zone, masks.relax_cells, masks.bdy_mask_cell,
                self.mesh_scaling_cell, mesh.n_edges_on_cell,
                mesh.edges_on_cell, mesh.cells_on_edge,
                mesh.edge_sign_on_cell, mesh.dv_edge,
                self.context.inv_dc_edge, rho_zz, theta_m,
                values_rho, values_rt, tend_rho, tend_rt,
            ),
        )
        self.kernels.launch(
            "regional_relaxzone_filter_edge_v841",
            max(nlev * masks.n_relax_edges, 1),
            (
                np.int32(nlev), np.int32(ncells), np.int32(nedges),
                np.int32(mesh.n_vertices), np.int32(mesh.max_edges),
                np.int32(mesh.vertex_degree), np.int32(masks.n_relax_edges),
                ten_dt, np.float32(self.config_relax_zone_divdamp_coef),
                relax_zone, masks.relax_edges, masks.bdy_mask_edge,
                self.mesh_scaling_edge, mesh.n_edges_on_cell,
                mesh.edges_on_cell, mesh.cells_on_edge, mesh.vertices_on_edge,
                mesh.edges_on_vertex, mesh.edge_sign_on_cell,
                self.edge_sign_on_vertex, mesh.dc_edge, mesh.dv_edge,
                self.context.inv_dc_edge, self.context.inv_dv_edge,
                self.context.inv_area_cell, self.context.inv_area_triangle,
                rho_u, values_ru, tend_ru,
            ),
        )

    # -- graft: the specified-zone velocity overwrite ----------------------

    def overwrite_speczone_rho_u(self, rho_u: Any, rk_step: int) -> Any:
        """``atm_srk3:2442-2485``, the ``ru`` half.

        The CPU authority never calls ``overwrite_speczone_u_ru`` as a unit:
        its two assignments straddle the candidate validation, the flux
        accumulation and the transport call (``driver.py:2935-2937`` and
        ``3006-3009``).  This half writes ``ru`` and sends the ``u`` write to
        a scratch sink; :meth:`overwrite_speczone_u` does the reverse with
        the SAME interpolated driving values, so the pair is exactly the
        native pair split at the native point.
        """

        values_u = self.driving_values("u", rk_step)
        values_ru = self.driving_values("ru", rk_step)
        masks = self.masks
        self.kernels.launch(
            "regional_speczone_u_ru_v841",
            max(self.nlev * masks.n_spec_edges, 1),
            (
                np.int32(self.nlev), np.int32(int(rho_u.shape[-1])),
                np.int32(masks.n_spec_edges), masks.spec_edges,
                values_u, values_ru, self._u_sink(rho_u), rho_u,
            ),
        )
        return values_u

    def _u_sink(self, like: Any) -> Any:
        sink = self._u_scratch
        if sink is None or sink.shape != like.shape:
            sink = self.cp.zeros_like(like)
            self._u_scratch = sink
        return sink

    def overwrite_speczone_u(self, normal_velocity: Any, values_u: Any) -> None:
        """``atm_srk3:2442-2485``, the ``u`` half, after recover."""

        masks = self.masks
        self.kernels.launch(
            "regional_speczone_u_ru_v841",
            max(self.nlev * masks.n_spec_edges, 1),
            (
                np.int32(self.nlev),
                np.int32(int(normal_velocity.shape[-1])),
                np.int32(masks.n_spec_edges), masks.spec_edges,
                values_u, values_u, normal_velocity,
                self._u_sink(normal_velocity),
            ),
        )

    # -- shared-kernel launches on the regional path -----------------------

    def shared_launch(
        self, name: str, count: int, args: tuple[Any, ...]
    ) -> None:
        """Launch one BYTE-UNCHANGED ``cuda_acoustic_v841`` entrypoint."""

        from .cuda_acoustic_v841 import _kernel as acoustic_kernel

        threads = 128
        acoustic_kernel(name, self.kernel_cache)(
            ((int(count) + threads - 1) // threads,), (threads,), args
        )

    def shared_transport_launch(
        self, name: str, count: int, args: tuple[Any, ...]
    ) -> None:
        """Launch one BYTE-UNCHANGED ``cuda_transport`` entrypoint."""

        from .cuda_transport import _kernel as transport_kernel

        threads = 128
        transport_kernel(name, self.kernel_cache)(
            ((int(count) + threads - 1) // threads,), (threads,), args
        )

    def shared_transport_v841_launch(
        self, name: str, count: int, args: tuple[Any, ...]
    ) -> None:
        """Launch one BYTE-UNCHANGED ``cuda_transport_v841`` entrypoint."""

        from .cuda_transport_v841 import _kernel as transport_kernel

        threads = 128
        transport_kernel(name, self.kernel_cache)(
            ((int(count) + threads - 1) // threads,), (threads,), args
        )

    def zero_speczone_w(self, vertical_velocity: Any) -> None:
        """``atm_zero_gradient_w_bdy`` (F:7868-7902)."""

        masks = self.masks
        self.kernels.launch(
            "regional_zero_speczone_w_v841",
            max((self.nlev + 1) * masks.n_spec_cells, 1),
            (
                np.int32(self.nlev + 1),
                np.int32(int(vertical_velocity.shape[-1])),
                np.int32(masks.n_spec_cells), masks.spec_cells,
                vertical_velocity,
            ),
        )

    # -- graft: the scalar boundary stages ---------------------------------

    def bdy_adjust_scalars(self, scalars: Any, rk_step: int) -> None:
        """``atm_srk3:2688-2717``."""

        dt_rk = transport_rk_timestep_f32(
            outer_dt=self.outer_dt, rk_step=int(rk_step),
            order=self.time_integration_order,
        )
        if self._ring_scalars_on() and hasattr(self.driving, "state_at_support"):
            driving, support = self.driving.state_at_support("scalars", self.clock, dt_rk)
            if support is not None:
                self._bdy_adjust_scalars_ring(scalars, driving, dt_rk)
                return
            del driving, support
        driving = self.driving.state_at("scalars", self.clock, dt_rk)
        masks = self.masks
        mesh = self.atmosphere.mesh
        ntracers = self._driven_tracer_count(scalars)
        ncells = int(scalars.shape[-1])
        nedges = int(mesh.n_edges)
        ten_dt = np.float32(np.float32(10.0) * np.float32(self.outer_dt))
        updates = self.cp.zeros_like(scalars)
        self.kernels.launch(
            "regional_bdy_adjust_scalars_compute_v841",
            max(ntracers * self.nlev
                * (masks.n_relax_cells + masks.n_spec_cells), 1),
            (
                np.int32(ntracers), np.int32(self.nlev), np.int32(ncells),
                np.int32(nedges), np.int32(mesh.max_edges),
                np.int32(masks.n_relax_cells), np.int32(masks.n_spec_cells),
                ten_dt, np.float32(dt_rk), np.float32(N_RELAX_ZONE),
                masks.relax_cells, masks.spec_cells, masks.bdy_mask_cell,
                self.mesh_scaling_cell, mesh.n_edges_on_cell,
                mesh.edges_on_cell, mesh.cells_on_edge,
                mesh.edge_sign_on_cell, mesh.dv_edge,
                self.context.inv_dc_edge, scalars, driving, updates,
            ),
        )
        self.kernels.launch(
            "regional_bdy_adjust_scalars_copyback_v841",
            max(ntracers * self.nlev * masks.n_nudged_cells, 1),
            (
                np.int32(ntracers), np.int32(self.nlev), np.int32(ncells),
                np.int32(masks.n_nudged_cells), masks.nudged_cells,
                updates, scalars,
            ),
        )

    # THE BREAKAGE THIS PREVENTS (conus3km hrrrv5 4 x RTX PRO 6000, V8X6000 2026-10-29 05:30Z, lease c4p-c3km2 arm
    # B): rank 2 ran out of memory at step 30 inside bdy_adjust_scalars, allocating cp.zeros_like(scalars) (1.81 GB)
    # at 98.8 GB held, right after driving.state_at('scalars') had scattered the compact driving pool back to a
    # second full-extent 1.81 GB array.  Both arrays spanned every cell of the rank although only the seven boundary
    # rings and their first neighbours are ever read or written (~2-3 % of conus3km's columns).  The ring path below
    # runs the SAME kernel on the boundary support (the driving pool's own compact index set): the stage scalars are
    # gathered there, the compute kernel sees support-local cell numbers (cellsOnEdge remapped, per-cell tables
    # gathered, edge tables unchanged), and the nudged cells are copied back.  Every value the kernel reads is the
    # same word in the same order of operations, so the output is word-identical; the transients are support-sized.
    # The end-of-step specified-zone set (reset_speczone_values) takes the same support values.
    # WOOF_HEX_REGIONAL_RING_SCALARS=0 restores the full-extent path for A/B.

    @staticmethod
    def _ring_scalars_on() -> bool:
        import os

        return os.environ.get("WOOF_HEX_REGIONAL_RING_SCALARS", "1").strip().lower() not in (
            "0", "off", "false", "no")

    def _ring_plan(self, ncells: int) -> dict[str, Any]:
        """Support-local mesh tables for the boundary scalar kernel, built once per rank."""

        plan = getattr(self, "_ring_plan_cache", None)
        if plan is not None and plan["ncells"] == ncells:
            return plan
        cp = self.cp
        driving = self.driving
        if int(driving.ncells) != int(ncells):
            raise RuntimeError(
                f"the boundary driving pool spans {int(driving.ncells)} cells and the stage scalars {int(ncells)}"
            )
        support = cp.ascontiguousarray(driving._support["cell"].astype(cp.int64))
        nsup = int(support.size)
        to_local = cp.full(int(ncells), -1, dtype=cp.int32)
        to_local[support] = cp.arange(nsup, dtype=cp.int32)
        garbage_local = int(to_local[int(driving._garbage_cell)])
        if garbage_local < 0:
            raise RuntimeError("the garbage cell is outside the boundary support")
        to_local = cp.where(to_local < 0, cp.int32(garbage_local), to_local)
        mesh = self.atmosphere.mesh
        masks = self.masks
        max_edges = int(mesh.max_edges)

        def gather_cells(values: Any) -> Any:
            flat = values.reshape(-1)
            return cp.ascontiguousarray(flat[cp.minimum(support, int(flat.size) - 1)])

        def gather_rows(values: Any) -> Any:
            table = values.reshape(-1, max_edges)
            return cp.ascontiguousarray(table[cp.minimum(support, int(table.shape[0]) - 1)])

        def localise(cells: Any) -> Any:
            flat = cells.reshape(-1).astype(cp.int64)
            valid = (flat >= 0) & (flat < int(ncells))
            local = cp.where(valid, to_local[cp.where(valid, flat, 0)], cp.int32(garbage_local))
            return cp.ascontiguousarray(local.astype(cp.int32).reshape(cells.shape))

        relax_local = localise(masks.relax_cells)
        spec_local = localise(masks.spec_cells)
        nudged_local = localise(masks.nudged_cells)
        for label, cells, local in (("relaxation", masks.relax_cells, relax_local),
                                    ("specified", masks.spec_cells, spec_local),
                                    ("nudged", masks.nudged_cells, nudged_local)):
            if int(cells.size) and not bool(cp.all(support[local.reshape(-1).astype(cp.int64)]
                                                   == cells.reshape(-1).astype(cp.int64))):
                raise RuntimeError(f"a {label}-zone cell is outside the boundary support")
        plan = {
            "ncells": int(ncells),
            "nsupport": nsup,
            "support": support,
            "relax": relax_local,
            "spec": spec_local,
            "spec_global": cp.ascontiguousarray(masks.spec_cells.reshape(-1).astype(cp.int64)),
            "spec_local64": cp.ascontiguousarray(spec_local.reshape(-1).astype(cp.int64)),
            "nudged_global": cp.ascontiguousarray(masks.nudged_cells.reshape(-1).astype(cp.int64)),
            "nudged_local64": cp.ascontiguousarray(nudged_local.reshape(-1).astype(cp.int64)),
            "bdy_mask_cell": gather_cells(masks.bdy_mask_cell),
            "mesh_scaling_cell": gather_cells(self.mesh_scaling_cell),
            "n_edges_on_cell": gather_cells(mesh.n_edges_on_cell),
            "edges_on_cell": gather_rows(mesh.edges_on_cell),
            "edge_sign_on_cell": gather_rows(mesh.edge_sign_on_cell),
            "cells_on_edge": localise(mesh.cells_on_edge),
        }
        self._ring_plan_cache = plan
        return plan

    def _bdy_adjust_scalars_ring(self, scalars: Any, driving: Any, dt_rk: Any) -> None:
        """``atm_bdy_adjust_scalars_work`` on the boundary support (see the breakage note above)."""

        cp = self.cp
        masks = self.masks
        mesh = self.atmosphere.mesh
        ntracers = self._driven_tracer_count(scalars)
        plan = self._ring_plan(int(scalars.shape[-1]))
        nsup = plan["nsupport"]
        if int(driving.shape[-1]) != nsup or int(driving.shape[0]) < ntracers:
            raise RuntimeError(
                f"compact driving scalars {tuple(driving.shape)} do not match {ntracers} tracers on {nsup} cells"
            )
        ten_dt = np.float32(np.float32(10.0) * np.float32(self.outer_dt))
        local = cp.ascontiguousarray(scalars[:ntracers, :, plan["support"]])
        updates = cp.zeros_like(local)
        self.kernels.launch(
            "regional_bdy_adjust_scalars_compute_v841",
            max(ntracers * self.nlev
                * (masks.n_relax_cells + masks.n_spec_cells), 1),
            (
                np.int32(ntracers), np.int32(self.nlev), np.int32(nsup),
                np.int32(int(mesh.n_edges)), np.int32(mesh.max_edges),
                np.int32(masks.n_relax_cells), np.int32(masks.n_spec_cells),
                ten_dt, np.float32(dt_rk), np.float32(N_RELAX_ZONE),
                plan["relax"], plan["spec"], plan["bdy_mask_cell"],
                plan["mesh_scaling_cell"], plan["n_edges_on_cell"],
                plan["edges_on_cell"], plan["cells_on_edge"],
                plan["edge_sign_on_cell"], mesh.dv_edge,
                self.context.inv_dc_edge, local, driving, updates,
            ),
        )
        del local
        if masks.n_nudged_cells:
            scalars[:ntracers, :, plan["nudged_global"]] = updates[:, :, plan["nudged_local64"]]

    def clamp_negative_scalars(self, scalars: Any) -> None:
        """``atm_srk3:2798-2800``."""

        self.kernels.launch(
            "regional_clamp_negative_scalars_v841",
            int(scalars.size),
            (np.int32(scalars.size), scalars),
        )

    # -- graft: the end-of-step resets -------------------------------------

    def reset_speczone_values(
        self,
        *,
        theta_m: Any,
        rho_theta: Any,
        rho_theta_perturbation: Any,
        scalars: Any,
    ) -> None:
        """``atm_srk3:2828-2878`` plus the F:8238 perturbation write."""

        dt_f32 = np.float32(self.outer_dt)
        rt_values = self.driving.state_at("rtheta_m", self.clock, dt_f32)
        rho_values = self.driving.state_at("rho_zz", self.clock, dt_f32)
        masks = self.masks
        ncells = int(theta_m.shape[-1])
        self.kernels.launch(
            "regional_reset_speczone_values_v841",
            max(self.nlev * masks.n_spec_cells, 1),
            (
                np.int32(self.nlev), np.int32(ncells),
                np.int32(masks.n_spec_cells), masks.spec_cells,
                rt_values, rho_values,
                self.atmosphere.reference.rho_theta_base,
                theta_m, rho_theta, rho_theta_perturbation,
            ),
        )
        if self._ring_scalars_on() and hasattr(self.driving, "state_at_support"):
            driving, support = self.driving.state_at_support("scalars", self.clock, dt_f32)
            if support is not None:
                ntracers = self._driven_tracer_count(scalars)
                plan = self._ring_plan(int(scalars.shape[-1]))
                if masks.n_spec_cells:
                    scalars[:ntracers, :, plan["spec_global"]] = driving[:ntracers, :, plan["spec_local64"]]
                return
            del driving, support
        driving = self.driving.state_at("scalars", self.clock, dt_f32)
        self.kernels.launch(
            "regional_bdy_set_scalars_v841",
            max(self._driven_tracer_count(scalars) * self.nlev
                * masks.n_spec_cells, 1),
            (
                np.int32(self._driven_tracer_count(scalars)), np.int32(self.nlev),
                np.int32(int(scalars.shape[-1])), np.int32(masks.n_spec_cells),
                masks.spec_cells, driving, scalars,
            ),
        )

    def _driven_tracer_count(self, scalars: Any) -> int:
        """Bound the configured native boundary scalar inventory.

        MPAS-A v8.4.2 Registry.xml:2150-2196; mpas_atm_boundaries.F:398-419.
        Missing stream variables retain native allocated/recycled pool values;
        they still participate in the specified and relaxation boundary loops.
        A configured inventory wider than the model would index beyond its
        scalar array, the measured r4.75.11020 failure this check prevents.
        """

        model = int(scalars.shape[0])
        if self.driven_scalars > model:
            raise RegionalAdmissionError(
                f"the lateral-boundary stream drives {self.driven_scalars} "
                f"scalar species and this run integrates {model}; the extra "
                "driving fields belong to species no scheme in this "
                "configuration carries"
            )
        return self.driven_scalars

    def moist_coefficients(self, scalars: Any) -> Any:
        """``atm_compute_moist_coefficients`` (F:3188-3283) for one species.

        The CPU authority computes ``qtot``/``cqw``/``cqu`` on EVERY regional
        step (``driver.py:3090-3105``): the pinned record is
        ``config_moist_physics=true`` with a single ``qv``.

        GUARD WHOSE REASON EXPIRED 2026-09-01, handed up rather than
        retired here.  This numpy path existed because
        ``moist_cell_coefficients_v841_f32`` and
        ``moist_edge_coefficients_v841_f32`` hardwired the six WSM6
        species "so they cannot serve a one-species regional block without
        moving ``cuda_driver``'s CUDA source".  Both kernels now take
        ``nmass`` from the species row, so a one-species block is exactly
        what they can serve, and this workaround is the kind a fix is
        supposed to retire.

        It is NOT retired in this commit, deliberately: swapping the numpy
        path for the device kernels changes the regional step's arithmetic,
        which is the one thing this lane has been careful not to do while
        the regional mint is already lapsed and owed a re-run.  Retiring it
        is a change that must land WITH its own re-mint, not beside
        somebody else's.  The three expressions are formed here
        with the same operations in the same order --
        ``qtot = 0 + q``, ``cqw(k>=1) = 1/(1 + 0.5*(qtot(k)+qtot(k-1)))``,
        ``cqu = 1/(1 + 0.5*(q(c1)+q(c2)))`` -- as separate float32 device
        operations, which is exactly what the numpy authority does.

        The edge gather runs through the REMAPPED connectivity on the padded
        block, so a one-cell ring-7 edge reads the garbage column's zero,
        which is what ``pad_cells_column(scalars, 0.0)`` gives the authority.
        """

        from .cuda_driver import _CudaV841MoistDynamicsCoefficients

        cp = self.cp
        q = cp.ascontiguousarray(scalars[0])
        qtot = cp.zeros_like(q)
        qtot = cp.ascontiguousarray(qtot + q)
        cqw = cp.ones_like(qtot)
        half = np.float32(0.5)
        one = np.float32(1.0)
        cqw[1:] = one / (one + half * (qtot[1:] + qtot[:-1]))
        cqw = cp.ascontiguousarray(cqw)
        coe = self.cells_on_edge
        edge_total = half * (q[:, coe[:, 0]] + q[:, coe[:, 1]])
        cqu = cp.ascontiguousarray(one / (one + edge_total))
        return _CudaV841MoistDynamicsCoefficients(qtot=qtot, cqw=cqw, cqu=cqu)

    def validate_recovered(self, state: Any, saved: Any, flag: Any) -> None:
        """``validate_recovered_v841_f32``'s contract, over the SOLVE region.

        The shared kernel is launched over the padded extent and demands
        ``rho``, ``rho_theta``, ``theta`` and ``exner`` be strictly POSITIVE.
        The garbage element is not a solved element -- native never enters a
        loop for it and the port holds it at the native allocation, which is
        zero for three of those four -- so the shared kernel would refuse a
        correct regional step for a value that means nothing.  The regional
        lane therefore runs the identical test, term for term, over the
        elements native solves, and is no weaker: every real element is
        checked for exactly what the shared kernel checks it for.

        Doing it here rather than by loosening the kernel is what keeps
        ``cuda_dynamics_v841``'s source byte-unchanged.

        The test runs on the device and sets the step's flag there; nothing
        is read back until the step reads its flag once before publishing.
        It used to be eleven ``cp.all(cp.isfinite(x) & (x > 0))`` reductions
        on strided views, each with its own temporaries and its own drain of
        the stream, per RK stage.
        """

        cells = self.n_cells_solve
        edges = self.n_edges_solve
        self.solve_region.validate(
            flag,
            positive=(
                (state.rho, cells),
                (state.rho_theta, cells),
                (saved.theta_m, cells),
                (saved.exner, cells),
            ),
            finite=(
                (saved.density_perturbation, cells),
                (saved.rho_theta_perturbation, cells),
                (saved.pressure_perturbation, cells),
                (state.rho_w, cells),
                (saved.vertical_velocity, cells),
                (state.rho_u, edges),
                (saved.normal_velocity, edges),
            ),
        )

    def validate_density(self, density: Any, flag: Any) -> None:
        """``_check_density``'s contract, over the SOLVE region, on the device.

        Three calls a step (one per transport stage); each was a full-size
        ``isfinite`` temporary, a comparison temporary, an ``all`` and a
        drain -- 10 ms of host wall per call measured on the point mesh.
        """

        self.solve_region.validate(flag, positive=((density, self.n_cells_solve),))

    def history_slice(self, array: Any) -> Any:
        """Strip the garbage element off a field before it leaves the card."""

        extent = int(array.shape[-1])
        if extent == self.n_cells_solve + 1:
            return array[..., : self.n_cells_solve]
        if extent == self.n_edges_solve + 1:
            return array[..., : self.n_edges_solve]
        return array

    def receipt(self) -> dict[str, Any]:
        return {
            "n_cells_solve": self.n_cells_solve,
            "n_edges_solve": self.n_edges_solve,
            "spec_cells": int(self.masks.n_spec_cells),
            "spec_edges": int(self.masks.n_spec_edges),
            "relax_cells": int(self.masks.n_relax_cells),
            "relax_edges": int(self.masks.n_relax_edges),
            "nudged_cells": int(self.masks.n_nudged_cells),
            "acoustic_one_cell_edge_flag": int(
                self.cp.asnumpy(self.acoustic_invalid)[0]
            ),
            "garbage": self.discipline.receipt(),
        }


# ---------------------------------------------------------------------------
# the front door
# ---------------------------------------------------------------------------


class _PaddedDeformationWeights:
    __slots__ = ("coef_c2", "coef_s2", "coef_cs")

    def __init__(self, weights: object) -> None:
        for name in self.__slots__:
            setattr(self, name, _pad_axis(getattr(weights, name), 0, 0.0))


def _measured_zone_width(mesh: object) -> int:
    """How many boundary rings this cull carries, read off the mesh.

    The regional stages are indexed by ring, so the zone's WIDTH is part of
    what a forecast mint measured and part of the class key.  Measured rather
    than declared: a row asserting seven rings over a mesh carrying five is
    exactly the two-classifiers-disagreeing shape this program has lost a day
    to.
    """

    mask = _mesh_value(mesh, "bdyMaskCell")
    if mask is None:
        return 0
    return int(np.max(np.asarray(mask)))


def _measured_finest_edge_m(mesh: object) -> float:
    """``min(dcEdge)``, in metres, off the mesh in hand.

    A cull moves no cell centre, so every cull containing its parent's fine
    core carries the parent's own finest edge in identical float64 bits.  That
    is why this length identifies a CLASS of culls and their cell counts do
    not.
    """

    return float(np.min(np.asarray(_mesh_value(mesh, "dcEdge"))))


def open_regional_forecast_v841(
    mesh: object,
    state: object,
    vertical: object,
    reference: object,
    saved: object,
    terrain: object,
    config: object,
    *,
    reference_wind_profiles: object,
    lbc_paths: Sequence[str],
    start_time: datetime,
    kernel_cache: Any = None,
    measure_garbage: bool = False,
    scalar_names: Sequence[str] = ("lbc_qv",),
    les_dissipation: Any = None,
    driver_type: Any = None,
) -> Any:
    """Open one regional configuration on the card, recording its anchor.

    RETIRED 2026-10-10 as a gate (owner ruling).  This used to refuse every regional configuration whose class
    held no forecast mint, and the only way past it was a measurement
    instrument that swapped this function for the assembler.  It now asks
    the same question, records the answer on the driver
    (``driver.regional_v841.anchor``: the admitted row, or
    ``{"matched": False, "reason": ...}``) for the receipt, and assembles the
    run either way.

    THIS IS THE SITE THAT KNOWS THE CONFIGURATION, and since the 2026-08-27
    re-keying that matters.  An anchor has two halves keyed two different
    ways -- a per-geometry contract deck and a per-CLASS forecast mint -- and
    the class key is the set of inputs the mint's own instrument reads:
    boundary-zone width, column count, the mesh's finest edge, the timestep
    and the regional kernel set's own bytes.  Every one of those is in scope
    here, so the key handed to the gate is MEASURED off this run rather than
    transcribed from a table.
    """

    from .cuda_backend.regional_admission import require_regional_anchor
    from .cuda_driver import regional_bdy_mask_digest

    # An internal partition cut is not a new physical regional boundary.
    # Admission measures the source geometry; construction uses its exact
    # local entity view, with the physical masks and native pool retained.
    source_mesh = getattr(mesh, "partition_source_mesh", mesh)
    masks = derive_regional_masks(source_mesh, np.dtype(np.float32))
    mesh_row = _mesh_value(source_mesh, "registry_row")
    try:
        anchor = require_regional_anchor(
            None if mesh_row is None else str(mesh_row),
            bdy_mask_sha256=regional_bdy_mask_digest(source_mesh),
            n_cells=int(masks.n_cells),
            boundary_zone_width=_measured_zone_width(source_mesh),
            n_vert_levels=int(np.asarray(getattr(state, "rho")).shape[0]),
            finest_edge_m=_measured_finest_edge_m(source_mesh),
            dt_seconds=float(config.config_dt),
        )
    except RuntimeError as error:
        anchor = {"matched": False, "reason": str(error)}
    driver = assemble_regional_driver_v841(
        mesh,
        state,
        vertical,
        reference,
        saved,
        terrain,
        config,
        reference_wind_profiles=reference_wind_profiles,
        lbc_paths=lbc_paths,
        start_time=start_time,
        kernel_cache=kernel_cache,
        measure_garbage=measure_garbage,
        scalar_names=tuple(scalar_names),
        les_dissipation=les_dissipation,
        driver_type=driver_type,
    )
    driver.regional_v841.anchor = anchor
    return driver


def assemble_regional_driver_v841(
    mesh: object,
    state: object,
    vertical: object,
    reference: object,
    saved: object,
    terrain: object,
    config: object,
    *,
    reference_wind_profiles: object,
    lbc_paths: Sequence[str],
    start_time: datetime,
    kernel_cache: Any = None,
    measure_garbage: bool = False,
    scalar_names: Sequence[str] = ("lbc_qv",),
    les_dissipation: Any = None,
    driver_type: Any = None,
) -> Any:
    """Build the regional device driver.  Carries NO admission decision."""

    from .cuda_backend import KernelCache, require_cuda
    from .cuda_driver import CudaDryDycoreDriver
    from .cuda_transport import CudaAdvectionCoefficients
    from .damping_v841 import build_v841_vertical_velocity_damping
    from .offcentering_v841 import build_v841_acoustic_offcentering
    from .mixing_v841 import initialize_deformation_weights_v841
    from .transport import build_advection_coefficients

    dtype = np.dtype(np.float32)
    masks = derive_regional_masks(mesh, dtype)
    require_cuda(min_compute=(12, 0))
    cache = KernelCache() if kernel_cache is None else kernel_cache
    padded = PaddedRegionalHostMesh(mesh, masks)

    n_vert_levels = int(np.asarray(getattr(state, "rho")).shape[0])
    offcentering = build_v841_acoustic_offcentering(
        np.asarray(vertical.rdzw),
        minimum=config.config_epssm_minimum,
        maximum=config.config_epssm_maximum,
        transition_bottom_z=config.config_epssm_transition_bottom_z,
        transition_top_z=config.config_epssm_transition_top_z,
    )
    # Core always rebuilds dss from config_xnutr/config_zd; it is built on the
    # UNPADDED zgrid and padded, because the garbage column of zgrid is the
    # pool zero and a damping profile of it means nothing.
    selected_dss = build_v841_vertical_velocity_damping(
        np.asarray(vertical.zgrid),
        xnutr=config.config_xnutr,
        damping_start_z=config.config_zd,
    )

    class _VerticalWithDss:
        def __getattr__(self, name: str) -> Any:
            if name == "dss":
                return selected_dss
            return getattr(vertical, name)

    device_atmosphere = build_regional_device_atmosphere(
        padded, state, _VerticalWithDss(), reference, saved, terrain
    )
    context = regional_padded_context_v841(
        mesh, offcentering, reference_wind_profiles, n_vert_levels=n_vert_levels
    )
    host_coefficients = build_advection_coefficients(
        mesh,
        config_scalar_adv_order=config.config_scalar_adv_order,
        n_vert_levels=n_vert_levels,
        source_order_v841=True,
        allow_regional_sentinels=True,
    )
    device_coefficients = CudaAdvectionCoefficients.from_host(
        regional_padded_advection_coefficients(
            host_coefficients, padded.garbage_cell
        )
    )
    driver = (CudaDryDycoreDriver if driver_type is None else driver_type)(
        device_atmosphere,
        config,
        device_coefficients,
        kernel_cache=cache,
        v841_context=context,
        les_dissipation=les_dissipation,
    )
    if driver.mixing_config_v841 is not None:
        driver.deformation_weights_receipt_v841 = (
            driver.horizontal.attach_deformation_weights_v841(
                _PaddedDeformationWeights(
                    initialize_deformation_weights_v841(mesh, dtype=dtype)
                )
            )
        )
    runtime = CudaRegionalRuntimeV841(
        mesh,
        padded,
        lbc_paths=lbc_paths,
        start_time=start_time,
        config_h_scale_with_mesh=bool(config.config_h_ScaleWithMesh),
        config_apply_lbcs=bool(config.config_apply_lbcs),
        # Registry.xml:284 default is retained until the shared config schema
        # exposes the regional coefficient; consume it as soon as it does.
        config_relax_zone_divdamp_coef=float(
            getattr(config, "config_relax_zone_divdamp_coef", 6.0)
        ),
        outer_dt=float(config.config_dt),
        dynamics_split=int(config.config_dynamics_split_steps),
        time_integration_order=int(config.config_time_integration_order),
        n_vert_levels=n_vert_levels,
        kernel_cache=cache,
        device_atmosphere=device_atmosphere,
        v841_context=context,
        measure_garbage=measure_garbage,
        scalar_names=tuple(scalar_names),
    )
    driver.regional_v841 = runtime
    return driver
