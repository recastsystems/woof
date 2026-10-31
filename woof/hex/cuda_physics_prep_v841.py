"""Resident CUDA preparation of MPAS-A v8.4.1 columns for physics.

This module is the MPAS-to-physics half of the column boundary.  It does not
run a parameterization and it does not couple returned tendencies.  It
reproduces the released preparation in ``MPAS_to_physics`` and the preceding
normal-wind reconstruction while keeping every forecast field resident.

The source authority is the official MPAS-Model v8.4.1 tag:

* ``mpas_atmphys_interface.F:301-355,427-555``;
* ``mpas_vector_reconstruction.F:195-320``.

All public CUDA arrays are packed FP32 ``cupy.ndarray`` objects.  Mass fields
use ``[level, cell]`` and interface fields use ``[interface, cell]``.  The six
WSM6 water carriers are copied through ``max(+0,q)`` scratch arrays; the
prognostic scalar storage is never changed.  Authority arithmetic is confined
to explicit RawKernels compiled through :class:`KernelCache`.  A successful
preparation performs exactly one final four-byte device-to-host validation.
"""

from __future__ import annotations
from woof.hex import numerics as _hex_numerics

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
import json
import math
import time
from types import MappingProxyType
from typing import Any

import numpy as np

from .species_row import WSM6_SPECIES_ROW, species_row_for_names
from .cuda_backend.containers import TransferStats, require_resident_array
from .cuda_backend.runtime import KernelCache
from .cuda_fp32 import CUDA_FTZ_HELPERS


CUDA_PHYSICS_PREP_V841_SCHEMA = "mpas-port.cuda-physics-prep-v841/v1"
MPAS_ATMPHYS_INTERFACE_V841_SHA256 = (
    "165b21ecfb4e599ce512a46ebc07a38d24abe4f76e4bf47388d15706fbc49ff2"
)
MPAS_VECTOR_RECONSTRUCTION_V841_SHA256 = (
    "5f4b7b2819e06acc82202aeeac7e40f4470e2e07e0da939cb23b11327662175f"
)
MPAS_NOAHMP_SOUNDING_V841_SHA256 = (
    "000a926306737d78e29f997ca972f87fe5d55bc500281f4379b8114112b11929"
)
#: The frozen lane's species, taken from the row rather than restated.  The
#: tuple is unchanged in content and order; what changed is that it now has
#: ONE definition in the tree (woof.hex.species_row) instead of three.
WSM6_SCALAR_NAMES = WSM6_SPECIES_ROW.names()
GRAVITY_F32 = np.float32(9.80616)
RD_F32 = np.float32(287.0)
RV_F32 = np.float32(461.6)
RV_OVER_RD_F32 = np.float32(RV_F32 / RD_F32)

def _authority_document(species_names: Sequence[str]) -> dict:
    """The preparation contract for ONE species row.

    The document is generated from the row rather than restating it, and the
    two species-bearing entries -- the scalar-input layout and the water
    scratch's names -- are the only places the row appears.  The WSM6 row
    regenerates today's document BYTE FOR BYTE, which is asserted below, so
    the frozen lane's contract digest does not move and its receipts are
    unchanged; a different row simply gets its own digest.
    """

    names = tuple(species_names)
    return {
    "schema": CUDA_PHYSICS_PREP_V841_SCHEMA,
    "mpas_version": "8.4.1",
    "layout": {
        "mass": "float32 C [level,cell]",
        "interface": "float32 C [interface,cell]",
        "scalar_input": (
            "float32 C [" + ",".join(names) + ",level,cell]"),
        "vertical_order": "surface_to_top",
    },
    "cuda_authority": {
        "module_key": "hexcore.cuda_physics_prep_v841",
        "compile_options": ["--std=c++17", "--fmad=false"],
        "compile_options_policy": "exact ordered base_options; no overrides",
        "implicit_cupy_authority_arithmetic": False,
        "successful_d2h_bytes": 4,
    },
    "geometry_integrity": {
        "mode": "private live buffers versus private sealed device mirrors",
        "check": "bitwise in reconstruct kernel before any geometry use",
        "recurring_d2h_bytes": 0,
        "upload_copies": 2,
    },
    "source": {
        "mpas_atmphys_interface.F": {
            "sha256": MPAS_ATMPHYS_INTERFACE_V841_SHA256,
            "lines": "301-355,427-555",
        },
        "mpas_vector_reconstruction.F": {
            "sha256": MPAS_VECTOR_RECONSTRUCTION_V841_SHA256,
            "lines": "195-320",
        },
        "mpas_atmphys_driver_lsm_noahmp.F": {
            "sha256": MPAS_NOAHMP_SOUNDING_V841_SHA256,
            "lines": "561-645",
        },
    },
    "pressure_families": {
        "eos_mass": "pres_p",
        "eos_interface": "pres2_p",
        "hydrostatic_moist_mass": "pres_hyd_p",
        "hydrostatic_moist_interface": "pres2_hyd_p",
        "hydrostatic_dry_mass": "pres_hydd_p",
        "hydrostatic_dry_interface": "pres2_hydd_p",
        "eos_surface": "psfc_p",
        "hydrostatic_moist_surface": "psfc_hyd_p",
        "hydrostatic_dry_surface": "psfc_hydd_p",
    },
    "water_scratch": {
        "names": list(names),
        "operation": "bitwise max(+0,q), preserving positive subnormals",
        "mutates_prognostic_input": False,
    },
    "wind_reconstruction": {
        "accumulation": "cell, then edge slot sequentially, then level",
        "rotation": "released spherical zonal/meridional equations",
    },
    "noahmp_sounding": {
        "qv": "raw prognostic qv; never common max(+0,qv) scratch",
        "fields": ["dz8w", "qv_curr", "t_phy", "u_phy", "v_phy", "p8w"],
    },
    "post_rk_wsm6_view": {
        "source": "mpas_atmphys_interface.F:637-672",
        "density": "rho_p=zz*rho_zz (dry, not phase-one moist rho_p)",
        "requires_candidate_scalar_clamp": True,
        "negative_test": "reject any negative nonzero FP32 bit pattern",
    },
}


def _contract_sha256(species_names: Sequence[str]) -> str:
    """This preparation contract's digest for one row."""

    return sha256(
        json.dumps(
            _authority_document(species_names),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


_AUTHORITY_DOCUMENT = _authority_document(WSM6_SCALAR_NAMES)

CUDA_PHYSICS_PREP_V841_CONTRACT_SHA256 = _contract_sha256(WSM6_SCALAR_NAMES)

#: The digest this contract carried before the row generalization, measured
#: at 6652552.  The generator is faithful or this module refuses to import:
#: WSM6's document has not moved a byte, so its receipts and every pin that
#: quotes them are unchanged.
_FROZEN_WSM6_CONTRACT_SHA256 = (
    "a9436205305c4127dc38399288ad47839dc797d0c786e1118e9e800cc7223c1d"
)
if CUDA_PHYSICS_PREP_V841_CONTRACT_SHA256 != _FROZEN_WSM6_CONTRACT_SHA256:
    raise AssertionError(
        "the species-row generator no longer reproduces the frozen WSM6 "
        "preparation contract: "
        f"{CUDA_PHYSICS_PREP_V841_CONTRACT_SHA256} != "
        f"{_FROZEN_WSM6_CONTRACT_SHA256}"
    )
CUDA_PHYSICS_PREP_V841_EVIDENCE: Mapping[str, Any] = MappingProxyType(
    _AUTHORITY_DOCUMENT
)


def physics_prep_contract_evidence_v841() -> dict[str, Any]:
    """Return a detached JSON-ready copy of the frozen preparation contract."""

    result = json.loads(json.dumps(_AUTHORITY_DOCUMENT, sort_keys=True))
    result["contract_sha256"] = CUDA_PHYSICS_PREP_V841_CONTRACT_SHA256
    result["kernel_sha256"] = CUDA_PHYSICS_PREP_V841_KERNEL_SHA256
    return result


def _cp() -> Any:
    import cupy as cp

    return cp


def _mesh_value(mesh: object, name: str, default: Any = None) -> Any:
    if hasattr(mesh, name):
        return getattr(mesh, name)
    if isinstance(mesh, Mapping) and name in mesh:
        return mesh[name]
    arrays = getattr(mesh, "arrays", None)
    if isinstance(arrays, Mapping) and name in arrays:
        return arrays[name]
    attrs = getattr(mesh, "attrs", None)
    if isinstance(attrs, Mapping) and name in attrs:
        return attrs[name]
    return default


def _regional_donor_for_extent(mesh: object, garbage_cell: int | None) -> Any:
    """Ordering provenance describes a donor only on a padded regional view."""
    if garbage_cell is None:
        return None
    return _mesh_value(mesh, "regional_garbage_donor_cell")


def _array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _canonical_edges_on_cell(
    value: Any, *, n_cells: int, max_edges: int
) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape == (n_cells, max_edges):
        result = raw
    elif raw.shape == (max_edges, n_cells):
        result = raw.T
    else:
        raise ValueError(
            "edgesOnCell must have shape "
            f"{(n_cells, max_edges)} (cell,slot), got {raw.shape}"
        )
    if result.dtype.kind not in "iu":
        raise TypeError("edgesOnCell must be integer")
    info = np.iinfo(np.int32)
    if result.size and (
        int(np.min(result)) < info.min or int(np.max(result)) > info.max
    ):
        raise OverflowError("edgesOnCell cannot be represented by int32")
    return np.ascontiguousarray(result, dtype=np.int32)


def _canonical_coefficients(
    value: Any, *, n_cells: int, max_edges: int
) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape == (n_cells, max_edges, 3):
        result = raw
    elif raw.shape == (3, max_edges, n_cells):
        result = raw.transpose(2, 1, 0)
    elif raw.shape == (n_cells, 3, max_edges):
        result = raw.transpose(0, 2, 1)
    else:
        raise ValueError(
            "coeffs_reconstruct must have shape "
            f"{(n_cells, max_edges, 3)}, got {raw.shape}"
        )
    result = np.ascontiguousarray(result, dtype=np.float32)
    if not np.all(np.isfinite(result)):
        raise ValueError("coeffs_reconstruct contains non-finite values")
    return result


_GEOMETRY_SEAL = object()


@dataclass(frozen=True, slots=True)
class CudaMpasToPhysGeometryV841:
    """One-time, source-checked upload of reconstruction geometry.

    The constructor is intentionally sealed.  ``from_host`` validates the
    zero-based active topology and canonical padding before any CUDA kernel can
    index it, records hashes of the uploaded canonical buffers, and then seals
    the immutable geometry object.
    """

    _edges_on_cell: Any = field(repr=False, compare=False)
    _n_edges_on_cell: Any = field(repr=False, compare=False)
    _coeffs_reconstruct: Any = field(repr=False, compare=False)
    _lat_cell: Any = field(repr=False, compare=False)
    _lon_cell: Any = field(repr=False, compare=False)
    _sealed_edges_on_cell: Any = field(repr=False, compare=False)
    _sealed_n_edges_on_cell: Any = field(repr=False, compare=False)
    _sealed_coeffs_reconstruct: Any = field(repr=False, compare=False)
    _sealed_lat_cell: Any = field(repr=False, compare=False)
    _sealed_lon_cell: Any = field(repr=False, compare=False)
    n_cells: int
    n_edges: int
    max_edges: int
    sealed_device_bytes: int
    h2d: TransferStats
    array_sha256: Mapping[str, str]
    _seal: object = field(repr=False, compare=False)
    #: Native's garbage element on a limited-area mesh -- the LAST column of
    #: the padded extent, the one an MPAS allocation always carries and never
    #: solves.  ``None`` on a closed sphere, where every column is solved.
    #: The mesh view declares it (``PaddedRegionalHostMesh.garbage_cell``);
    #: nothing else has to know, which is what keeps the preparation one
    #: routine instead of two.
    regional_garbage_cell: int | None = None
    # Partition-local index of the same real column the one-card view lends.
    # Only byte selection changes; every preparation kernel is unchanged.
    regional_garbage_donor_cell: int | None = None
    #: maxEdges of the host geometry.  ``max_edges`` is the slot stride of the
    #: device copies, which keep only the slots ``slot < nEdgesOnCell`` loops
    #: reach (``cuda_backend/compaction.py``).  0 means "not compacted".
    source_max_edges: int = 0
    edges_on_cell: Any = field(init=False, repr=False, compare=False)
    n_edges_on_cell: Any = field(init=False, repr=False, compare=False)
    coeffs_reconstruct: Any = field(init=False, repr=False, compare=False)
    lat_cell: Any = field(init=False, repr=False, compare=False)
    lon_cell: Any = field(init=False, repr=False, compare=False)

    @property
    def contract_sha256(self) -> str:
        return CUDA_PHYSICS_PREP_V841_CONTRACT_SHA256

    def __post_init__(self) -> None:
        for name in (
            "edges_on_cell",
            "n_edges_on_cell",
            "coeffs_reconstruct",
            "lat_cell",
            "lon_cell",
        ):
            object.__setattr__(self, name, getattr(self, f"_{name}"))

    @classmethod
    def from_host(cls, mesh: object) -> "CudaMpasToPhysGeometryV841":
        """Validate and upload the exact spherical reconstruction carriers."""

        lat = np.asarray(_mesh_value(mesh, "latCell"))
        lon = np.asarray(_mesh_value(mesh, "lonCell"))
        if lat.ndim != 1 or lon.shape != lat.shape or lat.size == 0:
            raise ValueError("latCell/lonCell must be matching non-empty vectors")
        n_cells = int(lat.size)
        counts_raw = np.asarray(_mesh_value(mesh, "nEdgesOnCell"))
        if counts_raw.shape != (n_cells,) or counts_raw.dtype.kind not in "iu":
            raise ValueError("nEdgesOnCell must be an integer nCells vector")
        counts64 = np.asarray(counts_raw, dtype=np.int64)

        raw_edges = np.asarray(_mesh_value(mesh, "edgesOnCell"))
        if raw_edges.ndim != 2 or n_cells not in raw_edges.shape:
            raise ValueError("edgesOnCell geometry is missing or has invalid rank")
        max_edges = int(
            raw_edges.shape[1] if raw_edges.shape[0] == n_cells else raw_edges.shape[0]
        )
        if max_edges <= 0 or np.any(counts64 < 0) or np.any(counts64 > max_edges):
            raise ValueError("nEdgesOnCell lies outside the stored edge-slot extent")
        counts = np.ascontiguousarray(counts64, dtype=np.int32)
        edges = _canonical_edges_on_cell(
            raw_edges, n_cells=n_cells, max_edges=max_edges
        )

        raw_cells_on_edge = np.asarray(_mesh_value(mesh, "cellsOnEdge"))
        if raw_cells_on_edge.ndim != 2 or 2 not in raw_cells_on_edge.shape:
            raise ValueError("cellsOnEdge is required to bind nEdges")
        n_edges = int(
            raw_cells_on_edge.shape[0]
            if raw_cells_on_edge.shape[1] == 2
            else raw_cells_on_edge.shape[1]
        )
        if n_edges <= 0:
            raise ValueError("reconstruction geometry must contain edges")
        slots = np.arange(max_edges, dtype=np.int64)[None, :]
        active = slots < counts64[:, None]
        if np.any(edges[active] < 0) or np.any(edges[active] >= n_edges):
            raise ValueError("active edgesOnCell entries must be zero-based in range")
        if np.any(edges[~active] != -1):
            raise ValueError("edgesOnCell padding must be canonical -1")

        coeffs = _canonical_coefficients(
            _mesh_value(mesh, "coeffs_reconstruct"),
            n_cells=n_cells,
            max_edges=max_edges,
        )
        if not np.any(coeffs[active]):
            raise ValueError("active coeffs_reconstruct entries are all zero")
        if np.any(coeffs[~active] != np.float32(0.0)):
            raise ValueError("coeffs_reconstruct padding must be bitwise zero")

        lat32 = np.ascontiguousarray(lat, dtype=np.float32)
        lon32 = np.ascontiguousarray(lon, dtype=np.float32)
        if not np.all(np.isfinite(lat32)) or not np.all(np.isfinite(lon32)):
            raise ValueError("latCell/lonCell contains non-finite values")
        if np.any(np.abs(lat32) > np.float32(np.pi / 2.0 + 1.0e-5)):
            raise ValueError("latCell must be in radians")
        on_sphere = _mesh_value(mesh, "on_a_sphere", True)
        if isinstance(on_sphere, str):
            on_sphere = on_sphere.strip().lower() in {"true", "t", "yes", "1"}
        if not bool(on_sphere):
            raise ValueError("the v8.4.1 CUDA physics seam requires spherical MPAS")

        canonical = {
            "edges_on_cell": edges,
            "n_edges_on_cell": counts,
            "coeffs_reconstruct": coeffs,
            "lat_cell": lat32,
            "lon_cell": lon32,
        }
        hashes = MappingProxyType(
            {name: _array_sha256(value) for name, value in canonical.items()}
        )
        # The hashes above identify the host canonical layout.  The device
        # copies keep the slots the reconstruction loops can reach: padding
        # was just proved canonical (-1 and +0), so nothing read changes.
        source_max_edges = 0
        from .cuda_backend.compaction import compaction_enabled, used_width

        if compaction_enabled("PREP_GEOMETRY"):
            keep = used_width(counts, max_edges)
            if keep < max_edges:
                canonical["edges_on_cell"] = np.ascontiguousarray(edges[:, :keep])
                canonical["coeffs_reconstruct"] = np.ascontiguousarray(
                    coeffs[:, :keep, :]
                )
                source_max_edges = max_edges
                max_edges = keep
        garbage_value = _mesh_value(mesh, "garbage_cell")
        garbage_cell = None if garbage_value is None else int(garbage_value)
        if garbage_cell is not None and garbage_cell != n_cells - 1:
            raise ValueError(
                "a limited-area mesh's garbage element is the LAST column of "
                f"its padded extent; this view declares {garbage_cell} with "
                f"{n_cells} columns"
            )
        cp = _cp()
        started = time.perf_counter()
        live = {
            f"_{name}": cp.asarray(value) for name, value in canonical.items()
        }
        sealed = {
            f"_sealed_{name}": cp.asarray(value)
            for name, value in canonical.items()
        }
        cp.cuda.get_current_stream().synchronize()
        elapsed = time.perf_counter() - started
        sealed_bytes = int(sum(int(value.nbytes) for value in sealed.values()))
        result = cls(
            **live,
            **sealed,
            n_cells=n_cells,
            n_edges=n_edges,
            max_edges=max_edges,
            sealed_device_bytes=sealed_bytes,
            h2d=TransferStats(
                int(sum(int(value.nbytes) for value in (*live.values(), *sealed.values()))),
                elapsed,
            ),
            array_sha256=hashes,
            _seal=_GEOMETRY_SEAL,
            regional_garbage_cell=garbage_cell,
            regional_garbage_donor_cell=_regional_donor_for_extent(mesh, garbage_cell),
            source_max_edges=source_max_edges,
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self._seal is not _GEOMETRY_SEAL:
            raise TypeError("geometry must be constructed by from_host")
        if self.n_cells <= 0 or self.n_edges <= 0 or self.max_edges <= 0:
            raise ValueError("geometry dimensions must be positive")
        if self.regional_garbage_cell is not None and (
            self.regional_garbage_cell != self.n_cells - 1
        ):
            raise ValueError(
                "the regional garbage column must be the last of the extent"
            )
        if self.regional_garbage_donor_cell is not None and (
            self.regional_garbage_cell is None
            or not 0 <= int(self.regional_garbage_donor_cell) < self.regional_garbage_cell
        ):
            raise ValueError("the regional garbage donor must be a real local column")
        shapes = {
            "edges_on_cell": (self.n_cells, self.max_edges),
            "n_edges_on_cell": (self.n_cells,),
            "coeffs_reconstruct": (self.n_cells, self.max_edges, 3),
            "lat_cell": (self.n_cells,),
            "lon_cell": (self.n_cells,),
        }
        dtypes = {
            "edges_on_cell": np.int32,
            "n_edges_on_cell": np.int32,
            "coeffs_reconstruct": np.float32,
            "lat_cell": np.float32,
            "lon_cell": np.float32,
        }
        sealed_bytes = 0
        for name in shapes:
            live = getattr(self, f"_{name}")
            sealed = getattr(self, f"_sealed_{name}")
            require_resident_array(
                f"physics_prep_geometry.live.{name}",
                live,
                dtype=dtypes[name],
                shape=shapes[name],
            )
            require_resident_array(
                f"physics_prep_geometry.sealed.{name}",
                sealed,
                dtype=dtypes[name],
                shape=shapes[name],
            )
            if int(live.data.ptr) == int(sealed.data.ptr):
                raise ValueError("live and sealed geometry buffers must not alias")
            sealed_bytes += int(sealed.nbytes)
        if self.sealed_device_bytes != sealed_bytes:
            raise ValueError("sealed geometry byte inventory changed")
        if self.h2d.bytes != 2 * sealed_bytes:
            raise ValueError("geometry upload must contain live and sealed copies")
        expected = set(shapes)
        if set(self.array_sha256) != expected or any(
            len(str(value)) != 64 for value in self.array_sha256.values()
        ):
            raise ValueError("geometry array hash inventory is incomplete")

    def receipt(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": CUDA_PHYSICS_PREP_V841_SCHEMA,
            "contract_sha256": self.contract_sha256,
            "regional_garbage_donor_cell": self.regional_garbage_donor_cell,
            "shape": {
                "nCells": self.n_cells,
                "nEdges": self.n_edges,
                "maxEdges": self.source_max_edges or self.max_edges,
                "coeffs_reconstruct": [
                    self.n_cells, self.source_max_edges or self.max_edges, 3
                ],
            },
            "device_slot_stride": self.max_edges,
            "layout": "zero-based edgesOnCell with canonical -1 padding",
            "edge_slot_order": "sequential source order",
            "array_sha256": dict(self.array_sha256),
            "h2d": self.h2d.as_dict(),
            "integrity": {
                "mode": "private-live-versus-private-sealed-device-mirror",
                "comparison": "bitwise-before-reconstruction-use",
                "sealed_device_bytes": self.sealed_device_bytes,
                "upload_copies": 2,
                "recurring_d2h_bytes": 0,
            },
            "authority_sha256": MPAS_VECTOR_RECONSTRUCTION_V841_SHA256,
        }


#: The mass-level outputs that are NOT species.  The species outputs are
#: named by the scheme's row and prepended by ``_mass_fields``, so this
#: tuple no longer grows when a scheme with more scalars arrives -- which
#: is what "arbitrary" has to mean at this layer.
_MASS_NONSPECIES_FIELDS = (
    "u_p",
    "v_p",
    "zz_p",
    "rho_dry",
    "rho_p",
    "th_p",
    "t_p",
    "pi_p",
    "pres_p",
    "zmid_p",
    "dz_p",
    "pres_hyd_p",
    "pres_hydd_p",
    "znu_p",
    "znu_hyd_p",
)


def _device_pointer_table(cp: Any, arrays: Sequence[Any]) -> Any:
    """A resident table of device pointers, one per array, in order.

    This is what replaces the six welded kernel parameters.  The arrays must
    stay alive for the launch; every caller here holds them in ``output``
    for the whole body, so the table can never outlive its targets.
    """

    if not arrays:
        raise ValueError("a pointer table needs at least one array")
    return cp.asarray(
        np.asarray([int(item.data.ptr) for item in arrays], dtype=np.uint64)
    )


def _mass_fields(species_names: Sequence[str]) -> tuple[str, ...]:
    """Every mass-level output name for a row: species first, then the 15."""

    return tuple(f"{name}_p" for name in species_names) + _MASS_NONSPECIES_FIELDS


#: The frozen lane's list, unchanged in content and order from the constant
#: this replaced, so a reader diffing the two sees no reordering.
_MASS_FIELDS = _mass_fields(WSM6_SCALAR_NAMES)
_INTERFACE_FIELDS = (
    "w_p",
    "z_p",
    "t2_p",
    "pres2_p",
    "pres2_hyd_p",
    "pres2_hydd_p",
)
_SURFACE_FIELDS = ("psfc_p", "psfc_hyd_p", "psfc_hydd_p", "plrad")


@dataclass(frozen=True, slots=True)
class CudaNoahmpSoundingV841:
    """Separate released NoahMP sounding; raw qv is part of its identity."""

    dz8w: Any
    qv_curr: Any
    t_phy: Any
    u_phy: Any
    v_phy: Any
    p8w: Any

    def validate(self, *, n_vert_levels: int, n_cells: int) -> None:
        shape = (n_vert_levels, n_cells)
        for name in ("dz8w", "qv_curr", "t_phy", "u_phy", "v_phy", "p8w"):
            require_resident_array(
                f"noahmp_sounding.{name}",
                getattr(self, name),
                dtype=np.float32,
                shape=shape,
            )

    def receipt(self) -> dict[str, Any]:
        return {
            "schema": CUDA_PHYSICS_PREP_V841_SCHEMA,
            "contract_sha256": CUDA_PHYSICS_PREP_V841_CONTRACT_SHA256,
            "fields": ["dz8w", "qv_curr", "t_phy", "u_phy", "v_phy", "p8w"],
            "qv_semantics": "raw-unclamped-prognostic-qv",
            "authority": {
                "file": "mpas_atmphys_driver_lsm_noahmp.F",
                "lines": "561-645",
                "sha256": MPAS_NOAHMP_SOUNDING_V841_SHA256,
            },
        }


@dataclass(frozen=True, slots=True)
class CudaWsm6InputViewV841:
    """Dry post-RK WSM6 view, valid only after candidate scalar clamping."""

    rho_dry: Any
    theta_dry: Any
    exner: Any
    p_eos: Any
    dz: Any
    z: Any
    w: Any
    #: Per-species dry-preparation aliases, keyed by the row's names.
    species: Mapping[str, Any]


    def __getattr__(self, name: str) -> Any:
        """A species name still resolves as an attribute."""

        species = object.__getattribute__(self, "species")
        value = species.get(name)
        if value is not None:
            return value
        raise AttributeError(
            f"{type(self).__name__!r} has no attribute {name!r}")
    def validate(self) -> None:
        shape = tuple(self.rho_dry.shape)
        if len(shape) != 2 or shape[0] < 2 or shape[1] <= 0:
            raise ValueError("WSM6 view must be [level,cell]")
        for name in (
            "rho_dry", "theta_dry", "exner", "p_eos", "dz", "z", "w",
            *self.species,
        ):
            require_resident_array(
                f"wsm6_input.{name}",
                getattr(self, name),
                dtype=np.float32,
                shape=shape,
            )

    def receipt(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": CUDA_PHYSICS_PREP_V841_SCHEMA,
            "contract_sha256": CUDA_PHYSICS_PREP_V841_CONTRACT_SHA256,
            "shape": list(self.rho_dry.shape),
            "rho_semantics": "dry metric-removed rho_p=zz*rho_zz",
            "candidate_scalars_clamped": True,
            "authority": "mpas_atmphys_interface.F:637-672",
        }


@dataclass(frozen=True, slots=True)
class CudaMpasToPhysColumnsV841:
    """Resident output using the exact native MPAS physics field names."""

    #: Per-species clamped scratch, keyed by the row's names IN ROW ORDER.
    #: This replaced six literal ``q*_p`` fields; ``__getattr__`` keeps the
    #: old spelling working for every existing reader.
    species_p: Mapping[str, Any]
    u_p: Any
    v_p: Any
    zz_p: Any
    rho_dry: Any
    rho_p: Any
    th_p: Any
    t_p: Any
    pi_p: Any
    pres_p: Any
    zmid_p: Any
    dz_p: Any
    pres_hyd_p: Any
    pres_hydd_p: Any
    znu_p: Any
    znu_hyd_p: Any
    w_p: Any
    z_p: Any
    t2_p: Any
    pres2_p: Any
    pres2_hyd_p: Any
    pres2_hydd_p: Any
    psfc_p: Any
    psfc_hyd_p: Any
    psfc_hydd_p: Any
    plrad: Any
    noahmp_sounding: CudaNoahmpSoundingV841 | None
    wsm6_ready: bool
    _source_scalars: Any = field(repr=False, compare=False)
    time_seconds: float
    validation_d2h: TransferStats
    geometry_receipt: Mapping[str, Any]

    def __getattr__(self, name: str) -> Any:
        """``<species>_p`` still resolves, whatever the row declares.

        The six welded fields became one ordered mapping, and every existing
        reader -- the GWDO adapter's ``prepared.qv_p``, the sealed adapter's
        six -- keeps working without knowing that.  A name the row does not
        declare raises AttributeError, so a typo is still a typo.
        """

        if name.endswith("_p"):
            species = object.__getattribute__(self, "species_p")
            value = species.get(name[:-2])
            if value is not None:
                return value
        raise AttributeError(
            f"{type(self).__name__!r} has no attribute {name!r}")

    @property
    def contract_sha256(self) -> str:
        return CUDA_PHYSICS_PREP_V841_CONTRACT_SHA256

    # Semantic aliases deliberately return the exact native carriers rather
    # than allocating substitute pressure arrays.
    @property
    def p_eos(self) -> Any:
        return self.pres_p

    @property
    def p2_eos(self) -> Any:
        return self.pres2_p

    @property
    def p_hyd(self) -> Any:
        return self.pres_hyd_p

    @property
    def p2_hyd(self) -> Any:
        return self.pres2_hyd_p

    @property
    def p_hyd_dry(self) -> Any:
        return self.pres_hydd_p

    @property
    def p2_hyd_dry(self) -> Any:
        return self.pres2_hydd_p

    @property
    def psfc_eos(self) -> Any:
        return self.psfc_p

    @property
    def psfc_hyd(self) -> Any:
        return self.psfc_hyd_p

    @property
    def psfc_hyd_dry(self) -> Any:
        return self.psfc_hydd_p

    @property
    def znu_eos(self) -> Any:
        return self.znu_p

    @property
    def znu_hyd(self) -> Any:
        return self.znu_hyd_p

    @property
    def scalar_scratch(self) -> Mapping[str, Any]:
        return MappingProxyType(
            dict(self.species_p)
        )

    def wsm6_input_view(self) -> CudaWsm6InputViewV841:
        """Return aliases to dry preparation fields after a proven raw clamp."""

        if not self.wsm6_ready:
            raise ValueError(
                "WSM6 input view requires post_rk_wsm6=True after scalar clamp"
            )
        result = CudaWsm6InputViewV841(
            rho_dry=self.rho_dry,
            theta_dry=self.th_p,
            exner=self.pi_p,
            p_eos=self.pres_p,
            dz=self.dz_p,
            z=self.z_p[:-1],
            w=self.w_p[:-1],
            species={
                name: self._source_scalars[index]
                for index, name in enumerate(self.species_p)
            },
        )
        result.validate()
        return result

    @property
    def n_vert_levels(self) -> int:
        return int(self.rho_p.shape[0])

    @property
    def n_cells(self) -> int:
        return int(self.rho_p.shape[1])

    def validate(self) -> None:
        if not math.isfinite(float(self.time_seconds)) or self.time_seconds < 0.0:
            raise ValueError("time_seconds must be finite and non-negative")
        nlev, ncells = tuple(self.rho_p.shape)
        if nlev < 2 or ncells <= 0:
            raise ValueError("prepared physics columns need at least two levels")
        if (self.u_p is None) != (self.v_p is None):
            raise ValueError("prepared horizontal winds must be present together")
        if not self.wsm6_ready and (self.u_p is None or self.noahmp_sounding is None):
            raise ValueError("phase-one preparation requires winds and NoahMP sounding")
        if self.noahmp_sounding is not None and self.u_p is None:
            raise ValueError("NoahMP sounding requires horizontal winds")
        for name in _mass_fields(self.species_p):
            if name in ("u_p", "v_p") and getattr(self, name) is None:
                continue
            require_resident_array(
                f"physics_prep.{name}",
                getattr(self, name),
                dtype=np.float32,
                shape=(nlev, ncells),
            )
        for name in _INTERFACE_FIELDS:
            require_resident_array(
                f"physics_prep.{name}",
                getattr(self, name),
                dtype=np.float32,
                shape=(nlev + 1, ncells),
            )
        for name in _SURFACE_FIELDS:
            require_resident_array(
                f"physics_prep.{name}",
                getattr(self, name),
                dtype=np.float32,
                shape=(ncells,),
            )
        require_resident_array(
            "physics_prep.source_scalars",
            self._source_scalars,
            dtype=np.float32,
            shape=(len(self.species_p), nlev, ncells),
        )
        if self.noahmp_sounding is not None:
            self.noahmp_sounding.validate(n_vert_levels=nlev, n_cells=ncells)
        if not isinstance(self.wsm6_ready, bool):
            raise TypeError("wsm6_ready must be bool")
        if self.wsm6_ready:
            self.wsm6_input_view().validate()
        if self.validation_d2h.bytes != 4:
            raise ValueError("physics preparation validation must transfer exactly 4 bytes")
        if self.geometry_receipt.get("contract_sha256") != self.contract_sha256:
            raise ValueError("prepared columns are not bound to their geometry contract")

    def receipt(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": CUDA_PHYSICS_PREP_V841_SCHEMA,
            "contract_sha256": self.contract_sha256,
            "kernel_sha256": CUDA_PHYSICS_PREP_V841_KERNEL_SHA256,
            "phase": "MPAS_to_physics",
            "time_seconds": float(self.time_seconds),
            "shape": {
                "mass": [self.n_vert_levels, self.n_cells],
                "interface": [self.n_vert_levels + 1, self.n_cells],
            },
            "pressure_field_identity": dict(
                _AUTHORITY_DOCUMENT["pressure_families"]
            ),
            "validation_d2h": self.validation_d2h.as_dict(),
            "noahmp_sounding": (
                None if self.noahmp_sounding is None else self.noahmp_sounding.receipt()),
            "horizontal_winds_prepared": self.u_p is not None,
            "post_rk_wsm6": (
                self.wsm6_input_view().receipt() if self.wsm6_ready else None
            ),
            "geometry": dict(self.geometry_receipt),
            "authority": json.loads(json.dumps(_AUTHORITY_DOCUMENT["source"])),
        }


@dataclass(frozen=True, slots=True)
class CpuNoahmpSoundingV841:
    """Readable NoahMP sounding with raw, explicitly unclamped qv."""

    dz8w: np.ndarray
    qv_curr: np.ndarray
    t_phy: np.ndarray
    u_phy: np.ndarray
    v_phy: np.ndarray
    p8w: np.ndarray

    def validate(self) -> None:
        shape = self.qv_curr.shape
        if len(shape) != 2 or shape[0] < 2 or shape[1] <= 0:
            raise ValueError("NoahMP sounding must be [level,cell]")
        for name in ("dz8w", "qv_curr", "t_phy", "u_phy", "v_phy", "p8w"):
            value = getattr(self, name)
            if value.shape != shape or value.dtype != np.float32:
                raise ValueError(f"NoahMP {name} must be FP32 {shape}")
            if not value.flags.c_contiguous or not np.all(np.isfinite(value)):
                raise ValueError(f"NoahMP {name} must be finite and C-contiguous")
        if np.any(self.dz8w <= np.float32(0.0)) or np.any(
            self.t_phy <= np.float32(0.0)
        ):
            raise ValueError("NoahMP sounding thickness/temperature must be positive")
        if np.any(self.p8w <= np.float32(0.0)) or np.any(
            np.diff(self.p8w, axis=0) >= np.float32(0.0)
        ):
            raise ValueError("NoahMP p8w must decrease from surface to top")

    def validate_against_source(self, **source: Any) -> None:
        expected = prepare_noahmp_sounding_cpu_oracle_v841(**source)
        for name in ("dz8w", "qv_curr", "t_phy", "u_phy", "v_phy", "p8w"):
            if not np.array_equal(
                getattr(self, name).view(np.uint32),
                getattr(expected, name).view(np.uint32),
            ):
                raise ValueError(
                    f"NoahMP {name} no longer matches the raw-qv sounding source"
                )


@dataclass(frozen=True, slots=True)
class CpuWsm6InputViewV841:
    """CPU oracle for the dry post-RK microphysics preparation view."""

    rho_dry: np.ndarray
    theta_dry: np.ndarray
    exner: np.ndarray
    p_eos: np.ndarray
    dz: np.ndarray
    z: np.ndarray
    w: np.ndarray
    species: Mapping[str, np.ndarray]


    def __getattr__(self, name: str) -> Any:
        """A species name still resolves as an attribute."""

        species = object.__getattribute__(self, "species")
        value = species.get(name)
        if value is not None:
            return value
        raise AttributeError(
            f"{type(self).__name__!r} has no attribute {name!r}")
    def validate(self) -> None:
        shape = self.rho_dry.shape
        for name in (
            "rho_dry", "theta_dry", "exner", "p_eos", "dz", "z", "w",
            *self.species,
        ):
            value = getattr(self, name)
            if value.shape != shape or value.dtype != np.float32:
                raise ValueError(f"WSM6 {name} must be FP32 {shape}")
            if not value.flags.c_contiguous or not np.all(np.isfinite(value)):
                raise ValueError(f"WSM6 {name} must be finite and C-contiguous")
        for name in self.species:
            bits = getattr(self, name).view(np.uint32)
            negative_nonzero = (
                ((bits & np.uint32(0x80000000)) != 0)
                & ((bits & np.uint32(0x7FFFFFFF)) != 0)
            )
            if np.any(negative_nonzero):
                raise ValueError("WSM6 view requires candidate scalar clamp")
        if np.any(self.rho_dry <= 0.0) or np.any(self.theta_dry <= 0.0):
            raise ValueError("WSM6 dry density/theta must be positive")


@dataclass(frozen=True, slots=True)
class CpuMpasToPhysColumnsV841:
    """Binary32 source-loop oracle with the same native field inventory."""

    species_p: Mapping[str, np.ndarray]
    u_p: np.ndarray
    v_p: np.ndarray
    zz_p: np.ndarray
    rho_dry: np.ndarray
    rho_p: np.ndarray
    th_p: np.ndarray
    t_p: np.ndarray
    pi_p: np.ndarray
    pres_p: np.ndarray
    zmid_p: np.ndarray
    dz_p: np.ndarray
    pres_hyd_p: np.ndarray
    pres_hydd_p: np.ndarray
    znu_p: np.ndarray
    znu_hyd_p: np.ndarray
    w_p: np.ndarray
    z_p: np.ndarray
    t2_p: np.ndarray
    pres2_p: np.ndarray
    pres2_hyd_p: np.ndarray
    pres2_hydd_p: np.ndarray
    psfc_p: np.ndarray
    psfc_hyd_p: np.ndarray
    psfc_hydd_p: np.ndarray
    plrad: np.ndarray
    noahmp_sounding: CpuNoahmpSoundingV841
    _source_scalars: np.ndarray = field(repr=False, compare=False)

    def __getattr__(self, name: str) -> Any:
        """``<species>_p`` still resolves, whatever the row declares.

        The six welded fields became one ordered mapping, and every existing
        reader -- the GWDO adapter's ``prepared.qv_p``, the sealed adapter's
        six -- keeps working without knowing that.  A name the row does not
        declare raises AttributeError, so a typo is still a typo.
        """

        if name.endswith("_p"):
            species = object.__getattribute__(self, "species_p")
            value = species.get(name[:-2])
            if value is not None:
                return value
        raise AttributeError(
            f"{type(self).__name__!r} has no attribute {name!r}")

    @property
    def p_eos(self) -> np.ndarray:
        return self.pres_p

    @property
    def p2_eos(self) -> np.ndarray:
        return self.pres2_p

    @property
    def p_hyd(self) -> np.ndarray:
        return self.pres_hyd_p

    @property
    def p2_hyd(self) -> np.ndarray:
        return self.pres2_hyd_p

    @property
    def p_hyd_dry(self) -> np.ndarray:
        return self.pres_hydd_p

    @property
    def p2_hyd_dry(self) -> np.ndarray:
        return self.pres2_hydd_p

    @property
    def psfc_eos(self) -> np.ndarray:
        return self.psfc_p

    @property
    def psfc_hyd(self) -> np.ndarray:
        return self.psfc_hyd_p

    @property
    def psfc_hyd_dry(self) -> np.ndarray:
        return self.psfc_hydd_p

    @property
    def znu_eos(self) -> np.ndarray:
        return self.znu_p

    @property
    def znu_hyd(self) -> np.ndarray:
        return self.znu_hyd_p

    @property
    def scalar_scratch(self) -> Mapping[str, np.ndarray]:
        return MappingProxyType(
            dict(self.species_p)
        )

    def wsm6_input_view(self) -> CpuWsm6InputViewV841:
        result = CpuWsm6InputViewV841(
            rho_dry=self.rho_dry,
            theta_dry=self.th_p,
            exner=self.pi_p,
            p_eos=self.pres_p,
            dz=self.dz_p,
            z=np.ascontiguousarray(self.z_p[:-1]),
            w=np.ascontiguousarray(self.w_p[:-1]),
            **{
                name: np.ascontiguousarray(self._source_scalars[index])
                for index, name in enumerate(self.species_p)
            },
        )
        result.validate()
        return result

    def validate(self) -> None:
        nlev, ncells = self.rho_p.shape
        for name in _mass_fields(self.species_p):
            value = getattr(self, name)
            if value.shape != (nlev, ncells) or value.dtype != np.float32:
                raise ValueError(f"{name} is not FP32 [level,cell]")
            if not value.flags.c_contiguous or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be finite and C-contiguous")
        for name in _INTERFACE_FIELDS:
            value = getattr(self, name)
            if value.shape != (nlev + 1, ncells) or value.dtype != np.float32:
                raise ValueError(f"{name} is not FP32 [interface,cell]")
            if not value.flags.c_contiguous or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be finite and C-contiguous")
        for name in _SURFACE_FIELDS:
            value = getattr(self, name)
            if value.shape != (ncells,) or value.dtype != np.float32:
                raise ValueError(f"{name} is not an FP32 column vector")
            if not value.flags.c_contiguous or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be finite and C-contiguous")
        if self._source_scalars.shape != (
                len(self.species_p), nlev, ncells):
            raise ValueError(
                "source scalars must retain the declared row's layout")
        if self._source_scalars.dtype != np.float32 or not self._source_scalars.flags.c_contiguous:
            raise ValueError("source scalars must remain FP32 C-contiguous")
        self.noahmp_sounding.validate()
        for index, name in enumerate(self.species_p):
            value = getattr(self, f"{name}_p")
            source_bits = self._source_scalars[index].view(np.uint32)
            expected_bits = np.where(
                (source_bits & np.uint32(0x80000000)) == 0,
                source_bits,
                np.uint32(0),
            ).astype(np.uint32, copy=False)
            if not np.array_equal(value.view(np.uint32), expected_bits):
                raise ValueError("water scratch changed bitwise max(+0,q) semantics")
        if np.any(self.rho_p <= 0.0) or np.any(self.th_p <= 0.0):
            raise ValueError("prepared density and dry theta must be positive")
        for name in (
            "pres_p",
            "pres2_p",
            "pres_hyd_p",
            "pres2_hyd_p",
            "pres_hydd_p",
            "pres2_hydd_p",
        ):
            value = getattr(self, name)
            if np.any(value <= 0.0) or np.any(np.diff(value, axis=0) >= 0.0):
                raise ValueError(f"{name} must decrease from surface to top")
        if np.any(self.dz_p <= 0.0) or np.any(np.diff(self.z_p, axis=0) <= 0.0):
            raise ValueError("source vertical order must be surface_to_top")
        for cell in range(ncells):
            if (
                self.pres2_hyd_p[-1, cell].view(np.uint32)
                != self.pres2_p[-1, cell].view(np.uint32)
                or self.pres2_hydd_p[-1, cell].view(np.uint32)
                != self.pres2_p[-1, cell].view(np.uint32)
            ):
                raise ValueError("hydrostatic pressure families lost the EOS top anchor")
            for level in range(nlev - 1, -1, -1):
                moist_expected = _add(
                    self.pres2_hyd_p[level + 1, cell],
                    _mul(
                        _mul(GRAVITY_F32, self.rho_p[level, cell]),
                        self.dz_p[level, cell],
                    ),
                )
                rho_dry = _div(
                    self.rho_p[level, cell],
                    _add(np.float32(1.0), self.qv_p[level, cell]),
                )
                dry_expected = _add(
                    self.pres2_hydd_p[level + 1, cell],
                    _mul(_mul(GRAVITY_F32, rho_dry), self.dz_p[level, cell]),
                )
                moist_mass_expected = _mul(
                    np.float32(0.5),
                    _add(
                        self.pres2_hyd_p[level + 1, cell],
                        self.pres2_hyd_p[level, cell],
                    ),
                )
                dry_mass_expected = _mul(
                    np.float32(0.5),
                    _add(
                        self.pres2_hydd_p[level + 1, cell],
                        self.pres2_hydd_p[level, cell],
                    ),
                )
                actual_expected = (
                    (self.pres2_hyd_p[level, cell], moist_expected),
                    (self.pres2_hydd_p[level, cell], dry_expected),
                    (self.pres_hyd_p[level, cell], moist_mass_expected),
                    (self.pres_hydd_p[level, cell], dry_mass_expected),
                )
                if any(
                    actual.view(np.uint32) != expected.view(np.uint32)
                    for actual, expected in actual_expected
                ):
                    raise ValueError("hydrostatic pressure source recurrence changed")
                znu_expected = _div(
                    self.pres_hyd_p[level, cell], self.psfc_hyd_p[cell]
                )
                if (
                    self.znu_hyd_p[level, cell].view(np.uint32)
                    != znu_expected.view(np.uint32)
                ):
                    raise ValueError("znu_hyd_p no longer uses pres_hyd_p")
            if (
                self.psfc_hyd_p[cell].view(np.uint32)
                != self.pres2_hyd_p[0, cell].view(np.uint32)
                or self.psfc_hydd_p[cell].view(np.uint32)
                != self.pres2_hydd_p[0, cell].view(np.uint32)
                or self.plrad[cell].view(np.uint32)
                != self.pres2_p[-1, cell].view(np.uint32)
            ):
                raise ValueError("surface/model-top pressure identity changed")


def _f32(value: Any) -> np.float32:
    return np.float32(value)


def _add(left: Any, right: Any) -> np.float32:
    return _f32(_f32(left) + _f32(right))


def _sub(left: Any, right: Any) -> np.float32:
    return _f32(_f32(left) - _f32(right))


def _mul(left: Any, right: Any) -> np.float32:
    return _f32(_f32(left) * _f32(right))


def _div(left: Any, right: Any) -> np.float32:
    return _f32(_f32(left) / _f32(right))


def _positive_q(value: np.float32) -> np.float32:
    raw = np.float32(value)
    bits = raw.view(np.uint32)
    if (bits & np.uint32(0x80000000)) != 0:
        return np.float32(0.0)
    return raw


def _require_cpu_f32(name: str, value: Any, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.float32:
        raise TypeError(f"{name} must have dtype float32")
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not array.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def reconstruct_winds_cpu_oracle_v841(
    normal_velocity: Any,
    *,
    edges_on_cell: Any,
    n_edges_on_cell: Any,
    coeffs_reconstruct: Any,
    lat_cell: Any,
    lon_cell: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce reconstruction.F's sequential edge-slot accumulation."""

    velocity = np.asarray(normal_velocity)
    if velocity.dtype != np.float32 or velocity.ndim != 2:
        raise TypeError("normal_velocity must be FP32 [level,edge]")
    if not velocity.flags.c_contiguous or not np.all(np.isfinite(velocity)):
        raise ValueError("normal_velocity must be finite and C-contiguous")
    nlev, n_edges = velocity.shape
    lat = np.asarray(lat_cell)
    lon = np.asarray(lon_cell)
    if lat.dtype != np.float32 or lon.dtype != np.float32 or lat.shape != lon.shape:
        raise TypeError("latCell/lonCell must be matching FP32 vectors")
    n_cells = int(lat.size)
    counts = np.asarray(n_edges_on_cell)
    if counts.shape != (n_cells,) or counts.dtype.kind not in "iu":
        raise ValueError("nEdgesOnCell must be an integer nCells vector")
    raw_edges = np.asarray(edges_on_cell)
    if raw_edges.ndim != 2 or n_cells not in raw_edges.shape:
        raise ValueError("edgesOnCell reconstruction geometry is missing")
    max_edges = int(raw_edges.shape[1] if raw_edges.shape[0] == n_cells else raw_edges.shape[0])
    edges = _canonical_edges_on_cell(raw_edges, n_cells=n_cells, max_edges=max_edges)
    coeffs = _canonical_coefficients(
        coeffs_reconstruct, n_cells=n_cells, max_edges=max_edges
    )
    x = np.zeros((nlev, n_cells), dtype=np.float32)
    y = np.zeros_like(x)
    z = np.zeros_like(x)
    for cell in range(n_cells):
        count = int(counts[cell])
        if count < 0 or count > max_edges:
            raise ValueError("nEdgesOnCell lies outside the slot extent")
        for slot in range(count):
            edge = int(edges[cell, slot])
            if edge < 0 or edge >= n_edges:
                raise ValueError("active edgesOnCell entry is out of range")
            for level in range(nlev):
                value = velocity[level, edge]
                x[level, cell] = _add(
                    x[level, cell], _mul(coeffs[cell, slot, 0], value)
                )
                y[level, cell] = _add(
                    y[level, cell], _mul(coeffs[cell, slot, 1], value)
                )
                z[level, cell] = _add(
                    z[level, cell], _mul(coeffs[cell, slot, 2], value)
                )
    u = np.empty_like(x)
    v = np.empty_like(x)
    for cell in range(n_cells):
        clat = _f32(np.cos(lat[cell]))
        slat = _f32(np.sin(lat[cell]))
        clon = _f32(np.cos(lon[cell]))
        slon = _f32(np.sin(lon[cell]))
        for level in range(nlev):
            u[level, cell] = _add(
                _mul(_sub(np.float32(0.0), x[level, cell]), slon),
                _mul(y[level, cell], clon),
            )
            horizontal = _add(
                _mul(x[level, cell], clon), _mul(y[level, cell], slon)
            )
            v[level, cell] = _add(
                _mul(_sub(np.float32(0.0), horizontal), slat),
                _mul(z[level, cell], clat),
            )
    return np.ascontiguousarray(u), np.ascontiguousarray(v)


def prepare_noahmp_sounding_cpu_oracle_v841(
    *,
    qv: Any,
    theta_m: Any,
    exner: Any,
    pressure_base: Any,
    pressure_p: Any,
    zgrid: Any,
    u: Any,
    v: Any,
) -> CpuNoahmpSoundingV841:
    """Follow ``lsm_noahmp_sounding_fromMPAS`` in released source order."""

    raw_qv = np.asarray(qv)
    if raw_qv.dtype != np.float32 or raw_qv.ndim != 2:
        raise TypeError("NoahMP raw qv must be FP32 [level,cell]")
    nlev, ncells = raw_qv.shape
    shape = (nlev, ncells)
    interface_shape = (nlev + 1, ncells)
    raw_qv = _require_cpu_f32("NoahMP.qv", raw_qv, shape)
    theta = _require_cpu_f32("NoahMP.theta_m", theta_m, shape)
    exner_in = _require_cpu_f32("NoahMP.exner", exner, shape)
    pressure_b = _require_cpu_f32("NoahMP.pressure_base", pressure_base, shape)
    pressure_perturbation = _require_cpu_f32(
        "NoahMP.pressure_p", pressure_p, shape
    )
    heights = _require_cpu_f32("NoahMP.zgrid", zgrid, interface_shape)
    zonal = _require_cpu_f32("NoahMP.u", u, shape)
    meridional = _require_cpu_f32("NoahMP.v", v, shape)
    if np.any(np.diff(heights, axis=0) <= np.float32(0.0)):
        raise ValueError("NoahMP zgrid must be surface_to_top")

    dz8w = np.empty(shape, dtype=np.float32)
    qv_curr = np.empty(shape, dtype=np.float32)
    t_phy = np.empty(shape, dtype=np.float32)
    u_phy = np.empty(shape, dtype=np.float32)
    v_phy = np.empty(shape, dtype=np.float32)
    p8w = np.empty(shape, dtype=np.float32)
    for cell in range(ncells):
        for level in range(nlev):
            dz8w[level, cell] = _sub(
                heights[level + 1, cell], heights[level, cell]
            )
            qv_curr[level, cell] = raw_qv[level, cell]
            denominator = _add(
                np.float32(1.0), _mul(RV_OVER_RD_F32, raw_qv[level, cell])
            )
            t_phy[level, cell] = _mul(
                _div(theta[level, cell], denominator), exner_in[level, cell]
            )
            u_phy[level, cell] = zonal[level, cell]
            v_phy[level, cell] = meridional[level, cell]

        z0 = heights[0, cell]
        z1 = _mul(
            np.float32(0.5), _add(heights[0, cell], heights[1, cell])
        )
        z2 = _mul(
            np.float32(0.5), _add(heights[1, cell], heights[2, cell])
        )
        w1 = _div(_sub(z0, z2), _sub(z1, z2))
        w2 = _sub(np.float32(1.0), w1)
        totm = _add(pressure_perturbation[0, cell], pressure_b[0, cell])
        totp = _add(pressure_perturbation[1, cell], pressure_b[1, cell])
        p8w[0, cell] = _add(_mul(w1, totm), _mul(w2, totp))
        for level in range(1, nlev):
            totm = _add(
                pressure_perturbation[level - 1, cell],
                pressure_b[level - 1, cell],
            )
            totp = _add(
                pressure_perturbation[level, cell], pressure_b[level, cell]
            )
            mult = _div(
                np.float32(1.0),
                _sub(heights[level + 1, cell], heights[level - 1, cell]),
            )
            fzm = _mul(
                mult, _sub(heights[level, cell], heights[level - 1, cell])
            )
            fzp = _mul(
                mult, _sub(heights[level + 1, cell], heights[level, cell])
            )
            p8w[level, cell] = _add(_mul(fzm, totp), _mul(fzp, totm))

    result = CpuNoahmpSoundingV841(
        dz8w=dz8w,
        qv_curr=qv_curr,
        t_phy=t_phy,
        u_phy=u_phy,
        v_phy=v_phy,
        p8w=p8w,
    )
    result.validate()
    return result


def prepare_mpas_to_phys_cpu_oracle_v841(
    *,
    rho_zz: Any,
    theta_m: Any,
    w: Any,
    scalars: Any,
    exner: Any,
    pressure_base: Any,
    pressure_p: Any,
    zgrid: Any,
    zz: Any,
    normal_velocity: Any,
    edges_on_cell: Any,
    n_edges_on_cell: Any,
    coeffs_reconstruct: Any,
    lat_cell: Any,
    lon_cell: Any,
    scalar_names: Sequence[str] = WSM6_SCALAR_NAMES,
) -> CpuMpasToPhysColumnsV841:
    """Readable FP32 oracle that follows the released loops and assignments."""

    names = tuple(str(name).strip().lower() for name in scalar_names)
    # The oracle follows whatever row it is given, and refuses one nobody
    # declared -- the same rule the device path applies.
    species_row_for_names(names)
    density = np.asarray(rho_zz)
    if density.dtype != np.float32 or density.ndim != 2:
        raise TypeError("rho_zz must be FP32 [level,cell]")
    nlev, ncells = density.shape
    if nlev < 2 or ncells <= 0:
        raise ValueError("MPAS_to_physics requires at least two levels")
    shape = (nlev, ncells)
    interface_shape = (nlev + 1, ncells)
    density = _require_cpu_f32("rho_zz", density, shape)
    theta = _require_cpu_f32("theta_m", theta_m, shape)
    vertical_velocity = _require_cpu_f32("w", w, interface_shape)
    exner_in = _require_cpu_f32("exner", exner, shape)
    pressure_b = _require_cpu_f32("pressure_base", pressure_base, shape)
    pressure_perturbation = _require_cpu_f32("pressure_p", pressure_p, shape)
    heights = _require_cpu_f32("zgrid", zgrid, interface_shape)
    metric = _require_cpu_f32("zz", zz, shape)
    scalar_input = _require_cpu_f32(
        "scalars", scalars, (len(names), nlev, ncells)
    )
    if np.any(np.diff(heights, axis=0) <= np.float32(0.0)):
        raise ValueError("zgrid vertical order must be surface_to_top")
    if np.any(density <= np.float32(0.0)) or np.any(theta <= np.float32(0.0)):
        raise ValueError("rho_zz and theta_m must be positive")

    q_fields = [np.empty(shape, dtype=np.float32) for _ in names]
    u_p, v_p = reconstruct_winds_cpu_oracle_v841(
        normal_velocity,
        edges_on_cell=edges_on_cell,
        n_edges_on_cell=n_edges_on_cell,
        coeffs_reconstruct=coeffs_reconstruct,
        lat_cell=lat_cell,
        lon_cell=lon_cell,
    )
    if u_p.shape != shape:
        raise ValueError("reconstruction geometry cell count does not match rho_zz")
    zz_p = np.empty(shape, dtype=np.float32)
    rho_dry = np.empty(shape, dtype=np.float32)
    rho_p = np.empty(shape, dtype=np.float32)
    th_p = np.empty(shape, dtype=np.float32)
    t_p = np.empty(shape, dtype=np.float32)
    pi_p = np.empty(shape, dtype=np.float32)
    pres_p = np.empty(shape, dtype=np.float32)
    zmid_p = np.empty(shape, dtype=np.float32)
    dz_p = np.empty(shape, dtype=np.float32)
    for level in range(nlev):
        for cell in range(ncells):
            for scalar_index in range(len(names)):
                q_fields[scalar_index][level, cell] = _positive_q(
                    scalar_input[scalar_index, level, cell]
                )
            qv = q_fields[0][level, cell]
            zz_p[level, cell] = metric[level, cell]
            dry_density = _mul(metric[level, cell], density[level, cell])
            rho_dry[level, cell] = dry_density
            rho_p[level, cell] = _mul(dry_density, _add(np.float32(1.0), qv))
            denominator = _add(np.float32(1.0), _mul(RV_OVER_RD_F32, qv))
            th_p[level, cell] = _div(theta[level, cell], denominator)
            t_p[level, cell] = _mul(th_p[level, cell], exner_in[level, cell])
            pi_p[level, cell] = exner_in[level, cell]
            pres_p[level, cell] = _add(
                pressure_perturbation[level, cell], pressure_b[level, cell]
            )
            zmid_p[level, cell] = _mul(
                np.float32(0.5),
                _add(heights[level + 1, cell], heights[level, cell]),
            )
            dz_p[level, cell] = _sub(
                heights[level + 1, cell], heights[level, cell]
            )

    psfc_p = np.empty(ncells, dtype=np.float32)
    znu_p = np.empty(shape, dtype=np.float32)
    t2_p = np.empty(interface_shape, dtype=np.float32)
    pres2_p = np.empty(interface_shape, dtype=np.float32)
    pres2_hyd_p = np.empty(interface_shape, dtype=np.float32)
    pres2_hydd_p = np.empty(interface_shape, dtype=np.float32)
    pres_hyd_p = np.empty(shape, dtype=np.float32)
    pres_hydd_p = np.empty(shape, dtype=np.float32)
    psfc_hyd_p = np.empty(ncells, dtype=np.float32)
    psfc_hydd_p = np.empty(ncells, dtype=np.float32)
    znu_hyd_p = np.empty(shape, dtype=np.float32)
    plrad = np.empty(ncells, dtype=np.float32)
    for cell in range(ncells):
        tem1 = _sub(heights[1, cell], heights[0, cell])
        tem2 = _sub(heights[2, cell], heights[1, cell])
        rho1 = _mul(
            _mul(density[0, cell], metric[0, cell]),
            _add(np.float32(1.0), q_fields[0][0, cell]),
        )
        rho2 = _mul(
            _mul(density[1, cell], metric[1, cell]),
            _add(np.float32(1.0), q_fields[0][1, cell]),
        )
        correction = _div(
            _mul(_mul(np.float32(0.5), _sub(rho2, rho1)), tem1),
            _add(tem1, tem2),
        )
        hydro = _mul(
            _mul(_mul(np.float32(0.5), GRAVITY_F32), tem1),
            _sub(rho1, correction),
        )
        psfc_p[cell] = _add(hydro, pres_p[0, cell])
        for level in range(nlev):
            znu_p[level, cell] = _div(pres_p[level, cell], psfc_p[cell])

        for interface in range(1, nlev):
            inverse = _div(
                np.float32(1.0),
                _sub(heights[interface + 1, cell], heights[interface - 1, cell]),
            )
            fzm = _mul(
                _sub(heights[interface, cell], heights[interface - 1, cell]),
                inverse,
            )
            fzp = _mul(
                _sub(heights[interface + 1, cell], heights[interface, cell]),
                inverse,
            )
            t2_p[interface, cell] = _add(
                _mul(fzm, t_p[interface, cell]),
                _mul(fzp, t_p[interface - 1, cell]),
            )
            pres2_p[interface, cell] = _add(
                _mul(fzm, pres_p[interface, cell]),
                _mul(fzp, pres_p[interface - 1, cell]),
            )

        z0 = heights[nlev, cell]
        z1 = _mul(
            np.float32(0.5),
            _add(heights[nlev, cell], heights[nlev - 1, cell]),
        )
        z2 = _mul(
            np.float32(0.5),
            _add(heights[nlev - 1, cell], heights[nlev - 2, cell]),
        )
        w1 = _div(_sub(z0, z2), _sub(z1, z2))
        w2 = _sub(np.float32(1.0), w1)
        t2_p[nlev, cell] = _add(
            _mul(w1, t_p[nlev - 1, cell]), _mul(w2, t_p[nlev - 2, cell])
        )
        logarithm = _add(
            _mul(w1, _f32(np.log(pres_p[nlev - 1, cell]))),
            _mul(w2, _f32(np.log(pres_p[nlev - 2, cell]))),
        )
        pres2_p[nlev, cell] = _f32(np.exp(logarithm))

        z0 = heights[0, cell]
        z1 = _mul(
            np.float32(0.5), _add(heights[0, cell], heights[1, cell])
        )
        z2 = _mul(
            np.float32(0.5), _add(heights[1, cell], heights[2, cell])
        )
        w1 = _div(_sub(z0, z2), _sub(z1, z2))
        w2 = _sub(np.float32(1.0), w1)
        t2_p[0, cell] = _add(_mul(w1, t_p[0, cell]), _mul(w2, t_p[1, cell]))
        pres2_p[0, cell] = _add(
            _mul(w1, pres_p[0, cell]), _mul(w2, pres_p[1, cell])
        )

        pres2_hyd_p[nlev, cell] = pres2_p[nlev, cell]
        pres2_hydd_p[nlev, cell] = pres2_p[nlev, cell]
        for level in range(nlev - 1, -1, -1):
            rho_a = _div(
                rho_p[level, cell],
                _add(np.float32(1.0), q_fields[0][level, cell]),
            )
            pres2_hyd_p[level, cell] = _add(
                pres2_hyd_p[level + 1, cell],
                _mul(_mul(GRAVITY_F32, rho_p[level, cell]), dz_p[level, cell]),
            )
            pres2_hydd_p[level, cell] = _add(
                pres2_hydd_p[level + 1, cell],
                _mul(_mul(GRAVITY_F32, rho_a), dz_p[level, cell]),
            )
        for level in range(nlev - 1, -1, -1):
            pres_hyd_p[level, cell] = _mul(
                np.float32(0.5),
                _add(
                    pres2_hyd_p[level + 1, cell], pres2_hyd_p[level, cell]
                ),
            )
            pres_hydd_p[level, cell] = _mul(
                np.float32(0.5),
                _add(
                    pres2_hydd_p[level + 1, cell], pres2_hydd_p[level, cell]
                ),
            )
        psfc_hyd_p[cell] = pres2_hyd_p[0, cell]
        psfc_hydd_p[cell] = pres2_hydd_p[0, cell]
        for level in range(nlev - 1, -1, -1):
            znu_hyd_p[level, cell] = _div(
                pres_hyd_p[level, cell], psfc_hyd_p[cell]
            )
        plrad[cell] = pres2_p[nlev, cell]

    noahmp_sounding = prepare_noahmp_sounding_cpu_oracle_v841(
        qv=scalar_input[0],
        theta_m=theta,
        exner=exner_in,
        pressure_base=pressure_b,
        pressure_p=pressure_perturbation,
        zgrid=heights,
        u=u_p,
        v=v_p,
    )
    result = CpuMpasToPhysColumnsV841(
        species_p={
            name: q_fields[index] for index, name in enumerate(names)
        },
        u_p=u_p,
        v_p=v_p,
        zz_p=zz_p,
        rho_dry=rho_dry,
        rho_p=rho_p,
        th_p=th_p,
        t_p=t_p,
        pi_p=pi_p,
        pres_p=pres_p,
        zmid_p=zmid_p,
        dz_p=dz_p,
        pres_hyd_p=pres_hyd_p,
        pres_hydd_p=pres_hydd_p,
        znu_p=znu_p,
        znu_hyd_p=znu_hyd_p,
        w_p=np.array(vertical_velocity, copy=True, order="C"),
        z_p=np.array(heights, copy=True, order="C"),
        t2_p=t2_p,
        pres2_p=pres2_p,
        pres2_hyd_p=pres2_hyd_p,
        pres2_hydd_p=pres2_hydd_p,
        psfc_p=psfc_p,
        psfc_hyd_p=psfc_hyd_p,
        psfc_hydd_p=psfc_hydd_p,
        plrad=plrad,
        noahmp_sounding=noahmp_sounding,
        _source_scalars=np.array(scalar_input, copy=True, order="C"),
    )
    result.validate()
    return result


_CUDA_SOURCE = CUDA_FTZ_HELPERS + r"""
__device__ __forceinline__ unsigned int prep_positive_q_bits_v841(
    const float raw)
{
    const unsigned int bits = __float_as_uint(raw);
    return (bits & 0x80000000u) == 0u ? bits : 0u;
}

__device__ __forceinline__ float prep_positive_q_v841(const float raw)
{
    return __uint_as_float(prep_positive_q_bits_v841(raw));
}

__device__ __forceinline__ bool prep_finite_bits_v841(const float raw)
{
    return (__float_as_uint(raw) & 0x7fffffffu) < 0x7f800000u;
}

__device__ __forceinline__ bool prep_negative_nonzero_bits_v841(
    const float raw)
{
    const unsigned int bits = __float_as_uint(raw);
    return (bits & 0x80000000u) != 0u
        && (bits & 0x7fffffffu) != 0u;
}

extern "C" __global__ void prep_mass_v841_f32(
    const int nlev, const int ncells, const int nspecies, const float rvord,
    const float *rho_zz, const float *theta_m, const float *scalars,
    const float *exner, const float *pressure_base, const float *pressure_p,
    const float *zgrid, const float *zz, float *const *qout, float *zz_p,
    float *rho_dry, float *rho_p, float *th_p, float *t_p, float *pi_p,
    float *pres_p, float *zmid_p, float *dz_p, int *invalid)
{
    const int cell = blockDim.x * blockIdx.x + threadIdx.x;
    if (cell >= ncells) return;
    const int plane = nlev * ncells;
    // qout is a DEVICE table of per-species output pointers, sized by the
    // scheme's species row, in the row's order.  It replaces six welded
    // parameters; the arithmetic below is unchanged, which is what the
    // WSM6 byte-identity arm measures.  Slot 0 is water vapour by the row's
    // own invariant -- the moist density, the dry-theta conversion, the
    // NoahMP sounding and the interface extrapolation all read it.
    for (int k = 0; k < nlev; ++k) {
        const int i = k * ncells + cell;
        for (int s = 0; s < nspecies; ++s) {
            const float raw = scalars[s * plane + i];
            if (!prep_finite_bits_v841(raw)) atomicExch(invalid, 1);
            qout[s][i] = prep_positive_q_v841(raw);
        }
        const float qv = qout[0][i];
        zz_p[i] = zz[i];
        const float dry_rho = mpas_mul(zz[i], rho_zz[i]);
        rho_dry[i] = dry_rho;
        rho_p[i] = mpas_mul(dry_rho, mpas_add(1.0f, qv));
        const float denominator = mpas_add(1.0f, mpas_mul(rvord, qv));
        th_p[i] = mpas_div(theta_m[i], denominator);
        t_p[i] = mpas_mul(th_p[i], exner[i]);
        pi_p[i] = exner[i];
        pres_p[i] = mpas_add(pressure_p[i], pressure_base[i]);
        const float bottom = zgrid[k * ncells + cell];
        const float top = zgrid[(k + 1) * ncells + cell];
        zmid_p[i] = mpas_mul(0.5f, mpas_add(top, bottom));
        dz_p[i] = mpas_sub(top, bottom);
        if (!isfinite(rho_zz[i]) || rho_zz[i] <= 0.0f
                || !isfinite(theta_m[i]) || theta_m[i] <= 0.0f
                || !isfinite(zz[i]) || !isfinite(exner[i])
                || !isfinite(pressure_p[i]) || !isfinite(pressure_base[i]))
            atomicExch(invalid, 1);
    }
}

extern "C" __global__ void reconstruct_winds_v841_f32(
    const int nlev, const int ncells, const int nedges, const int maxedges,
    const float *normal_velocity, const int *edges_on_cell,
    const int *n_edges_on_cell, const float *coeffs_reconstruct,
    const float *lat_cell, const float *lon_cell,
    const int *sealed_edges_on_cell, const int *sealed_n_edges_on_cell,
    const float *sealed_coeffs_reconstruct,
    const float *sealed_lat_cell, const float *sealed_lon_cell,
    float *x, float *y, float *z, float *u_p, float *v_p, int *invalid)
{
    // One lane per level/cell; each lane retains the original slot reduction.
    const int element = blockDim.x * blockIdx.x + threadIdx.x;
    if (element >= nlev * ncells) return;
    const int cell = element % ncells;
    const int k = element / ncells;
    bool geometry_changed =
        n_edges_on_cell[cell] != sealed_n_edges_on_cell[cell]
        || __float_as_uint(lat_cell[cell])
            != __float_as_uint(sealed_lat_cell[cell])
        || __float_as_uint(lon_cell[cell])
            != __float_as_uint(sealed_lon_cell[cell]);
    for (int slot = 0; slot < maxedges; ++slot) {
        const int edge_index = cell * maxedges + slot;
        const int coeff_index = edge_index * 3;
        geometry_changed = geometry_changed
            || edges_on_cell[edge_index] != sealed_edges_on_cell[edge_index]
            || __float_as_uint(coeffs_reconstruct[coeff_index])
                != __float_as_uint(sealed_coeffs_reconstruct[coeff_index])
            || __float_as_uint(coeffs_reconstruct[coeff_index + 1])
                != __float_as_uint(sealed_coeffs_reconstruct[coeff_index + 1])
            || __float_as_uint(coeffs_reconstruct[coeff_index + 2])
                != __float_as_uint(sealed_coeffs_reconstruct[coeff_index + 2]);
    }
    if (geometry_changed) {
        atomicExch(invalid, 1);
        return;
    }
    {
        const int i = k * ncells + cell;
        x[i] = 0.0f;
        y[i] = 0.0f;
        z[i] = 0.0f;
    }
    const int count = n_edges_on_cell[cell];
    if (count < 0 || count > maxedges) {
        atomicExch(invalid, 1);
        return;
    }
    for (int slot = 0; slot < count; ++slot) {
        const int edge = edges_on_cell[cell * maxedges + slot];
        if (edge < 0 || edge >= nedges) {
            atomicExch(invalid, 1);
            continue;
        }
        const int ci = (cell * maxedges + slot) * 3;
        const float cx = coeffs_reconstruct[ci];
        const float cy = coeffs_reconstruct[ci + 1];
        const float cz = coeffs_reconstruct[ci + 2];
        if (!isfinite(cx) || !isfinite(cy) || !isfinite(cz))
            atomicExch(invalid, 1);
        {
            const int i = k * ncells + cell;
            const float value = normal_velocity[k * nedges + edge];
            if (!isfinite(value)) atomicExch(invalid, 1);
            x[i] = mpas_add(x[i], mpas_mul(cx, value));
            y[i] = mpas_add(y[i], mpas_mul(cy, value));
            z[i] = mpas_add(z[i], mpas_mul(cz, value));
        }
    }
    const float clat = cosf(lat_cell[cell]);
    const float slat = sinf(lat_cell[cell]);
    const float clon = cosf(lon_cell[cell]);
    const float slon = sinf(lon_cell[cell]);
    if (!isfinite(clat) || !isfinite(slat) || !isfinite(clon) || !isfinite(slon))
        atomicExch(invalid, 1);
    {
        const int i = k * ncells + cell;
        u_p[i] = mpas_add(
            mpas_mul(mpas_sub(0.0f, x[i]), slon), mpas_mul(y[i], clon));
        const float horizontal = mpas_add(
            mpas_mul(x[i], clon), mpas_mul(y[i], slon));
        v_p[i] = mpas_add(
            mpas_mul(mpas_sub(0.0f, horizontal), slat), mpas_mul(z[i], clat));
    }
}

extern "C" __global__ void prep_noahmp_sounding_v841_f32(
    const int nlev, const int ncells, const float rvord,
    const float *raw_qv, const float *theta_m, const float *exner,
    const float *pressure_base, const float *pressure_p,
    const float *zgrid, const float *u_p, const float *v_p,
    float *dz8w, float *qv_curr, float *t_phy, float *u_phy,
    float *v_phy, float *p8w, int *invalid)
{
    const int cell = blockDim.x * blockIdx.x + threadIdx.x;
    if (cell >= ncells) return;
    for (int k = 0; k < nlev; ++k) {
        const int i = k * ncells + cell;
        const float qraw = raw_qv[i];
        const float thickness = mpas_sub(
            zgrid[i + ncells], zgrid[i]);
        const float denominator = mpas_add(
            1.0f, mpas_mul(rvord, qraw));
        const float temperature = mpas_mul(
            mpas_div(theta_m[i], denominator), exner[i]);
        dz8w[i] = thickness;
        qv_curr[i] = qraw;
        t_phy[i] = temperature;
        u_phy[i] = u_p[i];
        v_phy[i] = v_p[i];
        if (!isfinite(qraw) || !isfinite(temperature)
                || temperature <= 0.0f || !isfinite(thickness)
                || thickness <= 0.0f || !isfinite(u_p[i])
                || !isfinite(v_p[i]))
            atomicExch(invalid, 1);
    }

    float z0 = zgrid[cell];
    float z1 = mpas_mul(
        0.5f, mpas_add(zgrid[cell], zgrid[ncells + cell]));
    float z2 = mpas_mul(
        0.5f, mpas_add(zgrid[ncells + cell], zgrid[2 * ncells + cell]));
    float w1 = mpas_div(mpas_sub(z0, z2), mpas_sub(z1, z2));
    float w2 = mpas_sub(1.0f, w1);
    float totm = mpas_add(pressure_p[cell], pressure_base[cell]);
    float totp = mpas_add(
        pressure_p[ncells + cell], pressure_base[ncells + cell]);
    p8w[cell] = mpas_add(mpas_mul(w1, totm), mpas_mul(w2, totp));
    for (int k = 1; k < nlev; ++k) {
        const int i = k * ncells + cell;
        totm = mpas_add(pressure_p[i - ncells], pressure_base[i - ncells]);
        totp = mpas_add(pressure_p[i], pressure_base[i]);
        const float mult = mpas_div(
            1.0f,
            mpas_sub(zgrid[i + ncells], zgrid[i - ncells]));
        const float fzm = mpas_mul(
            mult, mpas_sub(zgrid[i], zgrid[i - ncells]));
        const float fzp = mpas_mul(
            mult, mpas_sub(zgrid[i + ncells], zgrid[i]));
        p8w[i] = mpas_add(mpas_mul(fzm, totp), mpas_mul(fzp, totm));
    }
    for (int k = 0; k < nlev; ++k) {
        const float value = p8w[k * ncells + cell];
        if (!isfinite(value) || value <= 0.0f)
            atomicExch(invalid, 1);
    }
}

extern "C" __global__ void prep_interfaces_v841_f32(
    const int nlev, const int ncells, const float gravity,
    const float *rho_zz, const float *zz, const float *qv_p,
    const float *pres_p, const float *t_p, const float *rho_p,
    const float *dz_p, const float *zgrid, const float *w,
    float *w_p, float *z_p, float *t2_p, float *pres2_p,
    float *pres_hyd_p, float *pres2_hyd_p, float *pres_hydd_p,
    float *pres2_hydd_p, float *psfc_p, float *psfc_hyd_p,
    float *psfc_hydd_p, float *znu_p, float *znu_hyd_p,
    float *plrad, int *invalid)
{
    const int cell = blockDim.x * blockIdx.x + threadIdx.x;
    if (cell >= ncells) return;
    for (int k = 0; k <= nlev; ++k) {
        const int i = k * ncells + cell;
        w_p[i] = w[i];
        z_p[i] = zgrid[i];
    }
    const float tem1 = mpas_sub(zgrid[ncells + cell], zgrid[cell]);
    const float tem2 = mpas_sub(zgrid[2 * ncells + cell], zgrid[ncells + cell]);
    const float rho1 = mpas_mul(
        mpas_mul(rho_zz[cell], zz[cell]), mpas_add(1.0f, qv_p[cell]));
    const float rho2 = mpas_mul(
        mpas_mul(rho_zz[ncells + cell], zz[ncells + cell]),
        mpas_add(1.0f, qv_p[ncells + cell]));
    const float correction = mpas_div(
        mpas_mul(mpas_mul(0.5f, mpas_sub(rho2, rho1)), tem1),
        mpas_add(tem1, tem2));
    const float hydro = mpas_mul(
        mpas_mul(mpas_mul(0.5f, gravity), tem1),
        mpas_sub(rho1, correction));
    psfc_p[cell] = mpas_add(hydro, pres_p[cell]);
    for (int k = 0; k < nlev; ++k) {
        const int i = k * ncells + cell;
        znu_p[i] = mpas_div(pres_p[i], psfc_p[cell]);
    }

    for (int k = 1; k < nlev; ++k) {
        const float inverse = mpas_div(
            1.0f, mpas_sub(zgrid[(k + 1) * ncells + cell],
                           zgrid[(k - 1) * ncells + cell]));
        const float fzm = mpas_mul(
            mpas_sub(zgrid[k * ncells + cell],
                     zgrid[(k - 1) * ncells + cell]), inverse);
        const float fzp = mpas_mul(
            mpas_sub(zgrid[(k + 1) * ncells + cell],
                     zgrid[k * ncells + cell]), inverse);
        const int i = k * ncells + cell;
        t2_p[i] = mpas_add(
            mpas_mul(fzm, t_p[i]), mpas_mul(fzp, t_p[i - ncells]));
        pres2_p[i] = mpas_add(
            mpas_mul(fzm, pres_p[i]), mpas_mul(fzp, pres_p[i - ncells]));
    }

    const int top = nlev * ncells + cell;
    float z0 = zgrid[top];
    float z1 = mpas_mul(0.5f, mpas_add(zgrid[top], zgrid[top - ncells]));
    float z2 = mpas_mul(
        0.5f, mpas_add(zgrid[top - ncells], zgrid[top - 2 * ncells]));
    float w1 = mpas_div(mpas_sub(z0, z2), mpas_sub(z1, z2));
    float w2 = mpas_sub(1.0f, w1);
    t2_p[top] = mpas_add(
        mpas_mul(w1, t_p[top - ncells]),
        mpas_mul(w2, t_p[top - 2 * ncells]));
    pres2_p[top] = expf(mpas_add(
        mpas_mul(w1, logf(pres_p[top - ncells])),
        mpas_mul(w2, logf(pres_p[top - 2 * ncells]))));

    z0 = zgrid[cell];
    z1 = mpas_mul(0.5f, mpas_add(zgrid[cell], zgrid[ncells + cell]));
    z2 = mpas_mul(
        0.5f, mpas_add(zgrid[ncells + cell], zgrid[2 * ncells + cell]));
    w1 = mpas_div(mpas_sub(z0, z2), mpas_sub(z1, z2));
    w2 = mpas_sub(1.0f, w1);
    t2_p[cell] = mpas_add(mpas_mul(w1, t_p[cell]),
                          mpas_mul(w2, t_p[ncells + cell]));
    pres2_p[cell] = mpas_add(mpas_mul(w1, pres_p[cell]),
                             mpas_mul(w2, pres_p[ncells + cell]));

    pres2_hyd_p[top] = pres2_p[top];
    pres2_hydd_p[top] = pres2_p[top];
    for (int k = nlev - 1; k >= 0; --k) {
        const int i = k * ncells + cell;
        const float rho_a = mpas_div(rho_p[i], mpas_add(1.0f, qv_p[i]));
        pres2_hyd_p[i] = mpas_add(
            pres2_hyd_p[i + ncells],
            mpas_mul(mpas_mul(gravity, rho_p[i]), dz_p[i]));
        pres2_hydd_p[i] = mpas_add(
            pres2_hydd_p[i + ncells],
            mpas_mul(mpas_mul(gravity, rho_a), dz_p[i]));
    }
    for (int k = nlev - 1; k >= 0; --k) {
        const int i = k * ncells + cell;
        pres_hyd_p[i] = mpas_mul(
            0.5f, mpas_add(pres2_hyd_p[i + ncells], pres2_hyd_p[i]));
        pres_hydd_p[i] = mpas_mul(
            0.5f, mpas_add(pres2_hydd_p[i + ncells], pres2_hydd_p[i]));
    }
    psfc_hyd_p[cell] = pres2_hyd_p[cell];
    psfc_hydd_p[cell] = pres2_hydd_p[cell];
    for (int k = nlev - 1; k >= 0; --k) {
        const int i = k * ncells + cell;
        znu_hyd_p[i] = mpas_div(pres_hyd_p[i], psfc_hyd_p[cell]);
    }
    plrad[cell] = pres2_p[top];
}

extern "C" __global__ void validate_prep_v841_f32(
    const int nlev, const int ncells, const int nedges, const int maxedges,
    const int wsm6_ready, const int prepare_winds, const int prepare_noahmp,
    const float gravity, const float rvord,
    const int *edges_on_cell, const int *n_edges_on_cell,
    const float *coeffs_reconstruct, const float *lat_cell,
    const float *lon_cell, const float *source_scalars,
    const float *source_theta_m, const float *source_exner,
    const int nspecies, const float *const *qspecies,
    const float *u_p, const float *v_p,
    const float *zz_p, const float *rho_dry, const float *rho_p,
    const float *th_p, const float *t_p, const float *pi_p,
    const float *pres_p, const float *zmid_p, const float *dz_p,
    const float *pres_hyd_p, const float *pres_hydd_p,
    const float *znu_p, const float *znu_hyd_p, const float *w_p,
    const float *z_p, const float *t2_p, const float *pres2_p,
    const float *pres2_hyd_p, const float *pres2_hydd_p,
    const float *psfc_p,
    const float *psfc_hyd_p, const float *psfc_hydd_p,
    const float *plrad, const float *noah_dz8w,
    const float *noah_qv_curr, const float *noah_t_phy,
    const float *noah_u_phy, const float *noah_v_phy,
    const float *noah_p8w, int *invalid)
{
    const int cell = blockDim.x * blockIdx.x + threadIdx.x;
    if (cell >= ncells) return;
    const int plane = nlev * ncells;
    const int count = n_edges_on_cell[cell];
    if (count < 0 || count > maxedges || !isfinite(lat_cell[cell])
            || !isfinite(lon_cell[cell])) atomicExch(invalid, 1);
    for (int slot = 0; slot < maxedges; ++slot) {
        const int edge = edges_on_cell[cell * maxedges + slot];
        if (slot < count) {
            if (edge < 0 || edge >= nedges) atomicExch(invalid, 1);
            const int ci = (cell * maxedges + slot) * 3;
            if (!isfinite(coeffs_reconstruct[ci])
                    || !isfinite(coeffs_reconstruct[ci + 1])
                    || !isfinite(coeffs_reconstruct[ci + 2]))
                atomicExch(invalid, 1);
        } else if (edge != -1) {
            atomicExch(invalid, 1);
        }
    }
    // The 15 non-species mass-level fields.  The species are checked from
    // qspecies, whose length is the row's, so this table no longer grows
    // with the scheme.
    const float *mass[15] = {
        u_p, v_p, zz_p,
        rho_dry, rho_p, th_p, t_p, pi_p, pres_p, zmid_p, dz_p,
        pres_hyd_p, pres_hydd_p, znu_p, znu_hyd_p};
    for (int k = 0; k < nlev; ++k) {
        const int i = k * ncells + cell;
        if (prepare_winds && (!isfinite(u_p[i]) || !isfinite(v_p[i])))
            atomicExch(invalid, 1);
        for (int f = 2; f < 15; ++f)
            if (!isfinite(mass[f][i])) atomicExch(invalid, 1);
        for (int q = 0; q < nspecies; ++q) {
            if (!isfinite(qspecies[q][i])) atomicExch(invalid, 1);
            const float raw = source_scalars[q * plane + i];
            const float value = qspecies[q][i];
            const unsigned int expected = prep_positive_q_bits_v841(raw);
            if (!prep_finite_bits_v841(raw)
                    || __float_as_uint(value) != expected)
                atomicExch(invalid, 1);
            if ((__float_as_uint(value) & 0x80000000u) != 0u)
                atomicExch(invalid, 1);
            if (wsm6_ready && prep_negative_nonzero_bits_v841(raw))
                atomicExch(invalid, 1);
        }
        const float expected_rho = mpas_mul(
            rho_dry[i], mpas_add(1.0f, qspecies[0][i]));
        const float expected_theta = mpas_div(
            source_theta_m[i],
            mpas_add(1.0f, mpas_mul(rvord, qspecies[0][i])));
        const float expected_temp = mpas_mul(expected_theta, source_exner[i]);
        const float expected_dz = mpas_sub(z_p[i + ncells], z_p[i]);
        const float expected_zmid = mpas_mul(
            0.5f, mpas_add(z_p[i + ncells], z_p[i]));
        if (__float_as_uint(rho_p[i]) != __float_as_uint(expected_rho)
                || __float_as_uint(th_p[i]) != __float_as_uint(expected_theta)
                || __float_as_uint(t_p[i]) != __float_as_uint(expected_temp)
                || __float_as_uint(pi_p[i])
                    != __float_as_uint(source_exner[i])
                || __float_as_uint(dz_p[i]) != __float_as_uint(expected_dz)
                || __float_as_uint(zmid_p[i])
                    != __float_as_uint(expected_zmid))
            atomicExch(invalid, 1);
        if (prepare_noahmp) {
            const float expected_noah_temp = mpas_mul(
                mpas_div(
                    source_theta_m[i],
                    mpas_add(1.0f, mpas_mul(rvord, source_scalars[i]))),
                source_exner[i]);
            if (__float_as_uint(noah_qv_curr[i])
                        != __float_as_uint(source_scalars[i])
                    || __float_as_uint(noah_t_phy[i])
                        != __float_as_uint(expected_noah_temp)
                    || __float_as_uint(noah_dz8w[i])
                        != __float_as_uint(dz_p[i])
                    || __float_as_uint(noah_u_phy[i]) != __float_as_uint(u_p[i])
                    || __float_as_uint(noah_v_phy[i]) != __float_as_uint(v_p[i])
                    || __float_as_uint(noah_p8w[i])
                        != __float_as_uint(pres2_p[i]))
                atomicExch(invalid, 1);
        }
        if (rho_dry[i] <= 0.0f || rho_p[i] <= 0.0f
                || th_p[i] <= 0.0f || t_p[i] <= 0.0f
                || pi_p[i] <= 0.0f || pres_p[i] <= 0.0f
                || pres_hyd_p[i] <= 0.0f || pres_hydd_p[i] <= 0.0f
                || dz_p[i] <= 0.0f || znu_p[i] <= 0.0f
                || znu_hyd_p[i] <= 0.0f) atomicExch(invalid, 1);
        if (prepare_noahmp && (!isfinite(noah_t_phy[i]) || noah_t_phy[i] <= 0.0f
                || !isfinite(noah_p8w[i]) || noah_p8w[i] <= 0.0f))
            atomicExch(invalid, 1);
        // MPAS detects exactly this pressure inversion in MPAS_to_physics and
        // does NOT refuse.  mpas_atmphys_interface.F:458-474 tests only pres_p,
        // only across the two lowest levels
        //   :462  if(pres_p(i,1,j) .le. pres_p(i,2,j)) then
        // writes diagnostic lines at :463-472, and its single fatal escalation
        // at :473 is COMMENTED OUT in the official source:
        //   !      call mpas_log_write('pressure increasing with height', &
        //   !                          messageType=MPAS_LOG_CRIT)
        // so execution continues past :474 unconditionally.  Refusing the step
        // was stricter than the authority, and stricter at more sites: native
        // never tests pres_hyd_p or pres_hydd_p for monotonicity at all.
        // Non-finite and non-positive pressure remain refused above.
    }

    const float *interfaces[6] = {
        w_p, z_p, t2_p, pres2_p, pres2_hyd_p, pres2_hydd_p};
    for (int k = 0; k <= nlev; ++k) {
        const int i = k * ncells + cell;
        for (int f = 0; f < 6; ++f)
            if (!isfinite(interfaces[f][i])) atomicExch(invalid, 1);
        if (t2_p[i] <= 0.0f || pres2_p[i] <= 0.0f
                || pres2_hyd_p[i] <= 0.0f || pres2_hydd_p[i] <= 0.0f)
            atomicExch(invalid, 1);
        if (k < nlev) {
            const int above = i + ncells;
            // z_p ordering is RETAINED: the t2_p/fzm/fzp reconstruction below
            // divides by (z_p[k+1] - z_p[k-1]), which is native's own
            // expression at mpas_atmphys_interface.F:479
            //   tem1 = 1./(zgrid(k+1,i)-zgrid(k-1,i))
            // so a non-monotonic column is a genuine division hazard rather
            // than a state native tolerates.  The interface PRESSURE ordering
            // refusals beside it are removed for the reason cited above:
            // native has no interface pressure monotonicity test whatsoever.
            if (!(z_p[above] > z_p[i])) atomicExch(invalid, 1);
        }
    }

    for (int k = 1; k < nlev; ++k) {
        const int i = k * ncells + cell;
        const float inverse = mpas_div(
            1.0f, mpas_sub(z_p[i + ncells], z_p[i - ncells]));
        const float fzm = mpas_mul(
            mpas_sub(z_p[i], z_p[i - ncells]), inverse);
        const float fzp = mpas_mul(
            mpas_sub(z_p[i + ncells], z_p[i]), inverse);
        const float expected_t2 = mpas_add(
            mpas_mul(fzm, t_p[i]), mpas_mul(fzp, t_p[i - ncells]));
        const float expected_p2 = mpas_add(
            mpas_mul(fzm, pres_p[i]), mpas_mul(fzp, pres_p[i - ncells]));
        if (__float_as_uint(t2_p[i]) != __float_as_uint(expected_t2)
                || __float_as_uint(pres2_p[i])
                    != __float_as_uint(expected_p2))
            atomicExch(invalid, 1);
    }
    const int top = nlev * ncells + cell;
    float z0 = z_p[top];
    float z1 = mpas_mul(0.5f, mpas_add(z_p[top], z_p[top - ncells]));
    float z2 = mpas_mul(
        0.5f, mpas_add(z_p[top - ncells], z_p[top - 2 * ncells]));
    float w1 = mpas_div(mpas_sub(z0, z2), mpas_sub(z1, z2));
    float w2 = mpas_sub(1.0f, w1);
    const float expected_top_t2 = mpas_add(
        mpas_mul(w1, t_p[top - ncells]),
        mpas_mul(w2, t_p[top - 2 * ncells]));
    const float expected_top_p2 = expf(mpas_add(
        mpas_mul(w1, logf(pres_p[top - ncells])),
        mpas_mul(w2, logf(pres_p[top - 2 * ncells]))));
    z0 = z_p[cell];
    z1 = mpas_mul(0.5f, mpas_add(z_p[cell], z_p[ncells + cell]));
    z2 = mpas_mul(
        0.5f, mpas_add(z_p[ncells + cell], z_p[2 * ncells + cell]));
    w1 = mpas_div(mpas_sub(z0, z2), mpas_sub(z1, z2));
    w2 = mpas_sub(1.0f, w1);
    const float expected_bottom_t2 = mpas_add(
        mpas_mul(w1, t_p[cell]), mpas_mul(w2, t_p[ncells + cell]));
    const float expected_bottom_p2 = mpas_add(
        mpas_mul(w1, pres_p[cell]), mpas_mul(w2, pres_p[ncells + cell]));
    if (__float_as_uint(t2_p[top]) != __float_as_uint(expected_top_t2)
            || __float_as_uint(pres2_p[top])
                != __float_as_uint(expected_top_p2)
            || __float_as_uint(t2_p[cell])
                != __float_as_uint(expected_bottom_t2)
            || __float_as_uint(pres2_p[cell])
                != __float_as_uint(expected_bottom_p2))
        atomicExch(invalid, 1);

    if (__float_as_uint(pres2_hyd_p[top])
                != __float_as_uint(pres2_p[top])
            || __float_as_uint(pres2_hydd_p[top])
                != __float_as_uint(pres2_p[top]))
        atomicExch(invalid, 1);
    for (int k = nlev - 1; k >= 0; --k) {
        const int i = k * ncells + cell;
        const float rho_a = mpas_div(
            rho_p[i], mpas_add(1.0f, qspecies[0][i]));
        const float expected_p2_hyd = mpas_add(
            pres2_hyd_p[i + ncells],
            mpas_mul(mpas_mul(gravity, rho_p[i]), dz_p[i]));
        const float expected_p2_hydd = mpas_add(
            pres2_hydd_p[i + ncells],
            mpas_mul(mpas_mul(gravity, rho_a), dz_p[i]));
        const float expected_p_hyd = mpas_mul(
            0.5f, mpas_add(expected_p2_hyd, pres2_hyd_p[i + ncells]));
        const float expected_p_hydd = mpas_mul(
            0.5f, mpas_add(expected_p2_hydd, pres2_hydd_p[i + ncells]));
        const float expected_znu = mpas_div(pres_p[i], psfc_p[cell]);
        const float expected_znu_hyd = mpas_div(
            expected_p_hyd, psfc_hyd_p[cell]);
        if (__float_as_uint(pres2_hyd_p[i])
                    != __float_as_uint(expected_p2_hyd)
                || __float_as_uint(pres2_hydd_p[i])
                    != __float_as_uint(expected_p2_hydd)
                || __float_as_uint(pres_hyd_p[i])
                    != __float_as_uint(expected_p_hyd)
                || __float_as_uint(pres_hydd_p[i])
                    != __float_as_uint(expected_p_hydd)
                || __float_as_uint(znu_p[i])
                    != __float_as_uint(expected_znu)
                || __float_as_uint(znu_hyd_p[i])
                    != __float_as_uint(expected_znu_hyd))
            atomicExch(invalid, 1);
    }
    if (!isfinite(psfc_p[cell]) || psfc_p[cell] <= 0.0f
            || !isfinite(psfc_hyd_p[cell]) || psfc_hyd_p[cell] <= 0.0f
            || !isfinite(psfc_hydd_p[cell]) || psfc_hydd_p[cell] <= 0.0f
            || !isfinite(plrad[cell]) || plrad[cell] <= 0.0f
            || __float_as_uint(psfc_hyd_p[cell])
                != __float_as_uint(pres2_hyd_p[cell])
            || __float_as_uint(psfc_hydd_p[cell])
                != __float_as_uint(pres2_hydd_p[cell])
            || __float_as_uint(plrad[cell])
                != __float_as_uint(pres2_p[top]))
        atomicExch(invalid, 1);
}
"""

CUDA_PHYSICS_PREP_V841_KERNEL_SHA256 = sha256(
    _CUDA_SOURCE.encode("utf-8")
).hexdigest()
_KERNEL_NAMES = (
    "prep_mass_v841_f32",
    "reconstruct_winds_v841_f32",
    "prep_noahmp_sounding_v841_f32",
    "prep_interfaces_v841_f32",
    "validate_prep_v841_f32",
)
_KERNELS: dict[tuple[int, str], Any] = {}
_REQUIRED_KERNEL_BASE_OPTIONS = ("--std=c++17", "--fmad=false")


def _validate_kernel_cache(cache: KernelCache) -> None:
    if not isinstance(cache, KernelCache):
        raise TypeError("kernel_cache must be a KernelCache")
    options = tuple(cache.base_options)
    if options != _hex_numerics.kernel_base():
        raise TypeError(
            "physics prep KernelCache base_options must be exactly "
            f"{_REQUIRED_KERNEL_BASE_OPTIONS}; got {options}"
        )


def _kernel(cache: KernelCache, name: str) -> Any:
    if name not in _KERNEL_NAMES:
        raise KeyError(name)
    _validate_kernel_cache(cache)
    key = (id(cache), name)
    result = _KERNELS.get(key)
    if result is None:
        result = cache.raw_kernel(
            name, _CUDA_SOURCE, module_key="hexcore.cuda_physics_prep_v841"
        )
        _KERNELS[key] = result
    return result


def _launch(cache: KernelCache, name: str, count: int, args: tuple[Any, ...]) -> None:
    # Resident coupling also launches reconstruction directly with a cell count.
    # Keep that caller contract while covering every level in the new mapping.
    if name == "reconstruct_winds_v841_f32":
        count = int(args[0]) * int(args[1])
    threads = 128
    _kernel(cache, name)(((count + threads - 1) // threads,), (threads,), args)


def _validate_scalar_names(names: Sequence[str], count: int, *,
                           registry_scalar_names: Sequence[str] | None = None) -> tuple[str, ...]:
    """The block's names must be a DECLARED row, in that row's order.

    This replaced a refusal that admitted exactly WSM6's six.  It still
    refuses a block nobody declared -- which is the state substitution the
    old text was guarding against -- but it now asks the species-row table
    rather than one hardcoded tuple, so a scheme with a different inventory
    is admitted on its own declaration instead of being impossible.
    """

    result = tuple(str(name).strip().lower() for name in names)
    if count != len(result):
        raise ValueError(
            f"resident physics preparation was given {count} scalar slots "
            f"and {len(result)} names {result}"
        )
    if registry_scalar_names is None:
        species_row_for_names(result)
    else:
        # MPAS Registry.xml:1701-1754; mpas_atmphys_interface.F:284-305.
        # Common preparation copies every selected Registry scalar and uses
        # index_qv for thermodynamics; it does not sum moisture-loading mass.
        declared = tuple(registry_scalar_names)
        if result != declared or not declared or declared[0] != 'qv' or len(set(declared)) != len(declared):
            raise ValueError('resident scalar block differs from the active MPAS Registry declaration')
    return result


#: Prepared fields a column physics scheme cannot interpret at or below zero.
_PREP_STRICTLY_POSITIVE = (
    "pres_p", "pres2_p", "pres_hyd_p", "pres2_hyd_p", "p8w",
    "t_p", "t2_p", "t_phy", "th_p", "rho_p", "rho_dry", "psfc_p",
    "dz_p", "dz8w",
)


def _diagnose_prep_failure(
    cp: Any,
    output: Mapping[str, Any],
    noahmp_output: Mapping[str, Any],
    *,
    n_cells: int,
    regional_garbage_cell: int | None,
) -> str:
    """Name the field and column the preparation validation refused."""

    findings: list[str] = []
    for carrier in (output, noahmp_output):
        for name, array in carrier.items():
            if array is None:
                continue
            host = cp.asnumpy(array)
            bad = ~np.isfinite(host)
            if name in _PREP_STRICTLY_POSITIVE:
                bad = bad | (host <= np.float32(0.0))
            count = int(np.count_nonzero(bad))
            if count == 0:
                continue
            columns = np.unique(np.argwhere(bad)[:, -1])
            findings.append(
                f"{name}: {count} value(s) at {columns.size} column(s), "
                f"first column {int(columns[0])}, "
                f"range [{float(np.nanmin(host))}, {float(np.nanmax(host))}]"
            )
    if not findings:
        return (
            "every prepared field is finite and positive on the host, so the "
            "kernel refused a geometry or consistency law rather than a value "
            "-- read validate_prep_v841_f32 for the exact test"
        )
    note = ""
    if regional_garbage_cell is not None:
        note = (
            f"  (column {regional_garbage_cell} of {n_cells} is native's "
            f"garbage element, which this preparation fills from column "
            f"{regional_garbage_cell - 1}; a finding confined to it is a "
            f"defect in that fill, not in the model state)"
        )
    return "; ".join(findings) + note


def prepare_mpas_to_phys_cuda_v841(
    atmosphere: Any,
    *,
    scalar_names: Sequence[str],
    geometry: CudaMpasToPhysGeometryV841,
    kernel_cache: KernelCache,
    post_rk_wsm6: bool = False,
    phase2_only: bool = False,
    phase2_winds: bool = False,
    registry_scalar_names: Sequence[str] | None = None,
    regional_garbage_cell: int | None = None,
) -> CudaMpasToPhysColumnsV841:
    """Prepare one resident MPAS state for the aggregate column backend.

    ``phase2_only`` omits the NoahMP sounding and horizontal wind reconstruction
    after RK. Microphysics that consumes winds sets ``phase2_winds``. History
    capture uses the full default preparation. No omitted field is used by
    thermodynamic preparation or the post-RK scalar aliases.

    ``regional_garbage_cell`` names native's garbage element on a
    limited-area mesh -- the extra column an MPAS allocation always carries
    and never solves.  Passing it makes that column a copy of the last real
    one in the DERIVED view only; see the block beside the validation launch
    for why it is neither skipped nor left as the dycore's pool values.
    """

    if type(post_rk_wsm6) is not bool:
        raise TypeError("post_rk_wsm6 must be bool")
    if type(phase2_only) is not bool or type(phase2_winds) is not bool:
        raise TypeError("phase2_only and phase2_winds must be bool")
    if phase2_only and not post_rk_wsm6:
        raise ValueError("phase2_only requires post_rk_wsm6=True")
    prepare_winds = not phase2_only or phase2_winds
    prepare_noahmp = not phase2_only
    if geometry is None or not isinstance(geometry, CudaMpasToPhysGeometryV841):
        raise TypeError("a validated CudaMpasToPhysGeometryV841 is required")
    geometry.validate()
    _validate_kernel_cache(kernel_cache)
    if hasattr(atmosphere, "validate"):
        atmosphere.validate()
    cp = _cp()
    state = atmosphere.state
    saved = atmosphere.saved
    vertical = atmosphere.vertical
    reference = atmosphere.reference
    if np.dtype(state.dtype) != np.dtype(np.float32):
        raise TypeError("MPAS_to_physics CUDA preparation requires FP32 state")
    nlev, ncells = tuple(state.rho.shape)
    nedges = int(saved.normal_velocity.shape[1])
    if nlev < 2:
        raise ValueError("MPAS_to_physics requires at least two vertical levels")
    if (ncells, nedges) != (geometry.n_cells, geometry.n_edges):
        raise ValueError(
            "preparation geometry/state mismatch: "
            f"state={(ncells, nedges)}, geometry={(geometry.n_cells, geometry.n_edges)}"
        )
    species_names = _validate_scalar_names(
        scalar_names, int(state.scalars.shape[0]), registry_scalar_names=registry_scalar_names)

    mass_shape = (nlev, ncells)
    interface_shape = (nlev + 1, ncells)
    require_resident_array("state.rho", state.rho, dtype=np.float32, shape=mass_shape)
    require_resident_array(
        "state.scalars",
        state.scalars,
        dtype=np.float32,
        shape=(len(species_names), nlev, ncells),
    )
    for name, value in (
        ("saved.theta_m", saved.theta_m),
        ("saved.exner", saved.exner),
        ("saved.pressure_perturbation", saved.pressure_perturbation),
        ("vertical.zz", vertical.zz),
        ("reference.pressure_base", reference.pressure_base),
    ):
        require_resident_array(name, value, dtype=np.float32, shape=mass_shape)
    require_resident_array(
        "saved.normal_velocity",
        saved.normal_velocity,
        dtype=np.float32,
        shape=(nlev, nedges),
    )
    require_resident_array(
        "saved.vertical_velocity",
        saved.vertical_velocity,
        dtype=np.float32,
        shape=interface_shape,
    )
    require_resident_array(
        "vertical.zgrid", vertical.zgrid, dtype=np.float32, shape=interface_shape
    )

    output: dict[str, Any] = {
        name: (None if name in ("u_p", "v_p") and not prepare_winds
               else cp.empty(mass_shape, dtype=cp.float32))
        for name in _mass_fields(species_names)
    }
    output.update(
        {name: cp.empty(interface_shape, dtype=cp.float32) for name in _INTERFACE_FIELDS}
    )
    output.update(
        {name: cp.empty((ncells,), dtype=cp.float32) for name in _SURFACE_FIELDS}
    )
    noahmp_output = {
        name: cp.empty(mass_shape, dtype=cp.float32) if prepare_noahmp else None
        for name in ("dz8w", "qv_curr", "t_phy", "u_phy", "v_phy", "p8w")
    }
    reconstructed_x = cp.empty(mass_shape, dtype=cp.float32) if prepare_winds else None
    reconstructed_y = cp.empty(mass_shape, dtype=cp.float32) if prepare_winds else None
    reconstructed_z = cp.empty(mass_shape, dtype=cp.float32) if prepare_winds else None
    invalid = cp.zeros((1,), dtype=cp.int32)

    # THE LIMITED-AREA GARBAGE COLUMN, LENT AND GIVEN BACK.
    #
    # Native allocates nCells+1 and runs its column physics over
    # 1..nCellsSolve; the extra element exists so an absent neighbour has
    # somewhere to point.  A device launch cannot skip its last element, so
    # the port's garbage column goes through every kernel below -- and the
    # DYCORE's pool values are not a column.  rho_zz is 1.0 there because
    # native writes 1.0 there (atm_recover_large_step_variables, F:4385-4392)
    # while rho_theta, exner and zgrid are the pool zero, so the column
    # arrives as a 0 K, zero-pressure, zero-thickness atmosphere.
    #
    # MEASURED on r4.75.11020 (2026-08-26): validate_prep_v841_f32 refuses it,
    # and it is right to.  Two of its laws bite and they bite in opposite
    # directions, which is why nothing done to the DERIVED arrays alone can
    # satisfy both: it requires rho_dry/th_p/t_p/pres_p strictly positive
    # (kernel :1988-1992), and it requires every derived value to be the
    # BITWISE function of the SOURCE state it came from (:1970-1987).  Pool
    # zeros fail the first.  Copying a real column into the derived arrays
    # fails the second.
    #
    # So the source column is lent a real column for the duration of the
    # preparation and given back before this function returns.  Everything
    # downstream is then consistent by construction, because it was DERIVED
    # from a well-posed column rather than patched afterwards, and the
    # dycore's state is bit-identical on the way out -- restored in a finally,
    # so a refusal cannot leave it lent.  Nothing real reads the garbage
    # column's physics: the only path from it into a real element is the
    # two-cell average in project_edges_v841_f32 at a ring-7 one-cell edge,
    # and every ring-7 edge is a SPECIFIED-zone edge whose momentum is
    # overwritten with the driving value in the same step (F:2442-2485).
    lent: list[tuple[Any, Any]] = []
    if regional_garbage_cell is None:
        regional_garbage_cell = geometry.regional_garbage_cell
    if regional_garbage_cell is not None:
        garbage = int(regional_garbage_cell)
        if garbage != ncells - 1:
            raise ValueError(
                "the regional garbage column is the LAST element of the "
                f"padded extent; got {garbage} with {ncells} columns"
            )
        for source in (
            state.rho,
            state.scalars,
            saved.theta_m,
            saved.exner,
            saved.pressure_perturbation,
            saved.vertical_velocity,
            reference.pressure_base,
            vertical.zz,
            vertical.zgrid,
        ):
            lent.append((source, source[..., garbage].copy()))
            donor = garbage - 1 if geometry.regional_garbage_donor_cell is None else int(geometry.regional_garbage_donor_cell)
            source[..., garbage] = source[..., donor]
    try:
        return _prepare_mpas_to_phys_body_v841(
            cp=cp,
            atmosphere=atmosphere,
            state=state,
            saved=saved,
            vertical=vertical,
            reference=reference,
            geometry=geometry,
            kernel_cache=kernel_cache,
            scalar_names=scalar_names,
            post_rk_wsm6=post_rk_wsm6,
            prepare_winds=prepare_winds,
            prepare_noahmp=prepare_noahmp,
            regional_garbage_cell=regional_garbage_cell,
            nlev=nlev,
            ncells=ncells,
            nedges=nedges,
            output=output,
            noahmp_output=noahmp_output,
            reconstructed_x=reconstructed_x,
            reconstructed_y=reconstructed_y,
            reconstructed_z=reconstructed_z,
            invalid=invalid,
        )
    finally:
        for source, column in lent:
            source[..., int(regional_garbage_cell)] = column


def _prepare_mpas_to_phys_body_v841(
    *,
    cp: Any,
    atmosphere: Any,
    state: Any,
    saved: Any,
    vertical: Any,
    reference: Any,
    geometry: CudaMpasToPhysGeometryV841,
    kernel_cache: KernelCache,
    scalar_names: Sequence[str],
    post_rk_wsm6: bool,
    prepare_winds: bool,
    prepare_noahmp: bool,
    regional_garbage_cell: int | None,
    nlev: int,
    ncells: int,
    nedges: int,
    output: dict[str, Any],
    noahmp_output: dict[str, Any],
    reconstructed_x: Any,
    reconstructed_y: Any,
    reconstructed_z: Any,
    invalid: Any,
) -> CudaMpasToPhysColumnsV841:
    """The launches themselves, split out so the lend above owns a finally."""

    species_names = tuple(scalar_names)
    species_outputs = frozenset(f"{name}_p" for name in species_names)
    species_arrays = [output[f"{name}_p"] for name in species_names]
    species_table = _device_pointer_table(cp, species_arrays)

    _launch(
        kernel_cache,
        "prep_mass_v841_f32",
        ncells,
        (
            np.int32(nlev),
            np.int32(ncells),
            np.int32(len(species_names)),
            RV_OVER_RD_F32,
            state.rho,
            saved.theta_m,
            state.scalars,
            saved.exner,
            reference.pressure_base,
            saved.pressure_perturbation,
            vertical.zgrid,
            vertical.zz,
            species_table,
            output["zz_p"],
            output["rho_dry"],
            output["rho_p"],
            output["th_p"],
            output["t_p"],
            output["pi_p"],
            output["pres_p"],
            output["zmid_p"],
            output["dz_p"],
            invalid,
        ),
    )
    if prepare_winds:
        _launch(
            kernel_cache,
            "reconstruct_winds_v841_f32",
            ncells,
            (
                np.int32(nlev),
                np.int32(ncells),
                np.int32(nedges),
                np.int32(geometry.max_edges),
                saved.normal_velocity,
                geometry._edges_on_cell,
                geometry._n_edges_on_cell,
                geometry._coeffs_reconstruct,
                geometry._lat_cell,
                geometry._lon_cell,
                geometry._sealed_edges_on_cell,
                geometry._sealed_n_edges_on_cell,
                geometry._sealed_coeffs_reconstruct,
                geometry._sealed_lat_cell,
                geometry._sealed_lon_cell,
                reconstructed_x,
                reconstructed_y,
                reconstructed_z,
                output["u_p"],
                output["v_p"],
                invalid,
            ),
        )
        if regional_garbage_cell is not None:
            # The garbage column's WIND cannot be reconstructed, and lending it a
            # thermodynamic column does not fix that.  ``reconstruct_winds`` sums
            # over ``edgesOnCell[cell, :nEdgesOnCell[cell]]``, and the garbage
            # cell has nEdgesOnCell = 0 by construction -- it is not on the mesh,
            # so it has no edges to gather a velocity from.  It therefore comes
            # out of that kernel at exactly u = v = 0 while carrying a real
            # column's temperature, moisture and pressure.
            #
            # THE BREAKAGE THAT NAMED THIS, MEASURED on r4.75.11020 (2026-08-26):
            # a column with a realistic sounding and identically zero wind at
            # every level has zero shear everywhere, and YSU forms a bulk
            # Richardson number over it.  The limited-area full-physics forecast
            # committed steps 1 and 2 and refused step 3 with "YSU returned
            # non-finite du tendency on domain 1; every producer input was finite
            # and in range, so the scheme's own arithmetic produced it" -- which
            # was true, and the input that produced it was a wind field no cell
            # has.  The wind is lent from the same real column as the rest of the
            # sounding, so the garbage column is coherent rather than half real,
            # and the NoahMP sounding below derives u_phy/v_phy from it, which
            # keeps the validation's derived-vs-derived equalities exact.
            garbage_column = int(regional_garbage_cell)
            donor = garbage_column - 1 if geometry.regional_garbage_donor_cell is None else int(geometry.regional_garbage_donor_cell)
            for name in ("u_p", "v_p"):
                output[name][..., garbage_column] = output[name][
                    ..., donor
                ]
    if prepare_noahmp:
        _launch(
            kernel_cache,
            "prep_noahmp_sounding_v841_f32",
            ncells,
            (
                np.int32(nlev),
                np.int32(ncells),
                RV_OVER_RD_F32,
                state.scalars,
                saved.theta_m,
                saved.exner,
                reference.pressure_base,
                saved.pressure_perturbation,
                vertical.zgrid,
                output["u_p"],
                output["v_p"],
                noahmp_output["dz8w"],
                noahmp_output["qv_curr"],
                noahmp_output["t_phy"],
                noahmp_output["u_phy"],
                noahmp_output["v_phy"],
                noahmp_output["p8w"],
                invalid,
            ),
        )
    _launch(
        kernel_cache,
        "prep_interfaces_v841_f32",
        ncells,
        (
            np.int32(nlev),
            np.int32(ncells),
            GRAVITY_F32,
            state.rho,
            vertical.zz,
            output["qv_p"],
            output["pres_p"],
            output["t_p"],
            output["rho_p"],
            output["dz_p"],
            vertical.zgrid,
            saved.vertical_velocity,
            output["w_p"],
            output["z_p"],
            output["t2_p"],
            output["pres2_p"],
            output["pres_hyd_p"],
            output["pres2_hyd_p"],
            output["pres_hydd_p"],
            output["pres2_hydd_p"],
            output["psfc_p"],
            output["psfc_hyd_p"],
            output["psfc_hydd_p"],
            output["znu_p"],
            output["znu_hyd_p"],
            output["plrad"],
            invalid,
        ),
    )
    _launch(
        kernel_cache,
        "validate_prep_v841_f32",
        ncells,
        (
            np.int32(nlev),
            np.int32(ncells),
            np.int32(nedges),
            np.int32(geometry.max_edges),
            np.int32(1 if post_rk_wsm6 else 0),
            np.int32(prepare_winds),
            np.int32(prepare_noahmp),
            GRAVITY_F32,
            RV_OVER_RD_F32,
            geometry._edges_on_cell,
            geometry._n_edges_on_cell,
            geometry._coeffs_reconstruct,
            geometry._lat_cell,
            geometry._lon_cell,
            state.scalars,
            saved.theta_m,
            saved.exner,
            np.int32(len(species_names)),
            species_table,
            *(output[name] if output[name] is not None else np.uint64(0)
              for name in _MASS_NONSPECIES_FIELDS),
            *(output[name] for name in _INTERFACE_FIELDS),
            *(output[name] for name in _SURFACE_FIELDS),
            *(noahmp_output[name] if prepare_noahmp else np.uint64(0) for name in (
                "dz8w", "qv_curr", "t_phy", "u_phy", "v_phy", "p8w"
            )),
            invalid,
        ),
    )
    started = time.perf_counter()
    invalid_value = int(cp.asnumpy(invalid)[0])
    validation = TransferStats(int(invalid.nbytes), time.perf_counter() - started)
    if invalid_value != 0:
        # A refusal has to name what it refused.  The kernel sets one flag for
        # the whole launch, so the flag alone says "some column, some level,
        # some field" -- which is what a caller used to get, and it is not
        # actionable.  This runs ONLY on the failure path, so it costs a
        # passing step nothing, and it reports the offending field, the worst
        # column and whether that column is the limited-area garbage element.
        raise FloatingPointError(
            "MPAS_to_physics CUDA preparation failed numeric/geometry "
            "validation: "
            + _diagnose_prep_failure(
                cp,
                output,
                noahmp_output,
                n_cells=ncells,
                regional_garbage_cell=regional_garbage_cell,
            )
        )
    result = CudaMpasToPhysColumnsV841(
        species_p={
            name: output[f"{name}_p"] for name in species_names
        },
        **{
            name: value for name, value in output.items()
            if name not in species_outputs
        },
        noahmp_sounding=(CudaNoahmpSoundingV841(**noahmp_output)
                         if prepare_noahmp else None),
        wsm6_ready=post_rk_wsm6,
        _source_scalars=state.scalars,
        time_seconds=float(state.time_seconds),
        validation_d2h=validation,
        geometry_receipt=MappingProxyType(geometry.receipt()),
    )
    result.validate()
    return result


__all__ = [
    "CUDA_PHYSICS_PREP_V841_CONTRACT_SHA256",
    "CUDA_PHYSICS_PREP_V841_EVIDENCE",
    "CUDA_PHYSICS_PREP_V841_KERNEL_SHA256",
    "CUDA_PHYSICS_PREP_V841_SCHEMA",
    "CpuNoahmpSoundingV841",
    "CpuMpasToPhysColumnsV841",
    "CpuWsm6InputViewV841",
    "CudaNoahmpSoundingV841",
    "CudaMpasToPhysColumnsV841",
    "CudaMpasToPhysGeometryV841",
    "CudaWsm6InputViewV841",
    "GRAVITY_F32",
    "MPAS_ATMPHYS_INTERFACE_V841_SHA256",
    "MPAS_NOAHMP_SOUNDING_V841_SHA256",
    "MPAS_VECTOR_RECONSTRUCTION_V841_SHA256",
    "RD_F32",
    "RV_F32",
    "RV_OVER_RD_F32",
    "WSM6_SCALAR_NAMES",
    "physics_prep_contract_evidence_v841",
    "prepare_mpas_to_phys_cpu_oracle_v841",
    "prepare_noahmp_sounding_cpu_oracle_v841",
    "prepare_mpas_to_phys_cuda_v841",
    "reconstruct_winds_cpu_oracle_v841",
]


_MPAS_NUMBER_CARRIERS_V842_SOURCE = r'''
extern "C" {
// mpas_atmphys_interface.F:357-425. Input words are the existing max(+0,q)
// preparation outputs; select-case order and MYNN's zeroing are preserved.
__global__ void prep_numbers_v842_u32(
    int nlev, int ncells, int solved, int aerosol, int mynn, int ice, int trio,
    const unsigned int *nc, const unsigned int *ni,
    const unsigned int *nifa, const unsigned int *nwfa,
    unsigned int *out_nc, unsigned int *out_ni,
    unsigned int *out_nifa, unsigned int *out_nwfa)
{
    const int cell = blockDim.x * blockIdx.x + threadIdx.x;
    if (cell >= solved) return;
    for (int k=0; k<nlev; ++k) {
        const int i=k*ncells+cell;
        if (aerosol) { out_nifa[i]=nifa[i]; out_nwfa[i]=nwfa[i]; }
        if (mynn) {
            out_nc[i]=0u; out_ni[i]=0u; out_nifa[i]=0u; out_nwfa[i]=0u;
            if (ice) out_ni[i]=ni[i];
            if (trio) { out_nc[i]=nc[i]; out_nifa[i]=nifa[i]; out_nwfa[i]=nwfa[i]; }
        }
    }
}
}
'''
MPAS_NUMBER_CARRIERS_V842_KERNEL_SHA256 = sha256(
    _MPAS_NUMBER_CARRIERS_V842_SOURCE.encode('ascii')).hexdigest()


def prepare_mpas_number_carriers_v842(
    prepared: CudaMpasToPhysColumnsV841, *, microphysics: str, pbl: str,
    scalar_flags: Mapping[str, bool], n_cells_solve: int,
    kernel_cache: KernelCache,
) -> Mapping[str, Any]:
    """Native number scratch selection; mpas_atmphys_interface.F:357-425.

    The generalized preparation already supplies max(+0,q) for every species
    in its row. Reuse those words and apply only the native allocation/flag
    selection. Returned arrays are distinct scratch, never prognostic state.
    MYNN resets aerosol scratch before its grouped flag test. Unallocated
    native number fields are absent from the returned mapping.
    """
    if microphysics not in ('off','mp_kessler','mp_wsm6','mp_thompson','mp_thompson_aerosols'):
        raise ValueError(f'unknown MPAS microphysics selector {microphysics!r}')
    if pbl not in ('off','bl_ysu','bl_mynn'):
        raise ValueError(f'unknown MPAS PBL selector {pbl!r}')
    _validate_kernel_cache(kernel_cache)
    cp = _cp()
    template = prepared.species_p['qv']
    nlev, ncells = map(int, template.shape)
    if type(n_cells_solve) is not int or not 0 < n_cells_solve <= ncells:
        raise ValueError('number scratch requires a nonempty native solve extent')
    aerosol, mynn = microphysics == 'mp_thompson_aerosols', pbl == 'bl_mynn'
    names = ('nc','ni','nifa','nwfa')
    for name in names:
        if type(scalar_flags.get(name,False)) is not bool:
            raise TypeError(f'native f_{name} requires bool')
    ice = scalar_flags.get('ni',False)
    trio = all(scalar_flags.get(name,False) for name in ('nc','nifa','nwfa'))
    required = set(('nifa','nwfa')) if aerosol else set()
    if mynn:
        if ice: required.add('ni')
        if trio: required.update(('nc','nifa','nwfa'))
    missing = required.difference(prepared.species_p)
    if missing:
        raise ValueError(f'native number scratch would read absent species: {sorted(missing)}')
    allocated = names if mynn else ('nifa','nwfa') if aerosol else ()
    if not allocated:
        return MappingProxyType({})
    output = {name:cp.zeros_like(template) for name in allocated}
    # Unused pointers are never dereferenced by the selected native branches.
    inputs = [prepared.species_p.get(name,template) for name in names]
    for name,array in zip(names,inputs):
        require_resident_array(f'{name}_p', array, ndim=2, dtype=np.float32,
                               shape=(nlev,ncells))
    targets = [output.get(name,template) for name in names]
    kernel = kernel_cache.raw_kernel(
        'prep_numbers_v842_u32', _MPAS_NUMBER_CARRIERS_V842_SOURCE,
        module_key='hexcore.mpasphys.number_carriers_v842')
    kernel(
        ((n_cells_solve+127)//128,),(128,),
        (np.int32(nlev),np.int32(ncells),np.int32(n_cells_solve),np.int32(aerosol),
         np.int32(mynn),np.int32(ice),np.int32(trio),*inputs,*targets))
    return MappingProxyType(output)
