"""One-time host/device transfer containers for the CUDA atmosphere port."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache
from operator import attrgetter, is_
import time
from typing import Any, Mapping

import numpy as np

from ..host_parallel import all_finite, contiguous_copy
from ..moving_upload_cache import ACTIVE as _UPLOAD_CACHE, upload as _upload

from .compaction import (
    compact_host,
    compaction_enabled,
    restore_device,
    uniform_tail,
    used_width,
)
from .runtime import CudaRefusal, require_cuda


def _cp() -> Any:
    import cupy as cp

    return cp


def require_resident_array(
    name: str,
    value: Any,
    *,
    dtype: Any | None = None,
    shape: tuple[int, ...] | None = None,
    ndim: int | None = None,
) -> Any:
    """Return one packed device array or refuse implicit transfer/layout drift.

    MPAS CUDA kernels address logical ``[level, entity]`` fields as
    ``level * n_entity + entity``.  Accepting a NumPy array, a strided CuPy
    view, or a different dtype would make that raw indexing silently wrong.
    Public resident APIs therefore share this exact fail-closed boundary.
    """

    cp = _cp()
    if not isinstance(value, cp.ndarray):
        raise TypeError(f"{name} must be a resident cupy.ndarray")
    if dtype is not None and value.dtype != cp.dtype(dtype):
        raise TypeError(f"{name} must have dtype {cp.dtype(dtype)}")
    if shape is not None and tuple(value.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {value.ndim}")
    if not bool(value.flags.c_contiguous):
        raise ValueError(f"{name} must be C-contiguous with entities fastest")
    return value


def _mesh_value(mesh: object, name: str, default: Any = None) -> Any:
    if hasattr(mesh, name):
        return getattr(mesh, name)
    arrays = getattr(mesh, "arrays", None)
    if isinstance(arrays, Mapping) and name in arrays:
        return arrays[name]
    attrs = getattr(mesh, "attrs", None)
    if isinstance(attrs, Mapping) and name in attrs:
        return attrs[name]
    return default


def _as_float(name: str, value: Any, dtype: np.dtype[Any]) -> np.ndarray:
    array = contiguous_copy(np.asarray(value), dtype=np.dtype(dtype))
    if not all_finite(array):
        raise ValueError(f"{name} contains non-finite values")
    return array


@lru_cache(maxsize=2)
def _finite_word_kernel(word_bytes):
    """Flag IEEE infinity/NaN words without a mask allocation or reduction JIT."""
    word, mask = {
        4: ("unsigned int", "0x7f800000U"),
        8: ("unsigned long long", "0x7ff0000000000000ULL"),
    }[word_bytes]
    name = f"hex_nonfinite_words_{word_bytes}"
    return _cp().RawKernel(
        f'''extern "C" __global__ void {name}(
            const {word}* words, unsigned long long count,
            unsigned int* flags, unsigned int field) {{
            unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
            unsigned long long stride = (unsigned long long)blockDim.x * gridDim.x;
            for (; i < count; i += stride) {{
                if ((words[i] & {mask}) == {mask}) {{
                    atomicExch(flags + field, 1U);
                    return;
                }}
            }}
        }}''',
        name,
    )


def _device_float_fields(cp, owner, names, dtype, *, exact_dtype=False, finite=True):
    """Upload source words, then pack and check them on the GPU.

    Keep a contiguous source's C or Fortran order during transfer. Packing a
    transposed vertical field on the host first otherwise leaves the GPU
    waiting for one NumPy loop. Arbitrary strides use bounded host copy
    workers. Precision conversions use bounded NumPy workers too: ordinary
    CuPy casts flush subnormal words. Authority checks precede conversion.
    """
    device, checks = {}, []
    word_flags = (cp.zeros(len(names), dtype=cp.uint32)
                  if finite and dtype in (np.dtype(np.float32), np.dtype(np.float64))
                  else None)
    for index, name in enumerate(names):
        source = np.asarray(getattr(owner, name))
        if exact_dtype and source.dtype != dtype:
            raise TypeError(f"{name} must have dtype {dtype}, got {source.dtype}")
        if source.dtype != dtype or not (source.flags.c_contiguous or source.flags.f_contiguous):
            source = contiguous_copy(source, dtype=dtype)
        uploaded = _upload(cp, source, name)
        value = cp.ascontiguousarray(uploaded, dtype=dtype)
        device[name] = value
        if word_flags is not None and value.size:
            _finite_word_kernel(dtype.itemsize)(
                (min(65535, (value.size + 255) // 256),), (256,),
                (value, np.uint64(value.size), word_flags, np.uint32(index)),
            )
        elif finite and word_flags is None:
            checks.append(cp.isfinite(value).all())
    if word_flags is not None or checks:
        # One small result transfer, after every field's device scan.
        valid = cp.asnumpy(word_flags) == 0 if word_flags is not None else cp.asnumpy(cp.stack(checks))
        for name, ok in zip(names, valid):
            if not ok:
                raise ValueError(f"{name} contains non-finite values")
    return device


def _as_index(name: str, value: Any, dtype: np.dtype[Any]) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "iu":
        raise TypeError(f"{name} must be integer")
    info = np.iinfo(dtype)
    if array.size and (np.min(array) < info.min or np.max(array) > info.max):
        raise OverflowError(f"{name} cannot be represented by {dtype}")
    return np.ascontiguousarray(array, dtype=dtype)


def _device_edge_signs(cp, edges, endpoints, counts, dtype):
    """Construct signs on the GPU, preserving +0 in every inactive slot."""
    signs = cp.empty(edges.shape, dtype=dtype)
    cp.ElementwiseKernel(
        "raw I edges, raw I endpoints, raw I counts, int64 width",
        "T signs",
        """long long cell = i / width;
           long long slot = i % width;
           signs = slot < counts[cell]
               ? (endpoints[2 * edges[i]] == cell ? (T)1 : (T)-1)
               : (T)0;""",
        "hex_mesh_edge_signs",
    )(edges, endpoints, counts, edges.shape[1], signs)
    return signs


@dataclass(frozen=True, slots=True)
class TransferStats:
    bytes: int
    seconds: float

    @property
    def gib_per_second(self) -> float:
        return self.bytes / max(self.seconds, np.finfo(np.float64).tiny) / 2**30

    def as_dict(self) -> dict[str, float | int]:
        return {
            "bytes": self.bytes,
            "seconds": self.seconds,
            "gib_per_second": self.gib_per_second,
        }


#: Every resident array DeviceMesh.validate checks, in its check order.
_MESH_ARRAY_NAMES = (
    "cells_on_edge", "edges_on_cell", "n_edges_on_cell", "cells_on_cell",
    "vertices_on_edge", "edges_on_edge", "n_edges_on_edge", "vertices_on_cell",
    "edges_on_vertex", "cells_on_vertex", "edge_sign_on_cell", "weights_on_edge",
    "dc_edge", "dv_edge", "area_cell", "area_triangle", "kite_areas_on_vertex",
    "lat_cell", "lon_cell", "lat_edge", "lon_edge", "angle_edge", "mesh_density",
    "defc_a", "defc_b", "f_vertex", "f_edge", "spec_zone_mask_edge",
)
_MESH_ARRAYS = attrgetter(*_MESH_ARRAY_NAMES)
_SHAPE = attrgetter("shape")


@dataclass(slots=True)
class DeviceMesh:
    cells_on_edge: Any
    edges_on_cell: Any
    n_edges_on_cell: Any
    cells_on_cell: Any
    vertices_on_edge: Any
    edges_on_edge: Any
    n_edges_on_edge: Any
    vertices_on_cell: Any
    edges_on_vertex: Any
    cells_on_vertex: Any
    edge_sign_on_cell: Any
    weights_on_edge: Any
    dc_edge: Any
    dv_edge: Any
    area_cell: Any
    area_triangle: Any
    kite_areas_on_vertex: Any
    lat_cell: Any
    lon_cell: Any
    lat_edge: Any
    lon_edge: Any
    angle_edge: Any
    mesh_density: Any
    defc_a: Any
    defc_b: Any
    f_vertex: Any
    f_edge: Any
    spec_zone_mask_edge: Any
    n_cells: int
    n_edges: int
    n_vertices: int
    max_edges: int
    max_edges2: int
    vertex_degree: int
    nominal_min_dc: float
    dtype: np.dtype[Any]
    index_dtype: np.dtype[Any]
    h2d: TransferStats
    #: ``maxEdges2`` of the host mesh.  ``max_edges2`` is the stride the
    #: device ``edges_on_edge``/``weights_on_edge`` are stored with, which is
    #: smaller when the edge stencil is compacted (``compaction.py``); every
    #: kernel takes ``max_edges2`` as its stride.  0 means "not compacted".
    source_max_edges2: int = 0
    edges_on_edge_fill: int = 0
    weights_on_edge_fill: float = 0.0
    #: Compacted ``edgesOnCell``/``edgesOnCell_sign`` per slot stride, built
    #: on first request by :meth:`slot_tables` (the terrain metrics use them).
    _slot_tables: dict = field(default_factory=dict, repr=False, compare=False)
    #: What :meth:`validate` last admitted: the layout scalars, the very
    #: arrays it checked (held, so a reused address can never match) and
    #: their shapes.  See :meth:`validate`.
    _validated: Any = field(default=None, init=False, repr=False, compare=False)

    @property
    def file_max_edges2(self) -> int:
        return int(self.source_max_edges2 or self.max_edges2)

    def full_edge_stencil(self) -> tuple[Any, Any]:
        """``edges_on_edge`` and ``weights_on_edge`` in the host's padded layout.

        Equal to the device arrays when nothing was compacted; otherwise a
        transient rebuild whose padding is the exact constant the host held.
        """

        width = self.file_max_edges2
        return (
            restore_device(self.edges_on_edge, width, self.edges_on_edge_fill, axis=1),
            restore_device(
                self.weights_on_edge, width, self.weights_on_edge_fill, axis=1
            ),
        )

    def slot_tables(self, stride: int) -> tuple[Any, Any]:
        """``edges_on_cell`` and ``edge_sign_on_cell`` at slot stride ``stride``.

        Readers loop ``slot < n_edges_on_cell[cell]``; a stride no smaller than
        the largest count reads the same words as the full table.
        """

        stride = int(stride)
        if stride == int(self.max_edges):
            return self.edges_on_cell, self.edge_sign_on_cell
        if not 1 <= stride < int(self.max_edges):
            raise ValueError(
                f"slot stride {stride} is outside [1, maxEdges={self.max_edges}]"
            )
        # Copies of this mesh (multi-member sharing, dataclasses.replace) share
        # the cache; an entry is used only for the very arrays it was cut
        # from, and holds them, so a reused address can never match.
        key = (stride, id(self.edges_on_cell), id(self.edge_sign_on_cell))
        cached = self._slot_tables.get(key)
        if (
            cached is None
            or cached[0] is not self.edges_on_cell
            or cached[1] is not self.edge_sign_on_cell
        ):
            cp = _cp()
            cached = (
                self.edges_on_cell,
                self.edge_sign_on_cell,
                cp.ascontiguousarray(self.edges_on_cell[:, :stride]),
                cp.ascontiguousarray(self.edge_sign_on_cell[:, :stride]),
            )
            self._slot_tables.pop(key, None)
            while len(self._slot_tables) >= 8:
                self._slot_tables.pop(next(iter(self._slot_tables)))
            self._slot_tables[key] = cached
        return cached[2], cached[3]

    @classmethod
    def from_host(
        cls,
        mesh: object,
        *,
        dtype: Any = np.float32,
        index_dtype: Any = np.int32,
    ) -> "DeviceMesh":
        require_cuda()
        cp = _cp()
        float_dtype = np.dtype(dtype)
        integer_dtype = np.dtype(index_dtype)
        if float_dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise TypeError("DeviceMesh dtype must be float32 or float64")
        if integer_dtype not in (np.dtype(np.int32), np.dtype(np.int64)):
            raise TypeError("DeviceMesh index_dtype must be int32 or int64")
        dimensions = getattr(mesh, "dimensions", {})
        n_cells = int(
            dimensions.get("nCells", np.asarray(_mesh_value(mesh, "latCell")).size)
        )
        n_edges = int(
            dimensions.get("nEdges", np.asarray(_mesh_value(mesh, "dcEdge")).size)
        )
        n_vertices = int(
            dimensions.get(
                "nVertices", np.asarray(_mesh_value(mesh, "areaTriangle")).size
            )
        )
        max_edges = int(
            dimensions.get(
                "maxEdges", np.asarray(_mesh_value(mesh, "edgesOnCell")).shape[1]
            )
        )
        max_edges2 = int(
            dimensions.get(
                "maxEdges2", np.asarray(_mesh_value(mesh, "edgesOnEdge")).shape[1]
            )
        )
        vertex_degree = int(
            dimensions.get(
                "vertexDegree", np.asarray(_mesh_value(mesh, "cellsOnVertex")).shape[1]
            )
        )
        index_names = (
            "cellsOnEdge",
            "edgesOnCell",
            "nEdgesOnCell",
            "cellsOnCell",
            "verticesOnEdge",
            "edgesOnEdge",
            "nEdgesOnEdge",
            "verticesOnCell",
            "edgesOnVertex",
            "cellsOnVertex",
        )
        host_index = {
            name: _as_index(name, _mesh_value(mesh, name), integer_dtype)
            for name in index_names
        }
        float_map = {
            "weights_on_edge": "weightsOnEdge",
            "dc_edge": "dcEdge",
            "dv_edge": "dvEdge",
            "area_cell": "areaCell",
            "area_triangle": "areaTriangle",
            "kite_areas_on_vertex": "kiteAreasOnVertex",
            "lat_cell": "latCell",
            "lon_cell": "lonCell",
            "lat_edge": "latEdge",
            "lon_edge": "lonEdge",
            "angle_edge": "angleEdge",
            "mesh_density": "meshDensity",
            "defc_a": "defc_a",
            "defc_b": "defc_b",
            "f_vertex": "fVertex",
            "f_edge": "fEdge",
        }
        host_float = {
            target: _as_float(source, _mesh_value(mesh, source), float_dtype)
            for target, source in float_map.items()
        }
        raw_zone = _mesh_value(mesh, "spec_zone_mask_edge")
        if raw_zone is None:
            raw_zone = np.zeros(n_edges, dtype=float_dtype)
        host_float["spec_zone_mask_edge"] = _as_float(
            "spec_zone_mask_edge", raw_zone, float_dtype
        )
        nominal = _mesh_value(mesh, "nominalMinDc")
        if nominal is None:
            nominal = _mesh_value(mesh, "nominal_min_dc")
        if nominal is None or not np.isfinite(float(np.asarray(nominal))):
            raise CudaRefusal(
                "mesh.nominalMinDc is required by CUDA scale-aware filters"
            )
        # Edge stencil at the stride the mesh uses (compaction.py): readers
        # loop slot < nEdgesOnEdge[edge] with max_edges2 as the stride.
        source_max_edges2 = 0
        eoe_fill: Any = 0
        woe_fill: Any = 0.0
        eoe_width = int(host_index["edgesOnEdge"].shape[1])
        if compaction_enabled("EDGE_STENCIL") and eoe_width == max_edges2 and (
            host_float["weights_on_edge"].shape == host_index["edgesOnEdge"].shape
        ):
            keep = used_width(host_index["nEdgesOnEdge"], eoe_width)
            if keep < eoe_width:
                eoe_tail = uniform_tail(host_index["edgesOnEdge"], keep, axis=1)
                woe_tail = uniform_tail(host_float["weights_on_edge"], keep, axis=1)
                if eoe_tail is not None and woe_tail is not None:
                    host_index["edgesOnEdge"] = compact_host(
                        host_index["edgesOnEdge"], keep, axis=1
                    )
                    host_float["weights_on_edge"] = compact_host(
                        host_float["weights_on_edge"], keep, axis=1
                    )
                    source_max_edges2 = max_edges2
                    max_edges2 = keep
                    eoe_fill = int(eoe_tail)
                    woe_fill = float(woe_tail)
        started = time.perf_counter()
        device_index = {name: _upload(cp, value, name) for name, value in host_index.items()}
        device_float = {name: _upload(cp, value, name) for name, value in host_float.items()}
        device_signs = _device_edge_signs(cp, device_index["edgesOnCell"],
                                          device_index["cellsOnEdge"],
                                          device_index["nEdgesOnCell"], float_dtype)
        cp.cuda.get_current_stream().synchronize()
        elapsed = time.perf_counter() - started
        # Signs were constructed on device and did not cross the host bus.
        arrays = (*device_index.values(), *device_float.values())
        byte_count = int(sum(int(value.nbytes) for value in arrays))
        return cls(
            cells_on_edge=device_index["cellsOnEdge"],
            edges_on_cell=device_index["edgesOnCell"],
            n_edges_on_cell=device_index["nEdgesOnCell"],
            cells_on_cell=device_index["cellsOnCell"],
            vertices_on_edge=device_index["verticesOnEdge"],
            edges_on_edge=device_index["edgesOnEdge"],
            n_edges_on_edge=device_index["nEdgesOnEdge"],
            vertices_on_cell=device_index["verticesOnCell"],
            edges_on_vertex=device_index["edgesOnVertex"],
            cells_on_vertex=device_index["cellsOnVertex"],
            edge_sign_on_cell=device_signs,
            **device_float,
            n_cells=n_cells,
            n_edges=n_edges,
            n_vertices=n_vertices,
            max_edges=max_edges,
            max_edges2=max_edges2,
            vertex_degree=vertex_degree,
            nominal_min_dc=float(np.asarray(nominal)),
            dtype=float_dtype,
            index_dtype=integer_dtype,
            h2d=TransferStats(byte_count, elapsed),
            source_max_edges2=source_max_edges2,
            edges_on_edge_fill=eoe_fill,
            weights_on_edge_fill=woe_fill,
        )

    def validate(self) -> None:
        """Validate the committed packed topology used by raw CUDA indexing.

        The full check runs once per mesh; afterwards a call only confirms
        that the mesh still holds the very arrays it admitted, with the same
        shapes and layout scalars, and returns.  CuPy arrays cannot change
        type or dtype in place, and their contiguity can change only through
        a shape change, so an identical array with an identical shape still
        passes every check below; anything else is checked again in full.
        THE BREAKAGE THIS PREVENTS: the dycore calls this on every theta, w
        and momentum launch and the acoustic and horizontal paths do too,
        re-checking 28 resident arrays each time; on 8 same-host ranks
        (r700, 36k cells per card) that host work was about a third of each
        rank thread's samples (VRFS vrfs-8card-diag, 2026-10-28), while the
        cards sat 22-34 % busy.
        """

        arrays = _MESH_ARRAYS(self)
        layout = (
            self.n_cells, self.n_edges, self.n_vertices, self.max_edges,
            self.max_edges2, self.vertex_degree, self.dtype, self.index_dtype,
        )
        seen = self._validated
        if (
            seen is not None
            and seen[0] == layout
            and all(map(is_, arrays, seen[1]))
            and seen[2] == tuple(map(_SHAPE, arrays))
        ):
            return
        index_shapes = {
            "cells_on_edge": (self.n_edges, 2),
            "edges_on_cell": (self.n_cells, self.max_edges),
            "n_edges_on_cell": (self.n_cells,),
            "cells_on_cell": (self.n_cells, self.max_edges),
            "vertices_on_edge": (self.n_edges, 2),
            "edges_on_edge": (self.n_edges, self.max_edges2),
            "n_edges_on_edge": (self.n_edges,),
            "vertices_on_cell": (self.n_cells, self.max_edges),
            "edges_on_vertex": (self.n_vertices, self.vertex_degree),
            "cells_on_vertex": (self.n_vertices, self.vertex_degree),
        }
        for name, shape in index_shapes.items():
            require_resident_array(
                f"mesh.{name}", getattr(self, name), dtype=self.index_dtype, shape=shape
            )
        float_shapes = {
            "edge_sign_on_cell": (self.n_cells, self.max_edges),
            "weights_on_edge": (self.n_edges, self.max_edges2),
            "dc_edge": (self.n_edges,),
            "dv_edge": (self.n_edges,),
            "area_cell": (self.n_cells,),
            "area_triangle": (self.n_vertices,),
            "kite_areas_on_vertex": (self.n_vertices, self.vertex_degree),
            "lat_cell": (self.n_cells,),
            "lon_cell": (self.n_cells,),
            "lat_edge": (self.n_edges,),
            "lon_edge": (self.n_edges,),
            "angle_edge": (self.n_edges,),
            "mesh_density": (self.n_cells,),
            "defc_a": (self.n_cells, self.max_edges),
            "defc_b": (self.n_cells, self.max_edges),
            "f_vertex": (self.n_vertices,),
            "f_edge": (self.n_edges,),
            "spec_zone_mask_edge": (self.n_edges,),
        }
        for name, shape in float_shapes.items():
            require_resident_array(
                f"mesh.{name}", getattr(self, name), dtype=self.dtype, shape=shape
            )
        self._validated = (layout, arrays, tuple(map(_SHAPE, arrays)))

    def to_host(self) -> dict[str, np.ndarray | float | int]:
        cp = _cp()
        result: dict[str, np.ndarray | float | int] = {}
        for name in (
            "cells_on_edge",
            "edges_on_cell",
            "n_edges_on_cell",
            "cells_on_cell",
            "vertices_on_edge",
            "edges_on_edge",
            "n_edges_on_edge",
            "vertices_on_cell",
            "edges_on_vertex",
            "cells_on_vertex",
            "edge_sign_on_cell",
            "weights_on_edge",
            "dc_edge",
            "dv_edge",
            "area_cell",
            "area_triangle",
            "kite_areas_on_vertex",
            "lat_cell",
            "lon_cell",
            "lat_edge",
            "lon_edge",
            "angle_edge",
            "mesh_density",
            "defc_a",
            "defc_b",
            "f_vertex",
            "f_edge",
            "spec_zone_mask_edge",
        ):
            result[name] = cp.asnumpy(getattr(self, name))
        # The host layout: a compacted edge stencil goes back to maxEdges2.
        full_eoe, full_woe = self.full_edge_stencil()
        result["edges_on_edge"] = cp.asnumpy(full_eoe)
        result["weights_on_edge"] = cp.asnumpy(full_woe)
        result.update(
            n_cells=self.n_cells,
            n_edges=self.n_edges,
            n_vertices=self.n_vertices,
            max_edges=self.max_edges,
            max_edges2=self.file_max_edges2,
            vertex_degree=self.vertex_degree,
            nominal_min_dc=self.nominal_min_dc,
        )
        return result


@dataclass(slots=True)
class DeviceVerticalGrid:
    zw: Any
    dzw: Any
    rdzw: Any
    zu: Any
    dzu: Any
    rdzu: Any
    rdzwp: Any
    rdzwm: Any
    fzp: Any
    fzm: Any
    ah: Any
    #: ``hx`` (terrain influence on the coordinate surfaces) is an init-time
    #: field: no forecast kernel reads it.  It stays on the host until
    #: something asks for ``.hx`` (regional residency holds, dual-run
    #: fingerprints), which uploads it once (``compaction.py``,
    #: ``GPUWM_HEX_MESH_COMPACT_VERTICAL_HX=0`` uploads it at once).
    _hx: Any
    zgrid: Any
    zz: Any
    zxu: Any
    dss: Any
    cf1: float
    cf2: float
    cf3: float
    first_height_level: int
    n_vert_levels: int
    dtype: np.dtype[Any]
    h2d: TransferStats
    _hx_host: Any = field(default=None, repr=False, compare=False)

    @property
    def hx(self) -> Any:
        if self._hx is None:
            if self._hx_host is None:
                raise AttributeError("vertical.hx was never supplied")
            self._hx = _cp().asarray(self._hx_host)
        return self._hx

    @hx.setter
    def hx(self, value: Any) -> None:
        self._hx = value

    @property
    def hx_resident(self) -> bool:
        """True once ``hx`` is on the device."""

        return self._hx is not None

    @classmethod
    def from_host(
        cls, vertical: object, *, dtype: Any | None = None
    ) -> "DeviceVerticalGrid":
        require_cuda()
        cp = _cp()
        selected = np.dtype(
            np.asarray(getattr(vertical, "zz")).dtype if dtype is None else dtype
        )
        if selected not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise TypeError("DeviceVerticalGrid dtype must be float32 or float64")
        names = (
            "zw",
            "dzw",
            "rdzw",
            "zu",
            "dzu",
            "rdzu",
            "rdzwp",
            "rdzwm",
            "fzp",
            "fzm",
            "ah",
            "hx",
            "zgrid",
            "zz",
            "zxu",
            "dss",
        )
        lazy_hx = compaction_enabled("VERTICAL_HX")
        started = time.perf_counter()
        device = _device_float_fields(
            cp, vertical, tuple(name for name in names if name != "hx"), selected,
            finite=False,
        )
        hx_host = None
        if lazy_hx:
            hx_host = contiguous_copy(getattr(vertical, "hx"), dtype=selected)
            device["_hx"] = None
        else:
            device["_hx"] = _device_float_fields(
                cp, vertical, ("hx",), selected, finite=False,
            )["hx"]
        cp.cuda.get_current_stream().synchronize()
        elapsed = time.perf_counter() - started
        byte_count = int(
            sum(int(value.nbytes) for value in device.values() if value is not None)
        )
        return cls(
            **device,
            _hx_host=hx_host if lazy_hx else None,
            cf1=float(getattr(vertical, "cf1")),
            cf2=float(getattr(vertical, "cf2")),
            cf3=float(getattr(vertical, "cf3")),
            first_height_level=int(getattr(vertical, "first_height_level")),
            n_vert_levels=int(np.asarray(getattr(vertical, "zz")).shape[0]),
            dtype=selected,
            h2d=TransferStats(byte_count, elapsed),
        )

    def validate(self, *, n_cells: int, n_edges: int) -> None:
        one_dimensional = {
            "zw": self.n_vert_levels + 1,
            "dzw": self.n_vert_levels,
            "rdzw": self.n_vert_levels,
            "zu": self.n_vert_levels,
            "dzu": self.n_vert_levels,
            "rdzu": self.n_vert_levels,
            "rdzwp": self.n_vert_levels,
            "rdzwm": self.n_vert_levels,
            "fzp": self.n_vert_levels,
            "fzm": self.n_vert_levels,
            "ah": self.n_vert_levels + 1,
        }
        for name, length in one_dimensional.items():
            require_resident_array(
                f"vertical.{name}",
                getattr(self, name),
                dtype=self.dtype,
                shape=(length,),
            )
        hx_shape = (self.n_vert_levels + 1, n_cells)
        if self._hx is None:
            # Not yet uploaded: check the host copy without uploading it.
            host_hx = self._hx_host
            if (
                not isinstance(host_hx, np.ndarray)
                or host_hx.dtype != self.dtype
                or tuple(host_hx.shape) != hx_shape
                or not bool(host_hx.flags.c_contiguous)
            ):
                raise ValueError(f"vertical.hx must be a {self.dtype} host array of shape {hx_shape}")
        else:
            require_resident_array(
                "vertical.hx", self._hx, dtype=self.dtype, shape=hx_shape
            )
        for name, shape in {
            "zgrid": (self.n_vert_levels + 1, n_cells),
            "zz": (self.n_vert_levels, n_cells),
            "zxu": (self.n_vert_levels, n_edges),
            "dss": (self.n_vert_levels, n_cells),
        }.items():
            require_resident_array(
                f"vertical.{name}", getattr(self, name), dtype=self.dtype, shape=shape
            )


@dataclass(slots=True)
class DeviceReferenceState:
    rho_base: Any
    rho_theta_base: Any
    pressure_base: Any
    exner_base: Any
    dtype: np.dtype[Any]
    h2d: TransferStats

    @classmethod
    def from_host(
        cls, reference: object, *, dtype: Any | None = None
    ) -> "DeviceReferenceState":
        require_cuda()
        cp = _cp()
        selected = np.dtype(
            np.asarray(getattr(reference, "rho_base")).dtype if dtype is None else dtype
        )
        names = ("rho_base", "rho_theta_base", "pressure_base", "exner_base")
        started = time.perf_counter()
        device = _device_float_fields(cp, reference, names, selected)
        cp.cuda.get_current_stream().synchronize()
        elapsed = time.perf_counter() - started
        return cls(
            **device,
            dtype=selected,
            h2d=TransferStats(
                sum(int(value.nbytes) for value in device.values()), elapsed
            ),
        )

    def validate(self, *, n_vert_levels: int, n_cells: int) -> None:
        shape = (n_vert_levels, n_cells)
        for name in ("rho_base", "rho_theta_base", "pressure_base", "exner_base"):
            require_resident_array(
                f"reference.{name}", getattr(self, name), dtype=self.dtype, shape=shape
            )


@dataclass(slots=True)
class DevicePrognosticState:
    rho: Any
    rho_theta: Any
    rho_u: Any
    rho_w: Any
    scalars: Any
    time_seconds: float
    dtype: np.dtype[Any]
    h2d: TransferStats

    @classmethod
    def from_host(
        cls, state: object, *, dtype: Any | None = None
    ) -> "DevicePrognosticState":
        require_cuda()
        cp = _cp()
        selected = np.dtype(
            np.asarray(getattr(state, "rho")).dtype if dtype is None else dtype
        )
        names = ("rho", "rho_theta", "rho_u", "rho_w", "scalars")
        started = time.perf_counter()
        device = _device_float_fields(cp, state, names, selected, exact_dtype=True)
        cp.cuda.get_current_stream().synchronize()
        elapsed = time.perf_counter() - started
        return cls(
            **device,
            time_seconds=float(getattr(state, "time_seconds", 0.0)),
            dtype=selected,
            h2d=TransferStats(
                sum(int(value.nbytes) for value in device.values()), elapsed
            ),
        )

    def validate(self, *, n_vert_levels: int, n_cells: int, n_edges: int) -> None:
        for name, shape in {
            "rho": (n_vert_levels, n_cells),
            "rho_theta": (n_vert_levels, n_cells),
            "rho_u": (n_vert_levels, n_edges),
            "rho_w": (n_vert_levels + 1, n_cells),
        }.items():
            require_resident_array(
                f"state.{name}", getattr(self, name), dtype=self.dtype, shape=shape
            )
        require_resident_array("state.scalars", self.scalars, dtype=self.dtype, ndim=3)
        if tuple(self.scalars.shape[1:]) != (n_vert_levels, n_cells):
            raise ValueError(
                "state.scalars must have shape "
                f"(nTracers, {n_vert_levels}, {n_cells}), got {tuple(self.scalars.shape)}"
            )

    def to_host(self) -> Any:
        from woof.hex.state import PrognosticState

        cp = _cp()
        return PrognosticState(
            rho=cp.asnumpy(self.rho),
            rho_theta=cp.asnumpy(self.rho_theta),
            rho_u=cp.asnumpy(self.rho_u),
            rho_w=cp.asnumpy(self.rho_w),
            scalars=cp.asnumpy(self.scalars),
            time_seconds=self.time_seconds,
        )

    def to_host_timed(self) -> tuple[Any, TransferStats]:
        started = time.perf_counter()
        state = self.to_host()
        elapsed = time.perf_counter() - started
        byte_count = int(
            sum(
                int(getattr(self, name).nbytes)
                for name in ("rho", "rho_theta", "rho_u", "rho_w", "scalars")
            )
        )
        return state, TransferStats(byte_count, elapsed)


@dataclass(slots=True)
class DeviceSavedDiagnostics:
    theta_m: Any
    exner: Any
    density_perturbation: Any
    rho_theta_perturbation: Any
    pressure_perturbation: Any
    normal_velocity: Any
    vertical_velocity: Any
    dtype: np.dtype[Any]
    h2d: TransferStats

    @classmethod
    def empty(
        cls,
        *,
        n_vert_levels: int,
        n_cells: int,
        n_edges: int,
        dtype: Any = np.float32,
    ) -> "DeviceSavedDiagnostics":
        """Allocate a device-only target for self-derived t0 diagnostics."""

        require_cuda()
        cp = _cp()
        selected = np.dtype(dtype)
        if selected not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise TypeError("DeviceSavedDiagnostics dtype must be float32 or float64")

        def cell_field() -> Any:
            return cp.empty((n_vert_levels, n_cells), dtype=selected)

        result = cls(
            theta_m=cell_field(),
            exner=cell_field(),
            density_perturbation=cell_field(),
            rho_theta_perturbation=cell_field(),
            pressure_perturbation=cell_field(),
            normal_velocity=cp.empty((n_vert_levels, n_edges), dtype=selected),
            vertical_velocity=cp.empty(
                (n_vert_levels + 1, n_cells), dtype=selected
            ),
            dtype=selected,
            h2d=TransferStats(0, 0.0),
        )
        result.validate(
            n_vert_levels=n_vert_levels,
            n_cells=n_cells,
            n_edges=n_edges,
        )
        return result

    @classmethod
    def from_host(
        cls, saved: object, *, dtype: Any | None = None
    ) -> "DeviceSavedDiagnostics":
        require_cuda()
        cp = _cp()
        selected = np.dtype(
            np.asarray(getattr(saved, "theta_m")).dtype if dtype is None else dtype
        )
        names = (
            "theta_m",
            "exner",
            "density_perturbation",
            "rho_theta_perturbation",
            "pressure_perturbation",
            "normal_velocity",
            "vertical_velocity",
        )
        started = time.perf_counter()
        device = _device_float_fields(cp, saved, names, selected, exact_dtype=True)
        cp.cuda.get_current_stream().synchronize()
        elapsed = time.perf_counter() - started
        return cls(
            **device,
            dtype=selected,
            h2d=TransferStats(
                sum(int(value.nbytes) for value in device.values()), elapsed
            ),
        )

    def validate(self, *, n_vert_levels: int, n_cells: int, n_edges: int) -> None:
        cell_shape = (n_vert_levels, n_cells)
        for name in (
            "theta_m",
            "exner",
            "density_perturbation",
            "rho_theta_perturbation",
            "pressure_perturbation",
        ):
            require_resident_array(
                f"saved.{name}", getattr(self, name), dtype=self.dtype, shape=cell_shape
            )
        require_resident_array(
            "saved.normal_velocity",
            self.normal_velocity,
            dtype=self.dtype,
            shape=(n_vert_levels, n_edges),
        )
        require_resident_array(
            "saved.vertical_velocity",
            self.vertical_velocity,
            dtype=self.dtype,
            shape=(n_vert_levels + 1, n_cells),
        )


@dataclass(slots=True)
class DeviceTerrainMetrics:
    zb_cell: Any
    zb3_cell: Any
    dtype: np.dtype[Any]
    h2d: TransferStats
    #: Slot stride ``zb_cell``/``zb3_cell`` are stored with.  The host
    #: metrics carry ``maxEdges`` slots per cell (10 in the standard grid
    #: files); a compacted upload keeps the mesh's largest ``nEdgesOnCell``
    #: (6 on a quasi-uniform mesh), and the readers index with this stride
    #: through :meth:`DeviceMesh.slot_tables`.  0 means the full host stride.
    slot_stride: int = 0
    #: ``maxEdges`` of the host metrics, and the constant the dropped slots
    #: held, so the host layout can be rebuilt exactly (:meth:`full_layout`).
    source_max_edges: int = 0
    slot_fill: float = 0.0

    @classmethod
    def from_host(
        cls,
        terrain: object,
        *,
        dtype: Any | None = None,
        slot_stride: int | None = None,
    ) -> "DeviceTerrainMetrics":
        """Upload the cell-oriented slope metrics.

        ``slot_stride`` (the mesh's largest ``nEdgesOnCell``) stores only the
        slots any reader can reach; it is applied only when every dropped slot
        holds one constant, so :meth:`full_layout` rebuilds the host bytes.
        """

        require_cuda()
        cp = _cp()
        started = time.perf_counter()
        selected = np.dtype(
            np.asarray(getattr(terrain, "zb_cell")).dtype if dtype is None else dtype
        )
        host_zb = _as_float("zb_cell", getattr(terrain, "zb_cell"), selected)
        host_zb3 = _as_float("zb3_cell", getattr(terrain, "zb3_cell"), selected)
        width = int(host_zb.shape[2]) if host_zb.ndim == 3 else 0
        stride = width
        fill = 0.0
        if (
            slot_stride is not None
            and host_zb.ndim == 3
            and host_zb3.shape == host_zb.shape
            and 1 <= int(slot_stride) < width
        ):
            keep = int(slot_stride)
            zb_tail = uniform_tail(host_zb, keep, axis=2)
            zb3_tail = uniform_tail(host_zb3, keep, axis=2)
            if (
                zb_tail is not None
                and zb3_tail is not None
                and np.asarray(zb_tail).tobytes() == np.asarray(zb3_tail).tobytes()
            ):
                host_zb = compact_host(host_zb, keep, axis=2)
                host_zb3 = compact_host(host_zb3, keep, axis=2)
                stride = keep
                fill = float(zb_tail)
        zb = _upload(cp, host_zb, 'zb')
        zb3 = _upload(cp, host_zb3, 'zb3')
        cp.cuda.get_current_stream().synchronize()
        elapsed = time.perf_counter() - started
        return cls(
            zb,
            zb3,
            selected,
            TransferStats(int(zb.nbytes + zb3.nbytes), elapsed),
            stride,
            width,
            fill,
        )

    @property
    def stride(self) -> int:
        """The slot stride of the resident arrays."""

        return int(self.zb_cell.shape[2])

    def full_layout(self) -> tuple[Any, Any]:
        """``zb_cell``/``zb3_cell`` in the host's ``maxEdges`` layout (transient)."""

        width = int(self.source_max_edges or self.stride)
        return (
            restore_device(self.zb_cell, width, self.slot_fill, axis=2),
            restore_device(self.zb3_cell, width, self.slot_fill, axis=2),
        )

    def validate(self, *, n_vert_levels: int, n_cells: int, max_edges: int) -> None:
        if getattr(self.zb_cell, "ndim", 0) != 3:
            raise ValueError("terrain.zb_cell must have shape (nlev+1, nCells, slots)")
        stride = self.stride
        if not 1 <= stride <= int(max_edges):
            raise ValueError(
                f"terrain slot stride {stride} is outside [1, maxEdges={max_edges}]"
            )
        if self.slot_stride and int(self.slot_stride) != stride:
            raise ValueError("terrain slot stride does not match its arrays")
        shape = (n_vert_levels + 1, n_cells, stride)
        require_resident_array(
            "terrain.zb_cell", self.zb_cell, dtype=self.dtype, shape=shape
        )
        require_resident_array(
            "terrain.zb3_cell", self.zb3_cell, dtype=self.dtype, shape=shape
        )


def terrain_slot_tables(mesh: Any, terrain: Any) -> tuple[Any, Any, int]:
    """``edgesOnCell``, its sign and the slot stride that index ``zb_cell``.

    The terrain kernels take one ``max_edges`` stride for all three tables.
    With a full-stride ``zb_cell`` these are the mesh's own tables; with a
    compacted one they are the mesh's tables cut to the same stride.  Either
    way the words a ``slot < nEdgesOnCell`` loop reads are identical.
    """

    stride = int(terrain.zb_cell.shape[2])
    if stride == int(mesh.max_edges):
        return mesh.edges_on_cell, mesh.edge_sign_on_cell, stride
    edges, signs = mesh.slot_tables(stride)
    return edges, signs, stride


@dataclass(slots=True)
class DeviceAtmosphere:
    mesh: DeviceMesh
    state: DevicePrognosticState
    vertical: DeviceVerticalGrid
    reference: DeviceReferenceState
    saved: DeviceSavedDiagnostics
    terrain: DeviceTerrainMetrics | None
    h2d: TransferStats

    @classmethod
    def from_host(
        cls,
        mesh: object,
        state: object,
        vertical: object,
        reference: object,
        saved_diagnostics: object | None,
        terrain_metrics: object | None = None,
        *,
        dtype: Any | None = None,
        index_dtype: Any = np.int32,
        upload_cache: Any = None,
    ) -> "DeviceAtmosphere":
        require_cuda()
        started = time.perf_counter()
        selected = np.asarray(getattr(state, "rho")).dtype if dtype is None else dtype
        cp = _cp()
        device_id = int(cp.cuda.runtime.getDevice())

        def upload(function, *args, **kwargs):
            # Device and stream state are thread-local in CUDA. Every component
            # already synchronizes its stream before returning resident words.
            # Three workers overlap host preparation with independent transfers
            # without unbounded conversion buffers or one stream per field.
            with cp.cuda.Device(device_id), cp.cuda.Stream(non_blocking=True) as stream:
                token = _UPLOAD_CACHE.set(None if upload_cache is None else
                                          (upload_cache, function.__qualname__))
                try:
                    return function(*args, **kwargs)
                except BaseException:
                    # A refusal can occur after an earlier field was queued.
                    # Finish that stream before discarding its temporaries.
                    stream.synchronize()
                    raise
                finally:
                    _UPLOAD_CACHE.reset(token)

        with ThreadPoolExecutor(max_workers=3, thread_name_prefix="hex-upload") as pool:
            mesh_future = pool.submit(upload, DeviceMesh.from_host, mesh,
                                      dtype=selected, index_dtype=index_dtype)
            state_future = pool.submit(upload, DevicePrognosticState.from_host,
                                       state, dtype=selected)
            vertical_future = pool.submit(upload, DeviceVerticalGrid.from_host,
                                          vertical, dtype=selected)
            # Resolve in the original order so invalid inputs keep their named
            # failure priority, even when a later component fails sooner.
            device_mesh = mesh_future.result()
            slot_stride = None
            if terrain_metrics is not None and compaction_enabled("TERRAIN"):
                counts = _mesh_value(mesh, "nEdgesOnCell")
                if counts is not None:
                    slot_stride = used_width(counts, device_mesh.max_edges)
            terrain_future = (None if terrain_metrics is None else pool.submit(
                upload, DeviceTerrainMetrics.from_host, terrain_metrics,
                dtype=selected, slot_stride=slot_stride))
            # Terrain is ready once mesh dimensions are known. Queue it before
            # the shorter reference/diagnostic uploads so it does not become
            # the last outstanding conversion and transfer.
            reference_future = pool.submit(upload, DeviceReferenceState.from_host,
                                           reference, dtype=selected)
            saved_future = (None if saved_diagnostics is None else pool.submit(
                upload, DeviceSavedDiagnostics.from_host, saved_diagnostics,
                dtype=selected))
            device_state = state_future.result()
            device_vertical = vertical_future.result()
            device_reference = reference_future.result()
            device_saved = (
                DeviceSavedDiagnostics.empty(
                    n_vert_levels=device_vertical.n_vert_levels,
                    n_cells=device_mesh.n_cells,
                    n_edges=device_mesh.n_edges,
                    dtype=selected,
                )
                if saved_future is None else saved_future.result()
            )
            device_terrain = None if terrain_future is None else terrain_future.result()
        if device_terrain is not None and device_terrain.stride != device_mesh.max_edges:
            # Built here, at upload, so the step's allocation sequence is the
            # same as with the full-stride tables.
            device_mesh.slot_tables(device_terrain.stride)
        _cp().cuda.get_current_stream().synchronize()
        elapsed = time.perf_counter() - started
        components = (
            device_mesh,
            device_state,
            device_vertical,
            device_reference,
            device_saved,
            device_terrain,
        )
        byte_count = int(sum(item.h2d.bytes for item in components if item is not None))
        result = cls(
            device_mesh,
            device_state,
            device_vertical,
            device_reference,
            device_saved,
            device_terrain,
            TransferStats(byte_count, elapsed),
        )
        result.validate()
        return result

    def validate(self) -> None:
        """Fail closed before any raw kernel can observe an inconsistent atmosphere."""

        self.mesh.validate()
        nlev = self.vertical.n_vert_levels
        ncells = self.mesh.n_cells
        nedges = self.mesh.n_edges
        expected_dtype = np.dtype(self.state.dtype)
        for component_name, component in (
            ("mesh", self.mesh),
            ("vertical", self.vertical),
            ("reference", self.reference),
            ("saved", self.saved),
        ):
            if np.dtype(component.dtype) != expected_dtype:
                raise TypeError(
                    f"{component_name}.dtype must match state dtype {expected_dtype}"
                )
        if self.terrain is not None and np.dtype(self.terrain.dtype) != expected_dtype:
            raise TypeError(f"terrain.dtype must match state dtype {expected_dtype}")
        self.vertical.validate(n_cells=ncells, n_edges=nedges)
        self.reference.validate(n_vert_levels=nlev, n_cells=ncells)
        self.state.validate(n_vert_levels=nlev, n_cells=ncells, n_edges=nedges)
        self.saved.validate(n_vert_levels=nlev, n_cells=ncells, n_edges=nedges)
        if self.terrain is not None:
            self.terrain.validate(
                n_vert_levels=nlev,
                n_cells=ncells,
                max_edges=self.mesh.max_edges,
            )


__all__ = [
    "DeviceAtmosphere",
    "DeviceMesh",
    "DevicePrognosticState",
    "DeviceReferenceState",
    "DeviceSavedDiagnostics",
    "DeviceTerrainMetrics",
    "DeviceVerticalGrid",
    "TransferStats",
    "require_resident_array",
    "terrain_slot_tables",
]
