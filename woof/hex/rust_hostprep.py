"""Stream exact initialized terrain and momentum arrays through Rust."""

from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path
import struct
import subprocess

import numpy as np

from .engines import HOSTPREP, EngineRefusal, resolve
from .host_parallel import contiguous_copy
from .prep_cache import cached

ABI_MARKER = b"rw_mpas_hostprep --protocol hex-hostprep-v1"


@lru_cache(maxsize=8)
def _check_binary(path: str, size: int, modified_ns: int) -> str:
    digest = hashlib.sha256()
    found = False
    tail = b""
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
            found = found or ABI_MARKER in tail + block
            tail = block[-len(ABI_MARKER):]
    if not found:
        raise EngineRefusal(
            "the host preparation bridge has no hex-hostprep-v1 marker; "
            "a stale executable would interpret initialized arrays with the wrong ABI. "
            "Build rw_mpas_hostprep from the current woof source"
        )
    return digest.hexdigest()


def hostprep_engine() -> tuple[Path, str]:
    path = resolve(HOSTPREP)
    stat = path.stat()
    return path, _check_binary(str(path), stat.st_size, stat.st_mtime_ns)


def _widen_pipe(stream) -> None:
    """Give the bridge's pipe 1 MiB instead of the kernel's 64 KiB default (the same bytes cross it).

    THE COST THIS REMOVES (30 km globe over 4 cards, V5090 box, 2026-10-27): the momentum correction's ~1.5 GB of
    arrays crossed a 64 KiB pipe in 1 MiB writes, ~6 s of the main thread's start-up on a loaded box, each write
    waiting on 16 reader wake-ups.  A host that refuses the size (pipe-max-size) keeps the default.
    """
    try:
        import fcntl
        fcntl.fcntl(stream.fileno(), getattr(fcntl, "F_SETPIPE_SZ", 1031), 1 << 20)
    except (ImportError, OSError, ValueError, AttributeError):
        pass


def _run(*, mode: int, dtype: np.dtype, counts, edges, n_edges: int,
         n_levels: int, arrays, outputs):
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError("host preparation requires float32 or float64 arrays")
    n_cells, max_edges = edges.shape
    engine, _ = hostprep_engine()
    process = subprocess.Popen([str(engine), "--protocol", "hex-hostprep-v1"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdin is not None and process.stdout is not None and process.stderr is not None
    _widen_pipe(process.stdin)
    _widen_pipe(process.stdout)
    values = []
    try:
        process.stdin.write(struct.pack("<8sQQQQBB", b"HEXHOST1", n_cells,
            n_edges, max_edges, n_levels, dtype.itemsize, mode))
        for array, wire_dtype in [(counts, "<i8"), (edges, "<i8"), *arrays]:
            contiguous = contiguous_copy(array, dtype=wire_dtype)
            data = memoryview(contiguous).cast("B")
            for offset in range(0, len(data), 1024 * 1024):
                process.stdin.write(data[offset:offset + 1024 * 1024])
        process.stdin.close()
        if outputs is None:
            values = json.loads(process.stdout.read())
        for shape, wire_dtype in outputs or ():
            value = np.empty(shape, dtype=wire_dtype)
            data = memoryview(value).cast("B")
            offset = 0
            while offset < len(data):
                read = process.stdout.readinto(data[offset:offset + 1024 * 1024])
                if not read:
                    break
                offset += read
            if offset != len(data):
                message = process.stderr.read().decode("utf-8", errors="replace").strip().removeprefix("host preparation: ")
                raise ValueError(message or "host preparation returned incomplete arrays")
            values.append(value)
        if process.stdout.read(1):
            raise EngineRefusal("host preparation returned excess bytes; the array ABI is inconsistent")
        message = process.stderr.read().decode("utf-8", errors="replace").strip().removeprefix("host preparation: ")
        if process.wait():
            raise ValueError(message or "host preparation bridge failed")
    except BrokenPipeError as error:
        message = process.stderr.read().decode("utf-8", errors="replace").strip().removeprefix("host preparation: ")
        raise ValueError(message or "host preparation input was rejected") from error
    finally:
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except BrokenPipeError:
                    pass
        if process.poll() is None:
            process.terminate()
        process.wait()
    return values


def _engine_digest() -> str:
    return hostprep_engine()[1]


@lru_cache(maxsize=8)
def _deformation_capable(path: str, size: int, modified_ns: int) -> bool:
    return b"hex-hostprep-gnu-deformation-v1" in Path(path).read_bytes()


def deformation_engine_digest() -> str:
    """Bind cached weights to the optional bridge, including older installations."""
    from .native_geometry_math import strict_binding
    strict = strict_binding()
    if strict is not None:
        return strict
    try:
        path, digest = hostprep_engine()
    except EngineRefusal:
        return "python"
    stat = path.stat()
    return digest if _deformation_capable(str(path), stat.st_size, stat.st_mtime_ns) else "python"


def deformation_weights(*, counts, edges, cells_on_cell, vertices_on_cell,
                        cells_on_edge, cell_xyz, vertex_xyz, radius, pi):
    """All five GNU binary32 deformation tables, parallel across cell rows.

    An older engine returns None so its Python authority remains available.
    Mode 10 is deliberately unused: it has different intrinsics and only
    three outputs, which cannot supply the native LES gradient coefficients.
    """
    from .native_geometry_math import strict_binding
    if strict_binding() is not None or deformation_engine_digest() == "python":
        return None
    dtype = np.dtype(np.float32)
    wire = dtype.newbyteorder("<")
    arrays = [(np.asarray([vertex_xyz[0].size]), "<i8"),
              (cells_on_cell, "<i8"), (vertices_on_cell, "<i8"),
              (cells_on_edge, "<i8"), (np.asarray([radius, pi]), wire)]
    arrays.extend((a, wire) for a in (*cell_xyz, *vertex_xyz))
    return _run(mode=15, dtype=dtype, counts=counts, edges=edges,
                n_edges=cells_on_edge.shape[0], n_levels=1, arrays=arrays,
                outputs=[(edges.shape, wire)] * 5)


# THE COST THIS REMOVES (NOAA RRFS v2 conus3km, 2,341,819 cells x 60 interfaces, V8X6000 2026-10-29, host-only):
# the vertical grid (17.5 s) and the initial state (18.8 s) ran on one thread, most of it in these two calls:
# about 15 GB of zb/zb3 in and zb_cell/zb3_cell out (terrain) and 10.7 GB in (momentum) crossed the bridge's
# pipe on one thread, and the prep cache hashed 6.8 GB of inputs and saved 7.9 GB of outputs on that thread.
# Both bridge modes are gathers and per-slot float operations (woof tools/rustwx/crates/rw-mpas/src/hostprep.rs
# process(), modes 0 and 1): the same words are now formed in-process on host threads over cell blocks, each
# element by the same operations in the same order (woof.hex.host_parallel's exactness argument), so the outputs
# are bit-identical (tests/test_rust_hostprep.py holds them to the scalar authority).  Mixed-precision momentum
# (mode 3) still goes through the bridge.

def _active_slots(counts, me: int):
    return np.arange(me, dtype=np.int64)[None, :] < counts[:, None]


def _first_refusal(flags, messages):
    """The message of the first (cell, slot) in row order with any refusal, checks in the bridge's order."""
    stacked = np.stack(flags)
    hit = stacked.any(axis=0)
    if not hit.any():
        return None
    flat = int(np.flatnonzero(hit.reshape(-1))[0])
    cell, slot = divmod(flat, hit.shape[1])
    kind = int(np.flatnonzero(stacked[:, cell, slot])[0])
    return messages[kind](cell, slot)


def terrain_metrics(*, counts, edges, cells_on_edge, zb, zb3, coefficient):
    """rw_mpas_hostprep mode 0, word for word, on host threads (see above)."""
    from .host_parallel import run_blocks
    dtype = np.dtype(zb.dtype)
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError("host preparation requires float32 or float64 arrays")
    counts = np.asarray(counts, dtype=np.int64)
    eoc = np.asarray(edges, dtype=np.int64)
    coe = np.asarray(cells_on_edge, dtype=np.int64)
    nc, me = eoc.shape
    nl, _, ne = zb.shape
    if np.any(counts < 0) or np.any(counts > me):
        raise ValueError("n_edges_on_cell entries must be in [0, maxEdges]")
    active = _active_slots(counts, me)
    in_range = (eoc >= 0) & (eoc < ne)
    safe = np.where(active & in_range, eoc, 0)
    duplicate = np.zeros_like(active)
    for slot in range(1, me):
        for earlier in range(slot):
            duplicate[:, slot] |= eoc[:, earlier] == eoc[:, slot]
    cell_index = np.arange(nc, dtype=np.int64)[:, None]
    first = coe[safe, 0] == cell_index
    second = coe[safe, 1] == cell_index
    message = _first_refusal(
        [active & ~in_range, active & duplicate, active & ~first & ~second],
        [lambda c, s: f"active edges_on_cell[{c}, {s}]={eoc[c, s]} is out of range",
         lambda c, s: f"cell {c} lists edge {eoc[c, s]} more than once",
         lambda c, s: f"cell {c} is not present in cells_on_edge for edge {eoc[c, s]}"])
    if message is not None:
        raise ValueError(message)
    canonical = np.where(active, eoc, np.int64(-1))
    signs = np.zeros((nc, me), dtype=dtype)
    signs[active & first] = dtype.type(1.0)
    signs[active & ~first] = dtype.type(-1.0)
    side = np.where(first, 0, 1)
    coefficient = dtype.type(coefficient)
    # (edge, side, interface) order when zb is the logical view of the native netCDF array: one row per edge side.
    native_zb = np.transpose(zb, (2, 1, 0))
    native_zb3 = np.transpose(zb3, (2, 1, 0))
    zb_cell = np.zeros((nl, nc, me), dtype=dtype)
    zb3_cell = np.zeros((nl, nc, me), dtype=dtype)

    def fill(block: slice) -> None:
        for slot in range(me):
            on = active[block, slot]
            if not on.any():
                continue
            cells = np.flatnonzero(on) + block.start
            rows = (safe[cells, slot], side[cells, slot])
            zb_cell[:, cells, slot] = native_zb[rows].T
            zb3_cell[:, cells, slot] = (coefficient * native_zb3[rows]).T

    run_blocks(fill, nc, elements=nl * nc * me)
    return [canonical, signs, zb_cell, zb3_cell]


def correct_vertical_momentum(*, counts, edges, fzm, fzp, ru, interface_zz,
                              signs, zb_cell, zb3_cell, rw):
    dtype = np.dtype(rw.dtype)
    wire = dtype.newbyteorder("<")
    nl, ne = ru.shape
    values = [np.asarray(value) for value in
              (fzm, fzp, ru, interface_zz, signs, zb_cell, zb3_cell, rw)]
    if any(value.dtype not in (np.dtype(np.float32), np.dtype(np.float64)) for value in values):
        raise TypeError("momentum preparation requires float32 or float64 fields")
    mixed = any(value.dtype != dtype for value in values)
    if not mixed:
        return _momentum_threads(counts, edges, *values)
    arrays = ([(np.asarray([value.dtype.itemsize for value in values], dtype=np.uint8), "u1")]
              if mixed else [])
    arrays.extend((value, value.dtype.newbyteorder("<")) for value in values)
    return _run(mode=3 if mixed else 1, dtype=dtype, counts=counts, edges=edges, n_edges=ne,
        n_levels=nl, arrays=arrays,
        outputs=[(rw.shape, wire)])[0]


def _momentum_threads(counts, edges, fzm, fzp, ru, interface_zz, signs, zb_cell, zb3_cell, rw):
    """rw_mpas_hostprep mode 1, word for word, on host threads (see terrain_metrics).

    Per cell the slot corrections apply in slot order; each is
    ``value - signs*(zb + copysign(1, flux)*zb3)*flux*interface_zz`` with
    ``flux = fzm[k]*ru[k, edge] + fzp[k]*ru[k-1, edge]``, every operation rounded in the field dtype.
    """
    from .host_parallel import run_blocks, copy_into
    counts = np.asarray(counts, dtype=np.int64)
    eoc = np.asarray(edges, dtype=np.int64)
    nc, me = eoc.shape
    nl, ne = ru.shape
    if np.any(counts < 0) or np.any(counts > me):
        raise ValueError("n_edges_on_cell entries must be in [0, maxEdges]")
    active = _active_slots(counts, me)
    message = _first_refusal([active & ((eoc < 0) | (eoc >= ne))],
                             [lambda c, s: f"active edges_on_cell[{c}, {s}]={eoc[c, s]} is out of range"])
    if message is not None:
        raise ValueError(message)
    safe = np.where(active, eoc, 0)
    dtype = rw.dtype
    out = np.empty(rw.shape, dtype=dtype)
    copy_into(out, rw)
    if nl < 2:
        return out
    weight_m = fzm[1:nl, None]
    weight_p = fzp[1:nl, None]
    one = dtype.type(1.0)

    def correct(block: slice) -> None:
        value = out[1:nl, block].copy()
        izz = interface_zz[1:nl, block]
        with np.errstate(all="ignore"):
            for slot in range(me):
                on = active[block, slot]
                if not on.any():
                    continue
                edge = safe[block, slot]
                flux = weight_m * ru[1:nl, edge] + weight_p * ru[0:nl - 1, edge]
                terrain = zb_cell[1:nl, block, slot] + np.copysign(one, flux) * zb3_cell[1:nl, block, slot]
                correction = signs[block, slot][None, :] * terrain * flux * izz
                value = np.where(on[None, :], value - correction, value)
        out[1:nl, block] = value

    run_blocks(correct, nc, elements=nl * nc * me)
    return out


# Called four times per run on the same topology (0.75 s each at 30 km).
@cached("rust-topology-errors", salt=_engine_digest)
def topology_errors(*, n_vertices: int, max_edges2: int, vertex_degree: int,
                    regional: bool, counts, edges, cells_on_cell,
                    vertices_on_cell, cells_on_edge, vertices_on_edge,
                    cells_on_vertex, edges_on_vertex, edge_counts, edges_on_edge):
    """Return the original ordered topology refusals, including graph closure."""
    arrays = [np.asarray([max_edges2, vertex_degree, regional], dtype=np.int64),
        cells_on_cell, vertices_on_cell, cells_on_edge, vertices_on_edge,
        cells_on_vertex, edges_on_vertex, edge_counts, edges_on_edge]
    return _run(mode=2, dtype=np.dtype(np.float64), counts=counts, edges=edges,
        n_edges=cells_on_edge.shape[0], n_levels=n_vertices,
        arrays=[(value, "<i8") for value in arrays], outputs=None)


def compress_advection(*, counts, edges, cells_on_cell, cells_on_edge, ids, deriv, dc, dv, source_order, garbage):
    """Use the shipped parallel Rust compression, preserving each edge's operation order."""
    dtype=deriv.dtype; wire=dtype.newbyteorder('<'); ne=dc.size; nc,me=edges.shape; width=deriv.shape[2]
    return _run(mode=11,dtype=dtype,counts=counts,edges=edges,n_edges=ne,n_levels=1,
        arrays=[(np.asarray([width,source_order,garbage],np.int64),'<i8'),
                (cells_on_cell,'<i8'),(cells_on_edge,'<i8'),(ids,'<i8'),
                (deriv,wire),(dc,wire),(dv,wire)],
        outputs=[((ne,width),wire),((ne,width),wire),((ne,),'<i8'),((ne,width),'<i8')])


def vertical_velocity_damping(heights, *, xnutr, start, pi):
    """Use the shipped RKIND sine and source-order Rust damping loop."""
    dtype=heights.dtype; wire=dtype.newbyteorder('<'); nl=heights.shape[0]-1; nc=heights.shape[1]
    return _run(mode=12,dtype=dtype,counts=np.zeros(nc,np.int64),edges=np.zeros((nc,1),np.int64),
        n_edges=1,n_levels=nl,arrays=[(np.asarray([xnutr,start,pi],dtype=dtype),wire),(heights,wire)],
        outputs=[((nl,nc),wire)])[0]
