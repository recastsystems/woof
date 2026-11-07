"""Replay real native inputs through canonical and physical-neighbor handoffs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cupy as cp
import numpy as np

from woof.core.kernels import get_kernel, module_options, module_source
from woof.wrf_exact import ENABLED
from replay import read_capture


def replay(path: Path, label: str, edge: int | None):
    data, dims, hour, carriers, surface, moisture, expected, allocated = read_capture(path)
    nx, ny, nz, levels, theta_m, unstagger = dims
    if (nx, ny, unstagger) != (2, 2, 1):
        raise ValueError("staged proof needs a 2x2 native unstaggered capture")
    host = [a.copy() for a in carriers]
    u = host[0].reshape(nz, ny, nx + 1)
    v = host[1].reshape(nz, ny + 1, nx)
    jobs = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 1, 0]], dtype=np.int32)
    ghosts = cp.empty((1, 2, levels), dtype=cp.float32)
    phase_equal = None
    if edge is not None:
        phase_file = path.parent / f"tslist-preboundary-{label}.bin"
        phase = np.frombuffer(phase_file.read_bytes(), dtype=">f4").astype(np.float32).reshape(2, nz)
        native_ghost = v[:, 0, 1] if edge == 0 else u[:, 1, 0]
        phase_equal = dict(
            source_ghost_equal=bool(np.array_equal(phase[0].view("u4"), phase[1].view("u4"))),
            saved_native_ghost_equal=bool(np.array_equal(phase[0].view("u4"), native_ghost.view("u4"))),
            words=nz,
        )
        assert phase_equal["source_ghost_equal"] and phase_equal["saved_native_ghost_equal"]
        # Strip the native tangential halo from this real 2x2 capture. The
        # production kernel must obtain it from the actual pre-forcing phase.
        # Preserve the final normal face and final edge mass values.
        if edge == 0:
            nx = 1
            u, v = u[:, :, :2].copy(), v[:, :, :1].copy()
            jobs[0] = [0, 0, 0, 0, 1]
        else:
            ny = 1
            u, v = u[:, :1, :].copy(), v[:, :2, :].copy()
            jobs[1] = [0, 0, 0, 1, 1]
        for index in (2, 3, 4, 5, 6):
            shape = (nz + 1 if index in (2, 3, 4) else nz, 2, 2)
            host[index] = host[index].reshape(shape)[:, :ny, :nx].copy().ravel()
        for index in (7, 8):
            host[index] = host[index].reshape(2, 2)[:ny, :nx].copy().ravel()
        surface = [a.reshape(2, 2)[:ny, :nx].copy().ravel() for a in surface]
        moisture = [a.reshape(nz, 2, 2)[:, :ny, :nx].copy().ravel() for a in moisture]
        preu, prev = u.copy(), v.copy()
        if edge == 0:
            prev[:, 0, 0] = phase[0]
        else:
            preu[:, 0, 0] = phase[0]
        # Include an inactive job used by the optional fixed-address graph plan.
        edge_jobs = cp.asarray(np.concatenate((jobs[edge:edge + 1], np.full((1, 5), -1, np.int32))))
        get_kernel("tslist", "tslist_wind_boundary")((2,), (128,), (
            edge_jobs, np.int32(2), cp.asarray(preu), cp.asarray(prev), ghosts,
            np.int32(nx), np.int32(ny), np.int32(levels)))
    host[0], host[1] = u.ravel(), v.ravel()
    dc, ds, dm = ([cp.asarray(a) for a in arrays] for arrays in (host, surface, moisture))
    sp = cp.asarray([a.data.ptr for a in ds], dtype=cp.uint64)
    mp = cp.asarray([a.data.ptr for a in dm], dtype=cp.uint64)
    output = cp.empty(expected.size, dtype=cp.float32)
    ij = cp.asarray([0, 0], dtype=cp.int32)
    arguments = (
        ij, sp, mp, *dc, output,
        *(np.int32(i) for i in (nx, ny, nz, 1, levels, 0, 0, 0, theta_m)),
        hour, np.uint64(0), np.int32(0), np.uint64(0), np.int32(1))
    get_kernel("tslist", "tslist_gather")((1,), (128,), arguments, shared_mem=4 * (nz + 1))
    neighbors = cp.empty_like(ghosts)
    get_kernel("tslist", "tslist_wind_neighbors")((2,), (128,), (
        cp.asarray(jobs), np.int32(2), dc[0], dc[1], dc[7], dc[8], ghosts, neighbors,
        np.int32(nx), np.int32(ny), np.int32(levels)))
    get_kernel("tslist", "tslist_wind_finish")((1,), (128,), (
        output, neighbors, np.int32(1), np.int32(levels), np.int32(0)))
    actual = output.get()
    different = np.flatnonzero(actual.view("u4") != expected.view("u4"))
    receipt = dict(
        case=label, capture=str(path), capture_sha256=hashlib.sha256(data).hexdigest(),
        source_sha256=hashlib.sha256(module_source("tslist").encode()).hexdigest(),
        compiler_options=module_options("tslist"), strict=True, words=expected.size,
        uncomputed_allocation_tail_words_excluded=7 * (allocated - levels),
        differing_words=len(different), differences=different.tolist(), phase_witness=phase_equal)
    print(receipt, flush=True)
    if len(different):
        raise AssertionError(receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("interior", "east", "north", "receipt"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    if not ENABLED:
        parser.error("staged native proof requires GPUWM_WRF_EXACT=1")
    receipts = [replay(path, label, edge) for label, path, edge in (
        ("interior", args.interior, None), ("east", args.east, 0), ("north", args.north, 1))]
    args.receipt.write_text(json.dumps(receipts, indent=2) + "\n")


if __name__ == "__main__":
    main()
