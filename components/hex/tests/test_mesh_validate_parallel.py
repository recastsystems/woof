"""Mesh.validate on host threads gives the one-thread refusal list, word for word, in the same order.

The checks run at once and the row-wise range checks run in row blocks (conus3km start-up, 2026-10-29); the
refusals, their order and any exception a check raises must not depend on the schedule.  Each case corrupts a few
words of a global or a regional fixture and compares one host thread (every check inline, one block) with four
threads cutting every array into many small blocks.
"""

from __future__ import annotations

import numpy as np
import pytest

from woof.hex import host_parallel
from woof.hex.mesh import Mesh

from test_mesh_validation import graded_sphere_mesh
from test_regional_mesh_admission import cull_regional_mesh


def _outcome(mesh: Mesh) -> str | None:
    try:
        mesh.validate()
    except Exception as error:  # noqa: BLE001 - the text and the type are what is compared
        return f"{type(error).__name__}: {error}"
    return None


def _cases(mesh: Mesh, seed: int, count: int) -> list[list[tuple[str, int, float]]]:
    rng = np.random.default_rng(seed)
    n_cells, n_edges = int(mesh.dimensions["nCells"]), int(mesh.dimensions["nEdges"])
    index_names = [name for name in (
        "cellsOnCell", "edgesOnCell", "verticesOnCell", "cellsOnEdge", "verticesOnEdge", "cellsOnVertex",
        "edgesOnVertex", "edgesOnEdge", "nEdgesOnCell", "nEdgesOnEdge", "bdyMaskCell", "bdyMaskEdge",
        "bdyMaskVertex") if name in mesh.arrays]
    value_names = [name for name in (
        "dcEdge", "dvEdge", "areaCell", "areaTriangle", "kiteAreasOnVertex", "weightsOnEdge", "angleEdge",
        "xCell", "latCell", "lonEdge", "zVertex") if name in mesh.arrays]
    cases: list[list[tuple[str, int, float]]] = [[]]
    for _ in range(count):
        edits = []
        for _ in range(int(rng.integers(1, 4))):
            if rng.random() < 0.55:
                name = index_names[int(rng.integers(len(index_names)))]
                size = mesh.arrays[name].size
                value = [-1, -2, 0, n_cells, n_edges + 5][int(rng.integers(5))]
                if name.startswith("bdyMask"):
                    value = int(rng.integers(-1, 9))
            else:
                name = value_names[int(rng.integers(len(value_names)))]
                size = mesh.arrays[name].size
                value = [np.nan, np.inf, -1.0, 1.001, 1.0000001][int(rng.integers(5))]
            edits.append((name, int(rng.integers(size)), value))
        cases.append(edits)
    return cases


def _apply(mesh: Mesh, edits: list[tuple[str, int, float]]) -> Mesh:
    corrupt = mesh.copy(deep=True)
    for name, flat, value in edits:
        words = corrupt.arrays[name].reshape(-1)
        if words.dtype.kind == "f" and value in (-1.0, 1.001, 1.0000001):
            words[flat] = words[flat] * value
        else:
            words[flat] = value
    return corrupt


@pytest.mark.parametrize("shape", ["global", "regional"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_threaded_validation_lists_the_one_thread_refusals(monkeypatch, shape, dtype):
    sphere = graded_sphere_mesh(level=4, zoom=1.5, dtype=dtype)
    mesh = sphere if shape == "global" else cull_regional_mesh(sphere, seed_cell=0, disk_hops=12)
    corrupted = [_apply(mesh, edits) for edits in _cases(mesh, 841 + (shape == "regional"), 60)]

    monkeypatch.setenv(host_parallel.ENVIRONMENT, "1")
    serial = [_outcome(case) for case in corrupted]

    monkeypatch.setenv(host_parallel.ENVIRONMENT, "4")
    monkeypatch.setattr(host_parallel, "MIN_PARALLEL_ELEMENTS", 0)
    monkeypatch.setattr(host_parallel, "MIN_BLOCK_COLUMNS", 19)
    threaded = [_outcome(case) for case in corrupted]

    assert serial[0] is None
    assert sum(outcome is not None for outcome in serial) > 40
    assert threaded == serial
