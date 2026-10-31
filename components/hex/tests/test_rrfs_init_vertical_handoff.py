"""Actual RRFS worker must consume boundary meteorology through the public hook.

RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_cases.F:4468-4483,
10100-10295 at 1d6e370e. Artifact-gated interface regression; complete native
file replay is the independent numerical authority.
"""
from dataclasses import fields
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from woof.hex.vertical import (
    _producer_vertical_grid,
    _producer_vertical_request,
    build_vertical_grid,
)


def test_real_worker_boundary_terrain_reaches_vertical_hook():
    worker = os.environ.get("R10_RRFS_VERTICAL_WORKER")
    met = os.environ.get("R10_RRFS_BOUNDARY_MET")
    if not worker or not met:
        pytest.skip("requires RRFS initializer worker and real decoded SOILHGT input")
    assert Path(worker).is_file() and Path(met).is_file()
    mesh = SimpleNamespace(
        areaCell=np.full(4, 1e6), nEdgesOnCell=np.full(4, 3),
        edgesOnCell=np.array([[0, 1, 2], [0, 3, 4], [1, 3, 5], [2, 4, 5]]),
        cellsOnCell=np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]),
        cellsOnEdge=np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]),
        dcEdge=np.full(6, 1000.), dvEdge=np.full(6, 1000.),
        deriv_two=np.zeros((15, 2, 6)),
        latCell=np.full(4, np.deg2rad(39.), dtype=np.float32),
        lonCell=np.full(4, np.deg2rad(-105.), dtype=np.float32),
        bdyMaskCell=np.array([0, 1, 5, 6]),
    )
    terrain = np.zeros(4, dtype=np.float32)
    producer = dict(executable=worker, stages=["vertical"], abi=(
        "rw_mpas_init_v831 stdin: hex-init-job/v1 JSON; "
        "stage=case7|case9|static|terrain|vertical|artifact|prepare|gwd|surface"))
    options = dict(n_vert_levels=3, ztop=30000., terrain_smoothing_passes=0,
        smooth_surfaces=False, surface_smoothing_passes=0, hybrid_coordinate=False)
    baseline = build_vertical_grid(mesh, terrain, producer=producer, **options)
    actual = build_vertical_grid(mesh, terrain, producer=producer,
        boundary_met_path=met, **options)
    # Real mountains and distinct native mask weights make dropping the file
    # scientifically observable, independent of request serialization details.
    assert actual.zgrid[0, 0] == baseline.zgrid[0, 0]
    assert np.all(actual.zgrid[0, 1:] > 0)
    assert actual.zgrid[0, 1] < actual.zgrid[0, 2]
    assert actual.zgrid[0, 2] == actual.zgrid[0, 3]
    assert actual.zgrid.tobytes() != baseline.zgrid.tobytes()

    request = _producer_vertical_request(mesh, terrain, 3, top=30000.,
        terrain_passes=0, smooth=False, surface_passes=0, minimum_fraction=.3,
        hybrid=False, transition=30000., damping=0., damping_start=22000.,
        regional_halo=False, order=3)
    request["boundary"] = dict(met_path=met, lat=mesh.latCell.tolist(),
        lon=mesh.lonCell.tolist(), mask=mesh.bdyMaskCell.tolist())
    expected = _producer_vertical_grid(producer, request)
    for field in fields(actual):
        left, right = getattr(actual, field.name), getattr(expected, field.name)
        if isinstance(left, np.ndarray):
            assert left.dtype == right.dtype and left.shape == right.shape
            assert left.tobytes() == right.tobytes(), field.name
        else:
            assert left == right, field.name
    assert np.count_nonzero(terrain) == 0
