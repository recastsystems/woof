"""The mapped exporter may omit only authenticated mesh support's unused corners."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from woof.hex.packaged_intermediate import _target_support


def test_mesh_and_generated_plan_seal_the_same_reader_support(tmp_path):
    mesh = tmp_path / "mesh.nc"
    mesh.write_bytes(b"mesh identity")
    box = "-2,-2,1,1,7,7"
    direct = _target_support({"from_grid": str(mesh)}, box, tmp_path)
    doc = json.loads(direct[0].read_text())
    assert doc["mesh"] == {"path": str(mesh), "sha256": hashlib.sha256(mesh.read_bytes()).hexdigest()}
    assert doc["consumer"] == "mpas-case7-f32"
    assert doc["target_latlon"] == box
    assert direct[1] == hashlib.sha256(direct[0].read_bytes()).hexdigest()
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"cull": {"grid": str(mesh)}}))
    assert _target_support({"from_plan": str(plan)}, box, tmp_path) == direct
    assert _target_support({}, box, tmp_path) is None
    plan.write_text(json.dumps({"plan": {"cull_region": {}}}))
    assert _target_support({"from_plan": str(plan)}, box, tmp_path) is None


@pytest.fixture
def engine():
    from woof.mapped_engine_bridge import require_engine
    try:
        binary = require_engine()
    except (FileNotFoundError, RuntimeError) as error:
        pytest.skip(str(error))
    capabilities = subprocess.run([str(binary), "capabilities"], capture_output=True, text=True, check=True)
    if not json.loads(capabilities.stdout.splitlines()[-1]).get("wps_mesh_support_v1"):
        pytest.skip("mapped engine predates mesh support")
    return binary


def _mesh(path, *, edge=0.):
    nc = pytest.importorskip("netCDF4")
    with nc.Dataset(path, "w", format="NETCDF3_64BIT_DATA") as dataset:
        for label in ["Cell", "Edge"]:
            dataset.createDimension("n" + label, 1)
            for axis in ["lat", "lon"]:
                variable = dataset.createVariable(axis + label, "f8", ("n" + label,))
                variable[:] = edge if label == "Edge" else 0.
    return path


def _run(engine, support, tmp_path, *, box="-2,-2,1,1,7,7", digest=None):
    result = subprocess.run([str(engine), "wps", "--target-latlon", box,
                             "--target-support", str(support[0]),
                             "--target-support-sha256", digest or support[1],
                             "--frameset", str(tmp_path / "absent"),
                             "--out", str(tmp_path / "output")], capture_output=True, text=True)
    assert result.returncode == 2
    return json.loads(result.stderr.splitlines()[-1])["message"]


def test_real_engine_rejects_changed_contract_lattice_and_mesh(tmp_path, engine):
    mesh = _mesh(tmp_path / "mesh.nc")
    support = _target_support({"from_grid": str(mesh)}, "-2,-2,1,1,7,7", tmp_path)
    assert "contract digest differs" in _run(engine, support, tmp_path, digest="0" * 64)
    assert "lattice differs" in _run(engine, support, tmp_path, box="-3,-2,1,1,7,7")
    mesh.write_bytes(mesh.read_bytes() + b"changed")
    assert "mesh digest differs" in _run(engine, support, tmp_path)


def test_real_engine_includes_edges_when_cells_alone_would_fit(tmp_path, engine):
    mesh = _mesh(tmp_path / "mesh.nc", edge=1.)
    support = _target_support({"from_grid": str(mesh)}, "-2,-2,1,1,7,7", tmp_path)
    assert "mesh interpolation support leaves the target lattice" in _run(engine, support, tmp_path)


def test_generated_box_retains_outer_edge_stencils(tmp_path, engine):
    from woof.hex.hrrr_intermediate import target_from_mesh
    mesh = _mesh(tmp_path / "mesh.nc", edge=1.)
    target, basis = target_from_mesh(mesh, None, margin_km=0., spacing_deg=1.)
    box = f"{target.south},{target.west},{target.dlat},{target.dlon},{target.nx},{target.ny}"
    support = _target_support({"from_grid": str(mesh)}, box, tmp_path)
    assert basis["cells"] == basis["edges"] == 1
    frameset = tmp_path / "absent"
    frameset.mkdir()
    (frameset / "frames.json").write_text("null")
    # Reaching frameset decoding proves the real consumer accepted every
    # cell/edge stencil, the mesh digest and the generated lattice together.
    assert "invalid type: null" in _run(engine, support, tmp_path, box=box)
