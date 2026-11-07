"""A generated grid's static holds the binary64 coordinates it declares.

THE BREAKAGE (2026-10-15).  gpuwm 2.9.0's
``rw_mpas_static``, and every build of woof integrate/hex-mpas before
46c8a71513e (24b6f08c among them), reads every grid through binary32.  A
grid the point door GENERATES asks its static for binary64 coordinates
(``rw_static_coordinate_representation``), so its static declared
``binary64_earth_centred`` while holding binary32 positions up to 0.44 m off
the grid.  The door's vertical check refused the pair ("dcEdge disagrees with
spherical cell-center arc length", 2.6e-4 against 9.5e-7) and, past it,
``rw_mpas_mesh --cull-parent`` refused the static's cut ("provenance grid
latCell differs"), so no 3 km regional mesh could be built.

The fix is in the engine (gpuwm 46c8a71513e reads a binary64 grid at
binary64) and the wheel bundles that builder.  What runs here:

* the door's own check, :func:`woof.hex.mesh_point.static_coordinate_errors`,
  on small real netCDF pairs: a static as the old builder writes it is refused
  naming the cause; a static as the fixed builder writes it passes the check
  and the door's plain load (``Mesh.from_netcdf(grid, static)``, what
  ``check_halo_vertical`` runs).  CPU, every machine.
* the wheel's pins carry a fixed builder.  CPU, every machine.
* the builder itself, on a generated grid and a synthetic terrain archive,
  round-tripped through the check and the door's load: the pinned bytes when
  they are on this machine (an installed wheel's ``woof/hex/libexec``, or
  ``$WOOF_HEX_ENGINE_BRIDGES`` as setup.py reads it), or any builder named
  by ``$WOOF_HEX_TEST_STATIC_BUILDER`` -- which is how the old builder is
  shown to fail it.  Elsewhere it skips, naming what is missing.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from woof.hex import engines, mesh_point
from woof.hex.errors import MeshValidationError
from woof.hex.mesh import Mesh

from test_mesh_validation import graded_sphere_mesh

netCDF4 = pytest.importorskip("netCDF4")

ROOT = Path(__file__).resolve().parents[1]
PINS = ROOT / "../../woof/hex/data/engine-bridges.json"
#: The radius rw_mpas_static carries a unit-sphere grid to (MPAS_EARTH_RADIUS_M).
RADIUS = 6_371_229.0
BINARY64 = "binary64_earth_centred"

_DIMENSIONS = {
    "Cell": ("nCells",), "Edge": ("nEdges",), "Vertex": ("nVertices",),
    "nEdgesOnCell": ("nCells",), "areaCell": ("nCells",), "meshDensity": ("nCells",),
    "cellsOnCell": ("nCells", "maxEdges"), "edgesOnCell": ("nCells", "maxEdges"),
    "verticesOnCell": ("nCells", "maxEdges"),
    "dcEdge": ("nEdges",), "dvEdge": ("nEdges",), "nEdgesOnEdge": ("nEdges",),
    "angleEdge": ("nEdges",),
    "cellsOnEdge": ("nEdges", "TWO"), "verticesOnEdge": ("nEdges", "TWO"),
    "edgesOnEdge": ("nEdges", "maxEdges2"), "weightsOnEdge": ("nEdges", "maxEdges2"),
    "areaTriangle": ("nVertices",),
    "cellsOnVertex": ("nVertices", "vertexDegree"), "edgesOnVertex": ("nVertices", "vertexDegree"),
    "kiteAreasOnVertex": ("nVertices", "vertexDegree"),
}
_CONNECTIVITY = ("cellsOnCell", "edgesOnCell", "verticesOnCell", "cellsOnEdge", "verticesOnEdge",
                 "edgesOnEdge", "cellsOnVertex", "edgesOnVertex")


def _dimensions_of(name: str) -> tuple[str, ...]:
    if name in _DIMENSIONS:
        return _DIMENSIONS[name]
    for kind in ("Cell", "Edge", "Vertex"):
        if name.endswith(kind) and name[: -len(kind)] in ("x", "y", "z", "lat", "lon"):
            return _DIMENSIONS[kind]
    raise KeyError(name)


def _write(path: Path, mesh: Mesh, attrs: dict, arrays: dict[str, np.ndarray]) -> Path:
    """An MPAS file: 1-based connectivity, 0 padding, the arrays at their own dtypes."""

    with netCDF4.Dataset(str(path), "w", format="NETCDF3_64BIT_OFFSET") as ds:
        for name, size in {**mesh.dimensions, "TWO": 2}.items():
            ds.createDimension(name, size)
        for name, value in attrs.items():
            ds.setncattr(name, value)
        for name, values in arrays.items():
            values = np.asarray(values)
            if name in _CONNECTIVITY:
                values = np.where(values < 0, 0, values + 1).astype(np.int32)
            elif values.dtype.kind in "iu":
                values = values.astype(np.int32)
            ds.createVariable(name, values.dtype, _dimensions_of(name))[:] = values
    return path


@pytest.fixture(scope="module")
def unit_mesh() -> Mesh:
    """A defect-free graded sphere on the UNIT sphere, in binary64: what the generator writes."""

    return graded_sphere_mesh(level=4, zoom=12.0, radius=1.0, dtype=np.float64)


def _generated_grid(directory: Path, mesh: Mesh) -> Path:
    arrays = dict(mesh.arrays)
    arrays["angleEdge"] = np.zeros(mesh.dimensions["nEdges"])
    arrays["meshDensity"] = np.ones(mesh.dimensions["nCells"])
    path = _write(directory / "generated.grid.nc", mesh,
                  {"on_a_sphere": "YES", "sphere_radius": 1.0,
                   "rw_static_coordinate_representation": BINARY64}, arrays)
    with netCDF4.Dataset(str(path), "a") as ds:
        ds.createVariable("nominalMinDc", "f8", ())
        ds.variables["nominalMinDc"].assignValue(float(np.min(mesh.arrays["dcEdge"])))
    return path


def _static(directory: Path, name: str, mesh: Mesh, *, binary32_read: bool) -> Path:
    """A static as rw_mpas_static writes one for a generated grid.

    ``binary32_read`` is gpuwm 2.9.0's / 24b6f08c's arithmetic: every value
    rounded to binary32 on read and carried to the sphere in binary32, then
    stored as binary64 under the binary64 declaration.  Without it, the fixed
    builder's: the grid's binary64 values times the radius in binary64.
    Metrics are binary32 either way.
    """

    arrays = {}
    for key, values in mesh.arrays.items():
        values = np.asarray(values)
        if key in _CONNECTIVITY or values.dtype.kind in "iu":
            arrays[key] = values
            continue
        cartesian = key[:1] in "xyz" and key[1:] in ("Cell", "Edge", "Vertex")
        angle = key[:3] in ("lat", "lon") and key[3:] in ("Cell", "Edge", "Vertex")
        if cartesian or key in ("dcEdge", "dvEdge"):
            power = 1
        elif key in ("areaCell", "areaTriangle", "kiteAreasOnVertex"):
            power = 2
        else:  # the angles and weightsOnEdge
            power = 0
        if binary32_read:
            r32 = np.float32(RADIUS)
            scale32 = (np.float32(1.0), r32, r32 * r32)[power]
            stored = (values.astype(np.float32) * scale32).astype(np.float64)
        else:
            stored = values * (1.0, RADIUS, RADIUS * RADIUS)[power]
        arrays[key] = stored if (cartesian or angle) else stored.astype(np.float32)
    return _write(directory / name, mesh,
                  {"on_a_sphere": "YES", "sphere_radius": np.float32(RADIUS),
                   "rw_coordinate_representation": BINARY64,
                   "rw_coordinate_origin_xyz": np.zeros(3)}, arrays)


def test_a_static_holding_binary32_under_a_binary64_declaration_is_refused_naming_the_builder(
        tmp_path: Path, unit_mesh: Mesh) -> None:
    grid = _generated_grid(tmp_path, unit_mesh)
    static = _static(tmp_path, "old-builder.static.nc", unit_mesh, binary32_read=True)
    errors = mesh_point.static_coordinate_errors(grid, static)
    assert len(errors) == 1, errors
    message = errors[0]
    assert "old-builder.static.nc declares binary64_earth_centred" in message
    assert "reads its grid through binary32" in message and "46c8a71513e" in message
    offset = float(message.split("positions are up to ")[1].split(" m")[0])
    # Two binary32 roundings per component: the unit value (<= 0.19 m once scaled) and the
    # product at the Earth's radius (<= 0.25 m).  The timing lane measured radii 0.56 m off.
    assert 0.0 < offset < 1.0, offset
    # The symptom the timing lane met: the door's plain load refuses the pair, blaming the mesh.
    with pytest.raises(MeshValidationError, match="disagrees with spherical"):
        Mesh.from_netcdf(grid, static)


def test_a_static_that_holds_its_binary64_declaration_passes_the_check_and_the_door_load(
        tmp_path: Path, unit_mesh: Mesh) -> None:
    grid = _generated_grid(tmp_path, unit_mesh)
    static = _static(tmp_path, "fixed-builder.static.nc", unit_mesh, binary32_read=False)
    assert mesh_point.static_coordinate_errors(grid, static) == []
    loaded = Mesh.from_netcdf(grid, static)
    assert loaded.arrays["xCell"].dtype == np.float64


def test_a_native_static_is_not_judged(tmp_path: Path, unit_mesh: Mesh) -> None:
    """No declaration: native init_atmosphere rounds by design (the i2 replay proves it)."""

    grid = _generated_grid(tmp_path, unit_mesh)
    static = _static(tmp_path, "native.static.nc", unit_mesh, binary32_read=True)
    with netCDF4.Dataset(str(static), "a") as ds:
        ds.delncattr("rw_coordinate_representation")
    assert mesh_point.static_coordinate_errors(grid, static) == []


def _setup_module():
    import setuptools

    original = setuptools.setup
    setuptools.setup = lambda *args, **kwargs: None
    try:
        spec = importlib.util.spec_from_file_location("_hex_setup_static_precision", ROOT / "setup.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        setuptools.setup = original
    return module


def test_the_wheel_bundles_a_static_builder_that_holds_binary64() -> None:
    """The pins name rw_mpas_static from a woof source carrying the fix, and the
    wheel build only bundles or builds that one (setup.py's source mark)."""

    pins = json.loads(PINS.read_text(encoding="utf-8"))
    row = pins["bridges"]["rw_mpas_static"]
    assert row["source"] == "tools/rustwx/crates/rw-mpas/src/bin/rw_mpas_static/main.rs"  # a bin directory since gpuwm 2.9.7
    assert row["builder_source"] == "tools/rustwx/crates/rw-mpas/src/static_builder.rs"
    assert len(row["builder_source_sha256"]) == 64
    assert "binary64" in row["holds"]
    setup = _setup_module()
    assert "rw_mpas_static" + (".exe" if os.name == "nt" else "") in setup.ENGINE_BRIDGES
    assert setup.STATIC_BUILDER_MARK == b"fn read_as("


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _builder_under_test() -> Path:
    """The builder this machine can grade: named, else the pinned bytes, else skip."""

    named = os.environ.get("WOOF_HEX_TEST_STATIC_BUILDER")
    if named:
        return Path(named)
    pinned = json.loads(PINS.read_text(encoding="utf-8"))["bridges"]["rw_mpas_static"]["sha256"]
    candidates = [engines.bundled_candidate(engines.STATIC)]
    if os.environ.get("WOOF_HEX_ENGINE_BRIDGES"):
        candidates.append(Path(os.environ["WOOF_HEX_ENGINE_BRIDGES"]) / "rw_mpas_static")
    for candidate in candidates:
        if candidate.is_file():
            assert _sha256(candidate) == pinned, f"{candidate} is not the pinned rw_mpas_static"
            return candidate
    pytest.skip("the pinned rw_mpas_static is not on this machine (no woof/hex/libexec copy and "
                "$WOOF_HEX_ENGINE_BRIDGES unset): point WOOF_HEX_ENGINE_BRIDGES at "
                "tools/build_engine_bridges.sh's output to run the builder itself")


def _terrain_archive(root: Path) -> Path:
    """A WPS_GEOG root holding one global 2-degree terrain tile (all the overlay reads)."""

    terrain = root / "topo_gmted2010_30s"
    terrain.mkdir(parents=True)
    (terrain / "index").write_text(
        "type=continuous\nprojection=regular_ll\ndx=2.0\ndy=2.0\nknown_x=1.0\nknown_y=1.0\n"
        "known_lat=-89.0\nknown_lon=-179.0\nwordsize=2\ntile_x=180\ntile_y=90\ntile_z=1\n"
        "tile_bdr=0\nendian=big\nsigned=yes\nscale_factor=1.0\nunits=\"meters\"\n", encoding="ascii")
    y, x = np.mgrid[0:90, 0:180]
    heights = (500.0 + 400.0 * np.sin(np.radians(2 * x)) * np.cos(np.radians(2 * y - 90))).astype(">i2")
    (terrain / "00001-00180.00001-00090").write_bytes(heights.tobytes())
    return root


def test_the_static_builder_round_trips_a_generated_grid_through_the_door(
        tmp_path: Path, unit_mesh: Mesh) -> None:
    builder = _builder_under_test()
    grid = _generated_grid(tmp_path, unit_mesh)
    overlay = tmp_path / "generated-terrain.static.nc"
    done = subprocess.run([str(builder), "--grid", str(grid), "--out", str(overlay),
                           "--geog", str(_terrain_archive(tmp_path / "geog")), "--terrain-only"],
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-2000:]
    with netCDF4.Dataset(str(overlay)) as ds:
        assert ds.getncattr("rw_coordinate_representation") == BINARY64
    assert mesh_point.static_coordinate_errors(grid, overlay) == []
    Mesh.from_netcdf(grid, overlay)  # the door's parent load (check_halo_vertical)
