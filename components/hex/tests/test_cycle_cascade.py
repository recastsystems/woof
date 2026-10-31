"""The cycling cascade: the delayed-start seam, the cascade row, the door.

Card-free by construction.  What needs a device -- the contract deck and the
fine forecast -- is proved on hardware and its receipts are the evidence;
what is tested here is every decision and every refusal around them.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import re
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from woof.hex import cascade_row  # noqa: E402
from woof.hex.cycle import chain, delayed_start  # noqa: E402
from woof.hex.cycle.errors import CycleRefusal, DelayedStartRefusal  # noqa: E402
from _layout import PACKAGE_DIR

netCDF4 = pytest.importorskip("netCDF4")

PARENT_CELLS = 12
PARENT_EDGES = 20
CHILD_PICK = (7, 2, 9, 0, 5)
CHILD_EDGE_PICK = (3, 11, 0, 17)
LEVELS = 4
SOIL = 2


def _coords(count: int, offset: float) -> tuple[np.ndarray, np.ndarray]:
    lat = np.linspace(-0.4, 0.4, count).astype(np.float64) + offset
    lon = np.linspace(0.1, 1.1, count).astype(np.float64) + offset
    return lat, lon


@pytest.fixture
def bundle(tmp_path: Path) -> dict[str, Path]:
    """A parent grid, a parent history frame, a cull grid and a culled init.

    The cull's coordinates are the parent's own float64 bits for the cells it
    kept -- which is what a real ``rw_mpas_mesh --cull-parent`` produces,
    because a cull moves no cell centre -- and it renumbers, which is also
    what a real cull does.  So the map has to be built on coordinates.
    """

    parent_lat, parent_lon = _coords(PARENT_CELLS, 0.0)
    parent_elat, parent_elon = _coords(PARENT_EDGES, 5.0)

    parent_grid = tmp_path / "parent.grid.nc"
    with netCDF4.Dataset(str(parent_grid), "w", format="NETCDF3_CLASSIC") as ds:
        ds.createDimension("nCells", PARENT_CELLS)
        ds.createDimension("nEdges", PARENT_EDGES)
        ds.createVariable("latCell", "f8", ("nCells",))[:] = parent_lat
        ds.createVariable("lonCell", "f8", ("nCells",))[:] = parent_lon
        ds.createVariable("latEdge", "f8", ("nEdges",))[:] = parent_elat
        ds.createVariable("lonEdge", "f8", ("nEdges",))[:] = parent_elon

    history = tmp_path / "parent.history.nc"
    with netCDF4.Dataset(str(history), "w", format="NETCDF3_CLASSIC") as ds:
        ds.createDimension("Time", 1)
        ds.createDimension("nCells", PARENT_CELLS)
        ds.createDimension("nEdges", PARENT_EDGES)
        ds.createDimension("nVertLevels", LEVELS)
        ds.createDimension("nVertLevelsP1", LEVELS + 1)
        ds.createDimension("nSoilLevels", SOIL)
        ds.createVariable("theta", "f4", ("Time", "nCells", "nVertLevels"))[:] = (
            np.arange(PARENT_CELLS * LEVELS, dtype=np.float32).reshape(
                1, PARENT_CELLS, LEVELS
            )
            + 300.0
        )
        ds.createVariable("rho", "f4", ("Time", "nCells", "nVertLevels"))[:] = 1.0 + (
            np.arange(PARENT_CELLS * LEVELS, dtype=np.float32).reshape(
                1, PARENT_CELLS, LEVELS
            )
            / 1000.0
        )
        ds.createVariable("qv", "f4", ("Time", "nCells", "nVertLevels"))[:] = 0.001
        ds.createVariable("w", "f4", ("Time", "nCells", "nVertLevelsP1"))[:] = 0.25
        ds.createVariable("normal_u", "f4", ("Time", "nEdges", "nVertLevels"))[:] = (
            np.arange(PARENT_EDGES * LEVELS, dtype=np.float32).reshape(
                1, PARENT_EDGES, LEVELS
            )
        )
        ds.createVariable("t2", "f4", ("Time", "nCells"))[:] = 288.0
        ds.createVariable("tslb", "f4", ("Time", "nCells", "nSoilLevels"))[:] = 280.0
        # Published by the parent and with NO slot in the init stream: the
        # ice-phase condensate a delayed start cannot carry.
        ds.createVariable("qi", "f4", ("Time", "nCells", "nVertLevels"))[:] = 1e-5
        for name, value in {"smois": .25, "sh2o": .2}.items():
            ds.createVariable(name, "f4", ("Time", "nCells", "nSoilLevels"))[:] = value
        for name, value in {"skintemp": 301, "canwat": .01, "snow": 1,
                            "snowh": .05, "snowc": .1, "snotime": 300}.items():
            ds.createVariable(name, "f4", ("Time", "nCells"))[:] = value

    child_lat = parent_lat[list(CHILD_PICK)]
    child_lon = parent_lon[list(CHILD_PICK)]
    child_elat = parent_elat[list(CHILD_EDGE_PICK)]
    child_elon = parent_elon[list(CHILD_EDGE_PICK)]

    child_grid = tmp_path / "child.grid.nc"
    with netCDF4.Dataset(str(child_grid), "w", format="NETCDF3_CLASSIC") as ds:
        ds.createDimension("nCells", len(CHILD_PICK))
        ds.createDimension("nEdges", len(CHILD_EDGE_PICK))
        ds.createVariable("latCell", "f8", ("nCells",))[:] = child_lat
        ds.createVariable("lonCell", "f8", ("nCells",))[:] = child_lon
        ds.createVariable("latEdge", "f8", ("nEdges",))[:] = child_elat
        ds.createVariable("lonEdge", "f8", ("nEdges",))[:] = child_elon
        # A cull RENUMBERS: its indexToCellID is 1..N, not the parent's ids.
        ds.createVariable("indexToCellID", "i4", ("nCells",))[:] = np.arange(
            1, len(CHILD_PICK) + 1
        )

    init = tmp_path / "child.init.nc"
    with netCDF4.Dataset(str(init), "w", format="NETCDF3_CLASSIC") as ds:
        ds.createDimension("Time", 1)
        ds.createDimension("StrLen", 64)
        ds.createDimension("nCells", len(CHILD_PICK))
        ds.createDimension("nEdges", len(CHILD_EDGE_PICK))
        ds.createDimension("nVertLevels", LEVELS)
        ds.createDimension("nVertLevelsP1", LEVELS + 1)
        ds.createDimension("nSoilLevels", SOIL)
        stamp = ds.createVariable("xtime", "S1", ("Time", "StrLen"))
        stamp[0] = np.array(list("2026-08-12_06:00:00".ljust(64)), dtype="S1")
        # An init's clock lives in three places and the door asserts against
        # THIS one.
        ds.setncattr("config_start_time", "2026-08-12_06:00:00")
        ds.createVariable("theta", "f4", ("Time", "nCells", "nVertLevels"))[:] = 0.0
        ds.createVariable("rho", "f4", ("Time", "nCells", "nVertLevels"))[:] = 0.0
        ds.createVariable("qv", "f4", ("Time", "nCells", "nVertLevels"))[:] = 0.0
        ds.createVariable("w", "f4", ("Time", "nCells", "nVertLevelsP1"))[:] = 0.0
        ds.createVariable("u", "f4", ("Time", "nEdges", "nVertLevels"))[:] = 0.0
        ds.createVariable("t2m", "f4", ("Time", "nCells"))[:] = 0.0
        ds.createVariable("skintemp", "f4", ("Time", "nCells"))[:] = 280.0
        ds.createVariable("tslb", "f4", ("Time", "nCells", "nSoilLevels"))[:] = 0.0
        for name, value in {"isltyp": 7, "ivgtyp": 7, "xland": 1, "xice": 0}.items():
            ds.createVariable(name, "f4", ("Time", "nCells"))[:] = value
        ds.createVariable("isice", "i4", ())[:] = 15
        ds.createVariable("mminlu", "S1", ("StrLen",))[:] = np.array(list("USGS".ljust(64)), dtype="S1")
        # Mandatory snow memory replaces the cold-init value.
        ds.createVariable("snowh", "f4", ("Time", "nCells"))[:] = 0.42
        ds.createVariable("tke", "f4", ("Time", "nCells", "nVertLevels"))[:] = 0.11

    return {
        "parent_grid": parent_grid,
        "history": history,
        "child_grid": child_grid,
        "init": init,
    }


# ---------------------------------------------------------------------------
# the delayed-start seam
# ---------------------------------------------------------------------------
def test_a_mid_window_state_lands_on_the_right_cells(bundle, tmp_path):
    """The transplant, checked value by value against the parent it came from.

    A cull renumbers, so the gather is by COORDINATE and a wrong map would put
    one column's atmosphere on another column.  This asserts the map, not the
    plumbing: every child cell must hold exactly its own parent cell's profile.
    """

    valid = datetime(2026, 8, 12, 12, 0, 0)
    report = delayed_start.compose_mid_window_init(
        child_init=bundle["init"],
        child_grid=bundle["child_grid"],
        parent_grid=bundle["parent_grid"],
        parent_history=bundle["history"],
        valid_time=valid,
        receipt_path=tmp_path / "delayed-start.json",
    )
    assert report.cells_matched == len(CHILD_PICK)
    assert report.edges_matched == len(CHILD_EDGE_PICK)
    assert report.valid_time == "2026-08-12_12:00:00"
    assert report.init_time_before.startswith("2026-08-12_06:00:00")

    with netCDF4.Dataset(str(bundle["history"])) as history, netCDF4.Dataset(
        str(bundle["init"])
    ) as init:
        history.set_auto_maskandscale(False)
        init.set_auto_maskandscale(False)
        parent_theta = np.asarray(history.variables["theta"][0])
        child_theta = np.asarray(init.variables["theta"][0])
        for child_index, parent_index in enumerate(CHILD_PICK):
            assert np.array_equal(child_theta[child_index], parent_theta[parent_index])
        parent_u = np.asarray(history.variables["normal_u"][0])
        child_u = np.asarray(init.variables["u"][0])
        for child_index, parent_index in enumerate(CHILD_EDGE_PICK):
            assert np.array_equal(child_u[child_index], parent_u[parent_index])
        # The clock moved with the state, and only after it.
        raw = np.asarray(init.variables["xtime"][0])
        assert raw.tobytes().decode().strip() == "2026-08-12_12:00:00"
        # ALL THREE CLOCKS MOVE TOGETHER.  Moving two of the three produces a
        # file whose own global attribute contradicts its own variable, and
        # the forecast door refuses it by name -- measured on the first real
        # cascade run, and the cause was in the transplant.
        assert init.getncattr("config_start_time") == "2026-08-12_12:00:00"
        assert (
            init.getncattr("gpuwm_hex_delayed_start_config_start_time_before")
            == "2026-08-12_06:00:00"
        )
        assert float(np.asarray(init.variables["snowh"][0])[0]) == pytest.approx(0.05)
        for name in ("smois", "sh2o", "tslb", "skintemp", "canwat", "snow", "snowh", "snowc", "snotime"):
            np.testing.assert_array_equal(init[name][0], history[name][0][list(CHILD_PICK)])


    receipt = json.loads((tmp_path / "delayed-start.json").read_text(encoding="utf-8"))
    carried = {row["field"] for row in receipt["carried"]}
    assert {"theta", "rho", "qv", "w", "u", "t2m", "tslb"} <= carried
    not_carried = {row["field"] for row in receipt["not_carried"]}
    # The ice phase, named, with the parent's own publication state recorded.
    assert {"qi", "qs", "qg"} <= not_carried
    assert "tke" in not_carried
    assert "snowh" in carried
    ice = [row for row in receipt["not_carried"] if row["field"] == "qi"][0]
    assert ice["present_in_parent_history"] is True
    assert "ice" in ice["reason"]


def test_the_hex_history_names_are_read(bundle):
    """The hex history publishes u, t2m, skintemp, sh2o and snowh under those
    names, and the composed init takes them.  Breakage it prevents: the table
    read normal_u/t2/tsk and had no sh2o/snow/snowh, so a cycled 24-member
    ensemble's backgrounds kept the cold start's 2 m and skin temperature
    (0 K member spread, 2026-10-28)."""

    with netCDF4.Dataset(str(bundle["history"]), "a") as ds:
        ds.renameVariable("normal_u", "u")
        ds.renameVariable("t2", "t2m")

    with netCDF4.Dataset(str(bundle["init"]), "a") as ds:
        ds.variables["skintemp"][:] = 0.0
        ds.createVariable("sh2o", "f4", ("Time", "nCells", "nSoilLevels"))[:] = 0.0
    report = delayed_start.compose_mid_window_init(
        child_init=bundle["init"],
        child_grid=bundle["child_grid"],
        parent_grid=bundle["parent_grid"],
        parent_history=bundle["history"],
        valid_time=datetime(2026, 8, 12, 12, 0, 0),
    )
    carried = {row["field"]: row["from"] for row in report.carried}
    for name in ("u", "t2m", "skintemp", "sh2o", "snowh"):
        assert carried[name] == name
    assert not {"u", "t2m", "skintemp", "sh2o", "snowh"} & {row["field"] for row in report.not_carried}
    with netCDF4.Dataset(str(bundle["history"])) as history, netCDF4.Dataset(str(bundle["init"])) as init:
        parent_u = np.asarray(history.variables["u"][0])
        child_u = np.asarray(init.variables["u"][0])
        for child_index, parent_index in enumerate(CHILD_EDGE_PICK):
            assert np.array_equal(child_u[child_index], parent_u[parent_index])
        assert np.all(np.asarray(init.variables["t2m"][0]) == np.float32(288.0))
        assert np.all(np.asarray(init.variables["skintemp"][0]) == np.float32(301.0))
        assert np.all(np.asarray(init.variables["sh2o"][0]) == np.float32(0.2))
        assert np.all(np.asarray(init.variables["snowh"][0]) == np.float32(0.05))


def test_a_one_dimensional_initial_time_moves_whole(bundle):
    """The port's inits (hex >= 0.4.6) write ``initial_time`` as ``(StrLen,)``.

    ``[0]`` on that shape is ONE CHARACTER: the transplant wrote only it and
    left ``' 025-05-19_06:00:00'``, and every member started from the
    composed init stopped with "Invalid isoformat string".  The whole stamp
    moves, on both shapes, and the 2-D ``xtime`` beside it still moves too.
    """

    with netCDF4.Dataset(str(bundle["init"]), "a") as ds:
        stamp = ds.createVariable("initial_time", "S1", ("StrLen",))
        stamp[:] = np.array(list("2026-08-12_06:00:00".ljust(64)), dtype="S1")
    delayed_start.compose_mid_window_init(
        child_init=bundle["init"],
        child_grid=bundle["child_grid"],
        parent_grid=bundle["parent_grid"],
        parent_history=bundle["history"],
        valid_time=datetime(2026, 8, 12, 12, 0, 0),
    )
    with netCDF4.Dataset(str(bundle["init"])) as init:
        init.set_auto_maskandscale(False)
        for name in ("initial_time", "xtime"):
            text = np.asarray(init.variables[name][:]).tobytes().decode()
            assert text.strip() == "2026-08-12_12:00:00", (name, text)
            datetime.fromisoformat(text.strip().replace("_", "T"))


def test_a_cell_the_parent_does_not_have_refuses_by_name(bundle):
    """A miss is a refusal, never a nearest neighbour."""

    with netCDF4.Dataset(str(bundle["child_grid"]), "a") as ds:
        ds.variables["latCell"][2] = 88.123456789
    with pytest.raises(DelayedStartRefusal) as excinfo:
        delayed_start.compose_mid_window_init(
            child_init=bundle["init"],
            child_grid=bundle["child_grid"],
            parent_grid=bundle["parent_grid"],
            parent_history=bundle["history"],
            valid_time=datetime(2026, 8, 12, 12, 0, 0),
        )
    message = str(excinfo.value)
    assert "exact coordinate bits" in message
    assert "another column" in message


def test_a_frame_from_another_mesh_refuses_by_name(bundle, tmp_path):
    """The gather is by parent index; a frame of a different size would read
    another mesh's rows."""

    other = tmp_path / "other.history.nc"
    with netCDF4.Dataset(str(other), "w", format="NETCDF3_CLASSIC") as ds:
        ds.createDimension("Time", 1)
        ds.createDimension("nCells", PARENT_CELLS + 3)
        ds.createDimension("nEdges", PARENT_EDGES)
        ds.createDimension("nVertLevels", LEVELS)
        ds.createVariable("theta", "f4", ("Time", "nCells", "nVertLevels"))[:] = 1.0
    with pytest.raises(DelayedStartRefusal, match="different mesh"):
        delayed_start.compose_mid_window_init(
            child_init=bundle["init"],
            child_grid=bundle["child_grid"],
            parent_grid=bundle["parent_grid"],
            parent_history=other,
            valid_time=datetime(2026, 8, 12, 12, 0, 0),
        )


def test_the_history_stream_carries_no_valid_time_and_that_is_reported(bundle):
    """The delayed-start seam, as a measurement rather than an assumption.

    The port's history stream is a PRODUCT stream: no ``xtime``.  The reader
    returns ``None`` so a caller states the time it is composing for instead
    of inferring one that is not there.
    """

    assert delayed_start.frame_valid_time(bundle["history"]) is None


# ---------------------------------------------------------------------------
# the cascade row
# ---------------------------------------------------------------------------
def _row(**overrides):
    base = dict(
        name="cascade-c01-s01",
        parent_row="x1.40962",
        n_cells=10,
        n_edges=20,
        n_levels=4,
        n_interfaces=5,
        n_soil_levels=2,
        nominal_dx_m=4000.0,
        dt_seconds=20.0,
        grid="grid.nc",
        grid_bytes=1,
        grid_sha256="a" * 64,
        static="static.nc",
        static_bytes=1,
        static_sha256="b" * 64,
        boundary_zone_width=7,
        bdy_mask_sha256="c" * 64,
        lbc_source="parent/lbc",
        cull_receipt="receipt.json",
        cycle_index=1,
        slot_id="s01",
        cull_pad_scale=1.35,
    )
    base.update(overrides)
    return base


def _registry_stub():
    import mpas_mesh_binding as binding

    return {"x1.40962": binding.MESH_BINDINGS["x1.40962"]}, binding.MeshBinding


def _write(tmp_path: Path, rows) -> Path:
    path = tmp_path / "rows.json"
    path.write_text(
        json.dumps({"schema": "gpuwm-hex.cascade-rows/v1", "rows": rows}),
        encoding="utf-8",
    )
    return path


def test_a_cull_of_an_unregistered_parent_refuses_by_name(tmp_path):
    """Lineage stops at a row a person registered."""

    registry, binding_type = _registry_stub()
    path = _write(tmp_path, [_row(parent_row="not-a-row")])
    with pytest.raises(cascade_row.CascadeRowRefusal) as excinfo:
        cascade_row.apply_rows(registry, binding_type, path)
    assert "not a registered mesh row" in str(excinfo.value)
    assert "nobody ever looked at" in str(excinfo.value)


def test_a_cascade_row_may_not_shadow_a_shipped_one(tmp_path):
    registry, binding_type = _registry_stub()
    path = _write(tmp_path, [_row(name="x1.40962")])
    with pytest.raises(cascade_row.CascadeRowRefusal, match="may never shadow"):
        cascade_row.apply_rows(registry, binding_type, path)


def test_bytes_that_moved_under_the_row_refuse(tmp_path):
    """The row is written FROM the cull receipt, so a mismatch means the file
    moved under it -- which is the whole reason a registry pins bytes."""

    registry, binding_type = _registry_stub()
    grid = tmp_path / "grid.nc"
    grid.write_bytes(b"0123456789")
    static = tmp_path / "static.nc"
    static.write_bytes(b"0123456789")
    path = _write(
        tmp_path,
        [_row(grid=str(grid), static=str(static), grid_bytes=10, static_bytes=10)],
    )
    with pytest.raises(cascade_row.CascadeRowRefusal, match="SHA-256"):
        cascade_row.apply_rows(registry, binding_type, path)


def test_a_boundary_zone_with_no_boundary_series_refuses(tmp_path):
    registry, binding_type = _registry_stub()
    grid = tmp_path / "grid.nc"
    grid.write_bytes(b"x")
    static = tmp_path / "static.nc"
    static.write_bytes(b"x")
    import hashlib

    digest = hashlib.sha256(b"x").hexdigest()
    path = _write(
        tmp_path,
        [
            _row(
                grid=str(grid), static=str(static),
                grid_bytes=1, static_bytes=1,
                grid_sha256=digest, static_sha256=digest,
                lbc_source="",
            )
        ],
    )
    with pytest.raises(cascade_row.CascadeRowRefusal, match="unforced boundary"):
        cascade_row.apply_rows(registry, binding_type, path)


def test_a_good_cascade_row_registers_and_says_what_it_is(tmp_path):
    import hashlib

    registry, binding_type = _registry_stub()
    grid = tmp_path / "grid.nc"
    grid.write_bytes(b"grid")
    static = tmp_path / "static.nc"
    static.write_bytes(b"static")
    path = _write(
        tmp_path,
        [
            _row(
                grid=str(grid), static=str(static),
                grid_bytes=4, static_bytes=6,
                grid_sha256=hashlib.sha256(b"grid").hexdigest(),
                static_sha256=hashlib.sha256(b"static").hexdigest(),
            )
        ],
    )
    patched = cascade_row.apply_rows(registry, binding_type, path)
    row = patched["cascade-c01-s01"]
    assert row.n_cells == 10
    assert row.dt_seconds == 20.0
    assert cascade_row.CASCADE_ROW_MARKER in row.notes
    # The row says, in its own words, what it does and does not have.
    assert "Nobody hand-wrote this row" in row.notes
    assert "does NOT have is a person who read it" in row.notes
    assert "x1.40962" in row.notes
    # The shipped rows are untouched.
    assert patched["x1.40962"] is registry["x1.40962"]


def test_no_rows_leaves_the_registry_alone():
    registry, binding_type = _registry_stub()
    assert cascade_row.apply_rows(registry, binding_type, None) is registry


# ---------------------------------------------------------------------------
# the loop and its door
# ---------------------------------------------------------------------------
FORBIDDEN_TOKENS = re.compile(
    r"("
    r"cyclone|hurricane|typhoon|tornado|derecho|convectiv|fire[_ -]?weather|"
    r"atmospheric[_ -]?river|winter[_ -]?storm|blizzard|"
    r"conus|ohio|real74|hrrr"
    r")",
    re.IGNORECASE,
)


def _executable_tokens(source: str) -> list[tuple[int, str]]:
    """Every name and every live string constant, docstrings excluded.

    The scan has to be over what the module DOES, not over what it says.  A
    docstring that names four phenomena in order to state that none of them
    reaches the code is the design working; an ``if threat_class ==`` on any
    one of them is the design failing.  Comments are excluded for the same
    reason -- ``ast`` drops them -- and docstrings are dropped explicitly.
    """

    import ast as _ast

    tree = _ast.parse(source)
    docstrings: set[int] = set()
    for node in _ast.walk(tree):
        if isinstance(
            node, (_ast.Module, _ast.FunctionDef, _ast.AsyncFunctionDef, _ast.ClassDef)
        ):
            first = node.body[0] if node.body else None
            if (
                isinstance(first, _ast.Expr)
                and isinstance(first.value, _ast.Constant)
                and isinstance(first.value.value, str)
            ):
                docstrings.add(id(first.value))
    found: list[tuple[int, str]] = []
    for node in _ast.walk(tree):
        line = getattr(node, "lineno", 0)
        if isinstance(node, _ast.Name):
            found.append((line, node.id))
        elif isinstance(node, _ast.Attribute):
            found.append((line, node.attr))
        elif isinstance(node, _ast.arg):
            found.append((line, node.arg))
        elif isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef, _ast.ClassDef)):
            found.append((line, node.name))
        elif isinstance(node, _ast.keyword) and node.arg:
            found.append((line, node.arg))
        elif isinstance(node, _ast.Constant) and isinstance(node.value, str):
            if id(node) not in docstrings:
                found.append((line, node.value))
    return found


def test_the_cascade_knows_nothing_about_any_phenomenon():
    """THE ARBITRARY ACCEPTANCE TEST, as a property of the executable source.

    A tropical cyclone, a convective area, a fire-weather region and an
    atmospheric river are ROWS in threat-metrics.v3.  They reach this loop as
    admitted entries carrying a cull region, a mesh spec, an ignition hour and
    a pad, and the loop cannot tell them apart.  If any phenomenon -- or any
    case, region or storm name -- appears in an identifier, an argument name
    or a live string of the cascade's own code, the design failed and the next
    phenomenon will need a code path.
    """

    offences: list[str] = []
    surfaces = sorted((PACKAGE_DIR / "cycle").glob("*.py"))
    surfaces.append(PACKAGE_DIR / "cascade_row.py")
    for path in surfaces:
        for line, token in _executable_tokens(path.read_text(encoding="utf-8")):
            if FORBIDDEN_TOKENS.search(token):
                offences.append(f"{path.name}:{line}: {token!r}")
    assert not offences, (
        "phenomenon or case names in the cascade's executable source:\n"
        + "\n".join(offences)
    )


def test_a_late_swath_with_no_parent_stream_refuses_rather_than_starting_early():
    """The refusal names what it prevents: hours of fine forecast over
    atmosphere nobody placed a grid for."""

    config = chain.CascadeConfig(
        out=Path("."), parent_row="x1.40962",
        parent_grid=Path("g"), parent_static=Path("s"), parent_init=Path("i"),
        parent_history=None, coarse_history=Path("c"),
        coarse_parent_grid=Path("p"), gpuwm_checkout=Path("a"), repo=ROOT,
    )
    with pytest.raises(CycleRefusal) as excinfo:
        chain.run_slot(
            config,
            {
                "slot_id": "s01",
                "cull_region": {"kind": "cap", "center_deg": [0.0, 0.0],
                                "radius_km": 100.0},
                "ignite_at_seconds": 6 * 3600.0,
            },
            cycle_index=1,
            cycle_start=datetime(2026, 8, 12, 6, 0, 0),
            coarse=chain.ParentStream(Path("c"), Path("g"), Path("s"), ()),
            parent=None,
            work=Path("."),
            ledger=Path("."),
        )
    message = str(excinfo.value)
    assert "--parent-history" in message
    assert "nobody placed a grid for" in message


def test_a_parent_stream_with_no_frame_at_the_wanted_hour_refuses():
    stream = chain.ParentStream(
        Path("receipt.json"), Path("g"), Path("s"),
        (
            (datetime(2026, 8, 12, 6, 0, 0), Path("a.nc"), ""),
            (datetime(2026, 8, 12, 7, 0, 0), Path("b.nc"), ""),
        ),
    )
    with pytest.raises(CycleRefusal) as excinfo:
        stream.at(datetime(2026, 8, 12, 6, 30, 0))
    assert "publishes no frame valid at" in str(excinfo.value)
    assert "history cadence" in str(excinfo.value)


def test_the_window_receipt_carries_its_own_limit(tmp_path):
    """One parent integration windowed per cycle is NOT N parent runs, and the
    receipt has to say so rather than letting a reader assume it."""

    frames = tuple(
        (datetime(2026, 8, 12, 6 + hour, 0, 0), tmp_path / f"f{hour}.nc", "")
        for hour in range(6)
    )
    for _, path, _ in frames:
        path.write_bytes(b"x")
    stream = chain.ParentStream(tmp_path / "r.json", Path("g"), Path("s"), frames)
    out = chain.window_receipt(
        stream, datetime(2026, 8, 12, 8, 0, 0), datetime(2026, 8, 12, 11, 0, 0),
        tmp_path / "window.json",
    )
    document = json.loads(out.read_text(encoding="utf-8"))
    assert document["forecast"]["history_labels"]["0"] == "2026-08-12_08.00.00"
    assert len(document["forecast"]["snapshot_files"]) == 4
    assert "older initial condition" in document["what_this_is"]
    assert "worse guidance, not better" in document["what_this_is"]


def test_the_door_has_no_per_phenomenon_flag():
    """A mode per threat is the failure this design exists to avoid."""

    from woof.hex.cli import build_parser

    parser = build_parser()
    assert "cycle" in parser.format_help()
    door = argparse.ArgumentParser()
    from woof.hex.cycle.door import _add_arguments

    _add_arguments(door)
    flags = [
        option
        for action in door._actions
        for option in action.option_strings
    ]
    assert "--parent-history" in flags
    offending = [flag for flag in flags if FORBIDDEN_TOKENS.search(flag)]
    assert not offending, f"per-phenomenon flags on the cycle door: {offending}"


def test_the_fine_forecast_carries_the_callers_forecast_flags_last(tmp_path, monkeypatch):
    """hex parity run group 7, #27: `cycle run` had no flag that could ask for a physics suite."""

    from woof.hex.cycle.door import _add_arguments, _config

    door = argparse.ArgumentParser()
    _add_arguments(door)
    paths = {flag: str(tmp_path / flag.strip("-")) for flag in (
        "--out", "--parent-grid", "--parent-static", "--parent-init", "--coarse-history",
        "--coarse-parent-grid")}
    arguments = door.parse_args([item for pair in paths.items() for item in pair] + [
        "--parent-row", "parent", "--forecast-arg=--physics-suite=convection_permitting"])
    config = _config(arguments)
    assert config.forecast_args == ("--physics-suite=convection_permitting",)
    seen = {}
    monkeypatch.setattr(chain, "_run", lambda argv, **_: seen.setdefault("argv", argv) and 0)
    chain.run_fine_forecast(
        config, mesh_row="cull", grid=tmp_path / "g.nc", static=tmp_path / "s.nc", init=tmp_path / "i.nc",
        lbc_dir=tmp_path / "lbc", start=datetime(2025, 5, 19, 12), out=tmp_path / "o", scratch=tmp_path / "x",
        receipt=tmp_path / "r.json", label="slot", rows_file=tmp_path / "rows.json", ledger_dir=tmp_path / "l",
        log=tmp_path / "log")
    assert seen["argv"][-1] == "--physics-suite=convection_permitting"
    assert seen["argv"].index("--dt") < len(seen["argv"]) - 1


def test_missing_land_refuses_before_child_write(bundle):
    with netCDF4.Dataset(bundle['history'], 'a') as ds:
        ds.renameVariable('canwat', 'missing_canwat')
    before = bundle['init'].read_bytes()
    with pytest.raises(DelayedStartRefusal, match='lacks canwat'):
        delayed_start.compose_mid_window_init(
            child_init=bundle['init'], child_grid=bundle['child_grid'],
            parent_grid=bundle['parent_grid'], parent_history=bundle['history'],
            valid_time=datetime(2026, 8, 12, 12))
    assert bundle['init'].read_bytes() == before
