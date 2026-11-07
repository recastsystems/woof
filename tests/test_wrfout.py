# tests/test_wrfout.py
import importlib.util
import os
import gc
from pathlib import Path
import queue
import threading
import time
import weakref
from contextlib import nullcontext, suppress
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import netCDF4
import pytest
from conftest import requires_gpu
from woof.io.wrfout import WrfoutWriter


_HANDOFF_POLL_SECONDS = 0.05


class _DoneEvent:
    @staticmethod
    def synchronize():
        return None


def _manual_async_writer(abort_event, *, ticket_queue=None):
    """CPU-only AsyncDomainWrfoutWriter shell (no CuPy stream allocation)."""
    from woof.io.wrfout import AsyncDomainWrfoutWriter

    writer = object.__new__(AsyncDomainWrfoutWriter)
    writer.nx = writer.ny = writer.nz = 1
    writer.dx = writer.dy = 1.0
    writer.soil_layers = 4
    writer.title = "test"
    writer.global_attrs = {}
    writer._queue = (AsyncDomainWrfoutWriter._new_ticket_queue()
                     if ticket_queue is None else ticket_queue)
    writer.stream = nullcontext()
    writer._condition = threading.Condition()
    writer._pending = 0
    writer._failure = None
    writer._closed = False
    writer._abort_event = abort_event
    writer.paths = []
    writer._thread = threading.Thread(target=writer._worker, daemon=True)
    writer._thread.start()
    return writer


def _cpu_ticket(path, *, fields=None, device_refs=(), pinned_refs=()):
    from woof.io.wrfout import _AsyncFrame

    return _AsyncFrame(
        path=Path(path), time_str="1974-04-03_12:00:00",
        fields=({"T": np.zeros((1, 1, 1), np.float32)}
                if fields is None else fields),
        event=_DoneEvent(), device_refs=device_refs,
        pinned_refs=pinned_refs)


def _queue_cpu_ticket(writer, path, *, fields=None, device_refs=(),
                      pinned_refs=()):
    ticket = _cpu_ticket(
        path, fields=fields, device_refs=device_refs,
        pinned_refs=pinned_refs)
    writer._admit(ticket)
    return ticket


def _await_worker(event, writer, *, reaching):
    """Block until the worker fires ``event``, or until it can no longer.

    The properties these handoff tests assert are orderings and object
    lifetimes.  None of them has a wall-clock component, so none of them
    should be decided by one.  A fixed deadline here decides "was the box
    busy", and it decides it inside a suite that takes ~37 minutes and runs
    on loaded machines -- so it reports a scheduling delay as a failure of
    the writer, in the one neighbourhood (corruption detection) where a
    test nobody believes is worse than no test.

    The wait is therefore bounded by the WORKER's liveness rather than by
    the clock: a slow machine waits longer and still passes, while a worker
    that stopped fails at once and with its own recorded cause instead of a
    timeout.  Only a worker that is alive and permanently wedged can hang,
    which is a real defect and the one outcome worth hanging for.
    """
    while not event.wait(timeout=_HANDOFF_POLL_SECONDS):
        if not writer._thread.is_alive():
            if event.is_set():
                break
            # Prefer the worker's own exception over a bare assertion.
            writer._raise_failure()
            raise AssertionError(
                f"the wrfout worker stopped without {reaching}")
    return True


def _dead_full_async_writer(failure):
    """Minimal writer whose sole worker has stopped with one queued ticket."""
    from woof.io.wrfout import AsyncDomainWrfoutWriter

    writer = object.__new__(AsyncDomainWrfoutWriter)
    writer._queue = queue.Queue(maxsize=1)
    writer._queue.put(object())
    writer._condition = threading.Condition()
    writer._pending = 1
    writer._failure = failure
    writer._failure_traceback = None
    writer._closed = False
    writer._abort_event = threading.Event()
    writer._thread = threading.Thread(target=lambda: None)
    writer._thread.start()
    writer._thread.join(timeout=1.0)
    assert not writer._thread.is_alive()
    return writer

def test_wrfout_roundtrip(tmp_path):
    # Colons are illegal in Windows filenames, so the on-disk name uses dashes.
    p = tmp_path / "wrfout_d01_1974-04-03_18-00-00.nc"
    nz, ny, nx = 4, 3, 5
    with WrfoutWriter(p, nx=nx, ny=ny, nz=nz, dx=100.0, dy=100.0) as w:
        w.write_frame("1974-04-03_18:00:00", {
            "T": np.zeros((nz, ny, nx), np.float32),
            "U": np.ones((nz, ny, nx + 1), np.float32),
            "W": np.zeros((nz + 1, ny, nx), np.float32),
            "MU": np.zeros((ny, nx), np.float32),
        })
        w.write_frame("1974-04-03_18:05:00", {
            "T": np.full((nz, ny, nx), 2.0, np.float32),
            "U": np.ones((nz, ny, nx + 1), np.float32),
            "W": np.zeros((nz + 1, ny, nx), np.float32),
            "MU": np.zeros((ny, nx), np.float32),
        })
    ds = netCDF4.Dataset(p)
    assert ds.dimensions["west_east"].size == nx
    assert ds.dimensions["west_east_stag"].size == nx + 1
    assert ds.variables["T"].dimensions == ("Time", "bottom_top", "south_north", "west_east")
    assert ds.variables["U"].dimensions[-1] == "west_east_stag"
    times = ["".join(c.decode() for c in row) for row in ds.variables["Times"][:]]
    assert times == ["1974-04-03_18:00:00", "1974-04-03_18:05:00"]
    assert float(ds.variables["T"][1].max()) == 2.0
    assert ds.getncattr("DX") == 100.0
    ds.close()


def test_wrfout_publication_fsyncs_the_directory_that_names_the_frame(
        monkeypatch, tmp_path):
    """The frame's DATA was durable; its NAME was not.

    ``close`` open-codes the produce / fsync / validate / rename sequence
    that ``supervisor.atomic_publish_file`` performs, and dropped that
    helper's last step.  POSIX ``rename(2)`` is atomic for concurrent
    readers and is not durable until the containing directory's metadata
    is synced, so a machine that loses power seconds after a frame is
    published can come back with the file's bytes intact and its
    directory entry still naming the hidden temporary -- which the next
    run's ``quarantine_orphan_wrfouts`` sweeps into ``.quarantine`` on the
    ``.wrfout*.tmp*`` glob.  The frame-ready marker beside it IS published
    through ``supervisor.atomic_write_json``, which fsyncs its own
    directory, so the ordering can invert and the documented invariant "a
    marker that exists names a frame that is complete and readable"
    becomes false.

    Driven through a writer SHELL rather than a real tape: the property
    under test is the publication sequence, and every other step of it is
    covered by the round trips above.

    RED before the fix: ``wrfout._fsync_directory`` does not exist, so
    there is nothing to record.
    """
    from woof.io import wrfout as wrfout_module

    class _Ds:
        variables: dict = {}

        def setncattr(self, name, value):
            pass

        def close(self):
            pass

    writer = object.__new__(WrfoutWriter)
    writer._closed = False
    writer._n = 1
    writer._times = ["1974-04-03_18:00:00"]
    writer._temp_path = tmp_path / ".wrfout_d01.tmp.1.0"
    writer._final_path = tmp_path / "wrfout_d01_1974-04-03_18-00-00.nc"
    writer._temp_path.write_bytes(b"a published frame")
    writer.ds = _Ds()

    synced = []
    real_fsync_directory = wrfout_module._fsync_directory
    monkeypatch.setattr(wrfout_module, "validate_wrfout_file",
                        lambda *args, **kwargs: None)
    monkeypatch.setattr(
        wrfout_module, "_fsync_directory",
        lambda directory: (synced.append(Path(directory)),
                           real_fsync_directory(directory))[1])

    writer.close()

    assert writer._final_path.read_bytes() == b"a published frame"
    assert not writer._temp_path.exists()
    assert synced == [writer._final_path.parent]


def test_wrfout_moist_terrain_fields_and_attrs(tmp_path):
    """Phase 2 Task 12: QVAPOR/QCLOUD/QRAIN, HGT, and the terrain-consistent
    (per-column 3-D) PHB base roundtrip, and every variable carries the WRF
    attribute set (stagger / MemoryOrder / description / units /
    FieldType)."""
    p = tmp_path / "wrfout_moist.nc"
    nz, ny, nx = 4, 3, 5
    rng = np.random.default_rng(12)
    # Terrain-consistent base geopotential: per-column, monotone in k.
    phb = np.cumsum(rng.uniform(1.0, 2.0, (nz + 1, ny, nx)),
                    axis=0).astype(np.float32)
    with WrfoutWriter(p, nx=nx, ny=ny, nz=nz, dx=100.0, dy=100.0) as w:
        w.write_frame("0001-01-01_00:00:00", {
            "U": np.ones((nz, ny, nx + 1), np.float32),
            "V": np.ones((nz, ny + 1, nx), np.float32),
            "W": np.zeros((nz + 1, ny, nx), np.float32),
            "PHB": phb,
            "HGT": np.full((ny, nx), 250.0, np.float32),
            "QVAPOR": np.full((nz, ny, nx), 0.014, np.float32),
            "QCLOUD": np.zeros((nz, ny, nx), np.float32),
            "QRAIN": np.zeros((nz, ny, nx), np.float32),
        })
    ds = netCDF4.Dataset(p)
    try:
        v = ds.variables
        # Dimensions/staggering per WRF conventions.
        assert v["QVAPOR"].dimensions == ("Time", "bottom_top",
                                          "south_north", "west_east")
        assert v["QCLOUD"].dimensions == v["QVAPOR"].dimensions
        assert v["QRAIN"].dimensions == v["QVAPOR"].dimensions
        assert v["HGT"].dimensions == ("Time", "south_north", "west_east")
        assert v["PHB"].dimensions == ("Time", "bottom_top_stag",
                                       "south_north", "west_east")
        # stagger attribute: "X"/"Y"/"Z" on staggered fields, "" otherwise.
        assert v["U"].stagger == "X"
        assert v["V"].stagger == "Y"
        assert v["W"].stagger == "Z"
        assert v["PHB"].stagger == "Z"
        assert v["QVAPOR"].stagger == ""
        assert v["HGT"].stagger == ""
        # MemoryOrder / FieldType / units / description (WRF Registry).
        assert v["QVAPOR"].MemoryOrder == "XYZ"
        assert v["HGT"].MemoryOrder == "XY "
        assert int(v["QVAPOR"].FieldType) == 104
        assert v["QVAPOR"].units == "kg kg-1"
        assert v["HGT"].units == "m"
        assert v["PHB"].units == "m2 s-2"
        for name in ("QVAPOR", "QCLOUD", "QRAIN", "HGT", "PHB"):
            assert v[name].description != ""
        # Terrain-consistent PHB base: per-column values roundtrip exactly.
        np.testing.assert_array_equal(np.asarray(v["PHB"][0]), phb)
        assert float(np.asarray(v["QVAPOR"][0]).max()) == np.float32(0.014)
    finally:
        ds.close()


def test_wrfout_roundtrips_opted_in_physics_surface_fields(tmp_path):
    p = tmp_path / "wrfout_physics.nc"
    nz, ny, nx = 2, 3, 4
    fields = {
        "RAINC": np.arange(ny * nx, dtype=np.float32).reshape(ny, nx) / 10,
        "SWDOWN": np.full((ny, nx), 625.5, dtype=np.float32),
        "GLW": np.full((ny, nx), 312.25, dtype=np.float32),
        "OLR": np.full((ny, nx), 241.75, dtype=np.float32),
    }
    with WrfoutWriter(p, nx=nx, ny=ny, nz=nz, dx=12000.0, dy=12000.0) as w:
        w.write_frame("1974-04-03_12:00:00", fields)

    with netCDF4.Dataset(p) as ds:
        expected_metadata = {
            "RAINC": ("mm", "ACCUMULATED TOTAL CUMULUS PRECIPITATION"),
            "SWDOWN": ("W m-2", "DOWNWARD SHORT WAVE FLUX AT GROUND SURFACE"),
            "GLW": ("W m-2", "DOWNWARD LONG WAVE FLUX AT GROUND SURFACE"),
            # Registry.EM_COMMON:1839, the row a wrf-python/wrf-rust OLR
            # recipe reads.  Same 2-D mass grid as its two neighbours.
            "OLR": ("W m-2", "TOA OUTGOING LONG WAVE"),
        }
        for name, expected in fields.items():
            assert name in ds.variables
            variable = ds.variables[name]
            assert variable.dtype == np.dtype(np.float32)
            assert (variable.units, variable.description) == expected_metadata[name]
            np.testing.assert_array_equal(np.asarray(variable[0]), expected)
        olr = ds.variables["OLR"]
        assert olr.dimensions == ("Time", "south_north", "west_east")
        assert olr.stagger == ""
        assert olr.MemoryOrder == "XY "
        assert int(olr.FieldType) == 104


def test_live_state_history_fields_cover_mp10_frozen_precip_and_noah():
    """Every live WRF-history array is handed to both frame builders.

    Use NumPy sentinels so this inventory regression remains CPU-only; the
    host and device frame paths consume the same pure mapping helper.
    """
    from woof.io.wrfout import _live_state_history_fields

    nz, ny, nx = 3, 2, 4
    atmospheric = {
        name: np.full((nz, ny, nx), value, dtype=np.float32)
        for value, name in enumerate(
            ("qi", "qs", "qg", "nc", "nr", "ni", "ns", "ng"), 1)
    }
    surface = {
        name: np.full((ny, nx), value, dtype=np.float32)
        for value, name in enumerate(("snow", "snowh", "snowc"), 21)
    }
    soil = {
        name: np.full((4, ny, nx), value, dtype=np.float32)
        for value, name in enumerate(("tslb", "smois", "sh2o"), 31)
    }
    microphysics = SimpleNamespace(
        rainnc=np.full((ny, nx), 41, dtype=np.float32),
        snownc=np.full((ny, nx), 42, dtype=np.float32),
        graupelnc=np.full((ny, nx), 43, dtype=np.float32))
    state = SimpleNamespace(
        **atmospheric,
        physics=SimpleNamespace(
            fields={**surface, **soil}, microphysics=microphysics,
            noah_params=object()))

    history = _live_state_history_fields(state)
    expected = {
        "QICE": state.qi, "QSNOW": state.qs, "QGRAUP": state.qg,
        "QNCLOUD": state.nc, "QNRAIN": state.nr, "QNICE": state.ni,
        "QNSNOW": state.ns, "QNGRAUPEL": state.ng,
        "RAINNC": microphysics.rainnc, "SNOWNC": microphysics.snownc,
        "GRAUPELNC": microphysics.graupelnc,
        "SNOW": surface["snow"], "SNOWH": surface["snowh"],
        "SNOWC": surface["snowc"], "TSLB": soil["tslb"],
        "SMOIS": soil["smois"], "SH2O": soil["sh2o"],
    }
    assert set(history) == set(expected)
    for name, sentinel in expected.items():
        assert history[name] is sentinel, name


def test_wrfout_mp10_precip_snow_and_noah_schema(tmp_path):
    """The added history inventory uses WRF names, dimensions, and values."""
    p = tmp_path / "wrfout_history_schema"
    nz, ny, nx = 3, 2, 4
    fields = {
        name: np.full((nz, ny, nx), value, dtype=np.float32)
        for value, name in enumerate((
            "QICE", "QSNOW", "QGRAUP", "QNCLOUD", "QNRAIN", "QNICE",
            "QNSNOW", "QNGRAUPEL"), 1)
    }
    fields.update({
        name: np.full((ny, nx), value, dtype=np.float32)
        for value, name in enumerate((
            "RAINNC", "SNOWNC", "GRAUPELNC", "SNOW", "SNOWH", "SNOWC"),
            21)
    })
    fields.update({
        name: np.full((4, ny, nx), value, dtype=np.float32)
        for value, name in enumerate(("TSLB", "SMOIS", "SH2O"), 31)
    })
    with WrfoutWriter(
            p, nx=nx, ny=ny, nz=nz, dx=12000.0, dy=12000.0,
            field_schema=fields) as writer:
        assert set(fields) <= set(writer.ds.variables)
        writer.write_frame("1974-04-03_12:00:00", fields)

    with netCDF4.Dataset(p) as ds:
        atmosphere_dims = (
            "Time", "bottom_top", "south_north", "west_east")
        surface_dims = ("Time", "south_north", "west_east")
        soil_dims = (
            "Time", "soil_layers_stag", "south_north", "west_east")
        assert ds.dimensions["soil_layers_stag"].size == 4
        for name in ("QICE", "QSNOW", "QGRAUP", "QNCLOUD", "QNRAIN",
                     "QNICE", "QNSNOW", "QNGRAUPEL"):
            assert ds[name].dimensions == atmosphere_dims
        for name in ("RAINNC", "SNOWNC", "GRAUPELNC", "SNOW", "SNOWH",
                     "SNOWC"):
            assert ds[name].dimensions == surface_dims
        for name in ("TSLB", "SMOIS", "SH2O"):
            assert ds[name].dimensions == soil_dims
            assert ds[name].stagger == "Z"
        for name, expected in fields.items():
            np.testing.assert_array_equal(np.asarray(ds[name][0]), expected)
            assert ds[name].description != ""


@pytest.mark.parametrize(
    "grid_id,parent_id,i_start,j_start,ratio,dt,expected_step",
    [
        (1, 0, 1, 1, 1, 60.0, 75),
        (2, 1, 63, 51, 4, 15.0, 300),
        (3, 2, 167, 117, 3, 5.0, 900),
        (4, 3, 151, 151, 3, 1.6666666, 2700),
    ])
def test_wrfout_real74_time_topology_vertical_and_coordinate_metadata(
        tmp_path, grid_id, parent_id, i_start, j_start, ratio, dt,
        expected_step):
    """The generic output identity matches WRF's four-domain metadata."""
    from datetime import timedelta
    from woof.io.wrfout import wrf_global_attrs

    start = datetime(1974, 4, 3, 12)
    valid = start + timedelta(minutes=75)
    grid = SimpleNamespace(
        truelat1=30.0, truelat2=60.0, stand_lon=-98.0,
        ref_lat=35.0, ref_lon=-97.0, cen_lat=35.0, cen_lon=-97.0,
        moad_cen_lat=35.0)
    attrs = wrf_global_attrs(
        grid, start, grid_id=grid_id, parent_id=parent_id,
        i_parent_start=i_start, j_parent_start=j_start,
        parent_grid_ratio=ratio, dt=dt, hybrid_opt=2, etac=0.2)
    nz, ny, nx = 3, 2, 4
    fields = {
        "T": np.zeros((nz, ny, nx), dtype=np.float32),
        "U": np.zeros((nz, ny, nx + 1), dtype=np.float32),
        "V": np.zeros((nz, ny + 1, nx), dtype=np.float32),
        "W": np.zeros((nz + 1, ny, nx), dtype=np.float32),
        "XLAT": np.zeros((ny, nx), dtype=np.float32),
        "XLONG": np.zeros((ny, nx), dtype=np.float32),
        "XLAT_U": np.zeros((ny, nx + 1), dtype=np.float32),
        "XLONG_U": np.zeros((ny, nx + 1), dtype=np.float32),
        "XLAT_V": np.zeros((ny + 1, nx), dtype=np.float32),
        "XLONG_V": np.zeros((ny + 1, nx), dtype=np.float32),
        "P_TOP": np.asarray(10000.0, dtype=np.float32),
        "ZNU": np.asarray([0.8, 0.5, 0.2], dtype=np.float32),
        "ZNW": np.asarray([1.0, 0.65, 0.35, 0.0], dtype=np.float32),
    }
    p = tmp_path / f"wrfout_d{grid_id:02d}"
    with WrfoutWriter(
            p, nx=nx, ny=ny, nz=nz, dx=12000.0 / ratio,
            dy=12000.0 / ratio, global_attrs=attrs) as writer:
        writer.write_frame(valid.strftime("%Y-%m-%d_%H:%M:%S"), fields)

    with netCDF4.Dataset(p) as ds:
        assert tuple(int(getattr(ds, name)) for name in (
            "GRID_ID", "PARENT_ID", "I_PARENT_START", "J_PARENT_START",
            "PARENT_GRID_RATIO", "HYBRID_OPT")) == (
                grid_id, parent_id, i_start, j_start, ratio, 2)
        assert float(ds.DT) == np.float32(dt)
        assert float(ds.ETAC) == np.float32(0.2)
        assert ds["XTIME"].dtype == np.dtype(np.float32)
        assert ds["ITIMESTEP"].dtype == np.dtype(np.int32)
        assert float(ds["XTIME"][0]) == 75.0
        assert int(ds["ITIMESTEP"][0]) == expected_step
        assert ds["XTIME"].units == "minutes since 1974-04-03 12:00:00"
        assert ds["P_TOP"].dimensions == ("Time",)
        assert ds["ZNU"].dimensions == ("Time", "bottom_top")
        assert ds["ZNW"].dimensions == ("Time", "bottom_top_stag")
        assert ds["ZNW"].stagger == "Z"
        assert ds["T"].coordinates == "XLONG XLAT XTIME"
        assert ds["U"].coordinates == "XLONG_U XLAT_U XTIME"
        assert ds["V"].coordinates == "XLONG_V XLAT_V XTIME"
        assert ds["W"].coordinates == "XLONG XLAT XTIME"
        assert ds["XLAT"].coordinates == "XLONG XLAT"
        assert ds["XLONG_U"].coordinates == "XLONG_U XLAT_U"
        assert ds["XLAT_V"].coordinates == "XLONG_V XLAT_V"


def _v461_reference_globals():
    """The pinned v4.6.1 WRF product's own global attributes.

    ``woof/wrf_direct_v461_contract.json`` is this repository's extracted
    contract of a REAL WRF v4.6.1 file, so it is an oracle rather than a
    transcription: an attribute this exporter is supposed to write the way
    WRF writes it can be compared against the value WRF actually wrote.
    """
    import json
    from pathlib import Path as _Path

    path = (_Path(__file__).resolve().parents[1] / "woof"
            / "wrf_direct_v461_contract.json")
    return json.loads(path.read_text())["wrfinput"]["global_attributes"]


def test_the_wrf_time_globals_and_land_category_count_match_the_reference():
    """GMT/JULYR/JULDAY/NUM_LAND_CAT: four globals stock WRF always writes.

    The emitted set was 46 attributes against the pinned v4.6.1 reference
    file's 92, and four of the differences are attributes stock WRF writes
    into EVERY history file and that this run already knows.  ARWpost and
    the older NCL post-processing chain place a file in time by
    ``GMT``/``JULYR``/``JULDAY`` rather than by the ``START_DATE`` string,
    and a consumer handed ``MMINLU`` and ``LU_INDEX`` with no
    ``NUM_LAND_CAT`` has to ASSUME a category count to size a table --
    which is the guess the ``ISOILWATER`` entry beside it was added to
    retire.

    Driven off the reference file's own values: for the instant the
    reference file starts at, this writer must produce the numbers the
    reference file carries.

    RED before the fix: ``wrf_global_attrs`` emits none of the four, so the
    first lookup raises ``KeyError``.
    """
    from datetime import datetime as _datetime

    from woof.io.wrfout import wrf_global_attrs

    reference = _v461_reference_globals()
    start = _datetime.strptime(reference["SIMULATION_START_DATE"],
                               "%Y-%m-%d_%H:%M:%S")
    grid = SimpleNamespace(
        truelat1=38.5, truelat2=38.5, stand_lon=-97.5,
        ref_lat=35.5, ref_lon=-98.0, cen_lat=35.5, cen_lon=-98.0,
        moad_cen_lat=35.5)
    attrs = wrf_global_attrs(grid, start)

    assert float(attrs["GMT"]) == reference["GMT"]
    assert int(attrs["JULYR"]) == reference["JULYR"]
    assert int(attrs["JULDAY"]) == reference["JULDAY"]
    # NC_FLOAT and NC_INT, like WRF's own -- not the NC_DOUBLE/NC_INT64 a
    # bare Python float or int would become.
    assert attrs["GMT"].dtype == np.dtype(np.float32)
    assert attrs["JULYR"].dtype == np.dtype(np.int32)
    assert attrs["JULDAY"].dtype == np.dtype(np.int32)
    # The land-use identity group states its own category count, and the
    # count belongs to the table the other five members name.
    for name in ("NUM_LAND_CAT", "MMINLU", "ISWATER", "ISLAKE", "ISICE",
                 "ISURBAN", "ISOILWATER"):
        assert attrs[name] == reference[name], name

    # A time of day, so GMT is not accidentally right at midnight only.
    noon_thirty = wrf_global_attrs(
        grid, _datetime(2026, 7, 18, 12, 30, 36))
    assert float(noon_thirty["GMT"]) == np.float32(12.51)
    assert int(noon_thirty["JULDAY"]) == 199


def test_the_patch_extent_globals_match_the_reference_file():
    """The twelve ``*_PATCH_*`` globals WRF's own I/O layer reads.

    ``WEST-EAST_GRID_DIMENSION`` says how big the DOMAIN is; the patch
    group says which slab of it THIS FILE holds, and it is what WRF's
    netCDF I/O layer and the ndown-class tools read for extent.  woof
    writes one undecomposed patch per domain, so start is 1 on every axis
    and the two ends bracket the mass/staggered counts -- which is exactly
    what the reference file carries, and the relationship is checked
    against ITS dimensions rather than restated here.

    RED before the fix: ``woof.io.wrfout`` has no
    ``_wrf_patch_extent_attrs``, so the import raises ``ImportError``.
    """
    from woof.io.wrfout import _wrf_patch_extent_attrs

    reference = _v461_reference_globals()
    # The reference's own domain size, read the way the file states it.
    nx = int(reference["WEST-EAST_GRID_DIMENSION"]) - 1
    ny = int(reference["SOUTH-NORTH_GRID_DIMENSION"]) - 1
    nz = int(reference["BOTTOM-TOP_GRID_DIMENSION"]) - 1

    emitted = _wrf_patch_extent_attrs(nx, ny, nz)
    expected = {name: reference[name] for name in reference
                if "_PATCH_" in name}
    assert len(expected) == 12, (
        "the reference file lost its patch group; this test would then "
        "compare nothing")
    assert emitted == expected

    # A different domain moves the ends and never the starts.
    other = _wrf_patch_extent_attrs(7, 5, 3)
    assert other["WEST-EAST_PATCH_END_UNSTAG"] == 7
    assert other["WEST-EAST_PATCH_END_STAG"] == 8
    assert other["SOUTH-NORTH_PATCH_END_UNSTAG"] == 5
    assert other["SOUTH-NORTH_PATCH_END_STAG"] == 6
    assert other["BOTTOM-TOP_PATCH_END_UNSTAG"] == 3
    assert other["BOTTOM-TOP_PATCH_END_STAG"] == 4
    assert set(other) == set(emitted)
    assert all(value == 1 for name, value in other.items()
               if name.endswith(("_PATCH_START_UNSTAG", "_PATCH_START_STAG")))


@pytest.mark.gpu
@requires_gpu
def test_state_frame_from_domain_state():
    """state_frame builds the standard wrfout frame from a DomainState:
    winds / T and dry-theta THM (theta - 300) / PH / MU plus the
    terrain-consistent base fields PHB (a flat base state's 1-D column
    broadcast to 3-D) / MUB /
    HGT, P_TOP/ZNU/ZNW vertical identity, and QVAPOR/QCLOUD/QRAIN exactly
    when the state carries moisture."""
    from woof.config import RunConfig
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.state import init_at_rest
    from woof.io.wrfout import state_frame

    vc = make_vertical_coord(4)

    def sounding(z):
        return np.full_like(np.asarray(z, dtype=np.float64), 300.0)

    cfg = RunConfig(nx=6, ny=5, nz=4, dx=100.0, dy=100.0, ztop=4000.0,
                    dt=0.5, run_seconds=1.0, moist=True)
    b = make_base_state(vc, sounding, p_surf=cfg.p_surf, ztop=cfg.ztop)
    s = init_at_rest(cfg, vc, b)
    s.qv[...] = 0.004
    f = state_frame(s)
    assert set(f) == {"T", "THM", "U", "V", "W", "PH", "MU", "PHB", "MUB",
                      "HGT", "QVAPOR", "QCLOUD", "QRAIN", "P_TOP", "ZNU", "ZNW"}
    # USE_THETA_M=0 publishes the actual dry-theta prognostic as THM.
    np.testing.assert_array_equal(f["THM"], f["T"])
    assert all(isinstance(a, np.ndarray) and a.dtype == np.float32
               for a in f.values())
    assert f["PHB"].shape == (cfg.nz + 1, cfg.ny, cfg.nx)
    np.testing.assert_array_equal(f["PHB"][:, 2, 3],
                                  np.asarray(b.phb, np.float32))
    assert f["HGT"].shape == (cfg.ny, cfg.nx) and not f["HGT"].any()
    np.testing.assert_array_equal(f["MUB"],
                                  np.full((cfg.ny, cfg.nx),
                                          np.float32(b.mub)))
    # Diagnostic pressure requires a diagnosed state (every production
    # writer path runs the EOS before writing; a fresh init has p == 0).
    from woof.core.diagnostics import update_diagnostics
    update_diagnostics(s)
    diagnostic = state_frame(s, include_diagnostic_pressure=True)
    assert set(diagnostic) == set(f) | {"P", "PB", "PSFC"}
    assert diagnostic["P"].shape == (cfg.nz, cfg.ny, cfg.nx)
    assert diagnostic["PB"].shape == (cfg.nz, cfg.ny, cfg.nx)
    # No physics driver attached: PSFC is WRF phy_prep's linear-in-z
    # surface extrapolation of the FULL (moist) pressure
    # (module_big_step_utilities_em.F:5566-5578).  Recompute it in
    # float64 from the frame's own output fields.
    z_if = (diagnostic["PH"].astype(np.float64)
            + diagnostic["PHB"]) / 9.81
    z_mid = 0.5 * (z_if[:-1] + z_if[1:])
    w1 = (z_if[0] - z_mid[1]) / (z_mid[0] - z_mid[1])
    p_full = diagnostic["P"].astype(np.float64) + diagnostic["PB"]
    np.testing.assert_allclose(
        diagnostic["PSFC"], w1 * p_full[0] + (1.0 - w1) * p_full[1],
        rtol=5.0e-6)
    # Downward extrapolation from an at-rest hydrostatic column: PSFC
    # exceeds the lowest half-level pressure and sits near p_surf (the
    # painted qv shifts the diagnosed moist pressure ~1% off the dry
    # base, so this is a sanity bound, not a pin).
    assert np.all(diagnostic["PSFC"] > p_full[0])
    np.testing.assert_allclose(diagnostic["PSFC"], cfg.p_surf, rtol=0.03)
    np.testing.assert_array_equal(f["QVAPOR"],
                                  np.full((cfg.nz, cfg.ny, cfg.nx),
                                          np.float32(0.004)))
    # isothermal-theta base at rest: T = theta - 300 = 0
    assert not f["T"].any()

    # dry state: no moisture variables in the frame
    dry = RunConfig(nx=6, ny=5, nz=4, dx=100.0, dy=100.0, ztop=4000.0,
                    dt=0.5, run_seconds=1.0)
    fd = state_frame(init_at_rest(dry, vc, b))
    assert not ({"QVAPOR", "QCLOUD", "QRAIN"} & set(fd))


#: ``state_frame`` imports cupy to normalise whatever the state carries.
#: It is not a device import here -- see the test below -- but it is still
#: an import, and the CPU tier of an install with no GPU extra has no cupy
#: to give it.
_CUPY_INSTALLED = importlib.util.find_spec("cupy") is not None


def _host_prepared_state(nz=4, ny=5, nx=6):
    """A state whose every array is numpy, as CPU preparation leaves it.

    The shape a host-prepared state actually has, not a convenient one:
    ``phb`` and ``pb`` are the 1-D base-state columns a flat base state
    carries, which is what puts the broadcasts on the branch under test.
    """
    heights = np.array([0.0, 500.0, 1200.0, 2100.0, 3200.0][:nz + 1],
                       dtype=np.float32)
    pb = np.array([100000.0, 94000.0, 87000.0, 79000.0][:nz],
                  dtype=np.float32)
    perturbation = np.linspace(-40.0, 40.0, nz * ny * nx,
                               dtype=np.float32).reshape(nz, ny, nx)
    return SimpleNamespace(
        mup=np.full((ny, nx), -25.0, np.float32),
        mub2d=np.full((ny, nx), 90000.0, np.float32),
        ht=np.zeros((ny, nx), np.float32),
        phb=heights * np.float32(9.81),
        php=np.zeros((nz + 1, ny, nx), np.float32),
        u=np.full((nz, ny, nx + 1), 7.0, np.float32),
        v=np.full((nz, ny + 1, nx), -3.0, np.float32),
        w=np.zeros((nz + 1, ny, nx), np.float32),
        total_theta=lambda: np.full((nz, ny, nx), 301.5, np.float32),
        pb=pb,
        p=pb[:, None, None] + perturbation,
        p_top=np.float32(5000.0),
        qv=None,
        physics=None)


@pytest.mark.skipif(not _CUPY_INSTALLED,
                    reason="state_frame imports cupy to normalise arrays")
def test_state_frame_broadcasts_the_base_pressure_on_a_host_state():
    """A state prepared on the CPU writes PB like every other field.

    ``state_frame``'s diagnostic-pressure branch built PB with
    ``cp.broadcast_to(pb3, state.p.shape)``.  That is a device call, and a
    state whose arrays were prepared on the host carries numpy arrays,
    which ``cp.broadcast_to`` refuses with ``TypeError`` -- while
    ``cp.asnumpy``, on every other line of the same branch, takes either
    kind.  So a host-prepared state could write every field of a history
    frame except that one, and the failure was a type error out of the
    writer rather than a wrong number.

    This test opens no device and is deliberately not a ``gpu`` test: every
    array it hands in is numpy, ``cp.asnumpy`` passes numpy straight
    through, and the branch does no device arithmetic.  That is the whole
    point -- the defect lives on the host route, so the guard has to run
    where that route runs.

    Both halves of the fix are pinned.  ``np.broadcast_to`` on the host
    result is what accepts the host state; ``np.ascontiguousarray`` around
    it is what hands the writer a dense, writeable array instead of a
    read-only stride-zero view, which is what the device call used to
    return and what the netCDF writer is given.
    """
    from woof.io.wrfout import state_frame

    state = _host_prepared_state()
    nz, ny, nx = state.p.shape

    frame = state_frame(state, include_diagnostic_pressure=True)

    assert isinstance(frame["PB"], np.ndarray)
    assert frame["PB"].shape == (nz, ny, nx)
    assert frame["PB"].dtype == np.float32
    # Every column carries the base-state column, which is what the
    # broadcast is for.
    for j in range(ny):
        for i in range(nx):
            np.testing.assert_array_equal(frame["PB"][:, j, i], state.pb)
    # Dense and writeable: a bare broadcast view is neither, and the frame
    # is handed to a writer that expects a real array.
    assert frame["PB"].flags["C_CONTIGUOUS"] and frame["PB"].flags["WRITEABLE"]
    assert frame["PB"].strides[-1] == frame["PB"].dtype.itemsize
    # P is the perturbation the same branch computes, so the two halves of
    # the diagnosed pressure still add back up to the state's own field.
    np.testing.assert_array_equal(frame["P"] + frame["PB"], state.p)
    # And the rest of the frame is there, on the same host arrays: the
    # branch used to abort before any of it was reached.
    assert {"T", "U", "V", "W", "PH", "MU", "PHB", "MUB", "HGT", "P_TOP",
            "PSFC"} <= set(frame)
    assert frame["PHB"].shape == (nz + 1, ny, nx)
    assert np.isfinite(frame["PSFC"]).all()
    assert (frame["PSFC"] > state.p[0]).all()


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("frame_name", ["state_frame", "_device_state_frame"])
def test_psfc_source_follows_the_surface_switch_not_the_driver(frame_name):
    """PSFC comes from the driver only when the surface refreshes it.

    A physics driver allocates ``psfc`` at 100000 Pa and rewrites it from
    ``p_interface[0]`` only inside the surface/PBL cadence block, which a
    microphysics-only composition never enters.  Both writer frames used
    to select the driver's array by asking whether a driver existed, so
    an mp-only ideal case at ``p_surf = 85000`` published the untouched
    seed: 100000.00 Pa against a true 84998.16 Pa, a 150 hPa fabrication
    on the default forecast history path.

    Both directions are pinned at both sites: surface off takes WRF's
    ``phy_prep`` extrapolation, surface on takes the driver's field
    (proved with a sentinel no computation could produce).
    """
    import cupy as cp
    from woof.config import RunConfig
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.state import init_at_rest
    from woof.core.diagnostics import update_diagnostics
    from woof.core.physics import initialize_physics
    from woof.io import wrfout as wrfout_module

    frame = getattr(wrfout_module, frame_name)

    def _psfc(cfg):
        vc = make_vertical_coord(cfg.nz)
        b = make_base_state(
            vc, lambda z: np.full_like(np.asarray(z, dtype=np.float64), 300.0),
            p_surf=cfg.p_surf, ztop=cfg.ztop)
        s = init_at_rest(cfg, vc, b)
        update_diagnostics(s)
        driver = initialize_physics(s, cfg, glw=0.0)
        out = frame(s, include_diagnostic_pressure=True)
        return driver, np.asarray(cp.asnumpy(out["PSFC"]), dtype=np.float64)

    common = dict(nx=6, ny=5, nz=90, dx=1000.0, dy=1000.0, ztop=20000.0,
                  dt=1.0, run_seconds=1.0, moist=True, p_surf=85000.0)

    # Direction 1: microphysics only.  The seed is never refreshed, so the
    # frame must NOT publish it.
    mp_only, psfc = _psfc(RunConfig(**common, mp_physics=6))
    assert mp_only.surface_enabled is False
    assert float(cp.asnumpy(mp_only.fields["psfc"]).ravel()[0]) == 100000.0
    np.testing.assert_allclose(psfc, 84998.16, atol=0.01)
    assert abs(psfc - 100000.0).min() > 14000.0    # the fabrication is gone

    # Direction 2: a surface-bearing composition.  The driver's array is
    # the authority, sentinel-proved.
    surf_cfg = RunConfig(**common, mp_physics=6, sf_sfclay_physics=1,
                         sf_surface_physics=2, bl_pbl_physics=1)
    vc = make_vertical_coord(surf_cfg.nz)
    b = make_base_state(
        vc, lambda z: np.full_like(np.asarray(z, dtype=np.float64), 300.0),
        p_surf=surf_cfg.p_surf, ztop=surf_cfg.ztop)
    s = init_at_rest(surf_cfg, vc, b)
    update_diagnostics(s)
    driver = initialize_physics(s, surf_cfg, glw=0.0)
    assert driver.surface_enabled is True
    driver.fields["psfc"][...] = np.float32(12345.0)
    out = frame(s, include_diagnostic_pressure=True)
    np.testing.assert_array_equal(
        np.asarray(cp.asnumpy(out["PSFC"])),
        np.full((surf_cfg.ny, surf_cfg.nx), np.float32(12345.0)))


def test_wrf_time_str():
    """WRF Times formatting: seconds from 0001-01-01_00:00:00, calendar
    rollover included (shared helper single-sources what straka/igw
    duplicated).

    The month and year roll too, which they did not before v1.1.3: the
    helper incremented only the day field, so this test stopping at day two
    was the whole reason ``0001-01-32`` and ``0001-01-366`` could ship.  The
    cases below cross a month end, a leap-less February, and a year end.
    """
    from woof.io.wrfout import wrf_time_str
    assert wrf_time_str(0.0) == "0001-01-01_00:00:00"
    assert wrf_time_str(305.0) == "0001-01-01_00:05:05"
    assert wrf_time_str(7200.0) == "0001-01-01_02:00:00"
    assert wrf_time_str(86400.0 + 3661.0) == "0001-01-02_01:01:01"
    assert wrf_time_str(30 * 86400.0) == "0001-01-31_00:00:00"
    assert wrf_time_str(31 * 86400.0) == "0001-02-01_00:00:00"
    assert wrf_time_str(58 * 86400.0) == "0001-02-28_00:00:00"
    assert wrf_time_str(59 * 86400.0) == "0001-03-01_00:00:00"
    assert wrf_time_str(365 * 86400.0) == "0002-01-01_00:00:00"
    # And the year is FOUR digits wide, always.  strftime("%Y") is unpadded
    # on glibc and MSVCRT, so the idealized epoch used to emit a 16-character
    # record that write_frame null-padded into the 19-wide Times array
    # WrfoutWriter declares -- shifting every field of WRF's fixed
    # YYYY-MM-DD_HH:MM:SS layout.  Width is the property, so assert width.
    for seconds in (0.0, 305.0, 365 * 86400.0, 999 * 365 * 86400.0):
        assert len(wrf_time_str(seconds)) == 19, seconds
    with pytest.raises(ValueError, match="non-negative"):
        wrf_time_str(-1.0)
    with pytest.raises(ValueError, match="overflows"):
        wrf_time_str(1.0e15)


def test_async_writer_netCDF_sessions_are_process_serial(monkeypatch,
                                                         tmp_path):
    import woof.io.wrfout as wrfout

    active = 0
    peak = 0
    guard = threading.Lock()

    class SlowWriter:
        def __init__(self, path, **_kwargs):
            self.path = path

        def __enter__(self):
            return self

        def write_frame(self, _time_str, _fields):
            nonlocal active, peak
            with guard:
                active += 1
                peak = max(peak, active)
            time.sleep(0.02)
            with guard:
                active -= 1

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(wrfout, "WrfoutWriter", SlowWriter)
    abort = threading.Event()
    first = _manual_async_writer(abort)
    second = _manual_async_writer(abort)
    _queue_cpu_ticket(first, tmp_path / "d01")
    _queue_cpu_ticket(second, tmp_path / "d02")
    first.close()
    second.close()
    assert peak == 1
    assert not first._thread.is_alive()
    assert not second._thread.is_alive()


def test_async_worker_releases_device_refs_after_d2h_before_write(
        monkeypatch, tmp_path):
    import woof.io.wrfout as wrfout

    write_started = threading.Event()
    finish_write = threading.Event()

    class BlockingWriter:
        def __init__(self, _path, **_kwargs):
            pass

        def __enter__(self):
            return self

        def write_frame(self, _time_str, _fields):
            write_started.set()
            # No deadline, and no assertion, on purpose.  This wait spans the
            # main thread's gc.collect(), whose cost is a function of the heap
            # the rest of the suite has already built -- not of this writer.
            # A bound here used to fail INSIDE the worker, where _worker's
            # `except BaseException` files it as self._failure and sets the
            # experiment-wide abort event, so a slow collection was reported
            # as `RuntimeError: per-domain wrfout writer failed`: a false
            # corruption alarm raised by the corruption detector's own test.
            # The release below is set unconditionally in this test's finally,
            # so an unbounded wait cannot outlive the test.
            finish_write.wait()

        def __exit__(self, *_args):
            return False

    class DeviceArray:
        pass

    monkeypatch.setattr(wrfout, "WrfoutWriter", BlockingWriter)
    writer = _manual_async_writer(threading.Event())
    try:
        device = DeviceArray()
        pinned_owner = np.zeros((1, 1, 1), dtype=np.float32)
        field = pinned_owner.view()
        assert field.base is pinned_owner
        device_ref = weakref.ref(device)
        pinned_ref = weakref.ref(pinned_owner)
        # `field` is a view of `pinned_owner`, so numpy's own base chain keeps
        # the backing alive independently of ticket.pinned_refs -- and in the
        # production path too, where host_fields come from
        # np.frombuffer(memory, ...).  Watching only `pinned_owner` therefore
        # cannot tell "pinned_refs retained it" from "the view's .base did":
        # dropping ticket.pinned_refs early leaves this test green.  The
        # unviewed buffer below is reachable ONLY through ticket.pinned_refs,
        # so it holds the explicit mechanism to its stated contract.
        unviewed = np.zeros((1, 1, 1), dtype=np.float32)
        unviewed_ref = weakref.ref(unviewed)
        ticket = _queue_cpu_ticket(
            writer, tmp_path / "d01", fields={"T": field},
            device_refs=(device,), pinned_refs=(pinned_owner, unviewed))
        ticket_ref = weakref.ref(ticket)
        del device, pinned_owner, field, ticket, unviewed

        _await_worker(write_started, writer, reaching="entering write_frame")
        gc.collect()
        assert device_ref() is None
        assert pinned_ref() is not None
        assert unviewed_ref() is not None
        assert ticket_ref() is not None
        finish_write.set()
        writer.drain()
        gc.collect()
        assert pinned_ref() is None
        assert unviewed_ref() is None
        assert ticket_ref() is None
    finally:
        finish_write.set()
        writer.close()


def test_queued_device_ownership_survives_unfinished_download(tmp_path):
    from woof.io.wrfout import AsyncDomainWrfoutWriter

    event = SimpleNamespace(done=False)
    device = SimpleNamespace()
    pinned = np.zeros(1, np.float32)
    ticket = _cpu_ticket(tmp_path / "queued", device_refs=(device,), pinned_refs=(pinned,))
    ticket.event = event
    tickets = AsyncDomainWrfoutWriter._new_ticket_queue()
    tickets.put(ticket)
    tickets.release_completed_device_refs(nullcontext())
    assert ticket.device_refs == (device,)
    event.done = True
    tickets.release_completed_device_refs(nullcontext())
    assert ticket.device_refs == ()
    assert ticket.pinned_refs == (pinned,)
    assert tickets.get_nowait() is ticket


def test_async_write_failure_does_not_retain_field_backing(
        monkeypatch, tmp_path):
    import woof.io.wrfout as wrfout

    class FailingWriter:
        def __init__(self, _path, **_kwargs):
            pass

        def __enter__(self):
            return self

        def write_frame(self, _time_str, fields):
            arr = fields["T"]
            assert arr.shape == (1, 1, 1)
            raise OSError("injected field write failure")

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(wrfout, "WrfoutWriter", FailingWriter)
    writer = _manual_async_writer(threading.Event())
    pinned_owner = np.zeros((1, 1, 1), dtype=np.float32)
    field = pinned_owner.view()
    assert field.base is pinned_owner
    pinned_ref = weakref.ref(pinned_owner)
    ticket = _queue_cpu_ticket(
        writer, tmp_path / "d01", fields={"T": field},
        pinned_refs=(pinned_owner,))
    ticket_ref = weakref.ref(ticket)
    del pinned_owner, field, ticket

    try:
        with pytest.raises(
                RuntimeError, match="per-domain wrfout writer failed") as raised:
            writer.drain()
        cause = raised.value.__cause__
        assert isinstance(cause, OSError)
        assert str(cause) == "injected field write failure"
        assert cause is writer._failure
        gc.collect()
        assert ticket_ref() is None
        assert pinned_ref() is None
        assert cause.__traceback__ is None
        assert "in write_frame" in writer._failure_traceback
        assert "injected field write failure" in writer._failure_traceback
    finally:
        with suppress(RuntimeError):
            writer.close()


def test_async_worker_holds_no_completed_ticket_while_idle(monkeypatch,
                                                           tmp_path):
    import woof.io.wrfout as wrfout

    class NoOpWriter:
        def __init__(self, _path, **_kwargs):
            pass

        def __enter__(self):
            return self

        def write_frame(self, _time_str, _fields):
            pass

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(wrfout, "WrfoutWriter", NoOpWriter)
    writer = _manual_async_writer(threading.Event())
    ticket = _queue_cpu_ticket(writer, tmp_path / "d01")
    ticket_ref = weakref.ref(ticket)
    del ticket

    try:
        writer.drain()
        gc.collect()
        assert ticket_ref() is None
    finally:
        writer.close()


def test_async_admission_raises_recorded_failure_when_worker_dies_with_full_queue(
        tmp_path):
    from woof.io.wrfout import AsyncDomainWrfoutWriter

    attempted = threading.Event()

    class SignallingQueue(queue.Queue):
        def put(self, item, block=True, timeout=None):
            if timeout is not None:
                attempted.set()
            return super().put(item, block=block, timeout=timeout)

    writer = object.__new__(AsyncDomainWrfoutWriter)
    writer._queue = SignallingQueue(maxsize=1)
    # Use the base implementation so filling the queue does not signal the
    # simulated worker-death transition before admission starts.
    queue.Queue.put(writer._queue, object())
    writer._condition = threading.Condition()
    writer._pending = 0
    writer._failure = None
    writer._failure_traceback = None
    writer._closed = False
    writer._abort_event = threading.Event()
    failure = OSError("worker failed while admission was blocked")

    def fail_worker():
        if attempted.wait(timeout=1.0):
            writer._failure = failure

    writer._thread = threading.Thread(target=fail_worker)
    writer._thread.start()
    with pytest.raises(RuntimeError, match="per-domain wrfout writer failed") as raised:
        writer._admit(_cpu_ticket(tmp_path / "blocked"))
    writer._thread.join(timeout=1.0)

    assert raised.value.__cause__ is failure
    assert writer.pending == 0
    assert writer._queue.qsize() == 1


def test_async_interrupted_admission_rolls_back_pending_and_drain_returns(
        tmp_path):
    from woof.io.wrfout import AsyncDomainWrfoutWriter

    class InterruptingQueue:
        @staticmethod
        def put(_item, *, timeout):
            assert timeout > 0.0
            raise KeyboardInterrupt("injected interrupted put")

    class LiveWorker:
        @staticmethod
        def is_alive():
            return True

    writer = object.__new__(AsyncDomainWrfoutWriter)
    writer._queue = InterruptingQueue()
    writer._condition = threading.Condition()
    writer._pending = 0
    writer._failure = None
    writer._failure_traceback = None
    writer._closed = False
    writer._abort_event = threading.Event()
    writer._thread = LiveWorker()

    with pytest.raises(KeyboardInterrupt, match="injected interrupted put"):
        writer._admit(_cpu_ticket(tmp_path / "interrupted"))
    assert writer.pending == 0
    writer.drain()


def test_async_interrupted_admission_after_insert_stays_balanced_and_drains(
        monkeypatch, tmp_path):
    import woof.io.wrfout as wrfout
    from woof.io.wrfout import AsyncDomainWrfoutWriter

    consumed = threading.Event()

    class NoOpWriter:
        def __init__(self, _path, **_kwargs):
            pass

        def __enter__(self):
            return self

        def write_frame(self, _time_str, _fields):
            consumed.set()

        def __exit__(self, *_args):
            return False

    ticket_queue_type = type(AsyncDomainWrfoutWriter._new_ticket_queue())

    class InsertThenInterruptQueue(ticket_queue_type):
        def __init__(self):
            super().__init__(maxsize=1)
            self._interrupt_next_ticket = True

        def put(self, item, block=True, timeout=None):
            super().put(item, block=block, timeout=timeout)
            if item is not None and self._interrupt_next_ticket:
                self._interrupt_next_ticket = False
                raise KeyboardInterrupt("injected interrupt after insertion")

    monkeypatch.setattr(wrfout, "WrfoutWriter", NoOpWriter)
    writer = _manual_async_writer(
        threading.Event(), ticket_queue=InsertThenInterruptQueue())
    drained = threading.Event()
    drainer = None
    pending_after_consume = None
    try:
        ticket = _cpu_ticket(tmp_path / "inserted")
        with pytest.raises(
                KeyboardInterrupt, match="injected interrupt after insertion"):
            writer._admit(ticket)
        assert ticket.admitted
        _await_worker(consumed, writer, reaching="consuming the ticket")
        pending_after_consume = writer.pending

        def drain():
            writer.drain()
            drained.set()

        drainer = threading.Thread(target=drain, daemon=True)
        drainer.start()
        drain_terminated = drained.wait(timeout=1.0)
    finally:
        # Keep the deliberately anti-hang regression self-cleaning while it
        # is red against an implementation that drives _pending negative.
        with writer._condition:
            if writer._pending < 0:
                writer._pending = 0
                writer._condition.notify_all()
        if drainer is not None:
            drainer.join(timeout=1.0)
        writer.close()

    assert pending_after_consume == 0
    assert drain_terminated


@pytest.mark.parametrize("operation", ["drain", "close"])
def test_async_shutdown_terminates_for_dead_worker_with_full_queue(operation):
    failure = OSError("worker died with queued work")
    writer = _dead_full_async_writer(failure)
    finished = threading.Event()
    raised = []

    def shut_down():
        try:
            getattr(writer, operation)()
        except BaseException as exc:
            raised.append(exc)
        finally:
            finished.set()

    caller = threading.Thread(target=shut_down, daemon=True)
    caller.start()
    assert finished.wait(timeout=1.0), f"{operation} blocked on a dead worker"
    caller.join(timeout=1.0)
    assert len(raised) == 1
    assert isinstance(raised[0], RuntimeError)
    assert raised[0].__cause__ is failure


def test_per_domain_writer_builds_static_metadata_once(
        monkeypatch, tmp_path):
    import woof.io.wrfout as wrfout
    import woof.runtime as runtime

    metadata = {"XLAT": np.arange(4, dtype=np.float64).reshape(2, 2)}
    metadata_calls = []
    global_start_calls = []

    def build_metadata(grid, static_fields):
        metadata_calls.append((grid, static_fields))
        return metadata

    class CapturingWriter:
        def __init__(self, **_kwargs):
            self.pending = 0
            self.paths = []
            self.submissions = []

        def submit(self, *args, **kwargs):
            self.submissions.append((args, kwargs))

        def drain(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(runtime, "_metadata_frame", build_metadata)
    def global_attrs(_grid, start_time, *_args, **_kwargs):
        global_start_calls.append(start_time)
        return {}
    monkeypatch.setattr(runtime, "_global_wrf_attrs", global_attrs)
    monkeypatch.setattr(wrfout, "AsyncDomainWrfoutWriter", CapturingWriter)

    cfg = SimpleNamespace(
        grid_id=1,
        start_time=datetime(1974, 4, 3, 12, 5),
        # sf_surface_physics is not optional in a RunConfig stand-in any more:
        # the soil axis is resolved from the SCHEME (config.soil_layer_count),
        # so a fake that declares a layer count without declaring whose it is
        # can no longer be asked for its geometry.
        run=__import__("woof.config", fromlist=["RunConfig"]).RunConfig(
            nx=2, ny=2, nz=1, dx=1.0, dy=1.0, ztop=10000.,
            dt=30., run_seconds=0., sf_surface_physics=0, num_soil_layers=4))
    grid = object()
    static_fields = {"HGT_M": np.zeros((2, 2), dtype=np.float32)}
    case = SimpleNamespace(
        static_fields=static_fields, geog_selection=None,
        initial_result=SimpleNamespace(
            coord=SimpleNamespace(hybrid_opt=2, etac=0.2)))
    node = SimpleNamespace(
        cfg=cfg, grid=grid, state=object(),
        # `spec` is not optional in a clock stand-in any more: the writer
        # stamps the CONFIGURED step on the tape (`configured_dt`), which
        # under an adaptive clock is the only place the step the run was
        # asked for survives -- the live one moves every root step.  This
        # fake carried only `tick_den` and the suite went red on the line
        # the moment that argument landed, unseen because no battery file
        # covered the per-domain writer's construction.
        clock=SimpleNamespace(tick_den=1,
                              spec=SimpleNamespace(dt_fp32=30.0)))
    model = SimpleNamespace(
        _prepared_by_grid_id={1: case},
        walk_parent_first=lambda: (node,))

    writers = wrfout.PerDomainWrfoutWriters(
        model, tmp_path, start_time=datetime(1974, 4, 3, 12),
        title="test")
    writers.submit(node, 0)
    writers.submit(node, 900)

    assert metadata_calls == [(grid, static_fields)]
    assert global_start_calls == [datetime(1974, 4, 3, 12, 5)]
    submissions = writers._writers[1].submissions
    assert submissions[0][1]["extra_fields"] is metadata
    assert submissions[1][1]["extra_fields"] is metadata


def test_failing_writer_close_joins_every_domain_and_aborts_publication(
        monkeypatch, tmp_path):
    import woof.io.wrfout as wrfout

    published = []

    class FailingWriter:
        def __init__(self, path, **_kwargs):
            self.path = Path(path)

        def __enter__(self):
            return self

        def write_frame(self, _time_str, _fields):
            if self.path.name == "d01":
                raise OSError("injected netCDF failure")
            published.append(self.path.name)

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(wrfout, "WrfoutWriter", FailingWriter)
    abort = threading.Event()
    first = _manual_async_writer(abort)
    second = _manual_async_writer(abort)
    _queue_cpu_ticket(first, tmp_path / "d01")
    _await_worker(abort, first, reaching="signalling the publication abort")
    with pytest.raises(RuntimeError, match="wrfout writing was aborted"):
        _queue_cpu_ticket(second, tmp_path / "d02")

    writers = object.__new__(wrfout.PerDomainWrfoutWriters)
    writers._writers = {1: first, 2: second}
    writers._abort_event = abort
    writers.last_durable_wrfout = None
    with pytest.raises(RuntimeError, match="per-domain wrfout writer failed"):
        writers.close()
    assert not first._thread.is_alive()
    assert not second._thread.is_alive()
    assert published == []
    time.sleep(0.02)
    assert published == []


_REFERENCE_WRFOUT = os.path.join(
    os.environ.get("WOOF_TEST_WRF74_BUNDLE",
                   "gpuwm-fixture-unset/wrf74-bundle"),
    "wrfout_reference", "wrfout_d01_1974-04-03_13_00_00")


@pytest.mark.skipif(not __import__("pathlib").Path(_REFERENCE_WRFOUT).is_file(),
                    reason="WRF_1974_MP55 reference bundle not present")
def test_var_meta_matches_reference_wrfout_attributes():
    """Every _VAR_META description/units string matches what the group's
    WRF v4.6.1 actually wrote (the Registry strings, via the reference
    wrfout).  Guards against drift for all variables the reference
    carries; gpuwm-only extensions (e.g. PSIM/PSIH) are exempt."""
    from woof.io.wrfout import _VAR_META

    with netCDF4.Dataset(_REFERENCE_WRFOUT) as ds:
        checked = 0
        for name, (desc, units) in _VAR_META.items():
            if name not in ds.variables:
                continue
            var = ds.variables[name]
            assert var.description == desc, (
                f"{name}: {var.description!r} != {desc!r}")
            assert var.units == units, f"{name}: {var.units!r} != {units!r}"
            checked += 1
    assert checked >= 40


def test_p3s_rime_pair_is_published_and_its_carriers_are_not():
    """mp=50's history inventory, and the two fields WRF keeps out of it.

    WRF gives QIR/QIB the history ``h`` in their IO string
    (``Registry.EM_COMMON:555-558``, ``i0rhusdf``), and without them an
    mp=50 wrfout is a one-moment ice field: rime fraction QIR/QICE and rime
    density QIR/QIB are the two indices P3's whole ice inventory is built
    on, and neither is recoverable from QICE alone.  th_old/qv_old are
    restart-only in WRF (``:1598-1599``, ``rusd`` -- no ``h``), so
    publishing them would be a woof invention, not a transcription.

    Presence-guarded, so no other scheme's inventory moves.
    """
    from woof.io.wrfout import _live_state_history_fields

    class _P3:
        qi = np.zeros((4, 3, 2), np.float32)
        ni = np.zeros((4, 3, 2), np.float32)
        nr = np.zeros((4, 3, 2), np.float32)
        qir = np.full((4, 3, 2), 1.0e-5, np.float32)
        qib = np.full((4, 3, 2), 2.0e-8, np.float32)
        th_old = np.full((4, 3, 2), 300.0, np.float32)
        qv_old = np.full((4, 3, 2), 1.0e-3, np.float32)

    fields = _live_state_history_fields(_P3())
    assert fields["QIR"] is _P3.qir
    assert fields["QIB"] is _P3.qib
    assert fields["QICE"] is _P3.qi
    # P3 has ONE ice category: these are absent, not zero.
    assert not ({"QSNOW", "QGRAUP", "QHAIL"} & set(fields))
    # Restart-only, following WRF's own IO strings.
    assert not ({"TH_OLD", "QV_OLD"} & set(fields))

    class _Morrison:
        qi = np.zeros((4, 3, 2), np.float32)
        nc = np.zeros((4, 3, 2), np.float32)

    assert not ({"QIR", "QIB"} & set(_live_state_history_fields(_Morrison())))


def _incomplete_wrfout(path):
    """A syntactically valid history file with no completion attribute."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("Time", 1)
    return path


def _complete_wrfout(path):
    """The same file, stamped the way a published frame is."""
    from woof.io.wrfout import _COMPLETION_ATTR

    path.parent.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("Time", 1)
        ds.setncattr(_COMPLETION_ATTR, 1)
    return path


def test_quarantine_sweeps_episode_foldered_frames_and_not_its_own(tmp_path):
    """The sweep reaches the tree this module itself writes.

    ``PerDomainWrfoutWriters.submit`` files a lifecycle domain's frames at
    ``<out>/d05/episode-002/wrfout_...``, taking the segment from
    ``render_layout.episode_segment``; the sweep globbed one level deep,
    so every orphan temporary and every incomplete final frame of an
    episode-foldered run stayed exactly where it was, named like a
    published product, for the next reader to pick up.

    Second half: the sweep is idempotent.  ``supervisor.quarantine_file``
    names the moved file ``wrfout_....incomplete-wrfout.<stamp>.<pid>``,
    which still begins with ``wrfout``, so a recursive walk without the
    dot-directory exclusion would re-sweep its own quarantine on every
    run and nest ``.quarantine`` inside ``.quarantine`` without bound.

    RED on the flat sweep: ``directory.glob`` never descends into
    ``d05/``, so the returned tuple is empty and both files stay put.
    """
    from woof.io.wrfout import quarantine_orphan_wrfouts

    out = tmp_path / "out"
    episode = out / "d05" / "episode-002"
    episode.mkdir(parents=True)

    orphan = episode / ".wrfout_d05_1974-04-03_18-00-00.nc.tmp.0"
    orphan.write_bytes(b"an interrupted frame")
    incomplete = _incomplete_wrfout(
        episode / "wrfout_d05_1974-04-03_18-00-00.nc")
    published = _complete_wrfout(
        out / "wrfout_d01_1974-04-03_18-00-00.nc")

    moved = quarantine_orphan_wrfouts(out)

    quarantine = episode / ".quarantine"
    assert {path.parent for path in moved} == {quarantine}
    assert len(moved) == 2
    assert not orphan.exists()
    assert not incomplete.exists()
    swept = sorted(path.name for path in quarantine.iterdir())
    assert any(name.startswith(".wrfout_d05_") and "orphan-wrfout-tmp" in name
               for name in swept), swept
    assert any(name.startswith("wrfout_d05_") and "incomplete-wrfout" in name
               for name in swept), swept
    # The complete frame at the run root is a product, not an orphan.
    assert published.is_file()

    # Idempotent: what the sweep already quarantined is not a candidate.
    assert quarantine_orphan_wrfouts(out) == tuple()
    assert sorted(path.name for path in quarantine.iterdir()) == swept
    assert not (quarantine / ".quarantine").exists()
    assert published.is_file()


def test_sweep_reaches_episode_frames_and_spares_readiness_markers(tmp_path):
    """Discovery names what a frame is; it does not open files to find out.

    Two halves of one walk, and the tree exercises both at once.

    Recursion: an interrupted lifecycle episode's unfinished frame sits
    at ``<out>/d05/episode-002/wrfout_...`` and must be swept.  A
    one-level glob never descends, so it stays under a published name.

    What a frame IS: ``progress_log.write_frame_marker`` publishes
    ``<out>/ready/<frame>.json``, a receipt that deliberately repeats the
    history basename, and its in-flight temporary carries ``.tmp`` as
    well.  Both match the sweep's patterns.  A recursive walk that
    handed them to netCDF4 and read "cannot open" as "incomplete
    product" would quarantine a valid receipt under a false label and
    destroy the one positive signal ``write_frame_marker`` documents a
    consumer may trust (a marker that exists names a frame that is
    complete and readable; absence means "not yet", never "corrupt").
    The same walk fed to a renderer would hand it JSON, which is the
    "NetCDF: Unknown file format" exit 2 that ``go_cli.WRFOUT_GLOB``
    records itself as existing to prevent, so the last assertion reads
    the listing directly and not only the sweep.

    RED on the flat sweep: the episode frame is never reached.
    RED on a recursive sweep with no frame rule: the marker is moved to
    ``ready/.quarantine/...incomplete-wrfout...`` and its temporary with
    it.
    """
    from woof import progress_log
    from woof.io.wrfout import quarantine_orphan_wrfouts

    out = tmp_path / "out"
    episode = out / "d05" / "episode-002"
    episode.mkdir(parents=True)

    stranded = _incomplete_wrfout(
        episode / "wrfout_d05_2026-08-15_06_00_00.nc")
    published = _complete_wrfout(out / "wrfout_d01_2026-08-15_00_00_00.nc")

    marker = progress_log.write_frame_marker(
        out / progress_log.FRAME_MARKER_DIRNAME, domain=1,
        valid_time="2026-08-15_00:00:00", path=published)
    assert marker.parent == out / progress_log.FRAME_MARKER_DIRNAME
    # A marker publication caught mid-flight: tmp + fsync + os.replace,
    # so this name is on disk for the length of a write.
    marker_temp = marker.with_name(f"{marker.name}.tmp.4242.0")
    marker_temp.write_text("{}\n", encoding="utf-8")

    moved = quarantine_orphan_wrfouts(out)

    # One move, and it is the stranded episode frame.  A flat sweep
    # moves nothing; a recursive sweep with no frame rule also carries
    # off the receipt and its temporary.
    assert len(moved) == 1, sorted(path.name for path in moved)
    assert moved[0].name.startswith(f"{stranded.name}.incomplete-wrfout.")
    assert moved[0].parent == episode / ".quarantine"
    assert not stranded.exists()

    # The receipt and its in-flight temporary are not products and are
    # not swept; nothing was filed under the marker directory.
    assert marker.is_file()
    assert marker_temp.is_file()
    assert not (marker.parent / ".quarantine").exists()
    assert published.is_file()

    # The same listing is what a render feed would read.
    from woof.io.wrfout import iter_wrfout_files

    assert iter_wrfout_files(out) == [published]


@pytest.mark.skipif(os.name != "nt", reason="native Windows junction contract")
@pytest.mark.parametrize("selected_root_is_junction", [False, True])
def test_orphan_sweep_does_not_follow_descendant_junctions(tmp_path, selected_root_is_junction):
    import _winapi
    from woof.io.wrfout import iter_wrfout_files, quarantine_orphan_wrfouts

    run = tmp_path / "run"
    episode = run / "d02" / "episode-001"
    episode.mkdir(parents=True)
    external = tmp_path / "external"
    external.mkdir()
    outside = external / ".wrfout_external.tmp.1"
    outside.write_bytes(b"belongs to a different run")
    _winapi.CreateJunction(str(external), str(run / "external-link"))
    inside = episode / ".wrfout_owned.tmp.2"
    inside.write_bytes(b"owned interrupted frame")
    selected = run
    if selected_root_is_junction:
        selected = tmp_path / "selected-run"
        _winapi.CreateJunction(str(run), str(selected))

    moved = quarantine_orphan_wrfouts(selected)

    assert outside.is_file(), "a descendant junction let the sweep move another run's file"
    assert outside.read_bytes() == b"belongs to a different run"
    assert not (external / ".quarantine").exists()
    assert not inside.exists()
    assert len(moved) == 1
    assert moved[0].parent == selected / "d02" / "episode-001" / ".quarantine"
    assert iter_wrfout_files(selected, ".wrfout*.tmp*") == []
