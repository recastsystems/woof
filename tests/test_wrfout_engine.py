"""The wrfout write path runs on the Rust writer by default.

2.5.0's law is that data-path processing is Rust; the product tape is the
data path's most important artifact.  These tests pin the flip itself:

* the DEFAULT engine is ``rust`` -- no flag, no env, a bare
  ``WrfoutWriter`` writes lossless compressed NetCDF-4 through
  ``woof.io.nc_writer_bridge``; explicit compression ``none`` keeps CDF-2;
* the Python/netCDF4 engine remains reachable ONLY as an explicit escape
  (``WOOF_WRFOUT_WRITER=python`` or ``engine="python"``), documented as
  a workaround, and still writes exactly what it always wrote;
* a box where the Rust library is missing gets a refusal that names the
  concrete breakage and both remedies -- never a silent engine downgrade,
  which would change the product tape's container with build state.

The dual-write equivalence of the two engines is proved separately in
``tests/test_wrfout_dual_write.py``.
"""

from __future__ import annotations

import numpy as np
import netCDF4
import pytest

from woof.io import nc_writer_bridge
from woof.io import wrfout as wrfout_module
from woof.io.wrfout import (
    HISTORY_COMPRESSION_ENV, WRFOUT_WRITER_ENV, WrfoutWriter, resolve_wrfout_engine,
)

_RUST_UNAVAILABLE = nc_writer_bridge.unavailable_reason()


def _require_rust_writer():
    if _RUST_UNAVAILABLE:
        pytest.skip(f"the Rust NetCDF writer library is not built: "
                    f"{_RUST_UNAVAILABLE}")


def _frame(nz=3, ny=4, nx=5):
    return {
        "T": np.arange(nz * ny * nx, dtype=np.float32).reshape(nz, ny, nx),
        "U": np.ones((nz, ny, nx + 1), np.float32),
        "W": np.zeros((nz + 1, ny, nx), np.float32),
        "MU": np.full((ny, nx), 3.25, np.float32),
    }


# ---------------------------------------------------------------------------
# Engine resolution: the default is the flip.
# ---------------------------------------------------------------------------

def test_the_default_engine_is_rust(monkeypatch):
    monkeypatch.delenv(WRFOUT_WRITER_ENV, raising=False)
    assert resolve_wrfout_engine() == "rust"
    assert resolve_wrfout_engine(None) == "rust"


def test_the_env_escape_selects_python(monkeypatch):
    monkeypatch.setenv(WRFOUT_WRITER_ENV, "python")
    assert resolve_wrfout_engine() == "python"


def test_the_explicit_argument_wins_over_the_env(monkeypatch):
    monkeypatch.setenv(WRFOUT_WRITER_ENV, "python")
    assert resolve_wrfout_engine("rust") == "rust"


def test_an_unknown_engine_is_refused_by_name(monkeypatch):
    monkeypatch.setenv(WRFOUT_WRITER_ENV, "fortran")
    with pytest.raises(ValueError, match="fortran"):
        resolve_wrfout_engine()


# ---------------------------------------------------------------------------
# Both compression modes use the Rust seam and preserve the frame words.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("compression,container", [
    (None, "NETCDF4_CLASSIC"), ("none", "NETCDF3_64BIT_OFFSET"),
])
def test_rust_writer_preserves_words_in_each_container(
        tmp_path, monkeypatch, compression, container):
    _require_rust_writer()
    monkeypatch.delenv(WRFOUT_WRITER_ENV, raising=False)
    monkeypatch.delenv(HISTORY_COMPRESSION_ENV, raising=False)
    path = tmp_path / "wrfout_d01_rust"
    nz, ny, nx = 3, 4, 5
    with WrfoutWriter(path, nx=nx, ny=ny, nz=nz, dx=100.0, dy=100.0,
                      compression=compression) as w:
        assert w.engine == "rust"
        w.write_frame("1974-04-03_18:00:00", _frame(nz, ny, nx))
        w.write_frame("1974-04-03_18:05:00", _frame(nz, ny, nx))
    with netCDF4.Dataset(path) as ds:
        assert ds.file_format == container
        assert ds.dimensions["Time"].isunlimited()
        assert len(ds.dimensions["Time"]) == 2
        assert int(getattr(ds, "GPUWM_WRITE_COMPLETE")) == 1
        var = ds.variables["T"]
        var.set_auto_maskandscale(False)
        if compression is None:
            assert var.filters()["zlib"]
        assert var.dimensions == (
            "Time", "bottom_top", "south_north", "west_east")
        for frame in range(2):
            np.testing.assert_array_equal(
                np.asarray(var[frame]).view(np.uint32),
                _frame(nz, ny, nx)["T"].view(np.uint32))
        times = ["".join(c.decode() for c in row)
                 for row in ds.variables["Times"][:]]
        assert times == ["1974-04-03_18:00:00", "1974-04-03_18:05:00"]


def test_python_escape_still_writes_netcdf4_classic(tmp_path, monkeypatch):
    monkeypatch.setenv(WRFOUT_WRITER_ENV, "python")
    path = tmp_path / "wrfout_d01_python"
    with WrfoutWriter(path, nx=5, ny=4, nz=3, dx=100.0, dy=100.0) as w:
        assert w.engine == "python"
        w.write_frame("1974-04-03_18:00:00", _frame())
    with netCDF4.Dataset(path) as ds:
        assert ds.file_format == "NETCDF4_CLASSIC"
        assert int(getattr(ds, "GPUWM_WRITE_COMPLETE")) == 1


def test_a_missing_rust_library_refuses_with_both_remedies(
        tmp_path, monkeypatch):
    """No silent downgrade: the refusal names the breakage and the escape.

    The concrete breakage a fallback would cause: the product tape's
    container would silently depend on which box built what, and the
    'default' writer would quietly stop being the one that is tested.
    """
    monkeypatch.delenv(WRFOUT_WRITER_ENV, raising=False)
    # The bridge caches its loaded library for the life of the process
    # (correct: the env override is read at first load), so an unloadable
    # box is simulated at the probe the writer consults rather than by
    # unloading a DLL Windows will not release.
    monkeypatch.setattr(
        nc_writer_bridge, "unavailable_reason",
        lambda: "FileNotFoundError: the Rust NetCDF writer library was "
                "not found (simulated)")
    with pytest.raises(RuntimeError) as refusal:
        WrfoutWriter(tmp_path / "wrfout_refused", nx=2, ny=2, nz=2,
                     dx=1.0, dy=1.0)
    message = str(refusal.value)
    assert "Rust NetCDF writer" in message
    assert WRFOUT_WRITER_ENV in message           # the documented escape
    assert "workaround" in message                # named as one
    # A refusal must not leave a temp file behind for the quarantine
    # sweep to find.
    assert not list(tmp_path.glob(".wrfout*"))


# ---------------------------------------------------------------------------
# The classic header is immutable after the first data write, and the
# refusals say so by name rather than corrupting the tape.
# ---------------------------------------------------------------------------

def test_a_new_variable_after_the_first_frame_is_refused_by_name(
        tmp_path, monkeypatch):
    monkeypatch.delenv(WRFOUT_WRITER_ENV, raising=False)
    _require_rust_writer()
    writer = WrfoutWriter(tmp_path / "wrfout_growing", nx=5, ny=4, nz=3,
                          dx=100.0, dy=100.0)
    try:
        writer.write_frame("1974-04-03_18:00:00", _frame())
        grown = dict(_frame())
        grown["HGT"] = np.zeros((4, 5), np.float32)
        with pytest.raises(Exception, match="HGT"):
            writer.write_frame("1974-04-03_18:05:00", grown)
    finally:
        writer.abort()


def test_a_frame_missing_a_variable_is_refused_at_close(
        tmp_path, monkeypatch):
    """A hole in the classic data section reads back as garbage; the
    writer refuses to finish such a tape rather than publish it."""
    monkeypatch.delenv(WRFOUT_WRITER_ENV, raising=False)
    _require_rust_writer()
    path = tmp_path / "wrfout_hole"
    writer = WrfoutWriter(path, nx=5, ny=4, nz=3, dx=100.0, dy=100.0)
    shrunk = dict(_frame())
    del shrunk["MU"]
    writer.write_frame("1974-04-03_18:00:00", _frame())
    writer.write_frame("1974-04-03_18:05:00", shrunk)
    with pytest.raises(Exception, match="MU"):
        writer.close()
    # The failed publication was quarantined, not left under the final name.
    assert not path.exists()


@pytest.mark.parametrize("engine", ["rust", "python"])
def test_a_tape_with_no_frames_is_refused_at_close(
        tmp_path, monkeypatch, engine):
    """Zero Times records must not publish, on either engine.

    Every publication step succeeds on an empty tape: the classic header
    freezes with ``GPUWM_WRITE_COMPLETE = 1``, ``finish()`` sees no
    unwritten region because there are no records, and
    ``validate_wrfout_file`` passes with ``times == ()`` and every shape
    carrying ``Time = 0``.  Measured on the pre-fix writer, the published
    file reported ``GPUWM_WRITE_COMPLETE = 1`` with ``Times`` of shape
    ``(0, 19)``.  The completion attribute is what
    ``quarantine_orphan_wrfouts`` trusts, so nothing downstream would ever
    look at it again.

    The engine parametrization is essential: the guard belongs to
    ``WrfoutWriter.close()``, ahead of the engine branch, and a repair
    pushed down into the classic tape would leave the netCDF4 escape
    publishing empty files.
    """
    monkeypatch.delenv(WRFOUT_WRITER_ENV, raising=False)
    if engine == "rust":
        _require_rust_writer()
    path = tmp_path / f"wrfout_empty_{engine}"
    writer = WrfoutWriter(path, nx=5, ny=4, nz=3, dx=100.0, dy=100.0,
                          engine=engine)
    with pytest.raises(ValueError, match="zero Times records"):
        writer.close()
    assert not path.exists()
    # And the temp file went to quarantine rather than being left for the
    # orphan sweep to reason about.
    assert not list(tmp_path.glob(".wrfout*"))
    # close() is still idempotent after the refusal, as it is after any
    # other publication failure.
    writer.close()

    # A one-frame tape on the same engine still publishes.
    good = tmp_path / f"wrfout_one_{engine}"
    with WrfoutWriter(good, nx=5, ny=4, nz=3, dx=100.0, dy=100.0,
                      engine=engine) as writer:
        writer.write_frame("1974-04-03_18:00:00", _frame())
    assert good.exists()


def test_a_relative_output_path_publishes_and_validates(tmp_path, monkeypatch):
    """The writer pins its paths to absolute at construction.

    The concrete breakage: this box's netCDF-C build refuses to open a
    CLASSIC file through a drive-relative path (``\\tmp\\...`` ->
    ``NetCDF: Unknown file format``) while opening the same bytes
    absolutely works -- so the publish-time self-validation failed on a
    correctly written tape.  ``tilestream.test_history`` writes to
    ``/tmp`` and found it.  Relative spellings must therefore be
    resolved once, at construction, for both engines.
    """
    monkeypatch.delenv(WRFOUT_WRITER_ENV, raising=False)
    _require_rust_writer()
    monkeypatch.chdir(tmp_path)
    with WrfoutWriter("history/wrfout_d01_rel", nx=5, ny=4, nz=3,
                      dx=100.0, dy=100.0) as w:
        w.write_frame("1974-04-03_18:00:00", _frame())
    published = tmp_path / "history" / "wrfout_d01_rel"
    assert published.is_file()
    with netCDF4.Dataset(published) as ds:
        assert int(getattr(ds, "GPUWM_WRITE_COMPLETE")) == 1


def test_an_aborted_default_write_leaves_no_final_file(tmp_path, monkeypatch):
    monkeypatch.delenv(WRFOUT_WRITER_ENV, raising=False)
    _require_rust_writer()
    path = tmp_path / "wrfout_aborted"
    with pytest.raises(ValueError, match="deliberate"):
        with WrfoutWriter(path, nx=5, ny=4, nz=3, dx=100.0, dy=100.0) as w:
            w.write_frame("1974-04-03_18:00:00", _frame())
            raise ValueError("deliberate")
    assert not path.exists()


# ---------------------------------------------------------------------------
# The async production path inherits the default.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("compression,container", [
    (None, "NETCDF4_CLASSIC"), ("none", "NETCDF3_64BIT_OFFSET"),
])
def test_the_async_worker_writes_through_the_default_engine(
        tmp_path, monkeypatch, compression, container):
    """The worker constructs the plain ``WrfoutWriter``, so one default
    governs the synchronous and asynchronous paths alike."""
    import threading
    from test_wrfout import _manual_async_writer, _queue_cpu_ticket

    monkeypatch.delenv(WRFOUT_WRITER_ENV, raising=False)
    monkeypatch.delenv(HISTORY_COMPRESSION_ENV, raising=False)
    if compression is not None:
        monkeypatch.setenv(HISTORY_COMPRESSION_ENV, compression)
    _require_rust_writer()
    writer = _manual_async_writer(threading.Event())
    try:
        _queue_cpu_ticket(writer, tmp_path / "wrfout_d01_async")
        writer.drain()
    finally:
        writer.close()
    with netCDF4.Dataset(tmp_path / "wrfout_d01_async") as ds:
        assert ds.file_format == container
        if compression is None:
            assert ds.variables["T"].filters()["zlib"]
        assert int(getattr(ds, "GPUWM_WRITE_COMPLETE")) == 1
