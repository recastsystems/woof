"""The ctypes seam onto the Rust classic-NetCDF writer.

ACCEPTANCE (1) for that writer lives here: a file written THROUGH THE SEAM
is read back by netCDF4-python -- the Unidata C library, a third
independent implementation after the Rust writer and the Rust reader used
in ``tools/rustwx/crates/netcdf-writer/tests/classic_roundtrip.rs`` -- and
its structure and values must come back unchanged.

Values are compared bitwise rather than with a tolerance.  A tolerance
would hide the defects this test exists to catch: a byte-swap that lands
near the right value, a NaN whose payload moved, a signed zero that lost
its sign.
"""

from __future__ import annotations

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")

from woof.io import nc_writer_bridge as bridge


def _schema(container: str = "cdf2"):
    try:
        return bridge.ClassicSchema(container)
    except (FileNotFoundError, OSError, bridge.NcWriteError) as error:
        pytest.skip(f"the Rust NetCDF writer library is not built: {error}")


def test_seam_writes_a_wrfout_shaped_tape_netcdf4_python_reads_back(tmp_path):
    """The whole shape a product tape needs, through the seam, in one go."""

    path = tmp_path / "wrfout_shaped.nc"
    ny, nx, nz = 4, 5, 3
    schema = _schema()

    time = schema.def_dim("Time", 0, unlimited=True)
    strlen = schema.def_dim("DateStrLen", 19)
    south_north = schema.def_dim("south_north", ny)
    west_east = schema.def_dim("west_east", nx)
    bottom_top = schema.def_dim("bottom_top", nz)

    schema.put_global_attr("TITLE", " OUTPUT FROM GPUWM")
    schema.put_global_attr("DX", np.float32(12000.0))
    schema.put_global_attr("MAP_PROJ", np.int32(1))
    schema.put_global_attr("CEN_LAT", np.float32(38.5))
    schema.put_global_attr("BOTTOM-TOP_GRID_DIMENSION", np.int32(nz + 1))

    times = schema.def_var("Times", "S1", (time, strlen))
    t2 = schema.def_var("T2", "f4", (time, south_north, west_east))
    schema.put_var_attr(t2, "units", "K")
    schema.put_var_attr(t2, "description", "TEMP at 2 M")
    schema.put_var_attr(t2, "_FillValue", np.float32(-9999.0))
    theta = schema.def_var("T", "f4", (time, bottom_top, south_north, west_east))
    itimestep = schema.def_var("ITIMESTEP", "i4", (time,))
    # A fixed (non-record) variable, so the file exercises both regions.
    hgt = schema.def_var("HGT_M", "f4", (south_north, west_east))
    schema.put_var_attr(hgt, "units", "m")

    stamps = [b"2026-08-16_00:00:00", b"2026-08-16_01:00:00"]
    t2_fields = [
        np.arange(ny * nx, dtype="f4").reshape(ny, nx) + 280.0,
        np.arange(ny * nx, dtype="f4").reshape(ny, nx) * -0.5 + 290.0,
    ]
    # A NaN and a negative zero in real data, on purpose.
    t2_fields[0][1, 1] = np.nan
    t2_fields[1][0, 0] = -0.0
    theta_fields = [
        np.linspace(-30.0, 30.0, nz * ny * nx, dtype="f4").reshape(nz, ny, nx),
        np.linspace(30.0, -30.0, nz * ny * nx, dtype="f4").reshape(nz, ny, nx),
    ]
    terrain = np.arange(ny * nx, dtype="f4").reshape(ny, nx) * 3.25

    with schema.create(path) as writer:
        writer.write_var(hgt, terrain)
        for record, stamp in enumerate(stamps):
            writer.write_record(record, times, stamp)
            writer.write_record(record, t2, t2_fields[record])
            writer.write_record(record, theta, theta_fields[record])
            writer.write_record(record, itimestep,
                                np.array([record * 60], dtype="i4"))

    with netCDF4.Dataset(path) as ds:
        assert ds.file_format == "NETCDF3_64BIT_OFFSET"
        assert ds.dimensions["Time"].isunlimited()
        assert len(ds.dimensions["Time"]) == len(stamps)
        assert len(ds.dimensions["DateStrLen"]) == 19
        assert len(ds.dimensions["south_north"]) == ny
        assert len(ds.dimensions["west_east"]) == nx
        assert len(ds.dimensions["bottom_top"]) == nz

        assert ds.TITLE == " OUTPUT FROM GPUWM"
        assert ds.DX.dtype == np.float32 and ds.DX == np.float32(12000.0)
        assert ds.MAP_PROJ.dtype == np.int32 and ds.MAP_PROJ == 1
        assert ds.getncattr("BOTTOM-TOP_GRID_DIMENSION") == nz + 1

        assert list(ds.variables) == [
            "Times", "T2", "T", "ITIMESTEP", "HGT_M"]

        read_times = ds.variables["Times"]
        assert read_times.dtype == np.dtype("S1")
        assert read_times.dimensions == ("Time", "DateStrLen")
        joined = [b"".join(row).decode() for row in read_times[:]]
        assert joined == [stamp.decode() for stamp in stamps]

        var = ds.variables["T2"]
        var.set_auto_maskandscale(False)
        assert var.dimensions == ("Time", "south_north", "west_east")
        assert var.units == "K"
        assert var.description == "TEMP at 2 M"
        assert var.getncattr("_FillValue") == np.float32(-9999.0)
        stored = np.asarray(var[:])
        assert stored.dtype == np.dtype("f4")
        want = np.stack(t2_fields)
        assert stored.tobytes() == want.tobytes(), "bitwise, NaN and -0.0 included"

        theta_read = ds.variables["T"]
        theta_read.set_auto_maskandscale(False)
        assert np.asarray(theta_read[:]).tobytes() == np.stack(theta_fields).tobytes()

        steps = ds.variables["ITIMESTEP"]
        steps.set_auto_maskandscale(False)
        assert np.asarray(steps[:]).tolist() == [0, 60]

        fixed = ds.variables["HGT_M"]
        fixed.set_auto_maskandscale(False)
        assert fixed.dimensions == ("south_north", "west_east")
        assert np.asarray(fixed[:]).tobytes() == terrain.tobytes()


def test_seam_writes_every_classic_type(tmp_path):
    path = tmp_path / "types.nc"
    schema = _schema()
    n = schema.def_dim("n", 3)
    cases = {
        "vbyte": ("i1", np.array([-128, 0, 127], dtype="i1")),
        "vshort": ("i2", np.array([-32768, 7, 32767], dtype="i2")),
        "vint": ("i4", np.array([-2147483648, 0, 2147483647], dtype="i4")),
        "vfloat": ("f4", np.array([1.5, -2.25, np.nan], dtype="f4")),
        "vdouble": ("f8", np.array([1e300, -0.0, 3.25], dtype="f8")),
    }
    varids = {name: schema.def_var(name, dtype, (n,))
              for name, (dtype, _) in cases.items()}
    chars = schema.def_var("vchar", "S1", (n,))
    with schema.create(path) as writer:
        for name, (_, values) in cases.items():
            writer.write_var(varids[name], values)
        writer.write_var(chars, b"abc")

    with netCDF4.Dataset(path) as ds:
        for name, (dtype, values) in cases.items():
            var = ds.variables[name]
            var.set_auto_maskandscale(False)
            assert var.dtype == np.dtype(dtype), name
            assert np.asarray(var[:]).tobytes() == values.tobytes(), name
        assert b"".join(np.asarray(ds.variables["vchar"][:])) == b"abc"


def test_cdf5_carries_the_extended_types(tmp_path):
    path = tmp_path / "cdf5.nc"
    schema = _schema("cdf5")
    n = schema.def_dim("n", 2)
    u64 = schema.def_var("big", "u8", (n,))
    i64 = schema.def_var("signed", "i8", (n,))
    with schema.create(path) as writer:
        writer.write_var(u64, np.array([0, 2**64 - 1], dtype="u8"))
        writer.write_var(i64, np.array([-(2**63), 2**63 - 1], dtype="i8"))

    with netCDF4.Dataset(path) as ds:
        assert ds.file_format == "NETCDF3_64BIT_DATA"
        assert np.asarray(ds.variables["big"][:]).tolist() == [0, 2**64 - 1]
        assert np.asarray(ds.variables["signed"][:]).tolist() == [
            -(2**63), 2**63 - 1]


def test_seam_refuses_a_payload_of_the_wrong_dtype(tmp_path):
    # The concrete breakage: f8 values pushed into an NC_FLOAT variable
    # would double every element's byte width and shift the data section.
    schema = _schema()
    n = schema.def_dim("n", 2)
    var = schema.def_var("a", "f4", (n,))
    with schema.create(tmp_path / "dtype.nc") as writer:
        with pytest.raises(bridge.NcWriteError, match="declared"):
            writer.write_var(var, np.array([1.0, 2.0], dtype="f8"))
        writer.write_var(var, np.array([1.0, 2.0], dtype="f4"))


def test_seam_relays_the_rust_refusal_text(tmp_path):
    # The Rust message, not a Python paraphrase: the seam has no second
    # copy of the writer's rules to drift from.
    schema = _schema()
    time = schema.def_dim("Time", 0, unlimited=True)
    y = schema.def_dim("y", 2)
    with pytest.raises(bridge.NcWriteError, match="first dimension"):
        schema.def_var("transposed", "f4", (y, time))


def test_a_block_that_raises_leaves_no_finished_tape(tmp_path):
    # A partial file must not be mistaken for a finished one: aborting
    # leaves numrecs at 0, so a reader sees an empty tape rather than a
    # half-written frame presented as whole.
    path = tmp_path / "aborted.nc"
    schema = _schema()
    time = schema.def_dim("Time", 0, unlimited=True)
    y = schema.def_dim("y", 2)
    var = schema.def_var("T2", "f4", (time, y))
    with pytest.raises(ValueError, match="deliberate"):
        with schema.create(path) as writer:
            writer.write_record(0, var, np.array([1.0, 2.0], dtype="f4"))
            raise ValueError("deliberate")
    with netCDF4.Dataset(path) as ds:
        assert len(ds.dimensions["Time"]) == 0


# ---------------------------------------------------------------------------
# The read-back sweep: which float variables hold a non-finite value.
#
# ACCEPTANCE (2): the sweep is asked about files netCDF4-python WROTE, in
# all three classic containers, so the header parser is proved against the
# Unidata C library's own header rather than only against the one this
# crate emits.  A parser that only ever reads its own output is a parser
# that agrees with itself.
# ---------------------------------------------------------------------------

def _require_seam():
    try:
        bridge.load()
    except (FileNotFoundError, OSError, bridge.NcWriteError) as error:
        pytest.skip(f"the Rust NetCDF writer library is not built: {error}")


def _c_library_file(path, container, *, poison=None):
    """A wrfinput-shaped classic file, written by netCDF4-python."""
    ds = netCDF4.Dataset(path, "w", format=container)
    ds.createDimension("Time", None)
    ds.createDimension("DateStrLen", 19)
    ds.createDimension("south_north", 3)
    ds.createDimension("west_east", 4)
    ds.TITLE = " OUTPUT FROM WRF"
    ds.MAP_PROJ = np.int32(1)
    times = ds.createVariable("Times", "S1", ("Time", "DateStrLen"))
    t2 = ds.createVariable("T2", "f4", ("Time", "south_north", "west_east"))
    t2.units = "K"
    znu = ds.createVariable("ZNU", "f8", ("south_north",))
    step = ds.createVariable("ITIMESTEP", "i4", ("Time",))
    field = np.arange(12, dtype="f4").reshape(3, 4)
    if poison is not None:
        field[1, 2] = poison
    times[0] = np.frombuffer(b"2026-08-18_00:00:00", dtype="S1")
    t2[0] = field
    znu[:] = [0.9959, 0.9875, 0.9713]
    step[0] = 3
    ds.close()
    return path


@pytest.mark.parametrize("container", ["NETCDF3_CLASSIC",
                                       "NETCDF3_64BIT_OFFSET",
                                       "NETCDF3_64BIT_DATA"])
def test_the_sweep_reads_a_c_library_file_and_finds_its_nan(
        tmp_path, container):
    _require_seam()
    path = _c_library_file(tmp_path / f"{container}.nc", container,
                           poison=np.nan)
    assert bridge.scan_nonfinite(path) == ("T2",)


@pytest.mark.parametrize("container", ["NETCDF3_CLASSIC",
                                       "NETCDF3_64BIT_OFFSET",
                                       "NETCDF3_64BIT_DATA"])
def test_a_clean_c_library_file_sweeps_clean(tmp_path, container):
    _require_seam()
    path = _c_library_file(tmp_path / f"clean-{container}.nc", container)
    assert bridge.scan_nonfinite(path) == ()


def test_the_sweep_finds_an_infinity_the_seam_itself_wrote(tmp_path):
    path = tmp_path / "infinite.nc"
    schema = _schema()
    time = schema.def_dim("Time", 0, unlimited=True)
    y = schema.def_dim("y", 2)
    clean = schema.def_var("CLEAN", "f4", (time, y))
    dirty = schema.def_var("DIRTY", "f4", (time, y))
    with schema.create(path) as writer:
        writer.write_record(0, clean, np.array([1.0, 2.0], dtype="f4"))
        writer.write_record(0, dirty, np.array([1.0, 2.0], dtype="f4"))
        writer.write_record(1, clean, np.array([3.0, 4.0], dtype="f4"))
        # The defect is in the SECOND record only, which is the shape a
        # wrfbdy defect takes: one boundary time out of many.
        writer.write_record(1, dirty, np.array([3.0, -np.inf], dtype="f4"))
    assert bridge.scan_nonfinite(path) == ("DIRTY",)


def test_a_truncated_file_refuses_rather_than_sweeping_clean(tmp_path):
    """"Found nothing" and "could not look" must not read the same.

    The concrete breakage: a wrfinput cut short by a full disk would
    otherwise pass the export's finiteness gate and reach stock WRF,
    where a short read is undefined behaviour rather than an error.
    """
    _require_seam()
    path = _c_library_file(tmp_path / "short.nc", "NETCDF3_64BIT_OFFSET")
    payload = path.read_bytes()
    path.write_bytes(payload[:-16])
    with pytest.raises(bridge.NcWriteError, match="truncated"):
        bridge.scan_nonfinite(path)


def test_an_hdf5_container_is_refused_and_names_the_reader_that_owns_it(
        tmp_path):
    _require_seam()
    path = tmp_path / "netcdf4.nc"
    ds = netCDF4.Dataset(path, "w", format="NETCDF4")
    ds.createDimension("n", 2)
    ds.createVariable("T2", "f4", ("n",))[:] = [1.0, 2.0]
    ds.close()
    with pytest.raises(bridge.NcWriteError, match="rw_netcdf"):
        bridge.scan_nonfinite(path)


def test_the_writer_is_a_bundled_artifact_with_one_spelling_everywhere():
    """The staging machinery and this seam must name one contract.

    The bundle row, the ABI-marker table and the library handshake are
    each declared once in the packaging layer; this binds them to the
    seam's own constants so the two layers cannot drift into loading the
    same library under two contracts.
    """
    from woof import bridge_assets, bridges

    artifact, = [a for a in bridge_assets.BUNDLED_ARTIFACTS
                 if a.name == "netcdf_writer"]
    assert artifact.kind == "library"
    assert artifact.env_var == bridge.NCWRITE_BRIDGE_ENV
    assert not artifact.vendored          # gpuwm-authored: stamp-proved
    assert bridges.BRIDGE_ABI_MARKERS["netcdf_writer"] == bridge.ABI_MARKER
    symbol, version = bridge_assets.library_abi_for("netcdf_writer")
    assert symbol == "gpuwm_ncwrite_abi_version"
    assert version == bridge.NCWRITE_ABI
    # The staged filename is the one the seam's own loader looks for.
    assert bridge_assets.artifact_filename(artifact, "win-x86_64") \
        == "netcdf_writer.dll"
    assert bridge_assets.artifact_filename(artifact, "linux-x86_64") \
        == "libnetcdf_writer.so"


def test_abi_marker_is_present_in_the_built_library():
    # The BRIDGE_ABI_MARKERS discipline, applied to this library: a build
    # predating the record dimension exports the rest, loads cleanly and
    # answers the version probe, then cannot write a `Times` variable.
    try:
        path = bridge.resolve_ncwrite_bridge()
    except FileNotFoundError as error:
        pytest.skip(f"the Rust NetCDF writer library is not built: {error}")
    assert bridge.ABI_MARKER in path.read_bytes()


#: The writer's checkout build as each shell must receive it, written out
#: rather than derived so a generator that loses its shell rule cannot
#: also rewrite what it is judged against.  Windows PowerShell 5.1
#: rejects `&&` with a parser error.
NC_WRITER_BUILD_FOR_SHELL = {
    False: "cd tools/rustwx && cargo build --release --offline && cd ../..",
    True: "cd tools/rustwx; cargo build --release --offline; cd ../..",
}


@pytest.mark.parametrize("windows", (False, True))
def test_the_missing_library_refusal_spells_the_build_for_the_shell(
        monkeypatch, tmp_path, windows):
    """The not-found refusal reads the one shell rule every remedy reads.

    It chose its separator from ``os.name``, so no test could force its
    Windows spelling and a regression to `&&` there would pass on every
    Linux runner.
    """

    from woof import bridges

    monkeypatch.setattr(bridges, "WINDOWS_SHELL", windows)
    monkeypatch.delenv(bridge.NCWRITE_BRIDGE_ENV, raising=False)
    monkeypatch.setattr(bridge, "library_candidates",
                        lambda: (tmp_path / "absent" / "netcdf_writer.dll",))
    with pytest.raises(FileNotFoundError) as caught:
        bridge.resolve_ncwrite_bridge()
    message = str(caught.value)
    assert NC_WRITER_BUILD_FOR_SHELL[windows] in message, message
    if windows:
        assert "&&" not in message, (
            f"Windows PowerShell 5.1 cannot parse '&&': {message}")


def test_history_frames_are_lossless_compressed_netcdf4_by_default(tmp_path):
    """Project ruling, 2026-10-28: "Lossless compression is the default."  A history
    frame written by WrfoutWriter is a NetCDF-4 (classic model) file whose
    every dimension, attribute and word equals the uncompressed frame's;
    ``none`` still writes the classic CDF-2 frame."""

    from woof.io import wrfout
    if bridge.netcdf4_unavailable_reason() is not None:
        pytest.skip(bridge.netcdf4_unavailable_reason())
    nx, ny, nz = 37, 29, 6
    rng = np.random.default_rng(3)
    fields = {
        "T2": (280 + rng.normal(0, 2, (ny, nx))).astype(np.float32),
        "T": rng.normal(0, 3, (nz, ny, nx)).astype(np.float32),
        "U": rng.normal(5, 3, (nz, ny, nx + 1)).astype(np.float32),
        "W": rng.normal(0, 0.3, (nz + 1, ny, nx)).astype(np.float32),
        "RAINNC": np.zeros((ny, nx), np.float32),
    }
    fields["T"][0, 0, 0] = np.float32(np.nan)
    paths = {}
    for mode in ("none", "deflate"):
        path = tmp_path / f"wrfout_d01_2026-10-28_12_00_00.{mode}"
        with wrfout.WrfoutWriter(path, nx=nx, ny=ny, nz=nz, dx=1000.0, dy=1000.0,
                                 compression=mode) as writer:
            writer.write_frame("2026-10-28_12:00:00", fields)
        paths[mode] = path
    assert wrfout.resolve_history_compression() == "deflate"
    with netCDF4.Dataset(paths["none"]) as a, netCDF4.Dataset(paths["deflate"]) as b:
        assert a.data_model == "NETCDF3_64BIT_OFFSET"
        assert b.data_model == "NETCDF4_CLASSIC"
        assert b.dimensions["Time"].isunlimited()
        assert {k: len(d) for k, d in a.dimensions.items()} == {
            k: len(d) for k, d in b.dimensions.items()}
        for name in a.ncattrs():
            assert np.array_equal(np.asarray(a.getncattr(name)),
                                  np.asarray(b.getncattr(name))), name
        assert set(a.variables) == set(b.variables)
        for name, va in a.variables.items():
            vb = b.variables[name]
            va.set_auto_maskandscale(False)
            vb.set_auto_maskandscale(False)
            assert va.dimensions == vb.dimensions
            xa, xb = np.asarray(va[:]), np.asarray(vb[:])
            if xa.dtype.kind == "f":
                xa, xb = xa.view(f"u{xa.itemsize}"), xb.view(f"u{xb.itemsize}")
            assert np.array_equal(xa, xb), name
        if b.variables["T"].filters() is not None:
            assert b.variables["T"].filters()["zlib"]
            assert b.variables["T"].filters()["shuffle"]
