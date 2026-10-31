"""ctypes seam onto the Rust classic-NetCDF writer.

This is the write-side counterpart to :mod:`woof.netcdf_bridge`, which is
the read-side facade over ``rw_netcdf``.  The library it loads is
``tools/rustwx/crates/netcdf-writer``, a dependency-free emitter for the
NetCDF classic containers (CDF-1, CDF-2, CDF-5) with the full classic type
set, a record dimension and attributes -- everything a wrfout needs and
none of which the previous Rust writer (``rw-store``'s ``netcdf3.rs``,
CDF-2, ``NC_FLOAT`` only, no record dimension) could express.

Why ctypes rather than pyo3
---------------------------
Every Rust library woof already drives from Python is a cdylib behind
ctypes: ``gpuwm_preprocess_cpu`` (:mod:`woof.ingest.cpu_backend`) and
``region_global_dealias`` (:mod:`woof.obs.dealias_region`,
:mod:`woof.obs.coarse_cost`).  One loading discipline, one staging path
in ``tools/stage_wheel_bridges.py``, one ABI-marker rule in
:mod:`woof.bridges`, and no build-time bind to a CPython version.  A
pyo3 module would need a maturin build per interpreter and would be the
only artifact of its kind in the estate.

What this module does NOT do
----------------------------
It does not change what :class:`woof.io.wrfout.WrfoutWriter` writes with
today.  Flipping the product tape onto this writer is an integration step
with its own dual-write verification, and doing it here would mean
changing the product tape in the same commit that first introduces the
library that writes it.

Shape
-----
The calls mirror the netCDF C API on purpose, because the caller being
ported is written against netCDF4-python::

    from woof.io.nc_writer_bridge import ClassicSchema

    schema = ClassicSchema()                       # CDF-2 by default
    time = schema.def_dim("Time", 0, unlimited=True)
    strlen = schema.def_dim("DateStrLen", 19)
    south_north = schema.def_dim("south_north", ny)
    west_east = schema.def_dim("west_east", nx)
    schema.put_global_attr("TITLE", " OUTPUT FROM GPUWM")
    times = schema.def_var("Times", "S1", (time, strlen))
    t2 = schema.def_var("T2", "f4", (time, south_north, west_east))
    schema.put_var_attr(t2, "units", "K")

    with schema.create(path) as writer:
        writer.write_record(0, times, b"2026-08-16_00:00:00")
        writer.write_record(0, t2, field)          # C-contiguous f4

Adding a variable is a row in the schema table driving ``def_var``.  There
is no per-variable code path on either side of the seam.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Final

import numpy as np

#: The C ABI version this module speaks.  A library reporting anything
#: else is refused rather than called: the signatures are positional and
#: a mismatched build would be interpreted, not detected.
NCWRITE_ABI: Final[int] = 1

#: Environment override for the library path, first rung of the ladder.
NCWRITE_BRIDGE_ENV: Final[str] = "WOOF_NCWRITE_BRIDGE"

#: The exported symbol that identifies THIS contract, for
#: :data:`woof.bridges.BRIDGE_ABI_MARKERS`.  It names the newest
#: capability a DEFAULT path depends on: ``gpuwm_ncwrite_write_record``
#: held it while the record dimension was that capability (a build
#: predating it exports every other entry point, loads cleanly, answers
#: the version probe -- and then cannot write a ``Times`` variable), and
#: the read-back sweep is now it, because the default wrfinput/wrfbdy
#: export verifies every float variable through
#: :func:`scan_nonfinite` before publishing.
ABI_MARKER: Final[bytes] = b"gpuwm_ncwrite_scan_nonfinite"

# Classic external type codes, as the format writes them.
NC_BYTE: Final[int] = 1
NC_CHAR: Final[int] = 2
NC_SHORT: Final[int] = 3
NC_INT: Final[int] = 4
NC_FLOAT: Final[int] = 5
NC_DOUBLE: Final[int] = 6
NC_UBYTE: Final[int] = 7
NC_USHORT: Final[int] = 8
NC_UINT: Final[int] = 9
NC_INT64: Final[int] = 10
NC_UINT64: Final[int] = 11

#: Container version codes accepted by ``gpuwm_ncwrite_schema_new``.
FORMATS: Final[dict[str, int]] = {"cdf1": 1, "cdf2": 2, "cdf5": 5}

#: ``(numpy kind, itemsize) -> classic type code``.  A table, so adding a
#: type is a row.  ``S`` is bytes, which is how numpy spells ``NC_CHAR``.
_NUMPY_TO_NC: Final[dict[tuple[str, int], int]] = {
    ("i", 1): NC_BYTE,
    ("S", 1): NC_CHAR,
    ("i", 2): NC_SHORT,
    ("i", 4): NC_INT,
    ("f", 4): NC_FLOAT,
    ("f", 8): NC_DOUBLE,
    ("u", 1): NC_UBYTE,
    ("u", 2): NC_USHORT,
    ("u", 4): NC_UINT,
    ("i", 8): NC_INT64,
    ("u", 8): NC_UINT64,
}

_NC_TO_NUMPY: Final[dict[int, str]] = {
    NC_BYTE: "i1",
    NC_CHAR: "S1",
    NC_SHORT: "i2",
    NC_INT: "i4",
    NC_FLOAT: "f4",
    NC_DOUBLE: "f8",
    NC_UBYTE: "u1",
    NC_USHORT: "u2",
    NC_UINT: "u4",
    NC_INT64: "i8",
    NC_UINT64: "u8",
}


class NcWriteError(RuntimeError):
    """The Rust writer refused, with its own message."""


def library_names() -> tuple[str, ...]:
    if os.name == "nt":
        return ("netcdf_writer.dll",)
    if os.uname().sysname == "Darwin":  # pragma: no cover - platform route
        return ("libnetcdf_writer.dylib",)
    return ("libnetcdf_writer.so",)


def library_candidates() -> tuple[Path, ...]:
    """Deterministic candidate paths, best first.

    The same ladder :mod:`woof.rustwx` walks for ``rw_wrfbatch``, because
    the library is built by the same workspace: environment override,
    checkout release, checkout debug, ``libexec/bridges`` beside the
    package, the wheel-bundled directory, the user-level default.
    """

    from woof.bridges import (default_bridge_dir, legacy_bridge_candidates,
                               packaged_bridge_dir)
    from woof.rustwx import crate_dir

    filename = library_names()[0]
    candidates: list[Path] = []
    override = os.environ.get(NCWRITE_BRIDGE_ENV)
    if override:
        candidates.append(Path(override))
    root = Path(__file__).resolve().parent.parent.parent
    candidates.extend((
        crate_dir() / "target" / "release" / filename,
        crate_dir() / "target" / "debug" / filename,
        root / "libexec" / "bridges" / filename,
        packaged_bridge_dir() / filename,
        default_bridge_dir() / filename,
        *legacy_bridge_candidates(filename),
    ))
    return tuple(candidates)


def checkout_build_command() -> str:
    """The checkout build of this library, spelled for the reader's shell.

    The not-found refusal below and the observation-product writer's
    refusal (:mod:`woof.obs.grid_product`) both print it, from here, so
    the two cannot drift apart.  The separator comes from
    :data:`woof.bridges.WINDOWS_SHELL`, the one shell rule every remedy
    reads, because Windows PowerShell 5.1 rejects ``&&`` with a parser
    error: the product writer's refusal hard-coded ``&&``, so on Windows
    its build line could not be pasted.  It ends with the ``cd`` back so
    a pasted block leaves the shell where it started.
    """

    from woof import bridges

    separator = ";" if bridges.WINDOWS_SHELL else " &&"
    return (f"cd {bridges.RUSTWX_CRATE_RELATIVE}{separator} cargo build "
            f"--release --offline{separator} cd ../..")


def resolve_ncwrite_bridge() -> Path:
    """The first existing candidate, or a refusal listing every path.

    An override naming a missing file is a hard error: explicit
    configuration must fail loudly rather than fall through to a
    different library.
    """

    override = os.environ.get(NCWRITE_BRIDGE_ENV)
    for candidate in library_candidates():
        if candidate.is_file():
            from woof.bridges import accept_resolved

            return accept_resolved(candidate.resolve(),
                                   executable=False)
        if override and candidate == Path(override):
            raise FileNotFoundError(
                f"{NCWRITE_BRIDGE_ENV} names a missing file: {candidate}")
    rendered = "\n  ".join(str(candidate) for candidate in library_candidates())
    raise FileNotFoundError(
        "the Rust NetCDF writer library was not found; searched:\n  "
        + rendered
        + "\n  # build it from a checkout:\n  "
        + checkout_build_command())


_LIBRARY: ctypes.CDLL | None = None


def unavailable_reason() -> str | None:
    """Why the Rust writer is not loadable here, or None when it is."""

    try:
        load()
    except (FileNotFoundError, OSError, NcWriteError) as error:
        return f"{type(error).__name__}: {error}"
    return None


def load() -> ctypes.CDLL:
    """Load the library once and bind every signature."""

    global _LIBRARY
    if _LIBRARY is not None:
        return _LIBRARY
    path = resolve_ncwrite_bridge()
    library = ctypes.CDLL(str(path))

    library.gpuwm_ncwrite_abi_version.argtypes = []
    library.gpuwm_ncwrite_abi_version.restype = ctypes.c_uint32
    observed = int(library.gpuwm_ncwrite_abi_version())
    if observed != NCWRITE_ABI:
        raise NcWriteError(
            f"{path} speaks NetCDF-writer ABI {observed}, this woof needs "
            f"{NCWRITE_ABI}; rebuild tools/rustwx")

    u8p = ctypes.POINTER(ctypes.c_uint8)
    u64p = ctypes.POINTER(ctypes.c_uint64)
    void = ctypes.c_void_p
    size = ctypes.c_size_t

    for name, args, result in (
            ("gpuwm_ncwrite_last_error", [u8p, size], size),
            ("gpuwm_ncwrite_schema_new", [ctypes.c_uint32], void),
            ("gpuwm_ncwrite_schema_free", [void], None),
            ("gpuwm_ncwrite_def_dim",
                [void, ctypes.c_char_p, size, ctypes.c_uint64, ctypes.c_int32], ctypes.c_int64),
            ("gpuwm_ncwrite_def_var",
                [void, ctypes.c_char_p, size, ctypes.c_uint32, size, u64p], ctypes.c_int64),
            ("gpuwm_ncwrite_put_att",
                [void, ctypes.c_int64, ctypes.c_char_p, size, ctypes.c_uint32, size, ctypes.c_char_p], ctypes.c_int32),
            ("gpuwm_ncwrite_create", [void, ctypes.c_char_p, size], void),
            ("gpuwm_ncwrite_write_var", [void, ctypes.c_uint64, ctypes.c_uint32, ctypes.c_char_p, size], ctypes.c_int32),
            ("gpuwm_ncwrite_write_record",
                [void, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint32, ctypes.c_char_p, size], ctypes.c_int32),
            ("gpuwm_ncwrite_finish", [void], ctypes.c_int32),
            ("gpuwm_ncwrite_abort", [void], None),
            ("gpuwm_ncwrite_num_records", [void], ctypes.c_uint64),
    ):
        function = getattr(library, name)
        function.argtypes = args
        function.restype = result
    # The read-back sweep. Bound here rather than probed at the call site
    # so a library predating it is refused at LOAD, with the rebuild
    # remedy, instead of half-way through an export.
    try:
        library.gpuwm_ncwrite_scan_nonfinite.argtypes = [ctypes.c_char_p, size]
        library.gpuwm_ncwrite_scan_nonfinite.restype = ctypes.c_int32
        library.gpuwm_ncwrite_last_scan.argtypes = [u8p, size]
        library.gpuwm_ncwrite_last_scan.restype = size
    except AttributeError:
        raise NcWriteError(
            f"{path} exports no gpuwm_ncwrite_scan_nonfinite, so it predates "
            f"this release's netcdf-writer contract (see ABI_MARKER): it can "
            f"write a file but cannot read one back, and the default "
            f"wrfinput/wrfbdy export verifies every float variable before it "
            f"publishes.  Rebuild tools/rustwx (cargo build --release "
            f"--locked --offline -p netcdf-writer), or run "
            f"`woof fetch-bridges` on an installed wheel") from None

    _bind_netcdf4(library)
    _LIBRARY = library
    return library


#: Whether the loaded library exports the compressed NetCDF-4 frame writer
#: (gpuwm_ncwrite_create_netcdf4).  Bound at load, refused at USE: the
#: classic tape and the wrfinput/wrfbdy export never need it, so a library
#: predating it still writes those, and the history writer refuses by name.
_NETCDF4_BOUND = False


def _bind_netcdf4(library: ctypes.CDLL) -> None:
    global _NETCDF4_BOUND
    void = ctypes.c_void_p
    size = ctypes.c_size_t
    u64p = ctypes.POINTER(ctypes.c_uint64)
    try:
        for name, args, result in (
                ("gpuwm_ncwrite_create_netcdf4",
                    [void, ctypes.c_char_p, size, ctypes.c_uint32, ctypes.c_int32, ctypes.c_uint32], void),
                ("gpuwm_ncwrite_nc4_write_var",
                    [void, ctypes.c_uint64, ctypes.c_uint32, void, size], ctypes.c_int32),
                ("gpuwm_ncwrite_nc4_write_record",
                    [void, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint32, void, size], ctypes.c_int32),
                ("gpuwm_ncwrite_nc4_num_records", [void], ctypes.c_uint64),
                ("gpuwm_ncwrite_nc4_bytes", [void, u64p, u64p], ctypes.c_int32),
                ("gpuwm_ncwrite_nc4_finish", [void], ctypes.c_int32),
                ("gpuwm_ncwrite_nc4_abort", [void], None),
                ("gpuwm_ncwrite_nc4_pool_threads", [], ctypes.c_uint32),
        ):
            function = getattr(library, name)
            function.argtypes = args
            function.restype = result
    except AttributeError:
        _NETCDF4_BOUND = False
        return
    _NETCDF4_BOUND = True


def netcdf4_unavailable_reason() -> str | None:
    """Why compressed NetCDF-4 frames cannot be written here, or None."""

    reason = unavailable_reason()
    if reason is not None:
        return reason
    try:
        load()
    except Exception as error:  # noqa: BLE001 - the reason is the message
        return str(error)
    if not _NETCDF4_BOUND:
        return (f"{resolve_ncwrite_bridge()} exports no "
                "gpuwm_ncwrite_create_netcdf4, so it predates compressed "
                "history frames; rebuild tools/rustwx (cargo build --release "
                "--locked --offline -p netcdf-writer), run `woof "
                "fetch-bridges`, or write uncompressed frames with "
                "GPUWM_HISTORY_COMPRESSION=none")
    return None


def _last_error(library: ctypes.CDLL) -> str:
    needed = int(library.gpuwm_ncwrite_last_error(None, 0))
    if needed == 0:
        return "the Rust writer reported a failure with no message"
    buffer = (ctypes.c_uint8 * needed)()
    library.gpuwm_ncwrite_last_error(
        ctypes.cast(buffer, ctypes.POINTER(ctypes.c_uint8)), needed)
    return bytes(buffer).decode("utf-8", "replace")


def scan_nonfinite(path) -> tuple[str, ...]:
    """Read a finished classic file back; name its non-finite variables.

    Returns the float/double variables holding a value that is not
    finite, in definition order -- empty for a clean file.  Anything that
    stops the sweep (a truncated data section, an HDF5 container, an
    unreadable path) is an :class:`NcWriteError`, never an empty tuple:
    "found nothing" and "could not look" must not read the same.

    This is the write side's own verification, kept where the geometry
    is.  ``rw_netcdf`` is the front door for READING a NetCDF file; it
    decodes one variable per process launch through an f64 temp file,
    which is the right shape for resolving a selector and the wrong one
    for sweeping all 183 variables of a ``wrfinput`` on the preparation
    path.
    """

    library = load()
    raw = os.fspath(path).encode("utf-8")
    if library.gpuwm_ncwrite_scan_nonfinite(raw, len(raw)) != 0:
        raise NcWriteError(_last_error(library))
    needed = int(library.gpuwm_ncwrite_last_scan(None, 0))
    if needed == 0:
        return ()
    buffer = (ctypes.c_uint8 * needed)()
    library.gpuwm_ncwrite_last_scan(
        ctypes.cast(buffer, ctypes.POINTER(ctypes.c_uint8)), needed)
    report = bytes(buffer).decode("utf-8", "replace")
    return tuple(name for name in report.split("\n") if name)


def nc_type_of(dtype) -> int:
    """The classic type code for a numpy dtype, or a named refusal."""

    dtype = np.dtype(dtype)
    key = (dtype.kind, dtype.itemsize)
    code = _NUMPY_TO_NC.get(key)
    if code is None:
        raise NcWriteError(
            f"numpy dtype {dtype!r} has no classic NetCDF external type; "
            f"classic supports {sorted(set(_NC_TO_NUMPY.values()))}")
    return code


def _payload(values, nc_type: int, what: str) -> bytes:
    """C-contiguous, native-endian bytes of the declared external type."""

    want = np.dtype(_NC_TO_NUMPY[nc_type])
    if nc_type == NC_CHAR and isinstance(values, (bytes, bytearray)):
        return bytes(values)
    array = np.asarray(values)
    if array.dtype.kind != want.kind or array.dtype.itemsize != want.itemsize:
        raise NcWriteError(
            f"{what} is declared {want.str} but the payload is "
            f"{array.dtype.str}; the writer will not reinterpret it")
    # A non-native byte order would be written back to front: normalise
    # rather than refuse, because a big-endian view of the caller's own
    # array is a legitimate thing to hold.
    array = np.ascontiguousarray(array.astype(want.newbyteorder("="),
                                              copy=False))
    return array.tobytes()


class ClassicWriter:
    """An open file. Use as a context manager, or call :meth:`finish`."""

    def __init__(self, library: ctypes.CDLL, handle: int,
                 var_types: dict[int, int], path: Path) -> None:
        self._library = library
        self._handle = handle
        self._var_types = var_types
        self._path = path

    @property
    def path(self) -> Path:
        return self._path

    def _check(self, status: int) -> None:
        if status != 0:
            raise NcWriteError(_last_error(self._library))

    def _live(self) -> int:
        if self._handle is None:
            raise NcWriteError(f"{self._path} is already closed")
        return self._handle

    def write_var(self, varid: int, values) -> None:
        """Write a whole fixed (non-record) variable."""

        nc_type = self._var_types[varid]
        payload = _payload(values, nc_type, f"variable {varid}")
        self._check(self._library.gpuwm_ncwrite_write_var(
            self._live(), varid, nc_type, payload, len(payload)))

    def write_record(self, recno: int, varid: int, values) -> None:
        """Write one record's slab of a record variable."""

        nc_type = self._var_types[varid]
        payload = _payload(values, nc_type,
                           f"record {recno} of variable {varid}")
        self._check(self._library.gpuwm_ncwrite_write_record(
            self._live(), recno, varid, nc_type, payload, len(payload)))

    def num_records(self) -> int:
        return int(self._library.gpuwm_ncwrite_num_records(self._live()))

    def finish(self) -> None:
        """Patch ``numrecs``, flush, fsync, and release the handle."""

        handle = self._live()
        self._handle = None
        status = int(self._library.gpuwm_ncwrite_finish(handle))
        if status != 0:
            raise NcWriteError(_last_error(self._library))

    def abort(self) -> None:
        """Release the handle without finishing. The file stays partial."""

        if self._handle is not None:
            handle, self._handle = self._handle, None
            self._library.gpuwm_ncwrite_abort(handle)

    def __enter__(self) -> "ClassicWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        # An exception inside the block leaves a partial file that must
        # not be mistaken for a finished tape, so abort rather than
        # finish -- and let the original exception through.
        if exc_type is not None:
            self.abort()
            return False
        self.finish()
        return False


def _native_view(values, nc_type: int, what: str):
    """A C-contiguous native-endian array of the declared type, uncopied
    when the caller's array already is one (the NetCDF-4 writer copies the
    bytes itself before it returns)."""

    want = np.dtype(_NC_TO_NUMPY[nc_type])
    if nc_type == NC_CHAR and isinstance(values, (bytes, bytearray)):
        return np.frombuffer(bytes(values), dtype=np.uint8)
    array = np.asarray(values)
    if array.dtype.kind != want.kind or array.dtype.itemsize != want.itemsize:
        raise NcWriteError(
            f"{what} is declared {want.str} but the payload is "
            f"{array.dtype.str}; the writer will not reinterpret it")
    return np.ascontiguousarray(array.astype(want.newbyteorder("="),
                                             copy=False))


class Netcdf4Writer:
    """A compressed NetCDF-4 frame being written (lossless shuffle +
    deflate, chunks compressed on the library's worker pool).  Same calls
    as :class:`ClassicWriter`."""

    def __init__(self, library: ctypes.CDLL, handle: int,
                 var_types: dict[int, int], path: Path) -> None:
        self._library = library
        self._handle = handle
        self._var_types = var_types
        self._path = path

    @property
    def path(self) -> Path:
        return self._path

    def _check(self, status: int) -> None:
        if status != 0:
            raise NcWriteError(_last_error(self._library))

    def _live(self) -> int:
        if self._handle is None:
            raise NcWriteError(f"{self._path} is already closed")
        return self._handle

    def write_var(self, varid: int, values) -> None:
        nc_type = self._var_types[varid]
        array = _native_view(values, nc_type, f"variable {varid}")
        self._check(self._library.gpuwm_ncwrite_nc4_write_var(
            self._live(), varid, nc_type, array.ctypes.data, array.nbytes))

    def write_record(self, recno: int, varid: int, values) -> None:
        nc_type = self._var_types[varid]
        array = _native_view(values, nc_type,
                             f"record {recno} of variable {varid}")
        self._check(self._library.gpuwm_ncwrite_nc4_write_record(
            self._live(), recno, varid, nc_type, array.ctypes.data,
            array.nbytes))

    def num_records(self) -> int:
        return int(self._library.gpuwm_ncwrite_nc4_num_records(self._live()))

    def stored_bytes(self) -> tuple[int, int]:
        """(raw, stored) bytes of the chunks written so far."""

        raw, stored = ctypes.c_uint64(0), ctypes.c_uint64(0)
        self._check(self._library.gpuwm_ncwrite_nc4_bytes(
            self._live(), ctypes.byref(raw), ctypes.byref(stored)))
        return int(raw.value), int(stored.value)

    def finish(self) -> None:
        handle = self._live()
        self._handle = None
        status = int(self._library.gpuwm_ncwrite_nc4_finish(handle))
        if status != 0:
            raise NcWriteError(_last_error(self._library))

    def abort(self) -> None:
        if self._handle is not None:
            handle, self._handle = self._handle, None
            self._library.gpuwm_ncwrite_nc4_abort(handle)

    def __enter__(self) -> "Netcdf4Writer":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is not None:
            self.abort()
            return False
        self.finish()
        return False


class ClassicSchema:
    """Dimensions, attributes and variables, before the file exists."""

    def __init__(self, container: str = "cdf2") -> None:
        if container not in FORMATS:
            raise NcWriteError(
                f"unknown container {container!r}; expected one of "
                f"{sorted(FORMATS)}")
        self._library = load()
        handle = self._library.gpuwm_ncwrite_schema_new(FORMATS[container])
        if not handle:
            raise NcWriteError(_last_error(self._library))
        self._handle = handle
        self._var_types: dict[int, int] = {}
        self._container = container

    def _check(self, status: int) -> None:
        if status != 0:
            raise NcWriteError(_last_error(self._library))

    def _live(self):
        if self._handle is None:
            raise NcWriteError("this schema was already handed to create()")
        return self._handle

    def def_dim(self, name: str, length: int = 0, *,
                unlimited: bool = False) -> int:
        """Define a dimension; returns its dimid."""

        raw = name.encode("utf-8")
        dimid = int(self._library.gpuwm_ncwrite_def_dim(
            self._live(), raw, len(raw), int(length), 1 if unlimited else 0))
        if dimid < 0:
            raise NcWriteError(_last_error(self._library))
        return dimid

    def def_var(self, name: str, dtype, dimids=()) -> int:
        """Define a variable; returns its varid.

        ``dtype`` is anything ``numpy.dtype`` accepts, so a schema table
        can carry ``"f4"`` / ``"i4"`` / ``"S1"`` exactly as it does for
        netCDF4-python today.
        """

        nc_type = nc_type_of(dtype)
        raw = name.encode("utf-8")
        ids = list(int(value) for value in dimids)
        buffer = (ctypes.c_uint64 * max(len(ids), 1))(*ids) if ids else None
        varid = int(self._library.gpuwm_ncwrite_def_var(
            self._live(), raw, len(raw), nc_type, len(ids),
            ctypes.cast(buffer, ctypes.POINTER(ctypes.c_uint64))
            if buffer is not None else None))
        if varid < 0:
            raise NcWriteError(_last_error(self._library))
        self._var_types[varid] = nc_type
        return varid

    def _put_att(self, varid: int, name: str, value, dtype=None) -> None:
        raw = name.encode("utf-8")
        if isinstance(value, str):
            payload = value.encode("utf-8")
            nc_type, nelems = NC_CHAR, len(payload)
        elif isinstance(value, (bytes, bytearray)):
            payload = bytes(value)
            nc_type, nelems = NC_CHAR, len(payload)
        else:
            array = np.asarray(value if dtype is None else
                               np.asarray(value, dtype=dtype))
            array = np.atleast_1d(array)
            nc_type = nc_type_of(array.dtype)
            array = np.ascontiguousarray(
                array.astype(array.dtype.newbyteorder("="), copy=False))
            payload, nelems = array.tobytes(), int(array.size)
        self._check(self._library.gpuwm_ncwrite_put_att(
            self._live(), varid, raw, len(raw), nc_type, nelems, payload))

    def put_global_attr(self, name: str, value, dtype=None) -> None:
        self._put_att(-1, name, value, dtype)

    def put_var_attr(self, varid: int, name: str, value, dtype=None) -> None:
        self._put_att(int(varid), name, value, dtype)

    def create(self, path) -> ClassicWriter:
        """Create the file and hand back the writer. Consumes the schema."""

        raw = str(path).encode("utf-8")
        handle = self._live()
        self._handle = None  # the Rust side owns it from here
        writer = self._library.gpuwm_ncwrite_create(handle, raw, len(raw))
        if not writer:
            raise NcWriteError(_last_error(self._library))
        return ClassicWriter(self._library, writer, dict(self._var_types),
                             Path(str(path)))

    def create_netcdf4(self, path, *, deflate: int = 4, shuffle: bool = True,
                       threads: int = 0) -> Netcdf4Writer:
        """Create a compressed NetCDF-4 frame (lossless) from this schema.
        Consumes the schema."""

        reason = netcdf4_unavailable_reason()
        if reason is not None:
            raise NcWriteError(reason)
        raw = str(path).encode("utf-8")
        handle = self._live()
        self._handle = None
        writer = self._library.gpuwm_ncwrite_create_netcdf4(
            handle, raw, len(raw), int(deflate), 1 if shuffle else 0,
            int(threads))
        if not writer:
            raise NcWriteError(_last_error(self._library))
        return Netcdf4Writer(self._library, writer, dict(self._var_types),
                             Path(str(path)))

    def __del__(self) -> None:  # pragma: no cover - interpreter teardown
        handle, self._handle = getattr(self, "_handle", None), None
        if handle is not None:
            try:
                self._library.gpuwm_ncwrite_schema_free(handle)
            except Exception:
                pass


#: Whether :func:`_bind_patch` has bound the patch entry points of the
#: one library :func:`load` holds.
_PATCH_BOUND = False


def _bind_patch(library: ctypes.CDLL) -> None:
    """Bind the patch entry points, once, on first use.

    They are newer than :data:`ABI_MARKER` and no default path depends on
    them, so :func:`load` does not require them: an install whose library
    predates them still writes every wrfout and every wrfinput.  The
    caller that DOES need them is told which build to make.
    """

    global _PATCH_BOUND
    if _PATCH_BOUND:
        return
    void = ctypes.c_void_p
    size = ctypes.c_size_t
    try:
        library.gpuwm_ncwrite_patch_open.argtypes = [
            ctypes.c_char_p, size, ctypes.c_char_p, size]
        library.gpuwm_ncwrite_patch_open.restype = void
        library.gpuwm_ncwrite_patch_describe.argtypes = [void]
        library.gpuwm_ncwrite_patch_describe.restype = ctypes.c_int32
        library.gpuwm_ncwrite_patch_put.argtypes = [
            void, ctypes.c_char_p, size, ctypes.c_int64, ctypes.c_uint32,
            ctypes.c_char_p, size]
        library.gpuwm_ncwrite_patch_put.restype = ctypes.c_int32
        library.gpuwm_ncwrite_patch_finish.argtypes = [void]
        library.gpuwm_ncwrite_patch_finish.restype = ctypes.c_int32
        library.gpuwm_ncwrite_patch_abort.argtypes = [void]
        library.gpuwm_ncwrite_patch_abort.restype = None
    except AttributeError:
        raise NcWriteError(
            "the loaded netcdf-writer library exports no "
            "gpuwm_ncwrite_patch_open, so it predates the in-place patch "
            "of an existing classic file: it can write a new file but "
            "cannot hand an analysis file back with its layout intact.  "
            "Rebuild tools/rustwx (cargo build --release --locked "
            "--offline -p netcdf-writer), or run `woof fetch-bridges` on "
            "an installed wheel") from None
    _PATCH_BOUND = True


class PatchVariable:
    """One variable of a patch template, as its header declares it."""

    __slots__ = ("name", "nc_type", "is_record", "elements")

    def __init__(self, name: str, nc_type: int, is_record: bool,
                 elements: int) -> None:
        self.name = name
        self.nc_type = nc_type
        self.is_record = is_record
        #: Elements in the whole variable (fixed) or in one record slab.
        self.elements = elements

    @property
    def dtype(self) -> np.dtype:
        """The numpy dtype the stored type corresponds to."""

        return np.dtype(_NC_TO_NUMPY[self.nc_type])

    def __repr__(self) -> str:
        return (f"PatchVariable({self.name!r}, {self.dtype.str}, "
                f"record={self.is_record}, elements={self.elements})")


class ClassicPatch:
    """A copy of an existing classic file, open for rewriting variables.

    The copy is written beside ``target`` under a partial name and appears
    at ``target`` only when :meth:`finish` succeeds.  Every byte a
    :meth:`put` does not rewrite is the template's own, so the header, the
    data-section alignment the producing library chose and every variable
    the caller never names come back unchanged.  A variable put back with
    the values it was read with leaves the file byte-identical.

    A patch changes values only.  It cannot add, remove or retype a
    variable or add a record; the Rust side refuses each by name.
    """

    def __init__(self, template, target) -> None:
        self._library = load()
        _bind_patch(self._library)
        self._template = Path(os.fspath(template))
        self._target = Path(os.fspath(target))
        raw_template = os.fspath(self._template).encode("utf-8")
        raw_target = os.fspath(self._target).encode("utf-8")
        handle = self._library.gpuwm_ncwrite_patch_open(
            raw_template, len(raw_template), raw_target, len(raw_target))
        if not handle:
            raise NcWriteError(_last_error(self._library))
        self._handle = handle
        try:
            self._num_records, self._variables = self._describe()
        except BaseException:
            self.abort()
            raise

    def _describe(self) -> tuple[int, dict[str, PatchVariable]]:
        if self._library.gpuwm_ncwrite_patch_describe(self._handle) != 0:
            raise NcWriteError(_last_error(self._library))
        needed = int(self._library.gpuwm_ncwrite_last_scan(None, 0))
        buffer = (ctypes.c_uint8 * max(needed, 1))()
        self._library.gpuwm_ncwrite_last_scan(
            ctypes.cast(buffer, ctypes.POINTER(ctypes.c_uint8)), needed)
        lines = bytes(buffer)[:needed].decode("utf-8").split("\n")
        head = lines[0].split("\t")
        if len(head) != 2 or head[0] != "numrecs":
            raise NcWriteError(
                f"the patch description does not start with the record "
                f"count: {lines[0]!r}")
        variables: dict[str, PatchVariable] = {}
        for line in lines[1:]:
            name, code, record, elements = line.rsplit("\t", 3)
            variables[name] = PatchVariable(
                name, int(code), record == "1", int(elements))
        return int(head[1]), variables

    @property
    def target(self) -> Path:
        return self._target

    @property
    def num_records(self) -> int:
        """Records the template holds; a patch cannot change this."""

        return self._num_records

    @property
    def variables(self) -> dict[str, PatchVariable]:
        """Every template variable, by name, in definition order."""

        return dict(self._variables)

    def _live(self):
        if self._handle is None:
            raise NcWriteError(f"the patch of {self._target} is already closed")
        return self._handle

    def put(self, name: str, values, *, record: int | None = None) -> None:
        """Rewrite one variable, or one record's slab of a record variable.

        ``values`` in the variable's stored type are written as they are.
        Any other numeric dtype crosses as float64 and the Rust side
        narrows it to the stored type only when every value survives the
        narrowing exactly; a value that would be rounded is refused by
        name.  That is the shape a reader that widens everything to
        float64 hands back, and it is why a round trip through such a
        reader is still byte-identical.
        """

        variable = self._variables.get(name)
        if variable is None:
            raise NcWriteError(
                f"'{name}' is not a variable of {self._template}; a patch "
                "rewrites existing variables and cannot add one")
        if variable.is_record and record is None:
            raise NcWriteError(
                f"'{name}' is a record variable; say which record to rewrite")
        if not variable.is_record and record is not None:
            raise NcWriteError(
                f"'{name}' is a fixed variable, so record {record} names "
                "nothing")
        stored = variable.dtype
        if variable.nc_type == NC_CHAR:
            payload = _payload(values, NC_CHAR, f"variable {name}")
            nc_type = NC_CHAR
        else:
            array = np.asarray(values)
            if array.dtype.kind not in "iuf":
                raise NcWriteError(
                    f"'{name}' is stored as {stored.str}; a {array.dtype.str} "
                    "payload is not numeric")
            if (array.dtype.kind == stored.kind
                    and array.dtype.itemsize == stored.itemsize):
                nc_type = variable.nc_type
            else:
                widened = np.asarray(array, dtype=np.float64)
                if array.dtype.kind in "iu" and not np.array_equal(
                        widened.astype(array.dtype), array):
                    raise NcWriteError(
                        f"'{name}': a {array.dtype.str} payload holds values "
                        "float64 cannot carry exactly, so it cannot cross "
                        f"to the stored {stored.str} without rounding")
                array, nc_type = widened, NC_DOUBLE
            payload = _payload(array, nc_type, f"variable {name}")
        raw = name.encode("utf-8")
        status = self._library.gpuwm_ncwrite_patch_put(
            self._live(), raw, len(raw), -1 if record is None else int(record),
            nc_type, payload, len(payload))
        if status != 0:
            raise NcWriteError(_last_error(self._library))

    def finish(self) -> Path:
        """Flush, fsync and rename the patched copy onto the target."""

        handle = self._live()
        self._handle = None
        if int(self._library.gpuwm_ncwrite_patch_finish(handle)) != 0:
            raise NcWriteError(_last_error(self._library))
        return self._target

    def abort(self) -> None:
        """Drop the patched copy. Nothing appears at the target path."""

        if self._handle is not None:
            handle, self._handle = self._handle, None
            self._library.gpuwm_ncwrite_patch_abort(handle)

    def __enter__(self) -> "ClassicPatch":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is not None:
            self.abort()
            return False
        self.finish()
        return False

    def __del__(self) -> None:  # pragma: no cover - interpreter teardown
        try:
            self.abort()
        except Exception:
            pass


__all__ = [
    "ABI_MARKER",
    "ClassicPatch",
    "ClassicSchema",
    "ClassicWriter",
    "FORMATS",
    "NCWRITE_ABI",
    "NCWRITE_BRIDGE_ENV",
    "NcWriteError",
    "PatchVariable",
    "checkout_build_command",
    "library_candidates",
    "load",
    "nc_type_of",
    "resolve_ncwrite_bridge",
    "scan_nonfinite",
    "unavailable_reason",
]
