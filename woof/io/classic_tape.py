"""A ``netCDF4.Dataset``-shaped facade over the Rust classic writer.

Two products write NetCDF from Python-shaped code: the wrfout history
tape (:mod:`woof.io.wrfout`) and the ``wrfinput``/``wrfbdy`` export
(:mod:`woof.wrf_direct`).  Both were written against netCDF4-python's
surface -- ``createDimension`` / ``createVariable`` / ``setncatts`` /
``variable[t] = array`` -- and both now put their bytes on disk through
:mod:`woof.io.nc_writer_bridge`.  This is the ONE facade that maps that
surface onto the seam, so the two products cannot drift into two
different ideas of what "the classic writer" does.

What the classic container carries, and what it does not
--------------------------------------------------------
Everything that describes the DATA survives: dimensions (including the
record dimension), variable names, types, dimension order, definition
order, global and per-variable attributes, and every value bit for bit.

What has no classic spelling at all is HDF5 storage tuning -- ``zlib``,
``shuffle``, ``complevel``, ``fletcher32``, ``chunksizes`` -- and the
``endian`` of the stored representation, because the classic format is
big-endian by definition.  None of those change a value; a caller
porting a netCDF4 ``createVariable`` call drops them rather than passing
them, and this facade refuses them by name rather than accepting and
ignoring them.  Silently swallowing ``zlib=True`` is how a caller comes
to believe a file is compressed when it is not.

The header freezes at the first data byte
-----------------------------------------
The classic formats put the entire header -- dimensions, attributes,
variables -- ahead of the data section, so it is IMMUTABLE from the
first data byte on.  This facade therefore collects the schema exactly as
the caller declares it and freezes it into a
:class:`woof.io.nc_writer_bridge.ClassicSchema` at the FIRST data write.
Schema mutation after that point is refused with the breakage named,
never absorbed: absorbing it would mean rewriting the file underneath a
caller who thinks it is appending.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

#: The container name for a lossless, compressed NetCDF-4 frame written
#: through the same schema (see :meth:`ClassicTape._freeze`).
NETCDF4_DEFLATE = "netcdf4-deflate"

#: ``createVariable`` keywords that describe HDF5 storage rather than
#: data.  Named here so the refusal can say which one was passed.
_HDF5_ONLY_KWARGS = ("zlib", "shuffle", "complevel", "fletcher32",
                     "chunksizes", "endian", "fill_value")


def classic_attr_value(value):
    """An attribute value as the classic container stores it.

    netCDF4-python in a classic-model file stores a bare Python ``int``
    as NC_INT (int32) and a bare ``float`` as NC_DOUBLE (measured on
    this repository's own files); the Rust seam takes numpy's word for
    the type, and numpy 2 spells a bare int ``int64`` -- which classic
    CDF-1/CDF-2 cannot hold at all.  Coercing here keeps the two engines'
    files attribute-identical and keeps a plain ``int`` writable.
    Numpy-typed values pass through untouched: a caller who chose a
    width keeps it.
    """
    if isinstance(value, (str, bytes, bytearray, np.generic, np.ndarray)):
        return value
    if isinstance(value, bool):
        return np.int32(int(value))
    if isinstance(value, int):
        return np.int32(value)          # OverflowError names the value
    if isinstance(value, float):
        return np.float64(value)
    return value


class ClassicDim:
    """One dimension of a :class:`ClassicTape`."""

    __slots__ = ("name", "length", "unlimited", "_tape")

    def __init__(self, tape, name, length, unlimited):
        self._tape = tape
        self.name = name
        self.length = int(length)
        self.unlimited = bool(unlimited)

    def __len__(self):
        if self.unlimited:
            return self._tape._numrecs
        return self.length

    @property
    def size(self):
        return len(self)

    def isunlimited(self):
        return self.unlimited


class ClassicVariable:
    """One variable of a :class:`ClassicTape`.

    Mimics exactly the slice of ``netCDF4.Variable`` the two callers
    use: attribute assignment (``var.units = ...``), ``setncattr`` /
    ``setncatts``, ``group()``, and per-record data assignment
    (``var[t] = arr``, or the fully-sliced ``var[t, :, :] = arr`` spelling
    ``woof.wrf_direct`` writes).  Attribute writes are schema work and
    are refused once the header is on disk.
    """

    _INTERNAL = frozenset({"_tape", "_name", "_dtype", "_dims", "_attrs",
                           "_varid"})

    def __init__(self, tape, name, dtype, dims):
        object.__setattr__(self, "_tape", tape)
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_dtype", np.dtype(dtype))
        object.__setattr__(self, "_dims", tuple(dims))
        object.__setattr__(self, "_attrs", {})
        object.__setattr__(self, "_varid", None)

    @property
    def name(self):
        return self._name

    @property
    def dtype(self):
        return self._dtype

    @property
    def dimensions(self):
        return self._dims

    @property
    def shape(self):
        return tuple(len(self._tape.dimensions[name])
                     for name in self._dims)

    def group(self):
        """The dataset this variable belongs to, as ``netCDF4`` spells it.

        ``woof.wrf_direct`` resolves a variable's declared extents with
        ``len(variable.group().dimensions[name])``, so the facade has to
        answer it rather than make that caller reach around it.
        """
        return self._tape

    def setncattr(self, name, value):
        self._tape._refuse_frozen(f"attribute {name!r} of variable "
                                  f"{self._name!r}")
        self._attrs[name] = value

    def setncatts(self, attrs):
        for name, value in dict(attrs).items():
            self.setncattr(name, value)

    def ncattrs(self):
        return list(self._attrs)

    def getncattr(self, name):
        return self._attrs[name]

    def __setattr__(self, name, value):
        if name in self._INTERNAL:
            object.__setattr__(self, name, value)
        else:
            self.setncattr(name, value)

    def __getattr__(self, name):
        try:
            return object.__getattribute__(self, "_attrs")[name]
        except KeyError:
            raise AttributeError(name) from None

    def _recno(self, key) -> int:
        """The record index a subscript names, or a named refusal.

        ``var[t]`` and ``var[t, :, :]`` both mean "this whole record".
        Anything narrower -- a partial slab, a slice of records -- is
        refused rather than translated: the seam writes whole slabs, and
        a partial write would leave the rest of the slab as whatever the
        filesystem had there, which the classic format cannot signal.
        """
        if isinstance(key, tuple):
            if not key:
                raise IndexError(
                    f"{self._name}: an empty subscript names no record")
            recno, rest = key[0], key[1:]
            if any(item != slice(None) for item in rest):
                raise IndexError(
                    f"{self._name}: the classic writer writes whole record "
                    f"slabs, so {key!r} cannot be written -- a partial slab "
                    f"would leave the rest of the record as whatever was on "
                    f"the disk, and the format has no way to say so.  Assign "
                    f"the whole record instead.")
            if len(rest) != max(len(self._dims) - 1, 0):
                raise IndexError(
                    f"{self._name}: subscript {key!r} names "
                    f"{len(rest) + 1} of {len(self._dims)} dimension(s); "
                    f"assign the whole record.")
        else:
            recno = key
        if isinstance(recno, slice) or not isinstance(recno, (int, np.integer)):
            raise IndexError(
                f"{self._name}: {recno!r} is not a record index; the classic "
                f"writer takes one whole record at a time.")
        return int(recno)

    def __setitem__(self, key, values):
        self._tape._write_record(self._recno(key), self, values)


class ClassicTape:
    """A ``netCDF4.Dataset``-shaped facade over the Rust classic writer.

    ``completion_attr``, when given, is written INTO the header (last,
    where a netCDF4 path's close-time ``setncattr`` also lands it).  That
    cannot mark a partial file as finished: publication is temp-file +
    rename, the rename happens only after ``finish()`` -- which refuses a
    tape with any unwritten region -- and an aborted tape keeps
    ``numrecs`` 0, so a reader of a stray temp sees an empty inventory,
    not a half-frame presented as whole.

    ``frozen_remedy`` is appended to the header-frozen refusal: what THIS
    caller should do instead, which differs per product and must not be
    guessed at from inside the facade.
    """

    def __init__(self, path, *, container: str = "cdf2",
                 completion_attr: str | None = None,
                 frozen_remedy: str = "",
                 netcdf4_options: dict | None = None) -> None:
        self._path = Path(path)
        self._container = container
        # container NETCDF4_DEFLATE: the same schema and values written as a
        # lossless shuffle+deflate NetCDF-4 frame (woof.io.wrfout's history
        # default); options go to ClassicSchema.create_netcdf4.
        self._netcdf4_options = dict(netcdf4_options or {})
        self._completion_attr = completion_attr
        self._frozen_remedy = frozen_remedy
        self.dimensions: dict[str, ClassicDim] = {}
        self.variables: dict[str, ClassicVariable] = {}
        self._attrs: dict[str, object] = {}
        self._writer = None
        self._numrecs = 0
        self._closed = False

    @property
    def path(self) -> Path:
        return self._path

    def filepath(self) -> str:
        """The file being written, as ``netCDF4`` spells it."""
        return str(self._path)

    # ------------------------------------------------------------ schema
    def _refuse_frozen(self, what: str) -> None:
        if self._writer is not None:
            remedy = f"  {self._frozen_remedy}" if self._frozen_remedy else ""
            raise RuntimeError(
                f"cannot add {what}: the classic NetCDF header of "
                f"{self._path} was written to disk when the first record "
                "landed, and the classic formats place the whole header "
                "ahead of the data section, so adding it now would mean "
                f"rewriting the file.{remedy}")

    def createDimension(self, name, size=None):
        self._refuse_frozen(f"dimension {name!r}")
        dim = ClassicDim(self, name, 0 if size is None else int(size),
                         size is None)
        self.dimensions[name] = dim
        return dim

    def setncattr(self, name, value):
        self._refuse_frozen(f"global attribute {name!r}")
        self._attrs[name] = value

    def setncatts(self, attrs):
        for name, value in dict(attrs).items():
            self.setncattr(name, value)

    def ncattrs(self):
        return list(self._attrs)

    def getncattr(self, name):
        return self._attrs[name]

    def __getattr__(self, name):
        try:
            return object.__getattribute__(self, "_attrs")[name]
        except KeyError:
            raise AttributeError(name) from None

    def createVariable(self, name, dtype, dims=(), **kwargs):
        self._refuse_frozen(f"variable {name!r}")
        if kwargs:
            offered = sorted(kwargs)
            raise TypeError(
                f"variable {name!r} was declared with {offered}, which the "
                f"classic NetCDF container has no spelling for: "
                f"{list(_HDF5_ONLY_KWARGS)} describe HDF5 storage, not data. "
                f"They are refused rather than ignored, because a caller who "
                f"believes a file is compressed when it is not has been told "
                f"something untrue about it.")
        for dim in dims:
            if dim not in self.dimensions:
                raise ValueError(
                    f"variable {name!r} names undefined dimension {dim!r}")
        variable = ClassicVariable(self, name, dtype, dims)
        self.variables[name] = variable
        return variable

    # ------------------------------------------------------------ data
    def _freeze(self) -> None:
        from woof.io import nc_writer_bridge

        netcdf4 = self._container == NETCDF4_DEFLATE
        # The NetCDF-4 frame takes the same schema; CDF-5 is only the
        # schema's (most permissive) classic validation, never written.
        schema = nc_writer_bridge.ClassicSchema(
            "cdf5" if netcdf4 else self._container)
        dimids: dict[str, int] = {}
        for name, dim in self.dimensions.items():
            dimids[name] = schema.def_dim(
                name, dim.length, unlimited=dim.unlimited)
        for name, value in self._attrs.items():
            if name == self._completion_attr:
                continue                # written once, below, always last
            schema.put_global_attr(name, classic_attr_value(value))
        if self._completion_attr is not None:
            # Last, exactly where the netCDF4 path's close() puts it.
            schema.put_global_attr(self._completion_attr, np.int32(1))
            self._attrs[self._completion_attr] = np.int32(1)
        for name, variable in self.variables.items():
            varid = schema.def_var(
                name, variable._dtype,
                tuple(dimids[dim] for dim in variable._dims))
            for attr_name, attr_value in variable._attrs.items():
                schema.put_var_attr(varid, attr_name,
                                    classic_attr_value(attr_value))
            object.__setattr__(variable, "_varid", varid)
        if netcdf4:
            self._writer = schema.create_netcdf4(self._path,
                                                 **self._netcdf4_options)
        else:
            self._writer = schema.create(self._path)

    def _write_record(self, recno: int, variable: ClassicVariable,
                      values) -> None:
        if self._closed:
            raise RuntimeError(f"{self._path} is already closed")
        if self._writer is None:
            self._freeze()
        declared = variable._dtype
        if declared.kind == "S":
            payload = values
        else:
            payload = np.asarray(values)
            if payload.dtype != declared:
                # netCDF4 casts on assignment; the seam refuses to
                # reinterpret, so the cast the caller has always relied
                # on happens here, explicitly.
                payload = payload.astype(declared)
        self._writer.write_record(recno, variable._varid, payload)
        self._numrecs = max(self._numrecs, recno + 1)

    # ------------------------------------------------------------ lifecycle
    def close(self) -> None:
        """Freeze if never written, then finish: numrecs, flush, fsync."""
        if self._closed:
            return
        if self._writer is None:
            self._freeze()
        writer, self._writer = self._writer, None
        self._closed = True
        try:
            writer.finish()
        except Exception:
            writer.abort()
            raise

    def abort(self) -> None:
        """Release without finishing; numrecs stays 0 on the partial file."""
        if self._closed:
            return
        self._closed = True
        if self._writer is not None:
            writer, self._writer = self._writer, None
            writer.abort()

    def __enter__(self) -> "ClassicTape":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is not None:
            self.abort()
            return False
        self.close()
        return False


__all__ = ["ClassicDim", "ClassicTape", "ClassicVariable",
           "NETCDF4_DEFLATE", "classic_attr_value"]
