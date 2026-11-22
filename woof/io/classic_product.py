"""A whole-file classic NetCDF product, written through WOOF-ArWen's Rust.

:mod:`woof.io.wrfout` drives :mod:`woof.io.nc_writer_bridge` as a *tape*:
one unlimited record dimension, frames appended over time, the header frozen
the moment the first frame lands.  The gridded observation products are the
other shape entirely -- no record dimension, every variable fixed, the whole
file produced in one call -- and they declare their schema INTERLEAVED with
their data (``createVariable`` immediately followed by ``variable[:] = ...``),
which the tape's freeze-on-first-write rule cannot express.

This module is that second shape.  It collects dimensions, attributes,
variables and their values, and emits them at :meth:`ClassicProduct.close` in
declaration order: header first, as the classic containers require, then every
variable's slab.  The surface is the slice of ``netCDF4.Dataset`` the product
writers already use, so a writer ported onto it keeps its own shape.

What it does NOT buffer is a copy.  ``variable[:] = array`` stores a
REFERENCE; the cast to the declared external type happens one variable at a
time inside :meth:`close`.  A radial-velocity volume is hundreds of megabytes
and there is no reason for a facade to hold a second copy of every one of them
at once.

Holes are impossible by construction: the Rust writer refuses ``finish`` with
any fixed variable never written, and the classic container has no in-band way
to say "this region was never filled" -- an unwritten slab would read back as
whatever the filesystem left there.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from woof.io.classic_tape import classic_attr_value as _classic_attr_value

#: The container these products are written in.  CDF-5 rather than the
#: wrfout tape's CDF-2 because a CDF-2 fixed variable is capped at 4 GiB
#: unless it is the last one, and a multi-radar velocity volume over a
#: wide domain is exactly the array that reaches that cap.  Nothing reads
#: these files but woof itself (``rw_netcdf`` and, for the character
#: variables, netCDF4), and both read CDF-5; the WRF-interop containers,
#: where io_form pins the version, are a different product.
DEFAULT_CONTAINER = "cdf5"

#: ``createVariable`` keywords that describe HDF5 STORAGE rather than data.
#: The classic container has none of them, and accepting one silently would
#: tell a caller their file is compressed or checksummed when it is not.
_HDF5_ONLY_KEYWORDS = ("zlib", "complevel", "shuffle", "fletcher32",
                       "contiguous", "chunksizes", "endian",
                       "least_significant_digit")


class ClassicProductError(RuntimeError):
    """The product could not be written as declared."""


class ProductVariable:
    """One fixed variable: attributes now, values on assignment."""

    _INTERNAL = frozenset({"_product", "_name", "_dtype", "_dims", "_attrs",
                           "_values"})

    def __init__(self, product: "ClassicProduct", name: str, dtype,
                 dims: tuple[str, ...]) -> None:
        object.__setattr__(self, "_product", product)
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_dtype", np.dtype(dtype))
        object.__setattr__(self, "_dims", tuple(dims))
        object.__setattr__(self, "_attrs", {})
        object.__setattr__(self, "_values", None)

    @property
    def name(self) -> str:
        return self._name

    @property
    def dtype(self):
        return self._dtype

    @property
    def dimensions(self) -> tuple[str, ...]:
        return self._dims

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self._product.dimensions[name] for name in self._dims)

    def setncattr(self, name: str, value) -> None:
        self._product._refuse_written(
            f"attribute {name!r} of variable {self._name!r}")
        self._attrs[name] = value

    def __setattr__(self, name: str, value) -> None:
        if name in self._INTERNAL:
            object.__setattr__(self, name, value)
        else:
            self.setncattr(name, value)

    def __getattr__(self, name: str):
        try:
            return object.__getattribute__(self, "_attrs")[name]
        except KeyError:
            raise AttributeError(name) from None

    def ncattrs(self) -> list[str]:
        return list(self._attrs)

    def __setitem__(self, selector, values) -> None:
        # ``variable[:] = array`` and ``variable[...] = array`` are the two
        # spellings the product writers use.  A PARTIAL assignment is
        # refused rather than absorbed: this facade writes each variable
        # exactly once, so a caller filling a region at a time would be
        # silently keeping only the last region.
        if not (selector is Ellipsis
                or (isinstance(selector, slice)
                    and selector == slice(None))):
            raise ClassicProductError(
                f"{self._name}: a classic product variable is written whole "
                f"(`variable[:] = array`); {selector!r} would leave the rest "
                f"of the variable unwritten, and the classic container has "
                f"no way to say a region was never filled")
        self._product._refuse_written(f"values of variable {self._name!r}")
        if self._values is not None:
            raise ClassicProductError(
                f"{self._name} was already assigned; the classic product "
                f"writes each variable once")
        object.__setattr__(self, "_values", values)

    def _payload(self):
        if self._values is None:
            raise ClassicProductError(
                f"{self._name} was declared but never assigned; the Rust "
                f"writer refuses a file with an unwritten region")
        if self._dtype.kind == "S":
            return self._values
        payload = np.asarray(self._values)
        if payload.dtype != self._dtype:
            # netCDF4 casts on assignment and the seam refuses to
            # reinterpret, so the cast a caller has always relied on
            # happens here, once, at write time.
            payload = payload.astype(self._dtype)
        return payload


class ClassicProduct:
    """A ``netCDF4.Dataset``-shaped facade for a one-shot classic file."""

    def __init__(self, path: Path | str, *,
                 container: str = DEFAULT_CONTAINER) -> None:
        self._path = Path(path)
        self._container = container
        self._order: list[str] = []
        self.dimensions: dict[str, int] = {}
        self.variables: dict[str, ProductVariable] = {}
        self._attrs: dict[str, object] = {}
        self._written = False

    # ------------------------------------------------------------ schema
    def _refuse_written(self, what: str) -> None:
        if self._written:
            raise ClassicProductError(
                f"cannot set {what}: {self._path} has already been written")

    def createDimension(self, name: str, size: int) -> int:
        self._refuse_written(f"dimension {name!r}")
        if size is None:
            raise ClassicProductError(
                f"dimension {name!r} has no length: a classic PRODUCT has no "
                f"record dimension -- use woof.io.wrfout's tape for that")
        self.dimensions[name] = int(size)
        return int(size)

    def setncattr(self, name: str, value) -> None:
        self._refuse_written(f"global attribute {name!r}")
        self._attrs[name] = value

    def setncatts(self, attrs: Mapping[str, object]) -> None:
        for name, value in dict(attrs).items():
            self.setncattr(name, value)

    def ncattrs(self) -> list[str]:
        return list(self._attrs)

    def getncattr(self, name: str):
        return self._attrs[name]

    def createVariable(self, name: str, dtype, dims: Iterable[str] = (), *,
                       fill_value=None, **unsupported) -> ProductVariable:
        self._refuse_written(f"variable {name!r}")
        offered = sorted(set(unsupported) & set(_HDF5_ONLY_KEYWORDS))
        if offered:
            raise ClassicProductError(
                f"variable {name!r} asks for {offered}, which is HDF5 storage "
                f"tuning; the classic container has none of it, and accepting "
                f"the keyword would tell you this file is stored a way it is "
                f"not")
        if unsupported:
            raise ClassicProductError(
                f"variable {name!r}: unknown createVariable keyword(s) "
                f"{sorted(unsupported)}")
        dims = tuple(str(dim) for dim in dims)
        for dim in dims:
            if dim not in self.dimensions:
                raise ClassicProductError(
                    f"variable {name!r} names undefined dimension {dim!r}")
        variable = ProductVariable(self, name, dtype, dims)
        if fill_value is not None:
            # netCDF4 turns `fill_value=` into the `_FillValue` ATTRIBUTE,
            # declared first, and pre-fills unwritten elements with it.
            # Only the attribute is reproduced here: this writer refuses a
            # file with an unwritten element, so there is nothing to
            # pre-fill and no way for the two to disagree.
            variable.setncattr("_FillValue",
                               np.asarray(fill_value,
                                          dtype=np.dtype(dtype)).reshape(())[
                                              ()])
        self.variables[name] = variable
        self._order.append(name)
        return variable

    # ------------------------------------------------------------ output
    def close(self) -> None:
        """Freeze the header, write every variable, fsync, release."""

        if self._written:
            return
        from woof.io import nc_writer_bridge      # noqa: PLC0415

        schema = nc_writer_bridge.ClassicSchema(self._container)
        dimids = {name: schema.def_dim(name, size)
                  for name, size in self.dimensions.items()}
        for name, value in self._attrs.items():
            schema.put_global_attr(name, _classic_attr_value(value))
        varids: dict[str, int] = {}
        for name in self._order:
            variable = self.variables[name]
            varid = schema.def_var(
                name, variable._dtype,
                tuple(dimids[dim] for dim in variable._dims))
            for attr_name, attr_value in variable._attrs.items():
                schema.put_var_attr(varid, attr_name,
                                    _classic_attr_value(attr_value))
            varids[name] = varid
        # Every payload is resolved BEFORE the file is created, so a
        # variable that was declared and never assigned refuses without
        # leaving a partial file on disk.
        payloads = [(varids[name], self.variables[name]._payload())
                    for name in self._order]
        writer = schema.create(self._path)
        try:
            for varid, payload in payloads:
                writer.write_var(varid, payload)
        except Exception:
            writer.abort()
            raise
        writer.finish()
        self._written = True

    def abort(self) -> None:
        """Discard the declaration; nothing has been written yet."""

        self._written = True

    def __enter__(self) -> "ClassicProduct":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is not None:
            self.abort()
            return False
        self.close()
        return False


__all__ = [
    "ClassicProduct",
    "ClassicProductError",
    "DEFAULT_CONTAINER",
    "ProductVariable",
]
