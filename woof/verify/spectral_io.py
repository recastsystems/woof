"""Fail-closed array and WRF-history readers for spectral verification.

The NetCDF decoder is reached only when a NetCDF source is opened, so the pure
NumPy synthetic controls and receipt validators remain CPU-only and importable
in a minimal test environment.  Every returned array is float64 and finite;
masked or missing values are refusals, not silently filled samples.

Gridded field VALUES come out of :mod:`woof.netcdf_bridge` -- ``rw_netcdf``,
WOOF-ArWen's Rust decoder -- and never out of a Python NetCDF library.  Pulling a
plane out of a forecast history file is decode work whoever wrote the file,
and a second decoder on this door would mean a spectral receipt could disagree
with the renderer and the field verifier about what the model wrote, with
nothing in the self-hashed receipt saying which library produced the numbers.
There is no fallback: a missing bridge is the bridge's own named refusal.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

import numpy as np

SUPPORTED_SUFFIXES = (".npy", ".npz", ".nc", ".nc4", ".cdf")
NETCDF_SUFFIXES = (".nc", ".nc4", ".cdf")

#: Leading bytes that identify a NetCDF container: classic, 64-bit offset,
#: CDF-5, and the HDF5 superblock that NetCDF-4 writes.  A forecast history
#: file is named ``wrfout_d0N_<valid time>`` and carries NO suffix at all, so
#: the name cannot decide the format and the bytes have to.
NETCDF_SIGNATURES = (b"CDF\x01", b"CDF\x02", b"CDF\x05", b"\x89HDF\r\n\x1a\n")

REDUCTIONS = ("plane", "level", "surface", "column_max",
              "column_max_abs", "column_mean")
DESTAGGER = ("none", "auto", "x", "y", "z")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: str | Path) -> dict[str, object]:
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"spectral source is not a readable file: {source}")
    stat = source.stat()
    return {
        "path": str(source.resolve()),
        "size_bytes": int(stat.st_size),
        "sha256": sha256_file(source),
    }


#: The ceiling on non-finite cells in a plane about to be transformed, and
#: the reason it is zero.  This is not conservatism: a two-dimensional FFT is
#: a dense sum over every cell, so ONE NaN reaches every retained Fourier
#: mode.  Every band power, every correlation, every gate on the whole plane
#: becomes NaN, and a receipt would then grade the entire field on the two
#: cells that were bad.  There is no fraction of the plane at which that
#: stops being true, so there is no fraction to tune.
NONFINITE_CELL_CEILING = 0

#: Cells listed by coordinate in the refusal before it says "and N more".
#: Enough to tell an edge artifact from a scattered fill value, short enough
#: that a whole-plane failure does not print a million tuples.
_REPORTED_CELLS = 6


def _locate(flags: np.ndarray) -> str:
    """Coordinates of the first offending cells, capped."""

    found = np.argwhere(flags)
    shown = ", ".join(
        "(" + ", ".join(str(int(axis)) for axis in cell) + ")"
        for cell in found[:_REPORTED_CELLS])
    if len(found) > _REPORTED_CELLS:
        shown += f", and {len(found) - _REPORTED_CELLS} more"
    return shown


def _finite_array(value: object, *, label: str) -> np.ndarray:
    masked = np.ma.asarray(value)
    if np.ma.isMaskedArray(masked) and np.any(np.ma.getmaskarray(masked)):
        flags = np.ma.getmaskarray(masked)
        count = int(np.count_nonzero(flags))
        raise ValueError(
            f"{label} carries {count} of {flags.size} cells masked or set to "
            f"a missing value, at {_locate(flags)}. A spectral score is a "
            "two-dimensional Fourier transform, so a single missing cell "
            "reaches every retained Fourier mode and every band power, "
            "correlation and gate on the whole plane becomes NaN -- the "
            "ceiling on missing cells is "
            f"{NONFINITE_CELL_CEILING} and cannot be raised. Either score a "
            "variable that is filled everywhere on this grid, or set "
            "crop_cells in the registration so the scored window excludes "
            "the masked frame, or fill the gap in a separately receipted "
            "product first -- this reader will not invent values.")
    array = np.asarray(masked, dtype=np.float64)
    if array.size == 0:
        raise ValueError(
            f"{label} carries no cells: its shape is {list(array.shape)}. "
            "A spectral score needs a plane at least 8x8 after cropping and "
            "reduction. Check the variable name, the time index, and the "
            "level index in the registration; a zero-length axis usually "
            "means one of them selected nothing.")
    nonfinite = ~np.isfinite(array)
    if np.any(nonfinite):
        count = int(np.count_nonzero(nonfinite))
        nan_count = int(np.count_nonzero(np.isnan(array)))
        raise ValueError(
            f"{label} carries {count} of {array.size} non-finite cells "
            f"(nan={nan_count}, inf={count - nan_count}), at "
            f"{_locate(nonfinite)}. A spectral score is a two-dimensional "
            "Fourier transform, so one non-finite cell reaches every "
            "retained Fourier mode and every band power, correlation and "
            "gate on the whole plane becomes NaN -- the ceiling on "
            f"non-finite cells is {NONFINITE_CELL_CEILING} and cannot be "
            "raised. Either score a variable that is finite everywhere on "
            "this grid, or set crop_cells in the registration so the scored "
            "window excludes the affected frame, or fix the field in the "
            "producing run -- this reader will not substitute a fill value "
            "and score it as data.")
    return array


def _netcdf_signature(path: Path) -> bytes | None:
    """Return the NetCDF/HDF5 signature this file starts with, or ``None``."""

    with path.open("rb") as stream:
        head = stream.read(max(len(item) for item in NETCDF_SIGNATURES))
    for signature in NETCDF_SIGNATURES:
        if head.startswith(signature):
            return signature
    return None


def _time_dimension(name: str) -> bool:
    normalized = str(name).strip().lower()
    return normalized in {"time", "times", "datetime", "date"} or (
        normalized.startswith("time_") or normalized.endswith("_time"))


def load_array(path: str | Path, *, variable: str | None = None,
               time_index: int = 0) -> tuple[np.ndarray, dict[str, object]]:
    """Load one source array and dimension metadata.

    ``.npy`` files contain one unnamed array.  ``.npz`` files require a key
    when they contain more than one array.  NetCDF readers remove a leading
    time dimension only when its dimension name explicitly identifies time.
    """

    source = Path(path)
    if not source.is_file():
        raise ValueError(f"spectral source is not a readable file: {source}")
    if int(time_index) != time_index or time_index < 0:
        raise ValueError("time_index must be a non-negative integer")
    time_index = int(time_index)
    suffix = source.suffix.lower()
    metadata: dict[str, object] = {
        "path": str(source.resolve()),
        "variable": variable,
        "time_index": time_index,
    }

    if suffix == ".npy":
        if time_index != 0:
            raise ValueError(f"{source} has no named time dimension")
        if variable not in (None, ""):
            raise ValueError("a .npy spectral source has no named variable")
        value = np.load(source, allow_pickle=False)
        metadata["format"] = "npy"
        metadata["dimensions"] = []
        return _finite_array(value, label=str(source)), metadata

    if suffix == ".npz":
        if time_index != 0:
            raise ValueError(f"{source} has no named time dimension")
        with np.load(source, allow_pickle=False) as archive:
            keys = tuple(archive.files)
            if variable is None:
                if len(keys) != 1:
                    raise ValueError(
                        f"{source} contains {list(keys)}; name one variable")
                key = keys[0]
            else:
                key = str(variable)
            if key not in archive:
                raise ValueError(f"{source} is missing array {key!r}")
            value = np.array(archive[key], copy=True)
        metadata["format"] = "npz"
        metadata["variable"] = key
        metadata["dimensions"] = []
        return _finite_array(value, label=f"{source}:{key}"), metadata

    # A model writes ``wrfout_d0N_<valid time>`` with no suffix, so the name
    # alone cannot admit or refuse it.  A declared NetCDF suffix is taken at
    # its word; anything else has to show a NetCDF/HDF5 signature, which keeps
    # GRIB messages, logs and truncated transfers out of the decoder.
    identified_by = "suffix"
    if suffix not in NETCDF_SUFFIXES:
        if _netcdf_signature(source) is None:
            raise ValueError(
                f"spectral source {source} carries no NetCDF or HDF5 "
                f"signature and its suffix {suffix!r} is not one of "
                f"{SUPPORTED_SUFFIXES}")
        identified_by = "signature"
    if not variable:
        raise ValueError("a NetCDF spectral source requires a variable name")
    # Reached only on the NetCDF path, and only ever this one: the Rust
    # decoder.  ``rw_netcdf`` missing is :class:`NetcdfBridgeMissing`, which
    # carries the exact command to build it -- deliberately not a fallback to
    # a Python NetCDF library.
    from woof import netcdf_bridge  # noqa: PLC0415

    with netcdf_bridge.open_dataset(source) as dataset:
        if variable not in dataset.variables:
            raise ValueError(f"{source} is missing variable {variable!r}")
        netcdf_variable = dataset.variables[str(variable)]
        if netcdf_variable.is_character:
            # Named here rather than left to the decoder, because the answer
            # is about the REQUEST: a spectral score is a Fourier transform
            # of a physical field, and ``Times`` is not one.
            raise ValueError(
                f"{source}:{variable} is a "
                f"{netcdf_variable.stored_dtype} variable; a spectral source "
                f"must be a numeric field")
        dimensions = list(netcdf_variable.dimensions)
        selector: tuple[object, ...]
        if dimensions and _time_dimension(dimensions[0]):
            if time_index >= netcdf_variable.shape[0]:
                raise ValueError(
                    f"time_index {time_index} is outside {source}:{variable}")
            selector = (time_index,) + (slice(None),) * (netcdf_variable.ndim - 1)
            dimensions = dimensions[1:]
        else:
            if time_index != 0:
                raise ValueError(
                    f"{source}:{variable} has no leading time dimension")
            selector = (slice(None),) * netcdf_variable.ndim
        value = netcdf_variable[selector]
        units = getattr(netcdf_variable, "units", None)
        stagger = getattr(netcdf_variable, "stagger", None)
    metadata.update({
        "format": "netcdf",
        "netcdf_identified_by": identified_by,
        "dimensions": dimensions,
        "units": None if units is None else str(units),
        "stagger_attribute": None if stagger is None else str(stagger),
    })
    return _finite_array(value, label=f"{source}:{variable}"), metadata


def destagger(array: np.ndarray, axis: int) -> np.ndarray:
    values = _finite_array(array, label="destagger input")
    normalized = axis if axis >= 0 else values.ndim + axis
    if normalized < 0 or normalized >= values.ndim:
        raise ValueError("destagger axis is outside the array")
    if values.shape[normalized] < 2:
        raise ValueError("destagger axis needs at least two points")
    lower = [slice(None)] * values.ndim
    upper = [slice(None)] * values.ndim
    lower[normalized] = slice(None, -1)
    upper[normalized] = slice(1, None)
    return 0.5 * (values[tuple(lower)] + values[tuple(upper)])


def apply_destagger(array: np.ndarray, dimensions: list[str] | tuple[str, ...],
                    request: str) -> tuple[np.ndarray, list[str]]:
    """Apply one requested stagger removal and return updated dimensions."""

    mode = str(request).lower()
    if mode not in DESTAGGER:
        raise ValueError(f"destagger must be one of {DESTAGGER}")
    values = _finite_array(array, label="destagger input")
    dims = [str(item) for item in dimensions]
    if dims and len(dims) != values.ndim:
        raise ValueError("dimension metadata does not match the source rank")
    if mode == "none":
        return values, dims

    axis: int | None = None
    if mode == "x":
        axis = values.ndim - 1
    elif mode == "y":
        axis = values.ndim - 2
    elif mode == "z":
        axis = values.ndim - 3
    elif mode == "auto":
        candidates = [index for index, name in enumerate(dims)
                      if name.lower().endswith("_stag")]
        if len(candidates) > 1:
            raise ValueError(
                f"auto destagger is ambiguous across dimensions {dims}")
        if candidates:
            axis = candidates[0]
        else:
            return values, dims
    assert axis is not None
    if axis < 0:
        raise ValueError(f"cannot {mode}-destagger a rank-{values.ndim} array")
    result = destagger(values, axis)
    if dims:
        dims[axis] = dims[axis].removesuffix("_stag")
    return result, dims


def reduce_to_plane(array: np.ndarray, *, reduction: str,
                    level_index: int | None = None) -> np.ndarray:
    """Reduce a finite 2-D/3-D field to one finite horizontal plane."""

    values = _finite_array(array, label="field reduction input")
    mode = str(reduction).lower()
    if mode not in REDUCTIONS:
        raise ValueError(f"reduction must be one of {REDUCTIONS}")
    if values.ndim == 2:
        if mode in ("plane", "column_max", "column_max_abs", "column_mean"):
            return values
        raise ValueError(f"reduction {mode!r} needs a vertical dimension")
    if values.ndim != 3:
        raise ValueError(
            "after time selection/destagger, a spectral field must be 2-D or 3-D")

    if mode == "plane":
        raise ValueError("reduction 'plane' cannot consume a 3-D field")
    if mode in ("level", "surface"):
        index = 0 if mode == "surface" else level_index
        if index is None or int(index) != index:
            raise ValueError("level reduction needs an integer level_index")
        index = int(index)
        if index < 0:
            index += values.shape[0]
        if index < 0 or index >= values.shape[0]:
            raise ValueError(
                f"level_index is outside a {values.shape[0]}-level field")
        return np.asarray(values[index], dtype=np.float64)
    if mode == "column_max":
        return np.max(values, axis=0)
    if mode == "column_max_abs":
        return np.max(np.abs(values), axis=0)
    if mode == "column_mean":
        return np.mean(values, axis=0, dtype=np.float64)
    raise AssertionError(mode)


def load_plane(source: Mapping[str, object]) -> tuple[np.ndarray, dict[str, object]]:
    """Load, optionally destagger, and reduce one declarative source."""

    path = source.get("path")
    if path is None:
        raise ValueError("spectral source is missing path")
    array, metadata = load_array(
        str(path),
        variable=(None if source.get("variable") is None
                  else str(source.get("variable"))),
        time_index=int(source.get("time_index", 0)),
    )
    values, dimensions = apply_destagger(
        array, list(metadata.get("dimensions", [])),
        str(source.get("destagger", "none")),
    )
    plane = reduce_to_plane(
        values,
        reduction=str(source.get("reduction", "plane")),
        level_index=(None if source.get("level_index") is None
                     else int(source["level_index"])),
    )
    metadata.update({
        "dimensions_after_destagger": dimensions,
        "destagger": str(source.get("destagger", "none")),
        "reduction": str(source.get("reduction", "plane")),
        "level_index": source.get("level_index"),
        "source_shape": list(array.shape),
        "plane_shape": list(plane.shape),
    })
    return _finite_array(plane, label="spectral scored plane"), metadata


__all__ = [
    "NONFINITE_CELL_CEILING",
    "DESTAGGER", "NETCDF_SIGNATURES", "NETCDF_SUFFIXES", "REDUCTIONS",
    "SUPPORTED_SUFFIXES", "apply_destagger", "destagger", "file_identity",
    "load_array", "load_plane", "reduce_to_plane", "sha256_file",
]
