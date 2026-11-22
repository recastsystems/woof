"""ctypes seam onto the Rust isobaric-height reader (``rw-isobaric``).

The library is ``tools/rustwx/crates/rw-isobaric``: the one implementation
of an isobaric height read between a WRF-family model's layer interfaces,
which every chart, sounding and ML export of a run already uses.  A
layer-mean height paired with the mass-level pressure reads 4 to 6 m high
at 500 hPa; this seam is how the Python consumers (the moving-nest vortex
tracker on a host state, the GNSS-RO refractivity operator, the
verification diagnostics and the flagship products) read heights the same
way, from the same code, instead of from a Python copy of it.  WOOF-ArWen's
Python boundary names regrid/transform as data-path processing, so there is
no Python implementation behind this seam and no fallback to one.

Why ctypes rather than pyo3: the ruling every cdylib seam in the tree
follows (:mod:`woof.obs_regrid_bridge`, :mod:`woof.io.nc_writer_bridge`,
:mod:`woof.static.rust_bridge`) -- one loading discipline, one staging
path, one ABI-marker rule, no per-interpreter builds.

Arrays cross as C-contiguous float64, level index outermost.  A caller's
float32 arrays are widened here, which is exact; every arithmetic step,
including WRF's PH + PHB, happens in the library.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Final

import numpy as np

#: The C ABI version this module speaks; a library reporting anything else
#: is refused rather than called.
ISOBARIC_ABI: Final[int] = 1

#: Environment override for the library path, first rung of the ladder.
ISOBARIC_BRIDGE_ENV: Final[str] = "WOOF_ISOBARIC_BRIDGE"

#: The exported symbol that identifies THIS contract, for
#: :data:`woof.bridges.BRIDGE_ABI_MARKERS`: a build that answers the
#: version probe but predates the height reader cannot read one surface.
ABI_MARKER: Final[bytes] = b"gpuwm_isobaric_heights"

#: Standard gravity (m s-2): ``rw_isobaric::STANDARD_GRAVITY``.  Geopotential
#: over this is geopotential metres, as in a GRIB height message.
STANDARD_GRAVITY: Final[float] = 9.80665


class IsobaricBridgeError(RuntimeError):
    """The Rust isobaric-height reader refused, with its own message."""


def library_names() -> tuple[str, ...]:
    if os.name == "nt":
        return ("rw_isobaric.dll",)
    if os.uname().sysname == "Darwin":  # pragma: no cover - platform route
        return ("librw_isobaric.dylib",)
    return ("librw_isobaric.so",)


def library_candidates() -> tuple[Path, ...]:
    """Deterministic candidate paths, best first (the obs-regrid ladder)."""
    from woof.bridges import (default_bridge_dir, legacy_bridge_candidates,
                               packaged_bridge_dir)
    from woof.rustwx import crate_dir

    filename = library_names()[0]
    candidates: list[Path] = []
    override = os.environ.get(ISOBARIC_BRIDGE_ENV)
    if override:
        candidates.append(Path(override))
    root = Path(__file__).resolve().parent.parent.parent.parent
    candidates.extend((
        crate_dir() / "target" / "release" / filename,
        crate_dir() / "target" / "debug" / filename,
        root / "libexec" / "bridges" / filename,
        packaged_bridge_dir() / filename,
        default_bridge_dir() / filename,
        *legacy_bridge_candidates(filename),
    ))
    return tuple(candidates)


def resolve_isobaric_bridge() -> Path:
    """First existing candidate, or a refusal listing every path."""
    override = os.environ.get(ISOBARIC_BRIDGE_ENV)
    for candidate in library_candidates():
        if candidate.is_file():
            from woof.bridges import accept_resolved

            return accept_resolved(candidate.resolve(), executable=False)
        if override and candidate == Path(override):
            raise FileNotFoundError(
                f"{ISOBARIC_BRIDGE_ENV} names a missing file: {candidate}")
    rendered = "\n  ".join(str(c) for c in library_candidates())
    from woof import bridges

    separator = ";" if bridges.WINDOWS_SHELL else " &&"
    raise FileNotFoundError(
        "the Rust isobaric-height library was not found, and without it no "
        "isobaric height can be read between layer interfaces (a layer-mean "
        "height at the mass-level pressure reads 4 to 6 m high at 500 hPa); "
        "searched:\n  " + rendered
        + "\n  # stage it with the rest of the bundle:\n"
        "  woof fetch-bridges\n"
        "  # or build it from a checkout:\n"
        f"  cd tools/rustwx{separator} cargo build --release "
        f"-p rw-isobaric --offline{separator} cd ../..")


_LIBRARY: ctypes.CDLL | None = None


def unavailable_reason() -> str | None:
    """Why the Rust reader is not loadable here, or None when it is."""
    try:
        load()
    except (FileNotFoundError, OSError, IsobaricBridgeError) as error:
        return f"{type(error).__name__}: {error}"
    return None


def load() -> ctypes.CDLL:
    """Load the library once and bind every signature."""
    global _LIBRARY
    if _LIBRARY is not None:
        return _LIBRARY
    path = resolve_isobaric_bridge()
    library = ctypes.CDLL(str(path))

    library.gpuwm_isobaric_abi_version.argtypes = []
    library.gpuwm_isobaric_abi_version.restype = ctypes.c_uint32
    observed = int(library.gpuwm_isobaric_abi_version())
    if observed != ISOBARIC_ABI:
        raise IsobaricBridgeError(
            f"{path} speaks isobaric ABI {observed}, this woof needs "
            f"{ISOBARIC_ABI}; rebuild tools/rustwx")

    u8p = ctypes.POINTER(ctypes.c_uint8)
    f64p = ctypes.POINTER(ctypes.c_double)
    size = ctypes.c_size_t
    f64 = ctypes.c_double

    library.gpuwm_isobaric_last_error.argtypes = [u8p, size]
    library.gpuwm_isobaric_last_error.restype = size
    library.gpuwm_isobaric_interfaces_from_layer_thickness.argtypes = [
        f64p, size, f64p]
    library.gpuwm_isobaric_interfaces_from_layer_thickness.restype = ctypes.c_int32
    library.gpuwm_isobaric_heights.argtypes = [
        f64p, f64p, f64, f64p, size, size, f64p, f64p, f64p, size, f64p]
    library.gpuwm_isobaric_heights.restype = ctypes.c_int32
    library.gpuwm_isobaric_interface_log_pressure.argtypes = [
        f64p, size, size, f64p, f64p, f64p]
    library.gpuwm_isobaric_interface_log_pressure.restype = ctypes.c_int32
    library.gpuwm_isobaric_mass_level_heights.argtypes = [
        f64p, f64p, f64, f64p, size, size, f64p, f64p, f64p]
    library.gpuwm_isobaric_mass_level_heights.restype = ctypes.c_int32

    _LIBRARY = library
    return library


def _last_error(library: ctypes.CDLL) -> str:
    length = int(library.gpuwm_isobaric_last_error(None, 0))
    if length == 0:
        return "the library reported failure without a message"
    buffer = (ctypes.c_uint8 * length)()
    library.gpuwm_isobaric_last_error(
        ctypes.cast(buffer, ctypes.POINTER(ctypes.c_uint8)), length)
    return bytes(buffer).decode("utf-8", "replace")


def _f64(array) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(array, dtype=np.float64))


def _ptr(array: np.ndarray | None):
    if array is None:
        return None
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def _columns(p_mass) -> tuple[np.ndarray, int, int, tuple[int, ...]]:
    p = _f64(p_mass)
    if p.ndim < 1 or p.shape[0] < 2:
        raise IsobaricBridgeError(
            f"mass-level pressure of shape {p.shape} has fewer than two levels")
    nz = int(p.shape[0])
    horizontal = tuple(int(n) for n in p.shape[1:])
    cells = int(np.prod(horizontal, dtype=np.int64)) if horizontal else 1
    return p, nz, cells, horizontal


def _interfaces(values, plus, nz: int, horizontal: tuple[int, ...]):
    shape = (nz + 1,) + horizontal

    def widen(array):
        if array is None:
            return None
        array = np.asarray(array)
        if array.ndim == 1 and horizontal:
            # A base state carried as one profile (WRF's PHB on some
            # routes): broadcast, which copies and computes nothing.
            array = np.broadcast_to(array.reshape((nz + 1,) + (1,) * len(horizontal)),
                                    shape)
        array = _f64(array)
        if array.shape != shape:
            raise IsobaricBridgeError(
                f"interface values of shape {array.shape} are not {shape}")
        return array

    return widen(values), widen(plus)


def _eta(eta_mass, eta_interface, nz: int):
    interface = _f64(eta_interface).ravel()
    if interface.size != nz + 1:
        raise IsobaricBridgeError(
            f"{interface.size} eta interfaces for {nz} mass levels")
    mass = None
    if eta_mass is not None:
        mass = _f64(eta_mass).ravel()
        if mass.size != nz:
            raise IsobaricBridgeError(f"{mass.size} eta mass levels, not {nz}")
    return mass, interface


def _check(library, code: int) -> None:
    if code != 0:
        raise IsobaricBridgeError(_last_error(library))


def interfaces_from_layer_thickness(dnw) -> np.ndarray:
    """The eta interfaces of a coordinate stated as WRF's ``DNW``."""
    library = load()
    dnw = _f64(dnw).ravel()
    out = np.empty(dnw.size + 1, dtype=np.float64)
    _check(library, library.gpuwm_isobaric_interfaces_from_layer_thickness(
        _ptr(dnw), dnw.size, _ptr(out)))
    return out


def isobaric_heights(interface, p_mass, levels, *, eta_interface,
                     eta_mass=None, interface_plus=None,
                     per_metre: float = 1.0) -> np.ndarray:
    """Height (m) of each surface in ``levels`` (``p_mass``'s unit) in
    every column, ``(len(levels),) + horizontal``, NaN where a column has
    no such surface.  ``interface`` (+ ``interface_plus``) over
    ``per_metre`` are the interface heights, ``(nz + 1,) + horizontal``."""
    library = load()
    p, nz, cells, horizontal = _columns(p_mass)
    values, plus = _interfaces(interface, interface_plus, nz, horizontal)
    mass, eta_w = _eta(eta_mass, eta_interface, nz)
    targets = _f64(levels).ravel()
    out = np.empty((targets.size,) + horizontal, dtype=np.float64)
    _check(library, library.gpuwm_isobaric_heights(
        _ptr(values), _ptr(plus), float(per_metre), _ptr(p), nz, cells,
        _ptr(mass), _ptr(eta_w), _ptr(targets), targets.size, _ptr(out)))
    return out


def interface_log_pressure(p_mass, *, eta_interface, eta_mass=None) -> np.ndarray:
    """ln p on every interface, ``(nz + 1,) + horizontal``."""
    library = load()
    p, nz, cells, horizontal = _columns(p_mass)
    mass, eta_w = _eta(eta_mass, eta_interface, nz)
    out = np.empty((nz + 1,) + horizontal, dtype=np.float64)
    _check(library, library.gpuwm_isobaric_interface_log_pressure(
        _ptr(p), nz, cells, _ptr(mass), _ptr(eta_w), _ptr(out)))
    return out


def mass_level_heights(interface, p_mass, *, eta_interface, eta_mass=None,
                       interface_plus=None, per_metre: float = 1.0) -> np.ndarray:
    """Height (m) of each mass level's own pressure, ``(nz,) + horizontal``."""
    library = load()
    p, nz, cells, horizontal = _columns(p_mass)
    values, plus = _interfaces(interface, interface_plus, nz, horizontal)
    mass, eta_w = _eta(eta_mass, eta_interface, nz)
    out = np.empty((nz,) + horizontal, dtype=np.float64)
    _check(library, library.gpuwm_isobaric_mass_level_heights(
        _ptr(values), _ptr(plus), float(per_metre), _ptr(p), nz, cells,
        _ptr(mass), _ptr(eta_w), _ptr(out)))
    return out


__all__ = ["ABI_MARKER", "ISOBARIC_ABI", "ISOBARIC_BRIDGE_ENV",
           "IsobaricBridgeError", "STANDARD_GRAVITY", "interface_log_pressure",
           "interfaces_from_layer_thickness", "isobaric_heights", "library_names",
           "load", "mass_level_heights", "resolve_isobaric_bridge",
           "unavailable_reason"]
