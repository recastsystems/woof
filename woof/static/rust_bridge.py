"""ctypes seam onto the Rust static-field builder (``static-fields``).

The library is ``tools/rustwx/crates/static-fields``, the port target
for everything in :mod:`woof.static` that processes bytes: projection
transforms, WPS_GEOG tile ingest, the geogrid-equivalent field build,
corridor geometry/crop, and the high-resolution overlay compute.  This
module owns ONLY the loading discipline and the low-level call
bindings; the public Python entry points (`build_static`,
`ProjectedGrid` methods, the highres appliers) keep their signatures
and route here by default once their lane lands.

Why ctypes rather than pyo3: same ruling as
:mod:`woof.io.nc_writer_bridge` -- every Rust library woof drives is
a cdylib behind ctypes; one loading discipline, one staging path, one
ABI-marker rule, no per-interpreter builds.

Default-on contract (WOOF's fixed-means-default law): once a lane
lands, the bare default path for its entry points is this bridge.  The
pure-Python implementation stays importable as the parity reference
and as an explicit fallback (``WOOF_STATIC_PYTHON=1``), and every use
of the fallback is a reported workaround, never silent.
"""

from __future__ import annotations

import ctypes
import json
import os
import sys
from pathlib import Path
from typing import Final

import numpy as np

#: The C ABI version this module speaks; a library reporting anything
#: else is refused rather than called.
STATIC_ABI: Final[int] = 1

#: Environment override for the library path, first rung of the ladder.
STATIC_BRIDGE_ENV: Final[str] = "WOOF_STATIC_BRIDGE"

#: Opt-out to the pure-Python implementation (a reported workaround).
STATIC_PYTHON_ENV: Final[str] = "WOOF_STATIC_PYTHON"

#: The exported symbol that identifies THIS contract, for
#: :data:`woof.bridges.BRIDGE_ABI_MARKERS`: a build that loads and
#: answers the version probe but predates portable sampling cannot
#: rebuild a prepared moving footprint with identical terrain and climatologies.
ABI_MARKER: Final[bytes] = b"gpuwm_static_sampling_portable_v1"
EXTRA_ABI_MARKER: Final[bytes] = b"gpuwm_static_extra_continuous_v1"

#: Stagger codes (`types::Stagger` in the crate).
STAGGER_MASS: Final[int] = 0
STAGGER_U: Final[int] = 1
STAGGER_V: Final[int] = 2
STAGGER_CORNER: Final[int] = 3

#: Array kinds for ``gpuwm_static_grid_array``.
ARRAY_LAT: Final[int] = 0
ARRAY_LON: Final[int] = 1
ARRAY_MAPFAC: Final[int] = 2
ARRAY_CORIOLIS_F: Final[int] = 3
ARRAY_CORIOLIS_E: Final[int] = 4
ARRAY_SINALPHA: Final[int] = 5
ARRAY_COSALPHA: Final[int] = 6


class StaticBridgeError(RuntimeError):
    """The Rust static builder refused, with its own message."""


#: Operations whose pure-Python fallback has already been reported this
#: process (fixed-means-default: a fallback run is a workaround and must
#: say so, once per operation, never silently and never on every call).
_REPORTED_FALLBACKS: set[str] = set()


def global_source_donors(values, *, period, first, out_nx,
                         south=0, north=0, vector=False):
    """Select a global ring and its pole donors in parallel Rust code."""
    source = np.ascontiguousarray(values, dtype=np.float64)
    ny, nx = source.shape[-2:]
    result = np.empty((*source.shape[:-2], ny + south + north, out_nx),
                      dtype=np.float64)
    library = load()
    try:
        select = library.gpuwm_static_global_donors_f64
    except AttributeError:
        raise StaticBridgeError(
            "Rust static-fields lacks periodic pole donors; rebuild with "
            + _checkout_build_command()) from None
    pointer = ctypes.POINTER(ctypes.c_double)
    select.argtypes = [pointer, pointer] + [ctypes.c_size_t] * 4 + [
        ctypes.c_int64, ctypes.c_size_t, ctypes.c_size_t,
        ctypes.c_size_t, ctypes.c_int32]
    select.restype = ctypes.c_int32
    code = select(source.ctypes.data_as(pointer), result.ctypes.data_as(pointer),
                  source.size // (ny * nx), ny, nx, period,
                  first, out_nx, south, north, int(vector))
    _check(library, code, "Rust global pole donor selection")
    return result


def floor_ruc_layer_moisture(values):
    """Copy and floor f32 RUC values through the Rust initialization seam."""
    result = np.array(values, dtype=np.float32, order="C", copy=True)
    library = load()
    try:
        floor = library.gpuwm_static_ruc_layer_moisture_floor
    except AttributeError:
        raise StaticBridgeError(
            "Rust static-fields lacks the RUC layer moisture floor; an old bridge "
            "would leave negative extrapolated SMOIS for cold start. Rebuild with "
            + _checkout_build_command()) from None
    floor.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
    floor.restype = ctypes.c_int32
    if floor(result.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), result.size) != 0:
        raise StaticBridgeError("Rust RUC layer moisture floor refused its array")
    return result


def route(operation: str):
    """This module when the Rust seam is the active default, else None.

    ``None`` means the caller must run its pure-Python body -- either
    because :data:`STATIC_PYTHON_ENV` explicitly opted out or because
    the library is not loadable -- and the reason has been printed once
    per operation as a WORKAROUND line.  Every default-on entry point in
    :mod:`woof.static` routes through this single decision so the env
    flag is read at call time (parity harnesses toggle it per call).
    """
    if python_fallback_requested():
        reason = f"{STATIC_PYTHON_ENV}=1"
    else:
        reason = unavailable_reason()
        if reason is None:
            return sys.modules[__name__]
    report_workaround(operation, reason)
    return None


def report_workaround(operation: str, reason: str) -> None:
    """Print one WORKAROUND line per operation per process."""
    if operation not in _REPORTED_FALLBACKS:
        _REPORTED_FALLBACKS.add(operation)
        print(f"[static] WORKAROUND: pure-Python {operation} "
              f"(Rust static-fields bridge not used: {reason})")


def python_fallback_requested() -> bool:
    """Has the caller explicitly opted out of the Rust path?

    The opt-out exists for parity debugging; any production route that
    honors it must say so in its receipt (fixed-means-default).
    """
    return os.environ.get(STATIC_PYTHON_ENV, "").strip() not in ("", "0")


def library_names() -> tuple[str, ...]:
    if os.name == "nt":
        return ("static_fields.dll",)
    if os.uname().sysname == "Darwin":  # pragma: no cover - platform route
        return ("libstatic_fields.dylib",)
    return ("libstatic_fields.so",)


def library_candidates() -> tuple[Path, ...]:
    """Deterministic candidate paths, best first (the nc-writer ladder)."""
    from woof.bridges import (default_bridge_dir, legacy_bridge_candidates,
                               packaged_bridge_dir)
    from woof.rustwx import crate_dir

    filename = library_names()[0]
    candidates: list[Path] = []
    override = os.environ.get(STATIC_BRIDGE_ENV)
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


def _checkout_build_command() -> str:
    """The checkout build of this library, spelled for the reader's shell.

    Both refusals below print it.  The separator comes from
    :data:`woof.bridges.WINDOWS_SHELL`, the one shell rule every remedy
    reads, because Windows PowerShell 5.1 rejects ``&&`` with a parser
    error: the stale-build refusal hard-coded ``&&``, so on Windows the
    line ``woof doctor`` printed for a stale staged library could not be
    pasted.  It ends with the ``cd`` back so a pasted block leaves the
    shell where it started.
    """

    from woof import bridges

    separator = ";" if bridges.WINDOWS_SHELL else " &&"
    return (f"cd {bridges.RUSTWX_CRATE_RELATIVE}{separator} cargo build "
            f"--release -p static-fields --offline{separator} cd ../..")


def resolve_static_bridge() -> Path:
    """First existing candidate, or a refusal listing every path."""
    override = os.environ.get(STATIC_BRIDGE_ENV)
    for candidate in library_candidates():
        if candidate.is_file():
            from woof.bridges import accept_resolved

            return accept_resolved(candidate.resolve(), executable=False)
        if override and candidate == Path(override):
            raise FileNotFoundError(
                f"{STATIC_BRIDGE_ENV} names a missing file: {candidate}")
    rendered = "\n  ".join(str(c) for c in library_candidates())
    raise FileNotFoundError(
        "the Rust static-field library was not found; searched:\n  "
        + rendered
        + "\n  # build it from a checkout:\n  "
        + _checkout_build_command())


_LIBRARY: ctypes.CDLL | None = None


def unavailable_reason() -> str | None:
    """Why the Rust builder is not loadable here, or None when it is."""
    try:
        load()
    except (FileNotFoundError, OSError, StaticBridgeError) as error:
        return f"{type(error).__name__}: {error}"
    return None


def load() -> ctypes.CDLL:
    """Load the library once and bind every signature."""
    global _LIBRARY
    if _LIBRARY is not None:
        return _LIBRARY
    path = resolve_static_bridge()
    library = ctypes.CDLL(str(path))

    library.gpuwm_static_abi_version.argtypes = []
    library.gpuwm_static_abi_version.restype = ctypes.c_uint32
    observed = int(library.gpuwm_static_abi_version())
    if observed != STATIC_ABI:
        raise StaticBridgeError(
            f"{path} speaks static-fields ABI {observed}, this woof "
            f"needs {STATIC_ABI}; rebuild tools/rustwx")

    u8p = ctypes.POINTER(ctypes.c_uint8)
    u64p = ctypes.POINTER(ctypes.c_uint64)
    f64p = ctypes.POINTER(ctypes.c_double)
    size = ctypes.c_size_t

    # EVERY binding below is inside one net, and the net names what it
    # caught.  The version handshake above cannot answer this question:
    # a library built before an entry point was ADDED still answers ABI
    # 1, so it passes, and then `getattr` on the entry point it does not
    # export raises a bare `AttributeError` -- which
    # `unavailable_reason` does not catch, because ctypes' lookup
    # failure is not one of the three error classes a missing or
    # unloadable library raises.
    #
    # The concrete breakage that prevented: a box carrying a bundle
    # staged before the high-resolution warp entry points existed
    # (`gpuwm_static_highres_resample`, `..._transform_points`) ran
    # `woof doctor` -- the tool whose entire purpose is diagnosing a
    # broken or partial install -- and got a ctypes traceback out of
    # `_static_builder_check` instead of a report.  Measured on this
    # tree, 2026-08-18, against a real staged
    # `~/.woof/bridges/static_fields.dll`.
    #
    # A missing entry point is the same FACT as a version mismatch --
    # this library predates this woof -- so it is reported the same
    # way: a StaticBridgeError, which makes the builder row say
    # `unusable` with a remedy, and makes the static path take the
    # announced `WOOF_STATIC_PYTHON` degradation rather than crash.
    stale_build_remedy = (
        "It is a build that predates this release. A moving nest whose statics "
        "were prepared by another build or machine can refuse at its first move "
        "on the overlap-statics equality check. Restage it with `woof "
        "fetch-bridges`, or rebuild from a checkout: "
        + _checkout_build_command())
    try:
        _bind_entry_points(library, u8p=u8p, u64p=u64p, f64p=f64p, size=size)
        marker = getattr(library, ABI_MARKER.decode("ascii"))
        marker.argtypes = []
        marker.restype = ctypes.c_uint32
        if marker() != 1:
            raise StaticBridgeError(
                f"{path} answers static-fields ABI {observed} but reports an "
                f"incompatible sampling contract. {stale_build_remedy}")
    except AttributeError as error:
        raise StaticBridgeError(
            f"{path} answers static-fields ABI {observed} but does not "
            f"export an entry point this woof binds ({error}). "
            f"{stale_build_remedy}") from None

    _LIBRARY = library
    return library


def _bind_entry_points(library: ctypes.CDLL, *, u8p, u64p, f64p, size) -> None:
    """Bind every signature this woof calls.

    Split out of :func:`load` so one ``try`` covers the whole set,
    including entry points added later: a binding that escapes the net
    is a binding that can still crash a report.
    """

    library.gpuwm_static_last_error.argtypes = [u8p, size]
    library.gpuwm_static_last_error.restype = size

    library.gpuwm_static_grid_new.argtypes = [u8p, size, u64p]
    library.gpuwm_static_grid_new.restype = ctypes.c_int32
    library.gpuwm_static_grid_nest.argtypes = [
        ctypes.c_uint64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_double, ctypes.c_double,
        u64p]
    library.gpuwm_static_grid_nest.restype = ctypes.c_int32
    library.gpuwm_static_grid_translated.argtypes = [
        ctypes.c_uint64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_int64, u64p]
    library.gpuwm_static_grid_translated.restype = ctypes.c_int32
    library.gpuwm_static_grid_free.argtypes = [ctypes.c_uint64]
    library.gpuwm_static_grid_free.restype = None
    library.gpuwm_static_grid_array.argtypes = [
        ctypes.c_uint64, ctypes.c_uint32, ctypes.c_uint32, f64p, size]
    library.gpuwm_static_grid_array.restype = ctypes.c_int32
    # Additive ABI: older libraries retain their original native calls.
    if hasattr(library, "gpuwm_static_grid_arrays"):
        library.gpuwm_static_grid_arrays.argtypes = [
            ctypes.c_uint64, ctypes.c_uint32, ctypes.c_uint32, f64p, size]
        library.gpuwm_static_grid_arrays.restype = ctypes.c_int32
    library.gpuwm_static_grid_transform.argtypes = [
        ctypes.c_uint64, ctypes.c_uint32, f64p, f64p, size]
    library.gpuwm_static_grid_transform.restype = ctypes.c_int32
    library.gpuwm_static_grid_identity_probes.argtypes = [
        ctypes.c_uint64, u8p, size]
    library.gpuwm_static_grid_identity_probes.restype = ctypes.c_int64

    library.gpuwm_static_build_fields.argtypes = [
        ctypes.c_uint64, u8p, size, ctypes.c_uint32, u64p]
    library.gpuwm_static_build_fields.restype = ctypes.c_int32
    library.gpuwm_static_build_terrain.argtypes = [ctypes.c_uint64, u8p, size, ctypes.c_uint32, u64p]
    library.gpuwm_static_build_terrain.restype = ctypes.c_int32
    library.gpuwm_static_fieldset_len.argtypes = [ctypes.c_uint64]
    library.gpuwm_static_fieldset_len.restype = ctypes.c_int64
    library.gpuwm_static_fieldset_name.argtypes = [
        ctypes.c_uint64, size, u8p, size]
    library.gpuwm_static_fieldset_name.restype = ctypes.c_int64
    library.gpuwm_static_field_dims.argtypes = [
        ctypes.c_uint64, u8p, size, u64p, u64p, u64p]
    library.gpuwm_static_field_dims.restype = ctypes.c_int32
    library.gpuwm_static_field_read.argtypes = [
        ctypes.c_uint64, u8p, size, f64p, size]
    library.gpuwm_static_field_read.restype = ctypes.c_int32
    library.gpuwm_static_field_coverage_json.argtypes = [
        ctypes.c_uint64, u8p, size, u8p, size]
    library.gpuwm_static_field_coverage_json.restype = ctypes.c_int64
    library.gpuwm_static_fieldset_free.argtypes = [ctypes.c_uint64]
    library.gpuwm_static_fieldset_free.restype = None

    library.gpuwm_static_highres_terrain.argtypes = [
        ctypes.c_uint64, u8p, size, u64p]
    library.gpuwm_static_highres_terrain.restype = ctypes.c_int32
    library.gpuwm_static_highres_overrides.argtypes = [
        ctypes.c_uint64, u8p, size, u64p]
    library.gpuwm_static_highres_overrides.restype = ctypes.c_int32
    library.gpuwm_static_highres_merge.argtypes = [
        ctypes.c_uint64, ctypes.c_uint64, u8p, size, u64p]
    library.gpuwm_static_highres_merge.restype = ctypes.c_int32
    library.gpuwm_static_highres_derive_window.argtypes = [
        u8p, size, u8p, size]
    library.gpuwm_static_highres_derive_window.restype = ctypes.c_int64
    library.gpuwm_static_highres_resample.argtypes = [u8p, size, u64p]
    library.gpuwm_static_highres_resample.restype = ctypes.c_int32
    library.gpuwm_static_highres_transform_points.argtypes = [
        u8p, size, f64p, f64p, size]
    library.gpuwm_static_highres_transform_points.restype = ctypes.c_int32
    library.gpuwm_static_highres_audit_json.argtypes = [
        ctypes.c_uint64, u8p, size]
    library.gpuwm_static_highres_audit_json.restype = ctypes.c_int64
    library.gpuwm_static_highres_audit_drop.argtypes = [ctypes.c_uint64]
    library.gpuwm_static_highres_audit_drop.restype = None


def last_error(library: ctypes.CDLL) -> str:
    length = int(library.gpuwm_static_last_error(None, 0))
    if length == 0:
        return "(no error recorded)"
    buffer = (ctypes.c_uint8 * length)()
    library.gpuwm_static_last_error(buffer, length)
    return bytes(buffer).decode("utf-8", "replace")


def _check(library: ctypes.CDLL, code: int, doing: str) -> None:
    if int(code) != 0:
        raise StaticBridgeError(f"{doing}: {last_error(library)}")


def _utf8(payload: str) -> tuple[ctypes.Array, int]:
    raw = payload.encode("utf-8")
    return (ctypes.c_uint8 * len(raw)).from_buffer_copy(raw), len(raw)


def grid_new(spec: dict) -> int:
    """Create a Rust grid handle from a GridSpec-shaped dict."""
    library = load()
    buffer, length = _utf8(json.dumps(spec))
    handle = ctypes.c_uint64(0)
    _check(library,
           library.gpuwm_static_grid_new(buffer, length,
                                         ctypes.byref(handle)),
           "grid_new")
    return int(handle.value)


def grid_free(handle: int) -> None:
    load().gpuwm_static_grid_free(ctypes.c_uint64(handle))


def grid_nest(parent, i, j, ratio, e_we, e_sn, dx, dy) -> int:
    """Construct a child with its exact integer placement lineage."""
    library = load()
    handle = ctypes.c_uint64(0)
    _check(library, library.gpuwm_static_grid_nest(
        parent, i, j, ratio, e_we, e_sn, dx, dy, ctypes.byref(handle)), "grid_nest")
    return int(handle.value)


def grid_translated(reference: int, di_cells: int, dj_cells: int,
                    e_we: int, e_sn: int) -> int:
    """Placement-translated grid handle from a reference handle."""
    library = load()
    handle = ctypes.c_uint64(0)
    _check(library,
           library.gpuwm_static_grid_translated(
               ctypes.c_uint64(reference), int(di_cells), int(dj_cells),
               int(e_we), int(e_sn), ctypes.byref(handle)),
           "grid_translated")
    return int(handle.value)


def grid_array(handle: int, stagger: int, kind: int,
               ny: int, nx: int) -> np.ndarray:
    """One derived array (row-major float64) from a grid handle."""
    library = load()
    out = np.empty((int(ny), int(nx)), dtype=np.float64)
    _check(library,
           library.gpuwm_static_grid_array(
               ctypes.c_uint64(handle), ctypes.c_uint32(stagger),
               ctypes.c_uint32(kind),
               out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
               out.size),
           f"grid_array[{stagger},{kind}]")
    return out



def grid_arrays(handle: int, stagger: int, kinds: tuple[int, ...],
                ny: int, nx: int) -> tuple[np.ndarray, ...]:
    """Read distinct derived fields with one native coordinate pass."""
    library = load()
    function = getattr(library, "gpuwm_static_grid_arrays", None)
    if (function is None or len(set(kinds)) != len(kinds)
            or any(kind < 0 or kind > 6 for kind in kinds)):
        if function is None and "derived array batch" not in _REPORTED_FALLBACKS:
            _REPORTED_FALLBACKS.add("derived array batch")
            print("static geometry: library lacks derived-array batching; "
                  "using the existing native array calls", file=sys.stderr)
        return tuple(grid_array(handle, stagger, kind, ny, nx) for kind in kinds)
    if not kinds:
        return ()
    ordered = sorted(kinds)
    out = np.empty((len(ordered), int(ny), int(nx)), dtype=np.float64)
    _check(library, function(
        ctypes.c_uint64(handle), ctypes.c_uint32(stagger),
        ctypes.c_uint32(sum(1 << kind for kind in ordered)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), out.size),
        f"grid_arrays[{stagger},{kinds}]")
    arrays = dict(zip(ordered, out))
    return tuple(arrays[kind] for kind in kinds)


def wrf_cell_geometry(dx, dy, msftx=None, msfty=None, *, shape=None):
    """Source-REAL WRF cell area and distance, computed in Rust once."""
    library = load()
    try:
        function = library.gpuwm_static_wrf_cell_geometry
    except AttributeError as error:
        raise StaticBridgeError("native WRF history cell geometry requires the rebuilt static-fields bridge") from error
    pointer = ctypes.POINTER(ctypes.c_float)
    function.argtypes = [ctypes.c_float, ctypes.c_float, pointer, pointer,
                         pointer, pointer, ctypes.c_size_t]
    function.restype = ctypes.c_int
    if shape is None:
        x, y = np.asarray(msftx), np.asarray(msfty)
        if x.shape != y.shape or x.ndim != 2:
            raise ValueError("WRF cell geometry needs a horizontal shape")
        shape = x.shape
    if len(shape) != 2 or any(int(n) <= 0 for n in shape):
        raise ValueError("WRF cell geometry needs a positive horizontal shape")
    area = np.empty(tuple(map(int, shape)), np.float32)
    distance = np.empty_like(area)
    _check(library, function(float(dx), float(dy),
           None, None, area.ctypes.data_as(pointer), distance.ctypes.data_as(pointer), area.size),
           "wrf_cell_geometry")
    return area, distance


def wrf_noah_snow_albedo(max_alb, ivgtyp):
    """Native Noah LSMINIT table replacement, seeded from Rust REAL4."""
    library = load()
    try:
        function = library.gpuwm_static_wrf_noah_snow_albedo
    except AttributeError as error:
        raise StaticBridgeError("native Noah LSMINIT requires the rebuilt static-fields bridge") from error
    pointer = ctypes.POINTER(ctypes.c_float)
    index_pointer = ctypes.POINTER(ctypes.c_int32)
    function.argtypes = [pointer, ctypes.c_size_t, index_pointer, pointer, ctypes.c_size_t]
    function.restype = ctypes.c_int
    table = np.ascontiguousarray(max_alb, dtype=np.float32)
    indices = np.ascontiguousarray(ivgtyp, dtype=np.int32)
    if table.ndim != 1 or indices.ndim != 2:
        raise ValueError("native Noah snow albedo needs a category table and horizontal indices")
    output = np.empty(indices.shape, np.float32)
    _check(library, function(table.ctypes.data_as(pointer), table.size,
           indices.ctypes.data_as(index_pointer), output.ctypes.data_as(pointer), output.size),
           "wrf_noah_snow_albedo")
    return output


def grid_hydrofabric_json(handle: int) -> str:
    """Setup-only projected mass-grid metadata from the executed native handle."""
    library = load()
    try:
        entry = library.gpuwm_static_grid_hydrofabric_json
    except AttributeError:
        raise StaticBridgeError(
            "Rust static-fields lacks the hydrofabric projected-grid export; "
            "rebuild with " + _checkout_build_command()) from None
    entry.argtypes = [ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
    entry.restype = ctypes.c_int64
    length = int(entry(ctypes.c_uint64(handle), None, 0))
    if length < 0:
        raise StaticBridgeError(f"grid_hydrofabric_json: {last_error(library)}")
    buffer = (ctypes.c_uint8 * length)()
    written = int(entry(ctypes.c_uint64(handle), buffer, length))
    if written != length:
        raise StaticBridgeError(
            f"grid_hydrofabric_json rendered {written} bytes on a {length}-byte probe: "
            + last_error(library))
    return bytes(buffer).decode("utf-8")


def grid_identity_probes_json(handle: int) -> str:
    """The corridor identity-probe JSON document for a grid handle."""
    library = load()
    length = int(library.gpuwm_static_grid_identity_probes(
        ctypes.c_uint64(handle), None, 0))
    if length < 0:
        raise StaticBridgeError(
            f"grid_identity_probes: {last_error(library)}")
    buffer = (ctypes.c_uint8 * length)()
    written = int(library.gpuwm_static_grid_identity_probes(
        ctypes.c_uint64(handle), buffer, length))
    if written != length:
        raise StaticBridgeError(
            f"grid_identity_probes rendered {written} bytes on a "
            f"{length}-byte probe")
    return bytes(buffer).decode("utf-8")


def field_coverage_json(handle: int, name: str) -> str:
    """One field's source-coverage receipt JSON from a fieldset handle."""
    library = load()
    name_buf, name_len = _utf8(name)
    length = int(library.gpuwm_static_field_coverage_json(
        ctypes.c_uint64(handle), name_buf, name_len, None, 0))
    if length < 0:
        raise StaticBridgeError(
            f"field_coverage_json[{name}]: {last_error(library)}")
    buffer = (ctypes.c_uint8 * length)()
    written = int(library.gpuwm_static_field_coverage_json(
        ctypes.c_uint64(handle), name_buf, name_len, buffer, length))
    if written != length:
        raise StaticBridgeError(
            f"field_coverage_json[{name}] rendered {written} bytes on "
            f"a {length}-byte probe")
    return bytes(buffer).decode("utf-8")


def build_fields(grid_handle: int, geog_paths: dict,
                 halo: int | None = None) -> int:
    """Run the full static build; returns a fieldset handle."""
    library = load()
    buffer, length = _utf8(json.dumps(
        {key: str(value) for key, value in geog_paths.items()}))
    handle = ctypes.c_uint64(0)
    halo_code = 0xFFFFFFFF if halo is None else int(halo)
    _check(library,
           library.gpuwm_static_build_fields(
               ctypes.c_uint64(grid_handle), buffer, length,
               ctypes.c_uint32(halo_code), ctypes.byref(handle)),
           "build_fields")
    return int(handle.value)


#: Exported by a library that builds WRF's sub-grid orographic statistics
#: (``gpuwm_static_build_orographic``).  Bound on first use, never in
#: ``_bind_entry_points``, so a library staged before it keeps serving
#: every build that does not ask for them.
OROGRAPHIC_MARKER: Final[str] = "gpuwm_static_orographic_v2"
CONTINUOUS_MARKER: Final[str] = "gpuwm_static_continuous_v1"


def build_orographic(grid_handle: int, request: dict,
                     halo: int | None = None) -> int:
    """Build the requested orographic statistics; returns a fieldset handle.

    ``request`` is the crate's ``OrographicRequest``: ``{"landuse": path,
    "fields": [{"name", "path", "gcell", "masked_water"}, ...]}``.
    """
    library = load()
    extended = {"gcell_ratio", "search_depth", "masked_land", "fill_missing"}
    if (any(extended.intersection(field) for field in request.get("fields", ()))
            and not hasattr(library, CONTINUOUS_MARKER)):
        raise StaticBridgeError(
            "the staged static-fields library lacks declared grid-cell "
            "ratios, bounded WPS search and land masking; LAKE_DEPTH would "
            "use different bathymetry. Restage the rebuilt bridge: "
            + _checkout_build_command())
    if not hasattr(library, OROGRAPHIC_MARKER):
        raise StaticBridgeError(
            "the staged static-fields library predates the corrected WPS "
            "single-precision sampling and post-interpolation scaling of sub-grid "
            "orographic statistics (VAR_SSO, CON, VAR, OA1-4, OL1-4 and the "
            "GSL sets) the terrain-drag options read; restage the rebuilt "
            "bridge or rebuild from this checkout: "
            + _checkout_build_command())
    entry = library.gpuwm_static_build_orographic
    u8p = ctypes.POINTER(ctypes.c_uint8)
    entry.argtypes = [ctypes.c_uint64, u8p, ctypes.c_size_t,
                      ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint64)]
    entry.restype = ctypes.c_int32
    buffer, length = _utf8(json.dumps(request))
    handle = ctypes.c_uint64(0)
    halo_code = 0xFFFFFFFF if halo is None else int(halo)
    _check(library,
           entry(ctypes.c_uint64(grid_handle), buffer, length,
                 ctypes.c_uint32(halo_code), ctypes.byref(handle)),
           "build_orographic")
    return int(handle.value)

def require_extra_fields(library=None):
    """Bind the optional contract without changing native library loading."""
    library = load() if library is None else library
    try:
        marker = getattr(library, EXTRA_ABI_MARKER.decode("ascii"))
        function = library.gpuwm_static_build_extra_fields
    except AttributeError as error:
        raise StaticBridgeError(
            "this Rust library cannot build extra static fields; rebuild "
            "static-fields and set WOOF_STATIC_BRIDGE to that library. "
            + _checkout_build_command()) from error
    marker.argtypes = []
    marker.restype = ctypes.c_uint32
    if marker() != 1:
        raise StaticBridgeError("unsupported extra static fields contract; rebuild static-fields")
    function.argtypes = [ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint8),
                         ctypes.c_size_t, ctypes.c_uint32,
                         ctypes.POINTER(ctypes.c_uint64)]
    function.restype = ctypes.c_int32
    return library


def scalar_reference_asin_f32(values) -> np.ndarray:
    """Evaluate ASIN with the qualified native binary32 intrinsic contract.

    The optional entry point leaves ordinary static loading unchanged. Three
    scalar word probes distinguish the required rounding from a native runtime
    that would change terrain fluxes or radiation-call solar declination.
    """
    library = load()
    try:
        entry = library.gpuwm_static_scalar_asin_f32
    except AttributeError:
        raise StaticBridgeError(
            "the static-fields bridge lacks the strict scalar ASIN contract; "
            "SLP_AZI and SWNORM before Noah, or solar COSZEN, would differ. Rebuild "
            "static-fields from this checkout: " + _checkout_build_command()) from None
    f32p = ctypes.POINTER(ctypes.c_float)
    entry.argtypes = [f32p, f32p, ctypes.c_size_t]
    entry.restype = ctypes.c_int32

    def evaluate(array):
        result = np.empty_like(array)
        _check(library, entry(array.ctypes.data_as(f32p), result.ctypes.data_as(f32p),
                              ctypes.c_size_t(array.size)), "scalar terrain rotation")
        return result

    probes = np.array([0xbd137442, 0xbd222103, 0xbd289002], dtype=np.uint32).view(np.float32)
    # Both scalar libm profiles are witnessed by native Fortran. The second
    # is gfortran's non-vectorized ASIN on the MP52 native forecast host;
    # see tools/p3_wrf461_oracle/native_asin_probe.f90. A single assumed
    # profile rejected that native runtime even though Rust matched it.
    qualified = ((0xbd137c6b, 0xbd222bdc, 0xbd289c33),
                 (0xbd137c6a, 0xbd222bdb, 0xbd289c32))
    if tuple(map(int, evaluate(probes).view(np.uint32))) not in qualified:
        raise StaticBridgeError(
            "this native scalar rounding changes SLP_AZI and SWNORM before "
            "Noah, or solar COSZEN; it cannot match the scalar reference")
    return evaluate(np.ascontiguousarray(values, dtype=np.float32))


def scalar_rotation_asin_f32(values) -> np.ndarray:
    """Retain the terrain caller while sharing its native scalar contract."""
    return scalar_reference_asin_f32(values)


def build_extra_fields(grid_handle: int, specs: list[dict],
                       halo: int | None = None) -> dict[str, np.ndarray]:
    """Build only the supplied rows, returning native float64 arrays."""
    library = require_extra_fields()
    buffer, length = _utf8(json.dumps(specs, allow_nan=False))
    handle = ctypes.c_uint64(0)
    _check(library, library.gpuwm_static_build_extra_fields(
        ctypes.c_uint64(grid_handle), buffer, length,
        ctypes.c_uint32(0xFFFFFFFF if halo is None else int(halo)),
        ctypes.byref(handle)), "build_extra_fields")
    try:
        return fieldset_to_dict(handle.value)
    finally:
        fieldset_free(handle.value)


def fieldset_to_dict(handle: int) -> dict[str, np.ndarray]:
    """Copy every field of a fieldset handle into float64 numpy arrays
    (2-D fields lose their leading singleton plane axis)."""
    library = load()
    count = int(library.gpuwm_static_fieldset_len(ctypes.c_uint64(handle)))
    if count < 0:
        raise StaticBridgeError(f"fieldset_len: {last_error(library)}")
    out: dict[str, np.ndarray] = {}
    for index in range(count):
        cap = 128
        buffer = (ctypes.c_uint8 * cap)()
        length = int(library.gpuwm_static_fieldset_name(
            ctypes.c_uint64(handle), ctypes.c_size_t(index), buffer, cap))
        if length < 0 or length > cap:
            raise StaticBridgeError(
                f"fieldset_name[{index}]: {last_error(library)}")
        name = bytes(buffer[:length]).decode("utf-8")
        name_buf, name_len = _utf8(name)
        planes = ctypes.c_uint64(0)
        ny = ctypes.c_uint64(0)
        nx = ctypes.c_uint64(0)
        _check(library,
               library.gpuwm_static_field_dims(
                   ctypes.c_uint64(handle), name_buf, name_len,
                   ctypes.byref(planes), ctypes.byref(ny),
                   ctypes.byref(nx)),
               f"field_dims[{name}]")
        shape = (int(planes.value), int(ny.value), int(nx.value))
        array = np.empty(shape, dtype=np.float64)
        _check(library,
               library.gpuwm_static_field_read(
                   ctypes.c_uint64(handle), name_buf, name_len,
                   array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                   array.size),
               f"field_read[{name}]")
        out[name] = array[0] if shape[0] == 1 else array
    return out


def build_terrain(handle, path, halo):
    library = load()
    buffer, length = _utf8(str(path))
    result = ctypes.c_uint64(0)
    _check(library, library.gpuwm_static_build_terrain(
        handle, buffer, length, int(halo), ctypes.byref(result)), "build_terrain")
    try:
        return fieldset_to_dict(result.value)["HGT_M"]
    finally:
        fieldset_free(result.value)


def fieldset_free(handle: int) -> None:
    load().gpuwm_static_fieldset_free(ctypes.c_uint64(handle))


def highres_audit_json(handle: int) -> dict:
    """The audit document a highres call remembered for its fieldset."""
    library = load()
    length = int(library.gpuwm_static_highres_audit_json(
        ctypes.c_uint64(handle), None, 0))
    if length < 0:
        raise StaticBridgeError(f"highres audit: {last_error(library)}")
    buffer = (ctypes.c_uint8 * length)()
    written = int(library.gpuwm_static_highres_audit_json(
        ctypes.c_uint64(handle), buffer, length))
    if written != length:
        raise StaticBridgeError(
            f"highres audit rendered {written} bytes on a {length}-byte "
            "probe")
    return json.loads(bytes(buffer).decode("utf-8"))


def highres_audit_drop(handle: int) -> None:
    load().gpuwm_static_highres_audit_drop(ctypes.c_uint64(handle))


def highres_resample(request: dict) -> tuple[dict[str, np.ndarray], dict]:
    """Run one warp-substrate request; returns (fields, audit).

    The crate's refusals are RAISED as :class:`ValueError` carrying the
    crate's own message, which the committed goldens pin byte-equal to
    the pure-Python messages -- so a caller cannot tell which engine
    refused from the text, only from the WORKAROUND line.
    """
    library = load()
    buffer, length = _utf8(json.dumps(request))
    handle = ctypes.c_uint64(0)
    code = library.gpuwm_static_highres_resample(
        buffer, length, ctypes.byref(handle))
    if code != 0:
        raise ValueError(last_error(library))
    try:
        fields = fieldset_to_dict(int(handle.value))
        audit = highres_audit_json(int(handle.value))
    finally:
        highres_audit_drop(int(handle.value))
        fieldset_free(int(handle.value))
    return fields, audit


def highres_transform_points(to_crs: str, x: np.ndarray, y: np.ndarray
                             ) -> tuple[np.ndarray, np.ndarray]:
    """Transform lon/lat degree arrays into ``to_crs`` (a recorded
    ``crs_override`` string), returning new float64 arrays."""
    library = load()
    xs = np.ascontiguousarray(np.asarray(x, dtype=np.float64)).reshape(-1)
    ys = np.ascontiguousarray(np.asarray(y, dtype=np.float64)).reshape(-1)
    if xs.size != ys.size:
        raise ValueError(
            f"transform_points x/y length mismatch: {xs.size} vs {ys.size}")
    xs, ys = xs.copy(), ys.copy()
    buffer, length = _utf8(json.dumps({"to": to_crs}))
    code = library.gpuwm_static_highres_transform_points(
        buffer, length,
        xs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ys.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), xs.size)
    if code != 0:
        raise ValueError(last_error(library))
    return xs, ys


class StaticCoverageRefusal(ValueError):
    """The crate refused because the footprint reaches past the source.

    Distinct from every other refusal because the caller's ``on_refuse``
    policy is allowed to answer it with the unchanged 30-arc-second
    baseline, and is NOT allowed to answer a decode fault that way.  The
    seam reports the distinction as its own return code (-2), so this is
    a fact off the wire rather than a sentence somebody pattern-matched.
    """


def highres_derive_window(request: dict) -> dict:
    """Derive one cached raster window (mosaic/clip/fill); returns the
    crate's audit document.  The GeoTIFF lands at ``request['out_path']``.

    Raises :class:`StaticCoverageRefusal` for a source-coverage refusal
    and :class:`ValueError` for anything else.
    """
    library = load()
    buffer, length = _utf8(json.dumps(request))
    # ONE call, not a length probe followed by a real one: this entry
    # point WRITES the derived GeoTIFF as a side effect, so probing
    # would derive the window twice.  The audit is a flat document of a
    # dozen scalars; 64 KiB is three orders of magnitude of headroom and
    # an overflow is refused rather than truncated.
    cap = 65536
    out = (ctypes.c_uint8 * cap)()
    written = int(library.gpuwm_static_highres_derive_window(
        buffer, length, out, cap))
    if written == -2:
        raise StaticCoverageRefusal(last_error(library))
    if written < 0:
        raise ValueError(last_error(library))
    if written > cap:
        raise StaticBridgeError(
            f"derive_window audit is {written} bytes, beyond the {cap}-byte "
            "buffer; the window was written but its audit was truncated")
    return json.loads(bytes(out[:written]).decode("utf-8"))


# ---------------------------------------------------------------------------
# Terrain smoothing (woof.static.terrain_smoothing).  Bound on first use,
# never in _bind_entry_points, so a library staged before these entries
# existed keeps loading and serving every default build.
# ---------------------------------------------------------------------------

def _smoothing_entry(name, argtypes):
    library = load()
    try:
        entry = getattr(library, name)
    except AttributeError:
        raise StaticBridgeError(
            "the staged static-fields library predates terrain-smoothing "
            "options; restage the rebuilt bridge or rebuild from this "
            "checkout: " + _checkout_build_command()) from None
    entry.argtypes = argtypes
    entry.restype = ctypes.c_int32
    return library, entry


#: Exported by a library whose smoothing JSON takes ``smooth_precision``
#: (``"wps-float32"`` on the default smoother).  A library without it
#: refuses the key as an unknown field; asking first names the remedy.
SMOOTHING_PRECISION_MARKER: Final[str] = (
    "gpuwm_static_terrain_smoothing_precision_v1")


def _smoothing_policy(library, smoothing):
    """``smoothing``'s JSON echo for the library, which must read it."""
    echo = smoothing.echo()
    if "smooth_precision" in echo and not hasattr(
            library, SMOOTHING_PRECISION_MARKER):
        raise StaticBridgeError(
            "the staged static-fields library predates terrain-smoothing "
            f"options for smooth_precision ({smoothing.label()}); restage "
            "the rebuilt bridge or rebuild from this checkout: "
            + _checkout_build_command())
    return _utf8(json.dumps(echo))


def _build_smoothed(name, grid_handle, payload, halo, smoothing):
    u8p = ctypes.POINTER(ctypes.c_uint8)
    library, entry = _smoothing_entry(name, [
        ctypes.c_uint64, u8p, ctypes.c_size_t, ctypes.c_uint32, u8p,
        ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint64)])
    buffer, length = _utf8(payload)
    policy, policy_len = _smoothing_policy(library, smoothing)
    result = ctypes.c_uint64(0)
    _check(library, entry(grid_handle, buffer, length, halo, policy,
                          policy_len, ctypes.byref(result)), name)
    return result.value


def build_fields_smoothed(grid_handle, geog_paths, halo, smoothing):
    """:func:`build_fields` with a non-default terrain smoothing."""
    return _build_smoothed(
        "gpuwm_static_build_fields_smoothed", grid_handle,
        json.dumps({k: str(v) for k, v in geog_paths.items()}), halo,
        smoothing)


def build_terrain_smoothed(grid_handle, path, halo, smoothing):
    """:func:`build_terrain` with a non-default terrain smoothing."""
    result = _build_smoothed("gpuwm_static_build_terrain_smoothed",
                             grid_handle, str(path), halo, smoothing)
    try:
        return fieldset_to_dict(result)["HGT_M"]
    finally:
        fieldset_free(result)


def terrain_smooth(extended, smoothing):
    """One halo-extended float64 plane through ``smoothing``."""
    f64p = ctypes.POINTER(ctypes.c_double)
    library, entry = _smoothing_entry("gpuwm_static_terrain_smooth", [
        f64p, ctypes.c_uint64, ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t, f64p])
    a = np.ascontiguousarray(extended, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(
            "terrain smoothing takes one two-dimensional extended plane")
    out = np.empty_like(a)
    policy, policy_len = _smoothing_policy(library, smoothing)
    _check(library, entry(a.ctypes.data_as(f64p), *a.shape, policy,
                          policy_len, out.ctypes.data_as(f64p)),
           "terrain_smooth")
    return out
