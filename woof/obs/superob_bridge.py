"""ctypes seam onto the Rust radar superob (``rw-superob``).

The library is ``tools/rustwx/crates/rw-superob``, the port of
:mod:`woof.obs.superob`: gate geometry, placement on the target grid,
the per-cell accumulators, correlation-coefficient QC, the four velocity
alias masks, the clear-air census, the multi-radar merge, the error model
and the windowed (v2) velocity layout -- and, linked into the same
library, the region-global dealias of each sweep including the per-sector
solve and reconciliation of a mixed-Nyquist sweep.

Why it exists, named: the whole stage was numpy, and the Python boundary
law (Python is orchestration and CUDA driver code only; every data-path
transform is Rust) does not allow a data path there.

Default-on, and refused rather than degraded.  A bare run superobs through
this library; if the library cannot be found the stage REFUSES with the
remedy, because falling back to numpy in silence would put the data path
back under Python on exactly the installs nobody checked.  The Python
module stays as the parity reference and as an explicit, stated opt-out:
``WOOF_SUPEROB_PYTHON=1``.

Why ctypes rather than pyo3: the ruling every woof cdylib follows (see
:mod:`woof.obs_regrid_bridge`) -- one loading discipline, one staging
path, one ABI-marker rule.  The seam is one entry point taking a JSON
request and a table of raw buffers; every float in the JSON is its
``repr`` string, parsed on the Rust side with a correctly rounded parser,
so no threshold changes by a bit crossing it.
"""

from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import threading
from typing import Final

import numpy as np

#: The C ABI version this module speaks.
SUPEROB_ABI: Final[int] = 1

#: Environment override for the library path, first rung of the ladder.
SUPEROB_BRIDGE_ENV: Final[str] = "WOOF_SUPEROB_BRIDGE"

#: Explicit opt-out to the numpy reference (a stated workaround).
SUPEROB_PYTHON_ENV: Final[str] = "WOOF_SUPEROB_PYTHON"

#: The exported symbol that identifies this contract, for
#: :data:`woof.bridges.BRIDGE_ABI_MARKERS`.
ABI_MARKER: Final[bytes] = b"gpuwm_superob_call"

_DTYPE_CODES: Final[dict[str, int]] = {
    "float32": 1, "float64": 2, "uint8": 3, "int8": 4, "int32": 5,
    "int64": 6, "int16": 7,
}


class SuperobBridgeError(RuntimeError):
    """The Rust superob refused a request, with its own message."""


class SuperobBridgeMissing(RuntimeError):
    """The Rust superob library is not on this install.

    A refusal, not a fallback: the numpy reference is the Python data path
    the boundary law retired, so it runs only when a caller says so.
    """


def python_reference_requested() -> bool:
    """Has the caller explicitly opted into the numpy reference?"""
    return os.environ.get(SUPEROB_PYTHON_ENV, "").strip() not in ("", "0")


def library_name() -> str:
    if os.name == "nt":
        return "rw_superob.dll"
    if os.uname().sysname == "Darwin":     # pragma: no cover - platform
        return "librw_superob.dylib"
    return "librw_superob.so"


def library_candidates() -> tuple[Path, ...]:
    """Deterministic candidate paths, best first (the shared ladder)."""
    from woof.bridges import default_bridge_dir, packaged_bridge_dir
    from woof.rustwx import crate_dir

    filename = library_name()
    candidates: list[Path] = []
    override = os.environ.get(SUPEROB_BRIDGE_ENV)
    if override:
        candidates.append(Path(override))
    root = Path(__file__).resolve().parent.parent.parent
    candidates.extend((
        crate_dir() / "target" / "release" / filename,
        crate_dir() / "target" / "debug" / filename,
        root / "libexec" / "bridges" / filename,
        packaged_bridge_dir() / filename,
        default_bridge_dir() / filename,
    ))
    return tuple(candidates)


def resolve_superob_bridge() -> Path:
    """First existing candidate, or a refusal naming every path searched."""
    override = os.environ.get(SUPEROB_BRIDGE_ENV)
    for attempt in (0, 1):
        for candidate in library_candidates():
            if candidate.is_file():
                from woof.bridges import accept_resolved

                return accept_resolved(candidate.resolve(), executable=False)
            if override and candidate == Path(override):
                raise SuperobBridgeMissing(
                    f"{SUPEROB_BRIDGE_ENV} names a missing file: {candidate}")
        # On demand since WOOF 1.1.6 (PyPI per-file limit): staged by the first
        # radar observation build from this release's bundle (never inside
        # bridges.inspection_only(), so doctor only reports).
        from woof.bridges import fetch_on_demand
        if attempt or override or not fetch_on_demand("rw_superob"):
            break
    rendered = "\n  ".join(str(c) for c in library_candidates())
    from woof import bridges

    separator = ";" if bridges.WINDOWS_SHELL else " &&"
    raise SuperobBridgeMissing(
        "the Rust radar superob library (rw-superob) was not found, and the "
        "superob stage does not fall back to numpy on its own: that is the "
        "Python data path the boundary law retired.  Searched:\n  "
        + rendered
        + "\n  # stage it with the rest of the bundle:\n"
        "  woof fetch-bridges\n"
        "  # or build it from a checkout:\n"
        f"  cd tools/rustwx{separator} cargo build --release -p rw-superob "
        f"--offline{separator} cd ../..\n"
        f"  # or run the numpy reference on purpose: {SUPEROB_PYTHON_ENV}=1")


class _SoArray(ctypes.Structure):
    _fields_ = [("ptr", ctypes.c_void_p), ("len", ctypes.c_uint64),
                ("dtype", ctypes.c_uint32), ("writable", ctypes.c_uint32)]


class SuperobLibrary:
    """A loaded, ABI-checked ``rw-superob`` library."""

    def __init__(self, path: Path):
        self.path = Path(path)
        library = ctypes.CDLL(str(self.path))
        library.gpuwm_superob_abi_version.argtypes = []
        library.gpuwm_superob_abi_version.restype = ctypes.c_uint32
        observed = int(library.gpuwm_superob_abi_version())
        if observed != SUPEROB_ABI:
            raise SuperobBridgeMissing(
                f"{self.path} reports superob ABI {observed}, this build "
                f"speaks {SUPEROB_ABI}; rebuild rw-superob from this checkout")
        library.gpuwm_superob_call.argtypes = [
            ctypes.c_char_p, ctypes.c_size_t, ctypes.POINTER(_SoArray),
            ctypes.c_size_t]
        library.gpuwm_superob_call.restype = ctypes.c_void_p
        library.gpuwm_superob_free.argtypes = [ctypes.c_void_p]
        library.gpuwm_superob_free.restype = None
        self._library = library

    def call(self, request: dict, arrays: list[tuple[np.ndarray, bool]]
             ) -> dict:
        """Run one op; ``arrays`` are ``(array, writable)`` in table order."""
        table = (_SoArray * max(1, len(arrays)))()
        for slot, (array, writable) in enumerate(arrays):
            code = _DTYPE_CODES.get(array.dtype.name)
            if code is None:
                raise SuperobBridgeError(
                    f"array {slot} has dtype {array.dtype}, which the seam "
                    "does not carry")
            if not (array.flags.c_contiguous and array.flags.aligned):
                raise SuperobBridgeError(
                    f"array {slot} is not C-contiguous and aligned")
            if writable and not array.flags.writeable:
                raise SuperobBridgeError(f"array {slot} is read-only")
            table[slot] = _SoArray(array.ctypes.data if array.size else None,
                                   array.size, code, 1 if writable else 0)
        payload = json.dumps(request, allow_nan=False).encode("utf-8")
        answer = self._library.gpuwm_superob_call(
            payload, len(payload), table, len(arrays))
        if not answer:
            raise SuperobBridgeError("rw-superob returned no answer")
        try:
            text = ctypes.string_at(answer).decode("utf-8")
        finally:
            self._library.gpuwm_superob_free(answer)
        return json.loads(text)


_LIBRARY: SuperobLibrary | None = None
_LOCK = threading.Lock()


def library() -> SuperobLibrary:
    """The loaded library, opened once per process (refuses if missing)."""
    global _LIBRARY
    with _LOCK:
        if _LIBRARY is None:
            _LIBRARY = SuperobLibrary(resolve_superob_bridge())
        return _LIBRARY


def route() -> SuperobLibrary | None:
    """The library when the Rust path is in force, ``None`` on the opt-out.

    Raises :class:`SuperobBridgeMissing` when the library is absent and the
    caller did not opt out.  That is the point: the default never degrades.
    """
    if python_reference_requested():
        return None
    return library()


def unavailable_reason() -> str | None:
    """Why the Rust path cannot run here, or None."""
    try:
        library()
    except (SuperobBridgeMissing, OSError) as error:
        return str(error)
    return None


def configure_threads(threads: int) -> dict:
    """Initialize the shared native pool without changing subprocess budgets."""
    try:
        answer = library().call({'op': 'thread_budget', 'threads': int(threads)}, [])
    except SuperobBridgeError as error:
        raise SuperobBridgeMissing('native thread-budget operation is unavailable; the legacy two-thread shared pool prevents site parallelism. Rebuild rw-superob.') from error
    if not answer.get('ok'):
        raise SuperobBridgeMissing('native thread-budget operation refused: ' + str(answer.get('message'))
            + '; the legacy shared pool prevents site parallelism. Rebuild rw-superob.')
    return answer


# ---------------------------------------------------------------------------
# request building
# ---------------------------------------------------------------------------

def f(value) -> str | None:
    """A float as the exact string the Rust side parses back to its bits."""
    if value is None:
        return None
    return repr(float(value))


class _Table:
    """Collects buffers and hands out their indices."""

    def __init__(self):
        self.arrays: list[tuple[np.ndarray, bool]] = []

    def read(self, array, dtype) -> int:
        array = np.require(np.asarray(array), dtype=dtype,
                           requirements=["C", "A"])
        self.arrays.append((array, False))
        return len(self.arrays) - 1

    def write(self, array: np.ndarray) -> int:
        if not (array.flags.c_contiguous and array.flags.writeable):
            raise SuperobBridgeError("an output buffer must be contiguous")
        self.arrays.append((array, True))
        return len(self.arrays) - 1


def _answer(lib: SuperobLibrary, request: dict, table: _Table) -> dict:
    answer = lib.call(request, table.arrays)
    if not answer.get("ok"):
        kind = answer.get("kind")
        message = answer.get("message", "no message")
        if kind == "window":
            from woof.obs.superob import SuperobError  # noqa: PLC0415

            raise SuperobError(message)
        if kind == "dealias_params":
            from woof.obs.dealias import DealiasParamsError  # noqa: PLC0415

            raise DealiasParamsError(message)
        if kind == "region_dealias":
            from woof.obs.dealias_region import (  # noqa: PLC0415
                RegionDealiasError)

            raise RegionDealiasError(message)
        if kind == "request":
            raise ValueError(f"rw-superob refused the request: {message}")
        raise SuperobBridgeError(f"rw-superob {kind}: {message}")
    return answer


_PARAM_FIELDS = (
    "nyquist_reject_fraction", "nyquist_min_ms", "nyquist_max_ms",
    "nyquist_spread_fraction", "shear_fold_fraction", "min_reflectivity_dbz",
    "max_range_km", "max_elevation_deg", "z_error_base_dbz",
    "vr_error_base_ms", "z_error_floor_dbz", "vr_error_floor_ms",
    "refraction_factor", "earth_radius_m", "clear_air_min_gates",
    "clear_air_error_dbz")


def params_request(params) -> dict:
    return {name: f(getattr(params, name)) for name in _PARAM_FIELDS}


def cc_request(cc) -> dict | None:
    if cc is None:
        return None
    payload = cc.to_payload()
    out = {}
    for key, value in payload.items():
        if isinstance(value, bool) or value is None or isinstance(value, int):
            out[key] = value
        else:
            out[key] = f(value)
    for key in ("rho_min_velocity", "rho_floor"):
        out[key] = f(payload[key])
    return out


def grid_request(grid, table: _Table) -> dict:
    """The target grid as the Rust placement needs it.

    Only a :class:`~woof.obs.target_grid.TargetGrid` (a projection plus
    column interfaces) can be placed natively; any other object's
    placement methods are Python and cannot be called from Rust.
    """
    projection = getattr(grid, "projection", None)
    z_w = getattr(grid, "z_w", None)
    if projection is None or z_w is None or not hasattr(projection,
                                                        "_rust_spec"):
        raise SuperobBridgeError(
            f"{type(grid).__name__} is not a TargetGrid: the Rust superob "
            "places gates with the grid's projection and column interfaces, "
            "and this object carries Python placement methods instead.  "
            f"Run the numpy reference for it explicitly ({SUPEROB_PYTHON_ENV}"
            "=1)")
    offsets = []
    while getattr(projection, "_translation_reference", None) is not None:
        di, dj = projection._translation_offset
        offsets.append([int(di), int(dj)])
        projection = projection._translation_reference
    spec = projection._rust_spec()
    spec = {key: (value if key in ("kind", "e_we", "e_sn") else f(value))
            for key, value in spec.items()}
    spec["e_we"] = int(spec["e_we"])
    spec["e_sn"] = int(spec["e_sn"])
    return {"spec": spec, "translations": list(reversed(offsets)),
            "nx": int(grid.nx), "ny": int(grid.ny), "nz": int(grid.nz),
            "z_w": table.read(z_w, np.float64)}


# ---------------------------------------------------------------------------
# operations
# ---------------------------------------------------------------------------

def horizontal_window(lib: SuperobLibrary, grid, site, params
                      ) -> tuple[int, int, int, int]:
    table = _Table()
    reach_m = float(params.max_range_km) * 1000.0 + 2 * max(
        float(grid.dx_m), float(grid.dy_m))
    request = {
        "op": "window",
        "lat": table.read(grid.lat, np.float64),
        "lon": table.read(grid.lon, np.float64),
        "ny": int(grid.ny), "nx": int(grid.nx),
        "site_lat_deg": f(site.lat_deg), "site_lon_deg": f(site.lon_deg),
        "earth_radius_m": f(params.earth_radius_m), "reach_m": f(reach_m),
    }
    answer = _answer(lib, request, table)
    return tuple(int(v) for v in answer["window"])


VOLUME_OUTPUTS = (
    ("z_linear_sum", np.float64), ("z_count", np.int64),
    ("z0_count", np.int64), ("z_max_dbz", np.float64),
    ("z_sumsq_dbz", np.float64), ("z_sum_dbz", np.float64),
    ("vr_sum", np.float64), ("vr_sumsq", np.float64),
    ("vr_count", np.int64), ("vr_min", np.float64), ("vr_max", np.float64),
    ("beam_east", np.float64), ("beam_north", np.float64),
    ("beam_up", np.float64), ("nyquist_min", np.float64),
    ("vr_rejected", np.int64))


def superob_volume(lib: SuperobLibrary, volume, grid, params, *,
                   window, clear_air_from_censor: bool, odim_census: bool,
                   dealiased: dict) -> tuple[dict, dict]:
    """Run the gridding pass natively.

    ``dealiased`` maps sweep position to ``(velocity, resolved)`` for every
    sweep the dealiaser ran on.  Returns ``(answer, accumulators)``.
    """
    table = _Table()
    j0, j1, i0, i1 = window
    shape = (int(grid.nz), j1 - j0 + 1, i1 - i0 + 1)
    accumulators = {name: np.empty(shape, dtype=dtype)
                    for name, dtype in VOLUME_OUTPUTS}
    sweeps = []
    for position, sweep in enumerate(volume.sweeps):
        moments = []
        for product, moment in sweep.moments.items():
            data = np.asarray(moment.data)
            entry = {
                "product": str(product),
                "gate_count": int(moment.gate_count),
                "columns": int(data.shape[1]) if data.ndim == 2 else -1,
                "first_gate_range_m": f(moment.first_gate_range_m),
                "gate_size_m": f(moment.gate_size_m),
                "data": table.read(data, np.float64),
            }
            if clear_air_from_censor and moment.censor is not None:
                entry["censor"] = table.read(moment.censor, np.uint8)
            if product == "VEL" and position in dealiased:
                velocity, resolved = dealiased[position]
                entry["dealiased"] = table.read(velocity, np.float64)
                entry["resolved"] = table.read(
                    np.asarray(resolved, dtype=bool).view(np.uint8),
                    np.uint8)
            moments.append(entry)
        sweeps.append({
            "sweep_index": int(sweep.sweep_index),
            "elevation_angle_deg": f(sweep.elevation_angle_deg),
            "nyquist_velocity_ms": f(sweep.nyquist_velocity_ms),
            "nyquist_radials_disagree": bool(sweep.nyquist_radials_disagree),
            "azimuth": table.read(sweep.azimuth_deg, np.float64),
            "elevation": table.read(sweep.elevation_deg, np.float64),
            "moments": moments,
        })
    dealias = None
    if params.dealias is not None:
        dealias = {"keep_beyond_reject_fraction":
                   bool(params.dealias.keep_beyond_reject_fraction),
                   "max_speed_ms": f(params.dealias.max_speed_ms)}
    request = {
        "op": "volume",
        "params": params_request(params),
        "dealias": dealias,
        "cc_qc": cc_request(params.cc_qc),
        "clear_air_from_censor": bool(clear_air_from_censor),
        "odim_census": bool(odim_census),
        "site": {"id": str(volume.site.id), "lat_deg": f(volume.site.lat_deg),
                 "lon_deg": f(volume.site.lon_deg),
                 "alt_m": f(volume.site.alt_m)},
        "grid": grid_request(grid, table),
        "window": [int(j0), int(j1), int(i0), int(i1)],
        "sweeps": sweeps,
        "outputs": {name: table.write(array)
                    for name, array in accumulators.items()},
    }
    return _answer(lib, request, table), accumulators


MERGE_INPUTS = (
    ("z_linear_sum", np.float64), ("z_count", np.int64),
    ("z0_count", np.int64), ("z_sum_dbz", np.float64),
    ("z_sumsq_dbz", np.float64), ("z_max_dbz", np.float64),
    ("vr_sum", np.float64), ("vr_sumsq", np.float64),
    ("vr_count", np.int64), ("beam_east", np.float64),
    ("beam_north", np.float64), ("beam_up", np.float64),
    ("vr_rejected", np.int64))


def merge(lib: SuperobLibrary, contributions, grid, params, *,
          z_reduce: str, windows) -> dict:
    """Reduce contributions natively; returns the output arrays by name."""
    table = _Table()
    shape = (int(grid.nz), int(grid.ny), int(grid.nx))
    max_nj = max(j1 - j0 + 1 for j0, j1, _, _ in windows)
    max_ni = max(i1 - i0 + 1 for _, _, i0, i1 in windows)
    vshape = (len(contributions), int(grid.nz), max_nj, max_ni)
    entries = []
    for contribution, window in zip(contributions, windows):
        entry = {"window": [int(v) for v in window]}
        for name, dtype in MERGE_INPUTS:
            entry[name] = table.read(getattr(contribution, name), dtype)
        entries.append(entry)
    outputs = {
        "z_obs": np.empty(shape, np.float64),
        "z_mask": np.empty(shape, np.int8),
        "z_err": np.empty(shape, np.float64),
        "z_max": np.empty(shape, np.float64),
        "z_mean": np.empty(shape, np.float64),
        "z_count": np.empty(shape, np.int32),
        "z0_mask": np.empty(shape, np.int8),
        "z0_count": np.empty(shape, np.int32),
        "z0_err": np.empty(shape, np.float64),
        "vr_obs": np.empty(vshape, np.float64),
        "vr_mask": np.empty(vshape, np.int8),
        "vr_err": np.empty(vshape, np.float64),
        "vr_count": np.empty(vshape, np.int32),
        "vr_rejected": np.empty(vshape, np.int32),
        "vr_beam_east": np.empty(vshape, np.float64),
        "vr_beam_north": np.empty(vshape, np.float64),
        "vr_beam_up": np.empty(vshape, np.float64),
        "vr_beam_coherence": np.empty(vshape, np.float64),
    }
    request = {
        "op": "merge",
        "params": params_request(params),
        "z_reduce": str(z_reduce),
        "nz": shape[0], "ny": shape[1], "nx": shape[2],
        "max_nj": max_nj, "max_ni": max_ni,
        "contributions": entries,
        "outputs": {name: table.write(array)
                    for name, array in outputs.items()},
    }
    _answer(lib, request, table)
    return outputs


def dealias_region(lib: SuperobLibrary, velocity, azimuth_deg, nyquist,
                   params, *, first_gate_m, gate_spacing_m,
                   nyquist_by_radial, nyquist_radials_disagree: bool):
    """One sweep through the linked region-global solver.

    Returns ``(output, state, reason, fold, stats)``; the stats carry the
    histogram and sector shifts as the reference spells them.
    """
    table = _Table()
    velocity = np.asarray(velocity, dtype=np.float64)
    rows, gates = velocity.shape
    planes = {"velocity": np.empty((rows, gates), np.float64),
              "state": np.empty((rows, gates), np.int8),
              "reason": np.empty((rows, gates), np.int8),
              "fold": np.empty((rows, gates), np.int16)}
    request = {
        "op": "dealias_region",
        "velocity": table.read(velocity, np.float64),
        "rows": int(rows), "gates": int(gates),
        "azimuth": table.read(np.asarray(azimuth_deg).ravel(), np.float64),
        "nyquist": f(nyquist),
        "nyquist_by_radial": (
            None if nyquist_by_radial is None
            else table.read(np.asarray(nyquist_by_radial).ravel(),
                            np.float64)),
        "nyquist_radials_disagree": bool(nyquist_radials_disagree),
        "max_speed_ms": f(getattr(params, "max_speed_ms", float("inf"))),
        "refinement": bool(getattr(params, "refinement", False)),
        "first_gate_m": f(first_gate_m),
        "gate_spacing_m": f(gate_spacing_m),
        "outputs": {name: table.write(array)
                    for name, array in planes.items()},
    }
    stats = _answer(lib, request, table)["stats"]
    stats["fold_histogram"] = {int(k): int(v)
                               for k, v in stats["fold_histogram"]}
    if "sector_shifts" in stats:
        stats["sector_shifts"] = {str(float(v)): int(k)
                                  for v, k in stats["sector_shifts"]}
    return (planes["velocity"], planes["state"], planes["reason"],
            planes["fold"], stats)


__all__ = [
    "ABI_MARKER", "SUPEROB_ABI", "SUPEROB_BRIDGE_ENV", "SUPEROB_PYTHON_ENV",
    "SuperobBridgeError", "SuperobBridgeMissing", "SuperobLibrary",
    "dealias_region", "horizontal_window", "library", "library_candidates",
    "merge", "python_reference_requested", "resolve_superob_bridge",
    "route", "superob_volume", "unavailable_reason",
]
