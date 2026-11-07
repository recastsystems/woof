"""ctypes transport for Rust observation scoring, without a Python math path."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np

OBSSCORE_ABI = 1
OBSSCORE_BRIDGE_ENV = "WOOF_OBSSCORE_BRIDGE"
ABI_MARKER = b"gpuwm_obsscore_masked_fss"
_LIBRARY: ctypes.CDLL | None = None


class ObsScoreBridgeError(RuntimeError):
    """The installed observation scoring library has an incompatible ABI."""


def library_names() -> tuple[str, ...]:
    if os.name == "nt":
        return ("obs_score.dll",)
    if os.uname().sysname == "Darwin":
        return ("libobs_score.dylib",)
    return ("libobs_score.so",)


def library_candidates() -> tuple[Path, ...]:
    from woof.bridges import (default_bridge_dir, legacy_bridge_candidates,
                               packaged_bridge_dir)
    from woof.rustwx import crate_dir

    filename = library_names()[0]
    override = os.environ.get(OBSSCORE_BRIDGE_ENV)
    candidates = [Path(override)] if override else []
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


def resolve_obsscore_bridge() -> Path:
    from woof.bridges import accept_resolved

    override = os.environ.get(OBSSCORE_BRIDGE_ENV)
    for candidate in library_candidates():
        if candidate.is_file():
            return accept_resolved(candidate.resolve(), executable=False)
        if override and candidate == Path(override):
            raise FileNotFoundError(
                f"{OBSSCORE_BRIDGE_ENV} names a missing file: {candidate}")
    raise FileNotFoundError(
        "the Rust observation scoring library was not found; scoring needs "
        "its native masked neighborhood and reduction kernels. Stage it with "
        "woof fetch-bridges or build tools/rustwx with cargo build --release "
        "-p obs-score --offline; searched:\n  "
        + "\n  ".join(str(path) for path in library_candidates()))


def load() -> ctypes.CDLL:
    global _LIBRARY
    if _LIBRARY is not None:
        return _LIBRARY
    path = resolve_obsscore_bridge()
    library = ctypes.CDLL(str(path))
    version_probe = _export(library, "gpuwm_obsscore_abi_version", path)
    version_probe.argtypes = []
    version_probe.restype = ctypes.c_uint32
    version = int(version_probe())
    if version != OBSSCORE_ABI:
        raise ObsScoreBridgeError(
            f"{path} speaks obs-score ABI {version}, this package needs "
            f"{OBSSCORE_ABI}; rebuild tools/rustwx")
    f64 = ctypes.POINTER(ctypes.c_double)
    u8 = ctypes.POINTER(ctypes.c_uint8)
    u64 = ctypes.POINTER(ctypes.c_uint64)
    i64 = ctypes.POINTER(ctypes.c_int64)
    size = ctypes.c_size_t
    code = ctypes.c_uint32
    value = ctypes.c_double
    signatures = {
        "boxcar": [f64, size, size, size, code, f64],
        "fraction": [u8, u8, size, size, size, code, f64, f64],
        "threshold": [f64, u8, size, value, f64],
        "masked_fss": [f64, f64, u8, u8, size, size, value, size, code,
                       code, size, f64, u64],
        "contingency": [f64, f64, u8, size, value, u64],
        "contingency_scores": [i64, f64, u8],
        "reduce": [f64, size, size, f64],
        "residuals": [f64, f64, size, f64, u64],
        "sample": [f64, size, size, value, value, code, f64],
        "shared_validity": [u8, size, size, u8],
        "match_reports": [i64, u64, size, i64, size, value, i64],
        "screen_report": [f64, u8, u8],
        "screen_reports": [f64, u8, size, u8],
    }
    for name, signature in signatures.items():
        function = _export(library, "gpuwm_obsscore_" + name, path)
        function.argtypes = signature
        function.restype = ctypes.c_int32
    last_error = _export(library, "gpuwm_obsscore_last_error", path)
    last_error.argtypes = [u8, size]
    last_error.restype = size
    _LIBRARY = library
    return library


def _export(library, name: str, path: Path):
    try:
        return getattr(library, name)
    except AttributeError as error:
        raise ObsScoreBridgeError(
            f"{path} is missing {name}; rebuild tools/rustwx so observation "
            "scoring uses a complete obs-score ABI") from error


def unavailable_reason() -> str | None:
    try:
        load()
    except (FileNotFoundError, OSError, ObsScoreBridgeError) as error:
        return f"{type(error).__name__}: {error}"
    return None


def _check(library, status: int) -> None:
    if status:
        length = int(library.gpuwm_obsscore_last_error(None, 0))
        buffer = (ctypes.c_uint8 * length)()
        library.gpuwm_obsscore_last_error(buffer, length)
        raise ValueError(bytes(buffer).decode("utf-8", "replace"))


def _array(value, dtype=np.float64):
    return np.ascontiguousarray(np.asarray(value, dtype=dtype))


def _ptr(array):
    return array.ctypes.data_as(ctypes.POINTER({
        np.dtype("float64"): ctypes.c_double,
        np.dtype("uint8"): ctypes.c_uint8,
        np.dtype("uint64"): ctypes.c_uint64,
        np.dtype("int64"): ctypes.c_int64,
    }[array.dtype]))


def shared_validity(masks) -> np.ndarray:
    stacked = _array(np.stack(masks), np.uint8)
    result = np.empty(np.asarray(masks[0]).shape, dtype=np.uint8)
    library = load()
    _check(library, library.gpuwm_obsscore_shared_validity(
        _ptr(stacked), result.size, len(masks), _ptr(result)))
    return result.view(np.bool_)


def boxcar(field, half_width: int, boundary: str) -> np.ndarray:
    array = _array(field)
    result = np.empty(array.shape, dtype=np.float64)
    library = load()
    _check(library, library.gpuwm_obsscore_boxcar(
        _ptr(array), *array.shape, half_width, int(boundary == "edge"),
        _ptr(result)))
    return result


def neighborhood_fraction(events, valid, half_width: int,
                          boundary: str) -> tuple[np.ndarray, np.ndarray]:
    events = _array(events, np.uint8)
    valid = _array(valid, np.uint8)
    fraction = np.empty(events.shape, dtype=np.float64)
    count = np.empty(events.shape, dtype=np.float64)
    library = load()
    _check(library, library.gpuwm_obsscore_fraction(
        _ptr(events), _ptr(valid), *events.shape, half_width,
        int(boundary == "edge"), _ptr(fraction), _ptr(count)))
    return fraction, count


def frequency_threshold(field, valid, target: float) -> float:
    field = _array(field)
    valid = _array(valid, np.uint8)
    if field.shape != valid.shape:
        raise IndexError("the frequency matching mask must match the field")
    result = np.empty(1, dtype=np.float64)
    library = load()
    _check(library, library.gpuwm_obsscore_threshold(
        _ptr(field), _ptr(valid), field.size, target, _ptr(result)))
    return float(result[0])


def _reduction_buffer(values: np.ndarray) -> int:
    """Use NumPy's reduction chunking for this contiguous float64 input.

    The grow-inner reduction iterator honors the transfer buffer on NumPy
    2.2 but spans a contiguous operand on newer builds. Ask that iterator
    for its chunk, without doing any scoring in Python or changing errstate.
    """
    if not values.size:
        return 1
    accumulator = np.zeros((), dtype=np.float64)
    with np.nditer(
        [accumulator, values.reshape(-1)],
        flags=["reduce_ok", "external_loop", "buffered", "grow_inner",
               "zerosize_ok"],
        op_flags=[["readwrite"], ["readonly"]],
        op_axes=[[-1], [0]], buffersize=np.getbufsize(),
    ) as chunks:
        return int(next(chunks)[1].size)


def masked_fss(model, obs, valid, scored, threshold: float, half_width: int,
               boundary: str, frequency_matched: bool):
    model = _array(model)
    obs = _array(obs)
    valid = _array(valid, np.uint8)
    scored = _array(scored, np.uint8)
    values = np.empty(6, dtype=np.float64)
    cells = np.empty(1, dtype=np.uint64)
    library = load()
    _check(library, library.gpuwm_obsscore_masked_fss(
        _ptr(model), _ptr(obs), _ptr(valid), _ptr(scored), *model.shape,
        threshold, half_width, int(boundary == "edge"),
        int(frequency_matched), _reduction_buffer(model), _ptr(values), _ptr(cells)))
    return values, int(cells[0])


def contingency_table(observed, forecast, valid, threshold: float):
    observed = _array(observed)
    forecast = _array(forecast)
    valid = _array(valid, np.uint8)
    result = np.empty(4, dtype=np.uint64)
    library = load()
    _check(library, library.gpuwm_obsscore_contingency(
        _ptr(observed), _ptr(forecast), _ptr(valid), observed.size, threshold,
        _ptr(result)))
    return tuple(int(value) for value in result)


def contingency_scores(counts):
    counts = _array(counts, np.int64)
    result = np.empty(8, dtype=np.float64)
    defined = np.empty(8, dtype=np.uint8)
    library = load()
    _check(library, library.gpuwm_obsscore_contingency_scores(
        _ptr(counts), _ptr(result), _ptr(defined)))
    return tuple(float(value) if flag else None
                 for value, flag in zip(result, defined))


def reduce(values):
    values = _array(values)
    result = np.empty(3, dtype=np.float64)
    library = load()
    _check(library, library.gpuwm_obsscore_reduce(
        _ptr(values), values.size, _reduction_buffer(values), _ptr(result)))
    return tuple(float(value) for value in result)


def residuals(forecast, observations):
    forecast = _array(forecast)
    observations = _array(observations)
    result = np.empty(forecast.shape, dtype=np.float64)
    index = np.empty(1, dtype=np.uint64)
    library = load()
    status = library.gpuwm_obsscore_residuals(
        _ptr(forecast), _ptr(observations), forecast.size, _ptr(result),
        _ptr(index))
    if status and int(index[0]) != 2 ** 64 - 1:
        return result, int(index[0])
    _check(library, status)
    return result, None


def sample(field, x: float, y: float, method: str) -> float:
    field = _array(field)
    result = np.empty(1, dtype=np.float64)
    library = load()
    _check(library, library.gpuwm_obsscore_sample(
        _ptr(field), *field.shape, x, y, int(method == "nearest"),
        _ptr(result)))
    return float(result[0])


def match_reports(report_seconds, tie_order, target_seconds,
                  tolerance: float):
    reports = _array(report_seconds, np.int64)
    order = _array(tie_order, np.uint64)
    targets = _array(target_seconds, np.int64)
    result = np.empty(targets.shape, dtype=np.int64)
    library = load()
    _check(library, library.gpuwm_obsscore_match_reports(
        _ptr(reports), _ptr(order), reports.size, _ptr(targets), targets.size,
        tolerance, _ptr(result)))
    return result


def screen_report(values, present):
    values = _array(values)
    present = _array(present, np.uint8)
    result = np.empty(3, dtype=np.uint8)
    library = load()
    _check(library, library.gpuwm_obsscore_screen_report(
        _ptr(values), _ptr(present), _ptr(result)))
    return result


def screen_reports(values, present):
    values = _array(values)
    present = _array(present, np.uint8)
    result = np.empty(values.shape, dtype=np.uint8)
    library = load()
    _check(library, library.gpuwm_obsscore_screen_reports(
        _ptr(values), _ptr(present), len(values), _ptr(result)))
    return result
