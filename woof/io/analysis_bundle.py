"""WRF analysis bundle, version 1: an analysed atmosphere a WOOF forecast can start from.

A bundle is one directory:

    manifest.json   schema "wrf-analysis-bundle/1" (below)
    analysis.nc     the analysed state at the valid time, WRF-named, on WOOF's own grid
    increment.nc    optional: analysis minus background, on WOOF's own staggering
    noda.nc         optional: the matched no-assimilation state at the same valid time

It is the one interface between whatever produced the analysis (an ensemble filter, a
variational system, a nudged run) and the forecast: the producer writes it, and
``woof run --wrfinput DIR --analysis BUNDLE`` reads it (:mod:`woof.da.analysis_door`).
Nothing here knows how the analysis was made.  The full description, field by field, is
``docs/analysis-bundle-wrf.md``.  The MPAS bundle of woof hex (``mpas-analysis-bundle/1``)
carries the same keys outside its identity block; this format's identity block is ``grid``
where that one's is ``mesh``.

``analysis.nc``
    ``analysis.form = "wrfinput"`` (the default): a real.exe-shaped initial-conditions file
    of the run's grid, the form ``woof run --wrfinput`` starts from, with ``Times`` at the
    valid time.  ``analysis.form = "wrfout"``: a history frame of the run (WOOF's or WRF's
    wrfout) at the valid time.  It carries no base-state inputs a cold start needs, so it
    starts a forecast only as a cycling restart (``--restart`` or ``--restart-from``), which
    adds the increment to the restored state.

``increment.nc``
    The analysis minus the background it was computed against, in WOOF's own prognostics
    on WOOF's own staggering, under the WRF names: ``U`` (bottom_top, south_north,
    west_east_stag), ``V``, ``W`` (bottom_top_stag ...), ``T`` (dry potential temperature
    perturbation), ``PH`` (perturbation geopotential), ``MU`` (perturbation dry column
    mass), ``QVAPOR`` (mixing ratio over dry air), and any other moisture species or number
    concentration the run carries.  The dycore core and vapour (:data:`REQUIRED_INCREMENT`)
    are required; write zeros for one the analysis left unchanged.  float64 (the default)
    or float32.  The background is ``analysis - increment``, formed in float64 and rounded
    to the analysis variable's type; exactness is measured per variable when the increment
    is written.  ``manifest.iau.window_s`` gives the window.

The manifest holds no paths and no names of places or cases: the grid is named by its
fingerprints, and every file by its SHA-256 and byte count.  ``notes`` (optional) is the
producer's own provenance: a reader records it and never acts on it.  ``forecast.args``
(optional) is the producer's own forecast configuration as ``woof run`` model flags
(:data:`FORECAST_FLAGS`; never a path or a host binding).  A reader refuses any file whose
bytes differ, and refuses a run whose grid, vertical coordinate or static fields differ
from the bundle's, by name.

``PH`` is the engine's, not the producer's: ``make`` re-integrates the analysed
geopotential with the engine's own hydrostatic operator (:mod:`woof.da.hydrostatic`) from
the ``T``, moisture and ``MU`` increments at the background's own dry mass (the default
``--ph balance``), so every bundle is balanced the same way, whatever produced it.

NetCDF bytes are read by the Rust decoder (:mod:`woof.netcdf_bridge`) and written by the
Rust classic writer (:mod:`woof.io.nc_writer_bridge`); bundle NetCDF files are classic
(CDF-1, CDF-2 or CDF-5), so a derived file (the background, an analysis made from an
increment) is a patched copy of its template with every other byte unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import contextvars
import functools
import json
from pathlib import Path
import re
import shutil
from typing import Any, Mapping, Sequence

import numpy as np

SCHEMA = "wrf-analysis-bundle/1"
MANIFEST = "manifest.json"
ANALYSIS = "analysis.nc"
INCREMENT = "increment.nc"
NODA = "noda.nc"
#: Every file name a v1 bundle may list, and whether it is required.
BUNDLE_FILES: dict[str, bool] = {ANALYSIS: True, INCREMENT: False, NODA: False}

ANALYSIS_KINDS = ("mean", "control", "member")
ANALYSIS_FORMS = ("wrfinput", "wrfout")
PHYSICS_MODES = ("correct", "exact")
#: WRF's own clock text (Times, START_DATE).
TIME_FORMAT = "%Y-%m-%d_%H:%M:%S"
#: What a producer writes in the manifest: UTC, ISO 8601.  A reader takes either form.
ISO_TIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

#: The dycore's own prognostics, by WRF name, with their staggered dimensions.
CORE_FIELDS: dict[str, tuple[str, ...]] = {
    "U": ("bottom_top", "south_north", "west_east_stag"),
    "V": ("bottom_top", "south_north_stag", "west_east"),
    "W": ("bottom_top_stag", "south_north", "west_east"),
    "T": ("bottom_top", "south_north", "west_east"),
    "PH": ("bottom_top_stag", "south_north", "west_east"),
    "MU": ("south_north", "west_east"),
}
MASS_LEVELS = ("bottom_top", "south_north", "west_east")
#: The increments every bundle carries (zeros where the analysis left a field alone): the
#: dycore core and vapour, so the background a reader rebuilds is the producer's in every
#: field the dynamics integrate.
REQUIRED_INCREMENT = tuple(CORE_FIELDS) + ("QVAPOR",)
#: Fields whose rebuilt background is clipped (float32 increment) or counted (float64)
#: below zero: mixing ratios, number concentrations and particle volumes.
_NON_NEGATIVE_PREFIX = "Q"


def host_threads() -> int:
    """Threads a bundle's host arithmetic may use: ``WOOF_BUNDLE_THREADS`` when set (a producer running several
    bundles at once gives each its share of the cores), else up to 8 of this process's CPUs."""

    import os

    try:
        asked = int(os.environ.get("WOOF_BUNDLE_THREADS") or 0)
    except ValueError:
        asked = 0
    if asked > 0:
        return asked
    try:
        cpus = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cpus = os.cpu_count() or 1
    return max(1, min(8, cpus))


def _ordered_map(function, rows, threads: int | None = None):
    """``function(row)`` for every row, yielded in row order.  With more than one thread the rows run on a pool
    (the Rust host kernels and NumPy release the GIL) with at most ``2 x threads`` rows held at once; every row's
    arithmetic is the one-thread loop's, so each result is that loop's to the last bit.  Single-core law
    (2026-10-27, DA cycle bundles: ~20 s of every member's bundle on one thread, the field loops one after another)."""

    threads = host_threads() if threads is None else max(1, int(threads))
    if threads == 1:
        for row in rows:
            yield function(row)
        return
    from concurrent.futures import ThreadPoolExecutor

    rows = iter(rows)
    with ThreadPoolExecutor(threads, thread_name_prefix="bundle-host") as pool:
        pending = []
        for row in rows:
            pending.append(pool.submit(function, row))
            if len(pending) >= 2 * threads:
                break
        while pending:
            result = pending.pop(0).result()
            for row in rows:
                pending.append(pool.submit(function, row))
                break
            yield result

#: What a wrfinput-form analysis carries for a cold start to read it (the WRF-input door's
#: reader checks the rest of the real.exe inventory by its own rules).
WRFINPUT_FORM_REQUIRED = ("Times", "U", "V", "W", "T", "PH", "MU", "PHB", "MUB", "P", "PB", "QVAPOR",
                          "XLAT", "XLONG", "HGT", "ZNW", "P_TOP", "MAPFAC_M", "MAPFAC_U", "MAPFAC_V",
                          "LANDMASK", "LU_INDEX")
#: What a wrfout-form analysis carries: the prognostics and the grid's identity.
WRFOUT_FORM_REQUIRED = ("Times", "U", "V", "W", "T", "PH", "MU", "QVAPOR", "XLAT", "XLONG", "HGT", "ZNW",
                        "P_TOP", "MAPFAC_M", "MAPFAC_U", "MAPFAC_V", "LANDMASK", "LU_INDEX")


def prognostic_dimensions(name: str) -> tuple[str, ...] | None:
    """The spatial dimensions of a prognostic an increment may carry, or None.

    The dycore core (:data:`CORE_FIELDS`) on its own staggering, and every moisture species,
    number concentration and particle volume the WRF-input door maps (the ``Q*`` names of
    :data:`woof.ingest.wrfinput.ALL_MOISTURE_WRFINPUT`) on the mass levels.
    """

    if name in CORE_FIELDS:
        return CORE_FIELDS[name]
    from woof.ingest.wrfinput import ALL_MOISTURE_WRFINPUT

    if name.startswith(_NON_NEGATIVE_PREFIX) and name in ALL_MOISTURE_WRFINPUT:
        return MASS_LEVELS
    return None


#: What identifies the run's horizontal grid: the geolocation and map factors, hashed as
#: little-endian float64 of the first record, so storage type does not change them.
HORIZONTAL_FIELDS = ("XLAT", "XLONG", "MAPFAC_M", "MAPFAC_U", "MAPFAC_V")
#: What identifies the static fields under the analysed columns (terrain and land).
STATIC_FIELDS = ("HGT", "LANDMASK", "LU_INDEX")
#: The projection, as the file's global attributes state it.
PROJECTION_ATTRIBUTES = ("MAP_PROJ", "TRUELAT1", "TRUELAT2", "STAND_LON", "MOAD_CEN_LAT",
                         "CEN_LAT", "CEN_LON")

_TOP_KEYS = {"schema", "valid_time", "grid", "static", "physics", "analysis", "iau", "versions", "files",
             "notes", "forecast"}
_REQUIRED_TOP = _TOP_KEYS - {"iau", "notes", "forecast"}
#: The model configuration a producer may hand a forecast (manifest ``forecast.args``): the
#: ``woof run --wrfinput`` flags that decide what model a forecast integrates, and how many
#: values each takes.  Never a host binding (devices, scratch, checkouts) and never a path.
FORECAST_FLAGS: dict[str, int] = {
    "--rrtmg-variant": 1, "--thompson-version": 1, "--urban-ucm-generation": 1,
    "--urban-treeoption": 1, "--urban-ttscheme": 1, "--urban-fvg": 1,
}
#: NetCDF containers a bundle file may use (``rw_netcdf`` inventory spelling): the classic
#: family, which the Rust classic writer patches byte for byte.
CLASSIC_FORMATS = ("Classic", "Offset64", "Cdf5")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+:-]{0,127}$")


class BundleRefusal(ValueError):
    """A bundle, or a run against it, that cannot be used; the message names why."""


def _refuse(message: str) -> BundleRefusal:
    return BundleRefusal(f"analysis bundle: {message}")


# ---------------------------------------------------------------------------
# hashing, reading, fingerprints
# ---------------------------------------------------------------------------
def sha256_file(path: str | Path) -> str:
    # Once per process for each unchanged file (woof.file_digest): a member leg's door checks the same
    # bundle, derived and checkpoint bytes several times between its plan and its worker.
    from woof.file_digest import sha256_file as digest_once
    return digest_once(path)


_RESIDENT_INPUTS = contextvars.ContextVar("analysis_bundle_resident_inputs", default=None)


def _reuse_input_reads(function):
    """Keep source/output decodes resident for one bundle, invalidating changed files."""
    @functools.wraps(function)
    def call(*args, **kwargs):
        paths = {str(Path(kwargs[name]).resolve()): None for name in ("background", "coordinate")
                 if kwargs.get(name) is not None}
        out = kwargs.get("out", args[0] if args else None)
        if out is not None:
            # The analysis is atomically replaced after PH balance. _open's inode,
            # size and timestamp check invalidates that first decode; subsequent
            # increment/fingerprint/reader checks share the final decoded arrays.
            paths[str((Path(out) / ANALYSIS).resolve())] = None
        token = _RESIDENT_INPUTS.set(paths)
        try:
            return function(*args, **kwargs)
        finally:
            _RESIDENT_INPUTS.reset(token)
    return call


def _open(path: str | Path):
    from woof import netcdf_bridge
    resident = _RESIDENT_INPUTS.get()
    if resident is not None:
        key = str(Path(path).resolve())
        if key in resident:
            stat = Path(path).stat()
            stamp = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
            entry = resident[key]
            if entry is None or entry[0] != stamp:
                entry = resident[key] = (stamp, netcdf_bridge.open_dataset(path))
            return entry[1]
    return netcdf_bridge.open_dataset(path)


def _spatial_dims(variable: Any) -> tuple[str, ...]:
    return tuple(d for d in variable.dimensions if d != "Time")


def _first_record(variable: Any) -> np.ndarray:
    """The variable's first record (or whole value without ``Time``), float64, unmasked."""

    variable.set_auto_mask(False)
    values = np.asarray(variable[...])
    if variable.dimensions and variable.dimensions[0] == "Time":
        values = values[0]
    return np.asarray(values, dtype=np.float64) if values.dtype.kind in "iuf" else values


def _prefetch_unmasked(dataset: Any, names: Sequence[str]) -> None:
    """Batch-decode ``names`` in one ``rw_netcdf`` run under the unmasked policy :func:`_first_record` reads with.

    ``prefetch`` caches each variable under its CURRENT mask policy, and ``_first_record`` then switches the
    variable to ``set_auto_mask(False)``, which drops a cache decoded masked.  So prefetching before that switch
    did nothing: every variable was decoded again in its own decoder process, one after another (93 process
    starts, about 5 s on one CPU thread, in every DA forecast leg's analysis door; measured 2026-10-16).
    Switching first makes the batch the read: the same decoder, the same flags, the same values.
    """

    for name in names:
        dataset.variables[name].set_auto_mask(False)
    dataset.prefetch(list(names))


def _times(dataset: Any) -> list[str]:
    if "Times" not in dataset.variables:
        return []
    variable = dataset.variables["Times"]
    variable.set_auto_mask(False)
    rows = np.asarray(variable[...])
    if rows.ndim == 1:
        rows = rows[None]
    return [b"".join(bytes(c) for c in row).decode("ascii", "replace").strip("\x00 ") for row in rows]


def _float_digest(dataset: Any, names: Sequence[str], what: str, path: Path) -> str:
    missing = [name for name in names if name not in dataset.variables]
    if missing:
        raise _refuse(f"{path.name} lacks {missing}, so its {what} cannot be identified")
    _prefetch_unmasked(dataset, list(names))
    return _float_digest_values((name, _first_record(dataset.variables[name])) for name in names)


def _float_digest_values(pairs) -> str:
    """SHA-256 over (name, values) pairs as little-endian float64: the one rule a file's fields
    and the same words held in memory are fingerprinted by."""

    digest = hashlib.sha256()
    for name, values in pairs:
        values = np.ascontiguousarray(np.asarray(values, dtype=np.float64), dtype="<f8")
        digest.update(name.encode("ascii") + b"\0" + repr(values.shape).encode("ascii") + b"\0")
        digest.update(values.tobytes())
    return digest.hexdigest()


def _attribute(dataset: Any, name: str, default: Any = None) -> Any:
    try:
        value = dataset.getncattr(name)
    except AttributeError:
        return default
    if isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    if isinstance(value, (np.generic,)):
        value = value.item()
    return value


def _number(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def file_format(path: str | Path) -> str:
    return str(_open(path).format)


def require_classic(path: str | Path, role: str) -> None:
    """Bundle NetCDF files are classic: the Rust writer patches them byte for byte."""

    found = file_format(path)
    if found not in CLASSIC_FORMATS:
        raise _refuse(f"{role} {Path(path).name} is a {found} NetCDF file; bundle files are NetCDF classic "
                      f"({', '.join(CLASSIC_FORMATS)}) so a derived file is a patched copy of its template "
                      "with every other byte unchanged.  Write it as classic (WRF io_form 2 with "
                      "NETCDF_classic=1, or a copy converted to CDF-2 or CDF-5)")


def grid_fingerprint(path: str | Path) -> dict[str, Any]:
    """The run grid a WRF-form file is on: sizes, projection, geolocation, vertical coordinate."""

    path = Path(path)
    dataset = _open(path)
    dims = {name: len(dim) for name, dim in dataset.dimensions.items()}
    for name in ("west_east", "south_north", "bottom_top"):
        if name not in dims:
            raise _refuse(f"{path.name} has no {name} dimension; it is not a WRF-form file")
    if "ZNW" not in dataset.variables:
        raise _refuse(f"{path.name} lacks ZNW, so its vertical coordinate cannot be identified")
    if "P_TOP" not in dataset.variables:
        raise _refuse(f"{path.name} lacks P_TOP, so its model top cannot be identified")
    return _grid_fingerprint(
        dims, _first_record(dataset.variables["ZNW"]), _first_record(dataset.variables["P_TOP"]),
        lambda name: _attribute(dataset, name),
        _float_digest(dataset, HORIZONTAL_FIELDS, "horizontal grid", path), path.name)


def _grid_fingerprint(dims: Mapping[str, int], znw: Any, p_top: Any, attribute, horizontal_sha256: str,
                      what: str) -> dict[str, Any]:
    """The grid block from its parts: one rule for a file and for the same words in memory."""

    znw = np.ascontiguousarray(np.asarray(znw, dtype=np.float64), dtype="<f8")
    p_top = float(np.asarray(p_top, dtype=np.float64).reshape(-1)[0])
    projection = {name.lower(): _number(attribute(name)) for name in PROJECTION_ATTRIBUTES}
    missing = [name for name in ("MAP_PROJ", "TRUELAT1", "TRUELAT2", "STAND_LON")
               if projection[name.lower()] is None]
    if missing:
        raise _refuse(f"{what} does not state {missing}, so its projection cannot be identified")
    projection["map_proj"] = int(projection["map_proj"])
    hybrid = attribute("HYBRID_OPT")
    etac = attribute("ETAC")
    return {
        "nx": int(dims["west_east"]), "ny": int(dims["south_north"]), "nz": int(dims["bottom_top"]),
        "dx_m": float(attribute("DX")), "dy_m": float(attribute("DY")),
        "projection": projection,
        "horizontal_sha256": horizontal_sha256,
        "vertical": {"znw_sha256": hashlib.sha256(znw.tobytes()).hexdigest(), "p_top_pa": p_top,
                     "hybrid_opt": None if hybrid is None else int(hybrid),
                     "etac": None if etac is None else float(etac)},
    }


def static_fingerprint(path: str | Path) -> str:
    """SHA-256 of the terrain and land fields under the analysed columns."""

    path = Path(path)
    return _float_digest(_open(path), STATIC_FIELDS, "static fields", path)


def describe_grid(grid_id: str, path: str | Path, *, geo_em_sha256: str | None = None) -> dict[str, Any]:
    """The manifest's ``grid`` block, measured from a WRF-form file of the grid."""

    measured = grid_fingerprint(path)
    return {"id": str(grid_id), "geo_em_sha256": geo_em_sha256, **measured}


# ---------------------------------------------------------------------------
# the manifest
# ---------------------------------------------------------------------------
def _no_paths(value: Any, where: str) -> None:
    """A manifest names things by hash, never by path or address (a path would tie a bundle to
    one machine and carry that machine's layout to every reader)."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            _no_paths(item, f"{where}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _no_paths(item, f"{where}[{index}]")
    elif isinstance(value, str):
        if (value.startswith(("/", "\\", "~")) or re.match(r"^[A-Za-z]:[\\/]", value)
                or "://" in value or "\\" in value):
            raise _refuse(f"{where} = {value!r} is a path or address; a manifest names files by "
                          "SHA-256 and holds no paths")


def _sha(value: Any, where: str) -> str:
    if not isinstance(value, str) or not _SHA256.match(value):
        raise _refuse(f"{where} must be a lowercase hex SHA-256, got {value!r}")
    return value


def _positive_int(value: Any, where: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise _refuse(f"{where} must be an integer >= {minimum}, got {value!r}")
    return value


def _positive_number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
        raise _refuse(f"{where} must be a positive number, got {value!r}")
    return float(value)


def _finite_number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise _refuse(f"{where} must be a finite number, got {value!r}")
    return float(value)


def _ident(value: Any, where: str) -> str:
    if not isinstance(value, str) or not _IDENT.match(value):
        raise _refuse(f"{where} must be a short identifier (letters, digits, . _ + : -), got {value!r}")
    return value


def _exact_keys(block: Any, where: str, required: set[str], optional: set[str] = frozenset()) -> Mapping:
    if not isinstance(block, Mapping):
        raise _refuse(f"{where} must be an object")
    missing = sorted(required - set(block))
    unknown = sorted(set(block) - required - set(optional))
    if missing:
        raise _refuse(f"{where} lacks {missing}")
    if unknown:
        raise _refuse(f"{where} carries {unknown}, which a {SCHEMA} reader does not know; a key "
                      "this reader would ignore could change how the bundle must be read")
    return block


def parse_valid_time(text: Any) -> datetime:
    """``YYYY-MM-DDThh:mm:ssZ`` (what producers write) or WRF's ``YYYY-MM-DD_hh:mm:ss``, UTC."""

    for form in (ISO_TIME_FORMAT, TIME_FORMAT):
        try:
            return datetime.strptime(str(text), form)
        except ValueError:
            continue
    raise _refuse(f"valid_time {text!r} is neither YYYY-MM-DDThh:mm:ssZ nor YYYY-MM-DD_hh:mm:ss")


def wrf_time(text: Any) -> str:
    return parse_valid_time(text).strftime(TIME_FORMAT)


def iso_time(text: Any) -> str:
    return parse_valid_time(text).strftime(ISO_TIME_FORMAT)


def _validate_grid(block: Any) -> None:
    grid = _exact_keys(block, "grid", {"id", "geo_em_sha256", "nx", "ny", "nz", "dx_m", "dy_m", "projection",
                                       "horizontal_sha256", "vertical"})
    _ident(grid["id"], "grid.id")
    if grid["geo_em_sha256"] is not None:
        _sha(grid["geo_em_sha256"], "grid.geo_em_sha256")
    for key in ("nx", "ny", "nz"):
        _positive_int(grid[key], f"grid.{key}")
    _positive_number(grid["dx_m"], "grid.dx_m")
    _positive_number(grid["dy_m"], "grid.dy_m")
    projection = _exact_keys(grid["projection"], "grid.projection",
                             {name.lower() for name in PROJECTION_ATTRIBUTES})
    _positive_int(projection["map_proj"], "grid.projection.map_proj", minimum=0)
    for name in ("truelat1", "truelat2", "stand_lon"):
        _finite_number(projection[name], f"grid.projection.{name}")
    for name in ("moad_cen_lat", "cen_lat", "cen_lon"):
        if projection[name] is not None:
            _finite_number(projection[name], f"grid.projection.{name}")
    _sha(grid["horizontal_sha256"], "grid.horizontal_sha256")
    vertical = _exact_keys(grid["vertical"], "grid.vertical", {"znw_sha256", "p_top_pa", "hybrid_opt", "etac"})
    _sha(vertical["znw_sha256"], "grid.vertical.znw_sha256")
    _positive_number(vertical["p_top_pa"], "grid.vertical.p_top_pa")
    if vertical["hybrid_opt"] is not None:
        _positive_int(vertical["hybrid_opt"], "grid.vertical.hybrid_opt", minimum=0)
    if vertical["etac"] is not None:
        _finite_number(vertical["etac"], "grid.vertical.etac")


def validate_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Every rule of the v1 manifest; returns it with ``analysis.form`` filled in."""

    top = _exact_keys(manifest, "manifest", _REQUIRED_TOP, {"iau", "notes", "forecast"})
    if top["schema"] != SCHEMA:
        raise _refuse(f"schema {top['schema']!r} is not {SCHEMA!r}")
    _no_paths(dict(top), "manifest")
    parse_valid_time(top["valid_time"])
    _validate_grid(top["grid"])
    static = _exact_keys(top["static"], "static", {"sha256", "builder"})
    _sha(static["sha256"], "static.sha256")
    if not isinstance(static["builder"], str) or not static["builder"]:
        raise _refuse("static.builder must name the static builder")
    physics = _exact_keys(top["physics"], "physics", {"suite", "mode"})
    if not isinstance(physics["suite"], str) or not physics["suite"]:
        raise _refuse("physics.suite must name the physics suite the analysis was cycled with")
    if physics["mode"] not in PHYSICS_MODES:
        raise _refuse(f"physics.mode must be one of {PHYSICS_MODES}, got {physics['mode']!r}")
    analysis = _exact_keys(top["analysis"], "analysis",
                           {"kind", "members", "recipe_id", "recipe_sha256", "cycles", "window"}, {"form"})
    if analysis["kind"] not in ANALYSIS_KINDS:
        raise _refuse(f"analysis.kind must be one of {ANALYSIS_KINDS}, got {analysis['kind']!r}")
    _positive_int(analysis["members"], "analysis.members")
    _ident(analysis["recipe_id"], "analysis.recipe_id")
    _sha(analysis["recipe_sha256"], "analysis.recipe_sha256")
    _positive_int(analysis["cycles"], "analysis.cycles", minimum=0)
    if analysis["window"] is not None:
        _positive_number(analysis["window"], "analysis.window")
    form = analysis.get("form", "wrfinput")
    if form not in ANALYSIS_FORMS:
        raise _refuse(f"analysis.form must be one of {ANALYSIS_FORMS}, got {form!r}")
    versions = _exact_keys(top["versions"], "versions", {"woof", "producer"})
    for key in ("woof", "producer"):
        if versions[key] is not None and not isinstance(versions[key], str):
            raise _refuse(f"versions.{key} must be a string or null")
    files = top["files"]
    if not isinstance(files, Mapping):
        raise _refuse("files must be an object of name -> {sha256, bytes}")
    unknown = sorted(set(files) - set(BUNDLE_FILES))
    if unknown:
        raise _refuse(f"files lists {unknown}; a v1 bundle holds only {sorted(BUNDLE_FILES)}")
    for name, required in BUNDLE_FILES.items():
        if required and name not in files:
            raise _refuse(f"files lacks {name}, the analysis itself")
    for name, entry in files.items():
        entry = _exact_keys(entry, f"files[{name}]", {"sha256", "bytes"})
        _sha(entry["sha256"], f"files[{name}].sha256")
        _positive_int(entry["bytes"], f"files[{name}].bytes")
    iau = top.get("iau")
    if (INCREMENT in files) != (iau is not None):
        raise _refuse("iau.window_s and increment.nc go together: an increment without a window (or a "
                      "window without an increment) defines no forcing")
    if iau is not None:
        iau = _exact_keys(iau, "iau", {"window_s"})
        _positive_number(iau["window_s"], "iau.window_s")
    if form == "wrfout" and INCREMENT not in files:
        raise _refuse("a wrfout-form analysis starts a forecast only as a cycling restart, which adds the "
                      "increment to the restored state; it needs increment.nc")
    if "notes" in top and not isinstance(top["notes"], Mapping):
        raise _refuse("notes must be an object (the producer's provenance; a reader never acts on it)")
    if "forecast" in top:
        forecast = _exact_keys(top["forecast"], "forecast", {"args"})
        parse_forecast_args(forecast["args"])
    result = json.loads(json.dumps(dict(top)))
    result["analysis"]["form"] = form
    return result


def parse_forecast_args(args: Any) -> list[tuple[str, str | None]]:
    """``forecast.args`` as (flag, value) pairs, each flag one of :data:`FORECAST_FLAGS`."""

    if not isinstance(args, list) or not all(isinstance(item, str) for item in args):
        raise _refuse("forecast.args must be a list of strings (woof run flags and their values)")
    pairs: list[tuple[str, str | None]] = []
    index = 0
    while index < len(args):
        flag = args[index]
        if flag not in FORECAST_FLAGS:
            raise _refuse(f"forecast.args names {flag!r}; a bundle hands the forecast only model configuration "
                          f"({', '.join(sorted(FORECAST_FLAGS))}), never a host binding or a path")
        arity = FORECAST_FLAGS[flag]
        if arity and (index + 1 >= len(args) or args[index + 1].startswith("--")):
            raise _refuse(f"forecast.args: {flag} takes a value")
        pairs.append((flag, args[index + 1] if arity else None))
        index += 1 + arity
    return pairs


def manifest_bytes(manifest: Mapping[str, Any]) -> bytes:
    """The canonical serialisation a producer writes."""

    return (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")


# ---------------------------------------------------------------------------
# the bundle on disk
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AnalysisBundle:
    root: Path
    manifest: Mapping[str, Any]
    manifest_sha256: str
    files: Mapping[str, Path]

    @property
    def valid_time(self) -> str:
        """WRF clock text (the analysis's Times, the run's start)."""
        return wrf_time(self.manifest["valid_time"])

    @property
    def form(self) -> str:
        return str(self.manifest["analysis"]["form"])

    @property
    def analysis(self) -> Path:
        return self.files[ANALYSIS]

    @property
    def increment(self) -> Path | None:
        return self.files.get(INCREMENT)

    @property
    def iau_window_seconds(self) -> float | None:
        iau = self.manifest.get("iau")
        return None if iau is None else float(iau["window_s"])

    def provenance_sentence(self) -> str:
        analysis = self.manifest["analysis"]
        return (f"analysis bundle {self.manifest_sha256[:12]} ({analysis['kind']} of "
                f"{analysis['members']}, recipe {analysis['recipe_id']}, {analysis['cycles']} cycles, "
                f"valid {self.valid_time})")

    def receipt(self) -> dict[str, Any]:
        return {"schema": SCHEMA, "root": str(self.root), "manifest_sha256": self.manifest_sha256,
                "valid_time": self.valid_time, "form": self.form,
                "grid": dict(self.manifest["grid"]), "static": dict(self.manifest["static"]),
                "physics": dict(self.manifest["physics"]), "analysis": dict(self.manifest["analysis"]),
                "iau": None if self.manifest.get("iau") is None else dict(self.manifest["iau"]),
                "versions": dict(self.manifest["versions"]),
                "files": {name: dict(entry) for name, entry in self.manifest["files"].items()},
                "files_verified": sorted(self.files),
                "forecast": self.manifest.get("forecast"),
                "notes": self.manifest.get("notes")}


def load_bundle(source: str | Path, *, check_contents: bool = True) -> AnalysisBundle:
    """Read and check a bundle: every listed file present, at its byte count and SHA-256.

    ``source`` is the bundle directory or its manifest.json.  ``check_contents`` also reads
    the analysis and increment and checks they carry what the manifest promises.
    """

    path = Path(source).expanduser()
    root = path.parent if path.name == MANIFEST else path
    if not (root / MANIFEST).is_file():
        raise _refuse(f"{root} holds no {MANIFEST}")
    raw = (root / MANIFEST).read_bytes()
    try:
        manifest = validate_manifest(json.loads(raw.decode("utf-8")))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _refuse(f"{MANIFEST} is not JSON: {error}") from None
    files: dict[str, Path] = {}
    # Every listed file hashes at once, each on its own thread; the checks below then read the
    # digests in manifest order, so the first file that fails is the one named, as before.
    from woof.file_digest import prefetch
    for name in manifest["files"]:
        if (root / name).is_file():
            prefetch(root / name)
    for name, entry in manifest["files"].items():
        file = root / name
        if not file.is_file():
            raise _refuse(f"{name} is listed in the manifest and absent from {root}")
        size = file.stat().st_size
        if size != entry["bytes"]:
            raise _refuse(f"{name} is {size} bytes; the manifest says {entry['bytes']}")
        digest = sha256_file(file)
        if digest != entry["sha256"]:
            raise _refuse(f"{name} has SHA-256 {digest}; the manifest says {entry['sha256']}.  A changed "
                          "or partial file would start the forecast from a state nobody analysed")
        files[name] = file.resolve()
    for name in BUNDLE_FILES:
        if name not in files and (root / name).exists():
            raise _refuse(f"{name} is in {root} but not listed in the manifest, so its bytes are not "
                          "covered by any digest; list it or remove it")
    bundle = AnalysisBundle(root=root.resolve(), manifest=manifest,
                            manifest_sha256=hashlib.sha256(raw).hexdigest(), files=files)
    if check_contents:
        check_analysis_file(bundle.analysis, valid_time=bundle.valid_time, form=bundle.form)
        measured = grid_fingerprint(bundle.analysis)
        _compare_grid(manifest["grid"], measured, f"{ANALYSIS}")
        static = static_fingerprint(bundle.analysis)
        if static != manifest["static"]["sha256"]:
            raise _refuse(f"{ANALYSIS}'s static fields hash to {static}; static.sha256 says "
                          f"{manifest['static']['sha256']}")
        if bundle.increment is not None:
            increment_variables(bundle.increment, bundle.analysis)
    return bundle


def _compare_grid(expected: Mapping[str, Any], measured: Mapping[str, Any], what: str) -> None:
    for key in ("nx", "ny", "nz", "horizontal_sha256"):
        if expected[key] != measured[key]:
            raise _refuse(f"{what} has {key} {measured[key]}; the bundle's grid {expected['id']} has "
                          f"{expected[key]}.  Starting from an analysis on another grid puts every field "
                          "on the wrong columns")
    for key in ("dx_m", "dy_m"):
        if float(expected[key]) != float(measured[key]):
            raise _refuse(f"{what} has {key} {measured[key]}; the bundle's grid has {expected[key]}")
    for key, value in expected["projection"].items():
        if value != measured["projection"].get(key):
            raise _refuse(f"{what} has projection {key} {measured['projection'].get(key)}; the bundle's grid "
                          f"has {value}")
    vertical = expected["vertical"]
    other = measured["vertical"]
    for key in ("znw_sha256", "p_top_pa", "hybrid_opt", "etac"):
        if vertical[key] != other[key]:
            raise _refuse(f"{what}'s vertical coordinate has {key} {other[key]}; the bundle's has "
                          f"{vertical[key]}.  The run would read the analysed columns on other levels")


def check_analysis_file(path: str | Path, *, valid_time: str, form: str = "wrfinput") -> dict[str, Any]:
    """The analysis file carries what its form promises, at the bundle's valid time."""

    path = Path(path)
    require_classic(path, "the analysis")
    dataset = _open(path)
    names = set(dataset.variables)
    required = WRFINPUT_FORM_REQUIRED if form == "wrfinput" else WRFOUT_FORM_REQUIRED
    missing = sorted(set(required) - names)
    if missing:
        raise _refuse(f"{path.name} ({form} form) lacks {missing}")
    times = _times(dataset)
    if valid_time not in times:
        raise _refuse(f"{path.name} holds no record at the valid time {valid_time} (records: {times})")
    if form == "wrfinput" and len(times) != 1:
        raise _refuse(f"{path.name} holds {len(times)} records; a wrfinput-form analysis is one record, the "
                      "state at the valid time")
    if form == "wrfinput":
        start = _attribute(dataset, "START_DATE")
        if start is not None and str(start) != valid_time:
            raise _refuse(f"{path.name} says START_DATE {start}; the bundle is valid {valid_time}.  A "
                          "wrfinput-form analysis starts the run, so its start is the valid time")
    return {"form": form, "times": times}


def increment_variables(path: str | Path, analysis: str | Path | None = None) -> dict[str, tuple[str, ...]]:
    """Every increment variable of ``path`` with its spatial dimensions.

    The required core must be there, every variable must be a known prognostic over its own
    staggered dimensions, finite, and, with ``analysis``, a variable of that file over the
    same dimensions.
    """

    path = Path(path)
    require_classic(path, "the increment")
    dataset = _open(path)
    out: dict[str, tuple[str, ...]] = {}
    for name, variable in dataset.variables.items():
        if name == "Times":
            continue
        expected = prognostic_dimensions(name)
        if expected is None:
            raise _refuse(f"{path.name} carries {name}, which is not a prognostic the forecast integrates "
                          f"(the dycore core {list(CORE_FIELDS)} and the moisture species and numbers)")
        dims = _spatial_dims(variable)
        if dims != expected:
            raise _refuse(f"{path.name}: {name} spans {variable.dimensions}; it increments a field over "
                          f"{expected}")
        if variable.dtype.kind != "f":
            raise _refuse(f"{path.name}: {name} is {variable.dtype}, not floating point")
        out[name] = dims
    missing = [name for name in REQUIRED_INCREMENT if name not in out]
    if missing:
        raise _refuse(f"{path.name} lacks {missing}; every increment carries the dycore core and vapour "
                      "(zeros where the analysis left a field alone)")
    if analysis is not None:
        other = _open(analysis)
        for name, dims in out.items():
            if name not in other.variables:
                raise _refuse(f"{path.name} carries {name} and {Path(analysis).name} has no {name}; the "
                              "background is formed as analysis minus increment, variable by variable")
            if _spatial_dims(other.variables[name]) != dims:
                raise _refuse(f"{name} spans {dims} in the increment and "
                              f"{_spatial_dims(other.variables[name])} in the analysis")
    return out


def read_increment(path: str | Path) -> dict[str, np.ndarray]:
    """Every increment variable, float64, by WRF name (spatial dimensions only)."""

    path = Path(path)
    names = list(increment_variables(path))
    dataset = _open(path)
    _prefetch_unmasked(dataset, names)
    out = {}
    for name in names:
        values = _first_record(dataset.variables[name])
        if not np.all(np.isfinite(values)):
            raise _refuse(f"{path.name}: {name} carries non-finite values")
        out[name] = np.ascontiguousarray(values)
    return out


def increment_precision(path: str | Path) -> dict[str, str]:
    dataset = _open(path)
    return {name: dataset.variables[name].dtype.str for name in increment_variables(path)}


# ---------------------------------------------------------------------------
# writing: increment, background, analysis
# ---------------------------------------------------------------------------
def _partial(path: Path) -> Path:
    return path.with_name(path.name + ".partial")


def _write_increment_file(out: Path, arrays: Mapping[str, np.ndarray], *, sizes: Mapping[str, int],
                          valid_time: str, precision: str) -> None:
    from woof.io.nc_writer_bridge import ClassicSchema

    schema = ClassicSchema("cdf2")
    dimids = {"Time": schema.def_dim("Time", 0, unlimited=True), "DateStrLen": schema.def_dim("DateStrLen", 19)}
    for name in ("west_east", "west_east_stag", "south_north", "south_north_stag", "bottom_top",
                 "bottom_top_stag"):
        dimids[name] = schema.def_dim(name, int(sizes[name]))
    schema.put_global_attr("TITLE", " ANALYSIS MINUS BACKGROUND (wrf-analysis-bundle/1 increment.nc)")
    schema.put_global_attr("START_DATE", valid_time)
    schema.put_global_attr("ANALYSIS_BUNDLE_SCHEMA", SCHEMA)
    times = schema.def_var("Times", "S1", (dimids["Time"], dimids["DateStrLen"]))
    varids = {}
    for name in sorted(arrays):
        dims = prognostic_dimensions(name)
        varids[name] = schema.def_var(name, precision, (dimids["Time"],) + tuple(dimids[d] for d in dims))
        schema.put_var_attr(varids[name], "increment_of", name)
        schema.put_var_attr(varids[name], "description", f"analysis {name} minus background {name}")
    partial = _partial(out)
    with schema.create(partial) as writer:
        writer.write_record(0, times, valid_time.encode("ascii"))
        for name, varid in varids.items():
            writer.write_record(0, varid, np.ascontiguousarray(arrays[name], dtype=precision))
    partial.replace(out)


def _sizes(dataset: Any) -> dict[str, int]:
    return {name: len(dim) for name, dim in dataset.dimensions.items()}


def candidate_prognostics(dataset: Any) -> list[str]:
    """The variables of a WRF-form file an increment may carry, in a fixed order."""

    names = []
    for name, variable in dataset.variables.items():
        expected = prognostic_dimensions(name)
        if expected is None or variable.dtype.kind != "f":
            continue
        if _spatial_dims(variable) != expected:
            continue
        names.append(name)
    return sorted(names)


def write_increment(analysis: str | Path, background: str | Path, out: str | Path, *,
                    precision: str = "f8", fields: str = "differing") -> dict[str, Any]:
    """analysis minus background in bundle form (the producer's side).

    ``precision`` "f8" (the default) stores the float64 difference, so the background a reader
    rebuilds as analysis minus increment is ``background`` bit for bit wherever the background
    value lies within the analysis value's float64 resolution (``exact_background`` in the
    receipt, measured per variable); "f4" stores float32.  ``fields`` "differing" (the
    default) writes the required core and vapour plus every other prognostic that differs;
    "all" writes every prognostic the two files share.
    """

    if precision not in ("f4", "f8"):
        raise _refuse(f"increment precision {precision!r} is not f4 or f8")
    if fields not in ("differing", "all"):
        raise _refuse(f"increment fields {fields!r} is not differing or all")
    analysis, background, out = Path(analysis), Path(background), Path(out)
    a, b = _open(analysis), _open(background)
    for name in ("west_east", "south_north", "bottom_top"):
        if len(a.dimensions[name]) != len(b.dimensions[name]):
            raise _refuse(f"analysis and background differ in {name}; an increment needs one grid")
    if grid_fingerprint(analysis) != grid_fingerprint(background):
        raise _refuse("analysis and background are on different grids (projection, geolocation or vertical "
                      "coordinate); an increment needs one grid")
    names = [name for name in candidate_prognostics(a) if name in b.variables]
    missing = [name for name in REQUIRED_INCREMENT if name not in names]
    if missing:
        raise _refuse(f"{missing} are missing from the analysis or the background")
    _prefetch_unmasked(a, names)
    _prefetch_unmasked(b, names)
    arrays, record = _increment_core(
        ((name, _first_record(a.variables[name]), a.variables[name].dtype, _first_record(b.variables[name]),
          b.variables[name].dtype) for name in names), precision=precision, fields=fields)
    stamp = _times(a)[0]
    _write_increment_file(out, arrays, sizes=_sizes(a), valid_time=stamp, precision=precision)
    return {"increment": str(out), "valid_time": stamp, **record}


def _increment_core(rows, *, precision: str, fields: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """analysis minus background, field by field: ``rows`` of (name, analysis float64, analysis
    type, background float64, background type), in the order the increment is written.  The one
    arithmetic of :func:`write_increment` (files) and :func:`make_exchange` (words in memory)."""

    from woof.da.host_analysis import binary

    def one(row):
        name, av, adtype, bv, bdtype = row
        try:
            delta, stats = binary(av, bv, subtract=True)
        except ValueError as exc:
            return name, exc, None, None, None
        if fields == "differing" and name not in REQUIRED_INCREMENT and not stats["nonzero"]:
            return name, None, None, None, None
        stored = delta if precision == "f8" else delta.astype(np.float32)
        rebuilt = binary(av, stored, subtract=True)[0].astype(adtype)
        background_words = bv.astype(bdtype)
        same = bool(np.array_equal(rebuilt.view(f"u{rebuilt.dtype.itemsize}"),
                                   background_words.view(f"u{background_words.dtype.itemsize}")))
        return name, None, stored, stats["rms"], same

    arrays: dict[str, np.ndarray] = {}
    rms: dict[str, float] = {}
    exact: dict[str, bool] = {}
    for name, error, stored, row_rms, same in _ordered_map(one, rows):
        if error is not None:
            raise _refuse(f"{name} increment is not finite") from error
        if stored is None:
            continue
        exact[name] = same
        arrays[name] = stored
        rms[name] = row_rms
    return arrays, {"precision": precision, "fields": sorted(arrays), "rms": rms, "exact_background": exact,
                    "exact_background_all": bool(exact) and all(exact.values())}


def _patched_copy(template: Path, out: Path, updates: Mapping[str, np.ndarray]) -> None:
    """``template`` copied to ``out`` with ``updates`` (record 0) rewritten in their stored type."""

    from woof.io.nc_writer_bridge import ClassicPatch

    partial = _partial(out)
    with ClassicPatch(template, partial) as patch:
        variables = patch.variables
        for name, values in updates.items():
            variable = variables[name]
            stored = np.ascontiguousarray(values, dtype=variable.dtype)
            patch.put(name, stored, record=0 if variable.is_record else None)
    partial.replace(out)


def write_background(analysis: str | Path, increment: str | Path, out: str | Path, *,
                     increment_read: dict[str, np.ndarray] | None = None) -> dict[str, Any]:
    """The background an IAU window starts from: the analysis minus its increment.

    Every increment variable changes its analysis variable, in float64 and rounded to the
    variable's stored type; every other byte is the analysis's.  A moisture species a float32
    increment leaves below zero by rounding is set to zero and counted; with a float64
    increment the producer's own values (transport roundoff below zero included) are kept,
    and counted.

    ``increment_read``, when given (an empty dict), receives :func:`read_increment`'s values
    of ``increment`` as this call read them, for a caller that needs them next (the analysis
    door's split): one decode of the increment file instead of two.  Nothing here writes them.
    """

    analysis, increment, out = Path(analysis), Path(increment), Path(out)
    require_classic(analysis, "the analysis")
    variables = increment_variables(increment, analysis)
    precision = increment_precision(increment)
    # The increment's decode runs on a host thread beside the analysis's: two independent rw_netcdf decodes
    # that ran one after the other on the analysis door's thread (single-core law,
    # 2026-10-29: DA member legs on woof 2.9.9 spent ~26 s at ~1 core before the engine
    # started; these two decodes were the largest part of the door).  The same reads; the increment's own
    # refusal is still raised first.
    import threading
    context = contextvars.copy_context()
    held: dict[str, Any] = {}

    def decode_increment():
        try:
            held["deltas"] = context.run(read_increment, increment)
        except BaseException as error:  # noqa: BLE001 - raised below, on the caller's thread
            held["error"] = error

    reader = threading.Thread(target=decode_increment, name="background-increment", daemon=True)
    reader.start()
    try:
        a = _open(analysis)
        _prefetch_unmasked(a, list(variables))
    except BaseException:
        reader.join()
        if "error" in held:
            raise held["error"] from None
        raise
    reader.join()
    if "error" in held:
        raise held["error"]
    deltas = held["deltas"]
    if increment_read is not None:
        increment_read.update(deltas)
    updates, record = _background_core(
        ((name, _first_record(a.variables[name]), a.variables[name].dtype) for name in sorted(variables)),
        deltas, precision)
    _patched_copy(analysis, out, updates)
    return {"background": str(out), **record}


def _background_core(rows, deltas: Mapping[str, np.ndarray],
                     precision: Mapping[str, str]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """analysis minus increment, rounded to each variable's type: ``rows`` of (name, analysis
    float64, analysis type); ``deltas`` float64; ``precision`` each increment's stored type
    (numpy ``dtype.str``).  The one arithmetic of :func:`write_background` and of the in-memory
    plan (:func:`woof.da.analysis_door.plan_from_exchange`)."""

    from woof.da.host_analysis import binary

    def one(row):
        name, a64, dtype = row
        clip = name.startswith(_NON_NEGATIVE_PREFIX) and precision[name] == np.dtype(np.float32).str
        values, stats = binary(a64, deltas[name], subtract=True, clip=clip)
        return name, clip, stats, values.astype(dtype)

    updates: dict[str, np.ndarray] = {}
    clipped: dict[str, dict[str, float]] = {}
    kept: dict[str, dict[str, float]] = {}
    for name, clip, stats, stored in _ordered_map(one, rows):
        if name.startswith(_NON_NEGATIVE_PREFIX) and stats["values"]:
            record = {key: stats[key] for key in ("values", "min")}
            (clipped if clip else kept)[name] = record
        updates[name] = stored
    return updates, {"changed": sorted(updates), "increment_precision": dict(precision),
                     "clipped_negative": clipped, "negative_kept_from_producer": kept,
                     "unchanged": "every variable with no increment (land and surface state among them): the "
                                  "analysis's own"}


def write_analysis(background: str | Path, increments: Mapping[str, np.ndarray], out: str | Path) -> dict[str, Any]:
    """The analysis a producer's increments make: background plus increment, rounded to each
    variable's stored type, every other byte the background's.

    A moisture species, number or volume (``Q*``) that background plus increment leaves below
    zero is set to zero and counted, the rule the door's at-once insertion already applies
    ("moisture clipped at zero").  Breakage it prevents: a direct-insertion start of such an
    analysis stops at the engine's initial health check (2026-10-15, a cycled convective analysis mean:
    ``qc`` -4.1e-14 from averaging 33 member increments, "violates lower bound >= 0.0").  The
    bundle's increment is then written from the clipped analysis, so analysis minus increment is
    still the producer's background.
    """

    background, out = Path(background), Path(out)
    updates, record = _analysis_updates(background, increments)
    _patched_copy(background, out, updates)
    return {"analysis": str(out), **record}


def _analysis_updates(background: Path, increments: Mapping[str, np.ndarray]):
    """:func:`write_analysis`'s values before they are written: (stored-type updates, record)."""

    require_classic(background, "the background")
    b = _open(background)
    _prefetch_unmasked(b, list(increments))
    for name in increments:
        if name not in b.variables:
            raise _refuse(f"the increment carries {name} and the background has no {name}")
    return _analysis_core(
        ((name, _first_record(b.variables[name]), b.variables[name].dtype, delta)
         for name, delta in sorted(increments.items())))


def _analysis_core(rows) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """background plus increment, rounded to each variable's type, moisture clipped at zero:
    ``rows`` of (name, background float64, background type, increment).  The one arithmetic of
    :func:`write_analysis` and :func:`make_exchange`."""

    from woof.da.host_analysis import binary

    def one(row):
        name, b64, dtype, delta = row
        try:
            values, stats = binary(b64, delta, clip=name.startswith(_NON_NEGATIVE_PREFIX))
        except ValueError as exc:
            return name, exc, None, None
        return name, None, stats, values.astype(dtype)

    updates = {}
    clipped: dict[str, dict[str, float]] = {}
    for name, error, stats, stored in _ordered_map(one, rows):
        if error is not None:
            raise _refuse(f"background {name} plus its increment is not finite") from error
        if name.startswith(_NON_NEGATIVE_PREFIX) and stats["values"]:
            clipped[name] = {key: stats[key] for key in ("values", "min")}
        updates[name] = stored
    return updates, {"changed": sorted(updates), "clipped_negative": clipped}


# ---------------------------------------------------------------------------
# the engine's geopotential (PH is never the producer's)
# ---------------------------------------------------------------------------
#: WRF moisture name -> the hydrostatic module's loading-mass name.
_LOADING = {"QVAPOR": "qv", "QCLOUD": "qc", "QRAIN": "qr", "QICE": "qi", "QSNOW": "qs", "QGRAUP": "qg",
            "QHAIL": "qh"}
#: What the column's state comes from: the time-varying fields of the background.
_COLUMN_STATE = ("P", "PB", "MUB", "MU", "T", "P_TOP")
#: The vertical coordinate's coefficients: one set per grid, so a background that does not
#: carry them (a wrfout) takes them from another file of the same grid (``coordinate``).
_COLUMN_COORDINATE = ("C1F", "C2F", "C1H", "C2H", "C3F", "C4F", "C3H", "C4H", "RDNW", "RDN", "DNW")


def column_from_file(path: str | Path, *, coordinate: str | Path | None = None):
    """The hydrostatic column of a WRF-form file, as :mod:`woof.da.hydrostatic` reads a state
    (float64 numpy): diagnosed pressure, dry mass, theta and the loading masses from ``path``,
    and the coordinate coefficients from ``coordinate`` (default ``path``), which must be on
    the same grid and vertical coordinate (a wrfout carries no coefficients; the run's
    wrfinput does)."""

    from types import SimpleNamespace

    path = Path(path)
    dataset = _open(path)
    missing = [name for name in _COLUMN_STATE if name not in dataset.variables]
    if missing:
        raise _refuse(f"{path.name} lacks {missing}, which the engine's hydrostatic geopotential reads")
    source = path if coordinate is None else Path(coordinate)
    if coordinate is not None and grid_fingerprint(source) != grid_fingerprint(path):
        raise _refuse(f"{source.name} is not on {path.name}'s grid and vertical coordinate, so its "
                      "coordinate coefficients are not this column's")
    coefficients = dataset if coordinate is None else _open(source)
    missing = [name for name in _COLUMN_COORDINATE if name not in coefficients.variables]
    if missing:
        raise _refuse(f"{source.name} lacks the coordinate coefficients {missing}; name a WRF-form file of the "
                      "same grid that carries them (the run's wrfinput) as the coordinate")
    loading = [name for name in _LOADING if name in dataset.variables]
    _prefetch_unmasked(dataset, list(_COLUMN_STATE) + loading)
    _prefetch_unmasked(coefficients, list(_COLUMN_COORDINATE))
    state = {name: _first_record(dataset.variables[name]) for name in list(_COLUMN_STATE) + loading}
    return column_from_arrays(state, {name: _first_record(coefficients.variables[name])
                                      for name in _COLUMN_COORDINATE})


def coordinate_of(path: str | Path) -> dict[str, np.ndarray]:
    """The vertical coordinate's coefficients of a WRF-form file (the run's wrfinput), as
    :func:`column_from_file` reads them: float64 by WRF name.  Read once per member by a caller
    that forms columns in memory."""

    path = Path(path)
    dataset = _open(path)
    missing = [name for name in _COLUMN_COORDINATE if name not in dataset.variables]
    if missing:
        raise _refuse(f"{path.name} lacks the coordinate coefficients {missing}; name a WRF-form file of the "
                      "same grid that carries them (the run's wrfinput) as the coordinate")
    _prefetch_unmasked(dataset, list(_COLUMN_COORDINATE))
    return {name: _first_record(dataset.variables[name]) for name in _COLUMN_COORDINATE}


def column_from_arrays(state: Mapping[str, Any], coordinate: Mapping[str, Any]):
    """:func:`column_from_file` on words held in memory: ``state`` maps the WRF names of
    :data:`_COLUMN_STATE` and every loading species the state carries to their values (any
    float type, read as float64 the way a file's record is), ``coordinate`` the coefficients of
    :data:`_COLUMN_COORDINATE` (the run's wrfinput's).  The loading species are those ``state``
    carries, exactly as a file's variables decide them, so a caller hands every species its
    history frame holds."""

    from types import SimpleNamespace

    missing = [name for name in _COLUMN_STATE if name not in state]
    if missing:
        raise _refuse(f"the state lacks {missing}, which the engine's hydrostatic geopotential reads")
    missing = [name for name in _COLUMN_COORDINATE if name not in coordinate]
    if missing:
        raise _refuse(f"the coordinate lacks the coefficients {missing} (the run's wrfinput carries them)")
    values = {name: np.asarray(state[name], dtype=np.float64) for name in _COLUMN_STATE}
    values.update({name: np.asarray(coordinate[name], dtype=np.float64) for name in _COLUMN_COORDINATE})
    loading = [name for name in _LOADING if name in state]
    values.update({name: np.asarray(state[name], dtype=np.float64) for name in loading})
    from woof.da.host_analysis import binary

    nz = values["T"].shape[0]
    column = SimpleNamespace(
        p=binary(values["P"], values["PB"])[0], mub2d=values["MUB"], mup=values["MU"],
        thb=np.full(nz, 300.0), thp=values["T"],
        c1f=values["C1F"], c2f=values["C2F"], c1h=values["C1H"], c2h=values["C2H"],
        c3f=values["C3F"], c4f=values["C4F"], c3h=values["C3H"], c4h=values["C4H"],
        rdnw=values["RDNW"], rdn=values["RDN"], dnw=values["DNW"],
        p_top=float(np.asarray(values["P_TOP"]).reshape(-1)[0]))
    for wrf_name in loading:
        setattr(column, _LOADING[wrf_name], values[wrf_name])
    return column


def balanced_geopotential(background: Any, increments: Mapping[str, np.ndarray], *,
                          hypsometric_opt: int = 2,
                          coordinate: str | Path | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """The ``PH`` increment that keeps the analysed columns hydrostatic at their own dry mass.

    :func:`woof.da.hydrostatic.mass_aware_geopotential_change` on the background's column
    and the ``T``, loading-moisture and ``MU`` increments (any other name is ignored).  Exactly
    zero in a column none of them touches.  ``background`` is a WRF-form file (its column read
    with :func:`column_from_file`) or a column already formed (:func:`column_from_arrays`;
    ``coordinate`` must then be None, the column carries its coefficients).
    """

    from woof.da import hydrostatic as hydro

    if isinstance(background, (str, Path)):
        column = column_from_file(background, coordinate=coordinate)
    else:
        if coordinate is not None:
            raise _refuse("a column formed in memory carries its own coordinate coefficients; name no "
                          "coordinate file beside it")
        column = background
    from woof.da.host_analysis import binary

    after = {"thp": binary(column.thp, increments.get("T", 0.0))[0]}
    if "MU" in increments and binary(increments["MU"], 0.0)[1]["nonzero"]:
        # Only a mass that moves takes the mass-aware integration; otherwise the operator is
        # the fixed-mass one the cycle's rebalance uses, byte for byte.
        after["mup"] = binary(column.mup, increments["MU"])[0]
    loads = []
    for wrf_name, name in _LOADING.items():
        current = getattr(column, name, None)
        if current is None:
            if wrf_name in increments and binary(increments[wrf_name], 0.0)[1]["nonzero"]:
                raise _refuse(f"the increment moves {wrf_name} and the background carries no {wrf_name}")
            continue
        loads.append((name, current, increments.get(wrf_name, 0.0)))
    # each species' analysed loading on the bundle threads (one kernel call per species, as before)
    for name, value in _ordered_map(lambda row: (row[0], binary(row[1], row[2], clip=2)[0]), loads):
        after[name] = value
    return hydro.mass_aware_geopotential_change(column, after, hypsometric_opt=hypsometric_opt)


def _balance_core(analysis: Mapping[str, np.ndarray], background: Mapping[str, np.ndarray], names: Sequence[str],
                  balance) -> tuple[np.ndarray, dict[str, Any]]:
    """``--ph balance``: the background's PH plus the engine's geopotential of the analysis's
    ``names`` changes (``balance(changes)`` is :func:`balanced_geopotential` on the background's
    column), as float64; the record carries how far the producer's own PH was from it.  The one
    arithmetic of :func:`make_bundle` and :func:`make_exchange`."""

    from woof.da.host_analysis import binary

    changes = dict(_ordered_map(lambda name: (name, binary(analysis[name], background[name], subtract=True)[0]),
                                names))
    dphp, record = balance(changes)
    balanced = binary(background["PH"], dphp)[0]
    record["producer_ph_departure_max_abs_m2s2"] = binary(balanced, analysis["PH"], subtract=True)[1]["max_abs"]
    return balanced, record


# ---------------------------------------------------------------------------
# make / verify / check_run
# ---------------------------------------------------------------------------
def _digest_beside(path: Path) -> None:
    """Start the file's SHA-256 on a thread now (woof.file_digest.prefetch); write_manifest then takes it."""

    from woof.file_digest import prefetch

    prefetch(path)


def _versions(producer: str) -> dict[str, Any]:
    from importlib import metadata

    try:
        gpuwm_version = metadata.version("recast-woof")
    except metadata.PackageNotFoundError:
        gpuwm_version = None
    return {"woof": gpuwm_version, "producer": producer}


def write_manifest(root: str | Path, *, valid_time: str, grid: Mapping[str, Any], static: Mapping[str, Any],
                   physics: Mapping[str, Any], analysis: Mapping[str, Any], versions: Mapping[str, Any],
                   iau: Mapping[str, Any] | None = None, notes: Mapping[str, Any] | None = None,
                   forecast_args: Sequence[str] | None = None) -> Path:
    """Hash the bundle's files and write its manifest (the producer's side).

    The manifest is validated before it is written, so a producer cannot publish one a reader
    would refuse.
    """

    root = Path(root)
    files = {}
    for name in BUNDLE_FILES:
        path = root / name
        if path.is_file():
            files[name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    manifest = {"schema": SCHEMA, "valid_time": iso_time(valid_time), "grid": dict(grid), "static": dict(static),
                "physics": dict(physics), "analysis": dict(analysis), "versions": dict(versions),
                "files": files}
    if iau is not None:
        manifest["iau"] = dict(iau)
    if notes is not None:
        manifest["notes"] = json.loads(json.dumps(dict(notes)))
    if forecast_args:
        manifest["forecast"] = {"args": [str(item) for item in forecast_args]}
    validate_manifest(manifest)
    target = root / MANIFEST
    partial = _partial(target)
    partial.write_bytes(manifest_bytes(manifest))
    partial.replace(target)
    return target


@_reuse_input_reads
def make_bundle(out: str | Path, *, background: str | Path, grid_id: str, static_builder: str,
                physics_suite: str, physics_mode: str, kind: str, members: int, recipe_id: str,
                recipe_sha256: str, cycles: int, window: float | None, producer: str,
                analysis: str | Path | None = None, increment: str | Path | None = None,
                iau_window_s: float | None = None, noda: str | Path | None = None,
                form: str = "wrfinput", ph: str = "balance", hypsometric_opt: int = 2,
                fields: str = "differing", increment_precision: str = "f8",
                geo_em_sha256: str | None = None, notes: Mapping[str, Any] | None = None,
                forecast_args: Sequence[str] | None = None,
                coordinate: str | Path | None = None,
                increment_arrays: Mapping[str, np.ndarray] | None = None) -> tuple[AnalysisBundle, dict[str, Any]]:
    """Assemble a bundle directory and write its manifest.

    ``background`` is the state the analysis was computed against (WRF form, classic).  Give
    the analysis itself (``analysis``) or the producer's increments (``increment``, a file in
    increment form); the analysis is then the background plus the increment.  ``ph``
    "balance" (the default) replaces the analysis's ``PH`` with the engine's own hydrostatic
    integration of the ``T``, moisture and ``MU`` changes at the background's dry mass;
    "analysis" keeps the producer's ``PH``.  ``iau_window_s`` writes increment.nc (analysis
    minus background, float64 by default) and the manifest's IAU window.  ``coordinate`` is a
    WRF-form file of the same grid carrying the vertical coordinate's coefficients, for a
    background that does not (a wrfout); default the background itself.
    ``increment_arrays`` accepts already resident producer arrays instead of serializing them to an intermediate
    increment file. It uses the same float64 addition, stored-type rounding, balance and bundle writer.
    """

    out = Path(out)
    background = Path(background)
    if sum(x is not None for x in (analysis, increment, increment_arrays)) != 1:
        raise _refuse("give exactly one of analysis, increment or increment_arrays")
    if ph not in ("balance", "analysis"):
        raise _refuse(f"--ph {ph!r} is not balance or analysis")
    if out.exists() and any(out.iterdir()):
        raise _refuse(f"{out} exists and is not empty")
    require_classic(background, "the background")
    out.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {"ph": ph}
    target = out / ANALYSIS
    if analysis is not None:
        require_classic(analysis, "the analysis")
        shutil.copyfile(analysis, target)
    else:
        if increment_arrays is None:
            increments = read_increment(increment)
        else:
            missing = sorted(set(REQUIRED_INCREMENT) - set(increment_arrays))
            if missing:
                raise _refuse(f"resident increments lack required fields {missing}")
            background_fields = _open(background)
            increments = {}
            for name, array in increment_arrays.items():
                expected = prognostic_dimensions(name)
                if expected is None or name not in background_fields.variables:
                    raise _refuse(f"resident increment {name} is not a background prognostic")
                variable = background_fields.variables[name]
                values = np.asarray(array)
                shape = tuple(n for d, n in zip(variable.dimensions, variable.shape) if d != "Time")
                if _spatial_dims(variable) != expected or values.shape != shape or values.dtype.kind != "f":
                    raise _refuse(f"resident increment {name} must be floating point over {expected}, shape {shape}")
                if not np.all(np.isfinite(values)):
                    raise _refuse(f"resident increment {name} is not finite")
                increments[name] = np.asarray(values, dtype=np.float64)
        updates, made = _analysis_updates(background, increments)
        record["analysis_from_increment"] = {"analysis": str(target), **made}
        if ph == "balance":
            # The analysis is written once, with the engine's PH already in it.  The balance reads the analysis's
            # T, moisture, MU and PH, which are exactly these stored words (or the background's where the increment
            # has none), so writing the file and then patching PH into a second full copy wrote the same bytes
            # twice and decoded them again in between (single-core law, 2026-10-27 DA cycle bundles).
            b = _open(background)
            names = [name for name in ("T", "MU", *_LOADING) if name in b.variables]
            _prefetch_unmasked(b, names + ["PH"])
            analysed = {name: (np.asarray(updates[name], dtype=np.float64) if name in updates
                               else _first_record(b.variables[name])) for name in names + ["PH"]}
            balanced, record["balance"] = _balance_core(
                analysed, {name: _first_record(b.variables[name]) for name in names + ["PH"]}, names,
                lambda changes: balanced_geopotential(background, changes, hypsometric_opt=hypsometric_opt,
                                                      coordinate=coordinate))
            updates = dict(updates, PH=balanced)
        _patched_copy(background, target, updates)
        _digest_beside(target)
    if ph == "balance" and analysis is not None:
        a, b = _open(target), _open(background)
        names = [name for name in ("T", "MU", *_LOADING) if name in a.variables and name in b.variables]
        _prefetch_unmasked(a, names + ["PH"])
        _prefetch_unmasked(b, names + ["PH"])
        balanced, record["balance"] = _balance_core(
            {name: _first_record(a.variables[name]) for name in names + ["PH"]},
            {name: _first_record(b.variables[name]) for name in names + ["PH"]}, names,
            lambda changes: balanced_geopotential(background, changes, hypsometric_opt=hypsometric_opt,
                                                  coordinate=coordinate))
        work = target.with_name(target.name + ".balanced")
        _patched_copy(target, work, {"PH": balanced})
        work.replace(target)
    if iau_window_s is not None:
        record["increment"] = write_increment(target, background, out / INCREMENT,
                                               precision=increment_precision, fields=fields)
        _digest_beside(out / INCREMENT)
    if noda is not None:
        require_classic(noda, "the no-DA state")
        shutil.copyfile(noda, out / NODA)
    stamp = _times(_open(target))[0]
    write_manifest(
        out, valid_time=stamp, grid=describe_grid(grid_id, target, geo_em_sha256=geo_em_sha256),
        static={"sha256": static_fingerprint(target), "builder": static_builder},
        physics={"suite": physics_suite, "mode": physics_mode},
        analysis={"kind": kind, "members": int(members), "recipe_id": recipe_id,
                  "recipe_sha256": recipe_sha256, "cycles": int(cycles),
                  "window": None if window is None else float(window), "form": form},
        versions=_versions(producer),
        iau=None if iau_window_s is None else {"window_s": float(iau_window_s)},
        notes=notes, forecast_args=forecast_args)
    return load_bundle(out), record


# ---------------------------------------------------------------------------
# the exchange: one analysis held in memory (wrf-analysis-exchange/1)
# ---------------------------------------------------------------------------
EXCHANGE_SCHEMA = "wrf-analysis-exchange/1"
#: ``analysis.form`` of an exchange: the analysis is words held in memory, not a file.
EXCHANGE_FORM = "exchange"
#: The integrity digest the record's ``arrays.<name>.digest`` names (XXH64 of each 1 MiB leaf of
#: the logical C-order bytes, then of the little-endian leaf digests; seed zero).
EXCHANGE_DIGEST = "xxh64-tree-1MiB"
_DIGEST16 = re.compile(r"^[0-9a-f]{16}$")


def _attribute_of(attributes: Mapping[str, Any], name: str) -> Any:
    """:func:`_attribute` on a mapping of global attributes held in memory."""

    value = attributes.get(name)
    if isinstance(value, np.ndarray) and value.size == 1:
        value = value.reshape(-1)[0]
    if isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    if isinstance(value, np.generic):
        value = value.item()
    return value


def frame_sizes(state: Mapping[str, Any]) -> dict[str, int]:
    """The six grid dimensions of a history frame held in memory, from its ``T`` (nz, ny, nx)."""

    if "T" not in state:
        raise _refuse("the state carries no T, so its grid dimensions cannot be read")
    nz, ny, nx = np.shape(state["T"])
    return {"west_east": nx, "west_east_stag": nx + 1, "south_north": ny, "south_north_stag": ny + 1,
            "bottom_top": nz, "bottom_top_stag": nz + 1}


def _shape_of(dims: Sequence[str], sizes: Mapping[str, int]) -> tuple[int, ...]:
    return tuple(int(sizes[name]) for name in dims)


def frame_prognostics(state: Mapping[str, Any], *, dimensions: Mapping[str, Sequence[str]] | None = None
                      ) -> list[str]:
    """:func:`candidate_prognostics` on words in memory: every floating field whose name is a
    prognostic over its own staggering (``dimensions`` when the producer states them, else the
    shape that staggering has on this grid), in a fixed order."""

    sizes = frame_sizes(state)
    names = []
    for name, values in state.items():
        expected = prognostic_dimensions(name)
        if expected is None or np.asarray(values).dtype.kind != "f":
            continue
        if dimensions is not None and name in dimensions:
            if tuple(dimensions[name]) != expected:
                continue
        elif tuple(np.shape(values)) != _shape_of(expected, sizes):
            continue
        names.append(name)
    return sorted(names)


def increment_arrays(increment: Mapping[str, Any], sizes: Mapping[str, int]) -> dict[str, np.ndarray]:
    """:func:`read_increment` on arrays: every field a prognostic over its own staggering, floating
    point, finite, the required core and vapour present; float64 by WRF name."""

    out: dict[str, np.ndarray] = {}
    for name in sorted(increment):
        expected = prognostic_dimensions(name)
        if expected is None:
            raise _refuse(f"the increment carries {name}, which is not a prognostic the forecast integrates "
                          f"(the dycore core {list(CORE_FIELDS)} and the moisture species and numbers)")
        values = np.asarray(increment[name])
        if tuple(values.shape) != _shape_of(expected, sizes):
            raise _refuse(f"the increment's {name} is {values.shape}; it increments a field over {expected} "
                          f"{_shape_of(expected, sizes)}")
        if values.dtype.kind != "f":
            raise _refuse(f"the increment's {name} is {values.dtype}, not floating point")
        out[name] = values
    missing = [name for name in REQUIRED_INCREMENT if name not in out]
    if missing:
        raise _refuse(f"the increment lacks {missing}; every increment carries the dycore core and vapour "
                      "(zeros where the analysis left a field alone)")
    for name, values in out.items():
        if not np.all(np.isfinite(values)):
            raise _refuse(f"the increment's {name} carries non-finite values")
    return {name: np.ascontiguousarray(np.asarray(values, dtype=np.float64)) for name, values in out.items()}


def grid_of_state(grid_id: str, state: Mapping[str, Any], attributes: Mapping[str, Any], *,
                  geo_em_sha256: str | None = None) -> dict[str, Any]:
    """:func:`describe_grid` on a frame held in memory (the same words, the same rule)."""

    missing = [name for name in ("ZNW", "P_TOP", *HORIZONTAL_FIELDS) if name not in state]
    if missing:
        raise _refuse(f"the state lacks {missing}, so its grid cannot be identified")
    measured = _grid_fingerprint(frame_sizes(state), state["ZNW"], state["P_TOP"],
                                 lambda name: _attribute_of(attributes, name),
                                 _float_digest_values((name, state[name]) for name in HORIZONTAL_FIELDS),
                                 "the state")
    return {"id": str(grid_id), "geo_em_sha256": geo_em_sha256, **measured}


def static_of_state(state: Mapping[str, Any]) -> str:
    """:func:`static_fingerprint` on a frame held in memory."""

    missing = [name for name in STATIC_FIELDS if name not in state]
    if missing:
        raise _refuse(f"the state lacks {missing}, so its static fields cannot be identified")
    return _float_digest_values((name, state[name]) for name in STATIC_FIELDS)


def check_exchange_run(grid: Mapping[str, Any], static_sha256: str, run_grid: Mapping[str, Any],
                       run_static_sha256: str, what: str = "the run") -> dict[str, Any]:
    """:func:`check_run` for an exchange: the run's grid (``grid_fingerprint`` of its wrfinput) and
    static fields against the analysis's, refused by name."""

    _compare_grid(grid, run_grid, what)
    if run_static_sha256 != static_sha256:
        raise _refuse(f"the analysis was made over static fields {static_sha256}; {what} has "
                      f"{run_static_sha256}.  Other terrain or land under the analysed columns")
    return {"run_file": what, "grid": dict(run_grid), "static_sha256": run_static_sha256}


@dataclass(frozen=True)
class AnalysisExchange:
    """One member's analysis held in memory: exactly the words :func:`make_bundle` writes.

    ``analysis`` holds the analysed words of every prognostic the analysis changed (``PH``
    included, the engine's balance), each in its stored type; every other field of the analysis
    is the background frame's own (``background``, borrowed, never copied).  ``increment`` holds
    what ``increment.nc`` holds, in its stored precision (None without IAU).  ``metadata`` is the
    bundle manifest's keys outside ``files`` with ``analysis.form = "exchange"``; ``form`` is the
    form :meth:`write` gives the bundle (a history frame's: ``wrfout``).
    """

    metadata: Mapping[str, Any]
    analysis: Mapping[str, np.ndarray]
    increment: Mapping[str, np.ndarray] | None
    background: Mapping[str, Any]
    attributes: Mapping[str, Any]
    dimensions: Mapping[str, Sequence[str]] | None
    form: str
    record: Mapping[str, Any]

    @property
    def valid_time(self) -> str:
        return wrf_time(self.metadata["valid_time"])

    @property
    def iau_window_seconds(self) -> float | None:
        iau = self.metadata.get("iau")
        return None if iau is None else float(iau["window_s"])

    @property
    def sizes(self) -> dict[str, int]:
        return frame_sizes(self.background)

    def state(self) -> Mapping[str, Any]:
        """The analysed state: the changed words over the background frame's."""

        from collections import ChainMap

        return ChainMap(dict(self.analysis), dict(self.background))

    def increment_float64(self) -> dict[str, np.ndarray] | None:
        """The increment as :func:`read_increment` hands it to the door: float64 by WRF name."""

        if self.increment is None:
            return None
        return {name: np.ascontiguousarray(np.asarray(values, dtype=np.float64))
                for name, values in self.increment.items()}

    def increment_precision(self) -> dict[str, str]:
        return {} if self.increment is None else {name: np.asarray(values).dtype.str
                                                  for name, values in self.increment.items()}

    def manifest_sha256(self) -> str:
        """SHA-256 of the canonical metadata: the exchange's identity in receipts (a bundle's is its
        manifest's, which also hashes its files)."""

        return hashlib.sha256(manifest_bytes(self.metadata)).hexdigest()

    def receipt(self) -> dict[str, Any]:
        """The ``bundle`` block of a door plan, for an exchange."""

        m = self.metadata
        return {"schema": EXCHANGE_SCHEMA, "root": None, "manifest_sha256": self.manifest_sha256(),
                "valid_time": self.valid_time, "form": EXCHANGE_FORM, "grid": dict(m["grid"]),
                "static": dict(m["static"]), "physics": dict(m["physics"]), "analysis": dict(m["analysis"]),
                "iau": None if m.get("iau") is None else dict(m["iau"]), "versions": dict(m["versions"]),
                "files": {}, "files_verified": [], "forecast": m.get("forecast"), "notes": m.get("notes")}

    def exchange_record(self, digest) -> dict[str, Any]:
        """The ``wrf-analysis-exchange/1`` record: the metadata with ``arrays`` (dtype, shape and the
        :data:`EXCHANGE_DIGEST` digest ``digest(array)`` of every increment field)."""

        if self.increment is None:
            raise _refuse("an exchange record lists the increment's arrays; this exchange carries none (no IAU)")
        record = json.loads(json.dumps(dict(self.metadata)))
        arrays = {}
        for name in sorted(self.increment):
            values = np.asarray(self.increment[name])
            value = digest(values)
            if not isinstance(value, str) or not _DIGEST16.fullmatch(value):
                raise _refuse(f"the digest of {name} is {value!r}, not a lowercase 16-digit {EXCHANGE_DIGEST}")
            arrays[name] = {"dtype": values.dtype.str[1:], "shape": [int(n) for n in values.shape], "digest": value}
        record["arrays"] = arrays
        return record

    def write(self, out: str | Path, *, template: str | Path | None = None,
              container: str = "cdf2") -> AnalysisBundle:
        """The bundle :func:`make_bundle` writes for this analysis.  ``template``: the background's own
        history file (the files path's): analysis.nc is its patched copy and the bundle is
        make_bundle's byte for byte.  Without it, analysis.nc is written from the frame's words
        (every variable the frame carries, ``dimensions`` naming each one's), so its arrays are
        make_bundle's and its bytes are not."""

        out = Path(out)
        if out.exists() and any(out.iterdir()):
            raise _refuse(f"{out} exists and is not empty")
        out.mkdir(parents=True, exist_ok=True)
        target = out / ANALYSIS
        if template is not None:
            template = Path(template)
            require_classic(template, "the background")
            _compare_grid(self.metadata["grid"], grid_fingerprint(template), template.name)
            if self.valid_time not in _times(_open(template)):
                raise _refuse(f"{template.name} holds no record at the exchange's valid time {self.valid_time}")
            _patched_copy(template, target, self.analysis)
        else:
            _write_state_file(target, self.state(), dimensions=self.dimensions, attributes=self.attributes,
                              valid_time=self.valid_time, container=container)
        if self.increment is not None:
            precision = self.record["increment"]["precision"]
            _write_increment_file(out / INCREMENT, self.increment, sizes=self.sizes, valid_time=self.valid_time,
                                  precision=precision)
        m = self.metadata
        analysis = dict(m["analysis"], form=self.form)
        write_manifest(out, valid_time=m["valid_time"], grid=m["grid"], static=m["static"], physics=m["physics"],
                       analysis=analysis, versions=m["versions"], iau=m.get("iau"), notes=m.get("notes"),
                       forecast_args=(m.get("forecast") or {}).get("args"))
        return load_bundle(out)


def _write_state_file(target: Path, state: Mapping[str, Any], *, dimensions, attributes, valid_time: str,
                      container: str) -> None:
    """A classic WRF-form file of one record from words in memory (every variable over Time and
    its own dimensions, the global attributes as given)."""

    from woof.io.nc_writer_bridge import ClassicSchema

    if dimensions is None:
        raise _refuse("writing an analysis from memory needs each variable's dimensions (the frame's own); "
                      "give the background's history file as the template instead")
    names = [name for name in state if name != "Times"]
    missing = [name for name in names if name not in dimensions]
    if missing:
        raise _refuse(f"the frame states no dimensions for {missing[:5]}")
    sizes: dict[str, int] = {}
    for name in names:
        for dim, n in zip(dimensions[name], np.shape(state[name])):
            if sizes.setdefault(dim, int(n)) != int(n):
                raise _refuse(f"{name}: dimension {dim} is {n} here and {sizes[dim]} elsewhere")
    schema = ClassicSchema(container)
    dimids = {"Time": schema.def_dim("Time", 0, unlimited=True), "DateStrLen": schema.def_dim("DateStrLen", 19)}
    for dim, n in sizes.items():
        if dim not in dimids:
            dimids[dim] = schema.def_dim(dim, n)
    for key, value in attributes.items():
        value = _attribute_of(attributes, key)
        if value is None:
            continue
        schema.put_global_attr(key, value if isinstance(value, str) else np.asarray(attributes[key]))
    times = schema.def_var("Times", "S1", (dimids["Time"], dimids["DateStrLen"]))
    varids = {name: schema.def_var(name, np.asarray(state[name]).dtype,
                                   (dimids["Time"],) + tuple(dimids[d] for d in dimensions[name]))
              for name in names}
    partial = _partial(target)
    with schema.create(partial) as writer:
        writer.write_record(0, times, valid_time.encode("ascii"))
        for name, varid in varids.items():
            writer.write_record(0, varid, np.ascontiguousarray(state[name]))
    partial.replace(target)


def make_exchange(*, state: Mapping[str, Any], attributes: Mapping[str, Any], valid_time: str,
                  coordinate: Mapping[str, Any], increment: Mapping[str, Any], grid_id: str, static_builder: str,
                  physics_suite: str, physics_mode: str, kind: str, members: int, recipe_id: str,
                  recipe_sha256: str, cycles: int, window: float | None, producer: str,
                  iau_window_s: float | None = None, form: str = "wrfout", ph: str = "balance",
                  hypsometric_opt: int = 2, fields: str = "differing", increment_precision: str = "f8",
                  geo_em_sha256: str | None = None, notes: Mapping[str, Any] | None = None,
                  forecast_args: Sequence[str] | None = None,
                  dimensions: Mapping[str, Sequence[str]] | None = None,
                  coordinate_grid: Mapping[str, Any] | None = None) -> AnalysisExchange:
    """:func:`make_bundle` (``increment=``, ``ph``, ``coordinate=``) on words held in memory.

    ``state`` is the background history frame (WRF name -> its stored words at ``valid_time``,
    no Time axis; every loading species it holds, as the column reads them); ``attributes`` its
    global attributes; ``coordinate`` the vertical coordinate's coefficients
    (:data:`_COLUMN_COORDINATE`, the run's wrfinput's) and ``coordinate_grid`` that file's
    ``grid_fingerprint`` (checked against the frame's, as :func:`column_from_file` checks a
    coordinate file); ``increment`` the producer's increment form as arrays (float64 or float32,
    the required core and vapour among them, ``PH`` zeros: the engine's balance replaces it).
    Every arithmetic step is the bundle's own function (:func:`_analysis_core`,
    :func:`_balance_core`, :func:`_increment_core`), so the exchange holds make_bundle's words.
    """

    if ph not in ("balance", "analysis"):
        raise _refuse(f"--ph {ph!r} is not balance or analysis")
    if increment_precision not in ("f4", "f8"):
        raise _refuse(f"increment precision {increment_precision!r} is not f4 or f8")
    if fields not in ("differing", "all"):
        raise _refuse(f"increment fields {fields!r} is not differing or all")
    if form not in ANALYSIS_FORMS:
        raise _refuse(f"form {form!r}: one of {ANALYSIS_FORMS}")
    valid = wrf_time(valid_time)
    sizes = frame_sizes(state)
    increments = increment_arrays(increment, sizes)
    for name in increments:
        if name not in state:
            raise _refuse(f"the increment carries {name} and the background has no {name}")
    record: dict[str, Any] = {"ph": ph}
    updates, record["analysis_from_increment"] = _analysis_core(
        (name, np.asarray(state[name], dtype=np.float64), np.asarray(state[name]).dtype, delta)
        for name, delta in sorted(increments.items()))
    analysis = dict(updates)

    def current(name):
        return analysis[name] if name in analysis else state[name]

    # describe_grid of the analysis: its identity fields are the background's (no increment touches
    # them), read through the analysed state all the same, as the files path reads analysis.nc
    grid = grid_of_state(grid_id, {name: current(name) for name in state}, attributes,
                         geo_em_sha256=geo_em_sha256)
    if ph == "balance":
        names = [name for name in ("T", "MU", *_LOADING) if name in state]
        if coordinate_grid is not None:
            measured = {key: value for key, value in grid.items() if key not in ("id", "geo_em_sha256")}
            if dict(coordinate_grid) != measured:
                raise _refuse("the coordinate is not on the state's grid and vertical coordinate, so its "
                              "coordinate coefficients are not this column's")
        column = column_from_arrays(state, coordinate)
        balanced, record["balance"] = _balance_core(
            {name: np.asarray(current(name), dtype=np.float64) for name in names + ["PH"]},
            {name: np.asarray(state[name], dtype=np.float64) for name in names + ["PH"]}, names,
            lambda changes: balanced_geopotential(column, changes, hypsometric_opt=hypsometric_opt))
        analysis["PH"] = np.ascontiguousarray(balanced, dtype=np.asarray(state["PH"]).dtype)
    increment_out = None
    if iau_window_s is not None:
        names = frame_prognostics(state, dimensions=dimensions)
        missing = [name for name in REQUIRED_INCREMENT if name not in names]
        if missing:
            raise _refuse(f"{missing} are missing from the analysis or the background")
        increment_out, made = _increment_core(
            ((name, np.asarray(current(name), dtype=np.float64), np.asarray(current(name)).dtype,
              np.asarray(state[name], dtype=np.float64), np.asarray(state[name]).dtype) for name in names),
            precision=increment_precision, fields=fields)
        record["increment"] = {"increment": None, "valid_time": valid, **made}
    metadata = {"schema": EXCHANGE_SCHEMA, "valid_time": iso_time(valid), "grid": grid,
                "static": {"sha256": static_of_state({name: current(name) for name in state}),
                           "builder": static_builder},
                "physics": {"suite": physics_suite, "mode": physics_mode},
                "analysis": {"kind": kind, "members": int(members), "recipe_id": recipe_id,
                             "recipe_sha256": recipe_sha256, "cycles": int(cycles),
                             "window": None if window is None else float(window), "form": EXCHANGE_FORM},
                "versions": _versions(producer)}
    if iau_window_s is not None:
        metadata["iau"] = {"window_s": float(iau_window_s)}
    if notes is not None:
        metadata["notes"] = json.loads(json.dumps(dict(notes)))
    if forecast_args:
        metadata["forecast"] = {"args": [str(item) for item in forecast_args]}
    validate_exchange_metadata(metadata, bundle_form=form)
    return AnalysisExchange(metadata=metadata, analysis=analysis, increment=increment_out, background=state,
                            attributes=dict(attributes), dimensions=None if dimensions is None else dict(dimensions),
                            form=form, record=record)


def validate_exchange_metadata(metadata: Mapping[str, Any], *, bundle_form: str = "wrfout") -> dict[str, Any]:
    """Every rule of the v1 manifest on an exchange's metadata: the bundle it writes as
    (``bundle_form``, with its files) must be one a reader takes."""

    if not isinstance(metadata, Mapping) or metadata.get("schema") != EXCHANGE_SCHEMA:
        raise _refuse(f"an exchange's schema is {EXCHANGE_SCHEMA}")
    if "files" in metadata or "arrays" in metadata:
        raise _refuse("an exchange's metadata names no files and no arrays (the record adds the arrays)")
    analysis = metadata.get("analysis")
    if not isinstance(analysis, Mapping) or analysis.get("form") != EXCHANGE_FORM:
        raise _refuse(f"an exchange's analysis.form is {EXCHANGE_FORM}")
    mirror = json.loads(json.dumps(dict(metadata)))
    mirror["schema"] = SCHEMA
    mirror["analysis"]["form"] = bundle_form
    mirror["files"] = {ANALYSIS: {"sha256": "0" * 64, "bytes": 1}}
    if mirror.get("iau") is not None:
        mirror["files"][INCREMENT] = {"sha256": "0" * 64, "bytes": 1}
    validate_manifest(mirror)
    return dict(metadata)


def check_run(bundle: AnalysisBundle, run_file: str | Path) -> dict[str, Any]:
    """Refuse, by name, a run whose grid, vertical coordinate or static fields are not the
    bundle's.  ``run_file`` is a WRF-form file of the run (its wrfinput, or its restart)."""

    measured = grid_fingerprint(run_file)
    _compare_grid(bundle.manifest["grid"], measured, Path(run_file).name)
    static = static_fingerprint(run_file)
    if static != bundle.manifest["static"]["sha256"]:
        raise _refuse(f"the analysis was made over static fields {bundle.manifest['static']['sha256']} "
                      f"(builder {bundle.manifest['static']['builder']}); {Path(run_file).name} has "
                      f"{static}.  Other terrain or land under the analysed columns")
    return {"run_file": str(run_file), "grid": measured, "static_sha256": static}


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    import shlex

    parser = argparse.ArgumentParser(prog="python -m woof.io.analysis_bundle", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("make", help="assemble a bundle directory and write its manifest")
    make.add_argument("--out", type=Path, required=True)
    make.add_argument("--background", type=Path, required=True,
                      help="the state the analysis was computed against (WRF form, NetCDF classic)")
    make.add_argument("--analysis", type=Path, default=None, help="the analysed state (WRF form)")
    make.add_argument("--increment", type=Path, default=None,
                      help="the producer's increments in increment form; the analysis is background + increment")
    make.add_argument("--noda", type=Path, default=None)
    make.add_argument("--grid-id", required=True)
    make.add_argument("--geo-em-sha256", default=None)
    make.add_argument("--static-builder", required=True)
    make.add_argument("--physics-suite", required=True)
    make.add_argument("--physics-mode", choices=PHYSICS_MODES, required=True)
    make.add_argument("--kind", choices=ANALYSIS_KINDS, required=True)
    make.add_argument("--members", type=int, required=True)
    make.add_argument("--recipe-id", required=True)
    make.add_argument("--recipe-sha256", required=True)
    make.add_argument("--cycles", type=int, required=True)
    make.add_argument("--window", type=float, default=None, help="assimilation window, seconds")
    make.add_argument("--iau-window-s", type=float, default=None,
                      help="write increment.nc (analysis minus background) and this IAU window")
    make.add_argument("--form", choices=ANALYSIS_FORMS, default="wrfinput")
    make.add_argument("--ph", choices=("balance", "analysis"), default="balance",
                      help="balance (default): the engine re-integrates PH from the T, moisture and MU changes; "
                           "analysis: keep the producer's PH")
    make.add_argument("--hypsometric-opt", type=int, choices=(1, 2), default=2)
    make.add_argument("--coordinate", type=Path, default=None,
                      help="a WRF-form file of the same grid carrying the vertical coordinate's coefficients "
                           "(the run's wrfinput), for a wrfout background; default the background")
    make.add_argument("--fields", choices=("differing", "all"), default="differing")
    make.add_argument("--increment-precision", choices=("f8", "f4"), default="f8")
    make.add_argument("--producer", required=True)
    make.add_argument("--forecast-args", default=None, metavar="'FLAGS'")
    verify = commands.add_parser("verify", help="check every file of a bundle against its manifest")
    verify.add_argument("bundle")
    increment = commands.add_parser("increment", help="write analysis minus background in bundle form")
    for flag in ("--analysis", "--background", "--out"):
        increment.add_argument(flag, type=Path, required=True)
    increment.add_argument("--precision", choices=("f8", "f4"), default="f8")
    increment.add_argument("--fields", choices=("differing", "all"), default="differing")
    background = commands.add_parser("background", help="write the background an IAU window starts from")
    background.add_argument("bundle")
    background.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "make":
            bundle, record = make_bundle(
                args.out, background=args.background, analysis=args.analysis, increment=args.increment,
                grid_id=args.grid_id, geo_em_sha256=args.geo_em_sha256, static_builder=args.static_builder,
                physics_suite=args.physics_suite, physics_mode=args.physics_mode, kind=args.kind,
                members=args.members, recipe_id=args.recipe_id, recipe_sha256=args.recipe_sha256,
                cycles=args.cycles, window=args.window, producer=args.producer, iau_window_s=args.iau_window_s,
                noda=args.noda, form=args.form, ph=args.ph, hypsometric_opt=args.hypsometric_opt,
                fields=args.fields, increment_precision=args.increment_precision, coordinate=args.coordinate,
                forecast_args=None if args.forecast_args is None else shlex.split(args.forecast_args))
            print(json.dumps({"bundle": bundle.receipt(), "make": record}, indent=2, default=str))
        elif args.command == "verify":
            print(json.dumps(load_bundle(args.bundle).receipt(), indent=2))
        elif args.command == "increment":
            print(json.dumps(write_increment(args.analysis, args.background, args.out, precision=args.precision,
                                             fields=args.fields), indent=2))
        else:
            bundle = load_bundle(args.bundle)
            if bundle.increment is None:
                raise _refuse("the bundle carries no increment.nc, so it has no background")
            print(json.dumps(write_background(bundle.analysis, bundle.increment, args.out), indent=2))
    except BundleRefusal as error:
        print(f"REFUSED {error}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
