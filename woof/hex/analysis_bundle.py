"""MPAS analysis bundle, version 1: an analysed atmosphere a hex forecast can start from.

A bundle is one directory, or the same files under one object-store prefix:

    manifest.json   schema "mpas-analysis-bundle/1" (below)
    analysis.nc     the analysed MPAS state at the valid time
    increment.nc    optional: analysis minus background, in MPAS IAU form
    noda.nc         optional: the matched no-assimilation state at the same valid time
    grid.nc         optional: the mesh's grid file (the mesh carried with the bundle)
    static.nc       optional: the mesh's static file
    mesh-row.json   optional: the woof hex mesh row the forecast binds the carried mesh with

It is the one interface between whatever produced the analysis (an ensemble filter, a
variational system, a nudged run) and the forecast: the producer writes it, and
``woof hex forecast --analysis BUNDLE`` reads it.  Nothing here knows how the analysis was
made.  The full description, field by field, is ``docs/analysis-bundle.md``.

``analysis.nc``
    ``analysis.form = "init"`` (the default): an MPAS initial-conditions file of the run's
    mesh, the form ``woof hex forecast --init`` takes.  It carries the uncoupled state
    (``rho`` dry density, ``theta``, ``u`` edge-normal wind, ``w``, the water species and any
    number concentrations the physics carries), the vertical grid (``zgrid``, ``zz`` ...), the
    reference state (``rho_base``, ``theta_base``) and the land and surface state, with
    ``xtime`` equal to ``valid_time``.  The forecast recouples it exactly as it recouples any
    initial-conditions file (MPAS-A ``atm_init_coupled_diagnostics``).
    ``analysis.form = "mpasout"``: only the fields of the RRFS v2 JEDI analysis stream
    (``woof.hex.jedi_contract.ANALYSIS_FIELDS``).  Such a file has no reference state of its
    own, so it starts a forecast only as a DA-cycling restart (``--restart-from``), which takes
    the reference state from the run's own init (``woof.hex.jedi_contract.recouple_da_cycling``).

``increment.nc``
    The fields of the MPAS-A v8.4.2 ``iau`` input stream (Registry ``tend_iau``):
    ``u_amb`` (edge-normal wind, m/s), ``rho_amb`` (dry density, kg/m3), ``theta_amb`` (K) and
    ``qv_amb qc_amb qr_amb qi_amb qs_amb qg_amb`` (kg/kg), each the analysis minus the
    background it was computed against; ``u_amb rho_amb theta_amb qv_amb`` are required.  Any
    other variable ``X`` of analysis.nc may carry ``X_amb`` too (number concentrations, ``w``).
    Values are float32 or float64; with float64 the background ``analysis - increment``
    (rounded to the analysis variable's type) is the producer's background bit for bit wherever
    the background value lies within the analysis value's float64 resolution (exactness is
    measured per variable when the increment is written).  A variable with no ``_amb`` is the
    analysis's own in the background.
    ``manifest.iau.window_s`` gives the window.

carried mesh (``grid.nc``, ``static.nc``, ``mesh-row.json``)
    A bundle may carry its mesh, so a forecast needs nothing but the bundle: ``static.nc``
    (SHA-256 ``static.sha256``), ``grid.nc`` (SHA-256 ``mesh.sha256``; omitted when the static
    file is also the grid, ``mesh.sha256 == static.sha256``) and optionally ``mesh-row.json``,
    a ``gpuwm-hex.mesh-rows/v1`` document with the one row named ``mesh.id`` whose ``grid`` and
    ``static`` are the bundle's own file names.

``noda.nc``
    The matched no-assimilation state (for example the ensemble mean of the same members never
    cycled), init form, for side-by-side forecasts.  Optional; the forecast never reads it.

The manifest holds no paths and no names of places or cases: the mesh and static are named by
their SHA-256, and every file by its SHA-256 and byte count.  ``notes`` (optional) is the
producer's own provenance: a reader records it and never acts on it.  ``forecast.args``
(optional) is the producer's own forecast configuration as door flags (timestep, executor,
physics; never a path or a host binding): the door applies each one the run does not name, so a
forecast from the bundle integrates the model the analysis was cycled with.  A reader refuses any file whose
bytes differ, and refuses a run whose mesh, static or vertical grid differs from the
bundle's, by name.

NetCDF here goes through netCDF4, as the init loader (``woof.hex.driver``) and the analysis
remap (``woof.hex.analysis_remap``) do.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import threading
from typing import Any, Mapping, Sequence

import numpy as np

SCHEMA = "mpas-analysis-bundle/1"
MANIFEST = "manifest.json"
ANALYSIS = "analysis.nc"
INCREMENT = "increment.nc"
NODA = "noda.nc"
GRID = "grid.nc"
STATIC = "static.nc"
MESH_ROW = "mesh-row.json"
#: Every file name a v1 bundle may list, and whether it is required.
BUNDLE_FILES: dict[str, bool] = {ANALYSIS: True, INCREMENT: False, NODA: False, GRID: False, STATIC: False,
                                 MESH_ROW: False}

ANALYSIS_KINDS = ("mean", "control", "member")
ANALYSIS_FORMS = ("init", "mpasout")
PHYSICS_MODES = ("correct", "exact")
#: MPAS's own clock text (xtime, --start-time).
TIME_FORMAT = "%Y-%m-%d_%H:%M:%S"
#: What a producer writes in the manifest: UTC, ISO 8601.  A reader takes either form.
ISO_TIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

#: The MPAS-A v8.4.2 ``iau`` stream (Registry.xml tend_iau; woof.hex.iau_v842.IAU_STREAM_FIELDS):
#: (increment variable, analysis variable, entity dimension).
IAU_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("u_amb", "u", "nEdges"),
    ("rho_amb", "rho", "nCells"),
    ("theta_amb", "theta", "nCells"),
    ("qv_amb", "qv", "nCells"),
    ("qc_amb", "qc", "nCells"),
    ("qr_amb", "qr", "nCells"),
    ("qi_amb", "qi", "nCells"),
    ("qs_amb", "qs", "nCells"),
    ("qg_amb", "qg", "nCells"),
)
#: The increments MPAS IAU cannot run without (woof.hex.iau_v842.iau_stream_request).
IAU_REQUIRED = ("u_amb", "rho_amb", "theta_amb", "qv_amb")
#: Suffix of every increment variable: ``X_amb`` is analysis ``X`` minus background ``X``.
AMB_SUFFIX = "_amb"
#: Variables an init-form analysis must carry for the forecast to start from it.
INIT_FORM_REQUIRED = ("xtime", "rho", "theta", "u", "w", "qv", "zgrid", "zz", "rho_base", "theta_base")
#: Variables the init loader prefers over rho/theta when present (woof.hex.driver
#: load_mpas_initial_state); a background rebuilt from rho/theta would be ignored under them.
COUPLED_ALIASES = ("rho_zz", "theta_m")

_TOP_KEYS = {"schema", "valid_time", "mesh", "static", "physics", "analysis", "iau", "versions", "files",
             "notes", "forecast"}
_REQUIRED_TOP = _TOP_KEYS - {"iau", "notes", "forecast"}
#: The model configuration a producer may hand a forecast (manifest ``forecast.args``): the door
#: flags that decide what model a forecast integrates, and how many values each takes.  Never a
#: host binding (devices, scratch, checkouts) and never a path.
FORECAST_FLAGS: dict[str, int] = {
    "--dt": 1, "--fullstep-local-timestep": 0, "--fullstep-lts-mode": 1, "--fullstep-lts-max-rate": 1,
    "--physics-suite": 1, "--physics": 1, "--physics-mode": 1, "--physics-correct-skip": 1,
    "--convection": 1, "--pbl-cadence": 1, "--preset": 1,
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+:-]{0,127}$")


class BundleRefusal(ValueError):
    """A bundle, or a run against it, that cannot be used; the message names why."""


# ---------------------------------------------------------------------------
# hashing and fingerprints
# ---------------------------------------------------------------------------
def sha256_file(path: str | Path) -> str:
    """The file's real SHA-256 (woof.hex.file_digest reuses it for an unchanged file version)."""

    from .file_digest import sha256_file as digest

    return digest(path)


def _zgrid_cells_by_interfaces(dataset: Any) -> np.ndarray:
    variable = dataset.variables["zgrid"]
    dims = tuple(d for d in variable.dimensions if d != "Time")
    key = tuple(0 if d == "Time" else slice(None) for d in variable.dimensions)
    values = np.asarray(variable[key], dtype=np.float64)
    if dims == ("nVertLevelsP1", "nCells"):
        values = values.T
    elif dims != ("nCells", "nVertLevelsP1"):
        raise BundleRefusal(f"zgrid has dimensions {variable.dimensions}; expected (nCells, nVertLevelsP1)")
    return np.ascontiguousarray(values)


def vertical_fingerprint(path: str | Path) -> dict[str, Any]:
    """What identifies a file's vertical grid: level count, the interface heights' digest, the top.

    ``zgrid_sha256`` is the SHA-256 of zgrid as little-endian float64 in (nCells,
    nVertLevelsP1) order, so the same grid stored as float32 or float64, with either axis
    first, has one fingerprint.  ``spec_sha256`` is the canonical hex vertical declaration a
    native-free init records (``gpuwm_hex_vertical_spec_sha256``), ``None`` otherwise.
    """

    from netCDF4 import Dataset

    with Dataset(path) as dataset:
        if "zgrid" not in dataset.variables:
            raise BundleRefusal(f"{Path(path).name} has no zgrid, so its vertical grid cannot be identified")
        zgrid = _zgrid_cells_by_interfaces(dataset)
        spec = (str(dataset.getncattr("gpuwm_hex_vertical_spec_sha256"))
                if "gpuwm_hex_vertical_spec_sha256" in dataset.ncattrs() else None)
    return {"levels": int(zgrid.shape[1] - 1),
            "zgrid_sha256": hashlib.sha256(zgrid.astype("<f8").tobytes()).hexdigest(),
            "z_top_m": float(np.max(zgrid[:, -1])),
            "spec_sha256": spec}


def describe_mesh(mesh_id: str, grid: str | Path, analysis: str | Path) -> dict[str, Any]:
    """The manifest's ``mesh`` block, measured from the grid file and the analysis."""

    from netCDF4 import Dataset

    with Dataset(grid) as dataset:
        cells = len(dataset.dimensions["nCells"])
    vertical = vertical_fingerprint(analysis)
    return {"id": str(mesh_id), "sha256": sha256_file(grid), "cells": int(cells),
            "levels": vertical["levels"], "vertical_spec": vertical}


# ---------------------------------------------------------------------------
# the manifest
# ---------------------------------------------------------------------------
def _refuse(message: str) -> BundleRefusal:
    return BundleRefusal(f"analysis bundle: {message}")


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
    """``YYYY-MM-DDThh:mm:ssZ`` (what producers write) or MPAS's ``YYYY-MM-DD_hh:mm:ss``, UTC."""

    for form in (ISO_TIME_FORMAT, TIME_FORMAT):
        try:
            return datetime.strptime(str(text), form)
        except ValueError:
            continue
    raise _refuse(f"valid_time {text!r} is neither YYYY-MM-DDThh:mm:ssZ nor YYYY-MM-DD_hh:mm:ss")


def mpas_time(text: Any) -> str:
    """A manifest time as MPAS's clock text (xtime, --start-time)."""

    return parse_valid_time(text).strftime(TIME_FORMAT)


def iso_time(text: Any) -> str:
    """A manifest or MPAS time as the manifest's ISO 8601 UTC text."""

    return parse_valid_time(text).strftime(ISO_TIME_FORMAT)


def validate_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Every rule of the v1 manifest; returns it with ``analysis.form`` filled in."""

    top = _exact_keys(manifest, "manifest", _REQUIRED_TOP, {"iau", "notes", "forecast"})
    if top["schema"] != SCHEMA:
        raise _refuse(f"schema {top['schema']!r} is not {SCHEMA!r}")
    _no_paths(dict(top), "manifest")
    parse_valid_time(top["valid_time"])
    mesh = _exact_keys(top["mesh"], "mesh", {"id", "sha256", "cells", "levels", "vertical_spec"})
    _ident(mesh["id"], "mesh.id")
    _sha(mesh["sha256"], "mesh.sha256")
    _positive_int(mesh["cells"], "mesh.cells")
    _positive_int(mesh["levels"], "mesh.levels")
    vertical = _exact_keys(mesh["vertical_spec"], "mesh.vertical_spec",
                           {"levels", "zgrid_sha256", "z_top_m"}, {"spec_sha256"})
    if vertical["levels"] != mesh["levels"]:
        raise _refuse(f"mesh.levels {mesh['levels']} and mesh.vertical_spec.levels {vertical['levels']} differ")
    _sha(vertical["zgrid_sha256"], "mesh.vertical_spec.zgrid_sha256")
    _positive_number(vertical["z_top_m"], "mesh.vertical_spec.z_top_m")
    if vertical.get("spec_sha256") is not None:
        _sha(vertical["spec_sha256"], "mesh.vertical_spec.spec_sha256")
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
    form = analysis.get("form", "init")
    if form not in ANALYSIS_FORMS:
        raise _refuse(f"analysis.form must be one of {ANALYSIS_FORMS}, got {form!r}")
    versions = _exact_keys(top["versions"], "versions", {"woof", "gpuwm_hex", "producer"})
    for key in ("woof", "gpuwm_hex", "producer"):
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
    if form == "mpasout" and INCREMENT in files:
        raise _refuse("an mpasout-form analysis restarts a cycled run (DA-cycling restart); it has no "
                      "background state to start an IAU window from, so it carries no increment.nc")
    _validate_carried_mesh(top, files)
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
        raise _refuse("forecast.args must be a list of strings (door flags and their values)")
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


def _validate_carried_mesh(top: Mapping[str, Any], files: Mapping[str, Any]) -> None:
    """A carried mesh is the manifest's own mesh and static, whole."""

    carried = [name for name in (GRID, STATIC, MESH_ROW) if name in files]
    if not carried:
        return
    if STATIC not in files:
        raise _refuse(f"files lists {carried} without {STATIC}; a carried mesh carries its static too, or a "
                      "forecast would bind a grid with no terrain under it")
    if files[STATIC]["sha256"] != top["static"]["sha256"]:
        raise _refuse(f"{STATIC} has SHA-256 {files[STATIC]['sha256']}; static.sha256 says "
                      f"{top['static']['sha256']}.  The carried static must be the one the analysis was made over")
    if GRID in files:
        if files[GRID]["sha256"] != top["mesh"]["sha256"]:
            raise _refuse(f"{GRID} has SHA-256 {files[GRID]['sha256']}; mesh.sha256 says {top['mesh']['sha256']}")
    elif top["mesh"]["sha256"] != top["static"]["sha256"]:
        raise _refuse(f"the bundle carries {STATIC} and no {GRID}, and mesh.sha256 is not static.sha256; the "
                      "static file serves as the grid only when it is the grid file")


def carried_grid_name(manifest: Mapping[str, Any]) -> str | None:
    """The bundle file that is the carried mesh's grid, or None when the mesh is not carried."""

    files = manifest["files"]
    if STATIC not in files:
        return None
    return GRID if GRID in files else STATIC


def manifest_bytes(manifest: Mapping[str, Any]) -> bytes:
    """The canonical serialisation the manifest's own digest is taken over."""

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
    source: str

    @property
    def valid_time(self) -> str:
        """MPAS clock text (the analysis's xtime, the forecast's --start-time)."""
        return mpas_time(self.manifest["valid_time"])

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
    def carried_grid(self) -> Path | None:
        name = carried_grid_name(self.manifest)
        return None if name is None else self.files[name]

    @property
    def carried_static(self) -> Path | None:
        return self.files.get(STATIC)

    @property
    def carried_row(self) -> Path | None:
        return self.files.get(MESH_ROW)

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
        return {"schema": SCHEMA, "source": self.source, "root": str(self.root),
                "manifest_sha256": self.manifest_sha256,
                "valid_time": self.valid_time, "form": self.form,
                "mesh": dict(self.manifest["mesh"]), "static": dict(self.manifest["static"]),
                "physics": dict(self.manifest["physics"]), "analysis": dict(self.manifest["analysis"]),
                "iau": None if self.manifest.get("iau") is None else dict(self.manifest["iau"]),
                "versions": dict(self.manifest["versions"]),
                "files": {name: dict(entry) for name, entry in self.manifest["files"].items()},
                "files_verified": sorted(self.files),
                "forecast": self.manifest.get("forecast"),
                "notes": self.manifest.get("notes")}


def _parse_s3(url: str) -> tuple[str, str]:
    rest = url[len("s3://"):]
    bucket, _, key = rest.partition("/")
    if not bucket:
        raise _refuse(f"{url} names no bucket")
    if key.endswith(MANIFEST):
        key = key[: -len(MANIFEST)]
    if key and not key.endswith("/"):
        key += "/"
    return bucket, key


def _s3_download(bucket: str, key: str, destination: Path) -> str:
    """One object to ``destination``: boto3 when installed, else the AWS CLI; refused otherwise."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    try:
        import boto3  # type: ignore[import-not-found]
    except ImportError:
        boto3 = None
    if boto3 is not None:
        boto3.client("s3").download_file(bucket, key, str(partial))
        partial.replace(destination)
        return "boto3"
    aws = shutil.which("aws")
    if aws is None:
        raise _refuse(f"s3://{bucket}/{key} cannot be fetched: neither boto3 nor the aws command is "
                      "installed here.  Install one (pip install boto3), or copy the bundle to a "
                      "directory and pass that directory")
    completed = subprocess.run([aws, "s3", "cp", "--only-show-errors", f"s3://{bucket}/{key}", str(partial)],
                               capture_output=True, text=True)
    if completed.returncode != 0:
        raise _refuse(f"aws s3 cp s3://{bucket}/{key} failed: {completed.stderr.strip()}")
    partial.replace(destination)
    return "aws-cli"


def fetch_bundle(url: str, destination: str | Path) -> Path:
    """Copy a bundle from ``s3://bucket/prefix/`` (or its manifest.json) into ``destination``.

    The manifest is read first and only the files it lists are fetched; each is checked by
    :func:`load_bundle` before anything reads it.
    """

    bucket, prefix = _parse_s3(url)
    destination = Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise _refuse(f"{destination} exists and is not empty; a fetched bundle goes into a fresh "
                      "directory so files of two bundles cannot mix")
    destination.mkdir(parents=True, exist_ok=True)
    _s3_download(bucket, prefix + MANIFEST, destination / MANIFEST)
    manifest = validate_manifest(json.loads((destination / MANIFEST).read_text(encoding="utf-8")))
    for name in manifest["files"]:
        _s3_download(bucket, prefix + name, destination / name)
    return destination


def load_bundle(source: str | Path, *, fetch_to: str | Path | None = None) -> AnalysisBundle:
    """Read and check a bundle: every listed file present, at its byte count and SHA-256.

    ``source`` is the bundle directory, its manifest.json, or ``s3://bucket/prefix/``
    (fetched into ``fetch_to`` first).
    """

    text = str(source)
    origin = text
    if text.startswith("s3://"):
        if fetch_to is None:
            raise _refuse(f"{text} is an object-store bundle and no local directory was given to fetch it into")
        root = fetch_bundle(text, fetch_to)
    else:
        path = Path(text).expanduser()
        root = path.parent if path.name == MANIFEST else path
        if not (root / MANIFEST).is_file():
            raise _refuse(f"{root} holds no {MANIFEST}")
        origin = str(root.resolve())
    raw = (Path(root) / MANIFEST).read_bytes()
    try:
        manifest = validate_manifest(json.loads(raw.decode("utf-8")))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _refuse(f"{MANIFEST} is not JSON: {error}") from None
    files: dict[str, Path] = {}
    for name, entry in manifest["files"].items():
        path = Path(root) / name
        if not path.is_file():
            raise _refuse(f"{name} is listed in the manifest and absent from {root}")
        size = path.stat().st_size
        if size != entry["bytes"]:
            raise _refuse(f"{name} is {size} bytes; the manifest says {entry['bytes']}")
        digest = sha256_file(path)
        if digest != entry["sha256"]:
            raise _refuse(f"{name} has SHA-256 {digest}; the manifest says {entry['sha256']}.  A changed "
                          "or partial file would start the forecast from a state nobody analysed")
        files[name] = path.resolve()
    for name in BUNDLE_FILES:
        if name not in files and (Path(root) / name).exists():
            raise _refuse(f"{name} is in {root} but not listed in the manifest, so its bytes are not "
                          "covered by any digest; list it or remove it")
    if MESH_ROW in files:
        read_carried_row(files[MESH_ROW], manifest, root)
    return AnalysisBundle(root=Path(root).resolve(), manifest=manifest,
                          manifest_sha256=hashlib.sha256(raw).hexdigest(), files=files, source=origin)


def write_manifest(root: str | Path, *, valid_time: str, mesh: Mapping[str, Any], static: Mapping[str, Any],
                   physics: Mapping[str, Any], analysis: Mapping[str, Any], versions: Mapping[str, Any],
                   iau: Mapping[str, Any] | None = None, notes: Mapping[str, Any] | None = None,
                   forecast_args: Sequence[str] | None = None) -> Path:
    """Hash the bundle's files and write its manifest (the producer's side).

    ``root`` already holds analysis.nc and any of increment.nc/noda.nc.  The manifest is
    validated before it is written, so a producer cannot publish one a reader would refuse.
    """

    root = Path(root)
    files = {}
    for name in BUNDLE_FILES:
        path = root / name
        if path.is_file():
            files[name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    manifest = {"schema": SCHEMA, "valid_time": iso_time(valid_time), "mesh": dict(mesh), "static": dict(static),
                "physics": dict(physics), "analysis": dict(analysis), "versions": dict(versions),
                "files": files}
    if iau is not None:
        manifest["iau"] = dict(iau)
    if notes is not None:
        manifest["notes"] = json.loads(json.dumps(dict(notes)))
    if forecast_args:
        manifest["forecast"] = {"args": [str(item) for item in forecast_args]}
    validate_manifest(manifest)
    if ANALYSIS in files:
        check_analysis_file(root / ANALYSIS, valid_time=mpas_time(valid_time),
                            form=str(manifest["analysis"].get("form", "init")))
    if INCREMENT in files:
        read_increment(root / INCREMENT)
        check_increment_matches(root / INCREMENT, root / ANALYSIS)
    if MESH_ROW in files:
        read_carried_row(root / MESH_ROW, manifest, root)
    target = root / MANIFEST
    partial = target.with_name(MANIFEST + ".partial")
    partial.write_bytes(manifest_bytes(manifest))
    partial.replace(target)
    return target


# ---------------------------------------------------------------------------
# file contents
# ---------------------------------------------------------------------------
def _xtime(dataset: Any, index: int = 0) -> str:
    variable = dataset.variables["xtime"]
    row = np.asarray(variable[index] if variable.dimensions[0] == "Time" else variable[:])
    return row.tobytes().decode("ascii", "replace").strip("\x00 ")


def check_analysis_file(path: str | Path, *, valid_time: str, form: str = "init") -> dict[str, Any]:
    """The analysis file carries what its form promises, at the bundle's valid time."""

    from netCDF4 import Dataset

    from .jedi_contract import RECOUPLED_INPUTS

    with Dataset(path) as dataset:
        names = set(dataset.variables)
        required = INIT_FORM_REQUIRED if form == "init" else ("xtime",) + tuple(
            n for n in RECOUPLED_INPUTS if n != "scalars") + ("qv",)
        missing = sorted(set(required) - names)
        if missing:
            raise _refuse(f"{Path(path).name} ({form} form) lacks {missing}")
        if "xtime" in names:
            times = [_xtime(dataset, i) for i in range(len(dataset.dimensions["Time"]))]
            if valid_time not in times:
                raise _refuse(f"{Path(path).name} holds no record at the valid time {valid_time} "
                              f"(records: {times})")
        dims = {name: len(dim) for name, dim in dataset.dimensions.items()}
    return {"form": form, "dimensions": {k: dims[k] for k in ("nCells", "nEdges", "nVertLevels") if k in dims}}


def _field(dataset: Any, name: str, entity: str, level: str = "nVertLevels") -> np.ndarray:
    """A [level, entity] float64 copy, whatever the stored axis order."""

    variable = dataset.variables[name]
    variable.set_auto_mask(False)
    dims = tuple(d for d in variable.dimensions if d != "Time")
    key = tuple(0 if d == "Time" else slice(None) for d in variable.dimensions)
    values = np.asarray(variable[key], dtype=np.float64)
    if dims == (entity, level):
        return values.T
    if dims == (level, entity):
        return values
    raise _refuse(f"{name} has dimensions {variable.dimensions}; expected ({entity}, {level})")


def _store(dataset: Any, name: str, values: np.ndarray, entity: str, level: str = "nVertLevels") -> None:
    variable = dataset.variables[name]
    dims = tuple(d for d in variable.dimensions if d != "Time")
    data = values.T if dims == (entity, level) else values
    key = tuple(0 if d == "Time" else slice(None) for d in variable.dimensions)
    variable[key] = data.astype(variable.dtype)


def read_increment(path: str | Path, *, n_cells: int | None = None, n_edges: int | None = None,
                   n_levels: int | None = None) -> dict[str, np.ndarray]:
    """The MPAS ``iau`` stream fields of an increment file, float32 [level, entity] (the model's
    precision, as MPAS reads its iau stream), named by their increment variable.  Any other
    ``X_amb`` the file carries is not part of MPAS's stream: :func:`increment_variables` lists it."""

    from netCDF4 import Dataset

    out: dict[str, np.ndarray] = {}
    with Dataset(path) as dataset:
        missing = [name for name in IAU_REQUIRED if name not in dataset.variables]
        if missing:
            raise _refuse(f"{Path(path).name} lacks {missing}; MPAS IAU needs u, rho, theta and qv increments")
        for name, _source, entity in IAU_FIELDS:
            if name not in dataset.variables:
                continue
            values = _field(dataset, name, entity)
            if not np.all(np.isfinite(values)):
                raise _refuse(f"{Path(path).name}: {name} carries non-finite values")
            out[name] = values.astype(np.float32)
        dims = {name: len(dim) for name, dim in dataset.dimensions.items()}
    for want, dim in ((n_cells, "nCells"), (n_edges, "nEdges"), (n_levels, "nVertLevels")):
        if want is not None and dims.get(dim) != int(want):
            raise _refuse(f"{Path(path).name}: {dim} is {dims.get(dim)}, this run's is {want}")
    return out


#: The dimensions an increment variable may span (a prognostic of the init stream).
_INCREMENT_ENTITIES = ("nCells", "nEdges")
_INCREMENT_LEVELS = ("nVertLevels", "nVertLevelsP1")


def _record(variable: Any) -> np.ndarray:
    """The variable's first record (or whole value when it has no Time), raw, unmasked."""

    variable.set_auto_mask(False)
    key = tuple(0 if d == "Time" else slice(None) for d in variable.dimensions)
    return np.asarray(variable[key])


def _spatial_dims(variable: Any) -> tuple[str, ...]:
    return tuple(d for d in variable.dimensions if d != "Time")


def _non_negative(name: str) -> bool:
    """Mixing ratios, number concentrations and volumes: a rebuilt background below zero is rounding."""

    return name.startswith(("q", "n", "vol"))


def increment_variables(path: str | Path, analysis: str | Path | None = None) -> dict[str, str]:
    """Every ``X_amb`` in an increment file, mapped to the analysis variable ``X`` it increments.

    With ``analysis``, each ``X`` must be a variable of that file over the same dimensions,
    because the background is formed variable by variable as analysis minus increment.
    """

    from netCDF4 import Dataset

    out: dict[str, str] = {}
    with Dataset(path) as dataset:
        for name, variable in dataset.variables.items():
            if not name.endswith(AMB_SUFFIX):
                continue
            out[name] = name[: -len(AMB_SUFFIX)]
            dims = _spatial_dims(variable)
            if (len(dims) != 2 or dims[0] not in _INCREMENT_ENTITIES or dims[1] not in _INCREMENT_LEVELS) \
                    and name not in {field for field, _s, _e in IAU_FIELDS}:
                raise _refuse(f"{Path(path).name}: {name} spans {variable.dimensions}; an increment variable "
                              "is a prognostic over (nCells|nEdges, nVertLevels|nVertLevelsP1)")
            if variable.dtype.kind != "f":
                raise _refuse(f"{Path(path).name}: {name} is {variable.dtype}, not floating point")
    if analysis is not None:
        with Dataset(path) as increment, Dataset(analysis) as dataset:
            for name, source in out.items():
                if source not in dataset.variables:
                    raise _refuse(f"{Path(path).name} carries {name} and {Path(analysis).name} has no {source}; the "
                                  "background is formed as analysis minus increment, variable by variable")
                if set(_spatial_dims(increment.variables[name])) != set(_spatial_dims(dataset.variables[source])):
                    raise _refuse(f"{name} spans {increment.variables[name].dimensions} and {source} spans "
                                  f"{dataset.variables[source].dimensions}")
    return out


def check_increment_matches(increment: str | Path, analysis: str | Path) -> dict[str, str]:
    """An increment file whose every variable increments a variable of the analysis."""

    return increment_variables(increment, analysis)


def write_increment(analysis: str | Path, background: str | Path, out: str | Path, *,
                    precision: str = "f8", fields: str = "differing") -> dict[str, Any]:
    """analysis minus background in MPAS IAU form (the producer's side, and the door's after a remap).

    ``precision`` "f8" (the default) stores the float64 difference, so the background a reader
    rebuilds as analysis minus increment is ``background`` bit for bit wherever the background
    value lies within the analysis value's float64 resolution (``exact_background`` in the
    receipt, measured per variable); "f4" stores what MPAS's single-precision iau stream holds.
    ``fields`` "differing" (the default) adds ``X_amb`` for every other prognostic (cells or
    edges by levels) that differs, w and number concentrations among them; "mpas" writes the
    MPAS iau stream fields only.  The MPAS stream's required four are always written.

    THE COST THIS REMOVES (own24 zoom globe, 167,254 cells x 55 levels, DA member leg on an RTX 5090,
    hex 0.4.9, 2026-10-28): the forecast door built every member's increment here on one thread for 7.5 s
    while the card waited.  The file was defined variable by variable after data had been written, so each
    ``createVariable`` and ``setncattr`` reopened the classic header, grew it and moved every word already
    written (about 41 % of the samples); whole-variable reads, float64 differences, the exactness check and
    the rms took the rest, one variable after another, each in freshly allocated arrays.  Now the file is
    defined once before any data, and every variable is compared, differenced, checked and written in
    blocks of words on host threads, each block read straight from the classic files into scratch its
    thread reuses (``_IncrementWords``).  The bytes are the ones the one-thread writer wrote: the same
    header, the same elementwise float64 arithmetic per word, and the rms is NumPy's own pairwise sum of
    the squares, assembled from its subtrees (``_pairwise_leaves``; checked against ``np.mean`` once per
    process, else the whole array is summed as before).
    """

    from netCDF4 import Dataset

    from . import classic_netcdf, host_parallel

    if precision not in ("f4", "f8"):
        raise _refuse(f"increment precision {precision!r} is not f4 or f8")
    if fields not in ("differing", "mpas"):
        raise _refuse(f"increment fields {fields!r} is not differing or mpas")
    out = Path(out)
    partial = out.with_name(out.name + ".partial")
    workers = host_parallel.threads()
    with Dataset(analysis) as a, Dataset(background) as b:
        for dim in ("nCells", "nEdges", "nVertLevels"):
            if len(a.dimensions[dim]) != len(b.dimensions[dim]):
                raise _refuse(f"analysis and background differ in {dim}; an increment needs one mesh")
        if "indexToCellID" in a.variables and "indexToCellID" in b.variables and not np.array_equal(
                np.asarray(a["indexToCellID"][:]), np.asarray(b["indexToCellID"][:])):
            raise _refuse("analysis and background order their cells differently")
        dimensions = [(dim, len(a.dimensions[dim])) for dim in ("nCells", "nEdges", "nVertLevels", "nVertLevelsP1")
                      if dim in a.dimensions]
        stamp = _xtime(a)
        mpas_sources = {source: name for name, source, _entity in IAU_FIELDS}
        names: list[tuple[str, str]] = []
        for name, source, _entity in IAU_FIELDS:
            if source not in a.variables or source not in b.variables:
                if name in IAU_REQUIRED:
                    raise _refuse(f"{source} is missing from the analysis or the background")
                continue
            names.append((name, source))
        candidates: list[str] = []
        if fields == "differing":
            for source, variable in a.variables.items():
                dims = _spatial_dims(variable)
                if (source in mpas_sources or source not in b.variables or variable.dtype.kind != "f"
                        or len(dims) != 2 or dims[0] not in _INCREMENT_ENTITIES or dims[1] not in _INCREMENT_LEVELS
                        or _spatial_dims(b.variables[source]) != dims):
                    continue
                candidates.append(source)
        wanted = [source for _name, source in names] + candidates
        words_a = _IncrementWords.open(a, analysis, wanted)
        words_b = _IncrementWords.open(b, background, wanted)
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="hex-increment") if workers > 1 else None
    scratch = _Scratch()
    try:
        if candidates:
            # np.array_equal: arrays of different shapes are not equal; otherwise every word compares equal.
            equal = {source: words_a[source].shape == words_b[source].shape for source in candidates}
            tasks = [(source, lo, hi) for source in candidates if equal[source]
                     for lo, hi in _blocks(words_a[source].size)]
            results = _map(pool, lambda task: _equal_block(words_a[task[0]], words_b[task[0]], task[1], task[2],
                                                           scratch), tasks)
            for (source, _lo, _hi), same in zip(tasks, results):
                if not same:
                    equal[source] = False
            names += [(source + AMB_SUFFIX, source) for source in candidates if not equal[source]]
        layout: list[tuple[str, str, tuple[str, ...]]] = []
        for name, source in names:
            va, vb = words_a[source], words_b[source]
            if va.dims != vb.dims:
                raise _refuse(f"{source} is stored over {va.dimensions} in the analysis and {vb.dimensions} in "
                              "the background")
            if va.shape != vb.shape:
                raise _refuse(f"{source} is {va.shape} in the analysis and {vb.shape} in the background")
            layout.append((name, source, va.dims))
        # Define everything, then write: a variable or attribute added after data reopens the classic header
        # and moves every word already written.  No fill: every word of every variable is written below.
        with Dataset(partial, "w", format="NETCDF3_64BIT_DATA") as o:
            o.set_fill_off()
            o.createDimension("Time", None)
            o.createDimension("StrLen", 64)
            for dim, size in dimensions:
                o.createDimension(dim, size)
            xtime = o.createVariable("xtime", "S1", ("Time", "StrLen"))
            for name, source, dims in layout:
                o.createVariable(name, precision, ("Time",) + dims).setncattr("increment_of", source)
            o.setncattr("increment_of", "analysis minus background (MPAS-A v8.4.2 iau stream, Registry tend_iau; "
                        "other X_amb: analysis X minus background X)")
            xtime[0, :] = np.array(list(stamp.ljust(64)), dtype="S1")
        found = classic_netcdf.extents(partial)
        if found is None:
            raise _refuse(f"{partial.name} is not a classic netCDF file after its definition")
        table = found[0]
        rms: dict[str, float] = {}
        exact: dict[str, bool] = {}
        pairwise = _pairwise_matches_numpy()
        handle = os.open(partial, os.O_WRONLY | getattr(os, "O_BINARY", 0))
        try:
            for name, source, _dims in layout:
                va, vb = words_a[source], words_b[source]
                ranges, tree = _pairwise_leaves(va.size)
                squares = None if pairwise else np.empty(va.size, dtype=np.float64)
                begin = table[name].begin
                results = _map(pool, lambda span: _increment_block(va, vb, span[0], span[1], precision, handle,
                                                                   begin, squares, scratch), ranges)
                if not all(finite for finite, _same, _leaf in results):
                    raise _refuse(f"{source} increment is not finite")
                exact[name] = all(same for _finite, same, _leaf in results)
                if pairwise:
                    rms[name] = float(np.sqrt(_combine(tree, [leaf for _f, _s, leaf in results]) / va.size))
                else:
                    rms[name] = float(np.sqrt(np.mean(squares.reshape(va.shape))))
        finally:
            os.close(handle)
    finally:
        if pool is not None:
            pool.shutdown(wait=True)
        for words in (words_a, words_b):
            for value in words.values():
                value.close()
    partial.replace(out)
    return {"increment": str(out), "valid_time": stamp, "precision": precision, "rms": rms,
            "exact_background": exact, "exact_background_all": bool(exact) and all(exact.values())}


#: Words per block of the increment writer's host-thread work.  2 MiB of float64 stays under NumPy's 4 MiB
#: huge-page advice, so each thread's scratch faults in once, in small pages, and is reused for every block.
_INCREMENT_BLOCK = 1 << 18


class _IncrementWords:
    """One variable's first record (``_record``'s array) as flat words, read in ranges on any thread.

    A variable the classic reader handles (``classic_netcdf.readable``, the record leading, netCDF4's
    shape and dtype) is read straight from the file at its offset and put in native order; any other is
    read whole through netCDF4 on the caller's thread, as ``_record`` reads it, and served from memory.
    """

    def __init__(self, variable: Any, *, path: Path | None = None, begin: int = 0, stored: Any = None,
                 array: np.ndarray | None = None) -> None:
        self.dimensions = tuple(variable.dimensions)
        self.dims = _spatial_dims(variable)
        self.dtype = np.dtype(variable.dtype)
        if array is not None:
            self.array = np.ascontiguousarray(array)
            self.shape = tuple(self.array.shape)
            self.dtype = self.array.dtype
        else:
            self.array = None
            self.shape = tuple(int(n) for n, d in zip(variable.shape, variable.dimensions) if d != "Time")
        self.size = int(np.prod(self.shape, dtype=np.int64))
        self.begin = begin
        self.swap = stored is not None and np.dtype(stored) != self.dtype
        self.fd = None if path is None else os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0))

    @classmethod
    def open(cls, dataset: Any, path: str | Path, names: Sequence[str]) -> dict[str, "_IncrementWords"]:
        from . import classic_netcdf

        path = Path(path)
        found = None
        if hasattr(os, "preadv") and not classic_netcdf.disabled():
            try:
                found = classic_netcdf.extents(path)
            except (OSError, ValueError, KeyError, struct.error):
                found = None
        table, numrecs = found if found is not None else ({}, 0)
        out: dict[str, _IncrementWords] = {}
        for name in names:
            variable = dataset.variables[name]
            extent = table.get(name)
            if (extent is not None and classic_netcdf.readable(extent, numrecs, extent.attributes)
                    and tuple(extent.shape) == tuple(variable.shape)
                    and np.dtype(extent.dtype).newbyteorder("=") == np.dtype(variable.dtype)
                    and ("Time" not in variable.dimensions or variable.dimensions[0] == "Time")):
                out[name] = cls(variable, path=path, begin=extent.begin, stored=extent.dtype)
            else:
                out[name] = cls(variable, array=_record(variable))
        return out

    def read(self, start: int, stop: int, scratch: np.ndarray) -> np.ndarray:
        """Words ``start:stop`` in native order (a view of ``scratch`` when read from the file)."""

        if self.array is not None:
            return self.array.reshape(-1)[start:stop]
        view = scratch[:stop - start]
        raw = memoryview(view.view(np.uint8))
        offset = self.begin + start * self.dtype.itemsize
        done = 0
        while done < len(raw):
            got = os.preadv(self.fd, [raw[done:]], offset + done)
            if got <= 0:
                raise EOFError(f"short read of a variable at byte {offset + done}")
            done += got
        if self.swap:
            view.byteswap(inplace=True)
        return view

    def close(self) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


class _Scratch(threading.local):
    """Per-thread scratch of ``_INCREMENT_BLOCK`` words by role and dtype, allocated once and reused."""

    def get(self, role: str, dtype: Any) -> np.ndarray:
        buffers = self.__dict__.setdefault("buffers", {})
        key = (role, np.dtype(dtype).str)
        value = buffers.get(key)
        if value is None:
            value = buffers[key] = np.empty(_INCREMENT_BLOCK, dtype=dtype)
        return value


def _blocks(size: int) -> list[tuple[int, int]]:
    return [(lo, min(size, lo + _INCREMENT_BLOCK)) for lo in range(0, size, _INCREMENT_BLOCK)]


def _map(pool: ThreadPoolExecutor | None, function: Any, items: Sequence[Any]) -> list[Any]:
    return [function(item) for item in items] if pool is None else list(pool.map(function, items))


def _equal_block(va: _IncrementWords, vb: _IncrementWords, lo: int, hi: int, scratch: _Scratch) -> bool:
    """``np.array_equal`` of the two variables' words ``lo:hi`` (elementwise ==: -0.0 equals 0.0, NaN never)."""

    x = va.read(lo, hi, scratch.get("a", va.dtype))
    y = vb.read(lo, hi, scratch.get("b", vb.dtype))
    flags = scratch.get("flag", np.bool_)[:hi - lo]
    np.equal(x, y, out=flags)
    return bool(flags.all())


def _increment_block(va: _IncrementWords, vb: _IncrementWords, lo: int, hi: int, precision: str, handle: int,
                     begin: int, squares: np.ndarray | None, scratch: _Scratch) -> tuple[bool, bool, float]:
    """Write words ``lo:hi`` of one increment variable; the one-thread writer's expressions, word for word.

    ``delta = av.astype(f8) - bv.astype(f8)``, ``stored`` (delta, or delta as float32),
    ``rebuilt = (av.astype(f8) - stored.astype(f8)).astype(av.dtype)`` compared bitwise with ``bv``, and the
    squares ``delta * delta``.  Returns whether every difference is finite, whether the rebuilt background
    is ``bv`` bit for bit, and NumPy's sum of these squares (a leaf of ``_pairwise_leaves``).
    """

    count = hi - lo
    x = va.read(lo, hi, scratch.get("a", va.dtype))
    y = vb.read(lo, hi, scratch.get("b", vb.dtype))
    delta = scratch.get("delta", np.float64)[:count]
    np.subtract(x, y, out=delta, dtype=np.float64)
    flags = scratch.get("flag", np.bool_)[:count]
    np.isfinite(delta, out=flags)
    finite = bool(flags.all())
    if precision == "f8":
        stored = delta
    else:
        stored = scratch.get("stored", np.float32)[:count]
        np.copyto(stored, delta, casting="unsafe")
    wide = scratch.get("wide", np.float64)[:count]
    np.subtract(x, stored, out=wide, dtype=np.float64)
    rebuilt = scratch.get("rebuilt", va.dtype)[:count]
    np.copyto(rebuilt, wide, casting="unsafe")
    if rebuilt.dtype.itemsize == y.dtype.itemsize:
        np.equal(rebuilt.view(f"u{rebuilt.dtype.itemsize}"), y.view(f"u{y.dtype.itemsize}"), out=flags)
        same = bool(flags.all())
    else:
        same = False
    np.multiply(delta, delta, out=wide)
    leaf = float(np.add.reduce(wide))
    if squares is not None:
        squares[lo:hi] = wide
    words = scratch.get("big", stored.dtype.newbyteorder(">"))[:count]
    np.copyto(words, stored)
    _write_at(handle, words, begin + lo * stored.dtype.itemsize)
    return finite, same, leaf


def _pairwise_leaves(size: int, leaf: int = _INCREMENT_BLOCK) -> tuple[list[tuple[int, int]], Any]:
    """NumPy's pairwise summation tree over ``size`` words, cut into subtrees of at most ``leaf`` words.

    DOUBLE_pairwise_sum (numpy/_core/src/umath/loops_utils.h.src) sums n > 128 words as the sum of its first
    n2 = n // 2 - (n // 2) % 8 words plus the rest, each the same way; a subtree's sum is what np.add.reduce
    gives for that range alone.  Returns the leaf ranges in order and the tree (leaf indices, nested pairs).
    """

    ranges: list[tuple[int, int]] = []

    def split(start: int, count: int) -> Any:
        if count <= leaf or count <= 128:
            ranges.append((start, start + count))
            return len(ranges) - 1
        half = count // 2
        half -= half % 8
        return split(start, half), split(start + half, count - half)

    return ranges, split(0, size)


def _combine(tree: Any, sums: Sequence[float]) -> float:
    if isinstance(tree, int):
        return sums[tree]
    return _combine(tree[0], sums) + _combine(tree[1], sums)


_PAIRWISE_EXACT: bool | None = None


def _pairwise_matches_numpy() -> bool:
    """Whether leaf sums combined in ``_pairwise_leaves``' tree give ``np.mean``'s sum bit for bit here.

    Checked once per process on a two-dimensional float64 array, larger than NumPy's iterator buffer,
    whose values span twelve orders of magnitude so that any other summation order shows.  False sends
    the writer back to one ``np.mean`` over the whole array of squares.
    """

    global _PAIRWISE_EXACT
    if _PAIRWISE_EXACT is None:
        rng = np.random.default_rng(20261028)
        values = (rng.standard_normal(9001 * 55) * 10.0 ** rng.uniform(-6.0, 6.0, 9001 * 55)) ** 2
        grid = values.reshape(9001, 55)
        ranges, tree = _pairwise_leaves(values.size, leaf=4096)
        total = _combine(tree, [float(np.add.reduce(values[lo:hi])) for lo, hi in ranges])
        _PAIRWISE_EXACT = bool(total == float(np.add.reduce(grid, axis=None))
                               and float(np.sqrt(total / values.size)) == float(np.sqrt(np.mean(grid))))
    return _PAIRWISE_EXACT


_SEEK_WRITE = threading.Lock()


def _write_at(handle: int, words: np.ndarray, offset: int) -> None:
    """Write ``words``' bytes at byte ``offset`` of the open file ``handle`` (threads may call it together)."""

    view = memoryview(words.reshape(-1).view(np.uint8))
    if not hasattr(os, "pwrite"):  # Windows: one writer at a time through the shared file position
        with _SEEK_WRITE:
            os.lseek(handle, offset, os.SEEK_SET)
            done = 0
            while done < len(view):
                done += os.write(handle, view[done:])
        return
    done = 0
    while done < len(view):
        wrote = os.pwrite(handle, view[done:], offset + done)
        if wrote <= 0:
            raise OSError(f"short write of an increment variable at byte {offset + done}")
        done += wrote


def write_background(analysis: str | Path, increment: str | Path, out: str | Path) -> dict[str, Any]:
    """The background an IAU window starts from: the analysis minus its increment.

    Every ``X_amb`` the increment carries changes ``X``, in float64 and rounded to ``X``'s type,
    so a float64 increment gives the producer's background bit for bit wherever the background
    value lies within the analysis value's float64 resolution; every other field (land
    and surface state, and any prognostic with no ``_amb``) stays the analysis's.  A mixing
    ratio, number concentration or volume a float32 increment leaves below zero by rounding is
    set to zero and counted; with a float64 increment the producer's own values (transport
    roundoff below zero included) are kept, and counted.
    """

    from netCDF4 import Dataset

    out = Path(out)
    partial = out.with_name(out.name + ".partial")
    read_increment(increment)  # the MPAS stream's required four, finite
    variables = increment_variables(increment, analysis)
    with Dataset(analysis) as source:
        aliases = sorted(set(COUPLED_ALIASES) & set(source.variables))
        if aliases:
            raise _refuse(f"{Path(analysis).name} carries {aliases}, which the init loader reads in "
                          "preference to rho/theta, so a background rebuilt from rho/theta would be "
                          "ignored; write the analysis in init form (rho and theta only)")
    shutil.copyfile(analysis, partial)
    clipped: dict[str, dict[str, float]] = {}
    kept: dict[str, dict[str, float]] = {}
    changed: list[str] = []
    precision: dict[str, str] = {}
    with Dataset(increment) as inc, Dataset(partial, "r+") as dataset:
        for name, source in sorted(variables.items()):
            target = dataset.variables[source]
            current = _record(target)
            delta_var = inc.variables[name]
            delta = _record(delta_var).astype(np.float64)
            order = [_spatial_dims(delta_var).index(d) for d in _spatial_dims(target)]
            delta = np.transpose(delta, order)
            if not np.all(np.isfinite(delta)):
                raise _refuse(f"{Path(increment).name}: {name} carries non-finite values")
            values = current.astype(np.float64) - delta
            if _non_negative(source):
                negative = values < 0
                if negative.any():
                    record = {"values": int(negative.sum()), "min": float(values[negative].min())}
                    if delta_var.dtype == np.float32:
                        # a float32 increment can leave rounding below zero that the producer's
                        # background never had
                        clipped[source] = record
                        values = np.where(negative, 0.0, values)
                    else:
                        # float64 rebuilds the producer's own background, whose transport
                        # roundoff (1e-19 kg/kg) the model meets on its own; clipping it would
                        # start the forecast from another state (measured: the frame-0 qr, qs
                        # and qg of a bundle run differed from the producer's own run)
                        kept[source] = record
            key = tuple(0 if d == "Time" else slice(None) for d in target.dimensions)
            target[key] = values.astype(current.dtype)
            changed.append(source)
            precision[source] = str(delta_var.dtype)
        dataset.setncattr("analysis_bundle_role", "background: the analysis minus increment.nc")
    partial.replace(out)
    return {"background": str(out), "changed": changed, "increment_precision": precision,
            "clipped_negative": clipped, "negative_kept_from_producer": kept,
            "unchanged": "every variable with no X_amb (land and surface state among them): the analysis's own"}


# ---------------------------------------------------------------------------
# a carried mesh
# ---------------------------------------------------------------------------
_ROW_PATH_KEYS = ("generator_receipt", "static_receipt", "lbc_source", "cull_receipt")


def read_carried_row(path: str | Path, manifest: Mapping[str, Any], root: str | Path) -> dict[str, Any]:
    """The bundle's mesh-row.json: one ``gpuwm-hex.mesh-rows/v1`` row for the carried mesh.

    Its ``grid`` and ``static`` are the bundle's own file names, its digests and sizes the
    manifest's, and no other field is a path (a path in a bundle would carry the producer's
    machine to every reader).
    """

    from .mesh_rows import ROWS_SCHEMA, MeshRow, MeshRowRefusal

    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise _refuse(f"{MESH_ROW} is not readable JSON: {error}") from None
    if not isinstance(document, Mapping) or document.get("schema") != ROWS_SCHEMA:
        raise _refuse(f"{MESH_ROW} is not a {ROWS_SCHEMA} document")
    rows = document.get("rows")
    if not isinstance(rows, list) or len(rows) != 1:
        raise _refuse(f"{MESH_ROW} must hold exactly one row, the bundle's mesh")
    raw = dict(rows[0])
    try:
        row = MeshRow.from_dict(raw)
    except MeshRowRefusal as error:
        raise _refuse(f"{MESH_ROW}: {error}") from None
    files = manifest["files"]
    grid_name = carried_grid_name(manifest)
    if grid_name is None:
        raise _refuse(f"{MESH_ROW} needs the mesh it names carried too ({STATIC}, and {GRID} unless the static is "
                      "the grid)")
    if row.name != manifest["mesh"]["id"]:
        raise _refuse(f"{MESH_ROW} names row {row.name!r}; the manifest's mesh.id is {manifest['mesh']['id']!r}")
    for field, want in (("grid", grid_name), ("static", STATIC)):
        if getattr(row, field) != want:
            raise _refuse(f"{MESH_ROW}: {field} is {getattr(row, field)!r}; in a bundle it is the bundle's own "
                          f"file {want!r}")
    for field, name in (("grid", grid_name), ("static", STATIC)):
        if getattr(row, field + "_sha256") != files[name]["sha256"] or getattr(row, field + "_bytes") != files[name]["bytes"]:
            raise _refuse(f"{MESH_ROW}: the row's {field} digest or size is not {name}'s")
    if row.n_cells != manifest["mesh"]["cells"] or row.n_levels != manifest["mesh"]["levels"]:
        raise _refuse(f"{MESH_ROW}: {row.n_cells} cells and {row.n_levels} levels; the manifest's mesh has "
                      f"{manifest['mesh']['cells']} and {manifest['mesh']['levels']}")
    _no_paths({key: value for key, value in raw.items() if key not in ("grid", "static")}, MESH_ROW)
    return raw


def bind_carried_mesh(bundle: "AnalysisBundle", work: str | Path) -> dict[str, Any]:
    """Register a bundle's carried mesh for this run: ``<work>/mesh-rows.json`` with the row's grid
    and static bound to the bundle's files.  Returns ``{row, grid, static, rows}`` (row None when
    the bundle carries the files and no row)."""

    from .mesh_rows import MeshRow, write_rows

    grid, static = bundle.carried_grid, bundle.carried_static
    if grid is None or static is None:
        raise _refuse("the bundle carries no mesh (grid.nc/static.nc); name it with --mesh, --grid and --static")
    record: dict[str, Any] = {"row": None, "grid": str(grid), "static": str(static), "rows": None}
    if bundle.carried_row is None:
        return record
    raw = read_carried_row(bundle.carried_row, bundle.manifest, bundle.root)
    raw.update(grid=str(grid), static=str(static))
    rows = Path(work) / "mesh-rows.json"
    write_rows(rows, [MeshRow.from_dict(raw)])
    record.update(row=raw["name"], rows=str(rows))
    return record


# ---------------------------------------------------------------------------
# the run's checks
# ---------------------------------------------------------------------------
def check_run(bundle: AnalysisBundle, *, grid: str | Path, static: str | Path,
              levels: int | None = None, vertical_sources: Sequence[str | Path] = ()) -> dict[str, Any]:
    """Refuse, by name, a run whose mesh, static or vertical grid is not the bundle's."""

    mesh = bundle.manifest["mesh"]
    grid_sha = sha256_file(grid)
    if grid_sha != mesh["sha256"]:
        raise _refuse(f"the analysis is on mesh {mesh['id']} (grid SHA-256 {mesh['sha256']}); this run's "
                      f"--grid has {grid_sha}.  Starting a forecast from an analysis on another mesh puts "
                      "every field on the wrong cells; give the bundle's grid, or --remap to move the "
                      "analysis to this mesh")
    static_sha = sha256_file(static)
    if static_sha != bundle.manifest["static"]["sha256"]:
        raise _refuse(f"the analysis was made over static SHA-256 {bundle.manifest['static']['sha256']} "
                      f"(builder {bundle.manifest['static']['builder']}); this run's --static has "
                      f"{static_sha}.  Another static means other terrain under the analysed columns")
    if levels is not None and int(levels) != int(mesh["levels"]):
        raise _refuse(f"the analysis has {mesh['levels']} levels; this run's mesh row declares {levels}")
    measured = vertical_fingerprint(bundle.analysis)
    expected = mesh["vertical_spec"]
    if measured["zgrid_sha256"] != expected["zgrid_sha256"] or measured["levels"] != expected["levels"]:
        raise _refuse(f"analysis.nc's vertical grid ({measured['levels']} levels, zgrid {measured['zgrid_sha256']}) "
                      f"is not the manifest's ({expected['levels']} levels, zgrid {expected['zgrid_sha256']})")
    others = []
    for path in vertical_sources:
        other = vertical_fingerprint(path)
        if other["zgrid_sha256"] != expected["zgrid_sha256"]:
            raise _refuse(f"{Path(path).name}'s vertical grid (zgrid {other['zgrid_sha256']}, top "
                          f"{other['z_top_m']:.1f} m, {other['levels']} levels) is not the analysis's "
                          f"(zgrid {expected['zgrid_sha256']}, top {expected['z_top_m']:.1f} m); the run "
                          "would read the analysed columns on other heights")
        others.append(str(path))
    return {"grid_sha256": grid_sha, "static_sha256": static_sha, "vertical": measured,
            "vertical_sources_checked": others}


# ---------------------------------------------------------------------------
# command line: make / verify / increment
# ---------------------------------------------------------------------------
def _versions(producer: str) -> dict[str, Any]:
    from importlib import metadata

    def version(name: str) -> str | None:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return None

    return {"woof": version("recast-woof"), "gpuwm_hex": version("recast-woof"), "producer": producer}


def carried_row_document(rows_file: str | Path, name: str, *, grid_name: str) -> dict[str, Any]:
    """The one row named ``name`` of a mesh-rows document, rewritten for a bundle: grid and static
    are the bundle's file names, and the receipt fields that are paths become digests."""

    from .mesh_rows import ROWS_SCHEMA, read_rows

    rows = [row for row in read_rows(Path(rows_file)) if row.name == name]
    if len(rows) != 1:
        raise _refuse(f"{rows_file} holds no single row named {name!r}")
    raw = rows[0].as_dict()
    raw.update(grid=grid_name, static=STATIC)
    for key in _ROW_PATH_KEYS:
        value = raw.get(key)
        if isinstance(value, str) and value:
            path = Path(value)
            raw[key] = f"sha256:{sha256_file(path)}" if path.is_file() else "not carried"
    return {"schema": ROWS_SCHEMA, "rows": [raw]}


def make_bundle(out: str | Path, *, analysis: str | Path, grid: str | Path, static: str | Path, mesh_id: str,
                static_builder: str, physics_suite: str, physics_mode: str, kind: str, members: int,
                recipe_id: str, recipe_sha256: str, cycles: int, window: float | None, producer: str,
                background: str | Path | None = None, increment: str | Path | None = None,
                iau_window_s: float | None = None, noda: str | Path | None = None,
                form: str = "init", link: bool = False, carry_mesh: bool = False,
                mesh_rows: str | Path | None = None, notes: Mapping[str, Any] | None = None,
                increment_precision: str = "f8", forecast_args: Sequence[str] | None = None) -> AnalysisBundle:
    """Assemble a bundle directory from files and write its manifest.

    ``background`` (an init-form file at the same time) makes increment.nc as analysis minus
    background (float64, every differing prognostic, so a reader rebuilds the background bit
    for bit); ``increment`` copies a ready increment file.  ``carry_mesh`` puts the grid and
    static in the bundle, and ``mesh_rows`` (a mesh-rows document holding a row named
    ``mesh_id``) the row a forecast binds them with.  ``link`` hard-links instead of copying
    where the filesystem allows it.
    """

    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise _refuse(f"{out} exists and is not empty")
    out.mkdir(parents=True, exist_ok=True)

    def place(source: str | Path, name: str) -> None:
        target = out / name
        if link:
            try:
                os.link(source, target)
                return
            except OSError:
                pass
        shutil.copyfile(source, target)

    place(analysis, ANALYSIS)
    if background is not None and increment is not None:
        raise _refuse("give the background or a ready increment, not both")
    if background is not None:
        write_increment(out / ANALYSIS, background, out / INCREMENT, precision=increment_precision)
    elif increment is not None:
        place(increment, INCREMENT)
    if noda is not None:
        place(noda, NODA)
    if mesh_rows is not None and not carry_mesh:
        raise _refuse("a carried mesh row needs the mesh carried too (carry_mesh)")
    if carry_mesh:
        place(static, STATIC)
        same = sha256_file(grid) == sha256_file(static)
        if not same:
            place(grid, GRID)
        if mesh_rows is not None:
            document = carried_row_document(mesh_rows, mesh_id, grid_name=STATIC if same else GRID)
            (out / MESH_ROW).write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    has_increment = (out / INCREMENT).is_file()
    if has_increment and iau_window_s is None:
        raise _refuse("an increment needs its IAU window (iau_window_s)")
    valid_time = None
    from netCDF4 import Dataset

    with Dataset(out / ANALYSIS) as dataset:
        valid_time = _xtime(dataset)
    write_manifest(
        out, valid_time=valid_time, mesh=describe_mesh(mesh_id, grid, out / ANALYSIS),
        static={"sha256": sha256_file(static), "builder": static_builder},
        physics={"suite": physics_suite, "mode": physics_mode},
        analysis={"kind": kind, "members": int(members), "recipe_id": recipe_id,
                  "recipe_sha256": recipe_sha256, "cycles": int(cycles),
                  "window": None if window is None else float(window), "form": form},
        versions=_versions(producer),
        iau=None if not has_increment else {"window_s": float(iau_window_s)},
        notes=notes, forecast_args=forecast_args)
    return load_bundle(out)


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m woof.hex.analysis_bundle", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("make", help="assemble a bundle directory and write its manifest")
    for flag in ("--out", "--analysis", "--grid", "--static"):
        make.add_argument(flag, type=Path, required=True)
    make.add_argument("--background", type=Path, default=None,
                      help="init-form background at the valid time; increment.nc = analysis - background")
    make.add_argument("--increment", type=Path, default=None, help="a ready MPAS IAU increment file")
    make.add_argument("--noda", type=Path, default=None)
    make.add_argument("--mesh-id", required=True)
    make.add_argument("--static-builder", required=True)
    make.add_argument("--physics-suite", required=True)
    make.add_argument("--physics-mode", choices=PHYSICS_MODES, required=True)
    make.add_argument("--kind", choices=ANALYSIS_KINDS, required=True)
    make.add_argument("--members", type=int, required=True)
    make.add_argument("--recipe-id", required=True)
    make.add_argument("--recipe-sha256", required=True)
    make.add_argument("--cycles", type=int, required=True)
    make.add_argument("--window", type=float, default=None, help="assimilation window, seconds")
    make.add_argument("--iau-window-s", type=float, default=None)
    make.add_argument("--form", choices=ANALYSIS_FORMS, default="init")
    make.add_argument("--producer", required=True)
    make.add_argument("--link", action="store_true", help="hard-link the files instead of copying them")
    make.add_argument("--carry-mesh", action="store_true",
                      help="put the grid and static in the bundle (grid.nc, static.nc), so a forecast needs "
                           "nothing but the bundle")
    make.add_argument("--mesh-rows", type=Path, default=None,
                      help="with --carry-mesh: a mesh-rows document holding the row named --mesh-id; it is "
                           "carried as mesh-row.json, bound to the bundle's own files")
    make.add_argument("--forecast-args", default=None, metavar="'FLAGS'",
                      help="the producer's own forecast configuration as door flags, one quoted string "
                           "(for example '--dt 18.75 --fullstep-local-timestep --fullstep-lts-mode active "
                           "--fullstep-lts-max-rate 8'); a forecast from the bundle applies each one it does not name")
    make.add_argument("--increment-precision", choices=("f8", "f4"), default="f8",
                      help="f8 (default): the background is rebuilt bit for bit wherever float64 resolves it "
                           "(measured per variable); f4: MPAS's own iau stream type")
    verify = commands.add_parser("verify", help="check every file of a bundle against its manifest")
    verify.add_argument("bundle")
    verify.add_argument("--fetch-to", type=Path, default=None)
    increment = commands.add_parser("increment", help="write analysis minus background in MPAS IAU form")
    for flag in ("--analysis", "--background", "--out"):
        increment.add_argument(flag, type=Path, required=True)
    increment.add_argument("--precision", choices=("f8", "f4"), default="f8")
    increment.add_argument("--fields", choices=("differing", "mpas"), default="differing")
    background = commands.add_parser("background", help="write the background an IAU window starts from")
    background.add_argument("bundle")
    background.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "make":
            bundle = make_bundle(
                args.out, analysis=args.analysis, grid=args.grid, static=args.static, mesh_id=args.mesh_id,
                static_builder=args.static_builder, physics_suite=args.physics_suite,
                physics_mode=args.physics_mode, kind=args.kind, members=args.members,
                recipe_id=args.recipe_id, recipe_sha256=args.recipe_sha256, cycles=args.cycles,
                window=args.window, producer=args.producer, background=args.background,
                increment=args.increment, iau_window_s=args.iau_window_s, noda=args.noda,
                form=args.form, link=args.link, carry_mesh=args.carry_mesh, mesh_rows=args.mesh_rows,
                increment_precision=args.increment_precision,
                forecast_args=None if args.forecast_args is None else __import__("shlex").split(args.forecast_args))
            print(json.dumps(bundle.receipt(), indent=2))
        elif args.command == "verify":
            print(json.dumps(load_bundle(args.bundle, fetch_to=args.fetch_to).receipt(), indent=2))
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
