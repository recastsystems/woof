"""The init front door: native-free construction or explicit capsule compatibility.

The Rust ``rw_mpas_init`` engine continues to own meteorological interpolation
and state construction.  This door owns fail-closed argument resolution,
construction of the vertical artifact when requested, compatibility-capsule
preflight, and provenance.

Normal native-free mode is:

``grid + static + WPS met + versioned vertical JSON -> constructed artifact -> rw_mpas_init``

The generated artifact is passed through the engine's existing capsule/reference
ABI as both arguments.  A native init file is opened only when the caller
explicitly selects compatibility mode with ``--capsule`` and ``--reference``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .host_identity import host_record
from .errors import MpasPortError

SURFACE_LEVEL_TAG = 200100.0
_PROFILE_FIELDS = {"TT", "RH", "SPECHUMD", "GHT", "PRES", "PRESSURE", "UU", "VV"}
_SOIL_PREFIXES = ("ST", "SM", "SOILT", "SOILM")
_REQUIRED_MET_FIELDS = (
    ("LANDSEA", "coastal soil and sea-ice interpolation would use the wrong surface type",
     "Use a Vtable that emits LANDSEA"),
    ("SOILHGT", "source terrain would remain zero, changing the terrain correction of skin and soil temperatures",
     "Declare the derived SOILHGT Vtable output and supply source terrain; a cycle-invariant terrain-only donor must share the primary mask's grid"),
)

_STATIC_VARIABLES = (
    "latCell",
    "lonCell",
    "latEdge",
    "lonEdge",
    "angleEdge",
    "cellsOnEdge",
    "edgesOnCell",
    "nEdgesOnCell",
    "landmask",
    "ter",
    "soiltemp",
    "ivgtyp",
    "isltyp",
    "snoalb",
    "isice_lu",
    "greenfrac",
    "albedo12m",
    "dcEdge",
    "dvEdge",
    "areaCell",
    "cellsOnCell",
    "deriv_two",
)
_COMPATIBILITY_VARIABLES = (
    "ter",
    "zgrid",
    "zz",
    "fzm",
    "fzp",
    "dzu",
    "rdzw",
    "zb",
    "zb3",
)


class InitDoorRefusal(MpasPortError):
    """A named refusal: wrong result prevented, then remedy."""


@dataclass(frozen=True)
class _Switch:
    flag: str
    namelist_key: str
    breakage: str


_SWITCHES = (
    _Switch("--nfglevels", "config_nfglevels", "it caps the first-guess level table; a wrong count truncates or over-allocates every profile"),
    _Switch("--nfgsoillevels", "config_nfgsoillevels", "it declares the first-guess soil column; a wrong count drops or zero-fills layers"),
    _Switch("--extrap-airtemp", "config_extrap_airtemp", "it selects temperature extrapolation past the first-guess column"),
    _Switch("--use-spechumd", "config_use_spechumd", "it selects the moisture source and changes qv everywhere"),
    _Switch("--theta-adv-order", "config_theta_adv_order", "it selects the z-edge/zb3 branch and initial vertical mass-flux correction"),
    _Switch("--coef-3rd-order", "config_coef_3rd_order", "it scales the third-order rw correction"),
    _Switch("--virtual-factor", "(reference build property)", "it selects the virtual-temperature arithmetic reproduced by the engine"),
    _Switch("--deep-soil-moisture", "(reference build property)", "it selects the deepest-soil first-guess anchor"),
    _Switch("--landuse-table", "config_landuse_data", "the wrong table silently renumbers vegetation categories"),
    _Switch("--frac-seaice", "config_frac_seaice", "it changes fractional versus binary sea-ice handling"),
    _Switch("--tsk-seaice-threshold", "config_tsk_seaice_threshold", "it controls cold-water conversion to ice-covered land"),
    _Switch("--oned-underflow", "(reference compiler property)", "it selects preserve versus ifx FTZ behavior at interpolation guards"),
    _Switch("--start-time", "config_start_time", "it stamps the init and selects monthly statics interpolation"),
)


def _refuse(message: str) -> None:
    raise InitDoorRefusal(message)


def _sha256(path: Path) -> str:
    # The real SHA-256, computed once per file version (woof.hex.file_digest),
    # so the forecast door that reads this init next does not hash it again.
    from .file_digest import sha256_file as file_version_sha256
    return file_version_sha256(path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _described(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().absolute()
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


# ---------------------------------------------------------------------------
# engine and met resolution
# ---------------------------------------------------------------------------
def resolve_engine(explicit: Path | None) -> Path:
    """The ``rw_mpas_init`` binary, or a refusal naming what supplies it.

    Delegates to :mod:`woof.hex.engines`, which is the ONE ladder every
    door in this distribution resolves through.  This door used to read
    ``$RW_MPAS_INIT`` and nothing else, which meant a user who had run
    ``woof fetch-bridges`` -- the single command that stages this exact
    binary onto the machine -- still met a refusal telling them to build
    it with cargo.  The shared ladder reads that staging directory, and
    the refusal now names the command that fills it.

    ``$RW_MPAS_INIT`` stays in the ladder permanently as a legacy
    spelling: an install line that already works must not be invalidated
    by a rename.
    """

    from .engines import INIT, EngineRefusal, resolve

    try:
        return resolve(INIT, explicit)
    except EngineRefusal as error:
        _refuse(str(error))
        raise  # pragma: no cover - _refuse always raises


def _probe_wps_header(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            head = handle.read(8)
    except OSError:
        return False
    if len(head) < 8:
        return False
    for endian in (">", "<"):
        marker, version = struct.unpack(f"{endian}ii", head)
        if marker == 4 and version in (3, 4, 5):
            return True
    return False


def resolve_met_source(met: Path) -> Path:
    if not met.exists():
        _refuse(
            f"met source {met} does not exist; point --met at ungrib output (one WPS "
            "intermediate file) or a directory containing exactly one valid-time file"
        )
    if met.is_file():
        return met
    candidates = sorted(
        child for child in met.iterdir() if child.is_file() and _probe_wps_header(child)
    )
    if not candidates:
        _refuse(
            f"no WPS intermediate file found in {met}; nothing there has a valid "
            "Fortran-framed WPS v3/v4/v5 header"
        )
    if len(candidates) > 1:
        _refuse(
            f"{met} holds {len(candidates)} candidate WPS files ({', '.join(p.name for p in candidates)}); "
            "the engine requires exactly one valid time, so pass the intended file explicitly"
        )
    return candidates[0]


@dataclass(frozen=True)
class MetInventory:
    fields: tuple[str, ...]
    valid_times: tuple[str, ...]
    upper_air_levels: int
    surface_level_present: bool
    map_sources: tuple[str, ...] = ()


#: The first guess's hydrometeor masses and numbers rw_mpas_init reads under
#: ``--first-guess-hydrometeors yes`` (rw-mpas init ``FIRST_GUESS_SPECIES``).
FIRST_GUESS_SPECIES = ("QC", "QR", "QI", "QS", "QG", "QNC", "QNI", "QNR")
HYDROMETEOR_FLAG = "--first-guess-hydrometeors"


def engine_reads_first_guess_hydrometeors(engine) -> bool:
    """Whether this rw_mpas_init takes ``--first-guess-hydrometeors``: its usage names it.

    An older binary reads unknown flags without complaint and would silently start the run dry, so it is never
    handed the flag; the door records the gap instead."""
    probe = subprocess.run([str(engine), "--help"], capture_output=True, text=True, timeout=60)
    return HYDROMETEOR_FLAG in (probe.stdout or "") + (probe.stderr or "")


def first_guess_hydrometeors(arguments, engine, fields) -> dict[str, Any]:
    """The init's first-guess hydrometeor choice: ``argv`` for the engine plus the receipt record.

    MPAS-A v8.4.2 case 7 reads none of the first guess's QC/QR/QI/QS/QG or QNC/QNI/QNR, so a native init starts
    with no condensate whatever the first guess holds; the RRFS producer (hrrrv5 and the other RRFS suites) reads
    them all.  THE BREAKAGE THIS PREVENTS (hex-meshes-sources review, 2026-10-29): the hex default route (MPAS-A
    physics) started from HRRR 2026-10-28 12Z produced no composite reflectivity at f000 and only weak (<25 dBZ) echo at
    f001 under observed 40-50 dBZ cells in Kansas, while the hrrrv5 route started with HRRR's own.

    Default yes (the improvement reaches both routes); no under strict numerics, the proof opt-in, where the init
    must be native case 7's.  With a first guess that carries none of them (GFS, ERA5, IFS as their rows stand)
    nothing is passed, so those inits keep the argument vector and the bytes they always had.  An engine that
    predates the flag is a loud named gap, never a refusal (woof.hex.correct_mode ruling)."""
    from . import numerics

    requested = getattr(arguments, "first_guess_hydrometeors", None)
    chosen_by = HYDROMETEOR_FLAG
    if requested is None:
        requested, chosen_by = ("no", "strict numerics (native case 7)") if numerics.is_strict() else ("yes", "default")
    carried = sorted(set(fields) & set(FIRST_GUESS_SPECIES))
    record: dict[str, Any] = {"requested": requested, "chosen_by": chosen_by, "first_guess_records": carried,
                              "applied": False, "argv": []}
    if requested != "yes" or not carried:
        return record
    if not engine_reads_first_guess_hydrometeors(engine):
        record["gap"] = (f"the first guess carries {', '.join(carried)} but rw_mpas_init ({engine}) predates "
                         f"{HYDROMETEOR_FLAG} (its usage does not name it); this init starts with no condensate, as "
                         "MPAS-A v8.4.2 case 7.  Install an engine inside this release's range to start from the "
                         "first guess's own clouds and precipitation")
        print(f"woof hex init: named gap: {record['gap']}", file=sys.stderr)
        return record
    record.update(applied=True, argv=[HYDROMETEOR_FLAG, "yes"])
    return record


def scan_met_file(met: Path, *, nfglevels: int, start_time: str) -> MetInventory:
    from .wps_intermediate import WpsIntermediateReader

    names: set[str] = set()
    map_sources: set[str] = set()
    valid_times: set[str] = set()
    profile_levels: set[float] = set()
    try:
        with WpsIntermediateReader(met) as reader:
            for field in reader.iter_fields(load_values=False):
                names.add(field.field)
                map_sources.add(field.map_source)
                valid_times.add(field.valid_time)
                if field.field in _PROFILE_FIELDS:
                    profile_levels.add(field.level)
    except MpasPortError as error:
        _refuse(
            f"{met} is not readable as a WPS intermediate file: {error}. Re-run ungrib "
            "or select its actual output; a truncated source cannot produce a trustworthy init"
        )
    for field, breakage, remedy in _REQUIRED_MET_FIELDS:
        if field not in names:
            _refuse(f"the intermediate file {met} carries no {field} field; {breakage}. {remedy}")
    if SURFACE_LEVEL_TAG not in profile_levels:
        _refuse(
            f"no profile level in {met} is tagged {SURFACE_LEVEL_TAG:.1f}; the case-7 "
            "surface column would be absent and t2m/rh2 would remain structurally wrong"
        )
    if len(profile_levels) > nfglevels:
        _refuse(
            f"the intermediate file has {len(profile_levels)} distinct first-guess levels but "
            f"--nfglevels declares {nfglevels}; declare at least {len(profile_levels)}"
        )
    if not any(name.startswith(_SOIL_PREFIXES) for name in names):
        _refuse(
            f"the intermediate file {met} carries no ST*/SM*/SOILT*/SOILM* soil layers; "
            "tslb/smois cannot be initialized. Use a source/Vtable with the declared soil column"
        )
    stamp = start_time[:13]
    stamps = sorted(value[:13] for value in valid_times)
    if stamps and stamp not in stamps:
        _refuse(
            f"--start-time {start_time} is not among met valid times {', '.join(stamps)}; "
            "using it would mislabel the meteorology"
        )
    return MetInventory(
        fields=tuple(sorted(names)),
        valid_times=tuple(sorted(valid_times)),
        upper_air_levels=len(profile_levels),
        surface_level_present=True,
        map_sources=tuple(sorted(map_sources)),
    )


# ---------------------------------------------------------------------------
# mesh and vertical-source contracts
# ---------------------------------------------------------------------------
def check_static_and_grid(static: Path, grid: Path | None) -> dict[str, Any]:
    try:
        from netCDF4 import Dataset
    except ImportError as error:  # pragma: no cover
        _refuse(
            f"python package netCDF4 is unavailable ({error}); the door will not launch "
            "without preflighting grid/static dimensions and variables"
        )
    if not static.is_file():
        _refuse(
            f"mesh static file {static} does not exist; it supplies the physical mesh and statics"
        )
    with Dataset(static) as dataset:
        missing_dims = [
            name
            for name in ("nCells", "nEdges", "maxEdges", "nMonths")
            if name not in dataset.dimensions
        ]
        missing_vars = [name for name in _STATIC_VARIABLES if name not in dataset.variables]
        if missing_dims or missing_vars:
            _refuse(
                f"static file {static} is missing {', '.join(missing_dims + missing_vars)}; "
                "vertical construction/interpolation would have no complete topology or statics. "
                "Use a complete generated static file"
            )
        summary = {
            "nCells": int(len(dataset.dimensions["nCells"])),
            "nEdges": int(len(dataset.dimensions["nEdges"])),
            "static_dcEdge_shape": list(dataset.variables["dcEdge"].shape),
        }
    if grid is not None:
        if not grid.is_file():
            _refuse(
                f"mesh grid file {grid} does not exist; native-free vertical construction requires "
                "the grid/static pair minted together"
            )
        with Dataset(grid) as dataset:
            for name in ("nCells", "nEdges"):
                if name not in dataset.dimensions:
                    _refuse(f"--grid {grid} has no {name} dimension; it is not an MPAS mesh")
            grid_cells = int(len(dataset.dimensions["nCells"]))
            grid_edges = int(len(dataset.dimensions["nEdges"]))
        if grid_cells != summary["nCells"] or grid_edges != summary["nEdges"]:
            _refuse(
                f"grid/static pairing mismatch: grid has nCells={grid_cells}, nEdges={grid_edges}; "
                f"static has nCells={summary['nCells']}, nEdges={summary['nEdges']}. "
                "Pass the pair generated together"
            )
        summary["grid_nCells"] = grid_cells
        summary["grid_nEdges"] = grid_edges
    return summary


def check_mesh_and_capsule(
    static: Path,
    capsule: Path,
    reference: Path,
    grid: Path | None,
) -> dict[str, Any]:
    """Compatibility-mode preflight; this function retains its old public name."""

    try:
        from netCDF4 import Dataset
    except ImportError as error:  # pragma: no cover
        _refuse(f"python package netCDF4 is unavailable ({error}); install it")
    summary = check_static_and_grid(static, grid)
    for label, path in (("capsule", capsule), ("reference", reference)):
        if not path.is_file():
            _refuse(
                f"compatibility {label} {path} does not exist; provide the native file or use "
                "--vertical-spec for the native-free path"
            )
    with Dataset(capsule) as dataset:
        missing = [
            name for name in ("nCells", "nVertLevels") if name not in dataset.dimensions
        ] + [name for name in _COMPATIBILITY_VARIABLES if name not in dataset.variables]
        if missing:
            _refuse(
                f"compatibility capsule {capsule} is missing {', '.join(missing)}; its vertical contract is incomplete"
            )
        capsule_cells = int(len(dataset.dimensions["nCells"]))
        if capsule_cells != summary["nCells"]:
            _refuse(
                f"capsule nCells={capsule_cells} differs from static nCells={summary['nCells']}; "
                "vertical arrays would be attached to the wrong columns"
            )
        capsule_shape = tuple(int(value) for value in dataset.variables["zgrid"].shape)
        summary["nVertLevels"] = int(len(dataset.dimensions["nVertLevels"]))
    with Dataset(reference) as dataset:
        if "zgrid" not in dataset.variables:
            _refuse(
                f"compatibility reference {reference} carries no zgrid; the capsule identity oracle cannot run"
            )
        reference_shape = tuple(int(value) for value in dataset.variables["zgrid"].shape)
    if capsule_shape != reference_shape:
        _refuse(
            f"compatibility capsule zgrid{capsule_shape} differs from reference zgrid{reference_shape}; "
            "pass the native reference that minted the capsule"
        )
    summary["zgrid_shape"] = list(capsule_shape)
    return summary


@dataclass(frozen=True)
class VerticalSource:
    mode: str
    capsule: Path
    reference: Path
    summary: dict[str, Any]
    vertical_spec: Path | None = None
    vertical_artifact_receipt: Path | None = None
    #: Held while a shared cached artifact is stamped with this run's clock
    #: (exclusive) and read by the engine (shared); released by run_init_command.
    lock: Any = None


#: Bumped whenever the cached artifact's contents would change for the same
#: grid, static, spec and declarations (key input, not a version claim).
VERTICAL_CACHE_SCHEMA = "gpuwm-hex.vertical-cache/v2"

#: Run declarations the cached artifact carries only as fixed-width global
#: attributes, so they are stamped per run instead of entering the key.
#:
#: THE BREAKAGE THIS PREVENTS, MEASURED (two rented hosts, 2026-10-12, 30 km
#: x1.655362): config_nfglevels and config_nfgsoillevels are facts of the
#: SOURCE's first-guess ladder (GFS 34, IFS 15, GEFS 32, ERA5 38, ICON 19 with
#: 8 soil levels), and with them in the key every source rebuilt the same
#: vertical arrays into its own 5.9 GB file -- five artifacts and 35 GB for
#: one mesh, which put a box under the GPU mutex's 80 GB free-disk guard.
#: The arrays depend on the grid, static and vertical spec alone; the
#: producer writes these values with setncattr and reads them nowhere.  Only
#: declarations whose encoded width cannot change (a text clock of fixed
#: length, int32) are stamped: rewriting a classic-format header in place is
#: safe only at constant size, so text and YES/NO declarations stay in the key.
STAMPED_DECLARATIONS = ("config_start_time", "config_nfglevels", "config_nfgsoillevels")


def vertical_cache_dir(arguments: argparse.Namespace) -> Path | None:
    """Where constructed vertical artifacts are kept between inits, or None.

    THE BREAKAGE THIS PREVENTS, MEASURED (8-card RTX PRO 6000 host, 2026-10-10, 15 km x1.2621442):
    every init rebuilt the same vertical artifact -- 231 s of the 448 s init,
    128 s of it hashing arrays for the artifact receipt -- and wrote it again
    as a 24.6 GB file beside the init, although it depends only on the grid,
    static, vertical spec and run declarations, none of which change between
    cycles on a mesh.  The one per-cycle value it carries, config_start_time,
    is stamped into the cached file in place under a lock before the engine
    reads it, so the init the engine writes carries the same attributes and
    the same variable bytes as a freshly constructed one.

    ``--vertical-cache DIR`` or ``WOOF_HEX_VERTICAL_CACHE``; default
    ``$WOOF_HEX_CACHE_ROOT/vertical`` or ``~/.cache/gpuwm-hex/vertical``;
    ``off`` builds a fresh artifact beside the init as before.  An explicit
    ``--vertical-artifact`` is always built fresh.
    """

    if getattr(arguments, "vertical_artifact", None) is not None:
        return None
    text = getattr(arguments, "vertical_cache", None) or os.environ.get("WOOF_HEX_VERTICAL_CACHE", "")
    text = str(text)
    if text.lower() in ("off", "0", "none"):
        return None
    if text:
        return Path(text).expanduser().absolute()
    root = os.environ.get("WOOF_HEX_CACHE_ROOT")
    return (Path(root) if root else Path.home() / ".cache" / "woof hex") / "vertical"


def _vertical_cache_key(arguments: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    from .vertical_spec import VerticalSpec
    from .rust_geometry import geometry_engine

    here = Path(__file__).parent
    declarations = dict(run_declarations(arguments))
    for stamped in STAMPED_DECLARATIONS:  # stamped per run, see STAMPED_DECLARATIONS
        declarations.pop(stamped)
    _, geometry_binary_sha = geometry_engine()
    material = {
        "schema": VERTICAL_CACHE_SCHEMA,
        "grid_sha256": _sha256(Path(arguments.grid).expanduser().absolute()),
        "static_sha256": _sha256(Path(arguments.static).expanduser().absolute()),
        "vertical_spec_canonical_sha256": VerticalSpec.from_file(
            Path(arguments.vertical_spec).expanduser().resolve(strict=True)).sha256(),
        "declarations": {key: declarations[key] for key in sorted(declarations)},
        "producer_sources": {name: _sha256(here / name)
                             for name in ("vertical_spec.py", "vector.py", "rust_geometry.py", "mesh.py")},
        "geometry_binary_sha256": geometry_binary_sha,
        "port_version": _port_version(),
    }
    key = hashlib.sha256(json.dumps(material, sort_keys=True).encode("utf-8")).hexdigest()
    return key, material


def _stamped_already(artifact: Path, declarations: dict[str, Any]) -> dict[str, Any] | None:
    """The stamp record when the cached artifact already carries this run's declarations, else None (read-only)."""
    from netCDF4 import Dataset

    stamp: dict[str, Any] = {}
    with Dataset(artifact, mode="r") as dataset:
        held = set(dataset.ncattrs())
        for key in STAMPED_DECLARATIONS:
            before = dataset.getncattr(key) if key in held else None
            before = before.item() if hasattr(before, "item") else before
            if before != declarations[key]:
                return None
            stamp[f"{key}_before"] = before
            stamp[key] = declarations[key]
    return stamp


def _stamp_run_declarations(artifact: Path, declarations: dict[str, Any]) -> dict[str, Any]:
    """Set the cached artifact's per-run declarations to this run's, in place."""
    from netCDF4 import Dataset
    from .vertical_spec import encode_config_attribute

    stamp: dict[str, Any] = {}
    with Dataset(artifact, mode="r+") as dataset:
        held = set(dataset.ncattrs())
        for key in STAMPED_DECLARATIONS:
            value = declarations[key]
            before = dataset.getncattr(key) if key in held else None
            before = before.item() if hasattr(before, "item") else before
            if before != value:
                dataset.setncattr(key, encode_config_attribute(value))
            stamp[f"{key}_before"] = before
            stamp[key] = value
    return stamp


def _cached_vertical_source(arguments: argparse.Namespace, cache: Path, summary: dict[str, Any]) -> VerticalSource:
    from .file_lock import lock as _flock, unlock as _funlock
    from .vertical_spec import materialize_vertical_artifact

    started = time.time()
    key, material = _vertical_cache_key(arguments)
    cache.mkdir(parents=True, exist_ok=True)
    artifact = cache / f"{key[:24]}.vertical.nc"
    receipt = artifact.with_name(artifact.name + ".receipt.json")
    complete = artifact.with_name(artifact.name + ".complete")
    lock = (cache / f"{key[:24]}.lock").open("a+")
    # Readers share the artifact; only a build or a re-stamp is exclusive.
    # THE BREAKAGE THIS PREVENTS, MEASURED (home RTX 5090 host, unified image
    # 2026-10-16, 6 GEFS members at 120 km): every member init held this lock
    # exclusively for its whole rw_mpas_init run, although all six stamp the
    # same cycle and first-guess ladder, so --init-jobs ran the inits one at a
    # time.  A run whose declarations the artifact already carries now holds
    # it shared while the engine reads; a run that must stamp takes it
    # exclusively (after readers of other declarations finish), stamps, then
    # re-checks under the shared lock (flock conversion is not atomic).
    declarations = run_declarations(arguments)
    hit: bool | None = None
    stamp: dict[str, Any] | None = None
    try:
        while True:
            _flock(lock, shared=True)
            ready = complete.is_file() and artifact.is_file() and receipt.is_file()
            hit = ready if hit is None else hit
            held = _stamped_already(artifact, declarations) if ready else None
            if held is not None:
                stamp = stamp or held
                break
            _funlock(lock)
            _flock(lock)
            if not (complete.is_file() and artifact.is_file() and receipt.is_file()):
                for stale in (artifact, receipt, complete):
                    stale.unlink(missing_ok=True)
                materialize_vertical_artifact(
                    grid=arguments.grid, static=arguments.static, spec_path=arguments.vertical_spec,
                    output=artifact, receipt_path=receipt, run_config=declarations)
                complete.write_text(json.dumps(material, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            stamp = _stamp_run_declarations(artifact, declarations)
            _funlock(lock)
        payload = json.loads(receipt.read_text(encoding="utf-8"))
    except BaseException:
        lock.close()
        raise
    summary = {
        **summary,
        "vertical_artifact": payload["output"],
        "vertical_spec_canonical_sha256": payload["inputs"]["vertical_spec"]["canonical_sha256"],
        "vertical_invariants": payload["invariants"],
        "vertical_cache": {"dir": str(cache), "key": key, "hit": hit, "seconds": round(time.time() - started, 3),
                           **stamp},
    }
    return VerticalSource(mode="constructed", capsule=artifact, reference=artifact, summary=summary,
                          vertical_spec=arguments.vertical_spec, vertical_artifact_receipt=receipt, lock=lock)


def select_vertical_mode(arguments: argparse.Namespace) -> str:
    constructed_requested = (
        getattr(arguments, "vertical_spec", None) is not None
        or getattr(arguments, "vertical_artifact", None) is not None
    )
    compatibility_requested = (
        getattr(arguments, "capsule", None) is not None
        or getattr(arguments, "reference", None) is not None
    )
    if constructed_requested and compatibility_requested:
        _refuse(
            "constructed and compatibility vertical sources were both declared. Mixing them can "
            "hide which bytes produced zgrid/zb. Use --vertical-spec (optionally --vertical-artifact) "
            "or use --capsule plus --reference, never both"
        )
    if constructed_requested:
        if getattr(arguments, "vertical_spec", None) is None:
            _refuse(
                "--vertical-artifact was given without --vertical-spec; there is no declaration to construct from"
            )
        if getattr(arguments, "grid", None) is None:
            _refuse(
                "native-free mode requires --grid as well as --static; topology and deriv_two must come "
                "from the grid/static pair generated together"
            )
        return "constructed"
    if compatibility_requested:
        if getattr(arguments, "capsule", None) is None or getattr(arguments, "reference", None) is None:
            _refuse(
                "compatibility mode requires both --capsule and --reference; a capsule asserted only "
                "against itself or without its oracle is not admitted"
            )
        return "compatibility-capsule"
    _refuse(
        "no vertical source was declared. Normal native-free use requires --vertical-spec and --grid; "
        "legacy compatibility requires both --capsule and --reference"
    )
    raise AssertionError("unreachable")


def _port_version() -> str:
    """This port's own version, for the init's lineage block."""

    from . import __version__

    return str(__version__)


def run_declarations(arguments: argparse.Namespace) -> dict[str, Any]:
    """The ``config_*`` global attributes this init declares about itself.

    They are what the forecast driver parses off the init (it asserts
    ``--start-time`` against ``config_start_time`` and refuses an init that
    carries none), and both routes through this door put them there: the
    constructed route writes them into the vertical artifact, whose global
    attributes the engine copies verbatim; the compatibility route stamps
    whichever of them the capsule did not carry (:func:`stamp_run_declarations`).
    """

    return {
        "config_start_time": arguments.start_time,
        "config_nfglevels": int(arguments.nfglevels),
        "config_nfgsoillevels": int(arguments.nfgsoillevels),
        "config_nsoillevels": 9 if getattr(arguments, "land_surface", "noah") == "ruc" else 4,
        "config_use_spechumd": arguments.use_spechumd == "yes",
        "config_frac_seaice": arguments.frac_seaice == "yes",
        "config_extrap_airtemp": arguments.extrap_airtemp,
        "config_tsk_seaice_threshold": float(arguments.tsk_seaice_threshold),
        "config_landuse_data": arguments.landuse_table,
    }


def stamp_run_declarations(init: Path, declarations: dict[str, Any]) -> dict[str, Any]:
    """Put the run declarations the engine's capsule copy left absent onto the init.

    ``rw_mpas_init`` writes the init's global attributes by copying the
    capsule's, and it does not stamp ``config_start_time`` itself.  A native
    capsule carries the attribute from its own namelist; a vertical artifact
    the constructed route minted carries it because that route wrote it; a
    capsule minted at mesh time -- ``mesh-plan --point`` culls the parent's
    vertical before any start time exists -- carries neither, and the init
    reached the forecast door without a clock.  MEASURED (2026-09-13, the
    first 937.5 m cull at a point, 43,884 cells): the forecast driver stopped
    with "init carries no config_start_time" after the contract deck had
    already spent seven minutes on the card.  The ``--start-time`` switch is
    documented as "it stamps the init"; this is where that becomes true on
    every route.

    Only ABSENT attributes are written.  An attribute the capsule carried is
    left in the file byte for byte (the native x4 anchor route carries all of
    them and is not touched) and recorded here with its value, so a receipt
    reader sees what the file says beside what the switch said.
    """

    try:
        from netCDF4 import Dataset
    except ImportError as error:  # pragma: no cover
        _refuse(f"python package netCDF4 is unavailable ({error}); install it")
    from .vertical_spec import encode_config_attribute

    stamped: dict[str, Any] = {}
    present: dict[str, Any] = {}
    with Dataset(init, mode="r+") as dataset:
        held = set(dataset.ncattrs())
        for key, value in declarations.items():
            if key in held:
                found = dataset.getncattr(key)
                present[key] = found.item() if hasattr(found, "item") else found
                continue
            dataset.setncattr(key, encode_config_attribute(value))
            stamped[key] = value
    return {"stamped": stamped, "already_present": present}


def prepare_vertical_source(arguments: argparse.Namespace) -> VerticalSource:
    mode = select_vertical_mode(arguments)
    if mode == "compatibility-capsule":
        summary = check_mesh_and_capsule(
            arguments.static, arguments.capsule, arguments.reference, arguments.grid
        )
        return VerticalSource(
            mode=mode,
            capsule=arguments.capsule,
            reference=arguments.reference,
            summary=summary,
        )

    summary = check_static_and_grid(arguments.static, arguments.grid)
    cache = vertical_cache_dir(arguments)
    if cache is not None:
        return _cached_vertical_source(arguments, cache, summary)
    from .vertical_spec import materialize_vertical_artifact

    out: Path = arguments.out
    artifact = (
        arguments.vertical_artifact
        if arguments.vertical_artifact is not None
        else out.with_name(out.name + ".vertical.nc")
    )
    artifact = artifact.expanduser().absolute()
    receipt = artifact.with_name(artifact.name + ".receipt.json")
    payload = materialize_vertical_artifact(
        grid=arguments.grid,
        static=arguments.static,
        spec_path=arguments.vertical_spec,
        output=artifact,
        receipt_path=receipt,
        # The emitter copies capsule global attributes verbatim and the
        # forecast driver parses them from the init, so the artifact must
        # carry THIS run's declarations, not the static template's
        # placeholders.
        run_config=run_declarations(arguments),
    )
    summary = {
        **summary,
        "vertical_artifact": payload["output"],
        "vertical_spec_canonical_sha256": payload["inputs"]["vertical_spec"]["canonical_sha256"],
        "vertical_invariants": payload["invariants"],
    }
    # The current Rust ABI performs a capsule/reference zgrid identity check.
    # Passing the one constructed artifact as both inputs keeps that check while
    # removing any native-file dependency.  The receipt names this explicitly.
    return VerticalSource(
        mode=mode,
        capsule=artifact,
        reference=artifact,
        summary=summary,
        vertical_spec=arguments.vertical_spec,
        vertical_artifact_receipt=receipt,
    )


# ---------------------------------------------------------------------------
# CLI and execution
# ---------------------------------------------------------------------------
def add_init_parser(commands: Any) -> None:
    parser = commands.add_parser(
        "init",
        help="build MPAS initial conditions with native-free vertical construction or explicit capsule compatibility",
        description=(
            "Drive rw_mpas_init. Normal mode constructs the vertical contract from --grid, "
            "--static, and --vertical-spec; compatibility mode is selected only by passing "
            "both --capsule and --reference. Every numerical switch remains explicit."
        ),
    )
    parser.add_argument("--met", type=Path, help="WPS intermediate file, or a directory holding exactly one")
    parser.add_argument("--sfc-met", type=Path, help="optional surface WPS input; enables config_input_sst")
    parser.add_argument("--seaice-met", type=Path, help="optional fractional sea-ice WPS input")
    parser.add_argument("--static", type=Path, help="generated/published static file")
    parser.add_argument("--grid", type=Path, default=None, help="grid file; required in native-free mode")
    parser.add_argument("--vertical-spec", type=Path, default=None, help="versioned JSON declaration for native-free construction")
    parser.add_argument("--vertical-artifact", type=Path, default=None, help="durable constructed artifact path, built fresh (default: the vertical cache)")
    parser.add_argument("--vertical-cache", default=None, metavar="DIR",
                        help="keep constructed vertical artifacts here between inits (default "
                             "$WOOF_HEX_VERTICAL_CACHE, else $WOOF_HEX_CACHE_ROOT/vertical or "
                             "~/.cache/gpuwm-hex/vertical; 'off' builds one beside the init)")
    parser.add_argument("--capsule", type=Path, default=None, help="explicit compatibility-only native init-class capsule")
    parser.add_argument("--reference", type=Path, default=None, help="explicit compatibility-only native reference")
    parser.add_argument("--out", type=Path, help="init NetCDF to write")
    parser.add_argument("--start-time", help="YYYY-MM-DD_HH:MM:SS (config_start_time)")
    parser.add_argument("--nfglevels", type=int, help="config_nfglevels")
    parser.add_argument("--nfgsoillevels", type=int, help="config_nfgsoillevels")
    parser.add_argument("--extrap-airtemp", choices=("constant", "linear", "lapse-rate"), help="config_extrap_airtemp")
    parser.add_argument("--use-spechumd", choices=("yes", "no"), help="config_use_spechumd")
    parser.add_argument("--theta-adv-order", type=int, help="config_theta_adv_order; must agree with vertical spec")
    parser.add_argument("--coef-3rd-order", type=float, help="config_coef_3rd_order")
    parser.add_argument("--virtual-factor", choices=("reproduce-fortran", "consistent"))
    parser.add_argument("--deep-soil-moisture", choices=("reproduce-fortran", "corrected"))
    parser.add_argument("--landuse-table", help="config_landuse_data")
    parser.add_argument("--frac-seaice", choices=("yes", "no"), help="config_frac_seaice")
    parser.add_argument("--tsk-seaice-threshold", type=float, help="config_tsk_seaice_threshold, K")
    parser.add_argument("--oned-underflow", choices=("preserve", "reproduce-ifx-ftz"))
    parser.add_argument("--engine", type=Path, default=None, help="rw_mpas_init executable (default $RW_MPAS_INIT)")
    parser.add_argument(HYDROMETEOR_FLAG, choices=("yes", "no"), default=None,
                        help="start from the first guess's own hydrometeors (QC QR QI QS QG) and numbers (QNC QNI "
                             "QNR) when it carries them (default yes; no under strict numerics, where the init is "
                             "MPAS-A v8.4.2 case 7's, which reads none)")
    parser.add_argument("--receipt", type=Path, default=None, help="provenance receipt path")
    parser.add_argument("--physics-profile", help="profile initialization row, including verified monthly aerosols")
    parser.add_argument("--land-surface", choices=("noah", "noahmp", "ruc"), default=None,
                        help="target soil geometry; defaults to the physics suite, or Noah")
    parser.add_argument("--physics-suite", default=None, help="initialize this suite's soil and complete species row")
    parser.add_argument("--wif-climatology", type=Path, default=None, help="explicit WIF dataset; missing or invalid named input refuses")
    rrfs = parser.add_argument_group(
        "suites with their own producer (RRFS-MPAS: hrrrv5, rrfs_*)",
        "the producer row in data/profiles/init-producers.v1.json carries the recipe; see docs/rrfs-init.md")
    rrfs.add_argument("--geog", type=Path, default=None,
                      help="WPS geography root: builds the suite's own static when --static lacks its fields, "
                           "and the GSL drag statistics when it holds topo_ugwp_30s and topo_ugwp_2.5m")
    rrfs.add_argument("--rrfs-worker", type=Path, default=None,
                      help="rw_mpas_init_v831 (default $WOOF_HEX_RW_MPAS_INIT_V831, then the engine ladder)")
    rrfs.add_argument("--physics-pin-set", default=None,
                      help="source pin set of the suite route whose producer initializes this case")
    rrfs.add_argument("--lbc-met", action="append", default=None, metavar="TIME=MET",
                      help="a boundary time and its WPS intermediate (repeat per hour); writes lbc.*.nc")
    rrfs.add_argument("--lbc-dir", type=Path, default=None, help="where lbc.*.nc go (default: beside --out)")
    rrfs.add_argument("--ugwp-oro", choices=("gsl-oro", "hex-ugwp-oro"), default=None,
                      help="who makes ugwp_oro_data.nc (default: the producer row's, the GSL generator)")
    rrfs.add_argument("--deep-soil-temperature", choices=("era5", "islscp-i"), default=None,
                      help="deep-soil temperature of a fork static built here (default era5: soiltemp_era5_1deg, "
                           "permissive; islscp-i is NOAA's soiltemp_1deg, licence unclear, selected by name only)")
    parser.add_argument("--soil-remap", choices=("auto", "on", "off"), default="auto",
                        help="table-selected cold soil hydraulic conversion: auto (default) applies it where its "
                             "contract holds and otherwise keeps the cold-start soil, naming why; on refuses instead; "
                             "off never applies it")
    parser.add_argument("--soil-donor", type=Path, default=None,
                        help="co-located native soil analysis required by the hydraulic contract")
    parser.add_argument("--soil-contract", default=None, help="explicit packaged soil hydraulic table id")
    parser.set_defaults(handler=run_init_command)


def _require_switches(arguments: argparse.Namespace) -> None:
    for switch in _SWITCHES:
        attribute = switch.flag.lstrip("-").replace("-", "_")
        if getattr(arguments, attribute, None) is None:
            _refuse(
                f"{switch.flag} was not given and has no default: {switch.breakage}. "
                f"State it explicitly; native key is {switch.namelist_key}"
            )
    for flag, value in (
        ("--met", arguments.met),
        ("--static", arguments.static),
        ("--out", arguments.out),
    ):
        if value is None:
            _refuse(f"{flag} was not given; it cannot be inferred")
    select_vertical_mode(arguments)


def _validate_spec_switch_agreement(arguments: argparse.Namespace) -> None:
    if arguments.vertical_spec is None:
        return
    from .vertical_spec import VerticalSpec

    spec = VerticalSpec.from_file(arguments.vertical_spec)
    if int(arguments.theta_adv_order) != spec.theta_adv_order:
        _refuse(
            f"--theta-adv-order={arguments.theta_adv_order} disagrees with vertical spec "
            f"theta_adv_order={spec.theta_adv_order}; zb/zb3 and the engine would use different branches. "
            "Make the two declarations identical"
        )
    if not abs(float(arguments.coef_3rd_order) - spec.coef_3rd_order) <= 8.0 * max(
        1.0, abs(spec.coef_3rd_order)
    ) * sys.float_info.epsilon:
        _refuse(
            f"--coef-3rd-order={arguments.coef_3rd_order} disagrees with vertical spec "
            f"coef_3rd_order={spec.coef_3rd_order}; declare one value in both contracts"
        )


def _soil_remap_off(arguments, reason):
    """The cold soil remap does not apply to this init: refuse under --soil-remap on, else keep the cold soil.

    THE BREAKAGE THE AUTO BRANCH PREVENTS (WOOF woof hex 0.4.10 bake and switch candidate, 2026-10-30/31): each case
    the remap cannot serve (a land model other than Noah, a forecast-hour start, an expired table date) refused the
    whole init, three times in a day. The init without the remap is the cold start native MPAS-A's init_atmosphere
    writes, and every hex init wrote before the remap, so a case the contract does not cover keeps it and the log
    and receipt say why. --soil-remap on is the opt-in that refuses instead.
    """
    if getattr(arguments, "soil_remap", "auto") == "on":
        _refuse(f"--soil-remap on: the cold soil remap does not apply: {reason}")
    arguments.soil_remap_reason = reason
    print(f"init: cold soil remap off: {reason}; the init keeps its cold-start soil", file=sys.stderr)


def prepare_soil_remap(arguments, map_sources):
    from .cycle.soil_wetness import source_contract
    if getattr(arguments, "soil_remap", "auto") == "off":
        return None
    try:
        contract = source_contract(map_sources, getattr(arguments, "soil_contract", None))
    except ValueError as error:
        _refuse(str(error))
    if contract is None:
        if getattr(arguments, "soil_remap", "auto") == "on":
            _refuse("--soil-remap on: no cold soil contract in the table covers this init's source "
                    f"({', '.join(map_sources) or 'none named'})")
        return None
    if arguments.land_surface != "noah":
        # The table's contract is Noah's hydraulics. Another land model (the ArWen route's Noah-MP, RUC) starts from
        # the interpolated soil, as every init did before the remap. init-source names the table's contract and
        # donor on every GFS init, so a named --soil-contract is no request to refuse (it stopped every ArWen-route
        # zoom globe, VR globe and region from GFS at its init, WOOF woof hex 0.4.10 bake, 2026-10-30).
        _soil_remap_off(arguments, f"the table's contract is Noah's, this init is {arguments.land_surface}")
        return None
    donor = getattr(arguments, "soil_donor", None)
    if donor is None or not Path(donor).is_file():
        _soil_remap_off(arguments, "no co-located native soil analysis (--soil-donor) was given; interpolated "
                                   "categorical soil classes cannot supply porosity (init-source acquires the donor)")
        return None
    return contract


def run_init_command(arguments: argparse.Namespace) -> int:
    started = time.time()
    from .init_physics import init_selection
    from .mpasphys.rrfs.cold_start_rrfs import RrfsInitError
    try:
        selection = init_selection(getattr(arguments, "physics_suite", None),
                                   getattr(arguments, "land_surface", None),
                                   profile=getattr(arguments, "physics_profile", None),
                                   pin_set=getattr(arguments, "physics_pin_set", None))
    except RrfsInitError as error:
        _refuse(str(error))
    if selection.get("producer") is not None:
        # A suite whose model initializes with its own producer (RRFS-MPAS: the fork's
        # init_atmosphere, ported to rw_mpas_init_v831). Its recipe is the table row, so
        # the v8.4.2 numerical switches below do not apply to it.
        from .mpasphys.rrfs.cold_start_rrfs import RrfsInitError, run_rrfs_init
        try:
            return run_rrfs_init(arguments, selection["producer"])
        except RrfsInitError as error:
            _refuse(str(error))
    _require_switches(arguments)
    arguments.land_surface = selection["land_surface"]
    _validate_spec_switch_agreement(arguments)

    out: Path = arguments.out.expanduser().absolute()
    if not out.parent.is_dir():
        _refuse(
            f"output directory {out.parent} does not exist; create it before an expensive init run"
        )
    receipt_path = (
        arguments.receipt.expanduser().absolute()
        if arguments.receipt is not None
        else out.with_name(out.name + ".provenance.json")
    )
    if not receipt_path.parent.is_dir():
        _refuse(f"receipt directory {receipt_path.parent} does not exist; create it")

    met = resolve_met_source(arguments.met)
    from .hrrr_intermediate import uncovered_cells_refusal
    uncovered = uncovered_cells_refusal(met, Path(arguments.static))
    if uncovered:
        _refuse(uncovered)
    inventory = scan_met_file(
        met, nfglevels=arguments.nfglevels, start_time=arguments.start_time
    )
    soil_contract = prepare_soil_remap(arguments, inventory.map_sources)
    engine = resolve_engine(arguments.engine)
    hydrometeors = first_guess_hydrometeors(arguments, engine, inventory.fields)
    vertical = prepare_vertical_source(arguments)

    argv = [
        str(engine),
        "--met", str(met),
        "--static", str(arguments.static),
        "--capsule", str(vertical.capsule),
        "--reference", str(vertical.reference),
        "--out", str(out),
        "--start-time", arguments.start_time,
        "--nfglevels", str(arguments.nfglevels),
        "--nfgsoillevels", str(arguments.nfgsoillevels),
        "--extrap-airtemp", arguments.extrap_airtemp,
        "--use-spechumd", arguments.use_spechumd,
        "--theta-adv-order", str(arguments.theta_adv_order),
        "--coef-3rd-order", repr(arguments.coef_3rd_order),
        "--virtual-factor", arguments.virtual_factor,
        "--deep-soil-moisture", arguments.deep_soil_moisture,
        "--landuse-table", arguments.landuse_table,
        "--frac-seaice", arguments.frac_seaice,
        "--tsk-seaice-threshold", repr(arguments.tsk_seaice_threshold),
        "--oned-underflow", arguments.oned_underflow,
        # WHO THIS FILE SAYS IT IS.  rw_mpas_lbc reads model_name, core_name,
        # version and git_version off its --grid and refuses rather than
        # invent one, so an init that carries none of them cannot drive a
        # boundary stream -- which, until 2026-08-26, was every init this
        # program minted, global or regional.  The name is THIS program's,
        # never "mpas": these files carry ArWen's physics and this port's
        # numerics, and a stream stamped mpas would claim a provenance no
        # file in the chain has.
        "--model-name", "woof hex",
        "--core-name", "atmosphere",
        "--model-version", str(_port_version()),
        "--git-version", f"gpuwm-hex-{_port_version()}",
    ]
    if arguments.land_surface != "noah":
        argv.extend(("--land-surface", arguments.land_surface))
    argv.extend(hydrometeors["argv"])
    # The inputs do not change while the engine reads them, so their digests
    # are taken on other cores while it runs instead of after it (the
    # constructed artifact alone is 24.6 GB at 15 km).
    from concurrent.futures import ThreadPoolExecutor
    input_paths = {"met": met, "static": arguments.static}
    for role, switch in (("sfc_met", "sfc-met"), ("seaice_met", "seaice-met")):
        source = getattr(arguments, role, None)
        if source is not None:
            source = resolve_met_source(Path(source).expanduser()).resolve(strict=True)
            argv.extend(("--" + switch, str(source)))
            input_paths[role] = source
    if arguments.grid:
        input_paths["grid"] = arguments.grid
    if vertical.mode == "constructed":
        input_paths.update(vertical_spec=vertical.vertical_spec, vertical_artifact=vertical.capsule,
                           vertical_artifact_receipt=vertical.vertical_artifact_receipt)
    else:
        input_paths.update(capsule=vertical.capsule, reference=vertical.reference)
    try:
        with ThreadPoolExecutor(max_workers=len(input_paths), thread_name_prefix="init-digest") as pool:
            described_futures = {role: pool.submit(_described, Path(path)) for role, path in input_paths.items()}
            completed = subprocess.run(argv, capture_output=True, text=True)
            described = {role: future.result() for role, future in described_futures.items()}
    finally:
        if vertical.lock is not None:
            # Released only after the engine read the stamped artifact and
            # its digest was taken; another run may stamp it now.
            vertical.lock.close()
    if completed.returncode != 0:
        sys.stderr.write(completed.stderr)
        sys.stderr.write(
            "woof hex init: rw_mpas_init refused or failed "
            f"(rc {completed.returncode}). No init/provenance receipt was written. "
            + (
                f"The constructed vertical artifact remains at {vertical.capsule} with its own receipt for diagnosis.\n"
                if vertical.mode == "constructed"
                else "\n"
            )
        )
        return int(completed.returncode)
    try:
        engine_receipt = json.loads(completed.stdout)
    except json.JSONDecodeError:
        engine_receipt = {"unparsed_stdout": completed.stdout}
    if not out.is_file():
        _refuse(
            f"the engine reported success but {out} does not exist; rebuild the current rw_mpas_init "
            "or inspect an ABI mismatch between door and engine"
        )
    expected_soil = 9 if arguments.land_surface == "ruc" else 4
    from netCDF4 import Dataset as _InitDataset
    with _InitDataset(out) as _written:
        written_soil = len(_written.dimensions["nSoilLevels"]) if "nSoilLevels" in _written.dimensions else None
    if written_soil != expected_soil:
        # An rw_mpas_init that predates --land-surface parses and ignores it,
        # writing Noah's four layers under a RUC declaration; RUC then steps
        # on a soil column it does not have.
        _refuse(
            f"rw_mpas_init wrote {written_soil} soil levels where --land-surface "
            f"{arguments.land_surface} needs {expected_soil}; the engine binary predates suite soil "
            "geometry. Rebuild or re-stage rw_mpas_init from woof 2.9.0 or later"
        )
    soil_receipt = {"mode": getattr(arguments, "soil_remap", "auto"), "applied": False}
    if getattr(arguments, "soil_remap_reason", None):
        soil_receipt["reason"] = arguments.soil_remap_reason
    if soil_contract is not None:
        from .cycle.soil_wetness import SoilRemapNotApplied, remap_cold_init
        try:
            result = remap_cold_init(out, arguments.soil_donor, soil_contract, arguments.start_time)
        except SoilRemapNotApplied as error:
            # Nothing was written: the init is the complete cold start. Under on it is removed with the refusal.
            if getattr(arguments, "soil_remap", "auto") == "on":
                out.unlink(missing_ok=True)
            _soil_remap_off(arguments, str(error))
            soil_receipt = {**soil_receipt, "contract": soil_contract["id"], "reason": str(error)}
        except Exception as error:
            # The remap failed while writing the init: this owned output is incomplete. It must not be forecast as an
            # admitted cold init when its soil completion has failed.
            out.unlink(missing_ok=True)
            _refuse(f"cold soil remap failed while writing and the incomplete init was removed: {error}")
        else:
            if not result.get("applied", True):
                print(f"init: cold soil remap had nothing to change: {result.get('reason')}", file=sys.stderr)
            soil_receipt = {**soil_receipt, "applied": True, **result}
    aerosol_receipt = None
    if selection["scalar_names"]:
        from .init_physics import initialize_species
        try:
            aerosol_receipt = initialize_species(engine, out, selection, arguments)
        except BaseException as error:
            # The engine's file is complete only once the suite's species are
            # in it. Left behind, it opens cleanly and starts the suite
            # without its aerosols, so a failed completion removes it.
            out.unlink(missing_ok=True)
            if isinstance(error, ValueError):
                _refuse(f"physics initialization did not complete and {out} was removed: {error}")
            raise
    declarations = stamp_run_declarations(out, run_declarations(arguments))
    carried_clock = declarations["already_present"].get("config_start_time")
    if carried_clock is not None and str(carried_clock) != arguments.start_time:
        sys.stderr.write(
            f"woof hex init: the capsule carried config_start_time {carried_clock!r} and "
            f"--start-time is {arguments.start_time!r}; the file keeps the capsule's, and the "
            "forecast door will refuse it by name unless the clock is moved (the cascade's "
            "delayed start does that) or the init is remade from a capsule of this run\n"
        )

    from . import __version__

    inputs: dict[str, Any] = {
        "met": described["met"],
        "sfc_met": described.get("sfc_met"),
        "seaice_met": described.get("seaice_met"),
        "static": described["static"],
        "grid": described.get("grid"),
    }
    if vertical.mode == "constructed":
        inputs["vertical_spec"] = described["vertical_spec"]
        inputs["vertical_artifact"] = described["vertical_artifact"]
        inputs["vertical_artifact_receipt"] = described["vertical_artifact_receipt"]
        inputs["capsule"] = None
        inputs["reference"] = None
    else:
        inputs["vertical_spec"] = None
        inputs["vertical_artifact"] = None
        inputs["vertical_artifact_receipt"] = None
        inputs["capsule"] = described["capsule"]
        inputs["reference"] = described["reference"]

    receipt = {
        "schema": "gpuwm-hex.init-provenance/v2",
        "tool": "woof hex init",
        "door_version": __version__,
        "created_utc": _utc_now(),
        "host": host_record(),
        "engine": _described(engine),
        "vertical_source": {
            "mode": vertical.mode,
            "native_runtime_dependency": vertical.mode != "constructed",
            "engine_abi_mapping": {
                "capsule_argument": str(vertical.capsule),
                "reference_argument": str(vertical.reference),
                "constructed_artifact_used_for_both": vertical.mode == "constructed",
            },
            "summary": vertical.summary,
        },
        "inputs": inputs,
        "met_inventory": {
            "fields": list(inventory.fields),
            "valid_times": list(inventory.valid_times),
            "distinct_upper_air_levels": inventory.upper_air_levels,
        },
        "engine_argv": argv,
        "engine_receipt": engine_receipt,
        "cold_soil_remap": soil_receipt,
        "first_guess_hydrometeors": {key: value for key, value in hydrometeors.items() if key != "argv"},
        "physics_initialization": (aerosol_receipt if aerosol_receipt is not None
                                   else selection.get("physics_initialization")),
        "run_declarations": declarations,
        "output": _described(out),
        "door_seconds": round(time.time() - started, 3),
    }
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "init": str(out),
                "provenance": str(receipt_path),
                "vertical_mode": vertical.mode,
                "vertical_artifact": (
                    str(vertical.capsule) if vertical.mode == "constructed" else None
                ),
                "n_cells": engine_receipt.get("n_cells"),
                "n_vert_levels": engine_receipt.get("n_vert_levels"),
                "met_records_used": engine_receipt.get("met_records_used"),
                "engine_seconds": engine_receipt.get("seconds"),
            },
            indent=2,
        )
    )
    return 0


# MPAS-A v8.4.2 mpas_init_atm_surface.F:29-394; data processing is Rust.
_SURFACE_ABI = 'rw_mpas_init --mode surface-update --sfc-prefix PREFIX --static STATIC.nc --out SURFACE.nc --start-time YYYY-MM-DD_HH:MM:SS --stop-time YYYY-MM-DD_HH:MM:SS --fg-interval SECONDS --surface-interval SECONDS [--receipt JSON]'


def add_surface_update_parser(commands: argparse._SubParsersAction) -> None:
    parser = commands.add_parser(
        "surface-update", help="build MPAS SST and sea-ice update records"
    )
    parser.add_argument("--engine", type=Path)
    parser.add_argument("--sfc-prefix", type=Path, required=True)
    parser.add_argument("--static", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start-time", required=True)
    parser.add_argument("--stop-time", required=True)
    parser.add_argument("--fg-interval", type=int, required=True)
    parser.add_argument("--surface-interval", type=int, required=True)
    parser.add_argument("--receipt", type=Path)
    parser.set_defaults(handler=run_surface_update_command)


def run_surface_update_command(arguments: argparse.Namespace) -> int:
    engine = resolve_engine(arguments.engine)
    try:
        probe = subprocess.run(
            [str(engine), "--mode", "surface-update", "--abi"],
            capture_output=True, text=True, check=False,
        )
        if probe.returncode or probe.stdout.strip() != _SURFACE_ABI:
            _refuse(
                "the init engine lacks the case-8 surface ABI; rebuild rw_mpas_init "
                "with the i5-init-sfc-update delivery before supplying inputs"
            )
        command = [str(engine), "--mode", "surface-update"]
        for flag in (
            "sfc-prefix", "static", "out", "start-time", "stop-time",
            "fg-interval", "surface-interval", "receipt",
        ):
            value = getattr(arguments, flag.replace("-", "_"))
            if value is not None:
                if isinstance(value, Path):
                    value = value.expanduser()
                command.extend(["--" + flag, str(value)])
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as error:
        _refuse(f"cannot run surface-update engine {engine}: {error}")
    if result.returncode:
        _refuse(result.stderr.strip() or "surface-update engine failed without a diagnostic")
    print(result.stdout.strip())
    return 0
