"""Direct woof prepared-cache -> WRF ``wrfinput``/``wrfbdy`` export.

This module is intentionally independent of WPS and WRF executables.  It
consumes the launch-ready native woof preparation cache, the matching native
static cache, and a small geometry receipt.  A frozen WRF-v4.6.1 declaration
contract supplies only NetCDF metadata; all gridded state is derived from the
native inputs.

Who writes the bytes: WOOF's Rust.  The DEFAULT engine is the classic
writer at ``tools/rustwx/crates/netcdf-writer``, driven through
:mod:`woof.io.nc_writer_bridge` and the shared
:class:`woof.io.classic_tape.ClassicTape` facade -- the same seam the
wrfout history tape writes through.  The export's closing verification
runs there too: the ``rw_netcdf`` reader for the header (attributes,
inventory, dimensions) and the writer crate's own read-back sweep for
the finiteness check over every float variable.  ``netCDF4`` survives in
this module for exactly one thing, the ``WOOF_WRFINPUT_WRITER=python``
workaround engine, which is announced as a workaround and never selected
silently.

The supported slice is a single specified projected domain
(lambert/mercator/polar; MAP_PROJ and MAP_PROJ_CHAR derive from the
geometry receipt) with an explicit validated hybrid eta coordinate and a
registry-resolved fixed physics profile.  Stock-WRF acceptance is proven for
Lambert; mercator/polar exports are oracle-gated but not yet stock-WRF-gated.
Unsupported cache/configuration combinations fail closed.
"""

from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from typing import Mapping, Sequence
import uuid
from weakref import WeakValueDictionary

import numpy as np

from woof.io.netcdf_serialization import netCDF4  # first use only (D-10)

from woof import netcdf_bridge
from woof.core import constants as _model_constants
from woof.io.classic_tape import ClassicTape
from woof.ingest.prepared_cache import (
    compare_prepared_domain_config,
    prepared_domain_config_identity,
    prepared_identity_refusal,
    undelayed_identity_defaults,
)
from woof.static.lambert import LambertGrid, grids_from_projection_config
from woof.static.projection import (MAP_PROJ_CHARS, WRF_MAP_PROJ_CODES,
                                     projection_class)
from woof.vertical_contract import (
    validate_coordinate_shapes,
    validate_explicit_eta_grid,
)
from woof.wrf_physics_inventory import (
    EXPORT_USE_THETA_M, WRFINPUT_2D_DIMS, stock_wrf_physics_inventory)


_CONTRACT_PATH = Path(__file__).with_name("wrf_direct_v461_contract.json")

#: Where the hierarchy exporter has d01's own exporter write, inside its
#: staging, before it moves the two root files up beside the children.
ROOT_EXPORT_DIRNAME = ".root-export"


def export_staging_path(output: Path, *, pid: int | None = None) -> Path:
    """The sibling an export is written in before one rename publishes it.

    One spelling for both exporters, so the depth a door measures before
    it prepares (:func:`woof.native_domain_artifacts.published_path_refusal`)
    is the depth they write at.  ``pid`` defaults to this process.
    """

    output = Path(output)
    return output.with_name(
        f"{output.name}.tmp-{os.getpid() if pid is None else int(pid)}")


def domain_artifacts_manifest_temporary(
        manifest_path: Path, *, pid: int | None = None,
        token: str | None = None) -> Path:
    """The name :func:`write_domain_artifacts_manifest` writes before its rename."""

    manifest_path = Path(manifest_path)
    pid = os.getpid() if pid is None else int(pid)
    token = uuid.uuid4().hex[:12] if token is None else token
    return manifest_path.with_name(f".{manifest_path.name}.tmp-{pid}-{token}")


# WRF's rvovrd (share/module_model_constants.F:41 = 1.6083624), not the 1.608
# this used to carry.  module_initialize_real.F writes T as
# theta*(1.+rvovrd*qv) - 300 on every column it produces, so an exporter that
# truncates rvovrd hands WRF an initial condition WRF would not have written:
# 2.3e-04 relative, ~1.1e-03 K of theta_m at qv = 0.01, and the same offset
# again through alpha.  No WRF code path in the stack woof mirrors uses 1.608
# (the only 1.608 in the tree is CAM-ZM's own virtual temperature, which is not
# in any woof template).
_RVOVRD = _model_constants.RVOVRD

_RD = 287.0
_RCP = 2.0 / 7.0
_P0 = 100000.0
_THETA_OFFSET_K = 300.0
_HRRR_SOIL_NODE_DEPTHS_M = np.array(
    [0.0, 0.01, 0.04, 0.10, 0.30, 0.60, 1.0, 1.6, 3.0],
    dtype=np.float64,
)
_NOAH_LAYER_MIDPOINTS_M = np.array(
    [0.05, 0.25, 0.70, 1.50], dtype=np.float64)
_NOAH_LAYER_THICKNESS_M = np.array(
    [0.10, 0.30, 0.60, 1.00], dtype=np.float64)
_CANONICAL_SURFACE_FIELDS = frozenset({
    "TSK", "TSLB", "SMOIS", "SH2O", "TMN", "SEAICE", "XLAND",
    "LANDMASK", "SNOW", "SNOWH",
})

#: Which library writes the ``wrfinput``/``wrfbdy`` pair.  ``rust`` (the
#: DEFAULT) is the dependency-free classic writer at
#: ``tools/rustwx/crates/netcdf-writer`` behind the
#: :mod:`woof.io.nc_writer_bridge` ctypes seam, emitting the CDF-2
#: (``NETCDF3_64BIT_OFFSET``) container -- which is what stock WRF's own
#: ``real.exe`` writes at ``io_form_input = 2``, so the pair stays a file
#: set unchanged WRF reads.  ``python`` is the netCDF4/HDF5 writer this
#: module used through 2.4, emitting the frozen contract's ``NETCDF4``
#: container -- kept reachable as an EXPLICIT WORKAROUND only, never
#: selected silently.  Their equivalence on the real contract is
#: ``tests/test_wrf_direct_writer.py::test_the_two_engines_write_the_same_file``.
#:
#: What the classic container cannot carry is the frozen contract's
#: per-variable HDF5 storage tuning -- ``zlib``/``shuffle``/``complevel``/
#: ``fletcher32``/``chunksizes`` and the stored ``endian``.  None of them
#: describes a value: names, types, dimensions, definition order,
#: attributes and every bit of every field survive the flip, and the
#: contract declares no fill value for any variable.  An uncompressed
#: pair is larger on disk and is exactly what ``real.exe`` produces.
WRFINPUT_WRITER_ENV = "WOOF_WRFINPUT_WRITER"

_WRFINPUT_ENGINES = ("rust", "python")

#: What a caller should do when the classic header is already on disk.
#: The whole schema is declared in ``_create_dataset`` before any value
#: is written, so this can only be reached by a change to this module.
_FROZEN_REMEDY = (
    "Every dimension, attribute and variable of this pair is declared in "
    "woof.wrf_direct._create_dataset before the first value is written; "
    "declare it there, or use the netCDF4 workaround "
    f"{WRFINPUT_WRITER_ENV}=python.")


def resolve_wrfinput_engine(engine: str | None = None) -> str:
    """The writer engine to use: explicit argument, else env, else rust."""

    value = engine if engine is not None else os.environ.get(
        WRFINPUT_WRITER_ENV, "rust")
    value = str(value).strip().lower()
    if value not in _WRFINPUT_ENGINES:
        raise ValueError(
            f"unknown wrfinput writer engine {value!r} "
            f"(from {WRFINPUT_WRITER_ENV!r} or the engine argument); "
            f"expected one of {_WRFINPUT_ENGINES}.  'rust' is the default "
            "classic writer; 'python' is the netCDF4 workaround.")
    return value


def _require_rust_writer() -> None:
    """Refuse, rather than downgrade, when the Rust writer is not here."""

    from woof.io import nc_writer_bridge

    reason = nc_writer_bridge.unavailable_reason()
    if reason is None:
        return
    raise RuntimeError(
        "the default wrfinput/wrfbdy engine is the Rust NetCDF writer "
        "(the netcdf-writer cdylib behind woof.io.nc_writer_bridge) and "
        "it is not loadable here, so the WRF handoff pair cannot be "
        f"written:\n  {reason}\n"
        "A silent fallback would change the pair's container with build "
        "state and quietly un-test the default, so it is refused instead. "
        "Remedies: build the library from a checkout (cd tools/rustwx; "
        "cargo build --release --locked --offline -p netcdf-writer; "
        "cd ../..), or run `woof fetch-bridges` on an installed wheel; "
        f"or set {WRFINPUT_WRITER_ENV}=python to select the netCDF4 "
        "writer as an explicit workaround.")


class StockWrfExportUnsupported(ValueError):
    """The prepared state is outside what the stock-WRF export represents.

    Every refusal in this module that means "woof can run this, but the
    unchanged-WRF file set cannot represent it" raises this rather than a
    bare ``ValueError``: mixed-domain microphysics, a microphysics
    selector with no WRF Registry inventory, and the physics slice the
    profile-free compatibility branch requires.

    It exists so a caller can tell export-representability apart from
    everything else that can go wrong here.  A caller whose product IS
    the export re-raises it and is unchanged.  A caller that is merely
    PREPARING a forecast catches it, records the refusal, and proceeds --
    which physics a domain tree may run is a decision the registry and
    the acknowledgement gate own, and answering it a second time from a
    downstream file-format contract is how a registry-reachable,
    explicitly acknowledged, profile-bound MYNN tree came to be
    unrunnable from any shipped GFS front door.

    ``unsupported`` carries the named per-selector deltas
    (``{selector: (observed, required)}``) when there are any, so the
    refusal a user reads names what to change.
    """

    def __init__(self, message: str, *,
                 unsupported: Mapping[str, object] | None = None) -> None:
        super().__init__(message)
        self.unsupported = dict(unsupported or {})


HIERARCHY_EXPORT_SCHEMA = "gpuwm-native-direct-wrf-hierarchy-export-v1"

#: The three states a hierarchy export can be in.  ``READY`` is a real
#: manifest with a ``files`` inventory; the other two are documents that
#: occupy the same slot and answer the same question -- did this
#: preparation publish unchanged-WRF inputs, and if not, why -- so no
#: consumer has to distinguish "absent" from "refused" by guessing.
STOCK_WRF_EXPORT_STATUSES = ("READY", "NOT_REQUESTED", "REFUSED")


def stock_wrf_export_not_requested(
        *, schema: str = HIERARCHY_EXPORT_SCHEMA) -> dict[str, object]:
    """The export slot for a preparation that never asked for an export."""

    return {
        "schema": schema,
        "status": "NOT_REQUESTED",
        "reason": "the caller did not request a stock-WRF export",
    }


def stock_wrf_export_refused(
        error: StockWrfExportUnsupported, *,
        schema: str = HIERARCHY_EXPORT_SCHEMA) -> dict[str, object]:
    """The export slot for an export refused on representability.

    The message is the export gate's own, unchanged -- it already names
    the selector deltas -- and ``unsupported`` repeats them as data so a
    receipt reader does not have to parse prose.  Tuples are flattened to
    lists so this document equals its own JSON round trip.
    """

    return {
        "schema": schema,
        "status": "REFUSED",
        "reason": str(error),
        "unsupported": {
            name: (list(value) if isinstance(value, tuple) else value)
            for name, value in error.unsupported.items()
        },
    }


def _direct_export_soil_geometry(
        sf_surface_physics: int, num_soil_layers: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the WRF ZS/DZS geometry selected by the land-surface scheme."""

    scheme = int(sf_surface_physics)
    count = int(num_soil_layers)
    if scheme in {2, 4}:
        if count != 4:
            raise ValueError(
                f"Noah/Noah-MP direct export requires four soil layers, "
                f"got sf_surface_physics={scheme}, num_soil_layers={count}")
        return _NOAH_LAYER_MIDPOINTS_M, _NOAH_LAYER_THICKNESS_M
    if scheme == 3:
        from woof.ingest.ruc_soil import ruc_soil_depths

        depths, thicknesses = ruc_soil_depths(count)
        return (
            np.asarray(depths, dtype=np.float64),
            np.asarray(thicknesses, dtype=np.float64),
        )
    raise ValueError(
        f"sf_surface_physics={scheme} has no source-driven direct-export "
        "soil geometry; woof/ingest/ruc_soil.py:"
        "preprocess_land_surface_soil implements selectors 2, 3, and 4")


@dataclass(frozen=True)
class PreparedDomainArtifacts:
    """Native preparation artifacts for one WRF domain.

    A hierarchy export accepts one of these records for every domain in the
    namelist-derived :class:`~woof.experiment.ExperimentConfig`.  Paths are
    explicit and digest-verified by the normal prepared/static/geometry
    contracts before any final output directory is published.
    """

    grid_id: int
    prepared_cache: Path
    static_cache: Path
    geometry_receipt: Path

    def __post_init__(self):
        if (isinstance(self.grid_id, bool)
                or not isinstance(self.grid_id, int)
                or self.grid_id < 1
                or self.grid_id > 99):
            raise ValueError(
                f"grid_id must be an integer in [1, 99], got {self.grid_id!r}")
        for name in ("prepared_cache", "static_cache", "geometry_receipt"):
            object.__setattr__(self, name, Path(getattr(self, name)))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def _array_sha256(array: np.ndarray) -> str:
    value = np.asarray(array)
    if not value.flags.c_contiguous:
        value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(b";")
    digest.update(_canonical(list(value.shape)).encode("ascii"))
    digest.update(b";")
    raw = memoryview(value).cast("B")
    block_bytes = 8 * 1024 * 1024
    for start in range(0, len(raw), block_bytes):
        digest.update(raw[start:start + block_bytes])
    return digest.hexdigest()


def _load_contract() -> Mapping[str, object]:
    payload = json.loads(_CONTRACT_PATH.read_text(encoding="utf-8"))
    if payload.get("schema") != "gpuwm-wrf-direct-contract-bundle-v1":
        raise RuntimeError("unsupported bundled WRF direct-export contract")
    return payload


#: The moist members the frozen WSM6 direct contract carries.  Named as data
#: so `_physics_contract_bundle` can prune what a package does not declare
#: without hard-coding species names inside the loop.
_FROZEN_CONTRACT_MOIST_MEMBERS = frozenset({
    "QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP",
})

_PACKAGE_FIELD_METADATA = {
    "CWM": ("Total condensate mixing ratio", "kg kg-1"),
    "QHAIL": ("Hail mixing ratio", "kg kg-1"),
    "QNDROP": ("Droplet number mixing ratio", "# kg-1"),
    "QNICE": ("Ice Number concentration", "# kg-1"),
    "QNRAIN": ("Rain Number concentration", "# kg(-1)"),
    "QNSNOW": ("Snow Number concentration", "# kg(-1)"),
    "QNGRAUPEL": ("Graupel Number concentration", "# kg(-1)"),
    "QNHAIL": ("Hail Number concentration", "# kg(-1)"),
    "QNCCN": ("CCN Number concentration", "# kg(-1)"),
    "QVGRAUPEL": ("Graupel Particle Volume", "m(3) kg(-1)"),
    "QVHAIL": ("Hail Particle Volume", "m(3) kg(-1)"),
    # P3 (mp_physics=50, Registry.EM_COMMON:555-558).  These two are why
    # the mp=50 row could not simply be declared in
    # woof/wrf_physics_inventory.py and left there: _physics_contract_bundle
    # below SKIPS any inventoried field this dict does not name, and
    # _write_wrfinput iterates the contract -- so without these entries a P3
    # export would have written QNICE and QNRAIN and silently dropped the
    # rime mass and rime volume, which are two of P3's four scalars and the
    # two that carry its whole ice-density prediction.  That is the exact
    # failure mode the mp=28 row's docstring names as a known downstream gap.
    "QIR": ("Rime ice mass-1 mixing ratio", "kg kg(-1)"),
    "QIB": ("Rime ice volume-1 mixing ratio", "m(3) kg(-1)"),
    # Thompson aerosol-aware (mp_physics=28, Registry.EM_COMMON:3036 for
    # the scalars and :492-493 for the two 2-D emission members).  These
    # six rows are audit R-054: without them an mp=28 stock export wrote 8
    # of its 14 inventoried members and dropped the six that carry the
    # scheme's entire aerosol state, silently -- the same failure the P3
    # comment above describes, met a second time by the package whose own
    # docstring had named it as a known gap.
    "QNCLOUD": ("cloud water Number concentration", "# kg(-1)"),
    "QNWFA": ("water-friendly aerosol number concentration", "# kg(-1)"),
    "QNIFA": ("ice-friendly aerosol number concentration", "# kg(-1)"),
    "QNBCA": ("black carbon aerosol number concentration", "# kg(-1)"),
    "QNWFA2D": ("water-friendly aerosol number tendency", "# kg(-1) s(-1)"),
    "QNIFA2D": ("ice-friendly aerosol number tendency", "# kg(-1) s(-1)"),
}

# Pure WRF FAST_SBM spectral fields, Registry/registry.sbm:318.
from woof.core.fast_sbm_contract import BIN_SPECIES as _FSBM_BIN_SPECIES
_PACKAGE_FIELD_METADATA.update({"EFF_RADIUS":("Effective Radius","Microns"),"ICE_EFF_RADIUS":("Ice Effective Radius","Microns"),"TOT_EFF_RADIUS":("Tot Effective Radius","Microns")})
_PACKAGE_FIELD_METADATA.update({
    name:("Spectral bin " + name, "  kg-1" if name.startswith("ff8") else "  kg kg-1")
    for name in _FSBM_BIN_SPECIES
})

#: Inventoried wrfinput members ``_PACKAGE_FIELD_METADATA`` has NO row for,
#: each with the defect that leaves it out.  ``_physics_contract_bundle``
#: skips an inventoried member this dict does not name WITHOUT A WORD, so a
#: row here is a silent truncation of somebody's stock export.
#:
#: EMPTY (audit R-054).  Its six rows were mp=28's, and they are now real
#: metadata rows above with a 2-D prototype branch beneath them.  The
#: agreement check below holds this set equal to the difference between the
#: registry's inventory and the metadata, so the truncation cannot come
#: back unnamed: a new package member with no row fails at import.
_PACKAGE_FIELDS_NOT_YET_EXPORTED: dict[str, str] = {}


def _require_agreement_with_the_registry() -> None:
    """Every inventoried wrfinput member is either exportable or cited."""

    import os

    from woof.physics_registry import (
        REGISTRY_REBUILD_ENV, consumer_rows_by_selector)

    if os.environ.get(REGISTRY_REBUILD_ENV) == "1":
        return
    inventoried: set[str] = set()
    for row in consumer_rows_by_selector("microphysics", "stock_wrf_export").values():
        if row.get("inventoried") is True:
            inventoried.update(
                field["netcdf_name"] for field in row["wrfinput_fields"])
    extensions = inventoried - _FROZEN_CONTRACT_MOIST_MEMBERS
    exportable = set(_PACKAGE_FIELD_METADATA)
    problems = []
    for name in sorted(extensions - exportable - set(_PACKAGE_FIELDS_NOT_YET_EXPORTED)):
        problems.append(
            f"{name} is inventoried and has neither a _PACKAGE_FIELD_METADATA "
            "row nor a cited reason; it would be dropped from a stock export "
            "silently")
    for name in sorted(set(_PACKAGE_FIELDS_NOT_YET_EXPORTED) & exportable):
        problems.append(
            f"{name} is cited as not yet exported but has a metadata row; "
            "retire the citation")
    for name in sorted(set(_PACKAGE_FIELDS_NOT_YET_EXPORTED) - extensions):
        problems.append(
            f"{name} is cited as not yet exported but no registry inventory "
            "names it; retire the citation")
    if problems:
        raise RuntimeError(
            "woof.wrf_direct._PACKAGE_FIELD_METADATA disagrees with the "
            "registry's stock_wrf_export inventory: " + "; ".join(problems))


_require_agreement_with_the_registry()


def _physics_contract_bundle(
        contract_bundle: Mapping[str, object], mp_physics: int,
        run_config: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Extend the frozen WSM6 NetCDF declarations for one WRF package.

    WRF v4.6.1's Registry marks scheme-specific mass and scalar members with
    input and boundary interpolation flags.  The frozen direct contract came
    from a WSM6 file and therefore cannot contain Thompson/Morrison number
    moments or NSSL-2 hail, number, CCN, and volume moments.  Clone the
    corresponding mass-scalar declaration mechanics while replacing the exact
    Registry name, description, and units.  Existing contract objects remain
    immutable and each returned bundle is independently hashable.
    """

    inventory = stock_wrf_physics_inventory(mp_physics)
    result = copy.deepcopy(dict(contract_bundle))
    input_contract = result["wrfinput"]
    bdy_contract = result["wrfbdy"]
    # THE FROZEN CONTRACT IS A WSM6 FILE, so it carries WSM6's six moist
    # members; every package inventoried before P3 declared those six plus
    # additions, and this loop only ever ADDED.  mp_physics=50 is the first
    # package that declares FEWER: Registry.EM_COMMON:3038 gives
    # p3_1category ``moist:qv,qc,qr,qi`` with no qs and no qg, because P3's
    # single ice category spans the snow/graupel continuum through rime mass
    # and rime volume instead of splitting it into species.  Without this
    # prune the exported wrfinput would carry a QSNOW and a QGRAUP for a
    # domain whose scheme has no snow and no graupel species at all -- two
    # zero-valued variables asserting hydrometeors that do not exist in the
    # run that wrote them.
    #
    # For mp_physics 6, 8, 10, 18 and 28 the declared moist list is exactly
    # (or a superset of) the frozen six, so `undeclared` is EMPTY and this
    # block is a provable no-op for every package that shipped before P3;
    # tests/test_wrf_physics_inventory.py measures that rather than asserting
    # it.
    declared_moist = {
        field.netcdf_name for field in inventory.wrfinput_fields
        if field.collection == "moist"
    }
    frozen_moist = {
        item["name"] for item in input_contract["variables"]
        if item["name"] in _FROZEN_CONTRACT_MOIST_MEMBERS
    }
    undeclared = frozen_moist - declared_moist
    if undeclared:
        input_contract["variables"] = [
            item for item in input_contract["variables"]
            if item["name"] not in undeclared
        ]
        bdy_contract["variables"] = [
            item for item in bdy_contract["variables"]
            if item["name"].split("_B")[0] not in undeclared
        ]
    extensions = [
        field
        for field in inventory.wrfinput_fields
        if field.netcdf_name in _PACKAGE_FIELD_METADATA
    ]
    extension_names = [field.netcdf_name for field in extensions]
    input_names = {item["name"] for item in input_contract["variables"]}
    input_prototype = next(
        item for item in input_contract["variables"]
        if item["name"] == "QCLOUD")
    # A package member is not necessarily 3-D.  mp=28's two surface
    # emission members are XY (Registry.EM_COMMON:492-493), and cloning
    # QCLOUD for them would declare QNWFA2D with a bottom_top dimension --
    # a variable stock WRF's own reader rejects, which is why adding the
    # metadata rows alone was never the whole fix (audit R-054).  The
    # frozen contract already carries 2-D prototypes; HGT is one, and it is
    # READ from the contract rather than constructed, so the rank, the
    # MemoryOrder and the stagger come from the same file every other
    # variable's do.
    surface_prototype = next(
        item for item in input_contract["variables"]
        if item["name"] == "HGT")
    two_dimensional = {
        field.netcdf_name for field in extensions
        if len(field.dimensions) == len(WRFINPUT_2D_DIMS)
    }
    for field in extensions:
        name = field.netcdf_name
        if name in input_names:
            continue
        description, units = _PACKAGE_FIELD_METADATA[name]
        spec = copy.deepcopy(
            surface_prototype if name in two_dimensional else input_prototype)
        spec["name"] = name
        spec["attributes"]["description"] = description
        spec["attributes"]["units"] = units
        input_contract["variables"].append(spec)

    # WRF Registry.EM_COMMON:91,1181-1210: the selected drag scheme
    # reads these real-valued mass-grid statistics from wrfinput.
    from woof.static.orographic import required_static_fields
    cfg = run_config or {}
    for name in required_static_fields(cfg.get("topo_wind", 0), cfg.get("gwd_opt", 0)):
        if name not in input_names:
            spec = copy.deepcopy(surface_prototype)
            spec["name"] = name
            spec["attributes"]["description"] = f"Subgrid orographic statistic {name}"
            spec["attributes"]["units"] = (
                "m2" if name == "VAR_SSO" else "m" if name.startswith("VAR") else "")
            input_contract["variables"].append(spec)

    bdy_names = {item["name"] for item in bdy_contract["variables"]}
    bdy_prototypes = [
        item for item in bdy_contract["variables"]
        if item["name"].startswith("QCLOUD_B")
    ]
    for name in extension_names:
        if name in two_dimensional:
            # A 2-D surface emission member has no lateral boundary arrays
            # in WRF's Registry either: there is nothing to force on the
            # face of a field with no vertical extent.
            continue
        description, units = _PACKAGE_FIELD_METADATA[name]
        for prototype in bdy_prototypes:
            suffix = prototype["name"].removeprefix("QCLOUD")
            variable_name = name + suffix
            if variable_name in bdy_names:
                continue
            spec = copy.deepcopy(prototype)
            spec["name"] = variable_name
            spec["attributes"]["description"] = description
            spec["attributes"]["units"] = units
            bdy_contract["variables"].append(spec)
    return result


def _contract_payload_sha256(contract: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical(contract).encode("utf-8")).hexdigest()


def resolved_physics_contract_sha256(mp_physics: int) -> str:
    """The ``resolved_physics_contract_sha256`` an export manifest records.

    The digest of the frozen contract resolved for one microphysics
    package, and nothing else: the exporters write it and the forecast's
    preflight (:func:`woof.prepared_single_domain_forecast.
    _resolved_wrf_direct_contract_sha256`) recomputes it from mp_physics
    alone, so both call this one function.  What the written files add to
    that contract from the run configuration -- the terrain-drag
    statistics ``topo_wind`` / ``gwd_opt`` read (``run_config`` of
    :func:`_physics_contract_bundle`) and the land categories the static
    has (:func:`_land_category_contract`) -- stays out of it; the export
    binds those through ``prepared_header_sha256``, whose cache identity
    records the run configuration.  Breakage this prevents: a digest taken
    over the drag-extended contract made every forecast whose run has
    ``gwd_opt`` 1 or 3 (the HRRR recipes' gwd_opt = 3) finish and then
    exit rc=1 at the seal's preflight with "proof export source hashes
    differ from preparation" (2.9.10, hrrr_v4_gsd41 on hrrr-native).
    """

    return _contract_payload_sha256(
        _physics_contract_bundle(_load_contract(), int(mp_physics)))


def _load_static_geometry_receipt(
        receipt_path: Path, static_path: Path, *,
        expected_geometry: Mapping[str, object] | None = None,
        expected_grid=None,
) -> tuple[dict[str, object], str]:
    """Verify the static receipt/cache binding and optional target geometry."""

    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (payload.get("schema") != "gpuwm-native-static-direct-v1"
            or payload.get("status") != "PASS"):
        raise ValueError("unrecognized or non-PASS native static receipt")
    static_sha256 = _sha256(static_path)
    expected_cache = {
        "path": static_path.name,
        "bytes": static_path.stat().st_size,
        "sha256": static_sha256,
    }
    if payload.get("cache") != expected_cache:
        raise ValueError("native static receipt does not bind the supplied cache")
    geometry = payload.get("geometry")
    if not isinstance(geometry, dict):
        raise ValueError("native static receipt lacks a geometry object")
    if expected_geometry is not None:
        from woof.native_wrf_contract import native_geometry_drift

        drift = {
            name: {"receipt": entry["recorded"],
                   "namelist": entry["expected"],
                   **{key: value for key, value in entry.items()
                      if key not in ("recorded", "expected")}}
            for name, entry in native_geometry_drift(
                geometry, expected_geometry, expected_grid).items()
        }
        if drift:
            raise ValueError(
                "native static receipt geometry differs from namelist: "
                f"{drift}")
    return geometry, static_sha256


def _acquire_output_lock(output: Path) -> tuple[int, Path]:
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.with_name(f".{output.name}.native-export.lock")
    try:
        descriptor = os.open(
            lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(
            f"another native export or unresolved lock owns {output}: "
            f"{lock_path}") from exc
    os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
    return descriptor, lock_path


def _release_output_lock(descriptor: int, lock_path: Path) -> None:
    os.close(descriptor)
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass


def _prepare_output_target(output: Path, *, overwrite: bool) -> Path:
    """Recover an interrupted backup and validate overwrite policy."""

    backup = output.with_name(output.name + ".previous-valid")
    if backup.exists() and output.exists():
        raise RuntimeError(
            f"both current and previous-valid exports exist for {output}; "
            "refusing to choose or delete either automatically")
    if backup.exists():
        os.replace(backup, output)
    if output.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {output}")
    return backup


def _publish_staging(staging: Path, output: Path, backup: Path) -> None:
    """Publish a validated tree while retaining rollback on replacement."""

    if not output.exists():
        os.replace(staging, output)
        return
    if backup.exists():
        raise RuntimeError(f"refusing to replace existing backup {backup}")
    os.replace(output, backup)
    try:
        os.replace(staging, output)
    except BaseException:
        os.replace(backup, output)
        raise
    # At this point a complete, validated new export is live.  A process crash
    # before cleanup leaves the former valid tree recoverable at `backup`.
    shutil.rmtree(backup)


@contextmanager
def _output_publication(output: Path, *, overwrite: bool):
    descriptor, lock_path = _acquire_output_lock(output)
    try:
        yield _prepare_output_target(output, overwrite=overwrite)
    finally:
        _release_output_lock(descriptor, lock_path)


def _restore_attribute(value, type_spec: Mapping[str, object]):
    kind = type_spec["kind"]
    if kind == "str":
        return str(value)
    if kind == "bytes":
        return str(value).encode("latin1")
    if kind != "numeric":
        raise RuntimeError(f"unsupported frozen attribute kind {kind!r}")
    array = np.asarray(value, dtype=np.dtype(type_spec["dtype"]))
    shape = tuple(type_spec["shape"])
    if shape:
        return array.reshape(shape)
    return array.reshape(())[()]


class PreparedCache:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.header_path = self.root / "header.json"
        self.header = json.loads(self.header_path.read_text(encoding="utf-8"))
        if self.header.get("schema") != "gpuwm-prepared-real-cache-v1":
            raise ValueError("prepared cache has an unsupported schema")
        if self.header.get("status") != "READY":
            raise ValueError("prepared cache is not READY")
        self._arrays = self.header["arrays"]
        basis = {
            "schema": self.header["schema"],
            "identity": self.header["identity"],
            "metadata": self.header["metadata"],
            "arrays": self._arrays,
            "payload_bytes": self.header["payload_bytes"],
        }
        observed_content = hashlib.sha256(
            _canonical(basis).encode("utf-8")).hexdigest()
        if observed_content != self.header.get("content_sha256"):
            raise ValueError("prepared cache header content digest mismatch")
        # Boundary export visits every forcing interval.  Retaining every
        # memmap here exhausted file handles on long forecasts.  Reuse
        # live arrays without owning them: consumers and their views
        # keep the mapping valid for exactly as long as they need it.
        self._loaded: WeakValueDictionary[str, np.ndarray] = (
            WeakValueDictionary())
        self._verified_files: set[Path] = set()

    def array(self, name: str) -> np.ndarray:
        cached = self._loaded.get(name)
        if cached is not None:
            return cached
        try:
            spec = self._arrays[name]
        except KeyError as exc:
            raise KeyError(f"prepared cache lacks {name!r}") from exc
        path = self.root / spec["file"]
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        if list(value.shape) != spec["shape"] or str(value.dtype) != spec["dtype"]:
            raise ValueError(f"prepared cache declaration drift for {name}")
        if path not in self._verified_files:
            if _array_sha256(value) != spec["sha256"]:
                raise ValueError(f"prepared cache digest drift for {name}")
            self._verified_files.add(path)
        self._loaded[name] = value
        return value


def load_domain_artifacts_manifest(path: str | Path) \
        -> tuple[PreparedDomainArtifacts, ...]:
    """Load the strict, relocatable per-domain artifact inventory.

    Relative paths resolve against the manifest directory.  The manifest is
    intentionally small: it names already provenance-bound native artifacts;
    their content hashes remain authoritative and are rechecked during export.
    """

    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if set(payload) != {"schema", "domains"}:
        raise ValueError(
            "domain-artifact manifest keys must be exactly schema/domains")
    if payload["schema"] != "gpuwm-native-domain-artifacts-v1":
        raise ValueError("unsupported domain-artifact manifest schema")
    domains = payload["domains"]
    if not isinstance(domains, list) or not domains:
        raise ValueError("domain-artifact manifest requires a non-empty domains list")
    root = manifest_path.parent
    resolved_root = root.resolve()
    result = []
    expected_keys = {
        "grid_id", "prepared_cache", "static_cache", "geometry_receipt",
    }
    for index, item in enumerate(domains):
        if not isinstance(item, dict) or set(item) != expected_keys:
            raise ValueError(
                f"domain-artifact entry {index} keys must be exactly "
                f"{sorted(expected_keys)}")

        def resolved(name: str) -> Path:
            value = item[name]
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"domain-artifact entry {index} {name} must be a path string")
            candidate = Path(value)
            if candidate.is_absolute():
                raise ValueError(
                    f"domain-artifact entry {index} {name} must be a "
                    "relocatable relative path")
            target = (root / candidate).resolve()
            try:
                target.relative_to(resolved_root)
            except ValueError as exc:
                raise ValueError(
                    f"domain-artifact entry {index} {name} escapes the "
                    "manifest directory") from exc
            return target

        result.append(PreparedDomainArtifacts(
            grid_id=item["grid_id"],
            prepared_cache=resolved("prepared_cache"),
            static_cache=resolved("static_cache"),
            geometry_receipt=resolved("geometry_receipt"),
        ))
    identifiers = [item.grid_id for item in result]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(
            f"domain-artifact manifest has duplicate grid ids {identifiers}")
    return tuple(result)


def write_domain_artifacts_manifest(
        path: str | Path,
        artifacts: Sequence[PreparedDomainArtifacts],
) -> dict[str, object]:
    """Atomically join published per-domain artifacts with relative paths."""

    manifest_path = Path(path)
    if manifest_path.exists():
        raise FileExistsError(
            f"refusing to overwrite domain-artifact manifest {manifest_path}")
    records = tuple(sorted(artifacts, key=lambda item: item.grid_id))
    identifiers = [item.grid_id for item in records]
    from woof.wps_domain_ids import validated_domain_ids
    validated_domain_ids(identifiers)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    root = manifest_path.parent.resolve()

    def relative(target: Path, *, directory: bool) -> str:
        resolved = Path(target).resolve()
        if directory:
            if not resolved.is_dir() or not (resolved / "header.json").is_file():
                raise FileNotFoundError(
                    f"prepared cache is missing or incomplete: {resolved}")
        elif not resolved.is_file():
            raise FileNotFoundError(f"domain artifact is missing: {resolved}")
        try:
            value = resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"domain artifact {resolved} is outside manifest root {root}") \
                from exc
        return value.as_posix()

    payload = {
        "schema": "gpuwm-native-domain-artifacts-v1",
        "domains": [{
            "grid_id": artifact.grid_id,
            "prepared_cache": relative(
                artifact.prepared_cache, directory=True),
            "static_cache": relative(artifact.static_cache, directory=False),
            "geometry_receipt": relative(
                artifact.geometry_receipt, directory=False),
        } for artifact in records],
    }
    temporary = domain_artifacts_manifest_temporary(manifest_path)
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
            + "\n", encoding="utf-8")
        os.replace(temporary, manifest_path)
        loaded = load_domain_artifacts_manifest(manifest_path)
        expected = tuple(
            (item.grid_id, item.prepared_cache.resolve(),
             item.static_cache.resolve(), item.geometry_receipt.resolve())
            for item in records)
        observed = tuple(
            (item.grid_id, item.prepared_cache.resolve(),
             item.static_cache.resolve(), item.geometry_receipt.resolve())
            for item in loaded)
        if observed != expected:
            raise RuntimeError("domain-artifact manifest round-trip drift")
    except BaseException:
        temporary.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
        raise
    return payload


def _interp_nodes(nodes: np.ndarray) -> np.ndarray:
    output = []
    for depth in _NOAH_LAYER_MIDPOINTS_M:
        lower = int(np.searchsorted(_HRRR_SOIL_NODE_DEPTHS_M, depth) - 1)
        weight = (
            (depth - _HRRR_SOIL_NODE_DEPTHS_M[lower])
            / (_HRRR_SOIL_NODE_DEPTHS_M[lower + 1]
               - _HRRR_SOIL_NODE_DEPTHS_M[lower])
        )
        output.append(
            nodes[lower] + weight * (nodes[lower + 1] - nodes[lower]))
    return np.stack(output)


def _wrf_noah_landuse(raw_lu: np.ndarray, *, iswater: int = 17,
                      islake: int = 21) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(raw_lu, dtype=np.int32)
    lake_mask = raw == int(islake)
    mapped = np.where(lake_mask, int(iswater), raw).astype(np.int32)
    return mapped, lake_mask


def _wrf_soil_category(sct_dom, lu: np.ndarray, *,
                       iswater: int = 17) -> np.ndarray:
    """ISLTYP as real.exe writes it for this land use (lakes merged)."""
    from woof.core.landuse import soil_category_matched_to_land

    raw = np.asarray(sct_dom, dtype=np.int32)
    return soil_category_matched_to_land(
        raw, np.asarray(lu, dtype=np.int32) != int(iswater))


def _moist_pressure(cache: PreparedCache) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the native/WRF moist hydrostatic pressure recurrence."""
    qv = np.asarray(cache.array("state/qv"), dtype=np.float64)
    mub = np.asarray(cache.array("base/mub"), dtype=np.float64)
    mup = np.asarray(cache.array("state/mup"), dtype=np.float64)
    c1f = np.asarray(cache.array("coord/c1f"), dtype=np.float64)
    c2f = np.asarray(cache.array("coord/c2f"), dtype=np.float64)
    rdn = np.asarray(cache.array("coord/rdn"), dtype=np.float64)
    rdnw = np.asarray(cache.array("coord/rdnw"), dtype=np.float64)
    perturbation = np.empty_like(qv, dtype=np.float64)
    nz = qv.shape[0]
    cq = 1.0 / (1.0 + qv[-1])
    load = qv[-1] * cq
    perturbation[-1] = (
        -0.5 * (c1f[nz] * mup
                + load * (c1f[nz] * mub + c2f[nz]))
        / rdnw[nz - 1] / cq
    )
    for k in range(nz - 2, -1, -1):
        kw = k + 1
        qbar = 0.5 * (qv[k] + qv[k + 1])
        cq = 1.0 / (1.0 + qbar)
        load = qbar * cq
        perturbation[k] = (
            perturbation[k + 1]
            - (c1f[kw] * mup
               + load * (c1f[kw] * mub + c2f[kw]))
            / cq / rdn[kw]
        )
    total = np.asarray(cache.array("base/pb"), dtype=np.float64) + perturbation
    if not np.isfinite(total).all() or np.any(total <= 0.0):
        raise ValueError("direct exporter diagnosed invalid moist pressure")
    return perturbation, total


def _date_text(value: datetime) -> str:
    if value.tzinfo is not None:
        value = value.astimezone(timezone.utc).replace(tzinfo=None)
    if value.microsecond:
        raise ValueError("valid time must resolve to a whole second")
    return value.strftime("%Y-%m-%d_%H:%M:%S")


def _dimensions(contract: Mapping[str, object], *, nx: int, ny: int,
                nz: int, num_soil_layers: int = 4,
                bdy_width: int | None = None,
                ) -> dict[str, int | None]:
    """The file's dimension extents: the frozen contract's, resized.

    ``bdy_width`` is the run's ``spec_bdy_width``, the wrfbdy strip
    width (real.exe sizes ``bdy_width`` from it).  ``None`` keeps the
    contract's own extent, which is what a wrfinput (no such dimension)
    and the tools that rebuild the frozen file ask for.
    """
    overrides = {
        "Time": None,
        "west_east": nx,
        "west_east_stag": nx + 1,
        "south_north": ny,
        "south_north_stag": ny + 1,
        "bottom_top": nz,
        "bottom_top_stag": nz + 1,
        "soil_layers_stag": num_soil_layers,
    }
    if bdy_width is not None:
        overrides["bdy_width"] = int(bdy_width)
    return {
        item["name"]: overrides.get(item["name"], int(item["length"]))
        for item in contract["dimensions"]
    }


def _land_category_contract(
        input_contract: Mapping[str, object],
        fields: Mapping[str, np.ndarray],
) -> tuple[Mapping[str, object], dict[str, object]]:
    """The wrfinput contract and global attributes for the land use written.

    ``fields`` are the wrfinput fields (:func:`_wrfinput_fields`), whose
    LANDUSEF is the prepared static's own.

    real.exe sizes ``land_cat_stag`` and states ``NUM_LAND_CAT`` from
    ``num_land_cat``, the category count of the met_em land use, so a
    stock wrfinput holds exactly the LANDUSEF planes its land use has.
    The frozen contract is the 21-category MODIS reference file.  An
    urban canopy run prepares MODIS plus WRF's built types 51-61
    (``woof.static.highres.landcover_legend``: 61 planes, the
    NUM_LAND_CAT the forecast's own history files state), and the
    15 arc-second MODIS set has 20 planes and no lake class
    (ISLAKE = -1, :func:`woof.native_wrf_contract.native_landuse_identity`).
    Writing either into the frozen 21 stopped the companion export after
    the forecast had run ("LANDUSEF source shape (61, 170, 170) !=
    (21, 170, 170)").  A 21-category static gets the contract back
    unchanged and no attribute, so its files keep their bytes; so do
    fields with no LANDUSEF, which the writer fills at the contract's own
    shape.  The resolved physics contract, whose digest the preparation
    records and checks, stays as it is: land use is not physics.
    """
    landusef = fields.get("LANDUSEF")
    if landusef is None:
        return input_contract, {}
    categories = int(np.shape(landusef)[0])
    frozen = {
        item["name"]: int(item["length"])
        for item in input_contract["dimensions"]
    }["land_cat_stag"]
    if categories == frozen:
        return input_contract, {}
    from woof.native_wrf_contract import native_landuse_identity

    contract = copy.deepcopy(dict(input_contract))
    for item in contract["dimensions"]:
        if item["name"] == "land_cat_stag":
            item["length"] = categories
    attributes = {"NUM_LAND_CAT": categories}
    islake = native_landuse_identity(fields)["ISLAKE"]
    if islake != contract["global_attributes"]["ISLAKE"]:
        attributes["ISLAKE"] = islake
    return contract, attributes


def _prepared_vertical_contract(
    cache: PreparedCache, *, nx: int, ny: int, nz: int,
    mp_physics: int | None = None,
) -> float:
    """Validate serialized vertical dimensions and return the bound p_top.

    ``mp_physics`` selects WHICH moist arrays the prepared cache must carry.
    It used to require WSM6's six unconditionally, which was correct while
    every exportable package declared those six; mp_physics=50 is the first
    that does not (Registry.EM_COMMON:3038 -- P3 carries one ice category and
    no snow or graupel species), and the prepared cache for a P3 domain
    therefore has no ``state/qs`` and no ``state/qg`` to offer.  Requiring
    them turned a correct cache into "direct-WRF prepared cache lacks
    required arrays ['state/qg', 'state/qs']" and refused the run.  ``None``
    keeps the historical six for callers that have not resolved a scheme.
    """

    coord_shapes = {
        name.removeprefix("coord/"): spec["shape"]
        for name, spec in cache._arrays.items()
        if name.startswith("coord/")
    }
    validate_coordinate_shapes(
        coord_shapes, nz=nz, context="direct-WRF prepared cache")

    expected_shapes = {
        "state/u": (nz, ny, nx + 1),
        "state/v": (nz, ny + 1, nx),
        "state/w": (nz + 1, ny, nx),
        "state/php": (nz + 1, ny, nx),
        "state/thp": (nz, ny, nx),
        "state/mup": (ny, nx),
        "base/mub": (ny, nx),
        "base/pb": (nz, ny, nx),
        "base/alb": (nz, ny, nx),
        "base/thb": (nz, ny, nx),
        "base/phb": (nz + 1, ny, nx),
        "base/terrain_z": (ny, nx),
    }
    # The moist half, from the package rather than from WSM6's habit.
    if mp_physics is None:
        moist_registry_names = ("qv", "qc", "qr", "qi", "qs", "qg")
    else:
        moist_registry_names = tuple(
            field.registry_name
            for field in stock_wrf_physics_inventory(
                int(mp_physics)).wrfinput_fields
            if field.collection == "moist"
        )
    for registry_name in moist_registry_names:
        expected_shapes[f"state/{registry_name}"] = (nz, ny, nx)
    missing = sorted(set(expected_shapes) - set(cache._arrays))
    if missing:
        raise ValueError(
            f"direct-WRF prepared cache lacks required arrays {missing}")
    drift = {
        name: {
            "actual": tuple(cache._arrays[name]["shape"]),
            "expected": shape,
        }
        for name, shape in expected_shapes.items()
        if tuple(cache._arrays[name]["shape"]) != shape
    }
    if drift:
        raise ValueError(
            f"direct-WRF prepared vertical/state shape drift: {drift}")

    metadata = cache.header.get("metadata", {})
    try:
        base_p_top = float(metadata["base_scalars"]["p_top"])
        coord_p_top = float(metadata["coord_scalars"]["p_top"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "direct-WRF prepared cache lacks numeric bound p_top metadata"
        ) from exc
    if coord_p_top != base_p_top:
        raise ValueError(
            "direct-WRF prepared coordinate/base p_top values differ")
    validate_explicit_eta_grid(
        cache.array("coord/znw"),
        nz=nz,
        p_top=base_p_top,
        context="direct-WRF prepared cache",
    )
    return base_p_top


def _global_updates(*, valid_time: datetime, nx: int, ny: int, nz: int,
                    dx: float, dy: float, dt: float,
                    geometry: Mapping[str, object],
                    physics_selection: Mapping[str, object] | None = None,
                    run_config: Mapping[str, object] | None = None,
                    ) -> dict[str, object]:
    stamp = _date_text(valid_time)
    cfg = dict(run_config or {})
    updates = {
        # WRF's input gate recognizes versioned initial data from the TITLE
        # token itself.  Preserve an accurate producer label while retaining
        # the required V4.x-compatible marker.
        "TITLE": " OUTPUT FROM GPUWM NATIVE WRF V4.6.1-COMPAT DIRECT EXPORTER",
        # The representation the fields below are written in; WRF's input
        # gate compares it with the namelist (EXPORT_USE_THETA_M).
        "USE_THETA_M": EXPORT_USE_THETA_M,
        "START_DATE": stamp,
        "SIMULATION_START_DATE": stamp,
        "WEST-EAST_GRID_DIMENSION": nx + 1,
        "SOUTH-NORTH_GRID_DIMENSION": ny + 1,
        "BOTTOM-TOP_GRID_DIMENSION": nz + 1,
        "WEST-EAST_PATCH_END_UNSTAG": nx,
        "WEST-EAST_PATCH_END_STAG": nx + 1,
        "SOUTH-NORTH_PATCH_END_UNSTAG": ny,
        "SOUTH-NORTH_PATCH_END_STAG": ny + 1,
        "BOTTOM-TOP_PATCH_END_UNSTAG": nz,
        "BOTTOM-TOP_PATCH_END_STAG": nz + 1,
        "GRID_ID": 1,
        "PARENT_ID": 0,
        "I_PARENT_START": 1,
        "J_PARENT_START": 1,
        "PARENT_GRID_RATIO": 1,
        "DX": dx,
        "DY": dy,
        "DT": dt,
        "GHG_INPUT": 0,
        # These are this run's choices, never the prototype file's values.
        # GHG_INPUT remains 0: the engine uses analytic year-dependent gas
        # concentrations, not WRF's optional CAMtr file (ghg_input=1).
        "SURFACE_INPUT_SOURCE": 3,
        "DIFF_OPT": int(cfg.get("diff_opt", 2)),
        "DIFF_6TH_SLOPEOPT": int(cfg.get("diff_6th_slopeopt", 0)),
        "CEN_LAT": geometry["center_lat"],
        "CEN_LON": geometry["center_lon"],
        "TRUELAT1": geometry["truelat1"],
        "TRUELAT2": geometry["truelat2"],
        # The mother domain's centre, which every domain of a hierarchy
        # shares.  `ref_lat` is that value only on a grid whose
        # reference IS its centre; on a nest it is the (1, 1) corner
        # latitude, and every consumer that rebuilds the projection from
        # the history file reads this attribute as the CRS origin.
        "MOAD_CEN_LAT": geometry.get("moad_cen_lat", geometry["ref_lat"]),
        "STAND_LON": geometry["stand_lon"],
        # Projection identity: WRF convention 1=lambert, 2=polar
        # stereographic, 3=mercator (a receipt without map_proj is a
        # legacy Lambert one).
        "MAP_PROJ": WRF_MAP_PROJ_CODES[
            str(geometry.get("map_proj", "lambert"))],
        "MAP_PROJ_CHAR": MAP_PROJ_CHARS[
            str(geometry.get("map_proj", "lambert"))],
        "GMT": (valid_time.hour + valid_time.minute / 60.0
                + valid_time.second / 3600.0),
        "JULYR": valid_time.year,
        "JULDAY": valid_time.timetuple().tm_yday,
    }
    if physics_selection is not None:
        selectors = physics_selection.get("selectors")
        if not isinstance(selectors, Mapping):
            raise ValueError(
                "front-door physics selection lacks selector provenance")
        global_names = _GLOBAL_PHYSICS_ATTRIBUTES
        missing = sorted(set(global_names) - set(selectors))
        if missing:
            raise ValueError(
                f"front-door physics selection lacks selectors {missing}")
        updates.update({
            global_name: selectors[selector]
            for selector, global_name in global_names.items()
        })
    return updates


#: The physics a stock wrfinput states in its global attributes, by the
#: selector each one carries.  The frozen contract holds the WSM6 + Dudhia
#: reference file's values for all seven, so an export that stamps none of
#: them states that file's physics, not its own.
_GLOBAL_PHYSICS_ATTRIBUTES = {
    "mp_physics": "MP_PHYSICS",
    "ra_lw_physics": "RA_LW_PHYSICS",
    "ra_sw_physics": "RA_SW_PHYSICS",
    "sf_sfclay_physics": "SF_SFCLAY_PHYSICS",
    "sf_surface_physics": "SF_SURFACE_PHYSICS",
    "bl_pbl_physics": "BL_PBL_PHYSICS",
    "cu_physics": "CU_PHYSICS",
}


def recorded_physics_selectors(cfg: Mapping[str, object], *,
                               label: str = "direct-export"
                               ) -> dict[str, object]:
    """The physics a prepared domain was prepared with, as global selectors.

    Read from the cache's own recorded configuration, the one every
    identity check binds, for an export no physics selection was handed
    to: the profile-free single-domain export (``python -m
    woof.wrf_direct`` without ``--physics-profile``, the ERA5 door) and
    every domain of the tree export.  Those used to stamp nothing, so the
    frozen contract's global MP_PHYSICS = 6, RA_LW_PHYSICS = 0,
    RA_SW_PHYSICS = 1 and CU_PHYSICS = 0 went out beside a Thompson,
    Morrison or RTE+RRTMGP cache's arrays, and stock WRF reads its scheme
    from those attributes.  The radiation pair is resolved through the
    engine's own rule, so the combined ``ra_physics`` spelling states the
    pair it runs.  A cache that records no value for one of them is
    refused by name rather than filled from the frozen file.
    """

    from woof.config import radiation_scheme_ids_from_settings

    missing = [key for key in _GLOBAL_PHYSICS_ATTRIBUTES
               if key not in ("ra_lw_physics", "ra_sw_physics")
               and cfg.get(key) is None]
    if (cfg.get("ra_physics") is None
            and (cfg.get("ra_lw_physics") is None
                 or cfg.get("ra_sw_physics") is None)):
        missing.append("ra_physics")
    if missing:
        raise ValueError(
            f"{label} prepared cache records no {sorted(missing)}, so the "
            "export cannot state the physics its arrays were prepared with "
            "and would write the frozen WSM6 reference file's instead; "
            "re-prepare the case")
    longwave, shortwave = radiation_scheme_ids_from_settings(cfg)
    selectors = {key: int(cfg[key]) for key in _GLOBAL_PHYSICS_ATTRIBUTES
                 if key not in ("ra_lw_physics", "ra_sw_physics")}
    selectors.update(ra_lw_physics=int(longwave), ra_sw_physics=int(shortwave))
    return selectors


def _create_dataset(path: Path, contract: Mapping[str, object],
                    dimensions: Mapping[str, int | None],
                    global_updates: Mapping[str, object],
                    engine: str | None = None):
    """Declare the whole frozen contract, then hand back the open file.

    The two engines differ only in who puts the bytes down: ``rust`` (the
    default) opens a :class:`~woof.io.classic_tape.ClassicTape` on the
    Rust seam, ``python`` opens the frozen contract's own NetCDF-4
    container with the C library.  Everything below is one code path, so
    the declaration cannot drift between them.
    """

    engine = resolve_wrfinput_engine(engine)
    if engine == "rust":
        _require_rust_writer()
        dataset = ClassicTape(path, container="cdf2",
                              frozen_remedy=_FROZEN_REMEDY)
    else:
        dataset = netCDF4.Dataset(path, "w", format=str(contract["format"]))
    try:
        for name, length in dimensions.items():
            dataset.createDimension(name, length)
        for name, value in contract["global_attributes"].items():
            restored = _restore_attribute(
                global_updates.get(name, value),
                contract["global_attribute_types"][name],
            )
            dataset.setncattr(name, restored)
        for spec in contract["variables"]:
            # The classic container has no spelling for HDF5 storage
            # tuning, so the rust engine declares the variable and
            # nothing else; the C library keeps the frozen contract's
            # chunking and compression exactly as 2.4 wrote them.
            kwargs: dict[str, object] = {}
            if engine == "python":
                compression = spec["compression"]
                kwargs = {
                    "zlib": bool(compression["zlib"]),
                    "shuffle": bool(compression["shuffle"]),
                    "complevel": int(compression["complevel"]),
                    "fletcher32": bool(compression["fletcher32"]),
                    "endian": spec["endian"],
                }
                if isinstance(spec["chunking"], list):
                    kwargs["chunksizes"] = tuple(
                        min(int(chunk), int(dimensions[dim]))
                        if dimensions[dim] is not None else int(chunk)
                        for chunk, dim in zip(spec["chunking"],
                                              spec["dimensions"])
                    )
                if spec["has_fill_value"]:
                    kwargs["fill_value"] = spec["fill_value"]
            variable = dataset.createVariable(
                spec["name"], np.dtype(spec["dtype"]),
                tuple(spec["dimensions"]), **kwargs)
            if spec["attributes"]:
                variable.setncatts({
                    name: _restore_attribute(
                        value, spec["attribute_types"][name])
                    for name, value in spec["attributes"].items()
                })
        return dataset
    except BaseException:
        _abandon_dataset(dataset, path)
        raise


def _abandon_dataset(dataset, path: Path) -> None:
    """Release a half-written file without publishing it.

    A classic tape is ABORTED, which leaves ``numrecs`` 0 so a reader of
    the leftover sees an empty inventory rather than a partial pair
    presented as whole; a netCDF4 dataset is closed, which is all that
    library offers.  The file is then removed, because unlike the wrfout
    tape this export publishes by directory rename and a leftover would
    travel with it.
    """

    with suppress(BaseException):
        if isinstance(dataset, ClassicTape):
            dataset.abort()
        else:
            dataset.close()
    with suppress(BaseException):
        Path(path).unlink(missing_ok=True)


def _shape_without_time(variable) -> tuple[int, ...]:
    return tuple(len(variable.group().dimensions[name])
                 for name in variable.dimensions if name != "Time")


def _write_time_value(variable, value) -> None:
    _write_time_value_at(variable, value, 0)


def _write_time_value_at(variable, value,
                         time_index: int) -> None:
    if not variable.dimensions or variable.dimensions[0] != "Time":
        raise ValueError(f"{variable.name} does not begin with Time")
    if time_index < 0:
        raise ValueError("time index must be non-negative")
    target_shape = _shape_without_time(variable)
    array = np.asarray(value)
    if (array.ndim == len(target_shape) + 1
            and array.shape[0] == 1
            and tuple(array.shape[1:]) == target_shape):
        array = array[0]
    if tuple(array.shape) != target_shape:
        raise ValueError(
            f"{variable.name} source shape {array.shape} != {target_shape}")
    variable[(time_index,) + (slice(None),) * len(target_shape)] = array


def _write_timestamp(variable, text: str) -> None:
    _write_timestamp_at(variable, text, 0)


def _write_timestamp_at(variable, text: str,
                        time_index: int) -> None:
    encoded = np.frombuffer(text.encode("ascii"), dtype="S1")
    _write_time_value_at(variable, encoded, time_index)


def _resolved_prototype_value(
    variable, prototype
):
    """Keep matching frozen prototypes; resize only all-zero scaffolds.

    The v4.6.1 metadata contract was captured from the authoritative
    49-level file.  Several diagnostic base-profile placeholders are all
    zero but therefore carry a prototype shape of 49.  Their semantic value
    is shape-independent zero, so derive that shape from the live NetCDF
    dimensions.  A nonzero mismatched prototype remains a hard error.
    """

    target_shape = _shape_without_time(variable)
    array = np.asarray(prototype)
    compatible = (
        tuple(array.shape) == target_shape
        or (array.ndim == len(target_shape) + 1
            and array.shape[0] == 1
            and tuple(array.shape[1:]) == target_shape)
    )
    if compatible:
        # Preserve the exact accepted 49-level write path and conversion.
        return prototype
    if array.size and np.all(array == 0):
        return np.zeros(target_shape, dtype=variable.dtype)
    raise ValueError(
        f"{variable.name} has a nonzero frozen prototype shape "
        f"{array.shape} that cannot serve derived shape {target_shape}")


def _run_start_vegetation(cache: PreparedCache,
                          static: Mapping[str, np.ndarray],
                          valid_time: datetime,
                          run_config: Mapping[str, object] | None,
                          ) -> dict[str, np.ndarray]:
    """VEGFRA, LAI and ALBBCK as the prepared forecast starts with them.

    The forecast takes the analyzed vegetation fraction when its start
    source carries one (``woof.ingest.vegetation.initial_vegetation_
    fraction``), otherwise GREENFRAC interpolated to the start date, and
    LAI and the background albedo interpolated to the date as real.exe
    does (module_initialize_real.F:1192, :1322-1335).  The export used the
    calendar month's raw value of each instead, so the pair described
    another land surface than the run that wrote it: on the 2024-05-21 18Z
    HRRR-start crop VEGFRA differed by 12 % RMS (up to 37 %) and stock WRF's
    latent heat flux by 89 W m-2 RMS after ten minutes.
    """
    from types import SimpleNamespace

    from woof.core.landuse import monthly_background_albedo, surface_leaf_area
    from woof.ingest.vegetation import initial_vegetation_fraction

    analyzed = ({"VEGFRA": np.asarray(cache.array("met/VEGFRA"))}
                if "met/VEGFRA" in cache._arrays else {})
    vegfra = initial_vegetation_fraction(
        SimpleNamespace(fields=analyzed), static, valid_time)
    run = SimpleNamespace(**dict(run_config or {}))
    return {
        "VEGFRA": np.asarray(vegfra, dtype=np.float64),
        "LAI": np.asarray(surface_leaf_area(run, static["LAI12M"], valid_time),
                          dtype=np.float64),
        "ALBBCK": np.asarray(monthly_background_albedo(
            static["ALBEDO12M"], static["LANDMASK"], valid_time),
            dtype=np.float64),
    }


def _surface_fields(cache: PreparedCache, static: Mapping[str, np.ndarray],
                    valid_time: datetime,
                    run_config: Mapping[str, object] | None = None,
                    ) -> dict[str, np.ndarray]:
    vegetation = _run_start_vegetation(cache, static, valid_time, run_config)
    declared = {
        name.removeprefix("surface/")
        for name in cache._arrays
        if name.startswith("surface/")
    }
    if declared and declared != _CANONICAL_SURFACE_FIELDS:
        raise ValueError(
            "prepared cache contains an incomplete canonical surface: "
            f"expected {sorted(_CANONICAL_SURFACE_FIELDS)}, "
            f"got {sorted(declared)}")
    if declared:
        result = {
            name: np.asarray(cache.array(f"surface/{name}"), dtype=np.float64)
            for name in sorted(_CANONICAL_SURFACE_FIELDS)
        }
        land = result["LANDMASK"] >= 0.5
        green = np.asarray(static["GREENFRAC"], dtype=np.float64)
        snoalb = np.asarray(static["SNOALB"], dtype=np.float64) / 100.0
        result.update({
            "VEGFRA": vegetation["VEGFRA"],
            "SHDMAX": 100.0 * np.max(green, axis=0),
            "SHDMIN": 100.0 * np.min(green, axis=0),
            "SHDAVG": 100.0 * np.mean(green, axis=0),
            "ALBBCK": vegetation["ALBBCK"],
            "LAI": vegetation["LAI"],
            "SNOALB": np.where(land, np.maximum(snoalb, 0.08), 0.08),
        })
        return result

    tsk = np.asarray(cache.array("met/SKINTEMP"), dtype=np.float64)
    land = np.asarray(static["LANDMASK"], dtype=np.float64) >= 0.5
    xice = np.asarray(cache.array("met/XICE"), dtype=np.float64).copy()
    xice[land] = 0.0
    xice = np.where((~land) & (xice >= 0.5), 1.0, 0.0)
    effective_land = land | (xice >= 0.5)
    soilt = _interp_nodes(np.asarray(cache.array("met/SOILT"), dtype=np.float64))
    soilw = _interp_nodes(np.asarray(cache.array("met/SOILW"), dtype=np.float64))
    soilt[:, ~land] = tsk[~land]
    soilw[:, ~land] = 1.0
    # This proof case is warm (all soil nodes > 285 K), so WRF's frozen-water
    # initialization is exactly SH2O == SMOIS.  Fail closed for colder cases
    # until the table-driven partition is exported here as well.
    #
    # EXPORT-REPRESENTABILITY, not preparation: woof's own Noah surface
    # layer carries frozen soil through the forecast; only the
    # unchanged-WRF wrfinput file set has no table-driven partition to
    # say SH2O with here.  That is precisely the category
    # StockWrfExportUnsupported's docstring reserves for itself, so a
    # caller whose product is a PREPARED FORECAST can record the refusal
    # and proceed while a caller whose product IS the export still
    # fails.  As a bare ValueError it was uncatchable: a measured
    # mountainous-west domain had 1438 of 65340 columns with a soil node
    # below freezing -- every one of them land, at the DEEPEST node,
    # between 1600 m and 3274 m of terrain -- and lost a complete,
    # verified GPU hierarchy to a missing oracle file.
    if np.any(soilt < 273.15):
        raise StockWrfExportUnsupported(
            "direct WRF export does not yet support frozen-soil SH2O setup")
    tmn = np.asarray(static["TMN"], dtype=np.float64).copy()
    valid_tmn = np.isfinite(tmn) & (tmn >= 170.0) & (tmn <= 400.0)
    tmn = np.where(land & valid_tmn, tmn, tsk)
    green = np.asarray(static["GREENFRAC"], dtype=np.float64)
    snoalb = np.asarray(static["SNOALB"], dtype=np.float64) / 100.0
    snoalb = np.where(land, np.maximum(snoalb, 0.08), 0.08)
    return {
        "TSK": tsk,
        "TSLB": soilt,
        "SMOIS": soilw,
        "SH2O": soilw.copy(),
        "TMN": tmn,
        "SEAICE": xice,
        "XLAND": np.where(effective_land, 1.0, 2.0),
        "LANDMASK": effective_land.astype(np.float64),
        "VEGFRA": vegetation["VEGFRA"],
        "SHDMAX": 100.0 * np.max(green, axis=0),
        "SHDMIN": 100.0 * np.min(green, axis=0),
        "SHDAVG": 100.0 * np.mean(green, axis=0),
        "ALBBCK": vegetation["ALBBCK"],
        "LAI": vegetation["LAI"],
        "SNOALB": snoalb,
    }


#: How far a rebuilt mass-grid footprint may sit from the extremes the
#: geometry receipt recorded, in degrees.  Same value as
#: ``woof/obs/target_grid.py``'s ``GEOREF_TOLERANCE_DEG``, which gates
#: the same question from the other end (a wrfout's own XLAT/XLONG
#: against the projection rebuilt from its global attributes).  The
#: failure this one exists for -- a projection anchor the receipt did
#: not carry -- displaces the footprint by half a domain (about 2.7 deg
#: on a 200x200 3 km nest) and can never displace it by less than half a
#: cell, so the band is orders below anything it has to catch and orders
#: above float64 noise on a reconstruction of the same arithmetic
#: (measured: bitwise equal for both a nest and a translated grid).
_FOOTPRINT_RANGE_TOLERANCE_DEG = 1.0e-5


def _require_rebuilt_footprint(geometry: Mapping[str, object],
                               latitude: np.ndarray,
                               longitude: np.ndarray) -> None:
    """Refuse a rebuilt projection that does not land where the receipt says.

    ``lat_range``/``lon_range`` are written by
    :func:`woof.native_wrf_contract.native_geometry_contract` from the
    AUTHORING grid's own mass points, and until this gate they had no
    consumer anywhere in the export path.  Everything else the emitted
    file carries for the same cells -- MAPFAC_M/U/V, F, E, SINALPHA,
    COSALPHA -- is read from the static cache built on that grid, so a
    rebuild that lands somewhere else emits one file whose latitudes and
    whose map factors describe two different places, which no inventory,
    dimension, attribute or finiteness check downstream can see.
    """
    drift = {}
    for name, value in (("lat_range", latitude), ("lon_range", longitude)):
        recorded = geometry.get(name)
        if recorded is None:
            raise ValueError(
                f"native geometry receipt carries no {name}; the rebuilt "
                "projection cannot be checked against the grid the statics "
                "were built on")
        expected = [float(item) for item in recorded]
        rebuilt = [float(np.min(value)), float(np.max(value))]
        offset = max(abs(a - b) for a, b in zip(expected, rebuilt))
        if offset > _FOOTPRINT_RANGE_TOLERANCE_DEG:
            drift[name] = {"receipt": expected, "rebuilt": rebuilt,
                           "degrees": offset}
    if drift:
        raise ValueError(
            "the rebuilt projection does not reproduce the footprint the "
            f"geometry receipt recorded: {drift}")


def _terrain_drag_export_fields(static, cfg, shape):
    """Carry required prepared statistics without inventing missing values."""
    from woof.static.orographic import required_static_fields
    result = {}
    for name in required_static_fields(cfg.get("topo_wind", 0), cfg.get("gwd_opt", 0)):
        if name not in static:
            raise ValueError(f"WRF export missing required terrain-drag statistic {name}")
        value = np.asarray(static[name], dtype=np.float32)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"WRF export invalid terrain-drag statistic {name}: {value.shape}, expected {shape}")
        result[name] = value
    if "VAR_SSO" in result:
        result["GOT_VAR_SSO"] = np.asarray(1, dtype=np.int32)
    return result


def _lake_export_fields(static, cfg, mask):
    """Preserve prepared bathymetry and its WRF presence flag."""
    depth = static.get("LAKE_DEPTH")
    if depth is None:
        if cfg.get("sf_lake_physics", 0) and cfg.get("use_lakedepth", 0) and np.any(mask):
            raise ValueError("WRF export missing required lake bathymetry LAKE_DEPTH")
        return {}
    depth = np.asarray(depth, dtype=np.float32)
    if depth.shape != mask.shape or not np.isfinite(depth).all():
        raise ValueError("WRF export invalid lake bathymetry LAKE_DEPTH")
    return {"LAKE_DEPTH": depth, "LAKE_DEPTH_FLAG": np.asarray(1, dtype=np.int32)}


def _real_exe_extrapolation_constants(dn, dnw, fnp, fnm) -> dict[str, np.ndarray]:
    """CF1, CF2, CF3, CFN and CFN1 as real.exe writes them for THIS column.

    ``module_initialize_real.F:3734-3742`` (v4.6.1), FP32 in Fortran's
    left-to-right order; Fortran index ``k`` is ``[k - 1]`` here.  wrf.exe
    never recomputes them: the surface (CF1-3) and model-top (CFN/CFN1)
    extrapolations of every step read the file's words.  The writer used to
    leave all five to the contract's prototype, a template's values for some
    other vertical grid (the UAE 44-level export carried CFN 1.4764977 where
    its own DN/DNW give 1.4612311), so wrf.exe ran a different top boundary
    than the levels it was handed.
    """
    f32 = np.float32
    dn = np.asarray(dn, dtype=f32).reshape(-1)
    dnw = np.asarray(dnw, dtype=f32).reshape(-1)
    fnp = np.asarray(fnp, dtype=f32).reshape(-1)
    fnm = np.asarray(fnm, dtype=f32).reshape(-1)
    dn2, dn3, dnw1 = dn[1], dn[2], dnw[0]
    cof1 = f32(f32(f32(f32(f32(2.0) * dn2) + dn3) / f32(dn2 + dn3)) * dnw1)
    cof1 = f32(cof1 / dn2)
    cof2 = f32(f32(f32(dn2 / f32(dn2 + dn3)) * dnw1) / dn3)
    top_dn, top_dnw = dn[-1], dnw[-1]
    return {
        "CF1": np.asarray(f32(fnp[1] + cof1)),
        "CF2": np.asarray(f32(f32(fnm[1] - cof1) - cof2)),
        "CF3": np.asarray(cof2),
        "CFN": np.asarray(f32(f32(f32(f32(0.5) * top_dnw) + top_dn) / top_dn)),
        "CFN1": np.asarray(f32(-f32(f32(f32(0.5) * top_dnw) / top_dn))),
    }


def _wrfinput_fields(cache: PreparedCache, static: Mapping[str, np.ndarray],
                     geometry: Mapping[str, object],
                     valid_time: datetime, *,
                     p_top: float, mp_physics: int = 6,
                     sf_surface_physics: int = 2,
                     num_soil_layers: int = 4,
                     run_config: Mapping[str, object] | None = None,
                     ) -> dict[str, np.ndarray]:
    u = cache.array("state/u")
    v = cache.array("state/v")
    nz, ny, nx1 = u.shape
    nx = nx1 - 1
    if v.shape != (nz, ny + 1, nx):
        raise ValueError("prepared U/V staggering is inconsistent")
    dx = float(geometry["dx_m"])
    dy = float(geometry["dy_m"])
    grid = projection_class(str(geometry.get("map_proj", "lambert")))(
        geometry["ref_lat"], geometry["ref_lon"], geometry["truelat1"],
        geometry["truelat2"], geometry["stand_lon"], dx, dy,
        nx + 1, ny + 1,
        # A nest's reference point is its (1, 1) mass point, not its
        # centre; rebuilding without the anchor took the centred WPS
        # default and translated every XLAT/XLONG here by half the
        # domain while MAPFAC/F/SINALPHA beside them stayed on the
        # correct grid.  A receipt written before the anchor was carried
        # leaves these None, which is exactly the centred default it
        # meant -- and `_require_rebuilt_footprint` below is what
        # refuses when that default is not what the grid was.
        known_x=geometry.get("known_x"), known_y=geometry.get("known_y"),
        moad_cen_lat=geometry.get("moad_cen_lat"),
        moad_cen_lon=geometry.get("moad_cen_lon"),
    )
    lat, lon = grid.latlon_mass()
    _require_rebuilt_footprint(geometry, lat, lon)
    lat_u, lon_u = grid.latlon_u()
    lat_v, lon_v = grid.latlon_v()

    qv = np.asarray(cache.array("state/qv"), dtype=np.float64)
    theta = (np.asarray(cache.array("base/thb"), dtype=np.float64)
             + np.asarray(cache.array("state/thp"), dtype=np.float64))
    # Boundary tables couple the model's float32 total_theta, rounded
    # before subtracting 300 K.  Preserve that order in the prognostic
    # records too; a float64 sum leaves extra bits in T that make this
    # export fail the reader's initial-boundary consistency check.
    dry_theta_perturbation = (
        np.asarray(cache.array("base/thb"), dtype=np.float32)
        + np.asarray(cache.array("state/thp"), dtype=np.float32)
    ) - np.float32(_THETA_OFFSET_K)
    theta_m = theta * (1.0 + _RVOVRD * qv)
    pressure_perturbation, total_pressure = _moist_pressure(cache)
    alpha = (_RD * theta_m * (total_pressure / _P0) ** _RCP
             / total_pressure)
    al_perturbation = (
        alpha - np.asarray(cache.array("base/alb"), dtype=np.float64))

    psfc = np.asarray(cache.array("result/surface_pressure"), dtype=np.float64)
    t2 = np.asarray(cache.array("met/T2"), dtype=np.float64)
    u10_face = np.asarray(cache.array("met/U10"), dtype=np.float64)
    v10_face = np.asarray(cache.array("met/V10"), dtype=np.float64)
    surface = _surface_fields(cache, static, valid_time, run_config)
    soil_depths, soil_thicknesses = _direct_export_soil_geometry(
        sf_surface_physics, num_soil_layers)
    soil_shape_drift = {
        name: tuple(np.asarray(surface[name]).shape)
        for name in ("TSLB", "SMOIS", "SH2O")
        if tuple(np.asarray(surface[name]).shape)
        != (int(num_soil_layers), ny, nx)
    }
    if soil_shape_drift:
        raise ValueError(
            f"prepared surface soil geometry differs from "
            f"sf_surface_physics={sf_surface_physics}, "
            f"num_soil_layers={num_soil_layers}: {soil_shape_drift}")
    raw_lu = np.asarray(static["LU_INDEX"], dtype=np.int32)
    # WRF real.exe remaps MODIS lake category 21 to the Noah water category
    # when sf_lake_physics=0.  Leaving IVGTYP=21 reaches Noah's fatal
    # "too many input landuse types" guard because VEGPARM has categories
    # 1..20.  Preserve LAKEMASK separately exactly as real.exe does.
    lu, lake_mask = _wrf_noah_landuse(raw_lu)
    # real.exe writes the soil category matched to the land mask
    # (module_initialize_real.F:3108-3131): a land column whose soil map
    # says water carries silty clay loam, never water soil under land.
    soil = _wrf_soil_category(static["SCT_DOM"], lu)
    # Match the model's existing WRF sea-ice category rule, including the
    # fractional-ice threshold when that policy is selected.
    from types import SimpleNamespace
    from woof.core.landuse import ruc_fractional_seaice
    fractional_ice = ruc_fractional_seaice(SimpleNamespace(**dict(run_config or {})))
    ice = np.asarray(surface["SEAICE"]) >= (0.02 if fractional_ice else 0.5)
    lu = np.where(ice, 15, lu).astype(np.int32)
    soil = np.where(ice, 16, soil).astype(np.int32)
    surface["SNOALB"] = np.where(ice, 0.75, surface["SNOALB"])
    landmask = surface["LANDMASK"]
    snow = np.asarray(
        surface["SNOW"] if "SNOW" in surface else cache.array("met/SNOW"),
        dtype=np.float64)
    snowh = np.asarray(
        surface["SNOWH"] if "SNOWH" in surface else cache.array("met/SNOWH"),
        dtype=np.float64)

    result: dict[str, np.ndarray] = {
        "XLAT": lat,
        "XLONG": lon,
        "XLAT_U": lat_u,
        "XLONG_U": lon_u,
        "XLAT_V": lat_v,
        "XLONG_V": lon_v,
        "CLAT": lat,
        "LU_INDEX": lu,
        "ZNU": cache.array("coord/znu"),
        "ZNW": cache.array("coord/znw"),
        "ZS": soil_depths,
        "DZS": soil_thicknesses,
        "U": u,
        "V": v,
        "W": cache.array("state/w"),
        "PH": cache.array("state/php"),
        "PHB": cache.array("base/phb"),
        "T": dry_theta_perturbation,
        # WRF's prognostic record.  Under use_theta_m = 0 it is the dry
        # perturbation itself, equal to T (EXPORT_USE_THETA_M); theta_m
        # above still enters the equation of state, as it does in WRF.
        "THM": dry_theta_perturbation,
        # T_INIT is not trajectory-active after input.  The prepared cache
        # retains the final dry-theta state but not real.exe's pre-adjustment
        # diagnostic, so bind it to the final dry theta explicitly.
        "T_INIT": dry_theta_perturbation,
        "MU": cache.array("state/mup"),
        "MUB": cache.array("base/mub"),
        "P": pressure_perturbation,
        "AL": al_perturbation,
        "ALB": cache.array("base/alb"),
        "PB": cache.array("base/pb"),
        "P_HYD": total_pressure,
        "FNM": cache.array("coord/fnm"),
        "FNP": cache.array("coord/fnp"),
        "RDNW": cache.array("coord/rdnw"),
        "RDN": cache.array("coord/rdn"),
        "DNW": cache.array("coord/dnw"),
        "DN": cache.array("coord/dn"),
        "C1H": cache.array("coord/c1h"),
        "C2H": cache.array("coord/c2h"),
        "C1F": cache.array("coord/c1f"),
        "C2F": cache.array("coord/c2f"),
        "C3H": cache.array("coord/c3h"),
        "C4H": cache.array("coord/c4h"),
        "C3F": cache.array("coord/c3f"),
        "C4F": cache.array("coord/c4f"),
        **_real_exe_extrapolation_constants(
            cache.array("coord/dn"), cache.array("coord/dnw"),
            cache.array("coord/fnp"), cache.array("coord/fnm")),
        "Q2": cache.array("result/surface_qv"),
        "T2": t2,
        "TH2": t2 * (_P0 / total_pressure[0]) ** _RCP,
        "PSFC": psfc,
        "U10": 0.5 * (u10_face[:, :-1] + u10_face[:, 1:]),
        "V10": 0.5 * (v10_face[:-1, :] + v10_face[1:, :]),
        "RDX": np.asarray(1.0 / dx),
        "RDY": np.asarray(1.0 / dy),
        "GOT_VAR_SSO": np.asarray(0, dtype=np.int32),
        "QVAPOR": cache.array("state/qv"),
        "QCLOUD": cache.array("state/qc"),
        "QRAIN": cache.array("state/qr"),
        "QICE": cache.array("state/qi"),
        "SHDMAX": surface["SHDMAX"],
        "SHDMIN": surface["SHDMIN"],
        "SHDAVG": surface["SHDAVG"],
        "SNOALB": surface["SNOALB"],
        "LANDUSEF": static["LANDUSEF"],
        "SOILCTOP": static["SOILCTOP"],
        "SOILCBOT": static["SOILCBOT"],
        "TSLB": surface["TSLB"],
        "SMOIS": surface["SMOIS"],
        "SH2O": surface["SH2O"],
        "SEAICE": surface["SEAICE"],
        "IVGTYP": lu,
        "ISLTYP": soil,
        "VEGFRA": surface["VEGFRA"],
        "SNOW": snow,
        "SNOWH": snowh,
        "LAI": surface["LAI"],
        "MAPFAC_M": static["MAPFAC_M"],
        "MAPFAC_U": static["MAPFAC_U"],
        "MAPFAC_V": static["MAPFAC_V"],
        "MAPFAC_MX": static["MAPFAC_M"],
        "MAPFAC_MY": static["MAPFAC_M"],
        "MAPFAC_UX": static["MAPFAC_U"],
        "MAPFAC_UY": static["MAPFAC_U"],
        "MAPFAC_VX": static["MAPFAC_V"],
        "MAPFAC_VY": static["MAPFAC_V"],
        # real.exe's word: 1./msfvx in FP32 of the stored FP32 map factor
        # (module_initialize_real.F:580).  wrf.exe couples v with this array
        # (couple_momentum, calc_ww_cp, rk_addtend_dry, the emdiv filter);
        # the float64 reciprocal rounded once differed from it in 4,560 of
        # 12,120 UAE d01 words and moved every coupled v flux.
        "MF_VX_INV": np.float32(1.0) / np.asarray(static["MAPFAC_V"], dtype=np.float32),
        "F": static["F"],
        "E": static["E"],
        "SINALPHA": static["SINALPHA"],
        "COSALPHA": static["COSALPHA"],
        "HGT": cache.array("base/terrain_z"),
        "TSK": surface["TSK"],
        "P_TOP": np.asarray(p_top),
        "ALBBCK": surface["ALBBCK"],
        "TMN": surface["TMN"],
        "XLAND": surface["XLAND"],
        "SNOWC": (snow >= 10.0).astype(np.float64),
        "LANDMASK": landmask,
        "LAKEMASK": lake_mask.astype(np.float64),
        "SST": surface["TSK"],
    }
    # Source analyses generally do not carry scheme-specific extra moments.
    # WRF real.exe's Registry ``i0`` policy initializes absent package
    # members to zero.  Preserve any explicitly prepared scheme state and
    # otherwise materialize zeros so the stock-WRF file is complete and its
    # package choice is auditable.
    inventory = stock_wrf_physics_inventory(mp_physics)
    # QSNOW/QGRAUP used to be listed unconditionally beside QCLOUD above.
    # They are package members like any other, and mp_physics=50 declares
    # neither (Registry.EM_COMMON:3038), so a hard-coded read of
    # ``state/qs`` raised KeyError on a perfectly valid P3 cache.  The loop
    # below already handles every declared member -- prepared state when the
    # cache carries it, real.exe's zeros when it does not -- so the two
    # species simply join it.
    for field in inventory.wrfinput_fields:
        if field.netcdf_name in result:
            continue
        # The key is DECLARED, not guessed.  It used to be
        # f"state/{field.registry_name}", and woof drops WRF's leading q
        # on mp=28's aerosol rows and on the Thompson, Morrison and P3
        # number moments (nc/nr/ni/ns/ng), so every one of them missed the
        # cache and exported zeros -- including on a run whose WIF
        # climatology lane had filled nwfa/nifa with real values (R-054)
        # and on a Thompson cold start that had seeded nr/ni (A108).
        # ``state_key`` is None only for a member woof has no species for
        # (QNBCA), where zeros ARE what real.exe writes.
        state_name = field.state_key
        state_key = None if state_name is None else f"state/{state_name}"
        # Rank comes from the inventory too: mp=28's two emission members
        # are XY, and a 3-D zero array for them would not match the
        # contract's own 2-D declaration.
        shape = (qv.shape if len(field.dimensions) != len(WRFINPUT_2D_DIMS)
                 else qv.shape[-2:])
        result[field.netcdf_name] = (
            cache.array(state_key)
            if state_key is not None and state_key in cache._arrays
            else np.zeros(shape, dtype=np.float32)
        )
    result.update(_terrain_drag_export_fields(static, run_config or {}, (ny, nx)))
    result.update(_lake_export_fields(static, run_config or {}, lake_mask))
    return result


def _side_to_wrf(value: np.ndarray, logical: str, side: str) -> np.ndarray:
    """One gpuwm-oriented side table in WRF's ``(width, z, side)`` layout."""
    value = np.asarray(value)
    if side in {"west", "east"}:
        if logical == "mu":
            return value[0].T
        return np.transpose(value, (2, 0, 1))
    if logical == "mu":
        return value[0]
    return np.transpose(value, (1, 0, 2))


def _lbc_to_wrf(cache: PreparedCache, interval_index: int, logical: str,
                side: str, kind: str) -> np.ndarray:
    return _side_to_wrf(
        cache.array(f"lbc/{interval_index}/{logical}/{side}/{kind}"),
        logical, side)


#: The wrfbdy fields the export carries, WRF name to prepared logical name.
_WRFBDY_EXPORT_FIELDS = {
    "U": "u", "V": "v", "PH": "phi", "T": "theta",
    "MU": "mu", "QVAPOR": "qv",
}
_WRFBDY_SUFFIX_SIDES = {"XS": "west", "XE": "east",
                        "YS": "south", "YE": "north"}


def _wrfbdy_export_fields(cache: PreparedCache, inventory,
                          contract: Mapping[str, object]) -> dict[str, str]:
    """The wrfbdy fields this cache's export carries, WRF to logical name.

    The dynamics and water vapour always.  The Thompson-aerosol numbers
    (QNWFA, QNIFA) when the prepared boundary carries them: the run forced
    them at its lateral boundary, which is what WRF does under
    ``aer_init_opt > 0`` (``use_aero_icbc``), real.exe writing their
    coupled tables (main/real_em.F:888-899) and wrf.exe relaxing toward
    them (dyn_em/solve_em.F:2804-2806).  Written as zeros instead, a stock
    run relaxed its aerosol numbers toward none at every boundary, and the
    WRF-input door refused the pair as a mixed one.
    """
    from woof.boundary_fields import AEROSOL_BOUNDARY_FIELDS

    fields = dict(_WRFBDY_EXPORT_FIELDS)
    by_state = {field.state_key: field.netcdf_name
                for field in inventory.wrfinput_fields
                if field.state_key is not None}
    declared = {item["name"] for item in contract["variables"]}
    for logical in AEROSOL_BOUNDARY_FIELDS:
        if f"lbc/0/{logical}/west/value" not in cache._arrays:
            continue
        wrf_name = by_state.get(logical)
        if wrf_name is None or f"{wrf_name}_BXS" not in declared:
            raise ValueError(
                f"the prepared boundary forces {logical} but the "
                f"mp_physics={inventory.mp_physics} wrfbdy contract has no "
                f"table for it; the export cannot carry that boundary")
        fields[wrf_name] = logical
    return fields


def _initial_boundary_head(wrfinput_fields: Mapping[str, np.ndarray],
                           export_fields: Mapping[str, str]
                           ) -> dict[str, np.ndarray]:
    """real.exe's first wrfbdy record for the wrfinput this export writes.

    The single definition the WRF-input door checks against
    (:func:`woof.ingest.wrfinput.wrf_initial_boundary_head`), applied to
    the FP32 values the file will hold, under the export's own
    ``USE_THETA_M``.
    """
    from woof.ingest.wrfinput import wrf_initial_boundary_head

    layouts = {logical: (wrf_name,)
               for wrf_name, logical in export_fields.items()}
    names = {"MU", "MUB", "C1H", "C2H", "C1F", "C2F",
             "MAPFAC_M", "MAPFAC_U", "MAPFAC_V", *export_fields}
    raw = {name: np.asarray(wrfinput_fields[name], dtype=np.float32)
           for name in names}
    return wrf_initial_boundary_head(
        raw, layouts, moist_theta=EXPORT_USE_THETA_M == 1)


def _wrfbdy_fields(cache: PreparedCache, interval_index: int, *,
                   initial_head: Mapping[str, np.ndarray] | None = None,
                   duration_seconds: float | None = None,
                   export_fields: Mapping[str, str] | None = None,
                   ) -> dict[str, np.ndarray]:
    """One wrfbdy record from the prepared boundary tables.

    ``initial_head`` (record 0 only) replaces the first record's values with
    the coupling of the exported wrfinput, as real.exe's first record always
    is (main/real_em.F:866-899), and re-derives that record's tendencies so
    the end of the first interval stays the prepared one: WRFDA's
    da_update_bc law for a start state that is not the one the boundary was
    built from, ``BT' = (B + BT*dt - B') / dt``.  The prepared head is a
    different atmosphere whenever the start state was replaced after the
    boundary was built (a separate initial analysis, ``--initial-inputs``),
    and a pair whose first record is not its own wrfinput is one no WRF
    tool writes: the WRF-input door refuses it as a mixed pair.
    """
    if initial_head is not None and (
            interval_index != 0 or duration_seconds is None
            or not float(duration_seconds) > 0.0):
        raise ValueError(
            "the initial boundary head replaces record 0 only and needs "
            "that record's positive duration")
    result: dict[str, np.ndarray] = {}
    if export_fields is None:
        export_fields = _WRFBDY_EXPORT_FIELDS
    for wrf_name, logical in export_fields.items():
        for suffix, side in _WRFBDY_SUFFIX_SIDES.items():
            prefix = f"lbc/{interval_index}/{logical}/{side}"
            if initial_head is None:
                value = cache.array(f"{prefix}/value")
                tendency = cache.array(f"{prefix}/tendency")
            else:
                from woof.ingest.wrfinput import _boundary_strip

                prepared = np.asarray(cache.array(f"{prefix}/value"),
                                      dtype=np.float64)
                prepared_tendency = np.asarray(
                    cache.array(f"{prefix}/tendency"), dtype=np.float64)
                width = (prepared.shape[-1] if side in {"west", "east"}
                         else prepared.shape[-2])
                value = _boundary_strip(
                    initial_head[logical], side, width)
                if value.shape != prepared.shape:
                    raise ValueError(
                        f"exported initial {wrf_name} {side} strip "
                        f"{value.shape} != prepared boundary table "
                        f"{prepared.shape}")
                duration = float(duration_seconds)
                end = prepared + duration * prepared_tendency
                tendency = (end - value.astype(np.float64)) / duration
            result[f"{wrf_name}_B{suffix}"] = _side_to_wrf(
                value, logical, side)
            result[f"{wrf_name}_BT{suffix}"] = _side_to_wrf(
                tendency, logical, side)
    return result


def _write_wrfinput(path: Path, contract: Mapping[str, object],
                    dimensions: Mapping[str, int | None],
                    global_updates: Mapping[str, object],
                    fields: Mapping[str, np.ndarray], stamp: str,
                    engine: str | None = None) -> None:
    dataset = _create_dataset(path, contract, dimensions, global_updates,
                              engine)
    try:
        specs = {item["name"]: item for item in contract["variables"]}
        for name, variable in dataset.variables.items():
            if name == "Times":
                _write_timestamp(variable, stamp)
            elif name in fields:
                _write_time_value(variable, fields[name])
            elif "prototype_value" in specs[name]:
                _write_time_value(
                    variable,
                    _resolved_prototype_value(
                        variable, specs[name]["prototype_value"]),
                )
            else:
                shape = _shape_without_time(variable)
                _write_time_value(variable, np.zeros(shape, dtype=variable.dtype))
    except BaseException:
        # A half-written pair must not be left where the publication
        # step would carry it forward as a finished export.
        _abandon_dataset(dataset, path)
        raise
    dataset.close()


def _write_wrfbdy(path: Path, contract: Mapping[str, object],
                  dimensions: Mapping[str, int | None],
                  global_updates: Mapping[str, object],
                  cache: PreparedCache,
                  boundary_times: list[datetime],
                  boundary_interval_seconds: int,
                  engine: str | None = None, *,
                  initial_head: Mapping[str, np.ndarray] | None = None,
                  export_fields: Mapping[str, str] | None = None,
                  ) -> None:
    """``initial_head`` (:func:`_initial_boundary_head` of the wrfinput
    written beside this file) makes record 0 that file's own coupling;
    every export passes it, with the ``export_fields`` it was built for."""
    dataset = _create_dataset(path, contract, dimensions, global_updates,
                              engine)
    try:
        for time_index, boundary_time in enumerate(boundary_times):
            fields = (_wrfbdy_fields(cache, time_index,
                                     export_fields=export_fields)
                      if time_index != 0 or initial_head is None else
                      _wrfbdy_fields(
                          cache, 0, initial_head=initial_head,
                          duration_seconds=boundary_interval_seconds,
                          export_fields=export_fields))
            stamp = _date_text(boundary_time)
            next_stamp = _date_text(
                boundary_time
                + timedelta(seconds=boundary_interval_seconds))
            for name, variable in dataset.variables.items():
                if name in {
                    "Times",
                    "md___thisbdytimee_x_t_d_o_m_a_i_n_m_e_t_a_data_",
                }:
                    _write_timestamp_at(variable, stamp, time_index)
                elif name == (
                    "md___nextbdytimee_x_t_d_o_m_a_i_n_m_e_t_a_data_"
                ):
                    _write_timestamp_at(variable, next_stamp, time_index)
                elif name in fields:
                    _write_time_value_at(
                        variable, fields[name], time_index)
                else:
                    shape = _shape_without_time(variable)
                    _write_time_value_at(
                        variable, np.zeros(shape, dtype=variable.dtype),
                        time_index)
    except BaseException:
        _abandon_dataset(dataset, path)
        raise
    dataset.close()


def _attribute_matches(actual, expected, type_spec=None) -> bool:
    """Does the file's attribute still say what the export meant it to?

    ``type_spec`` is the frozen contract's declaration for this
    attribute, and it is what makes the comparison survive the Rust
    reader: ``rw_netcdf`` reports every numeric attribute as a JSON
    number, so an NC_FLOAT ``CEN_LAT`` comes back as the f64 widening of
    an f32 and would differ from the f64 the caller asked for in the last
    bits.  Restoring the OBSERVED value to the width the contract
    declares compares like with like -- and the contract, not the reader,
    is the thing that says what width the file carries.
    """

    if type_spec is not None:
        with suppress(TypeError, ValueError, OverflowError):
            actual = _restore_attribute(actual, type_spec)
            expected = _restore_attribute(expected, type_spec)
    if isinstance(actual, bytes):
        return actual == (expected if isinstance(expected, bytes)
                          else str(expected).encode("latin1"))
    if isinstance(actual, str):
        return actual == str(expected)
    actual_array = np.asarray(actual)
    try:
        expected_array = np.asarray(expected, dtype=actual_array.dtype)
    except (TypeError, ValueError, OverflowError):
        return False
    return (actual_array.shape == expected_array.shape
            and np.array_equal(actual_array, expected_array))


def _domain_global_attributes(updates: Mapping[str, object]) \
        -> dict[str, object]:
    names = [
        "GRID_ID", "PARENT_ID", "I_PARENT_START", "J_PARENT_START",
        "PARENT_GRID_RATIO", "WEST-EAST_GRID_DIMENSION",
        "SOUTH-NORTH_GRID_DIMENSION", "BOTTOM-TOP_GRID_DIMENSION",
        "DX", "DY", "DT", "CEN_LAT", "CEN_LON", "MOAD_CEN_LAT",
        "TRUELAT1", "TRUELAT2", "STAND_LON",
        # Read back from both files: a pair whose header and fields name
        # two representations is the defect EXPORT_USE_THETA_M closed.
        "USE_THETA_M",
    ]
    names.extend(
        name for name in (
            "MP_PHYSICS", "RA_LW_PHYSICS", "RA_SW_PHYSICS",
            "SF_SFCLAY_PHYSICS", "SF_SURFACE_PHYSICS",
            "BL_PBL_PHYSICS", "CU_PHYSICS",
            # Stated only when the static's land use is not the frozen
            # 21 categories (:func:`_land_category_contract`).
            "NUM_LAND_CAT", "ISLAKE",
        )
        if name in updates
    )
    missing = [name for name in names if name not in updates]
    if missing:
        raise ValueError(
            f"domain global-attribute expectation lacks {missing}")
    return {name: updates[name] for name in names}


@contextmanager
def _open_for_validation(path: Path, engine: str):
    """Open the written pair for its read-back, on the matching reader.

    The default engine's file is a classic container and is read by
    :mod:`woof.netcdf_bridge` -- ``rw_netcdf``, WOOF's Rust -- in a
    metadata-only pass that decodes no value.

    The netCDF4 workaround engine's file is an HDF5 container, and there
    the Rust reader recovers the variable list through HDF5 dimension
    scales rather than a definition-order table, so it cannot answer the
    inventory-ORDER check this validation makes (measured: it returns the
    same 183 names in a different order).  A workaround-written file is
    therefore checked by the library that wrote it, which is part of the
    workaround rather than a second default.
    """

    if engine == "rust":
        with netcdf_bridge.open_dataset(path) as dataset:
            yield dataset
    else:
        with netCDF4.Dataset(path) as dataset:
            yield dataset


def _nonfinite_variables(path: Path, contract: Mapping[str, object],
                         engine: str) -> list[str]:
    """Every float variable of the written file, swept for finiteness.

    This is the gate that stops a NaN reaching stock WRF, where it
    surfaces hours later as a blown-up integration with no trace of where
    it came from, and it is a READ-BACK: it asks the bytes on disk, not
    the arrays that were handed to the writer, so a short or corrupted
    write is caught as well.

    On the default engine it runs inside the writer crate
    (:func:`woof.io.nc_writer_bridge.scan_nonfinite`), one sequential
    pass over a file whose geometry that crate owns.  ``rw_netcdf`` is
    the front door for READING a NetCDF file, and it is the wrong shape
    for this one job: it decodes one variable per process launch through
    an f64 temp file, which for a 183-variable ``wrfinput`` would put 183
    process launches and a few gigabytes of temporary writes on the
    preparation path.

    On the netCDF4 workaround engine the file is an HDF5 container the
    crate does not read, so the sweep stays with the library that wrote
    it -- part of the workaround, not a separate default.
    """

    if engine == "rust":
        from woof.io import nc_writer_bridge

        return list(nc_writer_bridge.scan_nonfinite(path))
    nonfinite = []
    with netCDF4.Dataset(path) as dataset:
        for name, variable in dataset.variables.items():
            if np.dtype(variable.dtype).kind == "f":
                if not np.isfinite(variable[:]).all():
                    nonfinite.append(name)
    return nonfinite


def _validate_file(path: Path, contract: Mapping[str, object], *,
                   nx: int, ny: int, nz: int,
                   num_soil_layers: int = 4,
                   expected_global_attributes: Mapping[str, object]
                   | None = None,
                   engine: str | None = None,
                   bdy_width: int | None = None) -> dict[str, object]:
    """Read the published file back and check it against the contract.

    The header half -- global attributes, variable inventory and order,
    dimension extents -- comes through :mod:`woof.netcdf_bridge`, the
    Rust reader, which is a metadata-only pass and decodes no value.  The
    value half is :func:`_nonfinite_variables`.
    """

    engine = resolve_wrfinput_engine(engine)
    expected_dimensions = _dimensions(
        contract, nx=nx, ny=ny, nz=nz,
        num_soil_layers=num_soil_layers, bdy_width=bdy_width)
    attribute_types = contract["global_attribute_types"]
    with _open_for_validation(path, engine) as dataset:
        if expected_global_attributes is not None:
            missing_attributes = sorted(
                set(expected_global_attributes) - set(dataset.ncattrs()))
            if missing_attributes:
                raise ValueError(
                    f"{path.name}: missing global attributes "
                    f"{missing_attributes}")
            drift = {
                name: {
                    "actual": dataset.getncattr(name),
                    "expected": expected,
                }
                for name, expected in expected_global_attributes.items()
                if not _attribute_matches(
                    dataset.getncattr(name), expected,
                    attribute_types.get(name))
            }
            if drift:
                raise ValueError(
                    f"{path.name}: global attribute drift {drift}")
        if list(dataset.variables) != [item["name"] for item in contract["variables"]]:
            raise ValueError(f"{path.name}: variable inventory/order drift")
        for name, expected in expected_dimensions.items():
            actual = dataset.dimensions[name]
            if expected is not None and len(actual) != expected:
                raise ValueError(f"{path.name}: dimension {name} mismatch")
            if expected is None and not actual.isunlimited():
                raise ValueError(f"{path.name}: dimension {name} is not unlimited")
    nonfinite = _nonfinite_variables(path, contract, engine)
    if nonfinite:
        raise ValueError(f"{path.name}: non-finite variables {nonfinite}")
    return {"bytes": path.stat().st_size, "sha256": _sha256(path)}


def _forcing_offsets_from_identity(
        identity: Mapping[str, object],
) -> tuple[list[int], str]:
    offsets = identity.get("forcing_offsets_seconds")
    if offsets is not None:
        key = "forcing_offsets_seconds"
        factor = 1
    else:
        offsets = identity.get("forcing_hours")
        key = "forcing_hours"
        factor = 3600
    if (not isinstance(offsets, list) or len(offsets) < 2
            or any(isinstance(value, bool) or not isinstance(value, int)
                   for value in offsets)):
        raise ValueError(
            "direct exporter requires at least two integer forcing offsets")
    seconds = [value * factor for value in offsets]
    if seconds[0] != 0 or any(
            later <= earlier
            for earlier, later in zip(seconds, seconds[1:])):
        raise ValueError(
            "direct exporter requires increasing forcing offsets beginning "
            "at zero")
    return seconds, key


def _forcing_interval_indices(cache: PreparedCache,
                              boundary_interval_seconds: int) -> list[int]:
    identity = cache.header["identity"]
    forcing_offsets, forcing_key = _forcing_offsets_from_identity(identity)
    expected_step_seconds = {
        later - earlier
        for earlier, later in zip(forcing_offsets, forcing_offsets[1:])
    }
    if expected_step_seconds != {boundary_interval_seconds}:
        raise ValueError(
            "prepared forcing cadence does not match the requested boundary "
            f"interval: {forcing_key}={identity[forcing_key]}, "
            f"boundary_interval_seconds={boundary_interval_seconds}")

    observed_indices = set()
    for name in cache.header["arrays"]:
        parts = name.split("/")
        if len(parts) >= 2 and parts[0] == "lbc":
            try:
                observed_indices.add(int(parts[1]))
            except ValueError as error:
                raise ValueError(
                    f"invalid prepared LBC interval key {name!r}") from error
    expected_indices = list(range(len(forcing_offsets) - 1))
    if sorted(observed_indices) != expected_indices:
        raise ValueError(
            "prepared LBC interval inventory does not span every forcing pair: "
            f"expected {expected_indices}, got {sorted(observed_indices)}")
    return expected_indices


def _validated_hierarchy(exp, artifacts: Sequence[PreparedDomainArtifacts]):
    domains = tuple(exp.domains)
    if not domains:
        raise ValueError("hierarchy export requires at least one domain")
    if len(domains) > 99:
        raise ValueError("hierarchy export supports at most 99 WRF domains")
    from woof.wps_domain_ids import validated_domain_ids
    expected_ids = list(validated_domain_ids([domain.grid_id for domain in domains]))
    if exp.projection is None or exp.projection.map_proj not in (
            "lambert", "mercator", "polar"):
        raise ValueError(
            "hierarchy direct export requires a lambert, mercator, or "
            "polar [projection] table")
    declared = {}
    for artifact in artifacts:
        if artifact.grid_id in declared:
            raise ValueError(
                f"duplicate artifacts for grid_id={artifact.grid_id}")
        declared[artifact.grid_id] = artifact
    if sorted(declared) != sorted(expected_ids):
        raise ValueError(
            "domain-artifact ids must exactly cover the WRF hierarchy: "
            f"expected {expected_ids}, got {sorted(declared)}")
    seen = set()
    pairs = []
    for domain in domains:
        root = domain.grid_id == 1
        if root:
            if (domain.parent_id != 0 or not domain.run.specified
                    or domain.run.nested):
                raise ValueError(
                    "d01 must be specified=true, nested=false, parent_id=0")
        elif (domain.parent_id not in seen or domain.run.specified
              or not domain.run.nested):
            raise ValueError(
                f"d{domain.grid_id:02d} must name an earlier parent and use "
                "specified=false, nested=true")
        seen.add(domain.grid_id)
        pairs.append((domain, declared[domain.grid_id]))
    return tuple(pairs)


def validate_stock_wrf_export_config(
        config, *, root: bool = True, configured_suite: bool = False,
        label: str = "direct-export"):
    """The exporter's selector/geometry admission, available before ingest.

    Prepared arrays still receive every existing identity, shape and value
    check when exported. This only answers configuration representability.
    Historical profile-free export keeps its original fixed physics slice.
    """
    from dataclasses import asdict, is_dataclass

    cfg = asdict(config) if is_dataclass(config) else dict(config)
    try:
        inventory = stock_wrf_physics_inventory(cfg.get("mp_physics"))
    except (TypeError, ValueError) as error:
        raise StockWrfExportUnsupported(
            f"unsupported {label} microphysics: {error}",
            unsupported={"mp_physics": (cfg.get("mp_physics"), None)},
        ) from None
    # spec_bdy_width is NOT pinned (U-E6).  It was required to be 5, the
    # width of the frozen v4.6.1 file the contract was read from, so every
    # HRRR-recipe preparation (spec_bdy_width = 10, spec_zone 1 +
    # relax_zone 9, as the operational HRRR namelist) had its companion
    # export refused.  The wrfbdy strip width is just the bdy_width
    # dimension: real.exe sizes it from spec_bdy_width, and the export
    # sizes it from the prepared boundary tables (_prepared_bdy_width).
    required = {"hybrid_opt": 2, "hypsometric_opt": 2,
                "specified": root, "nested": not root}
    if not configured_suite:
        required.update({"bl_pbl_physics": 1, "sf_sfclay_physics": 91,
                         "sf_surface_physics": 2})
    mismatch = {name: (cfg.get(name), expected)
                for name, expected in required.items() if cfg.get(name) != expected}
    width = cfg.get("spec_bdy_width")
    zones = (cfg.get("spec_zone", 1), cfg.get("relax_zone", 4))
    if (isinstance(width, bool) or not isinstance(width, int) or width < 1
            or not all(isinstance(zone, int) for zone in zones)
            or width < zones[0] + zones[1]):
        # The strip must hold the specified and relaxation zones, the
        # rule woof.config.validate_run_config applies to every run.
        mismatch["spec_bdy_width"] = (
            width, f">= spec_zone + relax_zone = {zones[0]} + {zones[1]}")
    if mismatch:
        raise StockWrfExportUnsupported(
            f"unsupported {label} configuration: {mismatch}", unsupported=mismatch)
    return inventory


def _prepared_bdy_width(cache: PreparedCache, cfg: Mapping[str, object],
                        interval_index: int) -> int:
    """The wrfbdy strip width: the prepared tables', which must be the run's.

    One width, read off the first boundary record's theta west strip
    (woof side layout ``(z, y, width)``), and refused if it is not the
    ``spec_bdy_width`` the cache's configuration records: a stock WRF
    namelist states that number, and a file sized from anything else is
    one wrf.exe would read with the wrong strip.
    """
    prefix = f"lbc/{interval_index}/theta/west/value"
    shape = cache._arrays[prefix]["shape"]
    width = int(shape[-1])
    if width != int(cfg["spec_bdy_width"]):
        raise ValueError(
            f"prepared boundary tables are {width} rows wide but the cache "
            f"records spec_bdy_width = {cfg['spec_bdy_width']}")
    return width


def _prepared_domain_context(artifact: PreparedDomainArtifacts, domain,
                             exp, expected_grid, valid_time: datetime):
    cache = PreparedCache(artifact.prepared_cache)
    identity = cache.header["identity"]
    expected_domain = prepared_domain_config_identity(domain)
    # Default-tolerant on fields that postdate the header, strict on
    # everything else: a package upgrade that adds an unused identity
    # field must not make an existing prepared bundle unreadable, and a
    # field carrying a real value must still bind.
    _, differing = compare_prepared_domain_config(
        identity.get("domain_config"), expected_domain,
        not_in_use=undelayed_identity_defaults(exp))
    if differing:
        raise ValueError(prepared_identity_refusal(
            subject=f"d{domain.grid_id:02d} prepared cache",
            header=cache.header, differing=differing))
    cfg = identity["domain_config"]["run"]
    validate_stock_wrf_export_config(
        cfg, root=domain.grid_id == 1, label=f"d{domain.grid_id:02d} direct-export")
    prepared_valid_time = datetime.fromisoformat(
        cache.header["metadata"]["user"]["initial_valid_time"])
    if _date_text(prepared_valid_time) != _date_text(valid_time):
        raise ValueError(
            f"d{domain.grid_id:02d} requested valid time does not match "
            "the prepared cache")

    nx, ny, nz = int(cfg["nx"]), int(cfg["ny"]), int(cfg["nz"])
    from woof.native_wrf_contract import native_geometry_contract

    expected_geometry = native_geometry_contract(expected_grid, domain.run)
    geometry, static_sha256 = _load_static_geometry_receipt(
        artifact.geometry_receipt, artifact.static_cache,
        expected_geometry=expected_geometry, expected_grid=expected_grid)
    if list(geometry["mass_shape"]) != [ny, nx] or int(geometry["nz"]) != nz:
        raise ValueError(
            f"d{domain.grid_id:02d} geometry receipt does not match cache")
    for name, geometry_name in (("dx", "dx_m"), ("dy", "dy_m")):
        if not math.isclose(
                float(geometry[geometry_name]), float(cfg[name]),
                rel_tol=1e-12):
            raise ValueError(
                f"d{domain.grid_id:02d} geometry/cache {name} mismatch")
    p_top = _prepared_vertical_contract(
        cache, nx=nx, ny=ny, nz=nz, mp_physics=cfg.get("mp_physics"))
    if p_top != float(exp.vertical.p_top):
        raise ValueError(
            f"d{domain.grid_id:02d} prepared p_top differs from namelist")
    expected_eta = np.asarray(
        exp.vertical.eta_levels, dtype=cache.array("coord/znw").dtype)
    if not np.array_equal(cache.array("coord/znw"), expected_eta):
        raise ValueError(
            f"d{domain.grid_id:02d} prepared eta levels differ from namelist")
    if static_sha256 != identity.get("static_cache_sha256"):
        raise ValueError(
            f"d{domain.grid_id:02d} static cache digest differs from "
            "prepared identity")
    return cache, cfg, geometry, p_top, static_sha256


def _configured_domain_start(exp, domain) -> datetime:
    if hasattr(exp, "domain_start_time"):
        return exp.domain_start_time(domain.grid_id)
    configured = getattr(domain, "start_time", None)
    return exp.start_time if configured is None else configured


def _hierarchy_global_updates(*, valid_time: datetime, cfg,
                              geometry: Mapping[str, object], domain):
    # Each nest states its OWN recorded physics: a child's cumulus is not
    # its parent's, and a stock WRF nest reads its schemes from its own
    # wrfinput's attributes.
    updates = _global_updates(
        valid_time=valid_time,
        nx=int(cfg["nx"]), ny=int(cfg["ny"]), nz=int(cfg["nz"]),
        dx=float(cfg["dx"]), dy=float(cfg["dy"]), dt=float(cfg["dt"]),
        geometry=geometry,
        physics_selection={"selectors": recorded_physics_selectors(
            cfg, label=f"d{domain.grid_id:02d} direct-export")},
        run_config=cfg,
    )
    updates.update({
        "GRID_ID": domain.grid_id,
        "PARENT_ID": domain.parent_id,
        "I_PARENT_START": domain.i_parent_start,
        "J_PARENT_START": domain.j_parent_start,
        "PARENT_GRID_RATIO": domain.parent_grid_ratio,
    })
    return updates


def validate_stock_wrf_export_hierarchy(exp):
    """Configuration-only hierarchy rule shared by early prep and export."""
    # Native prepared trees hold the unperturbed source states. Their
    # forecast runner applies bubbles after restoration, a step unchanged
    # stock WRF cannot recover from these files or their namelist.
    if getattr(exp, "perturbation", None) is not None:
        raise StockWrfExportUnsupported(
            "initial-state [perturbation] is deferred to the native prepared "
            "domain-tree forecast runner and is not stock-WRF exportable; "
            "the prepared arrays do not contain the configured bubbles",
            unsupported={"perturbation": "deferred native initial-state bubbles"})
    # Stock WRF v4.6.1 normalizes microphysics to one selector across the
    # hierarchy.  GPUWM's explicit MP8->MP18 edge is executable only in its
    # own one-way coupler and must never be mislabeled as a READY stock-WRF
    # export or silently normalized here.
    from woof.core.microphysics_transition import (
        resolve_microphysics_transition,
    )
    by_id = {domain.grid_id: domain for domain in exp.domains}
    for domain in exp.domains:
        if domain.parent_id == 0:
            continue
        transition = resolve_microphysics_transition(
            by_id[domain.parent_id].run, domain.run)
        if transition.mixed:
            raise StockWrfExportUnsupported(
                "mixed-domain microphysics is a GPUWM extension and is not "
                "stock-WRF exportable; choose one uniform stock-WRF "
                "mp_physics selector explicitly",
                unsupported={"mp_physics": (
                    int(domain.run.mp_physics),
                    int(by_id[domain.parent_id].run.mp_physics))})



def export_prepared_wrf_hierarchy(
        exp, domain_artifacts: Sequence[PreparedDomainArtifacts], output_dir,
        *, valid_time: datetime | None = None,
        boundary_interval_seconds: int = 3600,
        overwrite: bool = False,
        input_provenance: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Atomically export ``wrfinput_d01..dNN`` plus root ``wrfbdy_d01``.

    Every final initial-condition file is generated from its own prepared
    meteorology/static artifact set, allowing those expensive upstream
    preparations to run independently.  Only d01 receives external lateral
    boundaries; children are initialized as nested WRF domains and are forced
    by their declared parents at runtime.
    """

    # One writer decision governs the whole pair, resolved once here
    # rather than re-read per file: a hierarchy whose d01 was classic and
    # whose d03 was HDF5 because the environment moved mid-export is a
    # file set nobody declared.
    engine = resolve_wrfinput_engine()

    validate_stock_wrf_export_hierarchy(exp)

    pairs = _validated_hierarchy(exp, tuple(domain_artifacts))
    expected_grids = tuple(grids_from_projection_config(exp))
    if len(expected_grids) != len(pairs):
        raise ValueError(
            "namelist hierarchy did not resolve one grid per domain")
    if valid_time is None:
        valid_time = exp.start_time
    if _date_text(valid_time) != _date_text(exp.start_time):
        raise ValueError("export valid time differs from the namelist start time")
    if boundary_interval_seconds <= 0:
        raise ValueError("boundary interval must be positive")
    output = Path(output_dir)
    if output.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {output}")
    staging = export_staging_path(output)
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    contract_bundle = _load_contract()
    files = {}
    sources = {}
    root_manifest = None
    try:
        root_domain, root_artifact = pairs[0]
        root_output = staging / ROOT_EXPORT_DIRNAME
        root_manifest = export_prepared_wrf(
            root_artifact.prepared_cache, root_artifact.static_cache,
            root_artifact.geometry_receipt, root_output,
            valid_time=valid_time,
            boundary_interval_seconds=boundary_interval_seconds,
            overwrite=False,
        )
        for name in ("wrfinput_d01", "wrfbdy_d01"):
            os.replace(root_output / name, staging / name)
            files[name] = root_manifest["files"][name]
        shutil.rmtree(root_output)

        for (domain, artifact), expected_grid in zip(pairs, expected_grids):
            domain_valid_time = _configured_domain_start(exp, domain)
            cache, cfg, geometry, p_top, static_sha256 = \
                _prepared_domain_context(
                    artifact, domain, exp, expected_grid, domain_valid_time)
            physics_inventory = stock_wrf_physics_inventory(
                int(cfg["mp_physics"]))
            domain_contract_bundle = _physics_contract_bundle(
                contract_bundle, physics_inventory.mp_physics, cfg)
            input_contract = domain_contract_bundle["wrfinput"]
            name = f"wrfinput_d{domain.grid_id:02d}"
            if domain.grid_id != 1:
                updates = _hierarchy_global_updates(
                    valid_time=domain_valid_time, cfg=cfg, geometry=geometry,
                    domain=domain)
                with np.load(artifact.static_cache, allow_pickle=False) as static:
                    fields = _wrfinput_fields(
                        cache, static, geometry, domain_valid_time, p_top=p_top,
                        mp_physics=physics_inventory.mp_physics,
                        run_config=cfg)
                    input_contract, land_attributes = \
                        _land_category_contract(input_contract, fields)
                    updates.update(land_attributes)
                    dimensions = _dimensions(
                        input_contract, nx=int(cfg["nx"]), ny=int(cfg["ny"]),
                        nz=int(cfg["nz"]))
                    _write_wrfinput(
                        staging / name, input_contract, dimensions,
                        updates, fields, _date_text(domain_valid_time),
                        engine)
                files[name] = _validate_file(
                    staging / name, input_contract,
                    nx=int(cfg["nx"]), ny=int(cfg["ny"]), nz=int(cfg["nz"]),
                    expected_global_attributes=_domain_global_attributes(
                        updates),
                    engine=engine)
            sources[f"d{domain.grid_id:02d}"] = {
                "prepared_header_sha256": _sha256(cache.header_path),
                "prepared_content_sha256": cache.header["content_sha256"],
                "static_cache_sha256": static_sha256,
                "geometry_receipt_sha256": _sha256(
                    artifact.geometry_receipt),
                "mp_physics": physics_inventory.mp_physics,
                "microphysics": physics_inventory.scheme,
                "start_time": _date_text(domain_valid_time),
                # The package's contract, not the drag-extended one the
                # file was written to (resolved_physics_contract_sha256).
                "resolved_physics_contract_sha256":
                    resolved_physics_contract_sha256(
                        physics_inventory.mp_physics),
            }

        hierarchy = [{
            "grid_id": domain.grid_id,
            "parent_id": domain.parent_id,
            "i_parent_start": domain.i_parent_start,
            "j_parent_start": domain.j_parent_start,
            "parent_grid_ratio": domain.parent_grid_ratio,
            "parent_time_step_ratio": domain.parent_time_step_ratio,
            "nx": domain.run.nx,
            "ny": domain.run.ny,
            "nz": domain.run.nz,
            "dx_m": domain.run.dx,
            "dy_m": domain.run.dy,
            "dt_s": domain.run.dt,
            "mp_physics": domain.run.mp_physics,
            "start_time": _date_text(_configured_domain_start(exp, domain)),
        } for domain, _artifact in pairs]
        forcing_key = (
            "forcing_offsets_seconds"
            if "forcing_offsets_seconds" in root_manifest
            else "forcing_hours")
        manifest = {
            "schema": HIERARCHY_EXPORT_SCHEMA,
            "status": "READY",
            "valid_time": _date_text(valid_time),
            # What a stock namelist must say to run these files, for a
            # consumer that reads only the manifest (EXPORT_USE_THETA_M).
            "use_theta_m": EXPORT_USE_THETA_M,
            "boundary_interval_seconds": boundary_interval_seconds,
            "boundary_record_count": root_manifest["boundary_record_count"],
            "boundary_times": root_manifest["boundary_times"],
            "next_boundary_times": root_manifest["next_boundary_times"],
            forcing_key: root_manifest[forcing_key],
            "hierarchy": hierarchy,
            "vertical": {
                "e_vert": len(exp.vertical.eta_levels),
                "p_top_pa": exp.vertical.p_top,
                "hybrid_opt": exp.vertical.hybrid_opt,
                "eta_sha256": _array_sha256(np.asarray(
                    exp.vertical.eta_levels, dtype=np.float64)),
            },
            "source": {
                "domains": sources,
                "contract_sha256": _sha256(_CONTRACT_PATH),
                "input_provenance": dict(input_provenance or {}),
            },
            "files": files,
            "limitations": [
                "static, one-way nested projected hierarchy only (lambert stock-WRF-gated; mercator/polar exports oracle-gated, not yet stock-WRF-gated)",
                "all domains must use one shared explicit eta coordinate",
                "WRF-v4.6.1 Registry-inventoried "
                "WSM6/Thompson/Morrison/NSSL-2 "
                "microphysics with YSU+classic-MM5+Noah only",
                "T_INIT is bound to final dry theta and is diagnostic-only",
                "hydrometeor/W/HT_SHAD/PC external root LBC arrays are zero",
                "root forcing must be uniformly spaced and begin at f00",
            ],
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False)
            + "\n", encoding="utf-8")
        with _output_publication(output, overwrite=overwrite) as backup:
            _publish_staging(staging, output, backup)
        return manifest
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def export_prepared_wrf_namelists(
        wps_namelist: str | Path, namelist_input: str | Path,
        domain_artifacts_manifest: str | Path, output_dir, *,
        valid_time: datetime | None = None,
        boundary_interval_seconds: int = 3600,
        overwrite: bool = False,
) -> dict[str, object]:
    """Namelist-driven hierarchy entry point for the native final exporter."""

    import tomllib

    from woof.experiment import build_experiment_from_config_tables
    from woof.namelist_import import import_namelists
    from woof.native_wrf_contract import validate_native_lambert_contracts

    wps_path = Path(wps_namelist)
    input_path = Path(namelist_input)
    artifacts_path = Path(domain_artifacts_manifest)
    resolved_text, report = import_namelists(wps_path, input_path)
    # A WUDAPT geog_data_res imports with a [static] companion; the
    # exported statics are the prepared artifacts', so it is validated and
    # split off here, as every file door does.
    exp = build_experiment_from_config_tables(
        tomllib.loads(resolved_text),
        source=f"native export of {wps_path.name} + {input_path.name}",
        base_dir=wps_path.parent)
    validate_native_lambert_contracts(
        exp, wps_path, source_name="native hierarchy")
    provenance = {
        "wps_namelist_sha256": _sha256(wps_path),
        "namelist_input_sha256": _sha256(input_path),
        "domain_artifacts_manifest_sha256": _sha256(artifacts_path),
        "resolved_experiment_sha256": hashlib.sha256(
            resolved_text.encode("utf-8")).hexdigest(),
        "translation_report": asdict(report),
    }
    return export_prepared_wrf_hierarchy(
        exp, load_domain_artifacts_manifest(artifacts_path), output_dir,
        valid_time=valid_time,
        boundary_interval_seconds=boundary_interval_seconds,
        overwrite=overwrite,
        input_provenance=provenance,
    )


def export_prepared_wrf(prepared_cache, static_cache, geometry_receipt,
                        output_dir, *, valid_time: datetime,
                        boundary_interval_seconds: int = 3600,
                        overwrite: bool = False,
                        physics_profile: str | None = None,
                        expert_acknowledgements: Sequence[str] = (),
                        acknowledgement_provenance: Mapping[
                            str, object] | None = None,
                        experiment_config_suite: bool = False,
                        ) -> dict[str, object]:
    """Export wrfinput/wrfbdy under one of three physics contracts.

    - ``physics_profile`` named: the exporter re-validates that the
      cache-bound config IS that shipped suite and the v3 manifest
      carries the named-profile selection receipt.
    - ``experiment_config_suite=True`` (owner ruling 2026-07-31, the
      profileless preparation contract): the exporter recomputes the
      same source-neutral per-domain selection receipt the front door
      computed -- from its OWN cache-bound config, never from a caller
      claim -- and the v3 manifest carries it, so the caller's byte
      equality check proves the prepared physics came from this config
      under this registry.  Expert tuples retain their registry-owned
      acknowledgement here too.
    - Neither: the historical v2 contract, pinned to the stock
      YSU+classic-MM5+Noah slice with any inventoried microphysics; see
      test_the_profile_free_export_gate_keeps_its_exact_v2_stock_slice.
      Its global physics attributes are the cache's own recorded ones
      (:func:`recorded_physics_selectors`), so an mp_physics=8 cache
      states MP_PHYSICS = 8.
    """

    cache = PreparedCache(Path(prepared_cache))
    static_path = Path(static_cache)
    geometry_path = Path(geometry_receipt)
    output = Path(output_dir)
    if output.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {output}")
    if boundary_interval_seconds <= 0:
        raise ValueError("boundary interval must be positive")

    identity = cache.header["identity"]
    cfg = identity["domain_config"]["run"]
    physics_selection = None
    if physics_profile is not None:
        from woof.physics_compat import (
            validate_single_domain_physics_profile,
        )

        physics_selection = validate_single_domain_physics_profile(
            physics_profile, config=cfg,
            expert_acknowledgements=tuple(expert_acknowledgements),
            acknowledgement_provenance=acknowledgement_provenance)
    elif experiment_config_suite:
        from woof.physics_compat import single_domain_physics_selection

        physics_selection = single_domain_physics_selection(
            cfg,
            expert_acknowledgements=tuple(expert_acknowledgements),
            acknowledgement_provenance=acknowledgement_provenance)
    physics_inventory = validate_stock_wrf_export_config(
        cfg, configured_suite=physics_selection is not None)
    contract_bundle = _physics_contract_bundle(
        _load_contract(), physics_inventory.mp_physics, cfg)
    interval_indices = _forcing_interval_indices(
        cache, boundary_interval_seconds)
    _forcing_offsets, forcing_key = _forcing_offsets_from_identity(identity)
    prepared_valid_time = datetime.fromisoformat(
        cache.header["metadata"]["user"]["initial_valid_time"])
    if _date_text(prepared_valid_time) != _date_text(valid_time):
        raise ValueError("requested valid time does not match prepared cache")

    geometry, actual_static_sha256 = _load_static_geometry_receipt(
        geometry_path, static_path)
    nx, ny, nz = int(cfg["nx"]), int(cfg["ny"]), int(cfg["nz"])
    if list(geometry["mass_shape"]) != [ny, nx] or int(geometry["nz"]) != nz:
        raise ValueError("geometry receipt does not match prepared cache")
    if not math.isclose(float(geometry["dx_m"]), float(cfg["dx"]), rel_tol=1e-12):
        raise ValueError("geometry/cache dx mismatch")
    if not math.isclose(float(geometry["dy_m"]), float(cfg["dy"]), rel_tol=1e-12):
        raise ValueError("geometry/cache dy mismatch")
    p_top = _prepared_vertical_contract(
        cache, nx=nx, ny=ny, nz=nz, mp_physics=cfg.get("mp_physics"))
    expected_static_sha256 = identity.get("static_cache_sha256")
    if actual_static_sha256 != expected_static_sha256:
        raise ValueError("static cache digest does not match prepared identity")

    staging = export_staging_path(output)
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    stamp = _date_text(valid_time)
    boundary_times = [
        valid_time + timedelta(
            seconds=interval_index * boundary_interval_seconds)
        for interval_index in interval_indices
    ]
    boundary_stamps = [_date_text(value) for value in boundary_times]
    next_boundary_stamps = [
        _date_text(value + timedelta(seconds=boundary_interval_seconds))
        for value in boundary_times
    ]
    # The suite receipt is per-domain; this exporter writes exactly d01,
    # so the global attributes are stamped from that domain's selector
    # record.  A named-profile receipt already carries flat selectors.
    # With no selection the cache's own recorded physics is stamped, never
    # the frozen contract's WSM6 + Dudhia values.
    selection_selectors = physics_selection
    if physics_selection is not None and "domains" in physics_selection:
        selection_selectors = physics_selection["domains"]["1"]
    if selection_selectors is None:
        selection_selectors = {"selectors": recorded_physics_selectors(cfg)}
    updates = _global_updates(
        valid_time=valid_time, nx=nx, ny=ny, nz=nz,
        dx=float(cfg["dx"]), dy=float(cfg["dy"]), dt=float(cfg["dt"]),
        geometry=geometry,
        physics_selection=selection_selectors,
        run_config=cfg,
    )
    num_soil_layers = int(cfg.get("num_soil_layers", 4))
    sf_surface_physics = int(cfg["sf_surface_physics"])
    # One writer decision for wrfinput and wrfbdy alike: a pair with two
    # containers in it is a file set nobody declared.
    engine = resolve_wrfinput_engine()
    try:
        with np.load(static_path, allow_pickle=False) as static:
            wrfinput_fields = _wrfinput_fields(
                cache, static, geometry, valid_time, p_top=p_top,
                mp_physics=physics_inventory.mp_physics,
                sf_surface_physics=sf_surface_physics,
                num_soil_layers=num_soil_layers, run_config=cfg)
            input_contract, land_attributes = _land_category_contract(
                contract_bundle["wrfinput"], wrfinput_fields)
            updates.update(land_attributes)
            input_dimensions = _dimensions(
                input_contract, nx=nx, ny=ny, nz=nz,
                num_soil_layers=num_soil_layers)
            _write_wrfinput(
                staging / "wrfinput_d01", input_contract, input_dimensions,
                updates, wrfinput_fields, stamp, engine)
        bdy_contract = contract_bundle["wrfbdy"]
        bdy_width = _prepared_bdy_width(cache, cfg, interval_indices[0])
        bdy_dimensions = _dimensions(
            bdy_contract, nx=nx, ny=ny, nz=nz,
            num_soil_layers=num_soil_layers, bdy_width=bdy_width)
        export_fields = _wrfbdy_export_fields(
            cache, physics_inventory, bdy_contract)
        _write_wrfbdy(
            staging / "wrfbdy_d01", bdy_contract, bdy_dimensions,
            updates, cache, boundary_times, boundary_interval_seconds,
            engine, export_fields=export_fields,
            initial_head=_initial_boundary_head(
                wrfinput_fields, export_fields))
        files = {
            "wrfinput_d01": _validate_file(
                staging / "wrfinput_d01", input_contract,
                nx=nx, ny=ny, nz=nz,
                num_soil_layers=num_soil_layers,
                expected_global_attributes=_domain_global_attributes(
                    updates),
                engine=engine),
            "wrfbdy_d01": _validate_file(
                staging / "wrfbdy_d01", bdy_contract,
                nx=nx, ny=ny, nz=nz,
                num_soil_layers=num_soil_layers,
                expected_global_attributes=_domain_global_attributes(
                    updates),
                engine=engine, bdy_width=bdy_width),
        }
        limitations = [
            "single specified projected domain only (lambert stock-WRF-gated; mercator/polar exports oracle-gated, not yet stock-WRF-gated)",
            (
                f"{physics_inventory.scheme}+YSU+classic-MM5+Noah "
                "initialized state contract"
                if physics_selection is None else
                (f"{physics_selection['profile']} initialized state contract"
                 if physics_selection.get("profile") is not None else
                 "hash-bound experiment-config suite initialized state "
                 "contract")
            ),
            "T_INIT is bound to final dry theta and is diagnostic-only",
            "hydrometeor/W/HT_SHAD/PC external LBC arrays are zero",
            "forcing must be uniformly spaced and begin at f00",
        ]
        if not any(
                name.startswith("surface/") for name in cache._arrays):
            limitations.insert(2, "warm-soil SH2O initialization only")
        manifest = {
            "schema": (
                "gpuwm-native-direct-wrf-export-v2"
                if physics_selection is None
                else "gpuwm-native-direct-wrf-export-v3"
            ),
            "status": "READY",
            "valid_time": stamp,
            # What a stock namelist must say to run these files, for a
            # consumer that reads only the manifest (EXPORT_USE_THETA_M).
            "use_theta_m": EXPORT_USE_THETA_M,
            "next_boundary_time": next_boundary_stamps[0],
            "boundary_interval_seconds": boundary_interval_seconds,
            "boundary_record_count": len(boundary_times),
            "boundary_times": boundary_stamps,
            "next_boundary_times": next_boundary_stamps,
            forcing_key: identity[forcing_key],
            "dimensions": {"nx": nx, "ny": ny, "nz": nz},
            "source": {
                "prepared_header_sha256": _sha256(cache.header_path),
                "prepared_content_sha256": cache.header["content_sha256"],
                "static_cache_sha256": actual_static_sha256,
                "geometry_receipt_sha256": _sha256(geometry_path),
                "contract_sha256": _sha256(_CONTRACT_PATH),
                # The package's contract, not the drag-extended one the
                # files were written to (resolved_physics_contract_sha256).
                "resolved_physics_contract_sha256":
                    resolved_physics_contract_sha256(
                        physics_inventory.mp_physics),
            },
            "files": files,
            "limitations": limitations,
        }
        if physics_selection is not None:
            manifest["physics"] = physics_selection
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        with _output_publication(output, overwrite=overwrite) as backup:
            _publish_staging(staging, output, backup)
        return manifest
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-cache", type=Path)
    parser.add_argument("--static-cache", type=Path)
    parser.add_argument("--geometry-receipt", type=Path)
    parser.add_argument(
        "--domain-artifacts", type=Path,
        help="gpuwm-native-domain-artifacts-v1 JSON for d01..dNN",
    )
    parser.add_argument("--wps-namelist", type=Path)
    parser.add_argument("--namelist-input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--valid-time", help=(
            "UTC YYYY-MM-DD_HH:MM:SS; hierarchy mode defaults to and "
            "cross-checks the namelist start time"))
    parser.add_argument("--boundary-interval-seconds", type=int, default=3600)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--physics-profile",
        help="registered fixed physics template carried by the prepared cache")
    parser.add_argument(
        "--experiment-config-suite", action="store_true",
        help=("no named profile: recompute and carry the cache-bound "
              "config's own per-domain physics selection receipt "
              "(v3 manifest), instead of the legacy v2 stock slice"))
    parser.add_argument(
        "--ack", action="append", default=[],
        help="registry-owned expert acknowledgement id; repeat as needed")
    args = parser.parse_args()
    valid_time = (datetime.strptime(args.valid_time, "%Y-%m-%d_%H:%M:%S")
                  if args.valid_time is not None else None)
    if args.domain_artifacts is not None:
        missing = [name for name, value in (
            ("--wps-namelist", args.wps_namelist),
            ("--namelist-input", args.namelist_input),
        ) if value is None]
        incompatible = [name for name, value in (
            ("--prepared-cache", args.prepared_cache),
            ("--static-cache", args.static_cache),
            ("--geometry-receipt", args.geometry_receipt),
            ("--physics-profile", args.physics_profile),
            ("--experiment-config-suite",
             args.experiment_config_suite or None),
            ("--ack", args.ack or None),
        ) if value is not None]
        if missing or incompatible:
            parser.error(
                "hierarchy mode missing/incompatible options: "
                f"missing={missing}, incompatible={incompatible}")
        manifest = export_prepared_wrf_namelists(
            args.wps_namelist, args.namelist_input, args.domain_artifacts,
            args.output, valid_time=valid_time,
            boundary_interval_seconds=args.boundary_interval_seconds,
            overwrite=args.overwrite)
    else:
        missing = [name for name, value in (
            ("--prepared-cache", args.prepared_cache),
            ("--static-cache", args.static_cache),
            ("--geometry-receipt", args.geometry_receipt),
            ("--valid-time", valid_time),
        ) if value is None]
        incompatible = [name for name, value in (
            ("--wps-namelist", args.wps_namelist),
            ("--namelist-input", args.namelist_input),
        ) if value is not None]
        if missing or incompatible:
            parser.error(
                "single-domain mode missing/incompatible options: "
                f"missing={missing}, incompatible={incompatible}")
        if args.physics_profile is not None and args.experiment_config_suite:
            parser.error(
                "--experiment-config-suite means NO named profile; "
                "pass exactly one of it and --physics-profile")
        manifest = export_prepared_wrf(
            args.prepared_cache, args.static_cache, args.geometry_receipt,
            args.output, valid_time=valid_time,
            boundary_interval_seconds=args.boundary_interval_seconds,
            overwrite=args.overwrite,
            physics_profile=args.physics_profile,
            experiment_config_suite=args.experiment_config_suite,
            expert_acknowledgements=tuple(args.ack),
            acknowledgement_provenance={
                value: ["--ack"] for value in args.ack
            })
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
