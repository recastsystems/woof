"""Fail-closed ``rw-wps.mapping.v1`` source-frame materialization.

This module is the gpuwm-side consumer of RW-WPS's sealed declarative
mapping contract.  It deliberately stops at a canonical, hash-bound source
frame: interpolation, WRF-real initialization, static geography, nesting and
NetCDF export remain owned by their existing woof implementations.

The public ``mapped`` capability is implementation-runnable under this strict
contract. Stock-WRF certification remains keyed to exact retained mapping,
composition, source, and domain evidence; arbitrary mappings do not inherit
that evidence.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

import numpy as np

# NetCDF is decoded by the Rust bridge, exactly as GRIB2 already is
# through grib2_inventory/grib2_dump further down this module.  There is
# no `import netCDF4` here any more and no Python fallback behind it: a
# missing decoder is a named refusal, not a second implementation.
from woof import netcdf_bridge
from woof.core import portable_math as pm
from woof.explain import warn
from woof.ingest.grib import Era5Snapshot, build_rust_bridge, inspect_grib1_envelopes
from woof.ingest.quantization import admit_bounded
from woof.ingest.source_coverage import (
    ForcingSeriesRefusal, ScratchDiskRefusal, scratch_disk_refusal)
from woof.ingest.soil_contract import (
    MAPPED_SOIL_MOISTURE,
    MAPPED_SOIL_TEMPERATURE,
)
from woof.source_authorities import (
    BOUNDARY_MULTIPLES_KEY,
    boundary_interval_refusal,
)
from woof.source_frame import (
    COMPLETED_FIELD_REFERENCE_PREFIX,
    PORTABLE_HEADER_RULE,
    FieldDescriptor,
    GridDescriptor,
    SourceFrameHeader,
    TimeDescriptor,
    VerticalDescriptor,
    portable_frame_header_sha256,
    validate_source_frame,
)


MAPPING_SCHEMA = "rw-wps.mapping.v1"
FRAME_EVIDENCE_SCHEMA = "gpuwm-mapped-source-frames-v1"
INPUT_MANIFEST_SCHEMA = "gpuwm-mapped-source-inputs-v1"
INSPECTION_SCHEMA = "gpuwm-mapped-source-inspection-v1"
_MAX_RETAINED_AUTHORITY_BYTES = 16 * 1024 * 1024
_GRIB2_AUTHORITY_KEYS = (
    "center",
    "subcenter",
    "master_table_version",
    "local_table_version",
)

_FORMATS = {"grib1", "grib2", "netcdf"}
_AXES = {"time", "lead", "member", "vertical", "half_level", "y", "x", "soil"}

#: Seconds per unit of a ``coordinates.time.lead`` axis.
_LEAD_UNIT_SECONDS = {
    "days": 86400, "day": 86400, "d": 86400,
    "hours": 3600, "hour": 3600, "h": 3600,
    "minutes": 60, "minute": 60, "min": 60,
    "seconds": 1, "second": 1, "s": 1,
}


def _lead_unit_seconds(units: str) -> int:
    seconds = _LEAD_UNIT_SECONDS.get(str(units).strip().lower())
    if seconds is None:
        raise ValueError(
            f"coordinates.time.lead.units {units!r} is not days, hours, minutes or seconds")
    return seconds
_LOCATIONS = {"mass", "u_face", "v_face", "surface", "soil"}
_STAGGERING = {"none", "x", "y", "z"}
_VERTICAL_KINDS = {
    "pressure", "hybrid_sigma_pressure", "model_level", "height",
    "soil_depth", "embedded_levels",
}
_POLICY_FIELDS = {
    "cloud_water_mixing_ratio",
    "rain_water_mixing_ratio",
    "cloud_ice_mixing_ratio",
    "snow_mixing_ratio",
    "graupel_or_hail_mixing_ratio",
    "vertical_velocity",
    "snow_water_equivalent",
    "snow_depth",
    "sea_ice_fraction",
}
_DERIVATION_ARGUMENTS = {
    "copy": ({"source"}, set()),
    "height_from_interfaces": ({"source"}, set()),
    "mass_from_interfaces": ({"source"}, set()),
    "mass_fraction_rebase": ({"source"}, {"exclude"}),
    "wind_speed": ({"u", "v"}, set()),
    "specific_humidity_from_rh": (
        {"relative_humidity", "temperature", "pressure"}, set()
    ),
    "relative_humidity_from_dewpoint": ({"dewpoint", "temperature"}, set()),
    "geopotential_height": ({"geopotential"}, {"gravity_m_s2"}),
    "pressure_from_vertical_coordinate": (set(), set()),
    "specific_humidity_from_dewpoint": (
        {"dewpoint", "temperature", "pressure"}, set()
    ),
    # Air temperature from potential temperature and pressure,
    # T = theta * (p / p0) ** kappa.  A height-coordinate model (MPAS,
    # hex) publishes theta, not T.  kappa (R_d / c_p) and p0 are the
    # PRODUCER's constants, declared by the mapping: WRF's are 2/7 and
    # 1e5 Pa (the defaults), MPAS's are 287/1003 and 1e5 Pa, and the
    # temperature a model held is recovered only with its own pair.
    "air_temperature_from_potential_temperature": (
        {"potential_temperature", "pressure"},
        {"reference_pressure_pa", "kappa"},
    ),
    # Specific humidity from a water vapour mixing ratio against dry
    # air, q = r / (1 + r).  Its inverse r = q / (1 - q) is what
    # initialization applies, so the round trip returns r exactly
    # (to rounding) and condensate stays on the dry-air basis the
    # producer's own mixing ratios already use.
    "specific_humidity_from_mixing_ratio": ({"mixing_ratio"}, set()),
    # Column-integrated soil water (kg m-2 per declared layer) to
    # volumetric soil moisture (m3 m-3): arithmetic over the DECLARED
    # layer bounds, in the closed catalog because a contract must never
    # run unknown code.  Shared by every model publishing layer-mass
    # soil water (the DWD family, GEM/HRDPS).
    "volumetric_soil_moisture_from_layer_mass": (
        {"layer_mass"}, {"layer_bounds_m", "water_density_kg_m3"}
    ),
    # Extend a soil column to a 0 m surface sample by replicating the
    # shallowest value -- WRF's own moisture endpoint convention
    # (module_soil_pre.F brackets layer-form moisture by repeating its
    # shallowest layer at the surface).  Lets layer-published moisture
    # join a node ladder whose surface sample the provider publishes
    # only for temperature.
    "soil_surface_node_from_shallowest": ({"source"}, set()),
    # 3-D geopotential height built hydrostatically from surface
    # geopotential and virtual temperature up the hybrid half-level
    # pressure ladder -- ECMWF's own model-level build-up (their z is
    # not archivable on all 137 levels; the provider derives it the
    # same way).  Surface pressure rides in through the vertical
    # coordinate's declared surface_pressure_field, the same channel
    # the hybrid pressure derivation consumes.
    "geopotential_height_hydrostatic": (
        {"temperature", "specific_humidity", "surface_geopotential_height"},
        {"gravity_m_s2"},
    ),
    # Surface pressure at the declared surface height from mean-sea-level
    # pressure and the source's own level heights and pressures: WRF
    # real's sfcprs3 relation, for a source that publishes MSLP and no
    # surface pressure.  Every valid time reduces its own bytes, so the
    # column mass follows the forecast lead by lead.
    "surface_pressure_from_sea_level": (
        {"sea_level_pressure", "level_height", "pressure", "surface_height"},
        set(),
    ),
}
#: Surface fields a mapping may carry with their source's missing mask
#: intact (``preserve_mask``), besides the soil column: the water state.
#: Each is read only by a consumer that takes water values from water
#: cells -- the water-temperature assembly (woof/ingest/
#: water_temperature.py) reads the sea surface analysis from its own
#: water component, the masked water chain maps sea ice from water
#: sources only, and the lake mapping (woof/ingest/lake_temperature.py)
#: declines any cell a missing donor touches -- so a land cell the source
#: leaves missing never reaches the initial state.  The pressure-level
#: route decodes the same records with the same masks.
MASKED_WATER_STATE_FIELDS = frozenset({
    "sea_surface_temperature", "sea_ice_fraction",
    "lake_water_temperature", "lake_ice_temperature", "lake_ice_depth",
})

_CANONICAL_REQUIREMENTS = {
    "air_temperature": (("vertical", "y", "x"), "mass", "K"),
    "specific_humidity": (("vertical", "y", "x"), "mass", "kg kg-1"),
    "eastward_wind": (("vertical", "y", "x"), "mass", "m s-1"),
    "northward_wind": (("vertical", "y", "x"), "mass", "m s-1"),
    "geopotential_height": (("vertical", "y", "x"), "mass", "m"),
    "surface_pressure": (("y", "x"), "surface", "Pa"),
    "terrain_height": (("y", "x"), "surface", "m"),
    "skin_temperature": (("y", "x"), "surface", "K"),
    "air_temperature_2m": (("y", "x"), "surface", "K"),
    "specific_humidity_2m": (("y", "x"), "surface", "kg kg-1"),
    "eastward_wind_10m": (("y", "x"), "surface", "m s-1"),
    "northward_wind_10m": (("y", "x"), "surface", "m s-1"),
    "land_fraction": (("y", "x"), "surface", "1"),
    "soil_temperature": (("soil", "y", "x"), "soil", "K"),
    "volumetric_soil_moisture": (("soil", "y", "x"), "soil", "m3 m-3"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(array.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _reject_duplicate_object_pairs(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _load_json_document(path: Path, label: str) -> object:
    data = Path(path).read_bytes()
    return _load_json_bytes(data, label, path)


def read_input_list(path: str | Path) -> list[Path]:
    """The ``--input-list`` file: the repeated ``--input`` argv, as lines.

    One path per line, UTF-8, in exactly the deterministic time/file
    order the repeated flag spells; a line is taken verbatim (minus its
    terminator), and whitespace-only lines are skipped so a trailing
    newline or CRLF authoring is not an error.  Nothing else is
    interpreted -- no comments, no globbing -- because this file is a
    TRANSPORT for argv, not a second grammar: a field-per-file source
    needs hundreds of input files per prepared state, and Windows caps
    a whole command line at 32 KB, which is the only reason the flag
    exists.

    Raises ``ValueError`` naming the file for an unreadable or empty
    list, so both front doors refuse before any source byte is read.
    """

    list_path = Path(path)
    try:
        text = list_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"--input-list {list_path}: {error}") from None
    files = [Path(line) for line in text.splitlines() if line.strip()]
    if not files:
        raise ValueError(f"--input-list {list_path} names no input files")
    return files


def _load_json_bytes(data: bytes, label: str, path: Path | None = None) -> object:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} is not UTF-8: {error}") from error
    try:
        return json.loads(text, object_pairs_hook=_reject_duplicate_object_pairs)
    except json.JSONDecodeError as error:
        location = "" if path is None else f" {path}"
        raise ValueError(f"invalid {label} JSON{location}: {error}") from error


@dataclass(frozen=True)
class _AuthoritySnapshot:
    path: Path
    sha256: str
    size: int
    mtime_ns: int
    device: int
    inode: int
    mode: int
    data: bytes | None = None

    @property
    def identity(self) -> tuple[object, ...]:
        return (
            self.path,
            self.sha256,
            self.size,
            self.mtime_ns,
            self.device,
            self.inode,
            self.mode,
        )


def _snapshot_authority(
    path: str | Path,
    *,
    retain_bytes: bool = False,
) -> _AuthoritySnapshot:
    path = Path(path).resolve()
    path_before = path.stat()
    if not retain_bytes:
        # Inside a preparation's digest scope: a digest this preparation or
        # its fetch already holds for these exact bytes, with the full
        # re-read moved beside the work (woof.authority_digests).
        from woof import authority_digests
        held = authority_digests.known_digest(path, path_before)
        if held is not None:
            authority_digests.defer_verify(path, path_before, held[0], held[1])
            return _AuthoritySnapshot(
                path=path,
                sha256=held[0],
                size=int(path_before.st_size),
                mtime_ns=int(path_before.st_mtime_ns),
                device=int(path_before.st_dev),
                inode=int(path_before.st_ino),
                mode=int(path_before.st_mode),
            )
    digest = hashlib.sha256()
    retained = bytearray() if retain_bytes else None
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) \
        | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"authority is not a regular file: {path}")
        if retain_bytes and before.st_size > _MAX_RETAINED_AUTHORITY_BYTES:
            raise ValueError(
                f"retained authority exceeds {_MAX_RETAINED_AUTHORITY_BYTES} "
                f"bytes: {path}"
            )
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
            if retained is not None:
                retained.extend(block)
        after = os.fstat(stream.fileno())
    path_after = path.stat()

    def identity(status: os.stat_result) -> tuple[int, int, int, int]:
        return (
            int(status.st_dev),
            int(status.st_ino),
            int(status.st_size),
            int(status.st_mtime_ns),
        )

    # On Windows, ``fstat`` does not synthesize the execute permission bits
    # that ``Path.stat`` adds for ``.exe`` files.  Bind descriptor identity to
    # path identity, but compare permission modes within the same stat API.
    if (
        identity(before) != identity(after)
        or identity(path_before) != identity(path_after)
        or identity(after) != identity(path_after)
        or stat.S_IFMT(before.st_mode) != stat.S_IFMT(after.st_mode)
        or stat.S_IFMT(after.st_mode) != stat.S_IFMT(path_after.st_mode)
        or path_before.st_mode != path_after.st_mode
    ):
        raise ValueError(f"authority changed while snapshotting: {path}")
    data = None if retained is None else bytes(retained)
    if data is not None and len(data) != after.st_size:
        raise ValueError(f"authority changed while reading: {path}")
    from woof import authority_digests
    authority_digests.remember(path, path_after, digest.hexdigest())
    return _AuthoritySnapshot(
        path=path,
        sha256=digest.hexdigest(),
        size=int(after.st_size),
        mtime_ns=int(after.st_mtime_ns),
        device=int(after.st_dev),
        inode=int(after.st_ino),
        mode=int(path_after.st_mode),
        data=data,
    )


def _require_authority_snapshot(snapshot: _AuthoritySnapshot) -> None:
    from woof import authority_digests
    if authority_digests.require_unchanged(
            snapshot.path, sha256=snapshot.sha256, size=snapshot.size,
            mtime_ns=snapshot.mtime_ns, device=snapshot.device,
            inode=snapshot.inode, mode=snapshot.mode):
        return
    current = _snapshot_authority(snapshot.path)
    if current.identity != snapshot.identity:
        raise ValueError(f"authority changed after validation: {snapshot.path}")


def _object(
    value: object,
    label: str,
    *,
    allowed: Iterable[str],
    required: Iterable[str] = (),
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be an object")
    allowed_set = set(allowed)
    unknown = sorted(set(value) - allowed_set)
    if unknown:
        raise ValueError(f"{label} has unknown key(s): {unknown}")
    missing = sorted(set(required) - set(value))
    if missing:
        raise ValueError(f"{label} is missing required key(s): {missing}")
    return value


def _finite_json(value: object, label: str = "mapping") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} contains a non-finite number")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _finite_json(item, f"{label}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _finite_json(item, f"{label}.{key}")
        return
    raise TypeError(f"{label} contains unsupported JSON value {type(value).__name__}")


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _integer(
    value: object, label: str, *, minimum: int = 0, maximum: int | None = None
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        bound = f"{minimum}..{maximum}" if maximum is not None else f">={minimum}"
        raise ValueError(f"{label} must be in {bound}")
    return value


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _axes(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty axis list")
    result = tuple(_string(item, label) for item in value)
    unknown = sorted(set(result) - _AXES)
    if unknown:
        raise ValueError(f"{label} has unsupported axes {unknown}")
    if len(set(result)) != len(result):
        raise ValueError(f"{label} contains a duplicate axis")
    return result


def _validate_selector(selector: object, expected_format: str, label: str) -> dict[str, object]:
    value = _object(
        selector,
        label,
        allowed={
            "format", "parameter", "table_version", "center", "subcenter",
            "master_table_version", "local_table_version", "level_type",
            "level_value", "second_level_type", "second_level_value",
            "discipline", "category", "member", "name", "standard_name",
            "layer_dimension", "layer_value", "layer_units", "attributes",
            "pdt", "scale",
        },
        required={"format"},
    )
    if value["format"] != expected_format:
        raise ValueError(
            f"{label}.format={value['format']!r} differs from mapping format "
            f"{expected_format!r}"
        )
    allowed_by_format = {
        "grib1": {
            "format", "parameter", "table_version", "center", "level_type",
            "level_value", "scale",
        },
        "grib2": {
            # ``pdt`` binds the product-definition template as identity, so
            # an instantaneous record and its accumulation twin (HRRR's
            # WEASD `anl` beside WEASD `0-0 day acc fcst`) are DIFFERENT
            # records to a selector rather than a duplicate-message error.
            "format", "discipline", "category", "parameter", "level_type",
            "level_value", "second_level_type", "second_level_value", "member",
            "pdt", "scale", *_GRIB2_AUTHORITY_KEYS,
        },
        "netcdf": {
            "format", "name", "standard_name", "attributes",
            "layer_dimension", "layer_value", "layer_units",
            # The level a vertical-stack member variable holds.
            "level_value",
        },
    }[expected_format]
    incompatible = sorted(set(value) - allowed_by_format)
    if incompatible:
        raise ValueError(
            f"{label} has keys incompatible with {expected_format}: {incompatible}"
        )
    required = {
        "grib1": {"parameter"},
        "grib2": {"discipline", "category", "parameter"},
        "netcdf": set(),
    }[expected_format]
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"{label} is missing selector key(s): {missing}")
    if expected_format == "netcdf":
        if value.get("name") is not None:
            value["name"] = _selector_name_field(value["name"], f"{label}.name")
        if value.get("standard_name") is not None:
            _string(value["standard_name"], f"{label}.standard_name")
        if value.get("name") is None and value.get("standard_name") is None:
            raise ValueError(f"{label} needs name and/or standard_name")
        _validate_selector_attributes(value, label)
        _validate_layer_slice(value, label)
    else:
        integer_keys = (
            ("parameter", 0, 255), ("level_type", 0, 255),
            ("second_level_type", 0, 255),
            ("table_version", 0, 255),
            ("center", 0, 65535 if expected_format == "grib2" else 255),
            ("subcenter", 0, 65535),
            ("master_table_version", 0, 255),
            ("local_table_version", 0, 255),
            ("discipline", 0, 255), ("category", 0, 255),
            ("member", 0, 255),
            ("pdt", 0, 65535),
        )
        for key, minimum, maximum in integer_keys:
            if value.get(key) is not None:
                _integer(
                    value[key], f"{label}.{key}",
                    minimum=minimum, maximum=maximum,
                )
            elif key in required:
                raise ValueError(f"{label}.{key} must not be null")
        if value.get("level_value") is not None:
            _number(value["level_value"], f"{label}.level_value")
        if "scale" in value:
            # The factor that takes this selector's records to the
            # field's source units, for a field that lists a second
            # publication of the same quantity (geopotential beside
            # geopotential height) as a fallback selector.
            scale = _number(value["scale"], f"{label}.scale")
            if scale == 0.0:
                raise ValueError(f"{label}.scale must be finite and nonzero")
        second_keys = {
            key for key in ("second_level_type", "second_level_value")
            if value.get(key) is not None
        }
        if second_keys and expected_format != "grib2":
            raise ValueError(f"{label} second fixed surfaces require GRIB2")
        if second_keys and second_keys != {"second_level_type", "second_level_value"}:
            raise ValueError(
                f"{label} second_level_type and second_level_value are an atomic pair"
            )
        if value.get("second_level_value") is not None:
            _number(value["second_level_value"], f"{label}.second_level_value")
        missing_identifiers = {
            key
            for key in (
                ("parameter", "level_type", "table_version", "center")
                if expected_format == "grib1"
                else ("discipline", "category", "parameter", "second_level_type")
            )
            if value.get(key) == 255
        }
        if missing_identifiers:
            raise ValueError(
                f"{label} uses missing/undefined identifier code 255 for "
                f"{sorted(missing_identifiers)}"
            )
        if expected_format == "grib2" and value.get("level_type") == 255 and any(
            value.get(key) is not None
            for key in ("level_value", "second_level_type", "second_level_value")
        ):
            raise ValueError(
                f"{label} uses level_type=255 with fixed-surface metadata"
            )
        if expected_format == "grib2":
            local_identifiers = {
                key: identifier
                for key in (
                    "discipline", "category", "parameter", "level_type",
                    "second_level_type",
                )
                if isinstance((identifier := value.get(key)), int)
                and 192 <= identifier <= 254
            }
            if local_identifiers:
                missing_authority = sorted(
                    set(_GRIB2_AUTHORITY_KEYS) - set(value)
                )
                rendered = ", ".join(
                    f"{key}={identifier}"
                    for key, identifier in sorted(local_identifiers.items())
                )
                if missing_authority:
                    raise ValueError(
                        f"{label} uses GRIB2 local-use identifier(s) {rendered}; "
                        "selector must bind complete Section 1 table authority "
                        f"(missing {missing_authority})"
                    )
                if value["local_table_version"] == 255:
                    raise ValueError(
                        f"{label} uses GRIB2 local-use identifier(s) {rendered} "
                        "with local_table_version=255 (no local table)"
                    )
    return value


def _optional_number_overlap(
    left: Mapping[str, object],
    right: Mapping[str, object],
    key: str,
) -> bool:
    if key not in left or key not in right:
        return True
    return math.isclose(
        float(left[key]),
        float(right[key]),
        rel_tol=2e-9,
        abs_tol=2e-9,
    )


#: Depth units a NetCDF layer coordinate may be declared in, and the factor
#: that turns one of its values into metres.  Only lengths appear here: a
#: layer slice addresses a physical depth, so admitting a dimensionless
#: spelling would let a selector claim a depth it cannot name.
LAYER_UNIT_METRES = MappingProxyType({"m": 1.0, "cm": 0.01, "mm": 0.001})


def _validate_selector_attributes(
    selector: Mapping[str, object], label: str,
) -> None:
    """Validate the optional NetCDF attribute discriminator.

    One variable NAME can mean two different quantities in one producer's
    output.  NOAA's 20CRv3 publishes air temperature on pressure levels and
    at 2 m as ``air`` in two files, humidity as ``shum`` in two, and both
    wind components at model levels and at 10 m under the same names --
    which is not a defect, because each variable also says which it is
    (``level_desc = "Pressure Levels"`` against ``"2 m"``).

    So a selector may require named attributes to carry declared values.  It
    is a filter over the file's OWN self-description, never a rename: a
    variable that does not carry the attribute is simply not the one the
    mapping asked for.
    """

    declared = selector.get("attributes")
    if declared is None:
        return
    if not isinstance(declared, dict) or not declared:
        raise ValueError(
            f"{label}.attributes must be a non-empty attribute/value object"
        )
    for key, value in declared.items():
        _string(key, f"{label}.attributes key")
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            raise ValueError(
                f"{label}.attributes[{key!r}] must be a string or a number"
            )


def _attributes_match(variable, selector: Mapping[str, object]) -> bool:
    """Whether one variable carries every attribute the selector declares."""

    declared = selector.get("attributes")
    if not declared:
        return True
    for key, expected in declared.items():
        observed = getattr(variable, str(key), None)
        if isinstance(expected, str):
            if not isinstance(observed, str) or observed != expected:
                return False
        else:
            try:
                if observed is None or float(observed) != float(expected):
                    return False
            except (TypeError, ValueError):
                return False
    return True


def _validate_layer_slice(selector: Mapping[str, object], label: str) -> None:
    """Validate the optional NetCDF layer-slice keys, as an atomic trio.

    A meteorological producer is free to publish an N-layer quantity as ONE
    variable with its own layer dimension -- NOAA's 20CRv3 ``tsoil``/``soilw``
    do exactly that -- rather than as N separate variables the way ECMWF's
    ``stl1..stl4`` do.  Both shapes are the same declarative content, so a
    selector may address one slice of such a variable BY THE VALUE ON THE
    FILE'S OWN COORDINATE, never by position.

    The three keys are atomic on purpose.  ``layer_dimension`` alone cannot
    select; ``layer_value`` alone does not say which coordinate carries it;
    and ``layer_units`` is what lets a depth-bound contract (the soil
    composition) check that a selector is bound to the layer it claims,
    instead of trusting the order the selectors happen to be written in.
    """

    keys = {"layer_dimension", "layer_value", "layer_units"} & set(selector)
    if not keys:
        return
    if keys != {"layer_dimension", "layer_value", "layer_units"}:
        raise ValueError(
            f"{label} layer slice needs layer_dimension, layer_value and "
            f"layer_units together; it declares only {sorted(keys)}"
        )
    _string(selector["layer_dimension"], f"{label}.layer_dimension")
    if not str(selector["layer_dimension"]).strip():
        raise ValueError(f"{label}.layer_dimension must not be blank")
    _number(selector["layer_value"], f"{label}.layer_value")
    units = _string(selector["layer_units"], f"{label}.layer_units")
    if units not in LAYER_UNIT_METRES:
        raise ValueError(
            f"{label}.layer_units={units!r} is not one of "
            f"{sorted(LAYER_UNIT_METRES)}"
        )


def layer_slice_depth_metres(selector: Mapping[str, object]) -> float | None:
    """The depth in metres a selector's layer slice addresses, or None."""

    if "layer_value" not in selector:
        return None
    return float(selector["layer_value"]) * LAYER_UNIT_METRES[
        str(selector["layer_units"])
    ]


def _optional_exact_overlap(
    left: Mapping[str, object],
    right: Mapping[str, object],
    key: str,
) -> bool:
    return key not in left or key not in right or left[key] == right[key]


def _grib_selectors_overlap(
    left: Mapping[str, object],
    right: Mapping[str, object],
    source_format: str,
) -> bool:
    """Return whether one runtime GRIB record can satisfy both selectors."""

    required = (
        ("parameter",)
        if source_format == "grib1"
        else ("discipline", "category", "parameter")
    )
    if any(left[key] != right[key] for key in required):
        return False
    # A GRIB2 record carries exactly one product definition template, so
    # two selectors pinning different templates can never both match it;
    # that is what lets a field rank a PDT 1 selector above a PDT 0 form
    # of the same record (both engines already match ``pdt`` exactly).
    optional_exact = (
        ("table_version", "center", "level_type")
        if source_format == "grib1"
        else ("level_type", "member", "pdt", *_GRIB2_AUTHORITY_KEYS)
    )
    if any(
        not _optional_exact_overlap(left, right, key) for key in optional_exact
    ):
        return False
    if not _optional_number_overlap(left, right, "level_value"):
        return False
    if source_format == "grib1":
        return True
    left_second = "second_level_type" in left
    right_second = "second_level_type" in right
    if left_second != right_second:
        return False
    if not left_second:
        return True
    return left["second_level_type"] == right["second_level_type"] and math.isclose(
        float(left["second_level_value"]),
        float(right["second_level_value"]),
        rel_tol=2e-9,
        abs_tol=2e-9,
    )


def _validate_era_ladders(
    raw: object, levels: Sequence[float], source_format: object,
) -> None:
    """``vertical.era_ladders``: the ladders other publications carry.

    A publisher adds or drops pressure levels between releases of one
    product while every other byte of its files keeps the same shape, so
    a file from the other side of such a change lacks a level
    ``vertical.levels`` names.  Each entry here is the whole ladder one
    such publication carries, drawn from ``levels``; the GRIB decoders
    build the column from the largest declared ladder the records carry
    in full (:func:`_carried_ladder`), and a file missing a level that
    no declared ladder omits still refuses at that level.
    """

    if not levels:
        raise ValueError(
            "vertical.era_ladders needs vertical.levels: without declared "
            "levels the decoder already takes the ladder the file offers, "
            "so the era ladders would never be read")
    if source_format == "netcdf":
        raise ValueError(
            "vertical.era_ladders is read by the GRIB decoders only; the "
            "NetCDF decoder selects vertical.levels by coordinate value, so "
            "a file carrying one of these ladders would still refuse at its "
            "first absent level")
    if not isinstance(raw, list) or not raw:
        raise ValueError("vertical.era_ladders must be a non-empty list of level lists")
    declared = set(levels)
    for index, ladder in enumerate(raw):
        label = f"vertical.era_ladders[{index}]"
        if not isinstance(ladder, list) or not ladder:
            raise ValueError(f"{label} must be a non-empty numeric list")
        values = [
            _number(value, f"{label}[{position}]")
            for position, value in enumerate(ladder)
        ]
        if len(set(values)) != len(values):
            raise ValueError(f"{label} must be a unique numeric list")
        stray = [value for value in values if value not in declared]
        if stray:
            raise ValueError(
                f"{label} names levels {stray} that vertical.levels does not "
                "declare; an era ladder is drawn from the declared levels, "
                "which are the only ones the decoder admits")


#: The canonical fields ``fields.<name>.published_levels`` may declare: the
#: five hydrometeor mass mixing ratios.  A publisher that carries a
#: hydrometeor on part of its pressure ladder is stating that the mass it
#: did not write out there is negligible (NCEP's GFS pgrb2.0p25 writes
#: CLMR/ICMR/RWMR/SNMR/GRLE on 22 of the 33 isobaric levels its state
#: carries, 50..1000 hPa less 70 hPa), so the column is zero at the other
#: levels.  For any other field a zero inside the column is a wrong value
#: rather than an absent one, so the declaration is refused there.
PUBLISHED_LEVELS_ZERO_FIELDS = (
    "cloud_water_mixing_ratio", "rain_water_mixing_ratio",
    "cloud_ice_mixing_ratio", "snow_mixing_ratio",
    "graupel_or_hail_mixing_ratio",
)


def _validate_published_levels(
    name: str,
    field: Mapping[str, object],
    levels: Sequence[float],
    source_format: str,
) -> None:
    """``fields.<name>.published_levels``: the ladder levels a field is on.

    A publisher may carry a field on a subset of the ladder its state
    carries.  Without this the whole mapping must drop to that subset (a
    ladder is chosen for every stacked field at once), which would take
    temperature, humidity and wind off the levels above it.  The
    declaration names the levels the field is published on, held to
    every frame like the ladder itself; on the other declared levels the
    field is ``absent``, and ``"zero"`` is the one value it may take,
    for the fields where zero is what an unwritten mass is
    (:data:`PUBLISHED_LEVELS_ZERO_FIELDS`).
    """

    label = f"fields.{name}.published_levels"
    upper_boundary = field.get("published_levels", {}).get("absent") == "nearest_upper_boundary"
    if upper_boundary and name != "specific_humidity":
        raise ValueError(f"{label}: upper boundary extension is restricted to specific_humidity")
    if name not in PUBLISHED_LEVELS_ZERO_FIELDS and not upper_boundary:
        raise ValueError(
            f"{label} is restricted to the hydrometeor mass mixing ratios "
            f"({', '.join(PUBLISHED_LEVELS_ZERO_FIELDS)}): their zero is "
            "what a mass the publisher does not write out at a level is, "
            "and on any other field a zero inside the column is a wrong "
            "value rather than an absent one")
    if source_format not in {"grib1", "grib2"}:
        raise ValueError(
            f"{label} is read by the GRIB decoders only; the NetCDF decoder "
            "selects vertical.levels by coordinate value, so a file without "
            "one of these levels would still refuse there")
    if not levels:
        raise ValueError(
            f"{label} needs vertical.levels: the declaration is a subset of "
            "the declared ladder")
    if not field.get("selectors"):
        raise ValueError(
            f"{label} describes the records a publisher writes, so the field "
            "must be directly selected")
    if tuple(field["source_axes"]) != ("vertical", "y", "x") \
            or tuple(field["target_axes"]) != ("vertical", "y", "x"):
        raise ValueError(
            f"{label} is a vertical declaration; the field's source and "
            "target axes must be ('vertical', 'y', 'x')")
    if field["missing"]["kind"] != "reject":
        raise ValueError(
            f"{label} keeps the reject missing policy: a missing cell on a "
            "level the publisher writes is still a defect in the file, and "
            "only a level it does not write is zero")
    if float(field["units"].get("offset", 0.0)) != 0.0:
        raise ValueError(
            f"{label} needs a zero unit offset, or the declared zero would "
            "not stay zero through the unit conversion")
    value = _object(
        field["published_levels"], label,
        allowed={"levels", "absent"}, required={"levels", "absent"})
    if value["absent"] not in {"zero", "nearest_upper_boundary"}:
        raise ValueError(f"{label}.absent must be zero or nearest_upper_boundary")
    published = value["levels"]
    if not isinstance(published, list) or not published:
        raise ValueError(f"{label}.levels must be a non-empty numeric list")
    values = [_number(level, f"{label}.levels[{index}]")
              for index, level in enumerate(published)]
    if len(set(values)) != len(values):
        raise ValueError(f"{label}.levels must be a unique numeric list")
    declared = set(float(level) for level in levels)
    stray = [level for level in values if float(level) not in declared]
    if stray:
        raise ValueError(
            f"{label}.levels names levels {stray} that vertical.levels does "
            "not declare; the decoder admits the declared levels only")
    if upper_boundary and any(float(level) >= min(values) for level in declared - set(values)):
        raise ValueError(f"{label}: boundary extension cannot fill interior or lower levels")
    if set(float(level) for level in values) == declared:
        raise ValueError(
            f"{label}.levels names every declared level, so the declaration "
            "would change nothing; remove it")


def _published_levels(field: Mapping[str, object]) -> frozenset[float] | None:
    """The levels ``fields.<name>.published_levels`` names, or ``None``."""

    declaration = field.get("published_levels")
    if declaration and declaration.get("absent") == "nearest_upper_boundary":
        raise ValueError("nearest_upper_boundary requires the Rust mapped engine")
    if declaration is None:
        return None
    return frozenset(float(level) for level in declaration["levels"])


def _carried_ladder(
    mapping: Mapping[str, object],
    declared: Sequence[float],
    level_sets: Iterable[set[float]],
) -> tuple[float, ...]:
    """The declared ladder a decode stacks: the largest one carried in full.

    ``level_sets`` holds, per stacked vertical field and valid time, the
    declared levels its records carry.  ``vertical.levels`` wins when
    every set holds it; otherwise the largest ``vertical.era_ladders``
    entry every set holds, in the declared order.  When none fits, the
    declared ladder is returned and the coverage check names what is
    missing from it, exactly as it did before era ladders existed.
    """

    declared = tuple(float(value) for value in declared)
    raw = mapping["coordinates"]["vertical"].get("era_ladders") or ()
    carried = [set(values) for values in level_sets]
    if not raw or not carried:
        return declared

    def fits(ladder: Sequence[float]) -> bool:
        return all(level in values for values in carried for level in ladder)

    if fits(declared):
        return declared
    best: tuple[float, ...] | None = None
    for entry in raw:
        members = {float(value) for value in entry}
        ladder = tuple(level for level in declared if level in members)
        if fits(ladder) and (best is None or len(ladder) > len(best)):
            best = ladder
    return declared if best is None else best


class NothingMatched(ValueError):
    """A GRIB2 decode whose files carry no record any selector reads.

    A ``ValueError`` like every other mapping-versus-bytes refusal; its
    own type lets a caller with a declared answer for an absent field
    (``fields.terrain_height.when_absent``) take that answer for exactly
    this case and no other.
    """


#: Standard gravity for the surface-height derivation (m s-2).
_SURFACE_HEIGHT_GRAVITY = 9.80665


def _height_at_surface_pressure(
    levels_pa: Sequence[float],
    geopotential_height: np.ndarray,
    temperature: np.ndarray,
    specific_humidity: np.ndarray,
    surface_pressure: np.ndarray,
    surface_temperature: np.ndarray,
    surface_dewpoint: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    """Each column's height at its surface pressure, and how it was made.

    The hypsometric equation from the first level above the ground down
    to the surface pressure, with the layer's mean virtual temperature
    taken between that level and the surface.  At the surface the
    temperature and humidity are interpolated linearly in log pressure
    between the two levels that bracket it; below the deepest level they
    come from the 2 m temperature and the dewpoint's specific humidity,
    the surface fields WPS and real extrapolate from.  A surface above
    the highest level is refused: the column says nothing about the air
    below it.  The mapped engine's ``derive::height_at_surface_pressure``
    is the same arithmetic.
    """

    pressure = np.asarray(surface_pressure, dtype=np.float64)
    if pressure.ndim != 2:
        raise ValueError(
            "terrain_height from surface pressure needs (y, x) surface "
            f"fields; surface pressure has shape {pressure.shape}")
    levels = np.asarray(levels_pa, dtype=np.float64)
    column_shape = (levels.size, *pressure.shape)
    columns = {}
    for label, field in (("geopotential height", geopotential_height),
                         ("temperature", temperature),
                         ("specific humidity", specific_humidity)):
        array = np.asarray(field, dtype=np.float64)
        if array.shape != column_shape:
            raise ValueError(
                f"terrain_height from surface pressure needs {label} on the "
                f"{levels.size} declared levels over the surface grid "
                f"{column_shape}; got {array.shape}")
        columns[label] = array
    surface = {}
    for label, field in (("surface temperature", surface_temperature),
                         ("surface dewpoint", surface_dewpoint)):
        array = np.asarray(field, dtype=np.float64)
        if array.shape != pressure.shape:
            raise ValueError(
                f"terrain_height from surface pressure needs {label} on the "
                f"surface grid {pressure.shape}; got {array.shape}")
        surface[label] = array
    if not levels.size or not np.all(np.isfinite(levels) & (levels > 0.0)):
        raise ValueError(
            "terrain_height from surface pressure needs positive pressure levels")
    order = np.argsort(-levels, kind="stable")          # deepest first
    ladder = levels[order]
    heights = columns["geopotential height"][order]
    temperatures = columns["temperature"][order]
    humidities = columns["specific humidity"][order]
    cells = pressure.size
    unreadable = ~(np.isfinite(pressure) & (pressure > 0.0))
    safe = np.where(unreadable, ladder[0], pressure)
    above_any = ladder[None, None, :] < safe[..., None]   # (y, x, level)
    has_level = above_any.any(axis=-1)
    above_ladder = int(np.count_nonzero(~has_level & ~unreadable))
    if above_ladder:
        raise ValueError(
            f"terrain_height from surface pressure: {above_ladder} of {cells} "
            f"cells have a surface pressure at or above the highest level "
            f"({ladder[-1]:g} Pa), so the column says nothing about the air "
            "below their surface")
    position = np.argmax(above_any, axis=-1)            # first level above ground
    rows, cols = np.indices(pressure.shape)
    at = lambda field, level: field[level, rows, cols]  # noqa: E731
    t_above = at(temperatures, position)
    q_above = at(humidities, position)
    z_above = at(heights, position)
    p_above = ladder[position]
    below = np.maximum(position - 1, 0)
    p_below = ladder[below]
    beneath = position == 0
    weight = np.where(
        beneath, 0.0,
        (pm.log(safe) - pm.log(p_above))
        / np.where(beneath, 1.0, pm.log(p_below) - pm.log(p_above)))
    t_level = t_above + (at(temperatures, below) - t_above) * weight
    q_level = q_above + (at(humidities, below) - q_above) * weight
    dewpoint = surface["surface dewpoint"]
    vapour_hpa = (10.0 * 0.6112) * pm.exp(
        17.67 * (dewpoint - 273.15) / (dewpoint - 29.65))
    q_surface = 0.622 * vapour_hpa / (safe / 100.0 - 0.378 * vapour_hpa)
    surface_t = np.where(beneath, surface["surface temperature"], t_level)
    surface_q = np.where(beneath, q_surface, q_level)
    virtual_above = t_above * (1.0 + _HYDROSTATIC_VIRTUAL * q_above)
    virtual_surface = surface_t * (1.0 + _HYDROSTATIC_VIRTUAL * surface_q)
    scale = _HYDROSTATIC_RD / _SURFACE_HEIGHT_GRAVITY
    height = z_above - scale * (0.5 * (virtual_above + virtual_surface)) \
        * pm.log(safe / p_above)
    bad = int(np.count_nonzero(unreadable | ~np.isfinite(height)))
    if bad:
        raise ValueError(
            f"terrain_height from surface pressure: {bad} of {cells} cells "
            "have no finite surface pressure or column to derive from")
    return height, {
        "cells": int(cells),
        "cells_below_lowest_level": int(np.count_nonzero(beneath)),
        "minimum_m": float(height.min()),
        "maximum_m": float(height.max()),
    }


#: The identity keys a ``record_aliases`` entry compares (``record``) and
#: writes (``reads_as``).  ``pdt`` identifies a record but is never
#: rewritten: an alias renames a record, it does not change what kind of
#: product the record is.
_ALIAS_RECORD_KEYS = frozenset({
    "discipline", "category", "parameter", "level_type", "level_value",
    "second_level_type", "second_level_value", "pdt",
})
_ALIAS_READS_AS_KEYS = _ALIAS_RECORD_KEYS - {"pdt"}
_ALIAS_INTEGER_KEYS = (
    ("discipline", 0, 255), ("category", 0, 255), ("parameter", 0, 255),
    ("level_type", 0, 255), ("second_level_type", 0, 255), ("pdt", 0, 65535),
)


def _alias_part(raw: object, label: str, allowed: frozenset[str]) -> dict[str, object]:
    value = _object(
        raw, label, allowed=set(allowed),
        required={"discipline", "category", "parameter"},
    )
    for key, minimum, maximum in _ALIAS_INTEGER_KEYS:
        if key in value:
            _integer(value[key], f"{label}.{key}", minimum=minimum, maximum=maximum)
    for key in ("level_value", "second_level_value"):
        if key in value:
            _number(value[key], f"{label}.{key}")
    return value


def _alias_names(pattern: Mapping[str, object], record: "_GribRecord") -> bool:
    """Whether an alias's ``record`` names this record.

    Only the keys the alias declares are compared: a spelling is named by
    what distinguishes it, and a key it leaves out (the deepest soil
    layer's bottom, written as the all-ones missing value) is not.
    """

    def same(key: str, observed: object) -> bool:
        if key not in pattern:
            return True
        if observed is None:
            return False
        if key in {"level_value", "second_level_value"}:
            return math.isclose(float(observed), float(pattern[key]), abs_tol=1e-9)
        return int(observed) == int(pattern[key])

    return (
        same("discipline", record.discipline)
        and same("category", record.category)
        and same("parameter", record.parameter)
        and same("level_type", record.level_type)
        and same("level_value", record.level_value)
        and same("second_level_type", record.second_level_type)
        and same("second_level_value", record.second_level_value)
        and same("pdt", record.time_semantics[0] if record.time_semantics else None)
    )


def _alias_record(
    record: "_GribRecord", aliases: Sequence[Mapping[str, object]],
) -> tuple["_GribRecord", bool]:
    """The record under the spelling the mapping's selectors read.

    ``mapping.record_aliases`` names records an earlier publication of
    the product spelled differently; the first (and, by validation, the
    only) alias naming ``record`` rewrites the keys its ``reads_as``
    declares.  A ``reads_as`` without a second surface reads as a record
    that has none.  The mapped engine's ``grib::alias_identities`` makes
    the same rewrite.
    """

    for alias in aliases:
        if not _alias_names(alias["record"], record):
            continue
        reads_as = alias["reads_as"]
        changes: dict[str, object] = {
            key: int(reads_as[key])
            for key in ("discipline", "category", "parameter", "level_type")
            if key in reads_as
        }
        if "level_value" in reads_as:
            changes["level_value"] = float(reads_as["level_value"])
        if "second_level_type" in reads_as:
            changes["second_level_type"] = int(reads_as["second_level_type"])
            changes["second_level_value"] = float(reads_as["second_level_value"])
        else:
            changes["second_level_type"] = 255
        from dataclasses import replace

        return replace(record, **changes), True
    return record, False


def _alias_probe(values: Mapping[str, object]) -> "_GribRecord":
    """A data-less record carrying ``values``' identity, for validation."""

    return _GribRecord(
        source=Path("."), index=0,
        reference_time=datetime.min, valid_time=datetime.min, member=None,
        parameter=int(values["parameter"]),
        level_type=int(values.get("level_type", 255)),
        level_value=float(values.get("level_value", 0.0)),
        table_version=None, center=None, subcenter=None,
        master_table_version=None, local_table_version=None,
        discipline=int(values["discipline"]), category=int(values["category"]),
        second_level_type=int(values.get("second_level_type", 255)),
        second_level_value=float(values.get("second_level_value", 0.0)),
        process_identity=None, time_semantics=(int(values.get("pdt", 0)),),
        values=np.empty((0, 0)), latitude=np.empty(0), longitude=np.empty(0),
        grid_fingerprint="",
    )


def _alias_patterns_overlap(
    left: Mapping[str, object], right: Mapping[str, object],
) -> bool:
    """Whether one record could be named by both declared patterns."""

    for key in _ALIAS_RECORD_KEYS:
        if key not in left or key not in right:
            continue
        if key in {"level_value", "second_level_value"}:
            if not math.isclose(float(left[key]), float(right[key]), abs_tol=1e-9):
                return False
        elif int(left[key]) != int(right[key]):
            return False
    return True


def _selector_as_alias_pattern(selector: Mapping[str, object]) -> dict[str, object]:
    """A GRIB2 selector's identity in alias-pattern terms.

    A selector that declares no second surface reads only records that
    have none (code 255), so that is what it names here.
    """

    pattern = {
        key: selector[key] for key in _ALIAS_RECORD_KEYS if key in selector
    }
    pattern.setdefault("second_level_type", 255)
    return pattern


def _validate_record_aliases(
    raw: object,
    fields: Mapping[str, Mapping[str, object]],
    source_format: str,
) -> None:
    """``mapping.record_aliases``: records another publication spells apart.

    Each entry names one earlier spelling (``record``) and the spelling
    the mapping's selectors read (``reads_as``).  Refused, each for the
    breakage it would cause: an alias on a format whose decoder does not
    read it (it would never apply); a ``record`` a selector already reads
    as spelled (the alias would change what a file the mapping already
    decodes decodes to); two aliases that can name one record (which one
    applies would be an accident of order); and a ``reads_as`` that no
    selector, or more than one field, reads (the renamed record would be
    dropped or claimed twice).
    """

    if source_format != "grib2":
        raise ValueError(
            "mapping.record_aliases is read by the GRIB2 decoders only; on "
            f"a {source_format} mapping it would never rename a record")
    if not isinstance(raw, list) or not raw:
        raise ValueError("mapping.record_aliases must be a non-empty list")
    parsed: list[tuple[dict[str, object], dict[str, object]]] = []
    for index, entry in enumerate(raw):
        label = f"record_aliases[{index}]"
        alias = _object(entry, label, allowed={"record", "reads_as"},
                        required={"record", "reads_as"})
        record = _alias_part(alias["record"], f"{label}.record", _ALIAS_RECORD_KEYS)
        reads_as = _alias_part(
            alias["reads_as"], f"{label}.reads_as", _ALIAS_READS_AS_KEYS)
        if ("second_level_type" in reads_as) != ("second_level_value" in reads_as):
            raise ValueError(
                f"{label}.reads_as must declare second_level_type and "
                "second_level_value together, or neither for a record "
                "with no second surface")
        for previous_index, (previous, _reads) in enumerate(parsed):
            if _alias_patterns_overlap(previous, record):
                raise ValueError(
                    f"{label}.record and record_aliases[{previous_index}].record "
                    "can name the same record; which alias applied would "
                    "depend on their order")
        for field_name, field in fields.items():
            for position, selector in enumerate(field.get("selectors", [])):
                if _alias_patterns_overlap(
                        _selector_as_alias_pattern(selector), record):
                    raise ValueError(
                        f"{label}.record is read as spelled by "
                        f"fields.{field_name}.selectors[{position}]; an alias "
                        "names a spelling no selector reads, or it would "
                        "change what a file the mapping already decodes "
                        "decodes to")
        identity = {**record, **reads_as}
        if "second_level_type" not in reads_as:
            identity.pop("second_level_value", None)
            identity["second_level_type"] = 255
        probe = _alias_probe(identity)
        readers = sorted({
            field_name for field_name, field in fields.items()
            if any(_selector_matches_record(selector, probe, "grib2")
                   for selector in field.get("selectors", []))
        })
        if len(readers) != 1:
            raise ValueError(
                f"{label}.reads_as is read by {len(readers)} mapped fields "
                f"{readers}; a renamed record must answer exactly one field")
        parsed.append((record, reads_as))


#: Operations ``fields.terrain_height.when_absent`` may name, and the
#: fields each reads: three column fields, then three surface fields.
_WHEN_ABSENT_OPERATIONS = {
    "height_at_surface_pressure": (
        "geopotential_height", "temperature", "specific_humidity",
        "surface_pressure", "surface_temperature", "surface_dewpoint",
    ),
}


def _validate_when_absent(
    name: str,
    field: Mapping[str, object],
    fields: Mapping[str, Mapping[str, object]],
    vertical: Mapping[str, object],
    source_format: str,
) -> None:
    """``fields.<name>.when_absent``: the field derived when files lack it.

    Read for ``terrain_height`` supplied through a composition's terrain
    supplement, where the derived value is broadcast like a published
    analysis terrain; anywhere else nothing would read it, so it is
    refused rather than left to do nothing.
    """

    label = f"fields.{name}.when_absent"
    if name != "terrain_height":
        raise ValueError(
            f"{label} is read only for terrain_height, which a composition's "
            "terrain supplement derives when its files carry none")
    if source_format != "grib2":
        raise ValueError(
            f"{label} is read by the GRIB2 composition only; on a "
            f"{source_format} mapping it would never run")
    if not field.get("selectors"):
        raise ValueError(
            f"{label} is the answer when the selectors match nothing, so "
            "the field must also declare selectors")
    operation = field["when_absent"]
    kind = operation.get("operation") if isinstance(operation, dict) else None
    reads = _WHEN_ABSENT_OPERATIONS.get(kind) if isinstance(kind, str) else None
    if reads is None:
        raise ValueError(
            f"{label}.operation must be one of {sorted(_WHEN_ABSENT_OPERATIONS)}")
    value = _object(operation, label, allowed={"operation", *reads},
                    required={"operation", *reads})
    if vertical["kind"] != "pressure" or str(vertical["units"]) not in {"Pa", "hPa"}:
        raise ValueError(
            f"{label} ({kind}) reads a pressure ladder in Pa or hPa; the "
            f"mapping's vertical is {vertical['kind']!r} in "
            f"{vertical['units']!r}")
    for position, key in enumerate(reads):
        target = _string(value[key], f"{label}.{key}")
        mapped = fields.get(target)
        if mapped is None or not mapped.get("selectors"):
            raise ValueError(
                f"{label}.{key} names {target!r}, which is not a directly "
                "decoded field of this mapping; the derivation reads the "
                "primary's decoded records")
        axes = ("vertical", "y", "x") if position < 3 else ("y", "x")
        if tuple(mapped["target_axes"]) != axes:
            raise ValueError(
                f"{label}.{key} names {target!r}, whose axes "
                f"{tuple(mapped['target_axes'])} are not {axes}")


def load_mapping(
    path: str | Path,
    *,
    _raw: object | None = None,
) -> dict[str, object]:
    """Read and independently validate the executable subset of the schema.

    RW-WPS performs the full user-facing validation.  The engine repeats the
    structural and target-contract checks that affect array materialization so
    a mapping cannot become less safe when the frontend and engine are invoked
    separately.
    """

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = _load_json_document(path, "mapping") if _raw is None else _raw
    _finite_json(raw)
    mapping = _object(
        raw,
        "mapping",
        allowed={
            "schema", "name", "format", "coordinates", "fields", "derivations",
            "target", "grid", "record_aliases", "producer_attributes",
        },
        required={"schema", "name", "format", "coordinates", "fields", "target"},
    )
    if mapping.get("producer_attributes") is not None:
        if mapping["format"] != "netcdf":
            raise ValueError("producer_attributes is a NetCDF declaration")
        table = mapping["producer_attributes"]
        if not isinstance(table, dict) or not table:
            raise ValueError("mapping.producer_attributes must be a non-empty object")
        for variable_name, attributes in table.items():
            _string(variable_name, "producer_attributes variable name")
            if not isinstance(attributes, dict) or not attributes:
                raise ValueError(
                    f"producer_attributes.{variable_name} must be a non-empty object")
            for key, value in attributes.items():
                _string(key, f"producer_attributes.{variable_name} attribute name")
                _string(value, f"producer_attributes.{variable_name}.{key}")
    if mapping["schema"] != MAPPING_SCHEMA:
        raise ValueError(
            f"unsupported mapping schema {mapping['schema']!r}; expected {MAPPING_SCHEMA!r}"
        )
    _string(mapping["name"], "mapping.name")
    source_format = mapping["format"]
    if source_format not in _FORMATS:
        raise ValueError(f"unsupported mapping format {source_format!r}")
    if mapping.get("grid") is not None:
        mapping["grid"] = _validate_grid_declaration(
            mapping["grid"], source_format)

    coordinates = _object(
        mapping["coordinates"],
        "mapping.coordinates",
        allowed={"horizontal", "vertical", "time", "member"},
        required={"horizontal", "vertical", "time"},
    )
    horizontal = _object(
        coordinates["horizontal"],
        "mapping.coordinates.horizontal",
        allowed={"kind", "latitude", "longitude", "normalization"},
        required={"kind"},
    )
    if "normalization" in horizontal and (
            horizontal["kind"] != "embedded_grid" or source_format != "grib2"
            or horizontal["normalization"] not in {"coarsest_exact_subset", "largest_common_finite_rectangle"}):
        raise ValueError("unknown GRIB horizontal normalization")
    if horizontal["kind"] == "embedded_grid":
        if source_format == "netcdf":
            raise ValueError("NetCDF mappings require latitude/longitude selectors")
    elif horizontal["kind"] == "variables":
        for name in ("latitude", "longitude"):
            if name not in horizontal:
                raise ValueError(f"horizontal coordinates are missing {name}")
            _validate_selector(horizontal[name], source_format, f"horizontal.{name}")
    else:
        raise ValueError(f"unsupported horizontal coordinate kind {horizontal['kind']!r}")

    vertical = _object(
        coordinates["vertical"],
        "mapping.coordinates.vertical",
        allowed={
            "kind", "selector", "units", "positive", "levels", "era_ladders",
            "hybrid_a", "hybrid_b", "interface_levels", "interface_selector",
            "hybrid_a_field", "hybrid_b_field", "surface_pressure_field",
            "model_top_pressure_pa",
        },
        required={"kind", "units"},
    )
    if vertical["kind"] not in _VERTICAL_KINDS:
        raise ValueError(f"unsupported vertical kind {vertical['kind']!r}")
    if "model_top_pressure_pa" in vertical:
        if vertical["kind"] not in {"model_level", "hybrid_sigma_pressure"}:
            raise ValueError("model_top_pressure_pa requires a native model-level coordinate")
        if _number(vertical["model_top_pressure_pa"], "vertical.model_top_pressure_pa") <= 0.0:
            raise ValueError("vertical.model_top_pressure_pa must be positive")
    _string(vertical["units"], "vertical.units")
    if vertical.get("positive") not in {None, "up", "down"}:
        raise ValueError("vertical.positive must be 'up' or 'down'")
    if "selector" in vertical and vertical["selector"] is not None:
        _validate_selector(vertical["selector"], source_format, "vertical.selector")
    if source_format == "netcdf" and vertical.get("selector") is None:
        raise ValueError("NetCDF mappings require a vertical coordinate selector")
    if vertical.get("interface_selector") is not None:
        if source_format != "netcdf" or vertical.get("interface_levels") is None:
            raise ValueError(
                "vertical.interface_selector names a NetCDF interface coordinate and "
                "requires interface_levels")
        _validate_selector(
            vertical["interface_selector"], source_format, "vertical.interface_selector")
    levels = vertical.get("levels", [])
    if not isinstance(levels, list):
        raise ValueError("vertical.levels must be a unique numeric list")
    numeric_levels = [
        _number(value, f"vertical.levels[{index}]")
        for index, value in enumerate(levels)
    ]
    if len(set(numeric_levels)) != len(numeric_levels):
        raise ValueError("vertical.levels must be a unique numeric list")
    if vertical.get("era_ladders") is not None:
        _validate_era_ladders(
            vertical["era_ladders"], numeric_levels, source_format)
    interfaces = vertical.get("interface_levels")
    if interfaces is not None:
        if vertical.get("era_ladders") is not None:
            raise ValueError(
                "era_ladders cannot be combined with interface_levels: an era ladder can shorten "
                "the mass column while the fixed N+1 interfaces still bound the declared ladder")
        if vertical["kind"] != "model_level":
            raise ValueError("interface_levels requires a model_level coordinate")
        # GRIB carries each interface as its own record's level value.  A
        # NetCDF file carries them as a coordinate VARIABLE on their own
        # dimension (WRF's bottom_top_stag), so the mapping names it, the
        # same way it names the mass-level coordinate.
        if source_format == "netcdf" and vertical.get("interface_selector") is None:
            raise ValueError(
                "interface_levels on a NetCDF model_level coordinate requires "
                "vertical.interface_selector naming the interface coordinate variable")
        if not isinstance(interfaces, list) or len(interfaces) != len(levels) + 1:
            raise ValueError("interface_levels must bound every mass level with N+1 interfaces")
        values = [_number(value, "vertical.interface_levels") for value in interfaces]
        if len(set(values)) != len(values) or not all(b > a for a, b in zip(values, values[1:])):
            raise ValueError("interface_levels must be unique and increasing in column order")
    if vertical["kind"] == "hybrid_sigma_pressure":
        # The coefficient channels are the GRIB pv coordinate octets
        # (primary, read from the bytes at decode) and inline
        # vertical.hybrid_a/hybrid_b literal arrays (the declared-data
        # fallback that keeps pv-less providers on the table-work
        # path).  The retired *_field spellings named mapping fields,
        # and a coefficient VECTOR cannot be a mapping field (fields
        # require y/x axes), so those keys validated data no code
        # could ever consume: the L137 proof lane's gap G1.
        for legacy in ("hybrid_a_field", "hybrid_b_field"):
            if vertical.get(legacy) is not None:
                raise ValueError(
                    f"vertical.{legacy} is retired: hybrid A/B "
                    "coefficients arrive in the GRIB pv coordinate "
                    "octets or as inline vertical.hybrid_a/hybrid_b "
                    "literal arrays; a field name cannot carry a "
                    "coefficient vector"
                )
        if not vertical.get("surface_pressure_field"):
            raise ValueError(
                "hybrid vertical coordinate is incomplete: "
                "['surface_pressure_field']"
            )
        _string(
            vertical["surface_pressure_field"],
            "vertical.surface_pressure_field",
        )
        declared_literals = [
            name for name in ("hybrid_a", "hybrid_b")
            if vertical.get(name) is not None
        ]
        if len(declared_literals) == 1:
            raise ValueError(
                "vertical.hybrid_a and vertical.hybrid_b must be "
                "declared together; one half of a coefficient pair "
                "prices no pressure"
            )
        if declared_literals:
            arrays = {}
            for name in ("hybrid_a", "hybrid_b"):
                raw = vertical[name]
                if not isinstance(raw, list) or not raw:
                    raise ValueError(
                        f"vertical.{name} must be a non-empty numeric list"
                    )
                arrays[name] = [
                    _number(value, f"vertical.{name}[{index}]")
                    for index, value in enumerate(raw)
                ]
            if len(arrays["hybrid_a"]) != len(arrays["hybrid_b"]):
                raise ValueError(
                    "vertical.hybrid_a and vertical.hybrid_b must have "
                    f"the same length; got {len(arrays['hybrid_a'])} and "
                    f"{len(arrays['hybrid_b'])}"
                )
            if any(not math.isfinite(value) or value < 0.0
                   for value in arrays["hybrid_a"]):
                raise ValueError(
                    "vertical.hybrid_a must be finite and non-negative (Pa)"
                )
            if any(not math.isfinite(value) or not 0.0 <= value <= 1.0
                   for value in arrays["hybrid_b"]):
                raise ValueError(
                    "vertical.hybrid_b must be finite within [0, 1]"
                )
            if numeric_levels:
                count = len(arrays["hybrid_a"])
                nlevels = len(numeric_levels)
                if count not in (nlevels + 1, nlevels):
                    raise ValueError(
                        "hybrid coefficient count mismatch: "
                        f"vertical.hybrid_a declares {count} "
                        f"coefficients; {nlevels} declared levels accept "
                        f"{nlevels + 1} (half-level interfaces) or "
                        f"{nlevels} (full levels)"
                    )

    time = _object(
        coordinates["time"],
        "mapping.coordinates.time",
        allowed={"kind", "selector", "units", "calendar", "lead"},
        required={"kind"},
    )
    if time["kind"] == "dimension":
        if source_format != "netcdf":
            raise ValueError("dimension time coordinates are only supported for NetCDF")
        for name in ("selector", "units"):
            if name not in time:
                raise ValueError(f"time dimension is missing {name}")
        _validate_dimension_selector(time["selector"], "time.selector")
        _string(time["units"], "time.units")
        if time.get("calendar") is not None:
            _string(time["calendar"], "time.calendar")
        if time.get("lead") is not None:
            # A forecast file written as (reference time, lead time): the
            # time coordinate is the initialization, this axis the offset.
            lead = _object(
                time["lead"], "mapping.coordinates.time.lead",
                allowed={"selector", "units"}, required={"selector", "units"},
            )
            _validate_dimension_selector(lead["selector"], "time.lead.selector")
            _lead_unit_seconds(_string(lead["units"], "time.lead.units"))
    elif time.get("lead") is not None:
        raise ValueError("time.lead requires a NetCDF dimension time coordinate")
    elif time["kind"] != "embedded_metadata":
        raise ValueError(f"unsupported time coordinate kind {time['kind']!r}")
    if source_format == "netcdf" and time["kind"] != "dimension":
        raise ValueError("NetCDF mappings require a dimension time coordinate")

    member = coordinates.get("member")
    if member is not None:
        member = _object(
            member,
            "mapping.coordinates.member",
            allowed={"kind", "selector"},
            required={"kind"},
        )
        if member["kind"] == "dimension":
            if source_format != "netcdf" or "selector" not in member:
                raise ValueError("member dimension requires NetCDF and a selector")
            _validate_dimension_selector(member["selector"], "member.selector")
        elif member["kind"] != "embedded_metadata":
            raise ValueError(f"unsupported member coordinate kind {member['kind']!r}")
        if source_format == "netcdf" and member["kind"] != "dimension":
            raise ValueError("NetCDF member coordinates must use a dimension")

    fields = mapping["fields"]
    if not isinstance(fields, dict) or not fields:
        raise ValueError("mapping.fields must be a non-empty object")
    for name, raw_field in fields.items():
        _string(name, "field name")
        field = _object(
            raw_field,
            f"fields.{name}",
            allowed={
                "selectors", "derivation", "units", "source_axes", "target_axes",
                "location", "staggering", "missing", "selector_stack_axis",
                "time_binding", "provider", "when_absent", "dependency_only",
                "published_levels", "repeated_record", "optional",
            },
            required={"units", "source_axes", "target_axes", "location", "missing"},
        )
        if field.get("repeated_record", "last") != "last":
            raise ValueError(
                f"fields.{name}.repeated_record must be 'last'; it declares "
                "that a record the publisher writes twice at one level and "
                "valid time is read from its later message")
        # ``dependency_only``: the field is decoded for a derivation to
        # read and is not published on the frame (``_published_fields``).
        if not isinstance(field.get("dependency_only", False), bool):
            raise ValueError(f"fields.{name}.dependency_only must be true or false")
        selectors = field.get("selectors", [])
        if not isinstance(selectors, list):
            raise TypeError(f"fields.{name}.selectors must be a list")
        direct = bool(selectors)
        derived = isinstance(field.get("derivation"), str) and bool(field["derivation"])
        if not isinstance(field.get("optional", False), bool):
            raise ValueError(f"fields.{name}.optional must be true or false")
        if field.get("optional") and (source_format != "netcdf" or not direct):
            raise ValueError(f"fields.{name}.optional requires a directly selected NetCDF field")
        provider = field.get("provider")
        if provider is not None:
            # The third provider shape: the field's canonical contract is
            # declared here, but its VALUES come from another packaged
            # source through a cross-source composition binding.  The gap
            # is a declaration, never an absence -- and a mapping that
            # declares one cannot decode alone (refused at frame
            # materialization by name).
            if provider != "composition_bound":
                raise ValueError(
                    f"fields.{name}.provider={provider!r} is not "
                    "'composition_bound', the only declared external provider"
                )
            if direct or derived:
                raise ValueError(
                    f"fields.{name} is composition_bound and must not also "
                    "declare selectors or a derivation; a field has exactly "
                    "one provider"
                )
            if field.get("time_binding") is not None \
                    or field.get("selector_stack_axis") is not None:
                raise ValueError(
                    f"fields.{name} is composition_bound; its time and layer "
                    "shape belong to the contributing source's own mapping"
                )
        elif direct == derived:
            raise ValueError(
                f"fields.{name} must declare selectors or one derivation, exclusively"
            )
        # A producer that publishes an invariant surface identity (land
        # mask, orography, an ice analysis) once per cycle rather than at
        # every forecast hour declares that shape here; frame assembly
        # broadcasts the record to every dependent valid time after
        # proving it byte-identical wherever it does appear.  The default
        # binding is by valid time: absence at any valid time refuses.
        time_binding = field.get("time_binding")
        if time_binding is not None:
            binding = _string(time_binding, f"fields.{name}.time_binding")
            if binding not in {"valid_time", "cycle_invariant"}:
                raise ValueError(
                    f"fields.{name}.time_binding={binding!r} is not one of "
                    "'valid_time' or 'cycle_invariant'; interval statistics "
                    "and accumulations are not bindable initialization state"
                )
            if binding == "cycle_invariant":
                if source_format == "netcdf":
                    raise ValueError(
                        f"fields.{name}.time_binding='cycle_invariant' is a "
                        "GRIB frame-assembly capability; the NetCDF mapped "
                        "decoder binds time through coordinate variables and "
                        "has no cycle-invariant broadcast, so declaring it "
                        "there would silently do nothing"
                    )
                coordinate_height = (
                    field["source_axes"] == ["half_level", "y", "x"]
                    and field["target_axes"] == ["half_level", "y", "x"]
                    and field["units"].get("target") == "m"
                    and interfaces is not None
                )
                if not direct or (field["location"] != "surface" and not coordinate_height):
                    raise ValueError(
                        f"fields.{name}.time_binding='cycle_invariant' is "
                        "restricted to directly selected surface fields or interface heights: a "
                        "derived field takes its time from its dependencies, "
                        "and a 3-D or soil state declared invariant would "
                        "silently freeze prognostic state across the cycle"
                    )
        for index, selector in enumerate(selectors):
            _validate_selector(selector, source_format, f"fields.{name}.selectors[{index}]")
        if "half_level" in field["source_axes"] or "half_level" in field["target_axes"]:
            if interfaces is None or not direct:
                raise ValueError(f"fields.{name} half_level requires direct data and interface_levels")
        selector_stack_axis = field.get("selector_stack_axis")
        if selector_stack_axis is not None:
            if source_format != "netcdf" or not direct:
                raise ValueError(
                    f"fields.{name}.selector_stack_axis requires a direct NetCDF field"
                )
            selector_stack_axis = _string(
                selector_stack_axis, f"fields.{name}.selector_stack_axis"
            )
            if selector_stack_axis not in {"soil", "vertical"}:
                raise ValueError(
                    f"fields.{name}.selector_stack_axis supports 'soil' and 'vertical'"
                )
            if len(selectors) < 2 and not (selector_stack_axis == "vertical" and selectors
                    and field["missing"]["kind"] == "composition_fill"):
                raise ValueError(
                    f"fields.{name}.selector_stack_axis requires multiple selectors"
                )
            if selector_stack_axis == "vertical":
                # One variable per level (Earth2Studio's ``t50`` .. ``t1000``):
                # each selector names the level it IS, and the stack is the
                # declared ladder in the declared order.
                stacked = [selector.get("level_value") for selector in selectors]
                if any(isinstance(value, bool) or not isinstance(value, (int, float))
                       for value in stacked):
                    raise ValueError(
                        f"fields.{name} stacks on vertical, and every selector "
                        "needs a numeric level_value"
                    )
                declared = [float(value) for value in vertical.get("levels", [])]
                sparse = (field["missing"]["kind"] == "composition_fill"
                          and len(set(stacked)) == len(stacked)
                          and all(value in declared for value in stacked))
                if [float(value) for value in stacked] != declared and not sparse:
                    raise ValueError(
                        f"fields.{name} stacked level_values {stacked} differ from "
                        f"coordinates.vertical.levels {vertical.get('levels', [])}"
                    )
        elif any("level_value" in selector for selector in selectors) \
                and source_format == "netcdf":
            raise ValueError(
                f"fields.{name} selector level_value is a vertical stack member's "
                "level; declare selector_stack_axis 'vertical'"
            )
        unit = _object(
            field["units"], f"fields.{name}.units",
            allowed={"source", "target", "scale", "offset"},
            required={"source", "target"},
        )
        if not (isinstance(unit["source"], str) and not unit["source"].strip()
                and unit["target"] == "1"):
            _string(unit["source"], f"fields.{name}.units.source")
        _string(unit["target"], f"fields.{name}.units.target")
        _number(unit.get("scale", 1.0), f"fields.{name}.units.scale")
        _number(unit.get("offset", 0.0), f"fields.{name}.units.offset")
        source_axes = _axes(field["source_axes"], f"fields.{name}.source_axes")
        _axes(field["target_axes"], f"fields.{name}.target_axes")
        if selector_stack_axis is not None and selector_stack_axis not in source_axes:
            raise ValueError(
                f"fields.{name}.selector_stack_axis is absent from source_axes"
            )
        if field["location"] not in _LOCATIONS:
            raise ValueError(f"fields.{name} has unsupported location")
        if field.get("staggering", "none") not in _STAGGERING:
            raise ValueError(f"fields.{name} has unsupported staggering")
        missing = _object(
            field["missing"], f"fields.{name}.missing",
            allowed={"kind", "name", "value"}, required={"kind"},
        )
        if missing["kind"] == "attribute":
            if source_format != "netcdf" or not missing.get("name"):
                raise ValueError(
                    f"fields.{name} attribute missing policy requires NetCDF attribute name"
                )
        elif missing["kind"] == "value":
            _number(
                missing.get("value"),
                f"fields.{name} value missing policy",
            )
        elif missing["kind"] == "composition_fill":
            if selector_stack_axis != "vertical" or source_format != "netcdf":
                raise ValueError(f"fields.{name} composition_fill requires a NetCDF vertical selector stack")
        elif missing["kind"] == "preserve_mask":
            from woof.ingest.analyzed_numbers import CANONICAL_NUMBER_FIELDS
            if (field["location"] != "soil"
                    and name not in MASKED_WATER_STATE_FIELDS
                    and name not in CANONICAL_NUMBER_FIELDS
                    and not field.get("dependency_only", False)):
                raise ValueError(
                    f"fields.{name} preserve_mask is currently restricted to "
                    "soil fields repaired by the land/water-aware initializer "
                    "and analyzed number fields repaired by the metgrid interpolation chain "
                    "and the water-state fields read over water only ("
                    + ", ".join(sorted(MASKED_WATER_STATE_FIELDS)) + ")"
                )
        elif missing["kind"] == "landmask_zero_water":
            if field["location"] != "surface" or field["target_axes"] != ["y", "x"]:
                raise ValueError(f"fields.{name} landmask_zero_water requires a surface y,x field")
        elif missing["kind"] == "landmask_water":
            # Some providers publish soil state with WATER CELLS carrying
            # sentinel values instead of an encoded missing mask (DWD-style
            # regular-grid soil is 0 K over ocean, bitmap-free).  This
            # policy masks the field to missing wherever the mapping's own
            # land_fraction says water, and the existing land/water-aware
            # repair chain runs unchanged downstream.
            if field["location"] != "soil":
                raise ValueError(
                    f"fields.{name} landmask_water is restricted to soil "
                    "fields repaired by the land/water-aware initializer"
                )
        elif missing["kind"] != "reject":
            raise ValueError(f"fields.{name} has unsupported missing policy")

    landmask_masked = sorted(
        name for name, field in fields.items()
        if field["missing"]["kind"] in ("landmask_water", "landmask_zero_water")
    )
    if landmask_masked and "land_fraction" not in fields:
        raise ValueError(
            f"fields {landmask_masked} declare the landmask_water missing "
            "policy but the mapping does not map land_fraction, so there is "
            "no mask to key water cells from"
        )

    if source_format in {"grib1", "grib2"}:
        direct_selectors: list[
            tuple[str, int, Mapping[str, object]]
        ] = []
        for field_name, field in fields.items():
            for index, selector in enumerate(field.get("selectors", [])):
                for previous_field, previous_index, previous in direct_selectors:
                    if _grib_selectors_overlap(
                        previous,
                        selector,
                        str(source_format),
                    ):
                        raise ValueError(
                            f"fields.{field_name}.selectors[{index}] overlaps "
                            f"fields.{previous_field}.selectors[{previous_index}]; "
                            "one GRIB record cannot directly provide two mapped "
                            "selector slots"
                        )
                direct_selectors.append((field_name, index, selector))
    else:
        netcdf_selectors: dict[str, tuple[str, int]] = {}
        for field_name, field in fields.items():
            for index, selector in enumerate(field.get("selectors", [])):
                identity = json.dumps(
                    selector,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                previous = netcdf_selectors.get(identity)
                if previous is not None:
                    raise ValueError(
                        f"fields.{field_name}.selectors[{index}] duplicates "
                        f"fields.{previous[0]}.selectors[{previous[1]}]; derive "
                        "aliases explicitly"
                    )
                netcdf_selectors[identity] = (field_name, index)

    if mapping.get("record_aliases") is not None:
        _validate_record_aliases(
            mapping["record_aliases"], fields, str(source_format))
    for field_name, field in fields.items():
        if field.get("when_absent") is not None:
            _validate_when_absent(
                field_name, field, fields, vertical, str(source_format))
        if field.get("published_levels") is not None:
            if (isinstance(field["published_levels"], dict)
                    and field["published_levels"].get("absent") == "nearest_upper_boundary"
                    and vertical["kind"] != "pressure"):
                raise ValueError("upper boundary extension requires pressure-level specific_humidity")
            _validate_published_levels(
                field_name, field, numeric_levels, str(source_format))
        if (field.get("repeated_record") is not None
                and str(source_format) != "grib2"):
            raise ValueError(
                f"fields.{field_name}.repeated_record is a GRIB2 record "
                "declaration; other formats have no repeated messages")

    if vertical["kind"] == "hybrid_sigma_pressure":
        pressure_field = str(vertical["surface_pressure_field"])
        if pressure_field not in fields:
            raise ValueError(
                f"vertical.surface_pressure_field names {pressure_field!r}, "
                "which mapping.fields does not declare"
            )
        pressure_units = fields[pressure_field]["units"]["target"]
        if pressure_units != "Pa":
            raise ValueError(
                f"vertical.surface_pressure_field {pressure_field!r} must "
                f"resolve in Pa; fields.{pressure_field}.units.target is "
                f"{pressure_units!r}"
            )

    derivations = mapping.get("derivations", [])
    if not isinstance(derivations, list):
        raise TypeError("mapping.derivations must be a list")
    derivation_names: set[str] = set()
    derivation_dependencies: dict[str, tuple[str, ...]] = {}
    for index, raw_derivation in enumerate(derivations):
        derivation = _object(
            raw_derivation,
            f"derivations[{index}]",
            allowed={
                "name", "operation", "source", "u", "v", "relative_humidity",
                "temperature", "pressure", "dewpoint", "geopotential", "gravity_m_s2",
                "layer_mass", "layer_bounds_m", "water_density_kg_m3",
                "specific_humidity", "surface_geopotential_height", "exclude",
                "sea_level_pressure", "level_height", "surface_height",
                "potential_temperature", "reference_pressure_pa", "kappa",
                "mixing_ratio",
            },
            required={"name", "operation"},
        )
        name = _string(derivation["name"], f"derivations[{index}].name")
        if name in derivation_names:
            raise ValueError(f"duplicate derivation {name!r}")
        derivation_names.add(name)
        operation = _string(
            derivation["operation"], f"derivations[{index}].operation"
        )
        try:
            required_arguments, optional_arguments = _DERIVATION_ARGUMENTS[operation]
        except KeyError as error:
            raise ValueError(
                f"derivations[{index}] has unsupported operation {operation!r}"
            ) from error
        supplied_arguments = set(derivation) - {"name", "operation"}
        missing_arguments = sorted(required_arguments - supplied_arguments)
        extra_arguments = sorted(
            supplied_arguments - required_arguments - optional_arguments
        )
        if missing_arguments or extra_arguments:
            raise ValueError(
                f"derivations[{index}] arguments disagree with {operation}: "
                f"missing={missing_arguments}, extra={extra_arguments}"
            )
        dependencies = []
        for argument in sorted(required_arguments):
            dependency = _string(
                derivation[argument], f"derivations[{index}].{argument}"
            )
            dependencies.append(dependency)
        if operation == "mass_fraction_rebase":
            excluded = derivation.get("exclude")
            if not isinstance(excluded, list) or not excluded or len(set(excluded)) != len(excluded):
                raise ValueError("mass_fraction_rebase requires unique excluded mass-fraction fields")
            dependencies.extend(_string(value, "mass_fraction_rebase.exclude") for value in excluded)
        if "gravity_m_s2" in derivation:
            gravity = derivation["gravity_m_s2"]
            try:
                gravity_value = _number(
                    gravity,
                    f"derivations[{index}].gravity_m_s2",
                )
            except ValueError as error:
                raise ValueError(
                    f"derivations[{index}].gravity_m_s2 must be finite and positive"
                ) from error
            if gravity_value <= 0.0:
                raise ValueError(
                    f"derivations[{index}].gravity_m_s2 must be finite and positive"
                )
        if operation == "pressure_from_vertical_coordinate" \
                and vertical["kind"] not in {"pressure", "hybrid_sigma_pressure"}:
            raise ValueError(
                "pressure_from_vertical_coordinate requires a pressure or "
                "hybrid_sigma_pressure vertical coordinate"
            )
        if operation == "geopotential_height_hydrostatic" \
                and vertical["kind"] != "hybrid_sigma_pressure":
            raise ValueError(
                "geopotential_height_hydrostatic integrates the hybrid "
                "half-level ladder, which requires a hybrid_sigma_pressure "
                "vertical coordinate"
            )
        if operation == "air_temperature_from_potential_temperature":
            for key, default, upper in (
                ("reference_pressure_pa", 100000.0, None),
                ("kappa", 2.0 / 7.0, 1.0),
            ):
                value = _number(
                    derivation.get(key, default), f"derivations[{index}].{key}"
                )
                if value <= 0.0 or (upper is not None and value >= upper):
                    raise ValueError(
                        f"derivations[{index}].{key} must be finite and positive"
                        + ("" if upper is None else f" and below {upper:g}")
                    )
        if operation == "volumetric_soil_moisture_from_layer_mass":
            bounds = derivation.get("layer_bounds_m")
            if not isinstance(bounds, list) or not bounds:
                raise ValueError(
                    f"derivations[{index}].layer_bounds_m must be a non-empty "
                    "list of [top_m, bottom_m] pairs"
                )
            parsed_bounds = []
            for pair_index, pair in enumerate(bounds):
                if not isinstance(pair, list) or len(pair) != 2:
                    raise ValueError(
                        f"derivations[{index}].layer_bounds_m[{pair_index}] "
                        "must be a [top_m, bottom_m] pair"
                    )
                top = _number(
                    pair[0],
                    f"derivations[{index}].layer_bounds_m[{pair_index}][0]",
                )
                bottom = _number(
                    pair[1],
                    f"derivations[{index}].layer_bounds_m[{pair_index}][1]",
                )
                if bottom <= top or top < 0.0:
                    raise ValueError(
                        f"derivations[{index}].layer_bounds_m[{pair_index}] "
                        "must satisfy 0 <= top < bottom"
                    )
                if parsed_bounds and top < parsed_bounds[-1][1]:
                    raise ValueError(
                        f"derivations[{index}].layer_bounds_m must be "
                        "ascending and non-overlapping"
                    )
                parsed_bounds.append((top, bottom))
            density = _number(
                derivation.get("water_density_kg_m3", 1000.0),
                f"derivations[{index}].water_density_kg_m3",
            )
            if density <= 0.0:
                raise ValueError(
                    f"derivations[{index}].water_density_kg_m3 must be "
                    "finite and positive"
                )
        derivation_dependencies[name] = tuple(dependencies)

    field_derivations: dict[str, str] = {}
    for field_name, field in fields.items():
        derivation_name = field.get("derivation")
        if derivation_name is None:
            continue
        if derivation_name not in derivation_names:
            raise ValueError(
                f"field {field_name!r} names unknown derivation {derivation_name!r}"
            )
        field_derivations[field_name] = str(derivation_name)
    for derivation_name, dependencies in derivation_dependencies.items():
        missing_dependencies = sorted(set(dependencies) - set(fields))
        if missing_dependencies:
            raise ValueError(
                f"derivation {derivation_name!r} depends on unmapped fields "
                f"{missing_dependencies}"
            )

    complete: set[str] = set()

    def visit(field_name: str, active: list[str]) -> None:
        if field_name in active:
            start = active.index(field_name)
            cycle = active[start:] + [field_name]
            raise ValueError("derived fields form a cycle: " + " -> ".join(cycle))
        if field_name in complete or field_name not in field_derivations:
            return
        active.append(field_name)
        derivation_name = field_derivations[field_name]
        for dependency in derivation_dependencies[derivation_name]:
            visit(dependency, active)
        active.pop()
        complete.add(field_name)

    for field_name in field_derivations:
        visit(field_name, [])

    target = _object(
        mapping["target"],
        "mapping.target",
        allowed={
            "name", "physics_suite", "max_dom", "require_lateral_boundaries",
            "target_vertical_levels", "soil_layer_count", "boundary_interval_seconds",
            BOUNDARY_MULTIPLES_KEY,
            "required_fields", "pressure_requirement", "policy_controlled_fields",
            "initialization_policies", "pending_composition_requirements",
        },
        required={
            "name", "physics_suite", "max_dom", "require_lateral_boundaries",
            "required_fields",
        },
    )
    _string(target["name"], "target.name")
    _string(target["physics_suite"], "target.physics_suite")
    if not isinstance(target["max_dom"], int) or isinstance(target["max_dom"], bool) \
            or target["max_dom"] <= 0:
        raise ValueError("target.max_dom must be a positive integer")
    if not isinstance(target["require_lateral_boundaries"], bool):
        raise ValueError("target.require_lateral_boundaries must be boolean")
    # Some producers publish a complete atmosphere and NO land surface at
    # all (no soil, no mask, no orography, no skin state).  Such a source
    # is decodable but cannot initialize a WRF-like model until a
    # cross-source composition supplies the missing state, so its target
    # names that state here instead of mapping fields it does not have.
    # Only canonical requirements may be declared pending, a pending name
    # must not also be mapped or required, and the declaration never
    # relaxes anything for a mapping that does not make it.
    pending_raw = target.get("pending_composition_requirements", [])
    if not isinstance(pending_raw, list):
        raise TypeError(
            "target.pending_composition_requirements must be a list")
    pending_names: list[str] = []
    for index, raw_name in enumerate(pending_raw):
        label = f"target.pending_composition_requirements[{index}]"
        value = _string(raw_name, label)
        if value not in _CANONICAL_REQUIREMENTS:
            raise ValueError(
                f"{label}={value!r} is not a canonical requirement; only "
                "canonical initialization state can be declared pending an "
                "external composition"
            )
        if value in pending_names:
            raise ValueError(f"{label} repeats {value!r}")
        if value in fields:
            raise ValueError(
                f"target declares {value!r} pending external composition, "
                "but the mapping also maps it; a field cannot be both "
                "supplied here and awaited from elsewhere"
            )
        pending_names.append(value)
    pending = frozenset(pending_names)
    soil_pending = {"soil_temperature", "volumetric_soil_moisture"} & pending
    if soil_pending and len(soil_pending) != 2:
        raise ValueError(
            "soil_temperature and volumetric_soil_moisture are one soil "
            "column: declare both pending or neither, and set "
            "target.soil_layer_count to 0 exactly when both are pending"
        )
    for key in ("target_vertical_levels", "soil_layer_count"):
        if target.get(key) is None:
            raise ValueError(f"target.{key} is required")
    _integer(
        target["target_vertical_levels"],
        "target.target_vertical_levels", minimum=1,
    )
    soil_column_pending = len(soil_pending) == 2
    _integer(
        target["soil_layer_count"], "target.soil_layer_count",
        minimum=0 if soil_column_pending else 1,
    )
    if soil_column_pending and int(target["soil_layer_count"]) != 0:
        raise ValueError(
            "target.soil_layer_count must be 0 while the soil column is "
            "declared pending an external composition; a nonzero count "
            "promises layers this mapping cannot stack"
        )
    boundary_interval = target.get("boundary_interval_seconds")
    if target["require_lateral_boundaries"]:
        if boundary_interval is None:
            raise ValueError(
                "target.boundary_interval_seconds is required for lateral boundaries"
            )
        _integer(
            boundary_interval, "target.boundary_interval_seconds", minimum=1
        )
    elif boundary_interval is not None:
        _integer(
            boundary_interval, "target.boundary_interval_seconds", minimum=1
        )
    multiples = target.get(BOUNDARY_MULTIPLES_KEY, False)
    if not isinstance(multiples, bool):
        raise ValueError(f"target.{BOUNDARY_MULTIPLES_KEY} must be boolean")
    if multiples and boundary_interval is None:
        raise ValueError(
            f"target.{BOUNDARY_MULTIPLES_KEY} needs "
            "target.boundary_interval_seconds: the multiples it accepts are "
            "multiples of that declared spacing")
    requirements = target["required_fields"]
    if not isinstance(requirements, list):
        raise TypeError("target.required_fields must be a list")
    requirement_names: set[str] = set()
    requirement_contracts: dict[str, tuple[tuple[str, ...], str, str]] = {}
    for index, raw_requirement in enumerate(requirements):
        requirement = _object(
            raw_requirement,
            f"target.required_fields[{index}]",
            allowed={"name", "axes", "location", "target_units"},
            required={"name", "axes", "location", "target_units"},
        )
        field_name = _string(requirement["name"], "requirement.name")
        if field_name in requirement_names:
            raise ValueError(f"target repeats required field {field_name!r}")
        requirement_names.add(field_name)
        if field_name not in fields:
            raise ValueError(f"target requires unmapped field {field_name!r}")
        requirement_axes = _axes(
            requirement["axes"], f"target.required_fields[{index}].axes"
        )
        if requirement["location"] not in _LOCATIONS:
            raise ValueError(f"target requirement {field_name!r} has unsupported location")
        requirement_units = _string(
            requirement["target_units"],
            f"target.required_fields[{index}].target_units",
        )
        requirement_contracts[field_name] = (
            requirement_axes, str(requirement["location"]), requirement_units
        )
        mapped = fields[field_name]
        if mapped.get("optional"):
            raise ValueError(f"required field {field_name} cannot be optional")
        if requirement_axes != tuple(mapped["target_axes"]):
            raise ValueError(f"target axes disagree for {field_name}")
        if requirement["location"] != mapped["location"]:
            raise ValueError(f"target location disagrees for {field_name}")
        if requirement["target_units"] != mapped["units"]["target"]:
            raise ValueError(f"target units disagree for {field_name}")
    # A required field held off the stream would be written nowhere while
    # every check that it was derived still passed, so the reader would
    # meet a frame without it.
    withheld_required = sorted(
        requirement_names - set(_published_fields(mapping)))
    if withheld_required:
        raise ValueError(
            f"fields {withheld_required} are required by the target and "
            "marked dependency_only; a frame must publish every required "
            "field")
    pending_required = sorted(pending & requirement_names)
    if pending_required:
        raise ValueError(
            f"target.required_fields lists {pending_required}, which the "
            "target also declares pending external composition; a "
            "requirement cannot be awaited and required of this mapping "
            "at once"
        )
    for field_name, expected in _CANONICAL_REQUIREMENTS.items():
        if field_name in pending:
            continue
        if requirement_contracts.get(field_name) != expected:
            raise ValueError(
                f"target canonical requirement {field_name!r} must be "
                f"axes={expected[0]}, location={expected[1]!r}, units={expected[2]!r}"
            )
    pressure_requirement = target.get(
        "pressure_requirement", "air_pressure_or_hybrid_coordinate"
    )
    if pressure_requirement not in {
        "air_pressure", "hybrid_coordinate", "air_pressure_or_hybrid_coordinate",
    }:
        raise ValueError(f"unsupported target.pressure_requirement {pressure_requirement!r}")
    has_pressure = "air_pressure" in fields
    has_hybrid = vertical["kind"] == "hybrid_sigma_pressure"
    pressure_satisfied = {
        "air_pressure": has_pressure,
        "hybrid_coordinate": has_hybrid,
        "air_pressure_or_hybrid_coordinate": has_pressure or has_hybrid,
    }[pressure_requirement]
    if not pressure_satisfied:
        raise ValueError(
            f"target pressure requirement {pressure_requirement!r} is not satisfied"
        )
    controlled = target.get("policy_controlled_fields", [])
    policies = target.get("initialization_policies", {})
    if not isinstance(controlled, list) or not isinstance(policies, dict):
        raise TypeError("target policy fields/policies have invalid types")
    if len(set(controlled)) != len(controlled):
        raise ValueError("target policy_controlled_fields contains duplicates")
    omitted = _POLICY_FIELDS - set(controlled)
    if omitted:
        raise ValueError(f"target omits canonical policy-controlled fields {sorted(omitted)}")
    for name in controlled:
        _string(name, "target policy-controlled field")
        if name not in fields and not (
            isinstance(policies.get(name), str) and policies[name].strip()
        ):
            raise ValueError(f"absent policy-controlled field {name!r} has no policy")
    for name, policy in policies.items():
        _string(name, "target initialization policy field")
        _string(policy, f"target.initialization_policies.{name}")
    return mapping


def _selector_name_field(value: object, label: str) -> str | list[str]:
    """A selector name is one spelling, or an ordered list of accepted ones.

    A list is not an alias table for different variables: every spelling must
    denote the SAME variable, so that a descriptor survives a producer's
    rename without being rewritten.  Duplicates are refused because they hide
    a typo, and an empty list is refused because it silently accepts nothing.
    """

    if isinstance(value, str):
        return _string(value, label)
    if not isinstance(value, list) or not value:
        raise ValueError(
            f"{label} must be a variable name, or a non-empty list of "
            "accepted spellings of the same variable"
        )
    names = [_string(entry, f"{label}[{index}]")
             for index, entry in enumerate(value)]
    if len(set(names)) != len(names):
        raise ValueError(f"{label} repeats an accepted spelling")
    return names


def _validate_dimension_selector(value: object, label: str) -> dict[str, object]:
    selector = _object(
        value, label, allowed={"name", "standard_name"}
    )
    if selector.get("name") is not None:
        selector["name"] = _selector_name_field(selector["name"], f"{label}.name")
    if selector.get("standard_name") is not None:
        _string(selector["standard_name"], f"{label}.standard_name")
    if selector.get("name") is None and selector.get("standard_name") is None:
        raise ValueError(f"{label} needs name and/or standard_name")
    return selector


@dataclass(frozen=True)
class CanonicalField:
    name: str
    units: str
    axes: tuple[str, ...]
    location: str
    staggering: str
    values: np.ndarray
    missing_count: int
    source_references: tuple[str, ...]

    def __post_init__(self) -> None:
        array = np.asarray(self.values, dtype=np.float64)
        self._validate_values(array)
        copied = array.copy()
        copied.setflags(write=False)
        object.__setattr__(self, "values", copied)

    def _validate_values(self, array) -> None:
        if array.ndim != len(self.axes):
            raise ValueError(
                f"{self.name} rank {array.ndim} differs from axes {self.axes}"
            )
        if np.isinf(array).any():
            raise ValueError(f"{self.name} contains infinity")
        if int(np.isnan(array).sum()) != self.missing_count:
            raise ValueError(f"{self.name} missing count does not match its data")

    @classmethod
    def _take_owned_stream_array(cls, *, name, units, axes, location, staggering,
                                 values, missing_count, source_references):
        """Internal move from the frameset reader's fresh, verified allocation.

        The caller relinquishes its array and all views after this call. Public
        construction still copies borrowed inputs. A window reader must first
        verify the complete original field, then pass its separately declared
        retained shape and missing count. This never permits a memory map,
        shared buffer, or skipped canonical validation.
        """
        if (not isinstance(values, np.ndarray) or values.dtype != np.float64
                or not values.flags.owndata or not values.flags.c_contiguous
                or values.base is not None):
            raise ValueError("stream transfer requires an owning contiguous float64 array")
        result = object.__new__(cls)
        for key, value in (("name", name), ("units", units), ("axes", axes),
                           ("location", location), ("staggering", staggering),
                           ("values", values), ("missing_count", missing_count),
                           ("source_references", source_references)):
            object.__setattr__(result, key, value)
        result._validate_values(values)
        values.setflags(write=False)
        return result


def _validate_mapped_field_grid(field, ny: int, nx: int, nz: int) -> None:
    """One canonical field must share its owning frame's axes."""
    _validate_mapped_field_shape(field.name, field.axes, field.values.shape, ny, nx, nz)


def _validate_mapped_field_shape(name, axes, dimensions, ny, nx, nz):
    shape = dict(zip(axes, dimensions))
    if shape.get("y") != ny or shape.get("x") != nx:
        raise ValueError(f"{name} does not share the source horizontal grid")
    if "half_level" in shape and shape["half_level"] != nz + 1:
        raise ValueError(f"{name} does not bound the vertical coordinate")
    if "vertical" in shape and shape["vertical"] != nz:
        raise ValueError(f"{name} does not share the vertical coordinate")


@dataclass(frozen=True)
class MappedSourceFrame:
    valid_time: datetime
    member: str | None
    source_cycle: datetime
    latitude: np.ndarray
    longitude: np.ndarray
    vertical_kind: str
    vertical_units: str
    vertical_values: np.ndarray
    fields: Mapping[str, CanonicalField]
    mapping_sha256: str
    input_sha256: Mapping[str, str]
    grid_fingerprint: str
    header: SourceFrameHeader

    def __post_init__(self) -> None:
        if self.valid_time.tzinfo is not None or self.source_cycle.tzinfo is not None:
            raise ValueError("mapped source times must be naive UTC")
        if self.valid_time < self.source_cycle:
            raise ValueError("mapped valid time precedes source cycle")
        latitude = np.asarray(self.latitude, dtype=np.float64)
        longitude = np.asarray(self.longitude, dtype=np.float64)
        vertical = np.asarray(self.vertical_values, dtype=np.float64)
        for name, axis in (("latitude", latitude), ("longitude", longitude),
                           ("vertical", vertical)):
            if axis.ndim != 1 or axis.size == 0 or not np.isfinite(axis).all():
                raise ValueError(f"mapped {name} coordinate must be finite non-empty 1-D")
        if latitude.size < 2 or longitude.size < 2:
            raise ValueError("mapped horizontal grid must be at least 2x2")
        for name, axis in (("latitude", latitude), ("longitude", longitude)):
            difference = np.diff(axis)
            if not (np.all(difference > 0.0) or np.all(difference < 0.0)):
                raise ValueError(f"mapped {name} coordinate is not strictly monotonic")
            if not np.allclose(difference, difference[0], rtol=1e-10, atol=1e-12):
                raise ValueError(
                    f"mapped {name} coordinate is not a regular axis; "
                    "curvilinear export is not enabled"
                )
        copied_fields = MappingProxyType(dict(self.fields))
        for field in copied_fields.values():
            _validate_mapped_field_grid(field, latitude.size, longitude.size,
                                        vertical.size)
        for name, value in (("latitude", latitude), ("longitude", longitude),
                            ("vertical_values", vertical)):
            copied = value.copy()
            copied.setflags(write=False)
            object.__setattr__(self, name, copied)
        object.__setattr__(self, "fields", copied_fields)
        object.__setattr__(self, "input_sha256", MappingProxyType(dict(self.input_sha256)))
        validate_source_frame(self.header)


@dataclass(frozen=True)
class _DirectValue:
    name: str
    valid_time: datetime
    member: str | None
    source_cycle: datetime
    axes: tuple[str, ...]
    values: np.ndarray
    missing_count: int
    references: tuple[str, ...]


@dataclass(frozen=True)
class _DecodedCollection:
    latitude: np.ndarray
    longitude: np.ndarray
    vertical_values: np.ndarray
    direct: Mapping[tuple[datetime, str | None, str], _DirectValue]
    source_cycles: Mapping[tuple[datetime, str | None], datetime]
    grid_fingerprint: str
    #: Resolved hybrid A (Pa) / B coefficients for a hybrid_sigma_pressure
    #: vertical: N+1 half-level interfaces or N full-level values, top of
    #: the atmosphere first.  None on every other vertical kind.
    hybrid_a: np.ndarray | None = None
    hybrid_b: np.ndarray | None = None
    #: Selected records read through ``mapping.record_aliases``, counted
    #: per field they answer; ``None`` when none were.
    record_aliases: Mapping[str, int] | None = None


#: How a selector was satisfied.  Reported, never inferred silently.
NC_EVIDENCE_NAME = "name"
NC_EVIDENCE_STANDARD_NAME = "standard_name"


def _selector_names(selector: Mapping[str, object]) -> tuple[str, ...]:
    """Accepted spellings for a selector's variable name.

    A producer renaming a variable is not a new variable, so a descriptor may
    list every spelling it accepts.  ECMWF's `level` -> `pressure_level` and
    `time` -> `valid_time` are documented legacy-to-new renames, and files of
    both vintages are still in the wild.
    """

    configured = selector.get("name")
    if configured is None:
        return ()
    if isinstance(configured, str):
        return (configured,)
    return tuple(str(value) for value in configured)


def _match_nc_variables(
    dataset: netcdf_bridge.Dataset, selector: Mapping[str, object]
) -> list[tuple[object, str]]:
    """Resolve one selector, returning each match with the evidence for it.

    Requiring BOTH the name and the standard name to match defeats the whole
    purpose of a CF standard name.  The standard name is the stable identity;
    the variable name is a label the producer is free to change.  Under the
    old conjunction a correctly self-describing file broke on a pure rename --
    which is exactly what a current Copernicus delivery does: `pressure_level`
    still declares `standard_name = air_pressure`, and that could not rescue
    it.

    Precedence: an explicitly configured name that matches is authoritative.
    Only when no configured spelling matches does the standard name resolve
    the variable, and the caller reports that rescue rather than taking it
    silently, because it means the descriptor has drifted from the producer.
    """

    accepted_names = _selector_names(selector)
    expected_standard = selector.get("standard_name")

    by_name = [
        variable
        for name, variable in dataset.variables.items()
        if name in accepted_names and _attributes_match(variable, selector)
    ]
    if by_name:
        return [(variable, NC_EVIDENCE_NAME) for variable in by_name]

    if expected_standard is None:
        return []
    return [
        (variable, NC_EVIDENCE_STANDARD_NAME)
        for _name, variable in dataset.variables.items()
        if getattr(variable, "standard_name", None) == expected_standard
        and _attributes_match(variable, selector)
    ]


def _nc_selector_text(selector: Mapping[str, object]) -> str:
    parts = []
    names = _selector_names(selector)
    if names:
        parts.append(
            f"name={names[0]!r}" if len(names) == 1
            else f"name in {list(names)!r}"
        )
    if selector.get("standard_name") is not None:
        parts.append(f"standard_name={selector['standard_name']!r}")
    text = " or ".join(parts) if parts else "<empty selector>"
    attributes = selector.get("attributes")
    if attributes:
        text += " with " + ", ".join(
            f"{key}={value!r}" for key, value in sorted(attributes.items())
        )
    if "layer_value" in selector:
        text += (
            f" at {selector['layer_dimension']}={selector['layer_value']}"
            f" {selector['layer_units']}"
        )
    return text


def _apply_producer_attributes(dataset, mapping: Mapping[str, object], source) -> None:
    """Supply the attributes a producer's published vocabulary fixes.

    Earth2Studio's NetCDF writer emits ``t850(time, lead_time, lat, lon)``
    with no ``units`` on the field and none on ``lat``/``lon``: its lexicon
    fixes them.  The mapping's ``producer_attributes`` table carries that
    vocabulary as data, keyed by exact variable name.  An attribute the
    file lacks is supplied from it; one the file carries must agree.  The
    Rust engine's ``apply_producer_attributes`` is the same rule.
    """

    for name, attributes in (mapping.get("producer_attributes") or {}).items():
        variable = dataset.variables.get(name)
        if variable is None:
            continue
        for key, declared in attributes.items():
            observed = variable._attributes.get(key)
            if observed is None:
                variable._attributes[key] = declared
            elif observed != declared:
                raise ValueError(
                    f"NetCDF {source} variable {name!r} carries {key}={observed!r}, "
                    f"and the producer vocabulary this mapping declares says {declared!r}")


def _nc_vocabulary(dataset: netcdf_bridge.Dataset, limit: int = 40) -> str:
    """What the file actually offers, for both-vocabularies refusals."""

    entries = []
    for name, variable in dataset.variables.items():
        standard_name = getattr(variable, "standard_name", None)
        entries.append(
            name if not isinstance(standard_name, str)
            else f"{name} (standard_name={standard_name})"
        )
    entries.sort()
    if len(entries) > limit:
        return ", ".join(entries[:limit]) + f", ... ({len(entries)} total)"
    return ", ".join(entries)


def _resolve_nc_variable(
    dataset: netcdf_bridge.Dataset,
    selector: Mapping[str, object],
    label: str,
    report: list[dict[str, object]] | None = None,
):
    matches = _match_nc_variables(dataset, selector)
    if len(matches) != 1:
        # Naming only the count made a mapping/producer mismatch read as a
        # data fault.  Both vocabularies go in the message: what the mapping
        # asked for, and what this file actually contains.  Two matches never
        # collapse to a guess -- both are named.
        ambiguous = (
            ""
            if len(matches) < 2
            else "\n  ambiguous, all of: " + ", ".join(
                sorted(str(variable.name) for variable, _ in matches)
            )
        )
        raise ValueError(
            f"{label} selector resolved {len(matches)} NetCDF variables; "
            f"expected exactly one.{ambiguous}\n"
            f"  mapping asked for: {_nc_selector_text(selector)}\n"
            f"  file contains: {_nc_vocabulary(dataset)}"
        )
    variable, evidence = matches[0]
    _record_nc_resolution(report, label, selector, variable, evidence)
    return variable


def _record_nc_resolution(
    report: list[dict[str, object]] | None,
    label: str,
    selector: Mapping[str, object],
    variable,
    evidence: str,
) -> None:
    """Record how a selector was satisfied, with the evidence that did it."""

    if report is None:
        return
    report.append({
        "selector": label,
        "variable": str(variable.name),
        "evidence": evidence,
        "configured_names": list(_selector_names(selector)),
        "standard_name": getattr(variable, "standard_name", None),
        # Drift, not failure: the file is self-describing and was read
        # correctly, and the descriptor names a spelling this producer no
        # longer uses.  Surfaced so a user can see it, never absorbed.
        "drifted": bool(
            evidence == NC_EVIDENCE_STANDARD_NAME and _selector_names(selector)
        ),
    })


def _warn_nc_drift(report: Sequence[Mapping[str, object]], source: Path) -> None:
    drifted = [row for row in report if row.get("drifted")]
    if not drifted:
        return
    detail = "; ".join(
        f"{row['selector']} -> {row['variable']!r} by standard_name="
        f"{row['standard_name']!r} (descriptor names {row['configured_names']})"
        for row in drifted
    )
    warn(
        f"{len(drifted)} selector(s) in {source.name} resolved by CF "
        f"standard_name, not by the name the descriptor gives: {detail}",
        why="The file is self-describing and was read correctly.  The "
            "descriptor names a spelling this producer no longer uses -- "
            "ECMWF renamed `level` to `pressure_level` and `time` to "
            "`valid_time`, for instance.  Add the current spelling to that "
            "selector's name list so the match stops depending on the "
            "standard name alone.",
    )


def _nc_coordinate_dimension(variable: object, label: str) -> str:
    dimensions = tuple(variable.dimensions)
    if len(dimensions) != 1:
        raise ValueError(f"{label} coordinate must use exactly one NetCDF dimension")
    return str(dimensions[0])


def _resolve_nc_dimension(
    dataset: netcdf_bridge.Dataset,
    selector: Mapping[str, object],
    label: str,
    report: list[dict[str, object]] | None = None,
):
    accepted_names = _selector_names(selector)
    standard = selector.get("standard_name")
    evidence = NC_EVIDENCE_NAME
    candidates = [
        (dimension_name, dataset.variables.get(dimension_name))
        for dimension_name in dataset.dimensions
        if dimension_name in accepted_names
        and dataset.variables.get(dimension_name) is not None
    ]
    if not candidates and standard is not None:
        # Same precedence as a variable selector: the configured spelling is
        # authoritative, and the CF standard name rescues a renamed one.
        evidence = NC_EVIDENCE_STANDARD_NAME
        candidates = [
            (dimension_name, dataset.variables.get(dimension_name))
            for dimension_name in dataset.dimensions
            if dataset.variables.get(dimension_name) is not None
            and getattr(
                dataset.variables[dimension_name], "standard_name", None
            ) == standard
        ]
    if len(candidates) != 1 or candidates[0][1] is None:
        offered = sorted(
            dimension_name
            + (
                ""
                if dataset.variables.get(dimension_name) is None
                else " (has coordinate variable)"
            )
            for dimension_name in dataset.dimensions
        )
        raise ValueError(
            f"{label} selector resolved {len(candidates)} coordinate "
            f"dimensions; expected one.\n"
            f"  mapping asked for: {_nc_selector_text(selector)}\n"
            f"  file dimensions: {', '.join(offered)}"
        )
    _record_nc_resolution(report, label, selector, candidates[0][1], evidence)
    return candidates[0]


def _utc_datetime(value: object, label: str) -> datetime:
    if isinstance(value, datetime):
        result = value
    else:
        try:
            result = datetime(
                int(value.year), int(value.month), int(value.day),
                int(value.hour), int(value.minute), int(value.second),
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError(f"{label} is not a Gregorian UTC datetime") from error
    if result.tzinfo is not None:
        result = result.astimezone(timezone.utc).replace(tzinfo=None)
    return result


def _nc_coordinate_values(variable, *, expected_units: str | None, label: str) -> np.ndarray:
    if expected_units is not None and getattr(variable, "units", None) != expected_units:
        raise ValueError(
            f"{label} units {getattr(variable, 'units', None)!r} differ from "
            f"mapping {expected_units!r}"
        )
    value = np.ma.asarray(variable[:])
    if np.ma.getmaskarray(value).any():
        raise ValueError(f"{label} coordinate contains missing values")
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1 or result.size == 0 or not np.isfinite(result).all():
        raise ValueError(f"{label} coordinate must be finite non-empty 1-D")
    return result


def _cf_grid_mapping_names(dataset: netcdf_bridge.Dataset) -> dict[str, list[str]]:
    """Return declared CF ``grid_mapping_name`` values and their referrers.

    A CF grid-mapping container is discovered two ways: a data variable naming
    it through the ``grid_mapping`` attribute, and a variable that carries
    ``grid_mapping_name`` directly.  Both are reported so a refusal can name
    the projection *and* the variables that claim it.
    """

    declared: dict[str, list[str]] = {}
    for name, variable in dataset.variables.items():
        container_name = getattr(variable, "grid_mapping", None)
        if isinstance(container_name, str) and container_name.strip():
            # CF-1.7 extended syntax: "crs: x y" -- the container is token 0.
            token = container_name.split(":")[0].split()[0]
            container = dataset.variables.get(token)
            projection = getattr(container, "grid_mapping_name", None) \
                if container is not None else None
            key = str(projection) if isinstance(projection, str) else f"<unresolved {token!r}>"
            declared.setdefault(key, []).append(name)
        projection = getattr(variable, "grid_mapping_name", None)
        if isinstance(projection, str) and projection.strip():
            declared.setdefault(str(projection), [])
    return declared


_CF_DEGREES_NORTH = {
    "degrees_north", "degree_north", "degree_N", "degrees_N",
    "degreeN", "degreesN",
}
_CF_DEGREES_EAST = {
    "degrees_east", "degree_east", "degree_E", "degrees_E",
    "degreeE", "degreesE",
}
_CF_PROJECTION_STANDARD_NAMES = {
    "projection_x_coordinate", "projection_y_coordinate",
    "projection_x_angular_coordinate", "projection_y_angular_coordinate",
    "grid_latitude", "grid_longitude",
}
_SUPPORTED_GRID_MAPPING_NAMES = {"latitude_longitude"}


def _require_geographic_coordinate(
    variable,
    *,
    axis: str,
    label: str,
) -> None:
    """Refuse, by name, a horizontal coordinate that is not CF geographic.

    The descriptor is the authority for *which* variable is latitude, but the
    file is the evidence for what that variable actually holds.  A projected
    dataset's only 1-D horizontal coordinates are ``x``/``y`` in metres; read
    as degrees they mis-georeference the whole forecast silently.  This refuses
    instead, and says which unit and standard name were seen.
    """

    accepted_units = _CF_DEGREES_NORTH if axis == "latitude" else _CF_DEGREES_EAST
    canonical_unit = "degrees_north" if axis == "latitude" else "degrees_east"
    units = getattr(variable, "units", None)
    standard_name = getattr(variable, "standard_name", None)
    name = getattr(variable, "name", "<unnamed>")

    if isinstance(standard_name, str) and standard_name in _CF_PROJECTION_STANDARD_NAMES:
        raise ValueError(
            f"{label} selector resolved NetCDF variable {name!r} whose CF "
            f"standard_name is {standard_name!r}; that is a projection axis, "
            f"not geographic {axis}. A file's bare projection axes carry no "
            f"projection with them, so nothing here can say which plane they "
            f"are in and reading them as degrees would georeference the "
            f"source silently wrong. A source whose grid declaration states "
            f"its projection is paired through it and needs none of this. "
            f"Regrid the source to a regular latitude/longitude grid, or "
            f"supply 1-D geographic coordinate variables with units "
            f"{canonical_unit!r}."
        )

    units_ok = isinstance(units, str) and units in accepted_units
    standard_ok = standard_name == axis
    if not (units_ok or standard_ok):
        raise ValueError(
            f"{label} selector resolved NetCDF variable {name!r} with "
            f"units={units!r} and standard_name={standard_name!r}; RW-WPS "
            f"cannot confirm it holds geographic {axis} in degrees. Declare CF "
            f"units {canonical_unit!r} (or one of {sorted(accepted_units)}) or "
            f"standard_name={axis!r} on that variable."
        )


def _require_geographic_range(values: np.ndarray, *, axis: str, label: str) -> None:
    limit = 90.0 if axis == "latitude" else 360.0
    extreme = float(np.max(np.abs(values)))
    if extreme > limit:
        raise ValueError(
            f"{label} coordinate reaches {extreme:g}, outside the valid "
            f"geographic {axis} range +/-{limit:g} degrees; the source grid is "
            f"not a regular latitude/longitude grid."
        )


def _require_geographic_horizontal(
    dataset: netcdf_bridge.Dataset,
    latitude_variable,
    longitude_variable,
    latitude: np.ndarray,
    longitude: np.ndarray,
    source: Path,
) -> None:
    declared = _cf_grid_mapping_names(dataset)
    unsupported = sorted(set(declared) - _SUPPORTED_GRID_MAPPING_NAMES)
    if unsupported:
        detail = "; ".join(
            f"{projection!r}"
            + (
                f" (used by {', '.join(sorted(declared[projection])[:6])})"
                if declared[projection] else ""
            )
            for projection in unsupported
        )
        raise ValueError(
            f"{source} declares CF grid mapping(s) {detail}. RW-WPS mapped "
            f"geographic NetCDF path requires regular latitude/longitude grids "
            f"(grid_mapping_name={sorted(_SUPPORTED_GRID_MAPPING_NAMES)[0]!r} or "
            f"no grid mapping at all). A lambert_conformal_conic source needs "
            f"a declared Lambert grid in the mapping; other projections need "
            f"regridding before this geographic path can decode them."
        )
    _require_geographic_coordinate(
        latitude_variable, axis="latitude", label="latitude"
    )
    _require_geographic_coordinate(
        longitude_variable, axis="longitude", label="longitude"
    )
    _require_geographic_range(latitude, axis="latitude", label="latitude")
    _require_geographic_range(longitude, axis="longitude", label="longitude")


def _read_nc_values(
    variable,
    policy: Mapping[str, object],
    field_name: str,
    *,
    layer: tuple[int, int] | None = None,
    vertical: tuple[int, np.ndarray] | None = None,
) -> tuple[np.ndarray, int]:
    """Read one variable, sliced to what the mapping declared.

    ``layer`` drops one addressed slice of a producer's own layer dimension
    (``(axis, index)`` in the variable's own dimension order); ``vertical``
    keeps the declared subset of the file's vertical levels (``(axis,
    indices)`` in the order that survives the layer drop).  Both happen
    BEFORE the missing-value policy runs, because a ``reject`` policy must
    judge the data the mapping actually consumes: a source that publishes
    stratospheric levels its humidity does not reach must not fail a field
    whose declared levels are all present.
    """

    raw = np.ma.asarray(variable[:])
    if layer is not None:
        raw = np.ma.take(raw, layer[1], axis=layer[0])
    if vertical is not None:
        raw = np.ma.take(raw, vertical[1], axis=vertical[0])
    mask = np.ma.getmaskarray(raw).copy()
    data = np.asarray(raw.filled(np.nan), dtype=np.float64)
    if policy["kind"] == "attribute":
        attribute = str(policy["name"])
        if attribute not in variable.ncattrs():
            raise ValueError(f"{field_name} lacks declared missing attribute {attribute!r}")
        marker = float(variable.getncattr(attribute))
        if not math.isfinite(marker):
            raise ValueError(f"{field_name} missing attribute is non-finite")
        mask |= data == marker
    mask |= ~np.isfinite(data)
    if policy["kind"] == "reject" and mask.any():
        raise ValueError(f"{field_name} contains missing/non-finite source values")
    if policy["kind"] == "value":
        data[mask] = float(policy["value"])
        mask[...] = False
    else:
        data[mask] = np.nan
    if np.isinf(data).any():
        raise ValueError(f"{field_name} contains infinity")
    return data, int(mask.sum())


def _unit_transform(values: np.ndarray, field: Mapping[str, object], name: str) -> np.ndarray:
    unit = field["units"]
    scale = float(unit.get("scale", 1.0))
    offset = float(unit.get("offset", 0.0))
    converted = np.asarray(values, dtype=np.float64) * scale + offset
    finite = np.isfinite(values)
    if not np.isfinite(converted[finite]).all():
        raise ValueError(f"unit conversion for {name} produced non-finite data")
    return converted


def _transpose_to_target(
    values: np.ndarray,
    source_axes: Sequence[str],
    target_axes: Sequence[str],
    name: str,
) -> np.ndarray:
    source = tuple(source_axes)
    target = tuple(target_axes)
    if set(source) != set(target):
        raise ValueError(
            f"{name} cannot map source axes {source} to target axes {target} without "
            "an explicit stacking dimension"
        )
    return np.transpose(values, tuple(source.index(axis) for axis in target))


def _declared_level_indices(
    declared: np.ndarray, offered: np.ndarray, source: Path,
) -> np.ndarray | None:
    """Where the mapping's declared levels sit in this file's coordinate.

    ``None`` means "the file offers exactly the declared levels", which is
    the historical case and costs nothing.  Otherwise every declared level
    must appear exactly once in the file's coordinate, and the indices that
    address them -- IN THE MAPPING'S OWN ORDER -- are returned, so every
    field read from this file is cut and ordered to the mapping's vertical
    contract rather than to the producer's file order.

    The breakage this prevents is real and is the reason it exists: one
    producer's variables need not share a level set.  NOAA's 20CRv3
    publishes temperature, height and wind on 28 pressure levels and
    specific humidity on 21 of them, top-down.  Demanding equality made the
    humidity file, which is complete for every level the mapping declares,
    unreadable beside its own siblings; guessing an alignment instead would
    silently pair a level with the wrong data, which is why the lookup is by
    VALUE and a level that appears twice, or not at all, is refused.
    """

    if np.array_equal(declared, offered):
        return None
    positions: list[int] = []
    for value in declared:
        matches = np.flatnonzero(offered == value)
        if matches.size != 1:
            raise ValueError(
                f"{source} offers vertical levels {offered.tolist()}, in which "
                f"the mapping's declared level {value} appears {matches.size} "
                f"times; every declared level must appear exactly once "
                f"(declared: {declared.tolist()})"
            )
        positions.append(int(matches[0]))
    return np.asarray(positions, dtype=np.intp)


def _layer_slice_position(
    dataset: netcdf_bridge.Dataset,
    variable,
    selector: Mapping[str, object],
    field_name: str,
) -> tuple[int, int] | None:
    """Resolve a selector's layer slice against the file's own coordinate.

    Returns ``(axis, index)`` into the variable's own dimension order, or
    ``None`` when the selector addresses the whole variable.  The index is
    found by matching the declared value on the layer coordinate VARIABLE --
    never by position -- so a producer that reorders its layers cannot
    silently swap two of them.
    """

    if "layer_value" not in selector:
        return None
    dimension = str(selector["layer_dimension"])
    dimensions = tuple(variable.dimensions)
    if dimension not in dimensions:
        raise ValueError(
            f"{field_name} selector addresses layer dimension {dimension!r}, "
            f"which variable {variable.name!r} does not use "
            f"(it uses {list(dimensions)})"
        )
    coordinate = dataset.variables.get(dimension)
    if coordinate is None:
        raise ValueError(
            f"{field_name} selector addresses layer dimension {dimension!r} by "
            "value, but that dimension carries no coordinate variable to read "
            "the value from"
        )
    values = _nc_coordinate_values(
        coordinate, expected_units=str(selector["layer_units"]),
        label=f"{field_name} layer coordinate",
    )
    wanted = float(selector["layer_value"])
    matches = np.flatnonzero(values == wanted)
    if matches.size != 1:
        raise ValueError(
            f"{field_name} selector layer_value={wanted!r} matches "
            f"{matches.size} entries of {dimension!r} "
            f"({values.tolist()}); expected exactly one"
        )
    return dimensions.index(dimension), int(matches[0])


def _decode_netcdf(mapping: Mapping[str, object], files: Sequence[Path]) -> _DecodedCollection:
    coordinates = mapping["coordinates"]
    horizontal = coordinates["horizontal"]
    vertical = coordinates["vertical"]
    time_contract = coordinates["time"]
    member_contract = coordinates.get("member")
    direct: dict[tuple[datetime, str | None, str], _DirectValue] = {}
    cycles: dict[tuple[datetime, str | None], datetime] = {}
    reference_latitude = reference_longitude = reference_vertical = None
    grid_fingerprint = None
    # Which direct fields the mapping owes, and which files have paid.  One
    # producer routinely publishes one variable per file -- NOAA's 20CRv3
    # ships thirteen -- so "absent from THIS file" is not an error; "absent
    # from every file" is, and that is what gets reported, once, with the
    # whole inventory in the message.
    owed_direct_fields = {
        name for name, field in mapping["fields"].items()
        if field.get("derivation") is None and not field.get("optional", False)
    }
    supplied_by: dict[str, Path] = {}
    # A producer that writes one variable per pressure level (``t50`` ..
    # ``t1000``) and no level coordinate stacks them on ``vertical``; the
    # mapping's declared ladder is then the vertical coordinate.
    stacks_vertical = any(
        field.get("selector_stack_axis") == "vertical"
        for field in mapping["fields"].values()
    )
    declared_ladder = np.asarray(vertical.get("levels", []), dtype=np.float64)
    if stacks_vertical and not declared_ladder.size:
        raise ValueError(
            "a field stacks per-level variables on vertical, and "
            "coordinates.vertical declares no levels")

    for source in files:
        with netcdf_bridge.open_dataset(source) as dataset:
            _apply_producer_attributes(dataset, mapping, source)
            resolution: list[dict[str, object]] = []
            latitude_variable = _resolve_nc_variable(
                dataset, horizontal["latitude"], "latitude", resolution
            )
            longitude_variable = _resolve_nc_variable(
                dataset, horizontal["longitude"], "longitude", resolution
            )
            latitude = _nc_coordinate_values(
                latitude_variable, expected_units=None, label="latitude"
            )
            longitude = _nc_coordinate_values(
                longitude_variable, expected_units=None, label="longitude"
            )
            _require_geographic_horizontal(
                dataset, latitude_variable, longitude_variable,
                latitude, longitude, source,
            )
            # The vertical coordinate is resolved LAZILY, because a file that
            # carries only surface or soil quantities has none to resolve --
            # and refusing it there would make a multi-file source
            # undecodable for a reason that is not about the data.  The
            # failure is kept, not discarded: it is raised, unchanged, the
            # moment a field with a vertical axis is actually read from this
            # file, so nothing that needs the coordinate is let through
            # without it.
            vertical_variable = None
            vertical_values = None
            vertical_error: ValueError | None = None
            try:
                vertical_variable = _resolve_nc_variable(
                    dataset, vertical["selector"], "vertical", resolution
                )
                vertical_values = _nc_coordinate_values(
                    vertical_variable,
                    expected_units=str(vertical["units"]),
                    label="vertical",
                )
            except ValueError as error:
                vertical_variable = None
                vertical_values = None
                vertical_error = error
            latitude_dimension = _nc_coordinate_dimension(
                latitude_variable, "latitude"
            )
            longitude_dimension = _nc_coordinate_dimension(
                longitude_variable, "longitude"
            )
            # Interface (half-level) coordinate, resolved the same lazy way:
            # only a field with a half_level axis needs it.
            interface_dimension = None
            interface_selection: tuple[str, np.ndarray] | None = None
            interface_error: ValueError | None = None
            if vertical.get("interface_selector") is not None:
                try:
                    interface_variable = _resolve_nc_variable(
                        dataset, vertical["interface_selector"], "vertical interfaces",
                        resolution,
                    )
                    interface_values = _nc_coordinate_values(
                        interface_variable,
                        expected_units=str(vertical["units"]),
                        label="vertical interfaces",
                    )
                    interface_dimension = _nc_coordinate_dimension(
                        interface_variable, "vertical interfaces"
                    )
                    declared_interfaces = np.asarray(
                        vertical["interface_levels"], dtype=np.float64
                    )
                    indices = _declared_level_indices(
                        declared_interfaces, interface_values, source
                    )
                    if indices is not None:
                        interface_selection = (interface_dimension, indices)
                except ValueError as error:
                    interface_dimension = None
                    interface_error = error
            vertical_dimension = None
            vertical_selection: tuple[str, np.ndarray] | None = None
            if vertical_values is not None:
                vertical_dimension = _nc_coordinate_dimension(
                    vertical_variable, "vertical"
                )
                if len({
                    latitude_dimension, longitude_dimension, vertical_dimension,
                }) != 3:
                    raise ValueError(
                        "NetCDF latitude, longitude, and vertical coordinates "
                        "must use distinct dimensions"
                    )
                explicit_levels = np.asarray(
                    vertical.get("levels", []), dtype=np.float64
                )
                if explicit_levels.size:
                    indices = _declared_level_indices(
                        explicit_levels, vertical_values, source
                    )
                    if indices is not None:
                        vertical_selection = (vertical_dimension, indices)
                        vertical_values = explicit_levels
            if stacks_vertical and vertical_values is None:
                vertical_values = declared_ladder

            time_dimension, time_variable = _resolve_nc_dimension(
                dataset, time_contract["selector"], "time", resolution
            )
            declared_time_units = str(time_contract["units"])
            if getattr(time_variable, "units", None) != declared_time_units:
                raise ValueError("NetCDF time units differ from mapping")
            calendar = str(time_contract.get("calendar") or getattr(time_variable, "calendar", "standard"))
            if calendar not in {"standard", "gregorian", "proleptic_gregorian"}:
                raise ValueError(f"calendar {calendar!r} is not supported for WRF initialization")
            # The calendar arithmetic happens in the bridge, beside the
            # decode: turning "hours since 1970-01-01" and a number into
            # an instant is reading the file's own time convention, and
            # a second implementation of it here is exactly the kind of
            # duplicate this module no longer keeps.  The bridge folds a
            # reference time's UTC offset into the epoch itself, so what
            # comes back is already the naive-UTC this module works in.
            times = tuple(
                _utc_datetime(value, "time coordinate")
                for value in time_variable.times()
            )
            if len(set(times)) != len(times):
                raise ValueError(f"NetCDF file {source} contains duplicate valid times")
            # A forecast file written as (reference time, lead time): each
            # valid time is reference + lead, and its source cycle is the
            # reference.  One reference per file, so no valid time belongs
            # to two forecasts.
            lead_dimension = None
            valid_times = [(time, time) for time in times]
            lead_contract = time_contract.get("lead")
            if lead_contract is not None:
                lead_dimension, lead_variable = _resolve_nc_dimension(
                    dataset, lead_contract["selector"], "lead time", resolution
                )
                lead_units = str(lead_contract["units"])
                if getattr(lead_variable, "units", None) != lead_units:
                    raise ValueError(
                        f"NetCDF lead time units {getattr(lead_variable, 'units', None)!r} "
                        f"differ from mapping {lead_units!r}")
                seconds = _lead_unit_seconds(lead_units)
                if len(times) != 1:
                    raise ValueError(
                        f"NetCDF file {source} holds {len(times)} reference times beside "
                        "a lead axis; pass one initialization per file so every valid "
                        "time belongs to one forecast")
                reference = times[0]
                valid_times = []
                for lead_value in np.asarray(lead_variable[:], dtype=np.float64).reshape(-1):
                    offset = float(lead_value) * seconds
                    if not math.isfinite(offset) or offset != int(offset):
                        raise ValueError(
                            f"NetCDF file {source} lead time {lead_value} {lead_units} "
                            "is not a whole number of seconds")
                    valid_times.append(
                        (reference + timedelta(seconds=int(offset)), reference))
                if len({valid for valid, _ in valid_times}) != len(valid_times):
                    raise ValueError(f"NetCDF file {source} contains duplicate lead times")

            member_dimension = None
            member_value = None
            if member_contract is not None:
                if member_contract["kind"] != "dimension":
                    raise ValueError("NetCDF member coordinate must be a dimension")
                member_dimension, member_variable = _resolve_nc_dimension(
                    dataset, member_contract["selector"], "member", resolution
                )
                member_values = np.asarray(member_variable[:]).reshape(-1)
                if member_values.size != 1:
                    raise ValueError(
                        "mapped WRF initialization requires exactly one NetCDF ensemble member"
                    )
                member_value = str(member_values[0])

            if reference_latitude is None:
                reference_latitude = latitude
                reference_longitude = longitude
            elif not (
                np.array_equal(reference_latitude, latitude)
                and np.array_equal(reference_longitude, longitude)
            ):
                raise ValueError(
                    "NetCDF source horizontal coordinates change between files"
                )
            if vertical_values is not None:
                if reference_vertical is None:
                    reference_vertical = vertical_values
                    grid_fingerprint = hashlib.sha256(
                        np.ascontiguousarray(latitude).tobytes()
                        + np.ascontiguousarray(longitude).tobytes()
                        + np.ascontiguousarray(vertical_values).tobytes()
                    ).hexdigest()
                elif not np.array_equal(reference_vertical, vertical_values):
                    raise ValueError(
                        "NetCDF source vertical coordinate changes between files"
                    )

            soil_dimension = None
            claimed_direct_variables: dict[str, str] = {}
            for field_name, field in mapping["fields"].items():
                if field.get("derivation") is not None:
                    continue
                selectors = field.get("selectors", [])
                stack_axis = field.get("selector_stack_axis")
                resolved: list[tuple[object, Mapping[str, object]]] = []
                claimed_slices: set[tuple[str, object, object]] = set()
                stacked_absent = 0
                for selector_index, selector in enumerate(selectors):
                    matched = _match_nc_variables(dataset, selector)
                    for _variable, _evidence in matched:
                        _record_nc_resolution(
                            resolution, field_name, selector,
                            _variable, _evidence,
                        )
                    candidates = [variable for variable, _ in matched]
                    if len(candidates) > 1:
                        raise ValueError(
                            f"{field_name} selector resolves multiple NetCDF variables"
                        )
                    if not candidates:
                        # A stacked field whose members live in ONE file is
                        # all-or-nothing there; a stacked field that is
                        # wholly absent belongs to another file and is
                        # accounted for after every file has been read.
                        stacked_absent += 1
                        continue
                    variable = candidates[0]
                    slice_key = (
                        str(variable.name),
                        selector.get("layer_dimension"),
                        selector.get("layer_value"),
                    )
                    if slice_key in claimed_slices:
                        if stack_axis is not None:
                            raise ValueError(
                                f"{field_name} stacked selectors resolve duplicate "
                                f"variable {variable.name!r}"
                            )
                        continue
                    claimed_slices.add(slice_key)
                    resolved.append((variable, selector))
                if not resolved:
                    # Not in THIS file.  `owed_direct_fields` remembers it,
                    # and the refusal below names every file that was read.
                    continue
                if stack_axis is not None and stacked_absent:
                    raise ValueError(
                        f"{field_name} stacked selector inventory is split "
                        f"across files: {len(resolved)} of {len(selectors)} "
                        f"members resolve in {source}"
                    )
                if stack_axis is None and len(resolved) != 1:
                    raise ValueError(
                        f"{field_name} resolves multiple NetCDF variables; rw-wps.mapping.v1 "
                        "does not yet declare how alternatives become one field"
                    )
                if stack_axis is not None and len(resolved) != len(selectors):
                    raise ValueError(
                        f"{field_name} stacked selector inventory is incomplete"
                    )
                for variable, selector in resolved:
                    claim = (
                        str(variable.name)
                        if "layer_value" not in selector
                        else f"{variable.name}[{selector['layer_dimension']}"
                             f"={selector['layer_value']}]"
                    )
                    previous = claimed_direct_variables.get(claim)
                    if previous is not None and previous != field_name:
                        raise ValueError(
                            f"NetCDF variable {claim!r} directly provides "
                            f"both {previous!r} and {field_name!r}; derive aliases "
                            "explicitly"
                        )
                    claimed_direct_variables[claim] = field_name
                declared_source_units = str(field["units"]["source"])
                source_axes = list(_axes(field["source_axes"], f"fields.{field_name}.source_axes"))
                variable_axes = list(source_axes)
                if stack_axis is not None:
                    variable_axes.remove(str(stack_axis))
                if "vertical" in variable_axes and vertical_values is None:
                    # The lazy resolution above kept the reason; a field that
                    # needs the coordinate gets it, not a vaguer one.
                    raise ValueError(
                        f"{field_name} declares a vertical axis, and {source} "
                        f"has no readable vertical coordinate: {vertical_error}"
                    )
                if "half_level" in variable_axes and interface_dimension is None:
                    raise ValueError(
                        f"{field_name} declares a half_level axis, and {source} "
                        f"has no readable interface coordinate: "
                        f"{interface_error or 'vertical.interface_selector is not declared'}"
                    )
                expected_dimensions = {
                    "vertical": vertical_dimension,
                    "half_level": interface_dimension,
                    "y": latitude_dimension,
                    "x": longitude_dimension,
                    "lead": lead_dimension,
                }
                if lead_dimension is None and "lead" in variable_axes:
                    raise ValueError(
                        f"{field_name} declares a lead axis and coordinates.time "
                        "declares no lead")
                if lead_dimension is not None \
                        and ("time" in variable_axes) != ("lead" in variable_axes):
                    raise ValueError(
                        f"{field_name} must carry both the time and the lead axis, "
                        "or neither, in a file whose valid times are reference plus lead")
                arrays = []
                reference_dimensions = None
                reference_shape = None
                for variable, selector in resolved:
                    if getattr(variable, "units", None) != declared_source_units:
                        raise ValueError(
                            f"{field_name} source units "
                            f"{getattr(variable, 'units', None)!r} differ from mapping "
                            f"{declared_source_units!r}"
                        )
                    layer = _layer_slice_position(
                        dataset, variable, selector, field_name
                    )
                    dimensions = tuple(variable.dimensions)
                    if layer is not None:
                        dimensions = dimensions[:layer[0]] + dimensions[layer[0] + 1:]
                    vertical_take = None
                    if (vertical_selection is not None
                            and vertical_selection[0] in dimensions):
                        vertical_take = (
                            dimensions.index(vertical_selection[0]),
                            vertical_selection[1],
                        )
                    elif (interface_selection is not None
                            and interface_selection[0] in dimensions):
                        vertical_take = (
                            dimensions.index(interface_selection[0]),
                            interface_selection[1],
                        )
                    data, _ = _read_nc_values(
                        variable, field["missing"], field_name,
                        layer=layer, vertical=vertical_take,
                    )
                    if len(variable_axes) != data.ndim:
                        raise ValueError(
                            f"{field_name} source_axes rank differs from NetCDF "
                            f"variable {variable.name}"
                        )
                    for axis_role, dimension_name in expected_dimensions.items():
                        if axis_role not in variable_axes:
                            continue
                        axis = variable_axes.index(axis_role)
                        if dimensions[axis] != dimension_name:
                            raise ValueError(
                                f"{field_name} {axis_role} axis does not use the declared "
                                f"coordinate dimension {dimension_name!r}"
                            )
                    if "time" in variable_axes:
                        time_axis = variable_axes.index("time")
                        if dimensions[time_axis] != time_dimension:
                            raise ValueError(
                                f"{field_name} time axis does not use the declared dimension"
                            )
                    elif len(times) != 1 and lead_dimension is None:
                        raise ValueError(
                            f"static field {field_name} is ambiguous across a "
                            "multi-time NetCDF file"
                        )
                    if "member" in variable_axes:
                        if member_dimension is None:
                            raise ValueError(
                                f"{field_name} has a member axis without a member coordinate"
                            )
                        member_axis = variable_axes.index("member")
                        if dimensions[member_axis] != member_dimension:
                            raise ValueError(
                                f"{field_name} member axis uses the wrong dimension"
                            )
                    if reference_dimensions is None:
                        reference_dimensions = dimensions
                        reference_shape = data.shape
                    elif dimensions != reference_dimensions or data.shape != reference_shape:
                        raise ValueError(
                            f"{field_name} stacked NetCDF variables have different shapes "
                            "or dimensions"
                        )
                    arrays.append(data)
                if stack_axis == "vertical":
                    offered = [selector.get("level_value") for _variable, selector in resolved]
                    if [float(value) for value in offered] != [float(value) for value in declared_ladder]:
                        raise ValueError(
                            f"{field_name} stacked level_values {offered} differ from "
                            f"coordinates.vertical.levels {list(declared_ladder)}")
                    data = np.stack(arrays, axis=source_axes.index("vertical"))
                elif stack_axis is not None:
                    expected_soil_count = mapping["target"].get("soil_layer_count")
                    if expected_soil_count is None:
                        raise ValueError(
                            f"{field_name} stacks soil without target.soil_layer_count"
                        )
                    if len(arrays) != int(expected_soil_count):
                        raise ValueError(
                            f"{field_name} has {len(arrays)} stacked soil selectors; "
                            f"target declares {expected_soil_count}"
                        )
                    current_soil_dimension = "@selector_stack:soil"
                    if soil_dimension is None:
                        soil_dimension = current_soil_dimension
                    elif soil_dimension != current_soil_dimension:
                        raise ValueError(
                            "mapped NetCDF soil fields do not share one soil dimension"
                        )
                    data = np.stack(arrays, axis=source_axes.index(str(stack_axis)))
                else:
                    data = arrays[0]
                    if "soil" in source_axes:
                        soil_axis = source_axes.index("soil")
                        current_soil_dimension = str(
                            reference_dimensions[soil_axis]
                        )
                        expected_soil_count = mapping["target"].get(
                            "soil_layer_count"
                        )
                        if expected_soil_count is None:
                            raise ValueError(
                                f"{field_name} has a soil axis without "
                                "target.soil_layer_count"
                            )
                        observed_soil_count = len(
                            dataset.dimensions[current_soil_dimension]
                        )
                        if observed_soil_count != int(expected_soil_count):
                            raise ValueError(
                                f"{field_name} soil dimension has "
                                f"{observed_soil_count} layers; target declares "
                                f"{expected_soil_count}"
                            )
                        if soil_dimension is None:
                            soil_dimension = current_soil_dimension
                        elif soil_dimension != current_soil_dimension:
                            raise ValueError(
                                "mapped NetCDF soil fields do not share one soil dimension"
                            )

                for time_index, (valid_time, source_cycle) in enumerate(valid_times):
                    selected = data
                    selected_axes = list(source_axes)
                    if lead_dimension is not None:
                        if "lead" in selected_axes:
                            axis = selected_axes.index("lead")
                            selected = np.take(selected, time_index, axis=axis)
                            selected_axes.pop(axis)
                        if "time" in selected_axes:
                            axis = selected_axes.index("time")
                            selected = np.take(selected, 0, axis=axis)
                            selected_axes.pop(axis)
                    elif "time" in selected_axes:
                        axis = selected_axes.index("time")
                        selected = np.take(selected, time_index, axis=axis)
                        selected_axes.pop(axis)
                    if "member" in selected_axes:
                        axis = selected_axes.index("member")
                        selected = np.take(selected, 0, axis=axis)
                        selected_axes.pop(axis)
                    converted = _unit_transform(selected, field, field_name)
                    target_axes = _axes(field["target_axes"], f"fields.{field_name}.target_axes")
                    converted = _transpose_to_target(
                        converted, selected_axes, target_axes, field_name
                    )
                    missing_count = int(np.isnan(converted).sum())
                    key = (valid_time, member_value, field_name)
                    value = _DirectValue(
                        name=field_name,
                        valid_time=valid_time,
                        member=member_value,
                        source_cycle=source_cycle,
                        axes=target_axes,
                        values=converted,
                        missing_count=missing_count,
                        references=tuple(
                            f"{source.resolve()}:{variable.name}"
                            for variable, _selector in resolved
                        ),
                    )
                    if key in direct:
                        raise ValueError(f"duplicate mapped field {field_name} at {valid_time}")
                    direct[key] = value
                    cycles[(valid_time, member_value)] = source_cycle
                supplied_by[field_name] = source
        # Emitted once per file, after every selector in it has resolved, so
        # the notice lists the whole drift rather than one line per variable.
        _warn_nc_drift(resolution, source)

    unsupplied = sorted(owed_direct_fields - set(supplied_by))
    if unsupplied:
        raise ValueError(
            f"mapped field(s) {unsupplied} have no matching variable in any "
            f"of the {len(files)} supplied NetCDF file(s): "
            + ", ".join(str(path) for path in files)
        )
    if reference_latitude is None:
        raise ValueError("no NetCDF source data were decoded")
    if grid_fingerprint is None:
        # Nothing here has a vertical axis -- the composition's terrain-only
        # partition is exactly this shape -- so the grid identity is the
        # horizontal one.  Any field that DID declare a vertical axis was
        # already refused above with the coordinate's own error, so this is
        # never a quiet substitution for a missing coordinate.
        reference_vertical = np.zeros(0, dtype=np.float64)
        grid_fingerprint = hashlib.sha256(
            np.ascontiguousarray(reference_latitude).tobytes()
            + np.ascontiguousarray(reference_longitude).tobytes()
        ).hexdigest()
    hybrid_a = hybrid_b = None
    if str(mapping["coordinates"]["vertical"].get("kind")) == "hybrid_sigma_pressure" \
            and reference_vertical.size:
        # NetCDF bytes carry no pv channel; a hybrid NetCDF source rides
        # entirely on the mapping's inline literals.
        hybrid_a, hybrid_b = _resolve_hybrid_coefficients(
            mapping["coordinates"]["vertical"],
            int(reference_vertical.size),
            record_pv=(),
        )
    return _DecodedCollection(
        reference_latitude,
        reference_longitude,
        reference_vertical,
        MappingProxyType(direct),
        MappingProxyType(cycles),
        grid_fingerprint,
        hybrid_a=hybrid_a,
        hybrid_b=hybrid_b,
    )


@dataclass(frozen=True)
class _GribRecord:
    source: Path
    index: int
    reference_time: datetime
    valid_time: datetime
    member: str | None
    parameter: int
    level_type: int
    level_value: float
    table_version: int | None
    center: int | None
    subcenter: int | None
    master_table_version: int | None
    local_table_version: int | None
    discipline: int | None
    category: int | None
    second_level_type: int | None
    second_level_value: float | None
    process_identity: tuple[int, int] | None
    time_semantics: tuple[int, ...]
    values: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    grid_fingerprint: str
    #: Section 4's optional coordinate list (the pv octets): half-level
    #: A (Pa) then B hybrid coefficients on model-level records.  Empty
    #: when the message carries none; always empty for GRIB1, whose
    #: vertical-coordinate parameters live in the GDS this route does
    #: not read (inline mapping literals cover GRIB1 hybrid sources).
    coordinate_values: tuple[float, ...] = ()


def _embedded_valid_time(
    reference: datetime, unit: int, amount: int, *, edition: int
) -> datetime:
    if edition == 1:
        factors = {
            0: timedelta(minutes=1), 1: timedelta(hours=1),
            2: timedelta(days=1), 10: timedelta(hours=3),
            11: timedelta(hours=6), 12: timedelta(hours=12),
            254: timedelta(seconds=1),
        }
    else:
        factors = {
            0: timedelta(minutes=1), 1: timedelta(hours=1),
            2: timedelta(days=1), 10: timedelta(hours=3),
            11: timedelta(hours=6), 12: timedelta(hours=12),
            13: timedelta(seconds=1),
        }
    if unit not in factors:
        raise ValueError(f"unsupported GRIB{edition} forecast time unit {unit}")
    return reference + amount * factors[unit]


def _grib1_records(
    source: Path, bridge: Path, *, source_label: Path | None = None,
) -> tuple[_GribRecord, ...]:
    inspect_grib1_envelopes(source)
    with tempfile.TemporaryDirectory(prefix="gpuwm-mapped-grib1-") as temporary:
        dump = Path(temporary) / "dump"
        completed = subprocess.run(
            [os.fspath(bridge), os.fspath(source), os.fspath(dump)],
            text=True, capture_output=True, check=False,
        )
        if completed.returncode:
            detail = (completed.stderr or completed.stdout).strip()
            raise RuntimeError(f"GRIB1 bridge failed for {source}: {detail}")
        metadata = _load_json_document(dump / "metadata.json", "GRIB1 metadata")
        if not isinstance(metadata, dict):
            raise ValueError("GRIB1 bridge metadata must be a JSON object")
        if metadata.get("format_version") != 1 or metadata.get("edition") != 1:
            raise ValueError("GRIB1 bridge emitted an unsupported metadata schema")
        if metadata.get("dtype") != "<f8":
            raise ValueError("GRIB1 bridge emitted an unsupported dtype")
        values = np.fromfile(dump / "values.f64", dtype="<f8")
        latitude = np.asarray(metadata.get("latitude"), dtype=np.float64)
        longitude = np.asarray(metadata.get("longitude"), dtype=np.float64)
        shape = tuple(int(item) for item in metadata.get("shape", ()))
        if shape != (latitude.size, longitude.size):
            raise ValueError("GRIB1 bridge coordinate/grid shape mismatch")
        fingerprint = hashlib.sha256(
            np.ascontiguousarray(latitude).tobytes()
            + np.ascontiguousarray(longitude).tobytes()
        ).hexdigest()
        result = []
        for index, message in enumerate(metadata.get("messages", [])):
            count = int(message["count"])
            offset = int(message["offset_values"])
            message_shape = (int(message["ny"]), int(message["nx"]))
            if message_shape != shape:
                # Small control records are valid in concatenated CDO output,
                # but they can never satisfy a meteorological mapping field.
                continue
            if int(message["scan_mode"]) & 0x20:
                raise ValueError("GRIB1 j-consecutive scanning is unsupported")
            if count != shape[0] * shape[1] or offset < 0 or offset + count > values.size:
                raise ValueError("GRIB1 bridge message points outside its payload")
            array = np.asarray(values[offset:offset + count], dtype=np.float64).reshape(shape).copy()
            reference = datetime(
                int(message["year"]), int(message["month"]), int(message["day"]),
                int(message["hour"]), int(message["minute"]),
            )
            valid = _embedded_valid_time(
                reference, int(message["time_unit"]), int(message["p1"]), edition=1
            )
            result.append(_GribRecord(
                source=(source_label or source).resolve(), index=index,
                reference_time=reference, valid_time=valid, member=None,
                parameter=int(message["parameter"]),
                level_type=int(message["level_type"]),
                level_value=float(message["level"]),
                table_version=int(message["table_version"]),
                center=int(message["center"]), subcenter=None,
                master_table_version=None, local_table_version=None,
                discipline=None, category=None,
                second_level_type=None, second_level_value=None,
                process_identity=None,
                time_semantics=(
                    int(message["time_range_indicator"]),
                    int(message["p1"]), int(message["p2"]),
                ),
                values=array,
                latitude=latitude, longitude=longitude,
                grid_fingerprint=fingerprint,
            ))
        return tuple(result)


def _grib2_tools_crate() -> Path:
    """The checkout crate the cargo fallback builds in (may not exist).

    A function rather than an inline expression so the two callers that
    must agree on it -- this resolver and ``woof doctor``'s
    ``_grib2_route_crate`` -- can be bound by tests, and so a test of
    the no-crate refusal can exercise that branch on a tree that
    carries the crate.
    """

    return Path(__file__).resolve().parents[1] / "tools" / "grib1_bridge"


def _build_grib2_tools() -> tuple[Path, Path]:
    """The GRIB2 inventory and dump executables, resolved then built.

    Ladder first, cargo second, and that order is the whole fix.  This
    used to open by shelling ``cargo build`` in
    ``<package parent>/tools/grib1_bridge`` -- a directory a wheel does
    not have -- so ``woof-mapped-inspect`` and ``woof adapt`` died on
    a raw ``NotADirectoryError: [WinError 267] The directory name is
    invalid`` in 0.3 s on every pip install, with no message, while
    ``grib2_inventory`` and ``grib2_dump`` sat staged and pin-valid in
    ``~/.woof/bridges`` and ``woof doctor`` reported both ``ok``.  A
    green light over a hole, and the two binaries the command needed
    were on the disk the whole time.

    :mod:`woof.ingest.grib` had this right already -- it builds from a
    checkout crate when there is one and falls back to
    :func:`woof.bridges.find_bridge` otherwise -- so this is that
    module's shape, not a new one.  The one deliberate difference is
    which comes first: ``find_bridge`` already puts a checkout's own
    ``target/release`` build ahead of the staged copy, so consulting
    the ladder first still gives a developer their own rebuild and
    saves everyone else a cargo invocation that cannot succeed.

    This is the resolver OF RECORD for the two tools: the mapped
    engine, ``woof adapt``, and the ``rw-wps``/``woof prep`` front
    door (every GRIB2 route there, packaged profiles included) all
    resolve through this one function, so the pre-flight and the route
    cannot give different answers.

    A ladder hit is not the whole question.  Each found binary is held
    against its static contract marker
    (:data:`woof.bridges.BRIDGE_ABI_MARKERS`), because a copy staged
    before a decoder-contract change still exists, still launches, and
    then refuses inputs woof writes correctly -- the 1.1.0 GFS
    series-file failure, one route over.  A stale hit is treated as
    unresolved: where the checkout crate exists, the cargo build below
    REBUILDS it (the checkout's ``target/release`` outranks the staged
    copy on the ladder, so the rebuild wins); where nothing here can
    rebuild, the refusal names the staleness and the remedy.

    The refusal at the end names the remedy this install can actually
    take (the staged bundle, or a clone and a build) instead of
    relaying a cargo error about a missing directory.
    """

    from woof import bridges

    found: dict[str, Path | None] = {}
    stale: dict[str, tuple[Path, str]] = {}
    for name in ("grib2_inventory", "grib2_dump"):
        # A set environment override naming a missing file raises here,
        # by design: explicit configuration fails loudly rather than
        # falling through to a build that would shadow it.
        resolved = bridges.find_bridge(name)
        if resolved is not None:
            ok, evidence = bridges.bridge_abi_matches(name, resolved)
            if not ok:
                stale[name] = (resolved, evidence)
                resolved = None
        found[name] = resolved
    if found["grib2_inventory"] is not None and found["grib2_dump"] is not None:
        return found["grib2_inventory"], found["grib2_dump"]

    crate = _grib2_tools_crate()
    if (crate / "Cargo.toml").is_file():
        # The resolved toolchain, for the reason recorded on
        # woof.bridges.cargo_executable: a non-login shell has a working
        # rustup and no cargo on PATH, and this route refused to build on
        # a machine that could.
        command = [
            bridges.cargo_executable() or "cargo",
            "build", "--locked", "--offline", "--release",
            "--bin", "grib2_inventory", "--bin", "grib2_dump",
        ]
        try:
            completed = subprocess.run(
                command, cwd=crate, text=True, capture_output=True,
                check=False
            )
        except OSError:
            raise bridges.BridgeBuildError(
                bridges.cargo_missing_refusal(
                    "grib2_inventory/grib2_dump", bridges.CRATE_RELATIVE),
                failure_class="cargo-not-installed") from None
        if completed.returncode:
            # Same classifier as the GRIB1 route -- one table, two call
            # sites, so a newly-observed build failure is named
            # identically wherever it is met.
            detail = "\n".join(
                part for part in (completed.stdout, completed.stderr) if part)
            raise bridges.BridgeBuildError(
                bridges.cargo_build_refusal(
                    "grib2_inventory/grib2_dump", bridges.CRATE_RELATIVE,
                    returncode=completed.returncode, output=detail),
                failure_class=bridges.classify_cargo_failure(detail)[0])
        suffix = ".exe" if os.name == "nt" else ""
        inventory = crate / "target" / "release" / f"grib2_inventory{suffix}"
        dump = crate / "target" / "release" / f"grib2_dump{suffix}"
        if not inventory.is_file() or not dump.is_file():
            raise RuntimeError(
                "cargo succeeded but the GRIB2 tools are missing")
        return inventory, dump

    if stale:
        # Present, launchable, and wrong -- with nothing here that can
        # rebuild it.  Distinct from "not installed" because the two
        # have different remedies (replace this one versus install one),
        # and a caller that catches "missing" to offer a build must not
        # silently swallow "wrong".
        named = "; ".join(
            f"{name} at {path} {evidence}"
            for name, (path, evidence) in sorted(stale.items()))
        missing_too = [name for name in ("grib2_inventory", "grib2_dump")
                       if found[name] is None and name not in stale]
        if missing_too:
            named += f" (and {', '.join(missing_too)} is not installed at all)"
        raise bridges.DecoderContractError(
            f"the GRIB2 tools this route resolved are stale: {named}.\n"
            + bridges.install_aware_build_hint(bridges.CARGO_BUILD_HINT)
            + "\n  # --grib2-inventory / --grib2-dump override the "
            "resolved paths.")
    absent = [name for name in ("grib2_inventory", "grib2_dump")
              if found[name] is None]
    searched = ", ".join(
        str(candidate) for candidate in bridges.bridge_candidates(absent[0]))
    raise FileNotFoundError(
        "the GRIB2 inventory/dump tools this route needs are not "
        f"installed here: {', '.join(absent)} not found, and this "
        "installation has no tools/grib1_bridge crate to build them from "
        "(pip wheels ship no compiled Rust).  Searched, in order: "
        + searched + "\n" + bridges.bridge_remedy(absent[0])
        + "\n  # `woof doctor` checks this estate."
        + "\n  # --grib2-inventory / --grib2-dump override the resolved "
        "paths.")


# ---------------------------------------------------------------------------
# Declared source-grid families.
#
# The mapping's optional top-level ``grid`` block is what makes a projected
# source (HRRR's Lambert CONUS grid, RAP, NAM) TABLE WORK instead of a
# per-model module: the family and its parameters are data, the decoder
# cross-checks every observed GRIB grid octet against the declaration, and
# one generic engine executes the result.  Absent, the historical behaviour
# is unchanged: GRIB2 export requires regular latitude/longitude GDT 0.
# ---------------------------------------------------------------------------

GRID_FAMILY_REGULAR = "regular_latitude_longitude"
GRID_FAMILY_LAMBERT = "lambert_conformal"
#: GRIB2 GDT 3.1, a regular lattice in a rotated spherical frame (ECCC's
#: HRDPS and RDPS).  Projected like Lambert: canonical axes are distances
#: along the lattice in PROJECTED_AXIS_UNIT_M units, never degrees.
GRID_FAMILY_ROTATED = "rotated_latitude_longitude"
#: Every declared family whose axes are plane distances.
PROJECTED_GRID_FAMILIES = (GRID_FAMILY_LAMBERT, GRID_FAMILY_ROTATED)

_ROTATED_PARAMETER_KEYS = frozenset({
    "lat1", "lon1", "dlat_deg", "dlon_deg", "nx", "ny",
    "south_pole_lat", "south_pole_lon", "rotation_angle",
    "earth_radius_m", "shape_of_earth",
})

#: Axis unit for projected regular grids: one unit = 100 km along the
#: projection plane.  The whole downstream coordinate machinery
#: (:func:`woof.ingest.horiz._regular_coordinates` and every masked
#: operator above it) treats source axes as opaque monotone numbers plus a
#: 360-periodic longitude unwrap; expressing projected coordinates in
#: 100-km units keeps any Earth-sized LAM's span far below one period, so
#: the unwrap is a no-op and the global-ring detection can never trigger.
#: (A grid would need to span 36,000 km -- circumnavigation -- to alias.)
PROJECTED_AXIS_UNIT_M = 100_000.0

#: Canonical wind pairs a grid-relative source must rotate to the earth
#: basis at decode time.  ``eastward_wind`` NAMES the earth basis, so a
#: frame emitted without this rotation would be lying about its fields.
_ROTATED_WIND_PAIRS = (
    ("eastward_wind", "northward_wind"),
    ("eastward_wind_10m", "northward_wind_10m"),
)

def libm_dependent_fields(mapping: Mapping[str, object]) -> frozenset[str]:
    """The fields this mapping produces through a transcendental.

    Two productions in this engine call ``exp``/``sin``/``cos``, and a
    transcendental's last bit is the box's libm, not the decode's answer:

    * a field with a declared ``derivation`` (the humidity relations);
    * both wind pairs, when the declaration says the source publishes
      grid-relative components and the engine rotates them.

    Everything else is integer unpack plus IEEE add/multiply/divide, which
    is bit-reproducible on any conforming machine -- MEASURED 2026-08-20,
    Windows desktop against test host A: identical to the byte on every
    array but these.

    Read from the mapping's own declarations, so a new model is table
    work: nothing here names a source.
    """

    fields = mapping.get("fields", {})
    names = {
        name for name, specification in fields.items()
        if isinstance(specification, dict)
        and isinstance(specification.get("derivation"), str)
        and specification["derivation"]
    }
    declaration = _mapping_grid_declaration(mapping)
    if declaration.get("wind_basis") == "grid_relative_with_rotation":
        for pair in _ROTATED_WIND_PAIRS:
            names.update(name for name in pair if name in fields)
    return frozenset(names)


_LAMBERT_PARAMETER_KEYS = frozenset({
    "latin1", "latin2", "lov", "lat1", "lon1", "dx_m", "dy_m",
    "nx", "ny", "earth_radius_m", "shape_of_earth",
})


def _validate_grid_declaration(
    raw: object, source_format: str,
) -> dict[str, object]:
    """Validate the mapping's optional ``grid`` block."""

    grid = _object(
        raw,
        "mapping.grid",
        allowed={"family", "parameters", "wind_basis", "same_grid_pairing",
                 "published_first_points"},
        required={"family"},
    )
    family = grid["family"]
    same_grid_pairing = grid.get("same_grid_pairing")
    if same_grid_pairing is not None and (
            same_grid_pairing != "identity" or family != GRID_FAMILY_LAMBERT):
        raise ValueError(
            "mapping.grid.same_grid_pairing must be 'identity' on a "
            "declared Lambert grid; other sources retain projected pairing")
    if family == GRID_FAMILY_REGULAR:
        if grid.get("published_first_points") is not None:
            raise ValueError(
                "mapping.grid.published_first_points is not used by "
                f"{GRID_FAMILY_REGULAR}: the grid is read from the data")
        if grid.get("parameters") is not None:
            raise ValueError(
                "mapping.grid.parameters is not used by "
                f"{GRID_FAMILY_REGULAR}: the grid is read from the data"
            )
        if grid.get("wind_basis") not in {None, "earth_relative"}:
            raise ValueError(
                f"{GRID_FAMILY_REGULAR} sources carry earth-relative winds; "
                "a rotated basis requires a projected family"
            )
        return {"family": GRID_FAMILY_REGULAR, "wind_basis": "earth_relative"}
    if family == GRID_FAMILY_ROTATED:
        return _validate_rotated_declaration(grid, source_format)
    if family != GRID_FAMILY_LAMBERT:
        raise ValueError(
            f"unsupported mapping.grid.family {family!r}; supported: "
            f"{GRID_FAMILY_REGULAR}, {GRID_FAMILY_LAMBERT}, "
            f"{GRID_FAMILY_ROTATED}"
        )
    if source_format not in {"grib2", "netcdf"}:
        raise ValueError(
            f"mapping.grid.family {GRID_FAMILY_LAMBERT!r} requires GRIB2 or NetCDF"
        )
    wind_basis = grid.get("wind_basis")
    if wind_basis not in {"earth_relative", "grid_relative_with_rotation"}:
        raise ValueError(
            "a lambert_conformal grid must declare wind_basis "
            "'earth_relative' or 'grid_relative_with_rotation'"
        )
    parameters = _object(
        grid.get("parameters"),
        "mapping.grid.parameters",
        allowed=set(_LAMBERT_PARAMETER_KEYS),
        required=set(_LAMBERT_PARAMETER_KEYS),
    )
    for key in ("latin1", "latin2", "lov", "lat1", "lon1", "dx_m", "dy_m",
                "earth_radius_m"):
        parameters[key] = _number(parameters[key], f"mapping.grid.parameters.{key}")
    for key in ("nx", "ny"):
        parameters[key] = _integer(
            parameters[key], f"mapping.grid.parameters.{key}",
            minimum=2, maximum=1_000_000,
        )
    parameters["shape_of_earth"] = _integer(
        parameters["shape_of_earth"], "mapping.grid.parameters.shape_of_earth",
        minimum=0, maximum=11,
    )
    if parameters["dx_m"] <= 0.0 or parameters["dy_m"] <= 0.0:
        raise ValueError("mapping.grid.parameters dx_m/dy_m must be positive")
    if not 6.2e6 <= parameters["earth_radius_m"] <= 6.5e6:
        raise ValueError(
            "mapping.grid.parameters.earth_radius_m must be a plausible "
            "Earth radius in metres"
        )
    for key, low, high in (
        ("latin1", -90.0, 90.0), ("latin2", -90.0, 90.0),
        ("lat1", -90.0, 90.0), ("lov", 0.0, 360.0), ("lon1", 0.0, 360.0),
    ):
        if not low <= parameters[key] <= high:
            raise ValueError(
                f"mapping.grid.parameters.{key} must be within "
                f"[{low}, {high}] degrees (GRIB2 conventions)"
            )
    span_units = max(
        parameters["nx"] * parameters["dx_m"],
        parameters["ny"] * parameters["dy_m"],
    ) / PROJECTED_AXIS_UNIT_M
    if span_units >= 180.0:
        raise ValueError(
            "declared projected grid spans "
            f"{span_units * PROJECTED_AXIS_UNIT_M / 1000.0:.0f} km; the "
            "projected-axis representation covers grids below 18,000 km"
        )
    published = _validate_published_first_points(
        grid.get("published_first_points"), parameters)
    return {
        "family": GRID_FAMILY_LAMBERT,
        "wind_basis": wind_basis,
        "parameters": parameters,
        **({"same_grid_pairing": same_grid_pairing}
           if same_grid_pairing is not None else {}),
        **({"published_first_points": published} if published else {}),
    }


def _validate_rotated_declaration(
    grid: Mapping[str, object], source_format: str,
) -> dict[str, object]:
    """``mapping.grid`` for a rotated latitude/longitude lattice (GDT 3.1).

    The Rust mapped engine is the twin (``model::RotatedParameters``):
    the same keys, the same bounds, the same span limit.
    """

    if source_format != "grib2":
        raise ValueError(
            f"mapping.grid.family {GRID_FAMILY_ROTATED!r} requires GRIB2")
    if grid.get("published_first_points") is not None:
        raise ValueError(
            "mapping.grid.published_first_points is a Lambert admission; a "
            f"{GRID_FAMILY_ROTATED} grid declares one first point")
    wind_basis = grid.get("wind_basis")
    if wind_basis not in {"earth_relative", "grid_relative_with_rotation"}:
        raise ValueError(
            f"a {GRID_FAMILY_ROTATED} grid must declare wind_basis "
            "'earth_relative' or 'grid_relative_with_rotation'")
    parameters = _object(
        grid.get("parameters"),
        "mapping.grid.parameters",
        allowed=set(_ROTATED_PARAMETER_KEYS),
        required=set(_ROTATED_PARAMETER_KEYS),
    )
    for key in ("lat1", "lon1", "dlat_deg", "dlon_deg", "south_pole_lat",
                "south_pole_lon", "rotation_angle", "earth_radius_m"):
        parameters[key] = _number(parameters[key], f"mapping.grid.parameters.{key}")
    for key in ("nx", "ny"):
        parameters[key] = _integer(
            parameters[key], f"mapping.grid.parameters.{key}",
            minimum=2, maximum=1_000_000,
        )
    parameters["shape_of_earth"] = _integer(
        parameters["shape_of_earth"], "mapping.grid.parameters.shape_of_earth",
        minimum=0, maximum=11,
    )
    if parameters["dlat_deg"] <= 0.0 or parameters["dlon_deg"] <= 0.0:
        raise ValueError("mapping.grid.parameters dlat_deg/dlon_deg must be positive")
    if not 6.2e6 <= parameters["earth_radius_m"] <= 6.5e6:
        raise ValueError(
            "mapping.grid.parameters.earth_radius_m must be a plausible "
            "Earth radius in metres")
    for key in ("lat1", "south_pole_lat"):
        if not -90.0 <= parameters[key] <= 90.0:
            raise ValueError(
                f"mapping.grid.parameters.{key} must be within [-90, 90] degrees")
    metres_per_degree = math.pi / 180.0 * parameters["earth_radius_m"]
    span_units = max(
        parameters["nx"] * parameters["dlon_deg"],
        parameters["ny"] * parameters["dlat_deg"],
    ) * metres_per_degree / PROJECTED_AXIS_UNIT_M
    if span_units >= 180.0:
        raise ValueError(
            "declared projected grid spans "
            f"{span_units * PROJECTED_AXIS_UNIT_M / 1000.0:.0f} km; the "
            "projected-axis representation covers grids below 18,000 km")
    return {
        "family": GRID_FAMILY_ROTATED,
        "wind_basis": wind_basis,
        "parameters": parameters,
    }


def declared_projected_source_grid(projection: Mapping[str, object]):
    """The source grid a projected frame descriptor declares, by family.

    ``projection`` is ``{"family", "parameters"}`` as a frame header (or a
    snapshot) states it.  Both answers offer ``latlon_to_ij`` and
    ``ij_to_latlon`` in one-based grid coordinates whose spacing is the
    descriptor's ``dx_m``/``dy_m``, so a pairing never names a family.
    """

    family = str(projection["family"])
    parameters = projection["parameters"]
    if family == GRID_FAMILY_LAMBERT:
        return declared_lambert_source_grid(parameters)
    if family == GRID_FAMILY_ROTATED:
        from woof.static.rotated_latlon import RotatedLatLonGrid

        return RotatedLatLonGrid.from_parameters(parameters)
    raise ValueError(f"no declared projected source grid for family {family!r}")


def _validate_published_first_points(
    raw: object, parameters: Mapping[str, object],
) -> list[dict[str, float]]:
    """``mapping.grid.published_first_points``: other encodings of the
    DECLARED first grid point that a publisher writes for this same grid.

    HRRRv1 wrfprs and wrfnat files (to 2016-08-23) publish HRRR's corner
    rounded to millidegrees, 21.138, 237.28, under an otherwise identical
    grid definition; current files publish 21.138123, 237.280472.  The
    declaration admits such a record and decodes nothing differently: the
    frame keeps the declared parameters (the Rust engine's
    ``model::published_first_points`` is the twin).  An encoding farther
    than a tenth of a cell from the declared point is a different grid,
    not a spelling of this one, and is refused.
    """

    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError(
            "mapping.grid.published_first_points must be an array of "
            "{lat1, lon1}")
    tenth_cell_deg = 0.1 * min(
        float(parameters["dx_m"]), float(parameters["dy_m"])) / 111_000.0
    points = []
    for index, item in enumerate(raw):
        entry = _object(
            item, f"mapping.grid.published_first_points[{index}]",
            allowed={"lat1", "lon1"}, required={"lat1", "lon1"})
        lat1 = _number(entry["lat1"],
                       f"mapping.grid.published_first_points[{index}].lat1")
        lon1 = _number(entry["lon1"],
                       f"mapping.grid.published_first_points[{index}].lon1")
        if (abs(lat1 - float(parameters["lat1"])) > tenth_cell_deg
                or abs(lon1 - float(parameters["lon1"])) > tenth_cell_deg):
            raise ValueError(
                f"mapping.grid.published_first_points entry ({lat1}, {lon1}) "
                "is more than a tenth of a cell from the declared first point "
                f"({parameters['lat1']}, {parameters['lon1']}): that is a "
                "different grid, not an encoding of this one")
        points.append({"lat1": lat1, "lon1": lon1})
    return points


def _mapping_grid_declaration(
    mapping: Mapping[str, object],
) -> dict[str, object]:
    """The mapping's normalized grid declaration (defaulting to regular)."""

    declared = mapping.get("grid")
    if declared is None:
        return {"family": GRID_FAMILY_REGULAR, "wind_basis": "earth_relative"}
    return declared  # normalized by load_mapping


def _wrap180(value: float) -> float:
    return ((value + 180.0) % 360.0) - 180.0


def declared_lambert_source_grid(parameters: Mapping[str, object]):
    """A :class:`woof.static.lambert.LambertGrid` for a DECLARED source.

    The WPS projection math assumes its own fixed Earth radius, so the
    declared spacing is scaled by ``R_WPS / earth_radius_m`` -- geometry
    identical to the source's stated shape of Earth, expressed in WPS's
    coordinates.  This is the same reconciliation the native HRRR route
    performs (:func:`woof.ingest.hrrr.hrrr_source_grid`), stated once
    here for every declared Lambert source.
    """

    from woof.static.lambert import EARTH_RADIUS_M, LambertGrid

    nx = int(parameters["nx"])
    ny = int(parameters["ny"])
    scale = EARTH_RADIUS_M / float(parameters["earth_radius_m"])
    return LambertGrid(
        ref_lat=float(parameters["lat1"]),
        ref_lon=_wrap180(float(parameters["lon1"])),
        truelat1=float(parameters["latin1"]),
        truelat2=float(parameters["latin2"]),
        stand_lon=_wrap180(float(parameters["lov"])),
        dx=float(parameters["dx_m"]) * scale,
        dy=float(parameters["dy_m"]) * scale,
        e_we=nx + 1,
        e_sn=ny + 1,
        known_x=1.0,
        known_y=1.0,
    )


def _projected_axes(
    parameters: Mapping[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    """(y_axis, x_axis) of a declared projected grid, in 100-km units."""

    y = np.arange(int(parameters["ny"]), dtype=np.float64) \
        * float(parameters["dy_m"]) / PROJECTED_AXIS_UNIT_M
    x = np.arange(int(parameters["nx"]), dtype=np.float64) \
        * float(parameters["dx_m"]) / PROJECTED_AXIS_UNIT_M
    return y, x


def _declared_grid_rotation(
    parameters: Mapping[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    """Source-grid ``(SINALPHA, COSALPHA)`` over a declared Lambert grid.

    The same analytic cone rotation the native HRRR route computes on its
    bridge windows (``woof.ingest.hrrr._source_window_rotation``), over
    the full declared grid.
    """

    source = declared_lambert_source_grid(parameters)
    x = np.arange(1, int(parameters["nx"]) + 1, dtype=np.float64)
    y = np.arange(1, int(parameters["ny"]) + 1, dtype=np.float64)
    _, longitude = source.ij_to_latlon(*np.meshgrid(x, y))
    difference = source.stand_lon - longitude
    difference = np.where(difference > 180.0, difference - 360.0, difference)
    difference = np.where(difference < -180.0, difference + 360.0, difference)
    alpha = source.hemi * source.cone * np.pi / 180.0 * difference
    return np.sin(alpha), np.cos(alpha)


#: ``resolution_flags`` bit 0x08: vector components are grid-relative.
_GRIB2_GRID_RELATIVE_WIND_BIT = 0x08


def _require_declared_grib2_grid(
    row: Mapping[str, str], declaration: Mapping[str, object],
    source: Path,
) -> None:
    """Refuse a GRIB2 record whose observed grid differs from the table.

    Both vocabularies are printed -- what the mapping declared AND what the
    file contains -- because the remedies are opposite: a wrong declaration
    is contract work, a wrong file is input data.
    """

    parameters = declaration["parameters"]
    observed = {
        "nx": int(row["nx"]), "ny": int(row["ny"]),
        "lat1": float(row["lat1"]), "lon1": float(row["lon1"]),
        "dx_m": float(row["dx"]), "dy_m": float(row["dy"]),
        "latin1": float(row["latin1"]), "latin2": float(row["latin2"]),
        "lov": float(row["lov"]),
        "shape_of_earth": int(row["shape_of_earth"]),
    }
    declared = {key: parameters[key] for key in observed}

    def close(observed_value, declared_value) -> bool:
        return math.isclose(float(observed_value), float(declared_value),
                            rel_tol=0.0, abs_tol=1e-6)

    mismatched = {
        key for key in observed
        if (observed[key] != declared[key]
            if isinstance(declared[key], int)
            else not close(observed[key], declared[key]))
    }
    # The first point is one coordinate pair: the declared one, or one of
    # the mapping's published encodings of it, both halves together.
    if any(close(observed["lat1"], point["lat1"])
           and close(observed["lon1"], point["lon1"])
           for point in declaration.get("published_first_points", ())):
        mismatched -= {"lat1", "lon1"}
    if mismatched:
        raise ValueError(
            f"GRIB2 field {row['index']} in {source} is not on the "
            "declared lambert_conformal grid; declared "
            f"{ {key: declared[key] for key in sorted(mismatched)} }, "
            f"observed { {key: observed[key] for key in sorted(mismatched)} }"
        )
    flags = int(row["resolution_flags"], 0)
    grid_relative = bool(flags & _GRIB2_GRID_RELATIVE_WIND_BIT)
    declared_rotation = declaration["wind_basis"] == "grid_relative_with_rotation"
    if grid_relative != declared_rotation:
        raise ValueError(
            f"GRIB2 field {row['index']} in {source} declares "
            f"{'grid' if grid_relative else 'earth'}-relative vector "
            "components (resolution_flags bit 0x08) while the mapping "
            f"declares wind_basis={declaration['wind_basis']!r}"
        )


_GRIB2_INVENTORY_REQUIRED_COLUMNS = frozenset({
    "index", "discipline", "category", "parameter",
    *_GRIB2_AUTHORITY_KEYS,
    "reference_time", "forecast_unit", "forecast_time", "pdt",
    "level_type", "level_value", "second_level_type", "second_level_value",
    "member", "generating_process", "forecast_generating_process_id",
    "gdt", "nx", "ny", "lat1", "lon1", "dx", "dy", "latin1",
    "latin2", "lov", "scan_mode", "shape_of_earth", "resolution_flags",
    "drt", "bitmap",
    # Section 4's pv coordinate octets -- the hybrid A/B channel -- are
    # deliberately NOT here.  They are required where they are the
    # coordinate, which is :func:`_require_pv_where_hybrid` below, not on
    # every inventory: a sea-surface-temperature record has no vertical
    # coordinate to lose, and demanding pv of it made a stale
    # grib2_inventory refuse a round trip that was never about hybrid
    # levels at all.
})
_GRIB2_DUMP_REQUIRED_COLUMNS = frozenset({
    "index", "discipline", "category", "parameter",
    *_GRIB2_AUTHORITY_KEYS,
    "level_type", "level_value", "second_level_type", "second_level_value",
    "member", "pdt", "drt", "nx", "ny", "scan_mode", "bitmap",
    "count", "finite", "missing", "minimum", "maximum", "filename",
})
_GRIB2_INVENTORY_DUMP_PARITY_COLUMNS = (
    "discipline", "category", "parameter", *_GRIB2_AUTHORITY_KEYS,
    "level_type", "level_value", "second_level_type", "second_level_value",
    "member", "pdt", "drt", "nx", "ny", "scan_mode", "bitmap",
)


#: GRIB2 code table 4.5 level types whose vertical coordinate IS the
#: Section-4 pv octets -- the hybrid A/B coefficient pair a record's
#: pressure is rebuilt from.  105 is the hybrid level ECMWF's L137
#: ladder is published on; 118 and 119 are the hybrid-height and
#: hybrid-pressure spellings WMO defines beside it.  A record on any of
#: them has no vertical position at all without the coefficients.
_GRIB2_HYBRID_LEVEL_TYPES = frozenset({105, 118, 119})


def _require_pv_where_hybrid(
    rows: Sequence[Mapping[str, str]], *, label: str
) -> None:
    """Refuse a HYBRID inventory whose decoder cannot state the pv octets.

    The breakage this prevents is the L137 proof lane's gap G1: a
    grib2_inventory built before the pv column decodes a hybrid source
    into empty coefficient tuples, so every level lands at whatever the
    fallback puts it at and nothing says a word.

    It is asked of hybrid records ONLY, and that is the whole correction
    landed on 2026-08-20.  ``pv`` was briefly a required column of every
    GRIB2 inventory, which made a decoder without it refuse reads that
    have no vertical coordinate to lose -- measured on the 2.5.1 battery,
    where a sea-surface-temperature GRIB2 round trip failed with "missing
    required columns ['pv']" about a field on a single surface level.  A
    refusal has to name the breakage it prevents, and there is no hybrid
    coefficient to lose on a record that has none.
    """

    if not rows or "pv" in rows[0]:
        return
    hybrid = sorted({
        int(row["level_type"]) for row in rows
        if int(row["level_type"]) in _GRIB2_HYBRID_LEVEL_TYPES})
    if not hybrid:
        return
    raise ValueError(
        f"{label} carries hybrid-level records (level_type "
        + ", ".join(str(value) for value in hybrid)
        + ") and states no 'pv' column, so the Section-4 hybrid A/B "
          "coefficients cannot be read and those records have no vertical "
          "coordinate.  The decoder that wrote this inventory predates the "
          "pv octets: rebuild it (cargo build --release --manifest-path "
          "tools/grib1_bridge/Cargo.toml)."
    )


def _parse_grib2_tsv(
    lines: Sequence[str], *, required: frozenset[str], label: str
) -> list[dict[str, str]]:
    reader = csv.DictReader(lines, delimiter="\t")
    fields = reader.fieldnames or []
    if len(fields) != len(set(fields)):
        raise ValueError(f"{label} contains duplicate columns")
    missing = sorted(required - set(fields))
    if missing:
        raise ValueError(f"{label} is missing required columns {missing}")
    rows = list(reader)
    if not rows:
        raise ValueError(f"{label} contains no records")
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"{label} contains a malformed row")
    indices = [int(row["index"]) for row in rows]
    if len(indices) != len(set(indices)):
        raise ValueError(f"{label} contains duplicate field indices")
    for row in rows:
        for key, maximum in (
            ("center", 65535),
            ("subcenter", 65535),
            ("master_table_version", 255),
            ("local_table_version", 255),
        ):
            value = int(row[key])
            if not 0 <= value <= maximum:
                raise ValueError(
                    f"{label} field {row['index']} has invalid {key}={value}"
                )
    return rows


def _require_grib2_inventory_dump_parity(
    inventory: Mapping[str, str], dump: Mapping[str, str]
) -> None:
    mismatches = [
        key for key in _GRIB2_INVENTORY_DUMP_PARITY_COLUMNS
        if inventory[key] != dump[key]
    ]
    if mismatches:
        raise ValueError(
            f"GRIB2 dump metadata differs from inventory for field "
            f"{inventory['index']} in columns {mismatches}"
        )


def _grib2_inventory(source: Path, executable: Path) -> list[dict[str, str]]:
    completed = subprocess.run(
        [os.fspath(executable), os.fspath(source)],
        text=True, capture_output=True, check=False,
    )
    if completed.returncode:
        detail = (completed.stderr or completed.stdout).strip()
        if detail:
            # The tool RAN and named what is wrong with the bytes, so
            # this is a decode refusal about the input -- the same
            # condition the Rust engine refuses with `decode_failed`
            # (ValueError) -- not a broken installation.
            raise ValueError(
                f"GRIB2 inventory failed for {source}: {detail}")
        raise RuntimeError(
            f"GRIB2 inventory failed for {source}: the decoder exited "
            f"{completed.returncode} without a diagnostic, which is "
            "about the tool, not the bytes")
    rows = [line for line in completed.stdout.splitlines() if not line.startswith("#")]
    parsed = _parse_grib2_tsv(
        rows,
        required=_GRIB2_INVENTORY_REQUIRED_COLUMNS,
        label="GRIB2 inventory",
    )
    _require_pv_where_hybrid(parsed, label=f"GRIB2 inventory for {source}")
    return parsed


def _regular_latlon_frame(
    row: Mapping[str, str], raw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One canonical ascending-latitude frame from a regular GDT-0 record.

    Scan mode 0x40 stores rows south-to-north and passes through; 0x00
    stores them north-to-south (ECMWF's open data, NCEP's global grids)
    and is normalized by reversing the row axis together with its
    latitudes, so every (latitude, value) pair the producer stored is the
    pair a consumer reads.  This is the y-axis twin of the longitude
    unwrap below it: decode-time normalization into one declared
    orientation, at the one boundary that knows the source octets.  Any
    other scan mode -- reversed i, column-major, boustrophedon -- would
    silently permute the field under either convention, so those refuse
    by name.
    """

    gdt = int(row["gdt"])
    scan = int(row["scan_mode"], 16)
    if gdt != 0 or scan not in (0x40, 0x00):
        raise ValueError(
            "generic GRIB2 frame export requires regular latitude/longitude "
            "GDT 0 with scan mode 0x40 (rows south-to-north) or 0x00 (rows "
            "north-to-south, normalized at decode), or a mapping.grid "
            "declaration for a supported projected family; got GDT "
            f"{gdt} scan mode 0x{scan:02x}"
        )
    nx = int(row["nx"])
    ny = int(row["ny"])
    dy = float(row["dy"])
    if scan == 0x40:
        latitude = float(row["lat1"]) + np.arange(ny, dtype=np.float64) * dy
    else:
        latitude = float(row["lat1"]) - np.arange(ny, dtype=np.float64) * dy
    longitude_raw = float(row["lon1"]) + np.arange(nx, dtype=np.float64) * float(row["dx"])
    longitude_wrapped = (longitude_raw + 180.0) % 360.0 - 180.0
    spacing = abs(float(row["dx"]))
    if (nx + 0.5) * spacing < 360.0:
        # A regional crop across the antimeridian (the GDT-101 remap writes
        # one whenever the domain crosses it) is continuous in its own
        # column order.  Wrapping and sorting it put a jump of nearly a
        # full turn inside the axis, and every such domain was refused as
        # "not a regular axis".  Carrying the columns past the seam one
        # turn on keeps the crop continuous, which is the axis the target
        # pairing expects (horiz._regular_coordinates); a crop clear of
        # the seam has no descending step and is untouched.  A grid within
        # half a cell of the full circle is a whole ring whose spacing was
        # rounded to the octets' micro-degrees (a 1/12-degree ring stores
        # 4320 x 0.083333), and it keeps the -180 origin it always had.
        seam = np.flatnonzero(np.diff(longitude_wrapped) < 0.0)
        if seam.size == 1:
            longitude_wrapped = longitude_wrapped.copy()
            longitude_wrapped[seam[0] + 1:] += 360.0
    longitude_order = np.argsort(longitude_wrapped)
    longitude = longitude_wrapped[longitude_order]
    values = raw.reshape(ny, nx)[:, longitude_order]
    if scan == 0x00:
        latitude = latitude[::-1].copy()
        values = values[::-1, :]
    return latitude, longitude, values.copy()


def _coordinate_values(row: Mapping[str, str]) -> tuple[float, ...]:
    """A record's Section-4 coordinate octets, or the empty tuple.

    ``-`` is the decoder saying the record carries none.  An ABSENT
    column is a decoder that cannot say either way, and
    :func:`_require_pv_where_hybrid` has already refused that inventory
    if any record needed the answer -- so reaching here without the
    column means no record did, and the empty tuple is the truth rather
    than a guess.
    """

    raw = row.get("pv", "-")
    if raw == "-":
        return ()
    return tuple(float(value) for value in raw.split(","))


def _grib2_records(
    source: Path,
    inventory_executable: Path,
    dump_executable: Path,
    wanted_indices: set[int],
    grid_declaration: Mapping[str, object] | None = None,
    source_label: Path | None = None,
) -> tuple[_GribRecord, ...]:
    inventory = _grib2_inventory(source, inventory_executable)
    selected = [row for row in inventory if int(row["index"]) in wanted_indices]
    if not selected:
        return ()
    with tempfile.TemporaryDirectory(prefix="gpuwm-mapped-grib2-") as temporary:
        dump = Path(temporary) / "dump"
        command = [
            os.fspath(dump_executable), os.fspath(source), os.fspath(dump),
            *(str(int(row["index"])) for row in selected),
        ]
        completed = subprocess.run(
            command, text=True, capture_output=True, check=False
        )
        if completed.returncode:
            detail = (completed.stderr or completed.stdout).strip()
            if detail:
                # Same split as the inventory wrapper: a diagnostic
                # means undecodable bytes (ValueError, `decode_failed`);
                # silence means the installation broke.
                raise ValueError(
                    f"GRIB2 dump failed for {source}: {detail}")
            raise RuntimeError(
                f"GRIB2 dump failed for {source}: the decoder exited "
                f"{completed.returncode} without a diagnostic, which is "
                "about the tool, not the bytes")
        dump_rows = _parse_grib2_tsv(
            (dump / "metadata.tsv").read_text(encoding="utf-8").splitlines(),
            required=_GRIB2_DUMP_REQUIRED_COLUMNS,
            label="GRIB2 dump metadata",
        )
        dumped = {int(row["index"]): row for row in dump_rows}
        result = []
        for row in selected:
            index = int(row["index"])
            detail = dumped.get(index)
            if detail is None:
                raise ValueError(f"GRIB2 dump omitted selected field {index}")
            _require_grib2_inventory_dump_parity(row, detail)
            gdt = int(row["gdt"])
            scan = int(row["scan_mode"], 16)
            declared_family = (
                GRID_FAMILY_REGULAR if grid_declaration is None
                else str(grid_declaration["family"])
            )
            nx = int(row["nx"])
            ny = int(row["ny"])
            if declared_family == GRID_FAMILY_ROTATED:
                # The inventory/dump TSV this workaround engine reads has no
                # GDT 3.1 pole columns, so it cannot prove a record sits on
                # the declared rotated lattice; the default Rust engine
                # (gpuwm_mapped_engine) checks every Section 3 number.
                raise ValueError(
                    "the Python mapped engine cannot verify a "
                    f"{GRID_FAMILY_ROTATED} record (its GRIB2 inventory "
                    "carries no rotation pole); decode this source with the "
                    "default Rust engine (unset GPUWM_MAPPED_ENGINE)")
            if declared_family == GRID_FAMILY_LAMBERT:
                if gdt != 30 or scan != 0x40:
                    raise ValueError(
                        "the mapping declares a lambert_conformal source "
                        "grid, which requires GRIB2 GDT 30 with scan mode "
                        f"0x40; field {index} in {source} carries GDT "
                        f"{gdt} scan 0x{scan:02x}"
                    )
                _require_declared_grib2_grid(row, grid_declaration, source)
                latitude, longitude = _projected_axes(
                    grid_declaration["parameters"])
                values = np.fromfile(dump / detail["filename"], dtype="<f8")
                if values.size != nx * ny:
                    raise ValueError(
                        f"GRIB2 field {index} decoded count differs from grid")
                values = values.reshape(ny, nx).copy()
            else:
                raw = np.fromfile(dump / detail["filename"], dtype="<f8")
                if raw.size != nx * ny:
                    raise ValueError(f"GRIB2 field {index} decoded count differs from grid")
                latitude, longitude, values = _regular_latlon_frame(row, raw)
            second_type = int(row["second_level_type"])
            second_value = float(row["second_level_value"])
            reference = datetime.fromisoformat(row["reference_time"])
            valid = _embedded_valid_time(
                reference, int(row["forecast_unit"]), int(row["forecast_time"]), edition=2
            )
            fingerprint_payload = {
                key: row[key] for key in (
                    "gdt", "nx", "ny", "lat1", "lon1", "dx", "dy",
                    "latin1", "latin2", "lov", "scan_mode", "shape_of_earth",
                    "resolution_flags",
                )
            }
            fingerprint = hashlib.sha256(
                json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            result.append(_GribRecord(
                source=(source_label or source).resolve(), index=index,
                reference_time=reference, valid_time=valid,
                member=None if row["member"] == "-" else row["member"],
                parameter=int(row["parameter"]),
                level_type=int(row["level_type"]),
                level_value=float(row["level_value"]),
                table_version=None, center=int(row["center"]),
                subcenter=int(row["subcenter"]),
                master_table_version=int(row["master_table_version"]),
                local_table_version=int(row["local_table_version"]),
                discipline=int(row["discipline"]), category=int(row["category"]),
                second_level_type=second_type, second_level_value=second_value,
                process_identity=(
                    int(row["generating_process"]),
                    int(row["forecast_generating_process_id"]),
                ),
                time_semantics=(int(row["pdt"]),),
                values=values, latitude=latitude, longitude=longitude,
                grid_fingerprint=fingerprint,
                coordinate_values=_coordinate_values(row),
            ))
        return tuple(result)


#: Section-1 producer-identity octets a GRIB2 selector may pin.
_GRIB2_IDENTITY_KEYS = (
    "center", "subcenter", "master_table_version", "local_table_version",
)


def _selector_identity_refusal(
    mapping: Mapping[str, object],
    rows: Sequence[Mapping[str, str]],
    files: Sequence[Path],
) -> str:
    """Why zero records matched, named when the identity octets explain it.

    Some agencies publish two different products under IDENTICAL file
    names, distinguishable in-band only by the section-1 producer
    identity (an operational line and an experimental line can differ
    solely in ``subcenter``).  When every selector in a mapping pins one
    identity and every supplied message carries another, a plain
    "nothing matched" refusal would hide the one fact the operator
    needs, so this refusal states the pinned and observed octets side by
    side.  Octets that agree are never blamed.
    """

    pins: dict[str, set[int]] = {}
    for field in mapping["fields"].values():
        for selector in field.get("selectors", []):
            for key in _GRIB2_IDENTITY_KEYS:
                if selector.get(key) is not None:
                    pins.setdefault(key, set()).add(int(selector[key]))
    mismatched = []
    for key in _GRIB2_IDENTITY_KEYS:
        pinned = pins.get(key)
        if pinned is None or len(pinned) != 1:
            continue
        observed = sorted({
            int(row[key]) for row in rows if row.get(key) is not None
        })
        if observed and not set(observed) & pinned:
            mismatched.append(
                f"every mapping selector pins {key}={next(iter(pinned))} "
                f"but every supplied message observes {key}="
                + "/".join(str(value) for value in observed)
            )
    names = ", ".join(path.name for path in files)
    base = (
        f"0 of {len(rows)} GRIB message(s) in {names} match this "
        "mapping's selectors"
    )
    if not mismatched:
        return base
    return (
        base + "; the producer-identity octets explain it: "
        + "; ".join(mismatched)
        + " -- these bytes are a DIFFERENT product line published under "
        "the same file naming, and decoding them here would silently mix "
        "model versions; the profile's provenance document names the "
        "front door that serves the pinned identity"
    )


def _selector_matches_record(
    selector: Mapping[str, object], record: _GribRecord, source_format: str
) -> bool:
    if source_format == "grib1":
        checks = (
            record.parameter == int(selector["parameter"]),
            selector.get("table_version") is None
            or record.table_version == int(selector["table_version"]),
            selector.get("center") is None or record.center == int(selector["center"]),
            selector.get("level_type") is None
            or record.level_type == int(selector["level_type"]),
            selector.get("level_value") is None
            or math.isclose(record.level_value, float(selector["level_value"]), abs_tol=1e-9),
        )
    else:
        checks = (
            record.discipline == int(selector["discipline"]),
            record.category == int(selector["category"]),
            record.parameter == int(selector["parameter"]),
            selector.get("center") is None
            or record.center == int(selector["center"]),
            selector.get("subcenter") is None
            or record.subcenter == int(selector["subcenter"]),
            selector.get("master_table_version") is None
            or record.master_table_version
            == int(selector["master_table_version"]),
            selector.get("local_table_version") is None
            or record.local_table_version
            == int(selector["local_table_version"]),
            selector.get("level_type") is None
            or record.level_type == int(selector["level_type"]),
            selector.get("level_value") is None
            or math.isclose(record.level_value, float(selector["level_value"]), abs_tol=1e-9),
            (
                record.second_level_type in {None, 255}
                and selector.get("second_level_type") is None
            ) or (
                selector.get("second_level_type") is not None
                and record.second_level_value is not None
                and record.second_level_type == int(selector["second_level_type"])
                and math.isclose(
                    float(record.second_level_value),
                    float(selector["second_level_value"]),
                    abs_tol=1e-9,
                )
            ),
            selector.get("member") is None
            or record.member == str(selector["member"]),
            selector.get("pdt") is None
            or record.time_semantics[0] == int(selector["pdt"]),
        )
    return all(checks)


def _declared_vertical_admits(
    mapping: Mapping[str, object],
    field: Mapping[str, object],
    record: "_GribRecord",
) -> bool:
    """Whether a matched record's level is inside the declared ladder.

    The same general property the mapped NetCDF decoder already states:
    declared levels may be a SUBSET of a file's, selected by value.  A
    producer publishing extra diagnostic surfaces beside its ladder (HRRR
    adds a 1013.2 hPa standard-atmosphere set to its 40-level stack) is
    normal; a declared level with no record is still the error, refused by
    the vertical-coverage check downstream.  Fields without a vertical
    source axis are unaffected.
    """

    axes = tuple(field.get("source_axes", ()))
    if not {"vertical", "half_level"}.intersection(axes):
        return True
    key = "interface_levels" if "half_level" in axes else "levels"
    declared = mapping["coordinates"]["vertical"].get(key, [])
    if not declared:
        return True
    return any(
        math.isclose(record.level_value, float(level), rel_tol=0.0, abs_tol=1e-6)
        for level in declared
    )


def _grib2_wanted_indices(
    mapping: Mapping[str, object], rows: Sequence[Mapping[str, str]]
) -> set[int]:
    aliases = tuple(mapping.get("record_aliases") or ())
    wanted = set()
    for row in rows:
        # Lightweight selector record; data/coordinates are irrelevant here.
        record = _GribRecord(
            source=Path("."), index=int(row["index"]),
            reference_time=datetime.min, valid_time=datetime.min,
            member=None if row["member"] == "-" else row["member"],
            parameter=int(row["parameter"]), level_type=int(row["level_type"]),
            level_value=float(row["level_value"]), table_version=None,
            center=int(row["center"]), subcenter=int(row["subcenter"]),
            master_table_version=int(row["master_table_version"]),
            local_table_version=int(row["local_table_version"]),
            discipline=int(row["discipline"]), category=int(row["category"]),
            second_level_type=int(row["second_level_type"]),
            second_level_value=float(row["second_level_value"]),
            process_identity=None, time_semantics=(int(row["pdt"]),),
            values=np.empty((0, 0)),
            latitude=np.empty(0), longitude=np.empty(0), grid_fingerprint="",
        )
        record, _aliased = _alias_record(record, aliases)
        for field in mapping["fields"].values():
            if any(_selector_matches_record(selector, record, "grib2")
                   for selector in field.get("selectors", [])) \
                    and _declared_vertical_admits(mapping, field, record):
                wanted.add(record.index)
                break
    return wanted


#: Literal-vs-pv agreement tolerance.  The pv octets ride IEEE-754 f32
#: (about 7 significant digits) and inline literals are authored from
#: the provider's published table of the same numbers, so print
#: rounding can separate them by well under 1e-3 Pa in A and 1e-6 in B
#: -- while a wrong-model ladder moves adjacent coefficients by tens of
#: Pa or whole percent.  abs 1e-3 + rel 1e-6 admits the former and
#: names the latter.
_HYBRID_LITERAL_ABS_TOL = 1e-3
_HYBRID_LITERAL_REL_TOL = 1e-6


def _resolve_hybrid_coefficients(
    vertical: Mapping[str, object],
    nlevels: int,
    *,
    record_pv: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """The one A/B ladder a hybrid decode runs on.

    Primary channel: the records' pv coordinate octets (A half then B
    half, the GRIB encoding).  Declared-data fallback: inline
    ``vertical.hybrid_a``/``hybrid_b`` literal arrays, which keep
    providers whose bytes carry no pv on the table-work path.  When
    both exist they must agree; the bytes are then the values used.
    Counts are held to nlevels+1 (half-level interfaces) or nlevels
    (full levels).
    """

    literals: tuple[np.ndarray, np.ndarray] | None = None
    if vertical.get("hybrid_a") is not None:
        literals = (
            np.asarray([float(value) for value in vertical["hybrid_a"]],
                       dtype=np.float64),
            np.asarray([float(value) for value in vertical["hybrid_b"]],
                       dtype=np.float64),
        )
    if record_pv:
        pv = np.asarray(record_pv, dtype=np.float64)
        if pv.size % 2:
            raise ValueError(
                f"pv coordinate list length {pv.size} is not an even "
                "A+B split"
            )
        half = pv.size // 2
        if half not in (nlevels + 1, nlevels):
            raise ValueError(
                "hybrid coefficient count mismatch: the pv coordinate "
                f"octets carry {half} A and {half} B coefficients; "
                f"{nlevels} levels accept {nlevels + 1} (half-level "
                f"interfaces) or {nlevels} (full levels)"
            )
        a_values, b_values = pv[:half], pv[half:]
        if not np.isfinite(a_values).all() or np.any(a_values < 0.0):
            raise ValueError(
                "pv A coefficients must be finite and non-negative (Pa)"
            )
        if not np.isfinite(b_values).all() \
                or np.any(b_values < 0.0) or np.any(b_values > 1.0):
            raise ValueError(
                "pv B coefficients must be finite within [0, 1]"
            )
        if literals is not None:
            for label, declared, observed in (
                ("hybrid_a", literals[0], a_values),
                ("hybrid_b", literals[1], b_values),
            ):
                if declared.size != observed.size:
                    raise ValueError(
                        f"inline vertical.{label} declares {declared.size} "
                        "coefficients but the source's pv octets carry "
                        f"{observed.size}; the literals disagree with the "
                        "bytes"
                    )
                tolerance = _HYBRID_LITERAL_ABS_TOL \
                    + _HYBRID_LITERAL_REL_TOL * np.abs(observed)
                misfit = np.abs(declared - observed) > tolerance
                if misfit.any():
                    index = int(np.argmax(misfit))
                    raise ValueError(
                        f"inline vertical.{label} literals disagree with "
                        f"the source's pv octets at index {index}: literal "
                        f"{float(declared[index])!r} vs pv "
                        f"{float(observed[index])!r}"
                    )
        return a_values, b_values
    if literals is not None:
        count = int(literals[0].size)
        if count not in (nlevels + 1, nlevels):
            raise ValueError(
                "hybrid coefficient count mismatch: vertical.hybrid_a "
                f"declares {count} coefficients; {nlevels} levels accept "
                f"{nlevels + 1} (half-level interfaces) or {nlevels} "
                "(full levels)"
            )
        return literals
    raise ValueError(
        "hybrid_sigma_pressure source supplies no A/B coefficients: the "
        "selected records carry no pv coordinate octets and the mapping "
        "declares no inline vertical.hybrid_a/hybrid_b literals"
    )


def _assemble_grib(
    mapping: Mapping[str, object], records: Sequence[_GribRecord]
) -> _DecodedCollection:
    if not records:
        raise ValueError("no mapped GRIB records were decoded")
    if mapping["coordinates"]["horizontal"].get("normalization") == "largest_common_finite_rectangle":
        raise ValueError("finite-rectangle normalization requires the native mapped engine")
    source_format = str(mapping["format"])
    matched: dict[tuple[datetime, str | None, str], list[_GribRecord]] = {}
    for field_name, field in mapping["fields"].items():
        if field.get("derivation") is not None:
            continue
        selectors = field.get("selectors", [])
        for record in records:
            if any(_selector_matches_record(selector, record, source_format)
                   for selector in selectors) \
                    and _declared_vertical_admits(mapping, field, record):
                matched.setdefault((record.valid_time, record.member, field_name), []).append(record)

    if not matched:
        raise ValueError("no GRIB messages match the mapping selectors")
    selected_records = [record for group in matched.values() for record in group]
    grid_fingerprints = {record.grid_fingerprint for record in selected_records}
    if len(grid_fingerprints) != 1:
        raise ValueError("selected GRIB fields do not share one source grid")
    latitude = selected_records[0].latitude
    longitude = selected_records[0].longitude
    if any(not np.array_equal(record.latitude, latitude)
           or not np.array_equal(record.longitude, longitude)
           for record in selected_records):
        raise ValueError("selected GRIB coordinate axes differ")
    members = {record.member for record in selected_records}
    if len(members) != 1:
        raise ValueError("mapped WRF initialization requires exactly one GRIB member")
    processes_by_time: dict[
        tuple[datetime, str | None], set[tuple[int, int]]
    ] = {}
    for record in selected_records:
        if record.process_identity is not None:
            processes_by_time.setdefault(
                (record.valid_time, record.member), set()
            ).add(record.process_identity)
    mixed_processes = {
        key: identities for key, identities in processes_by_time.items()
        if len(identities) > 1
    }
    if mixed_processes:
        raise ValueError(
            "selected GRIB2 fields mix generating-process identities within "
            f"a valid time: {mixed_processes!r}"
        )
    unsupported_time_semantics = []
    for record in selected_records:
        if source_format == "grib1":
            time_range_indicator, _p1, p2 = record.time_semantics
            supported = time_range_indicator == 0 and p2 == 0
        else:
            supported = record.time_semantics[0] in {0, 1}
        if not supported:
            unsupported_time_semantics.append(
                (record.index, record.time_semantics)
            )
    if unsupported_time_semantics:
        raise ValueError(
            "selected GRIB fields use interval/derived time semantics that "
            "rw-wps.mapping.v1 cannot bind: "
            + repr(unsupported_time_semantics[:8])
        )

    # Fields a pressure-level frame completes where the source leaves
    # them out: their levels may be a subset of the ladder.
    completed = set(_completed_fields(mapping))
    explicit = tuple(float(value) for value in mapping["coordinates"]["vertical"].get("levels", []))
    if explicit:
        stacked = [
            key for key in matched
            if "vertical" in tuple(mapping["fields"][key[2]].get("source_axes", ()))
        ]
        # A field declared on part of the ladder (published_levels) holds
        # the levels it is zero on as well as the ones it carries.
        unpublished = {
            key: set(explicit) - published
            for key in stacked
            if (published := _published_levels(
                mapping["fields"][key[2]])) is not None
        }
        ladder = _carried_ladder(
            mapping, explicit,
            ({record.level_value for record in matched[key]}
             | unpublished.get(key, set()) for key in stacked),
        )
        # Records on a declared level the chosen era ladder omits are
        # not stacked: a publication that carries that level for some
        # fields and not others is read on the ladder all of them share.
        omitted = set(explicit) - set(ladder)
        if omitted:
            for key in stacked:
                matched[key] = [
                    record for record in matched[key]
                    if record.level_value not in omitted
                ]
        vertical_values = np.asarray(ladder, dtype=np.float64)
    else:
        level_sets = []
        # The ladder is read off the fields the source must publish whole;
        # a completed field states it only when nothing else does.
        ladder_from_completed = all(
            field_name in completed
            or "vertical" not in mapping["fields"][field_name]["target_axes"]
            for (_time, _member, field_name) in matched
        )
        for (_time, _member, field_name), group in matched.items():
            if field_name in completed and not ladder_from_completed:
                continue
            if "vertical" in mapping["fields"][field_name]["target_axes"]:
                level_sets.append(tuple(sorted({record.level_value for record in group})))
        if not level_sets or len(set(level_sets)) != 1:
            raise ValueError("GRIB atmospheric fields do not share one complete vertical inventory")
        vertical_values = np.asarray(level_sets[0], dtype=np.float64)

    hybrid_a = hybrid_b = None
    if str(mapping["coordinates"]["vertical"].get("kind")) == "hybrid_sigma_pressure":
        # Every selected vertical-bearing record states the whole ladder
        # in its pv octets; one source has one ladder, so disagreement
        # is a mixed-source input, not a choice to make.
        pv_lists = {
            tuple(record.coordinate_values)
            for (_time, _member, field_name), group in matched.items()
            if "vertical" in mapping["fields"][field_name]["target_axes"]
            for record in group
        }
        nonempty = {pv for pv in pv_lists if pv}
        if len(nonempty) > 1 or (nonempty and len(pv_lists) > 1):
            raise ValueError(
                "selected GRIB records do not share one pv coordinate "
                "list; hybrid A/B coefficients must be identical across "
                "the source"
            )
        hybrid_a, hybrid_b = _resolve_hybrid_coefficients(
            mapping["coordinates"]["vertical"],
            int(vertical_values.size),
            record_pv=next(iter(nonempty)) if nonempty else (),
        )

    direct: dict[tuple[datetime, str | None, str], _DirectValue] = {}
    cycles: dict[tuple[datetime, str | None], datetime] = {}
    for key, group in matched.items():
        valid_time, member, field_name = key
        field = mapping["fields"][field_name]
        source_axes = _axes(field["source_axes"], f"fields.{field_name}.source_axes")
        if "time" in source_axes or "member" in source_axes:
            raise ValueError(
                f"GRIB embedded time/member metadata must not also appear in {field_name}.source_axes"
            )
        stacking_axis = None
        stacking_values: Sequence[float] = ()
        if "vertical" in source_axes:
            stacking_axis = "vertical"
            stacking_values = vertical_values
        elif "half_level" in source_axes:
            stacking_axis = "half_level"
            stacking_values = mapping["coordinates"]["vertical"]["interface_levels"]
        elif "soil" in source_axes:
            stacking_axis = "soil"
            if mapping["target"].get("soil_layer_count") is None:
                raise ValueError("soil_layer_count is required to stack GRIB soil fields")
            selectors = field.get("selectors", [])
            if not selectors:
                raise ValueError(
                    f"{field_name} soil stacking requires ordered GRIB selectors"
                )
            # A soil-axis field stacks exactly its own ordered selector
            # ladder.  The CANONICAL pair's ladder length is separately
            # held to the target's soil_layer_count at frame
            # materialization; a non-canonical soil source (a
            # layer-integrated water column feeding a derivation) owns
            # its own length.
            if len(group) != len(selectors):
                raise ValueError(
                    f"{field_name} has {len(group)} GRIB soil records; the "
                    f"mapping declares {len(selectors)} ordered soil selectors"
                )
            stacking_values = tuple(range(len(group)))

        # Each record answers the FIRST of the field's selectors it
        # matches: a field lists its selectors in the order it prefers
        # them, so a record of a later selector stands in only where no
        # earlier one was published.  With one selector every record has
        # rank 0 and nothing here changes.
        field_selectors = field.get("selectors", [])
        policy = field["missing"]

        def rank_of(record: _GribRecord) -> int:
            for index, selector in enumerate(field_selectors):
                if _selector_matches_record(selector, record, source_format):
                    return index
            return 0

        def scaled(record: _GribRecord, rank: int) -> np.ndarray:
            # The record's values in the field's source units: a selector
            # may declare the factor that takes its record there.
            if rank < len(field_selectors) and "scale" in field_selectors[rank]:
                return record.values * float(field_selectors[rank]["scale"])
            return record.values

        used: list[_GribRecord] = list(group)
        absent_levels = 0
        # ``repeated_record: "last"``: the publisher writes this field's
        # record twice at one level and valid time (HRRRv1 wrfprs repeats
        # HGT at 1000 hPa and at the surface; the 1000 hPa pair differs,
        # and only the later one survives into HRRRv2 and every current
        # file), and the later message is the one this mapping means.
        # Without it a repeat is still refused.
        repeated_last = field.get("repeated_record") == "last"
        if stacking_axis is None:
            ranks = [rank_of(record) for record in group]
            best = min(ranks)
            chosen = [record for record, rank in zip(group, ranks) if rank == best]
            if len(chosen) > 1 and repeated_last:
                chosen = chosen[-1:]
            if len(chosen) != 1:
                raise ValueError(f"duplicate GRIB messages for scalar field {field_name} at {valid_time}")
            used = chosen
            values = scaled(chosen[0], best)
            materialized_axes = source_axes
        elif stacking_axis in {"vertical", "half_level"}:
            ranks = [rank_of(record) for record in group]
            best_rank: dict[float, int] = {}
            for record, rank in zip(group, ranks):
                best_rank[record.level_value] = min(
                    rank, best_rank.get(record.level_value, rank))
            by_level: dict[float, _GribRecord] = {}
            level_rank: dict[float, int] = {}
            for record, rank in zip(group, ranks):
                if best_rank[record.level_value] != rank:
                    continue
                if record.level_value in by_level and not repeated_last:
                    raise ValueError(
                        f"duplicate {field_name} GRIB level {record.level_value} at {valid_time}"
                    )
                by_level[record.level_value] = record
                level_rank[record.level_value] = rank
            # float(), not the raw numpy scalars: ``stacking_values`` is a
            # numpy array, so its elements repr as ``np.float64(100.0)``
            # under numpy 2 and as ``100.0`` under numpy 1.  That put the
            # installed numpy's version into a user-facing refusal -- the
            # same source refused with a different sentence depending on a
            # dependency the message is not about -- and it is a level
            # ladder being reported, not a dtype.
            missing = [float(level) for level in stacking_values
                       if level not in by_level]
            extra = [float(level) for level in by_level
                     if level not in set(stacking_values)]
            # A level the field is not published on (published_levels)
            # is zero when the file leaves it out; a published level the
            # file leaves out is still missing.  A record on such a level
            # is read like any other.
            published = (_published_levels(field)
                         if stacking_axis == "vertical" else None)
            zero_levels = set() if published is None else {
                level for level in missing if level not in published}
            missing = [level for level in missing if level not in zero_levels]
            # A field the frame completes may leave levels out; it is
            # derived there from the frame's own state.  Only under the
            # reject policy, whose NaN can mean nothing but "absent".
            completes = field_name in completed and policy["kind"] == "reject"
            if (missing and not completes) or extra:
                raise ValueError(
                    f"{field_name} vertical coverage mismatch; missing={missing}, extra={extra}"
                )
            absent_levels = len(missing)
            plane = next(iter(by_level.values())).values.shape
            used = []
            ordered = []
            for level in stacking_values:
                record = by_level.get(level)
                if record is None:
                    ordered.append(np.zeros(plane) if float(level) in zero_levels
                                   else np.full(plane, np.nan))
                    continue
                level_values = scaled(record, level_rank[level])
                if absent_levels and not np.isfinite(level_values).all():
                    raise ValueError(
                        f"{field_name} contains missing/non-finite GRIB data")
                used.append(record)
                ordered.append(level_values)
            axis = source_axes.index(stacking_axis)
            values = np.stack(ordered, axis=axis)
            materialized_axes = source_axes
        else:
            selectors = field["selectors"]
            by_selector: dict[int, _GribRecord] = {}
            for record in group:
                matched_selectors = [
                    index for index, selector in enumerate(selectors)
                    if _selector_matches_record(selector, record, source_format)
                ]
                if len(matched_selectors) != 1:
                    raise ValueError(
                        f"{field_name} GRIB soil record {record.index} matches "
                        f"{len(matched_selectors)} selectors; expected exactly one"
                    )
                selector_index = matched_selectors[0]
                if selector_index in by_selector:
                    raise ValueError(
                        f"duplicate {field_name} GRIB soil selector "
                        f"{selector_index} at {valid_time}"
                    )
                by_selector[selector_index] = record
            missing_selectors = sorted(set(range(len(selectors))) - set(by_selector))
            if missing_selectors:
                raise ValueError(
                    f"{field_name} is missing GRIB soil selectors {missing_selectors}"
                )
            ordered = [
                scaled(by_selector[index], index) for index in range(len(selectors))
            ]
            axis = source_axes.index(stacking_axis)
            values = np.stack(ordered, axis=axis)
            materialized_axes = source_axes

        missing_mask = ~np.isfinite(values)
        # A field with absent levels had each published level checked as
        # it was stacked, and its NaN planes are exactly the absent ones.
        if policy["kind"] == "reject" and not absent_levels and missing_mask.any():
            raise ValueError(f"{field_name} contains missing/non-finite GRIB data")
        values = np.asarray(values, dtype=np.float64).copy()
        if policy["kind"] == "value":
            values[missing_mask] = float(policy["value"])
            missing_mask[...] = False
        else:
            values[missing_mask] = np.nan
        values = _unit_transform(values, field, field_name)
        target_axes = _axes(field["target_axes"], f"fields.{field_name}.target_axes")
        values = _transpose_to_target(values, materialized_axes, target_axes, field_name)
        references = tuple(
            f"{record.source}:{record.index}" for record in sorted(used, key=lambda item: item.index)
        )
        cycles_for_group = {record.reference_time for record in used}
        if len(cycles_for_group) != 1:
            raise ValueError(f"{field_name} GRIB records mix source cycles at {valid_time}")
        cycle = cycles_for_group.pop()
        cycle_key = (valid_time, member)
        if cycle_key in cycles and cycles[cycle_key] != cycle:
            raise ValueError(f"mapped GRIB fields mix source cycles at {valid_time}")
        cycles[cycle_key] = cycle
        direct[key] = _DirectValue(
            name=field_name, valid_time=valid_time, member=member,
            source_cycle=cycle, axes=target_axes, values=values,
            missing_count=int(np.isnan(values).sum()), references=references,
        )

    direct, cycles = _broadcast_invariant_fields(mapping, direct, cycles)
    direct = _apply_landmask_water_missing(mapping, direct)

    declaration = _mapping_grid_declaration(mapping)
    if declaration.get("wind_basis") == "grid_relative_with_rotation":
        direct = _rotate_grid_relative_winds(direct, declaration)

    return _DecodedCollection(
        latitude=latitude,
        longitude=longitude,
        vertical_values=vertical_values,
        direct=MappingProxyType(direct),
        source_cycles=MappingProxyType(cycles),
        grid_fingerprint=next(iter(grid_fingerprints)),
        hybrid_a=hybrid_a,
        hybrid_b=hybrid_b,
    )


def _broadcast_invariant_fields(
    mapping: Mapping[str, object],
    direct: dict[tuple[datetime, str | None, str], "_DirectValue"],
    cycles: dict[tuple[datetime, str | None], datetime],
) -> tuple[
    dict[tuple[datetime, str | None, str], "_DirectValue"],
    dict[tuple[datetime, str | None], datetime],
]:
    """Bind declared time-invariant fields to every dependent valid time.

    Some providers publish invariant state ONCE per cycle (DWD-style
    ``time-invariant`` objects and ECMWF's analysis-frame statics: the
    land mask, the surface height, an ice analysis) while every
    prognostic field arrives per lead.  A field declaring
    ``time_binding: "cycle_invariant"`` is verified byte-invariant across
    every time it was actually supplied at, then bound to every valid
    time the time-dependent fields define.  Invariant fields never
    CREATE a valid time: a frame exists because prognostic state exists
    at it, so an invariant-only time (the publication time of a static
    object) is removed rather than materialized as a stateless frame.
    One broadcast belongs to one cycle: mixed source cycles refuse
    rather than hand one cycle's static to another that never proved it.
    """

    invariant = {
        name for name, field in mapping["fields"].items()
        if field.get("time_binding") == "cycle_invariant"
    }
    if not invariant:
        return direct, cycles
    dependent_keys = sorted(
        {(time, member) for (time, member, name) in direct if name not in invariant},
        key=lambda item: (item[0], str(item[1])),
    )
    if not dependent_keys:
        raise ValueError(
            "every decoded mapped field is declared time-invariant; there "
            "is no time-dependent state to define the forcing axis"
        )
    dependent_cycles = {
        str(cycles[key]) for key in dependent_keys if key in cycles
    }
    if len(dependent_cycles) > 1:
        raise ValueError(
            "cycle-invariant fields cannot broadcast across mixed source "
            f"cycles {sorted(dependent_cycles)}; one broadcast belongs to "
            "one cycle"
        )
    result = {
        key: value for key, value in direct.items() if key[2] not in invariant
    }
    kept_cycles = {
        key: cycle for key, cycle in cycles.items() if key in set(dependent_keys)
    }
    for name in sorted(invariant):
        instances = sorted(
            (
                (time, member, value)
                for (time, member, field_name), value in direct.items()
                if field_name == name
            ),
            key=lambda item: (item[0], str(item[1])),
        )
        if not instances:
            continue
        reference = instances[0][2]
        for _time, _member, value in instances[1:]:
            if not np.array_equal(reference.values, value.values):
                raise ValueError(
                    f"cycle-invariant field {name} changes across its "
                    "supplied valid times; the declaration promises one "
                    "array per source cycle"
                )
        for valid_time, member in dependent_keys:
            result[(valid_time, member, name)] = _DirectValue(
                name=name, valid_time=valid_time, member=member,
                source_cycle=kept_cycles[(valid_time, member)],
                axes=reference.axes, values=reference.values,
                missing_count=reference.missing_count,
                references=reference.references,
            )
    return result, kept_cycles


def _apply_landmask_water_missing(
    mapping: Mapping[str, object],
    direct: dict[tuple[datetime, str | None, str], "_DirectValue"],
) -> dict[tuple[datetime, str | None, str], "_DirectValue"]:
    """Mask declared landmask_water fields to missing on water cells.

    The provider filled water cells with sentinel values instead of an
    encoded missing mask, so the mapping's own land_fraction -- decoded
    from the same bytes, at the same valid time -- is the mask authority.
    Cells below the 0.5 land threshold become missing exactly as a
    bitmap would have made them, and the land/water-aware repair chain
    downstream runs unchanged.
    """

    masked_names = sorted(
        name for name, field in mapping["fields"].items()
        if field["missing"]["kind"] in ("landmask_water", "landmask_zero_water")
    )
    if not masked_names:
        return direct
    result = dict(direct)
    keys_by_time: dict[tuple[datetime, str | None], dict[str, tuple]] = {}
    for key in direct:
        keys_by_time.setdefault((key[0], key[1]), {})[key[2]] = key
    for (valid_time, member), by_name in sorted(
        keys_by_time.items(), key=lambda item: (item[0][0], str(item[0][1]))
    ):
        present = [name for name in masked_names if name in by_name]
        if not present:
            continue
        land_key = by_name.get("land_fraction")
        if land_key is None:
            raise ValueError(
                f"landmask_water fields {present} at {valid_time} have no "
                "decoded land_fraction to key water cells from"
            )
        land = direct[land_key].values
        if land.ndim != 2 or not np.isfinite(land).all():
            raise ValueError(
                "land_fraction must be a finite 2-D field to serve as the "
                "landmask_water mask"
            )
        water = land < 0.5
        for name in present:
            value = result[by_name[name]]
            masked = np.asarray(value.values, dtype=np.float64).copy()
            if mapping["fields"][name]["missing"]["kind"] == "landmask_zero_water":
                raise ValueError("landmask_zero_water requires the native mapped engine")
            masked[..., water] = np.nan
            result[by_name[name]] = _DirectValue(
                name=value.name, valid_time=value.valid_time,
                member=value.member, source_cycle=value.source_cycle,
                axes=value.axes, values=masked,
                missing_count=int(np.isnan(masked).sum()),
                references=value.references,
            )
    return result


def _rotate_grid_relative_winds(
    direct: dict[tuple[datetime, str | None, str], "_DirectValue"],
    declaration: Mapping[str, object],
) -> dict[tuple[datetime, str | None, str], "_DirectValue"]:
    """Rotate declared grid-relative wind pairs to the earth basis.

    The canonical names ``eastward_wind``/``northward_wind`` NAME the
    earth basis, so a projected source whose components ride the grid axes
    must rotate at decode time -- the same source-grid-to-earth step the
    native HRRR route performs before interpolation, with the same analytic
    cone rotation, computed here from the DECLARED Lambert parameters.
    Rotating one component without its partner is refused: half a vector
    has no basis.
    """

    sina, cosa = _declared_grid_rotation(declaration["parameters"])
    rotated = dict(direct)
    keys_by_time: dict[tuple[datetime, str | None], dict[str, tuple]] = {}
    for key in direct:
        keys_by_time.setdefault((key[0], key[1]), {})[key[2]] = key
    for (valid_time, member), by_name in keys_by_time.items():
        for u_name, v_name in _ROTATED_WIND_PAIRS:
            u_key = by_name.get(u_name)
            v_key = by_name.get(v_name)
            if u_key is None and v_key is None:
                continue
            if u_key is None or v_key is None:
                missing = u_name if u_key is None else v_name
                raise ValueError(
                    f"grid-relative wind rotation at {valid_time} needs "
                    f"both components of ({u_name}, {v_name}); {missing} "
                    "is not mapped"
                )
            u = direct[u_key]
            v = direct[v_key]
            if u.axes != v.axes or u.values.shape != v.values.shape:
                raise ValueError(
                    f"({u_name}, {v_name}) at {valid_time} disagree in "
                    "axes/shape; rotation needs one shared grid"
                )
            if u.axes[-2:] != ("y", "x"):
                raise ValueError(
                    f"({u_name}, {v_name}) must end in y/x axes to rotate"
                )
            if u.values.shape[-2:] != sina.shape:
                raise ValueError(
                    f"({u_name}, {v_name}) at {valid_time} do not share "
                    "the declared grid shape"
                )
            earth_u = u.values * cosa - v.values * sina
            earth_v = v.values * cosa + u.values * sina
            rotated[u_key] = _DirectValue(
                name=u.name, valid_time=u.valid_time, member=u.member,
                source_cycle=u.source_cycle, axes=u.axes, values=earth_u,
                missing_count=u.missing_count, references=u.references,
            )
            rotated[v_key] = _DirectValue(
                name=v.name, valid_time=v.valid_time, member=v.member,
                source_cycle=v.source_cycle, axes=v.axes, values=earth_v,
                missing_count=v.missing_count, references=v.references,
            )
    return rotated


def _decode_grib(
    mapping: Mapping[str, object], files: Sequence[Path], *,
    grib1_bridge: Path | None,
    grib2_inventory: Path | None,
    grib2_dump: Path | None,
) -> _DecodedCollection:
    from woof.ingest.codec import staged_decoded_object

    records: list[_GribRecord] = []
    aliased_records: list[_GribRecord] = []
    # Acquisition codec staging: an agency object wrapped in a registered
    # byte-level compression (DWD open data wraps every GRIB in bzip2)
    # decodes through a plainly staged twin whose lifetime is this decode;
    # provenance -- hashes, manifests, record references -- stays bound to
    # the supplied path, because those are the bytes acquisition delivered.
    #
    # This is the PYTHON engine's arm, and a bare default run no longer
    # reaches it: the mapped engine identifies the compression by magic
    # bytes and decodes it in process (`mapped-engine/src/codec.rs`), so
    # ICON-EU's 251 bz2-wrapped objects are decompressed in Rust on the
    # shipped route -- measured, as the source's compose parity row.
    # The staging below stays because this arm is the documented
    # workaround (`GPUWM_MAPPED_ENGINE=python`) and a workaround that
    # cannot read the same bytes is not one.
    with tempfile.TemporaryDirectory(prefix="gpuwm-acquisition-codec-") as codec_staging:
        payloads = [
            (Path(source), staged_decoded_object(source, codec_staging))
            for source in files
        ]
        if mapping["format"] == "grib1":
            bridge = Path(grib1_bridge) if grib1_bridge is not None else build_rust_bridge(release=True)
            if not bridge.is_file():
                raise FileNotFoundError(bridge)
            for source, payload in payloads:
                records.extend(_grib1_records(payload, bridge, source_label=source))
        else:
            if grib2_inventory is None or grib2_dump is None:
                inventory_executable, dump_executable = _build_grib2_tools()
            else:
                inventory_executable = Path(grib2_inventory)
                dump_executable = Path(grib2_dump)
            for executable in (inventory_executable, dump_executable):
                if not executable.is_file():
                    raise FileNotFoundError(executable)
            declaration = _mapping_grid_declaration(mapping)
            aliases = tuple(mapping.get("record_aliases") or ())
            inventoried_rows: list[Mapping[str, str]] = []
            for source, payload in payloads:
                rows = _grib2_inventory(payload, inventory_executable)
                inventoried_rows.extend(rows)
                wanted = _grib2_wanted_indices(mapping, rows)
                for record in _grib2_records(
                    payload, inventory_executable, dump_executable, wanted,
                    grid_declaration=declaration,
                    source_label=source,
                ):
                    record, aliased = _alias_record(record, aliases)
                    records.append(record)
                    if aliased:
                        aliased_records.append(record)
            if not records:
                # A total miss earns the identity diagnosis: the case of
                # two products under one filename, separable only by the
                # section-1 octets, must refuse by naming them.
                raise NothingMatched(_selector_identity_refusal(
                    mapping, inventoried_rows,
                    [source for source, _payload in payloads],
                ))
    collection = _assemble_grib(mapping, records)
    if aliased_records:
        from dataclasses import replace

        counts: dict[str, int] = {}
        for record in aliased_records:
            for name, field in mapping["fields"].items():
                if any(_selector_matches_record(selector, record, "grib2")
                       for selector in field.get("selectors", [])) \
                        and _declared_vertical_admits(mapping, field, record):
                    counts[name] = counts.get(name, 0) + 1
                    break
        collection = replace(collection, record_aliases=counts)
    return collection


def _resolve_direct_decoder_paths(
    source_format: str,
    *,
    grib1_bridge: str | Path | None,
    grib2_inventory: str | Path | None,
    grib2_dump: str | Path | None,
) -> dict[str, Path]:
    """Resolve the exact decoder paths that a direct decode will execute."""

    if source_format == "netcdf":
        if any(
            value is not None
            for value in (grib1_bridge, grib2_inventory, grib2_dump)
        ):
            raise ValueError("NetCDF mapped input does not use GRIB decoders")
        return {}
    if source_format == "grib1":
        if grib2_inventory is not None or grib2_dump is not None:
            raise ValueError("GRIB1 mapped input does not use GRIB2 decoders")
        bridge = (
            build_rust_bridge(release=True)
            if grib1_bridge is None
            else Path(grib1_bridge)
        ).resolve()
        if not bridge.is_file():
            raise FileNotFoundError(bridge)
        return {"grib1_bridge": bridge}
    if source_format != "grib2":
        raise ValueError(f"unsupported mapped source format {source_format!r}")
    if grib1_bridge is not None:
        raise ValueError("GRIB2 mapped input does not use a GRIB1 decoder")
    if (grib2_inventory is None) != (grib2_dump is None):
        raise ValueError(
            "GRIB2 inventory and dump decoders must be supplied together"
        )
    if grib2_inventory is None:
        inventory, dump = _build_grib2_tools()
    else:
        inventory, dump = Path(grib2_inventory), Path(grib2_dump)
    result = {
        "grib2_inventory": inventory.resolve(),
        "grib2_dump": dump.resolve(),
    }
    for path in result.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    return result


def _derivation_table(mapping: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    return {str(item["name"]): item for item in mapping.get("derivations", [])}


def _specific_humidity_from_rh(
    relative_humidity: np.ndarray,
    temperature: np.ndarray,
    pressure: np.ndarray,
) -> np.ndarray:
    # Reuse woof's WRF-real Bolton relation rather than maintaining another
    # humidity implementation in the generic adapter.
    from woof.ingest.real import _saturation_mixing_ratio

    mixing_ratio = _saturation_mixing_ratio(
        temperature, pressure, relative_humidity
    )
    return mixing_ratio / (1.0 + mixing_ratio)


#: Hydrostatic constants: ECMWF's own model-level geopotential build-up
#: (their compute-z-on-model-levels reference), so the derived heights
#: agree with the provider's archived pressure-level z.  Rd in J kg-1
#: K-1; the virtual factor is Rv/Rd - 1.
_HYDROSTATIC_RD = 287.06
_HYDROSTATIC_VIRTUAL = 0.609133
#: The provider's top-of-model clamp: the top interface pressure of a
#: full hybrid ladder is 0 Pa, whose logarithm does not exist, so the
#: top full level integrates against 0.1 Pa with alpha = ln 2.
_HYDROSTATIC_TOP_PA = 0.1

#: Canonical 3-D fields a pressure-level frame completes from its own
#: state when the source leaves them out, at some levels or at all of
#: them.  Keyed on canonical names, so every pressure-level source is
#: served without a line of per-source code: a source that publishes the
#: field keeps every value it publishes, and only the levels it does not
#: publish are derived.  ``mapped-engine``'s ``PRESSURE_COMPLETED_FIELDS``.
PRESSURE_COMPLETED_FIELDS = ("geopotential_height",)

#: What the hypsometric completion reads, by canonical name and unit.
HYPSOMETRIC_OPERANDS = (
    ("air_temperature", "K"),
    ("specific_humidity", "kg kg-1"),
    ("air_pressure", "Pa"),
    ("surface_pressure", "Pa"),
    ("terrain_height", "m"),
)

#: The first reference of a completed field, as
#: ``@completed.hypsometric:<values derived>``.  The preparation receipt
#: counts the derived values from it.
HYPSOMETRIC_COMPLETION_REFERENCE = COMPLETED_FIELD_REFERENCE_PREFIX + "hypsometric"

#: Standard gravity, the constant geopotential height is defined by.
_STANDARD_GRAVITY = 9.80665


def _completed_fields(mapping: Mapping[str, object]) -> tuple[str, ...]:
    """The pressure-level fields a frame of this mapping completes.

    Empty on every other vertical kind: a hybrid or model-level source
    declares its own hydrostatic derivation, and the completion integrates
    between pressure levels.
    """

    if str(mapping["coordinates"]["vertical"].get("kind")) != "pressure":
        return ()
    required = {
        str(item["name"])
        for item in mapping.get("target", {}).get("required_fields", [])
    }
    return tuple(name for name in PRESSURE_COMPLETED_FIELDS if name in required)


def _complete_geopotential_height(
    seed: CanonicalField | None,
    operands: Mapping[str, CanonicalField],
    name: str,
    units: str,
) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...]] | None:
    """Geopotential height where a pressure-level source leaves it out.

    ``seed`` is the source's own field, NaN on the levels (or cells) the
    source did not publish, or ``None`` when it published none of it.
    Every finite seed value is kept as it is.  Each missing value is
    placed by the hypsometric equation, dz = -(Rd / g) Tv dln(p), with
    virtual temperature taken linear in ln(p) between the published
    levels: from the nearest level of the same column that carries the
    source's own height (the level below on a tie), or from the surface
    when there is none, with the source's surface pressure and terrain
    height as the anchor and the virtual temperature at the surface
    interpolated between the levels that bracket it (held at the nearest
    level when the surface lies outside the ladder).

    The arithmetic is ``mapped-engine``'s ``complete_geopotential_height``
    operation for operation, and every logarithm is the C library's
    scalar ``log`` in both, so the two engines agree to the bit.
    ``None`` means an operand is absent and the frame keeps its
    missing-field refusal.
    """

    resolved = []
    for operand, unit in HYPSOMETRIC_OPERANDS:
        field = operands.get(operand)
        if field is None:
            return None
        if field.units != unit:
            raise ValueError(
                f"{name} is derived hydrostatically from {operand} in {unit}; "
                f"this frame carries it in {field.units}"
            )
        resolved.append(field)
    temperature, humidity, pressure, surface_pressure, terrain = resolved
    if units != "m":
        raise ValueError(
            f"{name} is completed in metres; the mapping declares {units}")
    column_axes = ("vertical", "y", "x")
    for field in (temperature, humidity, pressure, *(() if seed is None else (seed,))):
        if tuple(field.axes) != column_axes:
            raise ValueError(
                f"{name} is derived hydrostatically on ('vertical', 'y', 'x') "
                f"axes; {field.name} has {list(field.axes)}"
            )
    for field in (surface_pressure, terrain):
        if tuple(field.axes) != ("y", "x"):
            raise ValueError(
                f"{name} is derived hydrostatically from {field.name} on "
                f"('y', 'x') axes; it has {list(field.axes)}"
            )
    shape = temperature.values.shape
    levels = shape[0]
    for field in (humidity, pressure, *(() if seed is None else (seed,))):
        if field.values.shape != shape:
            raise ValueError(
                f"{name} hydrostatic operands disagree in shape: "
                f"{field.name} is {list(field.values.shape)}, "
                f"air_temperature is {list(shape)}"
            )
    for field in (surface_pressure, terrain):
        if field.values.shape != shape[1:]:
            raise ValueError(
                f"{name} hydrostatic operands disagree in shape: "
                f"{field.name} is {list(field.values.shape)}, the column "
                f"plane is {list(shape[1:])}"
            )
    plane = int(shape[1] * shape[2])
    pressure_rows = np.asarray(pressure.values, dtype=np.float64).reshape(levels, plane)
    level_pressure = []
    for level in range(levels):
        row = pressure_rows[level]
        first = float(row[0])
        if not math.isfinite(first) or first <= 0.0 or bool((row != first).any()):
            raise ValueError(
                f"{name} is derived hydrostatically on a pressure ladder with "
                f"one positive pressure per level; air_pressure level {level} "
                "is not one"
            )
        level_pressure.append(first)
    # Bottom first: the highest pressure is the lowest level.  Both sorts
    # are stable, the engine's too, so a tie keeps index order.
    order = sorted(range(levels), key=lambda level: -level_pressure[level])
    log_pressure = [math.log(level_pressure[level]) for level in order]
    candidates = [
        sorted(
            (other for other in range(levels) if other != position),
            key=lambda other, position=position: (
                abs(log_pressure[position] - log_pressure[other]), other),
        )
        for position in range(levels)
    ]
    temperature_rows = np.asarray(temperature.values, dtype=np.float64).reshape(levels, plane)
    humidity_rows = np.asarray(humidity.values, dtype=np.float64).reshape(levels, plane)
    surface_flat = np.asarray(surface_pressure.values, dtype=np.float64).reshape(plane)
    terrain_flat = np.asarray(terrain.values, dtype=np.float64).reshape(plane)
    if seed is None:
        values = np.full((levels, plane), np.nan)
    else:
        values = np.asarray(seed.values, dtype=np.float64).reshape(levels, plane).copy()
    known = np.isfinite(values[order])
    columns = np.flatnonzero(~known.all(axis=0))
    factor = _HYDROSTATIC_RD / _STANDARD_GRAVITY
    completed = 0
    if columns.size:
        known = known[:, columns]
        virtual = temperature_rows[order][:, columns] * (
            1.0 + _HYDROSTATIC_VIRTUAL * humidity_rows[order][:, columns])
        # The first failing column is refused, and within one column the
        # virtual temperature is checked before the surface, as the engine
        # walks them.
        bad_virtual = ~np.isfinite(virtual) | (virtual <= 0.0)
        surface = surface_flat[columns]
        height = terrain_flat[columns]
        bad_surface = ~np.isfinite(surface) | (surface <= 0.0) | ~np.isfinite(height)
        virtual_cells = np.flatnonzero(bad_virtual.any(axis=0))
        surface_cells = np.flatnonzero(bad_surface)
        first_virtual = int(virtual_cells[0]) if virtual_cells.size else columns.size
        first_surface = int(surface_cells[0]) if surface_cells.size else columns.size
        if first_virtual < columns.size and first_virtual <= first_surface:
            position = int(np.flatnonzero(bad_virtual[:, first_virtual])[0])
            raise ValueError(
                f"{name} is derived hydrostatically where the source leaves "
                "it out, which needs finite positive virtual temperature; "
                "air_temperature and specific_humidity give "
                f"{float(virtual[position, first_virtual])} at column "
                f"{int(columns[first_virtual])}"
            )
        if first_surface < columns.size:
            raise ValueError(
                f"{name} is derived hydrostatically where the source leaves "
                "it out, which needs finite positive surface_pressure and "
                f"finite terrain_height; column {int(columns[first_surface])} "
                f"has {float(surface[first_surface])} Pa and "
                f"{float(height[first_surface])} m"
            )
        thickness = np.empty_like(virtual)
        thickness[0] = 0.0
        for position in range(1, levels):
            thickness[position] = thickness[position - 1] + 0.5 * (
                virtual[position - 1] + virtual[position]
            ) * (log_pressure[position - 1] - log_pressure[position])
        log_surface = np.fromiter(
            (math.log(value) for value in surface.tolist()),
            dtype=np.float64, count=surface.size,
        )
        top = levels - 1
        surface_thickness = np.empty_like(log_surface)
        below_ladder = log_surface >= log_pressure[0]
        above_ladder = ~below_ladder & (log_surface <= log_pressure[top])
        surface_thickness[below_ladder] = (
            thickness[0][below_ladder]
            - virtual[0][below_ladder] * (log_surface[below_ladder] - log_pressure[0])
        )
        surface_thickness[above_ladder] = (
            thickness[top][above_ladder]
            + virtual[top][above_ladder] * (log_pressure[top] - log_surface[above_ladder])
        )
        inside = ~below_ladder & ~above_ladder
        for below in range(top):
            bracket = inside & (log_pressure[below] >= log_surface) & (
                log_surface > log_pressure[below + 1])
            if not bracket.any():
                continue
            weight = (log_pressure[below] - log_surface[bracket]) / (
                log_pressure[below] - log_pressure[below + 1])
            at_surface = virtual[below][bracket] + weight * (
                virtual[below + 1][bracket] - virtual[below][bracket])
            surface_thickness[bracket] = thickness[below][bracket] + 0.5 * (
                virtual[below][bracket] + at_surface
            ) * (log_pressure[below] - log_surface[bracket])
        for position in range(levels):
            wanted = ~known[position]
            if not wanted.any():
                continue
            placed = np.zeros(wanted.shape, dtype=bool)
            result = np.empty(wanted.shape, dtype=np.float64)
            for other in candidates[position]:
                take = wanted & ~placed & known[other]
                if not take.any():
                    continue
                result[take] = values[order[other]][columns[take]] + factor * (
                    thickness[position][take] - thickness[other][take])
                placed |= take
            rest = wanted & ~placed
            result[rest] = height[rest] + factor * (
                thickness[position][rest] - surface_thickness[rest])
            values[order[position], columns[wanted]] = result[wanted]
            completed += int(wanted.sum())
    references = [f"{HYPSOMETRIC_COMPLETION_REFERENCE}:{completed}"]
    for field in (*(() if seed is None else (seed,)), temperature, humidity,
                  pressure, surface_pressure, terrain):
        references.extend(field.source_references)
    return (
        values.reshape(shape),
        column_axes,
        tuple(dict.fromkeys(references)),
    )


def _complete_field(
    name: str,
    field: Mapping[str, object],
    seed: CanonicalField | None,
    operands: Mapping[str, CanonicalField],
) -> CanonicalField | None:
    """One completed field as a canonical field; ``None`` while an operand
    of its completion is absent."""

    units = str(field["units"]["target"])
    if name != "geopotential_height":
        raise AssertionError(f"{name} has no completion")
    completed = _complete_geopotential_height(seed, operands, name, units)
    if completed is None:
        return None
    values, axes, references = completed
    return CanonicalField(
        name=name, units=units, axes=axes, location=str(field["location"]),
        staggering=str(field.get("staggering", "none")), values=values,
        missing_count=int(np.isnan(values).sum()), source_references=references,
    )


def _hybrid_half_level_pressure(
    collection: _DecodedCollection,
    surface_pressure: np.ndarray,
    name: str,
) -> np.ndarray:
    """The (N+1, y, x) or (N, y, x) ladder p = A + B * ps, gated strict."""

    if collection.hybrid_a is None or collection.hybrid_b is None:
        raise ValueError(
            f"{name} requires resolved hybrid A/B coefficients, which this "
            "decoded collection does not carry"
        )
    if not np.isfinite(surface_pressure).all() \
            or np.any(surface_pressure <= 0.0):
        raise ValueError(
            f"{name} requires finite positive surface pressure to price "
            "the hybrid ladder"
        )
    ladder = collection.hybrid_a[:, None, None] \
        + collection.hybrid_b[:, None, None] * surface_pressure[None, :, :]
    if not np.all(ladder[1:] > ladder[:-1]):
        raise ValueError(
            f"{name} hybrid pressure must increase strictly from the top "
            "of the atmosphere downward at every cell; the resolved A/B "
            "ladder does not (check coefficient order against the "
            "declared levels)"
        )
    return ladder


def _libm_log(values: np.ndarray) -> np.ndarray:
    """The C library's scalar ``log`` over ``values``, one value at a time.

    The mapped engine's ``f64::ln`` is that same call, so the two engines
    agree to the bit; numpy's vectorized logarithm may round a last bit
    differently on some machines.
    """

    def one(value: float) -> float:
        if value > 0.0:
            return math.log(value)
        return -math.inf if value == 0.0 else math.nan

    return np.fromiter((one(value) for value in values.tolist()),
                       dtype=np.float64, count=values.size)


def _libm_exp(values: np.ndarray) -> np.ndarray:
    """The C library's scalar ``exp`` over ``values``; see _libm_log."""

    def one(value: float) -> float:
        try:
            return math.exp(value)
        except OverflowError:
            return math.inf

    return np.fromiter((one(value) for value in values.tolist()),
                       dtype=np.float64, count=values.size)


#: Below this surface height WRF reduces sea-level pressure along the
#: lowest layer's pressure gradient instead of interpolating.
_SEA_LEVEL_SHALLOW_M = 50.0


def _surface_pressure_from_sea_level(
    sea_level_pressure: np.ndarray,
    level_height: np.ndarray,
    pressure: np.ndarray,
    surface_height: np.ndarray,
    name: str,
) -> np.ndarray:
    """WRF real's ``sfcprs3`` relation in float64, every column at once.

    The mapped engine's ``derive::surface_pressure_from_sea_level``,
    operation for operation, so both engines publish the same bytes.
    Levels are ordered by the first column's pressure, highest first,
    and every column must keep that order strictly.  A surface under
    50 m takes sea-level pressure plus the lowest layer's gradient times
    its height; a higher one is interpolated in log pressure between the
    two levels whose heights bracket it, or, when it lies below every
    level, between sea level and the second level (WRF's own choice) or
    the first level under the sea-level pressure.  The search bounds
    keep WRF's excluded top levels.  The first column that cannot be
    reduced refuses by its row and column.
    """

    slp = np.asarray(sea_level_pressure, dtype=np.float64)
    zm = np.asarray(surface_height, dtype=np.float64)
    heights = np.asarray(level_height, dtype=np.float64)
    pressures = np.asarray(pressure, dtype=np.float64)
    if zm.ndim != 2 or zm.size == 0 or slp.shape != zm.shape:
        raise ValueError(
            f"{name} requires sea-level pressure and surface height on the "
            "same plane")
    if pressures.ndim != 3 or pressures.shape[0] < 2 \
            or heights.shape != pressures.shape \
            or pressures.shape[1:] != zm.shape:
        raise ValueError(
            f"{name} requires level heights and pressures on at least two "
            "common levels")
    levels = pressures.shape[0]
    order = np.argsort(-pressures[:, 0, 0], kind="stable")
    z = heights[order]
    p = pressures[order]
    with np.errstate(all="ignore"):
        surface_bad = ~(np.isfinite(zm) & np.isfinite(slp) & (slp > 0.0))
        level_bad = ~(np.isfinite(z) & np.isfinite(p) & (p > 0.0))
        order_bad = np.concatenate(
            [np.zeros((1, *zm.shape), dtype=bool), p[:-1] <= p[1:]])
        # Per column, the first level that fails either check decides
        # which sentence it earns, as the column walk does.
        first_level = np.where(level_bad.any(axis=0),
                               level_bad.argmax(axis=0), levels)
        first_order = np.where(order_bad.any(axis=0),
                               order_bad.argmax(axis=0), levels)

        # The lowest qualifying level wins, as the column walk's first
        # match does, so each search runs top down and the last write
        # is the lowest k.
        bracket = np.full(zm.shape, -1, dtype=np.int64)
        for k in range(levels - 3, -1, -1):
            bracket = np.where((z[k] <= zm) & (z[k + 1] > zm), k, bracket)
        fallback = np.full(zm.shape, -1, dtype=np.int64)
        for k in range(levels - 4, -1, -1):
            fallback = np.where((slp >= p[k + 1]) & (slp < p[k]), k, fallback)
        branch = np.select(
            [zm < _SEA_LEVEL_SHALLOW_M, bracket >= 0, slp >= p[0],
             fallback >= 0],
            [0, 1, 2, 3], default=-1)
        valid = ~surface_bad & (first_level >= levels) \
            & (first_order >= levels)
        values = np.full(zm.shape, np.nan)
        shallow = valid & (branch == 0)
        values[shallow] = (slp + (p[0] - p[1]) / (z[0] - z[1]) * zm)[shallow]
        # Log-pressure interpolation between (zl, pl) and (zu, pu): the
        # bracketing levels, or sea level and the second level, or sea
        # level and the first level under the sea-level pressure.
        low = np.clip(bracket, 0, max(levels - 2, 0))[None]
        up = np.clip(fallback + 1, 0, levels - 1)[None]
        zl = np.where(branch == 1, np.take_along_axis(z, low, 0)[0], 0.0)
        pl = np.where(branch == 1, np.take_along_axis(p, low, 0)[0], slp)
        zu = np.select(
            [branch == 1, branch == 2],
            [np.take_along_axis(z, low + 1, 0)[0], z[1]],
            default=np.take_along_axis(z, up, 0)[0])
        pu = np.select(
            [branch == 1, branch == 2],
            [np.take_along_axis(p, low + 1, 0)[0], p[1]],
            default=np.take_along_axis(p, up, 0)[0])
        cells = valid & (branch > 0)
        zl, zu, pl, pu, zc = (value[cells] for value in (zl, zu, pl, pu, zm))
        values[cells] = _libm_exp(
            (_libm_log(pl) * (zc - zu) + _libm_log(pu) * (zl - zc))
            / (zl - zu))
        unbracketed = valid & (branch == -1)
        result_bad = valid & ~(np.isfinite(values) & (values > 0.0))
    column_bad = surface_bad | (first_level < levels) \
        | (first_order < levels) | unbracketed | result_bad
    if column_bad.any():
        cell = int(np.flatnonzero(column_bad.ravel())[0])
        row, column = divmod(cell, zm.shape[1])
        if surface_bad.ravel()[cell]:
            reason = ("the sea-level pressure or surface height is not "
                      "finite and positive")
        elif first_level.ravel()[cell] < levels \
                and first_level.ravel()[cell] <= first_order.ravel()[cell]:
            reason = "a level height or pressure is not finite and positive"
        elif first_order.ravel()[cell] < levels:
            reason = "the level pressures do not keep one strict order"
        elif unbracketed.ravel()[cell]:
            reason = ("no level brackets the surface height or the "
                      "sea-level pressure")
        else:
            reason = "the reduced pressure is not finite and positive"
        raise ValueError(
            f"{name} cannot reduce sea-level pressure at row {row}, column "
            f"{column}: {reason}")
    return values


def _evaluate_derivation(
    operation: Mapping[str, object],
    available: Mapping[str, CanonicalField],
    collection: _DecodedCollection,
    field: Mapping[str, object],
    name: str,
    *,
    vertical: Mapping[str, object] | None = None,
) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...]]:
    kind = str(operation["operation"])

    def dependency(label: str) -> CanonicalField:
        dependency_name = str(operation[label])
        try:
            return available[dependency_name]
        except KeyError as error:
            raise KeyError(dependency_name) from error

    def surface_pressure_dependency() -> CanonicalField:
        if vertical is None:
            raise ValueError(
                f"{name} requires the mapping's vertical coordinate to "
                "resolve its declared surface pressure field"
            )
        pressure_name = str(vertical["surface_pressure_field"])
        try:
            resolved = available[pressure_name]
        except KeyError as error:
            raise KeyError(pressure_name) from error
        if tuple(resolved.axes) != ("y", "x"):
            raise ValueError(
                f"{name} requires the declared surface pressure field "
                f"{pressure_name!r} on ('y', 'x') axes; got {resolved.axes}"
            )
        return resolved

    if kind == "copy":
        source = dependency("source")
        raw = source.values
        axes = source.axes
        references = source.source_references
    elif kind == "wind_speed":
        u = dependency("u")
        v = dependency("v")
        if u.axes != v.axes or u.values.shape != v.values.shape:
            raise ValueError(f"{name} wind derivation dependencies disagree")
        raw = np.hypot(u.values, v.values)
        axes = u.axes
        references = (*u.source_references, *v.source_references)
    elif kind == "geopotential_height":
        geopotential = dependency("geopotential")
        gravity = float(operation.get("gravity_m_s2", 9.80665))
        if not math.isfinite(gravity) or gravity <= 0.0:
            raise ValueError(f"{name} declares invalid gravity")
        raw = geopotential.values / gravity
        axes = geopotential.axes
        references = geopotential.source_references
    elif kind == "mass_fraction_rebase":
        source = dependency("source")
        excluded = [available[str(key)] for key in operation["exclude"]]
        denominator = np.ones_like(source.values)
        references = list(source.source_references)
        for fraction in [source, *excluded]:
            if (fraction.units != "kg kg-1" or fraction.axes != source.axes
                    or fraction.values.shape != source.values.shape
                    or not np.isfinite(fraction.values).all() or np.any(fraction.values < 0)):
                raise ValueError(f"{name} requires finite nonnegative mass fractions on identical axes")
        for fraction in excluded:
            denominator -= fraction.values
            references.extend(fraction.source_references)
        if np.any(denominator <= 0):
            raise ValueError(f"{name} excluded mass fractions leave no positive reference mass")
        raw = source.values / denominator
        axes = source.axes
    elif kind == "mass_from_interfaces":
        source = dependency("source")
        if (source.axes != ("half_level", "y", "x")
                or source.units != field["units"]["source"]):
            raise ValueError(f"{name} requires matching units on half_level,y,x")
        if (source.values.shape[0] != collection.vertical_values.size + 1
                or not np.isfinite(source.values).all()):
            raise ValueError(f"{name} requires N+1 finite interface values")
        raw = 0.5 * (source.values[:-1] + source.values[1:])
        axes = ("vertical", "y", "x")
        references = source.source_references
    elif kind == "height_from_interfaces":
        source = dependency("source")
        if source.axes != ("half_level", "y", "x") or source.units != "m":
            raise ValueError(f"{name} requires interface heights in m on half_level,y,x axes")
        heights = source.values
        if heights.shape[0] != collection.vertical_values.size + 1 or not np.isfinite(heights).all():
            raise ValueError(f"{name} requires N+1 finite interface heights")
        thickness = np.diff(heights, axis=0)
        if not (np.all(thickness > 0) or np.all(thickness < 0)):
            raise ValueError(f"{name} interface heights must be strictly ordered without crossing layers")
        raw = 0.5 * (heights[:-1] + heights[1:])
        axes = ("vertical", "y", "x")
        references = source.source_references
    elif kind == "pressure_from_vertical_coordinate":
        axes = _axes(field["source_axes"], f"fields.{name}.source_axes")
        if axes != ("vertical", "y", "x"):
            raise ValueError(
                f"{name} pressure derivation currently requires source_axes "
                "['vertical','y','x']"
            )
        if vertical is not None \
                and str(vertical.get("kind")) == "hybrid_sigma_pressure":
            # p = A + B*ps on the resolved ladder: half-level interfaces
            # average to full levels; full-level coefficients state the
            # level pressure directly.
            pressure = surface_pressure_dependency()
            ladder = _hybrid_half_level_pressure(
                collection, pressure.values, name,
            )
            if ladder.shape[0] == collection.vertical_values.size + 1:
                raw = 0.5 * (ladder[:-1] + ladder[1:])
            else:
                raw = ladder
            references = (
                "@coordinate.vertical.hybrid", *pressure.source_references,
            )
        else:
            raw = np.broadcast_to(
                collection.vertical_values[:, None, None],
                (collection.vertical_values.size,
                 collection.latitude.size, collection.longitude.size),
            )
            references = ("@coordinate.vertical",)
    elif kind == "relative_humidity_from_dewpoint":
        dewpoint = dependency("dewpoint")
        temperature = dependency("temperature")
        if dewpoint.axes != temperature.axes:
            raise ValueError(f"{name} dewpoint/temperature axes disagree")
        from woof.ingest.real import _surface_relative_humidity

        raw = _surface_relative_humidity(dewpoint.values, temperature.values)
        axes = dewpoint.axes
        references = (*dewpoint.source_references, *temperature.source_references)
    elif kind in {"specific_humidity_from_rh", "specific_humidity_from_dewpoint"}:
        temperature = dependency("temperature")
        pressure = dependency("pressure")
        if kind == "specific_humidity_from_rh":
            humidity = dependency("relative_humidity")
            relative_humidity = humidity.values
            references = (
                *humidity.source_references, *temperature.source_references,
                *pressure.source_references,
            )
            axes = humidity.axes
        else:
            dewpoint = dependency("dewpoint")
            from woof.ingest.real import _surface_relative_humidity

            relative_humidity = _surface_relative_humidity(
                dewpoint.values, temperature.values
            )
            references = (
                *dewpoint.source_references, *temperature.source_references,
                *pressure.source_references,
            )
            axes = dewpoint.axes
        if axes != temperature.axes or axes != pressure.axes:
            raise ValueError(f"{name} humidity derivation dependency axes disagree")
        raw = _specific_humidity_from_rh(
            relative_humidity, temperature.values, pressure.values
        )
    elif kind == "air_temperature_from_potential_temperature":
        theta = dependency("potential_temperature")
        pressure = dependency("pressure")
        if theta.axes != pressure.axes or theta.values.shape != pressure.values.shape:
            raise ValueError(f"{name} potential temperature and pressure disagree in axes or shape")
        if theta.units != "K" or pressure.units != "Pa":
            raise ValueError(f"{name} requires potential temperature in K and pressure in Pa")
        valid = np.isfinite(theta.values) & np.isfinite(pressure.values)
        if np.any(theta.values[valid] <= 0.0) or np.any(pressure.values[valid] <= 0.0):
            raise ValueError(f"{name} requires positive potential temperature and pressure")
        reference = float(operation.get("reference_pressure_pa", 100000.0))
        kappa = float(operation.get("kappa", 2.0 / 7.0))
        raw = theta.values * np.power(pressure.values / reference, kappa)
        axes = theta.axes
        references = (*theta.source_references, *pressure.source_references)
    elif kind == "specific_humidity_from_mixing_ratio":
        ratio = dependency("mixing_ratio")
        if ratio.units != "kg kg-1":
            raise ValueError(f"{name} requires a mixing ratio in kg kg-1")
        valid = np.isfinite(ratio.values)
        if np.any(ratio.values[valid] <= -1.0):
            raise ValueError(f"{name} mixing ratio at or below -1 has no specific humidity")
        raw = ratio.values / (1.0 + ratio.values)
        axes = ratio.axes
        references = ratio.source_references
    elif kind == "volumetric_soil_moisture_from_layer_mass":
        layer_mass = dependency("layer_mass")
        axes = layer_mass.axes
        if "soil" not in axes:
            raise ValueError(
                f"{name} layer-mass derivation requires a soil axis on its "
                "layer_mass dependency"
            )
        bounds = [
            (float(pair[0]), float(pair[1]))
            for pair in operation["layer_bounds_m"]
        ]
        soil_axis = axes.index("soil")
        if layer_mass.values.shape[soil_axis] != len(bounds):
            raise ValueError(
                f"{name} declares {len(bounds)} soil layer bounds but its "
                f"layer_mass column has "
                f"{layer_mass.values.shape[soil_axis]} layers"
            )
        density = float(operation.get("water_density_kg_m3", 1000.0))
        thickness_shape = [1] * layer_mass.values.ndim
        thickness_shape[soil_axis] = len(bounds)
        thickness = np.asarray(
            [bottom - top for top, bottom in bounds], dtype=np.float64,
        ).reshape(thickness_shape)
        raw = layer_mass.values / (density * thickness)
        references = layer_mass.source_references
    elif kind == "surface_pressure_from_sea_level":
        sea_level = dependency("sea_level_pressure")
        height = dependency("level_height")
        pressure = dependency("pressure")
        surface = dependency("surface_height")
        if tuple(sea_level.axes) != ("y", "x") \
                or tuple(surface.axes) != ("y", "x"):
            raise ValueError(
                f"{name} requires sea-level pressure and surface height on "
                "('y', 'x') axes")
        if tuple(height.axes) != ("vertical", "y", "x") \
                or tuple(pressure.axes) != tuple(height.axes) \
                or pressure.values.shape != height.values.shape \
                or height.values.shape[1:] != surface.values.shape \
                or sea_level.values.shape != surface.values.shape:
            raise ValueError(
                f"{name} requires level heights and pressures on one "
                "('vertical', 'y', 'x') grid over the surface plane")
        raw = _surface_pressure_from_sea_level(
            sea_level.values, height.values, pressure.values, surface.values,
            name)
        axes = surface.axes
        references = (
            "@derived.sea_level_reduction",
            *sea_level.source_references,
            *height.source_references,
            *pressure.source_references,
            *surface.source_references,
        )
    elif kind == "soil_surface_node_from_shallowest":
        source = dependency("source")
        axes = source.axes
        if "soil" not in axes:
            raise ValueError(
                f"{name} surface-node derivation requires a soil axis on "
                "its source dependency"
            )
        soil_axis = axes.index("soil")
        shallowest = np.take(source.values, [0], axis=soil_axis)
        raw = np.concatenate([shallowest, source.values], axis=soil_axis)
        references = source.source_references
    elif kind == "geopotential_height_hydrostatic":
        temperature = dependency("temperature")
        humidity = dependency("specific_humidity")
        surface_height = dependency("surface_geopotential_height")
        pressure = surface_pressure_dependency()
        axes = _axes(field["source_axes"], f"fields.{name}.source_axes")
        if axes != ("vertical", "y", "x"):
            raise ValueError(
                f"{name} hydrostatic derivation currently requires "
                "source_axes ['vertical','y','x']"
            )
        if tuple(temperature.axes) != axes or tuple(humidity.axes) != axes:
            raise ValueError(
                f"{name} hydrostatic derivation requires temperature and "
                "specific humidity on ('vertical', 'y', 'x') axes"
            )
        if tuple(surface_height.axes) != ("y", "x"):
            raise ValueError(
                f"{name} hydrostatic derivation requires surface "
                "geopotential height on ('y', 'x') axes"
            )
        gravity = float(operation.get("gravity_m_s2", 9.80665))
        if not math.isfinite(gravity) or gravity <= 0.0:
            raise ValueError(f"{name} declares invalid gravity")
        nlevels = int(collection.vertical_values.size)
        ladder = _hybrid_half_level_pressure(collection, pressure.values, name)
        if ladder.shape[0] != nlevels + 1:
            raise ValueError(
                f"{name} hydrostatic integration requires half-level "
                f"interface coefficients: {nlevels} levels need "
                f"{nlevels + 1} A/B values, this source resolves "
                f"{ladder.shape[0]}"
            )
        # ECMWF's model-level build-up: virtual temperature per full
        # level, geopotential accumulated interface to interface from
        # the surface upward, the full level placed by its alpha.
        virtual = temperature.values * (
            1.0 + _HYDROSTATIC_VIRTUAL * humidity.values
        )
        if not np.isfinite(virtual).all() or np.any(virtual <= 0.0):
            raise ValueError(
                f"{name} hydrostatic integration requires finite positive "
                "virtual temperature"
            )
        log2 = math.log(2.0)
        phi_half = gravity * np.asarray(surface_height.values, dtype=np.float64)
        raw = np.empty(
            (nlevels, phi_half.shape[0], phi_half.shape[1]), dtype=np.float64,
        )
        for level in range(nlevels - 1, -1, -1):
            below = ladder[level + 1]
            above = ladder[level]
            positive_above = above > 0.0
            log_ratio = pm.log(
                below / np.where(positive_above, above, _HYDROSTATIC_TOP_PA)
            )
            alpha = np.where(
                positive_above,
                1.0 - (above / (below - above)) * log_ratio,
                log2,
            )
            energy = _HYDROSTATIC_RD * virtual[level]
            raw[level] = (phi_half + energy * alpha) / gravity
            phi_half = phi_half + energy * log_ratio
        references = (
            "@derived.hydrostatic",
            *temperature.source_references,
            *humidity.source_references,
            *surface_height.source_references,
            *pressure.source_references,
        )
    else:
        raise ValueError(f"unsupported derivation operation {kind!r}")

    source_axes = _axes(field["source_axes"], f"fields.{name}.source_axes")
    target_axes = _axes(field["target_axes"], f"fields.{name}.target_axes")
    if tuple(axes) != source_axes:
        raise ValueError(
            f"derived {name} produced axes {tuple(axes)}, expected {source_axes}"
        )
    converted = _unit_transform(raw, field, name)
    converted = _transpose_to_target(converted, source_axes, target_axes, name)
    return converted, target_axes, tuple(dict.fromkeys(references))


def _frame_header(
    mapping: Mapping[str, object],
    *,
    valid_time: datetime,
    source_cycle: datetime,
    latitude: np.ndarray,
    longitude: np.ndarray,
    vertical_values: np.ndarray,
    fields: Mapping[str, CanonicalField],
    source_id: str,
    hybrid_a: np.ndarray | None = None,
    hybrid_b: np.ndarray | None = None,
) -> SourceFrameHeader:
    vertical = mapping["coordinates"]["vertical"]
    vertical_kind = str(vertical["kind"])
    source_vertical_name = "atmosphere"
    descriptors: dict[str, VerticalDescriptor] = {
        source_vertical_name: VerticalDescriptor(
            coordinate={
                "hybrid_sigma_pressure": "hybrid",
                "embedded_levels": "model_level",
            }.get(vertical_kind, vertical_kind),
            level_count=int(vertical_values.size),
            level_values=tuple(float(value) for value in vertical_values),
            a_coefficients=(
                () if hybrid_a is None
                else tuple(float(value) for value in hybrid_a)
            ),
            b_coefficients=(
                () if hybrid_b is None
                else tuple(float(value) for value in hybrid_b)
            ),
            positive=str(vertical.get("positive", "down")),
            units=str(vertical["units"]),
        )
    }
    soil_fields = [field for field in fields.values() if "soil" in field.axes]
    if soil_fields:
        soil_count = soil_fields[0].values.shape[soil_fields[0].axes.index("soil")]
        descriptors["soil"] = VerticalDescriptor(
            coordinate="soil_depth", level_count=soil_count,
            units="index", positive="down",
        )
    lead = int((valid_time - source_cycle).total_seconds())
    time = TimeDescriptor(
        reference_time=source_cycle.replace(tzinfo=timezone.utc).isoformat(),
        valid_time=valid_time.replace(tzinfo=timezone.utc).isoformat(),
        lead_seconds=lead,
    )
    field_descriptors = []
    for field in fields.values():
        field_descriptors.append(FieldDescriptor(
            canonical_name=field.name,
            units=field.units,
            dimensions=field.axes,
            grid_location=field.location,
            vertical_coordinate=(
                source_vertical_name if "vertical" in field.axes
                else "soil" if "soil" in field.axes else None
            ),
            time=time,
            data_reference="sha256:" + _array_sha256(field.values),
            dtype=field.values.dtype.str,
            shape=field.values.shape,
            missing_value_policy=(
                "explicit_missing" if field.missing_count else "reject_nonfinite"
            ),
            source_field=";".join(field.source_references),
        ))
    declaration = _mapping_grid_declaration(mapping)
    if declaration["family"] == GRID_FAMILY_LAMBERT:
        parameters = declaration["parameters"]
        grid_descriptor = GridDescriptor(
            projection=GRID_FAMILY_LAMBERT,
            nx=int(longitude.size), ny=int(latitude.size),
            earth_shape=f"grib_shape_of_earth:{parameters['shape_of_earth']}",
            scan_order="+x,+y",
            # Grid-relative sources were rotated to the earth basis at
            # decode time (_rotate_grid_relative_winds); the frame states
            # what its arrays ARE, not what the producer published.
            wind_basis="earth_relative",
            parameters={
                **{key: parameters[key] for key in sorted(_LAMBERT_PARAMETER_KEYS)},
                "axis_unit_m": PROJECTED_AXIS_UNIT_M,
                "source_wind_basis": declaration["wind_basis"],
                **({"same_grid_pairing": declaration["same_grid_pairing"]}
                   if "same_grid_pairing" in declaration else {}),
            },
        )
    else:
        grid_descriptor = GridDescriptor(
            projection="regular_latitude_longitude",
            nx=int(longitude.size), ny=int(latitude.size),
            earth_shape="source_metadata_bound",
            scan_order=(
                ("+x" if longitude[-1] > longitude[0] else "-x") + ","
                + ("+y" if latitude[-1] > latitude[0] else "-y")
            ),
            wind_basis="earth_relative",
            parameters={
                "latitude_first": float(latitude[0]),
                "latitude_last": float(latitude[-1]),
                "longitude_first": float(longitude[0]),
                "longitude_last": float(longitude[-1]),
            },
        )
    return SourceFrameHeader(
        source_id=source_id,
        source_cycle=source_cycle.replace(tzinfo=timezone.utc).isoformat(),
        grid=grid_descriptor,
        vertical_coordinates=descriptors,
        fields=tuple(field_descriptors),
        initialization_policies=dict(
            mapping["target"].get("initialization_policies", {})
        ),
    )


def _published_fields(mapping: Mapping[str, object]) -> tuple[str, ...]:
    """The fields a frame publishes, in the mapping's declared order.

    Every declared field but those marked ``dependency_only``, which are
    decoded for a derivation to read and then left off the frame; the
    Rust engine's ``frames::plan_frames`` reads the same key.  Named
    breakage: ICON-D2's six raw mass fractions were written beside the
    mixing ratios rebased from them, 390 of 1,217 layers per valid time
    and about 16 GB of the 50 GB a 48 h 1 km run stages, read by nothing.
    """

    return tuple(
        name for name, field in mapping["fields"].items()
        if field.get("dependency_only") is not True
    )


def _materialize_frames(
    mapping: Mapping[str, object],
    collection: _DecodedCollection,
    *,
    mapping_sha256: str,
    input_sha256: Mapping[str, str],
    lead_batch: bool = False,
) -> tuple[MappedSourceFrame, ...]:
    unbound = sorted(
        name for name, field in mapping["fields"].items()
        if field.get("provider") == "composition_bound"
    )
    if unbound:
        raise ValueError(
            f"fields {unbound} are composition_bound: this mapping cannot "
            "materialize alone, because those values live in another packaged "
            "source's decode; a cross-source composition must bind each of "
            "them to a contributing source"
        )
    keys = sorted(collection.source_cycles, key=lambda item: (item[0], str(item[1])))
    if not keys:
        raise ValueError("mapped source has no valid times")
    members = {member for _time, member in keys}
    if len(members) != 1:
        raise ValueError("mapped WRF initialization requires exactly one member")
    derivations = _derivation_table(mapping)
    published = _published_fields(mapping)
    frames = []
    required_names = {
        str(item["name"]) for item in mapping["target"]["required_fields"]
    }
    completed_names = _completed_fields(mapping)
    # Required presence and a source mask are compatible: fields with a
    # declared downstream mask repair must retain holes until that repair.
    finite_required = {
        name for name in required_names
        if name not in {"soil_temperature", "volumetric_soil_moisture"}
        and mapping["fields"][name]["missing"]["kind"] != "preserve_mask"
    }
    # Fields declared ``time_binding: cycle_invariant`` were already
    # proven byte-invariant and bound to every dependent valid time by
    # ``_broadcast_invariant_fields`` at GRIB assembly, so every frame
    # below sees them as ordinary per-time fields.
    for valid_time, member in keys:
        available: dict[str, CanonicalField] = {}
        for (time_value, member_value, field_name), direct in collection.direct.items():
            if time_value != valid_time or member_value != member:
                continue
            field = mapping["fields"][field_name]
            available[field_name] = CanonicalField(
                name=field_name,
                units=str(field["units"]["target"]),
                axes=direct.axes,
                location=str(field["location"]),
                staggering=str(field.get("staggering", "none")),
                values=direct.values,
                missing_count=direct.missing_count,
                source_references=direct.references,
            )
        pending = {
            name for name, field in mapping["fields"].items()
            if field.get("derivation") is not None
        }
        # A field the frame completes, which the source published at only
        # some of its values or at none, is held back from every
        # derivation until it is whole; ``seeds`` keeps the values the
        # source did publish.
        seeds: dict[str, CanonicalField | None] = {}
        for name in completed_names:
            present = available.get(name)
            if name in pending or (present is not None and present.missing_count == 0):
                continue
            seeds[name] = available.pop(name, None)
        while pending or seeds:
            progress = False
            for name in sorted(tuple(pending)):
                field = mapping["fields"][name]
                derivation_name = str(field["derivation"])
                if derivation_name not in derivations:
                    raise ValueError(f"field {name} names unknown derivation {derivation_name!r}")
                operation = derivations[derivation_name]
                try:
                    values, axes, references = _evaluate_derivation(
                        operation, available, collection, field, name,
                        vertical=mapping["coordinates"]["vertical"],
                    )
                except KeyError:
                    continue
                available[name] = CanonicalField(
                    name=name, units=str(field["units"]["target"]), axes=axes,
                    location=str(field["location"]),
                    staggering=str(field.get("staggering", "none")),
                    values=values, missing_count=int(np.isnan(values).sum()),
                    source_references=references,
                )
                pending.remove(name)
                progress = True
            # A completion reads the final value of each operand, so it
            # waits while a derivation still has one to produce.
            for name in completed_names:
                if name not in seeds or any(
                        operand in pending for operand, _unit in HYPSOMETRIC_OPERANDS):
                    continue
                completed = _complete_field(
                    name, mapping["fields"][name], seeds[name], available)
                if completed is not None:
                    available[name] = completed
                    del seeds[name]
                    progress = True
            if not progress:
                # A completed field whose own derivation cannot run is
                # derived hydrostatically instead, and the loop goes on for
                # anything that reads it.
                for name in completed_names:
                    if name not in pending or any(
                            operand in pending
                            for operand, _unit in HYPSOMETRIC_OPERANDS):
                        continue
                    completed = _complete_field(
                        name, mapping["fields"][name], None, available)
                    if completed is not None:
                        available[name] = completed
                        pending.remove(name)
                        progress = True
            if not progress:
                if pending:
                    raise ValueError(
                        "derived fields have missing dependencies or a cycle: "
                        + ", ".join(sorted(pending))
                    )
                # What is left is a completion whose operands this source
                # does not carry: the frame lacks that field, and says so
                # by name below.
                break
        # State the frame's fields in the mapping's own declared order.
        # Assembly order above is decode order, which is a property of the
        # PRODUCER's record layout -- and a broadcast invariant lands after
        # the per-time records, so two frames with identical field SETS
        # could otherwise disagree about sequence, which both the
        # inventory-drift check below and the frame header would read as a
        # difference that is not one.  A dependency-only input has been
        # read by its derivations above and is not published.
        available = {
            name: available[name]
            for name in published if name in available
        }
        missing = sorted(required_names - set(available))
        if missing:
            raise ValueError(f"mapped frame at {valid_time} lacks required fields {missing}")
        for name in finite_required:
            if not np.isfinite(available[name].values).all():
                raise ValueError(f"required mapped field {name} is not finite at {valid_time}")
        soil_count = mapping["target"].get("soil_layer_count")
        # A declared count of 0 is an atmosphere-only source whose soil
        # column is pending an external composition (load_mapping enforces
        # that pairing); there are no soil fields to hold to a ladder.
        if soil_count is not None and int(soil_count) > 0:
            for name in ("soil_temperature", "volumetric_soil_moisture"):
                field = available[name]
                observed = field.values.shape[field.axes.index("soil")]
                if observed != int(soil_count):
                    raise ValueError(
                        f"{name} has {observed} layers, target declares {soil_count}"
                    )
        source_cycle = collection.source_cycles[(valid_time, member)]
        header = _frame_header(
            mapping, valid_time=valid_time, source_cycle=source_cycle,
            latitude=collection.latitude, longitude=collection.longitude,
            vertical_values=collection.vertical_values, fields=available,
            source_id=str(mapping["name"]),
            hybrid_a=collection.hybrid_a, hybrid_b=collection.hybrid_b,
        )
        frames.append(MappedSourceFrame(
            valid_time=valid_time, member=member, source_cycle=source_cycle,
            latitude=collection.latitude, longitude=collection.longitude,
            vertical_kind=str(mapping["coordinates"]["vertical"]["kind"]),
            vertical_units=str(mapping["coordinates"]["vertical"]["units"]),
            vertical_values=collection.vertical_values,
            fields=available, mapping_sha256=mapping_sha256,
            input_sha256=input_sha256,
            grid_fingerprint=collection.grid_fingerprint,
            header=header,
        ))

    times = tuple(frame.valid_time for frame in frames)
    if tuple(sorted(times)) != times or len(set(times)) != len(times):
        raise ValueError("mapped forcing times are not unique and increasing")
    target = mapping["target"]
    # One lead batch of a window (an as-posted preparation) may hold one
    # time; the window's series rules are its caller's, over every batch.
    if target.get("require_lateral_boundaries") and not lead_batch:
        if len(times) < 2:
            # The owned class, not a bare ValueError: this is a complete
            # statement about the bytes a user staged, and the door they
            # typed prints it as sentences instead of relaying it under
            # six frames of this module's internals.
            raise ForcingSeriesRefusal(
                "mapped lateral-boundary forcing requires at least two times")
        deltas = {
            int((later - earlier).total_seconds())
            for earlier, later in zip(times, times[1:])
        }
        if len(deltas) != 1 or next(iter(deltas)) <= 0:
            raise ValueError("mapped forcing cadence must be positive and uniform")
        refusal = boundary_interval_refusal(target, next(iter(deltas)))
        if refusal is not None:
            raise ValueError(refusal)
    # Tuple comparison is deliberate: every frame's fields were re-stated
    # in mapping-declared order above, so both SET and ORDER drift between
    # valid times refuse here.
    inventories = {tuple(frame.fields) for frame in frames}
    if len(inventories) != 1:
        raise ValueError("mapped field inventory changes between valid times")
    return tuple(frames)


def _verify_input_manifest(
    path: Path,
    expected_sha256: str,
    mapping_path: Path,
    files: Sequence[Path],
    *,
    _snapshots: Mapping[Path, _AuthoritySnapshot] | None = None,
    _recheck_snapshots: bool = True,
) -> dict[str, object]:
    path = path.resolve()
    supplied = {
        key.resolve(): value for key, value in (_snapshots or {}).items()
    }
    used: dict[Path, _AuthoritySnapshot] = {}

    def snapshot(actual: Path, *, retain_bytes: bool = False) -> _AuthoritySnapshot:
        actual = actual.resolve()
        value = supplied.get(actual)
        if value is None:
            value = _snapshot_authority(actual, retain_bytes=retain_bytes)
        if value.path != actual or (retain_bytes and value.data is None):
            raise ValueError(f"authority snapshot is incomplete for {actual}")
        used[actual] = value
        return value

    manifest_snapshot = snapshot(path, retain_bytes=True)
    observed = manifest_snapshot.sha256
    if observed != expected_sha256.lower():
        raise ValueError(
            f"mapped input-manifest SHA mismatch: expected {expected_sha256}, got {observed}"
        )
    raw = _load_json_bytes(
        manifest_snapshot.data,
        "mapped input manifest",
        path,
    )
    manifest = _object(
        raw, "input manifest",
        allowed={"schema", "mapping_sha256", "files"},
        required={"schema", "mapping_sha256", "files"},
    )
    if manifest["schema"] != INPUT_MANIFEST_SCHEMA:
        raise ValueError(f"unsupported mapped input manifest schema {manifest['schema']!r}")
    if manifest["mapping_sha256"] != snapshot(mapping_path).sha256:
        raise ValueError("input manifest mapping SHA does not match mapping bytes")
    rows = manifest["files"]
    if not isinstance(rows, list) or len(rows) != len(files):
        raise ValueError("input manifest file inventory differs from request")
    for index, (row, source) in enumerate(zip(rows, files)):
        spec = _object(
            row, f"input manifest files[{index}]",
            allowed={"path", "bytes", "sha256"},
            required={"path", "bytes", "sha256"},
        )
        declared = Path(str(spec["path"]))
        if not declared.is_absolute():
            declared = path.parent / declared
        if declared.resolve() != source.resolve():
            raise ValueError(f"input manifest path differs for file {index}")
        source_snapshot = snapshot(source)
        if (
            isinstance(spec["bytes"], bool)
            or not isinstance(spec["bytes"], int)
            or spec["bytes"] != source_snapshot.size
            or spec["sha256"] != source_snapshot.sha256
        ):
            raise ValueError(f"input manifest identity differs for {source}")
    if _recheck_snapshots:
        for value in used.values():
            _require_authority_snapshot(value)
    return manifest


#: Which engine a mapped call runs on, given the caller's decoder
#: arguments.  Named once so decode and inspect cannot disagree.
def _mapped_engine_choice(
    *,
    grib1_bridge: object,
    grib2_inventory: object,
    grib2_dump: object,
    subcommand: str = "decode",
    source_format: str | None = None,
    explicit: str | None = None,
) -> str:
    """``rust`` or ``python`` for this call.

    An EXPLICIT decoder tool argument selects the Python engine no matter
    what the default is.  Naming a subprocess tool is a statement about
    which binary must decode these bytes -- pinned-tool fixtures, an
    audit against a specific build, the 20CRv3 route's sealed decoder
    binding -- and the Rust engine decodes in-process, so honouring the
    default there would silently ignore the pin.  The workaround
    spelling (``GPUWM_MAPPED_ENGINE=python`` / ``--mapped-engine
    python``) is the way to ask for the Python engine WITHOUT pinning a
    tool.

    A path the Rust engine has not been taught yet (``compose``; NetCDF
    records) runs on the Python engine even though the default is Rust.
    That is a ROUTE, not a fallback: the capability table in
    :mod:`woof.mapped_engine_bridge` is checked against the built
    binary by the parity battery, the choice is recorded, and an
    EXPLICIT ``--mapped-engine rust`` still goes to the engine and gets
    its own ``not_implemented`` refusal rather than being quietly
    re-routed.  Asking for Rust and silently getting Python is the thing
    this must never do; refusing a composed source that decodes today,
    to advertise a finished migration, is the other.

    ``explicit`` is the ``--mapped-engine`` argument when the caller
    holds it on an argument namespace instead of in the environment --
    :mod:`woof.source_cli` does, because it decides what to FORWARD
    before the child publishes the variable.  Omitted, the environment
    is the only request, which is what every in-route caller wants:
    ``woof.mapped_direct`` publishes the flag to :data:`ENGINE_ENV`
    before it asks anything, so the flag and the documented workaround
    cannot mean different things.
    """

    from woof.mapped_engine_bridge import (ENGINE_PYTHON, ENGINE_RUST,
                                            engine_supports, resolve_engine)

    if any(value is not None
           for value in (grib1_bridge, grib2_inventory, grib2_dump)):
        return ENGINE_PYTHON
    asked = explicit if explicit is not None else _explicit_engine_request()
    chosen = resolve_engine(explicit)
    if chosen == ENGINE_RUST and asked is None \
            and not engine_supports(subcommand, source_format):
        return ENGINE_PYTHON
    return chosen


def _explicit_engine_request() -> str | None:
    """The engine the caller ASKED for, or ``None`` for a bare run."""

    import os

    from woof.mapped_engine_bridge import ENGINE_ENV

    request = os.environ.get(ENGINE_ENV)
    return request.strip().lower() or None if request else None


def _mapping_format(mapping_path: str | Path) -> str | None:
    """The mapping's declared ``format``, for capability routing only.

    Read with a plain JSON parse rather than :func:`load_mapping`: this
    runs BEFORE the route is chosen, and the validator of record must
    run inside the route that was chosen, not twice with different
    verdicts.  An unreadable document answers ``None`` and the call
    proceeds to whichever engine then refuses it properly.
    """

    try:
        document = json.loads(Path(mapping_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    value = document.get("format") if isinstance(document, dict) else None
    return str(value) if isinstance(value, str) else None


def _engine_scratch_directory() -> str | None:
    """Where an engine call with no preparation output stages its files.

    ``decode`` and ``inspect`` hand back in-memory products, so they
    have no output directory to stage beside; their scratch is placed
    by the compose scratch's own resolution with no destination
    (:func:`woof.mapped_composition._compose_scratch_base`):
    ``WOOF_COMPOSE_SCRATCH`` when it is set -- refused by name when it
    does not exist -- and the system temp otherwise.

    Named breakage: a 0.25-degree analysis decode on a box whose
    ``/tmp`` is a quota-limited tmpfs died with "cannot write the frame
    stream: Disk quota exceeded (os error 122)" while
    ``WOOF_COMPOSE_SCRATCH`` named a disk-backed directory, because the
    variable placed the compose scratch only and this route staged its
    frameset in the system temp; only ``TMPDIR`` moved it.  One
    documented variable now covers every temporary the engine route
    writes -- the input list and the frame stream both land under the
    directory handed to the engine, which reads no ``TMPDIR`` of its own.
    """

    import os

    from woof.mapped_composition import _compose_scratch_base

    base = _compose_scratch_base(None)
    return None if base is None else os.fspath(base)


def _decode_through_engine(
    mapping_path: Path,
    sources: tuple[Path, ...],
    *,
    input_manifest: Path | None,
    input_manifest_sha256: str | None,
) -> tuple[MappedSourceFrame, ...]:
    """Decode on the Rust engine, keeping the Python authority window.

    The hash-stability window, the input-manifest verification and the
    refusal grammar stay on this side of the seam (the design's §1.1
    split); the engine decodes bytes and writes a frameset.  The frames
    it wrote are then re-validated here by the same dataclasses the
    Python engine constructs, and its recorded authority hashes are
    checked against the ones Python snapshotted -- an engine that
    decoded a different mapping than the one this call verified is a
    silent-wrong-answer, and this is where it stops.
    """

    import tempfile

    from woof import mapped_engine_bridge

    mapping_snapshot = _snapshot_authority(mapping_path, retain_bytes=True)
    # The ONE grammar validator runs on this route too, before the
    # engine is launched: a broken document must earn the same sentence
    # whichever engine would have decoded it, and the Python validator
    # is the wording of record.  The engine's own validator stays behind
    # this as defense for a hand-run exe.
    load_mapping(mapping_path)
    snapshots = {mapping_path: mapping_snapshot}
    for source in sources:
        snapshots[source] = _snapshot_authority(source)
    before = {str(source): snapshots[source].sha256 for source in sources}
    if input_manifest is not None:
        snapshots[input_manifest] = _snapshot_authority(
            input_manifest, retain_bytes=True)
        _verify_input_manifest(
            input_manifest, str(input_manifest_sha256),
            mapping_path, sources,
            _snapshots=snapshots,
            _recheck_snapshots=False,
        )
    _require_authority_snapshot(mapping_snapshot)

    # The engine writes the WHOLE decoded frame stream under this
    # directory -- f64 frames, a few GB for a 0.25-degree analysis --
    # before it is read back, so its placement is the compose scratch's
    # (see :func:`_engine_scratch_directory`), not the bare system temp.
    scratch = _engine_scratch_directory()
    with tempfile.TemporaryDirectory(
            prefix="gpuwm-mapped-engine-", dir=scratch) as work:
        try:
            mapped_engine_bridge.run_engine(
                "decode",
                mapping=mapping_path,
                files=sources,
                output=Path(work) / "frameset",
                input_manifest=input_manifest,
                input_manifest_sha256=input_manifest_sha256,
            )
        except ScratchDiskRefusal as refusal:
            raise scratch_disk_refusal(refusal, scratch) from refusal
        frames = mapped_engine_bridge.read_frameset(
            Path(work) / "frameset")

    for frame in frames:
        if frame.mapping_sha256 != mapping_snapshot.sha256:
            raise RuntimeError(
                f"{mapped_engine_bridge.ENGINE_NAME} decoded mapping bytes "
                f"hashing to {frame.mapping_sha256}; this call verified "
                f"{mapping_path} at {mapping_snapshot.sha256}.  The engine "
                "and woof read different authority bytes, so nothing "
                "downstream is bound to the document that was checked")
        if dict(frame.input_sha256) != before:
            raise RuntimeError(
                f"{mapped_engine_bridge.ENGINE_NAME} recorded input hashes "
                "that differ from the ones this call verified, so the "
                "frames are not bound to the inputs the manifest covers")
    for snapshot in snapshots.values():
        _require_authority_snapshot(snapshot)
    return frames


def decode_mapped_source(
    mapping_path: str | Path,
    files: Sequence[str | Path],
    *,
    input_manifest: str | Path | None = None,
    input_manifest_sha256: str | None = None,
    grib1_bridge: str | Path | None = None,
    grib2_inventory: str | Path | None = None,
    grib2_dump: str | Path | None = None,
) -> tuple[MappedSourceFrame, ...]:
    """Decode arbitrary declared inputs into canonical source frames.

    All persistent authorities are hashed before decode and again afterward.
    The optional manifest is mandatory for the eventual public runner, but is
    optional here so small unit fixtures can exercise the materializer.

    The byte work runs on the Rust mapped engine
    (:mod:`woof.mapped_engine_bridge`); this signature is frozen and the
    return type is unchanged either way.  The Python engine below stays
    reachable as a documented WORKAROUND and for an explicitly pinned
    decoder tool -- see :func:`_mapped_engine_choice`.
    """

    from woof.mapped_engine_bridge import ENGINE_RUST

    if _mapped_engine_choice(
        grib1_bridge=grib1_bridge,
        grib2_inventory=grib2_inventory,
        grib2_dump=grib2_dump,
        subcommand="decode",
        source_format=_mapping_format(mapping_path),
    ) == ENGINE_RUST:
        sources = tuple(Path(path).resolve() for path in files)
        if not sources:
            raise ValueError("mapped source requires at least one input file")
        if len(set(sources)) != len(sources):
            raise ValueError("mapped source input list contains duplicates")
        for source in sources:
            if not source.is_file():
                raise FileNotFoundError(source)
        if (input_manifest is None) != (input_manifest_sha256 is None):
            raise ValueError(
                "input_manifest and input_manifest_sha256 are an atomic pair")
        return _decode_through_engine(
            Path(mapping_path).resolve(), sources,
            input_manifest=(
                None if input_manifest is None
                else Path(input_manifest).resolve()),
            input_manifest_sha256=input_manifest_sha256,
        )
    return _decode_mapped_source_python(
        mapping_path, files,
        input_manifest=input_manifest,
        input_manifest_sha256=input_manifest_sha256,
        grib1_bridge=grib1_bridge,
        grib2_inventory=grib2_inventory,
        grib2_dump=grib2_dump,
    )


def _decode_mapped_source_python(
    mapping_path: str | Path,
    files: Sequence[str | Path],
    *,
    input_manifest: str | Path | None = None,
    input_manifest_sha256: str | None = None,
    grib1_bridge: str | Path | None = None,
    grib2_inventory: str | Path | None = None,
    grib2_dump: str | Path | None = None,
) -> tuple[MappedSourceFrame, ...]:
    """The Python engine: the behaviour of record, and the parity referee.

    Reachable through ``GPUWM_MAPPED_ENGINE=python`` / ``--mapped-engine
    python`` -- a WORKAROUND, not the fix -- and through an explicit
    decoder-tool argument.  The parity battery decodes every staged
    source both ways and demands byte-identical field arrays, so this
    body is also the reference the Rust engine is measured against.
    """

    mapping_path = Path(mapping_path).resolve()
    sources = tuple(Path(path).resolve() for path in files)
    if not sources:
        raise ValueError("mapped source requires at least one input file")
    if len(set(sources)) != len(sources):
        raise ValueError("mapped source input list contains duplicates")
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(source)
    if (input_manifest is None) != (input_manifest_sha256 is None):
        raise ValueError("input_manifest and input_manifest_sha256 are an atomic pair")
    mapping_snapshot = _snapshot_authority(mapping_path, retain_bytes=True)
    mapping_raw = _load_json_bytes(
        mapping_snapshot.data,
        "mapping",
        mapping_path,
    )
    mapping = load_mapping(mapping_path, _raw=mapping_raw)
    mapping_digest = mapping_snapshot.sha256
    snapshots = {mapping_path: mapping_snapshot}
    for source in sources:
        snapshots[source] = _snapshot_authority(source)
    decoder_paths = _resolve_direct_decoder_paths(
        str(mapping["format"]),
        grib1_bridge=grib1_bridge,
        grib2_inventory=grib2_inventory,
        grib2_dump=grib2_dump,
    )
    for path in decoder_paths.values():
        snapshots[path] = _snapshot_authority(path)
    before = {
        str(source): snapshots[source].sha256 for source in sources
    }
    if input_manifest is not None:
        manifest_path = Path(input_manifest).resolve()
        snapshots[manifest_path] = _snapshot_authority(
            manifest_path,
            retain_bytes=True,
        )
        _verify_input_manifest(
            manifest_path, str(input_manifest_sha256),
            mapping_path, sources,
            _snapshots=snapshots,
            _recheck_snapshots=False,
        )
    _require_authority_snapshot(mapping_snapshot)
    if input_manifest is not None:
        _require_authority_snapshot(snapshots[manifest_path])
    if mapping["format"] == "netcdf":
        collection = _decode_netcdf(mapping, sources)
    else:
        collection = _decode_grib(
            mapping, sources,
            grib1_bridge=decoder_paths.get("grib1_bridge"),
            grib2_inventory=decoder_paths.get("grib2_inventory"),
            grib2_dump=decoder_paths.get("grib2_dump"),
        )
    frames = _materialize_frames(
        mapping, collection, mapping_sha256=mapping_digest,
        input_sha256=before,
    )
    for snapshot in snapshots.values():
        _require_authority_snapshot(snapshot)
    return frames


def inspect_mapped_source(
    mapping_path: str | Path,
    files: Sequence[str | Path],
    *,
    input_manifest: str | Path | None = None,
    input_manifest_sha256: str | None = None,
    grib1_bridge: str | Path | None = None,
    grib2_inventory: str | Path | None = None,
    grib2_dump: str | Path | None = None,
) -> dict[str, object]:
    """Decode real source bytes and report canonical materialization readiness.

    Inspection never upgrades an adapter's production status.  It is useful
    when a real file proves most selectors but lacks a required invariant or
    complementary product: direct arrays and their hashes are still recorded,
    while the exact materialization failure remains explicit.

    Signature frozen; routed like :func:`decode_mapped_source`.  The
    engine's ``inspect`` subcommand prints the same
    ``gpuwm-mapped-source-inspection-v1`` document this function returns,
    and the parity battery compares the two as canonical JSON under the
    enumerated engine-identity mask.
    """

    from woof.mapped_engine_bridge import ENGINE_RUST

    if _mapped_engine_choice(
        grib1_bridge=grib1_bridge,
        grib2_inventory=grib2_inventory,
        grib2_dump=grib2_dump,
        subcommand="inspect",
        source_format=_mapping_format(mapping_path),
    ) == ENGINE_RUST:
        return _inspect_through_engine(
            Path(mapping_path).resolve(),
            tuple(Path(path).resolve() for path in files),
            input_manifest=(
                None if input_manifest is None
                else Path(input_manifest).resolve()),
            input_manifest_sha256=input_manifest_sha256,
        )
    return _inspect_mapped_source_python(
        mapping_path, files,
        input_manifest=input_manifest,
        input_manifest_sha256=input_manifest_sha256,
        grib1_bridge=grib1_bridge,
        grib2_inventory=grib2_inventory,
        grib2_dump=grib2_dump,
    )


def _inspect_through_engine(
    mapping_path: Path,
    sources: tuple[Path, ...],
    *,
    input_manifest: Path | None,
    input_manifest_sha256: str | None,
) -> dict[str, object]:
    """Inspect on the Rust engine; the document is read off its stdout."""

    import tempfile

    from woof import mapped_engine_bridge

    if not sources:
        raise ValueError(
            "mapped source inspection requires at least one input file")
    if len(set(sources)) != len(sources):
        raise ValueError(
            "mapped source inspection input list contains duplicates")
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(source)
    if (input_manifest is None) != (input_manifest_sha256 is None):
        raise ValueError(
            "input_manifest and input_manifest_sha256 are an atomic pair")
    mapping_snapshot = _snapshot_authority(mapping_path, retain_bytes=True)
    # Same reasoning as :func:`_decode_through_engine`: the grammar
    # refusal must be the Python validator's sentence on both routes.
    load_mapping(mapping_path)
    snapshots = {mapping_path: mapping_snapshot}
    for source in sources:
        snapshots[source] = _snapshot_authority(source)
    if input_manifest is not None:
        snapshots[input_manifest] = _snapshot_authority(
            input_manifest, retain_bytes=True)
        _verify_input_manifest(
            input_manifest, str(input_manifest_sha256),
            mapping_path, sources,
            _snapshots=snapshots,
            _recheck_snapshots=False,
        )
    # Only the input list lands here (the document rides stdout), but
    # it is the same knob as every other engine temporary, so a scratch
    # override that steers the decode route steers this one too.
    with tempfile.TemporaryDirectory(
            prefix="gpuwm-mapped-inspect-",
            dir=_engine_scratch_directory()) as work:
        result = mapped_engine_bridge.run_engine(
            "inspect",
            mapping=mapping_path,
            files=sources,
            output=Path(work) / "inspection",
            input_manifest=input_manifest,
            input_manifest_sha256=input_manifest_sha256,
        )
        document = _inspection_from_stdout(str(result["stdout"]))
    for snapshot in snapshots.values():
        _require_authority_snapshot(snapshot)
    return document


def _inspection_from_stdout(stdout: str) -> dict[str, object]:
    """The inspection document among the engine's stdout lines.

    ``inspect`` shares its stdout with the progress stream, so the
    document is the one object carrying the INSPECTION schema rather
    than "the last line" -- a progress receipt printed after it would
    otherwise be returned as the report.
    """

    from woof.mapped_engine_bridge import ENGINE_NAME

    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            document = json.loads(line)
        except ValueError:
            continue
        if isinstance(document, dict) \
                and document.get("schema") == INSPECTION_SCHEMA:
            return _inspection_error_type(document)
    raise RuntimeError(
        f"{ENGINE_NAME} inspect printed no {INSPECTION_SCHEMA} object on "
        "stdout, so there is no report to return; rebuild the engine from "
        "a matching checkout")


def _inspection_error_type(document: dict[str, object]) -> dict[str, object]:
    """Name the materialization failure the way the Python engine does.

    A materialization that cannot complete is reported in the inspection
    document, and the Python engine names it by the EXCEPTION TYPE it
    would have raised.  The engine has no Python exceptions; it names a
    refusal class.  Translating here rather than teaching the engine
    Python's type names keeps exception-type selection on this side of
    the seam, which is where the design puts it -- the engine states
    what went wrong, woof decides what that is called and what a caller
    would catch.

    An unknown class is left exactly as the engine wrote it.  Rewriting
    it to something familiar would be a guess presented as a fact, and
    :func:`woof.mapped_engine_bridge.refusal_error` already refuses an
    unknown class by name on the failure path.
    """

    from woof.mapped_engine_bridge import REFUSAL_CLASSES

    materialization = document.get("materialization")
    if not isinstance(materialization, dict):
        return document
    name = materialization.get("error_class")
    if name is None:
        return document
    exception = REFUSAL_CLASSES.get(str(name))
    if exception is None:
        return document
    materialization = dict(materialization)
    materialization.pop("error_class")
    materialization["error_type"] = exception.__name__
    return {**document, "materialization": materialization}


def _inspect_mapped_source_python(
    mapping_path: str | Path,
    files: Sequence[str | Path],
    *,
    input_manifest: str | Path | None = None,
    input_manifest_sha256: str | None = None,
    grib1_bridge: str | Path | None = None,
    grib2_inventory: str | Path | None = None,
    grib2_dump: str | Path | None = None,
) -> dict[str, object]:
    """The Python engine's inspection: the behaviour of record."""

    mapping_path = Path(mapping_path).resolve()
    sources = tuple(Path(path).resolve() for path in files)
    if not sources:
        raise ValueError("mapped source inspection requires at least one input file")
    if len(set(sources)) != len(sources):
        raise ValueError("mapped source inspection input list contains duplicates")
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(source)
    if (input_manifest is None) != (input_manifest_sha256 is None):
        raise ValueError("input_manifest and input_manifest_sha256 are an atomic pair")

    mapping_snapshot = _snapshot_authority(mapping_path, retain_bytes=True)
    mapping_raw = _load_json_bytes(
        mapping_snapshot.data,
        "mapping",
        mapping_path,
    )
    mapping = load_mapping(mapping_path, _raw=mapping_raw)
    mapping_digest = mapping_snapshot.sha256
    snapshots = {mapping_path: mapping_snapshot}
    for source in sources:
        snapshots[source] = _snapshot_authority(source)
    decoder_paths = _resolve_direct_decoder_paths(
        str(mapping["format"]),
        grib1_bridge=grib1_bridge,
        grib2_inventory=grib2_inventory,
        grib2_dump=grib2_dump,
    )
    for path in decoder_paths.values():
        snapshots[path] = _snapshot_authority(path)
    before = {
        str(source): snapshots[source].sha256 for source in sources
    }
    if input_manifest is not None:
        manifest_path = Path(input_manifest).resolve()
        snapshots[manifest_path] = _snapshot_authority(
            manifest_path,
            retain_bytes=True,
        )
        _verify_input_manifest(
            manifest_path, str(input_manifest_sha256),
            mapping_path, sources,
            _snapshots=snapshots,
            _recheck_snapshots=False,
        )
    _require_authority_snapshot(mapping_snapshot)
    if input_manifest is not None:
        _require_authority_snapshot(snapshots[manifest_path])
    if mapping["format"] == "netcdf":
        collection = _decode_netcdf(mapping, sources)
    else:
        collection = _decode_grib(
            mapping, sources,
            grib1_bridge=decoder_paths.get("grib1_bridge"),
            grib2_inventory=decoder_paths.get("grib2_inventory"),
            grib2_dump=decoder_paths.get("grib2_dump"),
        )

    direct_field_names = {
        str(name) for name, field in mapping["fields"].items()
        if field.get("derivation") is None
    }
    frame_rows = []
    for valid_time, member in sorted(
        collection.source_cycles, key=lambda item: (item[0], str(item[1]))
    ):
        decoded = {
            field_name: value
            for (time_value, member_value, field_name), value
            in collection.direct.items()
            if time_value == valid_time and member_value == member
        }
        frame_rows.append({
            "valid_time": valid_time.isoformat(),
            "source_cycle": collection.source_cycles[(valid_time, member)].isoformat(),
            "member": member,
            "decoded_direct_fields": sorted(decoded),
            "unresolved_direct_fields": sorted(direct_field_names - set(decoded)),
            "fields": {
                name: {
                    "axes": list(value.axes),
                    "shape": list(value.values.shape),
                    "minimum": (
                        None if not np.isfinite(value.values).any()
                        else float(np.nanmin(value.values))
                    ),
                    "maximum": (
                        None if not np.isfinite(value.values).any()
                        else float(np.nanmax(value.values))
                    ),
                    "missing": value.missing_count,
                    "sha256": _array_sha256(value.values),
                    "source_references": list(value.references),
                }
                for name, value in sorted(decoded.items())
            },
        })

    materialization: dict[str, object]
    try:
        frames = _materialize_frames(
            mapping, collection, mapping_sha256=mapping_digest,
            input_sha256=before,
        )
    except (KeyError, TypeError, ValueError) as error:
        materialization = {
            "verdict": "INCOMPLETE",
            "error_type": type(error).__name__,
            "error": str(error),
        }
        status = "DECODED_INCOMPLETE_NOT_STOCK_WRF_CERTIFIED"
    else:
        materialization = {
            "verdict": "PASS",
            "frame_count": len(frames),
            "frame_header_sha256": [
                hashlib.sha256(
                    json.dumps(
                        frame.header.to_dict(), sort_keys=True,
                        separators=(",", ":"), allow_nan=False,
                    ).encode("utf-8")
                ).hexdigest()
                for frame in frames
            ],
            # Beside the raw digest, never instead of it: the raw one is an
            # identity of THIS box (input paths, and a derived field's libm
            # last bits); the portable one is what another box can
            # reproduce, and therefore what a recorded golden may assert.
            # See woof.source_frame.portable_frame_header for the rule.
            "frame_header_sha256_portable": [
                portable_frame_header_sha256(
                    frame.header, inputs=sources,
                    libm_dependent=libm_dependent_fields(mapping),
                )
                for frame in frames
            ],
            "portable_rule": PORTABLE_HEADER_RULE,
        }
        status = "CANONICAL_FRAMES_MATERIALIZED_NOT_STOCK_WRF_CERTIFIED"

    for snapshot in snapshots.values():
        _require_authority_snapshot(snapshot)
    return {
        "schema": INSPECTION_SCHEMA,
        "status": status,
        "stock_wrf_certified": False,
        "mapping": {"path": str(mapping_path), "sha256": mapping_digest},
        "inputs": [
            {
                "path": str(source), "bytes": source.stat().st_size,
                "sha256": before[str(source)],
            }
            for source in sources
        ],
        "decoders": {
            role: {
                "path": str(path),
                "bytes": snapshots[path].size,
                "sha256": snapshots[path].sha256,
            }
            for role, path in decoder_paths.items()
        },
        "source_format": mapping["format"],
        "grid": {
            "ny": int(collection.latitude.size),
            "nx": int(collection.longitude.size),
            "vertical_count": int(collection.vertical_values.size),
            "fingerprint": collection.grid_fingerprint,
        },
        "frames": frame_rows,
        "materialization": materialization,
    }


def _nearest_soil_column_repair(
    soil_t: np.ndarray,
    soil_m: np.ndarray,
    terrestrial: np.ndarray,
    longitude: np.ndarray,
    maximum_cells: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fill land cells whose soil column is missing from the nearest donor.

    Producers compute soil on their own surface tiling, and that tiling can
    disagree with the published land-cover field on a handful of coastal
    cells (ECMWF's fractional land mask against its soil tile mask).  A
    declared, BOUNDED repair takes the complete soil column -- temperature
    and moisture together -- from the nearest cell with a fully defined
    column, searching chebyshev rings out to ``maximum_cells`` and breaking
    ties by squared index distance then (di, dj) order, so the result is
    deterministic.  A land cell with no donor inside the radius stays
    missing, as it would with no declared repair, and
    :func:`_admit_source_land_soil_gaps` counts it.  On a full longitude
    ring the search wraps in x, because a coastal gap beside the seam is no
    farther from its neighbour than any other.
    """

    gap = terrestrial & ~(
        np.isfinite(soil_t).all(axis=0) & np.isfinite(soil_m).all(axis=0)
    )
    if not gap.any():
        return soil_t, soil_m
    donor = np.isfinite(soil_t).all(axis=0) & np.isfinite(soil_m).all(axis=0)
    ny, nx = gap.shape
    step = float(longitude[1] - longitude[0]) if nx > 1 else 0.0
    wrap_x = nx > 1 and abs(
        (float(longitude[-1]) + step) - float(longitude[0]) - 360.0
    ) < 1e-6
    soil_t = soil_t.copy()
    soil_m = soil_m.copy()
    for y, x in zip(*np.nonzero(gap)):
        best = None
        for radius in range(1, int(maximum_cells) + 1):
            for dy in range(-radius, radius + 1):
                yy = y + dy
                if yy < 0 or yy >= ny:
                    continue
                for dx in range(-radius, radius + 1):
                    if max(abs(dy), abs(dx)) != radius:
                        continue
                    xx = x + dx
                    if wrap_x:
                        xx %= nx
                    elif xx < 0 or xx >= nx:
                        continue
                    if not donor[yy, xx]:
                        continue
                    key = (dy * dy + dx * dx, dy, dx)
                    if best is None or key < best[0]:
                        best = (key, yy, xx)
            if best is not None:
                break
        if best is None:
            continue
        _key, yy, xx = best
        soil_t[:, y, x] = soil_t[:, yy, xx]
        soil_m[:, y, x] = soil_m[:, yy, xx]
    return soil_t, soil_m


#: Source soil gaps already announced, keyed on their counts, so a forcing
#: series whose every time carries the same gaps says it once.
_SOURCE_SOIL_GAPS_ANNOUNCED: set = set()


def _admit_source_land_soil_gaps(soil_t, soil_m, terrestrial):
    """Count source land cells with no soil value; refuse only no field.

    A producer computes soil on its own surface tiling, and that tiling can
    disagree with its published land cover: a land cell with no soil value
    beside a reservoir, a river mouth or a coast.  Refusing the whole
    preparation for it refused every domain that source reaches.  The
    masked horizontal mapping (woof/ingest/horiz.py) treats a missing
    value exactly as metgrid does: it is not a donor, and the target land
    near it takes the chain's answer from the source land around it, the
    WPS search's nearest value when none is near.  So a gap is counted and
    said, and left missing for that mapping.

    What no mapping can answer is a source that carries the field on NONE
    of its land cells: it has no soil field to initialize the land surface
    from.  That alone refuses, by name.
    """
    land_cells = int(np.count_nonzero(terrestrial))
    if land_cells == 0:
        return
    gaps = []
    for label, values in (("soil temperature", soil_t),
                          ("soil moisture", soil_m)):
        finite = np.isfinite(values[:, terrestrial])
        if not finite.any():
            raise ValueError(
                f"mapped {label} carries no value on any of the source's "
                f"{land_cells} land cell(s): the source has no {label} "
                "field to initialize the land surface from"
            )
        missing = int(np.count_nonzero(~finite))
        if missing:
            gaps.append(f"{missing} {label}")
    if not gaps:
        return
    signature = (land_cells, tuple(gaps))
    if signature in _SOURCE_SOIL_GAPS_ANNOUNCED:
        return
    _SOURCE_SOIL_GAPS_ANNOUNCED.add(signature)
    print(
        "mapped soil: " + " and ".join(gaps) + " source land value(s) of "
        f"{land_cells} land cell(s) carry no value (the source's soil "
        "tiling and its land cover disagree there); they are not donors, "
        "and target land near them takes the masked mapping's answer from "
        "the source land around them",
        file=sys.stderr)


#: The five canonical hydrometeors and the legacy names the regular-source
#: join packs them under.  One table, read twice: a plane the frame carries
#: is packed, a plane it does not carry is zero-filled under the header's
#: declared policy.
HYDROMETEOR_LEGACY_NAMES = {
    "cloud_water_mixing_ratio": "QC",
    "rain_water_mixing_ratio": "QR",
    "cloud_ice_mixing_ratio": "QI",
    "snow_mixing_ratio": "QS",
    "graupel_or_hail_mixing_ratio": "QG",
}

#: The two vertical-velocity planes the regular-source join packs, with
#: the units each must carry.  ``pressure_vertical_velocity`` is omega
#: (Pa s-1), what NCEP's native products publish (HRRR wrfnat and RAP
#: awp130bgrb: 0/2/8 on every hybrid level); ``vertical_velocity`` is
#: geometric W (m s-1).  The initializer (woof.ingest.real) converts
#: omega with the target column's own density and interpolates either
#: onto the W interfaces, so a start from an analysis keeps the updrafts
#: the analysis holds instead of restarting every storm from W = 0.
VERTICAL_VELOCITY_LEGACY_NAMES = {
    "pressure_vertical_velocity": ("OMEGA", "Pa s-1"),
    "vertical_velocity": ("WW", "m s-1"),
}

#: Canonical fields a mapping may legitimately declare that the
#: regular-source join has no consumer for.  Carrying more than the target
#: consumes is neither impossible nor self-contradictory nor missing, so it
#: is not refused: the plane is left out of the pack and named once.
#: Empty since the join took up vertical velocity (the drop it used to name
#: was the guard that kept every HRRR-analysis start at W = 0, the
#: hour-one spin-up defect); the machinery stays for the next field a
#: mapping carries ahead of its consumer.
REGULAR_JOIN_DROPPED_FIELDS: tuple[str, ...] = ()


def regular_join_dropped_fields(declared) -> tuple[str, ...]:
    """Which declared canonical fields the regular-source join drops."""

    names = set(declared)
    return tuple(name for name in REGULAR_JOIN_DROPPED_FIELDS
                 if name in names)


#: Drop notices already given in this process, keyed on the dropped
#: fields.  One preparation reaches this function from three places (the
#: plan review at both woof.mapped_direct.prepare_mapped_wrf call sites,
#: and the frame join), and ``explain.warn`` has no memory of its own, so
#: "named once" needs this: the first door to notice says it, the rest
#: stay quiet about the same drop.
_JOIN_DROPS_WARNED: set[tuple[str, ...]] = set()


def warn_regular_join_drops(declared, *, subject: str) -> tuple[str, ...]:
    """Name the dropped fields once per run, at whichever door asked first.

    ONE function, both doors: the mapped plan review calls it against the
    mapping document's declared field list, and the frame join calls it
    against the frames it was handed, so the two can never disagree about
    which declared field reaches the forecast.  Both still get the answer
    back for their receipts; only the sentence is said once.
    """

    dropped = regular_join_dropped_fields(declared)
    if not dropped:
        return ()
    if dropped in _JOIN_DROPS_WARNED:
        return dropped
    _JOIN_DROPS_WARNED.add(dropped)
    names = ", ".join(dropped)
    warn(
        f"{subject} declares {names}, which the regular-source join has no "
        f"consumer for: the field is decoded and then dropped, so it does "
        f"not reach the forecast",
        why=("The regular-source join packs the canonical fields the "
             "initialization ABI consumes.  A source that carries more "
             "than the target consumes is not an error; it is a drop, and "
             "this line is where it is stated.  Remove the field from the "
             "mapping to stop paying for its decode."),
    )
    return dropped


#: Canonical mapped field -> the regular-source ABI name it is packed as.
REGULAR_SNAPSHOT_LEGACY_NAMES = {
    "air_temperature": "T",
    "air_pressure": "PRES",
    "specific_humidity": "SPFH",
    "eastward_wind": "U",
    "northward_wind": "V",
    "geopotential_height": "GHT",
    "surface_pressure": "PSFC",
    "terrain_height": "SOURCE_OROGRAPHY",
    "skin_temperature": "SKINTEMP",
    "vegetation_fraction": "VEGFRA",
    "air_temperature_2m": "T2",
    "specific_humidity_2m": "Q2",
    "eastward_wind_10m": "U10",
    "northward_wind_10m": "V10",
    "land_fraction": "LANDSEA",
    # SNOW, not SNOW_EC.  ``SNOW_EC`` is bound in exactly one
    # decode table -- ECMWF GRIB1 (141, 1), metres of water
    # equivalent (``woof/ingest/grib.py``) -- and the soil
    # initializer multiplies that name by 1000 to reach kg m-2
    # (``woof/ingest/soil.py``).  The canonical
    # ``snow_water_equivalent`` is kg m-2 already, which is what
    # every other route carrying it calls ``SNOW`` and what
    # ``preprocess_noah_soil`` consumes unscaled.
    "snow_water_equivalent": "SNOW",
    "snow_depth": "SNOWH",
    "sea_ice_fraction": "SEAICE",
    # The water state the horizontal mapping assembles the water
    # temperature from (woof/ingest/water_temperature.py): the
    # analysis over the sea, and the lake model's own water and ice
    # (woof/ingest/lake_temperature.py) where the analysis has no
    # water of a lake's.  Without them a mapped source gave every
    # lake its coherent skin temperature while a direct route reading
    # the same file gave it the lake water.
    "sea_surface_temperature": "SST",
    "lake_water_temperature": "LAKE_WATER_TEMP",
    "lake_ice_temperature": "LAKE_ICE_TEMP",
    "lake_ice_depth": "LAKE_ICE_DEPTH",
}


def _regular_snapshot_field_items(frame, pressure, *, soil_land_repair,
                                  initialize_absent_hydrometeors):
    """Yield unchanged join arithmetic while retaining only its active inputs."""
    canonical = frame.fields
    pressure_shape = pressure.shape
    legacy_names = REGULAR_SNAPSHOT_LEGACY_NAMES
    for name, legacy in legacy_names.items():
        if name not in canonical:
            continue
        values = (pressure if name == "air_pressure" else
                  np.asarray(canonical[name].values, dtype=np.float64))
        if name == "air_pressure":
            pressure = None
        yield legacy, values
        del values
    from woof.ingest.analyzed_numbers import CANONICAL_NUMBER_FIELDS
    for name, legacy in CANONICAL_NUMBER_FIELDS.items():
        if name in canonical:
            if canonical[name].units != "kg-1":
                raise ValueError(f"{name} must carry number mixing ratio units kg-1")
            yield legacy, np.asarray(canonical[name].values, dtype=np.float64)
    # Packed as the number fields are: the plane as the frame carries it
    # (a windowed frame hands over its own window), with only the units
    # checked here; the initializer holds the shape to the mass grid.
    for name, (legacy, units) in VERTICAL_VELOCITY_LEGACY_NAMES.items():
        if name in canonical:
            if canonical[name].units != units:
                raise ValueError(f"{name} must carry {units}")
            yield legacy, np.asarray(canonical[name].values, dtype=np.float64)
    soil_t = np.asarray(
        canonical["soil_temperature"].values, dtype=np.float64,
    )
    soil_m = np.asarray(
        canonical["volumetric_soil_moisture"].values, dtype=np.float64,
    )
    if soil_t.ndim != 3 or soil_m.ndim != 3 or soil_t.shape != soil_m.shape:
        raise ValueError(
            "mapped soil temperature/moisture must share soil/y/x shape"
        )
    source_land = np.asarray(
        canonical["land_fraction"].values, dtype=np.float64,
    )
    if source_land.shape != soil_t.shape[1:] \
            or not np.isfinite(source_land).all():
        raise ValueError(
            "mapped land fraction must be finite and share the soil grid"
        )
    # A fraction, and now checked as one.  Finiteness alone let a
    # mis-scaled unit transform deliver 2.0 here, which the >= 0.5
    # threshold below reads as land without complaint; a value that
    # merely kisses 0 or 1 is decode rounding and clamps.
    source_land, _ = admit_bounded(
        source_land, name="land fraction", minimum=0.0, maximum=1.0,
        subject="mapped")
    terrestrial = source_land >= 0.5
    if soil_land_repair is not None:
        if str(soil_land_repair.get("kind")) \
                != "nearest_soil_column_within_cells":
            raise ValueError(
                "unsupported declared soil land repair "
                f"{soil_land_repair!r}"
            )
        soil_t, soil_m = _nearest_soil_column_repair(
            soil_t, soil_m, terrestrial, frame.longitude,
            int(soil_land_repair["maximum_cells"]),
        )
    _admit_source_land_soil_gaps(soil_t, soil_m, terrestrial)
    yield MAPPED_SOIL_TEMPERATURE, soil_t
    yield MAPPED_SOIL_MOISTURE, soil_m
    del soil_t, soil_m, source_land, terrestrial
    # Per field, not per block.  A frame that carries a hydrometeor gets
    # the decoded plane packed under its legacy name, exactly as the snow
    # and sea-ice names above are; one that does not gets the declared
    # zero.  A mixed inventory therefore composes (three real planes, two
    # zero-filled) instead of being refused for carrying what it has.
    for canonical_name, output in HYDROMETEOR_LEGACY_NAMES.items():
        if canonical_name in canonical:
            yield output, np.asarray(
                canonical[canonical_name].values, dtype=np.float64)
            continue
        if not initialize_absent_hydrometeors:
            continue
        if frame.header.initialization_policies.get(canonical_name) \
                != "explicit_zero_with_adapter_validation":
            raise ValueError(
                f"absent {canonical_name} lacks the supported "
                "explicit-zero policy"
            )
        # Era5Snapshot takes its own contiguous, immutable copy.
        yield output, np.broadcast_to(np.float64(0.0), pressure_shape)


def mapped_frames_to_regular_snapshots(
    frames: Sequence[MappedSourceFrame],
    *,
    soil_land_repair: Mapping[str, object] | None = None,
    initialize_absent_hydrometeors: bool = False,
    atmospheric_window=None,
    full_snapshot_factory=None,
) -> tuple[Era5Snapshot, ...]:
    """Translate canonical frames to the existing regular-source join ABI.

    This is naming/packing only.  It does not reimplement interpolation or
    initialization. Soil arrays retain canonical names; the independently
    validated composition contract supplies their depth/remapping semantics.
    ``soil_land_repair`` is the composition's declared missing.land policy
    when it is a bounded repair object rather than ``"reject"``.  Either
    way a source land cell still missing its soil value is not refused: it
    is counted and left missing, so the masked horizontal mapping does not
    take it as a donor (see :func:`_admit_source_land_soil_gaps`).

    Hydrometeors are packed per field.  A frame that CARRIES one of the five
    canonical hydrometeors has that decoded plane packed under its legacy
    name (QC/QR/QI/QS/QG); one it does not carry is zero-filled only when
    ``initialize_absent_hydrometeors`` is requested AND the header
    explicitly authorizes zero initialization for that field, so the flag's
    meaning is "zero whichever of the five this header declares absent"
    rather than "zero all five".  A mixed inventory composes.  The zero
    views and the decoded fields are copied once by the same immutable
    snapshot constructor; no temporary full-grid zero bank is allocated.

    Vertical velocity is packed per field as well: a frame carrying
    ``pressure_vertical_velocity`` (omega, Pa s-1) is packed as ``OMEGA``
    and one carrying ``vertical_velocity`` (m s-1) as ``WW``
    (VERTICAL_VELOCITY_LEGACY_NAMES); the initializer turns either into
    the start state's W.  A frame carrying neither leaves W to the
    initializer's zero policy.

    A canonical field the regular join has no consumer for
    (REGULAR_JOIN_DROPPED_FIELDS, empty today) is not refused.  It is
    dropped, and named once through :func:`warn_regular_join_drops`, which
    the mapped plan review calls on the same question.
    """

    frames = tuple(frames)
    if not frames:
        raise ValueError("at least one mapped frame is required")
    carried: set[str] = set()
    for frame in frames:
        carried.update(frame.fields)
    warn_regular_join_drops(carried, subject="this mapped source")
    result = []
    for frame in frames:
        canonical = frame.fields
        if "air_pressure" not in canonical:
            # The header may satisfy its pressure requirement through a
            # declared hybrid coordinate, but the regular-source join
            # interpolates on a 3-D pressure FIELD; a waiver cannot be
            # interpolated.  (The L137 proof lane's probe 6 died here as
            # a bare KeyError.)
            raise ValueError(
                "mapped frames carry no air_pressure field; a "
                "hybrid_sigma_pressure mapping materializes one with a "
                "'pressure_from_vertical_coordinate' derivation "
                "(p = A + B*ps on its resolved coefficient ladder)"
            )
        pressure = canonical["air_pressure"].values
        if not np.isfinite(pressure).all() or np.any(pressure <= 0.0):
            raise ValueError("mapped air pressure must be finite and positive")
        stored_window = getattr(frame, "atmospheric_window", None)
        if stored_window is not None:
            if atmospheric_window != stored_window:
                raise ValueError("mapped atmospheric provider/window descriptors differ")
            levels_hpa = frame.source_pressure_hpa
            if levels_hpa is None:
                raise ValueError("windowed pressure lacks its original full-source ladder")
        else:
            levels_hpa = np.median(pressure, axis=(1, 2)) / 100.0
        field_items = _regular_snapshot_field_items(
            frame, pressure, soil_land_repair=soil_land_repair,
            initialize_absent_hydrometeors=initialize_absent_hydrometeors)
        del pressure
        grid_descriptor = frame.header.grid
        projection = None
        if grid_descriptor.projection != "regular_latitude_longitude":
            projection = {
                "family": grid_descriptor.projection,
                "parameters": dict(grid_descriptor.parameters),
            }
        snapshot_class = Era5Snapshot
        snapshot_options = {}
        if atmospheric_window is not None:
            from woof.ingest.atmospheric_window import (
                ATMOSPHERIC_FIELDS, WindowedAtmosphericSnapshot,
            )
            snapshot_class = WindowedAtmosphericSnapshot
            snapshot_options = {"window": atmospheric_window,
                                "full_factory": full_snapshot_factory}

            def windowed_items(items):
                for name, values in items:
                    yield name, (atmospheric_window.crop(values)
                                 if name in ATMOSPHERIC_FIELDS else values)
                    del values
            if stored_window is None:
                field_items = windowed_items(field_items)
        result.append(snapshot_class.from_field_items(
            valid_time=frame.valid_time,
            levels_hpa=np.asarray(levels_hpa, dtype=np.float64),
            latitude=np.asarray(frame.latitude, dtype=np.float64),
            longitude=np.asarray(frame.longitude, dtype=np.float64),
            field_items=field_items,
            projection=projection,
            **snapshot_options,
        ))
        validate_remaining = getattr(frame, "validate_remaining_fields", None)
        if validate_remaining is not None:
            validate_remaining()
    return tuple(result)


def _frame_headers(frames) -> list[object]:
    """Each frame's header, from frames in hand or from a frameset read
    lazily, without reading a single array."""

    header = getattr(frames, "header", None)
    if callable(header):
        return [header(index) for index in range(len(frames))]
    return [frame.header for frame in frames]


def completed_field_summary(frames) -> dict[str, dict[str, object]]:
    """What the frames completed where the source left a field out.

    Read from each field's ``source_field``, whose first reference names
    the completion and the number of values it derived.  One entry per
    completed field: the method, how many frames carried a derived value,
    and the derived and total value counts over the frames.  Empty when
    the source published every value itself.
    """

    summary: dict[str, dict[str, object]] = {}
    headers = _frame_headers(frames)
    for header in headers:
        document = header.to_dict() if hasattr(header, "to_dict") else dict(header)
        for descriptor in document.get("fields", ()):
            first = str(descriptor.get("source_field", "")).split(";", 1)[0]
            if not first.startswith(COMPLETED_FIELD_REFERENCE_PREFIX):
                continue
            method, _, count = first[len(COMPLETED_FIELD_REFERENCE_PREFIX):].partition(":")
            derived = int(count)
            total = int(np.prod([int(size) for size in descriptor.get("shape", ())]))
            entry = summary.setdefault(str(descriptor["canonical_name"]), {
                "method": method, "frames": 0, "frame_count": len(headers),
                "values_derived": 0, "values_total": 0,
            })
            entry["frames"] = int(entry["frames"]) + (1 if derived else 0)
            entry["values_derived"] = int(entry["values_derived"]) + derived
            entry["values_total"] = int(entry["values_total"]) + total
    return {
        name: entry for name, entry in sorted(summary.items())
        if int(entry["values_derived"])
    }


def warn_completed_fields(summary: Mapping[str, Mapping[str, object]], *,
                          subject: str) -> None:
    """Say once, plainly, which values the source did not carry."""

    for name, entry in summary.items():
        warn(
            f"{subject} did not carry {name.replace('_', ' ')} for "
            f"{int(entry['values_derived']):,} of {int(entry['values_total']):,} "
            f"values ({entry['frames']} of {entry['frame_count']} times); "
            "those values were derived hydrostatically from its own "
            "temperature, humidity, surface pressure and terrain height, and "
            "the preparation receipt counts them",
            why=("Initialization needs geopotential height on every level. "
                 "Where the source publishes none, each missing value is "
                 "integrated with the hypsometric equation from the nearest "
                 "level of the same column that has the source's own height, "
                 "or from the surface when no level has one.  Against a "
                 "source that publishes height, the derived values agree to "
                 "a few metres through the troposphere."),
            once=True,
        )


def warn_unpublished_bindings(records, *, subject: str) -> None:
    """Say once which contributing fields start from a recorded fallback."""

    for record in records:
        alignment = record.get("alignment")
        if not isinstance(alignment, Mapping) or alignment.get("status") != "UNPUBLISHED":
            continue
        fallback = dict(alignment.get("fallback") or {})
        files = ", ".join(Path(str(row.get("path"))).name
                          for row in record.get("data", ()))
        for name, fallback_id in sorted(fallback.items()):
            warn(
                f"{subject}'s {record.get('binding')} files ({files}) publish no "
                f"{name.replace('_', ' ')} record; it starts from "
                f"{fallback_id}, recorded in the preparation receipt",
                why=("HRRR surface files before 2020-12-02 (HRRRv3) carry no "
                     "vegetation fraction.  The start then reads the static "
                     "GREENFRAC monthly climatology interpolated to the start "
                     "date, as WPS does, and the receipt's contributing source "
                     "says so (status UNPUBLISHED, with the fallback id)."),
                once=True,
            )


def mapped_frame_receipt(
    mapping_path: str | Path,
    files: Sequence[str | Path],
    frames: Sequence[MappedSourceFrame],
) -> dict[str, object]:
    """Return a compact immutable-evidence payload for tests and handoffs."""

    mapping_path = Path(mapping_path).resolve()
    sources = tuple(Path(path).resolve() for path in files)
    return {
        "schema": FRAME_EVIDENCE_SCHEMA,
        "status": "DECODED_DECODER_UNBOUND_NOT_STOCK_WRF_CERTIFIED",
        "mapping": {
            "path": str(mapping_path),
            "sha256": _sha256(mapping_path),
        },
        "inputs": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in sources
        ],
        "frames": [
            {
                "valid_time": frame.valid_time.isoformat(),
                "member": frame.member,
                "source_cycle": frame.source_cycle.isoformat(),
                "grid_fingerprint": frame.grid_fingerprint,
                "shape": [int(frame.latitude.size), int(frame.longitude.size)],
                "vertical_count": int(frame.vertical_values.size),
                "fields": {
                    name: {
                        "axes": list(field.axes),
                        "shape": list(field.values.shape),
                        "units": field.units,
                        "missing": field.missing_count,
                        "sha256": _array_sha256(field.values),
                    }
                    for name, field in frame.fields.items()
                },
            }
            for frame in frames
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    """This console script's parser, built without parsing anything.

    Exposed so the docs/CLI parity test can read the option surface of a
    documented door without running it.
    """

    parser = argparse.ArgumentParser(
        prog="woof-mapped-inspect",
        description=(
            "Decode an rw-wps.mapping.v1 source and print hash-bound "
            "canonical-frame readiness evidence. This does not certify or "
            "write WRF initialization files."
        ),
    )
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument(
        "--input", dest="inputs", type=Path, action="append", required=True
    )
    parser.add_argument("--input-manifest", type=Path)
    parser.add_argument("--input-manifest-sha256")
    parser.add_argument("--grib1-bridge", type=Path)
    parser.add_argument("--grib2-inventory", type=Path)
    parser.add_argument("--grib2-dump", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Inspect mapped source bytes without enabling production export."""

    args = build_parser().parse_args(argv)
    # The evidence this prints is only as bindable as the tree that
    # produced it, so the tree is named before the evidence.
    from woof.provenance_gate import announce_for_main

    refusal = announce_for_main("woof-mapped-inspect")
    if refusal is not None:
        print(f"woof-mapped-inspect: {refusal}", file=sys.stderr)
        return 2
    try:
        report = inspect_mapped_source(
            args.mapping, args.inputs,
            input_manifest=args.input_manifest,
            input_manifest_sha256=args.input_manifest_sha256,
            grib1_bridge=args.grib1_bridge,
            grib2_inventory=args.grib2_inventory,
            grib2_dump=args.grib2_dump,
        )
    except FileNotFoundError as error:
        # A named artifact or input that is not on this machine, with a
        # remedy already composed by whichever resolver refused.  Exit 2
        # (argparse's usage-error convention) and one message, never a
        # traceback: this is the layer a console script owes a reader,
        # and this command had none -- it relayed a raw
        # `NotADirectoryError` from a cargo invocation instead.
        print(f"woof-mapped-inspect: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True, indent=2, allow_nan=False))
    return 0


__all__ = [
    "CanonicalField", "FRAME_EVIDENCE_SCHEMA", "INPUT_MANIFEST_SCHEMA",
    "INSPECTION_SCHEMA", "MAPPING_SCHEMA", "MappedSourceFrame",
    "completed_field_summary", "warn_completed_fields",
    "decode_mapped_source", "inspect_mapped_source", "load_mapping",
    "mapped_frame_receipt", "mapped_frames_to_regular_snapshots",
]


if __name__ == "__main__":
    raise SystemExit(main())
