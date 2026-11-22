"""The generic native input-normalization stage, run before packaged authoring.

A packaged profile decodes a source whose bytes the mapped engine can already
read.  Some sources publish bytes it cannot: a grid whose coordinates travel in
separate records rather than inside the grid definition, for instance, which is
what WMO grid-definition template 101 (an unstructured mesh) is.  Such a source
names a NORMALIZATION DOCUMENT -- a fourth packaged authority beside its
mapping, composition and provenance -- and this module runs it before the
profile is authored.

Everything model-shaped lives in that document: the object-name grammar, the
cycle grid and lead cadence, every field's GRIB selector octets and remap
method, the level ladders and soil-depth semantics, the intermediate window's
spacing and envelope, and the supplement/provenance roles the stage rebinds.
This module owns only the mechanism: requests, paths, bounded lossless
transport, cryptographic receipts and an immutable cache.  The shipped native
remapper owns GRIB decoding, the spherical search, interpolation, missing-value
decisions, grid-identity checks and GRIB encoding.  There is deliberately no
CDO, ecCodes, scipy or Python interpolation fallback.

Adding another source on such a grid is therefore one more JSON document and
one row -- the same seam as the packaged profile itself, one stage earlier.
"""
from __future__ import annotations

from dataclasses import dataclass, field as _field
from datetime import datetime, timedelta
import bz2
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

#: The schema every normalization document declares.
NORMALIZATION_SCHEMA = "gpuwm-source-normalization-v1"

#: The key a normalization stage adds to its profile's provenance authority.
#: The rest of that document is carried through unchanged, and the forecast
#: checks exactly that: the packaged content, plus this one key, and nothing
#: else (:func:`woof.prepared_single_domain_forecast`).
NORMALIZATION_RECEIPT_KEY = "normalization"

#: The remap methods the native side implements.  A document naming anything
#: else is a table error, refused at load rather than by the binary.
REMAP_METHODS = frozenset({"idw4", "surface-idw4", "nearest", "soil-nearest",
                           "seaice-nearest"})

#: The object kind that carries no lead.  Every other kind a document declares
#: is part of the forecast series.
INVARIANT_KIND = "time-invariant"

#: The declared roles a field may carry.  ``geometry`` fields describe WHERE
#: the native cells are and are consumed by the plan rather than remapped;
#: ``land_fraction`` classifies them; ``terrain`` is the supplement the
#: composition binds.
FIELD_ROLES = frozenset({"geometry", "land_fraction", "terrain"})


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _path_token(path: Path) -> str:
    text = str(path.resolve())
    if any(c in text for c in "\n\r\t\0"):
        raise ValueError(f"path contains a native job/list delimiter: {path!r}")
    return text


# ---------------------------------------------------------------------------
# The document
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NormalizationSpec:
    """One loaded normalization document, checked at load rather than at use."""

    name: str
    contract: str
    bridge: str
    source_id: str
    profile: str
    path: Path
    document: Mapping[str, object]
    pattern: re.Pattern = _field(repr=False, default=None)  # type: ignore[assignment]

    @property
    def fields(self) -> Mapping[str, Mapping[str, object]]:
        return self.document["fields"]  # type: ignore[index]

    @property
    def target(self) -> Mapping[str, object]:
        return self.document["target"]  # type: ignore[index]

    @property
    def limits(self) -> Mapping[str, object]:
        return self.document["limits"]  # type: ignore[index]

    @property
    def cadence(self) -> Mapping[str, object]:
        return self.document["cadence"]  # type: ignore[index]

    @property
    def roles(self) -> Mapping[str, str]:
        return self.document["roles"]  # type: ignore[index]

    @property
    def native_grid(self) -> Mapping[str, object]:
        return self.document["native_grid"]  # type: ignore[index]

    @property
    def cycle_format(self) -> str:
        return self.document["objects"]["cycle_format"]  # type: ignore[index]

    def names_with_role(self, role: str) -> tuple[str, ...]:
        return tuple(sorted(name for name, row in self.fields.items()
                            if row.get("role") == role))

    def horizon_hours(self, cycle_hour: int) -> int:
        for row in self.cadence["horizons"]:  # type: ignore[index]
            if cycle_hour in row["cycle_hours"]:
                return int(row["max_lead_hours"])
        raise ValueError(
            f"{self.source_id} publishes cycles "
            f"{sorted(self.cadence['cycle_hours'])} UTC, not {cycle_hour:02d}")

    def levels_for(self, name: str) -> tuple[int, ...]:
        levels = self.fields[name].get("levels")
        if levels is None:
            return ()
        return tuple(int(x) for x in levels["values"])

    def selector_for(self, name: str, level: int | None):
        """The seven message-metadata values the native side checks."""

        row = self.fields[name]
        s = row["selector"]
        levels = row.get("levels")
        first, second = s["level_value"], s["second_level_value"]
        if levels is not None:
            if level is None or str(level) not in {str(x) for x in levels["values"]}:
                raise ValueError(
                    f"{self.name} field {name!r} has no level {level!r} on its "
                    f"declared ladder")
            if levels["kind"] == "scaled":
                first = float(level) * float(levels["scale"])
            elif levels["kind"] == "table":
                first = float(levels["values"][str(level)])
            else:
                top, bottom = levels["values"][str(level)]
                first, second = float(top), float(bottom)
        if first is None or second is None:
            raise ValueError(
                f"{self.name} field {name!r} needs a level to resolve its selector")
        return (int(s["discipline"]), int(s["category"]), int(s["parameter"]),
                int(s["level_type"]), float(first),
                int(s["second_level_type"]), float(second))

    def output_name(self, path: Path) -> str:
        rename = self.document["objects"]["output_rename"]  # type: ignore[index]
        return path.name.removesuffix(".bz2").replace(
            rename["from"], rename["to"], 1)


def _check_document(document, path: Path) -> None:
    schema = document.get("schema") if isinstance(document, dict) else None
    if schema != NORMALIZATION_SCHEMA:
        raise ValueError(
            f"{path.name} declares schema {schema!r}; this WOOF-ArWen reads "
            f"{NORMALIZATION_SCHEMA!r}")
    for key in ("name", "contract", "bridge", "source_id", "profile",
                "native_grid", "objects", "cadence", "roles", "plan_fields",
                "target", "limits", "fields", "methods"):
        if key not in document:
            raise ValueError(f"{path.name} omits the required key {key!r}")
    fields = document["fields"]
    for name, row in fields.items():
        if row.get("mode") not in REMAP_METHODS:
            raise ValueError(
                f"{path.name} field {name!r} names remap method "
                f"{row.get('mode')!r}; the native side implements "
                f"{sorted(REMAP_METHODS)}")
        if "role" in row and row["role"] not in FIELD_ROLES:
            raise ValueError(
                f"{path.name} field {name!r} declares role {row['role']!r}; "
                f"the stage binds {sorted(FIELD_ROLES)}")
        selector = row.get("selector")
        if not isinstance(selector, dict) or set(selector) != {
                "discipline", "category", "parameter", "level_type",
                "level_value", "second_level_type", "second_level_value"}:
            raise ValueError(
                f"{path.name} field {name!r} must pin exactly the seven "
                "selector values the native side checks")
        levels = row.get("levels")
        if levels is not None and levels.get("kind") not in {
                "scaled", "table", "bounds"}:
            raise ValueError(
                f"{path.name} field {name!r} declares an unknown level "
                f"ladder kind {levels.get('kind')!r}")
    for role in ("terrain", "provenance"):
        if not isinstance(document["roles"].get(role), str):
            raise ValueError(f"{path.name} must name its {role} role")
    for role in ("geometry", "terrain", "land_fraction"):
        if not any(row.get("role") == role for row in fields.values()):
            raise ValueError(f"{path.name} declares no {role} field")
    for name in document["plan_fields"]:
        if name not in fields:
            raise ValueError(
                f"{path.name} plans on {name!r}, which it does not declare")
    # The native side takes the plan records POSITIONALLY: latitude,
    # longitude, land fraction.  Nothing downstream can tell a latitude
    # record from a longitude one -- both are geometry, both are degrees,
    # both decode -- so a document that lists them the other way round
    # builds a plan with the axes swapped and every later check passes.
    # The roles are what distinguishes them, so the roles are checked here.
    plan_roles = [fields[name].get("role") for name in document["plan_fields"]]
    if plan_roles != ["geometry", "geometry", "land_fraction"]:
        raise ValueError(
            f"{path.name} must plan on exactly three records in the order "
            "the native side reads them: latitude and longitude (role "
            "'geometry'), then the land fraction (role 'land_fraction'); "
            f"it lists roles {plan_roles}")


def load_document(path: Path) -> NormalizationSpec:
    """Load and check one normalization document from an explicit path."""

    path = Path(path)
    document = json.loads(path.read_text(encoding="utf-8"))
    _check_document(document, path)
    return NormalizationSpec(
        name=str(document["name"]), contract=str(document["contract"]),
        bridge=str(document["bridge"]), source_id=str(document["source_id"]),
        profile=str(document["profile"]), path=path,
        document=MappingProxyType(document),
        pattern=re.compile(str(document["objects"]["pattern"])))


def load_normalization(name: str) -> NormalizationSpec:
    """Resolve a normalizer NAME through the packaged authority inventory.

    Closed dispatch: a name the distribution does not ship never becomes an
    arbitrary import or an arbitrary file read.
    """

    from woof.source_authorities import packaged_normalization
    spec = load_document(packaged_normalization(name))
    if spec.name != name:
        raise ValueError(
            f"packaged normalization document for {name!r} calls itself "
            f"{spec.name!r}")
    return spec


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InputObject:
    spec: NormalizationSpec
    path: Path
    cycle: str
    kind: str
    field: str
    lead: int | None
    level: int | None

    @property
    def identity(self):
        return self.cycle, self.kind, self.field, self.lead, self.level

    @property
    def output_name(self) -> str:
        # The name describes the output, and paths are never taken from an
        # upstream object or receipt without confinement checks.
        return self.spec.output_name(self.path)

    @property
    def mode(self) -> str:
        return str(self.spec.fields[self.field]["mode"])

    @property
    def role(self) -> str | None:
        role = self.spec.fields[self.field].get("role")
        return None if role is None else str(role)

    @property
    def selector(self):
        return self.spec.selector_for(self.field, self.level)


def parse_object(spec: NormalizationSpec, path: str | Path) -> InputObject:
    path = Path(path).resolve()
    _path_token(path)
    match = spec.pattern.fullmatch(path.name)
    if match is None:
        raise ValueError(
            f"{spec.source_id} requires its publisher's native object names, "
            f"not {path.name!r}")
    g = match.groupdict()
    try:
        cycle = datetime.strptime(g["cycle"], spec.cycle_format)
    except ValueError as exc:
        raise ValueError(f"invalid {spec.source_id} cycle in {path.name}") from exc
    if cycle.hour not in spec.cadence["cycle_hours"]:
        raise ValueError(
            f"{spec.source_id} cycles are "
            + ", ".join(f"{h:02d}" for h in spec.cadence["cycle_hours"])
            + " UTC")
    lead = None if g["lead"] is None else int(g["lead"])
    level = None if g["level"] is None else int(g["level"])
    field, kind = g["field"], g["kind"]
    row = spec.fields.get(field)
    valid = row is not None and row["kind"] == kind
    if valid and kind == INVARIANT_KIND:
        declared = spec.levels_for(field)
        invariant_level = g.get("invariant_level")
        if declared and invariant_level is not None:
            level = int(invariant_level)
        valid = lead is None and ((level in declared) if declared else (level is None and invariant_level in (None, "0")))
    elif valid:
        step = int(spec.cadence["step_hours"])
        horizon = spec.horizon_hours(cycle.hour)
        valid = lead is not None and 0 <= lead <= horizon and lead % step == 0
        declared = spec.levels_for(field)
        valid &= (level in declared) if declared else (level is None)
    if not valid:
        raise ValueError(
            f"object is outside the {spec.source_id} product/cadence "
            f"contract: {path.name}")
    return InputObject(spec, path, g["cycle"], kind, field, lead, level)


def _expected_state(spec: NormalizationSpec) -> set[tuple[str, str, int | None]]:
    expected = set()
    for name, row in spec.fields.items():
        if row["kind"] == INVARIANT_KIND:
            continue
        levels = spec.levels_for(name)
        if levels:
            expected |= {(row["kind"], name, level) for level in levels}
        else:
            expected.add((row["kind"], name, None))
    return expected


def validate_inventory(spec: NormalizationSpec,
                       paths: Iterable[str | Path]) -> tuple[InputObject, ...]:
    limit = int(spec.limits["max_input_files"])
    collected = []
    for path in paths:
        if len(collected) >= limit:
            raise ValueError(
                f"{spec.source_id} input inventory exceeds the bounded file count")
        collected.append(parse_object(spec, path))
    objects = tuple(collected)
    if not objects:
        raise ValueError(f"{spec.source_id} input inventory is empty")
    identities = [obj.identity for obj in objects]
    if len(set(identities)) != len(identities):
        raise ValueError(
            "duplicate object identity, even if its path or compression differs")
    if len({obj.cycle for obj in objects}) != 1:
        raise ValueError("mixed cycle references are not a forcing series")
    required = {(name, level) for name, row in spec.fields.items()
                if row["kind"] == INVARIANT_KIND
                for level in (spec.levels_for(name) or (None,))}
    missing_invariants = required - {(obj.field, obj.level) for obj in objects
                                     if obj.kind == INVARIANT_KIND}
    if missing_invariants:
        raise ValueError(
            f"missing coordinate/invariant objects: {sorted(missing_invariants)}")
    # A boundary series is uniform, at the publisher's spacing or a whole
    # multiple of it (A173): every lead in it is one the publisher wrote,
    # and nothing normalized here depends on the spacing, since each lead
    # is remapped alone through the same plan.  A gap or an uneven series
    # is refused because the decode reads one interval for the whole run.
    step = int(spec.cadence["step_hours"])
    leads = sorted({obj.lead for obj in objects if obj.lead is not None})
    gaps = {b - a for a, b in zip(leads, leads[1:])}
    if not leads or len(gaps) > 1 or any(gap % step for gap in gaps):
        raise ValueError(
            "forcing leads must be one uniform series at a whole multiple "
            f"of the publisher's {step}-hour spacing, not leads {leads}")
    expected = _expected_state(spec)
    for lead in leads:
        present = {(o.kind, o.field, o.level) for o in objects if o.lead == lead}
        if present != expected:
            missing = sorted(expected - present, key=str)
            raise ValueError(
                f"incomplete f{lead:03d} state; missing {missing[:8]}"
                f" ({len(missing)} missing records)")
    return tuple(sorted(objects,
                        key=lambda o: (o.lead if o.lead is not None else -1,
                                       o.kind, o.field, o.level or 0)))


# ---------------------------------------------------------------------------
# The regular intermediate window
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TargetWindow:
    spec: NormalizationSpec
    west: float
    south: float
    nx: int
    ny: int
    dx: float
    dy: float

    def __post_init__(self):
        limits = self.spec.target
        step = float(limits["step_degrees"])
        max_points = int(limits["max_points"])
        max_latitude = float(limits["max_abs_latitude"])
        max_span = float(limits["max_longitude_span_degrees"])
        if not all(math.isfinite(x)
                   for x in (self.west, self.south, self.dx, self.dy)):
            raise ValueError("nonfinite intermediate geometry")
        if (type(self.nx) is not int or type(self.ny) is not int
                or min(self.nx, self.ny) < 2):
            raise ValueError(
                "the intermediate must have at least two points per axis")
        if self.dx != step or self.dy != step:
            raise ValueError(
                f"{self.spec.source_id} intermediate spacing is fixed at "
                f"{step} degrees by its normalization document")
        if self.nx * self.ny > max_points:
            raise ValueError(
                f"intermediate exceeds {max_points:,} points; split the domain")
        if not -180. <= self.west < 180.:
            raise ValueError("the intermediate west must be in [-180,180)")
        if (self.nx - 1) * self.dx >= max_span:
            raise ValueError(
                f"a regional longitude window narrower than {max_span:g} "
                "degrees is required")
        if (self.south < -max_latitude
                or self.south + (self.ny - 1) * self.dy > max_latitude):
            raise ValueError(
                "this intermediate cannot include either pole or extend "
                f"beyond {max_latitude:g} degrees; a polar-vector route is "
                "required")

    def geometry(self) -> dict[str, object]:
        """The window itself, without the spec it was measured against."""

        return {"west": self.west, "south": self.south, "dx": self.dx,
                "dy": self.dy, "nx": self.nx, "ny": self.ny}

    def text(self) -> str:
        return (f"{self.west:.9f} {self.south:.9f} {self.dx:.9f} "
                f"{self.dy:.9f} {self.nx} {self.ny}\n")


def target_from_points(spec: NormalizationSpec, latitudes: Sequence[float],
                       longitudes: Sequence[float]) -> TargetWindow:
    # Extent planning only.  Numeric remapping remains native.  The minimum
    # covering longitude arc is independent of the -180/180 or 0/360 branch.
    if len(latitudes) != len(longitudes) or len(latitudes) == 0:
        raise ValueError(
            "target coordinate arrays must be nonempty and equal in length")
    step = float(spec.target["step_degrees"])
    halo = float(spec.target["halo_degrees"])
    lat = [float(x) for x in latitudes]
    lon = [float(x) for x in longitudes]
    if not all(math.isfinite(x) for x in lat + lon) or any(abs(x) > 90 for x in lat):
        raise ValueError("invalid target latitude/longitude")
    ordered = sorted(set(x % 360 for x in lon))
    gaps = [(ordered[(i + 1) % len(ordered)] + (360 if i == len(ordered) - 1 else 0)
             - x, i) for i, x in enumerate(ordered)]
    _, cut = max(gaps)
    west0 = ordered[(cut + 1) % len(ordered)]
    east0 = max(west0 + ((x - west0) % 360) for x in ordered)
    west = math.floor((west0 - halo) / step) * step
    east = math.ceil((east0 + halo) / step) * step
    south = math.floor((min(lat) - halo) / step) * step
    north = math.ceil((max(lat) + halo) / step) * step
    # Keep the western anchor in [-180,180); the east may pass +180.
    shift = math.floor((west + 180.) / 360.) * 360.
    return TargetWindow(spec, west - shift, south,
                        round((east - west) / step) + 1,
                        round((north - south) / step) + 1, step, step)


def target_from_wps(spec: NormalizationSpec, path: Path) -> TargetWindow:
    from woof.static.projection import grids_from_wps_namelist
    lat, lon = [], []
    try:
        grids = grids_from_wps_namelist(path)
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ValueError(
            "cannot resolve the target through the WPS projection owner: "
            f"{exc}") from exc
    for grid in grids:
        # All corner points, not just four extrema: a projected domain's
        # northernmost point can be in the middle of an edge or its interior.
        la, lo = grid.latlon_c()
        lat.extend(la.ravel().tolist())
        lon.extend(lo.ravel().tolist())
    return target_from_points(spec, lat, lon)


# ---------------------------------------------------------------------------
# Bounded transport and the native stage
# ---------------------------------------------------------------------------

def _expand(spec: NormalizationSpec, source: Path, destination: Path) -> None:
    """Bounded lossless transport codec, never scientific field decoding."""

    bound = int(spec.limits["max_expanded_object_bytes"])
    with source.open("rb") as stream:
        head = stream.read(4)
    if head.startswith(b"BZh"):
        opener = bz2.open
    elif head == b"GRIB":
        opener = open
    else:
        raise ValueError(f"not a GRIB2 or bz2-wrapped GRIB object: {source}")
    total = 0
    try:
        with opener(source, "rb") as incoming, destination.open("wb") as outgoing:
            while True:
                block = incoming.read(1024 * 1024)
                if not block:
                    break
                total += len(block)
                if total > bound:
                    raise ValueError(f"expanded object exceeds {bound} bytes")
                outgoing.write(block)
        with destination.open("rb") as stream:
            if stream.read(4) != b"GRIB":
                raise ValueError(f"expanded object is not GRIB2: {source}")
    except BaseException:
        destination.unlink(missing_ok=True)
        raise


def _resolve_bridge(spec: NormalizationSpec) -> Path:
    from woof import bridges
    path = bridges.find_bridge(spec.bridge)
    if path is None:
        raise RuntimeError(
            f"missing native {spec.bridge}; build the tools/grib1_bridge "
            "crate or install a bridge bundle containing this artifact")
    path = Path(path).resolve()
    okay, why = bridges.launchable(path)
    if not okay:
        raise RuntimeError(f"{spec.bridge} is not launchable: {why}")
    if spec.contract.encode() not in path.read_bytes():
        raise RuntimeError(
            f"{spec.bridge} has a stale or absent contract marker; rebuild "
            "the native bridge")
    return path


def _run(spec: NormalizationSpec, command: list[str]) -> None:
    timeout = int(spec.limits["native_stage_timeout_seconds"])
    try:
        done = subprocess.run(command, check=False, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"native normalization exceeded the {timeout}-second stage "
            "limit") from exc
    if done.returncode:
        raise RuntimeError(
            f"native normalization failed ({done.returncode}): "
            f"{done.stderr[-6000:].strip()}")


def _plan_records(spec: NormalizationSpec,
                  paths: Mapping[str, Path]) -> list[str]:
    """PATH and declared SELECTOR for each plan record, in document order.

    Every record the native side reads is addressed by the seven selector
    values THIS document declares -- the three the plan is built from
    exactly as much as the fields the plan is applied to.  Coordinate
    records are numbered in producer-local code tables, so a code number
    compiled into the converter would be a producer compiled into the
    converter, and a second source on the same grid template would need a
    second binary.
    """

    tokens: list[str] = []
    for name in spec.document["plan_fields"]:  # type: ignore[index]
        tokens.append(str(paths[name]))
        tokens.append(",".join(str(x) for x in spec.selector_for(name, None)))
    return tokens


def _job(obj: InputObject, source: Path, destination: Path) -> str:
    tokens = [_path_token(source), _path_token(destination),
              *(str(x) for x in obj.selector), obj.mode, obj.cycle,
              str((obj.lead or 0) * 3600)]
    return "\t".join(tokens) + "\n"


def _stage_workers(count: int) -> int:
    """Threads for COUNT independent per-object steps: the host's usable cores.

    Named breakage (WOOF canary gate 2026-10-24, ICON 3-1-0.5 km): the stage
    expanded, remapped and hashed ICON global's 234 objects one after another,
    54 s on one core of sixteen while the card waited, inside the 2.9 min
    preparation of the first-map path.  Every object is independent (its own
    expanded twin, its own job, its own output file), so they run side by
    side and each output is the same bytes the serial loop wrote.
    """

    try:
        from woof.ingest.preparation_workers import effective_workers
        cores = effective_workers()
    except Exception:  # the standalone preparation package
        try:
            cores = len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            cores = os.cpu_count() or 1
    return max(1, min(int(count), int(cores)))


def _parallel_map(function, items) -> list:
    """``list(map(function, items))`` on the host's cores, in item order."""

    items = list(items)
    if len(items) < 2:
        return [function(item) for item in items]
    with ThreadPoolExecutor(max_workers=_stage_workers(len(items)),
                            thread_name_prefix="normalize") as pool:
        return list(pool.map(function, items))


def _file_identity(path: Path) -> dict[str, object]:
    # Recheck basic identity around hashing.  A changed input is not a cache hit.
    first = path.stat()
    digest = _sha(path)
    last = path.stat()
    if ((first.st_size, first.st_mtime_ns, first.st_ino)
            != (last.st_size, last.st_mtime_ns, last.st_ino)):
        raise ValueError(f"input changed while it was being hashed: {path}")
    return {"path": _path_token(path), "bytes": last.st_size, "sha256": digest}


def _check_cache(spec: NormalizationSpec, directory: Path, request: dict) -> dict:
    if directory.is_symlink() or (directory / "fields").is_symlink():
        raise ValueError("normalization cache directories must not be symlinks")
    provenance_path = directory / "provenance.json"
    if (provenance_path.is_symlink()
            or provenance_path.stat().st_size > 32 * 1024 * 1024):
        raise ValueError("symlink or oversized normalization provenance")
    provenance = json.loads(provenance_path.read_text())
    normalization = (provenance.get("normalization")
                     if isinstance(provenance, dict) else None)
    if not isinstance(normalization, dict) or normalization.get("request") != request:
        raise ValueError("normalization cache identity mismatch")
    rows = normalization.get("outputs")
    if (not isinstance(rows, list) or not rows
            or not all(isinstance(row, dict) for row in rows)):
        raise ValueError("normalization cache lacks its output inventory")
    geometry = set(spec.names_with_role("geometry"))
    expected_names = {parse_object(spec, row["path"]).output_name
                      for row in request["inputs"]
                      if parse_object(spec, row["path"]).field not in geometry}
    if any(not isinstance(row.get("name"), str) for row in rows):
        raise ValueError("cache output names must be strings")
    if ({row["name"] for row in rows} != expected_names
            or len(rows) != len(expected_names)):
        raise ValueError(
            "normalization cache output inventory differs from the request")

    def verify(path: Path, row: dict):
        if (type(row.get("bytes")) is not int or row["bytes"] <= 0
                or not isinstance(row.get("sha256"), str)
                or re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) is None):
            raise ValueError("invalid cache file identity")
        if (path.is_symlink() or path.stat().st_size != row["bytes"]
                or _sha(path) != row["sha256"]):
            raise ValueError(f"normalization cache file was changed: {path.name}")

    for row in rows:
        name = row["name"]
        if Path(name).name != name or "/" in name or "\\" in name:
            raise ValueError(
                "normalization cache path escapes the output directory")
    _parallel_map(lambda row: verify(directory / "fields" / row["name"], row), rows)
    plan = normalization.get("plan")
    if not isinstance(plan, dict):
        raise ValueError("normalization cache lacks its plan identity")
    verify(directory / "weights.bin", plan)
    text = "".join(_path_token(directory / "fields" / row["name"]) + "\n"
                   for row in rows)
    listing = directory / "inputs.txt"
    if listing.is_symlink() or listing.read_text() != text:
        raise ValueError("normalized input-list was changed")
    return provenance


def _snapshot_expand(spec: NormalizationSpec, source: Path, destination: Path,
                     identity: dict) -> None:
    """Decode a private byte-for-byte snapshot, not a mutable upstream path."""

    snapshot = destination.with_suffix(".transport")
    digest = hashlib.sha256()
    count = 0
    bound = int(spec.limits["max_expanded_object_bytes"])
    try:
        with source.open("rb") as incoming, snapshot.open("xb") as outgoing:
            for block in iter(lambda: incoming.read(1024 * 1024), b""):
                count += len(block)
                if count > bound:
                    raise ValueError("transport object exceeds the bounded size")
                digest.update(block)
                outgoing.write(block)
        if count != identity["bytes"] or digest.hexdigest() != identity["sha256"]:
            raise ValueError(f"input changed before normalization: {source}")
        _expand(spec, snapshot, destination)
    finally:
        snapshot.unlink(missing_ok=True)


def normalize(spec: NormalizationSpec, objects: tuple[InputObject, ...],
              target: TargetWindow, cache_root: Path, *,
              bridge: Path | None = None) -> tuple[Path, dict]:
    """Create or verify a complete immutable regional normalization artifact."""

    from woof.source_authorities import packaged_authorities
    objects = validate_inventory(spec, [o.path for o in objects])
    authorities = dict(packaged_authorities(spec.profile))
    authorities["normalization"] = spec.path
    bridge = _resolve_bridge(spec) if bridge is None else Path(bridge).resolve()
    native = spec.native_grid
    request = {"contract": spec.contract, "normalizer": spec.name,
               "source_cells": int(native["cells"]),
               "target": target.geometry(), "converter": _file_identity(bridge),
               "inputs": _parallel_map(lambda o: _file_identity(o.path), objects),
               "authorities": {role: _sha(Path(path))
                               for role, path in authorities.items()}}
    key = hashlib.sha256(_canonical(request)).hexdigest()
    cache_root = Path(cache_root).resolve()
    final = cache_root / key
    if final.exists() or final.is_symlink():
        return final, _check_cache(spec, final, request)
    cache_root.mkdir(parents=True, exist_ok=True)
    geometry = set(spec.names_with_role("geometry"))
    plan_fields = tuple(str(x) for x in spec.document["plan_fields"])
    with tempfile.TemporaryDirectory(prefix=".normalizing-", dir=cache_root) as work:
        work = Path(work)
        (work / "fields").mkdir()
        target_path = work / "target.txt"
        target_path.write_text(target.text(), encoding="ascii")
        coordinate_paths = {}
        by_field = {o.field: o for o in objects if o.kind == INVARIANT_KIND}
        input_identities = {row["path"]: row for row in request["inputs"]}
        for name in plan_fields:
            destination = work / f"{name}.grib2"
            source = by_field[name].path
            _snapshot_expand(spec, source, destination,
                             input_identities[_path_token(source)])
            coordinate_paths[name] = destination
        plan_path = work / "weights.bin"
        _run(spec, [str(bridge), "plan",
                    *_plan_records(spec, coordinate_paths),
                    str(target_path), str(plan_path), objects[0].cycle,
                    str(int(native["cells"])),
                    str(int(native["originating_centre"]))])
        for path in coordinate_paths.values():
            path.unlink()
        target_path.unlink()
        remapped = [obj for obj in objects if obj.field not in geometry]

        def remap(indexed):
            # One object: its own expanded twin and job list in its own
            # folder, published to the same destination the serial loop
            # used.  The converter writes each output from that object's
            # bytes and the plan alone, so the order the objects run in
            # changes no byte (_stage_workers).
            index, obj = indexed
            staging = work / f".object-{index:05d}"
            staging.mkdir()
            unpacked = staging / "current.grib2"
            _snapshot_expand(spec, obj.path, unpacked,
                             input_identities[_path_token(obj.path)])
            destination = work / "fields" / obj.output_name
            jobs = staging / "current.jobs"
            jobs.write_text(_job(obj, unpacked, destination), encoding="utf-8")
            _run(spec, [str(bridge), "apply", str(plan_path), str(jobs)])
            row = {"name": obj.output_name,
                   "bytes": destination.stat().st_size,
                   "sha256": _sha(destination)}
            shutil.rmtree(staging)
            return row

        rows = _parallel_map(remap, enumerate(remapped))
        # A raw input changing during native work invalidates the whole stage.
        changed = _parallel_map(
            lambda expected: _file_identity(Path(expected["path"])) != expected,
            request["inputs"])
        for expected, moved in zip(request["inputs"], changed):
            if moved:
                raise ValueError(
                    f"input changed during normalization: {expected['path']}")
        if _file_identity(bridge) != request["converter"]:
            raise ValueError("converter changed during normalization")
        # The profile's provenance authority, carried through with this
        # stage's receipt added under one declared key.  The receipt names
        # the normalizer and pins, inside itself, the digest of every
        # authority it was produced under -- including the document it
        # extends -- so the forecast can check the extension rather than
        # having to refuse it.
        provenance = json.loads(Path(authorities["provenance"]).read_text())
        provenance[NORMALIZATION_RECEIPT_KEY] = {
            "request": request, "outputs": rows,
            "plan": {"bytes": plan_path.stat().st_size, "sha256": _sha(plan_path)},
            "methods": dict(spec.document["methods"]),
            "qualification": spec.document.get(
                "qualification",
                "No stock-WRF, GPU forecast, conservation or skill "
                "certification is implied"),
        }
        text = "".join(_path_token(final / "fields" / row["name"]) + "\n"
                       for row in rows)
        (work / "inputs.txt").write_text(text, encoding="utf-8", newline="\n")
        (work / "provenance.json").write_bytes(_canonical(provenance) + b"\n")
        try:
            os.rename(work, final)
        except OSError:
            # A concurrent producer may publish the same immutable request.
            # Never replace its directory or accept it merely because it exists.
            if not final.is_dir():
                raise
            provenance = _check_cache(spec, final, request)
    return final, _check_cache(spec, final, request)


def _check_wps_time_coverage(spec: NormalizationSpec, path: Path,
                             objects: tuple[InputObject, ...],
                             experiment_config: Path | None = None) -> None:
    from woof.static.projection import _parse_wps_namelist
    share = _parse_wps_namelist(path)
    declared_interval = int(spec.cadence["forcing_interval_seconds"])
    interval = share.get("interval_seconds")
    if isinstance(interval, (list, tuple)):
        interval = interval[0] if interval else None
    leads = sorted({o.lead for o in objects if o.lead is not None})
    # The namelist's interval is the series' own spacing (A173): a whole
    # multiple of the publisher's, and equal to the gap between the leads
    # these objects carry, so the forcing the decode reads every
    # interval_seconds exists at each of those times.
    gaps = {(b - a) * 3600 for a, b in zip(leads, leads[1:])}
    if (isinstance(interval, bool) or not isinstance(interval, int)
            or interval <= 0 or interval % declared_interval
            or (gaps and gaps != {interval})):
        spacing = (f"the {next(iter(gaps))} s spacing of these leads"
                   if len(gaps) == 1 else
                   f"a whole multiple of {declared_interval}")
        raise ValueError(
            f"{spec.source_id} requires WPS interval_seconds={spacing}, "
            f"not {interval!r}")
    cycle = datetime.strptime(objects[0].cycle, spec.cycle_format)
    earliest = cycle + timedelta(hours=min(leads))
    latest = cycle + timedelta(hours=max(leads))
    declared = []
    if experiment_config is not None:
        # ArWen's domain wizard deliberately omits WPS start_date/end_date:
        # the experiment owns the clock.  Use that same validated owner, not
        # a source-only TOML parser or an invented cycle/date default.
        from woof.experiment import load_experiment
        experiment = load_experiment(experiment_config)
        declared.extend(
            (("experiment start_time", experiment.start_time),
             ("experiment end",
              experiment.start_time + timedelta(seconds=experiment.run_seconds))))
    has_start, has_end = "start_date" in share, "end_date" in share
    if has_start != has_end:
        raise ValueError("WPS start_date/end_date must be supplied together")
    if has_start:
        parsed = {}
        for key in ("start_date", "end_date"):
            values = share[key]
            if not isinstance(values, (list, tuple)):
                values = [values]
            if not values:
                raise ValueError(f"empty WPS {key}")
            parsed[key] = []
            for text in values:
                try:
                    value = datetime.strptime(str(text), "%Y-%m-%d_%H:%M:%S")
                except ValueError as exc:
                    raise ValueError(f"invalid WPS {key}: {text!r}") from exc
                parsed[key].append(value)
                declared.append((f"WPS {key}", value))
        if len(parsed["start_date"]) != len(parsed["end_date"]):
            raise ValueError("WPS start_date/end_date have different domain counts")
        if any(end < start for start, end in zip(parsed["start_date"],
                                                 parsed["end_date"])):
            raise ValueError("WPS end_date precedes start_date")
    if not declared:
        raise ValueError(
            "time coverage needs --experiment-config or explicit WPS "
            "start_date/end_date")
    for label, value in declared:
        if not earliest <= value <= latest:
            raise ValueError(
                f"{spec.source_id} forcing {earliest}..{latest} does not "
                f"bracket {label}={value}")


def normalize_namespace(spec: NormalizationSpec, args) -> dict[str, object]:
    """Source front-door hook, after input-list expansion and before authoring."""

    if not getattr(args, "wps_namelist", None):
        raise ValueError(
            "native normalization requires --wps-namelist, including --author-only")
    objects = validate_inventory(spec, args.mapped_inputs)
    terrain_role = spec.roles["terrain"]
    terrain_names = spec.names_with_role("terrain")
    if len(terrain_names) != 1:
        raise ValueError(
            f"{spec.name} must declare exactly one terrain field, not "
            f"{list(terrain_names)}")
    # The supplement must be the terrain object travelling in this inventory.
    expected_terrain = next(o.path for o in objects if o.field == terrain_names[0])
    bindings = getattr(args, "supplement", ()) or ()
    if len(bindings) != 1:
        raise ValueError(
            f"{spec.source_id} expects exactly its native "
            f"{terrain_names[0]} supplement")
    role, sep, supplied = str(bindings[0]).partition("=")
    if not sep or role != terrain_role or Path(supplied).resolve() != expected_terrain:
        raise ValueError(
            f"the {terrain_names[0]} supplement must match the same-cycle "
            "primary invariant")
    target = target_from_wps(spec, Path(args.wps_namelist))
    _check_wps_time_coverage(spec, Path(args.wps_namelist), objects,
                             getattr(args, "experiment_config", None))
    result = {"contract": spec.contract, "normalizer": spec.name,
              "source": spec.source_id, "target": target.geometry(),
              "input_count": len(objects), "dry_run": bool(args.dry_run),
              "normalization_before_manifest": True}
    if args.dry_run:
        return result
    # The manifest location exists in both authoring and explicit-manifest use.
    manifest = (getattr(args, "author_input_manifest", None)
                or getattr(args, "source_sha256s", None))
    if manifest is None:
        raise ValueError(
            "normalization needs a manifest destination or a pinned existing "
            "manifest")
    directory, provenance = normalize(
        spec, objects, target,
        Path(manifest).resolve().parent / f"{spec.source_id}-normalized")
    normalized = [directory / "fields" / row["name"]
                  for row in provenance["normalization"]["outputs"]]
    # Materialize a compact list, but never modify the fetch's raw input-list.
    text = "".join(_path_token(path) + "\n" for path in normalized)
    input_list = directory / "inputs.txt"
    if input_list.is_symlink() or input_list.read_text() != text:
        raise ValueError("normalized input-list was changed")
    args.mapped_inputs = normalized
    args.input_list = input_list
    terrain = directory / "fields" / next(
        o.output_name for o in objects if o.field == terrain_names[0])
    args.supplement = [f"{terrain_role}={terrain}"]
    # The same declared role the profile binds, holding the same document
    # plus this stage's receipt: every raw input hash, the converter identity,
    # the plan, the target geometry and the normalized output inventory.  The
    # ordinary authoring path seals it like any other provenance row.
    args.provenance = [
        f"{spec.roles['provenance']}={directory / 'provenance.json'}"]
    result.update(output_count=len(normalized),
                  provenance=str(directory / "provenance.json"))
    return result


def declared_normalization(name: str) -> dict[str, object]:
    """What a dry run says: the stage that will run, and on what.

    A dry run opens no input here because no other source's dry run opens
    one either: `--dry-run` means the same thing at this door whichever
    source is named, and a source whose door alone refused an unreadable
    path would be a per-model dry run.
    """

    spec = load_normalization(name)
    return {"contract": spec.contract, "normalizer": spec.name,
            "source": spec.source_id, "bridge": spec.bridge,
            "native_grid": dict(spec.native_grid),
            "normalization_before_manifest": True, "dry_run": True}


def normalize_packaged_inputs(name: str, args) -> dict[str, object]:
    """Closed dispatch from a profile's declared normalizer name.

    Unknown names never become arbitrary Python imports or arbitrary file
    reads: only a document the distribution ships resolves.
    """

    return normalize_namespace(load_normalization(name), args)


__all__ = ["NORMALIZATION_SCHEMA", "NORMALIZATION_RECEIPT_KEY",
           "REMAP_METHODS", "declared_normalization", "NormalizationSpec",
           "InputObject", "TargetWindow", "load_document",
           "load_normalization", "parse_object", "validate_inventory",
           "target_from_points", "target_from_wps", "normalize",
           "normalize_namespace", "normalize_packaged_inputs"]
