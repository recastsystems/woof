"""Validate WPS met_em fields and expose the shared native forcing object.

All NetCDF payloads and unit conversions are decoded by the Rust reader.
The adapter performs no horizontal interpolation or wind rotation.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from difflib import get_close_matches
from pathlib import Path
import re
import hashlib
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from woof.ingest.horiz import HorizontalSnapshot
from woof.ingest.analyzed_numbers import METGRID_NUMBER_FIELDS
from woof.netcdf_bridge import open_dataset


class MetgridRefusal(ValueError):
    """A met_em file this module will not pretend to understand.

    Its own class rather than a bare ``ValueError`` so a door can print
    it as a user-facing sentence instead of a traceback, and so a test
    can tell "this file is wrong" from "this code is wrong".
    """


#: The single 3-D stack met_em carries per field, and the ArWen forcing
#: name each half of it becomes.  ``(met_em name, upper-level name,
#: surface name)``: the upper levels keep metgrid's own spelling because
#: ArWen already uses it, and the surface level takes WRF's 2 m / 10 m
#: spelling because that is what ``initialize_real`` asks for.
_SPLIT_STACKS = (
    ("TT", "TT", "T2"),
    ("RH", "RH", "RH2"),
    ("GHT", "GHT", None),      # GHT(0) is SOILHGT; carried separately
    ("UU", "UU", "U10"),
    ("VV", "VV", "V10"),
)

#: The specific-humidity lane's extra stack (metgrid ``SPECHUMD`` is
#: ArWen/WRF ``SPFH``; its surface level is ``Q2``), and the 3-D pressure
#: that lane consumes instead of a flat isobaric ladder.
_SPECIFIC_STACKS = (
    ("SPECHUMD", "SPFH", "Q2"),
    ("PRES", "PRES", None),    # PRES(0) is PSFC; carried separately
)

#: WRF's six flagged metgrid mass categories, including supplied hail.
_HYDROMETEOR_STACKS = ("QC", "QR", "QI", "QS", "QG", "QH")

#: 2-D fields read under their own names.  ``PSFC`` and ``SOILHGT`` are
#: essential -- ``initialize_real`` refuses without a source orography
#: for either pressure policy: moisture integrates on the analyzed surface.
_REQUIRED_2D = ("PSFC", "SOILHGT", "HGT_M")

#: 2-D fields carried through when present and named when absent only in
#: the receipt, never as a refusal: ArWen's real initialization does not
#: consume them, but the land-surface and water-temperature stages
#: downstream do.
_OPTIONAL_2D = (
    "SKINTEMP", "SST", "SNOW", "SNOWH", "SEAICE", "XICE", "LANDSEA",
    "LANDMASK", "LU_INDEX", "SOILTEMP", "SCT_DOM", "SCB_DOM", "SNOALB",
    "CON", "VAR", "VAR_SSO", "OA1", "OA2", "OA3", "OA4",
    "OL1", "OL2", "OL3", "OL4",
    "MAPFAC_M", "MAPFAC_U", "MAPFAC_V", "MAPFAC_MX", "MAPFAC_MY",
    "MAPFAC_UX", "MAPFAC_UY", "MAPFAC_VX", "MAPFAC_VY",
    "F", "E", "SINALPHA", "COSALPHA",
    "XLAT_M", "XLONG_M", "XLAT_U", "XLONG_U", "XLAT_V", "XLONG_V",
    "CLAT", "CLONG",
)

#: Category-dimensioned geogrid statics carried through as-is.
_OPTIONAL_3D_STATIC = (
    "LANDUSEF", "SOILCTOP", "SOILCBOT", "GREENFRAC", "ALBEDO12M",
    "LAI12M",
)

#: How far apart ``PRES(k)`` may be across the domain before the level
#: stops being isobaric.  float32 stores 1e5 Pa to about 0.008 Pa, and
#: the RH lane REPLACES the file's own 3-D pressure with a flat ladder
#: broadcast from ``levels_hpa`` (real.py:2694), so a level that is not
#: actually flat would be silently flattened.  1 Pa is ~120x float32
#: rounding at 1e5 Pa and ~5000x tighter than the thinnest standard
#: isobaric spacing (2500 Pa).
ISOBARIC_TOLERANCE_PA = 1.0

#: The lane of a met_em that carries RH and no SPECHUMD on levels that are
#: not isobaric (a model-level source: Arwen Global tapes, ECMWF model
#: levels, any hybrid ungrib table without SPECHUMD).  real.exe takes such
#: a file (rh_to_mxrat1 on the 3-D ``p_gc``, module_initialize_real.F:1113-
#: 1142); the RH lane cannot, because it replaces the 3-D pressure with a
#: flat ladder.  The reader forms the specific humidity real.exe's
#: conversion implies (:func:`woof.ingest.real.relative_humidity_to_specific_humidity`)
#: and hands it to the FLAG_SH lane, which reads ``PRES`` per column.
RH_DERIVED_LANE = "specific_humidity_from_rh"

#: ``PRES(0) == PSFC`` is how the surface pseudo-level is proven.  The
#: two are the same float32 bytes in both staged files (max |difference|
#: measured 0.0 Pa), so this tolerance only absorbs a producer that
#: rounded one of them.
SURFACE_LEVEL_TOLERANCE_PA = 1.0

#: One WPS date, in both spellings metgrid writes: YYYY-MM-DD_HH:MM:SS,
#: and YYYY-MM-DD_HH_MM_SS when namelist.wps sets nocolons = .true.
#: metgrid applies that spelling to the file name AND to the Times
#: record inside it, and real.exe reads either, so the name rule and the
#: Times rule are this one pattern.
_WPS_DATE = (r"(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})_"
             r"(?P<hour>\d{2})[:_](?P<minute>\d{2})[:_](?P<second>\d{2})")
_NAME_RE = re.compile(
    r"^met_em\.(?P<domain>d\d{2})\." + _WPS_DATE + r"(?:\.nc)?$")
_TIMES_RE = re.compile(r"^" + _WPS_DATE + r"$")


def _wps_instant(parts) -> datetime:
    return datetime(
        int(parts["year"]), int(parts["month"]), int(parts["day"]),
        int(parts["hour"]), int(parts["minute"]), int(parts["second"]))


def parse_met_em_times(record: str) -> datetime:
    """A met_em ``Times`` record, colon or nocolons spelling.

    Raises ``ValueError`` for anything else (a ``T`` separator, a space,
    dashes in the clock, an impossible date).
    """

    matched = _TIMES_RE.match(record)
    if matched is None:
        raise ValueError(f"{record!r} is not YYYY-MM-DD_HH:MM:SS or YYYY-MM-DD_HH_MM_SS")
    return _wps_instant(matched.groupdict())


def parse_met_em_name(path: Path | str) -> tuple[str, datetime]:
    """Parse WPS or Windows-safe spelling; the reader also checks Times."""

    name = Path(path).name
    matched = _NAME_RE.match(name)
    if matched is None:
        raise MetgridRefusal(
            f"{name!r} is not a met_em file name.  Expected "
            "met_em.d0N.YYYY-MM-DD_HH:MM:SS.nc (or the same name with "
            "underscores instead of colons).")
    parts = matched.groupdict()
    try:
        instant = _wps_instant(parts)
    except ValueError as error:
        raise MetgridRefusal(f"{name}: invalid filename time: {error}") from None
    return parts["domain"], instant


def met_em_series(root: Path | str, domain: str = "d01") -> tuple[Path, ...]:
    """Every met_em file for DOMAIN under ROOT, in valid-time order.

    ROOT may be the directory or a single file.  A directory holding no
    met_em file for the requested domain is a named refusal that says
    which domains it DOES hold, because "nothing happened" is the one
    outcome a user cannot debug.
    """

    root = Path(root)
    if root.is_file():
        found_domain, _ = parse_met_em_name(root)
        if found_domain != domain:
            raise MetgridRefusal(
                f"{root.name} is domain {found_domain}, not {domain}")
        return (root,)
    if not root.is_dir():
        raise MetgridRefusal(f"no such met_em file or directory: {root}")
    matched: list[tuple[datetime, Path]] = []
    other: set[str] = set()
    for candidate in sorted(root.iterdir()):
        if not candidate.is_file() or not candidate.name.startswith("met_em."):
            continue
        try:
            found_domain, valid_time = parse_met_em_name(candidate)
        except MetgridRefusal:
            continue
        if found_domain == domain:
            matched.append((valid_time, candidate))
        else:
            other.add(found_domain)
    if not matched:
        held = (f"; it does hold {', '.join(sorted(other))}" if other
                else "; it holds no met_em file at all")
        raise MetgridRefusal(
            f"{root} carries no met_em file for domain {domain}{held}")
    matched.sort()
    times = [time for time, _ in matched]
    if len(set(times)) != len(times):
        raise MetgridRefusal(f"{root}: duplicate valid times for {domain}; keep one file per time")
    return tuple(path for _, path in matched)


def _close(name: str, available: Sequence[str]) -> str:
    """The 'did you mean' tail of a missing-variable refusal."""

    near = get_close_matches(name, list(available), n=3, cutoff=0.6)
    if near:
        return f" The file's closest names are {', '.join(near)}."
    return ""


def _variable(dataset, name: str, *, why: str):
    if name not in dataset.variables:
        raise MetgridRefusal(
            f"met_em variable {name!r} is absent, and {why}."
            + _close(name, dataset.variables)
            + f" ({len(dataset.variables)} variables present.)")
    return dataset.variables[name]


def _read(dataset, name: str, *, why: str, want: tuple[int, ...]):
    """One variable, time record 0, checked by NAME and by SHAPE.

    Both refusals name the variable and print what was expected against
    what the file actually holds, because a met_em produced by a
    different WPS build, a different domain, or a hand edit fails
    exactly here and nowhere later where the cause would be lost.
    """

    variable = _variable(dataset, name, why=why)
    dimensions = tuple(variable.dimensions)
    if len(want) >= 2:
        ydim = "south_north_stag" if name in {"VV", "MAPFAC_V", "MAPFAC_VX", "MAPFAC_VY", "XLAT_V", "XLONG_V"} else "south_north"
        xdim = "west_east_stag" if name in {"UU", "MAPFAC_U", "MAPFAC_UX", "MAPFAC_UY", "XLAT_U", "XLONG_U"} else "west_east"
        if dimensions[-2:] != (ydim, xdim):
            raise MetgridRefusal(f"met_em {name}: dimensions {dimensions} must end in {(ydim, xdim)}")
    if len(want) == 3 and name in {"TT", "RH", "GHT", "PRES", "UU", "VV", "SPECHUMD", *_HYDROMETEOR_STACKS, *METGRID_NUMBER_FIELDS}:
        if dimensions != ("Time", "num_metgrid_levels", ydim, xdim):
            raise MetgridRefusal(f"met_em {name}: expected Time/num_metgrid_levels/C-grid dimensions, got {dimensions}")
    scale, offset = _unit_transform(variable, name)
    values = np.asarray(variable.read_transformed(scale=scale, offset=offset, cache=False))
    if values.ndim and values.shape[0] == 1 and len(values.shape) == len(want) + 1:
        values = values[0]
    if values.shape != want:
        raise MetgridRefusal(
            f"met_em variable {name!r} has shape {values.shape} on "
            f"dimensions {variable.dimensions}, but this domain's grid "
            # NOT .capitalize(): every `why` starts with a WRF field
            # name, and capitalize() lowercases the rest of the string --
            # it turned "UU must carry WRF's u staggering" into "Uu must
            # carry wrf's u staggering", which reads as a different
            # variable and a different acronym.
            f"requires {want}.  {why}.")
    if not np.isfinite(values).all():
        count = int((~np.isfinite(values)).sum())
        raise MetgridRefusal(
            f"met_em variable {name!r} carries {count} non-finite "
            f"value(s) out of {values.size}; WOOF-ArWen's real "
            "initialization has no fill-value convention and would "
            "propagate them into the state.")
    with np.errstate(over="ignore", invalid="ignore"):
        converted = np.ascontiguousarray(values, dtype=np.float32)
    if not np.isfinite(converted).all():
        raise MetgridRefusal(f"met_em {name}: finite input exceeds the forecast's float32 representation")
    return converted


def _global(dataset, name: str):
    attributes = dataset.global_attributes
    if name not in attributes:
        raise MetgridRefusal(
            f"met_em global attribute {name!r} is absent."
            + _close(name, attributes)
            + "  A met_em file always carries it; this file was either "
            "not written by metgrid or was stripped after it was.")
    return attributes[name]


def _integer_global(dataset, name):
    value = _global(dataset, name)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = float('nan')
    if not np.isfinite(numeric) or numeric != int(numeric):
        raise MetgridRefusal(f'met_em attribute {name} must be an integer, got {value!r}')
    return int(numeric)


@dataclass(frozen=True)
class MetgridCase:
    """One met_em valid time, as the objects WOOF-ArWen's ingest already takes.

    ``snapshot`` is the forcing object verbatim -- hand it straight to
    :func:`woof.ingest.real.initialize_real` with ``terrain`` and
    ``source_orography`` and nothing else is required.
    """

    path: Path
    domain: str
    valid_time: datetime
    #: The forcing object.  Nothing about it is met_em specific.
    snapshot: HorizontalSnapshot
    #: geogrid HGT_M -- the TARGET terrain the model runs on.
    terrain: np.ndarray
    #: met_em SOILHGT -- the SOURCE analysis terrain that the sfcprs2
    #: surface-pressure adjustment corrects from.
    source_orography: np.ndarray
    geometry: Mapping[str, object]
    landuse: Mapping[str, object]
    statics: Mapping[str, np.ndarray]
    soil: Mapping[str, object]
    #: ``"rh"`` (pressure-level RH, WRF's FLAG_QV-absent lane),
    #: ``"specific_humidity"`` (metgrid SPECHUMD present -- WRF FLAG_SH), or
    #: ``RH_DERIVED_LANE`` (RH on levels that are not isobaric, such as
    #: model levels: the specific humidity real.exe's rh_to_mxrat1 forms
    #: from RH, TT and the 3-D PRES, carried on the FLAG_SH lane).
    lane: str
    #: Which metgrid level was proven to be the surface pseudo-level, or
    #: ``None`` when the file carries no surface anchor.
    surface_level: int | None
    #: met_em name -> what it became, for the receipt and the door.
    field_map: Mapping[str, str]
    #: Per retained level, how far ``PRES`` spans across the domain (Pa).
    level_pressure_spread_pa: np.ndarray
    attributes: Mapping[str, object]
    variable_units: Mapping[str, str]
    notes: tuple[str, ...] = ()

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.terrain.shape)  # type: ignore[return-value]


def read_met_em_terrain(path: Path | str) -> np.ndarray:
    """Only ``HGT_M`` from one met_em file: the model terrain it carries.

    The vertical coordinate a run can use is fixed by the highest ground
    the run can touch, and that has to be known before the first
    coordinate is built -- which on this route means before the first
    met_em is turned into a state.  Reading one two-dimensional variable
    per domain answers it; reading each domain's whole snapshot to find
    out would cost the preparation twice over.
    """

    from woof import netcdf_bridge

    with netcdf_bridge.open_dataset(str(path)) as dataset:
        if "HGT_M" not in dataset.variables:
            raise ValueError(
                f"{Path(path).name}: no HGT_M, so the model terrain this "
                "file describes cannot be read; met_em files are written "
                "with the geogrid terrain the run integrates")
        return np.ascontiguousarray(
            np.asarray(dataset.variables["HGT_M"][0]), dtype=np.float64)


def read_met_em(path: Path | str) -> MetgridCase:
    """One met_em file -> the forcing object WOOF-ArWen's real ingest takes.

    No interpolation and no wind rotation happen here: metgrid already
    put the fields on this domain's C grid and already rotated the winds
    into the projection (its ``COSALPHA``/``SINALPHA``).  This function
    reads, names, splits the surface pseudo-level off the 3-D stacks,
    and refuses.
    """

    path = Path(path)
    domain, valid_time = parse_met_em_name(path)
    with open_dataset(path) as dataset:
        return _read_met_em_dataset(path, domain, valid_time, dataset)


def _read_met_em_dataset(path, domain, valid_time, dataset):
    if "Time" not in dataset.dimensions or len(dataset.dimensions["Time"]) != 1:
        raise MetgridRefusal(f"{path.name}: met_em requires one Time record")
    times = _variable(dataset, "Times", why="each file must identify its valid time")
    if tuple(times.dimensions) != ("Time", "DateStrLen") or tuple(times.shape) != (1, 19):
        raise MetgridRefusal(f"{path.name}: Times must have shape (1, 19) on Time/DateStrLen")
    try:
        record = b"".join(bytes(item) for item in np.asarray(times[...]).reshape(-1)).decode("ascii")
        record_time = parse_met_em_times(record)
    except (ValueError, UnicodeError, TypeError):
        raise MetgridRefusal(
            f"{path.name}: invalid ASCII Times record; metgrid writes "
            "YYYY-MM-DD_HH:MM:SS, or YYYY-MM-DD_HH_MM_SS with nocolons") from None
    if record_time != valid_time:
        raise MetgridRefusal(f"{path.name}: Times {record} disagrees with the filename; restore the correctly dated file")

    we_stag = _integer_global(dataset, "WEST-EAST_GRID_DIMENSION")
    sn_stag = _integer_global(dataset, "SOUTH-NORTH_GRID_DIMENSION")
    nx, ny = we_stag - 1, sn_stag - 1
    if nx < 1 or ny < 1:
        raise MetgridRefusal(
            f"met_em declares a {we_stag}x{sn_stag} staggered grid, which "
            f"is {nx}x{ny} mass points -- not a grid.")
    for dimension, size in (("west_east", nx), ("south_north", ny),
                            ("west_east_stag", nx + 1),
                            ("south_north_stag", ny + 1)):
        if dimension not in dataset.dimensions:
            raise MetgridRefusal(
                f"met_em dimension {dimension!r} is absent; the file's "
                f"dimensions are {sorted(dataset.dimensions)}.")
        held = len(dataset.dimensions[dimension])
        if held != size:
            raise MetgridRefusal(
                f"met_em dimension {dimension!r} is {held}, but the "
                f"file's own WEST-EAST/SOUTH-NORTH_GRID_DIMENSION "
                f"globals ({we_stag}x{sn_stag}) require {size}.  The "
                "file's geometry contradicts itself.")
    if "num_metgrid_levels" not in dataset.dimensions:
        raise MetgridRefusal(
            "met_em dimension 'num_metgrid_levels' is absent, so this "
            "file carries no vertical forcing stack at all; it may be a "
            "geo_em (geogrid static) file rather than a met_em.")
    nlev = len(dataset.dimensions["num_metgrid_levels"])
    if nlev < 3:
        raise MetgridRefusal(
            f"met_em carries {nlev} metgrid level(s); a surface anchor "
            "plus at least two levels above it are needed to interpolate "
            "a column.")

    mass3 = (nlev, ny, nx)
    u3 = (nlev, ny, nx + 1)
    v3 = (nlev, ny + 1, nx)
    mass2 = (ny, nx)

    flag_sh = _field_flag(dataset, "SH")
    if flag_sh == 1 and "SPECHUMD" not in dataset.variables:
        raise MetgridRefusal(f"{path.name}: FLAG_SH=1 but SPECHUMD is absent")
    specific = "SPECHUMD" in dataset.variables and flag_sh == 1
    stacks: dict[str, np.ndarray] = {}
    field_map: dict[str, str] = {}
    for name in ("TT", "GHT", "PRES", *(("RH",) if not specific else ())):
        stacks[name] = _read(
            dataset, name, want=mass3,
            why=f"WOOF-ArWen's real initialization interpolates {name} columns")
    stacks["UU"] = _read(
        dataset, "UU", want=u3,
        why="UU must carry WRF's u staggering (west_east_stag)")
    stacks["VV"] = _read(
        dataset, "VV", want=v3,
        why="VV must carry WRF's v staggering (south_north_stag)")

    two_d: dict[str, np.ndarray] = {}
    for name in _REQUIRED_2D:
        two_d[name] = _read(
            dataset, name, want=mass2,
            why=("PSFC anchors the surface pseudo-level; SOILHGT is the "
                 "source orography the sfcprs2 adjustment corrects from; "
                 "HGT_M is the model terrain"))

    # ---- the surface pseudo-level, proven rather than assumed ---------
    difference = np.abs(stacks["PRES"][0] - two_d["PSFC"])
    surface_gap = float(difference.max())
    first = stacks["PRES"][0]
    isobaric_first = float(first.max() - first.min())
    if surface_gap <= SURFACE_LEVEL_TOLERANCE_PA:
        surface_level = 0
    elif isobaric_first <= ISOBARIC_TOLERANCE_PA:
        surface_level = None
    else:
        raise MetgridRefusal(
            f"met_em level 0 of PRES is neither the surface (it differs "
            f"from PSFC by up to {surface_gap:.4g} Pa) nor an isobaric "
            f"level (it spans {isobaric_first:.4g} Pa across the "
            "domain).  WOOF-ArWen cannot tell what level 0 is, and guessing "
            "would put the wrong field at the bottom of every column.")
    if surface_level is None:
        raise MetgridRefusal(
            "met_em carries no surface pseudo-level: PRES level 0 is an "
            "isobaric level, so metgrid had no surface data to anchor "
            "the columns with.  WOOF-ArWen's real initialization requires "
            "PSFC/T2/RH2/U10/V10 (WRF's use_surface anchor, "
            "module_initialize_real.F), and this file supplies only the "
            "free atmosphere.  Re-run ungrib/metgrid with the source's "
            "surface fields included.")

    keep = slice(surface_level + 1, None)
    fields: dict[str, np.ndarray] = {}
    for met_name, upper, surface in _SPLIT_STACKS:
        if met_name not in stacks:
            continue
        fields[upper] = np.ascontiguousarray(stacks[met_name][keep])
        field_map[f"{met_name}[{surface_level + 1}:]"] = upper
        if surface is not None:
            fields[surface] = np.ascontiguousarray(
                stacks[met_name][surface_level])
            field_map[f"{met_name}[{surface_level}]"] = surface
    fields["PSFC"] = two_d["PSFC"]
    field_map["PSFC"] = "PSFC"
    if _field_flag(dataset, "SLP") == 1:
        fields["PMSL"] = _read(dataset, "PMSL", want=mass2,
            why="FLAG_SLP=1 declares sea-level pressure for sfcprs3")
        field_map["PMSL"] = "PMSL"
    field_map["SOILHGT"] = "source_orography"
    field_map["HGT_M"] = "terrain (initialize_real positional)"

    notes: list[str] = []
    if "RH" in stacks and _unit_transform(dataset.variables["RH"], "RH") != (1.0, 0.0):
        rh_units = str(dataset.variables["RH"].attributes.get("units", ""))
        notes.append(f"RH units {rh_units!r} converted to percent by the Rust reader")
    ght_gap = float(np.abs(stacks["GHT"][surface_level]
                           - two_d["SOILHGT"]).max())
    if ght_gap > 1.0:
        notes.append(
            f"GHT level {surface_level} differs from SOILHGT by up to "
            f"{ght_gap:.4g} m; metgrid normally writes the source "
            "terrain there.  SOILHGT is what WOOF-ArWen uses as the source "
            "orography either way.")

    # ---- which moisture lane this file can drive ---------------------
    retained = stacks["PRES"][keep]
    spread = retained.reshape(retained.shape[0], -1)
    level_spread = np.ascontiguousarray(spread.max(axis=1) - spread.min(axis=1))
    levels_pa = np.ascontiguousarray(spread.mean(axis=1, dtype=np.float64))
    lane = "rh"
    if specific:
        lane = "specific_humidity"
    elif np.any(level_spread > ISOBARIC_TOLERANCE_PA):
        # RH on levels that are not isobaric.  The RH lane would flatten
        # them onto levels_hpa, so the file goes the way real.exe takes
        # it: qv from RH on the 3-D pressure, here as the specific
        # humidity the FLAG_SH lane converts back, on that lane's PRES.
        from woof.ingest.real import relative_humidity_to_specific_humidity
        lane = RH_DERIVED_LANE
        k = int(np.flatnonzero(level_spread > ISOBARIC_TOLERANCE_PA)[0])
        with np.errstate(over="ignore", invalid="ignore"):
            derived = relative_humidity_to_specific_humidity(
                stacks["TT"], stacks["PRES"], stacks["RH"])
        if not np.isfinite(derived).all():
            raise MetgridRefusal(
                f"{path.name}: specific humidity derived from RH, TT and "
                "PRES is non-finite; TT or PRES holds a value no "
                "saturation relation can use")
        stacks["SPECHUMD"] = np.ascontiguousarray(derived, dtype=np.float32)
        del stacks["RH"], fields["RH"], fields["RH2"]
        for key in [key for key, value in field_map.items()
                    if value in ("RH", "RH2")]:
            del field_map[key]
        notes.append(
            f"RH on levels that are not isobaric (level {k + surface_level + 1} "
            f"of PRES spans {float(level_spread[k]):.4g} Pa): specific "
            "humidity formed from RH, TT and PRES by WRF's rh_to_mxrat1 "
            "relation and carried on the specific-humidity lane")
    if lane != "rh":
        for met_name, arwen, surface in _SPECIFIC_STACKS:
            stack = stacks.get(met_name)
            if stack is None:
                stack = _read(
                    dataset, met_name, want=mass3,
                    why=("the specific-humidity lane interpolates "
                         f"{met_name} columns"))
            source = met_name
            if lane == RH_DERIVED_LANE and met_name == "SPECHUMD":
                source = "rh_to_mxrat1(RH, TT, PRES)"
            fields[arwen] = np.ascontiguousarray(stack[keep])
            field_map[f"{source}[{surface_level + 1}:]"] = arwen
            if surface is not None:
                fields[surface] = np.ascontiguousarray(stack[surface_level])
                field_map[f"{source}[{surface_level}]"] = surface
    for name in (*_HYDROMETEOR_STACKS, *METGRID_NUMBER_FIELDS):
        flag = _field_flag(dataset, name)
        if flag == 1 and name not in dataset.variables:
            raise MetgridRefusal(f"{path.name}: FLAG_{name}=1 but {name} is absent")
        if flag == 1:
            stack = _read(dataset, name, want=mass3,
                          why=f"FLAG_{name}=1 declares analyzed {name}")
            if name in METGRID_NUMBER_FIELDS and np.any(stack < 0):
                raise MetgridRefusal(f"{path.name}: analyzed {name} contains negative number concentration")
            fields[name] = np.ascontiguousarray(stack[keep])
            field_map[f"{name}[{surface_level + 1}:]"] = name
            fields[name+"_SFC"] = np.ascontiguousarray(stack[surface_level])
            field_map[f"{name}[{surface_level}]"] = name+"_SFC"
        elif name in dataset.variables:
            notes.append(f"{name} is not flagged as analyzed input; retained WRF absent-field initialization")

    # ---- the level ladder (levels_pa, level_spread: above) ------------
    order = np.argsort(-levels_pa)
    if not np.array_equal(order, np.arange(levels_pa.size)) and not \
            np.array_equal(order, np.arange(levels_pa.size)[::-1]):
        raise MetgridRefusal(
            "met_em pressure levels are neither ascending nor "
            f"descending: {levels_pa.tolist()}.  A shuffled ladder "
            "would silently reorder every column.")
    if np.any(np.diff(np.sort(levels_pa)) <= 0.0):
        raise MetgridRefusal(
            "met_em carries duplicate pressure levels: "
            f"{levels_pa.tolist()}")

    snapshot = HorizontalSnapshot(
        valid_time=valid_time,
        levels_hpa=np.ascontiguousarray(levels_pa / 100.0),
        fields=fields)

    statics: dict[str, np.ndarray] = {}
    for name in _OPTIONAL_2D:
        if name in dataset.variables:
            shape = (ny + int(name.endswith(("_V", "_VX", "_VY"))),
                     nx + int(name.endswith(("_U", "_UX", "_UY"))))
            statics[name] = _read(dataset, name, want=shape, why="static fields share the declared C grid")
            field_map[name] = f"static {name}"
    for name in _OPTIONAL_3D_STATIC:
        if name in dataset.variables:
            shape = tuple(dataset.variables[name].shape)
            if len(shape) != 4 or shape[0] != 1:
                raise MetgridRefusal(f"{path.name}: {name} must have Time/category/y/x dimensions")
            statics[name] = _read(dataset, name, want=(shape[1], ny, nx), why="category fields share the declared C grid")
            field_map[name] = f"static {name}"
    statics["HGT_M"] = two_d["HGT_M"]
    statics["SOILHGT"] = two_d["SOILHGT"]

    soil = _read_soil(dataset, mass2, field_map)

    geometry = {
        "nx": nx, "ny": ny,
        "e_we": we_stag, "e_sn": sn_stag,
        "num_metgrid_levels": nlev,
        "dx": float(_global(dataset, "DX")),
        "dy": float(_global(dataset, "DY")),
        "map_proj": _integer_global(dataset, "MAP_PROJ"),
        "truelat1": float(_global(dataset, "TRUELAT1")),
        "truelat2": float(_global(dataset, "TRUELAT2")),
        "stand_lon": float(_global(dataset, "STAND_LON")),
        "cen_lat": float(_global(dataset, "CEN_LAT")),
        "cen_lon": float(_global(dataset, "CEN_LON")),
        "moad_cen_lat": float(_global(dataset, "MOAD_CEN_LAT")),
        "grid_id": _integer_global(dataset, "grid_id"),
        "title": str(dataset.global_attributes.get("TITLE", "")),
    }
    landuse = {
        "MMINLU": str(_global(dataset, "MMINLU")),
        "NUM_LAND_CAT": _integer_global(dataset, "NUM_LAND_CAT"),
        "ISWATER": _integer_global(dataset, "ISWATER"),
        "ISLAKE": _integer_global(dataset, "ISLAKE"),
        "ISICE": _integer_global(dataset, "ISICE"),
        "ISURBAN": _integer_global(dataset, "ISURBAN"),
        "ISOILWATER": _integer_global(dataset, "ISOILWATER"),
    }

    if geometry["grid_id"] != int(domain[1:]):
        raise MetgridRefusal(f"{path.name}: grid_id={geometry['grid_id']} disagrees with filename domain {domain}")
    geometry["projection_check"] = _check_projection(geometry, statics)
    return MetgridCase(
        path=path, domain=domain, valid_time=valid_time, snapshot=snapshot,
        terrain=np.ascontiguousarray(two_d["HGT_M"], dtype=np.float64),
        source_orography=np.ascontiguousarray(
            two_d["SOILHGT"], dtype=np.float64),
        geometry=MappingProxyType(geometry),
        landuse=MappingProxyType(landuse),
        statics=MappingProxyType(statics),
        soil=MappingProxyType(soil),
        lane=lane, surface_level=surface_level,
        field_map=MappingProxyType(field_map),
        level_pressure_spread_pa=level_spread,
        attributes=MappingProxyType(dict(dataset.global_attributes)),
        variable_units=MappingProxyType({name:str(var.attributes.get("units", ""))
                                        for name,var in dataset.variables.items()}),
        notes=tuple(notes))


def _read_soil(dataset, mass2, field_map) -> dict[str, object]:
    """The soil column, under whichever of metgrid's two spellings it uses.

    metgrid writes soil either as stacked ``ST``/``SM`` on
    ``num_st_layers``/``num_sm_layers`` with a ``SOIL_LAYERS`` depth
    array, or as one 2-D field per layer named for its depth range
    (``ST000007``) or its level (``SOILT001``).  Both are read; neither
    is required, because WOOF-ArWen's real initialization does not consume
    soil -- the land-surface stage after it does.
    """

    soil: dict[str, object] = {}
    for stacked, depth in (("ST", "num_st_layers"), ("SM", "num_sm_layers"),
                           ("SOILT", "num_soilt_levels"),
                           ("SOILM", "num_soilm_levels")):
        if stacked in dataset.variables and depth in dataset.dimensions:
            layers = len(dataset.dimensions[depth])
            soil[stacked] = _read(
                dataset, stacked, want=(layers, *mass2),
                why=f"the {stacked} soil stack rides on {depth}")
            field_map[stacked] = f"soil {stacked} ({layers} layers)"
    for name in ("SOIL_LAYERS", "SOIL_LEVELS"):
        if name in dataset.variables:
            variable = dataset.variables[name]
            shape = tuple(variable.shape)
            if not variable.dimensions:
                raise MetgridRefusal(f"{name} must carry a soil depth dimension, not a scalar")
            wanted = shape[1:] if variable.dimensions[0] == "Time" else shape
            soil[name] = _read(dataset, name, want=wanted, why="soil depths bind the sampled column")
            field_map[name] = f"soil depth axis {name}"
    per_layer = sorted(
        name for name in dataset.variables
        if re.fullmatch(r"(ST|SM)\d{6}", name)
        or re.fullmatch(r"SOIL[TM]\d{3}", name))
    for name in per_layer:
        soil[name] = _read(
            dataset, name, want=mass2,
            why=f"the per-layer soil field {name} is carried through")
        field_map[name] = f"soil layer {name}"
    if "NUM_METGRID_SOIL_LEVELS" in dataset.global_attributes:
        soil["NUM_METGRID_SOIL_LEVELS"] = int(
            dataset.global_attributes["NUM_METGRID_SOIL_LEVELS"])
    return soil


def _unit_transform(variable, name: str) -> tuple[float, float]:
    """Quantity units are metadata, never guessed from a field's range."""
    units = str(variable.attributes.get("units", "")).strip().lower()
    if name in METGRID_NUMBER_FIELDS:
        # WRF's own Registry spelling for these is "# kg(-1)"
        # (Registry/registry.new3d_wif:88), and the string a built WRF
        # actually writes into a file is the same one with the "#"
        # resolved away, "  kg(-1)".  Both are number per kilogram and
        # both reach this reader from a real metgrid WIF stream, so a
        # real met_em is not turned away on a units string.
        allowed = ("# kg-1", "# kg^-1", "#/kg", "kg-1", "1/kg",
                   "# kg(-1)", "kg(-1)")
        if units not in allowed:
            raise MetgridRefusal(f"met_em {name}: units {units!r} do not establish number per kilogram; supply {allowed}")
        return 1.0, 0.0
    catalog = {
        "TT": {"k": (1.0, 0.0), "kelvin": (1.0, 0.0)},
        "RH": {"%": (1.0, 0.0), "percent": (1.0, 0.0),
               "1": (100.0, 0.0), "fraction": (100.0, 0.0)},
        "SPECHUMD": {"kg kg-1": (1.0, 0.0), "kg/kg": (1.0, 0.0), "1": (1.0, 0.0)},
        # WPS synthesizes PRES in Pa and writes an empty units attribute;
        # PSFC has explicit units. Both are checked at the surface anchor.
        "PRES": {"": (1.0, 0.0), "pa": (1.0, 0.0), "hpa": (100.0, 0.0)},
        "PSFC": {"pa": (1.0, 0.0), "hpa": (100.0, 0.0)},
        "PMSL": {"pa": (1.0, 0.0), "hpa": (100.0, 0.0)},
        "GHT": {"m": (1.0, 0.0), "gpm": (1.0, 0.0)},
        "UU": {"m s-1": (1.0, 0.0), "m/s": (1.0, 0.0)},
        "VV": {"m s-1": (1.0, 0.0), "m/s": (1.0, 0.0)},
    }
    if name not in catalog:
        return 1.0, 0.0
    if units not in catalog[name]:
        raise MetgridRefusal(f"met_em {name}: units {units!r} do not establish this quantity; supply correctly labelled {sorted(catalog[name])}")
    return catalog[name][units]


def _check_projection(geometry, statics):
    from woof.static.projection import (WPS_MAP_PROJ_NAMES,
                                         WRF_LATLON_MAP_PROJ, latlon_refusal,
                                         projection_class)
    code = geometry["map_proj"]
    if code == WRF_LATLON_MAP_PROJ and code not in WPS_MAP_PROJ_NAMES:
        raise MetgridRefusal(f"MAP_PROJ={code}: {latlon_refusal()}")
    if code not in WPS_MAP_PROJ_NAMES:
        raise MetgridRefusal(f"MAP_PROJ={code} has no native projected-grid implementation; prepare a Lambert, polar, or Mercator grid")
    try:
        grid = projection_class(WPS_MAP_PROJ_NAMES[code])(
            ref_lat=geometry["cen_lat"], ref_lon=geometry["cen_lon"],
            truelat1=geometry["truelat1"], truelat2=geometry["truelat2"],
            stand_lon=geometry["stand_lon"], dx=geometry["dx"], dy=geometry["dy"],
            e_we=geometry["e_we"], e_sn=geometry["e_sn"],
            moad_cen_lat=geometry["moad_cen_lat"])
    except ValueError as error:
        raise MetgridRefusal(f"met_em projection: {error}") from None
    evidence = {}
    for suffix, coordinates, factor in (
            ("M", grid.latlon_mass, grid.mapfac_m),
            ("U", grid.latlon_u, grid.mapfac_u),
            ("V", grid.latlon_v, grid.mapfac_v)):
        names = (f"XLAT_{suffix}", f"XLONG_{suffix}", f"MAPFAC_{suffix}")
        missing = [name for name in names if name not in statics]
        if missing:
            raise MetgridRefusal(f"met_em lacks grid identity evidence {missing}; retain geogrid coordinates and map factors")
        lat, lon = coordinates()
        differences = (np.max(np.abs(lat-statics[names[0]])),
                       np.max(np.abs((lon-statics[names[1]]+180.0)%360.0-180.0)))
        # WPS publishes float32 coordinates. 0.005 degrees is the existing
        # wrfinput handoff tolerance; map factors have their own tighter check.
        if not np.isfinite(differences).all() or max(differences) > 0.005:
            raise MetgridRefusal(f"met_em projection attributes disagree with {names[:2]}: max differences {differences} degrees")
        mf = factor()
        if not np.allclose(mf, statics[names[2]], rtol=2e-5, atol=2e-6):
            raise MetgridRefusal(f"met_em projection attributes disagree with {names[2]}")
        evidence[suffix] = [float(item) for item in differences]
    return evidence


_SERIES_STATIC_FIELDS = ("HGT_M", "LANDMASK", "LU_INDEX", "SCT_DOM", "SCB_DOM",
    "XLAT_M", "XLONG_M", "XLAT_U", "XLONG_U", "XLAT_V", "XLONG_V",
    "MAPFAC_M", "MAPFAC_U", "MAPFAC_V", "F", "E", "SINALPHA", "COSALPHA")


def met_em_series_identity(case):
    """Keep only small identity evidence between streamed forcing times."""
    return replace(case, snapshot=replace(case.snapshot, fields={}), soil={},
        terrain=np.empty(0), source_orography=np.empty(0),
        statics={name:np.frombuffer(hashlib.sha256(np.ascontiguousarray(case.statics[name]).tobytes()).digest(),dtype=np.uint8)
                 for name in _SERIES_STATIC_FIELDS if name in case.statics})


def met_em_source_top_pressure_pa(path: Path | str) -> float:
    """The smallest analyzed pressure in one met_em file, in pascals.

    Read at plan review for exactly one purpose: a requested model top
    above the source atmosphere is a ladder the source cannot support,
    and :func:`woof.vertical_contract.validate_explicit_eta_grid`
    refuses it by name given this number.  Only ``PRES`` is decoded, and
    only from the first root file, so the answer costs one variable and
    is available before a GPU is selected or an output directory made.
    """
    path = Path(path)
    with open_dataset(path) as dataset:
        variable = _variable(dataset, "PRES",
            why="the source atmosphere's top pressure bounds the model top")
        scale, offset = _unit_transform(variable, "PRES")
        values = np.asarray(
            variable.read_transformed(scale=scale, offset=offset, cache=False),
            dtype=np.float64)
    finite = values[np.isfinite(values) & (values > 0.0)]
    if finite.size == 0:
        raise MetgridRefusal(
            f"{path.name}: PRES carries no finite positive pressure, so this "
            "file declares no source atmosphere and the model top has "
            "nothing to be checked against. Re-run metgrid for this valid "
            "time so PRES holds the analyzed pressures, or point --met-em at "
            "a directory whose files carry them")
    return float(finite.min())


def check_met_em_series(cases: Sequence[MetgridCase], *, interval_seconds: float,
                        start_time: datetime, end_time: datetime) -> None:
    """Keep a forcing series on one grid and its declared time/level axes."""
    if len(cases) < 2:
        raise MetgridRefusal("met_em forcing needs the initial file and a later boundary file")
    if not np.isfinite(interval_seconds) or interval_seconds <= 0:
        raise MetgridRefusal("met_em interval_seconds must be finite and positive")
    first = cases[0]
    if first.valid_time != start_time or cases[-1].valid_time < end_time:
        raise MetgridRefusal(f"met_em files cover {first.valid_time} to {cases[-1].valid_time}, requested {start_time} to {end_time}; supply the missing times")
    invariant_fields = _SERIES_STATIC_FIELDS
    geometry_keys = set(first.geometry) - {"title", "projection_check"}
    for previous, item in zip(cases, cases[1:]):
        gap = (item.valid_time-previous.valid_time).total_seconds()
        if gap != interval_seconds:
            raise MetgridRefusal(f"{item.path.name}: forcing gap is {gap:g}s, expected {interval_seconds:g}s; restore the missing or correctly dated file")
        if item.domain != first.domain or item.lane != first.lane:
            raise MetgridRefusal(f"{item.path.name}: domain or humidity representation changed within the series")
        if dict(item.landuse) != dict(first.landuse) or any(item.geometry.get(key) != first.geometry[key] for key in geometry_keys):
            raise MetgridRefusal(f"{item.path.name}: grid or land-use identity changed within the series")
        for name in invariant_fields:
            left, right = first.statics.get(name), item.statics.get(name)
            if (left is None) != (right is None) or (left is not None and not np.array_equal(left, right)):
                raise MetgridRefusal(f"{item.path.name}: static coordinate field {name} changed within the series")
        if item.snapshot.levels_hpa.shape != first.snapshot.levels_hpa.shape:
            raise MetgridRefusal(f"{item.path.name}: vertical forcing level count changed")
        # Only the RH lane flattens PRES onto levels_hpa; RH_DERIVED_LANE,
        # like the specific-humidity lane, reads PRES per column.
        if first.lane == "rh" and not np.allclose(item.snapshot.levels_hpa, first.snapshot.levels_hpa, rtol=0.0, atol=0.01):
            raise MetgridRefusal(f"{item.path.name}: isobaric forcing levels changed within the series")
        # Specific-humidity PRES is a physical, time-varying 3-D coordinate;
        # equality here would wrongly forbid ordinary evolving model levels.


def _field_flag(dataset, name):
    value = dataset.global_attributes.get(f"FLAG_{name}")
    if value is None:
        return None
    try:
        number = float(value)
    except (ValueError, TypeError):
        number = float("nan")
    if number not in (0.0, 1.0):
        raise MetgridRefusal(f"FLAG_{name} must be 0 or 1, got {value!r}")
    return int(number)
