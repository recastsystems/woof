"""The air above a source's published top, from a declared donor analysis.

The defect it fixes: ECCC's HRDPS publishes its isobaric column up to 50 hPa,
while the RRFS v2 vertical grid hrrrv5 initialises on (L60) tops at 25.9 km,
about 22 hPa.  The RRFS producer's case 7 (lapse-rate mode, the fork's own
recipe) refuses to extrapolate temperature above the first-guess column top
("temperature extrapolation above the first-guess column top is not
implemented for the lapse-rate mode", the Fortran's own fatal), so no hrrrv5
start or boundary could be made from HRDPS.

The fix is a TABLE declaration, the same shape as the deep-soil donor
(:mod:`woof.ingest.deep_soil_donor`): a composition may declare an
``upper_air_donor``, an analysis on pressure levels that supplies the column
above the source's own top.  The source keeps every level it publishes below
the declared blend layer; inside the layer (from ``blend.bottom_pa`` up to the
source's top level) each source level moves linearly in log-pressure toward
the donor, reaching the donor's value at the source's top, so the column has
no seam there; above it the donor's levels are appended.  Every 3-D field the
file carries at its top level must have a donor field, or the file is
refused by name.

It works on the written WPS v5 file (the box a limited-area run initialises
and is driven from, or a global file), so every consumer -- MPAS-A's case 7
and 9, the RRFS producer's -- reads one deeper column through its existing
code.  Donor cycle: the latest at or before the source's own cycle whose lead
reaches the valid time, then the cycle before it (the deep-soil rule).
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import tempfile
from datetime import datetime
from typing import Mapping

import numpy as np

UPPER_AIR_DONOR_KEY = "upper_air_donor"

_DONOR_KEYS = {
    "about", "source_id", "licence", "transport", "cycle_hours", "max_lead_hours",
    "levels_pa", "fields", "blend", "horizontal",
}
_TRANSPORT_KEYS = {"url", "index_suffix", "valid_from"}
_FIELD_KEYS = {"index", "discipline", "category", "parameter", "level_pressure", "scale"}
_BLEND_KINDS = {"log_pressure_linear"}
_HORIZONTAL_KINDS = {"bilinear"}
#: WPS ``xlvl`` codes that are not isobaric levels: the surface and mean sea level.
_NOT_ISOBARIC = (200100.0, 201300.0)
#: An isobaric WPS level is a pressure in Pa below this.
_ISOBARIC_MAX_PA = 110000.0


def _positive_int(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def validate_upper_air_donor(value: object) -> dict:
    """Validate a composition's ``upper_air_donor`` block."""

    if not isinstance(value, Mapping):
        raise TypeError("composition.upper_air_donor must be an object")
    unknown = sorted(set(value) - _DONOR_KEYS)
    if unknown:
        raise ValueError(f"composition.upper_air_donor has unknown key(s): {unknown}")
    missing = sorted(_DONOR_KEYS - {"about", "licence"} - set(value))
    if missing:
        raise ValueError(f"composition.upper_air_donor is missing key(s): {missing}")
    donor = copy.deepcopy(dict(value))
    if not isinstance(donor["source_id"], str) or not donor["source_id"].strip():
        raise ValueError("upper_air_donor.source_id must be a non-empty string")
    transport = donor["transport"]
    if not isinstance(transport, Mapping) or set(transport) - _TRANSPORT_KEYS \
            or {"url", "index_suffix"} - set(transport):
        raise ValueError(f"upper_air_donor.transport takes {sorted(_TRANSPORT_KEYS)}")
    if "{cycle" not in str(transport["url"]) or "{lead" not in str(transport["url"]):
        raise ValueError("upper_air_donor.transport.url must name {cycle:...} and {lead...}")
    _positive_int(donor["cycle_hours"], "upper_air_donor.cycle_hours")
    _positive_int(donor["max_lead_hours"], "upper_air_donor.max_lead_hours")
    levels = donor["levels_pa"]
    if (not isinstance(levels, list) or not levels
            or any(isinstance(p, bool) or not isinstance(p, (int, float)) or not 0 < p < _ISOBARIC_MAX_PA
                   for p in levels)
            or len(set(float(p) for p in levels)) != len(levels)):
        raise ValueError("upper_air_donor.levels_pa is a list of distinct isobaric levels in Pa")
    fields = donor["fields"]
    if not isinstance(fields, Mapping) or not fields:
        raise ValueError("upper_air_donor.fields maps WPS field names to donor records")
    for name, spec in fields.items():
        if not isinstance(spec, Mapping) or set(spec) - _FIELD_KEYS:
            raise ValueError(f"upper_air_donor.fields.{name} takes {sorted(_FIELD_KEYS)}")
        if spec.get("level_pressure") is True:
            if set(spec) != {"level_pressure"}:
                raise ValueError(f"upper_air_donor.fields.{name}: a level-pressure field reads no record")
            continue
        if {"index", "discipline", "category", "parameter"} - set(spec):
            raise ValueError(f"upper_air_donor.fields.{name} needs index, discipline, category and parameter")
        scale = spec.get("scale", 1.0)
        if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not scale > 0:
            raise ValueError(f"upper_air_donor.fields.{name}.scale must be a positive number (a units factor, "
                             "e.g. 9.80665 for geopotential from a height)")
    blend = donor["blend"]
    if not isinstance(blend, Mapping) or blend.get("kind") not in _BLEND_KINDS \
            or set(blend) - {"kind", "bottom_pa", "about"}:
        raise ValueError(f"upper_air_donor.blend.kind must be one of {sorted(_BLEND_KINDS)}, with bottom_pa")
    bottom = blend.get("bottom_pa")
    if isinstance(bottom, bool) or not isinstance(bottom, (int, float)) or not 0 < bottom < _ISOBARIC_MAX_PA:
        raise ValueError("upper_air_donor.blend.bottom_pa must be an isobaric level in Pa")
    horizontal = donor["horizontal"]
    if not isinstance(horizontal, Mapping) or horizontal.get("kind") not in _HORIZONTAL_KINDS:
        raise ValueError(f"upper_air_donor.horizontal.kind must be one of {sorted(_HORIZONTAL_KINDS)}")
    return donor


def has_upper_air_donor(composition: Mapping[str, object] | None) -> bool:
    return isinstance(composition, Mapping) and composition.get(UPPER_AIR_DONOR_KEY) is not None


# ---------------------------------------------------------------------------
# The WPS file.
# ---------------------------------------------------------------------------

def _groups(payload: bytes):
    """``[(field, xlvl, [5 record bodies])]`` of a WPS v5 file, in file order."""

    from woof.ingest.deep_soil_donor import _wps_records

    records = _wps_records(payload)
    if len(records) % 5:
        raise ValueError("not whole WPS v5 field groups")
    out = []
    for index in range(0, len(records), 5):
        header = records[index + 1]
        field = header[60:69].decode("ascii").strip()
        (level,) = struct.unpack(">f", header[140:144])
        out.append((field, float(level), list(records[index:index + 5])))
    return out


def _isobaric(level: float) -> bool:
    return 0.0 < level < _ISOBARIC_MAX_PA and level not in _NOT_ISOBARIC


def _level_name(pa: float) -> str:
    return f"{pa / 100.0:g} mb"


def _box(group) -> tuple[np.ndarray, np.ndarray, int, int]:
    nx, ny, iproj = struct.unpack(">iii", group[1][144:156])
    if iproj != 0:
        raise ValueError(f"the upper-air donor writes onto a regular lat-lon WPS grid (iproj 0); "
                         f"this file declares iproj {iproj}")
    startlat, startlon, deltalat, deltalon = struct.unpack(">ffff", group[2][8:24])
    latitude = np.repeat((startlat + deltalat * np.arange(ny, dtype=np.float64))[:, None], nx, axis=1)
    longitude = np.repeat((startlon + deltalon * np.arange(nx, dtype=np.float64))[None, :], ny, axis=0)
    return latitude, longitude, nx, ny


def plan_levels(source_levels, donor: Mapping[str, object]) -> dict:
    """Which levels the donor appends and which source levels it blends, with their weights.

    ``source_levels`` are the file's isobaric levels in Pa.  The weight is 0 at
    ``blend.bottom_pa`` and 1 at the source's top, linear in log-pressure.
    """

    levels = sorted({float(p) for p in source_levels}, reverse=True)
    if not levels:
        raise ValueError("the file carries no isobaric level to put a donor above")
    top = levels[-1]
    bottom = float(donor["blend"]["bottom_pa"])
    append = sorted((float(p) for p in donor["levels_pa"] if float(p) < top), reverse=True)
    if not append:
        return {"source_top_pa": top, "append_pa": [], "blend": []}
    if bottom <= top:
        raise ValueError(f"upper_air_donor.blend.bottom_pa {bottom:g} Pa is not below the source's top "
                         f"({top:g} Pa); the blend layer must hold the source's top")
    blend = [(p, math.log(bottom / p) / math.log(bottom / top)) for p in levels if top <= p < bottom]
    return {"source_top_pa": top, "append_pa": append, "blend": blend}


def _fetch(donor, cycle, lead: int, root: Path, records) -> tuple[Path, dict]:
    from woof.ingest.deep_soil_donor import _fetch as fetch_ranges

    digest = hashlib.sha256(json.dumps(sorted(records)).encode()).hexdigest()[:12]
    ranged = {"source_id": donor["source_id"],
              "transport": {"url": donor["transport"]["url"],
                            "index_suffix": donor["transport"]["index_suffix"],
                            "records": [list(row) for row in records]}}
    return fetch_ranges(ranged, cycle, lead, root, filename=f"upper-air-{digest}.grib2")


def _decode(donor, path: Path, wanted: Mapping[str, list[float]]):
    """``{field: {level_pa: (ny, nx) plane}}`` plus the donor's latitude and longitude axes."""

    from woof.grib2_stack import stack_grib2

    # One decode per donor record (two WPS names may read one record, e.g. GHT and GEOPT from HGT).
    def selector(name):
        spec = donor["fields"][name]
        return f"{int(spec['discipline'])}-{int(spec['category'])}-{int(spec['parameter'])}"
    union: dict[str, set] = {}
    for name, levels in wanted.items():
        union.setdefault(selector(name), set()).update(float(p) for p in levels)
    rows = []
    for key, levels in union.items():
        discipline, category, parameter = (int(part) for part in key.split("-"))
        rows.append(dict(key=key, discipline=discipline, category=category, parameter=parameter,
                         constituent_type=None, aerosol_type=None, level_type=100, levels=sorted(levels)))
    scratch = Path(tempfile.mkdtemp(prefix=".upper-air-decode-", dir=path.parent))
    try:
        stacked = {field.key: field for field in stack_grib2(path, rows, scratch / "stack")}
        out = {}
        latitude = longitude = None
        for name, levels in wanted.items():
            field = stacked[selector(name)]
            if field.projection is not None:
                raise ValueError("upper-air donor: only a regular latitude/longitude donor grid is declared")
            array = np.array(field.array, dtype=np.float64)
            held = np.asarray(field.levels, dtype=np.float64)
            planes = {}
            for level in levels:
                hit = np.flatnonzero(np.abs(held - float(level)) < 1e-3)
                if hit.size != 1:
                    raise ValueError(f"upper-air donor: {name} has no single level at {level:g} Pa "
                                     f"(levels {held.tolist()})")
                planes[float(level)] = array[hit[0]]
            out[name] = planes
            latitude = np.asarray(field.latitude, dtype=np.float64)
            longitude = np.asarray(field.longitude, dtype=np.float64)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return out, latitude, longitude


def _bilinear(plane: np.ndarray, lat_axis, lon_axis, latitude, longitude) -> np.ndarray:
    """``plane`` on a regular lat-lon donor grid, bilinear at the target points (periodic in longitude)."""

    ny, nx = plane.shape
    dlat = float(lat_axis[1] - lat_axis[0])
    dlon = float(lon_axis[1] - lon_axis[0])
    periodic = abs(nx * dlon - 360.0) < 1e-6
    values = np.concatenate([plane, plane[:, :1]], axis=1) if periodic else plane
    y = (np.asarray(latitude, dtype=np.float64) - float(lat_axis[0])) / dlat
    x = np.mod(np.asarray(longitude, dtype=np.float64) - float(lon_axis[0]), 360.0) / dlon
    if (y < 0).any() or (y > ny - 1).any() or (not periodic and ((x < 0).any() or (x > nx - 1).any())):
        raise ValueError("upper-air donor: the target points reach outside the donor grid")
    y0 = np.clip(np.floor(y).astype(np.int64), 0, ny - 2)
    x0 = np.clip(np.floor(x).astype(np.int64), 0, values.shape[1] - 2)
    wy = y - y0
    wx = x - x0
    out = (values[y0, x0] * (1 - wy) * (1 - wx) + values[y0, x0 + 1] * (1 - wy) * wx
           + values[y0 + 1, x0] * wy * (1 - wx) + values[y0 + 1, x0 + 1] * wy * wx)
    if not np.isfinite(out).all():
        raise ValueError("upper-air donor: a non-finite donor value reached the target points")
    return out


def append_upper_air_to_wps(path, donor: Mapping[str, object], *, valid_time: datetime,
                            primary_cycle: datetime | None = None, cache_dir=None) -> dict:
    """Blend the source's top layer toward the donor and append the donor's levels above it.

    Returns the receipt; ``appended_levels`` is 0 (and nothing is fetched) when the file
    already reaches every declared donor level.
    """

    from woof.ingest.deep_soil_donor import _utc_naive, _wps_record, donor_candidates

    path = Path(path)
    payload = path.read_bytes()
    groups = _groups(payload)
    by_key = {(field, level): body for field, level, body in groups}
    temperature_levels = [level for field, level, _ in groups if field == "TT" and _isobaric(level)]
    plan = plan_levels(temperature_levels, donor)
    top = plan["source_top_pa"]
    receipt = {"schema": "gpuwm-upper-air-donor-v1", "source_id": donor["source_id"],
               "source_top_pa": top, "appended_levels_pa": plan["append_pa"],
               "blended_levels": [{"level_pa": p, "donor_weight": round(w, 6)} for p, w in plan["blend"]],
               "valid_time": f"{_utc_naive(valid_time):%Y-%m-%dT%H:%M:%S}Z"}
    if not plan["append_pa"]:
        receipt.update(appended_levels=0, fields=[], note="the source already reaches every declared donor level")
        return receipt
    fields = [field for field, level, _ in groups if level == top and _isobaric(level)]
    undeclared = sorted(set(fields) - set(donor["fields"]))
    if undeclared:
        raise ValueError(f"{path}: the source carries {undeclared} at its top ({top:g} Pa) and the upper-air "
                         f"donor declares no field for them; the column above would be missing them")
    read = {name: sorted({*plan["append_pa"], *(p for p, _ in plan["blend"])})
            for name in fields if not donor["fields"][name].get("level_pressure")}
    records = sorted({(donor["fields"][name]["index"], _level_name(p))
                      for name, levels in read.items() for p in levels})
    root = Path(cache_dir) if cache_dir is not None else path.parent / ".upper-air"
    errors = []
    for cycle, lead in donor_candidates(donor, valid_time, primary_cycle):
        try:
            grib, fetched = _fetch(donor, cycle, lead, root, records)
            planes, lat_axis, lon_axis = _decode(donor, grib, read)
        except Exception as error:  # the next candidate cycle, then a refusal
            errors.append(f"{cycle:%Y-%m-%d %H}Z f{lead:03d}: {type(error).__name__}: {error}")
            continue
        break
    else:
        raise RuntimeError(f"upper-air donor {donor['source_id']} unavailable for "
                           f"{_utc_naive(valid_time):%Y-%m-%d %H}Z: " + "; ".join(errors))
    latitude, longitude, nx, ny = _box(by_key[("TT", top)])

    def donor_plane(name: str, level: float) -> np.ndarray:
        spec = donor["fields"][name]
        if spec.get("level_pressure"):
            return np.full((ny, nx), float(level))
        return float(spec.get("scale", 1.0)) * _bilinear(planes[name][level], lat_axis, lon_axis, latitude, longitude)

    weights = dict(plan["blend"])
    out = bytearray()
    blended = 0
    for field, level, body in groups:
        if field in fields and level in weights and not donor["fields"][field].get("level_pressure"):
            source = np.frombuffer(body[4], ">f4").reshape(ny, nx).astype(np.float64)
            w = weights[level]
            mixed = (1.0 - w) * source + w * donor_plane(field, level)
            body = [*body[:4], mixed.astype(">f4").tobytes()]
            blended += 1
        for record in body:
            out += _wps_record(record)
    appended = []
    for field in fields:
        template = by_key[(field, top)]
        for level in plan["append_pa"]:
            header = bytearray(template[1])
            header[140:144] = struct.pack(">f", float(level))
            text = f"{field} {level / 100.0:g} hPa, {donor['source_id']}"
            header[94:140] = text[:46].ljust(46).encode("ascii")
            data = donor_plane(field, level).astype(">f4").tobytes()
            for record in (template[0], bytes(header), template[2], template[3], data):
                out += _wps_record(record)
            appended.append(f"{field}@{level:g}")
    staging = path.with_name(path.name + ".upper-air.part")
    staging.write_bytes(bytes(out))
    os.replace(staging, path)
    receipt.update({
        "cycle": f"{cycle:%Y-%m-%dT%H}Z", "lead_hours": lead,
        "grib": str(grib), "grib_sha256": hashlib.sha256(grib.read_bytes()).hexdigest(),
        "fetch": fetched, "skipped_candidates": errors,
        "fields": fields, "appended_levels": len(plan["append_pa"]), "appended_records": len(appended),
        "blended_records": blended, "blend": dict(donor["blend"]),
        "top_mean_temperature_K": {f"{level:g}": round(float(donor_plane("TT", level).mean()), 2)
                                   for level in plan["append_pa"][:1] + plan["append_pa"][-1:]},
    })
    return receipt


__all__ = [
    "UPPER_AIR_DONOR_KEY", "append_upper_air_to_wps", "has_upper_air_donor", "plan_levels",
    "validate_upper_air_donor",
]
