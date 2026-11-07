"""The soil below a source's published layers, from a declared donor analysis.

The defect it fixes: ECCC's GEM sources (HRDPS 2.5 km, RDPS 10 km, GDPS)
publish ONE soil layer, 0-10 cm.  The RUC land surface (hrrrv5, the RRFS v2
suites) cannot start from one slab -- ``init_soil_3_real`` builds its 0 m
moisture anchor off the top TWO layers, and one 10 cm slab does not hold the
3 m column RUC integrates -- so every RUC suite was refused from every
Canadian source, on WOOF and on hex.

The fix is a TABLE declaration, not a per-source path (the arbitrary
acceptance test): a composition's ``soil_layers`` may declare a
``deep_donor``, an analysis that supplies the layers below the source's own.
The source keeps its own published layer(s); the donor's layers under them
are sampled at the same valid time onto the target points (land-only
bilinear, then the nearest donor land within a declared radius) and stacked
below.  The stacked column is then an ordinary layer contract, so every
consumer -- WOOF's Noah and RUC ingest, the WPS box export hex initialises
from -- runs its existing arm on it.  The donor layers carry their own soil
model (``noah`` for GFS), so the soil-moisture index remap
(:mod:`woof.ingest.land_state`) moves THEIR moisture onto the target land
model's hydraulics by relative wetness, while the source's own layer is
kept as published.

Which donor cycle: the latest donor cycle at or before the valid time whose
lead reaches it, then the cycle before it (a live start can come before the
donor posts its newest cycle).  The receipt names the cycle and lead used.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import copy
import hashlib
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Mapping

import numpy as np

DEEP_DONOR_KEY = "deep_donor"
#: The key the effective (stacked) contract carries so the soil-moisture
#: index remap knows which layers are the donor's and whose soil model they
#: are on.  It never appears in a packaged composition.
DONOR_LAYERS_KEY = "deep_donor_layers"

_DONOR_KEYS = {
    "about", "source_id", "soil_model", "licence", "layers", "transport",
    "land_selector", "cycle_hours", "max_lead_hours", "horizontal",
}
_TRANSPORT_KEYS = {"url", "index_suffix", "records", "valid_from"}
_HORIZONTAL_KINDS = {"land_bilinear_then_nearest"}


@dataclass(frozen=True)
class DeepSoil:
    temperature: np.ndarray   # (layers, *shape) K
    moisture: np.ndarray      # (layers, *shape) volumetric fraction
    found: np.ndarray         # shape: a donor land column reached the point
    receipt: dict


def _string(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def validate_deep_donor(value: object, primary_bounds) -> dict:
    """Validate a ``soil_layers.deep_donor`` block against the primary layers.

    The donor's layers must start exactly at the primary's deepest bottom and
    be contiguous: a gap would leave depths no layer describes, an overlap
    would give one depth two sources.
    """

    from woof.ingest.soil_contract import _layer_bounds, _selector

    if not isinstance(value, Mapping):
        raise TypeError("composition.soil_layers.deep_donor must be an object")
    unknown = sorted(set(value) - _DONOR_KEYS)
    if unknown:
        raise ValueError(f"composition.soil_layers.deep_donor has unknown key(s): {unknown}")
    missing = sorted({"source_id", "soil_model", "layers", "transport",
                      "land_selector", "cycle_hours", "max_lead_hours",
                      "horizontal"} - set(value))
    if missing:
        raise ValueError(f"composition.soil_layers.deep_donor is missing key(s): {missing}")
    donor = copy.deepcopy(dict(value))
    _string(donor["source_id"], "deep_donor.source_id")
    from woof.ingest.land_state import load_rules
    if donor["soil_model"] not in load_rules()["soil_models"]:
        raise ValueError(
            f"deep_donor.soil_model {donor['soil_model']!r} has no row in "
            "woof/data/soil/land-state.v1.json, so its moisture could not be "
            "moved onto the target land model's hydraulics")
    bounds = _layer_bounds(donor["layers"], "deep_donor.layers", bind_selectors=True)
    if not primary_bounds:
        raise ValueError("a deep_donor needs the source's own layers above it")
    if abs(bounds[0][0] - primary_bounds[-1][1]) > 1e-9:
        raise ValueError(
            f"deep_donor.layers begin at {bounds[0][0]} m but the source's own "
            f"layers end at {primary_bounds[-1][1]} m; the donor supplies the "
            "column BELOW the published layers, contiguously")
    transport = donor["transport"]
    if not isinstance(transport, Mapping) or set(transport) - _TRANSPORT_KEYS \
            or {"url", "index_suffix", "records"} - set(transport):
        raise ValueError(f"deep_donor.transport takes {sorted(_TRANSPORT_KEYS)}")
    url = _string(transport["url"], "deep_donor.transport.url")
    if "{cycle" not in url or "{lead" not in url:
        raise ValueError("deep_donor.transport.url must name {cycle:...} and {lead...}")
    records = transport["records"]
    if not isinstance(records, list) or not records or any(
            not isinstance(row, list) or len(row) != 2 for row in records):
        raise ValueError("deep_donor.transport.records is a list of [variable, level] index pairs")
    _selector(donor["land_selector"], "deep_donor.land_selector")
    for key in ("cycle_hours", "max_lead_hours"):
        number = donor[key]
        if isinstance(number, bool) or not isinstance(number, int) or number < 1:
            raise ValueError(f"deep_donor.{key} must be a positive integer")
    horizontal = donor["horizontal"]
    if not isinstance(horizontal, Mapping) or horizontal.get("kind") not in _HORIZONTAL_KINDS:
        raise ValueError(f"deep_donor.horizontal.kind must be one of {sorted(_HORIZONTAL_KINDS)}")
    radius = horizontal.get("search_radius_cells")
    if isinstance(radius, bool) or not isinstance(radius, int) or not 0 <= radius <= 8:
        raise ValueError("deep_donor.horizontal.search_radius_cells must be an integer in [0, 8]")
    return donor


def has_deep_donor(contract: Mapping[str, object] | None) -> bool:
    return isinstance(contract, Mapping) and contract.get(DEEP_DONOR_KEY) is not None


def effective_soil_layer_contract(contract: Mapping[str, object]) -> dict:
    """The stacked contract: the source's layers, then the donor's.

    The result declares no ``deep_donor`` (its layers are in
    ``source_layers``) and carries :data:`DONOR_LAYERS_KEY` naming where the
    donor's layers begin and whose soil model they are on.
    """

    if not has_deep_donor(contract):
        return dict(contract)
    effective = copy.deepcopy(dict(contract))
    donor = effective.pop(DEEP_DONOR_KEY)
    primary = list(effective["source_layers"])
    effective["source_layers"] = primary + [
        {"top": layer["top"], "bottom": layer["bottom"],
         "selectors": layer["selectors"]} for layer in donor["layers"]]
    effective[DONOR_LAYERS_KEY] = {
        "first_index": len(primary), "source_id": donor["source_id"],
        "soil_model": donor["soil_model"],
    }
    return effective


def primary_soil_layer_contract(contract: Mapping[str, object]) -> dict:
    """The source's own contract, the donor dropped (the one-layer arm)."""

    primary = copy.deepcopy(dict(contract))
    primary.pop(DEEP_DONOR_KEY, None)
    return primary


# ---------------------------------------------------------------------------
# Acquisition and decode.
# ---------------------------------------------------------------------------

def _cache_root(cache_dir) -> Path:
    if cache_dir is not None:
        return Path(cache_dir)
    explicit = os.environ.get("WOOF_LAND_STATE_CACHE")
    if explicit:
        return Path(explicit) / "deep-soil"
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "woof" / "deep-soil"


def _utc_naive(value: datetime) -> datetime:
    if value.tzinfo is not None:
        value = value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


def donor_candidates(donor: Mapping[str, object], valid_time: datetime,
                     primary_cycle: datetime | None = None):
    """``(cycle, lead)`` pairs to try, newest cycle first.

    The donor cycle is the latest at or before the primary's own cycle (the
    valid time when the caller has no cycle), so a later lead of a live run
    never asks for a donor cycle that has not run yet; the cycle before it
    is the fallback for a donor that posts later than the source.
    """

    valid = _utc_naive(valid_time)
    if valid.minute or valid.second or valid.microsecond:
        raise ValueError(f"deep soil donor: valid time {valid:%H:%M:%S}Z is not on the hour")
    step = int(donor["cycle_hours"])
    base = valid if primary_cycle is None else min(valid, _utc_naive(primary_cycle))
    newest = base.replace(hour=(base.hour // step) * step, minute=0, second=0, microsecond=0)
    out = []
    for back in range(2):
        cycle = newest - timedelta(hours=step * back)
        lead = int((valid - cycle).total_seconds() // 3600)
        if lead <= int(donor["max_lead_hours"]):
            out.append((cycle, lead))
    floor = donor["transport"].get("valid_from")
    if floor:
        start = _utc_naive(datetime.fromisoformat(str(floor).replace("Z", "+00:00")))
        out = [pair for pair in out if pair[0] >= start]
        if not out:
            raise ValueError(f"deep soil donor {donor['source_id']}: {valid:%Y-%m-%d %H}Z "
                             f"predates its archive ({floor})")
    return out


def _fetch(donor, cycle: datetime, lead: int, root: Path, *,
           filename: str = "deep-soil.grib2") -> tuple[Path, dict]:
    """The donor's declared records at one cycle and lead, by index byte ranges.

    ``filename`` keeps two donors' record sets apart in one cache (the
    upper-air donor reads other records from the same GFS object)."""

    from tools import download_hrrr_native_subset as transport

    url = str(donor["transport"]["url"]).format(cycle=cycle, lead=lead)
    folder = root / str(donor["source_id"]) / f"{cycle:%Y%m%d%H}" / f"f{lead:03d}"
    destination = folder / filename
    if destination.is_file():
        with destination.open("rb") as stream:
            head = stream.read(4)
            stream.seek(-4, os.SEEK_END)
            tail = stream.read(4)
        if head == b"GRIB" and tail == b"7777":
            return destination, {"url": url, "cached": True}
        destination.unlink()
    folder.mkdir(parents=True, exist_ok=True)
    headers = transport._head(url)
    size = int(headers["content-length"])
    index_payload, _ = transport._request_bytes(url + str(donor["transport"]["index_suffix"]))
    rows = transport._parse_index(index_payload, size)
    wanted = {(str(variable), str(level)) for variable, level in donor["transport"]["records"]}
    selected = tuple(i for i, row in enumerate(rows) if (row.variable, row.level) in wanted)
    found = {(rows[i].variable, rows[i].level) for i in selected}
    if found != wanted or len(selected) != len(wanted):
        raise ValueError(f"deep soil donor {url}: the index lacks {sorted(wanted - found)} "
                         f"or repeats a record ({len(selected)} rows for {len(wanted)})")
    ranges = transport._coalesce(rows, selected, size)
    staging = Path(tempfile.mkdtemp(prefix=".deep-soil-", dir=folder))
    try:
        parts = [staging / f"{i:03d}.part" for i in range(len(ranges))]
        with ThreadPoolExecutor(max_workers=min(8, len(ranges))) as pool:
            list(pool.map(lambda pair: transport._download_range(url, pair[0], pair[1], 5),
                          zip(ranges, parts)))
        assembled = staging / "assembled"
        with assembled.open("wb") as output:
            for part in parts:
                output.write(part.read_bytes())
        payload = assembled.read_bytes()
        if payload[:4] != b"GRIB" or payload[-4:] != b"7777":
            raise ValueError(f"deep soil donor {url}: the assembled ranges are not a GRIB2 stream")
        os.replace(assembled, destination)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return destination, {"url": url, "cached": False, "records": len(selected),
                         "bytes": destination.stat().st_size}


def _selector_row(key: str, spec: Mapping[str, object]) -> dict:
    return dict(key=key, discipline=int(spec["discipline"]), category=int(spec["category"]),
                parameter=int(spec["parameter"]), constituent_type=None, aerosol_type=None,
                level_type=int(spec["level_type"]), levels=None)


def _decode(donor, path: Path):
    """``(temperature, moisture, land, latitude, longitude)`` on the donor grid."""

    from woof.grib2_stack import stack_grib2

    first = donor["layers"][0]["selectors"]
    rows = [_selector_row("soil_temperature", first["soil_temperature"]),
            _selector_row("volumetric_soil_moisture", first["volumetric_soil_moisture"]),
            _selector_row("land", donor["land_selector"])]
    scratch = Path(tempfile.mkdtemp(prefix=".deep-soil-decode-", dir=path.parent))
    try:
        fields = {field.key: field for field in stack_grib2(path, rows, scratch / "stack")}
        out = {}
        tops = [float(layer["top"]) for layer in donor["layers"]]
        for key in ("soil_temperature", "volumetric_soil_moisture"):
            field = fields[key]
            if field.projection is not None:
                raise ValueError("deep soil donor: only a regular latitude/longitude donor grid is declared")
            levels = np.asarray(field.levels, dtype=np.float64)
            array = np.array(field.array, dtype=np.float32)
            picked = []
            for top in tops:
                hit = np.flatnonzero(np.abs(levels - top) < 1e-6)
                if hit.size != 1:
                    raise ValueError(f"deep soil donor: {key} has no single layer from {top} m "
                                     f"(levels {levels.tolist()})")
                picked.append(array[hit[0]])
            out[key] = np.stack(picked)
        land = np.array(fields["land"].array, dtype=np.float32)[0]
        latitude = np.asarray(fields["land"].latitude, dtype=np.float64)
        longitude = np.asarray(fields["land"].longitude, dtype=np.float64)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return out["soil_temperature"], out["volumetric_soil_moisture"], land, latitude, longitude


def _fractional_indices(latitude, longitude, lat_axis, lon_axis):
    """Fractional (y, x) of target points on a regular, periodic donor grid."""

    dlat = float(lat_axis[1] - lat_axis[0])
    dlon = float(lon_axis[1] - lon_axis[0])
    y = (np.asarray(latitude, dtype=np.float64) - float(lat_axis[0])) / dlat
    x = np.mod(np.asarray(longitude, dtype=np.float64) - float(lon_axis[0]), 360.0) / dlon
    return y, x


def sample_deep_soil(contract: Mapping[str, object], *, latitude, longitude,
                     valid_time: datetime, cache_dir=None,
                     primary_cycle: datetime | None = None) -> DeepSoil:
    """The donor's declared layers at ``valid_time`` on the target points."""

    from woof.ingest.land_state import _sample_land

    donor = contract[DEEP_DONOR_KEY]
    root = _cache_root(cache_dir)
    errors = []
    for cycle, lead in donor_candidates(donor, valid_time, primary_cycle):
        try:
            path, fetched = _fetch(donor, cycle, lead, root)
            temperature, moisture, land, lat_axis, lon_axis = _decode(donor, path)
        except Exception as error:  # the next candidate cycle, then a refusal
            errors.append(f"{cycle:%Y-%m-%d %H}Z f{lead:03d}: {type(error).__name__}: {error}")
            continue
        break
    else:
        raise RuntimeError(f"deep soil donor {donor['source_id']} unavailable for "
                           f"{_utc_naive(valid_time):%Y-%m-%d %H}Z: " + "; ".join(errors))
    # A periodic donor (a global grid that does not repeat its first column)
    # gets that column appended, so bilinear weights across 0 deg longitude
    # have both neighbours.
    nx = lon_axis.size
    periodic = abs((lon_axis[-1] - lon_axis[0]) + (lon_axis[1] - lon_axis[0]) - 360.0) < 1e-6
    values = np.concatenate([temperature, moisture], axis=0)
    if periodic:
        values = np.concatenate([values, values[:, :, :1]], axis=2)
        land = np.concatenate([land, land[:, :1]], axis=1)
    donor_land = ((land >= 0.5) & np.isfinite(values).all(axis=0)
                  & (values[: len(donor["layers"])] > 0.0).all(axis=0)
                  & (values[len(donor["layers"]):] > 0.0).all(axis=0)
                  & (values[len(donor["layers"]):] < 1.0 - 1e-6).all(axis=0))
    # The donor's water points carry no soil (GFS publishes them as bitmap
    # gaps, decoded NaN); a zero-weight NaN neighbour still poisons a
    # bilinear sum (NaN * 0 is NaN), so only donor land keeps its values.
    values = np.where(donor_land[None], values, 0.0).astype(np.float32)
    y, x = _fractional_indices(latitude, longitude, lat_axis, lon_axis)
    if not periodic:
        x = np.where(x > nx - 1, np.nan, x)
    shape = np.shape(latitude)
    sampled, found = _sample_land(values, donor_land, x.reshape(-1), y.reshape(-1),
                                  int(donor["horizontal"]["search_radius_cells"]))
    count = len(donor["layers"])
    sampled = sampled.reshape((values.shape[0],) + shape)
    found = found.reshape(shape)
    receipt = {
        "schema": "gpuwm-deep-soil-donor-v1",
        "source_id": donor["source_id"],
        "soil_model": donor["soil_model"],
        "cycle": f"{cycle:%Y-%m-%dT%H}Z",
        "lead_hours": lead,
        "valid_time": f"{_utc_naive(valid_time):%Y-%m-%dT%H:%M:%S}Z",
        "layers_m": [[float(l["top"]), float(l["bottom"])] for l in donor["layers"]],
        "grib": str(path),
        "grib_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "fetch": fetched,
        "skipped_candidates": errors,
        "points": int(found.size),
        "points_with_donor_land": int(np.count_nonzero(found)),
        "horizontal": dict(donor["horizontal"]),
    }
    return DeepSoil(sampled[:count], sampled[count:], found, receipt)


def stack_below(primary, donor_values, fallback_index: int = -1):
    """``primary`` (L, ...) then ``donor_values`` (D, ...), NaN donor points
    taking the primary's deepest layer (the one-layer arm's carry-down)."""

    primary = np.asarray(primary)
    donor_values = np.asarray(donor_values, dtype=primary.dtype if primary.dtype.kind == "f" else np.float32)
    carry = np.broadcast_to(primary[fallback_index], donor_values.shape)
    filled = np.where(np.isfinite(donor_values), donor_values, carry)
    return np.concatenate([primary, filled.astype(primary.dtype, copy=False)], axis=0)


def extend_mapped_soil_fields(fields: Mapping[str, object], contract: Mapping[str, object], *,
                              context, landmask=None):
    """Stack the donor's layers under the mapped soil arrays on the target grid.

    Returns ``(fields, effective contract, receipt)``.  ``context`` is the
    :class:`woof.ingest.land_state.LandStateContext` (where and when).
    """

    from woof.ingest.soil_contract import MAPPED_SOIL_MOISTURE, MAPPED_SOIL_TEMPERATURE

    if context is None:
        raise RuntimeError("this route passed no domain context (where and when), so the "
                           "declared deep soil donor cannot be sampled")
    deep = sample_deep_soil(contract, latitude=context.latitude, longitude=context.longitude,
                            valid_time=context.valid_time, cache_dir=getattr(context, "cache_dir", None))
    out = dict(fields)
    for name, values in ((MAPPED_SOIL_TEMPERATURE, deep.temperature),
                         (MAPPED_SOIL_MOISTURE, deep.moisture)):
        primary = fields[name]
        on_device = hasattr(primary, "get") and not isinstance(primary, np.ndarray)
        host = np.asarray(primary.get() if on_device else primary)
        if host.shape[1:] != values.shape[1:]:
            raise ValueError(f"deep soil donor sampled {values.shape[1:]} points but {name} "
                             f"is {host.shape[1:]}")
        stacked = stack_below(host, values)
        if on_device:
            import cupy
            stacked = cupy.asarray(stacked)
        out[name] = stacked
    land = None if landmask is None else (np.asarray(
        landmask.get() if hasattr(landmask, "get") else landmask) >= 0.5)
    receipt = dict(deep.receipt)
    if land is not None:
        receipt["land_points"] = int(np.count_nonzero(land))
        receipt["land_points_carried_down"] = int(np.count_nonzero(land & ~deep.found))
    tops = np.nanmean(np.where(land, deep.temperature, np.nan), axis=(1, 2)) if land is not None \
        and deep.temperature.ndim == 3 and land.any() else None
    if tops is not None:
        receipt["land_layer_mean_temperature_K"] = [round(float(v), 2) for v in tops]
        receipt["land_layer_mean_moisture"] = [round(float(v), 4) for v in np.nanmean(
            np.where(land, deep.moisture, np.nan), axis=(1, 2))]
    print(f"note: deep soil: {receipt['source_id']} {receipt['cycle']} f{receipt['lead_hours']:03d} "
          f"supplies {len(receipt['layers_m'])} layer(s) below the source's own "
          f"({receipt['points_with_donor_land']} of {receipt['points']} point(s) reached donor land)",
          file=sys.stderr, flush=True)
    return out, effective_soil_layer_contract(contract), receipt


# ---------------------------------------------------------------------------
# The WPS intermediate box (the hex limited-area road).
# ---------------------------------------------------------------------------

def _wps_records(payload: bytes):
    """Fortran sequential records of a big-endian WPS file."""

    import struct

    offset = 0
    out = []
    while offset < len(payload):
        (size,) = struct.unpack(">i", payload[offset:offset + 4])
        body = payload[offset + 4:offset + 4 + size]
        (tail,) = struct.unpack(">i", payload[offset + 4 + size:offset + 8 + size])
        if tail != size:
            raise ValueError("WPS intermediate record markers disagree")
        out.append(body)
        offset += size + 8
    return out


def _wps_record(body: bytes) -> bytes:
    import struct

    return struct.pack(">i", len(body)) + body + struct.pack(">i", len(body))


def _layer_cm(layer) -> tuple[int, int]:
    return int(round(float(layer["top"]) * 100)), int(round(float(layer["bottom"]) * 100))


def append_deep_soil_to_wps(path, contract: Mapping[str, object], *, valid_time: datetime,
                            primary_cycle: datetime | None = None, cache_dir=None) -> dict:
    """Append the donor's layers (``SMtttbbb``/``STtttbbb``) to a WPS v5 file.

    The file's own soil records name the source layers; the donor's layers
    are sampled on the file's regular lat-lon points and written in the
    same record form, so the init reads a four-layer first guess.  Water
    points and land the donor does not reach keep the source's deepest
    layer, the value the one-layer init carried down.
    """

    import struct

    path = Path(path)
    payload = path.read_bytes()
    records = _wps_records(payload)
    if len(records) % 5:
        raise ValueError(f"{path}: not whole WPS v5 field groups")
    groups = {}
    for index in range(0, len(records), 5):
        header = records[index + 1]
        groups[header[60:69].decode("ascii").strip()] = records[index:index + 5]
    donor = contract[DEEP_DONOR_KEY]
    top, bottom = _layer_cm(contract["source_layers"][-1])
    names = {prefix: f"{prefix}{top:03d}{bottom:03d}" for prefix in ("ST", "SM")}
    for name in (*names.values(), "LANDSEA"):
        if name not in groups:
            raise ValueError(f"{path}: no {name} record to stack the donor's layers under")
    head = groups[names["SM"]][1]
    nx, ny, iproj = struct.unpack(">iii", head[144:156])
    if iproj != 0:
        raise ValueError(f"{path}: the deep soil donor writes onto a regular lat-lon WPS grid "
                         f"(iproj 0); this file declares iproj {iproj}")
    startlat, startlon, deltalat, deltalon = struct.unpack(">ffff", groups[names["SM"]][2][8:24])
    latitude = np.repeat((startlat + deltalat * np.arange(ny, dtype=np.float64))[:, None], nx, axis=1)
    longitude = np.repeat((startlon + deltalon * np.arange(nx, dtype=np.float64))[None, :], ny, axis=0)
    deep = sample_deep_soil(contract, latitude=latitude, longitude=longitude,
                            valid_time=valid_time, cache_dir=cache_dir, primary_cycle=primary_cycle)
    land = np.frombuffer(groups["LANDSEA"][4], ">f4").reshape(ny, nx) >= 0.5
    use = land & deep.found
    appended = []
    out = bytearray(payload)
    for prefix, values, what in (("ST", deep.temperature, "Soil temperature"),
                                 ("SM", deep.moisture, "Soil moisture")):
        group = groups[names[prefix]]
        primary = np.frombuffer(group[4], ">f4").reshape(ny, nx)
        for layer, plane in zip(donor["layers"], values):
            layer_top, layer_bottom = _layer_cm(layer)
            field = f"{prefix}{layer_top:03d}{layer_bottom:03d}"
            data = np.where(use, plane, primary).astype(">f4")
            header = bytearray(group[1])
            header[60:69] = field.ljust(9).encode("ascii")
            text = f"{what} {layer_top}-{layer_bottom} cm, {donor['source_id']}"
            header[94:140] = text[:46].ljust(46).encode("ascii")
            for body in (group[0], bytes(header), group[2], group[3], data.tobytes()):
                out += _wps_record(body)
            appended.append(field)
    staging = path.with_name(path.name + ".deep-soil.part")
    staging.write_bytes(bytes(out))
    os.replace(staging, path)
    receipt = dict(deep.receipt)
    receipt.update({
        "appended_fields": appended,
        "land_points": int(np.count_nonzero(land)),
        "land_points_carried_down": int(np.count_nonzero(land & ~deep.found)),
        "land_layer_mean_temperature_K": [round(float(plane[use].mean()), 2)
                                          for plane in deep.temperature] if use.any() else [],
        "land_layer_mean_moisture": [round(float(plane[use].mean()), 4)
                                     for plane in deep.moisture] if use.any() else [],
    })
    return receipt


__all__ = [
    "DEEP_DONOR_KEY", "DONOR_LAYERS_KEY", "DeepSoil", "append_deep_soil_to_wps", "donor_candidates",
    "effective_soil_layer_contract", "extend_mapped_soil_fields", "has_deep_donor",
    "primary_soil_layer_contract", "sample_deep_soil", "stack_below", "validate_deep_donor",
]
