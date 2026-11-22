"""Select and remap soil before the RRFS producer's first-guess reader.

The projected intermediate also preserves native nodes as STxxxXXX/SMxxxXXX
(point depths, in cm).  Older callers declared four FG levels, so those
records were ignored and RUC rebuilt nine nodes from four interpolated layers.
This adapter chooses the richest complete soil column from the MET itself,
removes competing soil records, and feeds the worker its nine-node dialect.
Atmospheric record bytes and the worker's horizontal/terrain/ice treatment stay
unchanged.  No engine binary change is needed.
"""
from __future__ import annotations

from pathlib import Path
import re
import struct

import numpy as np

from .wps_intermediate import WpsIntermediateReader

RUC_DEPTHS_M = np.array([0., .01, .04, .10, .30, .60, 1., 1.60, 3.])
RUC_NAMES = tuple((f"SOILT{round(z*100):03d}", f"SOILM{round(z*100):03d}")
                  for z in RUC_DEPTHS_M)
_RUC_SOURCES = {"NCEP RAP Model", "NCEP HRRR Model", "NOAA GSD HRRR Model",
                "unknown model from NCEP"}
_LAYER = re.compile(r"^(ST|SM)(\d{3})(\d{3})$")
_NODE = re.compile(r"^SOIL([TM])(\d{3})$")


def node_bounds(depths):
    """Surface-to-deepest-node Voronoi control volumes (metres)."""
    z = np.asarray(depths, dtype=np.float64)
    if z.ndim != 1 or z.size < 2 or not np.isfinite(z).all() or z[0] < 0 or np.any(np.diff(z) <= 0):
        raise ValueError("soil node depths must be finite, nonnegative and strictly increasing")
    return np.r_[0., .5*(z[:-1]+z[1:]), z[-1]]


def remap_column(temperature, moisture, source_depths, source_bounds,
                 target_depths=RUC_DEPTHS_M):
    """Level-major columns; identity when depths AND support match.

    T: piecewise linear in depth, constant at either end.
    Water: piecewise constant in source control volumes, overlap averages on
    target volumes. If the target is deeper, extend the last value; then scale
    each entire column to retain its original integrated water. This explicit
    inventory convention introduces no unobserved deep water. The same scaling
    applies if a target is shallower. Zero-water columns remain zero. No floor
    or clipping is applied here. The receipt reports the scaling range.
    """
    t, q = np.asarray(temperature), np.asarray(moisture)
    z, b, target = (np.asarray(x, dtype=np.float64) for x in
                    (source_depths, source_bounds, target_depths))
    tb = node_bounds(target)
    if (z.ndim != 1 or z.size < 2 or not np.isfinite(z).all() or z[0] < 0
            or np.any(np.diff(z) <= 0) or b.shape != (len(z)+1,)
            or not np.isfinite(b).all() or b[0] != 0 or np.any(np.diff(b) <= 0)
            or np.any(z < b[:-1]) or np.any(z > b[1:])):
        raise ValueError("invalid soil source depths/control volumes")
    if t.shape != q.shape or t.shape[0] != len(z) or not np.isfinite(t).all() or not np.isfinite(q).all():
        raise ValueError("soil fields and depths disagree or contain missing values")
    if np.any(t <= 0) or np.any(q < 0):
        raise ValueError("nonpositive soil temperature or negative soil water")
    common = dict(source_depths_m=z.tolist(), source_bounds_m=b.tolist(),
                  target_depths_m=target.tolist(), target_bounds_m=tb.tolist())
    if np.array_equal(z, target) and np.array_equal(b, tb):
        return t.copy(), q.copy(), dict(path="native-nodes", **common,
                                       max_water_error_m=0., column_scale_range=[1., 1.])
    tf, qf = t.reshape(len(z), -1).astype(np.float64), q.reshape(len(z), -1).astype(np.float64)
    hi = np.clip(np.searchsorted(z, target, side="right"), 1, len(z)-1)
    a = np.clip((target-z[hi-1])/(z[hi]-z[hi-1]), 0., 1.)
    ot = (1-a[:, None])*tf[hi-1] + a[:, None]*tf[hi]
    overlap = np.maximum(0., np.minimum(tb[1:, None], b[None, 1:])
                         - np.maximum(tb[:-1, None], b[None, :-1]))
    overlap[:, -1] += np.maximum(0., tb[1:]-np.maximum(tb[:-1], b[-1]))
    oq = (overlap @ qf)/np.diff(tb)[:, None]
    water = np.diff(b) @ qf
    proposed = np.diff(tb) @ oq
    if np.any((water > 0) & (proposed <= 0)):
        raise ValueError("target soil support contains none of the source water")
    scale = np.divide(water, proposed, out=np.ones_like(water), where=proposed > 0)
    oq *= scale
    error = float(np.max(np.abs(np.diff(tb) @ oq-water)))
    shape = (len(target),)+t.shape[1:]
    return ot.reshape(shape), oq.reshape(shape), dict(
        path="conservative-depth-remap", **common, max_water_error_m=error,
        column_scale_range=[float(scale.min()), float(scale.max())],
        temperature_rule="linear in depth; constant ends",
        water_rule="control-volume overlaps; constant deep extension; column inventory rescale")


def _soil_name(name):
    return bool(_LAYER.fullmatch(name) or _NODE.fullmatch(name))


def _select(fields):
    """Select complete pairs, never mix node and layer representations."""
    by_name = {}
    for f in fields:
        if f.field in by_name:
            raise ValueError(f"duplicate soil field {f.field}")
        by_name[f.field] = f
    canonical = [name for pair in RUC_NAMES for name in pair]
    if all(name in by_name for name in canonical):
        # SOILT001 is ambiguous outside the RAP/HRRR dialect (ICON uses it
        # for its first node). The complete names AND source must agree.
        chosen = [by_name[n] for n in canonical]
        if all(f.map_source in _RUC_SOURCES for f in chosen):
            return list(RUC_NAMES), RUC_DEPTHS_M, node_bounds(RUC_DEPTHS_M), "ruc-node-names"
    pairs = {}
    for name in by_name:
        match = _LAYER.fullmatch(name)
        if match:
            kind, top, bot = match.groups()
            pairs.setdefault((int(top), int(bot)), {})[kind] = name
    incomplete = [k for k, v in pairs.items() if set(v) != {"ST", "SM"}]
    if incomplete:
        raise ValueError(f"incomplete soil temperature/moisture pairs: {incomplete}")
    nodes = sorted(k for k, v in pairs.items() if k[0] == k[1] and set(v) == {"ST", "SM"})
    if len(nodes) >= 2:
        z = np.array([a/100 for a, _ in nodes])
        if (all(f.map_source in _RUC_SOURCES for f in fields)
                and set(z).issubset(set(RUC_DEPTHS_M)) and len(z) != 9):
            raise ValueError("incomplete preserved RUC node column")
        return [(pairs[k]["ST"], pairs[k]["SM"]) for k in nodes], z, node_bounds(z), "preserved-point-names"
    layers = sorted(k for k, v in pairs.items() if k[0] < k[1] and set(v) == {"ST", "SM"})
    if (len(layers) >= 2 and layers[0][0] == 0
            and all(a[1] == b[0] for a, b in zip(layers, layers[1:]))):
        b = np.array([layers[0][0]]+[v for _, v in layers])/100
        return [(pairs[k]["ST"], pairs[k]["SM"]) for k in layers], .5*(b[:-1]+b[1:]), b, "layer-names"
    raise ValueError("no complete, unambiguous soil node or contiguous layer column in MET")


def _raw_record(stream, endian):
    head = stream.read(4)
    if len(head) != 4:
        raise ValueError("truncated WPS record")
    size = struct.unpack(endian+"i", head)[0]
    if size < 0:
        raise ValueError("negative WPS record length")
    body, tail = stream.read(size), stream.read(4)
    if len(body) != size or tail != head:
        raise ValueError("truncated or mismatched WPS record")
    return body


def _encoded_field(source, field, values, name, endian, map_source):
    """Retain the original grid header, including non-latlon projections."""
    with source.open("rb") as stream:
        stream.seek(field.record_offset)
        records = [_raw_record(stream, endian) for _ in range(5 if field.version == 5 else 4)]
    if field.version == 3:
        raise ValueError("soil handover requires WPS v4 or v5 source metadata")
    meta = bytearray(records[1])
    meta[28:60] = map_source.encode("ascii").ljust(32, b" ")
    meta[60:69] = name.encode("ascii").ljust(9, b" ")
    records[1] = bytes(meta)
    records[-1] = np.asarray(values, dtype=endian+"f4").tobytes(order="F")
    return b"".join(struct.pack(endian+"i", len(r))+r+struct.pack(endian+"i", len(r)) for r in records)


def prepare_soil_met(source, destination, *, target_depths=RUC_DEPTHS_M):
    """Return (MET path, FG level count, receipt), selecting nodes by default.

    Only RRFS RUC producers invoke this adapter. Non-soil records are copied
    byte for byte, including atmospheric source metadata. The worker treats
    soil map_source as a dialect selector: remapped non-RAP soil records use
    its RAP nine-node encoding, with original source identity in the receipt.
    This is an encoding label, not a claim that the donor model was RAP.
    """
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("soil normalization needs a separate destination")
    if not np.array_equal(np.asarray(target_depths), RUC_DEPTHS_M):
        raise ValueError("worker soil dialect supports the declared RUC nine-node target only")
    with WpsIntermediateReader(source) as reader:
        headers = list(reader.iter_fields(load_values=False))
        endian = reader.endian
    soil = [f for f in headers if _soil_name(f.field)]
    names, z, bounds, representation = _select(soil)
    wanted = {n for pair in names for n in pair}
    with WpsIntermediateReader(source) as reader:
        selected = {f.field: f for f in reader.iter_fields(include=wanted)}
    first = selected[names[0][0]]
    for f in selected.values():
        if (f.nx, f.ny, f.projection, f.valid_time, f.forecast_hour, f.map_source) != (
                first.nx, first.ny, first.projection, first.valid_time, first.forecast_hour, first.map_source):
            raise ValueError("soil records have different grids, times or source identities")
        valid_units = {"K"} if f.field in {p[0] for p in names} else {"m3 m-3", "m^3 m^-3", "m3/m3", "fraction", "1"}
        if f.units not in valid_units:
            raise ValueError(f"soil field {f.field} has unsupported units {f.units!r}")
    t = np.stack([selected[a].values for a, _ in names])
    q = np.stack([selected[b].values for _, b in names])
    t, q, receipt = remap_column(t, q, z, bounds, target_depths)
    receipt.update(source_met=str(source), source_representation=representation,
                   source_map_source=first.map_source, selected_fields=sorted(wanted),
                   discarded_fields=[f.field for f in soil if f.field not in wanted],
                   n_fg_soil_levels=9, applied=True)
    encoding = first.map_source if first.map_source in _RUC_SOURCES else "NCEP RAP Model"
    receipt["soil_transport_map_source"] = encoding
    receipt["soil_transport_note"] = "worker RAP/RUC node dialect; donor identity is source_map_source"
    # Already exactly what the worker expects: no rewrite or extra copy.
    if representation == "ruc-node-names" and len(soil) == 18:
        receipt["rewritten"] = False
        return source, 9, receipt
    offsets = [f.record_offset for f in headers]+[source.stat().st_size]
    with source.open("rb") as inp, destination.open("xb") as out:
        for f, end in zip(headers, offsets[1:]):
            if not _soil_name(f.field):
                inp.seek(f.record_offset)
                remaining = end-f.record_offset
                while remaining:
                    block = inp.read(min(1 << 20, remaining))
                    if not block:
                        raise ValueError("truncated MET during soil normalization")
                    out.write(block)
                    remaining -= len(block)
        for k, (tn, mn) in enumerate(RUC_NAMES):
            out.write(_encoded_field(source, first, t[k], tn, endian, encoding))
            out.write(_encoded_field(source, selected[names[0][1]], q[k], mn, endian, encoding))
    receipt["rewritten"] = True
    receipt["normalized_met"] = str(destination)
    receipt["max_water_error_after_float32_m"] = float(np.max(np.abs(
        np.diff(node_bounds(target_depths)) @ q.astype(np.float32).reshape(9, -1)
        - np.diff(bounds) @ np.stack([selected[b].values for _, b in names]).reshape(len(z), -1))))
    return destination, 9, receipt


def carries_ruc_nodes(source) -> bool:
    """True when the MET holds RUC's own nine soil nodes (an HRRR or RAP first guess).

    Only then does the producer hand the nodes over level for level. A first guess with
    layers only (GFS's four Noah layers and the other global sources) keeps the fork's own
    init_soil_properties_ruc interpolation, as NOAA's RRFS init does, unchanged.
    """
    try:
        with WpsIntermediateReader(Path(source)) as reader:
            soil = [f for f in reader.iter_fields(load_values=False) if _soil_name(f.field)]
        _, z, _, _ = _select(soil)
    except (OSError, ValueError):
        return False
    return len(z) == len(RUC_DEPTHS_M) and np.array_equal(np.asarray(z, dtype=np.float64), RUC_DEPTHS_M)


def handover_or_fork(met, destination, *, target_depths, requested):
    """(MET, FG soil level count, receipt) for the RRFS producer's case 7."""
    if target_depths is not None and carries_ruc_nodes(met):
        soil_met, levels, receipt = prepare_soil_met(met, destination, target_depths=target_depths)
        receipt["requested_nfgsoillevels"] = requested
        return soil_met, levels, receipt
    if requested is None:
        raise ValueError("the first guess carries no RUC nodes, so --nfgsoillevels must name its soil levels")
    return Path(met), int(requested), {
        "applied": False, "path": "fork-interpolation", "requested_nfgsoillevels": requested,
        "reason": ("the first guess carries no complete RUC nine-node column; the fork's own "
                   "init_soil_properties_ruc builds the nodes, as NOAA's RRFS init does")
                  if target_depths is not None else "the producer's land surface has no RUC nodes"}
