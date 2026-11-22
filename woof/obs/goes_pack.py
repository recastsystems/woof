"""The ``GPWMGOES`` container, read the way :mod:`woof.obs.sweeps` reads
``GPWMRDR1``: every self-describing field the writer promised is proven
before a single payload byte is interpreted.

Two pack families share this container and this reader, deliberately --
``gpuwm-obs.goes-cwp.v1`` on the 2 km ABI fixed grid and
``gpuwm-obs.goes-cloudtop.v1`` on the 10 km one.  The bridge refuses to
regrid one onto the other (``tools/rustwx/crates/rw-goes/src/cloudtop.rs``,
project decision 2026-08-06), so the two families are two files with two
grids, and the join between them is the consumer's explicit, recorded
choice.  This module does not make that choice; it hands back planes and
the navigation they are on.  :mod:`woof.obs.goes_cwp` makes it.

The framing, mirrored from ``tools/rustwx/crates/rw-goes/src/pack.rs``:

===========  ==========================================================
bytes 0-8    magic ``GPWMGOES``
bytes 8-12   version, ``<u4``
bytes 12-16  metadata length, ``<u4``
bytes 16-24  payload length, ``<u8``
bytes 24-64  zero padding
then         the metadata block, UTF-8 JSON
then         the payload: named ``<f4`` planes, contiguous
===========  ==========================================================

``content_sha256`` in the metadata is over the **payload only**, not the
whole file -- the same split ``sweeps.py`` uses, and the reason a
whole-file digest is reported separately as ``pack_sha256``.

In the payload, ``NaN`` means *no observation* and ``0.0`` means
*clear-sky zero*.  The bridge never writes one for the other, and neither
does anything downstream of this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

#: The 2 km cloud-water-path family.  v2 adds per-pixel DQF planes
#: (``cod_dqf``/``cps_dqf``/``actp_dqf``) and is otherwise additive, so a
#: name-based reader handles both; what the bump buys is the ability to
#: REQUIRE the planes and fail closed, rather than discover them missing
#: halfway through building an error model.
CWP_SCHEMA_V1 = "gpuwm-obs.goes-cwp.v1"
CWP_SCHEMA_V2 = "gpuwm-obs.goes-cwp.v2"
CWP_SCHEMAS = (CWP_SCHEMA_V1, CWP_SCHEMA_V2)

#: The 10 km cloud-top family, the CWP pack's deliberate sibling.
CLOUDTOP_SCHEMA_V1 = "gpuwm-obs.goes-cloudtop.v1"
CLOUDTOP_SCHEMA_V2 = "gpuwm-obs.goes-cloudtop.v2"
CLOUDTOP_SCHEMAS = (CLOUDTOP_SCHEMA_V1, CLOUDTOP_SCHEMA_V2)

#: Back-compatible aliases: these named the only schema that existed when
#: the reader was written, and callers still pass them.
CWP_SCHEMA = CWP_SCHEMA_V1
CLOUDTOP_SCHEMA = CLOUDTOP_SCHEMA_V1

#: Every family this reader knows.  A pack declaring anything else is
#: refused by name rather than by whichever field happens to be missing.
AOD_SCHEMA = "gpuwm-obs.goes-aod.v1"
KNOWN_SCHEMAS = CWP_SCHEMAS + CLOUDTOP_SCHEMAS + (AOD_SCHEMA,)

_MAGIC = b"GPWMGOES"
_VERSION = 1
_HEADER_BYTES = 64
_DTYPE = "<f4"

#: The status a pack must declare to be read as observations.  ``rw_goes``
#: writes ``EMPTY`` receipts for listings that found nothing; a pack that
#: is not ``READY`` is not an observation set.
_READY = "READY"

#: Metadata keys every family carries, whatever else it adds.
_REQUIRED_META = (
    "schema", "status", "satellite", "sector", "scan_start", "scan_end",
    "sources", "projection", "nx", "ny", "x_scan_rad", "y_scan_rad",
    "planes", "plane_order", "arrays", "payload_bytes", "content_sha256",
)

#: Projection fields that must agree, exactly, before two packs of
#: different families may be joined.  A join across two different
#: geostationary perspectives is not an interpolation, it is a fabrication.
PROJECTION_KEYS = (
    "perspective_point_height_m", "semi_major_axis_m", "semi_minor_axis_m",
    "longitude_of_projection_origin_deg", "sweep_angle_axis",
)


class GoesPackError(ValueError):
    """A GOES pack that does not satisfy the container contract.

    Always an error.  A partially decoded pack is never returned: the
    consumer either gets planes it can trust or an exception naming what
    was wrong.
    """


@dataclass(frozen=True)
class GoesPack:
    """One decoded ``GPWMGOES`` pack: its metadata and its planes.

    ``planes`` maps the writer's plane names -- ``cwp``, ``phase``,
    ``cod``, ``cps``, ``lat``, ``lon`` for the CWP family;
    ``cloud_top_height_m``, ``cloud_top_pressure_hpa``, ``lat``, ``lon``
    for the cloud-top family -- to ``(ny, nx)`` float32 arrays that are
    read-only views over the pack's payload.
    """

    path: Path
    meta: dict
    planes: dict
    pack_sha256: str
    pack_bytes: int

    @property
    def schema(self) -> str:
        return str(self.meta["schema"])

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.meta["ny"]), int(self.meta["nx"]))

    @property
    def family(self) -> str:
        """``"cwp"`` or ``"cloudtop"``, whatever the version."""

        if self.schema == AOD_SCHEMA:
            return "aod"
        return "cwp" if self.schema in CWP_SCHEMAS else "cloudtop"

    @property
    def schema_version(self) -> int:
        """The trailing version integer of the declared schema."""

        return int(str(self.schema).rsplit(".v", 1)[1])

    @property
    def has_dqf_planes(self) -> bool:
        """Whether every source row names a per-pixel DQF plane present here.

        All-or-nothing on purpose: a pack carrying DQF for two of three
        products would let an error model silently screen part of a scene
        under one rule and the rest under another.
        """

        sources = self.meta.get("sources") or []
        if not sources:
            return False
        return all(
            source.get("dqf_plane") in self.planes for source in sources)

    def dqf_plane(self, product: str) -> np.ndarray:
        """The per-pixel DQF word for one product, ``(ny, nx)`` uint16.

        NaN in the stored plane means the DQF pixel was itself fill or
        out of range -- the count the pack reports as ``dqf_missing`` --
        and is NEVER a stand-in for the real DQF value 0.  Those pixels
        come back with ``valid=False`` and a zero word that the caller
        must not read: ``astype(uint16)`` on a NaN is undefined and would
        hand back garbage silently rather than raising.

        Returns ``(word, valid)``.
        """

        rows = {str(source["product"]): source
                for source in self.meta.get("sources") or []}
        if product not in rows:
            raise GoesPackError(
                f"{self.path.name}: no source row for product {product!r}; "
                f"this pack was built from {sorted(rows)}")
        name = rows[product].get("dqf_plane")
        if name is None:
            raise GoesPackError(
                f"{self.path.name}: source row for {product!r} names no "
                f"dqf_plane. This is a {self.schema} pack; per-pixel DQF "
                "arrived in v2")
        plane = np.asarray(self.plane(name), dtype=np.float64)
        valid = np.isfinite(plane)
        word = np.zeros(plane.shape, dtype=np.uint16)
        word[valid] = plane[valid].astype(np.uint16)
        return word, valid

    @property
    def pairing_key(self) -> tuple[str, str, str]:
        """``(satellite, sector, scan_start)`` -- the key both families
        state at the top level, and the only thing that makes two packs
        two halves of one scan."""

        return (str(self.meta["satellite"]), str(self.meta["sector"]),
                str(self.meta["scan_start"]))

    def plane(self, name: str) -> np.ndarray:
        """One plane by name, refusing a missing one by name."""

        if name not in self.planes:
            raise GoesPackError(
                f"{self.path.name}: no plane {name!r}; this pack carries "
                f"{list(self.meta['plane_order'])}")
        return self.planes[name]

    def projection(self) -> dict:
        """The geostationary navigation of record, as a plain dict."""

        return {key: self.meta["projection"][key] for key in PROJECTION_KEYS}

    def dqf_policy(self) -> list:
        """What the bridge's DQF gate actually did, per source granule.

        This is the honoured-upstream half of the QC story: the gated
        pixels are already ``NaN`` in the planes, and this is the record
        of which rule and which condemn mask put them there.  A consumer
        that does not carry this into its own receipt cannot say how its
        observations were screened.
        """

        return [{"product": str(source["product"]),
                 "filename": str(source["filename"]),
                 "sha256": str(source["sha256"]),
                 "dqf_rule": str(source["dqf_rule"]),
                 "condemn_mask": (None if source.get("condemn_mask") is None
                                  else int(source["condemn_mask"])),
                 "dqf_plane": source.get("dqf_plane"),
                 "dqf": dict(source["dqf"])}
                for source in self.meta["sources"]]

    def provenance(self) -> dict:
        """Everything a downstream receipt needs to name this pack."""

        return {
            "pack_file": self.path.name,
            "pack_sha256": self.pack_sha256,
            "pack_bytes": int(self.pack_bytes),
            "pack_schema": self.schema,
            "content_sha256": str(self.meta["content_sha256"]),
            "satellite": str(self.meta["satellite"]),
            "sector": str(self.meta["sector"]),
            "scan_start": str(self.meta["scan_start"]),
            "scan_end": str(self.meta["scan_end"]),
            "nx": int(self.meta["nx"]),
            "ny": int(self.meta["ny"]),
            "window": self.meta.get("window"),
            "planes": list(self.meta["plane_order"]),
            "schema_version": self.schema_version,
            "has_per_pixel_dqf": self.has_dqf_planes,
            "projection": self.projection(),
            "dqf_policy": self.dqf_policy(),
        }


def read_goes_pack(path: str | Path, *,
                   expected_schema: str | None = None) -> GoesPack:
    """Read one ``GPWMGOES`` pack, checking the contract before the data.

    ``expected_schema`` is how a caller fails closed on *which family* it
    is holding.  Pass it: the two families share a container, so a
    cloud-top pack handed to a CWP consumer parses perfectly and yields
    heights where the consumer wanted water paths.
    """

    path = Path(path)
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise GoesPackError(f"{path}: cannot be read: {error}") from error
    return _decode(raw, path, expected_schema)


def _decode(raw: bytes, path: Path, expected_schema: str | None) -> GoesPack:
    if len(raw) < _HEADER_BYTES:
        raise GoesPackError(
            f"{path.name}: {len(raw)} bytes, the header alone is "
            f"{_HEADER_BYTES}")
    if raw[:8] != _MAGIC:
        raise GoesPackError(
            f"{path.name}: magic is {raw[:8]!r}, a GOES pack starts with "
            f"{_MAGIC!r}")
    version = int(np.frombuffer(raw[8:12], dtype="<u4")[0])
    if version != _VERSION:
        raise GoesPackError(
            f"{path.name}: pack version {version}, this reader reads "
            f"{_VERSION}")
    meta_len = int(np.frombuffer(raw[12:16], dtype="<u4")[0])
    payload_len = int(np.frombuffer(raw[16:24], dtype="<u8")[0])
    meta_end = _HEADER_BYTES + meta_len
    payload_end = meta_end + payload_len
    if payload_end != len(raw):
        raise GoesPackError(
            f"{path.name}: the header declares {payload_end} bytes, the "
            f"file has {len(raw)}")

    try:
        meta = json.loads(raw[_HEADER_BYTES:meta_end].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GoesPackError(
            f"{path.name}: metadata block is not UTF-8 JSON: {error}"
        ) from error
    if not isinstance(meta, dict):
        raise GoesPackError(
            f"{path.name}: metadata block is a "
            f"{type(meta).__name__}, not an object")

    schema = meta.get("schema")
    if schema not in KNOWN_SCHEMAS:
        raise GoesPackError(
            f"{path.name}: declares schema {schema!r}; this reader knows "
            f"{list(KNOWN_SCHEMAS)}")
    if expected_schema is not None:
        wanted = ((expected_schema,) if isinstance(expected_schema, str)
                  else tuple(expected_schema))
        if schema not in wanted:
            raise GoesPackError(
                f"{path.name}: is a {schema!r} pack but the caller demanded "
                f"{list(wanted)}. The two GOES families share a container "
                "on purpose; a pack of the wrong family decodes perfectly "
                "and answers a different question")
    if meta.get("status") != _READY:
        raise GoesPackError(
            f"{path.name}: status is {meta.get('status')!r}, only "
            f"{_READY!r} packs are observations")
    missing = [key for key in _REQUIRED_META if key not in meta]
    if missing:
        raise GoesPackError(
            f"{path.name}: metadata is missing {missing}")
    for key in PROJECTION_KEYS:
        if key not in meta["projection"]:
            raise GoesPackError(
                f"{path.name}: projection block is missing {key!r}; without "
                "the full geostationary perspective the planes have no "
                "navigation")

    payload = raw[meta_end:payload_end]
    if int(meta["payload_bytes"]) != len(payload):
        raise GoesPackError(
            f"{path.name}: metadata declares a {meta['payload_bytes']}-byte "
            f"payload, the header framed {len(payload)}")
    digest = hashlib.sha256(payload).hexdigest()
    if digest != meta.get("content_sha256"):
        raise GoesPackError(
            f"{path.name}: payload hashes to {digest}, metadata says "
            f"{meta.get('content_sha256')}")

    ny = int(meta["ny"])
    nx = int(meta["nx"])
    if ny <= 0 or nx <= 0:
        raise GoesPackError(
            f"{path.name}: declares a {ny} x {nx} grid")
    for axis, length, label in (("y_scan_rad", ny, "ny"),
                                ("x_scan_rad", nx, "nx")):
        if len(meta[axis]) != length:
            raise GoesPackError(
                f"{path.name}: {axis} has {len(meta[axis])} entries but "
                f"{label} is {length}; the navigation does not describe the "
                "planes")

    order = list(meta["plane_order"])
    if not order:
        raise GoesPackError(f"{path.name}: declares no planes")
    if sorted(order) != sorted(meta["planes"]):
        raise GoesPackError(
            f"{path.name}: plane_order is {order} but the planes table "
            f"holds {sorted(meta['planes'])}")
    if len(set(order)) != len(order):
        raise GoesPackError(
            f"{path.name}: plane_order repeats a name: {order}")

    planes: dict[str, np.ndarray] = {}
    for name in order:
        key = meta["planes"][name]
        if key not in meta["arrays"]:
            raise GoesPackError(
                f"{path.name}: plane {name!r} names array {key!r}, which "
                "the arrays table does not hold")
        entry = meta["arrays"][key]
        if entry.get("dtype") != _DTYPE:
            raise GoesPackError(
                f"{path.name}: plane {name!r} declares dtype "
                f"{entry.get('dtype')!r}, this reader reads {_DTYPE!r}")
        shape = tuple(int(value) for value in entry["shape"])
        if shape != (ny, nx):
            raise GoesPackError(
                f"{path.name}: plane {name!r} is shaped {shape}, the pack's "
                f"grid is {(ny, nx)}. Planes are only ever combined across "
                "granules whose navigation is identical")
        count = int(np.prod(shape))
        if count * 4 != int(entry["bytes"]):
            raise GoesPackError(
                f"{path.name}: plane {name!r} declares {entry['bytes']} "
                f"bytes for {count} float32 values")
        start = int(entry["offset"])
        stop = start + int(entry["bytes"])
        if start < 0 or stop > len(payload):
            raise GoesPackError(
                f"{path.name}: plane {name!r} spans bytes {start}:{stop} of "
                f"a {len(payload)}-byte payload")
        values = np.frombuffer(payload[start:stop],
                               dtype=_DTYPE).reshape(shape)
        values.flags.writeable = False
        planes[name] = values

    return GoesPack(path=path, meta=meta, planes=planes,
                    pack_sha256=hashlib.sha256(raw).hexdigest(),
                    pack_bytes=len(raw))
