"""The titan-rs volume containers, written and read from numpy.

titan-rs (the storm-object engine ``woof cells`` drives) takes
checksummed Cartesian volumes: one ``TFR1`` frame per scan, and a
``TFS1`` stream that concatenates frames.  Both layouts are fixed and
documented by that project (``FORMATS.md``: every integer and float is
little-endian, the trailing CRC-32 is the standard reflected polynomial
over every preceding byte, cell ``(x, y, z)`` lives at index
``(z * ny + y) * nx + x``, and NaN is a missing cell).  This module is
that layout in Python -- and nothing else: no interpolation, no file
discovery, no engine call.  It is pinned against the engine's own
``synthetic`` stream byte for byte, so a volume this module writes is a
volume the engine reads, and vice versa.

Kept dependency-free on purpose (``struct``, ``zlib``, ``numpy``): the
exporter runs wherever WOOF-ArWen history is, including a box that has no
titan binary at all, and writing the stream is not what needs the
engine.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

#: The frame and stream magics, and the one version both carry.
FRAME_MAGIC = b"TFR1"
STREAM_MAGIC = b"TFS1"
VERSION = 1

#: Field names in the fixed order the frame stores them, bit ``i`` of
#: the field mask naming ``FIELD_ORDER[i]``.  Reflectivity is bit 0 and
#: is never optional.
FIELD_ORDER = ("reflectivity", "velocity", "zdr", "kdp", "rhohv",
               "quality", "temperature", "auxiliary")

#: The fixed header: magic(4) version(2) flags(2) timestamp(8)
#: nx,ny,nz(12) origin_x,origin_y,dx,dy(32) projection_len,source_len(8).
_HEADER = struct.Struct("<4sHHqIIIddddII")
assert _HEADER.size == 68


class TitanVolumeError(ValueError):
    """A volume that titan-rs would refuse, named before it is written."""


@dataclass
class TitanVolume:
    """One Cartesian volume: geometry, provenance, and its fields.

    ``z_levels_m`` are cell-CENTRE heights above sea level, strictly
    increasing; ``origin_x_m``/``origin_y_m`` are the lower-left CORNER
    of cell (0, 0), so cell (x, y) is centred at
    ``origin + (index + 0.5) * spacing``.  Fields are ``(nz, ny, nx)``
    float32 arrays, which is the frame's own cell order in C layout.
    """

    timestamp_ms: int
    nx: int
    ny: int
    z_levels_m: np.ndarray
    origin_x_m: float
    origin_y_m: float
    dx_m: float
    dy_m: float
    projection: str
    source: str
    reflectivity: np.ndarray
    optional: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def nz(self) -> int:
        return int(len(self.z_levels_m))

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.nz, self.ny, self.nx)

    def validate(self) -> None:
        """Every rule the engine's ``GridGeometry::validate`` applies."""

        if self.nx <= 0 or self.ny <= 0 or self.nz <= 0:
            raise TitanVolumeError("all grid dimensions must be nonzero")
        z = np.asarray(self.z_levels_m, dtype=np.float64)
        if not np.all(np.isfinite(z)):
            raise TitanVolumeError("every z level must be finite")
        if self.nz > 1 and not np.all(np.diff(z) > 0):
            raise TitanVolumeError("z_levels_m must be strictly increasing")
        for name, value in (("origin_x_m", self.origin_x_m),
                            ("origin_y_m", self.origin_y_m)):
            if not np.isfinite(value):
                raise TitanVolumeError(f"{name} must be finite")
        for name, value in (("dx_m", self.dx_m), ("dy_m", self.dy_m)):
            if not (np.isfinite(value) and value > 0):
                raise TitanVolumeError(f"{name} must be finite and positive")
        for name, values in [("reflectivity", self.reflectivity),
                             *self.optional.items()]:
            if name not in FIELD_ORDER:
                raise TitanVolumeError(
                    f"{name!r} is not a titan field; the frame carries "
                    f"{', '.join(FIELD_ORDER)}")
            if tuple(values.shape) != self.shape:
                raise TitanVolumeError(
                    f"{name} has shape {tuple(values.shape)}, but the grid "
                    f"is (nz, ny, nx) = {self.shape}")
        quality = self.optional.get("quality")
        if quality is not None:
            finite = quality[np.isfinite(quality)]
            if finite.size and (finite.min() < 0.0 or finite.max() > 1.0):
                raise TitanVolumeError(
                    "finite quality values must lie in [0, 1]")

    def field_mask(self) -> int:
        mask = 1
        for bit, name in enumerate(FIELD_ORDER):
            if name != "reflectivity" and name in self.optional:
                mask |= 1 << bit
        return mask


def encode_frame(volume: TitanVolume) -> bytes:
    """One ``TFR1`` frame, checksummed, exactly as titan-io writes it."""

    volume.validate()
    projection = volume.projection.encode("utf-8")
    source = volume.source.encode("utf-8")
    out = bytearray()
    out += _HEADER.pack(
        FRAME_MAGIC, VERSION, volume.field_mask(), int(volume.timestamp_ms),
        volume.nx, volume.ny, volume.nz,
        float(volume.origin_x_m), float(volume.origin_y_m),
        float(volume.dx_m), float(volume.dy_m),
        len(projection), len(source))
    out += np.ascontiguousarray(volume.z_levels_m, dtype="<f8").tobytes()
    out += projection
    out += source
    out += np.ascontiguousarray(volume.reflectivity, dtype="<f4").tobytes()
    for name in FIELD_ORDER[1:]:
        if name in volume.optional:
            out += np.ascontiguousarray(
                volume.optional[name], dtype="<f4").tobytes()
    out += struct.pack("<I", zlib.crc32(bytes(out)) & 0xFFFFFFFF)
    return bytes(out)


def decode_frame(data: bytes) -> TitanVolume:
    """The inverse of :func:`encode_frame`, with the CRC checked first."""

    if len(data) < _HEADER.size + 4:
        raise TitanVolumeError("TFR1 frame is too short to hold a header")
    body, crc = data[:-4], struct.unpack_from("<I", data, len(data) - 4)[0]
    if crc != (zlib.crc32(body) & 0xFFFFFFFF):
        raise TitanVolumeError("TFR1 frame checksum mismatch")
    (magic, version, flags, timestamp_ms, nx, ny, nz, ox, oy, dx, dy,
     plen, slen) = _HEADER.unpack_from(body, 0)
    if magic != FRAME_MAGIC:
        raise TitanVolumeError("not a TFR1 frame")
    if version != VERSION:
        raise TitanVolumeError(f"unsupported TFR version {version}")
    if not flags & 1:
        raise TitanVolumeError("TFR1 frame has no reflectivity field")
    offset = _HEADER.size
    z = np.frombuffer(body, dtype="<f8", count=nz, offset=offset).copy()
    offset += 8 * nz
    projection = body[offset:offset + plen].decode("utf-8")
    offset += plen
    source = body[offset:offset + slen].decode("utf-8")
    offset += slen
    cells = nx * ny * nz

    def take() -> np.ndarray:
        nonlocal offset
        values = np.frombuffer(body, dtype="<f4", count=cells, offset=offset)
        offset += 4 * cells
        return values.reshape(nz, ny, nx).copy()

    reflectivity = take()
    optional: dict[str, np.ndarray] = {}
    for bit, name in enumerate(FIELD_ORDER):
        if bit and flags >> bit & 1:
            optional[name] = take()
    if offset != len(body):
        raise TitanVolumeError(
            f"TFR1 frame carries {len(body) - offset} trailing bytes")
    return TitanVolume(
        timestamp_ms=timestamp_ms, nx=nx, ny=ny, z_levels_m=z,
        origin_x_m=ox, origin_y_m=oy, dx_m=dx, dy_m=dy,
        projection=projection, source=source, reflectivity=reflectivity,
        optional=optional)


def encode_stream(frames: list[bytes]) -> bytes:
    """A ``TFS1`` stream over already-encoded frames."""

    out = bytearray(STREAM_MAGIC)
    out += struct.pack("<HHI", VERSION, 0, len(frames))
    for frame in frames:
        out += struct.pack("<q", len(frame))
        out += frame
    out += struct.pack("<I", zlib.crc32(bytes(out)) & 0xFFFFFFFF)
    return bytes(out)


def decode_stream(data: bytes) -> list[bytes]:
    """The frames of a ``TFS1`` stream, checksum verified, still encoded."""

    if len(data) < 16 or data[:4] != STREAM_MAGIC:
        raise TitanVolumeError("not a TFS1 stream")
    body, crc = data[:-4], struct.unpack_from("<I", data, len(data) - 4)[0]
    if crc != (zlib.crc32(body) & 0xFFFFFFFF):
        raise TitanVolumeError("TFS1 stream checksum mismatch")
    version, _reserved, count = struct.unpack_from("<HHI", body, 4)
    if version != VERSION:
        raise TitanVolumeError(f"unsupported TFS version {version}")
    offset = 12
    frames: list[bytes] = []
    for _ in range(count):
        (length,) = struct.unpack_from("<q", body, offset)
        offset += 8
        frames.append(body[offset:offset + length])
        offset += length
    if offset != len(body):
        raise TitanVolumeError(
            f"TFS1 stream carries {len(body) - offset} trailing bytes")
    return frames


class StreamWriter:
    """Write a ``TFS1`` stream one frame at a time, memory bounded.

    The engine's own ``StreamWriter`` contract: the frame count is
    declared up front (it lives in the fixed header, under the CRC), the
    bytes go to ``<path>.tmp``, and only :meth:`finish` -- having seen
    exactly the declared number of frames -- renames onto ``path``.
    """

    def __init__(self, path: Path | str, frame_count: int):
        self.path = Path(path)
        self.temporary = self.path.with_name(self.path.name + ".tmp")
        self.declared = int(frame_count)
        self.written = 0
        self._crc = 0
        self._file = open(self.temporary, "wb")
        header = STREAM_MAGIC + struct.pack("<HHI", VERSION, 0, self.declared)
        self._put(header)

    def _put(self, data: bytes) -> None:
        self._file.write(data)
        self._crc = zlib.crc32(data, self._crc) & 0xFFFFFFFF

    def push(self, volume: TitanVolume) -> None:
        if self.written == self.declared:
            raise TitanVolumeError(
                f"stream declared {self.declared} frames and all were written")
        frame = encode_frame(volume)
        self._put(struct.pack("<q", len(frame)))
        self._put(frame)
        self.written += 1

    def finish(self) -> Path:
        if self.written != self.declared:
            self._file.close()
            self.temporary.unlink(missing_ok=True)
            raise TitanVolumeError(
                f"stream declared {self.declared} frames but {self.written} "
                f"were written; the short stream was not published")
        self._file.write(struct.pack("<I", self._crc))
        self._file.close()
        self.temporary.replace(self.path)
        return self.path

    def abandon(self) -> None:
        try:
            self._file.close()
        finally:
            self.temporary.unlink(missing_ok=True)


def read_stream(path: Path | str) -> list[TitanVolume]:
    data = Path(path).read_bytes()
    if data[:4] == FRAME_MAGIC:
        return [decode_frame(data)]
    return [decode_frame(frame) for frame in decode_stream(data)]


def write_stream(path: Path | str, volumes: list[TitanVolume]) -> Path:
    writer = StreamWriter(path, len(volumes))
    try:
        for volume in volumes:
            writer.push(volume)
    except Exception:
        writer.abandon()
        raise
    return writer.finish()


def cell_index(nx: int, ny: int, x, y, z):
    """The engine's linear cell index for ``(x, y, z)``."""

    return (np.asarray(z) * ny + np.asarray(y)) * nx + np.asarray(x)


def cell_xyz(nx: int, ny: int, index):
    """``(x, y, z)`` for the engine's linear cell index (vectorised)."""

    index = np.asarray(index, dtype=np.int64)
    plane = nx * ny
    z = index // plane
    rem = index % plane
    return rem % nx, rem // nx, z


__all__ = [
    "FIELD_ORDER", "FRAME_MAGIC", "STREAM_MAGIC", "StreamWriter",
    "TitanVolume", "TitanVolumeError", "cell_index", "cell_xyz",
    "decode_frame", "decode_stream", "encode_frame", "encode_stream",
    "read_stream", "write_stream",
]
