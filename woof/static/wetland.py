"""Static maps of the Noah-MP wetland option, built at prep in Rust.

``FSATMX`` (maximum saturated fraction) from GLWD v2 combined wetland area
(Lehner et al. 2025, ESSD 17, 2277; CC BY 4.0, figshare doi
10.6084/m9.figshare.28519994, licence read at source 2026-11-09) and
``WCAP`` (storage capacity, mm) from Copernicus DEM GLO-90 (Copernicus
WorldDEM-90 licence, free with attribution; already verified in
``tools/hydro_world/data/sources.v1.json``).  The arithmetic is
``tools/rustwx/crates/static-fields/src/wetland.rs`` on every core; this
module only moves bytes and calls it.

Attribution carried with every map: "GLWD v2: Lehner, B. et al. (2025),
CC BY 4.0. Produced using Copernicus WorldDEM-90 (c) DLR e.V. 2010-2014 and
(c) Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the
European Union and ESA; all rights reserved."
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import zipfile

import numpy as np

ALGORITHM = "zhang2022-wetland-static-v1"
GLWD_ZIP_URL = "https://ndownloader.figshare.com/files/54001814"
GLWD_MEMBER = "GLWD_v2_0_combined_classes/GLWD_v2_0_area_pct.tif"
GLO90_URL = ("https://copernicus-dem-90m.s3.amazonaws.com/"
             "Copernicus_DSM_COG_30_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM/"
             "Copernicus_DSM_COG_30_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM.tif")
ATTRIBUTION = " ".join(
    __doc__.split("Attribution carried with every map: ")[1].split('"')[1].split())


def cache_root() -> Path:
    return Path(os.environ.get("WOOF_STATIC_CACHE",
                               os.path.expanduser("~/.woof/static-cache")))


def glwd_area_pct(root: Path | None = None) -> Path:
    """The GLWD v2 combined area-percent GeoTIFF, fetched once."""
    given = os.environ.get("WOOF_GLWD_AREA_PCT")
    if given:
        return Path(given)
    root = (root or cache_root()) / "glwd-v2"
    target = root / Path(GLWD_MEMBER).name
    if target.is_file():
        return target
    from woof.hydro.acquire import _download
    archive = root / "GLWD_v2_0_combined_classes_tif.zip"
    _download(GLWD_ZIP_URL, archive)
    with zipfile.ZipFile(archive) as z:
        part = target.with_suffix(".tif.part")
        with z.open(GLWD_MEMBER) as src, part.open("wb") as out:
            while chunk := src.read(16 << 20):
                out.write(chunk)
        os.replace(part, target)
    archive.unlink()
    return target


def glo90_tiles(south, north, west, east, root: Path | None = None) -> list[Path]:
    """Copernicus GLO-90 tiles over a box, fetched in parallel (sea tiles absent)."""
    from woof.hydro.acquire import _download
    tiles_dir = Path(os.environ.get("GPUWM_HYDRO_WORLD_TILES",
                                    os.path.expanduser("~/.woof/hydro-world"))
                     ) / "copernicus-dem-glo90" / "tiles"
    jobs = []
    for lat in range(math.floor(south), math.ceil(north)):
        for lon in range(math.floor(west), math.ceil(east)):
            ns, ew = ("N" if lat >= 0 else "S"), ("E" if lon >= 0 else "W")
            url = GLO90_URL.format(ns=ns, lat=abs(lat), ew=ew, lon=abs(lon))
            jobs.append((url, tiles_dir / url.rsplit("/", 1)[1]))
    with ThreadPoolExecutor(16) as pool:
        states = list(pool.map(lambda j: _download(j[0], j[1], absent_ok=True)[0], jobs))
    return [path for (_url, path), state in zip(jobs, states) if state != "absent"]


def build(lat, lon, dx_m: float):
    """``(fsatmx, wcap_mm, receipt)`` on the mass-point grid of ``lat``/``lon``."""
    from woof.static import rust_bridge as bridge
    lat64 = np.ascontiguousarray(np.asarray(lat, dtype=np.float64))
    lon64 = np.ascontiguousarray(np.asarray(lon, dtype=np.float64))
    if lat64.shape != lon64.shape:
        raise ValueError("wetland static: lat/lon shapes differ")
    key = hashlib.sha256(lat64.tobytes() + lon64.tobytes()
                         + np.float64(dx_m).tobytes() + ALGORITHM.encode()).hexdigest()[:24]
    cached = cache_root() / "wetland-static" / f"{key}.npz"
    if cached.is_file():
        with np.load(cached) as z:
            return z["fsatmx"], z["wcap"], json.loads(str(z["receipt"]))
    pad = 0.6 * dx_m / 111195.0
    coslat = max(0.01, math.cos(math.radians(float(np.abs(lat64).max()))))
    south, north = float(lat64.min()) - pad, float(lat64.max()) + pad
    west, east = float(lon64.min()) - pad / coslat, float(lon64.max()) + pad / coslat
    glwd, glwd_receipt = glwd_area_pct_window(south, north, west, east)
    dems = glo90_tiles(south, north, west, east)
    if glwd is None:
        # No GLWD v2 coverage (north of 84N or south of 56S): no mapped
        # wetland, so FSATMX = 0 and the scheme is a no-op on this domain.
        zero = np.zeros(lat64.shape, dtype=np.float32)
        receipt = {"algorithm": ALGORITHM, **glwd_receipt, "dem_tiles": len(dems),
                   "glwd_cell_coverage": 0.0, "dem_cell_coverage": 0.0,
                   "attribution": ATTRIBUTION}
        return zero, zero.copy(), receipt
    library = bridge.load()
    try:
        entry = library.gpuwm_static_wetland_v1
    except AttributeError:
        raise bridge.StaticBridgeError(
            "the static-fields library predates the wetland maps: rebuild it "
            "(noahmp_wetland=1 needs FSATMX and WCAP from Rust)") from None
    f64p, f32p = ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_float)
    entry.argtypes = [f64p, f64p, ctypes.c_size_t, ctypes.c_double,
                      ctypes.c_char_p, ctypes.c_char_p, f32p, f32p, f64p]
    entry.restype = ctypes.c_int32
    n = lat64.size
    fsatmx = np.zeros(n, dtype=np.float32)
    wcap = np.zeros(n, dtype=np.float32)
    coverage = np.zeros(2, dtype=np.float64)
    rc = entry(lat64.ctypes.data_as(f64p), lon64.ctypes.data_as(f64p), n,
               float(dx_m), str(glwd).encode(), "\n".join(map(str, dems)).encode(),
               fsatmx.ctypes.data_as(f32p), wcap.ctypes.data_as(f32p),
               coverage.ctypes.data_as(f64p))
    if rc != 0:
        raise bridge.StaticBridgeError(bridge.last_error(library))
    receipt = {"algorithm": ALGORITHM, "glwd": str(glwd), **glwd_receipt, "dem_tiles": len(dems),
               "glwd_cell_coverage": float(coverage[0]),
               "dem_cell_coverage": float(coverage[1]),
               "attribution": ATTRIBUTION}
    fsatmx, wcap = fsatmx.reshape(lat64.shape), wcap.reshape(lat64.shape)
    cached.parent.mkdir(parents=True, exist_ok=True)
    part = cached.with_suffix(".part.npz")
    np.savez(part, fsatmx=fsatmx, wcap=wcap, receipt=json.dumps(receipt))
    os.replace(part, cached)
    return fsatmx, wcap, receipt


# ---- GLWD v2 domain window, fetched without the 925 MB archive -------------
#
# The figshare record ships GLWD v2 only as one 925 MB zip of every product.
# Its ``GLWD_v2_0_area_pct.tif`` member is 252 MB deflated (289 MB inflated),
# and the rest is other products.  So the fetch reads the zip's central
# directory with two small HTTP range requests, streams ONLY that member's
# byte range through zlib, and keeps the member's tile tables plus the
# compressed 128x128 LZW tiles inside the domain window.  The window is
# written as its own small tiled GeoTIFF (tiles copied byte for byte, so no
# pixel is re-encoded) and cached; the full GeoTIFF never lands on disk.
# The authoritative TIFF directory sits at the END of the member (GDAL
# rewrote it there), so the whole member is streamed once; it is spooled to a
# temporary file and removed as soon as the window is written.
# The member CRC-32 from the central directory is checked before use.

GLWD_WINDOW_TILES = 8  # windows snap to 8x8-tile (1024 px, ~4.27 deg) blocks for cache reuse
_UA = {"User-Agent": "gpuwm-static/wetland"}


def _range(url, first, last=None, *, timeout=120):
    import urllib.request
    rng = f"bytes={first}-{'' if last is None else last}"
    return urllib.request.urlopen(urllib.request.Request(url, headers={**_UA, "Range": rng}),
                                  timeout=timeout)


def _zip_member(url, member):
    """``(local_header_offset, compressed_size, crc32, method)`` of one zip member."""
    import struct
    with _range(url, 0, 0) as r:
        total = int(r.headers["Content-Range"].rsplit("/", 1)[1])
    with _range(url, max(0, total - 65536), total - 1) as r:
        tail = r.read()
    at = tail.rfind(b"PK\x05\x06")
    if at < 0:
        raise RuntimeError("GLWD v2 archive: no zip end-of-central-directory record")
    cd_size, cd_off = struct.unpack("<II", tail[at + 12:at + 20])
    if cd_off == 0xFFFFFFFF:
        z64 = tail.rfind(b"PK\x06\x06")
        cd_size, cd_off = struct.unpack("<QQ", tail[z64 + 40:z64 + 56])
    with _range(url, cd_off, cd_off + cd_size - 1) as r:
        cd = r.read()
    p = 0
    while cd[p:p + 4] == b"PK\x01\x02":
        method, = struct.unpack("<H", cd[p + 10:p + 12])
        crc, csz, usz = struct.unpack("<III", cd[p + 16:p + 28])
        nl, el, cl = struct.unpack("<HHH", cd[p + 28:p + 34])
        off, = struct.unpack("<I", cd[p + 42:p + 46])
        if cd[p + 46:p + 46 + nl].decode() == member:
            return off, csz, crc, method
        p += 46 + nl + el + cl
    raise RuntimeError(f"GLWD v2 archive: member {member} not found")


def _bigtiff_ifd(f):
    """Tag dict ``tag -> (type, count, values_or_raw)`` of a little-endian BigTIFF."""
    import struct
    f.seek(0)
    head = f.read(16)
    if head[:4] != b"II+\x00":
        raise RuntimeError("GLWD v2 area_pct is no longer a little-endian BigTIFF")
    f.seek(struct.unpack("<Q", head[8:16])[0])
    n, = struct.unpack("<Q", f.read(8))
    entries = [struct.unpack("<HHQ8s", f.read(20)) for _ in range(n)]
    size = {1: 1, 2: 1, 3: 2, 4: 4, 12: 8, 16: 8}
    code = {1: "B", 2: "s", 3: "H", 4: "I", 12: "d", 16: "Q"}
    tags = {}
    for tag, typ, cnt, val in entries:
        nb = size[typ] * cnt
        if nb <= 8:
            raw = val[:nb]
        else:
            f.seek(struct.unpack("<Q", val)[0])
            raw = f.read(nb)
        tags[tag] = (typ, cnt, raw if typ == 2 else struct.unpack(f"<{cnt}{code[typ]}", raw))
    return tags


def _write_window(src, tags, tr0, tr1, tc0, tc1, out):
    """Write tiles rows tr0..tr1-1, cols tc0..tc1-1 of ``src`` as a BigTIFF."""
    import struct
    width, height = tags[256][2][0], tags[257][2][0]
    tw, th = tags[322][2][0], tags[323][2][0]
    across = -(-width // tw)
    offsets, counts = tags[324][2], tags[325][2]
    picked = [r * across + c for r in range(tr0, tr1) for c in range(tc0, tc1)]
    if any(counts[k] == 0 for k in picked):
        raise RuntimeError("GLWD v2 window holds a sparse (empty) tile; the copier "
                           "does not synthesise one")
    sx, sy, _ = tags[33550][2]
    tie = list(tags[33922][2])
    tie[3] += tc0 * tw * sx
    tie[4] -= tr0 * th * sy
    w_out = min(width - tc0 * tw, (tc1 - tc0) * tw)
    h_out = min(height - tr0 * th, (tr1 - tr0) * th)
    body = bytearray(b"II+\x00" + struct.pack("<HHQ", 8, 0, 0))
    new_off = []
    for k in picked:
        src.seek(offsets[k])
        new_off.append(len(body))
        body += src.read(counts[k])
    new_cnt = [counts[k] for k in picked]
    entries = {256: (16, [w_out]), 257: (16, [h_out]), 324: (16, new_off),
               325: (16, new_cnt), 33922: (12, tie)}
    for tag in (258, 259, 262, 277, 284, 317, 322, 323, 339, 33550, 34735, 34736, 34737, 42113):
        if tag in tags and tag not in entries:
            typ, cnt, vals = tags[tag]
            entries[tag] = (typ, vals)
    code = {3: "H", 4: "I", 12: "d", 16: "Q"}
    size = {2: 1, 3: 2, 4: 4, 12: 8, 16: 8}
    packed = []
    for tag in sorted(entries):
        typ, vals = entries[tag]
        raw = vals if typ == 2 else struct.pack(f"<{len(vals)}{code[typ]}", *vals)
        cnt = len(raw) // size[typ]
        if len(raw) <= 8:
            packed.append((tag, typ, cnt, raw.ljust(8, b"\x00")))
        else:
            if len(body) % 2:
                body += b"\x00"
            packed.append((tag, typ, cnt, struct.pack("<Q", len(body))))
            body += raw
    if len(body) % 2:
        body += b"\x00"
    ifd_at = len(body)
    body += struct.pack("<Q", len(packed))
    for tag, typ, cnt, val in packed:
        body += struct.pack("<HHQ", tag, typ, cnt) + val
    body += struct.pack("<Q", 0)
    body[8:16] = struct.pack("<Q", ifd_at)
    part = out.with_suffix(".part")
    part.write_bytes(bytes(body))
    os.replace(part, out)


def glwd_area_pct_window(south, north, west, east, root: Path | None = None):
    """``(path, receipt)`` of a GLWD v2 area_pct GeoTIFF covering the box.

    ``WOOF_GLWD_AREA_PCT`` or an already-extracted global GeoTIFF win; else a
    cached window; else the windowed fetch above.
    """
    import struct
    import tempfile
    import time
    import zlib
    given = os.environ.get("WOOF_GLWD_AREA_PCT")
    if given:
        return Path(given), {"glwd_source": "WOOF_GLWD_AREA_PCT"}
    root = (root or cache_root()) / "glwd-v2"
    full = root / Path(GLWD_MEMBER).name
    if full.is_file():
        return full, {"glwd_source": "cached global GeoTIFF"}
    # GLWD v2 area_pct lattice (TechDoc 4.1; checked against the member's own
    # tags on every fetch below): 15 arc-seconds, 86400 x 33600, 128-px tiles,
    # west -180, north 84.
    res, tile, west0, north0, width, height = 1.0 / 240.0, 128, -180.0, 84.0, 86400, 33600
    step = GLWD_WINDOW_TILES * tile
    across, down = -(-width // tile), -(-height // tile)
    tc0 = max(0, int(((west - west0) / res) // step) * GLWD_WINDOW_TILES)
    tc1 = min(across, (int(((east - west0) / res) // step) + 1) * GLWD_WINDOW_TILES)
    tr0 = max(0, int(((north0 - north) / res) // step) * GLWD_WINDOW_TILES)
    tr1 = min(down, (int(((north0 - south) / res) // step) + 1) * GLWD_WINDOW_TILES)
    if tc0 >= tc1 or tr0 >= tr1:
        return None, {"glwd_source": "domain outside GLWD v2 (84N-56S)"}
    target = root / f"GLWD_v2_0_area_pct.r{tr0}-{tr1}.c{tc0}-{tc1}.tif"
    if target.is_file():
        return target, {"glwd_source": f"cached window {target.name}"}
    root.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    off, csz, crc, method = _zip_member(GLWD_ZIP_URL, GLWD_MEMBER)
    if method != 8:
        raise RuntimeError(f"GLWD v2 member compression {method} is not deflate")
    fetched = 0
    with tempfile.TemporaryFile(dir=root) as spool:
        last_error = None
        for attempt in range(3):
            try:
                spool.seek(0)
                spool.truncate()
                with _range(GLWD_ZIP_URL, off, off + 30 + 65535 * 2 + csz, timeout=300) as r:
                    head = r.read(30)
                    if head[:4] != b"PK\x03\x04":
                        raise RuntimeError("GLWD v2 archive: bad local header")
                    nl, el = struct.unpack("<HH", head[26:30])
                    r.read(nl + el)
                    inflate, left, check, fetched = zlib.decompressobj(-15), csz, 0, 0
                    while left > 0:
                        chunk = r.read(min(8 << 20, left))
                        if not chunk:
                            raise RuntimeError("GLWD v2 member stream ended early")
                        left -= len(chunk)
                        fetched += len(chunk)
                        data = inflate.decompress(chunk)
                        check = zlib.crc32(data, check)
                        spool.write(data)
                    data = inflate.flush()
                    check = zlib.crc32(data, check)
                    spool.write(data)
                if check != crc:
                    raise RuntimeError("GLWD v2 member CRC-32 mismatch")
                break
            except Exception as error:  # network: retry the stream from the start
                last_error = error
        else:
            raise RuntimeError(f"GLWD v2 windowed fetch failed: {last_error}")
        spool.flush()
        tags = _bigtiff_ifd(spool)
        lattice = (tags[256][2][0], tags[257][2][0], tags[322][2][0],
                   tags[33922][2][3], tags[33922][2][4], tags[33550][2][0])
        expect = (width, height, tile, west0, north0, res)
        if lattice[:3] != expect[:3] or any(abs(a - b) > 1e-9 for a, b in zip(lattice[3:], expect[3:])):
            raise RuntimeError(f"GLWD v2 area_pct lattice changed: {lattice}")
        _write_window(spool, tags, tr0, tr1, tc0, tc1, target)
    receipt = {"glwd_source": "windowed fetch", "glwd_window": target.name,
               "glwd_fetched_bytes": fetched, "glwd_window_bytes": target.stat().st_size,
               "glwd_fetch_s": round(time.time() - t0, 1)}
    return target, receipt


__all__ = ["ALGORITHM", "ATTRIBUTION", "build", "glo90_tiles", "glwd_area_pct",
           "glwd_area_pct_window"]
