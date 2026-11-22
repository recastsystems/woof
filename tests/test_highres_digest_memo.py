"""A cached high-resolution payload is read for its digest once per process.

Breakage it prevents (WOOF gate t2rect-gate1, 2026-11-11): every build of a
sub-km grid's land cover read the 2.28 GB CGLC-MODIS-LCZ GeoTIFF in full three
times to check it against its sidecar and pins (4.2 s of a 5.1 s build on
an RTX 4090 test host), and a domain tree builds each grid twice.
"""
from __future__ import annotations

import hashlib
import json
import os

from woof.static import highres_fetch as hf


def _cached(tmp_path, body: bytes):
    path = tmp_path / "raster.tif"
    path.write_bytes(body)
    hf._sidecar(path).write_text(json.dumps({
        "url": "https://example.invalid/raster.tif",
        "sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body),
        "fetched_utc": "2026-11-11T00:00:00+00:00"}), encoding="utf-8")
    return path


def test_a_cached_payload_is_hashed_once_per_process(tmp_path, monkeypatch):
    calls = []
    real = hf.sha256_file
    monkeypatch.setattr(hf, "sha256_file",
                        lambda p: calls.append(p) or real(p))
    monkeypatch.setattr(hf, "_VERIFIED_DIGESTS", {})
    body = b"x" * 4096
    path = _cached(tmp_path, body)
    md5 = hashlib.md5(body, usedforsecurity=False).hexdigest()
    for _ in range(3):
        assert hf._read_sidecar(path) is not None
        assert hf._md5_file(path) == md5
    assert len(calls) == 1


def test_a_changed_payload_is_read_again_and_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(hf, "_VERIFIED_DIGESTS", {})
    path = _cached(tmp_path, b"a" * 4096)
    assert hf._read_sidecar(path) is not None
    stat = path.stat()
    path.write_bytes(b"b" * 4096)  # same size, new content
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    assert hf._read_sidecar(path) is None
