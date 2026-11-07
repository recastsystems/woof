"""A mirror that ignores Range, or an index out of step, never fails a start.

The defect (WOOF incident 2026-11-04, fix/ecmwf-open-data-range): a
customer start from the 2026-10-21 06Z IFS 0.1-degree cycle failed after
41 s with "fetch ecmwf-open-data-0p1: 20261021060000-12h-oper-fc-pl.grib2:
every endpoint refused in 5 rounds ... aws: the object did not verify --
the host ignored or changed the requested byte range 62979143-69846560
(asked 5 times)".  Only the AWS mirror still kept that cycle and the whole
object was there.  A host that answers a range with anything else now has
the object moved whole on one request and the selected records cut from
it (the same bytes the ranged transfer writes); an index whose offsets do
not land on the object's messages moves the object whole.
"""

from __future__ import annotations

from datetime import datetime
import hashlib
import io
import json

import pytest

from woof import fetch_ranges, fetch_routes, record_index


def _grib(body: bytes) -> bytes:
    length = 16 + len(body) + 4
    return b"GRIB" + b"\0\0\0\x02" + length.to_bytes(8, "big") + body + b"7777"


RECORDS = [("pl", "z", "500"), ("pl", "t", "500"), ("pl", "u", "500"),
           ("pl", "w", "500"), ("sfc", "msl", ""), ("sol", "sot", "1")]

SELECT = record_index.validate_declaration(
    {"format": "ecmwf-jsonl", "why": "test", "select": [
        {"levtype": "pl", "param": ["t", "u"]}, {"levtype": "sol", "param": ["sot"]}]}, "t")


def _object(shift: int = 0):
    """A GRIB object and its JSON-lines index (offsets moved by ``shift``)."""
    data = b""
    rows = []
    for levtype, param, level in RECORDS:
        message = _grib(f"{levtype}:{param}:{level}".encode() * 8)
        rows.append({"levtype": levtype, "param": param, "levelist": level,
                     "_offset": len(data) + shift, "_length": len(message)})
        data += message
    return data, "\n".join(json.dumps(row) for row in rows).encode()


def _wanted(data: bytes) -> bytes:
    out = b""
    offset = 0
    for levtype, param, _level in RECORDS:
        length = int.from_bytes(data[offset + 8:offset + 16], "big")
        if param in ("t", "u", "sot"):
            out += data[offset:offset + length]
        offset += length
    return out


class _Reply(io.BytesIO):
    def __init__(self, body, status, headers):
        super().__init__(body)
        self.status = status
        self.headers = headers


def _host(data, index, *, ranges="honour"):
    """A fake host: ``honour`` ranges, ``ignore`` them (200 whole) or ``shift`` them."""
    asked = []

    def opener(request, timeout=None):
        url = request.full_url
        wanted = request.headers.get("Range")
        asked.append((request.get_method(), url, wanted))
        if url.endswith(".index"):
            return _Reply(index, 200, {})
        whole = {"Content-Length": str(len(data)), "ETag": '"v1"',
                 "Accept-Ranges": "bytes"}
        if wanted is None or ranges == "ignore":
            return _Reply(data, 200, whole)
        first, last = (int(v) for v in wanted.split("=")[1].split("-"))
        last = min(last, len(data) - 1)  # as S3 answers a range past the end
        if ranges == "shift":
            first, last = first + 1, min(last + 1, len(data) - 1)
        return _Reply(data[first:last + 1], 206, {
            "Content-Range": f"bytes {first}-{last}/{len(data)}", "ETag": '"v1"'})

    return opener, asked


@pytest.mark.parametrize("ranges", ["ignore", "shift"])
def test_a_host_that_ignores_or_moves_a_range_has_the_records_cut_locally(tmp_path, ranges):
    data, index = _object()
    opener, asked = _host(data, index, ranges=ranges)
    said = []
    dest = tmp_path / "x-pl.grib2"
    entry = record_index.download_records(
        "https://h/x-pl.grib2", "https://h/x-pl.index", dest, declaration=SELECT,
        opener=opener, progress=said.append)
    # Byte for byte what a ranged transfer that honours the range writes.
    faithful, asked_faithful = _host(data, index)
    reference = tmp_path / "ref.grib2"
    record_index.download_records("https://h/x-pl.grib2", "https://h/x-pl.index", reference,
                                  declaration=SELECT, opener=faithful)
    assert dest.read_bytes() == reference.read_bytes() == _wanted(data)
    assert entry["sha256"] == hashlib.sha256(_wanted(data)).hexdigest()
    assert entry["selected_records"] == 3 and entry["object_bytes"] == len(data)
    assert "ignored or changed the requested byte range" in entry["range_fallback"]
    # One plain GET moved the object; no leftovers beside the file.
    assert [method for method, url, rng in asked if url.endswith(".grib2") and rng is None] \
        == ["GET"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ref.grib2", "x-pl.grib2"]
    assert any("moving the whole object" in line for line in said)


def test_a_stale_index_is_named_so_the_object_moves_whole(tmp_path):
    data, index = _object(shift=3)
    for ranges in ("honour", "ignore"):
        opener, _ = _host(data, index, ranges=ranges)
        with pytest.raises(record_index.IndexStale):
            record_index.download_records(
                "https://h/x-pl.grib2", "https://h/x-pl.index", tmp_path / "o.grib2",
                declaration=SELECT, opener=opener)
        assert list(tmp_path.iterdir()) == []


def _ifs_pl_object():
    plan = fetch_routes.resolve_request("ecmwf-open-data-0p1", cycle=datetime(2026, 10, 21, 6),
                                        hours=12, host="aws")
    (obj,) = [o for o in plan.objects if o.relpath.endswith("12h-oper-fc-pl.grib2")]
    return plan, obj


@pytest.mark.parametrize("shift,whole_moves", [(0, 0), (3, 1)])
def test_the_incident_object_lands_from_a_mirror_that_ignores_range(tmp_path, shift,
                                                                    whole_moves):
    """The exact customer start: the 0.1-degree pl object of 2026-10-21 06Z +12 h on AWS."""
    from woof.fetch_endpoints import Endpoint

    plan, obj = _ifs_pl_object()
    assert obj.record_index is not None
    data, index = _object(shift=shift)
    opener, asked = _host(data, index, ranges="ignore")
    moved = []

    def whole(url, dest, **kw):
        moved.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return {"name": dest.name, "bytes": len(data), "url": url}

    aws = Endpoint(name="aws", base="https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com",
                   retention_hours=None, why="test")
    said = []
    entry = fetch_routes._download_along_ladder(
        plan, obj, tmp_path / obj.relpath, magic="GRIB", fetch=whole, opener=opener,
        progress=said.append, ladder=(aws,))
    assert entry["endpoint"] == "aws"
    assert len(moved) == whole_moves
    landed = (tmp_path / obj.relpath).read_bytes()
    assert landed == (_wanted(data) if shift == 0 else data)
    assert all(url.startswith(aws.base) for _m, url, _r in asked)
    assert not any("refused" in line for line in said)


def test_a_whole_object_transfer_falls_back_to_one_stream(tmp_path, monkeypatch):
    """The parallel whole-object path: ranges refused mid-transfer, one plain stream instead."""
    monkeypatch.setattr(fetch_ranges, "MIN_PART", 64)
    data = bytes(range(256)) * 4
    calls = []

    def opener(request, timeout=None):
        calls.append(request.headers.get("Range"))
        return _Reply(data, 200, {"Content-Length": str(len(data)), "ETag": '"v1"',
                                  "Accept-Ranges": "bytes"})

    part = tmp_path / "o.grib2.part"
    written, digest = fetch_ranges.download("https://h/o.grib2", part, user_agent="t",
                                            streams=4, opener=opener)
    assert written == len(data) and part.read_bytes() == data
    assert digest == hashlib.sha256(data).hexdigest()
    assert calls[0] is None and calls[-1] is None and any(calls[1:-1])


def test_a_server_fault_on_a_range_is_a_fault_not_a_refused_range(tmp_path, monkeypatch):
    """S3's 503 SlowDown on the persistent-connection path is retried, never a 'refused range'."""
    from urllib.error import HTTPError

    from woof.fetch_endpoints import Endpoint

    data, index = _object()
    opener, asked = _host(data, index)
    faults = {"left": 1}

    def throttled(request, timeout=None):
        if request.headers.get("Range") and faults["left"]:
            faults["left"] -= 1
            return _Reply(b"<Error><Code>SlowDown</Code></Error>", 503, {})
        return opener(request, timeout)

    with pytest.raises(HTTPError) as caught:
        record_index.download_records("https://h/x-pl.grib2", "https://h/x-pl.index",
                                      tmp_path / "o.grib2", declaration=SELECT,
                                      opener=throttled, streams=1)
    assert caught.value.code == 503
    assert not any(rng is None and url.endswith(".grib2") for _m, url, rng in asked)
    # Along the ladder the fault is asked again and the records land.
    faults["left"] = 1
    monkeypatch.setattr(fetch_routes.fetch_pool, "sleep_unless_stopped",
                        lambda seconds, sleep=None: None)
    plan, obj = _ifs_pl_object()
    aws = Endpoint(name="aws", base="https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com",
                   retention_hours=None, why="test")
    entry = fetch_routes._download_along_ladder(
        plan, obj, tmp_path / obj.relpath, magic="GRIB", fetch=None, opener=throttled,
        progress=lambda *_: None, ladder=(aws,))
    assert entry["endpoint"] == "aws" and "range_fallback" not in entry
    assert (tmp_path / obj.relpath).read_bytes() == _wanted(data)
