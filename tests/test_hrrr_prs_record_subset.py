"""hrrr-prs takes the records its profile binds by the .idx (fix/hrrr-prs-record-subset).

The defect: hrrr-prs moved whole objects.  A wrfprs lead is 381-448 MB and
the packaged profile hrrr-prs-grib2-v1 with its composition reads 184-205 MB
of it; the step-0 wrfsfcf00 is 134-152 MB for its one 1.5 MB VEG record
(measured 2026-10-29).  These tests pin the table rows against real index
lines, and the whole-object fallbacks: an index that lists no record for a
selected field, and NOMADS, which paces every request.
"""

from __future__ import annotations

from datetime import datetime
import io
from pathlib import Path

from woof import fetch_routes, record_index


def _grib(body: bytes) -> bytes:
    length = 16 + len(body) + 4
    return b"GRIB" + b"\0\0\0\x02" + length.to_bytes(8, "big") + body + b"7777"


def _object(records):
    """A GRIB object and its NCEP .idx: one message per (VAR, LEVEL)."""
    data = b""
    idx = []
    for number, (variable, level) in enumerate(records, 1):
        message = _grib(f"{variable}:{level}".encode() * 8)
        idx.append(f"{number}:{len(data)}:d=2026102912:{variable}:{level}:anl:")
        data += message
    return data, "\n".join(idx).encode()


class _Reply(io.BytesIO):
    def __init__(self, body, status, headers):
        super().__init__(body)
        self.status = status
        self.headers = headers


def _host(data, index):
    asked = []

    def opener(request, timeout=None):
        url = request.full_url
        asked.append((request.get_method(), url, request.headers.get("Range")))
        if url.endswith(".idx"):
            return _Reply(index, 200, {})
        if request.get_method() == "HEAD":
            return _Reply(b"", 200, {"Content-Length": str(len(data))})
        first, last = (int(v) for v in request.headers["Range"].split("=")[1].split("-"))
        return _Reply(data[first:last + 1], 206, {
            "Content-Range": f"bytes {first}-{last}/{len(data)}", "ETag": '"v1"'})

    return opener, asked


#: Real lines of the 2026-10-29 12Z wrfprsf00 .idx (AWS archive): one whole
#: pressure level (whose height HRRR writes after the soil column) and the
#: start of the 1013.2 hPa block, the top and bottom soil nodes, and the
#: surface records the profile reads among ones it does not
#: (fix/hrrr-prs-record-subset).
HRRR_WRFPRS_IDX = """532:237474501:d=2026102912:GRLE:975 mb:anl:
533:237476580:d=2026102912:TMP:1000 mb:anl:
534:238120689:d=2026102912:RH:1000 mb:anl:
535:238984130:d=2026102912:DPT:1000 mb:anl:
536:240125806:d=2026102912:SPFH:1000 mb:anl:
537:241665076:d=2026102912:VVEL:1000 mb:anl:
538:242041299:d=2026102912:UGRD:1000 mb:anl:
539:242657355:d=2026102912:VGRD:1000 mb:anl:
540:243272829:d=2026102912:ABSV:1000 mb:anl:
541:243816316:d=2026102912:CLMR:1000 mb:anl:
542:243844164:d=2026102912:CIMIXR:1000 mb:anl:
543:243972075:d=2026102912:RWMR:1000 mb:anl:
544:243992229:d=2026102912:SNMR:1000 mb:anl:
545:243993080:d=2026102912:GRLE:1000 mb:anl:
546:243995358:d=2026102912:HGT:1013.2 mb:anl:
547:244847629:d=2026102912:TMP:1013.2 mb:anl:
560:251417965:d=2026102912:TSOIL:0-0 m below ground:anl:
561:253282387:d=2026102912:SOILW:0-0 m below ground:anl:
576:276273297:d=2026102912:TSOIL:3-3 m below ground:anl:
577:277831199:d=2026102912:SOILW:3-3 m below ground:anl:
578:278664186:d=2026102912:REFC:entire atmosphere:anl:
590:283592112:d=2026102912:HGT:1000 mb:anl:
607:287086273:d=2026102912:PRES:surface:anl:
608:288596038:d=2026102912:HGT:surface:anl:
609:290749733:d=2026102912:TMP:surface:anl:
613:293720202:d=2026102912:WEASD:surface:anl:
615:293922079:d=2026102912:SNOD:surface:anl:
616:294013446:d=2026102912:TMP:2 m above ground:anl:
618:296485220:d=2026102912:SPFH:2 m above ground:anl:
622:301973514:d=2026102912:UGRD:10 m above ground:anl:
623:304116986:d=2026102912:VGRD:10 m above ground:anl:
630:310305610:d=2026102912:WEASD:surface:0-0 day acc fcst:
703:384361455:d=2026102912:LAND:surface:anl:
704:384411931:d=2026102912:ICEC:surface:anl:
708:389695207:d=2026102912:SBT114:top of atmosphere:anl:
"""


def test_the_rows_select_what_the_profile_binds_and_nothing_else():
    plan = fetch_routes.resolve_request("hrrr-prs", cycle=datetime(2026, 10, 29, 12), hours=1,
                                        now=datetime(2026, 10, 29, 14))
    assert [obj.role for obj in plan.objects] == ["wrfprs", "vegetation_surface", "wrfprs"]
    assert all(obj.record_index and obj.index_key.endswith(".grib2.idx") for obj in plan.objects)
    assert fetch_routes.resolve_mode("hrrr-prs", "idx-subset") == "idx-subset"
    wrfprs, vegetation = fetch_routes.route_for("hrrr-prs").files
    ranges, _ = record_index.select_ncep(HRRR_WRFPRS_IDX.encode(),
                                         wrfprs.record_index["select"], 391_989_240)
    names = {line.split(":")[1]: line.split(":")[3] + ":" + line.split(":")[4]
             for line in HRRR_WRFPRS_IDX.splitlines()}
    kept = {names[str(offset)] for offset, _length in ranges}
    assert kept == {
        "GRLE:975 mb", "HGT:1000 mb", "TMP:1000 mb", "SPFH:1000 mb", "UGRD:1000 mb",
        "VGRD:1000 mb", "CLMR:1000 mb", "CIMIXR:1000 mb", "RWMR:1000 mb", "SNMR:1000 mb",
        "GRLE:1000 mb", "TSOIL:0-0 m below ground", "SOILW:0-0 m below ground",
        "TSOIL:3-3 m below ground", "SOILW:3-3 m below ground",
        "PRES:surface", "HGT:surface", "TMP:surface", "TMP:2 m above ground",
        "SPFH:2 m above ground", "UGRD:10 m above ground", "VGRD:10 m above ground",
        "LAND:surface", "ICEC:surface", "WEASD:surface", "SNOD:surface"}
    # Both WEASD records (the profile's instantaneous one and the 0-0 day
    # accumulation beside it) share VAR and LEVEL: 27 records for 26 names.
    assert len(ranges) == 27
    # An index that names cloud ice as HRRR did before July 2018 still selects.
    old = HRRR_WRFPRS_IDX.replace("CIMIXR", "CICE").encode()
    assert len(record_index.select_ncep(old, wrfprs.record_index["select"], 391_989_240)[0])         == len(ranges)
    # The vegetation file: VEG, with LAND so a pre-HRRRv4 index (no VEG) still matches.
    assert vegetation.record_index["select"] == [["VEG|LAND", "surface"]]


def _download(plan, obj, tmp_path, opener):
    whole = []

    def fetch(url, dest, **_):
        whole.append(url)
        dest.write_bytes(_grib(b"w" * 40))
        return {"name": dest.name, "bytes": dest.stat().st_size, "url": url}

    lines = []
    entry = fetch_routes._download_along_ladder(
        plan, obj, tmp_path / Path(obj.relpath).name, magic="GRIB", fetch=fetch,
        opener=opener, progress=lines.append, ladder=plan.ladder)
    return entry, whole, lines


def test_an_index_with_no_record_for_a_field_and_a_paced_host_move_the_object_whole(tmp_path):
    """A selection the index cannot serve fails no healthy object, on any endpoint."""
    assert issubclass(record_index.SelectionAbsent, record_index.IndexAbsent)
    assert issubclass(record_index.SelectionAbsent, ValueError)
    cycle, now = datetime(2026, 10, 29, 12), datetime(2026, 10, 29, 14)
    plan = fetch_routes.resolve_request("hrrr-prs", cycle=cycle, hours=0, host="aws", now=now)
    obj = next(obj for obj in plan.objects if obj.role == "vegetation_surface")
    data, idx = _object([("ICEC", "surface"), ("SNOD", "surface")])
    opener, asked = _host(data, idx)
    entry, whole, lines = _download(plan, obj, tmp_path, opener)
    assert whole == [obj.url] and entry["endpoint"] == "aws"
    assert [method for method, *_ in asked] == ["GET"]          # the index alone
    assert any("cannot serve the record selection" in line and "VEG|LAND" in line
               for line in lines)

    # The same index with a VEG record: one range, nothing whole.
    data, idx = _object([("ICEC", "surface"), ("VEG", "surface"), ("SNOD", "surface")])
    opener, asked = _host(data, idx)
    entry, whole, _ = _download(plan, obj, tmp_path, opener)
    assert not whole and entry["selected_records"] == 1

    nomads = fetch_routes.resolve_request("hrrr-prs", cycle=cycle, hours=0, host="nomads",
                                          now=now)
    obj = next(obj for obj in nomads.objects if obj.role == "vegetation_surface")
    asked.clear()
    entry, whole, _ = _download(nomads, obj, tmp_path, opener)
    assert whole == [obj.url] and not asked


def test_a_reset_range_is_asked_again_before_it_fails_the_object(tmp_path, monkeypatch):
    """One reset among a wrfprs selection's 127 ranges sent the whole object
    to NOMADS (412 MB instead of 186 MB, test host 2026-10-29)."""
    monkeypatch.setattr(record_index.time, "sleep", lambda _seconds: None)
    data, idx = _object([("TMP", "500 mb"), ("RH", "500 mb"), ("SPFH", "500 mb")])
    opener, asked = _host(data, idx)
    declaration = record_index.validate_declaration(
        {"format": "ncep-idx", "select": [["TMP|SPFH", "500 mb"]]}, "t")

    def flaky(resets):
        left = {"n": resets}

        def ask(request, timeout=None):
            if "Range" in request.headers and request.headers["Range"].startswith("bytes=0-") \
                    and left["n"]:
                left["n"] -= 1
                raise ConnectionResetError(104, "Connection reset by peer")
            return opener(request, timeout)
        return ask

    entry = record_index.download_records(
        "https://h/hrrr.wrfprsf01.grib2", "https://h/hrrr.wrfprsf01.grib2.idx",
        tmp_path / "o.grib2", declaration=declaration, opener=flaky(2), streams=1)
    assert entry["selected_records"] == 2
    assert record_index.grib_messages(tmp_path / "o.grib2") == 2

    asked.clear()
    try:
        record_index.download_records(
            "https://h/hrrr.wrfprsf01.grib2", "https://h/hrrr.wrfprsf01.grib2.idx",
            tmp_path / "p.grib2", declaration=declaration, opener=flaky(9), streams=1)
    except ConnectionResetError:
        pass
    else:
        raise AssertionError("a range that resets every time must fail the object")
    assert not (tmp_path / "p.grib2").exists()
    assert not (tmp_path / "p.grib2.part").exists()


def test_the_ranges_ride_one_persistent_connection_per_stream(tmp_path, monkeypatch):
    """A fresh connection per range moved a wrfprs selection's 127 ranges at
    7.5-9 MB/s where the whole object moved at 23 MB/s (test host, 2026-10-29);
    persistent connections moved them at 23 MB/s.  Same bytes either way."""
    import http.server
    import re as _re
    import threading

    monkeypatch.setattr(record_index, "getproxies", lambda: {})
    records = [(name, f"{level} mb") for level in range(100, 1000, 25)
               for name in ("TMP", "RH")]
    data, idx = _object(records)
    seen = {"connections": 0, "ranges": 0}

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def setup(self):
            super().setup()
            seen["connections"] += 1

        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path.endswith(".idx"):
                body, status, extra = idx, 200, {}
            else:
                first, last = (int(v) for v in _re.fullmatch(
                    r"bytes=(\d+)-(\d+)", self.headers["Range"]).groups())
                body, status = data[first:last + 1], 206
                extra = {"Content-Range": f"bytes {first}-{last}/{len(data)}", "ETag": '"v1"'}
                seen["ranges"] += 1
            self.send_response(status)
            for key, value in extra.items():
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_HEAD(self):
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}/hrrr.t12z.wrfprsf00.grib2"
        # Every other record: 36 separate ranges, as a wrfprs selection is.
        declaration = record_index.validate_declaration(
            {"format": "ncep-idx", "select": [["TMP", "[0-9]+ mb"]]}, "t")
        entry = record_index.download_records(base, base + ".idx", tmp_path / "o.grib2",
                                              declaration=declaration, streams=4)
    finally:
        server.shutdown()
        server.server_close()
    want = b"".join(_grib(f"TMP:{level} mb".encode() * 8) for level in range(100, 1000, 25))
    assert (tmp_path / "o.grib2").read_bytes() == want
    assert entry["selected_records"] == 36 and seen["ranges"] == 36
    # The index on its own connection, then at most one connection per stream.
    assert seen["connections"] <= 1 + 4
