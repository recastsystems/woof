"""GFS/GDAS records by .idx from the archive's objects (fix/gfs-aws-record-subset).

The defect: under the hosted AWS policy (WOOF) every GFS start moved whole
pgrb2.0p25 objects.  Measured on a test host for 2026-10-29 12Z f000/f003/f006:
1,556 MB in 70.3 s, of which the 124 records the decode reads are 266.5 MB.
The same records come from the objects by the byte ranges of their .idx;
these tests pin that selection to the grib-filter's own declaration and
run the transfer against a fake host.
"""

from __future__ import annotations

from datetime import datetime
import io
import json

import pytest

from woof import download_budget, fetch, fetch_bars, record_index
from woof.cli import build_parser
from tools import download_gfs_native_subset as gfs_transport

CYCLE = datetime(2026, 7, 28, 6)
#: The hydrometeor mixing ratios GDAS's packaged mapping binds.
HYDROMETEORS = ("CLMR", "ICMR", "RWMR", "SNMR", "GRLE")


def _grib(body: bytes) -> bytes:
    length = 16 + len(body) + 4
    return b"GRIB" + b"\0\0\0\x02" + length.to_bytes(8, "big") + body + b"7777"


def _records(source: str = "gfs") -> list[tuple[str, str]]:
    """An object's (VAR, LEVEL) list in pgrb2.0p25 order, kept and not."""
    moisture = "SPFH" if source == "gdas" else "RH"
    rows = [("PRMSL", "mean sea level")]
    for level in (50, 70) + gfs_transport.PRESSURE_LEVELS_HPA:
        for variable in ("HGT", "TMP", moisture, "VVEL", "UGRD", "VGRD", "ABSV"):
            rows.append((variable, f"{level} mb"))
        if source != "gdas":
            rows.append(("SPFH", f"{level} mb"))
    rows += [("HGT", "surface"), ("PRES", "surface"), ("TMP", "surface"),
             ("WEASD", "surface"), ("WEASD", "surface"), ("SNOD", "surface"),
             ("CAPE", "surface"), ("LAND", "surface"), ("ICEC", "surface"),
             ("TMP", "80 m above ground")]
    for layer in ("0-0.1", "0.1-0.4", "0.4-1", "1-2"):
        rows += [("TSOIL", f"{layer} m below ground"),
                 ("SOILW", f"{layer} m below ground"),
                 ("SOILL", f"{layer} m below ground")]
    rows += [("TMP", "2 m above ground"), (moisture, "2 m above ground"),
             ("UGRD", "10 m above ground"), ("VGRD", "10 m above ground")]
    # The hydrometeors GFS writes on 22 levels (50..1000 hPa, no 70 hPa).
    rows += [(name, f"{level} mb") for level in (50,) + gfs_transport.PRESSURE_LEVELS_HPA
             for name in HYDROMETEORS]
    return rows


def _object(hour: int, source: str = "gfs") -> tuple[bytes, str, list[bytes]]:
    data = b""
    lines = []
    messages = []
    for number, (variable, level) in enumerate(_records(source), 1):
        message = _grib(f"f{hour:03d}:{variable}:{level}".encode() * 4)
        lines.append(f"{number}:{len(data)}:d=2026072806:{variable}:{level}:anl:")
        messages.append(message)
        data += message
    return data, "\n".join(lines) + "\n", messages


class _Reply(io.BytesIO):
    def __init__(self, body, status, headers):
        super().__init__(body)
        self.status = status
        self.headers = headers


def _host(monkeypatch, *, absent_index=(), source="gfs"):
    """Serve every hour's object and .idx at its archive URL."""
    served = {}
    for hour in range(0, 25, 3):
        data, index, _ = _object(hour, source)
        url = fetch.gfs_object_url(CYCLE, hour, source)
        served[url] = data
        served[url + ".idx"] = index.encode()
    asked = []

    def urlopen(request, timeout=None, **_kwargs):
        url = request.full_url
        asked.append((request.get_method(), url, request.headers.get("Range")))
        if url.endswith(".idx"):
            if any(url.endswith(f"f{hour:03d}.idx") for hour in absent_index):
                from urllib.error import HTTPError
                raise HTTPError(url, 404, "Not Found", {}, io.BytesIO(b""))
            return _Reply(served[url], 200, {})
        data = served[url]
        if request.get_method() == "HEAD":
            return _Reply(b"", 200, {"Content-Length": str(len(data))})
        first, last = (int(v) for v in request.headers["Range"].split("=")[1].split("-"))
        return _Reply(data[first:last + 1], 206, {
            "Content-Range": f"bytes {first}-{last}/{len(data)}", "ETag": '"v1"'})

    whole = []

    def download(url, destination, **_kwargs):
        whole.append(url)
        destination.write_bytes(served[url])

    monkeypatch.setattr(record_index, "paced_urlopen", urlopen)
    # The fake host answers on the paced urllib path only: keep the ranges
    # off the persistent http.client connections (fix/hrrr-prs-record-subset),
    # which would otherwise ask the real archive.
    monkeypatch.setattr(record_index, "_reusable", lambda url, opener: False)
    monkeypatch.setattr(gfs_transport, "_download", download)
    monkeypatch.setattr(fetch, "gfs_live_index", lambda cycle, **kwargs: _object(
        0, kwargs.get("source", "gfs"))[1])
    monkeypatch.setattr(fetch, "_gfs_index_record_count",
                        lambda url, **kwargs: len(_records(source)))
    return asked, whole


def _kept(hour: int, source: str = "gfs") -> bytes:
    """The bytes the grib-filter's (variable, level) pairs keep, in file order."""
    _data, _index, messages = _object(hour, source)
    pairs = fetch_bars.nomads_selector_pairs(
        gfs_transport.nomads_variables(source), gfs_transport.NOMADS_LEVELS,
        fetch.container_subset_levels(
            source, available=(50.0, 70.0) + tuple(
                float(level) for level in gfs_transport.PRESSURE_LEVELS_HPA)))
    hydrometeors = source == "gdas"
    return b"".join(message for message, row in zip(messages, _records(source))
                    if row in pairs or (hydrometeors and row[0] in HYDROMETEORS))


@pytest.mark.parametrize("source", fetch.GFS_CONTAINER_SOURCES)
def test_the_selection_is_the_grib_filter_cross_product(source):
    levels = fetch.container_subset_levels(source)
    selection = fetch.gfs_record_selection(source, levels)
    index = _object(0, source)[1]
    pairs = fetch_bars.nomads_selector_pairs(
        gfs_transport.nomads_variables(source), gfs_transport.NOMADS_LEVELS, levels)
    # Two WEASD:surface records (instantaneous and accumulated) are both
    # kept, as the grib-filter keeps both; 50/70 mb, SPFH, VVEL, ABSV,
    # CAPE, SOILL, PRMSL and 80 m TMP are not.  For GDAS the grib-filter
    # asks for the five hydrometeors its mapping binds too (22 levels).
    expected = fetch_bars.count_index_selection(index, pairs)
    hydrometeors = sum(variable in HYDROMETEORS for variable, _level in _records(source)
                       if (variable, _level) in pairs)
    assert hydrometeors == (len(HYDROMETEORS) * 22 if source == "gdas" else 0)
    assert record_index.count_ncep(index.encode(), selection["select"]) == expected
    assert selection["format"] == "ncep-idx"


def test_records_fetch_moves_only_the_grib_filter_records(tmp_path, monkeypatch):
    asked, whole = _host(monkeypatch)
    out = tmp_path / "gfs"
    lines: list[str] = []
    manifest_path = fetch.fetch_gfs_fullfile(
        cycle=CYCLE, hours=(0, 3), area=None, out=out, progress=lines.append,
        records=True)
    assert whole == []
    manifest = json.loads(manifest_path.read_text())
    assert manifest["mode"] == "idx-subset"
    assert manifest["record_selection"]["format"] == "ncep-idx"
    payload = [item for item in manifest["files"] if item["forecast_hour"] is not None]
    assert [item["role"] for item in payload] == ["gfs-records"] * 2
    for item in payload:
        hour = item["forecast_hour"]
        # The records are copied unchanged: the file is exactly the
        # whole object's kept messages, in their order.
        assert (out / item["name"]).read_bytes() == _kept(hour)
        assert item["idx_records"] == item["grib2_messages"] == 125
        assert item["object_bytes"] == len(_object(hour)[0])
    assert all(method != "GET" or rng is not None or url.endswith(".idx")
               for method, url, rng in asked)
    assert any("125 records" in line and "idx-subset" in line for line in lines)

    # A re-run is answered from the receipt: no host is asked again.
    asked.clear()
    fetch.fetch_gfs_fullfile(cycle=CYCLE, hours=(0, 3), area=None, out=out,
                             progress=lines.append, records=True)
    assert asked == []
    # A whole-object request does not resume onto the records folder.
    with pytest.raises(ValueError, match="idx-subset fetch"):
        fetch.fetch_gfs_fullfile(cycle=CYCLE, hours=(0, 3), area=None, out=out,
                                 progress=lines.append)


def test_an_object_whose_host_keeps_no_index_moves_whole_and_says_so(tmp_path, monkeypatch):
    _asked, whole = _host(monkeypatch, absent_index=(3,))
    out = tmp_path / "gfs"
    lines: list[str] = []
    manifest = json.loads(fetch.fetch_gfs_fullfile(
        cycle=CYCLE, hours=(0, 3), area=None, out=out, progress=lines.append,
        records=True).read_text())
    roles = {item["forecast_hour"]: item["role"] for item in manifest["files"]
             if item["forecast_hour"] is not None}
    assert roles == {0: "gfs-records", 3: "gfs-full-file"}
    assert whole == [fetch.gfs_object_url(CYCLE, 3)]
    assert any("keeps no index" in line and "whole object" in line for line in lines)
    assert (out / "gfs.t06z.pgrb2.0p25.f003").read_bytes() == _object(3)[0]


def test_gdas_records_take_every_level_its_grib_filter_takes(tmp_path, monkeypatch):
    """GDAS's mapped preparation reads air temperature up to 1 hPa.

    A selection cut to the declared decode ladder (100 hPa and below) was
    refused at prepare on a test host ("air_temperature vertical coverage
    mismatch"); the grib-filter ladder is every published level.
    """
    _host(monkeypatch, source="gdas")
    out = tmp_path / "gdas"
    manifest = json.loads(fetch.fetch_gfs_fullfile(
        cycle=CYCLE, hours=(0, 3), area=None, out=out, progress=lambda line: None,
        source="gdas", records=True).read_text())
    payload = [item for item in manifest["files"] if item["forecast_hour"] is not None]
    assert [item["role"] for item in payload] == ["gdas-records"] * 2
    for item in payload:
        kept = _kept(item["forecast_hour"], "gdas")
        assert (out / item["name"]).read_bytes() == kept
        assert b"HGT:50 mb" in kept and b"TMP:70 mb" in kept
        # The mapping's hydrometeors ride along: a GDAS start without
        # them was refused ("lacks required fields").
        assert all(f"{name}:500 mb".encode() in kept for name in HYDROMETEORS)
    # GFS decodes through its bridge, which reads the grib-filter's records only.
    gfs = fetch.gfs_record_selection("gfs", fetch.container_subset_levels("gfs"))
    assert len(gfs["select"]) == 1


def _dispatch(tmp_path, monkeypatch, argv, *, policy):
    if policy:
        monkeypatch.setenv(fetch.fetch_endpoints.FETCH_POLICY_ENV, "aws")
    else:
        monkeypatch.delenv(fetch.fetch_endpoints.FETCH_POLICY_ENV, raising=False)
    monkeypatch.setenv("WOOF_FETCH_LOCK_ROOT", str(tmp_path / "locks"))
    monkeypatch.setattr(fetch, "require_published_cycle", lambda *a, **k: None)
    monkeypatch.setattr(fetch, "select_fetch_engine", lambda *a, **k: fetch.FetchEngineChoice(
        engine="python", binary=None, selection="explicit"))
    seen = []

    class Reached(Exception):
        pass

    def transfer(**kwargs):
        seen.append(kwargs.get("records"))
        raise Reached

    monkeypatch.setattr(fetch, "fetch_gfs_fullfile", transfer)
    args = build_parser().parse_args(
        ["fetch", "--source", "gfs", "--cycle", "2026-07-28T06", "--hours", "3",
         "--area", "30,-100,40,-90", "--whole-cycle",
         "--out", str(tmp_path / f"out-{len(argv)}-{policy}")] + argv)
    with pytest.raises(Reached):
        fetch.fetch_main(args)
    return seen


def test_the_hosted_policy_takes_records_and_a_named_full_file_stays_whole(tmp_path, monkeypatch):
    assert _dispatch(tmp_path, monkeypatch, [], policy=True) == [True]
    assert _dispatch(tmp_path, monkeypatch, ["--mode", "full-file"], policy=True) == [False]
    assert _dispatch(tmp_path, monkeypatch, ["--mode", "idx-subset"], policy=False) == [True]


def test_records_are_priced_as_records():
    request = {"source": "gfs", "cycle": "2025-06-01T00", "hours": 24, "cadence": 3,
               "area": "30,-100,40,-90", "mode": "idx-subset"}
    records = download_budget.download_estimate(request)
    whole = download_budget.download_estimate(dict(request, mode="full-file"))
    assert records["mode"] == "idx-subset" and records["objects"] == 9
    # About a sixth of the whole objects (88.5 of 513.1 MB per lead).
    assert 0.15 < records["bytes"] / whole["bytes"] < 0.2
