"""GDAS cycles the AWS archive keeps under a layout the engine does not read.

Door item 11 (an outside runner, gpuwm 2.9.13, 2026-11-11): ``woof fetch
--source gdas --cycle 2021-01-15T00 --hours 0 --readiness`` printed
"readiness waiting; expected ready at 2021-01-15T06:48:00Z" with exit 75,
and the plain fetch waited out its late budget and stopped with exit 75
("launch the same command again once f000 posts").  The archive keeps
that cycle under ``gdas.YYYYMMDD/HH/`` without ``atmos/``, and its pre-v16
analysis has relative humidity, not the specific humidity the GDAS mapping
requires, on pressure levels.  The GDAS row of
``rw-wps-archive-layouts.v1.json`` declares the ``atmos/`` layout from
2021-03-22 12Z, and every door refuses an earlier cycle by name with the
date, before asking a host.

No network: every probe here is a stub.
"""

from datetime import datetime, timedelta

import pytest

from woof import fetch, fetch_endpoints, source_readiness as readiness

NOW = datetime(2026, 11, 12, 3)


def _no_probe(url):
    pytest.fail(f"a cycle no layout serves was probed: {url}")


def test_gdas_row_declares_the_atmos_layout_from_its_first_cycle():
    rows = fetch_endpoints.archive_rows("gdas")
    assert [(row["transport"], row["record_start"], row["cycle_until"]) for row in rows] \
        == [("s3", "2021-03-22T12", None)]
    assert fetch.gfs_object_url(datetime(2021, 3, 22, 12), 0, "gdas", "s3") == (
        "https://noaa-gfs-bdp-pds.s3.amazonaws.com/"
        "gdas.20210322/12/atmos/gdas.t12z.pgrb2.0p25.f000")
    with pytest.raises(ValueError, match="no declared archive layout for 2021-03-22T06"):
        fetch.gfs_object_url(datetime(2021, 3, 22, 6), 0, "gdas", "s3")
    with pytest.raises(ValueError, match="outside the archive's declared forecast ladder"):
        fetch_endpoints.endpoint_named("gdas", "s3").url(
            "gdas.20260101/00/atmos/gdas.t00z.pgrb2.0p25.f010")
    # NOMADS keeps no dated rows: it serves every date it still keeps.
    assert fetch.gfs_object_url(datetime(2026, 11, 11, 0), 0, "gdas", "nomads").endswith(
        "/gdas.20261111/00/atmos/gdas.t00z.pgrb2.0p25.f000")


@pytest.mark.parametrize("cycle", [datetime(2021, 1, 15, 0), datetime(2021, 3, 22, 6),
                                   datetime(2019, 1, 1, 0)])
@pytest.mark.parametrize("as_posted", [True, False])
def test_readiness_refuses_a_gdas_cycle_before_the_layout_by_name(cycle, as_posted):
    document, code = readiness.readiness(
        "gdas", cycle=cycle, hours=0, as_posted=as_posted, probe=_no_probe, now=NOW)
    assert code == readiness.REFUSED_EXIT
    assert document["state"] == "refused"
    refusal = document["refusal"]
    assert f"GDAS cycle {cycle:%Y-%m-%dT%H}Z" in refusal
    assert "s3 from 2021-03-22T12Z" in refusal
    assert "relative humidity" in refusal and "--source gfs" in refusal


def test_a_pinned_archive_refuses_the_same_way_and_nomads_by_retention():
    cycle = datetime(2021, 1, 15, 0)
    document, code = readiness.readiness(
        "gdas", cycle=cycle, hours=0, transport="s3", probe=_no_probe, now=NOW)
    assert code == readiness.REFUSED_EXIT
    assert "s3 from 2021-03-22T12Z" in document["refusal"]
    document, code = readiness.readiness(
        "gdas", cycle=cycle, hours=0, transport="nomads", probe=_no_probe, now=NOW)
    assert code == readiness.REFUSED_EXIT
    assert "nomads" in document["refusal"]


def test_the_first_declared_cycle_is_asked_on_its_atmos_key():
    asked = []

    def probe(url):
        asked.append(url)
        return True

    document, code = readiness.readiness(
        "gdas", cycle=datetime(2021, 3, 22, 12), hours=0, probe=probe, now=NOW)
    assert code == readiness.READY_EXIT, document["refusal"]
    assert document["state"] == "ready"
    assert asked and all("/gdas.20210322/12/atmos/gdas.t12z.pgrb2.0p25.f000" in url
                         for url in asked)


def test_a_missing_archive_object_past_its_late_time_is_refused_not_waited():
    document, code = readiness.readiness(
        "gdas", cycle=datetime(2023, 1, 1, 0), hours=0, probe=lambda url: False, now=NOW)
    assert code == readiness.REFUSED_EXIT
    assert document["state"] == "refused"
    assert "lead 0" in document["refusal"]
    assert "gdas.20230101/00/atmos/" in document["refusal"]
    assert "does not fill a hole by waiting" in document["refusal"]


def test_a_fresh_cycle_the_live_host_still_keeps_is_waited_on():
    cycle = NOW.replace(hour=0) - timedelta(hours=6)
    document, code = readiness.readiness(
        "gdas", cycle=cycle, hours=0, probe=lambda url: False, now=NOW)
    assert code == readiness.NOT_YET_EXIT
    assert document["state"] == "waiting"


def test_a_fresh_cycle_on_a_live_route_with_only_unbounded_hosts_is_waited_on():
    """gfs-pressure (the mapped GFS route) posts live, and its only host keeps
    every cycle (retention_hours null).  A fresh cycle past its late time with
    every probe absent is late, not an archive hole: it still waits, exit 75
    (item 11 review: the first version refused it)."""
    cycle = datetime(2026, 11, 12, 0)
    for asked_at in (datetime(2026, 11, 12, 3, 30), datetime(2026, 11, 12, 12),
                     datetime(2026, 11, 13, 6)):
        document, code = readiness.readiness(
            "gfs-pressure", cycle=cycle, hours=0, probe=lambda url: False,
            now=asked_at)
        assert code == readiness.NOT_YET_EXIT, document.get("refusal")
        assert document["state"] == "waiting"


def test_the_fetch_refuses_before_asking_a_host(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "_head_answer", _no_probe)
    monkeypatch.setattr(fetch, "_head_ok", _no_probe)
    with pytest.raises(ValueError, match=r"GDAS cycle 2021-01-15T00Z is outside every "
                                         r"object layout.*s3 from 2021-03-22T12Z"):
        fetch.fetch_gfs_fullfile(cycle=datetime(2021, 1, 15, 0), hours=(0,), area=None,
                                 out=tmp_path / "out", source="gdas", all_levels=True,
                                 progress=lambda _line: None)
    with pytest.raises(RuntimeError, match="2021-01-15T00Z is outside every object layout"):
        fetch.require_published_cycle("gdas", datetime(2021, 1, 15, 0), 0,
                                      probe=_no_probe, now=NOW)
