from datetime import datetime
import pytest
from woof import fetch, fetch_endpoints as endpoints, source_readiness as readiness

def test_archive_routes_preserve_date_resolution_and_lead():
    cycle = datetime(2017, 9, 18, 12)
    expected = {
        "gdex": "https://data.gdex.ucar.edu/d084001/2017/20170918/gfs.0p25.2017091812.f006.grib2",
        "ncei-0p5": "https://www.ncei.noaa.gov/oa/prod-model/global-forecast-system/access/grid-004-0.5-degree/forecast/201709/20170918/gfs_4_20170918_1200_006.grb2",
        "ncei-1p0": "https://www.ncei.noaa.gov/oa/prod-model/global-forecast-system/access/grid-003-1.0-degree/forecast/201709/20170918/gfs_3_20170918_1200_006.grb2",
    }
    for host, url in expected.items():
        assert fetch.gfs_object_url(cycle, 6, transport=host) == url
        assert host in fetch.FETCH_TRANSPORTS
        assert fetch.transport_refusal("gfs", host) is None
    assert [r.name for r in endpoints.serving_ladder("gfs", cycle=cycle)] == list(expected)

def test_aws_layout_boundary():
    assert "/00/gfs.t00z" in fetch.gfs_object_url(datetime(2021, 1, 1), 0, transport="s3")
    assert "/12/atmos/gfs.t12z" in fetch.gfs_object_url(datetime(2021, 3, 22, 12), 0, transport="s3")
    with pytest.raises(ValueError, match="no declared"):
        fetch.gfs_object_url(datetime(2019, 1, 1), 0, transport="s3")

def test_missing_lead_is_not_a_catalogue_success():
    window = readiness.resolve_window("gfs", datetime(2017, 9, 18, 12), 6,
                                      cadence=3, transport="gdex")
    calls = []
    def probe(url):
        calls.append(url)
        return ".f003." not in url
    assert readiness.lead_answer(window, 0, probe=probe)["answer"] == "posted"
    assert readiness.lead_answer(window, 3, probe=probe)["answer"] == "not_posted"
    assert readiness.lead_answer(window, 6, probe=probe)["answer"] == "posted"
    assert len(calls) == 3
    assert all(not url.endswith(".idx") for url in calls)

def test_unsupported_archive_lead_does_not_probe_or_crash():
    window = readiness.Window("gfs", datetime(2017, 9, 18, 12), (1,), 1,
                              transport="gdex")
    def probe(url):
        pytest.fail("undeclared lead must not be probed")
    assert readiness.lead_answer(window, 1, probe=probe)["answer"] == "not_posted"

def test_whole_window_checks_interior_boundaries():
    calls = []
    def probe(url):
        calls.append(url)
        return ".f006." not in url
    document, code = readiness.readiness(
        "gfs", cycle=datetime(2017, 9, 18, 12), hours=12, cadence=3,
        transport="gdex", as_posted=False, probe=probe)
    assert code == readiness.NOT_YET_EXIT
    assert document["state"] != "ready"
    assert {row["lead"] for row in document["start_needs"]} == {0, 3, 6, 9, 12}
    assert len(calls) == 5

def test_go_dry_run_refuses_missing_archive_boundary_before_planning(tmp_path, monkeypatch, capsys):
    from woof import cli, go_cli
    import json
    config = tmp_path / "archive.toml"
    config.write_text('[fetch]\nsource="gfs"\ncycle="2017-09-18T12"\n'
                      'hours=12\ncadence=3\ntransport="gdex"\n')
    monkeypatch.setattr(fetch, "_head_answer", lambda url: ".f006." not in url)
    args = cli.build_parser().parse_args(["go", str(config), "--dry-run"])
    assert go_cli.go_main(args) == 75
    document = json.loads(capsys.readouterr().out)
    assert document["state"] != "ready"
    assert len(document["leads"]) == 5


@pytest.mark.parametrize("route", ["case_data", "prepared_root", "restart", "wps_namelist"])
def test_local_input_dry_run_never_probes_an_unused_archive(monkeypatch, route):
    """An old [fetch] label must not block review of declared or prepared inputs."""
    from types import SimpleNamespace
    from woof import go_cli

    payload = {"fetch": {"source": "gfs", "cycle": "1999-05-03T12", "hours": 6}}
    args = SimpleNamespace(dry_run=True)
    if route == "case_data":
        payload[route] = {"forcing": ["already-on-disk.grb"]}
    else:
        setattr(args, route, "already-on-disk")

    def unused(*args, **kwargs):
        pytest.fail("a local-input review contacted an archive it will not fetch")

    monkeypatch.setattr(go_cli, "_readiness_answer", unused)
    assert go_cli._archive_dry_readiness(args, payload, None) is None


@pytest.mark.parametrize("mutate", [False, True])
def test_archive_rename_preserves_verified_publication(tmp_path, monkeypatch, mutate):
    from woof import rustwx_fetch
    import hashlib
    landed = tmp_path / "gfs.0p25.2017091812.f000.grib2"
    payload = b"archive transfer test"
    landed.write_bytes(payload)
    receipt = {"name": landed.name, "bytes": len(payload),
               "publication": fetch._publication(fetch._stat_identity(landed)),
               "sha256": hashlib.sha256(payload).hexdigest(),
               "grib2_messages": 7, "wall_seconds": 1.0, "source": "gdex",
               "mode": "full-file", "mode_reason": "archive"}
    if mutate:
        landed.write_bytes(b"changed transfer test")
    monkeypatch.setattr(rustwx_fetch, "run_fetch",
                        lambda *a, **kw: {"files": [receipt]})
    kwargs = dict(binary=tmp_path / "rw_fetch", cycle=datetime(2017, 9, 18, 12),
                  hour=0, source="gfs", out=tmp_path, cache_dir=None,
                  progress=lambda _: None, transport="gdex")
    if mutate:
        with pytest.raises(ValueError, match="changed|differs"):
            fetch._rw_fetch_gfs_fullfile(**kwargs)
        assert landed.exists()
    else:
        result = fetch._rw_fetch_gfs_fullfile(**kwargs)
        canonical = tmp_path / "gfs.t12z.pgrb2.0p25.f000"
        assert canonical.read_bytes() == payload
        assert not landed.exists()
        assert result["publication"] == fetch._publication(fetch._stat_identity(canonical))
        monkeypatch.setattr(fetch, "sha256_file", lambda p: pytest.fail("lost receipt"))
        assert fetch._fetched_digest(canonical) == receipt["sha256"]
        assert fetch._fetched_records(canonical) == 7
