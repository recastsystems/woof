"""A run's download and its preparation are priced before it starts, everywhere its disk is priced.

Breakage these prevent, all measured on a development machine on 2026-09-26:

* the review estimate said download "no [fetch] in this plan" with null
  bytes, and null disk bytes, for an 18 hour HRRR plan whose fetch took
  38 files and 22.3 GB and whose run folder reached 58.8 GB; a 24 hour
  GFS plan that downloaded 9 files was told the same;
* the event page projected the 2021-12-10 tornado's 12 GB layout at
  15.2 GiB, picked it for a disk with 24 GiB free, and the run had
  written 28.9 GB by its first forecast step (18.9 GB of HRRR files and
  9.6 GB of preparation), taking the shared disk from 25 GB free to
  5.9 GB free;
* the same page promised 15 minutes to all pictures, 10 of them the
  forecast, for a run whose download and preparation alone took 707 s,
  because every unmeasured row was given the two minute mean of the
  measured ERA5 runs whatever its own source and size;
* a GFS start older than the grib-filter host keeps was priced as 9
  crops, 34 MB, while the fetch read it as 9 whole archive objects,
  4.6 GB, and a folder named by hand that held unrelated files priced
  its download at nothing;
* a GEFS fetch left 680 MB on disk and a GDPS fetch 1.1 GB, each priced
  at half of that, because the composed files those routes write beside
  the objects they keep were not counted.
"""
from __future__ import annotations

import functools
import json
import shutil
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from woof import disk_budget, download_budget, fetch, fetch_routes

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures" / "download_budget"
GIB = 1024 ** 3


@pytest.mark.parametrize("cycle", ["latest", None])
def test_ifs_current_download_is_priced(cycle):
    request = {"source": "ifs", "hours": 24}
    if cycle is not None:
        request["cycle"] = cycle
    # The IFS window (9 leads, each object's mapped records by its .index)
    # plus its declared sea-ice donor: the same cycle's GFS f000 ICEC record
    # by the .idx (ECMWF open data publishes sea-ice thickness only).
    assert download_budget.download_estimate(request)["bytes"] == 9 * 37_500_000 + GFS_ICEC_RECORD

#: The EF4 2021 run of the 12 GB layout: du of its run folder by part at
#: the first forecast step (downloads, chain/hrrr-root-prep, chain/run).
EF4_DOWNLOAD, EF4_PREPARATION, EF4_FIRST_FRAME = 18_879_788_730, 9_552_000_842, 547_618_443
EF4_REQUEST = {"source": "hrrr", "cycle": "2021-12-10T18", "hours": 15,
               "area": "29.50,-97.49,43.91,-79.35"}
#: The 18 hour HRRR run's fetch: 38 files, as its stage_finished event reported.
HRRR_FULL_FETCHED = 22_319_909_555


#: The clock the GFS review is read at: the fixture's cycle is a day old,
#: well inside the grib-filter host's rolling window.
REVIEW_NOW = datetime(2026, 9, 26, 12)
#: The whole-object size the full-file transport is priced at, per GFS and GDAS object.
GFS_OBJECT, GDAS_OBJECT = 513_100_000, 483_100_000
#: The grib-filter records the archive transport takes by .idx, per GFS and GDAS object.
GFS_RECORDS, GDAS_RECORDS = 88_500_000, 184_500_000
#: The ICEC:surface record a sea-ice donor takes from the GFS f000 by its .idx.
GFS_ICEC_RECORD = 101_000
GFS_BOX = "17.61,-116.54,53.24,-79.36"


def _clock(monkeypatch, now: datetime) -> None:
    """Read the fetch's own archive question at ``now``."""
    monkeypatch.setattr(fetch, "archive_only_cycle",
                        functools.partial(fetch.archive_only_cycle, now=now))


def _estimate(tmp_path, capsys, monkeypatch, fixture: str, **run_options) -> dict:
    from woof.cli import build_parser
    from woof.runplan import PLAN_SCHEMA, run_plan_main

    monkeypatch.setenv("GPUWM_NO_LOCAL_GPU", "1")
    config = tmp_path / fixture
    shutil.copyfile(FIXTURES / fixture, config)
    plan = tmp_path / "plan.json"
    document = {"schema": PLAN_SCHEMA, "name": "review", "route": "prepared",
                "config": {"path": str(config)}, "output_root": str(tmp_path / "run")}
    if run_options:
        document["run_options"] = run_options
    plan.write_text(json.dumps(document), encoding="utf-8")
    assert run_plan_main(build_parser().parse_args(["run-plan", "--estimate", str(plan)])) == 0
    return json.loads(capsys.readouterr().out)


def test_the_hrrr_review_prices_the_download_its_config_makes(tmp_path, capsys, monkeypatch):
    document = _estimate(tmp_path, capsys, monkeypatch, "central-us-hrrr.toml")
    download = document["download"]
    assert "no [fetch]" not in download["basis"]
    # The fetch default (auto) is priced at its upper bound, the whole objects.
    assert (download["source"], download["mode"]) == ("hrrr", "auto")
    assert "auto is priced as full-file" in download["basis"]
    assert (download["objects"], download["leads"]) == (38, 19)
    assert abs(download["bytes"] - HRRR_FULL_FETCHED) <= 0.25 * HRRR_FULL_FETCHED
    disk = document["disk"]
    assert disk["download_bytes"] == download["bytes"]
    assert disk["preparation_bytes"] > 0
    assert disk["bytes"] == (disk["download_bytes"] + disk["preparation_bytes"] + disk["history_bytes"]
                             + disk["checkpoint_bytes"] + disk["picture_bytes"])
    assert disk["unpriced"] == []


def test_the_gfs_review_says_it_downloads(tmp_path, capsys, monkeypatch):
    _clock(monkeypatch, REVIEW_NOW)
    document = _estimate(tmp_path, capsys, monkeypatch, "gfs-3km.toml")
    download = document["download"]
    assert "no [fetch]" not in download["basis"]
    assert (download["source"], download["mode"]) == ("gfs", "grib-filter")
    assert (download["objects"], download["leads"]) == (9, 9)
    # The run fetched 9 cropped files, about 34 MB.
    assert 0.75 * 34e6 <= download["bytes"] <= 1.25 * 34e6
    assert document["disk"]["bytes"] > download["bytes"]


def test_an_archived_gfs_start_is_priced_as_the_records_the_fetch_reads(monkeypatch):
    """The reviewed request with its cycle 16 days old: the fetch read 9 whole archive objects, 4.62 GB.

    Since fix/gfs-aws-record-subset it reads only the grib-filter's records
    of those 9 objects by their .idx, about 0.8 GB.
    """
    _clock(monkeypatch, REVIEW_NOW)
    request = {"source": "gfs", "cycle": "2026-09-10T06", "hours": 24, "cadence": 3, "area": GFS_BOX}
    estimate = download_budget.download_estimate(request)
    assert (estimate["mode"], estimate["objects"], estimate["leads"]) == ("idx-subset", 9, 9)
    assert estimate["bytes"] == 9 * GFS_RECORDS
    assert "older than the grib-filter host keeps" in estimate["basis"]
    # The same request a day old is the crop it always was.
    recent = download_budget.download_estimate(dict(request, cycle="2026-09-26T06"))
    assert recent["mode"] == "grib-filter" and recent["bytes"] < 50e6
    # A named mode is the request's own choice and is kept whatever the cycle's age.
    named = download_budget.download_estimate(dict(request, mode="full-file"))
    assert named["mode"] == "full-file" and "older than" not in named["basis"]


@pytest.mark.parametrize("source, size", [("gfs", GFS_RECORDS), ("gdas", GDAS_RECORDS)])
def test_a_far_past_container_cycle_is_priced_from_the_archive(source, size):
    """No clock pin needed: 2025-06-01 is past every rolling window; the fetch reads 3 objects' records."""
    request = {"source": source, "cycle": "2025-06-01T00", "hours": 6, "cadence": 3, "area": GFS_BOX}
    estimate = download_budget.download_estimate(request)
    assert (estimate["mode"], estimate["objects"]) == ("idx-subset", 3)
    assert estimate["bytes"] == 3 * size
    whole = download_budget.download_estimate(dict(request, mode="full-file"))
    assert whole["bytes"] == 3 * (GFS_OBJECT if source == "gfs" else GDAS_OBJECT)


def test_the_estimate_asks_the_fetchs_own_archive_question(monkeypatch):
    """One authority: the answer follows fetch.archive_only_cycle, not a copy of its retention."""
    request = {"source": "gdas", "cycle": "2026-09-26T00", "hours": 6, "cadence": 3, "area": GFS_BOX}
    asked = []
    monkeypatch.setattr(fetch, "archive_only_cycle",
                        lambda source, cycle, now=None: asked.append((source, cycle)) or True)
    assert download_budget.download_estimate(request)["mode"] == "idx-subset"
    assert asked and asked[0] == ("gdas", datetime(2026, 9, 26, 0))
    monkeypatch.setattr(fetch, "archive_only_cycle", lambda source, cycle, now=None: False)
    assert download_budget.download_estimate(request)["mode"] == "grib-filter"


def _ef4_layout(hours: int = 15):
    from woof.config import RunConfig

    # A190 replaces the full-history constant with writer inventory. This
    # saved moist forecast must retain its actual physics in the fixture.
    domain = SimpleNamespace(grid_id=1, history_interval_s=3600.0,
                             run=RunConfig(nx=386, ny=374, nz=49, spec_bdy_width=5,
                                           dx=3000, dy=3000, ztop=20000, dt=15,
                                           run_seconds=hours * 3600,
                                           moist=True, mp_physics=8,
                                           sf_sfclay_physics=91, sf_surface_physics=2,
                                           bl_pbl_physics=1, ra_lw_physics=4,
                                           ra_sw_physics=4, nwp_diagnostics=1))
    return SimpleNamespace(run_seconds=hours * 3600.0, restart_interval_s=3600.0, domains=(domain,))


def _ef4_floor() -> float:
    """What the run had written at its first step, plus the 15 frames and all pictures still to come."""
    return (EF4_DOWNLOAD + EF4_PREPARATION + EF4_FIRST_FRAME + 15 * EF4_FIRST_FRAME
            + disk_budget.projected_picture_bytes(386, 374, 15 * 3600, 3600))


@pytest.mark.usefixtures("classic_history_frames")
def test_the_projection_counts_what_the_ef4_run_wrote_before_its_first_step():
    p = disk_budget.projected_run_bytes(_ef4_layout(), keep_checkpoints=1, fetch=EF4_REQUEST,
                                        chain="prepared:hrrr", render=True)
    assert p["unpriced"] == []
    assert p["download_bytes"] >= 0.95 * EF4_DOWNLOAD
    assert p["preparation_bytes"] >= EF4_PREPARATION
    assert p["total_bytes"] >= _ef4_floor()
    # The figure the page used to compare with free disk left both out.
    assert p["total_bytes"] - p["download_bytes"] - p["preparation_bytes"] < 16.5e9


def test_every_route_and_transport_a_fetch_defaults_to_is_priced():
    """A source whose download is left out of every projection is the defect; adding one is a table row."""
    for source in fetch_routes.route_ids():
        route = fetch_routes.route_for(source)
        request = {"source": source, "cycle": None, "hours": route.default_cadence,
                   "cadence": route.default_cadence}
        estimate = download_budget.download_estimate(request)
        assert estimate["bytes"] is not None, (source, estimate["basis"])
    for source, legacy in download_budget.table()["legacy_sources"].items():
        for mode in (legacy["default_mode"], legacy.get("archive_mode")):
            if mode is None:
                continue
            request = {"source": source, "cycle": None, "hours": 6, "cadence": 6,
                       "area": "30,-100,40,-90", legacy["mode_key"]: mode}
            estimate = download_budget.download_estimate(request)
            assert estimate["bytes"] is not None, (source, mode, estimate["basis"])


def test_a_request_with_no_row_is_named_not_guessed(monkeypatch):
    table = dict(download_budget.table())
    table["downloads"] = [row for row in table["downloads"] if row["source"] != "rap"]
    monkeypatch.setattr(download_budget, "table", lambda: table)
    estimate = download_budget.download_estimate({"source": "rap", "cycle": "2026-09-26T00", "hours": 6})
    assert estimate["bytes"] is None and "no measured size for rap" in estimate["basis"]
    p = disk_budget.projected_run_bytes(_ef4_layout(), keep_checkpoints=1,
                                        fetch={"source": "rap", "cycle": "2026-09-26T00", "hours": 6},
                                        chain="prepared:staged", render=True)
    assert p["unpriced"] == ["download"]


def test_local_inputs_download_nothing_but_are_still_prepared():
    request = dict(EF4_REQUEST, source_root="/data/hrrr")
    estimate = download_budget.download_estimate(request)
    assert estimate["bytes"] == 0 and estimate["leads"] == 16
    p = disk_budget.projected_run_bytes(_ef4_layout(), keep_checkpoints=1, fetch=request,
                                        chain="prepared:hrrr", render=True)
    assert p["download_bytes"] == 0 and p["preparation_bytes"] >= EF4_PREPARATION


def test_a_download_already_on_disk_is_not_counted_again():
    full = disk_budget.projected_run_bytes(_ef4_layout(), keep_checkpoints=1, fetch=EF4_REQUEST,
                                           chain="prepared:hrrr", render=True)
    reused = disk_budget.projected_run_bytes(_ef4_layout(), keep_checkpoints=1, fetch=EF4_REQUEST,
                                             chain="prepared:hrrr", render=True,
                                             download_present_bytes=EF4_DOWNLOAD * 2)
    assert reused["download_bytes"] == 0
    assert full["total_bytes"] - reused["total_bytes"] == full["download_bytes"]


def test_the_refusal_names_the_download_and_the_preparation():
    p = disk_budget.projected_run_bytes(_ef4_layout(), keep_checkpoints=1, fetch=EF4_REQUEST,
                                        chain="prepared:hrrr", render=True)
    words = disk_budget.disk_refusal(p, 24 * GIB)
    assert "of download" in words and "of preparation" in words and "24.0 GiB free" in words
    # A download that lands on another disk is compared with that disk alone.
    assert disk_budget.disk_refusal(p, p["total_bytes"] - p["download_bytes"],
                                    download_free=p["download_bytes"]) is None
    words = disk_budget.disk_refusal(p, 10 ** 15, download_free=p["download_bytes"] - 1)
    assert "downloads about" in words


def test_run_plan_refuses_before_the_download_a_run_whose_download_does_not_fit(tmp_path, monkeypatch):
    """Between the old projection (history, checkpoints, pictures) and the whole one: refused, nothing fetched."""
    from woof import capabilities
    import woof.runplan as runplan
    from woof.runplan import EVENTS_FILENAME, PLAN_SCHEMA, EventStream, execute_plan, load_plan, read_events

    monkeypatch.setattr(capabilities, "require", lambda *args, **kwargs: None)
    fetched = []
    monkeypatch.setattr(runplan, "_run_fetch", lambda *args, **kwargs: fetched.append(args))
    monkeypatch.setattr(runplan, "_hrrr_chain", lambda *args, **kwargs: fetched.append("chain"))
    monkeypatch.setattr(disk_budget, "free_bytes", lambda path: 30 * GIB)
    config = tmp_path / "central-us-hrrr.toml"
    shutil.copyfile(FIXTURES / "central-us-hrrr.toml", config)
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({"schema": PLAN_SCHEMA, "name": "refuse", "route": "prepared",
                                "config": {"path": str(config)},
                                "output_root": str(tmp_path / "run")}), encoding="utf-8")
    plan = load_plan(path)
    plan.run_dir.mkdir(parents=True, exist_ok=True)
    with EventStream(plan.run_dir / EVENTS_FILENAME, mirror=None) as events:
        assert execute_plan(plan, events=events) == 1
    failed = read_events(plan.run_dir / EVENTS_FILENAME)[-1]
    assert failed["event"] == "failed"
    assert "of download" in failed["message"] and "Refused before the download" in failed["message"]
    assert fetched == []


def test_run_plan_refuses_before_the_download_an_archived_gfs_start_that_does_not_fit(
        tmp_path, capsys, monkeypatch):
    """Free disk the crop price fitted and the archive objects do not: refused, nothing fetched."""
    from woof import capabilities
    import woof.runplan as runplan
    from woof.runplan import EVENTS_FILENAME, PLAN_SCHEMA, EventStream, execute_plan, load_plan, read_events

    _clock(monkeypatch, REVIEW_NOW)
    text = (FIXTURES / "gfs-3km.toml").read_text(encoding="utf-8")
    old = text.replace("start_time = 2026-09-26T06:00:00", "start_time = 2026-09-10T06:00:00")
    old = old.replace('cycle = "2026-09-26T06"', 'cycle = "2026-09-10T06"')
    assert old.count("2026-09-10T06") == 2
    (tmp_path / "gfs-3km.toml").write_text(old, encoding="utf-8")
    monkeypatch.setenv("GPUWM_NO_LOCAL_GPU", "1")
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({"schema": PLAN_SCHEMA, "name": "archived", "route": "prepared",
                                "config": {"path": str(tmp_path / "gfs-3km.toml")},
                                "output_root": str(tmp_path / "run")}), encoding="utf-8")
    from woof.cli import build_parser

    assert runplan.run_plan_main(build_parser().parse_args(["run-plan", "--estimate", str(path)])) == 0
    document = json.loads(capsys.readouterr().out)
    download = document["download"]
    # The fixture's 100 hPa top needs one level above it (70 hPa): 129
    # records per lead against the 124 the row measured, priced per record.
    assert download["mode"] == "idx-subset"
    assert download["bytes"] == round(9 * GFS_RECORDS * 129 / 124)
    free = document["disk"]["bytes"] - download["bytes"] // 2

    monkeypatch.setattr(capabilities, "require", lambda *args, **kwargs: None)
    fetched = []
    monkeypatch.setattr(runplan, "_run_fetch", lambda *args, **kwargs: fetched.append(args))
    monkeypatch.setattr(runplan, "_staged_chain", lambda *args, **kwargs: fetched.append("chain"))
    monkeypatch.setattr(runplan, "_hrrr_chain", lambda *args, **kwargs: fetched.append("chain"))
    monkeypatch.setattr(disk_budget, "free_bytes", lambda path: free)
    plan = load_plan(path)
    plan.run_dir.mkdir(parents=True, exist_ok=True)
    with EventStream(plan.run_dir / EVENTS_FILENAME, mirror=None) as events:
        assert execute_plan(plan, events=events) == 1
    failed = read_events(plan.run_dir / EVENTS_FILENAME)[-1]
    assert failed["event"] == "failed"
    assert "Refused before the download" in failed["message"]
    assert "reads the archive" in failed["message"]
    assert fetched == []

GFS_REQUEST = {"source": "gfs", "cycle": "2026-09-26T06", "hours": 24, "cadence": 3, "area": GFS_BOX}


def _sized(path: Path, size: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.truncate(size)
    return path


def _legacy_receipt(folder: Path, *, cycle: str, area: dict | None, files: dict[str, int]) -> None:
    for name, size in files.items():
        _sized(folder / name, size)
    (folder / fetch.FETCH_MANIFEST_NAME).write_text(json.dumps({
        "schema": fetch.FETCH_MANIFEST_SCHEMA, "source": "gfs", "cycle": cycle,
        "forecast_hours": [0, 3], "area": area,
        "files": [{"name": name, "bytes": size} for name, size in files.items()],
    }), encoding="utf-8")


def test_a_folder_named_by_hand_counts_only_what_a_receipt_of_this_request_names(tmp_path):
    from woof.runplan import _present_download_bytes

    folder = tmp_path / "data"
    _legacy_receipt(folder, cycle="2026-09-26T06:00:00Z", area=fetch.parse_area(GFS_BOX).as_manifest(),
                    files={"gfs.t06z.f000.grib2": 3_000_000, "gfs.t06z.f003.grib2": 3_100_000})
    _sized(folder / "my-own-soundings.nc", 64_000_000)
    named = 6_100_000
    assert download_budget.present_bytes(folder, GFS_REQUEST) == named
    assert _present_download_bytes(GFS_REQUEST, folder, False) == named
    # Another cycle's or another area's receipt is not this download.
    assert download_budget.present_bytes(folder, dict(GFS_REQUEST, cycle="2026-09-25T06")) == 0
    assert download_budget.present_bytes(folder, dict(GFS_REQUEST, area="30,-100,40,-90")) == 0
    # A folder keyed to the request is all this request's, a half-finished download included.
    assert _present_download_bytes(GFS_REQUEST, folder, True) >= named + 64_000_000


def test_a_folder_with_no_receipt_is_downloaded_into_in_full(tmp_path):
    _sized(tmp_path / "data" / "my-own-soundings.nc", 64_000_000)
    assert download_budget.present_bytes(tmp_path / "data", GFS_REQUEST) == 0


def test_table_route_and_era5_receipts_are_read_too(tmp_path):
    from woof import era5_arco

    route = tmp_path / "rap"
    _sized(route / "rap.t00z.awip32f00.grib2", 1000)
    _sized(route / "stray.grib2", 5000)
    (route / fetch_routes.MANIFEST_NAME).write_text(json.dumps({
        "schema": fetch_routes.ROUTE_MANIFEST_SCHEMA,
        "request": {"source": "rap", "cycle": "2026-09-26T00Z", "host": "ladder", "member": None},
        "files": [{"relpath": "rap.t00z.awip32f00.grib2", "bytes": 1000}], "composed": [],
    }), encoding="utf-8")
    request = {"source": "rap", "cycle": "2026-09-26T00", "hours": 6}
    assert download_budget.present_bytes(route, request) == 1000
    assert download_budget.present_bytes(route, dict(request, cycle="2026-09-26T01")) == 0

    era5 = tmp_path / "era5"
    _sized(era5 / "era5-arco.nc", 2000)
    (era5 / era5_arco._RECEIPT).write_text(json.dumps({
        "schema": era5_arco._SCHEMA,
        "request": {"source": "era5", "provider": "arco", "cycle": "2021-12-10T18:00:00Z",
                    "hours": 6, "cadence_hours": 6,
                    "area": fetch.parse_area("30,-100,40,-90").as_manifest()},
        "artifact": {"name": "era5-arco.nc", "bytes": 2000},
    }), encoding="utf-8")
    request = {"source": "era5", "cycle": "2021-12-10T18", "hours": 6, "area": "30,-100,40,-90",
               "era5_provider": "arco"}
    assert download_budget.present_bytes(era5, request) == 2000
    assert download_budget.present_bytes(era5, dict(request, area="31,-100,40,-90")) == 0


def test_the_review_prices_a_download_into_a_folder_of_unrelated_files(tmp_path, capsys, monkeypatch):
    """The folder held more than the download; the review used to subtract all of it and price 0."""
    _clock(monkeypatch, REVIEW_NOW)
    folder = tmp_path / "my-data"
    _sized(folder / "my-own-soundings.nc", 64_000_000)
    document = _estimate(tmp_path, capsys, monkeypatch, "gfs-3km.toml", data_dir=str(folder))
    assert document["download"]["present_bytes"] == 0
    assert document["disk"]["download_bytes"] == document["download"]["bytes"] > 25e6


def test_a_moved_window_is_credited_only_with_the_leads_it_still_asks_for(tmp_path):
    """The fetch resumes a later start of the same cycle into the same folder; f000 is not this download."""
    folder = tmp_path / "data"
    sizes = {0: 3_000_000, 3: 3_100_000, 6: 3_200_000}
    for hour, size in sizes.items():
        _sized(folder / f"gfs.t06z.f{hour:03d}.grib2", size)
    _sized(folder / "gfs-series.tsv", 100)
    (folder / fetch.FETCH_MANIFEST_NAME).write_text(json.dumps({
        "schema": fetch.FETCH_MANIFEST_SCHEMA, "source": "gfs", "cycle": "2026-09-26T06:00:00Z",
        "forecast_hours": [0, 3, 6], "area": fetch.parse_area(GFS_BOX).as_manifest(),
        "files": [{"name": f"gfs.t06z.f{hour:03d}.grib2", "forecast_hour": hour, "bytes": size}
                  for hour, size in sizes.items()]
        + [{"name": "gfs-series.tsv", "role": "series", "forecast_hour": None, "bytes": 100}],
    }), encoding="utf-8")
    everything = sum(sizes.values()) + 100
    assert download_budget.present_bytes(folder, dict(GFS_REQUEST, hours=6)) == everything
    moved = dict(GFS_REQUEST, hours=3, forecast_start_hour=3)
    assert download_budget.present_bytes(folder, moved) == sizes[3] + sizes[6] + 100


def test_a_route_lead_the_request_does_not_ask_for_is_not_credited(tmp_path):
    route = tmp_path / "rap"
    for lead in (0, 1, 2):
        _sized(route / f"rap.t00z.awip32f{lead:02d}.grib2", 1000 + lead)
    (route / fetch_routes.MANIFEST_NAME).write_text(json.dumps({
        "schema": fetch_routes.ROUTE_MANIFEST_SCHEMA,
        "request": {"source": "rap", "cycle": "2026-09-26T00Z", "host": "ladder", "member": None},
        "files": [{"relpath": f"rap.t00z.awip32f{lead:02d}.grib2", "lead": lead, "bytes": 1000 + lead}
                  for lead in (0, 1, 2)], "composed": [],
    }), encoding="utf-8")
    request = {"source": "rap", "cycle": "2026-09-26T00", "hours": 2, "cadence": 1}
    assert download_budget.present_bytes(route, request) == 1000 + 1001 + 1002
    assert download_budget.present_bytes(route, dict(request, hours=1)) == 1000 + 1001


#: Real fetches of f000 to f006 at 3 h: what each left on disk.  GEFS member p01 of
#: 2026-10-28 00Z (fix/gefs-record-subset, each half's mapped records by its .idx)
#: kept 98,380,295 B of a and b records and wrote 98,380,295 B of pairs from them
#: (whole objects were 342,113,952 B); GDPS (2026-09-27 00Z) kept its 1,044
#: per-field objects and wrote 552,387,259 B of valid times.
COMPOSED_FETCHES = {"gefs": 196_760_590, "gem-gdps": 1_106_308_271}


@pytest.mark.parametrize("source", sorted(COMPOSED_FETCHES))
def test_a_route_that_composes_is_priced_with_the_copies_it_keeps(source):
    """Each composed file is a second copy of objects already counted; the estimate was half of the disk."""
    request = {"source": source, "cycle": "2026-09-27T00", "hours": 6, "cadence": 3}
    estimate = download_budget.download_estimate(request)
    assert 0.9 * COMPOSED_FETCHES[source] <= estimate["bytes"] <= 1.1 * COMPOSED_FETCHES[source]
    # The copies are written here, not moved over the network.
    assert estimate["transfer_bytes"] < 0.6 * estimate["bytes"]
    assert "composed file(s)" in estimate["basis"]


def test_a_route_that_composes_nothing_prices_no_copy():
    estimate = download_budget.download_estimate({"source": "rap", "cycle": "2026-09-26T00", "hours": 6})
    assert estimate["bytes"] == estimate["transfer_bytes"]
    assert "composed" not in estimate["basis"]


def test_an_era5_arco_window_moves_the_blocks_of_its_rows_not_whole_chunks():
    """rw_zarr reads the blosc blocks that hold the window's rows (fix/era5-arco-block-ranges).

    Measured 2026-10-29: two times over 30-42N, 105-90W received 174.1 MB, where the same
    times read whole received 835.6 MB.  The estimate said 551 MB per time whatever the area.
    """
    request = {"source": "era5", "cycle": "2026-10-01T00", "hours": 3, "cadence": 3,
               "era5_provider": "arco"}
    regional = download_budget.download_estimate(dict(request, area="30,-105,42,-90"))
    whole = download_budget.download_estimate(dict(request, area="-89,-180,89,180"))
    assert whole["transfer_bytes"] == 2 * 551_000_000
    assert 174.1e6 <= regional["transfer_bytes"] <= 0.25 * whole["transfer_bytes"]
    assert "latitude rows" in regional["basis"]
