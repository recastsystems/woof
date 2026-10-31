"""GDAS acquisition through the declared GFS container.

The provider publishes the f000 through f009 ladder. Requests beyond
that ladder refuse before download, while valid acquisitions publish
the structured preparation handoff for the composed packaged profile.
The separate network smoke checks real container and manifest bytes.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import shlex
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from woof import cli
from woof import fetch
from woof import fetch_bars
from woof import fetch_routes
from tools import download_gfs_native_subset as gfs_transport


def test_gdas_rides_the_gfs_container_declaration():
    assert fetch.GFS_CONTAINER_SOURCES == ("gfs", "gdas")
    assert set(gfs_transport.NOMADS_MODELS) == {"gfs", "gdas"}
    # One selector, two publishers: the variable/level declaration is
    # shared, so the two cannot drift apart on what a subset contains.
    assert (gfs_transport.NOMADS_MODELS["gdas"]["script"]
            == "filter_gdas_0p25.pl")


def test_the_gdas_query_requests_native_specific_humidity():
    box = dict(left_lon=260.0, right_lon=270.0, bottom_lat=30.0,
               top_lat=40.0)
    cycle = datetime(2026, 7, 29, 12)
    gfs = gfs_transport.nomads_query(cycle, 0, model="gfs", **box)
    gdas = gfs_transport.nomads_query(cycle, 0, model="gdas", **box)
    # Everything after the script name and the file/dir stems is the
    # same declaration, character for character.
    def selector(url: str) -> list[str]:
        return [part for part in url.split("&")
                if not part.startswith(("http", "file=", "dir="))]
    # Plus the hydrometeors the packaged GDAS mapping binds, appended in
    # the filter's own names (cloud mixing ratio is CLWMR to the filter).
    mapped = [f"var_{name}=on" for name in ("CLWMR", "GRLE", "ICMR", "RWMR", "SNMR")]
    assert [part.replace("var_RH=", "var_SPFH=") for part in selector(gfs)] + mapped == selector(gdas)
    assert "var_SPFH=on" in gdas and "var_RH=on" not in gdas
    assert "filter_gdas_0p25.pl" in gdas
    assert "file=gdas.t12z.pgrb2.0p25.f000" in gdas
    assert "gdas.20260729%2F12%2Fatmos" in gdas


def test_the_gdas_crop_asks_for_every_variable_its_mapping_binds():
    """rel/2.9.10's default GDAS route, this grib-filter crop, was refused at
    prepare: "mapped frame ... lacks required fields ['cloud_ice_mixing_ratio',
    'cloud_water_mixing_ratio', 'graupel_or_hail_mixing_ratio', ...]" (test host,
    woof go --prepare-only, GDAS 2026-10-29 06Z).  The query asked for none
    of the five hydrometeors the packaged GDAS mapping requires since 2.9.9.
    """
    levels = fetch.container_subset_levels("gdas")
    query = gfs_transport.nomads_query(
        datetime(2026, 10, 29, 6), 0, model="gdas", left_lon=243.33,
        right_lon=281.67, bottom_lat=17.81, top_lat=53.13,
        pressure_levels_hpa=levels)
    asked = {name for name, _level in fetch_bars.nomads_selector_pairs(
        tuple(part[:-len("=on")] for part in query.split("&")
              if part.startswith("var_")), ("lev_surface",), ())}
    mapped = fetch.mapped_record_levels("gdas")
    assert {"CLMR", "ICMR", "RWMR", "SNMR", "GRLE"} <= set(mapped)
    assert set(mapped) <= asked
    # The filter keeps the older wgrib2 name for cloud mixing ratio and
    # answers var_CLMR with HTTP 500 "invalid parameter: var_CLMR"
    # (filter_gdas_0p25.pl, 2026-10-29); the .idx says CLMR.
    assert "var_CLWMR=on" in query and "var_CLMR=on" not in query
    # The record bar the crop is held to counts them on the 22 levels the
    # mapping says GFS writes them on (50..1000 hPa, no 70 hPa): 41 levels
    # x 5 fields + 19 single-level records + 5 x 22.
    assert len(levels) == 41
    assert fetch.container_record_count("gdas", levels) == 224 + 110
    assert fetch.container_subset_record_count("gdas") == 334
    # GFS decodes through its bridge and binds no mapping: unchanged.
    assert fetch.container_subset_record_count("gfs") == 124


def test_an_unknown_model_is_refused():
    with pytest.raises(ValueError, match="unknown NOMADS model"):
        gfs_transport.nomads_query(
            datetime(2026, 7, 29, 12), 0, model="gefs",
            left_lon=260.0, right_lon=270.0, bottom_lat=30.0, top_lat=40.0)


def test_the_object_urls_and_cycle_cadence_follow_the_source():
    cycle = datetime(2026, 7, 29, 12)
    assert fetch.gfs_object_url(cycle, 0, "gdas").endswith(
        "gdas.20260729/12/atmos/gdas.t12z.pgrb2.0p25.f000")
    assert "/gdas." in fetch.gfs_object_url(cycle, 3, "gdas")
    # GDAS runs on the GFS synoptic grid, so the cadence rule is shared.
    assert fetch.parse_cycle("2026-07-29T12", "gdas") == cycle
    with pytest.raises(ValueError, match="GDAS cycles run at 00/06/12/18"):
        fetch.parse_cycle("2026-07-29T13", "gdas")


def test_the_gdas_ladder_stops_at_the_published_horizon():
    """The scope constant and the ladder agree, and nothing widens it.

    v1.0.1 pinned this at f000 against the analysis generating process
    alone, and widening it was treated as a re-certification event.  The
    span is a PUBLICATION fact now, not a claim about what was proved
    here: the assimilation cycle writes f000..f009 and nothing else, so
    the constant below records what the publisher writes.  The v1.1
    hygiene lane IS that event -- real NOMADS f000/f003/f006/f009
    subsets under ``tests/fixtures/gdas-process-id/`` with the declared
    81/96 processes, the last of them the endpoint itself -- so the span
    is the published f000..f009.  The gate did not become a flag: past
    f009 still refuses, in the same words.
    """

    assert fetch.GDAS_MAX_FORECAST_HOUR == 9
    assert fetch.GDAS_PUBLISHED_HOURS == tuple(range(10))
    assert fetch.gdas_forecast_hours(0) == (0,)
    assert fetch.gdas_forecast_hours(6, 3) == (0, 3, 6)
    assert fetch.gdas_forecast_hours(9, 3) == (0, 3, 6, 9)
    for beyond in (10, 12, 24, 384):
        with pytest.raises(ValueError, match="publishes"):
            fetch.gdas_forecast_hours(beyond)


def test_fetch_help_cannot_drift_from_the_registry_gdas_span(capsys):
    """`woof fetch --help` is pinned to the registry's max hour.

    v1.0.1's help said GDAS was "certified analysis-only, so it takes
    ``--hours 0``" and its `--cadence` line did not mention GDAS at all.
    Both survived the re-certification to f009, so an upgrader reading
    shipped `--help` got a strictly narrower scope than the registry,
    the refusal text, and DATA.md published -- and no test noticed.

    The bar here is the registry's own
    ``get_source_adapter("gdas").max_forecast_hour``, so help that
    hardcodes a hour, or a span that moves without the help moving with
    it, fails right here.  The scope words are pinned too: an endpoint
    without "no ingest route" beside it is the other half of the same
    contradiction.
    """

    from woof.source_adapters import get_source_adapter

    registry_max = get_source_adapter("gdas").max_forecast_hour

    with pytest.raises(SystemExit) as exit_info:
        cli.main(["fetch", "--help"])
    assert exit_info.value.code == 0
    # argparse rewraps to the terminal width, so collapse whitespace
    # before matching -- otherwise this test pins the wrap, not the text.
    help_text = " ".join(capsys.readouterr().out.split())

    # The endpoint, spelled the way the refusal and the docs spell it.
    assert f"f{registry_max:03d}" in help_text
    # ...and the scope that has to travel with it.
    assert "certified for fetch and decode" in help_text
    assert "native mapped GDAS preparation" in help_text
    # The superseded v1.0.1 claim is gone from every fetch help string,
    # not merely relocated.
    assert "analysis-only" not in help_text
    assert "certified analysis" not in help_text
    # --cadence used to list gfs, era5 and hrrr and silently skip gdas.
    # It states GDAS's rule, and states it as the ladder's own: whole
    # hours that divide the window, not a list of three values that
    # would have to be edited beside the ladder.
    assert "gdas any whole number of hours that divides --hours" in help_text
    assert f"f{registry_max:03d} ladder" in help_text

    # And `woof --help` alone must not leave a reader thinking GDAS
    # initializes a run.
    with pytest.raises(SystemExit):
        cli.main(["--help-all"])
    assert "native GDAS uses its mapped preparation" in " ".join(capsys.readouterr().out.split())


def test_a_gdas_fetch_prints_the_prep_line_it_published(tmp_path,
                                                        monkeypatch, capsys):
    """The GDAS handoff is a command, and it is the published argv.

    What this printed was a sentence with no command in it: fetch with
    --all-levels (which the default ladder already takes for this
    container) and bind the in-band supplement by hand, while the fetch
    had already written the whole bound argv beside the files.
    """

    def fetched(*, cycle, hours, out, source, **_):
        out.mkdir(parents=True, exist_ok=True)
        files = []
        for hour in hours:
            name = f"{source}.t{cycle:%H}z.pgrb2.0p25.f{hour:03d}.subset.grib2"
            (out / name).write_bytes(b"GRIB fixture 7777")
            files.append({"name": name, "role": f"{source}-subset",
                          "forecast_hour": hour, "bytes": 17,
                          "sha256": fetch.sha256_file(out / name),
                          "url": None})
        series = out / f"{source}-series.tsv"
        series.write_text("".join(
            f"{item['forecast_hour']}\t{item['name']}\t96\n"
            for item in files))
        fetch._write_gfs_front_door_files(
            out, source=source, cycle=cycle, files=files, series=series)
        manifest = out / fetch.FETCH_MANIFEST_NAME
        manifest.write_text("{}\n")
        return manifest

    monkeypatch.setattr(fetch, "fetch_gfs", fetched)
    monkeypatch.setattr(fetch, "require_published_cycle",
                        lambda *args, **kwargs: None)
    monkeypatch.setattr(fetch, "archive_only_cycle",
                        lambda *args, **kwargs: False)
    out = tmp_path / "gdas"
    assert cli.main(["fetch", "--source", "gdas", "--cycle", "2026-07-29T12",
                     "--hours", "3", "--area", "30,-100,40,-90",
                     "--out", str(out)]) == 0
    lines = capsys.readouterr().out.splitlines()
    document = json.loads(
        (out / fetch_routes.PREP_ARGUMENTS_NAME).read_text())
    commands = [line.strip() for line in lines
                if line.strip().startswith("woof prep ")]
    assert len(commands) == 1, lines
    assert shlex.split(commands[0]) == ["woof", "prep", *document["argv"]]
    comments = " ".join(line for line in lines
                        if line.strip().startswith("#"))
    for flag in document["caller_supplies"]:
        assert flag in comments, flag
    assert "--all-levels" not in " ".join(lines)


def test_a_gdas_request_past_the_published_span_refuses_up_front(tmp_path,
                                                                 capsys):
    """Refused at the CLI, before any download, in capability wording."""

    out = tmp_path / "gdas"
    rc = cli.main(["fetch", "--source", "gdas", "--cycle", "2026-07-29T12",
                   "--hours", "12", "--area", "30,-100,40,-90",
                   "--out", str(out)])
    assert rc == 2
    err = capsys.readouterr().err
    # Names the published limit and the absent f012 object.
    assert "f009" in err and "f012" in err
    # ...and says what to reach for instead, at the default width,
    # because a remedy never moves behind a flag.
    assert "f000..f009" in err
    assert "--source gfs" in err
    # The mechanism -- why no later object exists to fetch -- is one flag
    # away, and the refusal says so.  A remedy never moves behind a flag,
    # and the reason never crowds it out at the default width.
    assert "never written" not in err
    assert "--explain" in err
    assert "publishes" in err
    # Nothing was created, because nothing was attempted.
    assert not out.exists()

    rc = cli.main(["fetch", "--source", "gdas", "--cycle", "2026-07-29T12",
                   "--hours", "12", "--area", "30,-100,40,-90",
                   "--out", str(out), "--explain"])
    assert rc == 2
    explained = capsys.readouterr().err
    # Explain mode keeps the limit and the remedy, and restores the
    # mechanism the default width held back: the object is never
    # published, so no later fetch could find it.
    assert "f000..f009" in explained
    assert "--source gfs" in explained
    assert "never written" in explained
    assert not out.exists()


def test_the_library_boundary_refuses_the_same_request(tmp_path):
    """A caller reaching fetch_gfs directly hits the same gate."""

    with pytest.raises(ValueError, match="publishes"):
        fetch.fetch_gfs(
            cycle=datetime(2026, 7, 29, 12), hours=(0, 3, 12),
            area=fetch.parse_area("30,-100,40,-90"),
            out=tmp_path / "gdas", source="gdas", progress=lambda _: None)


def test_the_fetch_hint_table_accepts_gdas():
    fetch.validate_fetch_hints({"source": "gdas"}, source="gdas")
    fetch.validate_fetch_hints({"source": "gdas", "hours": 0},
                               source="gdas")


def test_the_fetch_hint_table_accepts_every_routed_source():
    """The validator's vocabulary IS the route table, name for name.

    It used to be a hand-typed four, and the drift that shipped was
    exactly this: ten table routes opened and every hand-written
    ``[fetch]`` table naming one of them was refused at config load,
    while `woof domain` was emitting those same tables.  The emitter
    (:func:`fetch.fetch_front_door_sources`) and this validator now read
    one list, so the assertion is that a table naming ANY source the
    wizard would emit loads -- not that a hand-picked few do.
    """

    routed = fetch.fetch_front_door_sources()
    assert routed, "no fetch route in the table; the seam is not answering"
    assert {"gdas", "gefs"} <= set(routed), routed
    for name in routed:
        fetch.validate_fetch_hints({"source": name}, source="experiment.toml")


@pytest.mark.parametrize("spelling", ["nam", "hrrr-ak", "gdass"])
def test_the_fetch_hint_table_refuses_an_unrouted_source_by_name(spelling):
    """An unrouted spelling is refused, and the refusal names it.

    The breakage: a config whose step 1 cannot be run.  ``nam`` and
    ``hrrr-ak`` are REGISTERED sources with no fetch route in this
    WOOF, and ``gdass`` is a typo for one that has a route -- all three
    describe a download `woof fetch` would not perform, so the table
    has to be refused at config load rather than at the download, and
    the refusal has to print the offending spelling and the vocabulary
    that would have worked.
    """

    routed = fetch.fetch_front_door_sources()
    assert spelling not in routed, (
        f"{spelling} now has a fetch route; this test needs an unrouted "
        f"spelling, pick one outside {routed}")
    with pytest.raises(ValueError) as refusal:
        fetch.validate_fetch_hints({"source": spelling},
                                   source="experiment.toml")
    message = str(refusal.value)
    assert repr(spelling) in message
    assert "is not one of" in message
    assert "gdas" in message  # the vocabulary that would have worked


def test_a_config_hint_past_the_published_span_is_refused():
    """The config surface is gated too, in the same words."""

    fetch.validate_fetch_hints({"source": "gdas", "hours": 6},
                               source="experiment.toml")
    with pytest.raises(ValueError, match="publishes"):
        fetch.validate_fetch_hints({"source": "gdas", "hours": 12},
                                   source="experiment.toml")


def test_gdas_prints_no_next_command_that_ends_in_a_refusal(tmp_path,
                                                            capsys):
    """One story across fetch, the registry, and the docs.

    GDAS runs through the generic mapped route as a packaged profile
    now, so the registry row must say so -- runnable, on the mapped
    runner, naming its packaged profile, no longer redirecting the user
    to ``--source gfs``.  The half that has not changed: the front-door
    manifest authoring must not print a GFS command over a GDAS series
    (that would be a source-identity lie rather than a dead end); it
    says what is missing instead.
    """

    from woof.source_adapters import AdapterStatus, get_source_adapter

    adapter = get_source_adapter("gdas")
    assert adapter.runnable
    assert adapter.runner == "mapped_composition_v1"
    assert adapter.status is AdapterStatus.RUNNABLE_NOT_CERTIFIED
    assert adapter.packaged_profile == "gdas-pgrb2-0p25-grib2-v1"
    assert adapter.max_forecast_hour == fetch.GDAS_MAX_FORECAST_HOUR
    assert "--source gfs" not in adapter.notes

    said: list[str] = []
    out = tmp_path / "gdas"
    out.mkdir()
    # The authoring seam prints the accurate stop, not an rw-wps line.
    try:
        fetch.author_gfs_front_door_manifest(
            out=out, bridge=None, wps_namelist=None,
            experiment_config=None, progress=said.append)
    except (ValueError, TypeError, OSError):
        pass  # no prior fetch here; the printing seam is asserted below
    assert not any("rw-wps --source gfs" in line for line in said)


@pytest.mark.network
@pytest.mark.skipif(os.environ.get("WOOF_NETWORK_TESTS") != "1",
                    reason="live network smoke; set WOOF_NETWORK_TESTS=1")
def test_live_gdas_ladder_matches_the_certified_gfs_container(tmp_path):
    """The whole certified ladder, live, checked against the claim.

    Fetches f000..f009 at the default three-hour cadence over a 10x10
    degree box -- four files, a couple of megabytes -- then asserts the
    census, the record bar, the manifest, the *declared* process ID on
    every row, and that the series seam the certified GFS bridge
    consumes is authored under the gdas names.

    It fetches the whole ladder because it asserts about the whole
    ladder.  Until v1.1.0 this test downloaded hour 0 alone, asserted
    the series had exactly one row, and then required that one row to
    carry both process IDs ``["81", "96"]`` -- unsatisfiable by
    construction, so enabling it could only ever fail, and the live
    certification of the forecast hours it was named for never
    happened.
    """

    # Eight hours back: the assimilation cycle has certainly published
    # its analysis and its short forecast.
    moment = datetime.now(timezone.utc) - timedelta(hours=8)
    cycle = datetime(moment.year, moment.month, moment.day,
                     (moment.hour // 6) * 6)
    # The published span, from the same constant the refusal uses -- so
    # a widened or narrowed span is exercised here, not merely declared.
    hours = fetch.gdas_forecast_hours(fetch.GDAS_MAX_FORECAST_HOUR, 3)
    assert hours == (0, 3, 6, 9)
    # f000 is the analysis; every forecast hour carries the other
    # declared process.  This is the expectation the series must state.
    expected_processes = ["81"] + ["96"] * (len(hours) - 1)

    out = tmp_path / "gdas"
    said: list[str] = []
    manifest = fetch.fetch_gfs(
        cycle=cycle, hours=hours,
        area=fetch.parse_area("30,-100,40,-90"),
        out=out, source="gdas", progress=said.append)

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["source"] == "gdas"
    subsets = [item for item in payload["files"]
               if item["role"] == "gdas-subset"]
    assert [item["forecast_hour"] for item in subsets] == list(hours)
    for item in subsets:
        assert item["name"].startswith("gdas.t")
        assert (out / item["name"]).is_file()
        # The certified census, verified against the file itself.
        assert fetch.count_grib2_messages(out / item["name"]) == payload["record_bars"][0]["derived"]

    # The record bar was derived from the live GDAS index and agreed
    # with the certified GFS constant -- that agreement IS the container
    # claim, so it is the assertion worth making.  (One bar covers the
    # fetch; the per-file census above is what checks each hour.)
    bar = payload["record_bars"][0]
    expected = fetch.container_record_count(
        "gdas", tuple(payload["pressure_levels_hpa"]))
    assert "var_SPFH" in payload["requested_variables"]
    assert {"var_CLWMR", "var_ICMR", "var_RWMR", "var_SNMR", "var_GRLE"} <= set(
        payload["requested_variables"])
    assert bar["certified"] == expected
    assert bar["derived"] == expected, (
        "the live GDAS inventory no longer yields the certified GFS "
        "census; the container has diverged and the mapping must be "
        "re-certified rather than reused")
    assert bar["inventory_change_accepted"] is False

    # The ingest seam: gdas-series.tsv under the gdas names, which the
    # unchanged gfs_grib2_bridge consumes.
    series = out / "gdas-series.tsv"
    assert series.is_file()
    rows = [line.split("\t") for line in
            series.read_text(encoding="utf-8").splitlines()]
    assert [row[0] for row in rows] == [str(hour) for hour in hours]
    assert all(row[1].startswith("gdas.t") for row in rows)
    assert [row[2] for row in rows] == expected_processes

    # Front-door dry contract check: author the manifest against a
    # stand-in bridge and confirm every role binds by digest.
    bridge = tmp_path / "gfs_grib2_bridge"
    bridge.write_bytes(b"stand-in for the digest binding")
    namelist = tmp_path / "namelist.wps"
    namelist.write_text("&share\n/\n", encoding="utf-8")
    config = tmp_path / "experiment.toml"
    config.write_text("[experiment]\n", encoding="utf-8")
    front, digest = fetch.author_gfs_front_door_manifest(
        out=out, bridge=bridge, wps_namelist=namelist,
        experiment_config=config, source="gdas", progress=said.append)
    document = json.loads(front.read_text(encoding="utf-8"))
    assert document["schema"] == fetch.GFS_FRONT_DOOR_MANIFEST_SCHEMA
    assert document["source"]["model"] == "GDAS"
    assert document["source"]["product"] == "pgrb2.0p25"
    assert set(document["files"]) == {
        "series", "bridge", "wps_namelist", "experiment_config",
    } | {f"grib-f{hour:03d}" for hour in hours}
    assert len(digest) == 64

    # ...and the seam still refuses to print an rw-wps command, because
    # there is still no GDAS front door to point at.
    assert not any("rw-wps --source gfs" in line for line in said)
    assert any("native mapped preparation route" in line for line in said)
