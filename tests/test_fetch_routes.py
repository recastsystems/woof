"""Resolution-level proofs for the packaged acquisition route table.

Nothing here moves a byte.  These tests hold the table (which host, which
key, which lead ladder, which file-set composition) against the shapes
measured on the live services on 2026-08-17, and hold the front door's
coverage against the registry: every runnable source either resolves a
route or refuses by name.

The live smoke -- one small real object per route -- is
``tests/test_fetch_routes_live.py``.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json
import os
import re
import shlex

import pytest

from woof import fetch_routes, source_adapters
from woof.filesystem_paths import io_path


# --------------------------------------------------------------------------
# Coverage: the registry and the route table agree about every source
# --------------------------------------------------------------------------

def test_every_runnable_source_either_routes_or_refuses_by_name():
    """No runnable source may fall off the front door silently.

    The breakage this prevents is the one the 6 h battery reported: a
    source with a runnable profile, a passing sim, and no way to get its
    bytes -- engine-proven, not shipped.
    """

    handled = set(fetch_routes.all_fetchable_sources()) | set(fetch_routes.refusal_ids())
    # gfs/hrrr/era5 keep their own hand-written routes in woof.fetch.
    handled |= set(fetch_routes.LEGACY_ROUTE_SOURCES)
    # A row whose bytes come from running a model names its door as an
    # acquisition owner (earth2 runs an Earth-2 model through earth2studio).
    handled |= {row.source_id for row in source_adapters.source_adapters()
                if row.acquisition_owner}
    from woof.cf_archive_fetch import sources
    handled |= set(sources())
    # Model-backed rows delegate acquisition to their declared owner rather
    # than a published-object route. Check the actual front-door contract.
    from importlib import import_module
    for row in source_adapters.source_adapters():
        if row.acquisition_owner:
            assert callable(getattr(import_module(row.acquisition_owner), "fetch", None))
            handled.add(row.source_id)
    runnable = {row.source_id for row in source_adapters.source_adapters()
                if row.runnable}
    # A runnable preparation can require caller-supplied files. Its fetch
    # boundary must explain that by source name rather than silently fall off.
    for source in runnable - handled:
        with pytest.raises(ValueError) as refusal:
            fetch_routes.route_for(source)
        assert source in str(refusal.value)
        assert "woof sources" in str(refusal.value)
        handled.add(source)
    assert runnable - handled == set()


def test_a_registered_source_with_no_route_refuses_naming_the_registry():
    """A non-runnable row is refused for the reason it is not runnable."""

    with pytest.raises(ValueError) as error:
        fetch_routes.route_for("nam")
    message = str(error.value)
    assert "nam" in message
    # Names WHY, not merely that it is unknown.
    assert "runnable" in message
    assert "woof sources" in message


def test_an_unregistered_name_is_refused_with_the_registry_listing():
    with pytest.raises(ValueError) as error:
        fetch_routes.route_for("not-a-model")
    message = str(error.value)
    assert "not-a-model" in message
    assert "rap" in message and "icon-eu" in message


def test_aliases_resolve_to_their_registry_row():
    assert fetch_routes.route_for("gdps").source_id == "gem-gdps"
    assert fetch_routes.route_for("ifs").source_id == "ecmwf-open-data"
    assert fetch_routes.route_for("hrrr-wrfprs").source_id == "hrrr-prs"


# --------------------------------------------------------------------------
# Table integrity
# --------------------------------------------------------------------------

def test_the_packaged_route_table_matches_its_pin():
    """The wheel's copy is the copy this code was written against."""

    assert (fetch_routes.packaged_route_table_sha256()
            == fetch_routes.ROUTE_TABLE_SHA256)


def test_every_route_declares_an_endpoint_ladder_and_known_tokens():
    """A route's hosts are an ORDERED ladder, and every rung says why.

    The `default: true` marker used to pick one host out of a set.  It
    now marks the ladder's HEAD, which the loader binds to position 0,
    so the table cannot declare a default it would never ask first.
    """

    for source_id in fetch_routes.route_ids():
        route = fetch_routes.route_for(source_id)
        assert route.hosts, source_id
        assert all(host.why.strip() for host in route.hosts), source_id
        assert route.host(None) is route.hosts[0], source_id
        for row in route.files:
            unknown = fetch_routes.unknown_tokens(row.path)
            assert unknown == (), (source_id, row.role, unknown)


def test_every_route_row_names_why_it_does_not_record_subset():
    """Full files are the default; the opt-out has to say what it costs."""

    for source_id in fetch_routes.route_ids():
        route = fetch_routes.route_for(source_id)
        if not route.record_subset_supported:
            assert route.record_subset_why.strip()


# --------------------------------------------------------------------------
# Cycle grammar
# --------------------------------------------------------------------------

def test_a_cycle_hour_the_publisher_does_not_run_is_refused_by_name():
    with pytest.raises(ValueError) as error:
        fetch_routes.resolve_request(
            "gem-gdps", cycle=datetime(2026, 8, 17, 6), hours=6)
    message = str(error.value)
    assert "06Z" in message
    assert "00, 12" in message


def test_icon_eu_runs_eight_cycles_a_day():
    plan = fetch_routes.resolve_request(
        "icon-eu", cycle=datetime(2026, 8, 17, 3), hours=2)
    assert plan.leads == (0, 1, 2)


@pytest.mark.parametrize("cycle_hour", [0, 3, 6, 9, 12, 15, 18, 21])
def test_icon_d2_runs_every_three_hours_each_to_f48_hourly(cycle_hour):
    """Measured on DWD's listing 2026-09-27: every cycle, 03Z included,
    publishes f000..f048 at one hour."""

    route = fetch_routes.route_for("icon-d2")
    ladder = fetch_routes.ladder_for(
        route, datetime(2026, 9, 27, cycle_hour))
    assert ladder == tuple(range(49))
    with pytest.raises(ValueError):
        fetch_routes.resolve_request(
            "icon-d2", cycle=datetime(2026, 9, 27, cycle_hour), hours=49)


def test_icon_d2_refuses_an_hour_it_does_not_run_and_takes_any_whole_hour():
    """A159: a 3 h cadence subsamples the hourly ladder and plans.  A173:
    so does 2 h, which only the retired cadences list refused; a spacing
    that would omit the window's final time is still refused, naming
    that."""

    with pytest.raises(ValueError) as error:
        fetch_routes.resolve_request(
            "icon-d2", cycle=datetime(2026, 9, 27, 1), hours=3)
    assert "01Z" in str(error.value)
    plan = fetch_routes.resolve_request(
        "icon-d2", cycle=datetime(2026, 9, 27, 0), hours=6, cadence=3)
    assert plan.leads == (0, 3, 6)
    plan = fetch_routes.resolve_request(
        "icon-d2", cycle=datetime(2026, 9, 27, 0), hours=6, cadence=2)
    assert plan.leads == (0, 2, 4, 6)
    with pytest.raises(ValueError, match="final time must not be silently"):
        fetch_routes.resolve_request(
            "icon-d2", cycle=datetime(2026, 9, 27, 0), hours=6, cadence=4)


# --------------------------------------------------------------------------
# Lead ladders (all measured against the live listings 2026-08-17)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("cycle_hour,last_lead", [(0, 21), (3, 51)])
def test_rap_extends_to_f51_on_the_off_synoptic_cycles(cycle_hour, last_lead):
    route = fetch_routes.route_for("rap")
    ladder = fetch_routes.ladder_for(
        route, datetime(2026, 8, 17, cycle_hour))
    assert ladder[-1] == last_lead


@pytest.mark.parametrize("cycle_hour,count,last_lead", [
    (0, 93, 120),    # hourly to 78, then three-hourly to 120
    (3, 34, 48),     # hourly to 30, then six-hourly to 48
])
def test_icon_eu_ladders_match_the_measured_dwd_listing(
        cycle_hour, count, last_lead):
    route = fetch_routes.route_for("icon-eu")
    ladder = fetch_routes.ladder_for(
        route, datetime(2026, 8, 17, cycle_hour))
    assert len(ladder) == count
    assert ladder[-1] == last_lead


def test_gefs_reaches_840_hours_only_on_the_00z_cycle():
    route = fetch_routes.route_for("gefs")
    assert fetch_routes.ladder_for(route, datetime(2026, 8, 17, 0))[-1] == 840
    assert fetch_routes.ladder_for(route, datetime(2026, 8, 17, 6))[-1] == 384


def test_a_lead_past_the_ladder_refuses_naming_the_horizon():
    with pytest.raises(ValueError) as error:
        fetch_routes.resolve_request(
            "rap", cycle=datetime(2026, 8, 17, 0), hours=30)
    message = str(error.value)
    assert "f021" in message or "f21" in message
    assert "03, 09, 15, 21" in message


def test_a_cadence_off_the_ladder_refuses_naming_the_ladder():
    with pytest.raises(ValueError) as error:
        fetch_routes.resolve_request(
            "gefs", cycle=datetime(2026, 8, 17, 0), hours=6, cadence=1)
    assert "cadence" in str(error.value)
    assert "3" in str(error.value)


def test_a_forecast_start_hour_shifts_the_window_not_its_length():
    plan = fetch_routes.resolve_request(
        "rap", cycle=datetime(2026, 8, 17, 3), hours=3, start_hour=12)
    assert plan.leads == (12, 13, 14, 15)


# --------------------------------------------------------------------------
# File-set composition, per source, against the measured key shapes
# --------------------------------------------------------------------------

def test_rap_resolves_the_measured_awip32_keys():
    plan = fetch_routes.resolve_request(
        "rap", cycle=datetime(2026, 8, 16, 0), hours=1)
    assert [obj.url for obj in plan.objects] == [
        "https://noaa-rap-pds.s3.amazonaws.com/rap.20260816/"
        "rap.t00z.awip32f00.grib2",
        "https://noaa-rap-pds.s3.amazonaws.com/rap.20260816/"
        "rap.t00z.awip32f01.grib2",
    ]
    # Flat layout: the key's basename is the file on disk.
    assert [obj.relpath for obj in plan.objects] == [
        "rap.t00z.awip32f00.grib2", "rap.t00z.awip32f01.grib2"]


def test_hrrr_prs_resolves_the_conus_wrfprs_keys():
    plan = fetch_routes.resolve_request(
        "hrrr-prs", cycle=datetime(2026, 8, 17, 0), hours=1)
    assert plan.objects[0].url == (
        "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.20260817/conus/"
        "hrrr.t00z.wrfprsf00.grib2")


def test_rrfs_pairs_prslev_with_the_2dfld_supplement():
    plan = fetch_routes.resolve_request(
        "rrfs", cycle=datetime(2026, 8, 17, 0), hours=1)
    roles = [obj.role for obj in plan.objects]
    assert roles == ["prslev", "2dfld", "prslev", "2dfld"]
    assert plan.objects[0].url.endswith(
        "rrfs.20260817/00/rrfs.t00z.prslev.3km.f000.conus.grib2")
    # Both files supply canonical fields: prslev has upper air, while
    # 2dfld has surface and soil state as well as the terrain supplement.
    assert len(plan.primary_files) == 4
    assert {path.name for path in plan.primary_files} == {Path(obj.relpath).name for obj in plan.objects}
    assert [path.name for path in plan.supplement_files] == [
        "rrfs.t00z.2dfld.3km.f000.conus.grib2",
        "rrfs.t00z.2dfld.3km.f001.conus.grib2",
    ]
    assert plan.supplement_role == "rrfs_prslev_2dfld_in_band_surface"


def test_gefs_keeps_member_identity_in_the_path_and_pairs_a_with_b():
    plan = fetch_routes.resolve_request(
        "gefs", cycle=datetime(2026, 8, 17, 0), hours=3)
    assert plan.leads == (0, 3)
    assert plan.objects[0].url == (
        "https://noaa-gefs-pds.s3.amazonaws.com/gefs.20260817/00/atmos/"
        "pgrb2ap5/gec00.t00z.pgrb2a.0p50.f000")
    # upstream layout: the whole key becomes the on-disk path, because the
    # member grammar addresses files by their upstream-relative path.
    assert plan.objects[0].relpath == (
        "upstream/gefs.20260817/00/atmos/pgrb2ap5/gec00.t00z.pgrb2a.0p50.f000")
    assert [step.name for step in plan.compose] == [
        "pairs/gec00.t00z.pair.0p50.f000.grib2",
        "pairs/gec00.t00z.pair.0p50.f003.grib2",
    ]
    # The concatenation order is a+b, and the pair is what prep consumes.
    assert [part.role for part in plan.compose[0].parts] == [
        "pgrb2a", "pgrb2b"]
    assert [path.name for path in plan.primary_files] == [
        "gec00.t00z.pair.0p50.f000.grib2",
        "gec00.t00z.pair.0p50.f003.grib2",
    ]


def test_gefs_perturbed_member_selects_its_own_token():
    plan = fetch_routes.resolve_request(
        "gefs", cycle=datetime(2026, 8, 17, 0), hours=3, member="p07")
    assert "gep07.t00z.pgrb2a.0p50.f000" in plan.objects[0].url


def test_an_unknown_member_refuses_naming_the_declared_set():
    with pytest.raises(ValueError) as error:
        fetch_routes.resolve_request(
            "gefs", cycle=datetime(2026, 8, 17, 0), hours=3, member="p99")
    assert "p99" in str(error.value)
    assert "c00" in str(error.value)


def test_aigefs_member_identity_is_only_in_the_path():
    plan = fetch_routes.resolve_request(
        "aigefs", cycle=datetime(2026, 8, 17, 0), hours=6)
    assert plan.objects[0].relpath == (
        "upstream/aigefs.20260817/00/mem000/model/atmos/grib2/"
        "aigefs.t00z.pres.f000.grib2")
    assert plan.member_set == "aigefs-ensemble-grib2-members-v1"


def test_aigfs_is_nomads_only_and_says_why_the_s3_copy_is_not_a_mirror():
    route = fetch_routes.route_for("aigfs")
    assert [host.name for host in route.hosts] == ["nomads"]
    with pytest.raises(ValueError) as error:
        fetch_routes.resolve_request(
            "aigfs", cycle=datetime(2026, 8, 17, 0), hours=6, host="aws")
    message = str(error.value)
    assert "aws" in message and "nomads" in message
    assert "subCentre" in route.host_note


def test_aigfs_plans_its_declared_same_cycle_gdas_donor():
    plan = fetch_routes.resolve_request(
        "aigfs", cycle=datetime(2026, 8, 17, 0), hours=6)
    assert len(plan.donors) == 1
    donor = plan.donors[0]
    assert donor.source == "gdas"
    assert donor.leads == (0,)
    assert donor.cycle == datetime(2026, 8, 17, 0)
    assert donor.role == "physical_analysis_surface_data"


def test_ecmwf_open_data_stamps_the_full_cycle_and_an_unpadded_lead():
    # Asked while the cycle is still on the publisher's door (it keeps
    # 72 h); a cycle older than that is planned from the mirror.
    plan = fetch_routes.resolve_request(
        "ecmwf-open-data", cycle=datetime(2026, 8, 16, 0), hours=3,
        now=datetime(2026, 8, 16, 12))
    assert [obj.url for obj in plan.objects] == [
        "https://data.ecmwf.int/forecasts/20260816/00z/ifs/0p25/oper/"
        "20260816000000-0h-oper-fc.grib2",
        "https://data.ecmwf.int/forecasts/20260816/00z/ifs/0p25/oper/"
        "20260816000000-3h-oper-fc.grib2",
    ]


def test_aifs_is_six_hourly_and_supplements_from_its_step0_object():
    plan = fetch_routes.resolve_request(
        "aifs", cycle=datetime(2026, 8, 17, 0), hours=6)
    assert plan.leads == (0, 6)
    assert plan.objects[0].url.endswith(
        "20260817/00z/aifs-single/0p25/oper/"
        "20260817000000-0h-oper-fc.grib2")
    # A window starting at f000: the step-0 object is also the first
    # input, and nothing extra is fetched.
    assert [obj.relpath for obj in plan.objects] == [
        "20260817000000-0h-oper-fc.grib2", "20260817000000-6h-oper-fc.grib2"]
    assert [path.name for path in plan.primary_files] == [
        "20260817000000-0h-oper-fc.grib2", "20260817000000-6h-oper-fc.grib2"]
    assert [path.name for path in plan.supplement_files] == [
        "20260817000000-0h-oper-fc.grib2"]
    assert plan.supplement_role == "aifs_single_in_band_surface"


def test_an_aifs_window_past_f000_fetches_the_cycle_step0_object_as_its_supplement():
    """AIFS publishes its land mask and surface geopotential at step 0 only.

    Named breakage (A133, WOOF 1.0.0): a start at f012 bound the f012
    object as the in-band surface, and preparation found 0 of its 122
    messages matching.  The step-0 object is planned beside the window,
    in the first lead's group, bound only as the supplement and never
    an input of the forcing series.
    """

    plan = fetch_routes.resolve_request(
        "aifs", cycle=datetime(2026, 9, 29, 12), hours=6, start_hour=12,
        now=datetime(2026, 9, 30, 0))
    assert plan.leads == (12, 18)
    assert [(obj.relpath, obj.lead, obj.role) for obj in plan.objects] == [
        ("20260929120000-12h-oper-fc.grib2", 12, "oper-fc"),
        ("20260929120000-0h-oper-fc.grib2", 0, "oper-fc"),
        ("20260929120000-18h-oper-fc.grib2", 18, "oper-fc"),
    ]
    assert plan.objects[1].url == (
        "https://data.ecmwf.int/forecasts/20260929/12z/aifs-single/0p25/"
        "oper/20260929120000-0h-oper-fc.grib2")
    assert [path.name for path in plan.primary_files] == [
        "20260929120000-12h-oper-fc.grib2", "20260929120000-18h-oper-fc.grib2"]
    assert [path.name for path in plan.supplement_files] == [
        "20260929120000-0h-oper-fc.grib2"]


def test_the_aifs_handoff_binds_the_step0_object_and_lists_only_the_window(tmp_path):
    plan = fetch_routes.resolve_request(
        "aifs", cycle=datetime(2026, 9, 29, 12), hours=6, start_hour=12,
        now=datetime(2026, 9, 30, 0))
    inputs, _command = fetch_routes.write_handoff(plan, tmp_path)
    assert [Path(line).name for line in
            inputs.read_text(encoding="utf-8").splitlines()] == [
        "20260929120000-12h-oper-fc.grib2", "20260929120000-18h-oper-fc.grib2"]
    argv = json.loads(
        (tmp_path / fetch_routes.PREP_ARGUMENTS_NAME).read_text(
            encoding="utf-8"))["argv"]
    bindings = [argv[index + 1] for index, token in enumerate(argv)
                if token == "--supplement"]
    # write_handoff binds through the I/O spelling (9e3980670: a deep cache
    # must stay openable on Windows), so on Windows the binding is the
    # extended-length path (prefix \\?\); elsewhere io_path is the path itself.
    step0 = io_path(tmp_path / "20260929120000-0h-oper-fc.grib2").resolve()
    assert bindings == [f"aifs_single_in_band_surface={step0}"]


def test_the_step0_object_is_priced_as_one_more_object_of_its_kind():
    """The extra object is an AIFS object like the others, so the download
    estimate a later start earns is three objects, not two."""

    from woof import download_budget

    request = {"source": "aifs", "cycle": "2026-09-29T12", "hours": 6}
    first = download_budget.download_estimate(request)
    later = download_budget.download_estimate(
        dict(request, forecast_start_hour=12))
    # Each estimate also carries the same cycle's GFS f000 ICEC record, the
    # declared sea-ice donor (AIFS publishes no sea-ice field), once: taken
    # by the .idx (fix/ecmwf-record-subset), not the whole 513 MB object.
    assert (first["objects"], later["objects"]) == (3, 4)
    aifs_object = later["bytes"] - first["bytes"]
    assert aifs_object and first["bytes"] == 2 * aifs_object + 101_000


def test_only_producers_with_step0_statics_declare_them():
    """The step-0 shapes are table rows, and exactly these routes have them.

    IFS open data carries its land mask and surface geopotential in every
    step (its f024 and f150 starts prepared, 2026-09-30); ICON's statics
    are cycle objects with no lead; AIGFS borrows a same-cycle analysis.
    AIFS and GDPS publish theirs at the analysis step only.
    """

    step0_supplement = sorted(
        source for source in fetch_routes.route_ids()
        if (fetch_routes.route_for(source).prep.get("supplement") or {}
            ).get("from") == "step0")
    step0_rows = sorted(
        source for source in fetch_routes.route_ids()
        if any(row.leads == "step0"
               for row in fetch_routes.route_for(source).files))
    assert step0_supplement == ["aifs"]
    assert step0_rows == ["gem-gdps", "hrrr-native", "hrrr-prs"]


def test_icon_eu_expands_125_field_objects_a_lead_plus_two_invariants():
    plan = fetch_routes.resolve_request(
        "icon-eu", cycle=datetime(2026, 8, 17, 0), hours=6)
    # The exact census the 6 h battery downloaded by hand: 125 x 7 + 2.
    assert len(plan.objects) == 877
    urls = {obj.url for obj in plan.objects}
    base = "https://opendata.dwd.de/weather/nwp/icon-eu/grib/00"
    assert (f"{base}/t/icon-eu_europe_regular-lat-lon_pressure-level_"
            "2026081700_000_850_T.grib2.bz2") in urls
    assert (f"{base}/t_2m/icon-eu_europe_regular-lat-lon_single-level_"
            "2026081700_003_T_2M.grib2.bz2") in urls
    assert (f"{base}/w_so/icon-eu_europe_regular-lat-lon_soil-level_"
            "2026081700_006_243_W_SO.grib2.bz2") in urls
    assert (f"{base}/hsurf/icon-eu_europe_regular-lat-lon_time-invariant_"
            "2026081700_HSURF.grib2.bz2") in urls
    # HSURF is the terrain supplement; FR_LAND rides in with the state.
    assert [path.name for path in plan.supplement_files] == [
        "icon-eu_europe_regular-lat-lon_time-invariant_2026081700_"
        "HSURF.grib2.bz2"]
    assert plan.supplement_role == "icon_eu_invariant_surface"


def test_icon_d2_expands_model_levels_and_all_height_interfaces():
    plan = fetch_routes.resolve_request(
        "icon-d2", cycle=datetime(2026, 9, 27, 12), hours=3)
    # 10 fields x 65 model levels + 8 single-level + 9 T_SO + 8 W_SO
    # per lead, then 66 height interfaces and five other invariants.
    assert plan.leads == (0, 1, 2, 3)
    assert len(plan.objects) == 675 * 4 + 71
    urls = {obj.url for obj in plan.objects}
    base = "https://opendata.dwd.de/weather/nwp/icon-d2/grib/12"
    # The names DWD publishes, lower-case, with "2d" for single-level
    # objects and "000_0" for the time-invariant ones.
    assert (f"{base}/t/icon-d2_germany_icosahedral_model-level_"
            "2026092712_000_1_t.grib2.bz2") in urls
    assert (f"{base}/t_2m/icon-d2_germany_icosahedral_single-level_"
            "2026092712_003_2d_t_2m.grib2.bz2") in urls
    assert (f"{base}/t_so/icon-d2_germany_icosahedral_soil-level_"
            "2026092712_002_5_t_so.grib2.bz2") in urls
    assert (f"{base}/w_so/icon-d2_germany_icosahedral_soil-level_"
            "2026092712_001_729_w_so.grib2.bz2") in urls
    for field in ("hsurf", "fr_land", "fr_ice", "clat", "clon"):
        assert (f"{base}/{field}/icon-d2_germany_icosahedral_time-invariant_"
                f"2026092712_000_0_{field}.grib2.bz2") in urls
    assert not any("regular-lat-lon" in url for url in urls)
    assert not any("_150_" in url or "_100_" in url for url in urls)
    assert [path.name for path in plan.supplement_files] == [
        "icon-d2_germany_icosahedral_time-invariant_2026092712_000_0_"
        "hsurf.grib2.bz2"]
    assert plan.supplement_role == "icon_d2_invariant_surface"


def test_icon_d2_aliases_resolve_to_its_route():
    for alias in ("icon-d2", "icon-2km", "dwd-icon-d2"):
        assert fetch_routes.route_for(alias).source_id == "icon-d2"


def test_gem_gdps_expands_174_a_lead_with_analysis_only_invariants():
    plan = fetch_routes.resolve_request(
        "gem-gdps", cycle=datetime(2026, 8, 16, 0), hours=3)
    # 174 state files at each of two leads, plus the two analysis-only
    # invariants and the terrain supplement, all at PT000H: 351 -- the
    # exact count the GEM staging lane fetched.
    assert len(plan.objects) == 351
    urls = {obj.url for obj in plan.objects}
    stem = "https://dd.weather.gc.ca/20260816/WXO-DD/model_gdps/15km/00"
    assert (f"{stem}/000/20260816T00Z_MSC_GDPS_AirTemp_IsbL-0850_"
            "LatLon0.15_PT000H.grib2") in urls
    assert (f"{stem}/003/20260816T00Z_MSC_GDPS_WindV_AGL-10m_"
            "LatLon0.15_PT003H.grib2") in urls
    assert (f"{stem}/000/20260816T00Z_MSC_GDPS_SeaIceFraction_Sfc_"
            "LatLon0.15_PT000H.grib2") in urls
    # The analysis-only invariants never appear at a later lead: their
    # /003/ URLs are 404s, which is the once-per-cycle publication shape.
    assert (f"{stem}/003/20260816T00Z_MSC_GDPS_SeaIceFraction_Sfc_"
            "LatLon0.15_PT003H.grib2") not in urls
    # Terrain is held OUT of the composed primary and passed once.
    assert [step.name for step in plan.compose] == [
        "composed/gdps_2026081600_PT000H.grib2",
        "composed/gdps_2026081600_PT003H.grib2",
    ]
    assert [path.name for path in plan.supplement_files] == [
        "20260816T00Z_MSC_GDPS_GeopotentialHeight_Sfc_"
        "LatLon0.15_PT000H.grib2"]
    assert plan.supplement_role == "gdps_analysis_invariant_surface"


def test_a_gdps_window_past_f000_takes_its_analysis_invariants_from_pt000h():
    """GDPS publishes land mask, sea ice and orography at PT000H only.

    Named breakage: the invariant rows rendered the WINDOW's first lead,
    so a start at f012 planned PT012H objects that Datamart answers 404
    (measured 2026-09-30 for all three).  They are the cycle's step-0
    objects, composed into the window's first valid time.
    """

    plan = fetch_routes.resolve_request(
        "gem-gdps", cycle=datetime(2026, 8, 16, 0), hours=3, start_hour=12)
    assert plan.leads == (12, 15)
    assert len(plan.objects) == 351
    stem = "https://dd.weather.gc.ca/20260816/WXO-DD/model_gdps/15km/00"
    urls = {obj.url for obj in plan.objects}
    for name in ("LandWaterProportion_Sfc", "SeaIceFraction_Sfc",
                 "GeopotentialHeight_Sfc"):
        assert (f"{stem}/000/20260816T00Z_MSC_GDPS_{name}_"
                "LatLon0.15_PT000H.grib2") in urls
        assert not any(f"_{name}_LatLon0.15_PT012H" in url for url in urls)
    first, second = plan.compose
    assert first.name == "composed/gdps_2026081600_PT012H.grib2"
    invariant_parts = [part.relpath for part in first.parts
                       if part.role == "analysis-invariant"]
    assert invariant_parts == [
        "20260816T00Z_MSC_GDPS_LandWaterProportion_Sfc_LatLon0.15_PT000H.grib2",
        "20260816T00Z_MSC_GDPS_SeaIceFraction_Sfc_LatLon0.15_PT000H.grib2"]
    assert not any(part.role == "analysis-invariant" for part in second.parts)
    assert all("PT015H" in part.relpath for part in second.parts)
    assert [path.name for path in plan.supplement_files] == [
        "20260816T00Z_MSC_GDPS_GeopotentialHeight_Sfc_"
        "LatLon0.15_PT000H.grib2"]


def test_a_gdps_window_from_f000_plans_what_it_always_planned():
    plan = fetch_routes.resolve_request(
        "gem-gdps", cycle=datetime(2026, 8, 16, 0), hours=3)
    first = plan.compose[0]
    assert [part.relpath for part in first.parts
            if part.role == "analysis-invariant"] == [
        "20260816T00Z_MSC_GDPS_LandWaterProportion_Sfc_LatLon0.15_PT000H.grib2",
        "20260816T00Z_MSC_GDPS_SeaIceFraction_Sfc_LatLon0.15_PT000H.grib2"]
    # The state parts precede the invariants, in the order they did.
    roles = [part.role for part in first.parts]
    assert roles == sorted(roles, key=("state", "analysis-invariant").index)


def test_a_retired_or_unknown_lead_rule_refuses_at_load():
    for rule in ("first", "firstt", "every"):
        with pytest.raises(ValueError, match="lead rule"):
            fetch_routes._file_row({"role": "x", "path": "{F}.grib2",
                                    "leads": rule})
    assert fetch_routes._file_row(
        {"role": "x", "path": "{F}.grib2", "leads": "step0"}).leads == "step0"


def test_an_unresolvable_supplement_origin_refuses_at_load(monkeypatch):
    document = json.loads(json.dumps(fetch_routes._load_table()))
    document["routes"]["aifs"]["prep"]["supplement"]["from"] = "first_input"
    monkeypatch.setattr(fetch_routes, "_load_table", lambda: document)
    with pytest.raises(ValueError, match="first_input"):
        fetch_routes._build_routes()
    document["routes"]["aifs"]["prep"]["supplement"]["from"] = "role:nope"
    with pytest.raises(ValueError, match="role:nope"):
        fetch_routes._build_routes()


# --------------------------------------------------------------------------
# Transport policy
# --------------------------------------------------------------------------

def test_idx_subset_refuses_with_the_row_s_own_reason():
    # GEFS was this example until fix/gefs-record-subset made its pair take
    # the mapped records by .idx; GDPS's refusal is structural (one record
    # per object), so it stays.
    with pytest.raises(ValueError) as error:
        fetch_routes.resolve_mode("gem-gdps", "idx-subset")
    message = str(error.value)
    assert "idx-subset" in message
    assert "the file IS the record" in message


def test_an_area_crop_refuses_naming_where_the_crop_actually_happens():
    with pytest.raises(ValueError) as error:
        fetch_routes.resolve_request(
            "rap", cycle=datetime(2026, 8, 17, 0), hours=1,
            area="30,-110,45,-90")
    message = str(error.value)
    assert "--area" in message
    assert "woof prep" in message


# --------------------------------------------------------------------------
# Named refusals for sources with no public bytes
# --------------------------------------------------------------------------

@pytest.mark.parametrize("source_id,fragment,door", [
    ("20crv3", "every-member", "--source-root DIR --author-only"),
    # The generic adapter has no folder layout to bind: its door is the
    # mapping and composition the caller brings with the files.
    ("mapped", "generic declarative adapter", "--mapping MAPPING.json"),
])
def test_a_source_without_fetchable_bytes_points_at_its_door(
        source_id, fragment, door):
    with pytest.raises(ValueError) as error:
        fetch_routes.route_for(source_id)
    message = str(error.value)
    assert fragment in message
    assert door in message


def test_the_20crv3_refusal_reaches_through_the_alias():
    with pytest.raises(ValueError) as error:
        fetch_routes.route_for("20cr")
    assert "--source-root" in str(error.value)


# --------------------------------------------------------------------------
# Transfer, composition and the handoff, over a fake transport
# --------------------------------------------------------------------------

def _grib(payload: bytes = b"body") -> bytes:
    return b"GRIB" + payload + b"7777"


def _fake_downloader(record):
    def download(url, dest, *, magic, opener=None):
        record.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        body = _grib(url.encode()) if magic == "GRIB" else b"BZh" + url.encode()
        dest.write_bytes(body)
        return {"name": dest.name, "bytes": len(body),
                "sha256": __import__("hashlib").sha256(body).hexdigest(),
                "url": url}
    return download


def test_a_table_route_moves_every_object_and_publishes_its_receipts(tmp_path):
    plan = fetch_routes.resolve_request(
        "rap", cycle=datetime(2026, 8, 16, 0), hours=1)
    seen: list[str] = []
    payload = fetch_routes.run_plan(
        plan, out=tmp_path, downloader=_fake_downloader(seen),
        progress=lambda *_: None)

    assert len(seen) == 2
    assert payload["schema"] == fetch_routes.ROUTE_MANIFEST_SCHEMA
    assert payload["request"]["cycle"] == "2026-08-16T00Z"
    assert payload["mode"] == "full-file"
    # The pool receipt rides in the manifest, so a slow fetch is
    # explicable after the fact.
    assert payload["concurrency"]["files"] == 2
    assert (payload["concurrency"]["workers_requested"]
            == fetch_routes.fetch_pool.DEFAULT_FILE_WORKERS)
    sums = (tmp_path / fetch_routes.SHA256SUMS_NAME).read_text().splitlines()
    assert len(sums) == 2
    assert all(len(line.split("  ")[0]) == 64 for line in sums)


def test_a_different_request_into_the_same_directory_refuses(tmp_path):
    first = fetch_routes.resolve_request(
        "rap", cycle=datetime(2026, 8, 16, 0), hours=1)
    fetch_routes.run_plan(first, out=tmp_path,
                          downloader=_fake_downloader([]),
                          progress=lambda *_: None)
    second = fetch_routes.resolve_request(
        "rap", cycle=datetime(2026, 8, 16, 12), hours=1)
    with pytest.raises(ValueError) as error:
        fetch_routes.run_plan(second, out=tmp_path,
                              downloader=_fake_downloader([]),
                              progress=lambda *_: None)
    assert "--force-refetch" in str(error.value)


def test_a_truncated_object_refuses_by_name(tmp_path, monkeypatch):
    from types import SimpleNamespace
    waits: list[float] = []
    monkeypatch.setattr(fetch_routes, "time",
                        SimpleNamespace(sleep=waits.append), raising=False)
    plan = fetch_routes.resolve_request(
        "rap", cycle=datetime(2026, 8, 16, 0), hours=1)

    def truncating(url, dest, *, magic, opener=None):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"GRIB" + b"half a message")
        fetch_routes._verify_payload(dest, magic=magic, label=dest.name)
        return {}

    with pytest.raises(ValueError) as error:
        fetch_routes.run_plan(plan, out=tmp_path, downloader=truncating,
                              progress=lambda *_: None)
    assert "end marker" in str(error.value)
    # Both hours are truncated, and the first to fail ends the request
    # (the other is stopped, not waited out), so the refusal names
    # whichever one that was.
    assert re.search(r"rap\.t00z\.awip32f0[01]\.grib2", str(error.value))
    # A payload that does not verify is asked for again on the shared
    # schedule before the refusal.
    assert waits and set(waits) <= {2, 4, 8, 16}


def test_gefs_concatenates_its_pair_in_the_declared_order(tmp_path):
    plan = fetch_routes.resolve_request(
        "gefs", cycle=datetime(2026, 8, 17, 0), hours=3)
    payload = fetch_routes.run_plan(
        plan, out=tmp_path, downloader=_fake_downloader([]),
        progress=lambda *_: None)
    assert len(payload["composed"]) == 2
    pair = tmp_path / "pairs" / "gec00.t00z.pair.0p50.f000.grib2"
    assert pair.is_file()
    body = pair.read_bytes()
    assert body.index(b"pgrb2ap5") < body.index(b"pgrb2bp5")


def test_the_handoff_binds_the_input_list_and_the_supplement_role(tmp_path):
    plan = fetch_routes.resolve_request(
        "rrfs", cycle=datetime(2026, 8, 17, 0), hours=1)
    fetch_routes.run_plan(plan, out=tmp_path,
                          downloader=_fake_downloader([]),
                          progress=lambda *_: None)
    inputs, command = fetch_routes.write_handoff(plan, tmp_path)
    listed = inputs.read_text().splitlines()
    assert len(listed) == 4
    assert all(line.endswith(".grib2") and ("prslev" in line or "2dfld" in line) for line in listed)
    assert sum("prslev" in line for line in listed) == sum("2dfld" in line for line in listed) == 2
    text = command.read_text()
    assert "woof prep \\" in text
    assert "--source rrfs" in text
    assert "--input-list" in text
    # Every continuation but the last carries its backslash, and no
    # comment ever rides on a continued line: the file runs as written.
    body = [line for line in text.splitlines()
            if line and not line.lstrip().startswith("#")]
    assert all(line.endswith("\\") for line in body[:-1])
    assert not body[-1].endswith("\\")
    assert "rrfs_prslev_2dfld_in_band_surface=" in text
    assert text.count("--supplement") == 2


def test_the_handoff_publishes_machine_readable_prep_arguments(tmp_path):
    """`prep-arguments.json` carries the SAME binding as the text file.

    The staged run-plan chain composes its preparation from this
    document, so it must hold argv TOKENS (no quoting convention to
    round-trip), the caller-owned flags by name, and an explicit list
    of any supplement role this fetch left unbound.
    """

    plan = fetch_routes.resolve_request(
        "rrfs", cycle=datetime(2026, 8, 17, 0), hours=1)
    fetch_routes.run_plan(plan, out=tmp_path,
                          downloader=_fake_downloader([]),
                          progress=lambda *_: None)
    inputs, command = fetch_routes.write_handoff(plan, tmp_path)
    document = json.loads(
        (tmp_path / fetch_routes.PREP_ARGUMENTS_NAME).read_text())
    assert document["schema"] == fetch_routes.PREP_ARGUMENTS_SCHEMA
    argv = document["argv"]
    assert argv[:4] == ["--source", "rrfs", "--input-list",
                        str(inputs.resolve())]
    assert argv.count("--supplement") == 2
    assert "--author-input-manifest" in argv
    assert document["unbound_supplement_roles"] == []
    assert document["caller_supplies"] == [
        "--wps-namelist", "--experiment-config", "--geog-root",
        "--output-root"]
    # Token for token, the text command is these argv pairs rendered.
    text = command.read_text()
    for flag, value in zip(argv[::2], argv[1::2]):
        assert flag in text
        assert value.split("=")[0] in text


def test_the_prep_arguments_name_an_unbound_donor_role(tmp_path):
    """A donor this fetch did not bring is a named hole, not a surprise."""

    plan = fetch_routes.resolve_request(
        "aigfs", cycle=datetime(2026, 8, 17, 0), hours=6)
    fetch_routes.run_plan(plan, out=tmp_path,
                          downloader=_fake_downloader([]),
                          progress=lambda *_: None)
    fetch_routes.write_handoff(plan, tmp_path)
    document = json.loads(
        (tmp_path / fetch_routes.PREP_ARGUMENTS_NAME).read_text())
    assert document["unbound_supplement_roles"] == [
        "physical_analysis_surface_data"]


def test_the_handoff_names_member_prep_for_an_ensemble_route(tmp_path):
    plan = fetch_routes.resolve_request(
        "aigefs", cycle=datetime(2026, 8, 17, 0), hours=6)
    fetch_routes.run_plan(plan, out=tmp_path,
                          downloader=_fake_downloader([]),
                          progress=lambda *_: None)
    _, command = fetch_routes.write_handoff(plan, tmp_path)
    text = command.read_text()
    assert "woof-member-prep" in text
    assert "aigefs-ensemble-grib2-members-v1" in text
    # The donor is declared even when this run did not fetch it.
    assert "physical_analysis_surface_data" in text


@pytest.mark.parametrize("hours", [6, 12, 24])
@pytest.mark.parametrize("member", [None, "mem001"])
def test_member_prep_handoff_steps_match_requested_leads(tmp_path, hours, member):
    from woof.member_prep import build_parser

    plan = fetch_routes.resolve_request(
        "aigefs", cycle=datetime(2026, 8, 17, 0), hours=hours, member=member)
    _, command = fetch_routes.write_handoff(plan, tmp_path)
    lines = command.read_text().splitlines()
    start = next(i for i, line in enumerate(lines)
                 if line.startswith("#   woof-member-prep"))
    parts = []
    for line in lines[start:]:
        parts.append(line.lstrip("#").strip().rstrip("\\").strip())
        if not line.rstrip().endswith("\\"):
            break
    argv = shlex.split(" ".join(parts), posix=False)
    arguments = build_parser().parse_args(argv[1:])
    document = json.loads(
        (tmp_path / fetch_routes.PREP_ARGUMENTS_NAME).read_text())
    assert arguments.steps == plan.leads
    assert list(arguments.steps) == document["member_prep"]["steps"]


def test_member_prep_handoff_preserves_paths_with_spaces(tmp_path):
    from woof.filesystem_paths import io_path
    from woof.member_prep import build_parser

    out = io_path(tmp_path / "member fetch")
    plan = fetch_routes.resolve_request(
        "aigefs", cycle=datetime(2026, 8, 17, 0), hours=6)
    fetch_routes.run_plan(plan, out=out,
                          downloader=_fake_downloader([]),
                          progress=lambda *_: None)
    _, command = fetch_routes.write_handoff(plan, out)
    lines = command.read_text().splitlines()
    start = next(i for i, line in enumerate(lines)
                 if line.startswith("#   woof-member-prep"))
    parts = []
    for line in lines[start:]:
        parts.append(line.lstrip("#").strip().rstrip("\\").strip())
        if not line.rstrip().endswith("\\"):
            break
    argv = shlex.split(" ".join(parts), posix=True)
    arguments = build_parser().parse_args(argv[1:])
    assert arguments.member_set == plan.member_set
    assert arguments.member == plan.member
    assert arguments.inputs == out.resolve() / "upstream"
    assert arguments.output == out.resolve() / "members"


def test_the_handoff_binds_a_donor_that_was_fetched(tmp_path):
    plan = fetch_routes.resolve_request(
        "aigfs", cycle=datetime(2026, 8, 17, 0), hours=6)
    fetch_routes.run_plan(plan, out=tmp_path,
                          downloader=_fake_downloader([]),
                          progress=lambda *_: None)
    donor = tmp_path / "donor-gdas" / "gdas.t00z.pgrb2.0p25.f000"
    donor.parent.mkdir(parents=True, exist_ok=True)
    donor.write_bytes(_grib())
    _, command = fetch_routes.write_handoff(
        plan, tmp_path, donor_files={"physical_analysis_surface_data": donor})
    text = command.read_text()
    assert "--supplement" in text
    assert "physical_analysis_surface_data=" in text
    assert "not fetched" not in text


@pytest.mark.parametrize("source", fetch_routes.route_ids())
def test_every_route_prints_the_whole_prep_line_it_published(source,
                                                             tmp_path):
    """The printed ``next:`` line is the published argv, token for token.

    It used to print ``--source`` and ``--input-list`` and point at
    ``prep-command.txt`` for the rest, so the line completed with the
    four flags it named was refused at the door for the supplement
    binding and the manifest flag it left out.  The argv in
    ``prep-arguments.json`` is what the chained doors compose from, so
    the printed line is held to exactly that, and the flags it leaves to
    the reader are exactly the document's ``caller_supplies`` plus any
    supplement role the fetch did not bind.
    """

    route = fetch_routes.route_for(source)
    cycle = datetime(2026, 8, 17, route.cycle_hours[0])
    if route.publication_eras and not any(
            era.valid_from <= cycle and
            (era.valid_until is None or cycle < era.valid_until)
            for era in route.publication_eras):
        cycle = fetch_routes.first_preparable_cycle(route)
        assert cycle is not None
    plan = fetch_routes.resolve_request(
        source, cycle=cycle, start_hour=route.first_lead,
        hours=min(route.default_cadence, fetch_routes.ladder_for(route, cycle)[-1]-route.first_lead))
    fetch_routes.write_handoff(plan, tmp_path)
    document = json.loads(
        (tmp_path / fetch_routes.PREP_ARGUMENTS_NAME).read_text())
    lines = fetch_routes.handoff_lines(plan, tmp_path)
    commands = [line.strip() for line in lines
                if line.strip().startswith("woof prep ")]
    assert len(commands) == 1, lines
    # The printed line is the platform's own quoting: on Windows the paths
    # are bare backslash paths, which POSIX splitting would eat.
    assert shlex.split(commands[0], posix=os.name != "nt") == [
        "woof", "prep", *document["argv"]]
    comments = " ".join(line for line in lines
                        if line.strip().startswith("#"))
    for flag in document["caller_supplies"]:
        assert flag in comments, flag
    for role in document["unbound_supplement_roles"]:
        assert f"--supplement {role}=" in comments, role
