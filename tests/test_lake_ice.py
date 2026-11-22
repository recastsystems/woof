"""Observed lake ice (woof.ingest.lake_ice): the table, the decision rule,
the field edit, the acquisition ladder and the loud fallback.

The Rust reader itself is tested in tools/grib1_bridge/src/ims_snow_ice.rs;
the end-to-end witness (a GFS start of 2025-02-20 12Z against IMS 1 km) is
recorded with the 2.9.1 lake-ice fix evidence.
"""
from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError

import numpy as np
import pytest

from woof.ingest import lake_ice


def test_the_table_rows_are_ims_1km_then_4km():
    rows = lake_ice.load_lake_ice_sources()
    assert [row.id for row in rows] == ["ims-1km", "ims-4km"]
    day = datetime(2025, 2, 20)
    assert rows[0].urls(day)[0].endswith("/1km/2025/ims2025051_00UTC_1km_v1.3.asc.gz")
    assert rows[1].urls(day)[0].endswith("/4km/2025/ims2025051_00UTC_4km_v1.3.asc.gz")


def _counts(*rows):
    return np.array(rows, dtype=np.uint32)


def test_water_pixels_decide_where_the_analysis_maps_water():
    counts = _counts(
        [0, 3, 0, 6, 0],   # mostly ice -> frozen
        [0, 6, 0, 3, 0],   # mostly open -> open
        [0, 4, 1, 4, 0],   # tie, centre ice -> frozen
        [0, 4, 1, 4, 0],   # tie, centre water -> open
        [0, 1, 0, 0, 8],   # one open pixel among snowy shore -> open (water decides)
    )
    centre = np.array([3, 1, 3, 1, 4])
    frozen, observed = lake_ice.classify_frozen(counts, centre)
    assert frozen.tolist() == [True, False, True, False, False]
    assert observed.all()


def test_snow_cover_decides_a_lake_the_water_mask_does_not_carry():
    counts = _counts([0, 0, 2, 0, 7], [0, 0, 7, 0, 2], [0, 0, 0, 0, 0])
    frozen, observed = lake_ice.classify_frozen(counts, np.array([4, 2, 0]))
    assert frozen.tolist() == [True, False, False]
    assert observed.tolist() == [True, True, False]


def test_window_radius_is_half_a_cell_in_pixels():
    assert lake_ice.window_radius(1000.0, 1000.0) == 0
    assert lake_ice.window_radius(3000.0, 1000.0) == 1
    assert lake_ice.window_radius(12000.0, 4000.0) == 1
    assert lake_ice.window_radius(1000.0, 4000.0) == 0


def test_apply_freezes_and_caps_and_keeps_the_forcing_ice():
    seaice = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    skin = np.array([[273.6, 250.0], [265.0, 280.0]])
    frozen = np.array([[True, True], [False, False]])
    fields = {"SEAICE": seaice, "SKINTEMP": skin.astype(np.float32)}
    out, capped, receipt = lake_ice.apply_lake_ice(
        fields, skin, frozen, skin_name="SKINTEMP")
    assert out["SEAICE"].dtype == np.float32
    assert out["SEAICE"].tolist() == [[1.0, 1.0], [1.0, 0.0]]
    assert capped.tolist() == [[271.4, 250.0], [265.0, 280.0]]
    assert np.allclose(out["SKINTEMP"], capped)
    assert fields["SEAICE"][0, 0] == 0.0  # the caller's field is not edited
    assert receipt["cells_newly_frozen"] == 2
    assert receipt["skin_capped_cells"] == 1


def test_apply_writes_both_ice_names_a_field_set_carries():
    fields = {"XICE": np.zeros((1, 2)), "SEAICE": np.zeros((1, 2))}
    out, _, _ = lake_ice.apply_lake_ice(
        fields, np.full((1, 2), 260.0), np.array([[True, False]]))
    assert out["XICE"].tolist() == [[1.0, 0.0]]
    assert out["SEAICE"].tolist() == [[1.0, 0.0]]


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_acquisition_walks_back_a_day_when_the_newest_is_not_posted(tmp_path):
    asked = []

    def opener(request, timeout):
        asked.append(request.full_url)
        if "2025051" in request.full_url:
            raise HTTPError(request.full_url, 404, "Not Found", None, None)
        return _Response(b"\x1f\x8b" + b"\0" * 32)

    observation, attempts = lake_ice.acquire_observed_lake_ice(
        datetime(2025, 2, 20, 12), cache_root=tmp_path, opener=opener)
    assert observation is not None
    assert observation.source.id == "ims-1km"
    assert observation.valid_time == datetime(2025, 2, 19)
    assert observation.path.parent == tmp_path / "ims-1km"
    assert "HTTP 404" in attempts[0]
    # A second preparation reads the cached file and asks nothing.
    asked.clear()

    def no_network(request, timeout):
        raise AssertionError("cached analysis was fetched again")

    again, _ = lake_ice.acquire_observed_lake_ice(
        datetime(2025, 2, 19, 18), cache_root=tmp_path, opener=no_network)
    assert again.path == observation.path


def test_no_network_stops_at_the_first_failure_and_says_why(tmp_path):
    calls = []

    def opener(request, timeout):
        calls.append(request.full_url)
        raise URLError("Name or service not known")

    observation, attempts = lake_ice.acquire_observed_lake_ice(
        datetime(2025, 2, 20, 12), cache_root=tmp_path, opener=opener)
    assert observation is None
    assert len(calls) == 1
    assert "Name or service not known" in attempts[0]


def test_offline_a_cached_older_day_or_4km_file_is_still_used(tmp_path):
    """Breakage (2.9.1 verify): offline, the first refused connection ended
    the search, so yesterday's cached 1 km file, or today's cached 4 km
    file, was never read and the run fell back to the forcing's ice, which
    the configuration reference says an offline machine avoids by putting
    the file in the cache."""
    rows = lake_ice.load_lake_ice_sources()
    calls = []

    def opener(request, timeout):
        calls.append(request.full_url)
        raise URLError("connection refused")

    # Yesterday's 1 km analysis is in the cache; today's is not.
    yesterday = datetime(2025, 2, 19)
    cached = tmp_path / rows[0].id / Path(rows[0].urls(yesterday)[0]).name
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"\x1f\x8b" + b"\0" * 32)
    observation, attempts = lake_ice.acquire_observed_lake_ice(
        datetime(2025, 2, 20, 12), cache_root=tmp_path, opener=opener)
    assert observation is not None and observation.path == cached
    assert observation.valid_time == yesterday
    assert len(calls) == 1 and "connection refused" in attempts[0]
    # Only the 4 km row's file for today is cached.
    cached.unlink()
    four = tmp_path / rows[1].id / Path(rows[1].urls(datetime(2025, 2, 20))[0]).name
    four.parent.mkdir(parents=True)
    four.write_bytes(b"\x1f\x8b" + b"\0" * 32)
    calls.clear()
    observation, _ = lake_ice.acquire_observed_lake_ice(
        datetime(2025, 2, 20, 12), cache_root=tmp_path, opener=opener)
    assert observation is not None and observation.source.id == rows[1].id
    assert len(calls) == 1


def test_a_failed_acquisition_falls_back_loudly(monkeypatch, capsys):
    monkeypatch.setattr(lake_ice, "acquire_observed_lake_ice",
                        lambda valid_time: (None, ["https://x/y: HTTP 404"]))
    provider = lake_ice.resolve_lake_ice_provider(
        "observed", datetime(2025, 2, 20, 12), where="a test start")
    # Nothing is asked until a domain with lake cells asks (door item 7).
    assert provider.observation is None and provider.fallback is None
    fields = {"SEAICE": np.zeros((1, 1))}
    out, skin, receipt = provider.freeze(
        fields, np.ones((1, 1)), lat=np.zeros((1, 1)), lon=np.zeros((1, 1)),
        lake_mask=np.ones((1, 1), bool), dx_m=3000.0, where="domain 1")
    assert "HTTP 404" in provider.fallback
    err = capsys.readouterr().err
    assert "warning: lake ice:" in err and "start as open water" in err
    assert "a test start" in err and "[ingest] gfs_lake_ice" in err
    assert out is fields and receipt["fallback"]


def test_wps_choice_reads_nothing(monkeypatch):
    monkeypatch.setattr(lake_ice, "acquire_observed_lake_ice",
                        lambda valid_time: pytest.fail("wps acquired an analysis"))
    provider = lake_ice.resolve_lake_ice_provider(
        "wps", datetime(2025, 2, 20, 12), where="x")
    assert provider.identity() == {"choice": "wps"}


def test_an_unknown_choice_is_refused_by_name():
    with pytest.raises(ValueError, match="gfs_lake_ice"):
        lake_ice.resolve_lake_ice_provider("ice", datetime(2025, 1, 1), where="x")


def test_the_gfs_policy_defaults_to_observed_and_keeps_wps(tmp_path):
    from woof.case_data import GFS_LAKE_ICE_CHOICES, gfs_preparation_policy

    assert GFS_LAKE_ICE_CHOICES == lake_ice.LAKE_ICE_CHOICES
    assert gfs_preparation_policy(None)["gfs_lake_ice"] == "observed"


def test_the_decision_receipt_carries_no_wall_time(monkeypatch, tmp_path):
    """The receipt is bound into the prepared cache's source identity.

    THE BREAKAGE (run_515470d250a1060a, 2026-10-26): it carried the
    sampling wall time, so two preparations of the same inputs differed
    (0.082 vs 0.08 s) and a checkpoint resume's forcing continuation
    refused with "continuation changes its immutable prepared head cache".
    """
    source = lake_ice.load_lake_ice_sources()[0]
    observation = lake_ice.ObservedLakeIce(
        source=source, path=tmp_path / "ims.asc.gz", sha256="0" * 64,
        valid_time=datetime(2026, 10, 25), url="https://example.invalid/ims")
    monkeypatch.setattr(lake_ice, "sample_ims_codes", lambda path, lat, lon, *, radius: (
        _counts(*[[0, 0, 0, 0, 0]] * len(lat)), np.zeros(len(lat), dtype=np.uint8),
        {"backend": "test"}))
    lat, lon = np.full((2, 3), 35.0), np.full((2, 3), -97.0)
    mask = np.array([[True, False, True], [False, True, False]])
    one = lake_ice.decide_lake_ice(observation, lat, lon, mask, dx_m=3000.0).receipt
    two = lake_ice.decide_lake_ice(observation, lat, lon, mask, dx_m=3000.0).receipt
    assert "seconds" not in one
    assert one == two
