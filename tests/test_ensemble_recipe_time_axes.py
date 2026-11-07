"""Recipe windows preserve forecast, analysis and supplied time semantics."""
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from woof.ensemble.recipes import SourceTrajectory, build_recipe

START = datetime(2024, 5, 21, 12, tzinfo=timezone.utc)


def singleton(source, *, start=START, hours=12):
    return build_recipe(source=source, cycle=start, start=start,
                        end=start + timedelta(hours=hours), count=1, base_seed=42)


@pytest.mark.parametrize("source", ["era5", "era5-l137", "20crv3"])
def test_analysis_singleton_is_not_limited_to_forecast_hour_zero(source):
    recipe = singleton(source)
    assert recipe.kind == "control"
    assert recipe.acquisitions() == (recipe.base,)
    assert recipe.base.window(recipe.start, recipe.end) == (0, 12)


def test_analysis_fetch_uses_valid_time_cadence_and_retrieves_bytes():
    recipe = singleton("era5")
    args = recipe.base.fetch_argv(recipe.start, recipe.end, "inputs")
    assert "--forecast-start-hour" not in args
    assert args[args.index("--cycle") + 1] == "2024-05-21T12"
    assert args[args.index("--hours") + 1] == "12"
    assert args[args.index("--cadence") + 1] == "6"
    assert "--retrieve" in args


def test_analysis_can_be_the_recentered_base():
    recipe = build_recipe(source="era5", cycle=START, start=START,
                          end=START + timedelta(hours=12), count=2, base_seed=42,
                          kind="recentered", donor=SourceTrajectory("gefs", START))
    assert recipe.acquisitions()[0].source == "era5"
    assert recipe.donor_population


def test_analysis_off_clock_knots_are_not_rounded_or_silently_dropped(monkeypatch):
    from woof import source_adapters
    from woof.source_cycles import CycleGrid
    ordinary = source_adapters.get_source_adapter
    six_hourly = replace(ordinary("era5-l137"), source_id="six-hourly-analysis", aliases=(),
                         forcing_interval_seconds=21600,
                         cycle_grid=CycleGrid((0, 6, 12, 18)))
    monkeypatch.setattr(source_adapters, "get_source_adapter",
                        lambda name: six_hourly if name == "six-hourly-analysis" else ordinary(name))
    with pytest.raises(ValueError, match="analysis valid-time knots"):
        singleton("six-hourly-analysis", start=START + timedelta(hours=1))
    # An end between analyses keeps its requested end and is bracketed by
    # the next analysis, as the single-run door fetches it (U-E2).
    bracketed = singleton("era5", hours=7)
    assert bracketed.end == START + timedelta(hours=7)
    assert bracketed.acquisition_window(bracketed.base) == (0, 12)
    # The hourly analysis row permits both same windows without a new path.
    assert singleton("era5-l137", start=START + timedelta(hours=1), hours=7).end.hour == 20


def test_analysis_cycle_labels_cannot_inflate_time_lag_ensemble_size():
    previous = SourceTrajectory("era5", START - timedelta(hours=6))
    with pytest.raises(ValueError, match="cannot create a distinct time-lagged forecast"):
        previous.window(START, START + timedelta(hours=12))
    with pytest.raises(ValueError, match="supplies 1 distinct trajectories"):
        build_recipe(source="era5", cycle=START, start=START,
                     end=START + timedelta(hours=12), count=2, base_seed=42,
                     kind="time-lagged", max_lag_hours=24)


@pytest.mark.parametrize("source", ["mapped", "era5-l137", "20crv3"])
def test_supplied_inputs_remain_plan_metadata_without_invented_acquisition(source):
    recipe = singleton(source)
    assert recipe.members[0].trajectory == recipe.base
    with pytest.raises(ValueError, match="no public recipe acquisition"):
        recipe.base.fetch_argv(recipe.start, recipe.end, "inputs")


def test_new_analysis_cadence_is_adapter_metadata(monkeypatch):
    from woof import source_adapters
    from woof.source_cycles import CycleGrid
    ordinary = source_adapters.get_source_adapter
    future = replace(ordinary("era5-l137"), source_id="future-analysis", aliases=(),
                     forcing_interval_seconds=10800,
                     cycle_grid=CycleGrid(tuple(range(0, 24, 3))))
    monkeypatch.setattr(source_adapters, "get_source_adapter",
                        lambda name: future if name == "future-analysis" else ordinary(name))
    assert singleton("future-analysis", hours=9).base.source == "future-analysis"
    bracketed = singleton("future-analysis", hours=10)
    assert bracketed.acquisition_window(bracketed.base) == (0, 12)


def test_forecast_window_still_emits_forecast_lead():
    trajectory = SourceTrajectory("gefs", START)
    args = trajectory.fetch_argv(START + timedelta(hours=3),
                                START + timedelta(hours=12), "inputs")
    assert args[args.index("--forecast-start-hour") + 1] == "3"
    assert args[args.index("--hours") + 1] == "9"
    assert "--as-posted" in args


def test_hourly_cam_keeps_exact_window_and_fetches_native_donor_brackets():
    cycle = START + timedelta(hours=6)
    start = cycle + timedelta(hours=1)
    recipe = build_recipe(source="hrrr", cycle=start, start=start,
        end=start + timedelta(hours=12), count=20, base_seed=42,
        kind="recentered", donor=SourceTrajectory("gefs", cycle))
    assert recipe.start == start and recipe.end == start + timedelta(hours=12)
    assert len(recipe.donor_population) == 30
    assert len(recipe.members) == 20
    assert recipe.acquisition_window(recipe.base) == (0, 12)
    for donor in recipe.donor_population:
        assert recipe.acquisition_window(donor) == (0, 15)
        args = recipe.fetch_argv(donor, "inputs")
        assert args[args.index("--forecast-start-hour") + 1] == "0"
        assert args[args.index("--hours") + 1] == "15"
        assert args[args.index("--cycle") + 1] == "2024-05-21T18"
    with pytest.raises(ValueError, match="does not publish"):
        recipe.members[0].trajectory.window(recipe.start, recipe.end)


def test_native_donor_brackets_do_not_extrapolate_or_relabel_analysis():
    trajectory = SourceTrajectory("gefs", START)
    with pytest.raises(ValueError, match="ends at"):
        trajectory.window(START, START + timedelta(hours=1000), native_bracketing=True)
    with pytest.raises(ValueError, match="complete initial"):
        trajectory.window(START - timedelta(hours=1), START + timedelta(hours=1),
                          native_bracketing=True)
    with pytest.raises(ValueError, match="cannot create a distinct time-lagged"):
        SourceTrajectory("era5", START).window(START + timedelta(hours=6),
            START + timedelta(hours=12), native_bracketing=True)


def test_legacy_native_fetch_owner_accepts_shifted_published_gfs_window():
    trajectory = SourceTrajectory("gfs", START)
    assert trajectory.window(START + timedelta(hours=1), START + timedelta(hours=13)) == (1, 13)
    args = trajectory.fetch_argv(START + timedelta(hours=1), START + timedelta(hours=13), "inputs")
    assert args[args.index("--forecast-start-hour") + 1] == "1"
    assert args[args.index("--hours") + 1] == "12"


def test_era5_hourly_analyses_start_off_the_six_hourly_clock():
    """WOOF Pickens, 2026-11-06: ERA5 2011-04-27T16, 12 h, refused by the WoFS
    ladder with "era5 2011-04-27T16Z does not cover the requested window".
    ERA5 analyses are hourly; the 6 h forcing spacing is taken from 16Z."""
    from woof.ensemble.wofs_recipe import WofsOptions, plan_ladder
    start = datetime(2011, 4, 27, 16, tzinfo=timezone.utc)
    end = start + timedelta(hours=12)
    trajectory = SourceTrajectory("era5", start)
    assert trajectory.window(start, end, native_bracketing=True) == (0, 12)
    args = trajectory.fetch_argv(start, end, "inputs")
    assert args[args.index("--cycle") + 1] == "2011-04-27T16"
    assert args[args.index("--hours") + 1] == "12"
    assert args[args.index("--cadence") + 1] == "6"
    ladder = plan_ladder(source="era5", cycle=start, start=start, end=end,
                         options=WofsOptions(members=12, provider="arco"))
    assert ladder.counts() == {"source-member": 1, "time-lagged": 0, "perturbation": 11}
    assert {m.cycle for m in ladder.members} == {start}


def test_era5_eda_has_no_member_off_its_three_hourly_clock(monkeypatch):
    from woof.ensemble.wofs_recipe import WofsOptions, plan_ladder, source_population
    monkeypatch.setattr("woof.fetch.cds_credentials_present", lambda: True)
    start = datetime(2011, 4, 27, 16, tzinfo=timezone.utc)
    assert len(source_population("era5", start)) == 1
    assert len(source_population("era5", start - timedelta(hours=1))) == 10
    ladder = plan_ladder(source="era5", cycle=start, start=start,
                         end=start + timedelta(hours=12), options=WofsOptions(members=4))
    assert ladder.counts() == {"source-member": 1, "time-lagged": 0, "perturbation": 3}
